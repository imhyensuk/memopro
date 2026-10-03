"""``memopro run script.py``: memopro's process-level features without changing code (v0.3, L0).

What it does (architecture §4.2, 0052 E7), and all of it is listed in ``report()``:

1. **Loading policy** for Hugging Face ``from_pretrained``. It applies only when the script chose
   none of ``device_map``, ``dtype``, ``quantization_config``, ``max_memory`` (and passed no
   positional model arguments), and only when the model would not fit as stored: then memopro
   loads it the way ``memopro.load`` would. An explicit choice in the script is never changed.
   If memopro's choice fails to load, the script's own call runs as written (fail-open).
2. **γ** (experimental, 0088, 0093): memory pressure is watched; ``memopro.load``/
   ``train_session`` in the script and the loading policy react to it. Off by default (``--elastic``
   turns it on): the macOS level marks pressure but not stalls a user feels (E013a), and on Linux
   the I/O of loading a model raised PSI to "warning", halving budgets (E021).
3. **census** (``--census``): saved activations of the whole run, summarised at the end.

This is the only mode where memopro patches another library globally (P4 exception); the patch
is installed when ``transformers.modeling_utils`` is imported and removed when the script ends.
The script runs in this process with ``__name__ == "__main__"`` and its own ``sys.argv``.
"""

from __future__ import annotations

import contextlib
import importlib.abc
import importlib.util
import runpy
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from memopro._errors import InvalidArgument
from memopro.report import report

__all__ = ["run"]

TARGET = "transformers.modeling_utils"


class _Loader(importlib.abc.Loader):
    def __init__(self, inner: Any, after: Callable[[Any], None]) -> None:
        self.inner = inner
        self.after = after

    def create_module(self, spec: Any) -> Any:
        return self.inner.create_module(spec)

    def exec_module(self, module: Any) -> None:
        self.inner.exec_module(module)
        self.after(module)


class _PatchOnImport(importlib.abc.MetaPathFinder):
    """Run ``after(module)`` once ``TARGET`` has been imported, however the script imports it."""

    def __init__(self, after: Callable[[Any], None]) -> None:
        self.after = after
        self._busy = False

    def find_spec(self, fullname: str, path: Any, target: Any = None) -> Any:
        if fullname != TARGET or self._busy:
            return None
        self._busy = True
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            self._busy = False
        if spec is None or spec.loader is None:
            return None
        spec.loader = _Loader(spec.loader, self.after)
        return spec


class _LoadingPolicy:
    """Wraps ``PreTrainedModel.from_pretrained`` while the script runs."""

    def __init__(self) -> None:
        self.cls: Any = None
        self.original: Any = None
        self.decisions: list[str] = []

    def install(self, module: Any) -> None:
        cls = module.PreTrainedModel
        if self.cls is not None:
            return
        self.cls, self.original = cls, cls.__dict__["from_pretrained"]
        policy = self

        def from_pretrained(klass: Any, name: Any, *args: Any, **kwargs: Any) -> Any:
            return policy.load(klass, name, args, kwargs)

        cls.from_pretrained = classmethod(from_pretrained)
        report().add("run.from_pretrained", "applied", "loading policy installed (P4 exception)")

    def remove(self) -> None:
        if self.cls is not None:
            self.cls.from_pretrained = self.original
            self.cls = None

    def load(self, klass: Any, name: Any, args: tuple, kwargs: dict[str, Any]) -> Any:
        from memopro.access._load import RESERVED, plan_load
        from memopro.orchestrator.candidates import from_pretrained_kwargs, post_load, speed_hint

        original = self.original.__func__
        explicit = sorted(set(kwargs) & set(RESERVED))
        if args or explicit:
            return original(klass, name, *args, **kwargs)
        try:
            plan = plan_load(name, revision=kwargs.get("revision"))
        except Exception as e:  # noqa: BLE001 - the script's own call must still run
            report().add("run.from_pretrained", "skipped", f"{name}: {type(e).__name__}: {e}"[:200])
            return original(klass, name, **kwargs)
        chosen = plan.chosen
        stored_fits = any(c.name == "stored" and c.usable for c in plan.candidates)
        if stored_fits:  # run intervenes only when the model does not fit as stored (0052 E7);
            chosen = next(c for c in plan.candidates if c.name == "stored")  # not for speed (D2)
        if chosen is None or chosen.name == "stored":
            why = "fits as stored" if chosen is not None else "nothing memopro may use fits"
            report().add("run.from_pretrained", "skipped", f"{name}: {why}; loaded as written")
            if chosen is None:
                _suggest(plan)
            return original(klass, name, **kwargs)
        extra = from_pretrained_kwargs(chosen, plan.ctx)
        post = post_load(chosen, plan.ctx)
        try:
            model = original(klass, name, **kwargs, **extra)
            if post is not None:
                model = post(model)
        except Exception as e:  # noqa: BLE001 - fail-open: load as the script asked
            report().add("run.from_pretrained", "failed", f"{name}: {type(e).__name__}: {e}"[:200])
            return original(klass, name, **kwargs)
        self.decisions.append(f"{name}: {chosen.describe()}")
        if hint := speed_hint(chosen, plan.candidates, plan.ctx):  # in run: the CLI option
            hint = hint.replace("quality='low'", "memopro run --quality low")
            report().add("run.from_pretrained", "suggested", f"{name}: {hint}")
        report().add(
            "run.from_pretrained",
            "applied",
            f"{name} does not fit as stored: loaded as {chosen.describe()} "
            f"({chosen.quality.name.lower()})",
        )
        return model


def _suggest(plan: Any) -> None:
    """Nothing fits: say which settings would (0064 D-a); the script's own call still runs."""
    from memopro.access._suggest import suggest_for_load

    try:
        suggestions = suggest_for_load(plan)
    except Exception:  # noqa: BLE001 - advice only; never break the script
        return
    if suggestions:
        advice = "; ".join(f"{s.call()} -> {s.describe}" for s in suggestions)
        report().add("run.from_pretrained", "suggested", f"{plan.info.source}: {advice}")


def plan_text(script: str, *, elastic: bool, census: bool) -> str:
    from memopro._units import format_size
    from memopro.access._common import setup
    from memopro.orchestrator.budget import describe_setting

    s = setup()
    device = "-" if s.budget.device is None else format_size(s.budget.device)
    setting = describe_setting(s.config.budget)
    cfg = s.config
    lines = [
        f"memopro run {script}",
        (
            f"  device         {s.device}, budget {setting} "
            f"(device {device}, host {format_size(s.budget.host)})"
        ),
        f"  quality        {cfg.quality}, prefer {cfg.prefer}, disk writes {cfg.disk_writes}",
        "  from_pretrained  only when the script chose no placement or precision and the model",
        "                   does not fit as stored: loaded as memopro.load would (P4 exception)",
        f"  elastic        {'on: memory pressure is watched (experimental)' if elastic else 'off'}"
        + ("" if elastic else " (default, 0088/0093; --elastic)"),
        f"  census         {'on: saved activations of the whole run' if census else 'off'}",
    ]
    return "\n".join(lines)


def _report_pager(pager: Any) -> None:
    from memopro._units import format_size

    st = pager.stats()
    report().add(
        "transparent",
        "applied",
        (
            f"budget {format_size(st['budget'])}, peak {format_size(st['peak_used'])}; "
            f"{st['faults']} faults, {st['evictions']} chunks compressed "
            f"({format_size(st['compress_in'])} to {format_size(st['compress_out'])}), "
            f"{st['restores']} brought back, {st['overruns']} over budget; nothing written"
        ),
    )


ELASTIC_OFF_NOTE = (
    "γ is off by default (experimental): the macOS pressure level marks pressure, not the stalls "
    "a user feels (E013a/0088), and Linux PSI rose above 'warning' from the I/O of loading a "
    "model, halving budgets so the loading policy stood aside (E021/0093); --elastic turns it on"
)


def default_elastic() -> bool:
    """γ's default in ``memopro run``: off everywhere (0088 G1 on macOS, E021 D4 on Linux)."""
    return False


def run(
    script: str,
    argv: Sequence[str] = (),
    *,
    elastic: bool | None = None,
    census: bool = False,
    dry_run: bool = False,
    transparent: str | None = None,
    report_json: str | None = None,
) -> None:
    """Run ``script`` with memopro's process-level features (settings come from `configure`).

    ``elastic=None`` takes the platform default (`default_elastic`). ``transparent`` (a budget,
    Linux) pages the script's large NumPy arrays within that budget (`memopro.rt.transparent`,
    0124). ``report_json`` writes the report (and the pager's counters) to that file at the end."""
    default = elastic is None
    if default:
        elastic = default_elastic()
    path = Path(script)
    if not path.is_file():
        raise InvalidArgument(f"no such script: {script}")
    if dry_run:
        print(plan_text(script, elastic=elastic, census=census))
        return
    if default and not elastic:
        report().add("elastic", "skipped", ELASTIC_OFF_NOTE)
    policy = _LoadingPolicy()
    hook = _PatchOnImport(policy.install)
    if TARGET in sys.modules:
        policy.install(sys.modules[TARGET])
    sys.meta_path.insert(0, hook)
    if elastic:
        from memopro import elastic as gamma

        gamma.enable(notebook=False)
    saved_argv, saved_path0 = sys.argv[:], sys.path[0] if sys.path else None
    sys.argv = [str(path), *argv]
    sys.path.insert(0, str(path.resolve().parent))
    recorder = None
    pager = None
    try:
        with contextlib.ExitStack() as stack:
            if transparent is not None:
                from memopro.rt import transparent as paged

                pager = stack.enter_context(paged(transparent))
            if census:
                from memopro.census import record

                recorder = stack.enter_context(record())
            runpy.run_path(str(path), run_name="__main__")
    finally:
        if pager is not None:
            _report_pager(pager)
        if report_json is not None:
            import json

            data = {
                "report": report().to_dict(),
                "transparent": None if pager is None else dict(pager.stats()),
            }
            Path(report_json).write_text(json.dumps(data, indent=1), encoding="utf-8")
        sys.argv = saved_argv
        if sys.path and sys.path[0] == str(path.resolve().parent) and saved_path0 != sys.path[0]:
            sys.path.pop(0)
        if hook in sys.meta_path:
            sys.meta_path.remove(hook)
        policy.remove()
        if elastic:
            from memopro import elastic as gamma

            gamma.disable()
        print(report().summary(), file=sys.stderr)
        if recorder is not None and recorder._result is not None:
            print(recorder.summary(), file=sys.stderr)
