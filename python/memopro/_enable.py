"""``memopro.enable()``: one call that fits this process to this machine (0228).

It measures the machine, sets one budget (``"auto"`` from what is free, or the user's cap such as
``"8GB"``) and turns on what this platform supports, so code written as usual stays within it:

- the budget becomes the process default: ``memopro.load``/``check``/``train_session`` and every
  ``budget="auto"`` of ``memopro.rt`` (``finetune``, ``generate``, ``stream_model``, ``Runtime``,
  ``transparent``) use it;
- Linux: NumPy arrays of 16 MiB or more are paged within the budget (`memopro.rt.transparent`);
- Hugging Face ``from_pretrained`` gets `memopro run`'s loading policy (only when the code chose
  no placement or precision and the model does not fit as stored);
- Apple silicon: PyTorch's MPS allocator is told to allocate exact sizes (0080), if torch has not
  started yet.

Everything it did or could not do is listed in the returned session. ``disable()`` undoes it.
Limit (0228): each part keeps to the budget on its own; memory outside them (small objects, C
extensions) is not counted, so the budget is not yet a ceiling on the whole process.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Self

__all__ = ["Session", "disable", "enable"]

_active: Session | None = None


@dataclass
class Session:
    machine: str
    budget: int
    setting: str
    done: list[tuple[str, str, str]] = field(default_factory=list)  # (part, status, detail)
    _stack: contextlib.ExitStack = field(default_factory=contextlib.ExitStack, repr=False)

    def _add(self, part: str, status: str, detail: str) -> None:
        self.done.append((part, status, detail))

    def summary(self) -> str:
        from memopro._units import format_size

        lines = [
            f"memopro.enable: {self.machine}",
            f"  budget        {format_size(self.budget)} ({self.setting})",
        ]
        lines += [f"  {part:<13} {status}: {detail}" for part, status, detail in self.done]
        return "\n".join(lines)

    __repr__ = summary

    def disable(self) -> None:
        """Undo everything `enable` did (arrays already paged keep their pager until freed)."""
        global _active
        self._stack.close()
        if _active is self:
            _active = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.disable()


def enable(
    budget: Any = None,
    *,
    numpy: bool = True,
    transformers: bool = True,
    **settings: Any,
) -> Session:
    """Measure this machine and keep memopro's work within one budget.

    ``budget``: ``None`` keeps the configured one (``"auto"`` unless set elsewhere); otherwise
    any budget form (``"8GB"``, ``0.5``, ``"-2GB"``, ``"2GB..8GB"``, ``"8GB!"``). Other settings
    (``quality``, ``budget_basis``, ...) go to `memopro.configure` for the session. ``numpy`` /
    ``transformers`` turn their part off. Calling it again replaces the earlier session."""
    from memopro import config as cfgmod
    from memopro.env import detect
    from memopro.orchestrator.budget import compute_budget, describe_setting

    if _active is not None:
        _active.disable()
    if budget is not None:
        settings["budget"] = budget
    saved = dict(cfgmod._overrides)
    cfgmod.configure(**settings)  # validates before anything changes
    mps = _mps_heap()  # before detection imports torch
    try:
        cfg = cfgmod.get_config()
        torch_here = importlib.util.find_spec("torch") is not None
        env = detect(devices=torch_here, config=cfg)
        b = compute_budget(env, cfg)
    except BaseException:
        cfgmod._overrides.clear()
        cfgmod._overrides.update(saved)
        raise
    setting = describe_setting(cfg.budget)
    if setting == "auto":
        setting = "auto: free memory minus headroom"
    elif not b.capped_by_setting and not b.forced:
        setting += " asked; only this much is free"
    s = Session(_machine(env), b.host, setting)
    s._stack.callback(_restore_config, saved)
    _rt_default(s, b.host)
    for shortfall in b.shortfalls:
        s._add("budget", "warning", f"{shortfall.pool} minimum not met")
    if b.device is not None and not b.unified:
        from memopro._units import format_size

        s._add("device", "applied", f"{b.device_name}: {format_size(b.device)} for load/train")
    if mps is not None:
        s._add("mps", *mps)
    if numpy:
        _numpy(s, b.host)
    if transformers:
        _transformers(s)
    if sys.platform == "darwin" and os.environ.get("MallocLargeCache") != "0":
        s._add("macOS", "note", "start Python with MallocLargeCache=0 so freed memory goes back")
    _set_active(s)
    return s


def _set_active(s: Session) -> None:
    global _active
    _active = s


def disable() -> None:
    """End the session `enable` started, if any."""
    if _active is not None:
        _active.disable()


def _machine(env: Any) -> str:
    from memopro._units import format_size

    gpus = ", ".join(d.name for d in env.devices) or "no GPU"
    return f"{env.os} {env.machine}, {format_size(env.host.total_bytes)} memory, {gpus}"


def _restore_config(saved: dict[str, Any]) -> None:
    from memopro import config as cfgmod

    cfgmod._overrides.clear()
    cfgmod._overrides.update(saved)


def _rt_default(s: Session, nbytes: int) -> None:
    from memopro import rt

    old = rt._default_budget
    rt._default_budget = nbytes
    s._stack.callback(setattr, rt, "_default_budget", old)
    s._add("rt", "applied", 'budget="auto" in finetune/generate/stream_model/Runtime uses it')


def _mps_heap() -> tuple[str, str] | None:
    from memopro.env import MPS_LOW_WATERMARK, MPS_LOW_WATERMARK_VAR, mps_heap_reserve_on

    if not mps_heap_reserve_on():
        return None
    if "torch" in sys.modules:
        return "skipped", "torch was imported first; call enable() before importing torch"
    os.environ[MPS_LOW_WATERMARK_VAR] = MPS_LOW_WATERMARK  # kept after disable: torch read it
    return "applied", f"{MPS_LOW_WATERMARK_VAR}={MPS_LOW_WATERMARK} (exact allocations)"


def _numpy(s: Session, nbytes: int) -> None:
    from memopro._errors import ModeUnavailable

    if importlib.util.find_spec("numpy") is None:
        return
    from memopro.rt import transparent

    try:
        s._stack.enter_context(transparent(nbytes))
    except ModeUnavailable:
        s._add("numpy", "skipped", "no transparent paging on this OS; use memopro.rt.Runtime")
        return
    s._add("numpy", "applied", "arrays >= 16 MiB are paged within the budget (compressed)")


def _transformers(s: Session) -> None:
    from memopro._run import TARGET, _LoadingPolicy, _PatchOnImport

    if importlib.util.find_spec("transformers") is None:
        return
    policy = _LoadingPolicy()
    hook = _PatchOnImport(policy.install)
    if TARGET in sys.modules:
        policy.install(sys.modules[TARGET])
    sys.meta_path.insert(0, hook)
    s._stack.callback(policy.remove)
    s._stack.callback(lambda: hook in sys.meta_path and sys.meta_path.remove(hook))
    s._add(
        "transformers", "applied", "from_pretrained loads models that do not fit as memopro.load"
    )
