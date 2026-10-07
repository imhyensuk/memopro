"""``memopro.enable()``: one call that fits this process to this machine (0228, 0229).

It measures the machine and sets one **ceiling on the whole process**: ``"auto"`` = what the
process holds now plus what is free, minus headroom; or the user's cap such as ``"8GB"`` (never
above what is measured unless forced with ``!``). Then it turns on what this platform supports,
so code written as usual stays within it:

- every runtime and pager made in the session keeps to the ceiling *together*: each one's limit
  shrinks by what the process holds outside it (macOS physical footprint, Linux resident set);
  ``budget="auto"`` of ``memopro.rt`` (``finetune``, ``generate``, ``stream_model``,
  ``Runtime``, ``transparent``) means the ceiling;
- NumPy arrays of 16 MiB or more are paged within it (`memopro.rt.transparent`: Linux
  userfaultfd, macOS signals);
- PyTorch: activations saved for backward move into runtime buffers once the process passes 75%
  of the ceiling (`memopro.rt.activations`), losslessly;
- Hugging Face ``from_pretrained`` gets `memopro run`'s loading policy (only when the code chose
  no placement or precision and the model does not fit as stored);
- Apple silicon: PyTorch's MPS allocator is told to allocate exact sizes (0080), if torch has not
  started yet.

`Session.estimate` predicts the slowdown of a repeated pass over data before it runs (from a
sample: compression ratio and speed measured here); `Session.measured` reports the time memopro
took so far. Everything it did or could not do is listed in the session; ``disable()`` undoes it.

Limits (0229): memory that no part can move (the interpreter, libraries, small objects, model
weights the code loaded itself) still counts; when it alone passes the ceiling, the pager
records overruns and runtimes refuse with ``BudgetExceeded``. The kernel cannot fault paged
NumPy memory in (``tofile``/``fromfile`` may raise ``OSError``).
"""

from __future__ import annotations

import atexit
import contextlib
import importlib.util
import os
import sys
import time
import weakref
from dataclasses import dataclass, field
from typing import Any, Self

__all__ = ["Session", "disable", "enable"]

_active: Session | None = None


@dataclass
class Session:
    machine: str
    budget: int  # the ceiling on the whole process, bytes
    setting: str
    start_footprint: int | None = None
    done: list[tuple[str, str, str]] = field(default_factory=list)  # (part, status, detail)
    pager: Any = None
    activations: Any = None
    runtimes: weakref.WeakSet = field(default_factory=weakref.WeakSet, repr=False)
    started: float = field(default_factory=time.perf_counter)
    _stack: contextlib.ExitStack = field(default_factory=contextlib.ExitStack, repr=False)

    def _add(self, part: str, status: str, detail: str) -> None:
        self.done.append((part, status, detail))

    def summary(self) -> str:
        from memopro._units import format_size

        lines = [
            f"memopro.enable: {self.machine}",
            f"  ceiling       {format_size(self.budget)} for the whole process ({self.setting})",
        ]
        if self.start_footprint is not None:
            lines.append(f"  process       {format_size(self.start_footprint)} in use at start")
        lines += [f"  {part:<13} {status}: {detail}" for part, status, detail in self.done]
        return "\n".join(lines)

    __repr__ = summary

    def disable(self) -> None:
        """Undo everything `enable` did (arrays already paged keep their pager until freed)."""
        global _active
        atexit.unregister(self.disable)
        self._stack.close()
        if _active is self:
            _active = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.disable()

    def measured(self) -> dict[str, Any]:
        """What memopro cost so far: seconds the program waited for memopro (pager faults,
        runtime restores), the slowdown that implies, the process peak against the ceiling."""
        from memopro.rt import process_footprint

        elapsed = time.perf_counter() - self.started
        waited = 0.0
        overruns = 0
        pager = None
        if self.pager is not None:
            pager = dict(self.pager.stats())
            waited += pager["restore_seconds"] + pager["compress_seconds"]
            overruns = pager["overruns"]
        runtimes = [r.stats() for r in list(self.runtimes)]
        waited += sum(s["restore_seconds"] for s in runtimes)
        work = max(elapsed - waited, 1e-9)
        return {
            "elapsed_seconds": elapsed,
            "memopro_seconds": waited,
            "slowdown": elapsed / work,
            "ceiling": self.budget,
            "footprint": process_footprint(),
            "overruns": overruns,
            "activations_moved": 0 if self.activations is None else self.activations.moved,
            "pager": pager,
            "runtimes": runtimes,
            "predictions": [p for r in list(self.runtimes) if (p := r.predict()) is not None],
        }

    def estimate(
        self,
        sample: Any,
        *,
        total: int | str | None = None,
        passes: int = 1,
        compute_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Predict, before running, what repeated passes over ``total`` bytes of data like
        ``sample`` (an array or tensor; up to 64 MiB of it is used) cost under the ceiling.

        Measures the sample's lossless compression ratio and speed here, takes the room the
        process has left, and assumes passes in a fixed order (each pass brings back what did
        not stay in memory once). Random access costs more (E040b). ``compute_seconds`` (one
        pass without memopro) turns the extra time into a slowdown factor."""
        from memopro._units import parse_size
        from memopro.rt import process_footprint

        raw = _bytes_of(sample)
        total_bytes = raw.nbytes if total is None else parse_size(total)
        ratio, compress_rate, decompress_rate = _codec_speed(raw)
        now = process_footprint() or 0
        room = max(self.budget - now, 0)
        if total_bytes <= room:
            resident, moved = total_bytes, 0
        else:
            # resident r plus the rest compressed must fit: r + (total - r) * ratio <= room
            resident = int((room - total_bytes * ratio) / (1 - ratio)) if ratio < 1 else -1
            moved = total_bytes - resident
        fits = resident >= 0
        per_pass = moved / compress_rate + moved / decompress_rate if fits else float("inf")
        out: dict[str, Any] = {
            "fits": fits,
            "room": room,
            "ratio": ratio,
            "resident_bytes": max(resident, 0),
            "moved_per_pass": moved if fits else None,
            "extra_seconds_per_pass": per_pass,
            "extra_seconds": per_pass * passes,
            "assumes": "passes in a fixed order; random access is slower (E040b)",
        }
        if compute_seconds is not None:
            out["slowdown"] = (compute_seconds + per_pass) / compute_seconds
        return out


def enable(
    budget: Any = None,
    *,
    numpy: bool = True,
    torch: bool = True,
    transformers: bool = True,
    **settings: Any,
) -> Session:
    """Measure this machine and keep this process within one memory ceiling.

    ``budget``: ``None`` keeps the configured one (``"auto"`` unless set elsewhere); otherwise
    any budget form (``"8GB"``, ``0.5``, ``"-2GB"``, ``"2GB..8GB"``, ``"8GB!"``), applied to the
    whole process. Other settings (``quality``, ``budget_basis``, ...) go to
    `memopro.configure` for the session. ``numpy`` / ``torch`` / ``transformers`` turn their
    part off. Calling it again replaces the earlier session."""
    from memopro import config as cfgmod
    from memopro.env import detect
    from memopro.orchestrator.budget import compute_budget, describe_setting
    from memopro.rt import process_footprint

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
        held = process_footprint()
        # what the process holds already counts: the ceiling is for all of it (0229)
        b = compute_budget(env, cfg, resident_host=held or 0)
    except BaseException:
        _restore_config(saved)
        raise
    setting = describe_setting(cfg.budget)
    if setting == "auto":
        setting = "auto: in use + free, minus headroom"
    elif not b.capped_by_setting and not b.forced:
        setting += " asked; only this much is there"
    s = Session(_machine(env), b.host, setting, held)
    s._stack.callback(_restore_config, saved)
    _rt_default(s, b.host)
    for shortfall in b.shortfalls:
        s._add("budget", "warning", f"{shortfall.pool} minimum not met")
    if held is not None and held >= b.host:
        s._add("budget", "warning", "the process already holds more than the ceiling")
    if b.device is not None and not b.unified:
        from memopro._units import format_size

        s._add("device", "applied", f"{b.device_name}: {format_size(b.device)} for load/train")
    if mps is not None:
        s._add("mps", *mps)
    if numpy:
        _numpy(s, b.host)
    if torch and torch_here:
        _torch(s, b.host)
    if transformers:
        _transformers(s)
    if sys.platform == "darwin" and os.environ.get("MallocLargeCache") != "0":
        s._add("macOS", "note", "start Python with MallocLargeCache=0 so freed memory goes back")
    atexit.register(s.disable)  # before NumPy and torch go away at exit
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

    old = rt._default_budget, rt._process_budget, rt._session_runtimes
    rt._default_budget, rt._process_budget, rt._session_runtimes = nbytes, nbytes, s.runtimes

    def restore() -> None:
        rt._default_budget, rt._process_budget, rt._session_runtimes = old

    s._stack.callback(restore)
    s._add(
        "rt",
        "applied",
        'runtimes and pagers share the ceiling; budget="auto" (finetune, generate, '
        "stream_model, Runtime) means it",
    )


def _mps_heap() -> tuple[str, str] | None:
    from memopro.env import MPS_LOW_WATERMARK, MPS_LOW_WATERMARK_VAR, mps_heap_reserve_on

    if not mps_heap_reserve_on():
        return None
    if "torch" in sys.modules:
        return "skipped", "torch was imported first; call enable() before importing torch"
    os.environ[MPS_LOW_WATERMARK_VAR] = MPS_LOW_WATERMARK  # kept after disable: torch read it
    return "applied", f"{MPS_LOW_WATERMARK_VAR}={MPS_LOW_WATERMARK} (exact allocations)"


def _numpy(s: Session, nbytes: int) -> None:
    from memopro._errors import MemoproError, ModeUnavailable

    if importlib.util.find_spec("numpy") is None:
        return
    from memopro.rt import transparent

    try:
        s.pager = s._stack.enter_context(transparent(nbytes, process_budget=nbytes))
    except ModeUnavailable as e:
        s._add("numpy", "skipped", f"no transparent paging here ({e.reason})")
        return
    except MemoproError as e:  # a ceiling too small for the pager
        s._add("numpy", "skipped", str(e))
        return
    s._add("numpy", "applied", "arrays >= 16 MiB are paged within the ceiling (compressed)")


def _torch(s: Session, nbytes: int) -> None:
    from memopro._errors import MemoproError
    from memopro.rt import Runtime
    from memopro.rt.activations import PRESSURE, SavedActivations

    try:
        runtime = Runtime(nbytes, process_budget=nbytes)
    except MemoproError as e:  # a ceiling too small for a runtime
        s._add("torch", "skipped", str(e))
        return
    saved = SavedActivations(runtime, nbytes)
    s._stack.enter_context(saved.hooks())
    s.activations = saved
    s._add(
        "torch",
        "applied",
        f"activations saved for backward move into runtime buffers above {PRESSURE:.0%} of "
        "the ceiling (this thread)",
    )


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


def _bytes_of(sample: Any) -> Any:
    """A sample as a flat uint8 NumPy array (at most 64 MiB) with its element size."""
    import numpy as np

    if hasattr(sample, "detach"):  # a torch tensor
        import torch

        t = sample.detach().reshape(-1)
        elem = t.element_size()
        raw = t.view(torch.uint8).cpu().numpy()
    else:
        a = np.ascontiguousarray(sample)
        elem = a.itemsize
        raw = a.reshape(-1).view(np.uint8)
    limit = (64 << 20) // elem * elem
    raw = raw[:limit]
    return _Sample(raw, elem)


@dataclass
class _Sample:
    data: Any
    elem: int

    @property
    def nbytes(self) -> int:
        return int(self.data.nbytes)


def _codec_speed(sample: _Sample) -> tuple[float, float, float]:
    """Compression ratio and compress/decompress bytes per second of the runtime's codec on the
    sample, measured here."""
    from memopro.rt import Runtime

    n = sample.nbytes
    elem = sample.elem if sample.elem in (1, 2, 4, 8) else 1
    r = Runtime(2 * n + (32 << 20), prefetch=False, process_budget=False, min_saving=0.0)
    b = r.alloc(n, dtype=f"uint{8 * elem}")
    b.apply(lambda a: a.reshape(-1).view("uint8").__setitem__(..., sample.data), write=True)
    b.evict()
    b.apply(len)
    st = r.stats()
    b.free()
    if not st["compress_in"]:
        return 1.0, float("inf"), float("inf")
    ratio = st["compress_out"] / st["compress_in"]
    compress = st["compress_in"] / max(st["compress_seconds"], 1e-9)
    decompress = st["decompress_bytes"] / max(st["decompress_seconds"], 1e-9)
    return ratio, compress, decompress
