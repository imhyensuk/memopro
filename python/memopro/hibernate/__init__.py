"""β hibernate: reclaim memory held by idle tensors, modules and optimizers (0032, 0036).

Methods, tried in this order by ``mode="auto"`` (no SSD writes before the last one):

1. ``source``   drop the memory; restore by re-reading the original file (bit-exact, verified)
2. ``host``     move a CUDA tensor to host RAM
3. ``compress`` lossless compression in RAM (only if it pays off)
4. ``spill``    write to SSD, only when ``disk_writes`` allows it (consent under "ask")

``bf16`` (lossy, halves float32) is never chosen automatically. Methods apply per tensor, so a
model can end up partly ``source`` and partly ``compress``.

Two styles::

    h = memopro.hibernate.now(model)      # explicit handle, no proxies (0032 I5)
    model = h.wake()                      # modules also wake by themselves when called

    %hibernate model                      # notebook magics (0032 H6)

The tensors keep their identity: memopro swaps their storage for an empty one while they sleep
(0036 B1), so optimizers and other holders see the data again after waking. Using a sleeping
tensor directly fails loudly (it has 0 elements); a sleeping module or optimizer wakes itself on
``forward`` or ``step``.
"""

from __future__ import annotations

import inspect
import threading
import weakref
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from memopro._errors import InvalidArgument, MemoproError, ModeUnavailable
from memopro.config import get_config
from memopro.hibernate._policy import MODES, Step, resolve_modes

__all__ = [
    "MODES",
    "Handle",
    "PlanRow",
    "Step",
    "Suggestion",
    "enable",
    "handles",
    "now",
    "plan",
    "register_source",
    "resolve_modes",
    "status",
    "suggest",
    "wake",
]

MIN_OBJECT_BYTES = 1 << 20  # objects under 1 MiB are not worth suggesting (architecture §5.1)

# Rough transfer rates for restore-time estimates in plan(); orders of magnitude only (U6).
_RATES = {"source": 2e9, "spill": 2e9, "host": 10e9, "compress": 1.5e9, "bf16": 20e9}


@dataclass(frozen=True)
class PlanRow:
    """One line of ``plan()``: what a method would do, without doing it."""

    mode: str
    available: bool
    reason: str  # why not, when unavailable
    reclaim_bytes: int
    restore_seconds: float | None  # rough estimate
    disk_write_bytes: int
    fidelity: str  # "exact" | "numerics"


@dataclass(frozen=True)
class Suggestion:
    name: str
    type: str
    nbytes: int
    idle_cells: int | None


@dataclass
class _Record:
    slot: Any
    sleeping: Any


def _measure() -> dict[str, int]:
    """Memory per pool now: process RSS and accelerator driver memory (after emptying caches)."""
    import torch

    from memopro import _core

    out = {"rss": _core.hwinfo_process_rss()}
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        out["cuda"] = int(
            sum(torch.cuda.memory_reserved(i) for i in range(torch.cuda.device_count()))
        )
    from memopro.env._torch import mps_usable

    if mps_usable():
        torch.mps.empty_cache()
        out["mps"] = int(torch.mps.driver_allocated_memory())
    return out


class Handle:
    """A hibernated object. ``wake()`` restores it (in place) and returns it."""

    def __init__(self, obj: Any, name: str, mode: str) -> None:
        self._obj = obj
        self.name = name
        self.requested_mode = mode
        self.records: list[_Record] = []
        self.kept: dict[str, list[str]] = defaultdict(list)  # reason -> tensor names left awake
        # Measured change per pool, signed (0045 F1): positive = freed, negative = added.
        # E.g. mode "host" frees CUDA memory and adds the same amount of host RAM.
        self.reclaimed: dict[str, int] = {}
        self.asleep = False
        self._lock = threading.RLock()  # concurrent forwards may all try to wake (0048 S5)
        self._guards: Any = None

    # ------------------------------------------------------------ info
    @property
    def object(self) -> Any:
        return self._obj

    @property
    def nbytes(self) -> int:
        return sum(r.sleeping.nbytes for r in self.records)

    def bytes_by_mode(self) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for r in self.records:
            out[r.sleeping.mode] += r.sleeping.nbytes
        return dict(out)

    @property
    def disk_write_bytes(self) -> int:
        return sum(r.sleeping.disk_write_bytes for r in self.records)

    def __repr__(self) -> str:
        state = "asleep" if self.asleep else "awake"
        modes = ", ".join(f"{m} {b}" for m, b in self.bytes_by_mode().items())
        return f"<memopro.hibernate.Handle {self.name!r} {state}: {modes or 'nothing'}>"

    # ------------------------------------------------------------ wake
    def wake(self) -> Any:
        from memopro.hibernate import _methods
        from memopro.hibernate._tensors import asleep
        from memopro.report import report

        with self._lock:
            if not self.asleep:
                return self._obj
            failed = []
            for r in self.records:
                if not asleep(r.slot):
                    continue
                try:
                    _methods.WAKE[r.sleeping.mode](r.slot, r.sleeping)
                except Exception as e:  # noqa: BLE001 - restore every other tensor, then report
                    failed.append(f"{r.slot.names[0] or 'tensor'}: {type(e).__name__}: {e}")
            if failed:  # keep the guards: the object stays protected while data is missing
                raise failed_error(failed)
            # data first, then guards: an interrupted wake leaves the rest still guarded
            if self._guards is not None:
                self._guards.remove()
                self._guards = None
            self.asleep = False
        report().add("hibernate", "reverted", f"{self.name} woke up")
        return self._obj

    def discard(self) -> None:
        """Give up on data that cannot be restored: remove the guards and leave those tensors
        empty. Only for after `wake()` raised `IntegrityError`; restorable tensors are restored."""
        from memopro.hibernate import _methods
        from memopro.hibernate._tensors import asleep

        with self._lock:
            for r in self.records:
                if asleep(r.slot):
                    try:
                        _methods.WAKE[r.sleeping.mode](r.slot, r.sleeping)
                    except Exception:  # noqa: BLE001, S110 - discarding: leave it empty
                        pass
            if self._guards is not None:
                self._guards.remove()
                self._guards = None
            self.asleep = False


def failed_error(failed: list[str]) -> MemoproError:
    from memopro._errors import IntegrityError

    by_reason: dict[str, list[str]] = defaultdict(list)
    for item in failed:
        name, _, reason = item.partition(": ")
        by_reason[reason].append(name)
    parts = [
        f"{len(names)} tensor(s) ({', '.join(names[:3])}{', ...' if len(names) > 3 else ''}): "
        f"{reason}"
        for reason, names in by_reason.items()
    ]
    return IntegrityError(
        "could not restore " + "; ".join(parts) + ". The object stays hibernated and guarded; "
        "handle.discard() gives up on the missing data and removes the guards"
    )


# id(object) -> handle. Weak: a sleeping object keeps its own handle alive (guards), and a woken
# object and its handle are freed as soon as the user drops them (0048 S4).
_handles: weakref.WeakValueDictionary[int, Handle] = weakref.WeakValueDictionary()
# Serialises now(): two threads hibernating the same object at once would each pack tensors the
# other had already emptied (0048 T1).
_now_lock = threading.RLock()


def _lookup(obj: Any) -> Handle | None:
    handle = _handles.get(id(obj))
    return handle if handle is not None and handle.object is obj else None


def handles() -> list[Handle]:
    """Handles of hibernated objects still alive in this process (asleep or woken)."""
    return list(_handles.values())


def _guess_name(obj: Any) -> str:
    frame = inspect.currentframe()
    try:
        caller = frame.f_back.f_back if frame and frame.f_back else None
        if caller is not None:
            for name, value in {**caller.f_globals, **caller.f_locals}.items():
                if value is obj and not name.startswith("_"):
                    return name
    finally:
        del frame
    return type(obj).__name__


def now(obj: Any, mode: str = "auto", *, allow_spill: bool = False, name: str | None = None):
    """Hibernate ``obj`` (tensor, ``nn.Module`` or optimizer) now and return a `Handle`.

    ``allow_spill=True`` gives consent to write to SSD when ``disk_writes="ask"``.
    With an explicit ``mode`` no other method is used; tensors it does not fit stay in memory
    and are listed in ``handle.kept``. If it fits none, `ModeUnavailable` explains why.
    Safe to call from several threads; calls run one at a time.
    """
    with _now_lock:
        return _now(obj, mode, allow_spill, name)


def _now(obj: Any, mode: str, allow_spill: bool, name: str | None) -> Handle:
    import torch

    from memopro.hibernate import _source
    from memopro.hibernate._tensors import collect
    from memopro.report import report

    cfg = get_config()
    steps = resolve_modes(mode, cfg, allow_spill=allow_spill)
    existing = _lookup(obj)
    if existing is not None and existing.asleep:
        return existing
    slots = collect(obj)
    handle = Handle(obj, name or _guess_name(obj), mode)
    regions = (
        _source.regions(obj)
        if isinstance(obj, torch.nn.Module) and any(s.mode == "source" for s in steps)
        else {}
    )
    explicit = mode != "auto"
    before = _measure()
    try:
        _sleep_all(handle, slots, steps, regions, cfg, explicit)
    except BaseException:
        _roll_back(handle)  # Ctrl-C or a bug midway: nothing may stay empty (0048 S3)
        raise
    if not handle.records:
        reasons = "; ".join(handle.kept) or "nothing to hibernate"
        others = () if mode == "auto" else tuple(m for m in MODES if m not in (mode, "auto"))
        raise ModeUnavailable(mode, reasons, others)
    after = _measure()
    handle.reclaimed = {k: before[k] - after.get(k, before[k]) for k in before}
    handle.asleep = True
    _handles[id(obj)] = handle
    detail = f"{handle.name}: " + ", ".join(f"{m} {b}" for m, b in handle.bytes_by_mode().items())
    detail += "; measured " + ", ".join(
        f"{pool} {'freed' if v >= 0 else 'added'} {abs(v)}" for pool, v in handle.reclaimed.items()
    )
    freed = max(handle.reclaimed.values(), default=0)
    global _malloc_note_shown
    if not _malloc_note_shown and _cpu_side_freed(handle):
        from memopro.env import MALLOC_CACHE_NOTE

        _malloc_note_shown = True  # once per process
        report().add("hibernate", "suggested", MALLOC_CACHE_NOTE)
    report().add("hibernate", "applied", detail, reclaimed_bytes=max(freed, 0))
    return handle


_malloc_note_shown = False


def _cpu_side_freed(handle: Handle) -> bool:
    """Whether macOS' allocator cache keeps what this handle released (0061 F4): CPU tensors and,
    for their host-side copies, MPS tensors, unless MallocLargeCache=0 is set."""
    from memopro.env import macos_malloc_cache_on

    if not macos_malloc_cache_on():
        return False
    return any(r.slot.device.type in ("cpu", "mps") for r in handle.records)


def _roll_back(handle: Handle) -> None:
    from memopro.hibernate import _methods

    if handle._guards is not None:
        handle._guards.remove()
        handle._guards = None
    for r in reversed(handle.records):
        try:
            _methods.WAKE[r.sleeping.mode](r.slot, r.sleeping)
        except Exception:  # noqa: BLE001, S110 - best effort; the original error is re-raised
            pass
    handle.records.clear()


def _sleep_all(handle: Handle, slots: list, steps: list, regions: dict, cfg: Any, explicit: bool):
    from memopro.hibernate._guards import Guards
    from memopro.hibernate._tensors import release, shared_reason

    for slot in slots:
        label = slot.names[0] or "tensor"
        if slot.tensor.numel() == 0:  # emptied meanwhile (e.g. already asleep in another handle)
            continue
        if slot.tensor.is_meta:
            handle.kept["meta tensor: it holds no memory"].append(label)  # D3
            continue
        if not slot.tensor.is_contiguous():
            handle.kept["not contiguous"].append(label)
            continue
        shared = shared_reason(slot.tensor)
        if shared is not None:
            handle.kept[shared].append(label)  # D1
            continue
        done = False
        reasons: list[str] = []
        for step in steps:
            try:
                if step.needs_confirmation:
                    raise ModeUnavailable(
                        "spill",
                        "SSD writes need consent (disk_writes='ask'): pass allow_spill=True "
                        "or use %hibernate --spill",
                    )
                sleeping = _sleep(step.mode, slot, regions, cfg, explicit)
            except ModeUnavailable as e:
                reasons.append(f"{step.mode}: {e.reason}")
                continue
            except Exception as e:  # noqa: BLE001 - fail-open: this tensor stays awake (D3)
                reasons.append(f"{step.mode}: failed ({type(e).__name__}: {e})")
                continue
            release(slot)
            handle.records.append(_Record(slot, sleeping))
            done = True
            break
        if not done:  # every method tried, with its own reason (0048)
            handle.kept["; ".join(reasons) or "no method fits"].append(label)
    if handle.records:
        handle._guards = Guards(handle)
        handle._guards.install()


def _sleep(mode: str, slot: Any, regions: dict, cfg: Any, explicit: bool) -> Any:
    from memopro.hibernate import _methods

    match mode:
        case "source":
            return _methods.sleep_source(slot, regions)
        case "host":
            return _methods.sleep_host(slot)
        case "compress":
            return _methods.sleep_compress(slot, explicit=explicit)
        case "bf16":
            return _methods.sleep_bf16(slot)
        case "spill":
            return _methods.sleep_spill(slot, cfg)
    raise InvalidArgument(f"unknown mode {mode!r}")


def wake(target: Any) -> Any:
    """Restore a hibernated object now: a `Handle`, the object itself, or a notebook proxy."""
    if isinstance(target, Handle):
        return target.wake()
    wake_proxy = getattr(type(target), "_memopro_wake", None)
    if wake_proxy is not None:
        return wake_proxy(target)
    handle = _lookup(target)
    if handle is None:
        raise InvalidArgument("this object is not hibernated")
    return handle.wake()


def register_source(model: Any, *paths: str) -> None:
    """Declare the safetensors files a model was loaded from, for write-free restore."""
    from memopro.hibernate import _source

    _source.register(model, *paths)


def plan(obj: Any) -> list[PlanRow]:
    """Compare every method for ``obj`` without running it (0032 H6)."""
    import torch

    from memopro import _core
    from memopro.hibernate import _source, _ssd
    from memopro.hibernate._methods import COMPRESS_WORTH
    from memopro.hibernate._tensors import collect, cpu_bytes

    existing = _lookup(obj)
    if existing is not None and existing.asleep:
        raise InvalidArgument(
            "this object is hibernated; plan() compares methods before hibernating"
        )
    cfg = get_config()
    slots = [s for s in collect(obj) if s.tensor.numel() and not s.tensor.is_meta]
    total = sum(s.nbytes for s in slots)
    rows = []

    regions = _source.regions(obj) if isinstance(obj, torch.nn.Module) else {}
    matched = sum(
        s.nbytes
        for s in slots
        if any(n in regions and _source.matches_shape(regions[n], s.shape) for n in s.names)
    )
    rows.append(
        PlanRow(
            "source",
            matched > 0,
            "" if matched else "no original file found",
            matched,
            matched / _RATES["source"],
            0,
            "exact",
        )
    )

    cuda = sum(s.nbytes for s in slots if s.device.type == "cuda")
    reason = "" if cuda else "no CUDA tensors (unified memory or already in RAM)"
    rows.append(PlanRow("host", cuda > 0, reason, cuda, cuda / _RATES["host"], 0, "exact"))

    sample = max(slots, key=lambda s: s.nbytes, default=None)
    ratio = 1.0
    if sample is not None:
        piece = sample.tensor.detach().reshape(-1)[
            : (4 << 20) // max(sample.tensor.element_size(), 1)
        ]
        live, view = cpu_bytes(piece)
        packed = _core.codec_pack(view, live.element_size())
        ratio = packed.stored_bytes / max(packed.raw_bytes, 1)
    saving = int(total * (1 - ratio))
    rows.append(
        PlanRow(
            "compress",
            ratio <= COMPRESS_WORTH,
            f"estimated {ratio:.0%} of size",
            saving,
            total / _RATES["compress"],
            0,
            "exact",
        )
    )

    f32 = sum(s.nbytes for s in slots if s.dtype in (torch.float32, torch.float64))
    rows.append(
        PlanRow(
            "bf16",
            f32 > 0,
            "" if f32 else "no float32 tensors",
            f32 // 2,
            f32 / _RATES["bf16"],
            0,
            "numerics",
        )
    )

    reason = ""
    ok = cfg.disk_writes != "never"
    if not ok:
        reason = "disk_writes='never'"
    else:
        try:
            _ssd.check_room(cfg, total)
        except ModeUnavailable as e:
            ok, reason = False, e.reason
    if ok and cfg.disk_writes == "ask":
        reason = "needs consent (--spill / allow_spill=True)"
    rows.append(PlanRow("spill", ok, reason, total, total / _RATES["spill"], total, "exact"))
    return rows


def suggest(namespace: dict[str, Any] | None = None, min_bytes: int = MIN_OBJECT_BYTES):
    """Objects worth hibernating in ``namespace`` (default: the notebook or the caller's globals),
    largest first, with how many cells they have been idle when the notebook tracker runs."""
    from memopro.hibernate import _tracker
    from memopro.hibernate._tensors import storage_bytes

    if namespace is None:
        namespace = _tracker.namespace() or inspect.currentframe().f_back.f_globals
    out = []
    for name, obj in list(namespace.items()):
        handle = _lookup(obj)
        if name.startswith("_") or (handle is not None and handle.asleep):
            continue
        n = storage_bytes(obj)
        if n >= min_bytes:
            out.append(Suggestion(name, type(obj).__name__, n, _tracker.idle_cells(name)))
    return sorted(out, key=lambda s: -s.nbytes)


def enable(auto: bool = False, idle_cells: int | None = None, idle_seconds: float | None = None):
    """Notebook tracking: suggestions after cells, and optionally automatic hibernation.

    An object is idle after ``idle_cells`` cells or ``idle_seconds`` seconds without use.
    Automatic hibernation only uses write-free methods unless ``disk_writes="allow"``.
    """
    from memopro.hibernate import _tracker

    _tracker.enable(auto=auto, idle_cells=idle_cells, idle_seconds=idle_seconds)


def status() -> dict[str, Any]:
    """Hibernated objects, reclaimed memory and SSD writes (0032 H5)."""
    from memopro.hibernate import _ssd

    cfg = get_config()
    try:
        directory = _ssd.spill_dir(cfg)
        today, total = _ssd.written_today(directory), _ssd.written_total(directory)
    except OSError:
        directory, today, total = None, 0, 0
    return {
        "objects": [
            {
                "name": h.name,
                "asleep": h.asleep,
                "bytes": h.nbytes,
                "by_mode": h.bytes_by_mode(),
                "reclaimed": h.reclaimed,
                "kept_awake": {k: len(v) for k, v in h.kept.items()},
            }
            for h in list(_handles.values())
        ],
        "ssd_written_today": today,
        "ssd_written_total": total,
        "spill_dir": str(directory) if directory else None,
        "disk_writes": cfg.disk_writes,
    }
