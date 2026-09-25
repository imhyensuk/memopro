"""The five hibernation methods, per tensor: put to sleep (keep a restorable form) and wake.

Every ``sleep_*`` function builds the restorable form of ONE tensor and returns it; the caller
releases that tensor's memory right away, so extra memory never exceeds one tensor (0013 V6).
A function raises ``ModeUnavailable`` when its method does not fit this tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from memopro import _core
from memopro._errors import IntegrityError, ModeUnavailable
from memopro.config import Config
from memopro.hibernate import _source, _ssd
from memopro.hibernate._tensors import Slot, cpu_bytes, empty_cpu, put_back

COMPRESS_WORTH = 0.9  # auto mode keeps compressing only if the result is at most 90% of the data
_OTHERS = ("source", "host", "compress", "spill")

# H3: spill files kept after waking, reused when the same bytes are hibernated again
_kept_spills: dict[int, tuple[str, int, bytes]] = {}


@dataclass
class Sleeping:
    mode: str
    form: Any  # mode-specific restorable form
    nbytes: int  # original bytes
    held_bytes: int  # bytes still held in memory by the form (host RAM or device)
    disk_write_bytes: int = 0


def _unavailable(mode: str, reason: str) -> ModeUnavailable:
    return ModeUnavailable(mode, reason, tuple(m for m in _OTHERS if m != mode))


# ---------------------------------------------------------------- source (H1)


def sleep_source(slot: Slot, regions: dict[str, _source.Region]) -> Sleeping:
    region = next((regions[n] for n in slot.names if n in regions), None)
    if region is None:
        raise _unavailable("source", "no original file found for this tensor")
    if not _source.matches_shape(region, slot.shape):
        raise _unavailable("source", "the file holds a different shape (weights were transformed)")
    if not _source.unchanged(region):
        raise _unavailable("source", "the original file changed since it was indexed")
    _live, live_view = cpu_bytes(slot.tensor)  # the view keeps its tensor alive
    buf, buf_view = empty_cpu(region.shape, region.dtype)
    _core.engine_read_source_into(region.path, region.offset, buf_view)
    if region.dtype != slot.dtype:
        buf = buf.to(slot.dtype)
        buf_view = buf.reshape(-1).view(torch.uint8).numpy()
    digest = _core.engine_digest(live_view)
    if _core.engine_digest(buf_view) != digest:
        raise _unavailable("source", "the file differs from the tensor (modified after loading)")
    return Sleeping("source", (region, digest), slot.nbytes, 0)


def wake_source(slot: Slot, sleeping: Sleeping) -> None:
    region, digest = sleeping.form
    if not _source.unchanged(region):
        raise IntegrityError(
            f"{region.path} changed while the tensor slept; its data cannot be restored"
        )
    buf, buf_view = empty_cpu(region.shape, region.dtype)
    _core.engine_read_source_into(region.path, region.offset, buf_view)
    if region.dtype != slot.dtype:
        buf = buf.to(slot.dtype)
    view = buf.reshape(-1).view(torch.uint8).numpy()
    if _core.engine_digest(view) != digest:
        raise IntegrityError(f"{region.path} no longer holds the tensor's data")
    put_back(slot, buf)


# ---------------------------------------------------------------- host


def sleep_host(slot: Slot) -> Sleeping:
    if slot.device.type != "cuda":
        reason = (
            "unified memory: moving to host RAM frees nothing"
            if slot.device.type == "mps"
            else "the tensor is already in host RAM"
        )
        raise _unavailable("host", reason)
    cpu = slot.tensor.detach().to("cpu")
    return Sleeping("host", cpu, slot.nbytes, slot.nbytes)


def wake_host(slot: Slot, sleeping: Sleeping) -> None:
    put_back(slot, sleeping.form)


# ---------------------------------------------------------------- compress


def sleep_compress(slot: Slot, *, explicit: bool) -> Sleeping:
    live, view = cpu_bytes(slot.tensor)
    packed = _core.codec_pack(view, live.element_size())
    if not explicit and packed.stored_bytes > COMPRESS_WORTH * packed.raw_bytes:
        raise _unavailable(
            "compress",
            f"compresses only to {packed.stored_bytes / max(packed.raw_bytes, 1):.0%} of its size",
        )
    return Sleeping("compress", packed, slot.nbytes, packed.stored_bytes)


def wake_compress(slot: Slot, sleeping: Sleeping) -> None:
    buf, view = empty_cpu(slot.shape, slot.dtype)
    sleeping.form.unpack_into(view)
    put_back(slot, buf)


# ---------------------------------------------------------------- bf16 (lossy, explicit only)


def sleep_bf16(slot: Slot) -> Sleeping:
    if slot.dtype not in (torch.float32, torch.float64):
        raise _unavailable("bf16", f"{slot.dtype} is already 16-bit or not floating point")
    low = slot.tensor.detach().to(torch.bfloat16)
    return Sleeping("bf16", low, slot.nbytes, low.numel() * 2)


def wake_bf16(slot: Slot, sleeping: Sleeping) -> None:
    put_back(slot, sleeping.form)


# ---------------------------------------------------------------- spill (SSD, last resort)


def sleep_spill(slot: Slot, config: Config) -> Sleeping:
    _live, view = cpu_bytes(slot.tensor)
    kept = _kept_spills.get(id(slot.tensor))
    if kept is not None and Path(kept[0]).exists() and _core.engine_digest(view) == kept[2]:
        return Sleeping("spill", kept, slot.nbytes, 0, 0)  # H3: unchanged, no rewrite
    directory = _ssd.check_room(config, slot.nbytes)
    if kept is not None:
        _ssd.remove(kept[0])
    written = _ssd.write(view, directory)
    _kept_spills[id(slot.tensor)] = written
    return Sleeping("spill", written, slot.nbytes, 0, written[1])


def wake_spill(slot: Slot, sleeping: Sleeping) -> None:
    path, _, digest = sleeping.form
    buf, view = empty_cpu(slot.shape, slot.dtype)
    try:
        _core.engine_read_source_into(path, 0, view, digest)
    except (RuntimeError, OSError) as e:
        raise IntegrityError(f"spill file {path} could not be restored: {e}") from None
    put_back(slot, buf)


def drop_kept(tensor: torch.Tensor) -> None:
    kept = _kept_spills.pop(id(tensor), None)
    if kept is not None:
        _ssd.remove(kept[0])


WAKE = {
    "source": wake_source,
    "host": wake_host,
    "compress": wake_compress,
    "bf16": wake_bf16,
    "spill": wake_spill,
}
