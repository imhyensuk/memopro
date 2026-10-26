"""The five hibernation methods, per tensor: put to sleep (keep a restorable form) and wake.

Every ``sleep_*`` function builds the restorable form of ONE tensor and returns it; the caller
releases that tensor's memory right away, so extra memory never exceeds one tensor (0013 V6).
A function raises ``ModeUnavailable`` when its method does not fit this tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from memopro import _core
from memopro._errors import IntegrityError, ModeUnavailable
from memopro.hibernate import _source
from memopro.hibernate._tensors import Slot, cpu_bytes, empty_cpu, put_back

COMPRESS_WORTH = 0.9  # auto mode keeps compressing only if the result is at most 90% of the data
_OTHERS = ("source", "host", "compress")


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


# Piece size for verifying a tensor against its file (0061 F1): a multiple of the digest chunk, so
# the pieces' chunk hashes combine to the digest of the whole tensor.
VERIFY_PIECE = 8 * _core.DIGEST_CHUNK


def sleep_source(slot: Slot, regions: dict[str, _source.Region]) -> Sleeping:
    region = next((regions[n] for n in slot.names if n in regions), None)
    if region is None:
        raise _unavailable("source", "no original file found for this tensor")
    if not _source.matches_shape(region, slot.shape):
        raise _unavailable("source", "the file holds a different shape (weights were transformed)")
    if not _source.unchanged(region):
        raise _unavailable("source", "the original file changed since it was indexed")
    digest = _verified_digest(slot, region)
    if digest is None:
        raise _unavailable("source", "the file differs from the tensor (modified after loading)")
    return Sleeping("source", (region, digest), slot.nbytes, 0)


def _verified_digest(slot: Slot, region: _source.Region) -> bytes | None:
    """Digest of the tensor's bytes if the file region (in the tensor's dtype) equals them.

    Works piece by piece (0061 F1): extra memory is a few `VERIFY_PIECE` buffers, not a copy of
    the tensor, so hibernating does not first grow the process (0060: +34% CPU, +62% MPS, kept by
    the macOS allocator cache). Device tensors are copied to the host one piece at a time.
    """
    live = slot.tensor.detach()
    if not live.is_contiguous():
        live = live.contiguous()  # rare for parameters; costs one copy
    flat = live.reshape(-1)
    out_size = flat.element_size()
    per = VERIFY_PIECE // out_size
    total = flat.numel()
    file_buf = torch.empty(min(per, total), dtype=region.dtype)
    in_size = file_buf.element_size()
    parts = []
    for start in range(0, total, per):
        n = min(per, total - start)
        piece = file_buf[:n]
        _core.engine_read_source_into(
            region.path, region.offset + start * in_size, piece.view(torch.uint8).numpy()
        )
        if region.dtype != slot.dtype:
            piece = piece.to(slot.dtype)
        mine = flat[start : start + n]
        if mine.device.type != "cpu":
            mine = mine.to("cpu")
        piece_bytes, mine_bytes = piece.view(torch.uint8), mine.view(torch.uint8)
        if not torch.equal(piece_bytes, mine_bytes):
            return None
        parts.append(_core.engine_digest_parts(piece_bytes.numpy()))
    return _core.engine_digest_combine(total * out_size, b"".join(parts))


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


WAKE = {
    "source": wake_source,
    "host": wake_host,
    "compress": wake_compress,
}
