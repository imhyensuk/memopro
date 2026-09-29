"""Accelerator facts from torch (imported here only, never at ``import memopro``).

CUDA: ``mem_get_info`` (free, total) per device, plus the whole segments torch's allocator
caches unused in this process (`releasable_cache`); note that this creates a CUDA context.
MPS: the recommended working-set limit is the device ceiling on unified memory (K5).
ROCm and XPU are detected and listed, but budgets for them are not supported yet (fail-open).
"""

from __future__ import annotations

from functools import cache
from importlib.util import find_spec

from memopro.env import Device

__all__ = ["mps_usable", "probe"]

MPS_UNUSABLE_NOTE = (
    "Apple MPS reports available but cannot allocate memory here (for example a virtual "
    "machine without GPU access): not used"
)


@cache
def mps_usable() -> bool:
    """MPS is usable only if a tiny allocation really works (0042).

    ``torch.backends.mps.is_available()`` is also true in virtual machines (e.g. CI runners)
    where every allocation fails with "MPS backend out of memory".
    """
    if find_spec("torch") is None:
        return False
    import torch

    mps = getattr(torch.backends, "mps", None)
    if mps is None or not mps.is_available():
        return False
    try:
        return float((torch.ones(4, device="mps") + 1).sum().cpu()) == 8.0
    except RuntimeError:
        return False


def releasable_cache(index: int = 0) -> int:
    """Bytes torch's CUDA caching allocator holds in segments with nothing allocated: the driver
    counts them as used, but this process gets them back (``empty_cache``, or on the next
    out-of-memory error). Free pieces of segments that are partly in use are not counted: they
    serve only allocations that fit them, and after a 4-bit load 1.67 of 1.99 GiB of cached
    memory was never reused (E023 F2, 0103)."""
    import torch

    try:
        stats = torch.cuda.memory_stats(index)
        reserved = stats["reserved_bytes.all.current"]
        active = stats["active_bytes.all.current"]
        pieces = stats["inactive_split_bytes.all.current"]
    except (KeyError, RuntimeError, AssertionError):
        return 0
    return max(0, int(reserved - active - pieces))


def probe() -> tuple[tuple[Device, ...], tuple[str, ...]]:
    if find_spec("torch") is None:
        return (), ("torch not installed: device (GPU) memory is not reported",)
    import torch

    devices: list[Device] = []
    notes: list[str] = []

    if torch.cuda.is_available():
        kind = "rocm" if getattr(torch.version, "hip", None) else "cuda"
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            devices.append(
                Device(
                    kind=kind,
                    name=f"{kind}:{i} {torch.cuda.get_device_name(i)}",
                    total_bytes=int(total),
                    available_bytes=int(free + releasable_cache(i)),
                    limit_bytes=None,
                    allocated_bytes=int(torch.cuda.memory_reserved(i)),
                    unified=False,
                )
            )
        if kind == "rocm":
            notes.append("ROCm GPU detected: listed only, budgets are not supported yet")

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available() and not mps_usable():
        notes.append(MPS_UNUSABLE_NOTE)
    elif mps is not None and mps.is_available():
        limit = int(torch.mps.recommended_max_memory())
        allocated = int(torch.mps.driver_allocated_memory())
        devices.append(
            Device(
                kind="mps",
                name="Apple GPU (MPS)",
                total_bytes=None,  # unified: physical memory is the host's
                available_bytes=max(0, limit - allocated),
                limit_bytes=limit,
                allocated_bytes=allocated,
                unified=True,
            )
        )

    xpu = getattr(torch, "xpu", None)
    if xpu is not None and xpu.is_available():
        for i in range(xpu.device_count()):
            devices.append(
                Device("xpu", f"xpu:{i} {xpu.get_device_name(i)}", None, None, None, None, False)
            )
        notes.append("Intel XPU detected: listed only, budgets are not supported yet")

    if not devices:
        notes.append(f"torch {torch.__version__} has no GPU backend available here (CPU only)")
    return tuple(devices), tuple(notes)
