"""Accelerator facts from torch (imported here only, never at ``import memopro``).

CUDA: ``mem_get_info`` (free, total) per device; note that this creates a CUDA context.
MPS: the recommended working-set limit is the device ceiling on unified memory (K5).
ROCm and XPU are detected and listed, but budgets for them are not supported yet (fail-open).
"""

from __future__ import annotations

from importlib.util import find_spec

from memopro.env import Device

__all__ = ["probe"]


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
                    available_bytes=int(free),
                    limit_bytes=None,
                    allocated_bytes=int(torch.cuda.memory_reserved(i)),
                    unified=False,
                )
            )
        if kind == "rocm":
            notes.append("ROCm GPU detected: listed only, budgets are not supported yet")

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
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
