"""Per-pool budget vector (architecture §3.2): {device, host, disk}.

- host:   usable host memory (conservative available, capped by the container) minus headroom
- device: CUDA free memory minus headroom; on unified memory (MPS) the smaller of the MPS limit
          left and usable host memory, minus headroom, and it shares one pool with ``host`` (K5)
- disk:   free space above the ``min_free_disk_fraction`` floor (0032 H4). Whether memopro may
          write there at all is a policy question (``disk_writes``), not part of the budget.
- an explicit ``budget`` setting caps device and host: a size caps both, a fraction scales both,
  a `PoolBudget` caps each pool on its own (0052 E3). It never raises a pool above what is there.
"""

from __future__ import annotations

from dataclasses import dataclass

from memopro.config import Config, PoolBudget
from memopro.env import Env

__all__ = ["Budget", "compute_budget"]

_BUDGETED_KINDS = ("cuda", "mps")


@dataclass(frozen=True)
class Budget:
    device: int | None  # None: no supported accelerator
    host: int
    disk: int
    unified: bool = False  # device and host draw from the same physical memory
    device_name: str | None = None
    headroom: float = 0.0
    disk_floor_bytes: int = 0
    capped_by_setting: bool = False


def describe_setting(spec: object) -> str:
    """The budget setting as people write it."""
    from memopro._units import format_size

    if isinstance(spec, PoolBudget):
        parts = [f"{k} {format_size(v)}" for k, v in (("device", spec.device), ("host", spec.host))]
        return ", ".join(p for p, v in zip(parts, (spec.device, spec.host)) if v is not None)
    if isinstance(spec, float):
        return f"{spec:.0%} of the measured budget"
    return str(spec) if spec == "auto" else format_size(int(spec))  # type: ignore[call-overload]


def compute_budget(env: Env, config: Config) -> Budget:
    keep = 1.0 - config.headroom
    host = int(env.host.usable_bytes * keep)

    device = None
    device_name = None
    unified = False
    primary = next((d for d in env.devices if d.kind in _BUDGETED_KINDS), None)
    if primary is not None and primary.available_bytes is not None:
        free = primary.available_bytes
        if primary.unified:
            free = min(free, env.host.usable_bytes)
            unified = True
        device = int(free * keep)
        device_name = primary.name

    floor = int(env.disk.total_bytes * config.min_free_disk_fraction)
    disk = max(0, env.disk.available_bytes - floor)

    measured_host, measured_device = host, device
    spec = config.budget
    if isinstance(spec, PoolBudget):
        if spec.host is not None:
            host = min(host, spec.host)
        if spec.device is not None and device is not None:
            device = min(device, spec.device)
    elif isinstance(spec, float):
        host = int(host * spec)
        device = None if device is None else int(device * spec)
    elif spec != "auto":
        host = min(host, int(spec))
        device = None if device is None else min(device, int(spec))
    capped = host < measured_host or (device is not None and device < measured_device)

    return Budget(
        device=device,
        host=host,
        disk=disk,
        unified=unified,
        device_name=device_name,
        headroom=config.headroom,
        disk_floor_bytes=floor,
        capped_by_setting=capped,
    )
