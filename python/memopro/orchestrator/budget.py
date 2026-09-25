"""Per-pool budget vector (architecture §3.2): {device, host, disk}.

- host:   usable host memory (conservative available, capped by the container) minus headroom
- device: CUDA free memory minus headroom; on unified memory (MPS) the smaller of the MPS limit
          left and usable host memory, minus headroom, and it shares one pool with ``host`` (K5)
- disk:   free space above the ``min_free_disk_fraction`` floor (0032 H4). Whether memopro may
          write there at all is a policy question (``disk_writes``), not part of the budget.
- an explicit ``budget`` setting caps device and host.
"""

from __future__ import annotations

from dataclasses import dataclass

from memopro.config import Config
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

    capped = False
    if config.budget != "auto":
        cap = int(config.budget)
        capped = host > cap or (device is not None and device > cap)
        host = min(host, cap)
        device = None if device is None else min(device, cap)

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
