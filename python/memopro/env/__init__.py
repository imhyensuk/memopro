"""Environment detection (architecture §3.1): what memory is really available, per pool.

Host facts (RAM, swap, cgroup limit, disk) come from the Rust `hwinfo` module; device facts
(CUDA free memory, MPS recommended limit) from torch when it is installed (0013 V12). Without
torch, device fields are None and the report says "torch not installed".

Status: skeleton. Next step: A1a hwinfo -> N1a doctor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from memopro._errors import NotYetImplemented

__all__ = ["Device", "Disk", "Env", "HostMemory", "detect"]


@dataclass(frozen=True)
class HostMemory:
    total_bytes: int
    available_bytes: int
    swap_total_bytes: int
    swap_free_bytes: int
    cgroup_limit_bytes: int | None  # container limit, if any (U7)


@dataclass(frozen=True)
class Disk:
    path: str
    total_bytes: int
    available_bytes: int


@dataclass(frozen=True)
class Device:
    kind: str  # "cuda" | "mps" | "cpu"
    name: str
    total_bytes: int | None
    available_bytes: int | None
    limit_bytes: int | None  # e.g. MPS recommended_max_memory (K5)
    unified: bool  # shares physical memory with the host


@dataclass(frozen=True)
class Env:
    os: str
    machine: str
    host: HostMemory
    disk: Disk
    devices: tuple[Device, ...] = ()
    notes: tuple[str, ...] = field(default=())


def detect() -> Env:
    """Detect the current environment."""
    raise NotYetImplemented("memopro.env.detect", "v0.1 (A1a hwinfo)", "docs/research/0032")
