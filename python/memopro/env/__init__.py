"""Environment detection (architecture §3.1): what memory is really available, per pool.

Host facts (RAM, swap, container limit, disk) come from the Rust `hwinfo` module; device facts
(CUDA free memory, MPS recommended limit) from torch when it is installed (0013 V12). Without
torch, no device is listed and a note says why.

Host ``available_bytes`` is conservative (0035): memory obtainable without compressing or
swapping anything. macOS' own, larger estimate is kept as ``kernel_available_bytes``.
"""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path

from memopro.config import Config, get_config, spill_location

__all__ = ["Device", "Disk", "Env", "HostMemory", "detect"]


@dataclass(frozen=True)
class HostMemory:
    total_bytes: int
    available_bytes: int  # conservative
    kernel_available_bytes: int  # the OS estimate; reference only
    swap_total_bytes: int
    swap_free_bytes: int
    cgroup_limit_bytes: int | None  # container limit, if any (U7)
    cgroup_free_bytes: int | None
    usable_bytes: int  # available, capped by the container

    @property
    def swap_used_bytes(self) -> int:
        return max(0, self.swap_total_bytes - self.swap_free_bytes)


@dataclass(frozen=True)
class Disk:
    spill_dir: str  # where spill files would go (may not exist yet)
    measured_path: str  # nearest existing directory, whose file system was measured
    total_bytes: int
    available_bytes: int

    @property
    def free_fraction(self) -> float:
        return self.available_bytes / self.total_bytes if self.total_bytes else 0.0


@dataclass(frozen=True)
class Device:
    kind: str  # "cuda" | "mps" | "rocm" | "xpu"
    name: str
    total_bytes: int | None
    available_bytes: int | None  # what this process can still allocate on it
    limit_bytes: int | None  # e.g. MPS recommended_max_memory (K5)
    allocated_bytes: int | None  # already held by this process
    unified: bool  # shares physical memory with the host


@dataclass(frozen=True)
class Env:
    os: str
    os_version: str
    machine: str
    cpu_brand: str
    logical_cpus: int
    host: HostMemory
    disk: Disk
    devices: tuple[Device, ...] = ()
    notes: tuple[str, ...] = field(default=())


def _nearest_existing(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if candidate.exists():
            return candidate
    return Path(path.anchor or ".")


def _os_version() -> str:
    if platform.system() == "Darwin":
        return f"macOS {platform.mac_ver()[0]}"
    return platform.release()


MALLOC_CACHE_NOTE = (
    "macOS keeps CPU memory freed by hibernate or `del` in the allocator cache until memory "
    "pressure (0060); start Python with MallocLargeCache=0 to return it to other apps at once "
    "(`memopro run` does this for you)"
)


MPS_LOW_WATERMARK_VAR = "PYTORCH_MPS_LOW_WATERMARK_RATIO"
MPS_LOW_WATERMARK = "0.1"
MPS_HEAP_NOTE = (
    "PyTorch's MPS allocator reserves a whole 1 GiB heap for any 10-512 MiB allocation (a long "
    "prompt, full logits) while it sees no memory pressure (0080); set "
    f"{MPS_LOW_WATERMARK_VAR}={MPS_LOW_WATERMARK} before torch first uses MPS to allocate exact "
    "sizes once MPS memory in use exceeds 10% of the recommended maximum (about 1 GiB less for "
    "Qwen2.5-1.5B/3B int4 on an 8 GB M1, same speed and output; `memopro run` does this for you)"
)


def mps_heap_reserve_on() -> bool:
    """True on Apple silicon unless this process set PyTorch's MPS low watermark ratio itself
    (0080 W2). Reads the environment only: no torch import."""
    return (
        sys.platform == "darwin"
        and platform.machine() == "arm64"
        and MPS_LOW_WATERMARK_VAR not in os.environ
    )


def macos_malloc_cache_on() -> bool:
    """True on macOS unless this process started with ``MallocLargeCache=0`` (0061 F4)."""
    return sys.platform == "darwin" and os.environ.get("MallocLargeCache") != "0"


def detect(*, devices: bool = True, config: Config | None = None) -> Env:
    """Detect the current environment.

    ``devices=False`` skips torch entirely (fast, no GPU context is created).
    """
    from memopro import _core

    cfg = config if config is not None else get_config()
    host = HostMemory(**_core.hwinfo_memory())
    cpu = _core.hwinfo_cpu()
    target = spill_location(cfg)
    measured = _nearest_existing(target)
    raw_disk = _core.hwinfo_disk(measured)
    disk = Disk(str(target), str(measured), raw_disk["total_bytes"], raw_disk["available_bytes"])

    found: tuple[Device, ...] = ()
    notes: tuple[str, ...] = ()
    if devices:
        from memopro.env._torch import probe

        found, notes = probe()
    else:
        notes = ("device detection skipped (devices=False)",)
    return Env(
        os=platform.system(),
        os_version=_os_version(),
        machine=platform.machine(),
        cpu_brand=cpu["brand"],
        logical_cpus=cpu["logical_cpus"],
        host=host,
        disk=disk,
        devices=found,
        notes=notes,
    )
