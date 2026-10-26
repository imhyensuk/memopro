"""Per-pool budget vector (architecture §3.2): {device, host, disk}.

The budget is the ceiling the user sets, from the machine's memory (0325 rule 2, 0327):

- host:   ``"auto"`` is 80% of the machine's memory (the container's limit when there is one);
          a size is that size; a fraction is that share of the memory.
- device: the same for a CUDA GPU's memory; on unified memory (MPS) it is the host budget (one
          pool, K5), at most the GPU's own limit.
- disk:   free space above the ``min_free_disk_fraction`` floor (0032 H4). Whether memopro may
          write there at all is a policy question (``disk_writes``), not part of the budget.

A ceiling above what is free right now is kept as given, with a note: other programs hold the
rest, and the operating system may swap to make room.
"""

from __future__ import annotations

from dataclasses import dataclass

from memopro.config import AUTO_FRACTION, Config, PoolBudget, PoolValue
from memopro.env import Env, HostMemory

__all__ = ["Budget", "Shortfall", "compute_budget", "describe_setting", "resolve_pool"]

_BUDGETED_KINDS = ("cuda", "mps")


@dataclass(frozen=True)
class Shortfall:
    """A pool whose budget is below a minimum the caller needs."""

    pool: str
    required: int
    available: int


@dataclass(frozen=True)
class Budget:
    device: int | None  # None: no supported accelerator
    host: int
    disk: int
    unified: bool = False  # device and host draw from the same physical memory
    device_name: str | None = None
    disk_floor_bytes: int = 0
    capped_by_setting: bool = False
    forced: tuple[str, ...] = ()  # pools the user set to a value; γ leaves them alone
    shortfalls: tuple[Shortfall, ...] = ()
    notes: tuple[str, ...] = ()


def _describe_value(value: PoolValue) -> str:
    from memopro._units import format_size

    if value == "auto":
        return f"auto ({AUTO_FRACTION:.0%} of the memory)"
    if isinstance(value, float):
        return f"{value:.0%} of the memory"
    return format_size(int(value))


def describe_setting(spec: object) -> str:
    """The budget setting as people write it."""
    if isinstance(spec, PoolBudget):
        pools = (("device", spec.device), ("host", spec.host), ("disk", spec.disk))
        return "; ".join(f"{k} {_describe_value(v)}" for k, v in pools if v is not None)
    return _describe_value(spec)  # type: ignore[arg-type]


def resolve_pool(value: PoolValue | None, memory: int) -> int:
    """One pool's ceiling from its setting and the pool's memory."""
    if value is None or value == "auto":
        return int(memory * AUTO_FRACTION)
    if isinstance(value, float):
        return int(memory * value)
    return int(value)


def host_memory(host: HostMemory) -> int:
    """The machine's memory, or the container's limit when there is one."""
    return host.cgroup_limit_bytes or host.total_bytes


def compute_budget(
    env: Env, config: Config, *, resident_device: int = 0, resident_host: int = 0
) -> Budget:
    """Budgets per pool. ``resident_*`` are bytes the caller already holds in that pool (a
    loaded model): they are part of the ceiling, and only used to tell whether the ceiling is
    above what the process can get now."""
    from memopro._units import format_size

    spec = config.budget
    if isinstance(spec, PoolBudget):
        specs = {"device": spec.device, "host": spec.host, "disk": spec.disk}
    else:
        specs = {"device": spec, "host": spec, "disk": None}
    notes: list[str] = []

    memory = host_memory(env.host)
    host = resolve_pool(specs["host"], memory)
    room = env.host.usable_bytes + resident_host
    if host > room:
        notes.append(
            f"the host ceiling {format_size(host)} is above the {format_size(room)} free now: "
            "other programs hold the rest, and the system may swap to make room"
        )

    device = None
    device_name = None
    unified = False
    primary = next((d for d in env.devices if d.kind in _BUDGETED_KINDS), None)
    if primary is not None:
        if primary.unified:  # one pool with the host (K5), at most the GPU's own limit
            device = resolve_pool(specs["device"], memory)
            device = min(device, host)
            if primary.limit_bytes:
                device = min(device, primary.limit_bytes)
            unified = True
        elif primary.total_bytes:
            device = resolve_pool(specs["device"], primary.total_bytes)
            free = (primary.available_bytes or 0) + resident_device
            if primary.available_bytes is not None and device > free:
                notes.append(
                    f"the device ceiling {format_size(device)} is above the "
                    f"{format_size(free)} free on {primary.name} now"
                )
        device_name = primary.name

    floor = int(env.disk.total_bytes * config.min_free_disk_fraction)
    disk_free = max(0, env.disk.available_bytes - floor)
    disk = (
        disk_free
        if specs["disk"] is None
        else min(disk_free, resolve_pool(specs["disk"], disk_free))
    )

    forced = tuple(p for p in ("device", "host") if specs[p] not in (None, "auto"))
    return Budget(
        device=device,
        host=host,
        disk=disk,
        unified=unified,
        device_name=device_name,
        disk_floor_bytes=floor,
        capped_by_setting=bool(forced) or specs["disk"] is not None,
        forced=forced,
        notes=tuple(notes),
    )
