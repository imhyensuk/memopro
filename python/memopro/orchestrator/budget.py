"""Per-pool budget vector (architecture §3.2): {device, host, disk}.

- host:   usable host memory minus headroom. What "usable" starts from is ``budget_basis``
          (0059 D3): conservative available (default, 0035), the OS estimate, or the total; the
          container limit always applies
- device: CUDA free memory minus headroom; on unified memory (MPS) the smaller of the MPS limit
          left and usable host memory, minus headroom, and it shares one pool with ``host`` (K5)
- disk:   free space above the ``min_free_disk_fraction`` floor (0032 H4). Whether memopro may
          write there at all is a policy question (``disk_writes``), not part of the budget.
- headroom is a fraction of the measured memory or a size (0059 D4).
- the ``budget`` setting then applies per pool (0059 D1/D2, `resolve_pool`): a cap, a fraction,
  memory to leave free, a range whose minimum stops memopro when unmet, or an exact forced size.
  Only a forced size and a non-conservative basis can go above what is measured.
"""

from __future__ import annotations

from dataclasses import dataclass

from memopro.config import Config, Limit, PoolBudget, PoolValue
from memopro.env import Env, HostMemory

__all__ = ["Budget", "Shortfall", "compute_budget", "describe_setting", "resolve_pool"]

_BUDGETED_KINDS = ("cuda", "mps")


@dataclass(frozen=True)
class Shortfall:
    """A pool whose budget is below the minimum the setting asks for (0059 D1)."""

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
    headroom: float = 0.0  # a fraction, or bytes when an int (0059 D4)
    disk_floor_bytes: int = 0
    capped_by_setting: bool = False
    basis: str = "conservative"
    forced: tuple[str, ...] = ()  # pools set to an exact size ("6GB!"); γ leaves them alone
    shortfalls: tuple[Shortfall, ...] = ()
    notes: tuple[str, ...] = ()


def _describe_value(value: PoolValue) -> str:
    from memopro._units import format_size

    if value == "auto":
        return "auto"
    if isinstance(value, float):
        return f"{value:.0%} of the measured budget"
    if isinstance(value, int):
        return format_size(value)
    assert isinstance(value, Limit)
    parts = []
    if value.force:
        parts.append(f"exactly {format_size(int(value.use))}")
    elif value.use != "auto":
        parts.append(_describe_value(value.use))
    if value.reserve:
        parts.append(f"leaving {format_size(value.reserve)} free")
    if value.minimum is not None:
        parts.append(f"at least {format_size(value.minimum)}")
    if value.maximum is not None:
        parts.append(f"at most {format_size(value.maximum)}")
    return ", ".join(parts) or "auto"


def describe_setting(spec: object) -> str:
    """The budget setting as people write it."""
    if isinstance(spec, PoolBudget):
        pools = (("device", spec.device), ("host", spec.host), ("disk", spec.disk))
        return "; ".join(f"{k} {_describe_value(v)}" for k, v in pools if v is not None)
    return _describe_value(spec)  # type: ignore[arg-type]


def resolve_pool(value: PoolValue | None, measured: int) -> tuple[int, bool, int | None]:
    """Apply one pool's setting to its measured budget: (budget, forced, unmet minimum or None)."""
    if value is None or value == "auto":
        return measured, False, None
    if isinstance(value, float):
        return int(measured * value), False, None
    if isinstance(value, int):
        return min(measured, value), False, None
    forced = value.force
    if forced:
        amount = int(value.use)
    else:
        amount = max(0, resolve_pool(value.use, measured)[0] - value.reserve)
    if value.maximum is not None:
        amount = min(amount, value.maximum)
    unmet = value.minimum if value.minimum is not None and amount < value.minimum else None
    return amount, forced, unmet


def measured_host(host: HostMemory, basis: str) -> int:
    """Host memory the budget starts from, capped by the container (0059 D3)."""
    if basis == "os":
        value = host.kernel_available_bytes
        if host.cgroup_free_bytes is not None:
            value = min(value, host.cgroup_free_bytes)
        return max(value, host.usable_bytes)
    if basis == "total":
        return host.cgroup_limit_bytes or host.total_bytes
    return host.usable_bytes


def _after_headroom(measured: int, headroom: float) -> int:
    if isinstance(headroom, int) and not isinstance(headroom, bool):
        return max(0, measured - headroom)
    return int(measured * (1.0 - headroom))


def compute_budget(env: Env, config: Config) -> Budget:
    from memopro._units import format_size

    usable = measured_host(env.host, config.budget_basis)
    host = _after_headroom(usable, config.headroom)

    device = None
    device_name = None
    unified = False
    primary = next((d for d in env.devices if d.kind in _BUDGETED_KINDS), None)
    if primary is not None and primary.available_bytes is not None:
        free = primary.available_bytes
        if primary.unified:
            free = min(free, usable)
            unified = True
        device = _after_headroom(free, config.headroom)
        device_name = primary.name

    floor = int(env.disk.total_bytes * config.min_free_disk_fraction)
    disk = max(0, env.disk.available_bytes - floor)

    spec = config.budget
    if isinstance(spec, PoolBudget):
        specs = {"device": spec.device, "host": spec.host, "disk": spec.disk}
    else:
        specs = {"device": spec, "host": spec, "disk": None}
    measured = {"device": device, "host": host, "disk": disk}
    result: dict[str, int | None] = {}
    forced, shortfalls, notes = [], [], []
    for pool, value in specs.items():
        amount = measured[pool]
        if amount is None:  # no accelerator: a device setting has nothing to apply to
            result[pool] = None
            continue
        budget, is_forced, unmet = resolve_pool(value, amount)
        result[pool] = budget
        if is_forced:
            forced.append(pool)
            if budget > amount:
                notes.append(
                    f"{pool} budget forced to {format_size(budget)}, above the measured "
                    f"{format_size(amount)}: expect swapping or out-of-memory errors"
                )
        if unmet is not None:
            shortfalls.append(Shortfall(pool, unmet, budget))
            notes.append(
                f"{pool} budget {format_size(budget)} is below the required minimum "
                f"{format_size(unmet)}: load, check, optimize and train_session will stop"
            )
    if config.budget_basis == "os":
        notes.append(
            "budget basis 'os': includes memory other apps are using; the OS may compress or "
            "swap to make room"
        )
    elif config.budget_basis == "total":
        notes.append(
            "budget basis 'total': counts all physical memory; other apps will be pushed to swap"
        )
    capped = any(
        result[p] is not None and measured[p] is not None and result[p] < measured[p]  # type: ignore[operator]
        for p in ("device", "host", "disk")
        if specs[p] is not None
    )

    return Budget(
        device=result["device"],
        host=int(result["host"]),  # type: ignore[arg-type]
        disk=int(result["disk"]),  # type: ignore[arg-type]
        unified=unified,
        device_name=device_name,
        headroom=config.headroom,
        disk_floor_bytes=floor,
        capped_by_setting=capped,
        basis=config.budget_basis,
        forced=tuple(forced),
        shortfalls=tuple(shortfalls),
        notes=tuple(notes),
    )
