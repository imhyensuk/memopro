"""Shared by the access-layer entry points: per-call settings, the device and the budget."""

from __future__ import annotations

import dataclasses
from typing import Any

from memopro.config import Config, _validate, get_config, spill_location
from memopro.env import Env, detect
from memopro.orchestrator.budget import Budget, compute_budget

__all__ = ["Setup", "resident_bytes", "setup"]


@dataclasses.dataclass(frozen=True)
class Setup:
    config: Config
    env: Env
    budget: Budget
    device: str  # "cuda" | "mps" | "cpu"
    half_dtype: Any

    @property
    def offload_dir(self) -> str:
        return str(spill_location(self.config) / "offload")


def settings(**options: Any) -> Config:
    """The effective configuration with per-call options (validated like `configure()`)."""
    cfg = get_config()
    changes = {k: _validate(k, v) for k, v in options.items() if v is not None}
    return dataclasses.replace(cfg, **changes)


def setup(*, device: str | None = None, holding: tuple[Any, ...] = (), **options: Any) -> Setup:
    """Settings, environment and budget; ``device`` ("cuda", "mps", "cpu") overrides detection.

    ``holding``: the caller's model and optimizer, already in memory. Their tensors count in the
    budget of the pool they are in (`resident_bytes`, E022 D8)."""
    import torch

    from memopro._errors import InvalidArgument
    from memopro.env._torch import mps_usable

    if device not in (None, "cuda", "mps", "cpu"):
        raise InvalidArgument(f"device must be 'cuda', 'mps' or 'cpu'; got {device!r}")
    cfg = settings(**options)
    env = detect(config=cfg)
    kinds = [d.kind for d in env.devices]
    budgeted = next((k for k in kinds if k in ("cuda", "mps")), None)
    budget = compute_budget(
        env,
        cfg,
        resident_device=resident_bytes(budgeted, *holding) if budgeted else 0,
        resident_host=resident_bytes("cpu", *holding),
    )
    _require_minimums(budget, cfg)
    budget = _under_pressure(budget)
    if device == "cpu":
        kinds = []
    elif device is not None and device not in kinds:
        raise InvalidArgument(f"device {device!r} is not available here (found: {kinds or 'cpu'})")
    if device in (None, "cuda") and "cuda" in kinds and torch.cuda.is_available():
        device = "cuda"
        major, _ = torch.cuda.get_device_capability(0)
        half = torch.bfloat16 if major >= 8 else torch.float16  # T4 (7.5): fp16 is native
    elif device in (None, "mps") and "mps" in kinds and mps_usable():
        device, half = "mps", torch.float16
    else:
        device, half = "cpu", torch.bfloat16
    if device == "cpu":
        budget = dataclasses.replace(budget, device=None, unified=False)
    return Setup(cfg, env, budget, device, half)


def resident_bytes(kind: str, *holding: Any) -> int:
    """Bytes of the parameters, buffers and optimizer state of ``holding`` (modules and
    optimizers) that live on device type ``kind`` ("cuda", "mps" or "cpu"), each tensor once."""
    seen: set[int] = set()
    total = 0

    def add(t: Any) -> None:
        nonlocal total
        if hasattr(t, "untyped_storage") and t.device.type == kind and id(t) not in seen:
            seen.add(id(t))
            total += t.numel() * t.element_size()

    for obj in holding:
        if hasattr(obj, "parameters"):
            for t in obj.parameters():
                add(t)
            for t in obj.buffers():
                add(t)
        for state in getattr(obj, "state", {}).values() if hasattr(obj, "param_groups") else ():
            for t in state.values():
                add(t)
    return total


def _require_minimums(budget: Budget, cfg: Config) -> None:
    """Stop when a pool is below the minimum the budget setting asks for (0059 D1)."""
    if not budget.shortfalls:
        return
    from memopro._errors import BudgetExceeded
    from memopro._units import format_size
    from memopro.orchestrator.budget import describe_setting

    short = "; ".join(
        f"{s.pool} has {format_size(s.available)}, needs at least {format_size(s.required)}"
        for s in budget.shortfalls
    )
    raise BudgetExceeded(
        f"the budget setting ({describe_setting(cfg.budget)}) asks for more than is available: "
        f"{short}. Free memory, lower the minimum, or choose budget_basis='os' to count memory "
        "the OS can reclaim by compressing or swapping"
    )


def _under_pressure(budget: Budget) -> Budget:
    """Shrink the budget while the OS reports memory pressure (γ, 0052 E6); forced pools keep
    the exact size the user asked for (0059 D1)."""
    try:
        from memopro.elastic import budget_factor
    except ImportError:
        return budget
    factor = budget_factor()
    if factor >= 1.0:
        return budget
    host = budget.host if "host" in budget.forced else int(budget.host * factor)
    device = budget.device
    if device is not None and "device" not in budget.forced:
        device = int(device * factor)
    return dataclasses.replace(budget, host=host, device=device)
