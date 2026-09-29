"""Shared by the access-layer entry points: per-call settings, the device and the budget."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
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
    held_device = resident_bytes(budgeted, *holding) if budgeted else 0
    held_host = resident_bytes("cpu", *holding)
    budget = compute_budget(env, cfg, resident_device=held_device, resident_host=held_host)
    _require_minimums(budget, cfg)
    budget = _under_pressure(budget, held_device, held_host)
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
    optimizers) on device type ``kind`` ("cuda" means the first GPU, the one budgets are for;
    "mps" or "cpu"), each memory block once. Quantized weights count their scales too
    (bitsandbytes ``quant_state``, int8 ``SCB``)."""
    seen: set[int] = set()
    total = 0

    def add(t: Any) -> None:
        nonlocal total
        if not hasattr(t, "untyped_storage") or t.device.type != kind:
            return
        if kind == "cuda" and t.device.index not in (None, 0):
            return
        ptr = t.data_ptr()
        if ptr and ptr not in seen:  # tied weights, and int8 CB sharing the weight's memory
            seen.add(ptr)
            total += t.numel() * t.element_size()

    for obj in holding:
        if hasattr(obj, "parameters"):
            for t in obj.parameters():
                add(t)
                for extra in _quantization_tensors(t):
                    add(extra)
            for t in obj.buffers():
                add(t)
        if hasattr(obj, "param_groups"):
            for state in getattr(obj, "state", {}).values():
                for t in state.values():
                    add(t)
    return total


def _quantization_tensors(param: Any) -> Iterator[Any]:
    """Tensors a quantized parameter keeps beside its data (bitsandbytes)."""
    state = getattr(param, "quant_state", None)
    while state is not None:  # nf4/fp4: absmax, code, offset; double quantization nests a state
        for name in ("absmax", "code", "offset"):
            if (t := getattr(state, name, None)) is not None:
                yield t
        state = getattr(state, "state2", None)
    for name in ("SCB", "CB"):  # int8 row scales (CB usually is the weight itself)
        if (t := getattr(param, name, None)) is not None:
            yield t


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


def _under_pressure(budget: Budget, held_device: int = 0, held_host: int = 0) -> Budget:
    """Shrink the budget while the OS reports memory pressure (γ, 0052 E6); forced pools keep
    the exact size the user asked for (0059 D1). Only what memopro may still take shrinks: the
    memory the caller's model already holds is not given back by shrinking (E022 D8)."""
    try:
        from memopro.elastic import budget_factor
    except ImportError:
        return budget
    factor = budget_factor()
    if factor >= 1.0:
        return budget

    def shrink(value: int, held: int) -> int:
        held = min(held, value)
        return held + int((value - held) * factor)

    host = budget.host if "host" in budget.forced else shrink(budget.host, held_host)
    device = budget.device
    if device is not None and "device" not in budget.forced:
        device = shrink(device, held_device)
    return dataclasses.replace(budget, host=host, device=device)
