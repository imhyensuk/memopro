"""``memopro.optimize``: fit a model you already have (architecture §4.1, 0052 E3).

Only run-time techniques apply to a loaded model, and only as far as needed:

- inference: CPU offload with accelerate (discrete GPUs only; exact). Nothing else: memopro
  never changes a model's precision (0328)
- training: activation checkpointing (exact); an 8-bit optimizer or LoRA are suggested only

The plan is checked before anything changes: if even the last allowed step does not fit,
`BudgetExceeded` is raised and the model is left untouched.
"""

from __future__ import annotations

from typing import Any

from memopro._errors import BudgetExceeded, InvalidArgument
from memopro._units import format_size
from memopro.access._common import setup
from memopro.access._info import model_info
from memopro.report import report

__all__ = ["optimize"]


def optimize(
    model: Any,
    goal: str = "infer",
    *,
    budget: Any = None,
    example: Any = None,
) -> Any:
    """Fit ``model`` to the budget in place and return it (see module docstring)."""
    if goal not in ("infer", "train"):
        raise InvalidArgument(f"goal must be 'infer' or 'train'; got {goal!r}")
    if not hasattr(model, "named_parameters"):
        raise InvalidArgument(f"optimize takes an nn.Module; got {type(model).__name__}")
    s = setup(
        budget=budget,
        device=_where(model),
        holding=(model,),  # the model is already in memory: counted once (E022 D8)
    )
    if goal == "infer":
        return _infer(model, s)
    return _train(model, s, example)


def _where(model: Any) -> str:
    p = next(model.parameters(), None)
    return p.device.type if p is not None else "cpu"


def _infer(model: Any, s: Any) -> Any:
    info = model_info(model)
    device = _where(model)
    budget = s.budget.device if device in ("cuda", "mps") else s.budget.host
    if budget is None:
        budget = s.budget.host
    runtime = info.weight_bytes(half=True) // 10
    steps: list[tuple[str, int]] = [("stored", info.weight_bytes())]
    if device == "cuda":
        steps.append(("offload.cpu", 0))
    needed = next((i for i, (_, size) in enumerate(steps) if size + runtime <= budget), None)
    if needed is None:
        need = steps[0][1] + runtime
        raise BudgetExceeded(
            f"{info.source}: needs {format_size(need)} as it is, the budget is "
            f"{format_size(budget)}; budget='{-(-need // 10**8) / 10:.1f}GB' would fit it"
        )
    model.eval()
    rep = report()
    for name, size in steps[1 : needed + 1]:
        if name == "offload.cpu":
            from accelerate import cpu_offload

            cpu_offload(model, execution_device=next(model.parameters()).device)
        rep.add(f"optimize.{name}", "applied", f"{info.source}: now about {format_size(size)}")
    if needed == 0:
        rep.add("optimize", "skipped", f"{info.source} already fits {format_size(budget)}")
    return model


def _train(model: Any, s: Any, example: Any) -> Any:
    from memopro.access._check import check

    info = model_info(model)
    rep = report()
    budget = s.budget.device if s.budget.device is not None else s.budget.host
    static = info.weight_bytes() * 4  # weights, gradients and two AdamW moments (fp32)
    enable = True
    if example is not None:
        result = check(model, goal="train", example=example)
        peak = result.training.get("peak")
        enable = peak is None or peak > budget
    if static > budget:
        rep.add(
            "optimize.suggest",
            "suggested",
            f"weights, gradients and AdamW state need about {format_size(static)} (budget "
            f"{format_size(budget)}): an 8-bit optimizer (numerics change) or LoRA (changes what "
            "is trained) would help; memopro does not switch them for you",
        )
    if enable and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        rep.add("optimize.checkpointing", "applied", f"{info.source}: activation checkpointing")
    elif enable:
        from memopro.access._train import checkpoint_blocks

        if checkpoint_blocks(model) is not None:
            rep.add("optimize.checkpointing", "applied", f"{info.source}: block checkpointing")
    return model
