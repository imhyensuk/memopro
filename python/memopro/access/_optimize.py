"""``memopro.optimize``: fit a model you already have (architecture §4.1, 0052 E3).

Only run-time techniques apply to a loaded model, and only as far as needed:

- inference: half precision (within ``quality``), then torchao int8 weight-only quantization
  (within ``quality``), then CPU offload with accelerate (discrete GPUs only; exact)
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
from memopro.orchestrator.candidates import QUALITY_LIMIT
from memopro.report import report
from memopro.techniques.base import QualityGrade
from memopro.techniques.integrations import loading

__all__ = ["optimize"]


def optimize(
    model: Any,
    goal: str = "infer",
    *,
    budget: Any = None,
    quality: str | None = None,
    prefer: str | None = None,
    example: Any = None,
    budget_basis: str | None = None,
    fallback: str | None = None,
) -> Any:
    """Fit ``model`` to the budget in place and return it (see module docstring)."""
    if goal not in ("infer", "train"):
        raise InvalidArgument(f"goal must be 'infer' or 'train'; got {goal!r}")
    if not hasattr(model, "named_parameters"):
        raise InvalidArgument(f"optimize takes an nn.Module; got {type(model).__name__}")
    s = setup(
        budget=budget,
        quality=quality,
        prefer=prefer,
        device=_where(model),
        budget_basis=budget_basis,
        fallback=fallback,
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
    limit = QUALITY_LIMIT[s.config.quality]
    device = _where(model)
    budget = s.budget.device if device in ("cuda", "mps") else s.budget.host
    if budget is None:
        budget = s.budget.host
    runtime = info.weight_bytes(half=True) // 10
    steps: list[tuple[str, int]] = [("stored", info.weight_bytes())]
    if limit.value >= QualityGrade.NEAR_LOSSLESS.value and info.stored_dtype == "float32":
        steps.append(("dtype.half", info.weight_bytes(half=True)))
    if limit.value >= QualityGrade.SMALL_LOSS.value and not loading._torchao_works(device, 8):
        steps.append(("quant.int8", info.weight_bytes(bits=8)))
    if device == "cuda":
        steps.append(("offload.cpu", 0))
    needed = next((i for i, (_, size) in enumerate(steps) if size + runtime <= budget), None)
    if needed is None:
        if s.config.fallback == "stored":
            detail = (
                f"{info.source}: no allowed step fits {format_size(budget)}; fallback='stored': "
                f"left as it is ({format_size(steps[0][1] + runtime)})"
            )
            import warnings

            warnings.warn(f"memopro.optimize: {detail}", UserWarning, stacklevel=3)
            report().add("optimize", "skipped", detail)
            return model
        raise BudgetExceeded(
            f"{info.source}: no allowed step fits {format_size(budget)} (quality "
            f"{s.config.quality!r}): "
            + ", ".join(f"{n} {format_size(b + runtime)}" for n, b in steps)
            + _optimize_suggestions(info, device, runtime, budget, s)
        )
    model.eval()
    rep = report()
    for name, size in steps[1 : needed + 1]:
        if name == "dtype.half":
            model.to(s.half_dtype)
        elif name == "quant.int8":
            from torchao.quantization import Int8WeightOnlyConfig, quantize_

            if s.half_dtype is not None:
                model.to(s.half_dtype)
            quantize_(model, Int8WeightOnlyConfig())
        elif name == "offload.cpu":
            from accelerate import cpu_offload

            cpu_offload(model, execution_device=next(model.parameters()).device)
        rep.add(f"optimize.{name}", "applied", f"{info.source}: now about {format_size(size)}")
    if needed == 0:
        rep.add("optimize", "skipped", f"{info.source} already fits {format_size(budget)}")
    return model


def _optimize_suggestions(info: Any, device: str, runtime: int, budget: int, s: Any) -> str:
    """Settings under which a step would fit (0064 D-a), from the same step sizes."""
    from memopro.config import QUALITIES

    grades = [("stored", info.weight_bytes(), "lossless")]
    if info.stored_dtype == "float32":
        grades.append(("dtype.half", info.weight_bytes(half=True), "high"))
    if not loading._torchao_works(device, 8):
        grades.append(("quant.int8", info.weight_bytes(bits=8), "balanced"))
    current = QUALITIES.index(s.config.quality)
    lines = []
    for name, size, quality in grades:
        need = size + runtime
        change = {}
        if QUALITIES.index(quality) > current:
            change["quality"] = quality
        if need > budget:
            change["budget"] = f"{-(-need // 10**8) / 10:.1f}GB!"
        if change:
            call = ", ".join(f"{k}={v!r}" for k, v in change.items())
            lines.append(f"  {call:<34} -> {name}, {format_size(need)}")
    if s.config.fallback == "none":
        lines.append("  fallback='stored'                  -> warn and leave the model as it is")
    return "\nSettings that would fit:\n" + "\n".join(lines) if lines else ""


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
