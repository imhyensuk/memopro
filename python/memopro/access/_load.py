"""``memopro.load``: load a Hugging Face model so it fits the budget (architecture §4.1, 0052 E3).

Sizes come from metadata before anything is downloaded or allocated. The first candidate that
is allowed, available and fits is loaded with ``from_pretrained``; if loading fails, memopro
loads again from the original with the next candidate (load-time techniques cannot be undone
in place, §3.4). Every attempt and the reason for the choice go to ``report()``.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any

from memopro._errors import BudgetExceeded, InvalidArgument
from memopro._units import format_size
from memopro.access._common import setup
from memopro.access._info import ModelInfo, model_info
from memopro.orchestrator.apply import fail_open
from memopro.orchestrator.candidates import (
    Configuration,
    from_pretrained_kwargs,
    infer_candidates,
)
from memopro.report import report
from memopro.techniques.integrations.loading import LoadContext

__all__ = ["RESERVED", "TASKS", "LoadPlan", "load", "plan_load"]

# memopro chooses these; passing them means "load it yourself" (use from_pretrained directly)
RESERVED = (
    "device_map",
    "dtype",
    "torch_dtype",
    "quantization_config",
    "max_memory",
    "offload_folder",
    "load_in_8bit",
    "load_in_4bit",
)
TASKS = {
    "text-generation": "AutoModelForCausalLM",
    "causal-lm": "AutoModelForCausalLM",
    "fill-mask": "AutoModelForMaskedLM",
    "text-classification": "AutoModelForSequenceClassification",
    "token-classification": "AutoModelForTokenClassification",
    "question-answering": "AutoModelForQuestionAnswering",
    "text2text-generation": "AutoModelForSeq2SeqLM",
    "image-classification": "AutoModelForImageClassification",
    "feature-extraction": "AutoModel",
}
GENERATION_TOKENS = 2048  # default generation length budgeted for the cache


def _model_class(info: ModelInfo, task: str | None) -> Any:
    import transformers

    if task is not None:
        if task not in TASKS:
            raise InvalidArgument(f"unknown task {task!r}; known: {', '.join(TASKS)}")
        return getattr(transformers, TASKS[task])
    for arch in getattr(info.config, "architectures", None) or ():
        cls = getattr(transformers, arch, None)
        if cls is not None:
            return cls
    return transformers.AutoModel


def _runtime_bytes(info: ModelInfo, half_bytes: int) -> int:
    """What inference keeps on the device besides weights: a generation cache (batch 1) and
    activations, taken as 10% of the half-precision weights (an estimate; `check` measures)."""
    seq = min(info.max_positions or GENERATION_TOKENS, GENERATION_TOKENS)
    return info.kv_cache_bytes(1, seq) + half_bytes // 10


@dataclass(frozen=True)
class LoadPlan:
    info: ModelInfo
    ctx: LoadContext
    candidates: list[Configuration]
    quality: str
    prefer: str

    @property
    def chosen(self) -> Configuration | None:
        return next((c for c in self.candidates if c.usable), None)


def plan_load(model_id: Any, **options: Any) -> LoadPlan:
    """Candidates for loading ``model_id`` with the current settings, best first."""
    revision = options.pop("revision", None)
    allow, deny = options.pop("allow", None), tuple(options.pop("deny", ()) or ())
    s = setup(**options)  # may include device=
    info = model_info(model_id, revision=revision)
    ctx = LoadContext(
        device=s.device,
        half_dtype=s.half_dtype,
        device_budget=s.budget.device,
        host_budget=s.budget.host,
        disk_writes=s.config.disk_writes,
        unified=s.budget.unified,
        runtime_bytes=_runtime_bytes(info, info.weight_bytes(half=True)),
        offload_dir=s.offload_dir,
    )
    candidates = infer_candidates(
        info,
        ctx,
        quality=s.config.quality,
        prefer=s.config.prefer,
        disk_budget=s.budget.disk,
        allow=tuple(allow) if allow is not None else None,
        deny=deny,
    )
    return LoadPlan(info, ctx, candidates, s.config.quality, s.config.prefer)


def _table(candidates: list[Configuration]) -> str:
    rows = []
    for c in candidates:
        need = c.needs.device + c.needs.host + c.needs.disk
        state = "fits" if c.usable else (c.why or "does not fit")
        rows.append(f"  {c.name:<13} {format_size(need):>11}  {state}")
    return "\n".join(rows)


def load(
    model_id: Any,
    *,
    task: str | None = None,
    tokenizer: bool = False,
    budget: Any = None,
    quality: str | None = None,
    prefer: str | None = None,
    allow: tuple[str, ...] | None = None,
    deny: tuple[str, ...] = (),
    revision: str | None = None,
    device: str | None = None,
    **from_pretrained: Any,
) -> Any:
    """Load ``model_id`` so it fits the budget; returns the model, or ``(model, tokenizer)``.

    ``quality`` bounds automatic loss ("lossless" < "high" < "balanced" < "low"), ``prefer``
    orders the rest ("speed", "quality", "memory"), ``allow``/``deny`` filter technique names
    (``dtype.half``, ``quant.int8``, ``quant.int4``, ``offload.cpu``, ``offload.disk``).
    Other keyword arguments go to ``from_pretrained`` unchanged.
    """
    reserved = sorted(set(from_pretrained) & set(RESERVED))
    if reserved:
        raise InvalidArgument(
            f"memopro.load chooses {', '.join(reserved)} itself; steer it with quality, prefer, "
            "allow, deny or budget, or call from_pretrained directly"
        )
    plan = plan_load(
        model_id,
        budget=budget,
        quality=quality,
        prefer=prefer,
        allow=allow,
        deny=deny,
        revision=revision,
        device=device,
    )
    info, ctx, candidates = plan.info, plan.ctx, plan.candidates
    usable = [c for c in candidates if c.usable]
    if not usable:
        raise BudgetExceeded(
            f"no configuration of {info.source} fits the budget with quality "
            f"{plan.quality!r}:\n{_table(candidates)}\n"
            "Try a lower quality (quality='low'), allow disk offload (disk_writes='allow') or "
            "a larger budget."
        )
    cls = _model_class(info, task)
    rep = report()
    for cfg in usable:
        kwargs = from_pretrained_kwargs(cfg, ctx) | from_pretrained
        outcome = fail_open(
            f"load.{cfg.name}", cls.from_pretrained, model_id, revision=revision, **kwargs
        )
        if outcome.ok:
            model = outcome.value
            need = cfg.needs.device + cfg.needs.host + cfg.needs.disk
            rep.add(
                f"load.{cfg.name}",
                "applied",
                f"{info.source}: {cfg.describe()} ({cfg.quality.name.lower()}), estimated "
                f"{format_size(need)}, weights {format_size(_footprint(model))} on {ctx.device}",
            )
            if tokenizer:
                from transformers import AutoTokenizer

                return model, AutoTokenizer.from_pretrained(model_id, revision=revision)
            return model
        # the exception's traceback holds the partly loaded model: drop it before the next try,
        # or a failed load keeps its memory while the next configuration loads (0054)
        del outcome
        _release()
    raise BudgetExceeded(
        f"every configuration that fits failed to load {info.source}; see memopro.report()"
    )


def _footprint(model: Any) -> int:
    try:
        return int(model.get_memory_footprint())
    except Exception:  # noqa: BLE001 - informational
        return sum(p.numel() * p.element_size() for p in model.parameters())


def _release() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
