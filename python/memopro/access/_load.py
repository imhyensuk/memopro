"""``memopro.load``: load a Hugging Face model so it fits the budget (architecture §4.1, 0052 E3).

Sizes come from metadata before anything is downloaded or allocated. The first candidate that
is allowed, available and fits is loaded with ``from_pretrained``; if loading fails, memopro
loads again from the original with the next candidate (load-time techniques cannot be undone
in place, §3.4). Every attempt and the reason for the choice go to ``report()``.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
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
    user_original,
)
from memopro.report import report
from memopro.techniques.integrations.loading import LoadContext

__all__ = ["ORIGINAL", "RESERVED", "TASKS", "LoadPlan", "load", "plan_load"]

# memopro chooses where the weights go; passing these means "place it yourself"
RESERVED = ("device_map", "max_memory", "offload_folder")
# these define the user's original model (0328): loaded exactly as given, never changed
ORIGINAL = ("dtype", "torch_dtype", "quantization_config", "load_in_8bit", "load_in_4bit")
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
    setup: Any = None  # the Setup it was planned with (settings, environment, budget)
    options: dict[str, Any] = field(default_factory=dict)  # what plan_load was given

    @property
    def chosen(self) -> Configuration | None:
        return next((c for c in self.candidates if c.usable), None)


def plan_load(model_id: Any, **options: Any) -> LoadPlan:
    """Candidates for loading ``model_id`` with the current settings, best first."""
    info = model_info(model_id, revision=options.get("revision"))
    return plan_from_info(info, **options)


def plan_from_info(info: ModelInfo, **options: Any) -> LoadPlan:
    """`plan_load` for a model already described (sizes from metadata; nothing is loaded)."""
    given = dict(options)
    options.pop("revision", None)
    allow, deny = options.pop("allow", None), tuple(options.pop("deny", ()) or ())
    s = setup(**options)  # may include device=
    ctx = LoadContext(
        device=s.device,
        half_dtype=s.half_dtype,
        device_budget=s.budget.device,
        host_budget=s.budget.host,
        unified=s.budget.unified,
        runtime_bytes=_runtime_bytes(info, info.weight_bytes(half=True)),
    )
    candidates = infer_candidates(
        info, ctx, allow=tuple(allow) if allow is not None else None, deny=deny
    )
    return LoadPlan(info, ctx, candidates, s, given)


def _table(candidates: list[Configuration]) -> str:
    rows = []
    for c in candidates:
        need = c.needs.device + c.needs.host
        state = "fits" if c.usable else (c.why or "does not fit")
        rows.append(f"  {c.name:<13} {format_size(need):>11}  {state}")
    return "\n".join(rows)


def load(
    model_id: Any,
    *,
    task: str | None = None,
    tokenizer: bool = False,
    budget: Any = None,
    allow: tuple[str, ...] | None = None,
    deny: tuple[str, ...] = (),
    revision: str | None = None,
    device: str | None = None,
    fallback: str | None = None,
    **from_pretrained: Any,
) -> Any:
    """Load ``model_id`` within the budget, exactly as it is; returns the model, or
    ``(model, tokenizer)``.

    The stored weights go on the device, or between the device and host RAM (``offload.cpu``),
    whichever fits first; when neither does they stay in their files and are streamed
    (``fallback="stream"``, the default; ``"none"`` stops instead). Nothing is converted.
    ``dtype``/``quantization_config`` (or ``load_in_8bit``/``load_in_4bit``) define *your*
    original (a half-precision or quantized model): it is loaded as given, so its results are
    exactly what ``from_pretrained`` with the same arguments gives (0328). Other keyword
    arguments go to ``from_pretrained`` unchanged.
    """
    from memopro.config import _REMOVED

    removed = sorted(set(from_pretrained) & set(_REMOVED))
    if removed:
        key = removed[0]
        raise InvalidArgument(f"memopro.load's {key!r} was removed in 0.2.0: {_REMOVED[key]}")
    reserved = sorted(set(from_pretrained) & set(RESERVED))
    if reserved:
        raise InvalidArgument(
            f"memopro.load places the weights itself ({', '.join(reserved)}); steer it with "
            "budget, allow or deny, or call from_pretrained directly"
        )
    given = {k: from_pretrained.pop(k) for k in ORIGINAL if k in from_pretrained}
    plan = plan_load(
        model_id,
        budget=budget,
        allow=allow,
        deny=deny,
        revision=revision,
        device=device,
        fallback=fallback,
    )
    info, ctx = plan.info, plan.ctx
    if given:
        yours = user_original(info, ctx, given)
        if not yours.fits:
            error = BudgetExceeded(
                f"{info.source} as you defined it ({yours.describe()}) needs about "
                f"{format_size(yours.needs.device + yours.needs.host)}: {yours.why}. Give a "
                "larger budget, or load the stored model without dtype/quantization_config "
                "(it is streamed when it does not fit)"
            )
            error.suggestions = []
            raise error
        usable = [yours]
    else:
        usable = [c for c in plan.candidates if c.usable]
    if not usable:
        if plan.setup.config.fallback == "stream" and not from_pretrained:
            # lossless streaming when nothing fits (0314); a budget too small even for the
            # runtime gets the usual explanation below
            try:
                return _load_streamed(plan, model_id, revision, task, tokenizer, from_pretrained)
            except InvalidArgument as e:
                if "too small" not in str(e) and "needs a host budget" not in str(e):
                    raise
        raise _nothing_fits(plan, f"{info.source} does not fit the budget as stored")
    cls = _model_class(info, task)
    # a hub model not on this disk is downloaded once with a notice and a free-disk check, one
    # layout of its weights (0312); transformers then reads the snapshot
    from memopro.rt.torch import _snapshot

    model_id, revision = _snapshot(model_id, revision)
    rep = report()
    for cfg in usable:
        kwargs = from_pretrained_kwargs(cfg, ctx) | from_pretrained
        outcome = fail_open(f"load.{cfg.name}", _attempt, cls, model_id, revision, kwargs, None)
        if outcome.ok:
            model = outcome.value
            need = cfg.needs.device + cfg.needs.host
            what = "as you defined it" if cfg.name == "yours" else "exactly as stored"
            detail = (
                f"{info.source}: {cfg.describe()} ({what}), estimated {format_size(need)}, "
                f"weights {format_size(_footprint(model))} on {ctx.device}"
            )
            rep.add(f"load.{cfg.name}", "applied", detail)
            if tokenizer:
                from transformers import AutoTokenizer

                return model, AutoTokenizer.from_pretrained(model_id, revision=revision)
            return model
        # the exception's traceback holds the partly loaded model: drop it before the next try,
        # or a failed load keeps its memory while the next configuration loads (0054)
        del outcome
        _release()
    raise _nothing_fits(
        plan, f"every configuration that fits failed to load {info.source}; see memopro.report()"
    )


def _load_streamed(
    plan: LoadPlan,
    model_id: Any,
    revision: Any,
    task: str | None,
    tokenizer: bool,
    from_pretrained: dict[str, Any],
) -> Any:
    """``fallback="stream"`` (0124): nothing fits, so the stored weights stay in their files and
    are streamed through `memopro.rt` on the CPU within the host budget (lossless, slower)."""
    import warnings

    from memopro.rt.torch import stream_model

    if from_pretrained:
        raise InvalidArgument(
            "fallback='stream' builds the model from its files itself; from_pretrained "
            f"arguments ({', '.join(sorted(from_pretrained))}) cannot be applied"
        )
    budget = plan.ctx.host_budget
    if not budget:
        raise InvalidArgument("fallback='stream' needs a host budget; none is available")
    model = stream_model(
        model_id, budget=int(budget), revision=revision, model_class=_model_class(plan.info, task)
    )
    detail = (
        f"{plan.info.source}: nothing fits the budget; fallback='stream': the stored weights stay "
        f"in their files and are streamed through memopro.rt on the CPU within "
        f"{format_size(int(budget))} (lossless, slower than running from memory)"
    )
    warnings.warn(f"memopro.load: {detail}", UserWarning, stacklevel=3)
    report().add("load.stream", "applied", detail)
    if tokenizer:
        from transformers import AutoTokenizer

        return model, AutoTokenizer.from_pretrained(model_id, revision=revision)
    return model


def _attempt(cls: Any, model_id: Any, revision: Any, kwargs: dict[str, Any], post: Any) -> Any:
    model = cls.from_pretrained(model_id, revision=revision, **kwargs)
    return post(model) if post is not None else model


def _nothing_fits(plan: LoadPlan, headline: str) -> BudgetExceeded:
    """`BudgetExceeded` with the candidate table and settings checked to load it (0064 D-a)."""
    from memopro.access._suggest import fallback_line, suggest_for_load, suggestions_text

    suggestions = suggest_for_load(plan)
    lines = [
        f"{headline}:",
        _table(plan.candidates),
        suggestions_text(plan.info.source, suggestions, fallback_line(plan)),
    ]
    error = BudgetExceeded("\n".join(line for line in lines if line))
    error.suggestions = suggestions
    return error


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
