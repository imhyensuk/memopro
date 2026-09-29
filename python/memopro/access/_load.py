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
    backend_of,
    from_pretrained_kwargs,
    infer_candidates,
    post_load,
    quality_note,
    speed_hint,
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
    return LoadPlan(info, ctx, candidates, s.config.quality, s.config.prefer, s, given)


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
    budget_basis: str | None = None,
    disk_writes: str | None = None,
    fallback: str | None = None,
    residency: str | None = None,
    **from_pretrained: Any,
) -> Any:
    """Load ``model_id`` so it fits the budget; returns the model, or ``(model, tokenizer)``.

    ``quality`` bounds automatic loss ("lossless" < "high" < "balanced" < "low"), ``prefer``
    orders the rest ("speed", "quality", "memory"), ``allow``/``deny`` filter technique names
    (``dtype.half``, ``quant.int8``, ``quant.int4``, ``offload.cpu``, ``offload.disk``).
    ``budget_basis``, ``disk_writes`` and ``fallback`` override those settings for this call
    (every setting a `BudgetExceeded` suggestion names can be passed here, 0064).
    ``residency="file"`` keeps the weights as clean file-backed pages on MPS (0072).
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
        budget_basis=budget_basis,
        disk_writes=disk_writes,
        fallback=fallback,
        residency=residency,
    )
    info, ctx, candidates = plan.info, plan.ctx, plan.candidates
    usable = [c for c in candidates if c.usable]
    fallback = None
    if not usable:
        fallback = _fallback(plan)
        if fallback is None:
            raise _nothing_fits(plan, f"no configuration of {info.source} fits the budget")
        usable = [fallback]
    cls = _model_class(info, task)
    rep = report()
    if plan.setup.config.residency == "file":
        return _load_file_backed(
            plan, usable[0], fallback, cls, model_id, revision, from_pretrained, tokenizer
        )
    for cfg in usable:
        kwargs = from_pretrained_kwargs(cfg, ctx) | from_pretrained
        outcome = fail_open(
            f"load.{cfg.name}", _attempt, cls, model_id, revision, kwargs, post_load(cfg, ctx)
        )
        if outcome.ok:
            model = outcome.value
            need = cfg.needs.device + cfg.needs.host + cfg.needs.disk
            backend = backend_of(cfg, ctx)
            detail = (
                f"{info.source}: {cfg.describe()}{f' via {backend}' if backend else ''} "
                f"({cfg.quality.name.lower()}), estimated {format_size(need)}, weights "
                f"{format_size(_footprint(model))} on {ctx.device}"
            )
            if note := quality_note(cfg, ctx):
                detail += f"; {note}"
            if cfg is fallback:
                detail += "; " + _fallback_note(plan, cfg)
                import warnings

                warnings.warn(f"memopro.load: {detail}", UserWarning, stacklevel=2)
            if hint := speed_hint(cfg, candidates, ctx):  # before "applied": the load is last
                rep.add("load", "suggested", hint)
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


_mps_heap_note_shown = False


def _suggest_mps_heap_setting() -> None:
    """0080 W2: after a file-backed load, say once per process that the MPS allocator may hold a
    1 GiB heap on top of the (file-backed) weights, and how to avoid it."""
    global _mps_heap_note_shown
    from memopro.env import MPS_HEAP_NOTE, mps_heap_reserve_on

    if not _mps_heap_note_shown and mps_heap_reserve_on():
        _mps_heap_note_shown = True
        report().add("load", "suggested", MPS_HEAP_NOTE)


def _load_file_backed(
    plan: LoadPlan,
    cfg: Configuration,
    fallback: Any,
    cls: Any,
    model_id: Any,
    revision: Any,
    from_pretrained: dict[str, Any],
    tokenizer: bool,
) -> Any:
    """``residency="file"`` (0072): an explicit choice, so failures raise instead of falling
    back to anonymous memory."""
    from memopro.residency import load_file_backed

    ctx, info = plan.ctx, plan.info

    def build() -> Any:
        kwargs = from_pretrained_kwargs(cfg, ctx) | from_pretrained
        return _attempt(cls, model_id, revision, kwargs, post_load(cfg, ctx))

    model, line = load_file_backed(plan, cfg, cls, model_id, revision, build)
    backend = backend_of(cfg, ctx)
    need = cfg.needs.device + cfg.needs.host + cfg.needs.disk
    detail = (
        f"{info.source}: {cfg.describe()}{f' via {backend}' if backend else ''} "
        f"({cfg.quality.name.lower()}), estimated {format_size(need)} on {ctx.device}; {line}"
    )
    if note := quality_note(cfg, ctx):
        detail += f"; {note}"
    if cfg is fallback:
        detail += "; " + _fallback_note(plan, cfg)
    _suggest_mps_heap_setting()  # before "applied", as hibernate's note (the last entry is the load)
    report().add(f"load.{cfg.name}", "applied", detail)
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
        f"{headline} with quality {plan.quality!r}:",
        _table(plan.candidates),
        suggestions_text(plan.info.source, suggestions, fallback_line(plan)),
    ]
    error = BudgetExceeded("\n".join(line for line in lines if line))
    error.suggestions = suggestions
    return error


def _fallback(plan: LoadPlan) -> Configuration | None:
    """The stored configuration when ``fallback="stored"`` and it is allowed (0064 D-d)."""
    if plan.setup is None or plan.setup.config.fallback != "stored":
        return None
    return next((c for c in plan.candidates if c.name == "stored" and c.ok), None)


def _fallback_note(plan: LoadPlan, cfg: Configuration) -> str:
    from memopro.access._suggest import over_free

    budget = plan.ctx.device_budget if plan.ctx.device_budget is not None else plan.ctx.host_budget
    over = over_free(plan, cfg)
    note = (
        f"fallback='stored': nothing fits the budget {format_size(budget or 0)}, loaded as stored "
        "anyway"
    )
    if over:
        note += f"; about {format_size(over)} more than is free may be compressed or swapped"
    return note


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
