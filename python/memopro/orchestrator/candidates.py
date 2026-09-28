"""Candidate configurations (architecture §3.3, 0052 E3): short lists of complete choices.

Every candidate is a whole configuration (stored weights, half precision, int8, int4, CPU or
disk offload) with its estimated bytes per pool, its quality grade and a speed class. The user's
``quality`` bounds what may be applied automatically (U4, U5); ``prefer`` orders the rest; the
first candidate that is available and fits the budget is chosen. Substitutes (int8 and int4)
never appear in one configuration. Rejected candidates keep their reason, so ``check`` and
``report()`` can say why something was not chosen.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from memopro.access._info import ModelInfo
from memopro.techniques.base import Fidelity, QualityGrade
from memopro.techniques.integrations import loading
from memopro.techniques.integrations.loading import LoadContext, Needs

__all__ = [
    "QUALITY_LIMIT",
    "Configuration",
    "backend_of",
    "from_pretrained_kwargs",
    "infer_candidates",
    "post_load",
    "select",
]

QUALITY_LIMIT = {
    "lossless": QualityGrade.LOSSLESS,
    "high": QualityGrade.NEAR_LOSSLESS,
    "balanced": QualityGrade.SMALL_LOSS,
    "low": QualityGrade.MODERATE_LOSS,
}


@dataclass(frozen=True)
class Configuration:
    name: str
    techniques: tuple[str, ...]
    fidelity: Fidelity
    quality: QualityGrade
    speed: int
    needs: Needs
    kwargs: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)
    ok: bool = True  # available and allowed
    fits: bool = False
    why: str = ""  # why it is not usable, or does not fit

    @property
    def usable(self) -> bool:
        return self.ok and self.fits

    def describe(self) -> str:
        return ", ".join(self.techniques) or "as stored"


def _device_map(ctx: LoadContext) -> Any:
    return {"": "cuda:0" if ctx.device == "cuda" else ctx.device}


def _needs_on_one_pool(ctx: LoadContext, weights: int) -> Needs:
    total = weights + ctx.runtime_bytes
    if ctx.device == "cpu":
        return Needs(host=total)
    return Needs(device=total)


def _quant_config(bits: int, ctx: LoadContext) -> Any:
    backend, _ = loading.quantization_backend(ctx.device, bits)
    if backend == "bitsandbytes":
        from transformers import BitsAndBytesConfig

        if bits == 8:
            return BitsAndBytesConfig(load_in_8bit=True)
        return BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=ctx.half_dtype
        )
    from torchao.quantization import Int4WeightOnlyConfig, Int8WeightOnlyConfig
    from transformers import TorchAoConfig

    quant = Int8WeightOnlyConfig() if bits == 8 else Int4WeightOnlyConfig(group_size=64)
    return TorchAoConfig(quant_type=quant)


def infer_candidates(
    info: ModelInfo,
    ctx: LoadContext,
    *,
    quality: str = "balanced",
    prefer: str = "speed",
    disk_budget: int = 0,
    allow: tuple[str, ...] | None = None,
    deny: tuple[str, ...] = (),
) -> list[Configuration]:
    """All inference configurations for ``info`` in ``ctx``, ordered by ``prefer``."""
    limit = QUALITY_LIMIT[quality]
    stored_half = info.stored_dtype in ("float16", "bfloat16")
    exact_dtype = "auto"
    base_kwargs = {"device_map": _device_map(ctx), "dtype": exact_dtype}
    half_kwargs = {"device_map": _device_map(ctx), "dtype": ctx.half_dtype}
    stored = info.weight_bytes()
    half = info.weight_bytes(half=True)

    raw: list[tuple[Configuration, tuple[loading.LoadTechnique, ...]]] = []

    def add(name, techs, quality_, speed, needs, kwargs, fidelity=Fidelity.EXACT):
        raw.append(
            (
                Configuration(
                    name, tuple(t.name for t in techs), fidelity, quality_, speed, needs, kwargs
                ),
                techs,
            )
        )

    add("stored", (), QualityGrade.LOSSLESS, 0, _needs_on_one_pool(ctx, stored), base_kwargs)
    if not stored_half:
        add(
            "half",
            (loading.HALF,),
            QualityGrade.NEAR_LOSSLESS,
            0,
            _needs_on_one_pool(ctx, half),
            half_kwargs,
            Fidelity.NUMERICS,
        )
    for tech, bits in ((loading.INT8, 8), (loading.INT4, 4)):
        grade = max(tech.quality, QualityGrade.NEAR_LOSSLESS, key=lambda g: g.value)
        add(
            tech.name,
            (tech,),
            grade,
            tech.speed,
            _needs_on_one_pool(ctx, info.weight_bytes(bits=bits)),
            {"device_map": _device_map(ctx), "dtype": ctx.half_dtype, "_quant_bits": bits},
            Fidelity.NUMERICS,
        )
    # offload keeps the stored precision unless half precision is allowed and needed
    offload_weights = half if (limit.value >= QualityGrade.NEAR_LOSSLESS.value) else stored
    offload_grade = (
        QualityGrade.NEAR_LOSSLESS
        if offload_weights == half and not stored_half
        else QualityGrade.LOSSLESS
    )
    offload_fidelity = Fidelity.NUMERICS if offload_grade.value else Fidelity.EXACT
    dtype = ctx.half_dtype if offload_grade.value else exact_dtype
    if ctx.device_budget is not None:
        on_device = max(0, min(offload_weights, ctx.device_budget - ctx.runtime_bytes))
        rest = offload_weights - on_device
        add(
            "offload.cpu",
            (loading.OFFLOAD_CPU,),
            offload_grade,
            loading.OFFLOAD_CPU.speed,
            Needs(device=on_device + ctx.runtime_bytes, host=rest),
            {
                "device_map": "auto",
                "dtype": dtype,
                "max_memory": {0: on_device or 1, "cpu": max(rest, ctx.host_budget)},
            },
            offload_fidelity,
        )
    place = ctx.device_budget if ctx.device_budget is not None else ctx.host_budget
    first = max(0, min(offload_weights, place - ctx.runtime_bytes))
    second = (
        0
        if ctx.device == "cpu" or ctx.unified
        else max(0, min(offload_weights - first, ctx.host_budget))
    )
    rest = offload_weights - first - second
    disk_needs = (
        Needs(host=first + ctx.runtime_bytes, disk=rest)
        if ctx.device == "cpu"
        else Needs(device=first + ctx.runtime_bytes, host=second, disk=rest)
    )
    max_memory: dict[Any, int] = {"cpu": max(1, second if ctx.device != "cpu" else first)}
    if ctx.device == "cuda":
        max_memory[0] = max(1, first)
    elif ctx.device == "mps":
        max_memory["mps"] = max(1, first)
    add(
        "offload.disk",
        (loading.OFFLOAD_DISK,),
        offload_grade,
        loading.OFFLOAD_DISK.speed,
        disk_needs,
        {
            "device_map": "auto",
            "dtype": dtype,
            "max_memory": max_memory,
            "offload_folder": ctx.offload_dir,
        },
        offload_fidelity,
    )

    out = []
    for cfg, techs in raw:
        why = ""
        if cfg.quality.value > limit.value:
            why = f"quality {quality!r} does not allow {cfg.quality.name.lower()} ({cfg.name})"
        elif allow is not None and any(t.name not in allow for t in techs):
            why = "not in allow"
        elif any(t.name in deny for t in techs):
            why = "in deny"
        else:
            for t in techs:
                a = loading.available(t, ctx)
                if not a.ok:
                    why = a.reason
                    break
        fits, fit_why = _fits(cfg.needs, ctx, disk_budget)
        out.append(replace(cfg, ok=not why, fits=fits, why=why or fit_why))
    return sorted(out, key=_order(prefer))


def _fits(needs: Needs, ctx: LoadContext, disk_budget: int) -> tuple[bool, str]:
    from memopro._units import format_size as fs

    if ctx.unified and ctx.device_budget is not None:
        total = needs.device + needs.host
        if total > ctx.device_budget:
            return False, f"needs {fs(total)}, device budget {fs(ctx.device_budget)}"
    elif ctx.device_budget is not None and needs.device > ctx.device_budget:
        return False, f"needs {fs(needs.device)} on the device, budget {fs(ctx.device_budget)}"
    if needs.host > ctx.host_budget:
        return False, f"needs {fs(needs.host)} of host RAM, budget {fs(ctx.host_budget)}"
    if needs.disk > disk_budget:
        return False, f"needs {fs(needs.disk)} of disk, budget {fs(disk_budget)}"
    return True, ""


def _order(prefer: str):
    def key(c: Configuration) -> tuple:
        size = c.needs.device + c.needs.host
        if prefer == "quality":
            return (c.quality.value, c.speed, size)
        if prefer == "memory":
            return (c.needs.device if c.needs.device else c.needs.host, c.quality.value, c.speed)
        return (c.speed, c.quality.value, size)

    return key


def select(candidates: list[Configuration]) -> Configuration | None:
    """The first usable candidate (already in preference order)."""
    return next((c for c in candidates if c.usable), None)


def backend_of(cfg: Configuration, ctx: LoadContext) -> str:
    """The quantization back end a quantized configuration uses here ("" if not quantized)."""
    bits = cfg.kwargs.get("_quant_bits")
    return loading.quantization_backend(ctx.device, bits)[0] if bits else ""


def from_pretrained_kwargs(cfg: Configuration, ctx: LoadContext) -> dict[str, Any]:
    """Keyword arguments for ``from_pretrained`` (quantization configs built only now)."""
    kwargs = {k: v for k, v in cfg.kwargs.items() if not k.startswith("_") and v is not None}
    bits = cfg.kwargs.get("_quant_bits")
    if bits and backend_of(cfg, ctx) == "torch-int4pack":
        # load on the CPU as stored (memory-mapped safetensors), convert layer by layer after
        kwargs["device_map"] = {"": "cpu"}
        kwargs["dtype"] = "auto"
    elif bits:
        kwargs["quantization_config"] = _quant_config(bits, ctx)
    return kwargs


def post_load(cfg: Configuration, ctx: LoadContext) -> Any:
    """What to apply to the model ``from_pretrained`` returned, or None (0069)."""
    if backend_of(cfg, ctx) != "torch-int4pack":
        return None
    from memopro.techniques.integrations import int4pack

    def apply(model: Any) -> Any:
        int4pack.convert(model, ctx.device)
        return model

    return apply
