"""Candidate configurations (architecture §3.3, 0052 E3): short lists of complete choices.

Since 0.2.0 every candidate keeps the model exactly as the user has it (0325 rule 4, 0328): the
stored weights on the device, or the same weights split between the device and host RAM
(accelerate's CPU offload). The first that is available and fits the budget is chosen; when none
does, ``load`` streams the stored weights instead. A model the user defines with ``dtype`` or
``quantization_config`` is their original: `user_original` sizes it, and it is loaded as given.
Rejected candidates keep their reason, so ``report()`` can say why something was not chosen.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from memopro.access._info import ModelInfo
from memopro.techniques.base import Fidelity, QualityGrade
from memopro.techniques.integrations import loading
from memopro.techniques.integrations.loading import LoadContext, Needs

__all__ = [
    "Configuration",
    "from_pretrained_kwargs",
    "infer_candidates",
    "select",
    "user_original",
]


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


def infer_candidates(
    info: ModelInfo,
    ctx: LoadContext,
    *,
    allow: tuple[str, ...] | None = None,
    deny: tuple[str, ...] = (),
) -> list[Configuration]:
    """The lossless inference configurations for ``info`` in ``ctx``, fastest first."""
    stored = info.weight_bytes()
    raw: list[tuple[Configuration, tuple[loading.LoadTechnique, ...]]] = [
        (
            Configuration(
                "stored",
                (),
                Fidelity.EXACT,
                QualityGrade.LOSSLESS,
                0,
                _needs_on_one_pool(ctx, stored),
                {"device_map": _device_map(ctx), "dtype": "auto"},
            ),
            (),
        )
    ]
    if ctx.device_budget is not None:
        on_device = max(0, min(stored, ctx.device_budget - ctx.runtime_bytes))
        rest = stored - on_device
        tech = loading.OFFLOAD_CPU
        raw.append(
            (
                Configuration(
                    tech.name,
                    (tech.name,),
                    Fidelity.EXACT,
                    QualityGrade.LOSSLESS,
                    tech.speed,
                    Needs(device=on_device + ctx.runtime_bytes, host=rest),
                    {
                        "device_map": "auto",
                        "dtype": "auto",
                        "max_memory": {0: on_device or 1, "cpu": max(rest, ctx.host_budget)},
                    },
                ),
                (tech,),
            )
        )
    out = []
    for cfg, techs in raw:
        why = ""
        if allow is not None and any(t.name not in allow for t in techs):
            why = "not in allow"
        elif any(t.name in deny for t in techs):
            why = "in deny"
        else:
            for t in techs:
                a = loading.available(t, ctx)
                if not a.ok:
                    why = a.reason
                    break
        fits, fit_why = _fits(cfg.needs, ctx)
        out.append(replace(cfg, ok=not why, fits=fits, why=why or fit_why))
    return sorted(out, key=lambda c: (c.speed, c.needs.device + c.needs.host))


def user_original(info: ModelInfo, ctx: LoadContext, given: dict[str, Any]) -> Configuration:
    """The configuration the user defined with ``dtype``/``quantization_config`` (0328): loaded as
    given, so its results are the user's original's. Sized from metadata: a 16-bit ``dtype``
    halves 32-bit weights, a bitsandbytes 8/4-bit config takes a byte or half a byte per linear
    weight; other quantized formats (GPTQ, AWQ, ...) are stored quantized and keep their size."""
    import torch

    quant = given.get("quantization_config")
    dtype = given.get("dtype", given.get("torch_dtype"))
    bits = None
    if quant is not None:
        if getattr(quant, "load_in_4bit", False):
            bits = 4
        elif getattr(quant, "load_in_8bit", False):
            bits = 8
    if bits:
        weights = info.weight_bytes(bits=bits)
    elif dtype in (torch.float16, torch.bfloat16, "float16", "bfloat16"):
        weights = info.weight_bytes(half=True)
    else:
        weights = info.weight_bytes()
    cfg = Configuration(
        "yours",
        tuple(f"{k}={v!r}" for k, v in given.items()),
        Fidelity.EXACT,
        QualityGrade.LOSSLESS,
        0,
        _needs_on_one_pool(ctx, weights),
        {"device_map": _device_map(ctx)} | given,
    )
    fits, why = _fits(cfg.needs, ctx)
    return replace(cfg, fits=fits, why=why)


def _fits(needs: Needs, ctx: LoadContext) -> tuple[bool, str]:
    from memopro._units import format_size as fs

    if ctx.unified and ctx.device_budget is not None:
        total = needs.device + needs.host
        if total > ctx.device_budget:
            return False, f"needs {fs(total)}, device budget {fs(ctx.device_budget)}"
    elif ctx.device_budget is not None and needs.device > ctx.device_budget:
        return False, f"needs {fs(needs.device)} on the device, budget {fs(ctx.device_budget)}"
    if needs.host > ctx.host_budget:
        return False, f"needs {fs(needs.host)} of host RAM, budget {fs(ctx.host_budget)}"
    return True, ""


def select(candidates: list[Configuration]) -> Configuration | None:
    """The first usable candidate (already in preference order)."""
    return next((c for c in candidates if c.usable), None)


def from_pretrained_kwargs(cfg: Configuration, ctx: LoadContext) -> dict[str, Any]:
    """Keyword arguments for ``from_pretrained``."""
    del ctx
    return {k: v for k, v in cfg.kwargs.items() if not k.startswith("_") and v is not None}
