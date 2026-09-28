"""Load-time techniques for Hugging Face models (architecture §3.3, 0052 E4).

Each technique contributes keyword arguments to ``from_pretrained`` and a size estimate per
pool. None of them is implemented here: transformers, bitsandbytes, torchao and accelerate do
the work (0010: never reimplement). Changing a load-time choice means loading again (§3.4).

Availability of quantization back ends is decided by a real test on a tiny layer on the target
device, because "installed" is not "works here" (torchao int4 needs an extra package, 0051).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any

from memopro.techniques.base import (
    Availability,
    Fidelity,
    Origin,
    Pool,
    QualityGrade,
    Stage,
    Timing,
)

__all__ = ["LoadContext", "LoadTechnique", "Needs", "load_techniques"]


@dataclass(frozen=True)
class Needs:
    """Estimated bytes per pool once loaded."""

    device: int = 0
    host: int = 0
    disk: int = 0


@dataclass(frozen=True)
class LoadContext:
    device: str  # "cuda" | "mps" | "cpu"
    half_dtype: Any  # torch.bfloat16 or torch.float16
    device_budget: int | None  # None on CPU
    host_budget: int
    disk_writes: str
    unified: bool
    runtime_bytes: int = 0  # activations and generation cache kept on the device
    offload_dir: str | None = None


@dataclass(frozen=True)
class LoadTechnique:
    name: str
    quality: QualityGrade
    speed: int  # 0 same, 1 a little slower, 3 slow, 4 very slow
    fidelity: Fidelity
    pools: frozenset[Pool]
    stages: frozenset[Stage] = frozenset({Stage.INFER})
    timing: Timing = Timing.LOAD_TIME
    origin: Origin = Origin.INTEGRATION
    backend: str = ""
    extra: dict[str, Any] = field(default_factory=dict, compare=False)


# ---------------------------------------------------------------- back end tests
@functools.cache
def _bnb_works(device: str, bits: int) -> str:
    """Empty if bitsandbytes runs a ``bits``-bit linear layer on ``device``, else the reason."""
    try:
        import bitsandbytes as bnb
        import torch

        if bits == 8:
            layer = bnb.nn.Linear8bitLt(64, 64, has_fp16_weights=False)
        else:
            layer = bnb.nn.Linear4bit(64, 64, compute_dtype=torch.float16)
        layer = layer.to(device)
        layer(torch.randn(2, 64, device=device, dtype=torch.float16))
        return ""
    except Exception as e:  # noqa: BLE001 - any failure means "not here"
        return f"bitsandbytes {bits}-bit does not run on {device}: {type(e).__name__}: {e}"[:200]


@functools.cache
def _torchao_works(device: str, bits: int) -> str:
    """Empty if torchao runs ``bits``-bit weight-only quantization on ``device``."""
    try:
        import torch
        from torchao.quantization import Int4WeightOnlyConfig, Int8WeightOnlyConfig, quantize_

        config = Int8WeightOnlyConfig() if bits == 8 else Int4WeightOnlyConfig(group_size=64)
        layer = torch.nn.Sequential(torch.nn.Linear(256, 256)).to(device, torch.bfloat16)
        quantize_(layer, config)
        layer(torch.randn(2, 256, device=device, dtype=torch.bfloat16))
        return ""
    except Exception as e:  # noqa: BLE001
        return f"torchao int{bits} does not run on {device}: {type(e).__name__}: {e}"[:200]


@functools.cache
def _int4pack_works(device: str, bits: int) -> str:
    """Empty if torch's own int4 kernel runs on ``device`` (MPS; 0069)."""
    if bits != 4:
        return "torch int4pack is an int4 back end"
    try:
        from memopro.techniques.integrations import int4pack
    except ImportError as e:  # torchao (for the packing) missing
        return f"torch int4pack needs torchao: {e}"
    return int4pack.works(device)


def quantization_backend(device: str, bits: int) -> tuple[str, str]:
    """(back end, "") for the first working back end, or ("", reasons).

    On MPS torch's own int4 kernel comes first: 1.63x bf16, where bitsandbytes nf4 ran at 0.35x
    (E015 Q4, 0069).
    """
    reasons = []
    order = (
        ("torch-int4pack", _int4pack_works),
        ("bitsandbytes", _bnb_works),
        ("torchao", _torchao_works),
    )
    for backend, works in order:
        reason = works(device, bits)
        if not reason:
            return backend, ""
        reasons.append(reason)
    return "", "; ".join(reasons)


# ---------------------------------------------------------------- techniques
HALF = LoadTechnique(
    "dtype.half", QualityGrade.NEAR_LOSSLESS, 0, Fidelity.NUMERICS, frozenset({Pool.DEVICE})
)
INT8 = LoadTechnique(
    "quant.int8", QualityGrade.SMALL_LOSS, 1, Fidelity.NUMERICS, frozenset({Pool.DEVICE})
)
INT4 = LoadTechnique(
    "quant.int4", QualityGrade.MODERATE_LOSS, 1, Fidelity.NUMERICS, frozenset({Pool.DEVICE})
)
OFFLOAD_CPU = LoadTechnique(
    "offload.cpu", QualityGrade.LOSSLESS, 3, Fidelity.EXACT, frozenset({Pool.DEVICE})
)
OFFLOAD_DISK = LoadTechnique(
    "offload.disk", QualityGrade.LOSSLESS, 4, Fidelity.EXACT, frozenset({Pool.DEVICE, Pool.HOST})
)


def load_techniques() -> tuple[LoadTechnique, ...]:
    return (HALF, INT8, INT4, OFFLOAD_CPU, OFFLOAD_DISK)


def available(tech: LoadTechnique, ctx: LoadContext) -> Availability:
    if tech.name in ("quant.int8", "quant.int4"):
        backend, reason = quantization_backend(ctx.device, 8 if tech is INT8 else 4)
        return Availability(bool(backend), reason or backend)
    if tech is OFFLOAD_CPU and (ctx.device == "cpu" or ctx.unified):
        return Availability(False, "unified memory or CPU: host RAM is the same pool")
    if tech is OFFLOAD_DISK and ctx.disk_writes != "allow":
        return Availability(
            False, f"writes weights to SSD, and disk_writes is {ctx.disk_writes!r} (not 'allow')"
        )
    return Availability(True)
