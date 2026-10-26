"""Load-time techniques for Hugging Face models (architecture §3.3, 0052 E4).

Each technique contributes keyword arguments to ``from_pretrained`` and a size estimate per
pool; accelerate does the work (0010: never reimplement). Since 0.2.0 only lossless ones remain
(0328): memopro never changes a model's precision, and never offloads weights to the disk.
"""

from __future__ import annotations

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


@dataclass(frozen=True)
class LoadContext:
    device: str  # "cuda" | "mps" | "cpu"
    half_dtype: Any  # torch.bfloat16 or torch.float16
    device_budget: int | None  # None on CPU
    host_budget: int
    unified: bool
    runtime_bytes: int = 0  # activations and generation cache kept on the device


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


# ---------------------------------------------------------------- techniques
OFFLOAD_CPU = LoadTechnique(
    "offload.cpu", QualityGrade.LOSSLESS, 3, Fidelity.EXACT, frozenset({Pool.DEVICE})
)


def load_techniques() -> tuple[LoadTechnique, ...]:
    return (OFFLOAD_CPU,)


def available(tech: LoadTechnique, ctx: LoadContext) -> Availability:
    if tech is OFFLOAD_CPU and (ctx.device == "cpu" or ctx.unified):
        return Availability(False, "unified memory or CPU: host RAM is the same pool")
    return Availability(True)
