"""Technique interface and registry shared by native techniques and integrations."""

from memopro.techniques.base import (
    Availability,
    Fidelity,
    Origin,
    Pool,
    QualityGrade,
    Registry,
    Stage,
    Technique,
    Timing,
    register_technique,
    registry,
)

__all__ = [
    "Availability",
    "Fidelity",
    "Origin",
    "Pool",
    "QualityGrade",
    "Registry",
    "Stage",
    "Technique",
    "Timing",
    "register_technique",
    "registry",
]
