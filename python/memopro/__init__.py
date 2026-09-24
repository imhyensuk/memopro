"""memopro: memory relief and redundancy diagnostics for PyTorch developers.

Status: early development. Only the package skeleton exists so far.
"""

from memopro._core import __version__, core_version
from memopro.report import Report, report
from memopro.techniques import (
    Availability,
    Fidelity,
    Origin,
    Pool,
    QualityGrade,
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
    "Report",
    "Stage",
    "Technique",
    "Timing",
    "__version__",
    "core_version",
    "register_technique",
    "registry",
    "report",
]
