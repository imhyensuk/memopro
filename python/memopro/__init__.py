"""memopro: memory relief and redundancy diagnostics for PyTorch developers.

``memopro.doctor()`` (and ``memopro doctor``) reports memory per pool and the budget.
Product core (0032 I7): β hibernate (reclaim idle memory, no SSD writes unless allowed) and
census (where memory goes and how much is redundant). The access layer (load, optimize,
train_session, check) follows in v0.2, γ elastic and ``memopro run`` in v0.3.

``import memopro`` is cheap and has no side effects (0032 I4): it loads only the Rust core and
this module. Everything else, and torch in particular, is imported on first use.

Status: skeleton. Features that are not built yet raise `NotYetImplemented` with the plan.
"""

from __future__ import annotations

import importlib
from typing import Any

from memopro._core import __version__, core_version
from memopro._errors import (
    ConfigError,
    InvalidArgument,
    MemoproError,
    ModeUnavailable,
    NotYetImplemented,
    PolicyError,
)
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

# name -> (module, attribute or None for the module itself), resolved on first access (PEP 562)
_LAZY: dict[str, tuple[str, str | None]] = {
    "load": ("memopro.access", "load"),
    "optimize": ("memopro.access", "optimize"),
    "train_session": ("memopro.access", "train_session"),
    "check": ("memopro.access", "check"),
    "doctor": ("memopro._doctor", "doctor"),
    "configure": ("memopro.config", "configure"),
    "get_config": ("memopro.config", "get_config"),
    "census": ("memopro.census", None),
    "config": ("memopro.config", None),
    "elastic": ("memopro.elastic", None),
    "env": ("memopro.env", None),
    "hibernate": ("memopro.hibernate", None),
    "integrations": ("memopro.integrations", None),
    "orchestrator": ("memopro.orchestrator", None),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module 'memopro' has no attribute {name!r}") from None
    module = importlib.import_module(module_name)
    value = module if attr is None else getattr(module, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


def load_ipython_extension(shell: Any) -> None:
    """Entry point for ``%load_ext memopro``."""
    from memopro.integrations.ipython import load_ipython_extension as load

    load(shell)


__all__ = [
    "Availability",
    "ConfigError",
    "Fidelity",
    "InvalidArgument",
    "MemoproError",
    "ModeUnavailable",
    "NotYetImplemented",
    "Origin",
    "PolicyError",
    "Pool",
    "QualityGrade",
    "Report",
    "Stage",
    "Technique",
    "Timing",
    "__version__",
    "census",
    "check",
    "configure",
    "core_version",
    "doctor",
    "elastic",
    "get_config",
    "hibernate",
    "load",
    "load_ipython_extension",
    "optimize",
    "register_technique",
    "registry",
    "report",
    "train_session",
]
