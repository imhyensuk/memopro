"""memopro: run work that needs more memory than the machine has, losslessly, within a ceiling.

The public API (0.2.0, 0327): ``enable``/``disable`` set one memory ceiling for the whole
process (``memopro run`` does it for a script); ``finetune``/``generate`` train and run LLMs
whose weights exceed memory; ``load`` loads a model that fits; ``configure``/``using`` change
settings; ``doctor()`` (and ``memopro doctor``) reports memory and the ceiling. ``memopro.rt``
is the advanced layer (runtimes, streamed models, paging).

``import memopro`` is cheap and has no side effects (0032 I4): it loads only the Rust core and
this module. Everything else, and torch in particular, is imported on first use.

Status: alpha (the API may change). Features that are not built yet raise `NotYetImplemented`
with the plan.
"""

from __future__ import annotations

import importlib
from typing import Any

from memopro._core import __version__ as _cargo_version
from memopro._core import core_version
from memopro._errors import (
    BudgetExceeded,
    ConfigError,
    IntegrityError,
    InvalidArgument,
    MemoproError,
    ModeUnavailable,
    NotYetImplemented,
    PolicyError,
)
from memopro.report import Report, report


def _pep440(cargo: str) -> str:
    """Cargo pre-release version -> PEP 440, as maturin writes it: 0.1.0-alpha.1 -> 0.1.0a1."""
    for cargo_tag, pep_tag in (("-alpha.", "a"), ("-beta.", "b"), ("-rc.", "rc")):
        cargo = cargo.replace(cargo_tag, pep_tag)
    return cargo


__version__ = _pep440(_cargo_version)

# name -> (module, attribute or None for the module itself), resolved on first access (PEP 562)
_LAZY: dict[str, tuple[str, str | None]] = {
    "load": ("memopro.access", "load"),
    "finetune": ("memopro.llm", "finetune"),
    "generate": ("memopro.llm", "generate"),
    "doctor": ("memopro._doctor", "doctor"),
    "enable": ("memopro._enable", "enable"),
    "disable": ("memopro._enable", "disable"),
    "configure": ("memopro.config", "configure"),
    "get_config": ("memopro.config", "get_config"),
    "using": ("memopro.config", "using"),
    "rt": ("memopro.rt", None),
}

# names removed from the public API in 0.2.0 (0327): what to use instead
_REMOVED = {
    "optimize": "use memopro.enable() (or memopro run), which applies the ceiling to any code",
    "train_session": "use memopro.finetune(...), or memopro.enable() around your own loop",
    "check": "memopro.doctor() shows the ceiling, and memopro.load(...) says what does not fit",
    "hibernate": "memopro.enable() keeps idle memory within the ceiling",
    "census": "use memopro.doctor() and the memopro.enable() session report",
    "elastic": "set the ceiling with memopro.enable(budget=...)",
    "Technique": "techniques are no longer registered from outside",
    "register_technique": "techniques are no longer registered from outside",
    "registry": "techniques are no longer registered from outside",
    "load_ipython_extension": "the %hibernate/%wake magics were removed; memopro.enable() keeps "
    "idle memory within the ceiling",
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr = _LAZY[name]
    except KeyError:
        if name in _REMOVED:
            raise AttributeError(f"memopro.{name} was removed in 0.2.0: {_REMOVED[name]}") from None
        raise AttributeError(f"module 'memopro' has no attribute {name!r}") from None
    module = importlib.import_module(module_name)
    value = module if attr is None else getattr(module, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "BudgetExceeded",
    "ConfigError",
    "IntegrityError",
    "InvalidArgument",
    "MemoproError",
    "ModeUnavailable",
    "NotYetImplemented",
    "PolicyError",
    "Report",
    "__version__",
    "configure",
    "core_version",
    "disable",
    "doctor",
    "enable",
    "finetune",
    "generate",
    "get_config",
    "load",
    "report",
    "rt",
    "using",
]
