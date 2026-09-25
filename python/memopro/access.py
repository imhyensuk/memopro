"""Universal access layer entry points (architecture §3–4, v0.2).

These wrap existing, proven techniques as optional backends; memopro never reimplements them
(0010). They are convenience features: the product core is β + census (0032 I7).

Status: skeleton.
"""

from __future__ import annotations

from typing import Any

from memopro._errors import NotYetImplemented

__all__ = ["check", "load", "optimize", "train_session"]

_PLANNED = "v0.2 (A2)"
_REF = "docs/design/architecture.md §4.1"


def load(model_id: str, **options: Any) -> Any:
    """Load a model so it fits the memory budget (load-time candidate configurations)."""
    raise NotYetImplemented("memopro.load", _PLANNED, _REF)


def optimize(model: Any, goal: str = "infer", **options: Any) -> Any:
    """Fit a model you already have to the budget (``goal="infer"`` or ``"train"``)."""
    raise NotYetImplemented("memopro.optimize", _PLANNED, _REF)


def train_session(model: Any, optimizer: Any, **options: Any) -> Any:
    """Context manager that fits model, optimizer and batches to the budget during training."""
    raise NotYetImplemented("memopro.train_session", _PLANNED, _REF)


def check(target: Any, **options: Any) -> Any:
    """Predict whether ``target`` fits the budget and with which configuration."""
    raise NotYetImplemented("memopro.check", _PLANNED, _REF)
