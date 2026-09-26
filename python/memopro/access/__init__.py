"""Universal access layer (architecture §3–4, v0.2): load, optimize, train_session, check.

These choose and apply existing, proven techniques (transformers, accelerate, bitsandbytes,
torchao, torch checkpointing and autocast) as optional back ends; memopro never reimplements
them (0010) and claims no novelty here (0052 E9). Every choice is explained in ``report()``.
"""

from __future__ import annotations

from typing import Any

__all__ = ["check", "load", "optimize", "train_session"]


def load(model_id: Any, **options: Any) -> Any:
    """Load a Hugging Face model so it fits the memory budget. See `memopro.access._load.load`."""
    from memopro.access._load import load as _load

    return _load(model_id, **options)


def optimize(model: Any, goal: str = "infer", **options: Any) -> Any:
    """Fit a model you already have to the budget (``goal="infer"`` or ``"train"``)."""
    from memopro.access._optimize import optimize as _optimize

    return _optimize(model, goal, **options)


def train_session(model: Any, optimizer: Any, **options: Any) -> Any:
    """Context manager that fits training to the budget (micro-batches, checkpointing, ...)."""
    from memopro.access._train import TrainSession

    return TrainSession(model, optimizer, **options)


def check(target: Any, **options: Any) -> Any:
    """Predict memory for inference and training, and which configuration memopro would use."""
    from memopro.access._check import check as _check

    return _check(target, **options)
