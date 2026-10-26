"""Loading a model within the budget (architecture §3–4): ``load``.

It chooses and applies existing, proven techniques (transformers, accelerate) as back ends; memopro never reimplements
them (0010) and claims no novelty here (0052 E9). Every choice is explained in ``report()``.
"""

from __future__ import annotations

from typing import Any

__all__ = ["load"]


def load(model_id: Any, **options: Any) -> Any:
    """Load a Hugging Face model so it fits the memory budget. See `memopro.access._load.load`."""
    from memopro.access._load import load as _load

    return _load(model_id, **options)
