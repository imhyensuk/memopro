"""census: where memory goes and how much of it is redundant (architecture §5.1 census, 0032).

Modes:

- ``fast``  bytes per category (parameters, gradients, optimizer state, saved activations),
            outliers and massive activations; lossless entropy as a secondary metric
- ``light`` + needed bits per category from sampled tensors at 8/4/2 bits (reconstruction
            error, 0036 B4) and the worst-case tensor (K3) (v0.1, 0032 Q1)
- ``deep``  gradient-impact precision study (v0.2)

Usage::

    with memopro.census.record(model, optimizer) as c:
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
    print(c.summary())

Saved activations are captured with ``torch.autograd.graph.saved_tensors_hooks`` while the block
runs; parameters, gradients and optimizer state are read when it ends. Storages are counted once
(views and tied weights are not double-counted). Findings become actionable advice (0032 P6).
census is not a quantization-sensitivity tool for inference models (0030 C3).
"""

from __future__ import annotations

import random
from typing import Any, Self

from memopro._errors import InvalidArgument, NotYetImplemented

MODES = ("fast", "light", "deep")

__all__ = ["MODES", "Census", "record"]


def record(
    model: Any = None,
    optimizer: Any = None,
    mode: str = "fast",
    *,
    samples_per_category: int = 32,
    sample_elements: int = 65536,
    seed: int = 0,
) -> Census:
    """Context manager that records memory while the block runs."""
    if mode not in MODES:
        raise InvalidArgument(f"unknown census mode {mode!r}; choose from {', '.join(MODES)}")
    if mode == "deep":
        raise NotYetImplemented(
            "memopro.census.record(mode='deep')", "v0.2 (A2)", "docs/research/0036"
        )
    return Census(model, optimizer, mode, samples_per_category, sample_elements, seed)


class Census:
    """Recording and its result. ``summary()`` for people, ``to_json()`` for tools."""

    def __init__(self, model, optimizer, mode, samples_per_category, sample_elements, seed):
        self.model = model
        self.optimizer = optimizer
        self.mode = mode
        self._k = samples_per_category
        self._elements = sample_elements
        self._rng = random.Random(seed)
        self._result: dict[str, Any] | None = None

    # ------------------------------------------------------------------ recording
    def __enter__(self) -> Self:
        import torch

        from memopro.census._collect import SavedTensorLog, allocator_snapshot

        self._torch = torch
        self._saved = SavedTensorLog(self._k, self._elements, self._rng)
        self._alloc_before = allocator_snapshot()
        self._hooks = torch.autograd.graph.saved_tensors_hooks(self._saved.pack, lambda t: t)
        self._hooks.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        from memopro.census._collect import build_result

        self._hooks.__exit__(*exc)
        self._result = build_result(self)
        self._saved = None  # drop samples of saved tensors

    # ------------------------------------------------------------------ results
    def _require(self) -> dict[str, Any]:
        if self._result is None:
            raise InvalidArgument("census result is available after the `with` block ends")
        return self._result

    def to_json(self) -> dict[str, Any]:
        return self._require()

    def advice(self) -> list[str]:
        return list(self._require()["advice"])

    def summary(self) -> str:
        from memopro.census._report import render

        return render(self._require())

    def __repr__(self) -> str:
        state = "done" if self._result is not None else "recording"
        return f"<memopro.census.Census mode={self.mode!r} {state}>"
