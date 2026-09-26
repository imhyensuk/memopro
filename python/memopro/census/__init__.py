"""census: where memory goes and how much of it is redundant (architecture §5.1 census, 0032).

Modes:

- ``fast``  bytes per category (parameters, gradients, optimizer state, saved activations),
            outliers and massive activations; lossless entropy as a secondary metric
- ``light`` + needed bits per category from sampled tensors at 8/4/2 bits (reconstruction
            error, 0036 B4) and the worst-case tensor (K3) (v0.1, 0032 Q1)
- ``deep``  + needed bits for *training* (v0.2, 0052 E5): each category perturbed as a whole,
            effect on the loss, the gradient and one optimizer step; needs ``probe``

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

from memopro._errors import InvalidArgument

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
    probe: Any = None,
    tolerance: tuple[float, float] | None = None,
) -> Census:
    """Context manager that records memory while the block runs.

    ``mode="deep"`` needs ``probe``: a function returning the loss of a fixed batch, e.g.
    ``lambda: model(**batch).loss``. ``tolerance=(loss, cosine)`` overrides (1e-3, 0.999).
    """
    if mode not in MODES:
        raise InvalidArgument(f"unknown census mode {mode!r}; choose from {', '.join(MODES)}")
    if mode == "deep" and (probe is None or model is None):
        raise InvalidArgument(
            "census deep mode needs the model and probe=<function returning a loss for a fixed "
            "batch>, e.g. probe=lambda: model(**batch).loss"
        )
    census = Census(model, optimizer, mode, samples_per_category, sample_elements, seed)
    census._probe = probe
    census._tolerance = tolerance
    census._seed = seed
    return census


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

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        from memopro._errors import MemoproError
        from memopro.census._collect import build_result

        self._hooks.__exit__(exc_type, exc, tb)
        try:
            self._result = build_result(self)
            if self.mode == "deep" and exc_type is None:
                self._result["deep"] = self._deep()
        except Exception as e:
            if exc_type is not None:  # never hide the user's own exception (0048)
                return
            if isinstance(e, MemoproError):
                raise
            raise MemoproError(f"census could not build its result: {type(e).__name__}: {e}") from e
        finally:
            self._saved = None  # drop samples of saved tensors

    def _deep(self) -> dict[str, Any]:
        from memopro.census import _deep

        tol = _deep.Tolerance(*self._tolerance) if self._tolerance else _deep.Tolerance()
        deep = _deep.run(self.model, self.optimizer, self._probe, tolerance=tol, seed=self._seed)
        cats = self._result["categories"] if self._result else {}
        for name in ("parameters", "gradients", "optimizer_state", "saved_activations"):
            entry, cat = deep.get(name), cats.get(name, {})
            stored, needed = cat.get("stored_bits"), (entry or {}).get("needed_bits")
            if entry is not None and stored and needed:
                entry["stored_bits"] = stored
                entry["waste_bits"] = max(0, stored - needed)
                entry["saveable_bytes"] = int(cat["bytes"] * max(0.0, 1 - needed / stored))
        return deep

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
