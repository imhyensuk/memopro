"""census: where memory goes and how much of it is redundant (architecture §5.1 census, 0032).

Modes:

- ``fast``  bytes per category (parameters, gradients, optimizer state, saved tensors), outliers
            and massive activations, idle time; lossless entropy as a secondary metric
- ``light`` + needed bits per category from a few sampled tensors at 2/4/8 bits, and the
            worst-case tensor (v0.1, 0032 Q1)
- ``deep``  full precision study (v0.2)

Findings are turned into actionable advice (0032 P6); census is not a quantization-sensitivity
tool for inference models (0030 C3).

Status: skeleton.
"""

from __future__ import annotations

from typing import Any

from memopro._errors import InvalidArgument, NotYetImplemented

MODES = ("fast", "light", "deep")
_REF = "docs/research/0032-adopt-group1-2.md"

__all__ = ["MODES", "Census", "record"]


class Census:
    """Result of a recording. ``summary()`` for people, ``to_json()`` for tools."""

    def summary(self) -> str:
        raise NotYetImplemented("memopro.census.Census.summary", "v0.1 (N1b)", _REF)

    def to_json(self) -> dict[str, Any]:
        raise NotYetImplemented("memopro.census.Census.to_json", "v0.1 (N1b)", _REF)


def record(model: Any = None, optimizer: Any = None, mode: str = "fast") -> Census:
    """Context manager that records memory while the block runs.

    ``with memopro.census.record(model, optimizer) as c: ...`` then ``print(c.summary())``.
    """
    if mode not in MODES:
        raise InvalidArgument(f"unknown census mode {mode!r}; choose from {', '.join(MODES)}")
    planned = "v0.2 (A2)" if mode == "deep" else "v0.1 (N1b)"
    raise NotYetImplemented(f"memopro.census.record(mode={mode!r})", planned, _REF)
