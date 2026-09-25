"""Hugging Face ``Trainer`` integration: census as a callback (0032 I8).

``trainer = Trainer(..., callbacks=[memopro.integrations.hf.census_callback()])``

transformers is imported inside the factory, never at module import.

Status: skeleton.
"""

from __future__ import annotations

from typing import Any

from memopro._errors import NotYetImplemented

__all__ = ["census_callback"]


def census_callback(mode: str = "fast", **options: Any) -> Any:
    raise NotYetImplemented(
        "memopro.integrations.hf.census_callback", "v0.1 (N1b)", "docs/research/0032"
    )
