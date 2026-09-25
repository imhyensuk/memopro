"""Lightning integration: census as a callback (0032 I8).

``trainer = L.Trainer(callbacks=[memopro.integrations.lightning.census_callback()])``

lightning is imported inside the factory, never at module import.

Status: skeleton.
"""

from __future__ import annotations

from typing import Any

from memopro._errors import NotYetImplemented

__all__ = ["census_callback"]


def census_callback(mode: str = "fast", **options: Any) -> Any:
    raise NotYetImplemented(
        "memopro.integrations.lightning.census_callback", "v0.1 (N1b)", "docs/research/0032"
    )
