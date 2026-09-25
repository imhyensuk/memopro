"""γ elastic: step the configuration down under OS memory pressure instead of failing (v0.3).

Write-free steps come first (release source-backed weights, shrink batch or KV cache); SSD
spill only if the policy allows it and within the daily write limit (0032 H3).

Status: skeleton. Before v0.3 starts, redundancy is re-checked (0030 C5).
"""

from __future__ import annotations

from memopro._errors import NotYetImplemented

__all__ = ["enable"]


def enable(**options: object) -> None:
    raise NotYetImplemented("memopro.elastic.enable", "v0.3 (N3)", "docs/design/architecture.md")
