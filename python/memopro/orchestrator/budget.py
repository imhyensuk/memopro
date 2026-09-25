"""Per-pool budget vector (architecture §3.2): {device, host, disk}.

On unified memory (Apple Silicon) device and host share one physical pool; the device part is
capped by the MPS recommended limit (K5).
"""

from __future__ import annotations

from dataclasses import dataclass

from memopro._errors import NotYetImplemented
from memopro.config import Config
from memopro.env import Env

__all__ = ["Budget", "compute_budget"]


@dataclass(frozen=True)
class Budget:
    device: int | None  # None: no accelerator
    host: int
    disk: int
    unified: bool = False


def compute_budget(env: Env, config: Config) -> Budget:
    raise NotYetImplemented(
        "memopro.orchestrator.compute_budget",
        "v0.1 (N1a doctor)",
        "docs/design/architecture.md §3.2",
    )
