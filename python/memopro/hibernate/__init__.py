"""β hibernate: reclaim memory held by idle tensors and modules (architecture §5.1 β, 0032).

Methods, tried in this order by ``mode="auto"`` (no SSD writes before the last one):

1. ``source``   drop the memory; restore by re-reading the original file (hash-verified)
2. ``host``     move a CUDA tensor to host RAM
3. ``compress`` lossless compression in RAM
4. ``spill``    write to SSD, only when ``disk_writes`` allows it

``bf16`` (lossy) is never chosen automatically. Two styles are offered: an explicit handle
(``h = now(obj)`` then ``obj = h.wake()``, no proxies) and the notebook magics
(``%hibernate``/``%wake``), which rebind the name through a proxy.

Status: skeleton. The policy is enforced already; the methods arrive in v0.1 after gate Gβ.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from memopro._errors import NotYetImplemented
from memopro.config import get_config
from memopro.hibernate._policy import MODES, Step, resolve_modes

__all__ = [
    "MODES",
    "Handle",
    "PlanRow",
    "Step",
    "enable",
    "now",
    "plan",
    "resolve_modes",
    "status",
    "suggest",
    "wake",
]

_PLANNED = "v0.1 (N1c, after gate Gβ)"
_REF = "docs/research/0032-adopt-group1-2.md"


@dataclass(frozen=True)
class PlanRow:
    """One line of ``plan()``: what a method would do, without doing it."""

    mode: str
    available: bool
    reason: str  # why not, when unavailable
    reclaim_bytes: int
    restore_seconds: float | None  # estimate
    disk_write_bytes: int
    fidelity: str  # "exact" | "numerics"


class Handle:
    """Explicit handle to a hibernated object (0032 I5). ``wake()`` returns the restored object."""

    def __init__(self, name: str, mode: str, nbytes: int) -> None:
        self.name = name
        self.mode = mode
        self.nbytes = nbytes

    def wake(self) -> Any:
        raise NotYetImplemented("memopro.hibernate.Handle.wake", _PLANNED, _REF)

    def __repr__(self) -> str:
        return f"Handle(name={self.name!r}, mode={self.mode!r}, nbytes={self.nbytes})"


def now(obj: Any, mode: str = "auto", *, allow_spill: bool = False) -> Handle:
    """Hibernate ``obj`` now and return a handle."""
    resolve_modes(mode, get_config(), allow_spill=allow_spill)  # policy errors surface today
    raise NotYetImplemented("memopro.hibernate.now", _PLANNED, _REF)


def plan(obj: Any) -> list[PlanRow]:
    """Compare every method for ``obj`` (reclaim, restore time, SSD writes) without running it."""
    raise NotYetImplemented("memopro.hibernate.plan", _PLANNED, _REF)


def wake(target: Any) -> Any:
    """Restore a hibernated object (a `Handle` or a proxy) right away."""
    raise NotYetImplemented("memopro.hibernate.wake", _PLANNED, _REF)


def suggest() -> list[Any]:
    """List idle objects worth hibernating, with the reclaimable amount per method."""
    raise NotYetImplemented("memopro.hibernate.suggest", _PLANNED, _REF)


def enable(auto: bool = False, idle_cells: int | None = None, idle_seconds: float | None = None):
    """Turn on suggestions (and, optionally, automatic hibernation) in a notebook session."""
    raise NotYetImplemented("memopro.hibernate.enable", _PLANNED, _REF)


def status() -> dict[str, Any]:
    """Hibernated objects, reclaimed bytes and SSD writes (today, total) (0032 H5)."""
    raise NotYetImplemented("memopro.hibernate.status", _PLANNED, _REF)
