"""Candidate configurations (architecture §3.3): short lists of complete technique combinations.

Selection sorts candidates by the user's preference and picks the first that fits the budget.
Substitutes (e.g. int8 vs int4) never appear together in one configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from memopro._errors import NotYetImplemented
from memopro.orchestrator.budget import Budget
from memopro.techniques.base import Fidelity, QualityGrade

__all__ = ["Configuration", "select"]


@dataclass(frozen=True)
class Configuration:
    name: str
    techniques: tuple[str, ...]  # registry names
    fidelity: Fidelity
    quality: QualityGrade


def select(target: Any, budget: Budget, goal: str) -> Configuration:
    raise NotYetImplemented(
        "memopro.orchestrator.select", "v0.2 (A2)", "docs/design/architecture.md §3.3"
    )
