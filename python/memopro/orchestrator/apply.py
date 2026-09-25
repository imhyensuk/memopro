"""Fail-open execution (architecture U3, §3.4).

memopro's own failures must never stop the user's work: a technique that raises is recorded in
the report and skipped. Interrupts (Ctrl-C) and exits are never swallowed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from memopro.report import Report, report

__all__ = ["Outcome", "fail_open"]


@dataclass(frozen=True)
class Outcome:
    ok: bool
    value: Any = None
    error: BaseException | None = None


def fail_open(
    technique: str, fn: Callable[..., Any], *args: Any, into: Report | None = None, **kwargs: Any
) -> Outcome:
    """Run ``fn``; on an ordinary exception record it as "failed" and return ``ok=False``."""
    rep = into if into is not None else report()
    try:
        value = fn(*args, **kwargs)
    except Exception as e:  # noqa: BLE001 - fail-open is the point
        rep.add(technique, "failed", f"{type(e).__name__}: {e}")
        return Outcome(False, error=e)
    return Outcome(True, value)
