"""Unified memory report (architecture section 3.5): what was applied, measured savings, fallbacks."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ReportEntry:
    technique: str
    action: str  # "applied" | "reverted" | "skipped" | "failed" | "suggested"
    detail: str = ""
    reclaimed_bytes: int | None = None  # measured, never logical (architecture 0013 V7)
    timestamp: float = field(default_factory=time.time)


class Report:
    def __init__(self) -> None:
        self.entries: list[ReportEntry] = []

    def add(
        self, technique: str, action: str, detail: str = "", reclaimed_bytes: int | None = None
    ):
        entry = ReportEntry(technique, action, detail, reclaimed_bytes)
        self.entries.append(entry)
        return entry

    def clear(self) -> None:
        self.entries.clear()

    def to_dict(self) -> dict[str, Any]:
        return {"entries": [asdict(e) for e in self.entries]}

    def summary(self) -> str:
        if not self.entries:
            return "memopro report: nothing applied yet."
        lines = ["memopro report:"]
        for e in self.entries:
            saved = "" if e.reclaimed_bytes is None else f" ({e.reclaimed_bytes / 2**20:.1f} MiB)"
            detail = f" - {e.detail}" if e.detail else ""
            lines.append(f"  [{e.action}] {e.technique}{saved}{detail}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


_global_report = Report()


def report() -> Report:
    """Return the process-wide report."""
    return _global_report
