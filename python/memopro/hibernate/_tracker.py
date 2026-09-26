"""Notebook tracker: which names each cell used, idle objects, and suggestions (0015 P4).

A name counts as used in a cell if it appears as an identifier in the cell's code or anywhere in
its raw text (as in the E006 probe), so idle time is never overstated. After a cell, memopro
prints one suggestion line when idle objects hold at least ``SUGGEST_MIN_BYTES``; each object is
suggested once until it is used again. With ``auto=True`` idle objects are hibernated right
away with write-free methods (SSD only under ``disk_writes="allow"``).

An object is idle after ``idle_cells`` cells without use, or after ``idle_seconds`` of wall-clock
time since the end of the last cell that used it (0052 E8), whichever comes first. A cell
boundary is also where γ may act on memory pressure (0052 E6).
"""

from __future__ import annotations

import ast
import re
import time
from typing import Any

SUGGEST_MIN_BYTES = 256 << 20
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SKIP = {"In", "Out", "exit", "quit", "get_ipython", "open"}


def _names_in(code: str, raw: str) -> set[str]:
    names = set(_IDENT.findall(raw))
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return names | set(_IDENT.findall(code))
    return names | {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}


class Tracker:
    def __init__(
        self, shell: Any, idle_cells: int, auto: bool, idle_seconds: float | None = None
    ) -> None:
        self.shell = shell
        self.idle_cells = idle_cells
        self.idle_seconds = idle_seconds
        self.auto = auto
        self.cell = 0
        self.last_used: dict[str, int] = {}
        self.first_seen: dict[str, int] = {}
        self.last_used_at: dict[str, float] = {}  # monotonic time at the end of that cell
        self.suggested: set[str] = set()
        self._pending: set[str] = set()

    def pre_run_cell(self, info: Any) -> None:
        raw = getattr(info, "raw_cell", "") or ""
        try:
            code = self.shell.transform_cell(raw)
        except Exception:  # noqa: BLE001 - never break the user's session
            code = raw
        self._pending = _names_in(code, raw)

    def post_run_cell(self, result: Any) -> None:
        try:
            self._after_cell()
        except Exception as e:  # noqa: BLE001 - never break the user's session
            print(f"[memopro] tracking skipped: {e}")

    def idle(self, name: str) -> int | None:
        if name not in self.first_seen:
            return None
        return self.cell - self.last_used.get(name, self.first_seen[name])

    def idle_time(self, name: str, now: float | None = None) -> float | None:
        if name not in self.last_used_at:
            return None
        return (time.monotonic() if now is None else now) - self.last_used_at[name]

    def is_idle(self, name: str, now: float | None = None) -> bool:
        cells = self.idle(name)
        if cells is not None and cells >= self.idle_cells:
            return True
        seconds = self.idle_time(name, now)
        return (
            self.idle_seconds is not None and seconds is not None and (seconds >= self.idle_seconds)
        )

    def _after_cell(self) -> None:
        from memopro import hibernate
        from memopro._units import format_size
        from memopro.hibernate._tensors import storage_bytes

        self.cell += 1
        now = time.monotonic()
        for name in self._pending:
            self.last_used[name] = self.cell
            self.last_used_at[name] = now
            self.suggested.discard(name)
        ns = self.shell.user_ns
        hidden = getattr(self.shell, "user_ns_hidden", {})
        idle = []
        for name, obj in list(ns.items()):
            if name.startswith("_") or name in _SKIP or name in hidden:
                continue
            self.first_seen.setdefault(name, self.cell)
            self.last_used_at.setdefault(name, now)
            if getattr(type(obj), "_memopro_wake", None) is not None:
                continue  # already a sleeping proxy
            h = hibernate._lookup(obj)
            if h is not None and h.asleep:
                continue
            if not self.is_idle(name, now):
                continue
            n = storage_bytes(obj)
            if n >= hibernate.MIN_OBJECT_BYTES:
                idle.append((name, obj, n, self.idle(name) or 0))
        _at_cell_boundary(self, idle)
        if self.auto:
            self._hibernate_idle(idle)
            return
        fresh = [(name, n, cells) for name, _, n, cells in idle if name not in self.suggested]
        if sum(n for _, n, _ in fresh) < SUGGEST_MIN_BYTES:
            return
        fresh.sort(key=lambda x: -x[1])
        parts = ", ".join(
            f"{name} {format_size(n)} ({cells} cells)" for name, n, cells in fresh[:4]
        )
        print(f"[memopro] idle: {parts} -> %hibernate {fresh[0][0]}  (%hibernate NAME --plan)")
        self.suggested.update(name for name, _, _ in fresh)

    def _hibernate_idle(self, idle: list[tuple[str, Any, int, int]]) -> None:
        from memopro import hibernate
        from memopro._errors import MemoproError
        from memopro._units import format_size
        from memopro.config import get_config
        from memopro.integrations.ipython import sleep_in_namespace

        allow = get_config().disk_writes == "allow"
        for name, obj, n, _ in idle:
            try:
                handle = hibernate.now(obj, allow_spill=allow, name=name)
            except MemoproError:
                continue
            sleep_in_namespace(self.shell, name, obj, handle)
            print(f"[memopro] auto-hibernated {name} ({format_size(n)})")


def _at_cell_boundary(tracker: Tracker, idle: list[tuple[str, Any, int, int]]) -> None:
    """γ acts here, between cells, never while user code runs (0052 E6)."""
    try:
        from memopro.elastic import at_cell_boundary
    except ImportError:
        return
    at_cell_boundary(tracker, idle)


_tracker: Tracker | None = None


def enable(
    auto: bool = False,
    idle_cells: int | None = None,
    shell: Any = None,
    idle_seconds: float | None = None,
) -> Tracker:
    global _tracker
    from memopro._errors import InvalidArgument
    from memopro.config import get_config

    if shell is None:
        try:
            from IPython import get_ipython
        except ImportError:
            get_ipython = None
        shell = get_ipython() if get_ipython else None
    if shell is None:
        raise InvalidArgument("hibernate.enable() needs an IPython/Jupyter session")
    cfg = get_config()
    cells = idle_cells if idle_cells is not None else cfg.idle_cells
    seconds = idle_seconds if idle_seconds is not None else cfg.idle_seconds
    if seconds is not None and seconds <= 0:
        raise InvalidArgument(f"idle_seconds must be > 0; got {seconds!r}")
    if _tracker is not None and _tracker.shell is shell:
        _tracker.idle_cells, _tracker.auto, _tracker.idle_seconds = cells, auto, seconds
        return _tracker
    _tracker = Tracker(shell, cells, auto, seconds)
    shell.events.register("pre_run_cell", _tracker.pre_run_cell)
    shell.events.register("post_run_cell", _tracker.post_run_cell)
    return _tracker


def namespace() -> dict[str, Any] | None:
    return _tracker.shell.user_ns if _tracker is not None else None


def idle_cells(name: str) -> int | None:
    return _tracker.idle(name) if _tracker is not None else None
