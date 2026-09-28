"""γ elastic: step memory use down under OS memory pressure instead of failing (v0.3, 0052 E6).

**Experimental** (0088): on an 8 GB M1 the macOS pressure level was on in 87% of seconds under
pressure and never when calm, but the stalls a user feels were rare (5 of 870 s), so acting on
"warning" shrinks budgets far more often than needed. `memopro run` leaves γ off on macOS unless
``--elastic`` is given; ``enable()`` still works when called explicitly.

A background thread reads the OS signal (``memopro._core.pressure_current``: macOS memory
pressure level, Linux PSI) and only records it. Nothing is changed from that thread, because
changing tensors that user code may be using at that moment would corrupt its work. Changes
happen at **safe points**, where none of the user's own code is running:

- a notebook cell boundary: empty allocator caches, then hibernate idle objects with write-free
  methods (source, host, compress); SSD spill only if ``disk_writes="allow"``
- a ``train_session`` step boundary: one exact step down per level (micro-batch, checkpointing,
  activation offload), stepping back up one level after ``up_after`` seconds without pressure
- ``memopro.load``: the budget shrinks while pressure lasts (x0.5 warning, x0.25 critical)
- ``memopro.elastic.checkpoint()`` in your own loop: empties caches and returns the level

Where the OS offers no signal, ``enable()`` reports it and stays off (fail-open). Prior work and
the difference to it are in docs/research/0051; memopro claims no more than "not found in our
survey" for this combination.
"""

from __future__ import annotations

import gc
import threading
import time
import weakref
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "at_cell_boundary",
    "at_safe_point",
    "budget_factor",
    "checkpoint",
    "current",
    "disable",
    "enable",
    "register_session",
    "status",
    "unregister_session",
]

LEVELS = ("normal", "warning", "critical")
BUDGET_FACTOR = {"normal": 1.0, "warning": 0.5, "critical": 0.25}


@dataclass
class _State:
    enabled: bool = False
    level: str = "normal"
    reading: dict[str, Any] | None = None
    error: str | None = None
    interval: float = 1.0
    up_after: float = 30.0
    calm_since: float | None = None  # when the level last dropped
    thread: threading.Thread | None = None
    stop: threading.Event = field(default_factory=threading.Event)
    history: list[tuple[float, str]] = field(default_factory=list)
    sessions: weakref.WeakSet[Any] = field(default_factory=weakref.WeakSet)
    steps_down: dict[int, int] = field(default_factory=dict)  # id(session) -> steps taken
    cell_actions: int = 0


_state = _State()
_lock = threading.Lock()


# ---------------------------------------------------------------- reading
def current() -> dict[str, Any]:
    """One reading now: ``{"level", "source", "some_avg10", "full_avg10", "raw_level"}``.

    Raises `ModeUnavailable` where the operating system offers no memory-pressure signal.
    """
    from memopro import _core
    from memopro._errors import ModeUnavailable

    try:
        return dict(_core.pressure_current())
    except (NotImplementedError, OSError) as e:
        raise ModeUnavailable("elastic", str(e)) from None


def _observe(reading: dict[str, Any]) -> None:
    from memopro.report import report

    level = reading["level"]
    with _lock:
        previous = _state.level
        _state.reading = reading
        if level == previous:
            return
        _state.level = level
        now = time.monotonic()
        _state.history.append((time.time(), level))
        del _state.history[:-100]
        if LEVELS.index(level) < LEVELS.index(previous):
            _state.calm_since = now
    detail = f"memory pressure {previous} -> {level} ({_describe(reading)})"
    report().add("elastic", "applied" if level != "normal" else "reverted", detail)


def _describe(reading: dict[str, Any]) -> str:
    if reading.get("raw_level") is not None:
        return f"macOS level {reading['raw_level']}"
    return f"PSI some {reading.get('some_avg10')}%, full {reading.get('full_avg10')}% ({reading.get('source')})"


def _watch() -> None:
    from memopro import _core

    while not _state.stop.is_set():
        try:
            _observe(dict(_core.pressure_current()))
        except Exception as e:  # noqa: BLE001 - the monitor must never take the process down
            with _lock:
                _state.error = f"{type(e).__name__}: {e}"
            return
        _state.stop.wait(_state.interval)


# ---------------------------------------------------------------- control
def enable(interval: float = 1.0, up_after: float = 30.0, notebook: bool = True) -> dict[str, Any]:
    """Start watching memory pressure; returns `status()`.

    ``notebook=True`` also turns on the notebook tracker (cell boundaries are safe points).
    """
    from memopro._errors import InvalidArgument, ModeUnavailable
    from memopro.report import report

    if interval <= 0 or up_after < 0:
        raise InvalidArgument("interval must be > 0 and up_after >= 0")
    try:
        first = current()
    except ModeUnavailable as e:
        report().add("elastic", "skipped", e.reason)
        with _lock:
            _state.enabled, _state.error = False, e.reason
        return status()
    disable()
    with _lock:
        _state.enabled = True
        _state.error = None
        _state.interval, _state.up_after = interval, up_after
        _state.level, _state.reading, _state.calm_since = "normal", None, time.monotonic()
        _state.history = []  # a new watch starts a new history
        _state.stop = threading.Event()
    _observe(first)
    thread = threading.Thread(target=_watch, name="memopro-elastic", daemon=True)
    _state.thread = thread
    thread.start()
    if notebook:
        _hook_notebook()
    report().add("elastic", "applied", f"watching memory pressure every {interval:g}s")
    return status()


def disable() -> None:
    """Stop watching. Steps already taken in running sessions stay until they end."""
    thread = _state.thread
    _state.stop.set()
    if thread is not None and thread is not threading.current_thread():
        thread.join(timeout=5)
    with _lock:
        _state.enabled = False
        _state.thread = None


def status() -> dict[str, Any]:
    with _lock:
        return {
            "enabled": _state.enabled,
            "level": _state.level,
            "reading": dict(_state.reading) if _state.reading else None,
            "error": _state.error,
            "interval": _state.interval,
            "up_after": _state.up_after,
            "history": list(_state.history),
            "sessions": len(_state.sessions),
            "cell_actions": _state.cell_actions,
        }


def budget_factor() -> float:
    """Multiplier for budgets while γ is on (1.0 when off or calm)."""
    with _lock:
        return BUDGET_FACTOR[_state.level] if _state.enabled else 1.0


def _hook_notebook() -> None:
    try:
        from IPython import get_ipython
    except ImportError:
        return
    shell = get_ipython()
    if shell is None:
        return
    from memopro.hibernate import _tracker

    if _tracker._tracker is None or _tracker._tracker.shell is not shell:
        _tracker.enable(shell=shell)


# ---------------------------------------------------------------- safe points
def register_session(session: Any) -> Any:
    with _lock:
        _state.sessions.add(session)
        _state.steps_down[id(session)] = 0
    return session


def unregister_session(session: Any) -> None:
    with _lock:
        _state.sessions.discard(session)
        _state.steps_down.pop(id(session), None)


def at_safe_point(session: Any) -> None:
    """A training step boundary: move the session toward the level the OS reports."""
    with _lock:
        if not _state.enabled or id(session) not in _state.steps_down:
            return
        want = LEVELS.index(_state.level)
        taken = _state.steps_down[id(session)]
        calm_for = time.monotonic() - (_state.calm_since or 0.0)
        up_after = _state.up_after
    if want > taken and session.degrade(f"memory pressure {LEVELS[want]}"):
        with _lock:
            _state.steps_down[id(session)] = taken + 1
    elif want < taken and calm_for >= up_after and session.upgrade():
        with _lock:
            _state.steps_down[id(session)] = taken - 1
            _state.calm_since = time.monotonic()  # one step per calm period


def at_cell_boundary(tracker: Any, idle: list[tuple[str, Any, int, int]]) -> None:
    """Between notebook cells: release caches and hibernate idle objects under pressure."""
    with _lock:
        level = _state.level if _state.enabled else "normal"
    if level == "normal":
        return
    from memopro import hibernate
    from memopro._errors import MemoproError
    from memopro._units import format_size
    from memopro.config import get_config
    from memopro.integrations.ipython import sleep_in_namespace
    from memopro.report import report

    _release_caches()
    allow_spill = level == "critical" and get_config().disk_writes == "allow"
    done = []
    for name, obj, n, _ in sorted(idle, key=lambda x: -x[2]):
        try:
            handle = hibernate.now(obj, allow_spill=allow_spill, name=name)
        except MemoproError:
            continue
        sleep_in_namespace(tracker.shell, name, obj, handle)
        done.append(f"{name} ({format_size(n)})")
    with _lock:
        _state.cell_actions += 1
    if done:
        text = ", ".join(done)
        report().add("elastic", "applied", f"memory pressure {level}: hibernated {text}")
        print(f"[memopro] memory pressure {level}: hibernated {text} (use them to wake them)")


def checkpoint() -> str:
    """A safe point in your own loop: under pressure, empty allocator caches. Returns the level."""
    with _lock:
        level = _state.level if _state.enabled else "normal"
    if level != "normal":
        _release_caches()
    return level


def _release_caches() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
