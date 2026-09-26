"""SSD write policy and bookkeeping (0032 H2–H5, P5, 0036 B5).

- spill files live in the spill directory (created ``0700``), are ``0600`` and hold raw bytes;
- files of this process are removed at exit, files of dead processes on the next start (P5);
- writes are refused when free space would fall under ``min_free_disk_fraction`` (H4) or the
  day's total would pass the daily limit (H3), and counted per day in ``writes.json`` (H5).
"""

from __future__ import annotations

import atexit
import datetime as _dt
import json
import os
import sys
from pathlib import Path

from memopro import _core
from memopro._errors import ModeUnavailable
from memopro.config import Config, spill_location

DEFAULT_DAILY_CAP = 20 * 10**9  # 0036 B5: min(20 GB, 2% of the disk) per day
DEFAULT_DAILY_FRACTION = 0.02

_own_files: set[str] = set()
_cleaned: set[str] = set()


def _pid_alive(pid: int) -> bool:
    if sys.platform == "win32":
        return _pid_alive_windows(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _pid_alive_windows(pid: int) -> bool:
    """Never ``os.kill(pid, 0)`` on Windows: signal 0 is CTRL_C_EVENT there (0052 E8)."""
    import ctypes

    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return kernel32.GetLastError() == 5  # access denied: it exists
    try:
        code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return True
        return code.value == 259  # STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def spill_dir(config: Config) -> Path:
    d = spill_location(config)
    d.mkdir(mode=0o700, parents=True, exist_ok=True)
    key = str(d)
    if key not in _cleaned:
        _cleaned.add(key)
        for f in d.glob("*.mpspill"):  # leftovers of processes that died (P5)
            try:
                pid = int(f.name.split("-", 1)[0])
            except ValueError:
                continue
            if pid != os.getpid() and not _pid_alive(pid):
                f.unlink(missing_ok=True)
    return d


def daily_limit(config: Config, disk_total: int) -> int:
    if config.daily_write_limit is not None:
        return config.daily_write_limit
    return min(DEFAULT_DAILY_CAP, int(disk_total * DEFAULT_DAILY_FRACTION))


def _counter(directory: Path) -> Path:
    return directory / "writes.json"


def _today() -> str:
    return _dt.datetime.now().astimezone().date().isoformat()  # the user's local day


def written_today(directory: Path) -> int:
    try:
        data = json.loads(_counter(directory).read_text())
    except (OSError, ValueError):
        return 0
    return int(data.get(_today(), 0))


def written_total(directory: Path) -> int:
    try:
        data = json.loads(_counter(directory).read_text())
    except (OSError, ValueError):
        return 0
    return int(sum(int(v) for v in data.values()))


def _record(directory: Path, nbytes: int) -> None:
    path = _counter(directory)
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        data = {}
    today = _today()
    data[today] = int(data.get(today, 0)) + nbytes
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    os.chmod(tmp, 0o600)
    tmp.replace(path)


def spilled_bytes() -> int:
    """Bytes this process holds in spill files right now."""
    total = 0
    for path in list(_own_files):
        try:
            total += os.path.getsize(path)
        except OSError:
            pass
    return total


def _disk_budget(config: Config, above_floor: int) -> int | None:
    """The disk pool's budget setting (0059 D2) applied to this disk, or None if it has none."""
    from memopro.config import PoolBudget
    from memopro.orchestrator.budget import resolve_pool

    spec = config.budget
    if not isinstance(spec, PoolBudget) or spec.disk is None:
        return None
    # measured before this process wrote anything, so its own files do not shrink the cap
    return resolve_pool(spec.disk, max(0, above_floor) + spilled_bytes())[0]


def check_room(config: Config, nbytes: int) -> Path:
    """Directory to spill into, or ModeUnavailable explaining why SSD writes are not possible."""
    directory = spill_dir(config)
    disk = _core.hwinfo_disk(directory)
    floor = int(disk["total_bytes"] * config.min_free_disk_fraction)
    if disk["available_bytes"] - nbytes < floor:
        raise ModeUnavailable(
            "spill",
            f"free disk space would fall below the {config.min_free_disk_fraction:.0%} floor",
            ("source", "host", "compress"),
        )
    cap = _disk_budget(config, disk["available_bytes"] - floor)
    if cap is not None and spilled_bytes() + nbytes > cap:
        raise ModeUnavailable(
            "spill",
            f"the disk budget ({cap} bytes, budget setting) would be exceeded",
            ("source", "host", "compress"),
        )
    limit = daily_limit(config, disk["total_bytes"])
    if written_today(directory) + nbytes > limit:
        raise ModeUnavailable(
            "spill", f"today's SSD write limit ({limit} bytes) would be exceeded", ("compress",)
        )
    return directory


def write(data, directory: Path) -> tuple[str, int, bytes]:
    path, n, digest = _core.engine_write(data, directory)
    _own_files.add(str(path))
    _record(directory, n)
    return str(path), n, digest


def remove(path: str) -> None:
    _own_files.discard(path)
    Path(path).unlink(missing_ok=True)


@atexit.register
def _cleanup() -> None:
    for path in list(_own_files):
        Path(path).unlink(missing_ok=True)
