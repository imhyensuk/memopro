"""Configuration: defaults < memopro.toml < MEMOPRO_* environment < memopro.configure().

Nothing is read at import time (0032 I4). Every `get_config()` call resolves the layers again, so
a change to the environment or the file is picked up without restarting.

The file is `$MEMOPRO_CONFIG` if set, otherwise `./memopro.toml` if it exists. Keys sit at the top
level of the file and use the same names as `configure()`.
"""

from __future__ import annotations

import dataclasses
import os
import sys
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memopro._errors import ConfigError, InvalidArgument
from memopro._units import parse_size

QUALITIES = ("lossless", "balanced")
PREFERENCES = ("quality", "speed", "memory")
DISK_WRITES = ("ask", "never", "allow")
# Write-free hibernation methods that `mode="auto"` may pick, in order (0032 H1). `bf16` is lossy
# and only ever explicit; `spill` writes to disk and is governed by `disk_writes` instead.
AUTO_MODES = ("source", "host", "compress")


@dataclass(frozen=True)
class Config:
    budget: str | int = "auto"  # "auto" or a byte count
    headroom: float = 0.10  # fraction of usable memory never budgeted (U3 prevention, 0035)
    quality: str = "balanced"
    prefer: str = "quality"
    hibernate_modes: tuple[str, ...] = AUTO_MODES
    disk_writes: str = "ask"
    daily_write_limit: int | None = None  # None: derived from the disk size (decided in A1b)
    min_free_disk_fraction: float = 0.20  # 0032 H4
    spill_dir: str | None = None
    idle_cells: int = 3


_FIELDS = {f.name for f in dataclasses.fields(Config)}
_overrides: dict[str, Any] = {}


def _choice(key: str, value: Any, choices: tuple[str, ...]) -> str:
    if value not in choices:
        raise InvalidArgument(f"{key} must be one of {', '.join(choices)}; got {value!r}")
    return value


def _modes(value: Any) -> tuple[str, ...]:
    items = value.split(",") if isinstance(value, str) else list(value)
    modes = tuple(str(m).strip() for m in items if str(m).strip())
    for m in modes:
        if m == "bf16":
            raise InvalidArgument("hibernate_modes cannot contain 'bf16': lossy, explicit only")
        if m == "spill":
            raise InvalidArgument("hibernate_modes cannot contain 'spill': use disk_writes")
        _choice("hibernate_modes", m, AUTO_MODES)
    if len(set(modes)) != len(modes):
        raise InvalidArgument(f"hibernate_modes has duplicates: {modes}")
    return modes


def _validate(key: str, value: Any) -> Any:
    if key not in _FIELDS:
        raise InvalidArgument(f"unknown setting {key!r}; known: {', '.join(sorted(_FIELDS))}")
    match key:
        case "budget":
            return "auto" if value == "auto" else parse_size(value)
        case "quality":
            return _choice(key, value, QUALITIES)
        case "prefer":
            return _choice(key, value, PREFERENCES)
        case "hibernate_modes":
            return _modes(value)
        case "disk_writes":
            return _choice(key, value, DISK_WRITES)
        case "daily_write_limit":
            return None if value in (None, "", "auto") else parse_size(value)
        case "headroom":
            fraction = float(value)
            if not 0.0 <= fraction < 0.9:
                raise InvalidArgument(f"headroom must be in [0, 0.9); got {value!r}")
            return fraction
        case "min_free_disk_fraction":
            fraction = float(value)
            if not 0.0 <= fraction < 1.0:
                raise InvalidArgument(f"min_free_disk_fraction must be in [0, 1); got {value!r}")
            return fraction
        case "spill_dir":
            return None if value in (None, "") else str(value)
        case "idle_cells":
            cells = int(value)
            if cells < 1:
                raise InvalidArgument(f"idle_cells must be >= 1; got {value!r}")
            return cells
    raise AssertionError(key)


def _layer(source: str, values: Mapping[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in values.items():
        try:
            out[key] = _validate(key, value)
        except (TypeError, ValueError) as e:  # includes InvalidArgument; no raw errors (0048)
            raise ConfigError(f"{source}: {key}={value!r}: {e}") from None
    return out


def _file_layer() -> dict[str, Any]:
    explicit = os.environ.get("MEMOPRO_CONFIG")
    path = Path(explicit) if explicit else Path("memopro.toml")
    if not path.is_file():
        if explicit:
            raise ConfigError(f"MEMOPRO_CONFIG points to a missing file: {path}")
        return {}
    try:
        data = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: {e}") from None
    return _layer(str(path), data)


def _env_layer() -> dict[str, Any]:
    values = {}
    for key in _FIELDS:
        env = f"MEMOPRO_{key.upper()}"
        if env in os.environ:
            values[key] = os.environ[env]
    return _layer("environment", values)


def default_spill_dir() -> Path:
    """Per-user cache location for spill files (created only when something is spilled)."""
    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Caches" / "memopro"
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", home / "AppData" / "Local")) / "memopro"
    return Path(os.environ.get("XDG_CACHE_HOME", home / ".cache")) / "memopro"


def spill_location(config: Config) -> Path:
    """Where spill files would go: ``spill_dir`` if set, otherwise `default_spill_dir()`."""
    return Path(config.spill_dir).expanduser() if config.spill_dir else default_spill_dir()


def get_config() -> Config:
    """Resolve the effective configuration from all layers."""
    merged: dict[str, Any] = {}
    merged.update(_file_layer())
    merged.update(_env_layer())
    merged.update(_overrides)
    return Config(**merged)


def configure(**settings: Any) -> Config:
    """Set runtime overrides (highest priority) and return the effective configuration.

    Example: ``memopro.configure(disk_writes="never", spill_dir="/Volumes/External/memopro")``
    """
    _overrides.update(_layer("configure()", settings))
    return get_config()


def reset_config() -> None:
    """Drop all runtime overrides set by `configure()`."""
    _overrides.clear()
