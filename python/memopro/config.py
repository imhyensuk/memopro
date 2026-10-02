"""Configuration: defaults < memopro.toml < MEMOPRO_* < configure() < using() < per-call options.

Nothing is read at import time (0032 I4). Every `get_config()` call resolves the layers again, so
a change to the environment or the file is picked up without restarting.

The file is `$MEMOPRO_CONFIG` if set, otherwise `./memopro.toml` if it exists. Keys sit at the top
level of the file and use the same names as `configure()`; a per-pool budget is a ``[budget]``
table. ``with using(...)`` changes settings inside one block only (0059 D5).
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import sys
import tomllib
from collections.abc import Iterator, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memopro._errors import ConfigError, InvalidArgument
from memopro._units import parse_size

# Most loss that may be applied automatically (0052 E3): exact only < half precision < int8 < int4
QUALITIES = ("lossless", "high", "balanced", "low")
PREFERENCES = ("speed", "quality", "memory")
DISK_WRITES = ("ask", "never", "allow")
# What the measured host budget starts from (0059 D3); only "conservative" never needs swap
BUDGET_BASES = ("conservative", "os", "total")
POOLS = ("device", "host", "disk")
# What load/optimize do when nothing fits the budget (0064 D-d): stop, or load as stored anyway
FALLBACKS = ("none", "stored")
# What kind of memory holds loaded weights (0072): anonymous, or clean file-backed pages (MPS)
RESIDENCIES = ("memory", "file")
# Write-free hibernation methods that `mode="auto"` may pick, in order (0032 H1). `bf16` is lossy
# and only ever explicit; `spill` writes to disk and is governed by `disk_writes` instead.
AUTO_MODES = ("source", "host", "compress")


@dataclass(frozen=True)
class Limit:
    """A pool budget beyond a plain size or fraction (0059 D1).

    ``use`` is "auto", a size in bytes or a fraction of the measured budget. ``reserve`` leaves that
    many bytes of the measured budget unused ("-2GB"). ``force`` takes the size ``use`` as it is,
    even above what is measured ("6GB!"). ``maximum`` caps the result; below ``minimum`` memopro
    stops with `BudgetExceeded` instead of trying ever smaller configurations ("2GB..6GB").
    """

    use: str | int | float = "auto"
    reserve: int = 0
    force: bool = False
    minimum: int | None = None
    maximum: int | None = None


# One pool's setting: "auto", bytes (a cap), a fraction of the measured budget, or a Limit
PoolValue = str | int | float | Limit


@dataclass(frozen=True)
class PoolBudget:
    """Settings per pool (0052 E3, 0059 D2); None leaves that pool to the measured budget."""

    device: PoolValue | None = None
    host: PoolValue | None = None
    disk: PoolValue | None = None


@dataclass(frozen=True)
class Config:
    # one pool value for device and host (see PoolValue), or settings per pool
    budget: PoolValue | PoolBudget = "auto"
    budget_basis: str = "conservative"
    # never budgeted (U3 prevention, 0035): a fraction of the measured memory, or bytes (0059 D4)
    headroom: float = 0.10
    quality: str = "balanced"
    prefer: str = "speed"
    fallback: str = "none"
    residency: str = "memory"
    hibernate_modes: tuple[str, ...] = AUTO_MODES
    disk_writes: str = "ask"
    daily_write_limit: int | None = None  # None: derived from the disk size (decided in A1b)
    min_free_disk_fraction: float = 0.20  # 0032 H4
    spill_dir: str | None = None
    idle_cells: int = 3
    idle_seconds: float | None = None  # also suggest objects unused this long (0052 E8)


_FIELDS = {f.name for f in dataclasses.fields(Config)}
_overrides: dict[str, Any] = {}
_scoped: ContextVar[tuple[dict[str, Any], ...]] = ContextVar("memopro_settings", default=())


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


def _fraction(value: Any) -> float | None:
    """0.5, "0.5" or "50%" as a fraction in (0, 1]; None if ``value`` is not written as one."""
    fraction = None
    if isinstance(value, float):
        fraction = value
    elif isinstance(value, str) and value.strip().endswith("%"):
        fraction = float(value.strip()[:-1]) / 100
    elif isinstance(value, str) and "." in value and value.strip().replace(".", "", 1).isdigit():
        fraction = float(value)
    if fraction is not None and not 0.0 < fraction <= 1.0:
        raise InvalidArgument(f"a budget fraction must be in (0, 1]; got {value!r}")
    return fraction


def _optional_size(text: str) -> int | None:
    return parse_size(text.strip()) if text.strip() else None


def _simple(limit: Limit) -> PoolValue:
    """A Limit that only says ``use`` is that plain value (keeps old settings comparable)."""
    if limit == Limit(use=limit.use):
        return limit.use
    return limit


def _pool_value(value: Any) -> PoolValue:
    """Parse one pool's budget (0059 D1): "auto", "6GB", 0.5 / "50%", "-2GB" (leave 2GB free),
    "6GB!" (exactly 6GB, even above what is measured), "2GB..6GB" (at least / at most), or a
    mapping {"use": ..., "min": ..., "max": ...}."""
    if isinstance(value, Limit):
        return _simple(value)
    if isinstance(value, Mapping):
        unknown = set(value) - {"use", "min", "max"}
        if unknown:
            raise InvalidArgument(f"a pool budget takes use, min and max; got {sorted(unknown)}")
        base = _pool_value(value.get("use", "auto"))
        if isinstance(base, Limit) and (base.minimum is not None or base.maximum is not None):
            raise InvalidArgument("use cannot be a range; give min and max instead")
        limit = base if isinstance(base, Limit) else Limit(use=base)
        bounds = {k: value[k] for k in ("min", "max") if k in value}
        limit = dataclasses.replace(
            limit,
            minimum=parse_size(bounds["min"]) if "min" in bounds else None,
            maximum=parse_size(bounds["max"]) if "max" in bounds else None,
        )
        return _checked(limit)
    if value == "auto":
        return "auto"
    if isinstance(value, str):
        text = value.strip()
        if ".." in text:
            low, high = (_optional_size(part) for part in text.split("..", 1))
            if low is None and high is None:
                raise InvalidArgument("a budget range needs a minimum, a maximum or both: 2GB..6GB")
            return _checked(Limit(minimum=low, maximum=high))
        if text.startswith("-"):
            return Limit(reserve=parse_size(text[1:]))
        if text.endswith("!"):
            return Limit(use=parse_size(text[:-1]), force=True)
    fraction = _fraction(value)
    if fraction is not None:
        return fraction
    return parse_size(value)


def _checked(limit: Limit) -> PoolValue:
    if limit.minimum is not None and limit.maximum is not None and limit.minimum > limit.maximum:
        raise InvalidArgument(
            f"budget minimum {limit.minimum} is larger than the maximum {limit.maximum}"
        )
    return _simple(limit)


def _budget(value: Any) -> PoolValue | PoolBudget:
    """Parse a budget: one pool value for device and host (see `_pool_value`), or settings per
    pool ({"device": "80%", "host": "-2GB", "disk": "20GB"} or "device=80%,host=-2GB")."""
    if isinstance(value, PoolBudget):
        return value
    if isinstance(value, str) and "=" in value:
        value = dict(item.split("=", 1) for item in value.split(",") if item.strip())
        value = {k.strip(): v.strip() for k, v in value.items()}
    if isinstance(value, Mapping) and not set(value) & {"use", "min", "max"}:
        unknown = set(value) - set(POOLS)
        if unknown:
            raise InvalidArgument(f"budget pools are {', '.join(POOLS)}; got {sorted(unknown)}")
        return PoolBudget(**{k: _pool_value(v) for k, v in value.items()})
    return _pool_value(value)


def _headroom(value: Any) -> float | int:
    """A fraction (0.1, "10%") or a size ("1GB", or an integer number of bytes >= 1)."""
    if isinstance(value, bool):
        raise InvalidArgument(f"headroom must be a fraction or a size; got {value!r}")
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("%"):
            fraction = float(text[:-1]) / 100
        elif any(c.isalpha() for c in text):
            return parse_size(text)
        else:
            fraction = float(text)
    elif isinstance(value, int) and value >= 1:
        return value
    else:
        fraction = float(value)
    if not 0.0 <= fraction < 0.9:
        raise InvalidArgument(f"headroom must be in [0, 0.9) or a size; got {value!r}")
    return fraction


def _validate(key: str, value: Any) -> Any:
    if key not in _FIELDS:
        raise InvalidArgument(f"unknown setting {key!r}; known: {', '.join(sorted(_FIELDS))}")
    match key:
        case "budget":
            return _budget(value)
        case "budget_basis":
            return _choice(key, value, BUDGET_BASES)
        case "quality":
            return _choice(key, value, QUALITIES)
        case "prefer":
            return _choice(key, value, PREFERENCES)
        case "fallback":
            return _choice(key, value, FALLBACKS)
        case "residency":
            return _choice(key, value, RESIDENCIES)
        case "hibernate_modes":
            return _modes(value)
        case "disk_writes":
            return _choice(key, value, DISK_WRITES)
        case "daily_write_limit":
            return None if value in (None, "", "auto") else parse_size(value)
        case "headroom":
            return _headroom(value)
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
        case "idle_seconds":
            if value in (None, "", "none"):
                return None
            seconds = float(value)
            if seconds <= 0:
                raise InvalidArgument(f"idle_seconds must be > 0; got {value!r}")
            return seconds
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
        data = tomllib.loads(path.read_text(encoding="utf-8"))
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
    for layer in _scoped.get():
        merged.update(layer)
    return Config(**merged)


def configure(**settings: Any) -> Config:
    """Set runtime overrides (highest priority) and return the effective configuration.

    Example: ``memopro.configure(disk_writes="never", spill_dir="/Volumes/External/memopro")``
    """
    _overrides.update(_layer("configure()", settings))
    return get_config()


@contextlib.contextmanager
def using(**settings: Any) -> Iterator[Config]:
    """Change settings inside one ``with`` block only (0059 D5); above `configure()`.

    Example: ``with memopro.using(budget="3GB", quality="high"): model = memopro.load(...)``.
    Each thread and asyncio task sees its own blocks (contextvars).
    """
    token = _scoped.set((*_scoped.get(), _layer("using()", settings)))
    try:
        yield get_config()
    finally:
        _scoped.reset(token)


def reset_config() -> None:
    """Drop all runtime overrides set by `configure()`."""
    _overrides.clear()
