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
import tomllib
from collections.abc import Iterator, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memopro._errors import ConfigError, InvalidArgument
from memopro._units import parse_size

# a model streamed by name and not on this disk is downloaded once into the Hugging Face cache
# ("auto", 0303) or refused ("never")
DOWNLOADS = ("auto", "never")
# What memopro does when data it holds must come back and the ceiling has no room for it (0320):
# stop with BudgetExceeded ("error"), or give the data back over the ceiling with a warning
# ("allow"; e.g. activations needed by backward, so a training run is not lost)
OVER_CEILINGS = ("error", "allow")
POOLS = ("device", "host")
# What load does when the model does not fit the budget as stored (0124, 0328): stream the
# stored weights from their files (memopro.rt), or stop
FALLBACKS = ("stream", "none")


# One pool's setting (0327): "auto" (80% of the machine's memory), bytes, or a fraction of it
PoolValue = str | int | float

# the default ceiling: this share of the machine's memory (0325 rule 2)
AUTO_FRACTION = 0.8


@dataclass(frozen=True)
class PoolBudget:
    """Settings per pool (0052 E3, 0059 D2); None leaves that pool at "auto"."""

    device: PoolValue | None = None
    host: PoolValue | None = None


@dataclass(frozen=True)
class Config:
    # one pool value for device and host (see PoolValue), or settings per pool
    budget: PoolValue | PoolBudget = "auto"
    fallback: str = "stream"
    download: str = "auto"
    over_ceiling: str = "error"
    # folders where the user keeps models (each one a model folder, or a folder of them):
    # generate(draft="auto") looks there first for a draft (0324)
    model_dirs: tuple[str, ...] = ()


_FIELDS = {f.name for f in dataclasses.fields(Config)}
_overrides: dict[str, Any] = {}
_scoped: ContextVar[tuple[dict[str, Any], ...]] = ContextVar("memopro_settings", default=())


def _choice(key: str, value: Any, choices: tuple[str, ...]) -> str:
    if value not in choices:
        raise InvalidArgument(f"{key} must be one of {', '.join(choices)}; got {value!r}")
    return value


def _dirs(value: Any) -> tuple[str, ...]:
    """Folders as a list, or one string with the platform's path separator (``:`` on macOS and
    Linux, ``;`` on Windows), as in ``MEMOPRO_MODEL_DIRS``."""
    import os

    items = value.split(os.pathsep) if isinstance(value, str) else list(value)
    return tuple(str(d).strip() for d in items if str(d).strip())


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


_OLD_FORMS = (
    'budget is "auto" (80% of this machine\'s memory), a size such as "6GB", or a share of the '
    'memory such as "50%" or 0.5, for all pools or per pool ("device=8GB,host=24GB"); the '
    "value is the ceiling as given (0327)"
)


def _pool_value(value: Any) -> PoolValue:
    """Parse one pool's budget (0327): "auto", "6GB" (exactly that), 0.5 / "50%" (of the
    machine's memory). Forms removed in 0.2.0 ("-2GB", "2GB..6GB", "6GB!", {use, min, max})
    say how to write them now."""
    if value == "auto":
        return "auto"
    if isinstance(value, Mapping):
        raise InvalidArgument(f"budget minimums and maximums were removed: {_OLD_FORMS}")
    if isinstance(value, str):
        text = value.strip()
        if ".." in text or text.startswith("-") or text.endswith("!"):
            raise InvalidArgument(f"{value!r} is a budget form removed in 0.2.0: {_OLD_FORMS}")
    fraction = _fraction(value)
    if fraction is not None:
        return fraction
    return parse_size(value)


def _budget(value: Any) -> PoolValue | PoolBudget:
    """Parse a budget: one pool value for device and host (see `_pool_value`), or settings per
    pool ({"device": "8GB", "host": "24GB"} or "device=8GB,host=24GB")."""
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


# settings removed in 0.2.0 (0327): what to do instead
_LOSSY = (
    "memopro never changes results; to run a half-precision or quantized model, pass it as "
    "your original: memopro.load(name, dtype=..., quantization_config=...) (0328)"
)
_DISK = "memopro writes nothing to the disk besides downloads (0325 rule 1, 0328)"
_HIBERNATE = (
    "hibernation was removed in 0.2.0; memopro.enable() keeps idle memory within the ceiling"
)
_REMOVED = {
    "hibernate_modes": _HIBERNATE,
    "idle_cells": _HIBERNATE,
    "idle_seconds": _HIBERNATE,
    "budget_basis": "the budget is now the ceiling as given, from the machine's memory",
    "headroom": 'the default "auto" already leaves 20% of the memory free; give a smaller budget',
    "quality": _LOSSY,
    "prefer": _LOSSY,
    "residency": _DISK,
    "disk_writes": _DISK,
    "spill_dir": _DISK,
    "daily_write_limit": _DISK,
    "min_free_disk_fraction": _DISK,
}


def _validate(key: str, value: Any) -> Any:
    if key in _REMOVED:
        raise InvalidArgument(f"the setting {key!r} was removed in 0.2.0: {_REMOVED[key]}")
    if key not in _FIELDS:
        raise InvalidArgument(f"unknown setting {key!r}; known: {', '.join(sorted(_FIELDS))}")
    match key:
        case "budget":
            return _budget(value)
        case "fallback":
            return _choice(key, value, FALLBACKS)
        case "download":
            return _choice(key, value, DOWNLOADS)
        case "over_ceiling":
            return _choice(key, value, OVER_CEILINGS)
        case "model_dirs":
            return _dirs(value)
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

    Example: ``memopro.configure(budget="6GB", download="never")``
    """
    _overrides.update(_layer("configure()", settings))
    return get_config()


@contextlib.contextmanager
def using(**settings: Any) -> Iterator[Config]:
    """Change settings inside one ``with`` block only (0059 D5); above `configure()`.

    Example: ``with memopro.using(budget="3GB"): model = memopro.load(...)``.
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
