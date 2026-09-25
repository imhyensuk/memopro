"""Byte-size parsing and formatting ("20GB", "512MiB", 1_000_000)."""

from __future__ import annotations

import re

from memopro._errors import InvalidArgument

_UNITS = {
    "": 1,
    "b": 1,
    "kb": 10**3,
    "mb": 10**6,
    "gb": 10**9,
    "tb": 10**12,
    "kib": 2**10,
    "mib": 2**20,
    "gib": 2**30,
    "tib": 2**40,
}
_SIZE = re.compile(r"\s*(\d+(?:\.\d+)?)\s*([a-zA-Z]*)\s*")


def parse_size(value: str | float) -> int:
    """Parse a byte size. Decimal units (GB) are powers of 10, binary units (GiB) powers of 2."""
    if isinstance(value, bool):
        raise InvalidArgument(f"not a size: {value!r}")
    if isinstance(value, int | float):
        if value < 0:
            raise InvalidArgument(f"size must not be negative: {value!r}")
        return int(value)
    match = _SIZE.fullmatch(value)
    if match is None or match.group(2).lower() not in _UNITS:
        raise InvalidArgument(f"not a size: {value!r} (examples: 512MiB, 20GB, 1073741824)")
    return int(float(match.group(1)) * _UNITS[match.group(2).lower()])


def format_size(nbytes: int) -> str:
    """Human-readable binary size, e.g. 2254857830 -> '2.10 GiB'."""
    size = float(nbytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(size) < 1024 or unit == "GiB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.2f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")
