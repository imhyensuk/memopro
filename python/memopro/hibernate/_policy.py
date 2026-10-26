"""Which hibernation methods may run, in which order (0032 H1, H2, H6).

This is pure policy: it looks only at the requested mode and the configuration, never at the
object. Whether a method actually fits an object (e.g. `source` needs an unchanged tensor with a
known original file) is decided later and reported as `ModeUnavailable`.
"""

from __future__ import annotations

from dataclasses import dataclass

from memopro._errors import InvalidArgument, PolicyError
from memopro.config import Config

MODES = ("auto", "source", "host", "compress")
# removed in 0.2.0 (0328): spill wrote to the disk (rule 1), bf16 changed the values (rule 4)
REMOVED_MODES = {
    "spill": "it wrote to the disk; memopro writes nothing besides downloads",
    "bf16": "it changed the values; memopro never changes results",
}


@dataclass(frozen=True)
class Step:
    mode: str


def resolve_modes(requested: str, config: Config) -> list[Step]:
    """Return the methods to try, in order: ``auto`` tries the configured ones (all lossless and
    write-free); an explicit mode is tried alone (memopro does not fall back on its own)."""
    if requested in REMOVED_MODES:
        raise PolicyError(f"mode {requested!r} was removed in 0.2.0: {REMOVED_MODES[requested]}")
    if requested not in MODES:
        raise InvalidArgument(f"unknown mode {requested!r}; choose from {', '.join(MODES)}")
    if requested == "auto":
        return [Step(m) for m in config.hibernate_modes]
    return [Step(requested)]
