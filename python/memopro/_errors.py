"""Exception hierarchy (docs/research/0034).

Every error memopro raises derives from `MemoproError`, so callers can catch memopro's own
problems without catching their own. Features that exist only as a skeleton raise
`NotYetImplemented`, which names the planned version instead of silently doing nothing.
"""

from __future__ import annotations


class MemoproError(Exception):
    """Base class for all memopro errors."""


class InvalidArgument(MemoproError, ValueError):
    """An argument or configuration value is not valid."""


class ConfigError(InvalidArgument):
    """A configuration source (memopro.toml, MEMOPRO_* variable, configure()) is invalid."""


class PolicyError(MemoproError):
    """The requested action is forbidden by the active policy (e.g. SSD writes are disabled)."""


class ModeUnavailable(MemoproError):
    """The requested method cannot be used for this object; carries the reason and alternatives.

    memopro never switches to another method silently when the user chose one (0032 H6).
    """

    def __init__(self, mode: str, reason: str, alternatives: tuple[str, ...] = ()) -> None:
        self.mode = mode
        self.reason = reason
        self.alternatives = alternatives
        hint = f" (other modes to try: {', '.join(alternatives)})" if alternatives else ""
        super().__init__(f"mode {mode!r} is not available: {reason}{hint}")


class IntegrityError(MemoproError):
    """Restored data does not match what was hibernated (changed source file, corrupted spill)."""


class NotYetImplemented(MemoproError, NotImplementedError):
    """The feature is part of the designed API but is not built yet."""

    def __init__(self, feature: str, planned: str, ref: str) -> None:
        self.feature = feature
        self.planned = planned
        self.ref = ref
        super().__init__(f"{feature} is not implemented yet (planned: {planned}; design: {ref})")
