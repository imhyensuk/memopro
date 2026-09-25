"""Which hibernation methods may run, in which order (0032 H1, H2, H6).

This is pure policy: it looks only at the requested mode and the configuration, never at the
object. Whether a method actually fits an object (e.g. `source` needs an unchanged tensor with a
known original file) is decided later and reported as `ModeUnavailable`.
"""

from __future__ import annotations

from dataclasses import dataclass

from memopro._errors import InvalidArgument, PolicyError
from memopro.config import Config

MODES = ("auto", "source", "host", "compress", "bf16", "spill")


@dataclass(frozen=True)
class Step:
    mode: str
    needs_confirmation: bool = False  # SSD write under disk_writes="ask" without consent


def resolve_modes(requested: str, config: Config, *, allow_spill: bool = False) -> list[Step]:
    """Return the methods to try, in order.

    - ``auto`` tries the configured write-free methods, never ``bf16``, and ends with ``spill``
      unless ``disk_writes="never"``.
    - An explicit mode is tried alone; memopro does not fall back to another method on its own.
    - ``spill`` under ``disk_writes="ask"`` needs consent (``allow_spill=True``, ``--spill``);
      ``disk_writes="never"`` refuses it whatever was requested.
    """
    if requested not in MODES:
        raise InvalidArgument(f"unknown mode {requested!r}; choose from {', '.join(MODES)}")
    ask = config.disk_writes == "ask" and not allow_spill
    if requested == "auto":
        steps = [Step(m) for m in config.hibernate_modes]
        if config.disk_writes != "never":
            steps.append(Step("spill", needs_confirmation=ask))
        return steps
    if requested == "spill":
        if config.disk_writes == "never":
            raise PolicyError("SSD writes are disabled (disk_writes='never'); 'spill' refused")
        return [Step("spill", needs_confirmation=ask)]
    return [Step(requested)]
