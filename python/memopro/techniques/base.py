"""How a technique memopro applies is described: stage, fidelity, timing, pools, origin."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Stage(Enum):
    """Workflow stage a technique applies to."""

    DEV = "dev"
    TRAIN = "train"
    INFER = "infer"


class Fidelity(Enum):
    """How a technique affects results (architecture U5)."""

    EXACT = "exact"  # mathematically equivalent: may be applied automatically
    NUMERICS = "numerics"  # changes numerics: only within the user's chosen quality grade
    SEMANTICS = "semantics"  # changes what is trained/computed: suggest only, never auto-apply


class Timing(Enum):
    """When a technique takes effect (architecture section 3.4)."""

    LOAD_TIME = "load_time"  # decided before loading; changing it means reloading
    RUN_TIME = "run_time"  # applied to live objects; must support revert()


class Pool(Enum):
    """Memory pool a technique relieves (architecture section 3.2)."""

    DEVICE = "device"
    HOST = "host"
    DISK = "disk"


class Origin(Enum):
    MEMOPRO_NATIVE = "memopro_native"
    INTEGRATION = "integration"


class QualityGrade(Enum):
    """Declared quality impact, ordered from none to largest."""

    LOSSLESS = 0
    NEAR_LOSSLESS = 1
    SMALL_LOSS = 2
    MODERATE_LOSS = 3


@dataclass(frozen=True)
class Availability:
    """Whether a technique can run in the current environment, and why not if it cannot."""

    ok: bool
    reason: str = ""
