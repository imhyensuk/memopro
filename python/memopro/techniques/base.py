"""Common contract for every memory technique (docs/design/architecture.md, section 5).

Both memopro-native techniques (hibernate, census, rfc, elastic) and integrations of existing
techniques (quantization, offload, checkpointing, ...) implement the same `Technique` protocol,
so a new research result can be added as a single registry entry.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, TypeVar, runtime_checkable


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


@runtime_checkable
class Technique(Protocol):
    name: str
    stages: frozenset[Stage]
    fidelity: Fidelity
    timing: Timing
    pools: frozenset[Pool]
    origin: Origin
    quality: QualityGrade

    def available(self, env: Any) -> Availability: ...

    def estimate(self, target: Any, env: Any) -> dict[str, Any]: ...

    def apply(self, target: Any, env: Any) -> Any: ...

    def revert(self, applied: Any) -> None: ...

    def report(self, applied: Any) -> dict[str, Any]: ...


T = TypeVar("T")


class Registry:
    """Name -> technique factory. Third-party techniques will register via entry points later."""

    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], Technique]] = {}

    def register(self, name: str, factory: Callable[[], Technique]) -> None:
        if name in self._factories:
            raise ValueError(f"technique {name!r} is already registered")
        self._factories[name] = factory

    def unregister(self, name: str) -> None:
        self._factories.pop(name, None)

    def create(self, name: str) -> Technique:
        try:
            factory = self._factories[name]
        except KeyError:
            raise KeyError(f"unknown technique {name!r}; registered: {self.names()}") from None
        return factory()

    def names(self) -> list[str]:
        return sorted(self._factories)

    def __contains__(self, name: object) -> bool:
        return name in self._factories

    def __len__(self) -> int:
        return len(self._factories)


registry = Registry()


def register_technique(name: str) -> Callable[[type[T]], type[T]]:
    """Class decorator: register a zero-argument technique class under `name`."""

    def decorator(cls: type[T]) -> type[T]:
        registry.register(name, cls)  # type: ignore[arg-type]
        return cls

    return decorator
