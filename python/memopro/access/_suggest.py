"""What would load a model that does not fit (0064 D-a): settings checked against the plan.

Nothing here guesses. Each candidate change of settings (the OS estimate as basis, a lower
quality, disk offload, or the exact forced budget a rejected configuration needs) is planned
again from the same metadata, and only changes under which a configuration is really chosen are
suggested, with how much of it would exceed memory free without compressing or swapping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from memopro._units import format_size
from memopro.config import QUALITIES

__all__ = ["Suggestion", "fallback_line", "over_free", "suggest_for_load", "suggestions_text"]

MAX_SUGGESTIONS = 4
# the lowest quality setting that allows each grade (QualityGrade.value -> quality)
_QUALITY_FOR_GRADE = {0: "lossless", 1: "high", 2: "balanced", 3: "low"}


@dataclass(frozen=True)
class Suggestion:
    settings: dict[str, str]
    config: str  # the configuration that would be chosen
    describe: str
    quality: str  # its quality grade
    needs: int
    over_free: int  # bytes above what is free without compressing or swapping (0: none)
    speed: int = field(default=0, compare=False)

    def call(self) -> str:
        return ", ".join(f"{k}={v!r}" for k, v in self.settings.items())


def _memory_needs(cfg: Any, unified_or_cpu: bool) -> tuple[int, int]:
    """(device bytes, host bytes) in memory; on unified memory or the CPU they share one pool."""
    if unified_or_cpu:
        return 0, cfg.needs.device + cfg.needs.host
    return cfg.needs.device, cfg.needs.host


def over_free(plan: Any, cfg: Any) -> int:
    """How much of ``cfg`` would not fit in memory free without compressing or swapping."""
    env = plan.setup.env
    shared = plan.ctx.unified or plan.ctx.device == "cpu"
    device_need, host_need = _memory_needs(cfg, shared)
    over = max(0, host_need - env.host.usable_bytes)
    if device_need:
        dev = next((d for d in env.devices if d.kind == plan.ctx.device), None)
        free = dev.available_bytes if dev is not None and dev.available_bytes else 0
        over += max(0, device_need - free)
    return over


def _forced_budget(cfg: Any, shared: bool) -> str:
    device_need, host_need = _memory_needs(cfg, shared)
    need = max(device_need, host_need)
    return f"{math.ceil(need / 1e8) / 10:.1f}GB!"


def _tries(plan: Any) -> list[dict[str, str]]:
    cfg = plan.setup.config
    current = QUALITIES.index(cfg.quality)
    lower = list(QUALITIES[current + 1 :])
    shared = plan.ctx.unified or plan.ctx.device == "cpu"
    tries: list[dict[str, str]] = []
    if cfg.budget_basis == "conservative":
        tries.append({"budget_basis": "os"})
    for q in lower:
        tries.append({"quality": q})
        if cfg.budget_basis == "conservative":
            tries.append({"quality": q, "budget_basis": "os"})
    if cfg.disk_writes != "allow":
        tries.append({"disk_writes": "allow"})
    for c in plan.candidates:
        if c.usable or c.name.startswith("offload"):
            continue  # offload is suggested through disk_writes, not by forcing memory
        blocked_by_quality = c.why.startswith("quality ")
        if not c.ok and not blocked_by_quality:
            continue  # not available here (back end, allow/deny)
        change = {"budget": _forced_budget(c, shared)}
        needed = _QUALITY_FOR_GRADE[c.quality.value]
        if QUALITIES.index(needed) > current:
            change = {"quality": needed, **change}
        tries.append(change)
    return tries


def suggest_for_load(plan: Any) -> list[Suggestion]:
    """Setting changes under which ``memopro.load`` would choose a configuration, best first."""
    from memopro._errors import MemoproError
    from memopro.access._load import plan_from_info

    if plan.setup is None:
        return []
    base = {k: v for k, v in plan.options.items() if v is not None}
    found: dict[tuple, Suggestion] = {}
    for change in _tries(plan):
        try:
            again = plan_from_info(plan.info, **(base | change))
        except (MemoproError, ValueError):
            continue
        chosen = again.chosen
        if chosen is None:
            continue
        over = over_free(again, chosen)
        if over and again.ctx.device == "cuda" and chosen.needs.device:
            dev = next((d for d in again.setup.env.devices if d.kind == "cuda"), None)
            if dev is not None and chosen.needs.device > (dev.available_bytes or 0):
                continue  # beyond the GPU's free memory: it would run out of memory, not swap
        s = Suggestion(
            settings=change,
            config=chosen.name,
            describe=chosen.describe(),
            quality=chosen.quality.name.lower(),
            needs=chosen.needs.device + chosen.needs.host + chosen.needs.disk,
            over_free=over,
            speed=chosen.speed,
        )
        key = (s.config, tuple(sorted(change)))
        if key not in found:
            found[key] = s
    # what fits in free memory first (least loss, fastest, fewest changes); past that, the least
    # swapping first: E014 showed a model far beyond free memory stalls the machine (0063)
    ranked = sorted(
        found.values(),
        key=lambda s: (s.over_free > 0, s.over_free, _grade(s.quality), s.speed, len(s.settings)),
    )
    # one suggestion per configuration: the simplest way to get it
    out, seen = [], set()
    for s in ranked:
        if s.config not in seen:
            seen.add(s.config)
            out.append(s)
    return out[:MAX_SUGGESTIONS]


def _grade(name: str) -> int:
    return {"lossless": 0, "near_lossless": 1, "small_loss": 2, "moderate_loss": 3}.get(name, 9)


def fallback_line(plan: Any) -> str:
    """The `fallback` options that would apply: "stored" (0064 D-d) and "stream" (0124)."""
    if plan.setup is None or plan.setup.config.fallback != "none":
        return ""
    lines = []
    stored = next((c for c in plan.candidates if c.name == "stored" and c.ok), None)
    if stored is not None:
        over = over_free(plan, stored)
        size = format_size(stored.needs.device + stored.needs.host)
        risk = f", about {format_size(over)} over what is free" if over else ""
        lines.append(f"  fallback='stored'  -> warn and load as stored anyway ({size}{risk})")
    if plan.ctx.host_budget:
        lines.append(
            "  fallback='stream'  -> lossless on the CPU: the stored weights stream from their "
            f"files within {format_size(int(plan.ctx.host_budget))} (slower)"
        )
    return "\n".join(lines)


def suggestions_text(source: str, suggestions: list[Suggestion], fallback: str = "") -> str:
    if not suggestions and not fallback:
        return "No setting change makes it fit here; free memory or use a smaller model."
    lines = [f"Settings that would load {source} (checked against the plan, nothing loaded):"]
    for s in suggestions:
        risk = (
            f"about {format_size(s.over_free)} over what is free, may compress or swap"
            if s.over_free
            else "fits in free memory"
        )
        lines.append(
            f"  {s.call():<38} -> {s.describe} ({s.quality}), {format_size(s.needs)}, {risk}"
        )
    if fallback:
        lines.append(fallback)
    return "\n".join(lines)
