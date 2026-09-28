"""``memopro doctor`` / ``memopro.doctor()``: available memory per pool and the resulting budget.

Everything shown is measured now; nothing is changed. ``devices=False`` skips torch.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from memopro._units import format_size
from memopro.config import Config, get_config
from memopro.env import (
    MALLOC_CACHE_NOTE,
    MPS_HEAP_NOTE,
    Env,
    detect,
    macos_malloc_cache_on,
    mps_heap_reserve_on,
)
from memopro.orchestrator.budget import Budget, compute_budget, describe_setting

__all__ = ["DoctorReport", "doctor"]

# The macOS estimate counts other apps' active memory as available; mention the gap when large.
_KERNEL_GAP_NOTE = 0.10


@dataclass(frozen=True)
class DoctorReport:
    version: str
    env: Env
    budget: Budget
    config: Config

    def warnings(self) -> list[str]:
        env, budget, cfg = self.env, self.budget, self.config
        out = []
        host = env.host
        if host.swap_used_bytes > 0:
            out.append(f"swap in use: {format_size(host.swap_used_bytes)}")
        gap = host.kernel_available_bytes - host.available_bytes
        if gap > _KERNEL_GAP_NOTE * host.total_bytes and cfg.budget_basis == "conservative":
            out.append(
                f"the OS reports {format_size(host.kernel_available_bytes)} available, including "
                "memory other apps are actively using; memopro budgets with the conservative "
                f"{format_size(host.available_bytes)} (budget_basis='os' counts the larger number)"
            )
        if env.disk.free_fraction < cfg.min_free_disk_fraction:
            out.append(
                f"free disk space is below the {cfg.min_free_disk_fraction:.0%} floor: memopro "
                "will not spill to this disk"
            )
        if budget.capped_by_setting:
            out.append(f"budget capped by the budget setting ({describe_setting(cfg.budget)})")
        if macos_malloc_cache_on():
            out.append(MALLOC_CACHE_NOTE)
        if mps_heap_reserve_on():
            out.append(MPS_HEAP_NOTE)
        return out + list(budget.notes) + list(env.notes)

    def to_json(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "env": dataclasses.asdict(self.env),
            "budget": dataclasses.asdict(self.budget),
            "policy": {
                "disk_writes": self.config.disk_writes,
                "hibernate_modes": list(self.config.hibernate_modes),
                "min_free_disk_fraction": self.config.min_free_disk_fraction,
                "headroom": self.config.headroom,
                "budget": describe_setting(self.config.budget),
                "budget_basis": self.config.budget_basis,
            },
            "warnings": self.warnings(),
        }

    def summary(self) -> str:
        env, budget, cfg = self.env, self.budget, self.config
        h, d = env.host, env.disk
        cpu = f", {env.cpu_brand} ({env.logical_cpus} logical CPUs)" if env.cpu_brand else ""
        lines = [
            f"memopro doctor {self.version} - {env.os_version} {env.machine}{cpu}",
            "",
            "Host memory",
            f"  total          {format_size(h.total_bytes)}",
            f"  available      {format_size(h.available_bytes)}  (without compressing or swapping)",
        ]
        if h.cgroup_limit_bytes is not None:
            lines.append(
                f"  container      limit {format_size(h.cgroup_limit_bytes)}, "
                f"free {format_size(h.cgroup_free_bytes or 0)}"
            )
        else:
            lines.append("  container      no limit")
        lines.append(
            f"  swap           {format_size(h.swap_used_bytes)} used of "
            f"{format_size(h.swap_total_bytes)}"
        )
        lines += [
            "",
            f"Disk (spill location {d.spill_dir})",
            (
                f"  free           {format_size(d.available_bytes)} of "
                f"{format_size(d.total_bytes)} ({d.free_fraction:.1%})"
            ),
            "",
            "Devices",
        ]
        if not env.devices:
            lines.append("  none")
        for dev in env.devices:
            parts = [f"  {dev.kind:<5} {dev.name}"]
            if dev.unified:
                parts.append("unified memory")
            if dev.limit_bytes is not None:
                parts.append(f"limit {format_size(dev.limit_bytes)}")
            if dev.total_bytes is not None:
                parts.append(f"total {format_size(dev.total_bytes)}")
            if dev.available_bytes is not None:
                parts.append(f"free {format_size(dev.available_bytes)}")
            lines.append(", ".join(parts))
        shared = " (same physical memory as host)" if budget.unified else ""
        device = "-" if budget.device is None else format_size(budget.device) + shared
        lines += [
            "",
            f"Budget ({_basis(budget.basis)}, headroom {_headroom(budget.headroom)})",
            f"  setting        {describe_setting(cfg.budget)}",
            f"  device         {device}",
            f"  host           {format_size(budget.host)}",
            (
                f"  disk           {format_size(budget.disk)}  (above the "
                f"{cfg.min_free_disk_fraction:.0%} free-space floor)"
            ),
            "",
            "Policy",
            f"  disk_writes    {cfg.disk_writes}"
            + {
                "ask": "  (SSD spill only with your consent)",
                "never": "  (SSD spill disabled)",
                "allow": "  (SSD spill allowed)",
            }[cfg.disk_writes],
            f"  auto methods   {' -> '.join(cfg.hibernate_modes)}",
        ]
        warnings = self.warnings()
        if warnings:
            lines += ["", "Notes"] + [f"  - {w}" for w in warnings]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def _headroom(value: float) -> str:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
        return format_size(value)
    return f"{value:.0%}"


def _basis(basis: str) -> str:
    return {
        "conservative": "from conservative available memory",
        "os": "from the OS estimate of available memory",
        "total": "from total physical memory",
    }[basis]


def doctor(*, devices: bool = True) -> DoctorReport:
    """Measure the environment and compute the per-pool budget."""
    from memopro import __version__

    cfg = get_config()
    env = detect(devices=devices, config=cfg)
    return DoctorReport(__version__, env, compute_budget(env, cfg), cfg)
