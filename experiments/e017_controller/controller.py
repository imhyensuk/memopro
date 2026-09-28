"""E017: a comfort controller for RCR F-class inference (0066 §5.3 ⑤). Research prototype.

Signals, per decision window (every `check_every` tokens):
  - neighbours: the system's swap growth rate. The pilot (0073 §1) showed the OS keeps our
    actively used clean pages resident and pushes idle neighbours' anonymous memory to swap, so
    the harm of a large working set shows up there, not in our own pages.
  - ourselves: the bytes we page back in (rusage), when our own pages are dropped (a working set
    beyond free memory, e.g. 3B bf16).
On the 8 GB M1 the OS pressure level is "warning" even when calm (E013 note), so it is not used.

Actions at token boundaries only (the safe points of 0052 E6), in the order of 0066 R4:
  1. precision first: switch from the bf16 file mapping to the int4 one (both mapped, both clean
     pages, so holding both costs nothing) when neighbours are swapping or we are paging in;
     switch back after a calm period when the OS has room for bf16 again. The KV cache is shared
     between the two (same architecture).
  2. pace: when already on int4 and still paging in, cap our reread rate so neighbours are not
     pushed to swap faster than the cap.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from experiments.e011_os_swap.common import rusage

PAGE = 16384
_libc = ctypes.CDLL(ctypes.util.find_library("c"))


class _XswUsage(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_uint64),
        ("avail", ctypes.c_uint64),
        ("used", ctypes.c_uint64),
        ("pagesize", ctypes.c_uint32),
        ("encrypted", ctypes.c_bool),
    ]


def swap_used() -> int:
    """System swap in use (sysctl vm.swapusage), without a subprocess."""
    usage = _XswUsage()
    size = ctypes.c_size_t(ctypes.sizeof(usage))
    _libc.sysctlbyname(b"vm.swapusage", ctypes.byref(usage), ctypes.byref(size), None, 0)
    return int(usage.used)


@dataclass
class Policy:
    check_every: int = 2  # tokens between decisions
    swap_mb_s: float = 20.0  # neighbours swapping faster than this: we take too much room
    pagein_mb: float = 64.0  # we paged in more than this in a window: our pages are dropped
    up_after_s: float = 10.0  # calm this long before stepping back up
    up_headroom: float = 1.2  # OS-estimated available must be this x the bf16 weights
    pace_mb_s: float = 200.0  # reread cap while paging in on the lowest precision
    pace: bool = True
    switch: bool = True


@dataclass
class Controller:
    models: dict[str, Any]  # "bf16" and/or "int4" -> model (file-backed)
    maps: dict[str, list[Any]]  # the same keys -> their Mapping objects
    weight_bytes: dict[str, int]
    policy: Policy = field(default_factory=Policy)
    active: str = ""
    log: list[dict] = field(default_factory=list)
    _calm_since: float = 0.0
    _pageins: int = 0
    _swap: int = 0
    _t: float = 0.0

    def __post_init__(self) -> None:
        self.active = self.active or ("bf16" if "bf16" in self.models else "int4")
        self.reset()
        self._calm_since = self._t

    def reset(self) -> None:
        """Start a fresh measurement window (after warm-up or between phases). The calm clock
        keeps running: calm time between phases counts toward stepping back up."""
        self._pageins = rusage(os.getpid())["pageins"]
        self._swap = swap_used()
        self._t = time.perf_counter()

    def step(self, token: int) -> None:
        """Called after each token; decides every `check_every` tokens."""
        if token % self.policy.check_every:
            return
        now = time.perf_counter()
        window = max(now - self._t, 1e-6)
        paged_mb = (rusage(os.getpid())["pageins"] - self._pageins) * PAGE / 1e6
        swap_mb = (swap_used() - self._swap) / 1e6
        crowding = swap_mb / window > self.policy.swap_mb_s
        thrashing = paged_mb > self.policy.pagein_mb
        if crowding or thrashing:
            self._calm_since = now
        action = ""
        if (
            self.policy.switch
            and self.active == "bf16"
            and "int4" in self.models
            and (crowding or thrashing)
        ):
            self.active, action = "int4", "down"
        elif (
            self.policy.switch
            and self.active == "int4"
            and "bf16" in self.models
            and now - self._calm_since >= self.policy.up_after_s
            and self._room_for("bf16")
        ):
            self.active, action = "bf16", "up"
        elif self.policy.pace and thrashing and paged_mb / window > self.policy.pace_mb_s:
            pause = min(paged_mb / self.policy.pace_mb_s - window, 2.0)
            if pause > 0:
                time.sleep(pause)
                action = f"pace {pause:.2f}s"
        self.log.append(
            {
                "token": token,
                "t": now,
                "active": self.active,
                "paged_mb": paged_mb,
                "swap_mb": swap_mb,
                "window_s": window,
                "action": action,
            }
        )
        self._pageins = rusage(os.getpid())["pageins"]
        self._swap = swap_used()
        self._t = time.perf_counter()

    def _room_for(self, key: str) -> bool:
        import memopro

        host = memopro.doctor(devices=False).env.host
        return host.kernel_available_bytes >= self.policy.up_headroom * self.weight_bytes[key]


def generate(
    controller: Controller | None, model: Any, ids: torch.Tensor, n: int, cache: Any = None
) -> tuple[list[int], Any]:
    """Greedy decoding with a shared KV cache; the controller may switch models between tokens.
    Returns the new tokens and the cache (to continue in a later phase)."""
    from transformers import DynamicCache

    first = cache is None
    cache = cache if cache is not None else DynamicCache()
    tokens: list[int] = []
    active = controller.models[controller.active] if controller else model
    with torch.no_grad():
        out = active(ids if first else ids[:, -1:], past_key_values=cache, use_cache=True)
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        for i in range(1, n + 1):
            tokens.append(int(nxt))
            if i == n:
                break
            if controller is not None:
                controller.step(i)
                active = controller.models[controller.active]
            out = active(nxt, past_key_values=cache, use_cache=True)
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    torch.mps.synchronize()
    return tokens, cache
