"""Collect tensors per category, sample them, and build the census result."""

from __future__ import annotations

import random
import statistics
from collections import defaultdict
from typing import Any

import torch

from memopro.census import _stats
from memopro.census._advice import advise

CATEGORIES = ("parameters", "gradients", "optimizer_state", "saved_activations")


def _key(t: torch.Tensor) -> tuple[str, int]:
    return (str(t.device), t.untyped_storage().data_ptr())


def _sample(t: torch.Tensor, elements: int) -> torch.Tensor:
    """Evenly spaced 1-D CPU sample of ``t`` without copying the whole tensor."""
    n = t.numel()
    if n == 0:
        return torch.empty(0, dtype=t.dtype)
    if n <= elements:
        return t.detach().reshape(-1).to("cpu", copy=True)
    t = t.detach()
    if t.is_contiguous():
        step = n // elements
        return t.reshape(-1)[::step][:elements].cpu()  # strided view: copies only the sample
    # works on every backend (torch.take is not implemented on MPS)
    flat = torch.linspace(0, n - 1, elements, device=t.device).long()
    return t[torch.unravel_index(flat, t.shape)].cpu()


class SavedTensorLog:
    """``saved_tensors_hooks`` pack hook: counts storages once, reservoir-samples tensors."""

    def __init__(self, k: int, elements: int, rng: random.Random) -> None:
        self.k = k
        self.elements = elements
        self.rng = rng
        self.entries: dict[tuple[str, int], torch.Tensor] = {}  # key -> representative tensor meta
        self.nbytes: dict[tuple[str, int], int] = {}
        self.samples: list[tuple[tuple[str, int], torch.Tensor]] = []
        self.seen = 0

    def pack(self, t: torch.Tensor) -> torch.Tensor:
        if not isinstance(t, torch.Tensor) or t.untyped_storage().data_ptr() == 0:
            return t
        key = _key(t)
        if key in self.nbytes:
            return t
        self.nbytes[key] = t.untyped_storage().nbytes()
        self.entries[key] = torch.empty(0, dtype=t.dtype, device="meta")
        self.seen += 1
        if len(self.samples) < self.k:
            self.samples.append((key, _sample(t, self.elements)))
        else:
            j = self.rng.randrange(self.seen)
            if j < self.k:
                self.samples[j] = (key, _sample(t, self.elements))
        return t


def cublas_workspace_bytes() -> int | None:
    """CUDA memory held by cuBLAS/cuBLASLt workspaces, which PyTorch allocates through its
    caching allocator (0045 F2): the drop in ``memory_allocated`` when they are cleared.

    Clearing is safe: PyTorch recreates a workspace at the next matrix multiply. None when the
    private hook is missing; 0 without CUDA.
    """
    if not torch.cuda.is_available():
        return 0
    clear = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
    if clear is None:
        return None
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    clear()
    return max(0, before - torch.cuda.memory_allocated())


def allocator_snapshot() -> dict[str, dict[str, int]]:
    from memopro import _core

    snap: dict[str, dict[str, int]] = {"cpu": {"rss": _core.hwinfo_process_rss()}}
    if torch.cuda.is_available():
        snap["cuda"] = {
            "allocated": sum(
                torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())
            ),
            "peak": sum(
                torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())
            ),
        }
    from memopro.env._torch import mps_usable

    if mps_usable():
        snap["mps"] = {
            "allocated": int(torch.mps.current_allocated_memory()),
            "driver": int(torch.mps.driver_allocated_memory()),
        }
    return snap


class _Category:
    def __init__(self) -> None:
        self.tensors: dict[tuple[str, int], torch.Tensor] = {}
        self.nbytes: dict[tuple[str, int], int] = {}

    def add(self, t: torch.Tensor) -> None:
        if t.untyped_storage().data_ptr() == 0:
            return
        key = _key(t)
        if key not in self.nbytes:
            self.nbytes[key] = t.untyped_storage().nbytes()
            self.tensors[key] = t


def _pick(tensors: list[torch.Tensor], k: int) -> list[torch.Tensor]:
    """Up to k tensors: the largest plus evenly spaced ones in model order."""
    if len(tensors) <= k:
        return tensors
    largest = max(tensors, key=lambda t: t.numel())
    step = len(tensors) / (k - 1)
    picked = [tensors[int(i * step)] for i in range(k - 1)]
    return [largest, *[t for t in picked if t is not largest]]


def _summarise(
    name: str,
    nbytes: dict[tuple[str, int], int],
    meta: dict[tuple[str, int], torch.Tensor],
    samples: list[torch.Tensor],
    light: bool,
) -> dict[str, Any]:
    devices: dict[str, int] = defaultdict(int)
    dtypes: dict[str, int] = defaultdict(int)
    for key, n in nbytes.items():
        devices[key[0].split(":")[0]] += n
        dtypes[str(meta[key].dtype).removeprefix("torch.")] += n
    stats = [_stats.analyse(s, light) for s in samples if s.numel()]
    out: dict[str, Any] = {
        "bytes": sum(nbytes.values()),
        "tensors": len(nbytes),
        "devices": dict(devices),
        "dtypes": dict(dtypes),
        "sampled": len(stats),
    }
    if stats:
        entropy = statistics.median(s.entropy_bits for s in stats)
        stored = statistics.median(s.stored_bits for s in stats)
        ratios = [s.stored_bits / s.entropy_bits for s in stats if s.entropy_bits > 0]
        peaks = [s.peak_to_median for s in stats if s.peak_to_median is not None]
        zeros = [s.zero_fraction for s in stats if s.zero_fraction is not None]
        out |= {
            "stored_bits": stored,
            "entropy_bits": round(entropy, 3),
            "lossless_ratio": round(statistics.median(ratios), 3) if ratios else None,
            "massive_tensors": sum(1 for p in peaks if p >= _stats.MASSIVE_RATIO),
            "max_peak_to_median": round(max(peaks), 1) if peaks else None,
            "zero_fraction": round(statistics.median(zeros), 4) if zeros else None,
        }
        if light:
            errors = [s.quant_rel_error for s in stats if s.quant_rel_error is not None]
            out["quant_error"] = (
                {
                    str(b): {
                        "median": round(statistics.median(e[b] for e in errors), 5),
                        "worst": round(max(e[b] for e in errors), 5),
                    }
                    for b in _stats.BITS
                }
                if errors
                else None
            )
            out["needed_bits"] = _stats.needed_bits(errors)
    return out


def build_result(census: Any) -> dict[str, Any]:
    light = census.mode in ("light", "deep")
    model, optimizer, log = census.model, census.optimizer, census._saved
    cats = {c: _Category() for c in CATEGORIES}

    params: list[torch.Tensor] = []
    if model is not None:
        params = list(model.parameters()) + list(model.buffers())
    elif optimizer is not None:
        params = [p for g in optimizer.param_groups for p in g["params"]]
    for p in params:
        cats["parameters"].add(p)
    for p in params:
        if getattr(p, "grad", None) is not None:
            cats["gradients"].add(p.grad)
    if optimizer is not None:
        for state in optimizer.state.values():
            for v in state.values():
                if isinstance(v, torch.Tensor):
                    cats["optimizer_state"].add(v)

    live_keys = set().union(*(c.nbytes for c in cats.values()))
    saved = cats["saved_activations"]
    for key, n in log.nbytes.items():
        if key not in live_keys:
            saved.nbytes[key] = n
            saved.tensors[key] = log.entries[key]

    categories = {}
    for name, cat in cats.items():
        if name == "saved_activations":
            samples = [s for key, s in log.samples if key not in live_keys]
        else:
            samples = [
                _sample(t, census._elements) for t in _pick(list(cat.tensors.values()), census._k)
            ]
        categories[name] = _summarise(name, cat.nbytes, cat.tensors, samples, light)

    after = allocator_snapshot()
    workspace = cublas_workspace_bytes()
    coverage, unclassified = {}, {}
    live_by_device: dict[str, int] = defaultdict(int)
    for name in ("parameters", "gradients", "optimizer_state"):
        for dev, n in categories[name]["devices"].items():
            live_by_device[dev] += n
    live_by_device["cuda"] += workspace or 0
    for dev in ("cuda", "mps"):
        if dev in after and after[dev]["allocated"] > 0:
            allocated = after[dev]["allocated"]
            coverage[dev] = round(min(1.0, live_by_device.get(dev, 0) / allocated), 4)
            unclassified[dev] = max(0, allocated - live_by_device.get(dev, 0))

    result: dict[str, Any] = {
        "mode": census.mode,
        "categories": categories,
        "allocator_before": census._alloc_before,
        "allocator_after": after,
        "framework_workspace": {"cuda": workspace} if workspace else {},
        "coverage_at_end": coverage,
        "unclassified_at_end": unclassified,
        "criteria": {
            "needed_bits_basis": "blockwise-absmax reconstruction error, not gradient impact",
            "tolerance": _stats.TOLERANCE,
            "block": _stats.BLOCK,
            "massive_ratio": _stats.MASSIVE_RATIO,
            "sample_elements": census._elements,
            "samples_per_category": census._k,
        },
    }
    result["advice"] = advise(result)
    return result
