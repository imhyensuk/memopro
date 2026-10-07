"""Activations saved for backward go into runtime buffers when the process nears its ceiling (0229).

Installed by `memopro.enable` through ``torch.autograd.graph.saved_tensors_hooks``, so any
PyTorch model trains unchanged: while the process occupies less than ``pressure`` of the
ceiling, a saved tensor stays as it is (no copy, no cost); above that, each saved activation of
``threshold`` bytes or more is copied byte for byte into a :class:`memopro.rt.Runtime` buffer
and its own memory is let go. The runtime compresses idle buffers losslessly when room is
needed, and backward gets the exact bytes back, so gradients do not change.

Parameters and other leaf tensors are never moved (their modules hold them anyway), nor are
non-contiguous or sparse tensors. If the runtime cannot take a tensor, it stays as it is.
Prior art: PyTorch ``save_on_cpu``, activation offloading and compression; no novelty claim.
"""

from __future__ import annotations

import weakref
from typing import Any

from memopro._errors import MemoproError

__all__ = ["SavedActivations"]

PRESSURE = 0.75
THRESHOLD = 1 << 20


class _Packed:
    __slots__ = ("__weakref__", "buf", "device", "dtype", "shape")

    def __init__(self, buf: Any, shape: Any, dtype: Any, device: Any) -> None:
        self.buf, self.shape, self.dtype, self.device = buf, shape, dtype, device


class SavedActivations:
    """The pack/unpack pair; ``hooks()`` is the context manager that installs it."""

    def __init__(
        self, runtime: Any, ceiling: int, *, pressure: float = PRESSURE, threshold: int = THRESHOLD
    ) -> None:
        self.runtime = runtime
        self.ceiling = ceiling
        self.pressure = pressure
        self.threshold = threshold
        self.moved = 0
        self.moved_bytes = 0

    def hooks(self) -> Any:
        import torch

        return torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack)

    def _under_pressure(self) -> bool:
        from memopro.rt import process_footprint

        now = process_footprint()
        return now is not None and now >= self.pressure * self.ceiling

    def pack(self, t: Any) -> Any:
        import torch

        if (
            t.is_leaf
            or t.layout != torch.strided
            or not t.is_contiguous()
            or t.numel() * t.element_size() < self.threshold
            or not self._under_pressure()
        ):
            return t
        nbytes = t.numel() * t.element_size()
        size = t.element_size()
        try:
            buf = self.runtime.alloc(
                nbytes, dtype=f"uint{8 * size}" if size in (1, 2, 4, 8) else "uint8"
            )
            raw = t.detach().reshape(-1).view(torch.uint8).cpu().numpy()
            with buf.view(write=True) as a:
                a.reshape(-1).view("uint8")[...] = raw
        except MemoproError:
            return t
        p = _Packed(buf, t.shape, t.dtype, t.device)
        weakref.finalize(p, buf.free)
        self.moved += 1
        self.moved_bytes += nbytes
        return p

    def unpack(self, p: Any) -> Any:
        import torch

        if not isinstance(p, _Packed):
            return p
        out = torch.empty(p.shape, dtype=p.dtype)
        with p.buf.view() as a:
            out.reshape(-1).view(torch.uint8).numpy()[...] = a.reshape(-1).view("uint8")
        return out if p.device.type == "cpu" else out.to(p.device)
