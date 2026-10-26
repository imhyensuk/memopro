"""Activations saved for backward go into runtime buffers when the process nears its ceiling (0229).

Installed by `memopro.enable` through ``torch.autograd.graph.saved_tensors_hooks``, so any
PyTorch model trains unchanged: while the process occupies less than ``pressure`` of the
ceiling, a saved tensor stays as it is (no copy, no cost); above that, each saved activation of
``threshold`` bytes or more is copied byte for byte into a :class:`memopro.rt.Runtime` buffer
and its own memory is let go. The runtime compresses idle buffers losslessly when room is
needed, and backward gets the exact bytes back, so gradients do not change.

On the Apple GPU the copies never pass through the CPU (0321): the runtime buffer is wrapped
as an MPS tensor (no copy, unified memory) and the GPU copies the activation into it, and back
out of it for backward, queued like any other GPU work; no step waits for the GPU per tensor.
Wrapped buffers are given back to the runtime (to be compressed) in batches behind one MPS
event, as for streamed weights.

Parameters and other leaf tensors are never moved (their modules hold them anyway), nor are
non-contiguous or sparse tensors. If the runtime cannot take a tensor, it stays as it is. If
backward needs a tensor back and the ceiling has no room for it, ``over_ceiling`` decides
(0320): ``"error"`` raises :class:`memopro.BudgetExceeded` (the ceiling holds, the step is
lost), ``"allow"`` gives the exact bytes back over the ceiling with a warning (counted in
``over_ceiling_count``).
Prior art: PyTorch ``save_on_cpu``, activation offloading and compression; no novelty claim.
"""

from __future__ import annotations

import weakref
from typing import Any

from memopro._errors import BudgetExceeded, MemoproError

__all__ = ["SavedActivations"]

PRESSURE = 0.75
THRESHOLD = 1 << 20
# GPU copies queued between two fences (one MPS event each, ~1 ms): wrapped buffers go back to
# the runtime this many bytes behind the GPU (0321)
FENCE_BYTES = 64 << 20


class _Packed:
    __slots__ = ("__weakref__", "buf", "device", "dtype", "shape")

    def __init__(self, buf: Any, shape: Any, dtype: Any, device: Any) -> None:
        self.buf, self.shape, self.dtype, self.device = buf, shape, dtype, device


class SavedActivations:
    """The pack/unpack pair; ``hooks()`` is the context manager that installs it."""

    def __init__(
        self,
        runtime: Any,
        ceiling: int,
        *,
        pressure: float = PRESSURE,
        threshold: int = THRESHOLD,
        over_ceiling: str = "error",
    ) -> None:
        from memopro.config import _validate

        self.runtime = runtime
        self.ceiling = ceiling
        self.pressure = pressure
        self.threshold = threshold
        self.over_ceiling = _validate("over_ceiling", over_ceiling)
        self.moved = 0
        self.moved_bytes = 0
        self.over_ceiling_count = 0
        self.over_ceiling_bytes = 0
        self._metal: Any = None  # wrapped runtime buffers for MPS tensors, made on first use
        self._dead: list[Any] = []  # buffers whose free waits for the GPU to let them go
        self._queued = 0  # bytes of GPU copies since the last fence

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
        gpu = self._gpu() if t.device.type == "mps" else None
        try:
            buf = self._alloc(nbytes, f"uint{8 * size}" if size in (1, 2, 4, 8) else "uint8")
            src = t.detach().reshape(-1).view(torch.uint8)
            if gpu is not None:
                # the GPU writes the runtime memory itself, queued behind the work that made t
                # ponytail: through the read pin _MetalPins keeps; nothing else views it
                gpu.tensor(buf, torch.uint8, nbytes).copy_(src)
                self._queue(nbytes)
            else:
                with buf.view(write=True) as a:
                    a.reshape(-1).view("uint8")[...] = src.cpu().numpy()
        except MemoproError:
            return t
        p = _Packed(buf, t.shape, t.dtype, t.device)
        weakref.finalize(p, self._release, buf)
        self.moved += 1
        self.moved_bytes += nbytes
        return p

    def unpack(self, p: Any) -> Any:
        import torch

        if not isinstance(p, _Packed):
            return p
        nbytes = p.buf.nbytes
        try:
            if p.device.type == "mps" and self._metal is not None:
                out = torch.empty(nbytes, dtype=torch.uint8, device=p.device)
                out.copy_(self._metal.tensor(p.buf, torch.uint8, nbytes))  # on the GPU
                self._queue(nbytes)
                return out.view(p.dtype).reshape(p.shape)
            out = torch.empty(p.shape, dtype=p.dtype)
            dst = out.reshape(-1).view(torch.uint8).numpy()
            with p.buf.view() as a:
                dst[...] = a.reshape(-1).view("uint8")
        except BudgetExceeded as e:
            if self.over_ceiling != "allow":
                raise BudgetExceeded(
                    f"{e} (backward needs a saved activation back; set "
                    "over_ceiling='allow' to get it over the ceiling instead)"
                ) from None
            out = torch.empty(p.shape, dtype=p.dtype)
            dst = out.reshape(-1).view(torch.uint8).numpy()
            p.buf.read_into(dst)
            self._over(dst.nbytes)
        return out if p.device.type == "cpu" else out.to(p.device)

    def _gpu(self) -> Any:
        if self._metal is None:
            from memopro.rt.torch import _MetalPins

            self._metal = _MetalPins(self.runtime)
        return self._metal

    def _queue(self, nbytes: int) -> None:
        """Fence the wrapped buffers every FENCE_BYTES of GPU copies, and give back those the
        GPU is done with: a saved activation is used once each way, so its wrap is not kept."""
        self._queued += nbytes
        if self._queued >= FENCE_BYTES:
            self._queued = 0
            self._metal.reap(fence=True)
            self._free_dead()

    def _alloc(self, nbytes: int, dtype: str) -> Any:
        """A runtime buffer; when the runtime is full of buffers the GPU still holds, wait for
        the GPU once and give them all back first."""
        self._free_dead()
        try:
            return self.runtime.alloc(nbytes, dtype=dtype)
        except BudgetExceeded:
            if self._metal is None:
                raise
            self._metal.reap(block=True)
            self._free_dead()
            return self.runtime.alloc(nbytes, dtype=dtype)

    def _release(self, buf: Any) -> None:
        try:
            buf.free()
        except MemoproError:  # still wrapped for the GPU: freed once given back
            self._dead.append(buf)

    def _free_dead(self) -> None:
        if not self._dead:
            return
        if self._metal is not None:
            self._metal.reap()
        dead, self._dead = self._dead, []
        for buf in dead:
            self._release(buf)

    def _over(self, nbytes: int) -> None:
        import warnings

        if self.over_ceiling_count == 0:
            warnings.warn(
                "memopro: backward needed saved activations back with no room under the "
                "ceiling; they are given back over it (over_ceiling='allow'). Results are "
                "unchanged; the process may pass the ceiling while this lasts.",
                UserWarning,
                stacklevel=3,
            )
        self.over_ceiling_count += 1
        self.over_ceiling_bytes += nbytes
