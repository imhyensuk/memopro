"""Tensor helpers for adopting objects into runtime buffers (0205)."""

from __future__ import annotations

import numpy as np
import torch


def cpu_bytes(t: torch.Tensor) -> tuple[torch.Tensor, np.ndarray]:
    """A contiguous CPU copy (or the tensor itself) and a uint8 view of its bytes.

    Keep the returned tensor alive while the view is used.
    """
    t = t.detach()
    if t.device.type != "cpu" or not t.is_contiguous():
        t = t.to("cpu", memory_format=torch.contiguous_format).contiguous()
    return t, t.reshape(-1).view(torch.uint8).numpy()


# References held by the tensor itself and by the storage object created while checking.
_OWN_REFS = 2


def _storage_use_count(t: torch.Tensor) -> int | None:
    use_count = getattr(torch._C, "_storage_Use_Count", None)  # private API: optional
    if use_count is None:
        return None
    storage = t.untyped_storage()
    return int(use_count(storage._cdata))


def shared_reason(t: torch.Tensor) -> str | None:
    """Why ``t`` must not move because its memory is shared (0041 D1), or None.

    Swapping ``t.data`` would not free memory that another tensor still holds, and afterwards
    ``t`` would no longer alias it: writes through one would silently miss the other.
    """
    if t._base is not None:
        return "a view of another tensor (waking would break the sharing)"
    count = _storage_use_count(t)
    if count is not None and count > _OWN_REFS:
        return "memory shared with other tensors (a view, or saved by autograd for backward)"
    return None
