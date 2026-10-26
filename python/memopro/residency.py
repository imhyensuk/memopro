"""MPS tensors over memory memopro already holds, without copying (0066, 0130).

`metal_tensor` wraps a no-copy ``MTLBuffer`` (made by the Rust core) as a torch MPS tensor via
DLPack; streamed weights and saved activations on Apple GPUs use it (`memopro.rt.torch`). The
file-backed residency that once lived here (``residency="file"``, 0072) wrote converted weights
to disk and was removed in 0.2.0 (0325 rule 1, 0328).
"""

from __future__ import annotations

import ctypes
from typing import Any

__all__ = ["metal_tensor"]


# ---------------------------------------------------------------- mapping as MPS tensors
class _Device(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class _DType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _Device),
        ("ndim", ctypes.c_int32),
        ("dtype", _DType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _Managed(ctypes.Structure):
    _fields_ = [
        ("dl_tensor", _DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.c_void_p),
    ]


def metal_tensor(buffer: int, nbytes: int) -> tuple[Any, Any]:
    """A 1-D uint8 MPS tensor over ``nbytes`` of the ``id<MTLBuffer>`` ``buffer`` (no copy),
    and the DLPack structures that must outlive it. The caller owns the buffer; torch never
    frees it (no deleter). Import each buffer once: two torch tensors over one buffer crash
    MPS (0136)."""
    import torch

    shape = (ctypes.c_int64 * 1)(nbytes)
    managed = _Managed()
    managed.dl_tensor.data = buffer
    managed.dl_tensor.device = _Device(8, 0)  # kDLMetal, as torch exports MPS tensors
    managed.dl_tensor.ndim = 1
    managed.dl_tensor.dtype = _DType(1, 8, 1)  # uint8
    managed.dl_tensor.shape = shape
    new = ctypes.pythonapi.PyCapsule_New
    new.restype = ctypes.py_object
    new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    tensor = torch.utils.dlpack.from_dlpack(new(ctypes.addressof(managed), b"dltensor", None))
    return tensor, (managed, shape)
