"""E015: Metal buffers from Python (no pyobjc): build native/metalbuf.m once and load it.

`nocopy_tensor(ptr, nbytes, shape, dtype)` wraps an MTLBuffer made with
``newBufferWithBytesNoCopy`` over existing memory as a torch MPS tensor through DLPack (device
type 8 = Metal, data = the buffer, as torch itself exports MPS tensors). Research instrument only.
"""

from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "native" / "metalbuf.m"
LIB = HERE / "native" / "build" / "libmetalbuf.dylib"
PURGE = {"keep": 1, "nonvolatile": 2, "volatile": 3, "empty": 4}  # MTLPurgeableState


def lib() -> ctypes.CDLL:
    if not LIB.exists() or LIB.stat().st_mtime < SRC.stat().st_mtime:
        LIB.parent.mkdir(exist_ok=True)
        subprocess.run(
            [
                "clang",
                "-fobjc-arc",
                "-shared",
                "-O2",
                "-framework",
                "Metal",
                "-framework",
                "Foundation",
                "-o",
                str(LIB),
                str(SRC),
            ],
            check=True,
        )
    so = ctypes.CDLL(str(LIB))
    for name, res, args in (
        ("mp_nocopy", ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_size_t]),
        ("mp_new", ctypes.c_void_p, [ctypes.c_size_t]),
        ("mp_contents", ctypes.c_void_p, [ctypes.c_void_p]),
        ("mp_purgeable", ctypes.c_int, [ctypes.c_void_p, ctypes.c_int]),
        ("mp_release", None, [ctypes.c_void_p]),
    ):
        fn = getattr(so, name)
        fn.restype, fn.argtypes = res, args
    return so


class _Device(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class _DType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class _Tensor(ctypes.Structure):
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
        ("dl_tensor", _Tensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.c_void_p),
    ]


_DTYPES = {
    "float16": (2, 16),
    "bfloat16": (4, 16),
    "float32": (2, 32),
    "uint8": (1, 8),
    "int8": (0, 8),
    "int32": (0, 32),
}
_keep: list = []  # the DLPack structs must outlive the tensors (no deleter: kept for the process)


def dlpack_capsule(buffer: int, shape: tuple[int, ...], dtype: str, byte_offset: int = 0):
    code, bits = _DTYPES[dtype]
    dims = (ctypes.c_int64 * len(shape))(*shape)
    m = _Managed()
    m.dl_tensor.data = buffer
    m.dl_tensor.device = _Device(8, 0)
    m.dl_tensor.ndim = len(shape)
    m.dl_tensor.dtype = _DType(code, bits, 1)
    m.dl_tensor.shape = dims
    m.dl_tensor.strides = None
    m.dl_tensor.byte_offset = byte_offset
    m.deleter = None
    _keep.extend([m, dims])
    new = ctypes.pythonapi.PyCapsule_New
    new.restype = ctypes.py_object
    new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    return new(ctypes.addressof(m), b"dltensor", None)


def nocopy_tensor(ptr: int, nbytes: int, shape: tuple[int, ...], dtype: str):
    import torch

    buffer = lib().mp_nocopy(ptr, nbytes)
    if not buffer:
        raise RuntimeError("newBufferWithBytesNoCopy returned nil (alignment? length?)")
    _keep.append(buffer)
    return torch.utils.dlpack.from_dlpack(dlpack_capsule(buffer, shape, dtype)), buffer
