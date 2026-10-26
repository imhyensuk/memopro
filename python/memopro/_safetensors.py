"""Safetensors headers: where each tensor's bytes are in its file (read without loading)."""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass

import torch

from memopro._errors import InvalidArgument

_DTYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


@dataclass(frozen=True)
class Region:
    path: str
    offset: int
    nbytes: int
    dtype: torch.dtype
    shape: tuple[int, ...]
    size: int  # file size, to detect a changed file
    mtime_ns: int


def read_header(path: str | os.PathLike[str]) -> dict[str, Region]:
    """Tensor name -> region, from a safetensors header (8-byte length + JSON)."""
    path = str(path)
    st = os.stat(path)
    with open(path, "rb") as f:
        (length,) = struct.unpack("<Q", f.read(8))
        if length > st.st_size - 8:  # not a safetensors file (or cut short): say so clearly
            raise InvalidArgument(
                f"{path}: not a safetensors file (header length {length} > file size {st.st_size})"
            )
        header = json.loads(f.read(length))
    base = 8 + length
    out = {}
    for name, info in header.items():
        if name == "__metadata__" or info.get("dtype") not in _DTYPES:
            continue
        start, end = info["data_offsets"]
        out[name] = Region(
            path,
            base + start,
            end - start,
            _DTYPES[info["dtype"]],
            tuple(info["shape"]),
            st.st_size,
            st.st_mtime_ns,
        )
    return out
