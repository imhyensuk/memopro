"""Find the original file of each tensor (safetensors) for write-free restore (0032 H1, 0036 B3).

Supported sources:

- Hugging Face models loaded with ``from_pretrained``: the local directory, or the snapshot in
  the Hugging Face cache (looked up offline, at the commit the model was loaded from).
- Any safetensors file registered explicitly with ``memopro.hibernate.register_source``.

A mapping is only a candidate: at hibernate time the file bytes are read, converted to the
tensor's dtype and compared bit for bit with the live tensor before any memory is released.
"""

from __future__ import annotations

import json
import os
import struct
import weakref
from dataclasses import dataclass
from pathlib import Path

import torch

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


def _hf_directory(model: torch.nn.Module) -> Path | None:
    config = getattr(model, "config", None)
    name = getattr(model, "name_or_path", None) or getattr(config, "_name_or_path", None)
    if not name:
        return None
    if Path(name).is_dir():
        return Path(name)
    try:
        from huggingface_hub import snapshot_download

        return Path(
            snapshot_download(
                name,
                revision=getattr(config, "_commit_hash", None),
                local_files_only=True,
                allow_patterns=["*.safetensors", "*.json"],
            )
        )
    except Exception:  # noqa: BLE001 - no cached snapshot: no source, other methods still apply
        return None


_registered: weakref.WeakKeyDictionary[torch.nn.Module, list[Path]] = weakref.WeakKeyDictionary()


def register(model: torch.nn.Module, *paths: str | os.PathLike[str]) -> None:
    _registered.setdefault(model, []).extend(Path(p) for p in paths)


def _files(model: torch.nn.Module) -> list[Path]:
    files = list(_registered.get(model, []))
    directory = _hf_directory(model)
    if directory is not None:
        files += sorted(directory.glob("*.safetensors"))
    return files


def regions(model: torch.nn.Module) -> dict[str, Region]:
    """State-dict name -> candidate region, for every tensor found in the model's files."""
    table: dict[str, Region] = {}
    for f in _files(model):
        try:
            table.update(read_header(f))
        except (OSError, ValueError):
            continue
    if not table:
        return {}
    prefix = getattr(model, "base_model_prefix", "") or ""
    out = {}
    for name, _ in list(model.named_parameters(remove_duplicate=False)) + list(
        model.named_buffers(remove_duplicate=False)
    ):
        candidates = [name]
        if prefix and name.startswith(prefix + "."):
            candidates.append(name[len(prefix) + 1 :])
        elif prefix:
            candidates.append(f"{prefix}.{name}")
        for c in candidates:
            if c in table:
                out[name] = table[c]
                break
    return out


def matches_shape(region: Region, shape: tuple[int, ...]) -> bool:
    return region.shape == shape or (
        torch.Size(region.shape).numel() == torch.Size(shape).numel() and len(shape) <= 1
    )


def unchanged(region: Region) -> bool:
    try:
        st = os.stat(region.path)
    except OSError:
        return False
    return st.st_size == region.size and st.st_mtime_ns == region.mtime_ns
