"""RCR F class (0066, 0072): hold model weights as clean file-backed pages on Apple GPUs.

``residency="file"`` does not change which configuration ``load`` chooses; it changes what kind
of memory holds the weights. They become views of a read-only file mapping wrapped as an MPS
tensor without copying (Rust `FileMap` + ``newBufferWithBytesNoCopy`` + DLPack), so they add
nothing to the process footprint, need no swap writes, and under pressure the OS drops them and
reads them back (E015 G-F, E016: Qwen2.5-3B int4 at 10.6 tokens/s with a 0.53 GB footprint).

- ``stored``: the original safetensors files are mapped as they are (no conversion, no disk),
  unless a tensor is misaligned for its dtype (then an aligned file cache, as below).
- ``half`` and ``quant.int4`` (torch int4pack): a file cache is built once from the normally
  loaded model (needs ``disk_writes="allow"`` and room in the disk budget) and then mapped.
"""

from __future__ import annotations

import ctypes
import glob
import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Any

from memopro._errors import ModeUnavailable

__all__ = ["FORMAT", "load_file_backed"]

FORMAT = 1  # bump when the cache layout changes
_PAGE = 16384
_ALTERNATIVES = ("memory",)
_DT = {
    "BF16": "bfloat16",
    "F16": "float16",
    "F32": "float32",
    "I32": "int32",
    "U8": "uint8",
    "I64": "int64",
    "I8": "int8",
    "BOOL": "bool",
}


def _unavailable(reason: str) -> ModeUnavailable:
    return ModeUnavailable("residency='file'", reason, _ALTERNATIVES)


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


class Mapping:
    """One file mapped read-only and seen by torch as a uint8 MPS tensor (no copy)."""

    def __init__(self, path: str) -> None:

        from memopro import _core

        if not hasattr(_core, "FileMap"):
            raise _unavailable("file mappings are not available on this platform")
        self.path = path
        self.map = _core.FileMap(path)
        try:
            buffer = self.map.metal_buffer()
        except (NotImplementedError, OSError, RuntimeError) as e:
            raise _unavailable(f"no Metal buffer over the file: {e}") from None
        # the mapping owns the buffer
        self.bytes, self._keep = metal_tensor(buffer, self.map.length)

    def tensor(self, offset: int, nbytes: int, dtype: Any, shape: tuple[int, ...]) -> Any:
        return self.bytes[offset : offset + nbytes].view(dtype).reshape(shape)

    def resident(self) -> float:
        return self.map.resident()


def _dtype(name: str) -> Any:
    import torch

    return getattr(torch, name)


# ---------------------------------------------------------------- model skeleton
def _skeleton(cls: Any, model_id: Any, revision: str | None) -> Any:
    from accelerate import init_empty_weights
    from transformers import AutoConfig, GenerationConfig

    config = AutoConfig.from_pretrained(model_id, revision=revision)
    with init_empty_weights(include_buffers=False):
        model = cls._from_config(config)
    try:  # as from_pretrained does
        model.generation_config = GenerationConfig.from_pretrained(model_id, revision=revision)
    except OSError:
        pass
    return model.eval()


def _assign(model: Any, name: str, value: Any) -> None:
    import torch

    owner, _, attr = name.rpartition(".")
    module = model.get_submodule(owner) if owner else model
    if attr in module._parameters:
        module._parameters[attr] = torch.nn.Parameter(value, requires_grad=False)
    else:
        module._buffers[attr] = value


def _finish(model: Any, keep: list[Mapping]) -> Any:
    model.tie_weights()
    for name, b in list(model.named_buffers()):  # non-persistent buffers built by the model
        if b.device.type != "mps":
            _assign(model, name, b.to("mps"))
    missing = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    if missing:
        raise _unavailable(f"weights not found in the files: {', '.join(missing[:5])}")
    model._memopro_mappings = keep  # the mappings live as long as the model
    return model


# ---------------------------------------------------------------- stored: the original files
def _safetensors_files(model_id: Any, revision: str | None) -> list[str]:
    from memopro.access._info import local_safetensors

    files = [str(p) for p in local_safetensors(model_id, revision)]
    if not files and not Path(str(model_id)).expanduser().is_dir():
        from huggingface_hub import snapshot_download

        root = snapshot_download(
            str(model_id), revision=revision, allow_patterns=["*.safetensors", "*.json"]
        )
        files = sorted(glob.glob(str(Path(root) / "*.safetensors")))
    if not files:
        raise _unavailable(f"{model_id} has no safetensors files to map")
    return files


def _header(path: str) -> tuple[int, dict]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return 8 + n, json.loads(f.read(n))


def _resolver(model: Any):
    """File key -> model name, as from_pretrained matches them: exact, then with or without the
    base model prefix (e.g. GPT-2 checkpoints lack "transformer."); None for keys the model does
    not hold (e.g. GPT-2's causal-mask buffers)."""
    names = {n for n, _ in model.named_parameters()} | {n for n, _ in model.named_buffers()}
    prefix = getattr(model, "base_model_prefix", "") or ""

    def resolve(key: str) -> str | None:
        if key in names:
            return key
        if prefix and f"{prefix}.{key}" in names:
            return f"{prefix}.{key}"
        if prefix and key.startswith(prefix + ".") and key[len(prefix) + 1 :] in names:
            return key[len(prefix) + 1 :]
        return None

    return resolve


def _aligned(model_id: Any, revision: str | None) -> bool:
    """Whether every tensor in the original files starts at a multiple of its element size
    (torch cannot view misaligned bytes, e.g. GPT-2's fp32 file)."""
    size = {"BF16": 2, "F16": 2, "F32": 4, "I32": 4, "I64": 8, "U8": 1, "I8": 1, "BOOL": 1}
    for path in _safetensors_files(model_id, revision):
        start, header = _header(path)
        for name, meta in header.items():
            if name != "__metadata__" and (start + meta["data_offsets"][0]) % size.get(
                meta["dtype"], 8
            ):
                return False
    return True


def _map_stored(cls: Any, model_id: Any, revision: str | None) -> tuple[Any, int]:
    model = _skeleton(cls, model_id, revision)
    resolve = _resolver(model)
    keep, mapped = [], 0
    for path in _safetensors_files(model_id, revision):
        start, header = _header(path)
        m = Mapping(path)
        keep.append(m)
        mapped += m.map.file_length
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            target = resolve(name)
            if target is None:
                continue
            a, b = meta["data_offsets"]
            t = m.tensor(start + a, b - a, _dtype(_DT[meta["dtype"]]), tuple(meta["shape"]))
            _assign(model, target, t)
    return _finish(model, keep), mapped


# ---------------------------------------------------------------- half / int4: a file cache
def _sources(model_id: Any, revision: str | None) -> list[dict]:
    out = []
    for p in _safetensors_files(model_id, revision):
        st = os.stat(p)
        out.append({"path": os.path.basename(p), "size": st.st_size, "mtime_ns": st.st_mtime_ns})
    return out


def cache_path(directory: Path, model_id: Any, revision: str | None, cfg_name: str) -> Path:
    import torch

    from memopro.techniques.integrations.int4pack import GROUP

    key = json.dumps([str(model_id), revision, cfg_name, GROUP, torch.__version__, FORMAT])
    return directory / f"{hashlib.sha256(key.encode()).hexdigest()[:24]}.f"


def _valid(path: Path, sources: list[dict]) -> bool:
    try:
        manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        manifest.get("format") == FORMAT
        and manifest.get("sources") == sources
        and path.exists()
        and path.stat().st_size == manifest.get("bytes")
    )


def _write_cache(model: Any, path: Path, sources: list[dict]) -> int:
    import torch

    entries, offset, seen = [], 0, {}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        for name, t in list(model.named_parameters()) + list(model.named_buffers()):
            key = (t.untyped_storage().data_ptr(), t.storage_offset(), tuple(t.shape))
            if key in seen:  # tied weights: one copy
                entries.append({**seen[key], "name": name})
                continue
            data = t.detach().contiguous().cpu()
            raw = data.reshape(-1).view(torch.uint8).numpy().tobytes() if data.numel() else b""
            entry = {
                "name": name,
                "dtype": str(data.dtype).removeprefix("torch."),
                "shape": list(data.shape),
                "offset": offset,
                "nbytes": len(raw),
            }
            f.write(raw)
            pad = (-len(raw)) % _PAGE
            f.write(b"\0" * pad)
            offset += len(raw) + pad
            seen[key] = entry
            entries.append(entry)
    int4 = {
        n: {
            "in": m.in_features,
            "out": m.out_features,
            "group": m.group,
            "bias": m.bias is not None,
            "dtype": str(m.compute_dtype).removeprefix("torch."),
        }
        for n, m in model.named_modules()
        if type(m).__name__ == "Int4PackedLinear"
    }
    manifest = {
        "format": FORMAT,
        "sources": sources,
        "entries": entries,
        "int4": int4,
        "bytes": offset,
    }
    tmp.replace(path)
    mpath = path.with_suffix(".json")
    fd = os.open(mpath, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(manifest, f)
    return offset


def _map_cache(cls: Any, model_id: Any, revision: str | None, path: Path) -> tuple[Any, int]:
    import torch

    from memopro.techniques.integrations.int4pack import Int4PackedLinear

    manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    model = _skeleton(cls, model_id, revision)
    for name, spec in manifest["int4"].items():  # shells whose buffers come from the file
        owner, _, attr = name.rpartition(".")
        empty = torch.empty(0)
        shell = Int4PackedLinear(
            empty, empty, spec["in"], spec["out"], None, spec["group"], _dtype(spec["dtype"])
        )
        if spec["bias"]:
            shell.bias = torch.nn.Parameter(torch.empty(0), requires_grad=False)
        setattr(model.get_submodule(owner) if owner else model, attr, shell)
    m = Mapping(str(path))
    for e in manifest["entries"]:
        _assign(
            model,
            e["name"],
            m.tensor(e["offset"], e["nbytes"], _dtype(e["dtype"]), tuple(e["shape"])),
        )
    return _finish(model, [m]), m.map.file_length


# ---------------------------------------------------------------- entry point from load()
def load_file_backed(
    plan: Any, cfg: Any, cls: Any, model_id: Any, revision: str | None, build: Any
) -> tuple[Any, str]:
    """The model for ``cfg`` with file-backed weights, and a line for the report.

    ``build()`` loads the configuration the normal way (used once to fill a file cache).
    """
    from memopro._units import format_size
    from memopro.config import spill_location
    from memopro.orchestrator.candidates import backend_of

    if plan.ctx.device != "mps":
        raise _unavailable(
            f"file-backed weights need Apple unified memory (MPS); the device is {plan.ctx.device}"
        )
    if cfg.name == "stored" and _aligned(model_id, revision):
        model, mapped = _map_stored(cls, model_id, revision)
        return model, f"residency file: the original safetensors mapped ({format_size(mapped)})"
    if cfg.name not in ("stored", "half", "quant.int4") or (
        cfg.name == "quant.int4" and backend_of(cfg, plan.ctx) != "torch-int4pack"
    ):
        raise _unavailable(
            f"file residency supports as stored, half and int4 (torch int4pack); {cfg.name} was "
            "chosen. Choose it with quality or budget, or keep residency='memory'"
        )
    setup = plan.setup
    path = cache_path(spill_location(setup.config) / "fcache", model_id, revision, cfg.name)
    sources = _sources(model_id, revision)
    if _valid(path, sources):
        model, mapped = _map_cache(cls, model_id, revision, path)
        return model, f"residency file: cache reused ({format_size(mapped)}, {path})"
    size = cfg.needs.device + cfg.needs.host - plan.ctx.runtime_bytes
    if setup.config.disk_writes != "allow":
        raise _unavailable(
            f"the file cache for {cfg.name} must be written once (about {format_size(size)} to "
            f"SSD) and disk_writes is {setup.config.disk_writes!r}; pass disk_writes='allow'"
        )
    if size > setup.budget.disk:
        raise _unavailable(
            f"the file cache needs about {format_size(size)} of disk and the disk budget is "
            f"{format_size(setup.budget.disk)} (free-space floor or disk budget setting)"
        )
    import gc

    model = build()
    written = _write_cache(model, path, sources)
    del model
    gc.collect()
    import torch

    torch.mps.empty_cache()
    model, mapped = _map_cache(cls, model_id, revision, path)
    return model, f"residency file: cache built ({format_size(written)}, {path})"
