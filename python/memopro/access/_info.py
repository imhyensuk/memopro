"""What a model holds, known before its weights are loaded (0052 E3).

Sources, first match wins:

1. an ``nn.Module`` you already have: its parameters and buffers
2. a local directory with safetensors files: the file headers
3. a Hugging Face model id whose safetensors files are in the local cache: the file headers
4. a Hugging Face model id online: the safetensors metadata API (headers only, no weights)
5. otherwise the model built from its ``config.json`` on the meta device (no memory)

Only names, shapes and dtypes are read. Sizes of other configurations (half precision, int8,
int4) are derived from them; see `ModelInfo.weight_bytes`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from memopro._errors import InvalidArgument

__all__ = ["ModelInfo", "TensorInfo", "model_info"]

_ST_DTYPES = {
    "F64": ("float64", 8),
    "F32": ("float32", 4),
    "F16": ("float16", 2),
    "BF16": ("bfloat16", 2),
    "F8_E4M3": ("float8_e4m3fn", 1),
    "F8_E5M2": ("float8_e5m2", 1),
    "I64": ("int64", 8),
    "I32": ("int32", 4),
    "I16": ("int16", 2),
    "I8": ("int8", 1),
    "U8": ("uint8", 1),
    "BOOL": ("bool", 1),
}
_FLOATS = {"float64", "float32", "float16", "bfloat16", "float8_e4m3fn", "float8_e5m2"}
# 2-D weights that quantization back ends keep in higher precision
_NOT_QUANTIZED = ("embed", "wte", "wpe", "lm_head", "shared", "position", "norm", "ln_")


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    dtype: str  # torch dtype name, e.g. "float32"
    itemsize: int

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    @property
    def nbytes(self) -> int:
        return self.numel * self.itemsize

    @property
    def floating(self) -> bool:
        return self.dtype in _FLOATS

    @property
    def quantizable(self) -> bool:
        """A linear layer's weight, which int8/int4 back ends quantize."""
        lowered = self.name.lower()
        return (
            self.floating
            and len(self.shape) == 2
            and lowered.endswith("weight")
            and not any(part in lowered for part in _NOT_QUANTIZED)
        )


@dataclass(frozen=True)
class ModelInfo:
    source: str
    origin: str  # "module" | "safetensors" | "hub-metadata" | "meta-model"
    tensors: tuple[TensorInfo, ...]
    config: Any = field(default=None, compare=False, repr=False)

    # ------------------------------------------------------------ sizes
    @property
    def n_params(self) -> int:
        return sum(t.numel for t in self.tensors if t.floating)

    @property
    def stored_bytes(self) -> int:
        return sum(t.nbytes for t in self.tensors)

    @property
    def stored_dtype(self) -> str:
        """The dtype holding most floating-point bytes."""
        totals: dict[str, int] = {}
        for t in self.tensors:
            if t.floating:
                totals[t.dtype] = totals.get(t.dtype, 0) + t.nbytes
        return max(totals, key=totals.__getitem__) if totals else "float32"

    def weight_bytes(self, *, half: bool = False, bits: int | None = None, group: int = 64) -> int:
        """Bytes of the weights as stored, in half precision, or with linear weights at ``bits``.

        Quantized weights also carry scales: about 4 bytes per output row for int8 and per
        ``group``-element block for int4 (64 in bitsandbytes, 32 in memopro's torch int4pack).
        """
        total = 0
        for t in self.tensors:
            if not t.floating:
                total += t.nbytes
                continue
            size = min(t.itemsize, 2) if (half or bits) else t.itemsize
            if bits and t.quantizable:
                scales = t.shape[0] * 4 if bits >= 8 else math.ceil(t.numel / group) * 4
                total += math.ceil(t.numel * bits / 8) + scales
            else:
                total += t.numel * size
        return total

    # ------------------------------------------------------------ architecture
    def _cfg(self, *names: str) -> int | None:
        config = getattr(self.config, "get_text_config", lambda: self.config)()
        for name in names:
            value = getattr(config, name, None)
            if isinstance(value, int):
                return value
        return None

    @property
    def num_layers(self) -> int | None:
        return self._cfg("num_hidden_layers", "n_layer", "num_layers")

    @property
    def max_positions(self) -> int | None:
        return self._cfg("max_position_embeddings", "n_positions", "n_ctx")

    def kv_cache_bytes(self, batch: int, seq: int, itemsize: int = 2) -> int:
        """Keys and values of a generation cache (0 if the config does not say)."""
        layers = self.num_layers
        hidden = self._cfg("hidden_size", "n_embd", "d_model")
        heads = self._cfg("num_attention_heads", "n_head")
        kv_heads = self._cfg("num_key_value_heads") or heads
        head_dim = self._cfg("head_dim") or (hidden // heads if hidden and heads else None)
        if not (layers and kv_heads and head_dim):
            return 0
        return 2 * layers * batch * seq * kv_heads * head_dim * itemsize


# ---------------------------------------------------------------- sources
def model_info(target: Any, *, revision: str | None = None) -> ModelInfo:
    """Describe ``target`` (an ``nn.Module``, a local directory or a Hugging Face model id)."""
    if hasattr(target, "named_parameters"):
        return _from_module(target)
    if not isinstance(target, str | Path):
        raise InvalidArgument(f"expected a model, a directory or a model id; got {type(target)}")
    path = Path(target).expanduser()
    if path.is_dir():
        files = sorted(path.glob("*.safetensors"))
        config = _config(str(path))
        if files:
            return ModelInfo(str(path), "safetensors", _from_files(files), config)
        return _from_meta(str(path), config)
    name = str(target)
    config = _config(name, revision)
    files = _cached_files(name, revision)
    if files:
        return ModelInfo(name, "safetensors", _from_files(files), config)
    tensors = _from_hub(name, revision)
    if tensors:
        return ModelInfo(name, "hub-metadata", tensors, config)
    return _from_meta(name, config)


def _from_module(model: Any) -> ModelInfo:
    seen: set[int] = set()
    tensors = []
    for kind in ("named_parameters", "named_buffers"):
        for name, t in getattr(model, kind)():
            if id(t) in seen:
                continue
            seen.add(id(t))
            dtype = str(t.dtype).removeprefix("torch.")
            tensors.append(TensorInfo(name, tuple(t.shape), dtype, t.element_size()))
    config = getattr(model, "config", None)
    source = getattr(config, "_name_or_path", None) or type(model).__name__
    return ModelInfo(source, "module", tuple(tensors), config)


def _from_files(files: list[Path]) -> tuple[TensorInfo, ...]:
    from memopro.hibernate._source import read_header  # header only

    tensors = []
    for f in files:
        for name, region in read_header(f).items():
            dtype = str(region.dtype).removeprefix("torch.")
            itemsize = region.nbytes // max(1, math.prod(region.shape)) if region.shape else 1
            tensors.append(TensorInfo(name, region.shape, dtype, max(1, itemsize)))
    return tuple(tensors)


def _config(name: str, revision: str | None = None) -> Any:
    try:
        from transformers import AutoConfig

        return AutoConfig.from_pretrained(name, revision=revision)
    except Exception:  # noqa: BLE001 - not a transformers model, or offline: sizes still work
        return None


def _cached_files(name: str, revision: str | None) -> list[Path]:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return []
    single = try_to_load_from_cache(name, "model.safetensors", revision=revision)
    if isinstance(single, str):
        return [Path(single)]
    index = try_to_load_from_cache(name, "model.safetensors.index.json", revision=revision)
    if not isinstance(index, str):
        return []
    shards = sorted(set(json.loads(Path(index).read_text(encoding="utf-8"))["weight_map"].values()))
    paths = [try_to_load_from_cache(name, s, revision=revision) for s in shards]
    return [Path(p) for p in paths] if all(isinstance(p, str) for p in paths) else []


def _from_hub(name: str, revision: str | None) -> tuple[TensorInfo, ...]:
    try:
        from huggingface_hub import get_safetensors_metadata

        meta = get_safetensors_metadata(name, revision=revision)
    except Exception:  # noqa: BLE001 - offline, private, or no safetensors files
        return ()
    tensors = []
    for file_meta in meta.files_metadata.values():
        for tname, info in file_meta.tensors.items():
            dtype, itemsize = _ST_DTYPES.get(info.dtype, ("float32", 4))
            tensors.append(TensorInfo(tname, tuple(info.shape), dtype, itemsize))
    return tuple(tensors)


def _from_meta(name: str, config: Any) -> ModelInfo:
    if config is None:
        raise InvalidArgument(
            f"cannot size {name!r}: no safetensors files and no config.json found "
            "(check the model id, or run once online so it is cached)"
        )
    import torch
    from transformers import AutoModel

    with torch.device("meta"):
        model = AutoModel.from_config(config)
    info = _from_module(model)
    return ModelInfo(name, "meta-model", info.tensors, config)
