"""E016: RCR F class prototype (0066 §5): model weights that are clean file-backed pages on MPS.

- `load_bf16(model_id)`: map the original safetensors files read-only and point every weight
  at its bytes through one no-copy Metal buffer per file (no conversion, no extra disk).
- `build_int4(model_id, path)` / `load_int4(model_id, path)`: convert once to an aligned F file
  (int4-packed linear weights for torch's kernel, bf16 for the rest), then map it the same way.
- `Prefetcher`: warms the page cache for the next layers with large reads (E015 Q1: faults
  alone reread at 0.28 GB/s, warming at 0.74 GB/s).

Research prototype, not library code.
"""

from __future__ import annotations

import glob
import json
import os
import queue
import struct
import threading
from pathlib import Path
from typing import Any

import torch

from experiments.e015_foundations import vm
from experiments.e015_foundations.metal import dlpack_capsule, lib

PAGE = vm.PAGE
_DT = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "I32": torch.int32,
    "U8": torch.uint8,
    "I64": torch.int64,
}
_DT_NAME = {
    torch.bfloat16: "BF16",
    torch.float16: "F16",
    torch.float32: "F32",
    torch.int32: "I32",
    torch.uint8: "U8",
    torch.int64: "I64",
}


def _snapshot(model_id: str) -> Path:
    from huggingface_hub import try_to_load_from_cache

    config = try_to_load_from_cache(model_id, "config.json")
    if not isinstance(config, str):
        raise FileNotFoundError(f"{model_id} is not in the local cache")
    return Path(config).parent


class MappedFile:
    """A read-only mapping of a whole file wrapped as one uint8 MPS tensor (no copy)."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.size = os.path.getsize(path)
        self.length = (self.size + PAGE - 1) // PAGE * PAGE  # Metal wants whole pages
        self.addr, self.fd = vm.mmap_file(path, self.length)
        buffer = lib().mp_nocopy(self.addr, self.length)
        if not buffer:
            raise RuntimeError(f"no-copy Metal buffer failed for {path}")
        self.bytes = torch.utils.dlpack.from_dlpack(dlpack_capsule(buffer, (self.length,), "uint8"))

    def tensor(self, offset: int, nbytes: int, dtype: torch.dtype, shape: tuple) -> torch.Tensor:
        return self.bytes[offset : offset + nbytes].view(dtype).reshape(shape)

    def resident(self) -> float:
        return vm.resident(self.addr, self.length)


def _skeleton(model_id: str) -> Any:
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig

    config = AutoConfig.from_pretrained(model_id)
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    try:  # as from_pretrained does: the model's own generation defaults
        model.generation_config = GenerationConfig.from_pretrained(model_id)
    except OSError:
        pass
    return model.eval()


def _assign(model: Any, name: str, value: torch.Tensor) -> None:
    owner, _, attr = name.rpartition(".")
    module = model.get_submodule(owner) if owner else model
    if attr in module._parameters:
        module._parameters[attr] = torch.nn.Parameter(value, requires_grad=False)
    else:
        module._buffers[attr] = value


def _finish(model: Any) -> Any:
    model.tie_weights()
    for name, b in list(model.named_buffers()):  # non-persistent buffers (e.g. rotary)
        if b.device.type != "mps":
            _assign(model, name, b.to("mps"))
    left = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    if left:
        raise RuntimeError(f"weights not found in the files: {left[:5]}")
    return model


def _safetensors_header(path: str) -> tuple[int, dict]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return 8 + n, json.loads(f.read(n))


def load_bf16(model_id: str) -> tuple[Any, list[MappedFile], dict]:
    """The stored model with every weight pointing into its original safetensors file."""
    model = _skeleton(model_id)
    files, ranges = [], {}
    for path in sorted(glob.glob(str(_snapshot(model_id) / "*.safetensors"))):
        start, header = _safetensors_header(path)
        mf = MappedFile(path)
        files.append(mf)
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            a, b = meta["data_offsets"]
            t = mf.tensor(start + a, b - a, _DT[meta["dtype"]], tuple(meta["shape"]))
            _assign(model, name, t)
            ranges[name] = (mf, start + a, b - a)
    return _finish(model), files, ranges


def build_int4(model_id: str, path: str) -> dict:
    """Convert once: int4-packed linear weights (torch's MPS layout) and bf16 for the rest,
    each tensor at a page-aligned offset, plus a JSON manifest."""
    from transformers import AutoModelForCausalLM

    from memopro.techniques.integrations import int4pack

    model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": "cpu"}, dtype="auto")
    int4pack.convert(model, "mps")
    entries, offset, seen = [], 0, {}
    with open(path, "wb") as f:
        for name, t in list(model.named_parameters()) + list(model.named_buffers()):
            key = t.untyped_storage().data_ptr()
            if key in seen:  # tied weights: one copy
                entries.append({**seen[key], "name": name})
                continue
            data = t.detach().contiguous().cpu()
            raw = data.view(torch.uint8).numpy().tobytes() if data.numel() else b""
            entry = {
                "name": name,
                "dtype": _DT_NAME[data.dtype],
                "shape": list(data.shape),
                "offset": offset,
                "nbytes": len(raw),
            }
            f.write(raw)
            pad = (-len(raw)) % PAGE
            f.write(b"\0" * pad)
            offset += len(raw) + pad
            seen[key] = entry
            entries.append(entry)
    layers = {
        n: {
            "in": m.in_features,
            "out": m.out_features,
            "group": m.group,
            "bias": m.bias is not None,
        }
        for n, m in model.named_modules()
        if type(m).__name__ == "Int4PackedLinear"
    }
    manifest = {"entries": entries, "int4": layers, "bytes": offset}
    Path(path + ".json").write_text(json.dumps(manifest))
    return manifest


def load_int4(model_id: str, path: str) -> tuple[Any, list[MappedFile], dict]:
    from memopro.techniques.integrations.int4pack import Int4PackedLinear

    manifest = json.loads(Path(path + ".json").read_text())
    model = _skeleton(model_id)
    for name, spec in manifest["int4"].items():  # shells; their buffers come from the file
        owner, _, attr = name.rpartition(".")
        empty = torch.empty(0)
        shell = Int4PackedLinear(
            empty, empty, spec["in"], spec["out"], None, spec["group"], torch.bfloat16
        )
        if spec["bias"]:
            shell.bias = torch.nn.Parameter(torch.empty(0), requires_grad=False)
        setattr(model.get_submodule(owner) if owner else model, attr, shell)
    mf = MappedFile(path)
    ranges = {}
    for e in manifest["entries"]:
        t = mf.tensor(e["offset"], e["nbytes"], _DT[e["dtype"]], tuple(e["shape"]))
        _assign(model, e["name"], t)
        ranges[e["name"]] = (mf, e["offset"], e["nbytes"])
    return _finish(model), [mf], ranges


class Prefetcher:
    """Warms the page cache for the next ``ahead`` decoder layers with large reads."""

    CHUNK = 8 << 20

    def __init__(self, model: Any, ranges: dict, ahead: int = 2) -> None:
        self.queue: queue.Queue = queue.Queue()
        layers = model.model.layers
        self.by_layer = []
        for i in range(len(layers)):
            prefix = f"model.layers.{i}."
            spans = [
                (mf.path, off, n)
                for name, (mf, off, n) in ranges.items()
                if name.startswith(prefix)
            ]
            self.by_layer.append(spans)
        self.fds: dict[str, int] = {}
        for i, layer in enumerate(layers):
            nxt = [(i + k) % len(layers) for k in range(1, ahead + 1)]
            layer.register_forward_pre_hook(lambda _m, _a, nxt=nxt: self._ask(nxt))
        threading.Thread(target=self._run, daemon=True).start()

    def _ask(self, layers: list[int]) -> None:
        for i in layers:
            self.queue.put(i)

    def _run(self) -> None:
        buf = bytearray(self.CHUNK)
        view = memoryview(buf)
        while True:
            i = self.queue.get()
            for path, off, n in self.by_layer[i]:
                fd = self.fds.get(path)
                if fd is None:
                    fd = self.fds[path] = os.open(path, os.O_RDONLY)
                done = 0
                while done < n:
                    k = min(self.CHUNK, n - done)
                    os.preadv(fd, [view[:k]], off + done)
                    done += k
