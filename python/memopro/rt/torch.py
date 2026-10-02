"""``memopro.rt.torch``: run PyTorch models whose weights do not fit in memory, losslessly (0115).

``stream_model(name)`` builds the model without weights (meta device) and registers every weight
as a runtime buffer over its region of the original safetensors files. Right before a module
runs, its weights are pinned and become tensors over the runtime's memory (no copy); when it
returns they are let go, and the runtime keeps as many as the budget allows, re-reading the rest
from the files (nothing is written, 0110) and prefetching the next module's weights while this
one computes. Results are bit-identical to loading the model normally: the same bytes go through
the same computation.

Training with frozen streamed weights (LoRA and similar): wrap forward and backward in
:func:`saved_weights` so the autograd graph keeps a *reference* to each weight it needs for the
backward pass instead of the weight itself; the weight is pinned again when backward uses it.

    model = memopro.rt.torch.stream_model("Qwen/Qwen2.5-3B-Instruct", budget="1.5GB")
    out = model.generate(**inputs, max_new_tokens=16)       # bf16 weights, no quantization

CPU only for now (weights are used where they are, in the runtime's memory).
"""

from __future__ import annotations

import contextlib
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from memopro._errors import InvalidArgument, ModeUnavailable
from memopro.rt import Runtime

__all__ = ["StreamedWeights", "saved_weights", "stream_model"]

warnings.filterwarnings("ignore", message="The given buffer is not writable")

_RT_DTYPE = {
    "float32": "float32",
    "float16": "float16",
    "bfloat16": "bfloat16",
    "float64": "float64",
    "int64": "int64",
    "int32": "int32",
    "int8": "int8",
    "uint8": "uint8",
    "bool": "uint8",
}


def _files(name_or_dir: str | Path, revision: str | None) -> list[Path]:
    path = Path(name_or_dir).expanduser()
    if path.is_dir():
        files = sorted(path.glob("*.safetensors"))
    else:
        from memopro.access._info import _cached_files

        files = _cached_files(str(name_or_dir), revision)
    if not files:
        raise ModeUnavailable(
            "memopro.rt.torch.stream_model",
            f"no safetensors files for {name_or_dir} on this disk",
            ("download the model once (e.g. huggingface-cli download)", "pass a local folder"),
        )
    return files


class StreamedWeights:
    """The weights of one model as runtime buffers, and the hooks that pin them around each
    module's forward. Reached as ``model.memopro_weights``."""

    def __init__(self, model: Any, runtime: Runtime, regions: dict[str, Any]) -> None:
        import torch

        self.runtime = runtime
        self.model = model
        prefix = getattr(model, "base_model_prefix", "") or ""
        # one runtime buffer per distinct parameter object (tied weights share one)
        self.buffers: dict[int, tuple[Any, torch.dtype, tuple[int, ...]]] = {}
        for name, param in model.named_parameters(remove_duplicate=False):
            if id(param) in self.buffers:
                continue
            region = _find(regions, name, prefix)
            if region is None:
                continue
            if tuple(region.shape) != tuple(param.shape):
                raise InvalidArgument(
                    f"{name}: shape {tuple(param.shape)} in the model, {region.shape} in the file"
                )
            if region.dtype != param.dtype:
                raise InvalidArgument(
                    f"{name}: dtype {param.dtype} in the model, {region.dtype} in the file "
                    "(streaming keeps the stored dtype)"
                )
            dtype = str(region.dtype).removeprefix("torch.")
            buf = runtime.add_file(
                region.path, region.offset, region.nbytes, dtype=_RT_DTYPE[dtype]
            )
            self.buffers[id(param)] = (buf, region.dtype, tuple(region.shape))
        missing = [n for n, p in model.named_parameters() if id(p) not in self.buffers]
        if missing:
            raise InvalidArgument(f"no weights in the files for {missing[:5]}")
        self._load_saved_buffers(regions, prefix)
        # storage address -> (buffer, dtype, numel) of weights pinned right now (for backward)
        self.pinned: dict[int, tuple[Any, torch.dtype, int]] = {}
        self._own: dict[Any, list[tuple[str, Any]]] = {}
        self._held: dict[Any, list[list[Any]]] = {}
        self._handles = []
        for module in model.modules():
            own = [
                (n, p) for n, p in module.named_parameters(recurse=False) if id(p) in self.buffers
            ]
            if own:
                self._own[module] = own
                self._held[module] = []
                self._handles.append(module.register_forward_pre_hook(self._before))
                self._handles.append(module.register_forward_hook(self._after))

    def _load_saved_buffers(self, regions: dict[str, Any], prefix: str) -> None:
        """Persistent buffers stored in the files (small): loaded once, normally."""
        import torch

        for name, buf in self.model.named_buffers():
            region = _find(regions, name, prefix)
            if region is None or tuple(region.shape) != tuple(buf.shape):
                continue
            with open(region.path, "rb") as f:
                f.seek(region.offset)
                data = bytearray(f.read(region.nbytes))
            value = torch.frombuffer(data, dtype=region.dtype).reshape(region.shape).clone()
            buf.data = value.to(buf.dtype)

    def _tensor(self, pid: int) -> Any:
        import torch

        buf, dtype, shape = self.buffers[pid]
        pin = self.runtime._rt.pin(buf.id, False)
        numel = 1
        for d in shape:
            numel *= d
        # the tensor keeps `pin` alive, and the pin keeps the memory put (0115)
        t = torch.frombuffer(pin, dtype=dtype, count=numel).view(shape)
        self.pinned[t.untyped_storage().data_ptr()] = (buf, dtype, numel)
        return t

    def _before(self, module: Any, args: Any) -> None:
        import torch

        swapped = []
        for name, param in self._own[module]:
            t = self._tensor(id(param))
            module._parameters[name] = torch.nn.Parameter(t, requires_grad=False)
            swapped.append((name, param, t.untyped_storage().data_ptr()))
        self._held[module].append(swapped)

    def _after(self, module: Any, args: Any, output: Any) -> None:
        for name, param, ptr in self._held[module].pop():
            module._parameters[name] = param
            self.pinned.pop(ptr, None)

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    # ---------------------------------------------------------------- backward

    def pack(self, t: Any) -> Any:
        if t.requires_grad or t.device.type != "cpu":
            return t
        info = self.pinned.get(t.untyped_storage().data_ptr())
        if info is None:
            return t
        buf, dtype, numel = info
        return (
            "memopro.rt",
            buf,
            dtype,
            numel,
            tuple(t.size()),
            tuple(t.stride()),
            t.storage_offset(),
        )

    def unpack(self, x: Any) -> Any:
        if not (isinstance(x, tuple) and len(x) == 7 and x[0] == "memopro.rt"):
            return x
        import torch

        _, buf, dtype, numel, size, stride, offset = x
        pin = self.runtime._rt.pin(buf.id, False)
        base = torch.frombuffer(pin, dtype=dtype, count=numel)
        return torch.as_strided(base, size, stride, offset)


def _tie(model: Any, config: Any) -> None:
    """Tie output to input embeddings as loading would (files store tied weights once)."""
    text = getattr(config, "get_text_config", lambda: config)()
    if not getattr(text, "tie_word_embeddings", getattr(config, "tie_word_embeddings", False)):
        return
    with contextlib.suppress(Exception):
        model.tie_weights()
    out, inp = model.get_output_embeddings(), model.get_input_embeddings()
    if out is not None and inp is not None and out.weight is not inp.weight:
        out.weight = inp.weight


def _find(regions: dict[str, Any], name: str, prefix: str) -> Any:
    if name in regions:
        return regions[name]
    if prefix and name.startswith(prefix + "."):
        return regions.get(name[len(prefix) + 1 :])
    if prefix:
        return regions.get(f"{prefix}.{name}")
    return None


def stream_model(
    name_or_dir: str | Path,
    *,
    budget: str | float = "auto",
    runtime: Runtime | None = None,
    revision: str | None = None,
    model_class: Any = None,
    prefetch: bool = True,
    lookahead: str | int = "64MiB",
) -> Any:
    """A Hugging Face model whose weights stay on disk and are streamed through a runtime with
    ``budget`` bytes (or the given ``runtime``). Weights keep their stored dtype; results equal
    those of loading the model normally."""
    import torch
    import transformers
    from accelerate import init_empty_weights

    from memopro.hibernate._source import read_header

    files = _files(name_or_dir, revision)
    regions: dict[str, Any] = {}
    for f in files:
        regions.update(read_header(f))
    floats = [r for r in regions.values() if r.dtype.is_floating_point]
    if not floats:
        raise InvalidArgument(f"{name_or_dir}: no floating-point weights in its files")
    by_dtype: dict[Any, int] = {}
    for r in floats:
        by_dtype[r.dtype] = by_dtype.get(r.dtype, 0) + r.nbytes
    stored = max(by_dtype, key=by_dtype.__getitem__)
    config = transformers.AutoConfig.from_pretrained(str(name_or_dir), revision=revision)
    cls = model_class or transformers.AutoModelForCausalLM
    with init_empty_weights(include_buffers=False):
        model = cls.from_config(config, dtype=stored)
    _tie(model, config)
    model.eval()
    rt = runtime or Runtime(budget=budget, prefetch=prefetch, lookahead=lookahead)
    weights = StreamedWeights(model, rt, regions)
    model.memopro_weights = weights
    model.memopro_runtime = rt
    # parameters live on the meta device between uses; the model computes on the CPU
    model.__class__ = type(
        f"Streamed{type(model).__name__}",
        (type(model),),
        {"device": property(lambda self: torch.device("cpu"))},
    )
    return model


@contextlib.contextmanager
def saved_weights(model: Any) -> Iterator[None]:
    """Inside this block, autograd saves streamed weights as references and pins them again for
    the backward pass, so training frozen weights (LoRA) keeps within the budget."""
    import torch

    weights: StreamedWeights | None = getattr(model, "memopro_weights", None)
    if weights is None:
        raise InvalidArgument("saved_weights needs a model from memopro.rt.torch.stream_model")
    with torch.autograd.graph.saved_tensors_hooks(weights.pack, weights.unpack):
        yield
