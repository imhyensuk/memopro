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

On the CPU the weights are used where they are, in the runtime's memory. On Apple GPUs
(``device="mps"``, G4 E1) the same memory is handed to the GPU without copying (a no-copy Metal
buffer per pinned weight, unified memory); a weight stays pinned until the GPU has finished with
it (an MPS event recorded after torch let go of it), so the runtime never moves memory the GPU
still reads.
"""

from __future__ import annotations

import contextlib
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from memopro._errors import BudgetExceeded, InvalidArgument, ModeUnavailable
from memopro.rt import Runtime

__all__ = [
    "StreamedWeights",
    "causal_lm_loss",
    "enable_checkpointing",
    "saved_weights",
    "stream_model",
]

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


class _MetalPins:
    """Pinned runtime memory seen by the GPU as MPS tensors without copying (G4 E1).

    Each weight gets a no-copy ``MTLBuffer`` over its pinned bytes, handed to torch through
    DLPack with a deleter. When torch frees the storage (every view gone) the deleter only
    notes it; at the next safe point an MPS event is recorded behind the work already queued,
    and the pin and the buffer are given back once that event has completed."""

    def __init__(self, runtime: Any) -> None:
        import ctypes
        import mmap

        from memopro import _core
        from memopro.residency import _Managed

        self.runtime = runtime
        self.core = _core
        self.page = mmap.PAGESIZE
        self.entries: dict[int, tuple[Any, int, Any, Any]] = {}
        self.freed: list[int] = []
        self.fenced: list[tuple[Any, list[int]]] = []
        self.next_id = 1

        def deleter(address: int) -> None:  # called by torch; must not raise
            try:
                self.freed.append(_Managed.from_address(address).manager_ctx)
            except Exception:  # noqa: BLE001, S110 - nothing may escape into torch
                pass

        self._deleter = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(deleter)

    def pin(self, buf: Any) -> Any:
        self.reap()
        try:
            return self.runtime._rt.pin(buf.id, False)
        except BudgetExceeded:
            self.reap(block=True)  # weights the GPU was still reading can go now
            return self.runtime._rt.pin(buf.id, False)

    def tensor(self, buf: Any, dtype: Any, numel: int) -> Any:
        """A 1-D MPS tensor of ``numel`` elements over the buffer's pinned bytes."""
        import ctypes

        import torch

        from memopro.residency import _Device, _DType, _Managed

        pin = self.pin(buf)
        nbytes = pin.nbytes
        length = -(-nbytes // self.page) * self.page
        mtl = self.core.metal_wrap(pin.address, length)
        eid = self.next_id
        self.next_id += 1
        shape = (ctypes.c_int64 * 1)(nbytes)
        managed = _Managed()
        managed.dl_tensor.data = mtl
        managed.dl_tensor.device = _Device(8, 0)  # kDLMetal
        managed.dl_tensor.ndim = 1
        managed.dl_tensor.dtype = _DType(1, 8, 1)  # uint8
        managed.dl_tensor.shape = shape
        managed.manager_ctx = eid
        managed.deleter = ctypes.cast(self._deleter, ctypes.c_void_p)
        self.entries[eid] = (pin, mtl, managed, shape)
        new = ctypes.pythonapi.PyCapsule_New
        new.restype = ctypes.py_object
        new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
        raw = torch.utils.dlpack.from_dlpack(new(ctypes.addressof(managed), b"dltensor", None))
        return raw.view(dtype)[:numel]

    def reap(self, block: bool = False) -> None:
        import torch

        if self.freed:
            event = torch.mps.event.Event()
            event.record()
            self.fenced.append((event, self.freed))
            self.freed = []
        waiting = []
        for event, ids in self.fenced:
            if block:
                event.synchronize()
            if block or event.query():
                for eid in ids:
                    pin, mtl, _, _ = self.entries.pop(eid)
                    self.core.metal_release(mtl)
                    pin.release()
            else:
                waiting.append((event, ids))
        self.fenced = waiting

    def held_bytes(self) -> int:
        return sum(e[0].nbytes for e in self.entries.values())


class StreamedWeights:
    """The weights of one model as runtime buffers, and the hooks that pin them around each
    module's forward. Reached as ``model.memopro_weights``."""

    def __init__(
        self, model: Any, runtime: Runtime, regions: dict[str, Any], device: str = "cpu"
    ) -> None:
        import torch

        self.runtime = runtime
        self.model = model
        self.device = torch.device(device)
        self.metal = _MetalPins(runtime) if self.device.type == "mps" else None
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
        if self.device.type != "cpu":
            for module in model.modules():
                for name, b in module._buffers.items():
                    if b is not None and b.device.type != "meta":
                        module._buffers[name] = b.to(self.device)
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
        numel = 1
        for d in shape:
            numel *= d
        if self.metal is not None:
            t = self.metal.tensor(buf, dtype, numel).view(shape)
        else:
            pin = self.runtime._rt.pin(buf.id, False)
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
        if t.requires_grad or t.device.type != self.device.type:
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
        if self.metal is not None:
            base = self.metal.tensor(buf, dtype, numel)
        else:
            pin = self.runtime._rt.pin(buf.id, False)
            base = torch.frombuffer(pin, dtype=dtype, count=numel)
        return torch.as_strided(base, size, stride, offset)

    def finish(self, release_cache: bool = True) -> None:
        """Wait for the GPU and give back every weight it held (end of a step or of use); on
        the GPU also give back what torch's MPS allocator keeps cached (E3, 0133: it held twice
        the memory in use)."""
        if self.metal is not None:
            self.metal.reap(block=True)
            if release_cache:
                import torch

                torch.mps.empty_cache()


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
    device: str = "cpu",
) -> Any:
    """A Hugging Face model whose weights stay on disk and are streamed through a runtime with
    ``budget`` bytes (or the given ``runtime``). Weights keep their stored dtype; results equal
    those of loading the model normally on the same device. ``device="mps"`` computes on the
    Apple GPU with the runtime's memory used in place (G4 E1)."""
    import torch
    import transformers
    from accelerate import init_empty_weights

    from memopro.hibernate._source import read_header

    if device not in ("cpu", "mps"):
        raise InvalidArgument(f"device must be 'cpu' or 'mps', not {device!r}")
    if device == "mps":
        from memopro.env._torch import mps_usable

        if not mps_usable():
            raise ModeUnavailable(
                "memopro.rt.torch.stream_model(device='mps')",
                "no usable Apple GPU here",
                ("device='cpu'",),
            )
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
        # Auto classes build from a config with from_config, model classes with _from_config
        build = getattr(cls, "from_config", None) or cls._from_config
        model = build(config, dtype=stored)
    _tie(model, config)
    model.eval()
    rt = runtime or Runtime(budget=budget, prefetch=prefetch, lookahead=lookahead)
    weights = StreamedWeights(model, rt, regions, device=device)
    model.memopro_weights = weights
    model.memopro_runtime = rt
    # parameters live on the meta device between uses; the model computes on `device`
    where = torch.device(device)
    model.__class__ = type(
        f"Streamed{type(model).__name__}",
        (type(model),),
        {"device": property(lambda self: where)},
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


def enable_checkpointing(model: Any) -> None:
    """Keep only each layer's input for the backward pass and recompute the rest (E3, 0133).

    Uses torch's reentrant checkpointing: the recomputation runs inside :func:`saved_weights`,
    so weights it saves become references too. (The non-reentrant kind saves recomputed
    tensors through its own hooks, which kept streamed weights pinned until the budget ran
    out.) Recomputing a layer pins its weights again, and the backward pass right after finds
    them still in memory, so the bytes read per step stay about the same."""
    if not hasattr(model, "memopro_weights"):
        raise InvalidArgument("enable_checkpointing needs a model from stream_model")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    model.enable_input_require_grads()  # the first layer's input must carry gradients


def causal_lm_loss(model: Any, input_ids: Any, labels: Any = None, chunk: int = 32) -> Any:
    """Next-token cross-entropy (mean over labels other than -100) computed ``chunk`` positions
    at a time, each chunk recomputed in the backward pass: the full float32 logits
    (positions x vocabulary) never exist at once (E3, 0133). Same mathematics as the model's
    own loss; the summation order differs, so the last bits can differ from it."""
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    labels = input_ids if labels is None else labels
    base = getattr(model, model.base_model_prefix)
    hidden = base(input_ids=input_ids).last_hidden_state[:, :-1]
    targets = labels[:, 1:]
    count = (targets != -100).sum()
    head = model.get_output_embeddings()

    def part(h: Any, t: Any) -> Any:
        logits = head(h).float()
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), t.reshape(-1), ignore_index=-100, reduction="sum"
        )

    total = torch.zeros((), device=hidden.device)
    for start in range(0, hidden.shape[1], chunk):
        h, t = hidden[:, start : start + chunk], targets[:, start : start + chunk]
        total = total + checkpoint(part, h, t, use_reentrant=True)
    return total / count.clamp_min(1)
