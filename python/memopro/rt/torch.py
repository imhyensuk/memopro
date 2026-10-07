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

Faster lossless generation (G4 E4): a small int4 copy of a model with the same vocabulary stays
in memory and proposes tokens; the streamed model checks several of them in one pass over its
weights (Hugging Face assisted generation, greedy: every token kept is the streamed model's own
choice).

    draft = memopro.rt.torch.draft_model("Qwen/Qwen2.5-1.5B-Instruct", target=model)
    out = memopro.rt.torch.generate(model, input_ids, draft=draft, max_new_tokens=64)

``generate`` returns exactly the tokens of ``model.generate(input_ids, do_sample=False)``: a GPU
computes a row of a matrix product with slightly different rounding depending on how many rows
it gets, so checking several proposed tokens at once could break a near-tie differently
(0141). It therefore computes every row after the prompt alone, as plain generation does, with
each layer's weights pinned once for all of them (0142).
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
    "draft_model",
    "enable_checkpointing",
    "generate",
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
    from memopro.access._info import local_safetensors

    files = local_safetensors(name_or_dir, revision)
    if not files:
        raise ModeUnavailable(
            "memopro.rt.torch.stream_model",
            f"no safetensors files for {name_or_dir} on this disk",
            ("download the model once (e.g. huggingface-cli download)", "pass a local folder"),
        )
    return files


class _MetalPins:
    """Pinned runtime memory seen by the GPU as MPS tensors without copying (G4 E1).

    Each runtime buffer in use has one pin, one no-copy ``MTLBuffer`` and one torch tensor over
    it; every use gets a view of that tensor. (Wrapping the same memory once per use, e.g. a
    tied embedding once per loss chunk, was counted again by macOS, 0135; importing one
    ``MTLBuffer`` into torch twice crashed MPS.)

    Wrapped buffers stay wrapped after use, so the next use costs nothing, until the runtime runs
    short of room (less than a quarter of its limit free): then every buffer nobody but this
    cache holds is fenced with one MPS event recorded behind the queued work, and given back
    once the event has completed if it was not used again meanwhile. (Fencing each buffer as soon
    as it was free recorded an event per layer; each one commits the GPU's command buffer, which
    cost about 40% of a training step, 0154.)"""

    def __init__(self, runtime: Any) -> None:
        import mmap

        import torch

        from memopro import _core

        self.runtime = runtime
        self.core = _core
        self.page = mmap.PAGESIZE
        self._use_count = torch._C._storage_Use_Count
        # buffer id -> {"pin", "mtl", "base", "keep", "gen"}
        self.shared: dict[int, dict[str, Any]] = {}
        # (event, [(buffer id, generation at fencing)]) per fenced batch
        self.fenced: list[tuple[Any, list[tuple[int, int]]]] = []
        self.fenced_ids: set[int] = set()

    def pin(self, buf: Any) -> Any:
        try:
            return self.runtime._rt.pin(buf.id, False)
        except BudgetExceeded:
            self.reap(block=True)  # weights the GPU was still reading can go now
            return self.runtime._rt.pin(buf.id, False)

    def _unused(self, record: dict[str, Any]) -> bool:
        # the cached tensor plus the temporary storage object: nobody else holds it
        return self._use_count(record["base"].untyped_storage()._cdata) <= 2

    def tensor(self, buf: Any, dtype: Any, numel: int) -> Any:
        """A 1-D MPS tensor of ``numel`` elements over the buffer's pinned bytes (a view)."""
        from memopro.residency import metal_tensor

        record = self.shared.get(buf.id)
        if record is None:
            self.reap()
            pin = self.pin(buf)
            length = -(-pin.nbytes // self.page) * self.page
            mtl = self.core.metal_wrap(pin.address, length)
            base, keep = metal_tensor(mtl, pin.nbytes)
            record = {"pin": pin, "mtl": mtl, "base": base, "keep": keep, "gen": 0}
            self.shared[buf.id] = record
        record["gen"] += 1
        return record["base"].view(dtype)[:numel]

    def _give_back(self, batch: list[tuple[int, int]]) -> None:
        for bid, gen in batch:
            self.fenced_ids.discard(bid)
            record = self.shared.get(bid)
            if record is None or record["gen"] != gen or not self._unused(record):
                continue  # used again after the fence: fenced anew when room runs short
            del self.shared[bid]
            record["base"] = None  # torch lets go of the storage
            self.core.metal_release(record["mtl"])
            record["pin"].release()

    def reap(self, block: bool = False) -> None:
        """Give back fenced buffers the GPU is done with; when room runs short (or ``block``),
        fence every unused buffer first."""
        import torch

        limit = self.runtime.limit  # changes when room is held back (0165)
        if block or limit - self.runtime._rt.stats()["used"] < limit // 4:
            batch = [
                (bid, r["gen"])
                for bid, r in self.shared.items()
                if bid not in self.fenced_ids and self._unused(r)
            ]
            if batch:
                event = torch.mps.event.Event()
                event.record()
                self.fenced.append((event, batch))
                self.fenced_ids.update(bid for bid, _ in batch)
        waiting = []
        for event, batch in self.fenced:
            if block:
                event.synchronize()
            elif not event.query():
                waiting.append((event, batch))
                continue
            self._give_back(batch)
        self.fenced = waiting

    def held_bytes(self) -> int:
        return sum(r["pin"].nbytes for r in self.shared.values())


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
        # a discrete GPU (CUDA) cannot use runtime memory in place: each use copies the weight to
        # the device and lets the runtime buffer go at once (0195)
        self.copy = self.device.type == "cuda"
        # copies kept on the device, up to `gpu_budget` bytes: with room for the whole model
        # every use after the first is a hit (0201)
        self.gpu_budget = 0
        self.gpu_cache: dict[int, Any] = {}
        self.gpu_cache_bytes = 0
        # the next module's copies, made while the device computes the current one (0201)
        self._prefetched: dict[int, Any] = {}
        self._order: list[Any] = []
        self._seen: set[Any] = set()
        self._stream: Any = None
        self.copy_stats = {"copies": 0, "copy_bytes": 0, "hits": 0, "prefetched": 0}
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
        self._pid_of = {buf.id: pid for pid, (buf, _, _) in self.buffers.items()}
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
        elif self.copy:
            t = self._device_copy(pid)
        else:
            pin = self.runtime._rt.pin(buf.id, False)
            # the tensor keeps `pin` alive, and the pin keeps the memory put (0115)
            t = torch.frombuffer(pin, dtype=dtype, count=numel).view(shape)
        self.pinned[t.untyped_storage().data_ptr()] = (buf, dtype, numel)
        return t

    def _copy_now(self, pid: int, non_blocking: bool = False) -> Any:
        """A device copy of one weight; the runtime buffer is pinned only while it is read."""
        import torch

        buf, dtype, shape = self.buffers[pid]
        numel = 1
        for d in shape:
            numel *= d
        pin = self.runtime._rt.pin(buf.id, False)
        host = torch.frombuffer(pin, dtype=dtype, count=numel).view(shape)
        # from pageable memory the host side of the copy is done when this returns, so the pin
        # can go with `pin` (0195)
        t = host.to(self.device, copy=True, non_blocking=non_blocking)
        self.copy_stats["copies"] += 1
        self.copy_stats["copy_bytes"] += buf.nbytes
        return t

    def _device_copy(self, pid: int) -> Any:
        """The device copy of a weight: from the cache, from the prefetch, or made now."""
        import torch

        t = self.gpu_cache.get(pid)
        if t is not None:
            self.copy_stats["hits"] += 1
            return t
        t = self._prefetched.pop(pid, None)
        if t is not None:
            self.copy_stats["prefetched"] += 1
            if self._stream is not None:  # the copy was queued on the side stream
                torch.cuda.current_stream(self.device).wait_stream(self._stream)
                t.record_stream(torch.cuda.current_stream(self.device))
        else:
            t = self._copy_now(pid)
        nbytes = t.numel() * t.element_size()
        if self.gpu_cache_bytes + nbytes <= self.gpu_budget:
            self.gpu_cache[pid] = t
            self.gpu_cache_bytes += nbytes
        return t

    def _prefetch_after(self, module: Any) -> None:
        """Copy the weights of the module that came after `module` last time (forward order),
        on a side stream while the device computes `module`."""
        import torch

        if not self._order or module not in self._seen:
            return
        i = self._order.index(module)
        nxt = self._order[(i + 1) % len(self._order)]
        self._prefetched.clear()
        want = [id(p) for _, p in self._own.get(nxt, ()) if id(p) not in self.gpu_cache]
        if not want:
            return
        if self.device.type == "cuda":
            if self._stream is None:
                self._stream = torch.cuda.Stream(self.device)
            self._stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(self._stream):
                for pid in want:
                    self._prefetched[pid] = self._copy_now(pid, non_blocking=True)
        else:  # the same bookkeeping without a device stream (tests on the CPU)
            for pid in want:
                self._prefetched[pid] = self._copy_now(pid)

    def drop_gpu_cache(self) -> None:
        """Give back the device copies kept by the cache and the prefetch."""
        self.gpu_cache.clear()
        self.gpu_cache_bytes = 0
        self._prefetched.clear()

    def _before(self, module: Any, args: Any) -> None:
        import torch

        if self.copy and module not in self._seen:  # learn the forward order once
            self._seen.add(module)
            self._order.append(module)
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
        if self.copy:
            self._prefetch_after(module)

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
        elif self.copy:
            base = self._device_copy(self._pid_of[buf.id])
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


# torch's MPS allocator gives any 10-512 MiB request a 1 GiB heap unless its own allocations
# pass a low watermark (this ratio of the recommended maximum); streamed weights are not its
# allocations, so with the default ratio long sequences (activations over 10 MiB) took up to
# 1 GiB more than the budget. 0.01 keeps the 3B LoRA step at seq 512 within budget + 512 MiB at
# the same speed (0162); a value set by the user is kept, the one `memopro run` set is not (0175).
MPS_LOW_WATERMARK = "0.01"


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
    gpu_budget: str | int = "auto",
) -> Any:
    """A Hugging Face model whose weights stay on disk and are streamed through a runtime with
    ``budget`` bytes (or the given ``runtime``). Weights keep their stored dtype; results equal
    those of loading the model normally on the same device. ``device="mps"`` computes on the
    Apple GPU with the runtime's memory used in place (G4 E1); ``device="cuda"`` copies each
    module's weights to the GPU for its forward and backward (0195); ``budget`` is host memory.
    On CUDA up to ``gpu_budget`` bytes of those copies stay on the GPU for the next use, and the
    next module's weights are copied while the current one computes (0201). ``"auto"`` keeps
    the GPU's free memory minus room for activations (2 GiB or 15% of the GPU, the larger); 0
    keeps one module's weights at a time."""
    import torch
    import transformers
    from accelerate import init_empty_weights

    from memopro.hibernate._source import read_header

    if device not in ("cpu", "mps", "cuda"):
        raise InvalidArgument(f"device must be 'cpu', 'mps' or 'cuda', not {device!r}")
    if device == "cuda" and not torch.cuda.is_available():
        raise ModeUnavailable(
            "memopro.rt.torch.stream_model(device='cuda')", "no CUDA GPU here", ("device='cpu'",)
        )
    if device == "mps":
        import os

        from memopro.env import MPS_LOW_WATERMARK_BY_RUN, MPS_LOW_WATERMARK_VAR
        from memopro.env._torch import mps_usable

        # before torch's MPS allocator starts (later it has no effect): see MPS_LOW_WATERMARK
        mine = os.environ.get(MPS_LOW_WATERMARK_VAR)
        if mine is None or mine == os.environ.get(MPS_LOW_WATERMARK_BY_RUN):
            os.environ[MPS_LOW_WATERMARK_VAR] = MPS_LOW_WATERMARK
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
    # generation settings as loading would set them (Qwen2.5: repetition_penalty 1.05; 0195)
    with contextlib.suppress(Exception):
        model.generation_config = transformers.GenerationConfig.from_pretrained(
            str(name_or_dir), revision=revision
        )
    regions = _renamed(regions, model)
    rt = runtime or Runtime(budget=budget, prefetch=prefetch, lookahead=lookahead)
    weights = StreamedWeights(model, rt, regions, device=device)
    if weights.copy:
        weights.gpu_budget = _gpu_budget(gpu_budget, torch.device(device))
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


def _gpu_budget(value: str | int, device: Any) -> int:
    """Bytes of weight copies kept on a CUDA device (0201)."""
    import torch

    if value != "auto":
        from memopro._units import parse_size

        return max(0, parse_size(value))
    free, total = torch.cuda.mem_get_info(device)
    return max(0, free - max(2 << 30, int(0.15 * total)))


def _renamed(regions: dict[str, Any], model: Any) -> dict[str, Any]:
    """Checkpoint names as the model names them: transformers renames some on loading (ViT in
    transformers 5: ``encoder.layer.N.attention.attention.query`` -> ``layers.N.attention.q_proj``,
    0195). Pure renames, and splits of one tensor into equal row blocks (DINOv2 in transformers
    5.18: ``mlp.weights_in`` -> ``mlp.gate_proj`` + ``mlp.up_proj``, 0221), whose parts are
    contiguous byte ranges of the file. A key that would need merging or another operation keeps
    its name, and the model then reports the weight as missing."""
    import dataclasses

    try:
        from transformers.conversion_mapping import get_model_conversion_mapping
        from transformers.core_model_loading import WeightRenaming, rename_source_key
    except ImportError:  # older transformers: checkpoint names are model names
        return regions
    mapping = get_model_conversion_mapping(model)
    if not mapping:
        return regions
    renames = [m for m in mapping if isinstance(m, WeightRenaming)]
    others = [m for m in mapping if not isinstance(m, WeightRenaming)]
    meta = model.state_dict()
    out = {}
    for key, region in regions.items():
        new, matched = rename_source_key(key, renames, others, model.base_model_prefix, meta)
        if matched is None:
            out[new] = region
            continue
        conv = next((c for c in others if matched in c.source_patterns), None)
        targets = _row_split_targets(conv)
        if not targets or not region.shape or region.shape[0] % len(targets):
            out[key] = region
            continue
        part = region.nbytes // len(targets)
        shape = (region.shape[0] // len(targets), *region.shape[1:])
        for i, t in enumerate(targets):
            name = new.replace(targets[0], t, 1) if i else new
            out[name] = dataclasses.replace(
                region, offset=region.offset + i * part, nbytes=part, shape=shape
            )
    return out


def _row_split_targets(conv: Any) -> list[str] | None:
    """The target patterns of a converter that only splits one tensor into equal blocks of rows
    (transformers ``Chunk(dim=0)``), else None."""
    if conv is None or len(conv.source_patterns) != 1 or len(conv.target_patterns) < 2:
        return None
    ops = getattr(conv, "operations", None) or []
    ok = all(
        type(op).__name__ == "Chunk"
        and getattr(op, "dim", None) == 0
        and getattr(op, "num_shards_attribute", None) is None
        for op in ops
    )
    return list(conv.target_patterns) if ops and ok else None


def _padded(forward: Any, pad: int) -> Any:
    """``forward`` with ``pad`` columns of -inf appended to its logits."""
    import torch

    def padded(x: Any) -> Any:
        # torch.cat, not F.pad: torch 2.14's MPS constant pad changed the values (0170)
        y = forward(x)
        return torch.cat([y, y.new_full((*y.shape[:-1], pad), -float("inf"))], dim=-1)

    return padded


def _vocabulary_padding(name_or_dir: Any, revision: Any, config: Any, target: Any) -> int:
    """Rows the target's vocabulary has beyond the draft's when both use the same tokenizer
    (Qwen2.5 pads 7B's to 152,064 and 1.5B's to 151,936, 0170); otherwise InvalidArgument."""
    import transformers

    try:
        mine = transformers.AutoTokenizer.from_pretrained(str(name_or_dir), revision=revision)
        theirs = transformers.AutoTokenizer.from_pretrained(
            target.config._name_or_path, revision=getattr(target.config, "_commit_hash", None)
        )
        vocab = mine.get_vocab()
        # a real tokenizer covers nearly all of the model's rows (Qwen2.5: 151,665 of 151,936);
        # a folder without tokenizer files yields a one-token default
        covers = 0.9 * config.vocab_size <= len(mine) <= config.vocab_size
        same = covers and vocab == theirs.get_vocab()
    except Exception:  # noqa: BLE001 - no tokenizer to compare: not shown to be shared
        same = False
    extra = target.config.vocab_size - config.vocab_size
    if extra < 0 or not same:
        raise InvalidArgument(
            f"{name_or_dir}: vocabulary {config.vocab_size} differs from the target's "
            f"{target.config.vocab_size} and the tokenizers differ; a draft must share the "
            "tokenizer"
        )
    return extra


def draft_model(
    name_or_dir: str | Path,
    *,
    target: Any = None,
    device: str = "mps",
    revision: str | None = None,
) -> Any:
    """An int4 copy of a Hugging Face model that stays in device memory, to propose tokens for
    ``target`` (a streamed model) through ``generate(assistant_model=...)`` (G4 E4, 0139).

    Linear layers become torch's int4 kernel layers (``int4pack``, group 32); embeddings, norms
    and the output head stay in bfloat16. Its size in bytes is ``memopro_draft_bytes``; it is
    device memory outside the streamed model's budget. The draft never changes what greedy
    generation returns, only how many passes over the streamed weights it takes."""
    import gc

    import torch
    import transformers

    from memopro.techniques.integrations import int4pack

    if device != "mps":
        raise ModeUnavailable(
            "memopro.rt.torch.draft_model",
            f"int4 drafts use torch's int4 kernel on Apple GPUs; not on {device!r}",
            ("pass a smaller model loaded normally as assistant_model",),
        )
    reason = int4pack.works(device)
    if reason:
        raise ModeUnavailable("memopro.rt.torch.draft_model", reason, ("device='cpu' streaming",))
    config = transformers.AutoConfig.from_pretrained(str(name_or_dir), revision=revision)
    pad = 0
    if target is not None and getattr(target.config, "vocab_size", None) != config.vocab_size:
        pad = _vocabulary_padding(name_or_dir, revision, config, target)
    # memory-mapped bf16 on the CPU, converted one layer at a time (0069)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        str(name_or_dir), revision=revision, dtype=torch.bfloat16
    )
    int4pack.convert(model, device)
    model.eval()
    if pad:  # the target's extra rows are padding: the draft never proposes them
        head = model.get_output_embeddings()
        head.forward = _padded(head.forward, pad)
        model.config.vocab_size += pad
    gc.collect()
    torch.mps.empty_cache()
    seen: set[int] = set()
    size = 0
    for t in [*model.parameters(), *model.buffers()]:
        key = t.untyped_storage().data_ptr()
        if key not in seen:
            seen.add(key)
            size += t.untyped_storage().nbytes()
    model.memopro_draft_bytes = size
    return model


def _rows(t: Any, start: int, stop: int) -> Any:
    """Rows ``start:stop`` as a tensor of their own, with the strides plain generation's tensors
    have: a view (or a plain ``clone``) keeps the batch stride of all rows, and on x86 CPUs
    oneDNN's bf16 GEMM can round a row differently by strides alone (0240)."""
    import torch

    return None if t is None else t[:, start:stop].clone(memory_format=torch.contiguous_format)


@contextlib.contextmanager
def _row_invariant(model: Any, prompt: int) -> Iterator[None]:
    """Inside this block, rows at positions >= ``prompt`` go through the decoder layers, the
    final norm and the output head one at a time, exactly as in plain generation; earlier rows
    keep the block shape of the prompt's own pass (0142)."""
    import torch

    base = getattr(model, getattr(model, "base_model_prefix", ""), None)
    layers = getattr(base, "layers", None)
    norm, head = getattr(base, "norm", None), model.get_output_embeddings()
    if layers is None or norm is None or head is None:
        raise ModeUnavailable(
            "memopro.rt.torch.generate",
            f"{type(model).__name__}: no decoder layers/norm/output head where Llama-like "
            "models keep them",
            ("model.generate(..., assistant_model=draft) (outputs may differ at near-ties)",),
        )
    weights = model.memopro_weights
    state = {"before": 0, "rows": 0}  # the cache length before this pass, its rows

    def hold(module: Any) -> list[Any]:
        """Pin the weights under ``module`` once for all rows (the tensors keep them pinned;
        each leaf's own hooks then reuse the pinned memory)."""
        return [
            weights._tensor(id(p))
            for m in module.modules()
            if m in weights._own
            for _, p in weights._own[m]
        ]

    def layer_forward(layer: Any, original: Any) -> Any:
        def forward(hidden: Any, *args: Any, **kw: Any) -> Any:
            cache = kw.get("past_key_values")
            q = hidden.shape[1]
            if layer is layers[0]:
                state["rows"] = q
                state["before"] = (
                    0 if cache is None else cache.get_seq_length(layer.self_attn.layer_idx)
                )
            if args or cache is None or q == 1:
                return original(hidden, *args, **kw)
            before = cache.get_seq_length(layer.self_attn.layer_idx)
            block = max(0, min(q, prompt - before))
            if block == q:
                return original(hidden, **kw)
            mask, pos, emb = (
                kw.get("attention_mask"),
                kw.get("position_ids"),
                kw.get("position_embeddings"),
            )
            held = hold(layer)
            try:
                outs = []
                if block:
                    part = dict(kw)
                    if mask is not None:
                        part["attention_mask"] = mask[..., :block, : before + block]
                    part["position_ids"] = _rows(pos, 0, block)
                    if emb is not None:
                        part["position_embeddings"] = tuple(_rows(e, 0, block) for e in emb)
                    outs.append(original(_rows(hidden, 0, block), **part))
                for i in range(block, q):
                    one = dict(kw)
                    one["attention_mask"] = None  # one row sees every cached key
                    one["position_ids"] = _rows(pos, i, i + 1)
                    if emb is not None:
                        one["position_embeddings"] = tuple(_rows(e, i, i + 1) for e in emb)
                    outs.append(original(_rows(hidden, i, i + 1), **one))
            finally:
                del held
            return torch.cat(outs, dim=1)

        return forward

    def norm_forward(original: Any) -> Any:
        def forward(hidden: Any) -> Any:
            q = hidden.shape[1]
            block = max(0, min(q, prompt - state["before"]))
            if q == 1 or block == q:
                return original(hidden)
            parts = [original(_rows(hidden, 0, block))] if block else []
            parts += [original(_rows(hidden, i, i + 1)) for i in range(block, q)]
            return torch.cat(parts, dim=1)

        return forward

    def head_forward(original: Any) -> Any:
        def forward(hidden: Any) -> Any:
            q = hidden.shape[1] if hidden.dim() == 3 else 1
            if q == 1:
                return original(hidden)
            # the prompt's last row (if this pass has it) as plain generation's prompt pass gives
            # it to the head: the last row of the prompt's block, a view with the block's strides
            last = prompt - 1 - state["before"] - (state["rows"] - q)
            held = hold(head)
            try:
                return torch.cat(
                    [
                        original(
                            _rows(hidden, 0, i + 1)[:, -1:]
                            if i == last
                            else _rows(hidden, i, i + 1)
                        )
                        for i in range(q)
                    ],
                    dim=1,
                )
            finally:
                del held

        return forward

    patched = [(layer, layer_forward(layer, layer.forward)) for layer in layers]
    patched += [(norm, norm_forward(norm.forward)), (head, head_forward(head.forward))]
    for module, forward in patched:
        module.forward = forward
    try:
        yield
    finally:
        for module, _ in patched:
            del module.forward


def generate(model: Any, input_ids: Any, *, draft: Any = None, **kwargs: Any) -> Any:
    """Greedy generation from a streamed model, with an optional resident ``draft``
    (:func:`draft_model`) proposing tokens: returns exactly what
    ``model.generate(input_ids, do_sample=False, **kwargs)`` returns, in fewer passes over the
    streamed weights (G4 E4, 0142). One sequence at a time, no padding."""
    import torch

    if not hasattr(model, "memopro_weights"):
        raise InvalidArgument("generate needs a model from memopro.rt.torch.stream_model")
    if kwargs.get("do_sample"):
        raise InvalidArgument("generate is greedy: sampling would not reproduce plain outputs")
    if input_ids.dim() != 2 or input_ids.shape[0] != 1:
        raise InvalidArgument("generate takes one sequence: input_ids of shape [1, length]")
    mask = kwargs.get("attention_mask")
    if mask is not None and not bool(mask.all()):
        raise InvalidArgument("generate takes no padding (an attention_mask of all ones)")
    kwargs["do_sample"] = False
    if draft is not None:
        kwargs["assistant_model"] = draft
    with torch.no_grad(), _row_invariant(model, input_ids.shape[1]):
        return model.generate(input_ids=input_ids, **kwargs)


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


# float32 logits of one loss chunk stay below this: on MPS any allocation of 10-512 MiB makes
# torch's allocator reserve a 1 GiB heap unless it sees pressure, and it never does when the
# weights are the runtime's own memory (0136)
LOSS_CHUNK_BYTES = 8 << 20
# torch 2.14 on MPS: a matmul whose inner dimension exceeds 2**17 (and is not a multiple of
# 16384) can return NaN depending on where its operands were allocated. The output head's
# backward (gradient x weight) has the vocabulary as its inner dimension (151,936 / 152,064 in
# Qwen2.5), so on MPS a large head runs in slices of VOCAB_SLICE rows (0159).
VOCAB_SLICE = 1 << 16


def causal_lm_loss(model: Any, input_ids: Any, labels: Any = None, chunk: int | None = None) -> Any:
    """Next-token cross-entropy (mean over labels other than -100) computed ``chunk`` positions
    at a time, each chunk recomputed in the backward pass: the full float32 logits
    (positions x vocabulary) never exist at once (E3, 0133). By default a chunk's float32 logits
    stay under ``LOSS_CHUNK_BYTES`` (0136). Same mathematics as the model's own loss; the
    summation order differs, so the last bits can differ from it."""
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    labels = input_ids if labels is None else labels
    base = getattr(model, model.base_model_prefix)
    hidden = base(input_ids=input_ids).last_hidden_state[:, :-1]
    targets = labels[:, 1:]
    count = (targets != -100).sum()
    head = model.get_output_embeddings()
    if chunk is None:
        row = hidden.shape[0] * head.out_features * 4
        chunk = max(1, LOSS_CHUNK_BYTES // row)

    slice_head = hidden.device.type == "mps" and head.out_features > 1 << 17

    def part(h: Any, t: Any) -> Any:  # also re-run by the checkpoint during backward
        if slice_head:
            head.forward = sliced
        try:
            logits = head(h).float()
        finally:
            if slice_head:
                del head.forward
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), t.reshape(-1), ignore_index=-100, reduction="sum"
        )

    def sliced(x: Any) -> Any:  # the head's own forward, in slices of its rows (same logits)
        w, b = head.weight, head.bias
        return torch.cat(
            [
                F.linear(x, w[i : i + VOCAB_SLICE], None if b is None else b[i : i + VOCAB_SLICE])
                for i in range(0, w.shape[0], VOCAB_SLICE)
            ],
            dim=-1,
        )

    total = torch.zeros((), device=hidden.device)
    for start in range(0, hidden.shape[1], chunk):
        h, t = hidden[:, start : start + chunk], targets[:, start : start + chunk]
        total = total + checkpoint(part, h, t, use_reentrant=True)
    return total / count.clamp_min(1)
