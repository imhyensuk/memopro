"""One-line LLM fine-tuning and generation beyond device memory, losslessly (G4 E8, 0149).

    import memopro

    result = memopro.finetune("Qwen/Qwen2.5-3B-Instruct", texts, budget="1GiB")
    result.adapter.save_pretrained("my-lora")                 # a standard PEFT adapter
    print(memopro.generate(result.model, "Hello!", draft="Qwen/Qwen2.5-1.5B-Instruct"))

Both stream the stored 16-bit weights from their files through a budgeted runtime
(:mod:`memopro.rt.torch`): nothing is quantized and nothing is written to disk. Fine-tuning is LoRA
through PEFT (wrapped, not reimplemented) with reentrant checkpointing and a chunked loss;
generation is greedy and returns exactly what plain greedy generation returns, faster with a
resident int4 ``draft`` (Apple GPUs).
"""

from __future__ import annotations

import time
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from memopro._errors import BudgetExceeded, InvalidArgument, MemoproError

__all__ = ["FinetuneResult", "finetune", "generate"]

# an automatic draft is about this fraction of the target's bytes, at most AUTO_DRAFT_MAX: the
# measured pairs were 1.5B for 7B (3.8x, E038b, 0174) and 0.5B for 3B (3.1x, E039, 0184)
AUTO_DRAFT_SIZE = 0.2
AUTO_DRAFT_MAX = 1 / 3
# free room a draft must leave, as a fraction of its own size: what the GPU keeps during
# drafted generation beyond the draft and the estimate (E061: more than 102 MiB of 473)
AUTO_DRAFT_SLACK = 0.5
# the target's vocabulary may have this many padding rows beyond a draft's (Qwen2.5: 128, 0170)
AUTO_DRAFT_PADDING = 1024
_UNSET = object()
# model -> its automatic draft (or None), kept off the model itself: a module set as an
# attribute would become one of its submodules (state_dict, .to())
_AUTO_DRAFTS: weakref.WeakKeyDictionary[Any, Any] = weakref.WeakKeyDictionary()

# the first step's room is this many times the estimate, until measured (0165). Measured over
# estimate was 1.32-1.37 for Qwen2.5-3B/7B at 512 and 2,048 tokens (E037c, E041; 0187)
_FIRST_STEP_MARGIN = 1.4
# memory that comes and goes between two measurements of the process (0344: an Apple GPU step of
# Qwen2.5-3B rose 10 MiB over a 2.5 GiB ceiling with nothing kept for it)
_TRANSIENT = 64 << 20
# GPU memory left free beside a whole model and its generation: what torch's allocator kept
# unused around 7B on a T4 loaded normally (reserved - allocated, 139 MiB in E042)
_GPU_SLACK = 128 << 20


def _device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    from memopro.env._torch import mps_usable

    if torch.cuda.is_available():
        return "cuda"
    from memopro.rt.torch import set_mps_watermark

    set_mps_watermark()  # the probe below starts MPS
    return "mps" if mps_usable() else "cpu"


def _streamed(model: Any, budget: Any, device: str) -> Any:
    import memopro.rt.torch as rtt

    if hasattr(model, "memopro_weights"):
        return model
    if isinstance(model, (str, Path)):
        import gc

        import torch

        from memopro.rt import resolve_budget

        # a model streamed by an earlier call and dropped since sits in a reference cycle until
        # the collector runs; until then its runtime and device copies count as the process's
        # and the next call is refused (E068, 0353: 6,138 MiB "held besides" at a 6 GiB ceiling)
        gc.collect()
        where = _device(device)
        if where == "cuda":
            torch.cuda.empty_cache()
        elif where == "mps":  # what it held must not count as this model's GPU memory
            torch.mps.empty_cache()
        ceiling = resolve_budget(budget)
        return rtt.stream_model(model, budget=ceiling, runtime=_runtime(ceiling), device=where)
    raise InvalidArgument(
        "pass a model name or folder, or a model from memopro.rt.torch.stream_model"
    )


def _runtime(ceiling: int) -> Any:
    """The runtime of a model streamed here. The ceiling is the whole process's (0344): besides
    what `_hold` keeps for the step and the rest of the process, the runtime shrinks by what the
    process holds while it runs (what the first step's warm-up adds before it can be measured)."""
    from memopro.rt import Runtime

    # the runtime measures the process when it makes room; what grows between two measurements
    # needs a little room (E068: 7B generation peaked 0.95 MiB over 7 GiB)
    return Runtime(budget=ceiling, process_budget=ceiling - min(128 << 20, ceiling // 16))


def _tokenizer(model: Any, tokenizer: Any) -> Any:
    if tokenizer is not None:
        return tokenizer
    import transformers

    # the revision the weights came from (a pinned download has no "main" ref to fall back to)
    revision = getattr(model.config, "_commit_hash", None)
    return transformers.AutoTokenizer.from_pretrained(model.config._name_or_path, revision=revision)


def _activation_estimate(config: Any, tokens: int, batch: int = 1) -> int:
    """Bytes a checkpointed training step of ``batch`` sequences of ``tokens`` keeps outside
    the weights: every layer's input, one layer recomputed with its gradients (bf16), that
    layer's float32 attention scores and their gradient (0187: this term is 16x larger at 2,048
    tokens than at 512), and a few loss chunks."""
    import memopro.rt.torch as rtt

    h, i, layers = config.hidden_size, config.intermediate_size, config.num_hidden_layers
    scores = 2 * config.num_attention_heads * tokens * tokens * 4
    chunk = min(rtt.LOSS_CHUNK_BYTES, batch * tokens * config.vocab_size * 4)  # float32 logits
    return batch * (2 * tokens * (layers * h + 4 * (4 * h + 3 * i)) + scores) + 4 * chunk


def _generation_estimate(config: Any, tokens: int) -> int:
    """Bytes generation keeps outside the weights for ``tokens`` positions: the key/value cache,
    one layer's prompt intermediates with its float32 attention scores, and the last row's
    float32 logits (bf16 elsewhere)."""
    h, i, layers = config.hidden_size, config.intermediate_size, config.num_hidden_layers
    heads = config.num_attention_heads
    kv = getattr(config, "num_key_value_heads", None) or heads
    head_dim = getattr(config, "head_dim", None) or h // heads
    cache = 2 * layers * kv * head_dim * tokens * 2
    scores = heads * tokens * tokens * 4
    return cache + 2 * tokens * (4 * h + 3 * i) + scores + 4 * config.vocab_size * 4


# batch="auto" stops here: 129-token steps of 7B were 1.86x faster at batch 4 than at 1 and
# compute already dominated (0255); larger batches mostly cost memory
AUTO_BATCH_MAX = 8


def _auto_batch(m: Any, seq_len: int, pieces: int) -> int:
    """The largest batch up to ``AUTO_BATCH_MAX`` (and the number of pieces) whose first-step
    activation estimate fits: next to the largest weight in the runtime's budget, or on CUDA in
    the GPU room left for activations."""
    if m.device.type == "cuda":
        import memopro.rt.torch as rtt

        room = rtt._gpu_activation_room(m.device)
    else:
        stats = m.memopro_runtime.stats()
        largest = max(b[0].nbytes for b in m.memopro_weights.buffers.values())
        room = stats["budget"] - stats["reserve"] - largest - _outside(m)
    batch = 1
    while batch < min(AUTO_BATCH_MAX, pieces) and (
        _FIRST_STEP_MARGIN * _activation_estimate(m.config, seq_len, batch + 1) <= room
    ):
        batch += 1
    return batch


def _gpu_kept(m: Any) -> int:
    """Apple GPU memory torch holds besides the wrapped streamed weights, not counting what other
    streamed models in this process hold back (0298)."""
    import memopro.rt.torch as rtt

    return rtt.gpu_kept(m.memopro_weights)


def _outside(m: Any) -> int:
    """Bytes the process holds outside the runtime of ``m`` and the GPU memory its step keeps:
    Python, torch, the tokenizer, other models (0344: the ceiling is the whole process's, as for
    MLX in 0343). 0 where the process cannot be measured."""
    from memopro import rt

    footprint = rt.process_footprint()
    if not footprint:
        return 0
    gpu = max(0, _gpu_kept(m)) if m.device.type == "mps" else 0
    return max(0, footprint - m.memopro_runtime.stats()["used"] - gpu)


def _warm_up(m: Any) -> int:
    """What the rest of the process may add before the first step or pass is measured."""
    return int((_FIRST_STEP_MARGIN - 1) * _outside(m))


def _hold(m: Any, nbytes: int, what: str = "the step's activations", lower: str = "seq_len") -> int:
    """Keep ``nbytes`` of the runtime's budget for ``what`` (memory outside the weights) and
    what the rest of the process holds, or say why not; returns all it keeps."""
    from memopro._errors import BudgetExceeded

    weights, rt = m.memopro_weights, m.memopro_runtime
    if weights.metal is not None:
        weights.metal.reap(block=True)  # wrapped weights nobody uses can go
    largest = max(b[0].nbytes for b in weights.buffers.values())
    stats = rt.stats()
    budget = stats["budget"]
    outside = _outside(m) + _TRANSIENT
    if budget - stats["reserve"] - nbytes - outside < largest:
        raise BudgetExceeded(
            f"a {budget >> 20} MiB ceiling cannot hold the largest weight ({largest >> 20} MiB), "
            f"{what} (about {nbytes >> 20} MiB) and the {outside >> 20} MiB the process holds "
            f"besides; raise the budget or lower {lower}"
        )
    rt.hold_back(nbytes + outside)
    return nbytes + outside


@dataclass
class FinetuneResult:
    model: Any  # the streamed model with LoRA layers, ready for `generate`
    adapter: Any  # the PEFT model: `adapter.save_pretrained(path)` writes a standard adapter
    losses: list[float] = field(default_factory=list)
    step_seconds: list[float] = field(default_factory=list)
    seconds: float = 0.0
    tokens: int = 0
    held_back: int = 0  # bytes of the budget kept for the step's activations and the process
    batch: int = 1  # sequences per step (what batch="auto" chose)


def finetune(
    model: Any,
    texts: list[str],
    *,
    tokenizer: Any = None,
    budget: Any = "auto",
    device: str = "auto",
    epochs: int = 1,
    seq_len: int = 512,
    batch: int | str = 1,
    lr: float = 2e-4,
    rank: int = 8,
    alpha: int = 16,
    targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj"),
    seed: int = 0,
    checkpointing: bool = True,
) -> FinetuneResult:
    """LoRA fine-tuning of a causal LM whose 16-bit weights stay in their files.

    ``model`` may also be a model loaded with ``mlx_lm.load`` (pass its tokenizer): then it is
    trained with mlx-lm's LoRA layers and loss, the same as training it whole, and memopro only
    reads again the layers that do not fit the ceiling (0339). A PyTorch model you loaded
    yourself (Unsloth's ``FastLanguageModel``, Transformers; pass its tokenizer) is trained as
    it is, with its own layers and loss (an Unsloth model gets Unsloth's LoRA layers and keeps
    its speed), within the ceiling for the whole process (0363).

    ``texts`` are joined (end-of-text token between them) and cut into ``seq_len``-token pieces,
    ``batch`` per step (the last step may hold fewer). Each streamed weight is read once per
    step whatever ``batch`` is, so a larger batch trains more tokens per read (0252); it is
    ordinary mini-batch training, so losses differ from ``batch=1`` but stay identical across
    budgets. ``batch="auto"`` takes the largest batch up to ``AUTO_BATCH_MAX`` whose step fits
    the budget (on CUDA, the GPU room ``stream_model`` leaves for activations); the result's
    ``batch`` says which (0262). Only the LoRA weights train (AdamW, float32); the base weights are streamed
    within ``budget`` and never change. Layer checkpointing keeps activations small and results
    reproducible; ``checkpointing=False`` saves a pass over the streamed weights (about 10% on an
    M1) but keeps every layer's activations and, on Apple GPUs, made results differ from run to
    run (0156)."""
    from memopro import _mlx

    if _mlx.is_mlx_model(model):  # the user's mlx-lm model is the original (0334 C, 0339)
        if device != "auto" or checkpointing is not True:
            raise InvalidArgument("an MLX model takes neither device nor checkpointing")
        return _mlx.finetune(
            model,
            texts,
            tokenizer=tokenizer,
            budget=budget,
            epochs=epochs,
            seq_len=seq_len,
            batch=batch,
            lr=lr,
            rank=rank,
            alpha=alpha,
            targets=targets,
            seed=seed,
        )
    from memopro import _loaded

    if _loaded.is_loaded(model):  # the user's own model (Unsloth, Transformers) is the original
        if device != "auto":
            raise InvalidArgument("a model you loaded computes where it is: leave device out")
        return _loaded.finetune(
            model,
            texts,
            tokenizer=tokenizer,
            budget=budget,
            epochs=epochs,
            seq_len=seq_len,
            batch=batch,
            lr=lr,
            rank=rank,
            alpha=alpha,
            targets=targets,
            seed=seed,
            checkpointing=checkpointing,
        )
    import peft
    import torch

    import memopro.rt.torch as rtt

    if not texts:
        raise InvalidArgument("finetune needs at least one text")
    if batch != "auto" and (not isinstance(batch, int) or batch < 1):
        raise InvalidArgument(f"batch must be 'auto' or at least 1, not {batch!r}")
    torch.manual_seed(seed)
    m = _streamed(model, budget, device)
    _leave_gpu_room(m.memopro_weights)
    tok = _tokenizer(m, tokenizer)
    adapter = peft.get_peft_model(
        m, peft.LoraConfig(r=rank, lora_alpha=alpha, target_modules=list(targets))
    )
    # PEFT puts the adapters where the base weights are: the meta device for streamed models
    for layer in m.modules():
        if isinstance(layer, peft.tuners.lora.LoraLayer):
            for name in layer.active_adapters:
                layer.lora_A[name].to_empty(device=m.device)
                layer.lora_B[name].to_empty(device=m.device)
                layer.reset_lora_parameters(name, True)
    for p in m.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    m.train()
    eos = tok.eos_token or ""
    ids = tok(eos.join(texts) + eos, return_tensors="pt").input_ids[0]
    if checkpointing:
        rtt.enable_checkpointing(m)
    pieces = [ids[i : i + seq_len] for i in range(0, len(ids) - 1, seq_len)]
    pieces = [p for p in pieces if len(p) > 1]
    if batch == "auto":
        batch = _auto_batch(m, seq_len, len(pieces))
    steps = [pieces[i : i + batch] for i in range(0, len(pieces), batch)]
    if steps and len({len(p) for p in steps[-1]}) > 1:  # only the last piece is shorter
        steps[-1:] = [steps[-1][:-1], steps[-1][-1:]]
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    result = FinetuneResult(m, adapter, batch=batch)
    gpu = m.device.type == "mps"
    # the budget covers the whole process: weights, what the step keeps (0165; CPU 0175) and
    # the rest of the process (0344). On CUDA the step's memory is the GPU's own, outside the
    # (host) budget (0195)
    step = 0
    if m.device.type != "cuda":
        step = int(
            _FIRST_STEP_MARGIN * _activation_estimate(m.config, seq_len, min(batch, len(pieces)))
        )
    # until the first step is measured, the rest of the process may grow by _FIRST_STEP_MARGIN
    # as well (0344: the Apple GPU's first step added 92 MiB of 469 outside the runtime)
    result.held_back = _hold(m, step + _warm_up(m), lower="seq_len or batch")
    measured = False
    start = time.perf_counter()
    for _ in range(epochs):
        for group in steps:
            x = torch.stack(group).to(m.device)
            t = time.perf_counter()
            with rtt.saved_weights(m):
                loss = rtt.causal_lm_loss(m, x)
                loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            # what the step really kept on the GPU, weights excluded (the CPU has no such count:
            # there the estimate stays), and what the rest of the process holds now
            if gpu:
                step = max(step, _gpu_kept(m))
            if not measured or step + _outside(m) > result.held_back:
                result.held_back = _hold(m, step, lower="seq_len or batch")
                measured = True
            m.memopro_weights.finish()
            result.losses.append(float(loss.detach()))
            result.step_seconds.append(time.perf_counter() - t)
            result.tokens += x.numel()
    result.seconds = time.perf_counter() - start
    m.eval()
    return result


def generate(
    model: Any,
    prompt: str | Any,
    *,
    tokenizer: Any = None,
    budget: Any = "auto",
    device: str = "auto",
    draft: Any = "auto",
    max_new_tokens: int = 256,
    chat: bool = True,
    streamer: Any = None,
    stopping_criteria: Any = None,
) -> str:
    """Greedy text from a model whose 16-bit weights stay in their files.

    ``model`` is a name, a folder or a streamed model (reused as is), or a model loaded with
    ``mlx_lm.load`` (pass its tokenizer): then mlx-lm runs it as given and memopro only keeps
    it within the ceiling (0337). A PyTorch model you loaded yourself (Unsloth, Transformers;
    pass its tokenizer) generates greedily as it is, Unsloth's in its inference mode (0363). A string ``prompt`` goes
    through the chat template when the tokenizer has one and ``chat`` is true. ``draft`` (a
    name, folder or :func:`memopro.rt.torch.draft_model`) proposes tokens on Apple and NVIDIA
    GPUs; the text is the same with or without it. ``"auto"`` (the default) takes a smaller model
    already on this disk that shares the tokenizer, about a fifth of the size (0322; nothing is
    downloaded for it; looked for in the ``model_dirs`` setting, next to a model loaded from a
    folder, and in the Hugging Face cache), and goes without one when there is none or when the budget
    cannot hold it next to the generation; on NVIDIA GPUs only while the weights do not all
    fit the GPU (0363); ``None`` never drafts. ``streamer`` (e.g. Transformers'
    ``TextIteratorStreamer``) and ``stopping_criteria`` go to ``model.generate`` as they are: the
    streamer receives the same tokens the returned text is decoded from.

    The budget covers the whole generation (0182): part of it is held back for the draft, the
    key/value cache and the prompt's intermediates (estimated first; on Apple GPUs measured
    after every pass and raised when more was kept), and budgets that cannot hold the largest
    weight next to that are refused with ``BudgetExceeded``."""
    from memopro import _mlx

    if _mlx.is_mlx_model(model):  # the user's mlx-lm model is the original (0334 C, 0337)
        if draft not in (None, "auto") or streamer is not None or stopping_criteria is not None:
            raise InvalidArgument(
                "an MLX model takes prompt, tokenizer, budget, max_new_tokens and chat only"
            )
        return _mlx.generate(
            model,
            prompt,
            tokenizer=tokenizer,
            budget=budget,
            max_new_tokens=max_new_tokens,
            chat=chat,
        )
    from memopro import _loaded

    if _loaded.is_loaded(model):  # the user's own model (Unsloth, Transformers) is the original
        if device != "auto" or draft not in (None, "auto"):
            raise InvalidArgument("a model you loaded takes neither device nor draft")
        return _loaded.generate(
            model,
            prompt,
            tokenizer=tokenizer,
            budget=budget,
            max_new_tokens=max_new_tokens,
            chat=chat,
            streamer=streamer,
            stopping_criteria=stopping_criteria,
        )
    import memopro.rt.torch as rtt

    m = _streamed(model, budget, device)
    tok = _tokenizer(m, tokenizer)
    if isinstance(prompt, str):
        if chat and tok.chat_template:
            msgs = [{"role": "user", "content": prompt}]
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
            ids = ids["input_ids"] if hasattr(ids, "keys") else ids
        else:
            ids = tok(prompt, return_tensors="pt").input_ids
    else:
        ids = prompt
    tokens = ids.shape[1] + max_new_tokens
    estimate = int(_FIRST_STEP_MARGIN * _generation_estimate(m.config, tokens))
    auto = isinstance(draft, str) and draft == "auto"
    whole_kept = (auto or draft is None) and _keep_whole_on_gpu(m, estimate)
    if auto:
        draft = _auto_draft(m, tok, _room(m, estimate), tokens)
    elif draft is not None and not hasattr(draft, "memopro_draft_bytes"):
        draft = rtt.draft_model(draft, target=m, device=m.device.type)
    held = [estimate + (draft.memopro_draft_bytes if draft is not None else 0)]
    what = (
        "the draft, cache and intermediates" if draft is not None else "the cache and intermediates"
    )
    if m.device.type == "cuda":  # the cache and draft live in GPU memory (0195)
        held[0] = 0
    try:  # and the rest of the process, on every device (0344), until measured with room to grow
        _hold(m, held[0] + _warm_up(m), what, "max_new_tokens")
    except BudgetExceeded:
        if not (auto and draft is not None):
            raise
        # a draft chosen for the caller must not cost them the generation
        _note("memopro: the budget cannot hold the draft as well; generating without it")
        _AUTO_DRAFTS[m] = draft = None
        held[0], what = (0 if m.device.type == "cuda" else estimate), "the cache and intermediates"
        _hold(m, held[0], what, "max_new_tokens")

    measured = [False]

    def measure(*_: Any) -> None:  # what generation really kept on the GPU
        kept = _gpu_kept(m)
        if not measured[0] or kept > held[0]:
            held[0], measured[0] = max(held[0], kept), True
            _hold(m, held[0], what, "max_new_tokens")

    sent = _Sent(streamer) if streamer is not None else None
    hook = m.register_forward_hook(measure) if m.device.type == "mps" else None
    try:
        out = rtt.generate(
            m,
            ids.to(m.device),
            draft=draft,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
            streamer=sent,
            stopping_criteria=stopping_criteria,
        )
    except BudgetExceeded:
        if not (auto and draft is not None):
            raise
        out = None  # retried below, once the exception (and the draft it holds) is gone
    except Exception as e:  # a CUDA out-of-memory with a room memopro chose (0366, 0370)
        if not ((whole_kept or (auto and draft is not None)) and _out_of_memory(e)):
            raise
        out = None
    finally:
        if hook is not None:
            hook.remove()
    if out is None:
        # measured on the GPU, the draft or the whole model did not fit next to the generation
        # after all: the same text without it; a streamer gets only the tokens it has not had yet
        weights = m.memopro_weights
        if whole_kept:
            _note("memopro: the GPU cannot hold the whole model as well; copying what is left")
            weights.gpu_budget, weights.auto_gpu_budget = weights.auto_gpu_budget, None
            weights.drop_gpu_cache()
        else:
            _note("memopro: the budget cannot hold the draft as well; generating without it")
            weights.gpu_budget += getattr(weights, "draft_taken", 0)  # the draft's room is free
            weights.draft_taken = 0
        _AUTO_DRAFTS[m] = draft = None
        _free_gpu(m)
        return generate(
            m,
            ids,
            tokenizer=tok,
            budget=budget,
            device=device,
            draft=None,
            max_new_tokens=max_new_tokens,
            chat=chat,
            streamer=sent.resumed() if sent is not None else None,
            stopping_criteria=stopping_criteria,
        )
    return tok.decode(out[0, ids.shape[1] :], skip_special_tokens=True)


def _keep_whole_on_gpu(m: Any, estimate: int) -> bool:
    """Give the CUDA cache all the room when the whole model fits next to this generation, as it
    does when loaded normally. The automatic budget leaves 2 GiB or 15% for training's
    activations; generation needs far less, and 7B on a T4 (15.3 GB) copied about 2 GB again for
    every token, 7.9x slower than the plain run (0369). ponytail: a model that does not fit keeps
    the automatic budget; a tighter one would have to leave room for its largest weights too."""
    weights = m.memopro_weights
    if m.device.type != "cuda" or weights.auto_gpu_budget is None:
        return False
    import torch

    whole = sum(buf.nbytes for buf, _, _ in weights.buffers.values())
    if whole <= weights.auto_gpu_budget:
        return False
    free, _ = torch.cuda.mem_get_info(m.device)
    unused = torch.cuda.memory_reserved(m.device) - torch.cuda.memory_allocated(m.device)
    fits = whole + estimate + _GPU_SLACK <= free + unused + weights.gpu_cache_bytes
    weights.gpu_budget = whole if fits else weights.auto_gpu_budget
    if weights.gpu_cache_bytes > weights.gpu_budget:
        weights.drop_gpu_cache()
    return fits


def _leave_gpu_room(weights: Any) -> None:
    """Back to the automatic CUDA cache: training's activations need the room generation gave up."""
    auto = weights.auto_gpu_budget
    if auto is not None and weights.gpu_budget > auto:
        weights.gpu_budget = auto
        weights.drop_gpu_cache()


def _out_of_memory(e: BaseException) -> bool:
    import torch

    kind = getattr(torch, "OutOfMemoryError", None) or torch.cuda.OutOfMemoryError
    return isinstance(e, kind)


def _free_gpu(m: Any) -> None:
    import gc

    import torch

    gc.collect()
    if m.device.type == "mps":
        torch.mps.empty_cache()
    elif m.device.type == "cuda":
        torch.cuda.empty_cache()


def _note(message: str) -> None:
    import warnings

    warnings.warn(message, UserWarning, stacklevel=4)


def _room(m: Any, estimate: int) -> int:
    """Budget bytes left beside the largest weight and ``estimate`` for a draft, less what the
    machine cannot give without swapping (0333: a self-draft beside a 4.5 GiB ceiling on 8 GB
    grew swap 1.2 GB)."""
    if m.device.type == "mps":  # what the GPU already keeps counts as it will be measured
        estimate = max(estimate, _gpu_kept(m))
    rt = m.memopro_runtime._rt
    stats = rt.stats()
    largest = max(b[0].nbytes for b in m.memopro_weights.buffers.values())
    return (
        stats["budget"] - stats["reserve"] - _machine_short(rt) - largest - estimate - _outside(m)
    )


def _machine_short(rt: Any) -> int:
    """Bytes of the runtime's limit the machine cannot give without swapping now (0333)."""
    return rt.limit() - rt.machine_limit()


def _auto_draft(m: Any, tok: Any, room: int | None = None, tokens: int = 2048) -> Any:
    """The draft ``generate(draft="auto")`` uses for ``m``: found once per model and kept on it
    (a chat calls generate again and again), or None (0322). A draft that would leave less than
    AUTO_DRAFT_SLACK of its size free in ``room`` is not loaded: measured on the GPU it would
    not fit the generation after all, and finding that out mid-way costs more than it saves
    (E061: 1.5B at 1 GiB, 2.0 instead of 1.5 s per token). On CUDA a draft is 16-bit and is
    used only when the target's weights do not all fit the GPU cache; its weights and its own
    generation memory for ``tokens`` positions come out of that cache (0363)."""
    import memopro.rt.torch as rtt

    found = _AUTO_DRAFTS.get(m, _UNSET)
    if found is not _UNSET:
        return found
    _AUTO_DRAFTS[m] = None
    if m.device.type == "cuda":
        return _cuda_draft(m, tok, tokens)
    if m.device.type != "mps":
        return None  # drafts stay in GPU memory
    choice = _local_draft(m, tok)
    if choice is None and _streams_every_token(m):
        # no sibling: an int4 copy of the model itself, kept in memory only (0329, E033 S3:
        # 3B 3.55 -> 0.89-1.04 s/token, the same text). Only when the weights do not fit the
        # budget: when they do, streaming is cheap and the draft is pure cost (1.5B at 3 GiB:
        # 0.29 -> 0.48 s/token)
        choice = _own_folder(m)
    if choice is None:
        _note(
            "memopro: no smaller model with the same tokenizer is on this disk, so this "
            "generation has no draft; draft='<a model of the same family, about a fifth "
            "of the size>' makes it faster (downloaded once, the text stays the same), or "
            "memopro.configure(model_dirs=[...]) names the folders where you keep models"
        )
        return None
    name, path = choice
    need = int(rtt.draft_size(path, m.device.type) * (1 + AUTO_DRAFT_SLACK))
    if room is not None and need > room:
        rt = m.memopro_runtime._rt
        budget = rt.stats()["budget"]
        more = (budget + need - room + (1 << 20) - 1) >> 20
        how = (
            f"a budget of at least {more} MiB lets it draft"
            if rt.machine_limit() >= rt.limit()
            else "this machine cannot give that much without swapping now; closing other "
            "programs may let it draft"
        )
        _note(
            f"memopro: {name} could draft, but the budget leaves {max(room, 0) >> 20} MiB "
            f"beside the largest weight and the generation, and the draft needs about "
            f"{need >> 20} MiB; {how} (the text is the same either way)"
        )
        return None
    try:
        draft = rtt.draft_model(path, target=m, device=m.device.type)
    except MemoproError as e:  # e.g. no int4 kernel on this GPU
        _note(f"memopro: {name} cannot draft here ({e}); generating without a draft")
        return None
    what = (
        "an int4 copy of the model itself, in memory only"
        if choice == _own_folder(m)
        else f"{name} found on this disk"
    )
    _note(
        f"memopro: drafting with {what} (the text is the same as without a draft; "
        "draft=None turns it off)"
    )
    _AUTO_DRAFTS[m] = draft
    return draft


def _draft_pays(whole: int, cache: int, need: int) -> bool:
    """Whether a CUDA draft taking ``need`` bytes of a ``cache``-byte GPU cache is worth it for
    ``whole`` bytes of weights: it saves passes (about 3 tokens per pass) but every pass then
    copies what the cache no longer holds. Measured on a T4 (0367): 7B (15 GB, cache 13 GB) went
    from 1.55 to 7.3 s/token with a 1.5B draft, an estimated 4x the bytes per pass. Worth it
    when a pass copies less than twice as much; only a model far larger than the GPU gets there
    (ponytail: the factor 2 comes from that one measurement; retune it with a 14B run)."""
    return whole - (cache - need) < 2 * (whole - cache)


def _cuda_draft(m: Any, tok: Any, tokens: int) -> Any:
    """The automatic draft on CUDA (0363): none while every target weight stays in the GPU cache
    (then a draft is pure cost: E068 0351, 3B 1.07 s/token with a 0.5B draft vs 0.46 without);
    else a smaller sibling on this disk whose 16-bit weights and own generation memory fit what
    the cache has (before 0363 only its weights were counted, and 7B with a 1.5B draft ran out
    of GPU memory on a T4)."""
    import transformers

    import memopro.rt.torch as rtt

    weights = m.memopro_weights
    whole = sum(buf.nbytes for buf, _, _ in weights.buffers.values())
    if whole <= weights.gpu_budget:
        return None
    choice = _local_draft(m, tok)
    if choice is None:
        return None
    name, path = choice
    config = transformers.AutoConfig.from_pretrained(str(path))
    working = int(_FIRST_STEP_MARGIN * _generation_estimate(config, tokens))
    # a verification holds one target layer's copies and the output head at once, beside the
    # copy of the next module (E068 0.2.0b4: 7B with a 1.5B draft ran out of GPU memory while
    # holding the 1 GiB head; plain generation never verifies several rows)
    largest = max(buf.nbytes for buf, _, _ in weights.buffers.values())
    layers = max(1, getattr(m.config, "num_hidden_layers", 1))
    working += 2 * largest + (whole - 2 * largest) // layers
    need = int((rtt.draft_size(path, "cuda") + working) * (1 + AUTO_DRAFT_SLACK))
    if not _draft_pays(whole, weights.gpu_budget, need):
        return None
    if need > weights.gpu_budget:
        _note(
            f"memopro: {name} could draft, but it needs about {need >> 20} MiB of GPU memory "
            f"and the cache has {weights.gpu_budget >> 20} MiB; generating without it (the "
            "text is the same either way)"
        )
        return None
    try:
        draft = rtt.draft_model(path, target=m, device="cuda")  # takes its size off the cache
    except MemoproError as e:
        _note(f"memopro: {name} cannot draft here ({e}); generating without a draft")
        return None
    weights.gpu_budget = max(0, weights.gpu_budget - working)
    weights.draft_taken = draft.memopro_draft_bytes + working  # given back if the draft goes
    if weights.gpu_cache_bytes > weights.gpu_budget:
        weights.drop_gpu_cache()
    _note(
        f"memopro: drafting with {name} found on this disk (the text is the same as without a "
        "draft; draft=None turns it off)"
    )
    _AUTO_DRAFTS[m] = draft
    return draft


def _local_draft(m: Any, tok: Any) -> tuple[str, Path] | None:
    """(name, folder) of the model, among those :func:`_model_folders` finds, that suits ``m``
    best as a draft: a causal LM with the same tokenizer, at most AUTO_DRAFT_MAX of ``m``'s bytes, the
    closest to AUTO_DRAFT_SIZE of them; None when there is none. Reads only config files and
    sizes until the tokenizer check."""
    import json
    import math
    import struct

    import transformers

    import memopro.rt.torch as rtt
    from memopro._safetensors import read_header

    target = sum(buf.nbytes for buf, _, _ in m.memopro_weights.buffers.values())
    vocab = getattr(m.config, "vocab_size", None)
    if not isinstance(vocab, int) or target == 0:
        return None
    try:  # loaded as the candidates are, so both get the same added tokens
        want = transformers.AutoTokenizer.from_pretrained(m.config._name_or_path).get_vocab()
    except Exception:  # noqa: BLE001 - no tokenizer files with the weights: the caller's
        want = tok.get_vocab()
    candidates = []
    for name, snap in _model_folders(m):
        try:
            cfg = json.loads((snap / "config.json").read_text())
            names = [p.name for p in snap.iterdir()]
            files = rtt._weight_files(names)
            size = sum((snap / n).stat().st_size for n in files)
            for n in files:  # a broken copy (E068: link text in a synced cache) is no draft
                if n.endswith(".safetensors"):
                    read_header(snap / n)
        except (OSError, ValueError, struct.error, MemoproError):
            continue
        causal = any(str(a).endswith("ForCausalLM") for a in cfg.get("architectures") or ())
        v = cfg.get("vocab_size")
        if not causal or not isinstance(v, int) or not 0 <= vocab - v <= AUTO_DRAFT_PADDING:
            continue
        if not 0 < size <= AUTO_DRAFT_MAX * target:
            continue
        candidates.append((abs(math.log(size / (AUTO_DRAFT_SIZE * target))), name, snap))
    for _, name, snap in sorted(candidates):
        try:
            same = transformers.AutoTokenizer.from_pretrained(str(snap)).get_vocab() == want
        except Exception:  # noqa: BLE001 - no usable tokenizer: not shown to be shared
            same = False
        if same:
            return name, snap
    return None


def _model_folders(m: Any) -> list[tuple[str, Path]]:
    """Folders that may hold a draft for ``m``, each once (0324): the user's ``model_dirs``
    (a model folder, or a folder of model folders), the folders next to ``m``'s own when it
    was loaded from one, and the Hugging Face cache when there is one."""
    from memopro.config import get_config

    found: dict[Path, tuple[str, Path]] = {}

    def add(name: str, folder: Path) -> None:
        if (folder / "config.json").is_file():
            found.setdefault(folder.resolve(), (name, folder))

    def under(root: Path) -> None:
        add(root.name, root)
        if root.is_dir():
            for f in sorted(root.iterdir()):
                if f.is_dir():
                    add(f.name, f)

    for d in get_config().model_dirs:
        under(Path(d).expanduser())
    own = Path(str(m.config._name_or_path))
    if own.is_dir():  # a model kept in a folder of its own: its neighbours (models/*)
        under(own.resolve().parent)
    try:
        import huggingface_hub
    except ImportError:  # no Hugging Face library: no cache to look in
        return list(found.values())
    root = Path(huggingface_hub.constants.HF_HUB_CACHE)
    for snap in sorted(root.glob("models--*/snapshots/*")):
        add(_hub_name(snap), snap)
    return list(found.values())


def _streams_every_token(m: Any) -> bool:
    """Whether ``m``'s weights are larger than its budget, so each token re-reads some."""
    weights = sum(buf.nbytes for buf, _, _ in m.memopro_weights.buffers.values())
    return weights > m.memopro_runtime.stats()["budget"]


def _own_folder(m: Any) -> tuple[str, Path] | None:
    """(name, folder) of ``m``'s own weights on this disk, or None."""
    name = str(m.config._name_or_path)
    if Path(name).is_dir():
        return Path(name).name, Path(name)
    return next(((n, p) for n, p in _model_folders(m) if n == name), None)


def _hub_name(snapshot: Path) -> str:
    """``Org/Name`` of a Hugging Face cache snapshot folder."""
    return snapshot.parent.parent.name.removeprefix("models--").replace("--", "/")


class _Sent:
    """A Transformers streamer passed through, counting the new tokens it got, so that a second
    generation (without the draft) gives it only those it has not had (0322)."""

    def __init__(self, inner: Any, *, prompt: bool = True, skip: int = 0) -> None:
        self.inner = inner
        self.prompt = prompt  # whether the prompt (the first put) goes through
        self.skip = skip  # new tokens the inner streamer already has
        self.prompt_seen = False
        self.seen = 0

    def put(self, value: Any) -> None:
        if not self.prompt_seen:
            self.prompt_seen = True
            if self.prompt:
                self.inner.put(value)
            return
        flat = value.reshape(-1)
        self.seen += flat.numel()
        drop = min(self.skip, flat.numel())
        self.skip -= drop
        if drop < flat.numel():
            self.inner.put(flat[drop:])

    def end(self) -> None:
        self.inner.end()

    def resumed(self) -> _Sent:
        """The streamer for the second generation: no prompt, and none of the tokens sent."""
        return _Sent(self.inner, prompt=False, skip=self.seen)
