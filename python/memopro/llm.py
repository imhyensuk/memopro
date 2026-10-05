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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from memopro._errors import InvalidArgument

__all__ = ["FinetuneResult", "finetune", "generate"]

# the first step's room is this many times the estimate, until measured (0165). Measured over
# estimate was 1.32-1.37 for Qwen2.5-3B/7B at 512 and 2,048 tokens (E037c, E041; 0187)
_FIRST_STEP_MARGIN = 1.4


def _device(device: str) -> str:
    if device != "auto":
        return device
    from memopro.env._torch import mps_usable

    return "mps" if mps_usable() else "cpu"


def _streamed(model: Any, budget: Any, device: str) -> Any:
    import memopro.rt.torch as rtt

    if hasattr(model, "memopro_weights"):
        return model
    if isinstance(model, (str, Path)):
        return rtt.stream_model(model, budget=budget, device=_device(device))
    raise InvalidArgument(
        "pass a model name or folder, or a model from memopro.rt.torch.stream_model"
    )


def _tokenizer(model: Any, tokenizer: Any) -> Any:
    if tokenizer is not None:
        return tokenizer
    import transformers

    # the revision the weights came from (a pinned download has no "main" ref to fall back to)
    revision = getattr(model.config, "_commit_hash", None)
    return transformers.AutoTokenizer.from_pretrained(model.config._name_or_path, revision=revision)


def _activation_estimate(config: Any, tokens: int) -> int:
    """Bytes a checkpointed training step keeps outside the weights: every layer's input, one
    layer recomputed with its gradients (bf16), that layer's float32 attention scores and their
    gradient (0187: this term is 16x larger at 2,048 tokens than at 512), and a few loss
    chunks."""
    import memopro.rt.torch as rtt

    h, i, layers = config.hidden_size, config.intermediate_size, config.num_hidden_layers
    scores = 2 * config.num_attention_heads * tokens * tokens * 4
    chunk = min(rtt.LOSS_CHUNK_BYTES, tokens * config.vocab_size * 4)  # float32 logits
    return 2 * tokens * (layers * h + 4 * (4 * h + 3 * i)) + scores + 4 * chunk


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


def _gpu_kept(m: Any) -> int:
    """Apple GPU memory torch holds besides the wrapped streamed weights."""
    import torch

    return torch.mps.driver_allocated_memory() - m.memopro_weights.metal.held_bytes()


def _hold(
    m: Any, nbytes: int, what: str = "the step's activations", lower: str = "seq_len"
) -> None:
    """Keep ``nbytes`` of the runtime's budget for ``what`` (memory outside the weights), or
    say why not."""
    from memopro._errors import BudgetExceeded

    weights, rt = m.memopro_weights, m.memopro_runtime
    if weights.metal is not None:
        weights.metal.reap(block=True)  # wrapped weights nobody uses can go
    largest = max(b[0].nbytes for b in weights.buffers.values())
    stats = rt.stats()
    budget = stats["budget"]
    if budget - stats["reserve"] - nbytes < largest:
        raise BudgetExceeded(
            f"a {budget >> 20} MiB budget cannot hold the largest weight ({largest >> 20} MiB) "
            f"and {what} (about {nbytes >> 20} MiB); raise the budget or lower {lower}"
        )
    rt.hold_back(nbytes)


@dataclass
class FinetuneResult:
    model: Any  # the streamed model with LoRA layers, ready for `generate`
    adapter: Any  # the PEFT model: `adapter.save_pretrained(path)` writes a standard adapter
    losses: list[float] = field(default_factory=list)
    step_seconds: list[float] = field(default_factory=list)
    seconds: float = 0.0
    tokens: int = 0
    held_back: int = 0  # bytes of the budget kept for the step's activations


def finetune(
    model: Any,
    texts: list[str],
    *,
    tokenizer: Any = None,
    budget: Any = "auto",
    device: str = "auto",
    epochs: int = 1,
    seq_len: int = 512,
    lr: float = 2e-4,
    rank: int = 8,
    alpha: int = 16,
    targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj"),
    seed: int = 0,
    checkpointing: bool = True,
) -> FinetuneResult:
    """LoRA fine-tuning of a causal LM whose 16-bit weights stay in their files.

    ``texts`` are joined (end-of-text token between them) and cut into ``seq_len``-token pieces,
    one per step. Only the LoRA weights train (AdamW, float32); the base weights are streamed
    within ``budget`` and never change. Layer checkpointing keeps activations small and results
    reproducible; ``checkpointing=False`` saves a pass over the streamed weights (about 10% on an
    M1) but keeps every layer's activations and, on Apple GPUs, made results differ from run to
    run (0156)."""
    import peft
    import torch

    import memopro.rt.torch as rtt

    if not texts:
        raise InvalidArgument("finetune needs at least one text")
    torch.manual_seed(seed)
    m = _streamed(model, budget, device)
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
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    result = FinetuneResult(m, adapter)
    gpu = m.device.type == "mps"
    # the budget covers the whole step: weights plus what the step keeps (0165; CPU 0175)
    result.held_back = int(_FIRST_STEP_MARGIN * _activation_estimate(m.config, seq_len))
    _hold(m, result.held_back)
    start = time.perf_counter()
    for _ in range(epochs):
        for piece in pieces:
            x = piece.unsqueeze(0).to(m.device)
            t = time.perf_counter()
            with rtt.saved_weights(m):
                loss = rtt.causal_lm_loss(m, x)
                loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            # what the step really kept on the GPU, weights excluded (the CPU has no such count:
            # there the estimate stays)
            if gpu:
                kept = _gpu_kept(m)
                if kept > result.held_back:
                    result.held_back = kept
                    _hold(m, kept)
            m.memopro_weights.finish()
            result.losses.append(float(loss.detach()))
            result.step_seconds.append(time.perf_counter() - t)
            result.tokens += x.shape[1]
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
    draft: Any = None,
    max_new_tokens: int = 256,
    chat: bool = True,
) -> str:
    """Greedy text from a model whose 16-bit weights stay in their files.

    ``model`` is a name, a folder or a streamed model (reused as is). A string ``prompt`` goes
    through the chat template when the tokenizer has one and ``chat`` is true. ``draft`` (a
    name, folder or :func:`memopro.rt.torch.draft_model`) proposes tokens on Apple GPUs; the
    text is the same with or without it.

    The budget covers the whole generation (0182): part of it is held back for the draft, the
    key/value cache and the prompt's intermediates (estimated first; on Apple GPUs measured
    after every pass and raised when more was kept), and budgets that cannot hold the largest
    weight next to that are refused with ``BudgetExceeded``."""
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
    if draft is not None and not hasattr(draft, "memopro_draft_bytes"):
        draft = rtt.draft_model(draft, target=m, device=m.device.type)
    tokens = ids.shape[1] + max_new_tokens
    held = [int(_FIRST_STEP_MARGIN * _generation_estimate(m.config, tokens))]
    if draft is not None:
        held[0] += draft.memopro_draft_bytes
    what = (
        "the draft, cache and intermediates" if draft is not None else "the cache and intermediates"
    )
    _hold(m, held[0], what, "max_new_tokens")

    def measure(*_: Any) -> None:  # what generation really kept on the GPU
        kept = _gpu_kept(m)
        if kept > held[0]:
            held[0] = kept
            _hold(m, kept, what, "max_new_tokens")

    hook = m.register_forward_hook(measure) if m.device.type == "mps" else None
    try:
        out = rtt.generate(
            m,
            ids.to(m.device),
            draft=draft,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
        )
    finally:
        if hook is not None:
            hook.remove()
    return tok.decode(out[0, ids.shape[1] :], skip_special_tokens=True)
