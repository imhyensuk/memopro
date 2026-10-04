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

    return transformers.AutoTokenizer.from_pretrained(model.config._name_or_path)


@dataclass
class FinetuneResult:
    model: Any  # the streamed model with LoRA layers, ready for `generate`
    adapter: Any  # the PEFT model: `adapter.save_pretrained(path)` writes a standard adapter
    losses: list[float] = field(default_factory=list)
    seconds: float = 0.0
    tokens: int = 0


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
) -> FinetuneResult:
    """LoRA fine-tuning of a causal LM whose 16-bit weights stay in their files.

    ``texts`` are joined (end-of-text token between them) and cut into ``seq_len``-token pieces,
    one per step. Only the LoRA weights train (AdamW, float32); the base weights are streamed
    within ``budget`` and never change."""
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
    rtt.enable_checkpointing(m)
    m.train()
    eos = tok.eos_token or ""
    ids = tok(eos.join(texts) + eos, return_tensors="pt").input_ids[0]
    pieces = [ids[i : i + seq_len] for i in range(0, len(ids) - 1, seq_len)]
    pieces = [p for p in pieces if len(p) > 1]
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    result = FinetuneResult(m, adapter)
    start = time.perf_counter()
    for _ in range(epochs):
        for piece in pieces:
            x = piece.unsqueeze(0).to(m.device)
            with rtt.saved_weights(m):
                loss = rtt.causal_lm_loss(m, x)
                loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            m.memopro_weights.finish()
            result.losses.append(float(loss.detach()))
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
    text is the same with or without it."""
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
    out = rtt.generate(
        m,
        ids.to(m.device),
        draft=draft,
        max_new_tokens=max_new_tokens,
        pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
    )
    return tok.decode(out[0, ids.shape[1] :], skip_special_tokens=True)
