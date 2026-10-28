"""A PyTorch model the user loaded themselves (with Unsloth's ``FastLanguageModel`` or
Transformers), passed to ``memopro.finetune`` / ``memopro.generate`` (0361, 0363).

The model is the original (0334 C): it is trained and run as given, with its own layers, kernels
and loss, so the results are those of training or generating with it directly under the same
settings. An Unsloth model gets its LoRA layers from ``FastLanguageModel.get_peft_model`` and
generates in Unsloth's inference mode, so it keeps Unsloth's speed. memopro adds the settings
and the ceiling: the whole process must stay within ``budget``, checked before and after every
step; a model that cannot be loaded within the device is passed by name instead and streamed.
"""

from __future__ import annotations

import sys
import time
from typing import Any

from memopro._errors import BudgetExceeded, InvalidArgument


def is_loaded(model: Any) -> bool:
    """A torch module that is not a memopro streamed model."""
    torch = sys.modules.get("torch")
    return (
        torch is not None
        and isinstance(model, torch.nn.Module)
        and not hasattr(model, "memopro_weights")
    )


def is_unsloth(model: Any) -> bool:
    """Loaded through Unsloth: its model class's forward is Unsloth's (it patches the class)."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    for m in (base, getattr(base, "model", None)):
        if m is not None and type(m).forward.__module__.startswith("unsloth"):
            return True
    return getattr(getattr(base, "config", None), "unsloth_version", None) is not None


def _check(ceiling: int, when: str) -> int:
    from memopro import rt

    held = rt.process_footprint() or 0
    if held > ceiling:
        raise BudgetExceeded(
            f"the process holds {held >> 20} MiB {when}, over the {ceiling >> 20} MiB ceiling; "
            "pass the model's name or folder instead to stream its weights within it"
        )
    return held


def finetune(
    model: Any,
    texts: list[str],
    *,
    tokenizer: Any,
    budget: Any,
    epochs: int,
    seq_len: int,
    batch: int | str,
    lr: float,
    rank: int,
    alpha: int,
    targets: tuple[str, ...],
    seed: int,
    checkpointing: bool,
) -> Any:
    import torch

    from memopro import rt
    from memopro.llm import FinetuneResult

    if tokenizer is None:
        raise InvalidArgument("a model you loaded needs its tokenizer: tokenizer=...")
    if not texts:
        raise InvalidArgument("finetune needs at least one text")
    if batch == "auto":
        batch = 1  # nothing streams, so there is no read per step to share
    if not isinstance(batch, int) or batch < 1:
        raise InvalidArgument(f"batch must be 'auto' or at least 1, not {batch!r}")
    ceiling = rt.resolve_budget(budget)
    _check(ceiling, "with the model loaded")
    torch.manual_seed(seed)
    unsloth = is_unsloth(model)
    if hasattr(model, "peft_config"):
        adapter = model  # adapters the user added (PEFT, Unsloth's get_peft_model): theirs
    elif unsloth:
        from unsloth import FastLanguageModel

        adapter = FastLanguageModel.get_peft_model(
            model,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=0,
            bias="none",
            target_modules=list(targets),
            use_gradient_checkpointing="unsloth" if checkpointing else False,
            random_state=seed,
        )
    else:
        import peft

        if checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        adapter = peft.get_peft_model(
            model, peft.LoraConfig(r=rank, lora_alpha=alpha, target_modules=list(targets))
        )
    device = next(adapter.parameters()).device
    eos = tokenizer.eos_token or ""
    ids = tokenizer(eos.join(texts) + eos, return_tensors="pt").input_ids[0]
    pieces = [ids[i : i + seq_len] for i in range(0, len(ids) - 1, seq_len)]
    pieces = [p for p in pieces if len(p) > 1]
    steps = [pieces[i : i + batch] for i in range(0, len(pieces), batch)]
    if steps and len({len(p) for p in steps[-1]}) > 1:  # only the last piece is shorter
        steps[-1:] = [steps[-1][:-1], steps[-1][-1:]]
    opt = torch.optim.AdamW([p for p in adapter.parameters() if p.requires_grad], lr=lr)
    result = FinetuneResult(adapter, adapter, batch=batch)
    adapter.train()
    start = time.perf_counter()
    for _ in range(epochs):
        for group in steps:
            x = torch.stack(group).to(device)
            t = time.perf_counter()
            loss = adapter(input_ids=x, labels=x).loss
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            result.losses.append(float(loss.detach()))
            result.step_seconds.append(time.perf_counter() - t)
            result.tokens += x.numel()
            _check(ceiling, "after a training step")
    result.seconds = time.perf_counter() - start
    adapter.eval()
    return result


def generate(
    model: Any,
    prompt: Any,
    *,
    tokenizer: Any,
    budget: Any,
    max_new_tokens: int,
    chat: bool,
    streamer: Any,
    stopping_criteria: Any,
) -> str:
    import torch

    from memopro import rt

    if tokenizer is None:
        raise InvalidArgument("a model you loaded needs its tokenizer: tokenizer=...")
    ceiling = rt.resolve_budget(budget)
    _check(ceiling, "with the model loaded")
    if is_unsloth(model):
        from unsloth import FastLanguageModel

        FastLanguageModel.for_inference(model)
    if isinstance(prompt, str):
        if chat and tokenizer.chat_template:
            msgs = [{"role": "user", "content": prompt}]
            ids = tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, return_tensors="pt"
            )
            ids = ids["input_ids"] if hasattr(ids, "keys") else ids
        else:
            ids = tokenizer(prompt, return_tensors="pt").input_ids
    else:
        ids = prompt
    device = next(model.parameters()).device
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    with torch.no_grad():
        out = model.generate(
            input_ids=ids.to(device),  # the mask as generate infers it, as a plain call
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad,
            streamer=streamer,
            stopping_criteria=stopping_criteria,
        )
    _check(ceiling, "after generating")
    return tokenizer.decode(out[0, ids.shape[1] :], skip_special_tokens=True)
