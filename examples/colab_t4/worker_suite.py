"""E042 worker (docs/research/0197): one case of the Colab T4 suite in a fresh process.

    worker_suite.py lm_gen    <model> <plain|memopro> <budget_bytes> <new_tokens>
    worker_suite.py lm_lora   <model> <plain|memopro> <budget_bytes> <steps> <seq>
    worker_suite.py vis_infer <model> <plain|memopro> <budget_bytes> <batch>
    worker_suite.py vis_lora  <model> <plain|memopro> <budget_bytes> <steps> <batch>

`plain` loads the model normally onto the GPU; `memopro` streams it from host memory within
`budget_bytes` (memopro.rt.torch.stream_model(device="cuda"), 0195).
"""

import hashlib
import sys
import time

import torch
from worker_common import DEVICE, MON, RESULT, cleanup, finish, guarded, model_path, sync

PROMPTS = [
    "Explain how a hash table handles collisions.",
    "Write a Python function that checks whether a string is a palindrome, with a short docstring.",
]
VISION_CLASSES = {"resnet": "ResNetForImageClassification", "dinov2": "Dinov2Model",
                  "vit": "ViTModel"}


def copy_stats(m):
    """Device copies, cache hits, prefetched weights and the cache size (0201), if streamed."""
    w = getattr(m, "memopro_weights", None)
    if w is None or not hasattr(w, "copy_stats"):
        return None
    return {**w.copy_stats, "gpu_budget": w.gpu_budget, "gpu_cache_bytes": w.gpu_cache_bytes}


def sha(t):
    return hashlib.sha256(t.detach().float().cpu().numpy().tobytes()).hexdigest()[:16]


def vision_class(model_id):
    import transformers

    for key, name in VISION_CLASSES.items():
        if key in model_id.lower():
            return getattr(transformers, name)
    return transformers.AutoModel


def load(kind, model_id, mode, budget):
    import transformers

    import memopro.rt.torch as rtt

    path = model_path(model_id)
    cls = transformers.AutoModelForCausalLM if kind == "lm" else vision_class(model_id)
    t = time.time()
    if mode == "plain":
        dtype = torch.bfloat16 if kind == "lm" else None
        m = cls.from_pretrained(path, dtype=dtype).to(DEVICE).eval()
    else:
        m = rtt.stream_model(path, budget=budget, device=DEVICE, model_class=cls)
    sync()
    RESULT["load_s"] = round(time.time() - t, 1)
    RESULT["memory_after_load"] = MON.snapshot()
    return m, path


def lora_layers(m, device):
    """PEFT adapters of a streamed model start on the meta device: put them on the GPU (as
    memopro.finetune does)."""
    import peft

    for layer in m.modules():
        if isinstance(layer, peft.tuners.lora.LoraLayer):
            for name in layer.active_adapters:
                if layer.lora_A[name].weight.device.type == "meta":
                    layer.lora_A[name].to_empty(device=device)
                    layer.lora_B[name].to_empty(device=device)
                    layer.reset_lora_parameters(name, True)


def lm_gen(model_id, mode, budget, new_tokens):
    import transformers

    import memopro

    m, path = load("lm", model_id, mode, budget)
    tok = transformers.AutoTokenizer.from_pretrained(path)
    texts, secs = [], []
    for p in PROMPTS:
        t = time.time()
        if mode == "plain":
            msgs = [{"role": "user", "content": p}]
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
            ids = (ids["input_ids"] if hasattr(ids, "keys") else ids).to(DEVICE)
            with torch.no_grad():
                out = m.generate(input_ids=ids, max_new_tokens=new_tokens, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            texts.append(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
        else:
            texts.append(memopro.generate(m, p, tokenizer=tok, max_new_tokens=new_tokens))
        sync()
        secs.append(time.time() - t)
    finish(texts=texts, seconds=secs, s_per_token=sum(secs) / (new_tokens * len(PROMPTS)),
           rt=getattr(getattr(m, "memopro_runtime", None), "stats", lambda: None)(),
           copy_stats=copy_stats(m))


def lm_lora(model_id, mode, budget, steps, seq):
    import transformers

    import memopro
    import memopro.rt.torch as rtt
    from worker_common import wikitext_ids

    if mode == "memopro":
        m, path = load("lm", model_id, mode, budget)
        tok = transformers.AutoTokenizer.from_pretrained(path)
        ids = wikitext_ids(tok, "train", (seq - 1) * steps)
        texts = [tok.decode(ids[i * (seq - 1):(i + 1) * (seq - 1)]) for i in range(steps)]
        r = memopro.finetune(m, texts, tokenizer=tok, seq_len=seq)
        finish(losses=r.losses, loss_bits=[float.hex(x) for x in r.losses],
               step_s=r.step_seconds, tokens=r.tokens, rt=m.memopro_runtime.stats())
        return
    import peft

    m, path = load("lm", model_id, mode, budget)
    tok = transformers.AutoTokenizer.from_pretrained(path)
    ids = wikitext_ids(tok, "train", (seq - 1) * steps)
    texts = [tok.decode(ids[i * (seq - 1):(i + 1) * (seq - 1)]) for i in range(steps)]
    eos = tok.eos_token or ""
    all_ids = tok(eos.join(texts) + eos, return_tensors="pt").input_ids[0]
    torch.manual_seed(0)
    m = peft.get_peft_model(m, peft.LoraConfig(r=8, lora_alpha=16,
                                               target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    for p in m.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    m.train()
    m.base_model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    m.base_model.model.enable_input_require_grads()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=2e-4)
    losses, step_s = [], []
    pieces = [all_ids[i:i + seq] for i in range(0, len(all_ids) - 1, seq)]
    for piece in [p for p in pieces if len(p) > 1]:
        t = time.time()
        loss = rtt.causal_lm_loss(m.base_model.model, piece.unsqueeze(0).to(DEVICE))
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        sync()
        losses.append(float(loss.detach()))
        step_s.append(time.time() - t)
    finish(losses=losses, loss_bits=[float.hex(x) for x in losses], step_s=step_s)


def images(batch, size):
    g = torch.Generator().manual_seed(0)
    return torch.rand(batch, 3, size, size, generator=g)


def features(m, x):
    out = m(pixel_values=x)
    logits = getattr(out, "logits", None)
    return logits if logits is not None else out.last_hidden_state[:, 0]


def vis_infer(model_id, mode, budget, batch):
    m, _ = load("vision", model_id, mode, budget)
    x = images(batch, 224).to(DEVICE)
    with torch.no_grad():
        t = time.time()
        y = features(m, x)
        sync()
        first = time.time() - t
        t = time.time()
        y2 = features(m, x)
        sync()
    finish(output_sha=sha(y), same_twice=bool(torch.equal(y, y2)), shape=list(y.shape),
           first_s=first, second_s=time.time() - t,
           rt=getattr(getattr(m, "memopro_runtime", None), "stats", lambda: None)(),
           copy_stats=copy_stats(m))


def vis_lora(model_id, mode, budget, steps, batch):
    """LoRA on the attention projections + a fresh linear head (10 classes) over the class token;
    the same random images and labels at every step."""
    import peft

    import memopro.rt.torch as rtt

    m, _ = load("vision", model_id, mode, budget)
    names = {n.split(".")[-1] for n, mod in m.named_modules() if isinstance(mod, torch.nn.Linear)}
    targets = [t for t in ("query", "value", "q_proj", "v_proj") if t in names]
    torch.manual_seed(0)
    m = peft.get_peft_model(m, peft.LoraConfig(r=8, lora_alpha=16, target_modules=targets))
    lora_layers(m, DEVICE)
    for p in m.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    torch.manual_seed(1)
    hidden = m.base_model.model.config.hidden_size
    head = torch.nn.Linear(hidden, 10).to(DEVICE)
    params = [p for p in m.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3)
    x = images(batch, 224).to(DEVICE)
    y = torch.arange(batch, device=DEVICE) % 10
    streamed = hasattr(m.base_model.model, "memopro_weights")
    losses, step_s = [], []
    for _ in range(steps):
        t = time.time()
        ctx = rtt.saved_weights(m.base_model.model) if streamed else torch.enable_grad()
        with ctx:
            f = features(m.base_model.model, x).float()
            loss = torch.nn.functional.cross_entropy(head(f), y)
            loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        sync()
        losses.append(float(loss.detach()))
        step_s.append(time.time() - t)
    finish(losses=losses, loss_bits=[float.hex(v) for v in losses], step_s=step_s,
           targets=targets,
           rt=m.base_model.model.memopro_runtime.stats() if streamed else None)


def main():
    kind, model_id, mode, budget, *rest = sys.argv[1:]
    RESULT.update(kind=kind, model=model_id, mode=mode, budget=int(budget))
    torch.manual_seed(0)
    if kind.endswith("_lora"):  # the same steps give the same bits (CUBLAS_WORKSPACE_CONFIG set)
        torch.use_deterministic_algorithms(True, warn_only=True)
    fn = {"lm_gen": lm_gen, "lm_lora": lm_lora, "vis_infer": vis_infer, "vis_lora": vis_lora}[kind]
    fn(model_id, mode, int(budget), *map(int, rest))
    cleanup()


if __name__ == "__main__":
    guarded(main)
