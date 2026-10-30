"""E053 worker (docs/research/0268): one case in a fresh process; prints one JSON line.

    worker.py lm_gen   <model_dir> <plain|memopro> <host_budget> <tokens> <prompts> [draft_dir]
    worker.py lm_lora  <model_dir> <plain|memopro> <host_budget> <steps> <seq> <batch|auto>
    worker.py vis_infer <model_dir> <class> <plain|memopro> <host_budget> <batch> <size>
    worker.py vis_lora  <model_dir> <class> <plain|memopro> <host_budget> <steps> <batch> <size>
    worker.py unsloth  <model_dir> <bits 16|4> <steps> <seq>   (run with Unsloth's own Python)

`plain` loads the model normally onto the GPU; `memopro` streams it within `host_budget` bytes
of host memory (memopro.rt.torch.stream_model / memopro.finetune). Self-contained: the notebook
writes this file into a fresh Colab runtime.
"""

import gc
import hashlib
import json
import os
import sys
import threading
import time
import traceback

if sys.argv[1] == "unsloth":
    import unsloth  # noqa: F401  (before transformers and peft, as Unsloth asks)
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULT = {"task": sys.argv[1], "argv": sys.argv[2:], "device": DEVICE, "status": None}
PROMPTS = [
    "Explain how a hash table handles collisions.",
    "Write a Python function that checks whether a string is a palindrome, with a short docstring.",
]


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def host_peak():
    """VmHWM: this process's own peak resident memory (ru_maxrss survives execve, 0199)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    import resource

    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r if sys.platform == "darwin" else r * 1024


class GpuMonitor:
    """Peak GPU memory in use as nvidia-smi sees it, sampled every 20 ms."""

    def __init__(self):
        self.peak = 0
        if DEVICE == "cuda":
            threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            free, total = torch.cuda.mem_get_info()
            self.peak = max(self.peak, total - free)
            time.sleep(0.02)


MON = GpuMonitor()


def finish(**kw):
    RESULT.update(kw)
    RESULT["status"] = RESULT["status"] or "ok"


def sha(t):
    return hashlib.sha256(t.detach().float().cpu().numpy().tobytes()).hexdigest()[:16]


def stats_of(m):
    rt = getattr(m, "memopro_runtime", None)
    return rt.stats() if rt is not None else None


def wikitext(tok, n_tokens):
    """The first `n_tokens` tokens of WikiText-2's training split (the same text every case)."""
    import datasets

    ds = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    text = "".join(t for t in ds["text"][:4000] if t.strip())
    return tok(text, return_tensors="pt").input_ids[0][:n_tokens]


def targets_of(m):
    """LoRA targets present in this architecture: separate q/k/v/o (Llama, Qwen, Mistral) or
    fused qkv (Phi-3)."""
    names = {n.split(".")[-1] for n, x in m.named_modules() if isinstance(x, torch.nn.Linear)}
    for group in (
        ("q_proj", "k_proj", "v_proj", "o_proj"),
        ("qkv_proj", "o_proj"),
        ("query", "key", "value"),
        ("q_proj", "v_proj"),
    ):
        if all(g in names for g in group):
            return list(group)
    raise RuntimeError(f"no known attention projections among {sorted(names)[:20]}")


def load_lm(path, mode, budget):
    import transformers

    import memopro.rt.torch as rtt

    t = time.time()
    if mode == "plain":
        m = transformers.AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
        m = m.to(DEVICE).eval()
    else:
        m = rtt.stream_model(path, budget=budget, device=DEVICE)
    sync()
    RESULT["load_s"] = round(time.time() - t, 1)
    return m, transformers.AutoTokenizer.from_pretrained(path)


def lm_gen(path, mode, budget, new_tokens, prompts=None, draft=None):
    import memopro
    import memopro.rt.torch as rtt

    m, tok = load_lm(path, mode, int(budget))
    if draft:  # a resident draft proposes tokens; the text must not change (0276)
        draft = rtt.draft_model(draft, target=m, device=DEVICE)
        RESULT["draft_bytes"] = draft.memopro_draft_bytes
        RESULT["draft_dtype"] = str(next(draft.parameters()).dtype)
    tokens, secs = [], []
    for p in PROMPTS[: int(prompts or len(PROMPTS))]:
        msgs = [{"role": "user", "content": p}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        ids = (ids["input_ids"] if hasattr(ids, "keys") else ids).to(DEVICE)
        kw = {
            "max_new_tokens": int(new_tokens),
            "min_new_tokens": int(new_tokens),
            "pad_token_id": tok.eos_token_id,
        }
        t = time.time()
        if mode == "plain":
            with torch.no_grad():
                out = m.generate(input_ids=ids, do_sample=False, **kw)
        else:
            out = rtt.generate(m, ids, draft=draft or None, **kw)
        sync()
        secs.append(time.time() - t)
        tokens.append(out[0, ids.shape[1] :].tolist())
    finish(
        tokens=tokens,
        text=tok.decode(tokens[0], skip_special_tokens=True)[:200],
        seconds=secs,
        s_per_token=sum(secs) / (int(new_tokens) * len(secs)),
        rt=stats_of(m),
        copy_stats=getattr(getattr(m, "memopro_weights", None), "copy_stats", None),
        memopro=memopro.__version__,
    )


def lm_lora(path, mode, budget, steps, seq, batch):
    import transformers

    import memopro
    import memopro.rt.torch as rtt

    steps, seq = int(steps), int(seq)
    tok = transformers.AutoTokenizer.from_pretrained(path)
    ids = wikitext(tok, (seq - 1) * steps)
    texts = [tok.decode(ids[i * (seq - 1) : (i + 1) * (seq - 1)]) for i in range(steps)]
    # decoding and tokenizing again can add tokens (Mistral, 0270): cut into `steps` pieces exactly
    eos = tok.eos_token or ""
    seq = -(-(len(tok(eos.join(texts) + eos).input_ids) - 1) // steps)
    if mode == "memopro":
        m = rtt.stream_model(path, budget=int(budget), device=DEVICE)
        targets = targets_of(m)  # before PEFT wraps them
        r = memopro.finetune(
            m,
            texts,
            tokenizer=tok,
            seq_len=seq,
            targets=tuple(targets),
            batch=batch if batch == "auto" else int(batch),
        )
        finish(
            losses=r.losses,
            loss_bits=[float.hex(x) for x in r.losses],
            batch=r.batch,
            step_s=r.step_seconds,
            tokens=r.tokens,
            rt=stats_of(m),
            targets=targets,
        )
        return
    import peft

    m = transformers.AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(DEVICE)
    all_ids = tok(eos.join(texts) + eos, return_tensors="pt").input_ids[0]
    torch.manual_seed(0)
    targets = targets_of(m)
    m = peft.get_peft_model(m, peft.LoraConfig(r=8, lora_alpha=16, target_modules=targets))
    for p in m.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    m.train()
    base = m.base_model.model
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    base.enable_input_require_grads()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=2e-4)
    losses, step_s = [], []
    for i in range(0, len(all_ids) - 1, seq):
        piece = all_ids[i : i + seq]
        if len(piece) < 2:
            continue
        t = time.time()
        loss = rtt.causal_lm_loss(base, piece.unsqueeze(0).to(DEVICE))
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        sync()
        losses.append(float(loss.detach()))
        step_s.append(time.time() - t)
    finish(losses=losses, loss_bits=[float.hex(x) for x in losses], step_s=step_s, targets=targets)


def vision(path, cls_name, mode, budget):
    import transformers

    import memopro.rt.torch as rtt

    cls = getattr(transformers, cls_name)
    t = time.time()
    if mode == "plain":
        m = cls.from_pretrained(path).to(DEVICE).eval()
    else:
        m = rtt.stream_model(path, budget=int(budget), device=DEVICE, model_class=cls)
    sync()
    RESULT["load_s"] = round(time.time() - t, 1)
    return m


def images(batch, size):
    g = torch.Generator().manual_seed(0)
    return torch.rand(int(batch), 3, int(size), int(size), generator=g)


def features(m, x):
    out = m(pixel_values=x)
    logits = getattr(out, "logits", None)
    if logits is not None:
        return logits
    pooled = getattr(out, "pooler_output", None)
    return pooled if pooled is not None else out.last_hidden_state[:, 0]


def vis_infer(path, cls_name, mode, budget, batch, size):
    m = vision(path, cls_name, mode, budget)
    x = images(batch, size).to(DEVICE)
    with torch.no_grad():
        t = time.time()
        y = features(m, x)
        sync()
        first = time.time() - t
        t = time.time()
        y2 = features(m, x)
        sync()
    finish(
        output_sha=sha(y),
        same_twice=bool(torch.equal(y, y2)),
        shape=list(y.shape),
        first_s=first,
        second_s=time.time() - t,
        rt=stats_of(m),
    )


def vis_lora(path, cls_name, mode, budget, steps, batch, size):
    """LoRA on the attention projections + a fresh 10-class head; deterministic kernels only
    (strict, cuBLAS workspace fixed, attention on the math kernel: 0270)."""
    import peft
    from torch.nn.attention import SDPBackend, sdpa_kernel

    import memopro.rt.torch as rtt

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # before cuBLAS starts
    torch.use_deterministic_algorithms(True)
    m = vision(path, cls_name, mode, budget)
    torch.manual_seed(0)
    m = peft.get_peft_model(m, peft.LoraConfig(r=8, lora_alpha=16, target_modules=targets_of(m)))
    for layer in m.modules():  # adapters of a streamed model start on the meta device
        if isinstance(layer, peft.tuners.lora.LoraLayer):
            for name in layer.active_adapters:
                if layer.lora_A[name].weight.device.type == "meta":
                    layer.lora_A[name].to_empty(device=DEVICE)
                    layer.lora_B[name].to_empty(device=DEVICE)
                    layer.reset_lora_parameters(name, True)
    for p in m.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    base = m.base_model.model
    torch.manual_seed(1)
    head = torch.nn.Linear(base.config.hidden_size, 10).to(DEVICE)
    opt = torch.optim.AdamW(
        [p for p in m.parameters() if p.requires_grad] + list(head.parameters()), lr=1e-3
    )
    x = images(batch, size).to(DEVICE)
    y = torch.arange(int(batch), device=DEVICE) % 10
    streamed = hasattr(base, "memopro_weights")
    losses, step_s = [], []
    for _ in range(int(steps)):
        t = time.time()
        with (
            sdpa_kernel(SDPBackend.MATH),
            rtt.saved_weights(base) if streamed else torch.enable_grad(),
        ):
            loss = torch.nn.functional.cross_entropy(head(features(base, x).float()), y)
            loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        sync()
        losses.append(float(loss.detach()))
        step_s.append(time.time() - t)
    finish(
        losses=losses, loss_bits=[float.hex(v) for v in losses], step_s=step_s, rt=stats_of(base)
    )


def unsloth_lora(path, bits, steps, seq):
    import unsloth as us
    from unsloth import FastLanguageModel

    steps, seq = int(steps), int(seq)
    RESULT["unsloth"] = getattr(us, "__version__", "?")
    t = time.time()
    m, tok = FastLanguageModel.from_pretrained(
        model_name=path, max_seq_length=seq, dtype=None, load_in_4bit=bits == "4"
    )
    RESULT["load_s"] = round(time.time() - t, 1)
    m = FastLanguageModel.get_peft_model(
        m,
        r=8,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=0,
    )
    ids = wikitext(tok, (seq - 1) * steps)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=2e-4)
    losses, step_s = [], []
    eos = torch.tensor([tok.eos_token_id])
    for i in range(steps):
        x = torch.cat([ids[i * (seq - 1) : (i + 1) * (seq - 1)], eos])[None].to(DEVICE)
        t = time.time()
        loss = m(input_ids=x, labels=x).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        sync()
        losses.append(float(loss.detach()))
        step_s.append(time.time() - t)
    finish(losses=losses, step_s=step_s, dtype=str(m.dtype))


def main():
    task, args = sys.argv[1], sys.argv[2:]
    fn = {
        "lm_gen": lm_gen,
        "lm_lora": lm_lora,
        "vis_infer": vis_infer,
        "vis_lora": vis_lora,
        "unsloth": unsloth_lora,
    }[task]
    try:
        fn(*args)
    except torch.cuda.OutOfMemoryError as e:
        RESULT.update(status="oom", error=str(e)[:400])
    except MemoryError as e:
        RESULT.update(status="oom", error=repr(e)[:400])
    except Exception as e:  # noqa: BLE001 - every failure is a recorded result
        RESULT.update(
            status="error",
            error=f"{type(e).__name__}: {e}"[:600],
            traceback=traceback.format_exc()[-2500:],
        )
    gc.collect()
    RESULT["host_peak_bytes"] = host_peak()
    RESULT["gpu_peak_bytes"] = MON.peak
    if DEVICE == "cuda":
        RESULT["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    print("\n" + json.dumps(RESULT, default=str), flush=True)


if __name__ == "__main__":
    main()
