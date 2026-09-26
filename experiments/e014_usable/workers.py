"""E014 workers: each case runs in a fresh process and prints one JSON line.

  infer      --model ID --arm naive|memopro|memopro_os|memopro_low|memopro_4gb
             load + greedy generation (memopro_4gb: budget "4GB!", the user accepts swapping)
  train      --arm naive|session --device mps|cpu --batch B --seq S --steps N
  alternate  --arm resident|reload|beta                     two 1.5B instances used in turns
  reclaim    --device cpu|mps                               footprint right after `source`

The orchestrator (run.py) sets MallocLargeCache and measures footprint, swap and responsiveness
from outside; the worker reports timings, what memopro chose, and outputs for comparison.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[2] / ".cache" / "hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch

PROMPT = "Explain in two sentences why a GPU can run out of memory."
GPT2 = ("openai-community/gpt2", "607a30d783dfa663caf39e06633721c8d4cfcd7e")
ALT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def footprint() -> int:
    from experiments.e011_os_swap.common import rusage

    return rusage(os.getpid())["phys_footprint"]


def device_of(model) -> str:
    return next(model.parameters()).device.type


def prompt_ids(tok, device):
    text = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}], tokenize=False, add_generation_prompt=True
    )
    return tok(text, return_tensors="pt").input_ids.to(device)


def generate(model, ids, n):
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=n, min_new_tokens=n, do_sample=False, pad_token_id=0
        )
    return out[0, ids.shape[1] :].tolist()


# ---------------------------------------------------------------- inference
def infer(args) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import memopro

    t0 = time.perf_counter()
    chosen = None
    if args.arm == "naive":
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto").to("mps")
    else:
        opts = {
            "memopro_os": {"budget_basis": "os"},
            "memopro_low": {"quality": "low"},
            "memopro_4gb": {"budget": "4GB!"},
        }.get(args.arm, {})
        model, tok = memopro.load(args.model, tokenizer=True, **opts)
        chosen = [e.technique for e in memopro.report().entries if e.action == "applied"]
    device = device_of(model)
    sync(device)
    load_s = time.perf_counter() - t0
    ids = prompt_ids(tok, device)
    generate(model, ids, 4)  # warm-up
    t1 = time.perf_counter()
    generate(model, ids, 1)
    ttft = time.perf_counter() - t1
    t2 = time.perf_counter()
    tokens = generate(model, ids, 64)
    gen_s = time.perf_counter() - t2
    return {
        "load_s": load_s,
        "ttft_s": ttft,
        "gen_s": gen_s,
        "tokens_per_s": 64 / gen_s,
        "device": device,
        "dtype": str(next(model.parameters()).dtype),
        "chosen": chosen,
        "tokens": tokens,
        "text": tok.decode(tokens),
    }


# ---------------------------------------------------------------- training
def train(args) -> dict:
    from transformers import GPT2LMHeadModel

    import memopro

    torch.manual_seed(0)
    model = GPT2LMHeadModel.from_pretrained(GPT2[0], revision=GPT2[1], dtype=torch.float32)
    model = model.to(args.device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    g = torch.Generator().manual_seed(0)
    batches = [
        torch.randint(0, 50257, (args.batch, args.seq), generator=g).to(args.device)
        for _ in range(args.steps)
    ]
    times, losses = [], []
    session = None
    if args.arm == "session":
        session = memopro.train_session(model, opt).__enter__()
    for x in batches:
        t = time.perf_counter()
        if session is None:
            loss = model(input_ids=x, labels=x).loss
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            value = float(loss)
        else:
            value = session.step({"input_ids": x, "labels": x}, lambda mb: model(**mb).loss)
        sync(args.device)
        times.append(time.perf_counter() - t)
        losses.append(value)
    out = {
        "step_s": times,
        "steady_step_s": statistics.median(times[1:]) if len(times) > 1 else times[0],
        "losses": losses,
    }
    if session is not None:
        out["micro_batch"] = session.micro
        out["report"] = [(e.technique, e.action) for e in memopro.report().entries]
        session.__exit__(None, None, None)
    return out


# ---------------------------------------------------------------- two models in turns (β)
def alternate(args) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from memopro import hibernate

    tok = AutoTokenizer.from_pretrained(ALT_MODEL)
    ids = prompt_ids(tok, "mps")
    load = lambda: AutoModelForCausalLM.from_pretrained(ALT_MODEL, dtype="auto").to("mps").eval()
    t0 = time.perf_counter()
    models: dict[str, object] = {}
    handles: dict[str, object] = {}
    uses, outputs = [], []
    for i, name in enumerate(["A", "B"] * 4):  # A, B, A, B, ... (the first two include loading)
        t = time.perf_counter()
        other = "B" if name == "A" else "A"
        if args.arm == "reload" and other in models:
            del models[other]
            gc.collect()
            torch.mps.empty_cache()
        if args.arm == "beta" and other in models and other not in handles:
            handles[other] = hibernate.now(models[other])
        handles.pop(name, None)  # a hibernated model wakes by itself when called
        if name not in models:
            models[name] = load()
        outputs.append(generate(models[name], ids, 32))
        torch.mps.synchronize()
        uses.append(
            {"use": i, "model": name, "s": time.perf_counter() - t, "footprint": footprint()}
        )
    return {
        "total_s": time.perf_counter() - t0,
        "turns_after_loading_s": sum(u["s"] for u in uses[2:]),
        "uses": uses,
        "outputs_equal": all(o == outputs[0] for o in outputs),
        "tokens": outputs[0],
    }


# ---------------------------------------------------------------- reclaim right after `source`
def reclaim(args) -> dict:
    from transformers import GPT2LMHeadModel

    from memopro import hibernate

    model = GPT2LMHeadModel.from_pretrained(GPT2[0], revision=GPT2[1], dtype=torch.float32)
    model = model.to(args.device).eval()
    if args.device == "cpu":
        for p in model.parameters():  # anonymous memory, as after training
            p.data = p.data.clone()
    nbytes = sum(p.numel() * p.element_size() for p in model.parameters())
    gc.collect()
    sync(args.device)
    before = footprint()
    t = time.perf_counter()
    h = hibernate.now(model, mode="source")
    gc.collect()
    if args.device == "mps":
        torch.mps.empty_cache()
    sleep_s = time.perf_counter() - t
    after = footprint()
    t = time.perf_counter()
    h.wake()
    sync(args.device)
    return {
        "model_bytes": nbytes,
        "reclaimed_frac": (before - after) / nbytes,
        "sleep_s": sleep_s,
        "wake_s": time.perf_counter() - t,
        "malloc_cache_off": os.environ.get("MallocLargeCache") == "0",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["infer", "train", "alternate", "reclaim"])
    parser.add_argument("--model")
    parser.add_argument("--arm", default="")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--steps", type=int, default=6)
    args = parser.parse_args()
    try:
        result = {"ok": True, **globals()[args.kind](args)}
    except Exception as e:  # noqa: BLE001 - a failure is a result (e.g. out of memory)
        result = {
            "ok": False,
            "error": f"{type(e).__name__}: {e}"[:2000],
            "trace": traceback.format_exc()[-3000:],
        }
    print("RESULT " + json.dumps(result), flush=True)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
