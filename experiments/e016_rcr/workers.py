"""E016 worker: one inference case in a fresh process, in two phases.

  --arm naive          from_pretrained(dtype="auto").to("mps")       (anonymous bf16, as E014)
  --arm memopro_int4   memopro.load(quality="low"), int4 via torch int4pack (anonymous, 0069)
  --arm rcr_bf16       RCR F class: the original safetensors mapped, no copy, + prefetch
  --arm rcr_int4       RCR F class: the int4 F file mapped, no copy, + prefetch
  --arm rcr_int4_nopf  the same without prefetch (ablation)

Phase A prints "A {json}" (load, first-token time, 64-token speed, tokens). The worker then waits
for "go" on stdin (the orchestrator starts the memory pressure), generates 64 tokens again and
prints "B {json}".
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[2] / ".cache" / "hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch

PROMPT = "Explain in two sentences why a GPU can run out of memory."


def generate(model, ids, n):
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=n, min_new_tokens=n, do_sample=False)
    torch.mps.synchronize()
    return out[0, ids.shape[1] :].tolist()


def load(args):
    files = []
    if args.arm == "naive":
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto").to("mps")
    elif args.arm == "memopro_int4":
        import memopro
        from memopro.access._load import plan_load

        int4 = next(
            c for c in plan_load(args.model, device="mps").candidates if c.name == "quant.int4"
        )
        need = int4.needs.device + int4.needs.host
        budget = f"{(need + 10**8) / 1e9:.1f}GB!"  # just enough for int4, the same every run
        model = memopro.load(args.model, quality="low", budget=budget)
    else:
        from experiments.e016_rcr.fclass import Prefetcher, load_bf16, load_int4

        if args.arm == "rcr_bf16":
            model, files, ranges = load_bf16(args.model)
        else:
            model, files, ranges = load_int4(args.model, args.ffile)
        if args.arm != "rcr_int4_nopf":
            Prefetcher(model, ranges)
    torch.mps.synchronize()
    return model, files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--ffile", default="")
    args = parser.parse_args()
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model)
        text = tok.apply_chat_template(
            [{"role": "user", "content": PROMPT}], tokenize=False, add_generation_prompt=True
        )
        ids = tok(text, return_tensors="pt").input_ids.to("mps")
        t0 = time.perf_counter()
        model, files = load(args)
        load_s = time.perf_counter() - t0
        generate(model, ids, 4)  # warm-up
        t = time.perf_counter()
        generate(model, ids, 1)
        ttft = time.perf_counter() - t
        t = time.perf_counter()
        tokens = generate(model, ids, 64)
        gen_s = time.perf_counter() - t
        resident = [f.resident() for f in files]
        print(
            "A "
            + json.dumps(
                {
                    "ok": True,
                    "load_s": load_s,
                    "ttft_s": ttft,
                    "gen_s": gen_s,
                    "tokens_per_s": 64 / gen_s,
                    "tokens": tokens,
                    "resident": resident,
                }
            ),
            flush=True,
        )
    except Exception as e:  # noqa: BLE001 - a failure is a result
        print(
            "A "
            + json.dumps(
                {
                    "ok": False,
                    "error": f"{type(e).__name__}: {e}"[:1500],
                    "trace": traceback.format_exc()[-2000:],
                }
            ),
            flush=True,
        )
        return
    if sys.stdin.readline().strip() != "go":
        return
    try:
        before = [f.resident() for f in files]
        t = time.perf_counter()
        tokens_b = generate(model, ids, 64)
        gen_b = time.perf_counter() - t
        print(
            "B "
            + json.dumps(
                {
                    "ok": True,
                    "gen_s": gen_b,
                    "tokens_per_s": 64 / gen_b,
                    "same_tokens": tokens_b == tokens,
                    "resident_before": before,
                    "resident_after": [f.resident() for f in files],
                }
            ),
            flush=True,
        )
    except Exception as e:  # noqa: BLE001
        print(
            "B " + json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"[:1500]}), flush=True
        )


if __name__ == "__main__":
    main()
