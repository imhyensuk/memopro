"""E020 follow-up (0085): before applying Q-b/Q-c, measure (1) decoding speed of the int4 kernel
at group 32 vs 64, (2) one forward pass over M input rows with the kernel only vs the dequantizing
path only, to place `int4pack.LONG_INPUT`. One fresh process per model and group.

Usage: .venv/bin/python -m experiments.e020_int4_quality.followup --model M --group 32|64
"""

from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch

from experiments.e020_int4_quality.evaluate import OUT

LENGTHS = (1, 32, 64, 96, 128, 192, 256, 384, 512, 1024)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--group", type=int, required=True)
    args = parser.parse_args()
    from transformers import AutoModelForCausalLM, DynamicCache

    from memopro.techniques.integrations import int4pack

    int4pack.GROUP = args.group  # convert() reads the module constant
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": "cpu"}
    )
    int4pack.convert(model, "mps")
    model.eval()
    rec = {"model": args.model, "group": args.group, "decode": [], "forward": []}
    with torch.no_grad():
        ids = torch.full((1, 16), 1000, device="mps")
        for _ in range(3):  # decoding: 64 tokens after a 16-token prompt, kernel path
            cache = DynamicCache()
            out = model(ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
            nxt = out.logits[:, -1:].argmax(-1)
            torch.mps.synchronize()
            t = time.perf_counter()
            for _ in range(64):
                nxt = model(nxt, past_key_values=cache, use_cache=True).logits[:, -1:].argmax(-1)
            torch.mps.synchronize()
            rec["decode"].append(64 / (time.perf_counter() - t))
        for m in LENGTHS:
            x = torch.full((1, m), 1000, device="mps")
            row = {"tokens": m}
            for name, threshold in (("kernel", 1 << 30), ("dequant", 0)):
                int4pack.LONG_INPUT = threshold
                times = []
                for _ in range(2):
                    torch.mps.synchronize()
                    t = time.perf_counter()
                    model(x, logits_to_keep=1)
                    torch.mps.synchronize()
                    times.append(time.perf_counter() - t)
                row[name] = min(times)
            rec["forward"].append(row)
            print(json.dumps(row), flush=True)
    rec["decode_tok_s"] = sorted(rec["decode"])[1]
    print(json.dumps({"decode_tok_s": rec["decode_tok_s"], "all": rec["decode"]}), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"followup_{args.model.split('/')[-1]}_g{args.group}.json"
    (OUT / name).write_text(json.dumps(rec, indent=1) + "\n")


if __name__ == "__main__":
    main()
