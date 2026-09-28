"""E020 (report only): time of one forward pass over M prompt tokens, bf16 original (file-backed)
vs memopro's int4 kernel (`_weight_int4pack_mm`, file cache) vs the same int4 codes dequantized
per layer to bf16 (`evaluate.DequantLinear`). One fresh process per model and variant.

Usage: .venv/bin/python -m experiments.e020_int4_quality.prefill --model M --variant V
         --cache-dir DIR [--lengths 1,16,64,256,1024,2048]
"""

from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.1")

import torch

from experiments.e020_int4_quality.evaluate import OUT, load_variant


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--variant", required=True)  # bf16, int4-cache, int4deq-g64
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--lengths", default="1,16,64,256,1024,2048")
    args = parser.parse_args()
    import memopro

    memopro.configure(spill_dir=args.cache_dir, min_free_disk_fraction=0.0)
    model, _ = load_variant(args.model, args.variant)
    model.eval()
    rows = []
    with torch.no_grad():
        model(torch.full((1, 8), 1000, device="mps"), logits_to_keep=1)  # warm-up
        for m in (int(x) for x in args.lengths.split(",")):
            ids = torch.full((1, m), 1000, device="mps")
            times = []
            for _ in range(2 if m >= 1024 else 3):
                torch.mps.synchronize()
                t = time.perf_counter()
                model(ids, logits_to_keep=1)
                torch.mps.synchronize()
                times.append(time.perf_counter() - t)
            best = min(times)
            rows.append({"tokens": m, "seconds": best, "tokens_per_s": m / best})
            print(json.dumps({"variant": args.variant, **rows[-1]}), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"prefill_{args.model.split('/')[-1]}_{args.variant}.json"
    (OUT / name).write_text(json.dumps({"model": args.model, "variant": args.variant, "rows": rows}, indent=1) + "\n")


if __name__ == "__main__":
    main()
