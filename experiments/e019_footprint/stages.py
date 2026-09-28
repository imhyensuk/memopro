"""E019: where does int4 `load(residency="file")` get its extra ~1 GB footprint (0078 §해석 3)?

One fresh process per run. Prints the process footprint and the MPS allocator's view after each
stage, for the models loaded in the given order (as the E017 workers do).

Usage: .venv/bin/python -m experiments.e019_footprint.stages --model M --order int4|bf16,int4|bf16
         --cache-dir DIR
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[2] / ".cache" / "hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

MiB = 2**20
_peak = [0]
MPS_STATS = True


def _watch() -> None:
    """Peak footprint between snapshots (E017's A peak was sampled from process start)."""
    import time

    from experiments.e011_os_swap.common import rusage

    while True:
        _peak[0] = max(_peak[0], rusage(os.getpid())["phys_footprint"])
        time.sleep(0.005)


def snap(stage: str, out: list) -> None:
    import torch

    from experiments.e011_os_swap.common import rusage

    r = rusage(os.getpid())
    rec = {
        "stage": stage,
        "footprint_mib": r["phys_footprint"] / MiB,
        "resident_mib": r["resident_size"] / MiB,
        "peak_since_last_mib": max(_peak[0], r["phys_footprint"]) / MiB,
    }
    _peak[0] = 0
    if MPS_STATS:
        rec["mps_alloc_mib"] = torch.mps.current_allocated_memory() / MiB
        rec["mps_driver_mib"] = torch.mps.driver_allocated_memory() / MiB
    out.append(rec)
    print(json.dumps({k: round(v, 1) if isinstance(v, float) else v for k, v in rec.items()}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--order", default="int4")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--no-mps-stats", action="store_true")
    parser.add_argument("--chat", action="store_true", help="the E017 prompt (chat template)")
    args = parser.parse_args()
    global MPS_STATS
    MPS_STATS = not args.no_mps_stats
    out: list = []
    import threading

    threading.Thread(target=_watch, daemon=True).start()
    import torch

    snap("torch imported", out)
    import memopro
    from experiments.e017_controller.controller import generate
    from experiments.e017_controller.workers import load

    memopro.configure(spill_dir=args.cache_dir, min_free_disk_fraction=0.0)
    snap("memopro imported", out)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    if args.chat:
        from experiments.e017_controller.workers import PROMPT

        text = tok.apply_chat_template(
            [{"role": "user", "content": PROMPT}], tokenize=False, add_generation_prompt=True
        )
    else:
        text = "Explain why a GPU can run out of memory."
    ids = tok(text, return_tensors="pt").input_ids.to("mps")
    snap("tokenizer + ids on mps", out)
    models = {}
    for kind in args.order.split(","):
        models[kind], _, _ = load(args.model, kind)
        snap(f"loaded {kind}", out)
    model = models[args.order.split(",")[-1]]
    generate(None, model, ids, 4)
    snap("warm-up 4 tokens", out)
    generate(None, model, ids, 64)
    snap("64 tokens", out)
    torch.mps.empty_cache()
    snap("empty_cache", out)


if __name__ == "__main__":
    main()
