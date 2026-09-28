"""E019: MPS memory kept after one forward pass, by prompt length (fresh process per length).

Usage: .venv/bin/python -m experiments.e019_footprint.prompt_len --model M --kind int4|bf16
         --cache-dir DIR --length N
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[2] / ".cache" / "hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--kind", default="int4")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--kv", action="store_true", help="prefill into a DynamicCache")
    parser.add_argument("--decode", type=int, default=0, help="then decode N tokens")
    parser.add_argument("--attn", default=None, help="attention implementation, e.g. eager")
    parser.add_argument("--hooks", action="store_true", help="print modules that grow MPS memory")
    parser.add_argument("--clone-head", action="store_true", help="lm_head weight as a normal tensor")
    parser.add_argument("--keep", type=int, default=0, help="logits_to_keep for the prefill")
    args = parser.parse_args()
    import torch

    import memopro
    from experiments.e011_os_swap.common import rusage
    from experiments.e017_controller.workers import load

    memopro.configure(spill_dir=args.cache_dir, min_free_disk_fraction=0.0)
    model, _, _ = load(args.model, args.kind)
    if args.clone_head:
        w = model.lm_head.weight
        model.lm_head.weight = torch.nn.Parameter(w.detach().clone(), requires_grad=False)
        torch.mps.synchronize()
    if args.attn:
        model.set_attn_implementation(args.attn)
    ids = torch.full((1, args.length), 1000, dtype=torch.long, device="mps")
    fp = lambda: rusage(os.getpid())["phys_footprint"] / 2**20
    before, drv0 = fp(), torch.mps.driver_allocated_memory() / 2**20
    marks = {}
    if args.hooks:
        last = [torch.mps.driver_allocated_memory()]

        def hook(mod, inp, out, name=""):
            torch.mps.synchronize()
            now = torch.mps.driver_allocated_memory()
            if now - last[0] > 50 * 2**20:
                print(json.dumps({"module": name, "type": type(mod).__name__,
                                  "grew_mib": round((now - last[0]) / 2**20)}))
            last[0] = now

        for name, mod in model.named_modules():
            if not list(mod.children()):
                mod.register_forward_hook(lambda m, i, o, n=name: hook(m, i, o, n))
    with torch.no_grad():
        if args.kv:
            from transformers import DynamicCache

            cache = DynamicCache()
            out = model(ids, past_key_values=cache, use_cache=True, logits_to_keep=args.keep)
            torch.mps.synchronize()
            marks["after_prefill_mib"] = round(fp() - before)
            nxt = out.logits[:, -1:].argmax(-1)
            del out
            for _ in range(args.decode):
                nxt = model(nxt, past_key_values=cache, use_cache=True).logits[:, -1:].argmax(-1)
            del cache, nxt
        else:
            model(ids, use_cache=False)
    torch.mps.synchronize()
    import gc

    gc.collect()
    rec = {**marks,
        "kind": args.kind,
        "length": args.length,
        "footprint_growth_mib": round(fp() - before),
        "driver_growth_mib": round(torch.mps.driver_allocated_memory() / 2**20 - drv0),
        "recommended_max_mib": round(torch.mps.recommended_max_memory() / 2**20),
    }
    torch.mps.empty_cache()
    rec["after_empty_cache_mib"] = round(fp() - before)
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
