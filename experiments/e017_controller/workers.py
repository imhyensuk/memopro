"""E017 worker: one case in a fresh process, three phases (A calm, B pressure, C released).

  --arm bf16      memopro.load(residency="file") as stored (bf16 original file), no controller
  --arm bf16_ctl  bf16 and int4 both file-backed, the controller may switch and pace
  --arm int4      memopro.load(residency="file", quality="low") int4 file cache, no controller
  --arm int4_ctl  the int4 file cache with the controller pacing rereads

Prints "A {json}", then for each "go B" / "go C" line on stdin prints "B {json}" / "C {json}".
Each phase decodes greedily from the prompt with a fresh KV cache (controller state persists).
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


PROMPT = "Explain in two sentences why a GPU can run out of memory."
TOKENS = {"A": 64, "B": 128, "C": 64}


def load(model_id: str, kind: str):
    import memopro
    from memopro.access._load import plan_load

    if kind == "bf16":
        model = memopro.load(model_id, residency="file", budget="64GB!")
    else:
        int4 = next(
            c for c in plan_load(model_id, device="mps").candidates if c.name == "quant.int4"
        )
        need = int4.needs.device + int4.needs.host + (100 << 20)
        # forced: the bf16 mapping may already be loaded and must not shrink the int4 budget
        model = memopro.load(
            model_id, residency="file", quality="low", budget=f"{need / 1e9:.2f}GB!"
        )
    maps = list(getattr(model, "_memopro_mappings", []))
    return model, maps, sum(m.map.file_length for m in maps)


def main() -> None:
    from experiments.e017_controller.controller import Controller, Policy, generate

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--mode", default="comfort", choices=["comfort", "quality"])
    args = parser.parse_args()
    import memopro

    memopro.configure(spill_dir=args.cache_dir, min_free_disk_fraction=0.0)
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model)
        text = tok.apply_chat_template(
            [{"role": "user", "content": PROMPT}], tokenize=False, add_generation_prompt=True
        )
        ids = tok(text, return_tensors="pt").input_ids.to("mps")
        t0 = time.perf_counter()
        kinds = {
            "bf16": ["bf16"],
            "bf16_ctl": ["bf16", "int4"],
            "int4": ["int4"],
            "int4_ctl": ["int4"],
        }[args.arm]
        models, maps, sizes = {}, {}, {}
        for k in kinds:
            models[k], maps[k], sizes[k] = load(args.model, k)
        load_s = time.perf_counter() - t0
        ctl = None
        if args.arm.endswith("_ctl"):
            ctl = Controller(models, maps, sizes, Policy(switch=args.arm == "bf16_ctl", pace=True, mode=args.mode))
        start = ctl.start() if ctl is not None else kinds[0]
        model = models[start]
        generate(None, model, ids, 4)  # warm-up: every weight of the start precision read once
        if ctl is not None:
            ctl.arm()
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
    phase = "A"
    while True:
        try:
            start_log = len(ctl.log) if ctl else 0
            if ctl is not None:
                ctl.reset()
            t = time.perf_counter()
            tokens, _ = generate(ctl, model, ids, TOKENS[phase])
            dt = time.perf_counter() - t
            rec = {
                "ok": True,
                "tokens_per_s": len(tokens) / dt,
                "gen_s": dt,
                "tokens": tokens,
                "active_end": ctl.active if ctl else kinds[0],
                "log": ctl.log[start_log:] if ctl else [],
                "resident": {
                    k: min((m.resident() for m in v), default=1.0) for k, v in maps.items()
                },
            }
            if phase == "A":
                rec["load_s"] = load_s
                rec["start"] = start
            print(f"{phase} " + json.dumps(rec), flush=True)
        except Exception as e:  # noqa: BLE001
            print(
                f"{phase} " + json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"[:1500]}),
                flush=True,
            )
        line = sys.stdin.readline().strip()
        if not line.startswith("go "):
            return
        phase = line.split()[1]


if __name__ == "__main__":
    main()
