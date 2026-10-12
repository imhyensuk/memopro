"""E051 (docs/research/0252): S2 — `finetune(batch=k)` trains k sequences per read of the
streamed weights. Qwen2.5-7B 16-bit LoRA on the 8 GB M1's GPU at a 2 GiB budget, 16 sequences
of 129 tokens (8 texts of E034, 2 epochs) with batch 1, 2 and 4, each twice in the order
1 2 4 4 2 1, and batch 2 once more at 2.5 GiB (losses identical across budgets).

    caffeinate -i .venv/bin/python -m experiments.e051_batch.run --all

Results: docs/research/data/e051/ (cases/*.json, summary.md, env.json).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

from experiments.e028_mps_lora.run import footprint, swap_used

ROOT = Path(__file__).resolve().parents[2]
MIB = 1 << 20
MODEL = "Qwen/Qwen2.5-7B-Instruct"
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
TEXTS, EPOCHS, SEQ = 8, 2, 129
# E051 (0252): 2 GiB, watermark 0.1 as E036b-E038b. E051b (0254): 2.5 GiB and the watermark
# stream_model itself sets (0.01): here MPS starts before stream_model (for the baseline
# footprint), too late for stream_model to set it, so the harness sets it.
VARIANT = os.environ.get("E051_VARIANT", "e051")
OUT = ROOT / "docs" / "research" / "data" / VARIANT
BUDGET, WATERMARK = {"e051": (2048 * MIB, "0.1"), "e051b": (2560 * MIB, "0.01")}[VARIANT]
# name -> (batch, budget)
CASES = [
    ("b1-1", 1, BUDGET),
    ("b2-1", 2, BUDGET),
    ("b4-1", 4, BUDGET),
    ("b4-2", 4, BUDGET),
    ("b2-2", 2, BUDGET),
    ("b1-2", 1, BUDGET),
    *([("b2-big", 2, 2560 * MIB)] if VARIANT == "e051" else [("b4-small", 4, 2048 * MIB)]),
]


def case(batch: int, budget: int) -> dict:
    import peft  # noqa: F401 - imported before the baseline, as torch and transformers
    import torch
    import transformers  # noqa: F401

    import memopro
    import memopro.rt.torch as rtt

    torch.ones(1, device="mps")
    torch.mps.synchronize()
    texts = json.loads((ROOT / "docs/research/data/e034/texts.json").read_text())[:TEXTS]
    base, _ = footprint()
    swap0 = swap_used()
    model = rtt.stream_model(MODEL, budget=budget, device="mps", revision=REVISION)
    r = memopro.finetune(model, texts, seq_len=SEQ, epochs=EPOCHS, batch=batch)
    _, peak = footprint()
    return {
        "batch": batch,
        "budget": budget,
        "limit": model.memopro_runtime.limit,
        "held_back": r.held_back,
        "tokens": r.tokens,
        "losses": r.losses,
        "step_s": r.step_seconds,
        "base_footprint": base,
        "peak_footprint": peak,
        "swap_before": swap0,
        "swap_after": swap_used(),
        "rt": model.memopro_runtime.stats(),
    }


def steady(r: dict) -> float:
    """Tokens per second over every step but the first."""
    per_step = r["tokens"] // len(r["step_s"])  # all steps hold the same number of sequences
    return per_step * (len(r["step_s"]) - 1) / sum(r["step_s"][1:])


def summarize() -> str:
    recs = {p.stem: json.loads(p.read_text()) for p in sorted((OUT / "cases").glob("*.json"))}
    lines = [f"# {VARIANT.upper()} summary (0252, 0254)", "",
             ("| case | batch | budget MiB | steps | step s (median) | tok/s (steps 2+) | "
              "re-read GiB | footprint +MiB | swap MiB |"), "|---|---|---|---|---|---|---|---|---|"]
    for name, r in recs.items():
        if "error" in r:
            lines.append(f"| {name} | error: {r['error'][-200:]!r} | | | | | | | |")
            continue
        lines.append(
            f"| {name} | {r['batch']} | {r['budget'] // MIB} | {len(r['step_s'])} | "
            f"{statistics.median(r['step_s']):.1f} | {steady(r):.1f} | "
            f"{r['rt']['reread_bytes'] / (1 << 30):.0f} | "
            f"{(r['peak_footprint'] - r['base_footprint']) // MIB} | "
            f"{(r['swap_after'] - r['swap_before']) // MIB:+d} |")
    ok = {n: r for n, r in recs.items() if "error" not in r}

    def same(a: str, b: str) -> bool:
        return a in ok and b in ok and ok[a]["losses"] == ok[b]["losses"]

    k1 = same("b1-1", "b1-2") and same("b2-1", "b2-2") and same("b4-1", "b4-2")
    k1b = same("b2-1", "b2-big") if VARIANT == "e051" else same("b4-1", "b4-small")
    rate = {k: statistics.median(steady(ok[n]) for n in ok if n.startswith(f"b{k}-") and
                                 ok[n]["budget"] == BUDGET) for k in (1, 2, 4)
            if any(n.startswith(f"b{k}-") for n in ok)}
    over = [n for n, r in ok.items()
            if r["peak_footprint"] - r["base_footprint"] > r["budget"] + 512 * MIB]
    lines += ["", f"- K1 repeated runs identical: {'yes' if k1 else 'NO'}",
              f"- K1 same batch, another budget: {'yes' if k1b else 'NO'}",
              f"- K2 footprint within budget + 512 MiB: {'yes' if not over else 'NO ' + str(over)}"]
    if 1 in rate:
        lines += ["- tok/s median: " + ", ".join(f"batch {k} {v:.1f} ({v / rate[1]:.2f}x)"
                                                 for k, v in rate.items())]
    if 1 in rate and 4 in rate:
        k3 = rate[4] / rate[1] >= 1.8
        lines.append(f"- K3 batch 4 >= 1.8x batch 1: {'yes' if k3 else 'NO'}")
        lines.append(f"\nDecision (0252/0254): {'PASS' if k1 and k1b and k3 else 'FAIL'}")
    return "\n".join(lines) + "\n"


def run_all() -> None:
    (OUT / "cases").mkdir(parents=True, exist_ok=True)
    env = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                 text=True, check=False).stdout.strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "swap_at_start": swap_used(),
    }
    (OUT / "env.json").write_text(json.dumps(env, indent=1))
    child = {**os.environ, "MallocLargeCache": "0", "PYTORCH_MPS_LOW_WATERMARK_RATIO": WATERMARK,
             "HF_HOME": str(ROOT / ".cache" / "hf"), "HF_HUB_OFFLINE": "1",
             "TOKENIZERS_PARALLELISM": "false"}
    for name, batch, budget in CASES:
        print(f"[{time.strftime('%H:%M:%S')}] {name}", flush=True)
        p = subprocess.run(
            [sys.executable, "-m", "experiments.e051_batch.run", "--case", str(batch), str(budget)],
            cwd=ROOT, capture_output=True, text=True, env=child, check=False)
        rec = ({"case": name, "error": p.stderr[-3000:]} if p.returncode != 0
               else {"case": name, **json.loads(p.stdout.strip().splitlines()[-1])})
        (OUT / "cases" / f"{name}.json").write_text(json.dumps(rec, indent=1))
        print(f"   -> {'error' if 'error' in rec else f'{steady(rec):.1f} tok/s'}", flush=True)
    text = summarize()
    (OUT / "summary.md").write_text(text)
    print(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--case", nargs=2, type=int)
    args = ap.parse_args()
    if args.case:
        print(json.dumps(case(*args.case)))
    elif args.summary:
        print(summarize())
    elif args.all:
        run_all()


if __name__ == "__main__":
    main()
