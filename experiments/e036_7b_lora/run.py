"""E036 (docs/research/0158): success criterion S1 — 16-bit LoRA on Qwen2.5-7B-Instruct (bf16,
14.2 GiB) on an 8 GB M1 with the one-line API `memopro.finetune`, at 2 GiB and 1.5 GiB budgets.

    .venv/bin/python -m experiments.e036_7b_lora.run --all

Results: docs/research/data/e036/ (cases/*.json, summary.md, env.json).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from experiments.e028_mps_lora.run import footprint, swap_used

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "research" / "data" / "e036"
MIB = 1 << 20
MODEL = "Qwen/Qwen2.5-7B-Instruct"
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
CASES = [("L7a", 2048 * MIB), ("L7b", 1536 * MIB)]
STEPS = 5


def case(budget: int) -> dict:
    import memopro
    import memopro.rt.torch as rtt

    texts = json.loads((ROOT / "docs/research/data/e034/texts.json").read_text())[:STEPS]
    base, _ = footprint()
    swap0 = swap_used()
    model = rtt.stream_model(MODEL, budget=budget, device="mps", revision=REVISION)
    r = memopro.finetune(model, texts, seq_len=129, epochs=1)
    _, peak = footprint()
    stats = model.memopro_runtime.stats()
    return {
        "budget": budget,
        "limit": model.memopro_runtime.limit,
        "losses": r.losses,
        "loss_bits": [float.hex(x) for x in r.losses],
        "step_s": r.step_seconds,
        "base_footprint": base,
        "peak_footprint": peak,
        "swap_before": swap0,
        "swap_after": swap_used(),
        "rt": stats,
    }


def summarize() -> str:
    recs = {p.stem: json.loads(p.read_text()) for p in sorted((OUT / "cases").glob("*.json"))}
    a, b = recs.get("L7a", {}), recs.get("L7b", {})
    done = [n for n, r in recs.items() if len(r.get("losses", [])) == STEPS]
    checks = [("K1 both budgets complete 5 steps", len(done) == 2, ", ".join(done) or "none")]
    same = "loss_bits" in a and "loss_bits" in b and a["loss_bits"] == b["loss_bits"]
    checks.append(("K2 losses bit-identical across budgets", same,
                   str([round(x, 4) for x in a.get("losses", [])])))
    parts, oks = [], []
    for name, r in recs.items():
        if "rt" not in r:
            oks.append(False)
            parts.append(f"{name}: {r.get('error', 'missing')[-120:]}")
            continue
        growth = r["peak_footprint"] - r["base_footprint"]
        swap = r["swap_after"] - r["swap_before"]
        oks.append(growth <= r["budget"] + 512 * MIB and swap <= 64 * MIB
                   and r["rt"]["peak_used"] <= r["limit"])
        parts.append(f"{name}: footprint +{growth / MIB:.0f} MiB (budget {r['budget'] // MIB}), "
                     f"swap {swap / MIB:+.0f} MiB, peak {r['rt']['peak_used'] / MIB:.0f} MiB")
    checks.append(("K3 memory", bool(oks) and all(oks), "; ".join(parts)))
    gate = all(ok for _, ok, _ in checks)
    rows = ["# E036 summary (0158)", "", f"S1 (7B 16-bit LoRA on 8 GB): **{'pass' if gate else 'fail'}**",
            "", "| check | result | detail |", "|---|---|---|"]
    rows += [f"| {n} | {'pass' if ok else 'fail'} | {d} |" for n, ok, d in checks]
    rows += ["", "| case | step s | tokens/s (steps 2-5) | re-read per run GiB |", "|---|---|---|---|"]
    for name, r in recs.items():
        if "step_s" in r:
            steady = sum(r["step_s"][1:]) / len(r["step_s"][1:])
            rows.append(f"| {name} | {[round(x, 1) for x in r['step_s']]} | {129 / steady:.1f} | "
                        f"{r['rt']['reread_bytes'] / 2**30:.1f} |")
    return "\n".join(rows) + "\n"


def run_all() -> None:
    (OUT / "cases").mkdir(parents=True, exist_ok=True)
    power = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, check=False).stdout
    low = subprocess.run(["pmset", "-g"], capture_output=True, text=True, check=False).stdout
    env = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "python": sys.version.split()[0],
           "platform": platform.platform(), "model": MODEL, "revision": REVISION,
           "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                    text=True, check=False).stdout.strip(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "power": power, "lowpowermode": [l for l in low.splitlines() if "lowpowermode" in l],
           "swap_at_start": swap_used()}
    (OUT / "env.json").write_text(json.dumps(env, indent=1))
    child = {**os.environ, "MallocLargeCache": "0", "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.1",
             "HF_HOME": str(ROOT / ".cache" / "hf"), "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
             "TOKENIZERS_PARALLELISM": "false"}
    for name, budget in CASES:
        first = None
        for attempt in (1, 2):
            print(f"[{time.strftime('%H:%M:%S')}] {name} (attempt {attempt})", flush=True)
            p = subprocess.run([sys.executable, "-m", "experiments.e036_7b_lora.run", "--case", str(budget)],
                               cwd=ROOT, capture_output=True, text=True, env=child, check=False)
            if p.returncode != 0:
                rec = {"case": name, "budget": budget, "error": p.stderr[-3000:]}
                break
            rec = json.loads(p.stdout.strip().splitlines()[-1]) | {"case": name, "attempt": attempt}
            if rec["swap_after"] - rec["swap_before"] <= 64 * MIB:
                break
            if attempt == 1:
                print("   swap grew: repeating once (contamination rule)", flush=True)
                first = rec
        if first is not None:
            rec["first_attempt"] = first
        (OUT / "cases" / f"{name}.json").write_text(json.dumps(rec, indent=1))
        print(f"   -> {'error' if 'error' in rec else 'ok'}", flush=True)
    text = summarize()
    (OUT / "summary.md").write_text(text)
    print(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--case")
    a = ap.parse_args()
    if a.case:
        print(json.dumps(case(int(a.case))))
    elif a.summary:
        print(summarize())
    elif a.all:
        run_all()


if __name__ == "__main__":
    main()
