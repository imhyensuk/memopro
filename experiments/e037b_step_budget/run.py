"""E037b (docs/research/0166): after 0165 the budget covers the whole training step on Apple GPUs —
`memopro.finetune` at 512 tokens, Qwen2.5-3B (1 GiB, 768 MiB) and 7B (2.5 GiB, 2 GiB), process
growth within budget + 256 MiB; a 7B step that cannot fit 1.5 GiB is refused.

    .venv/bin/python -m experiments.e037b_step_budget.run --all

Results: docs/research/data/e037b/ (cases/*.json, summary.md, env.json); texts from E037.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from experiments.e028_mps_lora.run import footprint, swap_used

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "research" / "data" / "e037b"
TEXTS = ROOT / "docs" / "research" / "data" / "e037" / "texts.json"
MIB = 1 << 20
MODELS = {
    "q3": ("Qwen/Qwen2.5-3B-Instruct", None),
    "q7": ("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28"),
}
CASES = [("T3a", "q3", 1024 * MIB), ("T3b", "q3", 768 * MIB), ("T7a", "q7", 2560 * MIB),
         ("T7b", "q7", 2048 * MIB), ("R7", "q7", 1536 * MIB)]
STEPS, SEQ = 5, 512


def texts() -> Path:
    path = TEXTS
    if not path.exists():
        import transformers

        from experiments.e026_rt_phase2.run import training_tokens

        tok = transformers.AutoTokenizer.from_pretrained(MODELS["q3"][0])
        ids, _ = training_tokens(tok, (SEQ - 1) * STEPS)
        path.write_text(json.dumps([tok.decode(ids[i * (SEQ - 1) : (i + 1) * (SEQ - 1)])
                                    for i in range(STEPS)]))
    return path


def case(key: str, budget: int) -> dict:
    import peft  # noqa: F401 - imported before the baseline; MPS is not started yet
    import torch  # noqa: F401
    import transformers  # noqa: F401

    import memopro
    import memopro.rt.torch as rtt

    name, revision = MODELS[key]
    t = json.loads(texts().read_text())
    base, _ = footprint()
    swap0 = swap_used()
    model = rtt.stream_model(name, budget=budget, device="mps", revision=revision)
    try:
        r = memopro.finetune(model, t, seq_len=SEQ, epochs=1)
    except memopro.BudgetExceeded as e:
        return {"budget": budget, "refused": str(e), "swap_before": swap0, "swap_after": swap_used()}
    _, peak = footprint()
    return {"budget": budget, "limit": model.memopro_runtime.limit, "losses": r.losses,
            "loss_bits": [float.hex(x) for x in r.losses], "step_s": r.step_seconds,
            "tokens": r.tokens, "base_footprint": base, "peak_footprint": peak,
            "swap_before": swap0, "swap_after": swap_used(), "rt": model.memopro_runtime.stats(),
            "watermark": os.environ.get("PYTORCH_MPS_LOW_WATERMARK_RATIO"),
            "held_back": r.held_back}


def summarize() -> str:
    recs = {p.stem: json.loads(p.read_text()) for p in sorted((OUT / "cases").glob("*.json"))}
    trained = ("T3a", "T3b", "T7a", "T7b")
    checks = []
    done = [n for n in trained if len(recs.get(n, {}).get("losses", [])) == STEPS]
    checks.append(("K1 the four training cases complete 5 steps", len(done) == 4, ", ".join(done)))
    for a, b in (("T3a", "T3b"), ("T7a", "T7b")):
        ra, rb = recs.get(a, {}), recs.get(b, {})
        ok = ("loss_bits" in ra and "loss_bits" in rb and ra["loss_bits"] == rb["loss_bits"]
              and all(math.isfinite(x) for x in ra["losses"]))
        checks.append((f"K2 {a}/{b} losses finite and bit-identical", ok,
                       str([round(x, 4) for x in ra.get("losses", [])])))
    parts, oks = [], []
    for name in trained:
        r = recs.get(name, {})
        if "rt" not in r:
            oks.append(False)
            parts.append(f"{name}: {(r.get('error') or r.get('refused') or 'missing')[-120:]}")
            continue
        growth = r["peak_footprint"] - r["base_footprint"]
        swap = r["swap_after"] - r["swap_before"]
        oks.append(growth <= r["budget"] + 256 * MIB and swap <= 64 * MIB
                   and r["rt"]["peak_used"] <= r["limit"] + r["held_back"])
        parts.append(f"{name}: +{growth / MIB:.0f} MiB (budget {r['budget'] // MIB}, held "
                     f"{r['held_back'] / MIB:.0f}), swap {swap / MIB:+.0f}")
    checks.append(("K3 footprint <= budget + 256 MiB, swap <= 64 MiB", bool(oks) and all(oks),
                   "; ".join(parts)))
    r7 = recs.get("R7", {})
    checks.append(("K4 7B at 1.5 GiB refused with BudgetExceeded", "refused" in r7,
                   (r7.get("refused") or r7.get("error") or "ran")[-160:]))
    gate = all(ok for _, ok, _ in checks)
    rows = ["# E037b summary (0166)", "", f"Gate (budget covers the step): **{'pass' if gate else 'fail'}**",
            "", "| check | result | detail |", "|---|---|---|"]
    rows += [f"| {n} | {'pass' if ok else 'fail'} | {d} |" for n, ok, d in checks]
    rows += ["", "| case | step s | tokens/s (steps 2-5) | held back MiB |", "|---|---|---|---|"]
    for name in trained:
        r = recs.get(name, {})
        if "step_s" in r:
            steady = sum(r["step_s"][1:]) / len(r["step_s"][1:])
            rows.append(f"| {name} | {[round(x, 1) for x in r['step_s']]} | {SEQ / steady:.1f} | "
                        f"{r['held_back'] / MIB:.0f} |")
    return "\n".join(rows) + "\n"


def run_all() -> None:
    (OUT / "cases").mkdir(parents=True, exist_ok=True)
    env_off = {**os.environ, "HF_HOME": str(ROOT / ".cache" / "hf"), "HF_HUB_OFFLINE": "1",
               "HF_DATASETS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "MallocLargeCache": "0"}
    env_off.pop("PYTORCH_MPS_LOW_WATERMARK_RATIO", None)
    os.environ.update({k: env_off[k] for k in ("HF_HOME", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")})
    texts()
    power = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, check=False).stdout
    low = subprocess.run(["pmset", "-g"], capture_output=True, text=True, check=False).stdout
    env = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "python": sys.version.split()[0],
           "platform": platform.platform(),
           "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                    text=True, check=False).stdout.strip(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "power": power, "lowpowermode": [l for l in low.splitlines() if "lowpowermode" in l],
           "swap_at_start": swap_used()}
    (OUT / "env.json").write_text(json.dumps(env, indent=1))
    for name, key, budget in CASES:
        first = None
        for attempt in (1, 2):
            print(f"[{time.strftime('%H:%M:%S')}] {name} (attempt {attempt})", flush=True)
            p = subprocess.run([sys.executable, "-m", "experiments.e037b_step_budget.run", "--case", key,
                                str(budget)], cwd=ROOT, capture_output=True, text=True, env=env_off,
                               check=False)
            if p.returncode != 0:
                rec = {"case": name, "budget": budget, "error": p.stderr[-3000:]}
                break
            rec = json.loads(p.stdout.strip().splitlines()[-1]) | {"case": name, "attempt": attempt}
            if "refused" in rec or rec["swap_after"] - rec["swap_before"] <= 64 * MIB:
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
    ap.add_argument("--case", nargs=2)
    a = ap.parse_args()
    if a.case:
        print(json.dumps(case(a.case[0], int(a.case[1]))))
    elif a.summary:
        print(summarize())
    elif a.all:
        run_all()


if __name__ == "__main__":
    main()
