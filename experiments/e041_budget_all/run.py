"""E041 (docs/research/0183): the budget covers everything — `memopro.generate` with an int4 draft
(draft, cache and intermediates held back, 0182) for Qwen2.5-3B and 7B, and `memopro.finetune`
at 2,048 tokens for 3B and 7B. Every case in a fresh process; E037c's measurement.

    .venv/bin/python -m experiments.e041_budget_all.run --all

Results: docs/research/data/e041/ (cases/*.json, summary.md, env.json).
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
OUT = ROOT / "docs" / "research" / "data" / "e041"
TEXTS = OUT / "texts.json"
MIB = 1 << 20
MODELS = {
    "q3": ("Qwen/Qwen2.5-3B-Instruct", None),
    "q7": ("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28"),
}
DRAFT = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPTS = [
    "Explain how a hash table handles collisions.",
    "Write a Python function that checks whether a string is a palindrome, with a short docstring.",
]
NEW = 32
# plain greedy tokens of the same prompts from earlier runs (same chat template, min_new_tokens)
REFERENCE = {"q3": ROOT / "docs/research/data/e033c/cases/P.json",
             "q7": ROOT / "docs/research/data/e038b/cases/P.json"}
STEPS, SEQ = 5, 2048
CASES = [  # (name, kind, model, budget)
    ("G3", "gen", "q3", 2048 * MIB), ("G7", "gen", "q7", 3072 * MIB),
    ("RG7", "gen", "q7", 2048 * MIB),
    ("T3a", "train", "q3", 2560 * MIB), ("T3b", "train", "q3", 2304 * MIB),
    ("T7a", "train", "q7", 3840 * MIB), ("T7b", "train", "q7", 3584 * MIB),
]


def texts() -> list[str]:
    if not TEXTS.exists():
        import transformers

        from experiments.e026_rt_phase2.run import training_tokens

        tok = transformers.AutoTokenizer.from_pretrained(MODELS["q3"][0])
        ids, _ = training_tokens(tok, (SEQ - 1) * STEPS)
        TEXTS.write_text(json.dumps([tok.decode(ids[i * (SEQ - 1) : (i + 1) * (SEQ - 1)])
                                     for i in range(STEPS)]))
    return json.loads(TEXTS.read_text())


def reference(key: str) -> list[str]:
    import transformers

    tok = transformers.AutoTokenizer.from_pretrained(MODELS[key][0], revision=MODELS[key][1])
    rows = json.loads(REFERENCE[key].read_text())["prompts"][: len(PROMPTS)]
    out = []
    for r in rows:
        ids = r["tokens"][:NEW]
        if tok.eos_token_id in ids:
            ids = ids[: ids.index(tok.eos_token_id)]
        out.append(tok.decode(ids, skip_special_tokens=True))
    return out


def case(name: str, kind: str, key: str, budget: int) -> dict:
    import peft  # noqa: F401 - imported before the baseline; MPS is not started yet
    import torch  # noqa: F401
    import transformers

    import memopro
    import memopro.rt.torch as rtt

    model_name, revision = MODELS[key]
    data = texts() if kind == "train" else None
    base, _ = footprint()
    swap0 = swap_used()
    model = rtt.stream_model(model_name, budget=budget, device="mps", revision=revision)
    rec = {"budget": budget, "base_footprint": base, "swap_before": swap0}
    try:
        if kind == "gen":
            draft = rtt.draft_model(DRAFT, target=model)
            tok = transformers.AutoTokenizer.from_pretrained(model_name, revision=revision)
            got, secs = [], []
            for p in PROMPTS:
                t = time.perf_counter()
                got.append(memopro.generate(model, p, tokenizer=tok, draft=draft,
                                            max_new_tokens=NEW))
                secs.append(time.perf_counter() - t)
            rec |= {"texts": got, "seconds": secs, "reference": reference(key),
                    "draft_bytes": draft.memopro_draft_bytes}
        else:
            r = memopro.finetune(model, data, seq_len=SEQ, epochs=1)
            rec |= {"losses": r.losses, "loss_bits": [float.hex(x) for x in r.losses],
                    "step_s": r.step_seconds, "tokens": r.tokens, "held_back": r.held_back}
    except memopro.BudgetExceeded as e:
        return rec | {"refused": str(e), "swap_after": swap_used()}
    _, peak = footprint()
    stats = model.memopro_runtime.stats()
    return rec | {"peak_footprint": peak, "swap_after": swap_used(), "rt": stats,
                  "limit": model.memopro_runtime.limit,
                  "held": stats["budget"] - stats["reserve"] - model.memopro_runtime.limit}


def summarize() -> str:
    recs = {p.stem: json.loads(p.read_text()) for p in sorted((OUT / "cases").glob("*.json"))}
    checks = []
    for n in ("G3", "G7"):
        r = recs.get(n, {})
        # equal, or a prefix when generation met the end token (the references suppressed it)
        same = [a == b or (len(a) < len(b) and b.startswith(a))
                for a, b in zip(r.get("texts", []), r.get("reference", []), strict=False)]
        ok = len(same) == len(PROMPTS) and all(same)
        checks.append((f"K1 {n} text equals plain greedy", ok, f"{sum(same)}/{len(PROMPTS)}"))
    for a, b in (("T3a", "T3b"), ("T7a", "T7b")):
        ra, rb = recs.get(a, {}), recs.get(b, {})
        ok = (len(ra.get("losses", [])) == STEPS and ra.get("loss_bits") == rb.get("loss_bits")
              and all(math.isfinite(x) for x in ra["losses"]))
        checks.append((f"K2 {a}/{b} complete, losses finite and bit-identical", ok,
                       str([round(x, 4) for x in ra.get("losses", [])])))
    parts, oks = [], []
    for n in ("G3", "G7", "T3a", "T3b", "T7a", "T7b"):
        r = recs.get(n, {})
        if "rt" not in r:
            oks.append(False)
            parts.append(f"{n}: {(r.get('error') or r.get('refused') or 'missing')[-120:]}")
            continue
        growth, swap = r["peak_footprint"] - r["base_footprint"], r["swap_after"] - r["swap_before"]
        oks.append(growth <= r["budget"] + 256 * MIB and swap <= 64 * MIB
                   and r["rt"]["peak_used"] <= r["budget"])
        parts.append(f"{n}: +{growth / MIB:.0f} MiB (budget {r['budget'] // MIB}, held "
                     f"{r['held'] / MIB:.0f}), swap {swap / MIB:+.0f}")
    checks.append(("K3 footprint <= budget + 256 MiB, swap <= 64 MiB", bool(oks) and all(oks),
                   "; ".join(parts)))
    rr = recs.get("RG7", {})
    checks.append(("K4 7B generation with the draft at 2 GiB refused", "refused" in rr,
                   (rr.get("refused") or rr.get("error") or "ran")[-160:]))
    gate = all(ok for _, ok, _ in checks)
    rows = ["# E041 summary (0183)", "", f"Gate (the budget covers everything): **{'pass' if gate else 'fail'}**",
            "", "| check | result | detail |", "|---|---|---|"]
    rows += [f"| {n} | {'pass' if ok else 'fail'} | {d} |" for n, ok, d in checks]
    rows += ["", "| case | s/token or step s | tokens/s |", "|---|---|---|"]
    for n, r in recs.items():
        if "seconds" in r:
            rows.append(f"| {n} | {sum(r['seconds']) / (NEW * len(PROMPTS)):.2f} s/token | |")
        if "step_s" in r:
            steady = sum(r["step_s"][1:]) / len(r["step_s"][1:])
            rows.append(f"| {n} | {[round(x, 1) for x in r['step_s']]} | {SEQ / steady:.1f} |")
    return "\n".join(rows) + "\n"


def run_all() -> None:
    (OUT / "cases").mkdir(parents=True, exist_ok=True)
    env_off = {**os.environ, "HF_HOME": str(ROOT / ".cache" / "hf"), "HF_HUB_OFFLINE": "1",
               "HF_DATASETS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "MallocLargeCache": "0"}
    env_off.pop("PYTORCH_MPS_LOW_WATERMARK_RATIO", None)
    os.environ.update({k: env_off[k] for k in ("HF_HOME", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")})
    texts()
    env = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "python": sys.version.split()[0],
           "platform": platform.platform(),
           "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                    text=True, check=False).stdout.strip(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "power": subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True,
                                   check=False).stdout,
           "swap_at_start": swap_used()}
    (OUT / "env.json").write_text(json.dumps(env, indent=1))
    for name, kind, key, budget in CASES:
        first = None
        for attempt in (1, 2):
            print(f"[{time.strftime('%H:%M:%S')}] {name} (attempt {attempt})", flush=True)
            p = subprocess.run([sys.executable, "-m", "experiments.e041_budget_all.run", "--case",
                                kind, key, str(budget)], cwd=ROOT, capture_output=True, text=True,
                               env=env_off, check=False)
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
    ap.add_argument("--case", nargs=3)
    a = ap.parse_args()
    if a.case:
        print(json.dumps(case("x", a.case[0], a.case[1], int(a.case[2]))))
    elif a.summary:
        print(summarize())
    elif a.all:
        run_all()


if __name__ == "__main__":
    main()
