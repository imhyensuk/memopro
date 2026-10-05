"""E035 (docs/research/0155): gate G4-E2 — the speed changes of 0154 (wrapped buffers kept until
room runs short, one fence per batch; checkpointing only when activations would not fit) against
the previous code (main at 9aa1fe6), interleaved old/new twice on the same machine.

    .venv/bin/python -m experiments.e035_speed.run --all

The old code runs from a git worktree of 9aa1fe6 at .cache/e035-old (same Rust extension: the
Rust sources did not change). Results: docs/research/data/e035/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

from experiments.e028_mps_lora.run import footprint, swap_used

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "research" / "data" / "e035"
OLD = ROOT / ".cache" / "e035-old"
OLD_COMMIT = "9aa1fe6"
MIB = 1 << 20
MODELS = {"q15": "Qwen/Qwen2.5-1.5B-Instruct", "q3": "Qwen/Qwen2.5-3B-Instruct"}
BUDGETS = {"q15": 768 * MIB, "q3": 1024 * MIB}
KINDS = [("T15", "train", "q15"), ("T3", "train", "q3"), ("G3", "gen", "q3")]
STEPS, NEW_TOKENS = 10, 32
PROMPT = "Explain how a hash table handles collisions."


def train(key: str) -> dict:
    import memopro

    texts = json.loads((ROOT / "docs/research/data/e034/texts.json").read_text())
    base, _ = footprint()
    r = memopro.finetune(MODELS[key], texts[:STEPS], budget=BUDGETS[key], seq_len=129, epochs=1)
    _, peak = footprint()
    return {
        "losses": r.losses,
        "step_s": r.step_seconds,
        "checkpointing": getattr(r, "checkpointing", True),
        "base_footprint": base,
        "peak_footprint": peak,
        "rt": r.model.memopro_runtime.stats(),
        "limit": r.model.memopro_runtime.limit,
    }


def gen(key: str) -> dict:
    import torch
    import transformers

    import memopro.rt.torch as rtt

    tok = transformers.AutoTokenizer.from_pretrained(MODELS[key])
    msgs = [{"role": "user", "content": PROMPT}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    ids = (ids["input_ids"] if hasattr(ids, "keys") else ids).to("mps")
    base, _ = footprint()
    m = rtt.stream_model(MODELS[key], budget=BUDGETS[key], device="mps")
    t = time.perf_counter()
    with torch.no_grad():
        out = m.generate(input_ids=ids, max_new_tokens=NEW_TOKENS, min_new_tokens=NEW_TOKENS,
                         do_sample=False, pad_token_id=tok.eos_token_id)
    torch.mps.synchronize()
    seconds = time.perf_counter() - t
    m.memopro_weights.finish()
    _, peak = footprint()
    return {"tokens": out[0, ids.shape[1] :].tolist(), "seconds": seconds, "base_footprint": base,
            "peak_footprint": peak, "rt": m.memopro_runtime.stats(), "limit": m.memopro_runtime.limit}


def worktree() -> None:
    if not (OLD / "python" / "memopro").exists():
        subprocess.run(["git", "worktree", "add", "--detach", str(OLD), OLD_COMMIT], cwd=ROOT,
                       check=True, capture_output=True)
    so = next((ROOT / "python" / "memopro").glob("_core*.so"))
    shutil.copy2(so, OLD / "python" / "memopro" / so.name)


def run_case(name: str, kind: str, key: str, code: str, env: dict) -> dict:
    pythonpath = str((OLD if code == "old" else ROOT) / "python")
    swap0 = swap_used()
    p = subprocess.run(
        [sys.executable, "-m", "experiments.e035_speed.run", "--case", kind, key],
        cwd=ROOT, capture_output=True, text=True, env={**env, "PYTHONPATH": pythonpath}, check=False,
    )
    rec = {"case": name, "kind": kind, "model": key, "code": code, "swap_before": swap0,
           "swap_after": swap_used()}
    if p.returncode != 0:
        return rec | {"error": p.stderr[-3000:]}
    return rec | json.loads(p.stdout.strip().splitlines()[-1])


def tok_s(r: dict) -> float:
    if r["kind"] == "train":
        return 129 * (len(r["step_s"]) - 1) / sum(r["step_s"][1:])
    return NEW_TOKENS / r["seconds"]


def summarize() -> str:
    recs = [json.loads(p.read_text()) for p in sorted((OUT / "cases").glob("*.json"))]
    ok = [r for r in recs if "error" not in r]

    def mean(name: str, code: str) -> float:
        v = [tok_s(r) for r in ok if r["case"].startswith(name) and r["code"] == code]
        return sum(v) / len(v) if len(v) == 2 else float("nan")

    head = "| run | code | tok/s | checkpointing | footprint growth MiB | swap MiB |"
    rows = ["# E035 summary (0155)", "", head, "|---|---|---|---|---|---|"]
    for r in recs:
        if "error" in r:
            rows.append(f"| {r['case']} | {r['code']} | error: {r['error'][-80:]} | | | |")
            continue
        rows.append(f"| {r['case']} | {r['code']} | {tok_s(r):.2f} | {r.get('checkpointing', '')} | "
                    f"{(r['peak_footprint'] - r['base_footprint']) / MIB:.0f} | "
                    f"{(r['swap_after'] - r['swap_before']) / MIB:+.0f} |")
    t15, t3, g3 = (mean(k, "new") / mean(k, "old") for k in ("T15", "T3", "G3"))
    new15 = [r for r in ok if r["case"].startswith("T15") and r["code"] == "new"]
    old15 = [r for r in ok if r["case"].startswith("T15") and r["code"] == "old"]
    mem = all(r["peak_footprint"] - r["base_footprint"] <= r["limit"] + 1024 * MIB
              and r["rt"]["peak_used"] <= r["limit"] for r in ok if r["code"] == "new")
    drift = max((abs(a - b) for n, o in zip(new15, old15, strict=False)
                 for a, b in zip(n["losses"], o["losses"], strict=True)), default=float("nan"))
    gens = [r["tokens"] for r in ok if r["kind"] == "gen"]
    same = len(gens) == 4 and all(g == gens[0] for g in gens)
    checks = [
        ("S1 1.5B training speed new/old >= 1.3", t15 >= 1.3, f"{t15:.2f}x"),
        ("S3 memory (new): footprint <= limit + 1 GiB, runtime peak <= limit", mem, ""),
        ("S4 losses new vs old within 1e-3", drift <= 1e-3, f"max |diff| {drift:.2e}"),
        ("S5 generation new/old >= 1.0 and same tokens", g3 >= 1.0 and same, f"{g3:.2f}x, same={same}"),
    ]
    gate = all(c[1] for c in checks)
    rows += ["", f"Gate G4-E2: **{'pass' if gate else 'fail'}**", "", "| check | result | detail |",
             "|---|---|---|"]
    rows += [f"| {n} | {'pass' if c else 'fail'} | {d} |" for n, c, d in checks]
    rows += ["", f"Report: 3B training new/old {t3:.2f}x (S2, no criterion)"]
    return "\n".join(rows) + "\n"


def run_all() -> None:
    (OUT / "cases").mkdir(parents=True, exist_ok=True)
    worktree()
    power = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, check=False).stdout
    low = subprocess.run(["pmset", "-g"], capture_output=True, text=True, check=False).stdout
    env = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "python": sys.version.split()[0],
           "platform": platform.platform(),
           "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                    text=True, check=False).stdout.strip(),
           "old_commit": OLD_COMMIT, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "power": power, "lowpowermode": [l for l in low.splitlines() if "lowpowermode" in l],
           "swap_at_start": swap_used()}
    (OUT / "env.json").write_text(json.dumps(env, indent=1))
    child = {**os.environ, "MallocLargeCache": "0", "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.1",
             "HF_HOME": str(ROOT / ".cache" / "hf"), "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
             "TOKENIZERS_PARALLELISM": "false"}
    for rep, order in ((1, ("old", "new")), (2, ("new", "old"))):
        for name, kind, key in KINDS:
            for code in order:
                print(f"[{time.strftime('%H:%M:%S')}] {name}-{rep} {code}", flush=True)
                rec = run_case(f"{name}-{rep}", kind, key, code, child)
                (OUT / "cases" / f"{name}-{rep}-{code}.json").write_text(json.dumps(rec, indent=1))
                print(f"   -> {'error' if 'error' in rec else round(tok_s(rec), 2)}", flush=True)
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
        import memopro

        rec = train(a.case[1]) if a.case[0] == "train" else gen(a.case[1])
        print(json.dumps(rec | {"memopro_file": memopro.__file__}))
    elif a.summary:
        print(summarize())
    elif a.all:
        run_all()


if __name__ == "__main__":
    main()
