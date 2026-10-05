"""E034 (docs/research/0150): success criterion S3 — the same 16-bit LoRA task with mlx-tune
(Unsloth-compatible API on MLX) and with memopro on an 8 GB M1: Qwen2.5-1.5B and 3B, LoRA r 8 on
q/k/v/o, 10 steps of 129 tokens, batch 1, lr 2e-4.

    .venv/bin/python -m experiments.e034_vs_mlx_tune.run --all

mlx-tune runs in the isolated venv .cache/venv-mlx (experiments/e034_vs_mlx_tune/mlx_case.py).
Results: docs/research/data/e034/ (cases/*.json, texts.json, summary.md, env.json).
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
OUT = ROOT / "docs" / "research" / "data" / "e034"
MLX = ROOT / ".cache" / "venv-mlx" / "bin" / "python"
MIB = 1 << 20
MODELS = {"q15": "Qwen/Qwen2.5-1.5B-Instruct", "q3": "Qwen/Qwen2.5-3B-Instruct"}
BUDGETS = {"q15": 768 * MIB, "q3": 1024 * MIB}
CASES = [("X15", "mlx", "q15"), ("P15", "memopro", "q15"), ("X3", "mlx", "q3"), ("P3", "memopro", "q3")]
STEPS, TIMEOUT = 10, 45 * 60


def texts() -> Path:
    path = OUT / "texts.json"
    if not path.exists():
        import transformers

        from experiments.e026_rt_phase2.run import training_tokens

        tok = transformers.AutoTokenizer.from_pretrained(MODELS["q15"])
        ids, _ = training_tokens(tok, 128 * STEPS)
        path.write_text(json.dumps([tok.decode(ids[i * 128 : (i + 1) * 128]) for i in range(STEPS)]))
    return path


def memopro_case(key: str) -> dict:
    import memopro

    base, _ = footprint()
    t = json.loads(texts().read_text())
    # each text is 128 tokens; joined with the end-of-text token, pieces of 129 = one text each
    r = memopro.finetune(MODELS[key], t, budget=BUDGETS[key], seq_len=129, epochs=1,
                         targets=("q_proj", "k_proj", "v_proj", "o_proj"))
    _, peak = footprint()
    return {
        "iters": [
            {"iter": i + 1, "loss": loss, "tok_s": 129 / s}
            for i, (loss, s) in enumerate(zip(r.losses, r.step_seconds, strict=True))
        ],
        "train_s": r.seconds,
        "base_footprint": base,
        "peak_footprint": peak,
        "rt": r.model.memopro_runtime.stats(),
        "limit": r.model.memopro_runtime.limit,
    }


def run_case(name: str, tool: str, key: str, env: dict) -> dict:
    swap0 = swap_used()
    if tool == "mlx":
        out = OUT / "mlx" / name
        out.mkdir(parents=True, exist_ok=True)
        cmd = [str(MLX), str(Path(__file__).with_name("mlx_case.py")), MODELS[key], str(texts()),
               str(out), str(STEPS)]
    else:
        cmd = [sys.executable, "-m", "experiments.e034_vs_mlx_tune.run", "--case", key]
    t = time.perf_counter()
    try:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env, timeout=TIMEOUT,
                           check=False)
    except subprocess.TimeoutExpired:
        return {"case": name, "error": f"timeout after {TIMEOUT} s", "swap_before": swap0,
                "swap_after": swap_used()}
    wall = time.perf_counter() - t
    if p.returncode != 0:
        return {"case": name, "error": p.stderr[-3000:], "swap_before": swap0, "swap_after": swap_used()}
    rec = json.loads(p.stdout.strip().splitlines()[-1])
    rec.update(case=name, tool=tool, model=key, wall_s=wall, swap_before=swap0, swap_after=swap_used())
    return rec


def steady(r: dict) -> float:
    rows = [x["tok_s"] for x in r["iters"] if x["iter"] >= 2]
    return sum(rows) / len(rows) if rows else float("nan")


def summarize() -> str:
    recs = {p.stem: json.loads(p.read_text()) for p in sorted((OUT / "cases").glob("*.json"))}
    head = "| case | tool | model | finished | steady tok/s | peak footprint growth MiB | swap MiB | losses |"
    rows = ["# E034 summary (0150)", "", head, "|---|---|---|---|---|---|---|---|"]
    for name, r in recs.items():
        swap = (r["swap_after"] - r["swap_before"]) / MIB
        if "error" in r:
            rows.append(f"| {name} | | | no: {r['error'][-80:]} | | | {swap:+.0f} | |")
            continue
        growth = (r["peak_footprint"] - r["base_footprint"]) / MIB
        rows.append(f"| {name} | {r['tool']} | {r['model']} | {len(r['iters'])} steps | "
                    f"{steady(r):.1f} | {growth:.0f} | {swap:+.0f} | "
                    f"{[round(x['loss'], 3) for x in r['iters']]} |")
    def get(n):
        return recs.get(n, {})
    checks = []
    p3, x3, p15, x15 = get("P3"), get("X3"), get("P15"), get("X15")
    if "iters" in p3:
        g = p3["peak_footprint"] - p3["base_footprint"]
        x3g = (x3["peak_footprint"] - x3["base_footprint"]) if "iters" in x3 else None
        checks.append(("Q1 3B memory", g <= 2560 * MIB and (x3g is None or x3g >= 5632 * MIB),
                       f"memopro +{g / MIB:.0f} MiB; mlx-tune "
                       + (f"+{x3g / MIB:.0f} MiB" if x3g is not None else "did not finish")))
    if "iters" in p3:
        ok = "iters" not in x3 or steady(p3) >= steady(x3)
        checks.append(("Q4 3B speed", ok, f"memopro {steady(p3):.1f} tok/s; mlx-tune "
                       + (f"{steady(x3):.1f}" if "iters" in x3 else "did not finish")))
    if "iters" in p15 and "iters" in x15:
        checks.append(("Q3 1.5B speed (mlx-tune >= 2x)", steady(x15) >= 2 * steady(p15),
                       f"mlx-tune {steady(x15):.1f}, memopro {steady(p15):.1f} tok/s"))
    if "swap_after" in x3:
        sw = x3["swap_after"] - x3["swap_before"]
        checks.append(("Q2 3B mlx-tune swap > 1 GiB or unfinished", "iters" not in x3 or sw > 1 << 30,
                       f"{sw / MIB:+.0f} MiB"))
    rows += ["", "| prediction | held | detail |", "|---|---|---|"]
    rows += [f"| {n} | {'yes' if ok else 'no'} | {d} |" for n, ok, d in checks]
    return "\n".join(rows) + "\n"


def run_all() -> None:
    (OUT / "cases").mkdir(parents=True, exist_ok=True)
    texts()
    env = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "python": sys.version.split()[0],
           "platform": platform.platform(),
           "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                    text=True, check=False).stdout.strip(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "mlx_case_sha256": hashlib.sha256(Path(__file__).with_name("mlx_case.py").read_bytes()).hexdigest(),
           "mlx_versions": subprocess.run([str(MLX), "-m", "pip", "list"], capture_output=True, text=True,
                                          check=False).stdout,
           "swap_at_start": swap_used()}
    (OUT / "env.json").write_text(json.dumps(env, indent=1))
    child = {**os.environ, "MallocLargeCache": "0", "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.1",
             "HF_HOME": str(ROOT / ".cache" / "hf"), "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
             "TOKENIZERS_PARALLELISM": "false"}
    for name, tool, key in CASES:
        print(f"[{time.strftime('%H:%M:%S')}] {name}", flush=True)
        rec = run_case(name, tool, key, child)
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
        print(json.dumps(memopro_case(a.case)))
    elif a.summary:
        print(summarize())
    elif a.all:
        run_all()


if __name__ == "__main__":
    main()
