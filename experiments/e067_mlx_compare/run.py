"""E067 (docs/research/0341): MLX 16-bit LoRA on an 8 GB M1 — the original (mlx-lm's own
whole-graph training step), memopro within a ceiling, and mlx-tune, on E034's ten texts
(129 tokens per step, batch 1, LoRA r 8 / alpha 16 on q/k/v/o, AdamW 2e-4).

    caffeinate -i .venv/bin/python -m experiments.e067_mlx_compare.run --all

Results: docs/research/data/e067/ (cases/*.json, summary.md, env.json).
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
OUT = ROOT / "docs" / "research" / "data" / "e067"
TEXTS = ROOT / "docs" / "research" / "data" / "e034" / "texts.json"
MLX_TUNE = ROOT / ".cache" / "venv-mlx" / "bin" / "python"
MLX_CASE = ROOT / "experiments" / "e034_vs_mlx_tune" / "mlx_case.py"
MIB = 1 << 20
MODELS = {"15": ROOT / ".cache/models/Qwen2.5-1.5B-Instruct", "3": ROOT / ".cache/models/Qwen2.5-3B-Instruct"}
CEILINGS = {"15": 1536 * MIB, "3": 2560 * MIB}
# (case, tool, model) in run order: memopro, the original, mlx-tune; the small model first
CASES = [("M15", "memopro", "15"), ("O15", "original", "15"), ("T15", "mlx-tune", "15"),
         ("M3", "memopro", "3"), ("O3", "original", "3"), ("T3", "mlx-tune", "3")]
SEQ, TIMEOUT = 129, 45 * 60


def pieces(tok) -> list[list[int]]:
    """The pieces memopro.finetune makes of the texts (batch 1): the original trains on them."""
    texts = json.loads(TEXTS.read_text())
    eos = tok.eos_token
    ids = tok.encode(eos.join(texts) + eos)
    return [p for p in (ids[i : i + SEQ] for i in range(0, len(ids) - 1, SEQ)) if len(p) > 1]


def digests(model) -> dict[str, str]:
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten

    return {n: hashlib.sha256(np.array(v.view(mx.uint8)).tobytes()).hexdigest()[:16]
            for n, v in sorted(tree_flatten(model.trainable_parameters()))}


def worker(name: str) -> dict:
    import mlx.core as mx
    import mlx_lm

    import memopro

    mx.eval(mx.zeros(1))  # Metal started: the baseline is after imports, as in E028c
    base, _ = footprint()
    tool, key = next((t, k) for c, t, k in CASES if c == name)
    path = str(MODELS[key])
    if tool == "memopro":
        model, tok = mlx_lm.load(path, lazy=True)
        texts = json.loads(TEXTS.read_text())
        r = memopro.finetune(model, texts, tokenizer=tok, budget=CEILINGS[key], seq_len=SEQ, batch=1,
                             lr=2e-4, rank=8, alpha=16)
        losses, steps, tokens = r.losses, r.step_seconds, [len(p) for p in pieces(tok)]
        detail = memopro.report().entries[-1].detail
    else:
        sys.path.insert(0, str(ROOT / "tests"))
        from test_mlx import _whole_graph_lora

        _, tok = mlx_lm.load(path, lazy=True)
        ps = pieces(tok)
        steps = []
        losses, model = _whole_graph_lora(path, [[p] for p in ps], lr=2e-4, times=steps)
        tokens, detail = [len(p) for p in ps], "mlx-lm whole-graph step (tests/test_mlx.py)"
    _, peak = footprint()
    return {"losses": losses, "step_s": steps, "tokens": tokens, "adapters": digests(model),
            "mlx_peak": mx.get_peak_memory(), "base_footprint": base, "peak_footprint": peak,
            "detail": detail}


def run_case(name: str, tool: str, key: str, env: dict) -> dict:
    swap0 = swap_used()
    if tool == "mlx-tune":
        out = OUT / "mlx-tune" / name
        out.mkdir(parents=True, exist_ok=True)
        cmd = [str(MLX_TUNE), str(MLX_CASE), str(MODELS[key]), str(TEXTS), str(out), "10"]
    else:
        cmd = [sys.executable, "-m", "experiments.e067_mlx_compare.run", "--case", name]
    t = time.perf_counter()
    try:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env, timeout=TIMEOUT,
                           check=False)
    except subprocess.TimeoutExpired:
        return {"case": name, "error": f"timeout after {TIMEOUT} s", "swap_before": swap0,
                "swap_after": swap_used()}
    rec = {"case": name, "tool": tool, "model": key, "wall_s": time.perf_counter() - t,
           "swap_before": swap0, "swap_after": swap_used(), "stderr_tail": p.stderr[-3000:]}
    if p.returncode != 0:
        rec["error"] = f"exit {p.returncode}"
        return rec
    rec.update(json.loads(p.stdout.strip().splitlines()[-1]))
    if tool == "mlx-tune":  # E034's case format: per-iteration rows from mlx-lm's report
        rec["losses"] = [x["loss"] for x in rec["iters"]]
    return rec


def steady(r: dict) -> float:
    """Tokens per second over steps 2..n (the first step loads the kept layers / compiles)."""
    if "iters" in r:
        rows = [x["tok_s"] for x in r["iters"] if x["iter"] >= 2]
        return sum(rows) / len(rows) if rows else float("nan")
    return sum(r["tokens"][1:]) / sum(r["step_s"][1:]) if len(r["step_s"]) > 1 else float("nan")


def ok(r: dict) -> bool:
    return "error" not in r and len(r.get("losses", [])) >= 10


def summarize() -> str:
    recs = {p.stem: json.loads(p.read_text()) for p in sorted((OUT / "cases").glob("*.json"))}
    rows = ["# E067 summary (0341)", "",
            "| case | tool | model | finished | steady tok/s | footprint growth MiB | MLX peak MiB | swap MiB |",
            "|---|---|---|---|---|---|---|---|"]
    for name, r in recs.items():
        swap = (r["swap_after"] - r["swap_before"]) / MIB
        if not ok(r):
            err = (r.get("error", "") + " " + r.get("stderr_tail", "")[-120:]).replace("\n", " ")
            rows.append(f"| {name} | {r.get('tool', '')} | {r.get('model', '')} | no: {err} | | | | {swap:+.0f} |")
            continue
        growth = (r["peak_footprint"] - r["base_footprint"]) / MIB
        peak = f"{r['mlx_peak'] / MIB:.0f}" if "mlx_peak" in r else ""
        rows.append(f"| {name} | {r['tool']} | {r['model']} | {len(r['losses'])} steps | {steady(r):.1f} | "
                    f"{growth:.0f} | {peak} | {swap:+.0f} |")
    g = recs.get
    checks = []

    def same(m, o):
        return m["losses"] == o["losses"] and m["adapters"] == o["adapters"]

    m15, o15, t15, m3, o3, t3 = (g(n, {}) for n in ("M15", "O15", "T15", "M3", "O3", "T3"))
    checks.append(("K1 1.5B identical to the original", ok(m15) and ok(o15) and same(m15, o15),
                   "losses and adapter digests" if ok(m15) and ok(o15) else "a case did not finish"))
    if ok(o3):
        checks.append(("K2 3B identical to the original", ok(m3) and same(m3, o3), "original finished"))
    else:
        checks.append(("K2 3B identical to the original", "n/a", "the original did not finish on this device"))
    for n, m in (("M15", m15), ("M3", m3)):
        if ok(m):
            c = CEILINGS[m["model"]]
            growth = m["peak_footprint"] - m["base_footprint"]
            checks.append((f"K3 {n} within the ceiling", m["mlx_peak"] <= c and growth <= c + 256 * MIB,
                           f"MLX peak {m['mlx_peak'] / MIB:.0f}, footprint +{growth / MIB:.0f}, ceiling {c / MIB:.0f} MiB"))
        if "swap_after" in m:
            sw = m["swap_after"] - m["swap_before"]
            checks.append((f"K4 {n} swap <= 512 MiB", sw <= 512 * MIB, f"{sw / MIB:+.0f} MiB"))
    checks.append(("K5 M3 finishes", ok(m3), f"{len(m3.get('losses', []))} steps"))
    if ok(m15) and ok(o15):
        checks.append(("P1 M15 >= 0.6x O15", steady(m15) >= 0.6 * steady(o15),
                       f"{steady(m15):.1f} vs {steady(o15):.1f}"))
    if ok(m15) and ok(t15):
        checks.append(("P2 T15 >= 1.5x M15", steady(t15) >= 1.5 * steady(m15),
                       f"{steady(t15):.1f} vs {steady(m15):.1f}"))
    if ok(m3):
        checks.append(("P3 M3 >= 24 tok/s", steady(m3) >= 24, f"{steady(m3):.1f}"))
    for n, r in (("O3", o3), ("T3", t3)):
        if r:
            sw = r["swap_after"] - r["swap_before"]
            checks.append((f"P4 {n} fails or swaps > 1 GiB", not ok(r) or sw > 1 << 30,
                           f"{'finished' if ok(r) else 'did not finish'}, swap {sw / MIB:+.0f} MiB"))
    for size, ms, others in (("1.5B", m15, (o15, t15)), ("3B", m3, (o3, t3))):
        if "swap_after" in ms and all("swap_after" in o for o in others):
            mine = ms["swap_after"] - ms["swap_before"]
            checks.append((f"P5 {size} memopro swaps least", all(mine < o["swap_after"] - o["swap_before"] for o in others),
                           f"memopro {mine / MIB:+.0f} MiB"))
    rows += ["", "| check | held | detail |", "|---|---|---|"]
    rows += [f"| {n} | {v if isinstance(v, str) else ('yes' if v else 'no')} | {d} |" for n, v, d in checks]
    return "\n".join(rows) + "\n"


def run_all() -> None:
    (OUT / "cases").mkdir(parents=True, exist_ok=True)
    def sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    env = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "python": sys.version.split()[0],
           "platform": platform.platform(),
           "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                                    check=False).stdout.strip(),
           "script_sha256": sha(__file__), "mlx_case_sha256": sha(MLX_CASE), "texts_sha256": sha(TEXTS),
           "test_mlx_sha256": sha(ROOT / "tests" / "test_mlx.py"),
           "venv": subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True,
                                  check=False).stdout,
           "mlx_tune_venv": subprocess.run([str(MLX_TUNE), "-m", "pip", "list"], capture_output=True,
                                           text=True, check=False).stdout,
           "swap_at_start": swap_used()}
    (OUT / "env.json").write_text(json.dumps(env, indent=1))
    child = {**os.environ, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
             "PATH": f"{MLX_TUNE.parent}:{os.environ['PATH']}"}  # mlx-tune's CLI fallback (0151)
    for name, tool, key in CASES:
        print(f"[{time.strftime('%H:%M:%S')}] {name}", flush=True)
        rec = run_case(name, tool, key, child)
        (OUT / "cases" / f"{name}.json").write_text(json.dumps(rec, indent=1))
        print(f"   -> {'ok' if ok(rec) else 'not finished'}", flush=True)
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
        print(json.dumps(worker(a.case)))
    elif a.summary:
        print(summarize())
    elif a.all:
        run_all()


if __name__ == "__main__":
    main()
