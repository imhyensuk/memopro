"""E050c (docs/research/0250): S1 (0248) against the code before it in the same machine state.
The extension binary is swapped between runs (old = Rust of 2a773d5, new = Rust of ef58ce6);
cases come from run.py. Orders alternate (ABBA...) so drift in machine state hits both.

    .venv/bin/python -m experiments.e050_read_path.ab --build   # both binaries
    caffeinate -i .venv/bin/python -m experiments.e050_read_path.ab --all

Results: docs/research/data/e050c/ (runs.json, summary.md, env.json).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

from experiments.e028_mps_lora.run import swap_used

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "research" / "data" / "e050c"
CORE = ROOT / "python" / "memopro" / "_core.abi3.so"
BIN = ROOT / ".cache" / "e050c"
COMMITS = {"old": "2a773d5", "new": "ef58ce6"}
# (case, rounds): each round runs old and new, alternating which goes first
PLAN = [("B-64", 4), ("G-P-64", 3)]


def build() -> None:
    BIN.mkdir(parents=True, exist_ok=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()
    env = {**os.environ, "VIRTUAL_ENV": str(ROOT / ".venv")}
    try:
        for name, commit in COMMITS.items():
            subprocess.run(["git", "checkout", "-q", commit, "--", "crates"], cwd=ROOT, check=True)
            subprocess.run([str(ROOT / ".venv/bin/maturin"), "develop", "--release", "-q"],
                           cwd=ROOT, env=env, check=True)
            shutil.copy2(CORE, BIN / f"{name}.so")
    finally:
        subprocess.run(["git", "checkout", "-q", head, "--", "crates"], cwd=ROOT, check=True)
        if (BIN / "new.so").exists():
            shutil.copy2(BIN / "new.so", CORE)


def run_case(case: str, version: str) -> dict:
    shutil.copy2(BIN / f"{version}.so", CORE)
    child = {**os.environ, "MallocLargeCache": "0", "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.1",
             "HF_HOME": str(ROOT / ".cache" / "hf"), "HF_HUB_OFFLINE": "1",
             "TOKENIZERS_PARALLELISM": "false"}
    swap0, load0 = swap_used(), os.getloadavg()[0]
    p = subprocess.run([sys.executable, "-m", "experiments.e050_read_path.run", "--case", case],
                       cwd=ROOT, capture_output=True, text=True, env=child, check=False)
    rec = {"case": case, "version": version, "swap_before": swap0, "swap_after": swap_used(),
           "load_before": load0, "time": time.strftime("%H:%M:%S")}
    if p.returncode != 0:
        return {**rec, "error": p.stderr[-3000:]}
    return {**rec, **json.loads(p.stdout.strip().splitlines()[-1])}


def speed(r: dict) -> float:
    """B: GiB/s over passes 2-3; G: seconds per token."""
    if "passes" in r:
        ps = r["passes"][1:]
        return sum(p["load_bytes"] + p["reread_bytes"] for p in ps) / sum(
            p["seconds"] for p in ps) / (1 << 30)
    return r["seconds"] / len(r["tokens"])


def summarize(runs: list[dict]) -> str:
    lines = ["# E050c summary (0250)", "", "| case | round | old | new |", "|---|---|---|---|"]
    out = {}
    for case, rounds in PLAN:
        old = [speed(r) for r in runs if r["case"] == case and r["version"] == "old"
               and "error" not in r]
        new = [speed(r) for r in runs if r["case"] == case and r["version"] == "new"
               and "error" not in r]
        for i in range(rounds):
            lines.append(f"| {case} | {i + 1} | {old[i]:.2f} | {new[i]:.2f} |"
                         if i < min(len(old), len(new)) else f"| {case} | {i + 1} | | |")
        if old and new:
            out[case] = (statistics.median(old), statistics.median(new))
    lines.append("")
    base = json.loads((ROOT / "docs/research/data/e050/cases/G-P-64.json").read_text())["tokens"]
    toks = [r["tokens"] for r in runs if "tokens" in r]
    same = bool(toks) and all(t == base for t in toks)
    errors = [f"{r['case']}/{r['version']}" for r in runs if "error" in r]
    lines.append(f"- tokens of every G-P-64 run = E050: {'yes' if same else 'NO'} ({len(toks)} runs)")
    lines.append(f"- errors: {', '.join(errors) or 'none'}")
    for r in runs:
        lines.append(f"- {r['time']} {r['case']}/{r['version']}: swap "
                     f"{(r['swap_after'] - r['swap_before']) >> 20:+d} MiB, load {r['load_before']:.1f}")
    if "B-64" in out:
        o, n = out["B-64"]
        lines.append(f"\nB-64 median GiB/s: old {o:.2f}, new {n:.2f} ({n / o - 1:+.0%})")
    if "G-P-64" in out:
        o, n = out["G-P-64"]
        gain = o / n - 1
        ok = same and not errors and gain >= 0.10
        lines.append(f"G-P-64 median s/token: old {o:.2f}, new {n:.2f} (gain {gain:+.0%})")
        lines.append(f"\nDecision (0250): {'PASS' if ok else 'FAIL'} "
                     "(identical tokens, no errors, gain >= 10%)")
    return "\n".join(lines) + "\n"


def run_all() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    env = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                 text=True, check=False).stdout.strip(),
        "binaries": {k: hashlib.sha256((BIN / f"{k}.so").read_bytes()).hexdigest()
                     for k in COMMITS},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "swap_at_start": swap_used(),
        "uptime": subprocess.run(["uptime"], capture_output=True, text=True, check=False).stdout.strip(),
    }
    (OUT / "env.json").write_text(json.dumps(env, indent=1))
    runs: list[dict] = []
    try:
        for case, rounds in PLAN:
            for i in range(rounds):
                for version in (("old", "new") if i % 2 == 0 else ("new", "old")):
                    print(f"[{time.strftime('%H:%M:%S')}] {case} {version}", flush=True)
                    r = run_case(case, version)
                    runs.append(r)
                    (OUT / "runs.json").write_text(json.dumps(runs, indent=1))
                    print(f"   -> {'error' if 'error' in r else f'{speed(r):.2f}'}", flush=True)
    finally:
        shutil.copy2(BIN / "new.so", CORE)
    text = summarize(runs)
    (OUT / "summary.md").write_text(text)
    print(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if args.build:
        build()
    if args.all:
        run_all()


if __name__ == "__main__":
    main()
