"""E020 runner: every (model, variant) of evaluate.py, then prefill.py, each in a fresh process.

Usage: caffeinate -is .venv/bin/python -m experiments.e020_int4_quality.run --ref-dir DIR
         --cache-dir DIR [--windows 32]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from experiments._harness.env import capture, save_json
from experiments.e020_int4_quality.evaluate import OUT

MODELS = ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct")
QUALITY = ("bf16", "int4deq-g64", "int4deq-g32", "int4deq-g128")
PREFILL = ("bf16", "int4-cache", "int4deq-g64")
ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--windows", type=int, default=32)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    save_json(capture(__file__, extra={"argv": sys.argv}), OUT / "env_run.json")
    base = [sys.executable, "-m"]
    for model in MODELS:
        for variant in QUALITY:
            cmd = base + ["experiments.e020_int4_quality.evaluate", "--model", model]
            cmd += ["--variant", variant, "--ref-dir", args.ref_dir, "--cache-dir", args.cache_dir]
            cmd += ["--windows", str(args.windows)]
            done = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
            last = [line for line in done.stdout.splitlines() if line.startswith("{")]
            print(model, variant, (last[-1] if last else done.stderr[-600:])[:300], flush=True)
    for model in MODELS:
        for variant in PREFILL:
            cmd = base + ["experiments.e020_int4_quality.prefill", "--model", model]
            cmd += ["--variant", variant, "--cache-dir", args.cache_dir]
            done = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
            print(model, variant, done.stdout.strip().replace("\n", " | ")[:600] or done.stderr[-600:], flush=True)


if __name__ == "__main__":
    main()
