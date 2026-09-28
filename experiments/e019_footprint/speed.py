"""E019: E017 worker phase A (calm, 64 tokens) with and without a low MPS low-watermark ratio.

Alternates the two settings, one fresh worker per run; samples the worker's footprint from
outside every 20 ms (as E016/E017 did every 200 ms). No pressure process, so no swap writes.

Usage: .venv/bin/python -m experiments.e019_footprint.speed --cache-dir DIR [--repeats 3]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from experiments._harness.env import capture, save_json
from experiments.e011_os_swap.common import rusage

OUT = Path(__file__).resolve().parents[2] / "docs" / "research" / "data" / "e019"
CASES = [
    ("Qwen/Qwen2.5-3B-Instruct", "int4"),
    ("Qwen/Qwen2.5-1.5B-Instruct", "int4"),
    ("Qwen/Qwen2.5-1.5B-Instruct", "bf16"),
]
SETTINGS = {"default": None, "low_watermark_0.3": "0.3"}


def run(model: str, arm: str, cache_dir: str, ratio: str | None) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in ("MallocLargeCache",)}
    env.pop("PYTORCH_MPS_LOW_WATERMARK_RATIO", None)
    if ratio is not None:
        env["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = ratio
    cmd = [sys.executable, "-m", "experiments.e017_controller.workers", "--model", model]
    cmd += ["--arm", arm, "--cache-dir", cache_dir]
    p = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env
    )
    peak, done = [0], threading.Event()

    def watch() -> None:
        while not done.is_set():
            try:
                peak[0] = max(peak[0], rusage(p.pid)["phys_footprint"])
            except OSError:
                pass
            time.sleep(0.02)

    threading.Thread(target=watch, daemon=True).start()
    line = p.stdout.readline()
    end = rusage(p.pid)["phys_footprint"]
    done.set()
    p.stdin.close()
    p.wait(timeout=60)
    a = json.loads(line[2:])
    return {
        "ok": a.get("ok"),
        "tokens_per_s": a.get("tokens_per_s"),
        "tokens": a.get("tokens"),
        "peak_mib": peak[0] / 2**20,
        "end_mib": end / 2**20,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    save_json(capture(__file__, extra={"argv": sys.argv}), OUT / "env_speed.json")
    rows = []
    for rep in range(args.repeats):
        for model, arm in CASES:
            names = list(SETTINGS) if rep % 2 == 0 else list(SETTINGS)[::-1]
            for name in names:
                r = run(model, arm, args.cache_dir, SETTINGS[name])
                r.update(repeat=rep, model=model, arm=arm, setting=name)
                rows.append(r)
                save_json({"runs": rows}, OUT / "speed.json")
                print(
                    f"{rep} {model.split('/')[-1]:22} {arm:5} {name:18} "
                    f"{r['tokens_per_s'] or 0:5.2f} tok/s peak {r['peak_mib']:6.0f} "
                    f"end {r['end_mib']:6.0f} MiB",
                    flush=True,
                )
                time.sleep(5)


if __name__ == "__main__":
    main()
