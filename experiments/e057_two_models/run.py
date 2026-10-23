"""E057 (docs/research/0298): two streamed models in one process on an Apple GPU, each within its
own budget. Model A generates, then B next to it, then A again; ``--count old`` puts back the
GPU count before 0298 (everything on the GPU but this model's wrapped weights) to compare.
Each count runs in its own process (0055):

    .venv/bin/python -m experiments.e057_two_models.run --count old A_DIR B_DIR
    .venv/bin/python -m experiments.e057_two_models.run --count new A_DIR B_DIR

A_DIR, B_DIR: model folders with an int4 file cache already written (E055); budgets
--budgets 0.8GB 1.0GB. Results: docs/research/data/e057/{count}_{budget A}_{budget B}.json,
env.json.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "research" / "data" / "e057"
PROMPT, MAX_NEW = "Name three colors.", 24


def held_back(m) -> int:
    import memopro.rt.torch as rtt

    return rtt._held_back(m.memopro_runtime)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", choices=("old", "new"), required=True)
    ap.add_argument("--budgets", nargs=2, default=["0.8GB", "1.0GB"])
    ap.add_argument("dirs", nargs=2)
    a = ap.parse_args()

    import torch

    import memopro
    import memopro.rt.torch as rtt
    from memopro import llm
    from memopro.rt import process_footprint

    if a.count == "old":  # the count before 0298

        def old(m):
            return torch.mps.driver_allocated_memory() - m.memopro_weights.metal.held_bytes()

        llm._gpu_kept = old
    models, steps = {}, []
    with memopro.using(disk_writes="allow"):
        for name, path, budget in (("A", a.dirs[0], a.budgets[0]), ("B", a.dirs[1], a.budgets[1])):
            models[name] = rtt.stream_model(path, budget=budget, device="mps", precision="int4")
    from transformers import AutoTokenizer

    toks = {n: AutoTokenizer.from_pretrained(p) for n, p in zip("AB", a.dirs)}
    for name in ("A", "B", "A"):
        m = models[name]
        t = time.perf_counter()
        row = {"model": name}
        try:
            row["text"] = memopro.generate(m, PROMPT, tokenizer=toks[name], max_new_tokens=MAX_NEW)
            row["ok"] = True
        except memopro.BudgetExceeded as e:
            row["ok"], row["error"] = False, str(e)
        row["seconds"] = time.perf_counter() - t
        row["held_back"] = {n: held_back(x) for n, x in models.items()}
        row["wrapped"] = {n: x.memopro_weights.metal.held_bytes() for n, x in models.items()}
        row["budget"] = {n: x.memopro_runtime.stats()["budget"] for n, x in models.items()}
        row["mps_driver"] = torch.mps.driver_allocated_memory()
        row["footprint"] = process_footprint()
        steps.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "text"}), row.get("text", ""))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.count}_{a.budgets[0]}_{a.budgets[1]}.json").write_text(
        json.dumps({"dirs": a.dirs, "budgets": a.budgets, "steps": steps}, indent=1)
    )
    import transformers

    env = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, check=False
        ).stdout.strip(),
        "dirty": subprocess.run(
            ["git", "status", "--porcelain", "python"],
            capture_output=True,
            text=True,
            cwd=ROOT,
            check=False,
        ).stdout.strip(),
    }
    (OUT / "env.json").write_text(json.dumps(env, indent=1))


if __name__ == "__main__":
    main()
