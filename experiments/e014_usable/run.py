"""E014: is memopro usable on a small machine? End-to-end cases, pre-registered in 0062.

Every case is a fresh worker process (workers.py). From outside, the orchestrator samples the
worker's physical footprint every 0.2 s (peak), the system swap in use (growth), and a probe
process's page-touch latency (how the rest of the system feels, probe.py). Arms rotate within
each repeat. Parts:

  reclaim    F1/F4 check: GPT-2 footprint right after `source` (cpu, mps) x (cache on, off)
  f4cost     CPU training step time with MallocLargeCache on vs off (the price of F4)
  infer      Qwen2.5-1.5B/3B-Instruct: naive from_pretrained().to("mps") vs memopro.load
             (default, budget_basis="os", quality="low", budget="4GB!")
  train      GPT-2 on MPS, batch 8 x 512, AdamW: plain loop vs train_session
  alternate  two instances of Qwen2.5-1.5B used in turns: both resident, del + reload, β
             (cache on) and β with MallocLargeCache=0

Usage: caffeinate -is .venv/bin/python -m experiments.e014_usable.run [--parts ...] [--repeats 3]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from experiments._harness.env import capture, save_json
from experiments.e011_os_swap.common import rusage, swap_used

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "research" / "data" / "e014"
MODELS = ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct")
MIN_FREE_DISK = 5 << 30  # stop before macOS runs out of room for swap
TIMEOUT = 10 * 60  # a case that takes longer is not comfortable by any measure (0062)


def cases(part: str) -> list[dict]:
    """The (worker arguments, environment) of one repeat of a part, in rotation order."""
    off = {"MallocLargeCache": "0"}
    if part == "reclaim":
        return [
            {"args": ["reclaim", "--device", d], "env": e, "label": f"{d}/{'off' if e else 'on'}"}
            for d in ("cpu", "mps")
            for e in ({}, off)
        ]
    if part == "f4cost":
        base = ["train", "--arm", "naive", "--device", "cpu", "--batch", "4", "--seq", "256"]
        return [{"args": base, "env": e, "label": "off" if e else "on"} for e in ({}, off)]
    if part == "infer":
        return [
            {"args": ["infer", "--model", m, "--arm", a], "env": {}, "label": f"{m}/{a}"}
            for m in MODELS
            for a in ("naive", "memopro", "memopro_os", "memopro_low", "memopro_4gb")
        ]
    if part == "train":
        return [
            {"args": ["train", "--arm", a, "--device", "mps"], "env": {}, "label": a}
            for a in ("naive", "session")
        ]
    if part == "alternate":
        return [
            {"args": ["alternate", "--arm", a], "env": e, "label": label}
            for a, e, label in (
                ("resident", {}, "resident"),
                ("reload", {}, "reload"),
                ("beta", {}, "beta"),
                ("beta", off, "beta_cache_off"),
            )
        ]
    raise ValueError(part)


class Probe:
    def __init__(self) -> None:
        self.p = subprocess.Popen(
            [sys.executable, "-m", "experiments.e014_usable.probe"],
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        self.p.stdout.readline()

    def ask(self, cmd: str) -> dict:
        self.p.stdin.write(cmd + "\n")
        self.p.stdin.flush()
        return json.loads(self.p.stdout.readline())

    def close(self) -> None:
        try:
            self.ask("quit")
        except (BrokenPipeError, json.JSONDecodeError):
            pass
        self.p.kill()


def run_case(case: dict, probe: Probe) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "MallocLargeCache"} | case["env"]
    probe.ask("mark")
    swap0 = swap_used()
    peak = {"footprint": 0, "swap": swap0}
    t0, clock = time.perf_counter(), time.time()
    p = subprocess.Popen(
        [sys.executable, "-m", "experiments.e014_usable.workers", *case["args"]],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    done = threading.Event()

    def sample() -> None:
        while not done.is_set():
            try:
                peak["footprint"] = max(peak["footprint"], rusage(p.pid)["phys_footprint"])
                peak["swap"] = max(peak["swap"], swap_used())
            except OSError:
                pass
            done.wait(0.2)

    threading.Thread(target=sample, daemon=True).start()
    try:
        out, err = p.communicate(timeout=TIMEOUT)
        timed_out = False
    except subprocess.TimeoutExpired:
        p.kill()
        out, err = p.communicate()
        timed_out = True
    done.set()
    line = next((x for x in out.splitlines() if x.startswith("RESULT ")), None)
    result = json.loads(line[7:]) if line else {"ok": False, "error": "no result"}
    if timed_out:
        result = {"ok": False, "error": f"timeout after {TIMEOUT} s"}
    return {
        "label": case["label"],
        "args": case["args"],
        "env": case["env"],
        "wall_s": time.perf_counter() - t0,
        "clock": [clock, time.time()],
        "returncode": p.returncode,
        "peak_footprint": peak["footprint"],
        "swap_growth": peak["swap"] - swap0,
        "probe": probe.ask("report"),
        "result": result,
        "stderr_tail": err[-1500:] if not result.get("ok") else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", default="reclaim,f4cost,infer,train,alternate")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--rest", type=float, default=10.0)
    parser.add_argument("--name", default="results")
    args = parser.parse_args()
    parts = args.parts.split(",")
    save_json(capture(__file__, extra={"argv": sys.argv}), OUT / f"env_{args.name}.json")
    probe = Probe()
    data: dict = {"parts": parts, "idle_probe": [], "cases": []}
    timed_out: set[str] = set()  # an arm that timed out once is not run again (0062)
    try:
        for part in parts:
            probe.ask("mark")
            time.sleep(10)
            data["idle_probe"].append({"part": part, **probe.ask("report")})
            items = cases(part)
            for rep in range(args.repeats):
                k = rep % len(items)
                for case in items[k:] + items[:k]:
                    free = shutil.disk_usage("/").free
                    key = f"{part}/{case['label']}"
                    if free < MIN_FREE_DISK:
                        rec = {"label": case["label"], "skipped": f"free disk {free} bytes"}
                    elif key in timed_out:
                        rec = {"label": case["label"], "skipped": "timed out in an earlier repeat"}
                    else:
                        rec = run_case(case, probe)
                        if rec["result"].get("error", "").startswith("timeout"):
                            timed_out.add(key)
                    rec.update(part=part, repeat=rep)
                    data["cases"].append(rec)
                    save_json(data, OUT / f"{args.name}.json")
                    r = rec.get("result", {})
                    brief = {
                        k: round(v, 3) if isinstance(v, float) else v
                        for k, v in r.items()
                        if k
                        in (
                            "ok",
                            "tokens_per_s",
                            "load_s",
                            "steady_step_s",
                            "total_s",
                            "reclaimed_frac",
                            "chosen",
                        )
                    }
                    print(
                        f"{part} {rep} {rec['label']:40} {brief} "
                        f"peak={rec.get('peak_footprint', 0) >> 20}MiB "
                        f"swap+={rec.get('swap_growth', 0) >> 20}MiB "
                        f"p95={rec.get('probe', {}).get('p95_ms', 0):.0f}ms "
                        f"{r.get('error', '')[:120]}",
                        flush=True,
                    )
                    time.sleep(args.rest)
    finally:
        probe.close()


if __name__ == "__main__":
    main()
