"""E016: RCR inference prototype against the E014 arms. Pre-registered in docs/research/0070.

Each case is a fresh worker (workers.py) and has two phases, measured from outside like E014
(worker footprint sampled every 0.2 s, system swap growth, responsiveness probe):

  A  load, warm up, first-token time, 64 tokens (the machine as it is)
  B  the E011 pressure process fills P bytes of random data, then 64 more tokens

Usage: caffeinate -is .venv/bin/python -m experiments.e016_rcr.run --ffiles DIR [--repeats 3]
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

import memopro
from experiments._harness.env import capture, save_json
from experiments.e011_os_swap.common import rusage, swap_used
from experiments.e014_usable.run import Probe

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "research" / "data" / "e016"
MODELS = {
    "Qwen/Qwen2.5-1.5B-Instruct": "qwen2.5-1.5b-int4.f",
    "Qwen/Qwen2.5-3B-Instruct": "qwen2.5-3b-int4.f",
}
ARMS = ("naive", "memopro_int4", "rcr_bf16", "rcr_int4", "rcr_int4_nopf")
TIMEOUT = 10 * 60
MIN_FREE_DISK = 5 << 30
GiB = 1 << 30


class Sampler:
    def __init__(self, pid: int) -> None:
        self.pid, self.peak, self.swap0 = pid, 0, swap_used()
        self.swap_peak = self.swap0
        self.done = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while not self.done.is_set():
            try:
                self.peak = max(self.peak, rusage(self.pid)["phys_footprint"])
                self.swap_peak = max(self.swap_peak, swap_used())
            except OSError:
                pass
            self.done.wait(0.2)

    def stop(self) -> dict:
        self.done.set()
        return {"peak_footprint": self.peak, "swap_growth": self.swap_peak - self.swap0}


def read_line(p: subprocess.Popen, prefix: str, timeout: float) -> dict:
    result: dict = {}

    def reader() -> None:
        for line in p.stdout:
            if line.startswith(prefix + " "):
                result.update(json.loads(line[len(prefix) + 1 :]))
                return

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        p.kill()
        return {"ok": False, "error": f"timeout after {timeout:.0f} s"}
    return result or {"ok": False, "error": f"worker exited ({p.poll()})"}


def run_case(model: str, arm: str, ffiles: Path, p_bytes: int, probe: Probe) -> dict:
    rec: dict = {"model": model, "arm": arm, "clock": [time.time()]}
    env = {k: v for k, v in os.environ.items() if k != "MallocLargeCache"}
    p = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "experiments.e016_rcr.workers",
            "--model",
            model,
            "--arm",
            arm,
            "--ffile",
            str(ffiles / MODELS[model]),
        ],
        cwd=ROOT,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    probe.ask("mark")
    s = Sampler(p.pid)
    t0 = time.perf_counter()
    rec["A"] = read_line(p, "A", TIMEOUT)
    rec["A_wall_s"] = time.perf_counter() - t0
    rec["A_mem"] = s.stop()
    rec["A_probe"] = probe.ask("report")
    if rec["A"].get("ok"):
        b = subprocess.Popen(
            [sys.executable, "-m", "experiments.e011_os_swap.pressure", "--bytes", str(p_bytes)],
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        rec["fill"] = json.loads(b.stdout.readline())
        time.sleep(5)
        probe.ask("mark")
        s = Sampler(p.pid)
        p.stdin.write("go\n")
        p.stdin.flush()
        rec["B"] = read_line(p, "B", TIMEOUT)
        rec["B_mem"] = s.stop()
        rec["B_probe"] = probe.ask("report")
        b.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
        b.stdin.flush()
        try:
            b.wait(timeout=30)
        except subprocess.TimeoutExpired:
            b.kill()
    try:
        p.stdin.close()
        p.wait(timeout=30)
    except (subprocess.TimeoutExpired, BrokenPipeError):
        p.kill()
    rec["clock"].append(time.time())
    return rec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ffiles", required=True, help="directory with the int4 F files")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--rest", type=float, default=10.0)
    parser.add_argument("--name", default="results")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--arms", default=",".join(ARMS))
    args = parser.parse_args()
    host = memopro.doctor(devices=False).env.host
    p_bytes = min(4 * GiB, host.kernel_available_bytes + GiB)
    save_json(
        capture(
            __file__,
            extra={
                "argv": sys.argv,
                "pressure_bytes": p_bytes,
                "available": host.available_bytes,
                "kernel_available": host.kernel_available_bytes,
            },
        ),
        OUT / f"env_{args.name}.json",
    )
    probe = Probe()
    data: dict = {"pressure_bytes": p_bytes, "idle_probe": [], "cases": []}
    timed_out: set[str] = set()
    arms = args.arms.split(",")
    try:
        probe.ask("mark")
        time.sleep(10)
        data["idle_probe"].append(probe.ask("report"))
        for rep in range(args.repeats):
            for m_i, model in enumerate(args.models.split(",")):
                k = (rep + m_i) % len(arms)
                for arm in arms[k:] + arms[:k]:
                    key = f"{model}/{arm}"
                    free = shutil.disk_usage("/").free
                    if free < MIN_FREE_DISK:
                        rec = {"model": model, "arm": arm, "skipped": f"free disk {free}"}
                    elif key in timed_out:
                        rec = {"model": model, "arm": arm, "skipped": "timed out earlier"}
                    else:
                        rec = run_case(model, arm, Path(args.ffiles), p_bytes, probe)
                        if str(rec["A"].get("error", "")).startswith("timeout") or str(
                            rec.get("B", {}).get("error", "")
                        ).startswith("timeout"):
                            timed_out.add(key)
                    rec["repeat"] = rep
                    data["cases"].append(rec)
                    save_json(data, OUT / f"{args.name}.json")
                    a, b = rec.get("A", {}), rec.get("B", {})
                    print(
                        f"{rep} {model.split('/')[-1]:22} {arm:14} "
                        f"A={a.get('tokens_per_s', 0):.1f}tok/s load={a.get('load_s', 0):.1f}s "
                        f"peak={rec.get('A_mem', {}).get('peak_footprint', 0) >> 20}MiB "
                        f"swap+={rec.get('A_mem', {}).get('swap_growth', 0) >> 20}MiB "
                        f"p95={rec.get('A_probe', {}).get('p95_ms', 0):.0f}ms | "
                        f"B={b.get('tokens_per_s', 0):.1f}tok/s "
                        f"p95={rec.get('B_probe', {}).get('p95_ms', 0):.0f}ms "
                        f"{a.get('error', '')[:80]}{b.get('error', '')[:80]} "
                        f"{rec.get('skipped', '')}",
                        flush=True,
                    )
                    time.sleep(args.rest)
    finally:
        probe.close()


if __name__ == "__main__":
    main()
