"""E011: β (memopro hibernate) against leaving idle memory to macOS (compressor + swap).

Pre-registered in docs/research/0058. One trial:

  1. A (holder.py) loads GPT-2 fp32 on the device and warms up. On the CPU the weights are either
     mapped from the file as transformers loads them (mmap) or copied to anonymous memory (anon)
  2. A goes idle by one arm: os (nothing), source, compress (memopro.hibernate), reload (del),
     and on the CPU the diagnostic source_nocache (source with macOS' malloc large cache off)
  3. B (pressure.py) fills P bytes of random data: other work that needs memory
  4. with B holding its memory: A's footprint and the weights' resident pages
  5. A resumes (wake / reload / nothing) and runs a forward pass: return latency, exactness
  6. B reads one byte per page: what A's return cost the other work

Arms rotate within each repeat and conditions alternate, so drift in the machine's state spreads
over all arms. Usage:

  .venv/bin/python -m experiments.e011_os_swap.run --repeats 5
  .venv/bin/python -m experiments.e011_os_swap.run --pilot   # 1 repeat, cpu/anon, 1 GiB; not judged
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import memopro
from experiments._harness.env import capture, save_json, vm_snapshot
from experiments.e011_os_swap.common import rusage, swap_used

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "research" / "data" / "e011"
ARMS = ("os", "source", "compress", "reload")
# (device, backing, arms); source_nocache only where malloc holds the memory: CPU anonymous weights
CONDITIONS = (
    ("cpu", "mmap", ARMS),
    ("cpu", "anon", (*ARMS, "source_nocache")),
    ("mps", "anon", ARMS),
)
MODEL_BYTES = 497_759_232  # GPT-2 parameters, fp32 (checked against the holder's report)
CAP = 4 << 30


class Proc:
    def __init__(self, *args: str, env: dict[str, str] | None = None) -> None:
        self.p = subprocess.Popen(
            [sys.executable, "-m", *args],
            cwd=ROOT,
            env=None if env is None else {**os.environ, **env},
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )

    @property
    def pid(self) -> int:
        return self.p.pid

    def read(self) -> dict:
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError(f"{self.p.args} exited with {self.p.wait()}")
        return json.loads(line)

    def ask(self, **cmd) -> dict:
        self.p.stdin.write(json.dumps(cmd) + "\n")
        self.p.stdin.flush()
        return self.read()

    def close(self) -> None:
        try:
            self.ask(cmd="quit")
        except (BrokenPipeError, RuntimeError, json.JSONDecodeError):
            pass
        try:
            self.p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.p.kill()


def host() -> dict[str, int]:
    h = memopro.doctor(devices=False).env.host
    return {
        "available": h.available_bytes,
        "kernel_available": h.kernel_available_bytes,
        "swap_used": swap_used(),
    }


def compressor(vm: dict) -> dict[str, int]:
    keys = ("Pages stored in compressor", "Pages occupied by compressor", "Swapouts", "Swapins")
    return {k: vm.get(k, 0) for k in keys}


def trial(device: str, backing: str, arm: str, pressure: int, settle: float) -> dict:
    rec: dict = {
        "device": device,
        "backing": backing,
        "arm": arm,
        "pressure_bytes": pressure,
        "host_before": host(),
    }
    env = {"MallocLargeCache": "0"} if arm == "source_nocache" else None
    a = Proc("experiments.e011_os_swap.holder", "--device", device, "--backing", backing, env=env)
    b = None
    try:
        rec["start"] = a.ask(cmd="start")
        rec["a_loaded"] = rusage(a.pid)
        rec["act"] = a.ask(cmd="act", arm=arm)
        rec["a_idle"] = rusage(a.pid)
        rec["probe_idle"] = a.ask(cmd="probe")
        vm0 = vm_snapshot()
        b = Proc("experiments.e011_os_swap.pressure", "--bytes", str(pressure))
        rec["fill"] = b.read()
        time.sleep(settle)
        rec["a_pressed"] = rusage(a.pid)
        rec["b_pressed"] = rusage(b.pid)
        rec["probe_pressed"] = a.ask(cmd="probe")
        rec["compressor_delta"] = {
            k: v - compressor(vm0)[k] for k, v in compressor(vm_snapshot()).items()
        }
        rec["host_pressed"] = host()
        before = rusage(a.pid)
        rec["resume"] = a.ask(cmd="resume")
        after = rusage(a.pid)
        rec["a_resumed"] = after
        rec["a_resume_io"] = {k: after[k] - before[k] for k in ("pageins", "diskio_bytesread")}
        rec["b_pass"] = b.ask(cmd="pass")
        rec["b_after"] = rusage(b.pid)
    finally:
        if b is not None:
            b.close()
        a.close()
    return rec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--pressure", default="auto", help="bytes, or auto (0058)")
    parser.add_argument("--settle", type=float, default=5.0)
    parser.add_argument("--rest", type=float, default=15.0, help="pause between trials")
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    if args.pilot:
        args.repeats, args.pressure = 1, str(1 << 30)
    conditions = CONDITIONS[1:2] if args.pilot else CONDITIONS

    start = host()
    if args.pressure == "auto":
        pressure = min(CAP, start["kernel_available"] + MODEL_BYTES)
    else:
        pressure = int(args.pressure)
    name = "pilot2" if args.pilot else "results"
    env = capture(
        __file__,
        extra={"argv": sys.argv, "pressure_bytes": pressure, "host_at_start": start},
    )
    save_json(env, OUT / f"env_{name}.json")

    trials = []
    for rep in range(args.repeats):
        for c_i, (device, backing, arms) in enumerate(conditions):
            k = (rep + c_i) % len(arms)
            for arm in arms[k:] + arms[:k]:
                t0 = time.perf_counter()
                try:
                    rec = trial(device, backing, arm, pressure, args.settle)
                except Exception as e:  # noqa: BLE001 - keep the run going, record the failure
                    rec = {"device": device, "backing": backing, "arm": arm, "error": repr(e)}
                rec["repeat"] = rep
                rec["wall_s"] = time.perf_counter() - t0
                trials.append(rec)
                save_json({"pressure_bytes": pressure, "trials": trials}, OUT / f"{name}.json")
                brief = {
                    k: rec.get(k, {}).get(v)
                    for k, v in (("fill", "fill_s"), ("resume", "resume_s"), ("b_pass", "pass_s"))
                }
                print(
                    f"rep {rep} {device}/{backing} {arm:14} {brief} {rec.get('error', '')}",
                    flush=True,
                )
                time.sleep(args.rest)


if __name__ == "__main__":
    main()
