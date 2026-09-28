"""E017: the comfort controller against uncontrolled RCR (pre-registered in docs/research/0073).

Each case is a fresh worker (workers.py) with three phases, measured from outside as in E016:
  A  calm: 64 tokens
  B  the E011 pressure process holds P bytes of random data: 128 tokens. Measured in two parts:
     the first 3 s (re-establishing a working set the pressure evicted) and the rest (steady)
  C  pressure released (process ended, 10 s rest): 64 tokens

Usage: caffeinate -is .venv/bin/python -m experiments.e017_controller.run --cache-dir DIR
       [--prep]    build the int4 file caches once (disk writes allowed, into DIR)
       [--resume]  continue an interrupted run: keep results.json, its pressure size and idle
                   probe, and run only the (repeat, model, arm) cases not recorded yet
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import memopro
from experiments._harness.env import capture, save_json
from experiments.e011_os_swap.common import swap_used
from experiments.e014_usable.run import Probe
from experiments.e016_rcr.run import Sampler, read_line

SETTLE_S = 3.0  # B is split: the first seconds re-establish a working set, the rest is steady

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "research" / "data" / "e017"
MODELS = ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct")
ARMS = ("bf16", "bf16_ctl", "int4", "int4_ctl")
TIMEOUT = 10 * 60
GiB = 1 << 30


def prep(cache_dir: str) -> None:
    from memopro.access._load import plan_load

    memopro.configure(spill_dir=cache_dir, min_free_disk_fraction=0.0, disk_writes="allow")
    for model in MODELS:
        int4 = next(c for c in plan_load(model, device="mps").candidates if c.name == "quant.int4")
        memopro.load(
            model,
            residency="file",
            quality="low",
            budget=int4.needs.device + int4.needs.host + (100 << 20),
        )
        print(model, memopro.report().entries[-1].detail[-90:], flush=True)


def start_pressure(p_bytes: int) -> subprocess.Popen:
    b = subprocess.Popen(
        [sys.executable, "-m", "experiments.e011_os_swap.pressure", "--bytes", str(p_bytes)],
        cwd=ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    b.stdout.readline()
    return b


def stop(b: subprocess.Popen) -> None:
    try:
        b.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
        b.stdin.flush()
        b.wait(timeout=30)
    except (BrokenPipeError, subprocess.TimeoutExpired):
        b.kill()


def run_case(model: str, arm: str, cache_dir: str, p_bytes: int, probe: Probe) -> dict:
    rec: dict = {"model": model, "arm": arm, "clock": [time.time()]}
    env = {k: v for k, v in os.environ.items() if k != "MallocLargeCache"}
    p = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "experiments.e017_controller.workers",
            "--model",
            model,
            "--arm",
            arm,
            "--cache-dir",
            cache_dir,
        ],
        cwd=ROOT,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    b = None
    try:
        for phase in ("A", "B", "C"):
            if phase == "B":
                b = start_pressure(p_bytes)
                time.sleep(5)
            if phase == "C":
                stop(b)
                b = None
                time.sleep(10)
            probe.ask("mark")
            s = Sampler(p.pid)
            swap0 = swap_used()
            if phase != "A":
                p.stdin.write(f"go {phase}\n")
                p.stdin.flush()
            if phase == "B":
                time.sleep(SETTLE_S)
                swap_settled = swap_used()
                rec["B_initial_probe"] = probe.ask("report")
                probe.ask("mark")
            rec[phase] = read_line(p, phase, TIMEOUT)
            rec[f"{phase}_mem"] = s.stop()
            rec[f"{phase}_probe"] = probe.ask("report")
            if phase == "B":
                end = swap_used()
                rec["B_swap"] = {"initial": swap_settled - swap0, "steady": end - swap_settled}
            if not rec[phase].get("ok"):
                break
    finally:
        if b is not None:
            stop(b)
        try:
            p.stdin.close()
            p.wait(timeout=30)
        except (subprocess.TimeoutExpired, BrokenPipeError):
            p.kill()
    rec["clock"].append(time.time())
    return rec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--prep", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--rest", type=float, default=10.0)
    parser.add_argument("--name", default="results")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        help="MODEL=ARM not to run (recorded as skipped), e.g. an amendment",
    )
    args = parser.parse_args()
    if args.prep:
        prep(args.cache_dir)
        return
    previous = OUT / f"{args.name}.json"
    if args.resume and previous.exists():
        resume(args, json.loads(previous.read_text()))
        return
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
    try:
        probe.ask("mark")
        time.sleep(10)
        data["idle_probe"].append(probe.ask("report"))
        run_all(args, data, probe, done=set())
    finally:
        probe.close()


def resume(args: argparse.Namespace, data: dict) -> None:
    """Continue with the same pressure size; cases already recorded are not run again."""
    done = {(c["repeat"], c["model"], c["arm"]) for c in data["cases"]}
    n = len(data.setdefault("resumed", [])) + 1
    data["resumed"].append({"at": time.time(), "cases_before": len(data["cases"])})
    save_json(data, OUT / f"{args.name}.json")
    save_json(
        capture(
            __file__,
            extra={"argv": sys.argv, "resume": n, "pressure_bytes": data["pressure_bytes"]},
        ),
        OUT / f"env_{args.name}_resume{n}.json",
    )
    probe = Probe()
    try:
        run_all(args, data, probe, done)
    finally:
        probe.close()


def run_all(args: argparse.Namespace, data: dict, probe: Probe, done: set) -> None:
    p_bytes = data["pressure_bytes"]
    arms = args.arms.split(",")
    for rep in range(args.repeats):
        for m_i, model in enumerate(args.models.split(",")):
            k = (rep + m_i) % len(arms)
            for arm in arms[k:] + arms[:k]:
                if (rep, model, arm) in done:
                    continue
                if f"{model}={arm}" in args.skip:
                    data["cases"].append(
                        {
                            "model": model,
                            "arm": arm,
                            "repeat": rep,
                            "skipped": "excluded by amendment (amendment_resume.md)",
                        }
                    )
                    save_json(data, OUT / f"{args.name}.json")
                    continue
                if shutil.disk_usage("/").free < (5 << 30):
                    rec = {"model": model, "arm": arm, "skipped": "free disk"}
                else:
                    rec = run_case(model, arm, args.cache_dir, p_bytes, probe)
                rec["repeat"] = rep
                data["cases"].append(rec)
                save_json(data, OUT / f"{args.name}.json")
                brief = " | ".join(
                    f"{ph} {rec.get(ph, {}).get('tokens_per_s', 0):.1f}tok/s "
                    f"{rec.get(ph, {}).get('active_end', '-')} "
                    f"swap+{rec.get(f'{ph}_mem', {}).get('swap_growth', 0) >> 20}MiB "
                    f"p95 {rec.get(f'{ph}_probe', {}).get('p95_ms', 0):.0f}ms"
                    for ph in ("A", "B", "C")
                )
                err = next(
                    (
                        rec[ph].get("error", "")
                        for ph in ("A", "B", "C")
                        if ph in rec and not rec[ph].get("ok")
                    ),
                    "",
                )
                print(f"{rep} {model.split('/')[-1]:22} {arm:9} {brief} {err[:100]}", flush=True)
                time.sleep(args.rest)


if __name__ == "__main__":
    main()
