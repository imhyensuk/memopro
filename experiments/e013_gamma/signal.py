"""E013a: is γ's pressure signal informative on this machine? (pre-registered in 0087)

Every second: macOS `kern.memorystatus_vm_pressure_level` (γ's signal, via memopro.elastic),
system swap in use, memopro's conservative available memory and the OS estimate, and the E014
responsiveness probe's slowest read in that second. Schedule, twice: calm 60 s, then for P in
1, 2, 3, 4 GiB the E011 pressure process holds P bytes of random data for 60 s and is released
for 30 s; a final calm 30 s. SSD bytes written (ioreg) are recorded before and after.

Usage: caffeinate -is .venv/bin/python -m experiments.e013_gamma.signal [--rounds 2]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import memopro
from experiments._harness.env import capture, save_json
from experiments.e014_usable.run import Probe
from experiments.e017_controller.controller import swap_used
from experiments.e017_controller.run import start_pressure, stop

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "research" / "data" / "e013"
GiB = 1 << 30


def ssd_written() -> int | None:
    out = subprocess.run(
        ["ioreg", "-c", "IOBlockStorageDriver", "-r", "-k", "Statistics"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    values = [int(v) for v in re.findall(r'"Bytes \(Write\)"=(\d+)', out)]
    return max(values) if values else None


def sample(phase: str, probe: Probe, t0: float) -> dict:
    from memopro import elastic

    report = probe.ask("report")
    probe.ask("mark")
    host = memopro.doctor(devices=False).env.host
    reading = elastic.current()
    return {
        "t": round(time.monotonic() - t0, 2),
        "phase": phase,
        "raw_level": reading.get("raw_level"),
        "level": reading.get("level"),
        "swap_used": swap_used(),
        "available": host.available_bytes,
        "kernel_available": host.kernel_available_bytes,
        "total": host.total_bytes,
        "probe_max_ms": report.get("max_ms"),
        "probe_n": report.get("n"),
    }


def hold(seconds: float, phase: str, probe: Probe, rows: list, t0: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        tick = time.monotonic()
        rows.append(sample(phase, probe, t0))
        time.sleep(max(0.0, 1.0 - (time.monotonic() - tick)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--name", default="signal")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    save_json(capture(__file__, extra={"argv": sys.argv}), OUT / f"env_{args.name}.json")
    written0 = ssd_written()
    probe = Probe()
    rows: list[dict] = []
    t0 = time.monotonic()
    try:
        probe.ask("mark")
        for r in range(args.rounds):
            hold(60, f"r{r}:calm", probe, rows, t0)
            for gib in (1, 2, 3, 4):
                b = start_pressure(gib * GiB)
                try:
                    hold(60, f"r{r}:P{gib}", probe, rows, t0)
                finally:
                    stop(b)
                hold(30, f"r{r}:after{gib}", probe, rows, t0)
                print(f"round {r} P{gib} done, {len(rows)} samples", flush=True)
        hold(30, "final:calm", probe, rows, t0)
    finally:
        probe.close()
        written1 = ssd_written()
        data = {
            "rows": rows,
            "ssd_written_bytes": None if written0 is None or written1 is None else written1 - written0,
        }
        (OUT / f"{args.name}.json").write_text(json.dumps(data) + "\n")
        print(f"saved {len(rows)} samples; SSD written {data['ssd_written_bytes']}", flush=True)


if __name__ == "__main__":
    main()
