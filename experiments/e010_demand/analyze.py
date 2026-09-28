"""E010 analysis: per-session demand indicators and the Gβ verdict pre-registered in 0086 §4.

Input: a directory with one subdirectory per participant (any label, e.g. p1, p2), each holding
the `memopro_e010_*.jsonl` files that participant sent.

Usage: .venv/bin/python -m experiments.e010_demand.analyze DIR [--out FILE]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

GiB = 1 << 30
MIN_CELLS = 20  # a work session
IDLE_BIG = 1 * GiB  # S2
IDLE_SHARE = 0.25  # S2: of the device's memory
IDLE_UNDER_PRESSURE = GiB // 2  # S3
SWAP_GROWTH = 1 * GiB  # pressure
GPU_FULL = 0.9  # pressure: reserved / total
OS_LOW = 0.10  # pressure: OS-available / total
MIN_PARTICIPANTS, MIN_SESSIONS = 3, 8


def pressure(c: dict) -> bool:
    if c.get("oom"):
        return True
    if c.get("swap_growth_bytes", 0) >= SWAP_GROWTH:
        return True
    total = c.get("cuda_total_bytes")
    if total and c.get("cuda_reserved_bytes", 0) >= GPU_FULL * total:
        return True
    host = c.get("total_bytes")
    return bool(host and "os_available_bytes" in c and c["os_available_bytes"] < OS_LOW * host)


def session_metrics(records: list[dict]) -> dict | None:
    cells = [r for r in records if r.get("kind") == "cell"]
    if len(cells) < MIN_CELLS:
        return None
    capacity = cells[-1].get("cuda_total_bytes") or cells[-1].get("total_bytes") or 0
    peak = max(cells, key=lambda c: c["idle_bytes"])
    return {
        "cells": len(cells),
        "peak_idle_bytes": peak["idle_bytes"],
        "peak_idle_share_of_tracked": peak["idle_bytes"] / peak["tracked_bytes"]
        if peak["tracked_bytes"]
        else 0.0,
        "S2_meaningful_idle": any(
            c["idle_bytes"] >= IDLE_BIG or (capacity and c["idle_bytes"] >= IDLE_SHARE * capacity)
            for c in cells
        ),
        "S3_idle_under_pressure": any(
            c["idle_bytes"] >= IDLE_UNDER_PRESSURE and pressure(c) for c in cells
        ),
        "S4_source_share_at_peak": peak["idle_source_bytes"] / peak["idle_bytes"]
        if peak["idle_bytes"]
        else None,
        "oom_cells": sum(bool(c.get("oom")) for c in cells),
        "pressure_cells": sum(pressure(c) for c in cells),
    }


def verdict(sessions: list[dict], participants: int) -> dict:
    n = len(sessions)
    if participants < MIN_PARTICIPANTS or n < MIN_SESSIONS:
        return {
            "gbeta": "no verdict (not enough data)",
            "participants": participants,
            "work_sessions": n,
        }
    s2 = sum(s["S2_meaningful_idle"] for s in sessions) / n
    s3 = sum(s["S3_idle_under_pressure"] for s in sessions) / n
    shares = [s["S4_source_share_at_peak"] for s in sessions if s["S4_source_share_at_peak"] is not None]
    s4 = statistics.median(shares) if shares else None
    if s3 >= 0.25:
        gbeta = "strong"
    elif s2 >= 0.5:
        gbeta = "moderate"
    else:
        gbeta = "weak"
    source = (
        None
        if s4 is None
        else "source first confirmed"
        if s4 >= 0.5
        else "reconsider mode order"
        if s4 < 0.2
        else "keep source first"
    )
    return {
        "gbeta": gbeta,
        "S2_share": s2,
        "S3_share": s3,
        "S4_median": s4,
        "source_mode": source,
        "participants": participants,
        "work_sessions": n,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    root = Path(args.directory)
    per_participant: dict[str, list[dict]] = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(d.glob("memopro_e010_*.jsonl")):
            records = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
            m = session_metrics(records)
            if m is not None:
                per_participant.setdefault(d.name, []).append({"file": f.name, **m})
    sessions = [s for ss in per_participant.values() for s in ss]
    result = {
        "per_participant": per_participant,
        "verdict": verdict(sessions, len(per_participant)),
    }
    text = json.dumps(result, indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(json.dumps(result["verdict"], indent=1))


if __name__ == "__main__":
    main()
