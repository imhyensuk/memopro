"""E015 analysis: the verdicts pre-registered in 0067 (G-F, G-P, Q1, Q4).

Usage: .venv/bin/python -m experiments.e015_foundations.analyze [results.json]
Writes docs/research/data/e015/summary_<name>.json and prints a report.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "docs" / "research" / "data" / "e015"
GiB = 1 << 30


def q1(rows: list[dict]) -> dict:
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["mode"], []).append(r)
    out = {}
    for mode, rs in by.items():
        gbps, secs = [], []
        for r in rs:
            s = r["reread"]["seconds"]
            res = r["reread"].get("resident_before")
            missing = GiB if res is None or mode == "read_nocache" else GiB * (1 - res)
            secs.append(s)
            gbps.append(missing / s / 1e9 if s > 0 else None)
        out[mode] = {
            "seconds_median": statistics.median(secs),
            "gbps_missing_median": statistics.median([g for g in gbps if g is not None]),
            "resident_before": [r["reread"].get("resident_before") for r in rs],
            "swap_growth_mib": [r.get("swap_growth", 0) >> 20 for r in rs],
        }
    mm = {m: v for m, v in out.items() if m.startswith("mmap")}
    best = max(mm, key=lambda m: mm[m]["gbps_missing_median"]) if mm else None
    warm = out.get("mmap_readwarm", {}).get("gbps_missing_median")
    plain = out.get("mmap_plain", {}).get("gbps_missing_median")
    return {
        "modes": out,
        "best_mapping_mode": best,
        "readwarm_vs_plain": (warm / plain) if warm and plain else None,
        "explicit_warm_needed": bool(warm and plain and warm >= 2 * plain),
    }


def q2(rows: list[dict]) -> dict:
    cases = []
    for r in rows:
        s = r["series"]
        t_anon = next((x["t"] for x in s if x["anon"] < 0.5), None)
        rec = {"t_anon_below_half": t_anon, "vm": r["vm"]}
        ok = True
        for key in ("mach", "metal"):
            t = next((x["t"] for x in s if x[key] == "empty"), None)
            rec[f"t_{key}_empty"] = t
            anon_at = next((x["anon"] for x in s if x[key] == "empty"), None)
            rec[f"anon_resident_at_{key}_empty"] = anon_at
            emptied = t is not None
            in_time = emptied and (t_anon is None or t <= t_anon + 0.5)
            rec[f"{key}_ok"] = emptied and in_time
            ok = ok and emptied and in_time
        rec["pass"] = ok
        cases.append(rec)
    return {"cases": cases, "G_P_pass": all(c["pass"] for c in cases) and len(cases) == 3}


def q3(rows: list[dict]) -> dict:
    cases = []
    for r in rows:
        p = r["prepare"]
        cond1 = bool(p.get("accepted") and p.get("exact") and p.get("int4_exact"))
        cond2 = p.get("footprint_delta", 1e18) < 0.10 * p.get("model_bytes", 1)
        cond3 = r["probe_pressed"]["resident"] < 0.9
        cond4 = bool(r["recheck"].get("exact"))
        cases.append(
            {
                "accepted_exact_int4": cond1,
                "footprint_delta_mib": p.get("footprint_delta", 0) / 2**20,
                "footprint_ok": cond2,
                "resident_pressed": r["probe_pressed"]["resident"],
                "dropped_ok": cond3,
                "recheck_exact": cond4,
                "recheck_s": r["recheck"].get("seconds"),
                "recheck_resident_before": r["recheck"].get("resident_before"),
                "swap_growth_mib": r.get("swap_growth", 0) >> 20,
                "first_compute_s": p.get("first_compute_s"),
                "pass": cond1 and cond2 and cond3 and cond4,
            }
        )
    return {"cases": cases, "G_F_pass": all(c["pass"] for c in cases) and len(cases) == 3}


def q4(runs: list[dict]) -> dict:
    shapes = list(runs[0])
    table = {}
    for name in shapes:
        row = {}
        for key in runs[0][name]:
            vals = [r[name][key] for r in runs if isinstance(r[name].get(key), float)]
            if vals:
                row[key] = statistics.median(vals)
        table[name] = row
    bf = sum(table[n]["bfloat16"] for n in shapes)
    int4 = min(sum(table[n][f"int4pack_g{g}"] for n in shapes) for g in (32, 128))
    return {
        "table_ms": {n: {k: v * 1e3 for k, v in r.items()} for n, r in table.items()},
        "bf16_sum_ms": bf * 1e3,
        "int4_sum_ms": int4 * 1e3,
        "speedup": bf / int4,
        "fast_int4_available": bf / int4 >= 1.5,
    }


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / "results.json"
    d = json.loads(path.read_text())
    summary = {"pressure_bytes": d["pressure_bytes"]}
    if "q1" in d:
        summary["Q1"] = q1(d["q1"])
    if "q2" in d:
        summary["G_P"] = q2(d["q2"])
    if "q3" in d:
        summary["G_F"] = q3(d["q3"])
    if "q4" in d:
        summary["Q4"] = q4(d["q4"])
    (OUT / f"summary_{path.stem}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
