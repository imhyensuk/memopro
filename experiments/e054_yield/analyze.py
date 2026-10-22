"""E054 analysis, exactly as pre-registered (docs/research/0282 §6, 0283 §3, 0295 §2-§4).

A second is harmful if the P1 probe's worst read in it exceeds 24.1 ms, or the video tab
reported a frame gap over 100 ms or 3+ dropped frames in it (P2). Per run: H0 (baseline), H
(generation), ΔH = H - H0, Dv (dropped/total frames while generating), tokens/s and time to first
token (TTFT); per (model, condition) the median of the valid runs. Validity: B and C must
generate the same tokens per prompt (lossless greedy), B4 runs must agree with each other.

    python -m experiments.e054_yield.analyze [--quick]
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict

from experiments.e054_yield.run import OUT

P1_MS, P2_GAP_MS, P2_DROPPED = 24.1, 100.0, 3


def seconds(rec: dict, phase: str) -> list[int]:
    return [int(r["wall"]) for r in rec["samples"] if r["phase"] == phase]


def harmful_fraction(rec: dict, phase: str) -> float | None:
    p1 = {int(r["wall"]): (r.get("probe_max_ms") or 0) > P1_MS
          for r in rec["samples"] if r["phase"] == phase}
    p2 = defaultdict(bool)
    for f in rec["frames"]:
        p2[int(f["wall"])] |= f["max_gap_ms"] > P2_GAP_MS or f["dropped"] >= P2_DROPPED
    if not p1:
        return None
    return sum(p1[s] or p2[s] for s in p1) / len(p1)


def dropped_fraction(rec: dict, phase: str) -> float | None:
    secs = seconds(rec, phase)
    if not secs:
        return None
    lo, hi = min(secs) - 1, max(secs)
    frames = [f for f in rec["frames"] if lo < f["wall"] <= hi]
    total = sum(f["total"] for f in frames)
    return sum(f["dropped"] for f in frames) / total if total else None


def speed(rec: dict) -> tuple[float | None, float | None]:
    """(tokens/s, TTFT s). Tokens/s = (n - 1) / (t_n - t_1) summed over generations (for a
    complete 256-token one this is 0282's (256 - 1) / (T256 - T1)); TTFT is the median time to
    the first token, or Ollama's prompt_eval_duration when it reported one."""
    gens = rec["model"]["generations"]
    span = sum(g["times"][-1] - g["times"][0] for g in gens if len(g["times"]) > 1)
    count = sum(len(g["times"]) - 1 for g in gens if len(g["times"]) > 1)
    ttft = [g.get("prompt_eval_duration_s") or (g["times"][0] - g["start"])
            for g in gens if g["times"]]
    return (count / span if span else None), (statistics.median(ttft) if ttft else None)


def per_run(rec: dict) -> dict:
    h0, h = harmful_fraction(rec, "baseline"), harmful_fraction(rec, "generate")
    tps, ttft = speed(rec) if rec.get("model") else (None, None)
    peak = max((r.get("model_footprint") or 0 for r in rec["samples"]
                if r["phase"] == "generate"), default=0)
    return {"index": rec["spec"]["index"], "model": rec["spec"]["model"],
            "cond": rec["spec"]["cond"], "H0": h0, "H": h,
            "dH": None if h is None or h0 is None else h - h0,
            "Dv": dropped_fraction(rec, "generate"), "tok_s": tps, "ttft_s": ttft,
            "model_peak_footprint": peak, "app_memory_total": rec.get("app_memory", {}).get("total"),
            "load_s": rec.get("load_s"), "ssd_written": _delta(rec, "ssd_written")}


def _delta(rec: dict, key: str):
    a, b = rec.get("before", {}).get(key), rec.get("after", {}).get(key)
    return None if a is None or b is None else b - a


def validity(recs: list[dict]) -> dict[int, str]:
    """Run index -> reason for runs whose tokens disagree (0282 §2, 0295 §1)."""
    bad: dict[int, str] = {}
    for model in ("S", "L"):
        for group in (("B", "C"), ("B4",)):
            ref: dict[int, list[int]] = {}
            for rec in sorted(recs, key=lambda r: r["spec"]["index"]):
                s = rec["spec"]
                if s["model"] != model or s["cond"] not in group or not rec.get("model"):
                    continue
                for g in rec["model"]["generations"]:
                    if not g.get("complete"):
                        continue
                    first = ref.setdefault(g["prompt"], g["tokens"])
                    if g["tokens"] != first:
                        bad[s["index"]] = f"prompt {g['prompt']} differs from earlier {group}"
    return bad


def med(rows: list[dict], key: str):
    vals = [r[key] for r in rows if r[key] is not None]
    return statistics.median(vals) if vals else None


def speed_class(tps, ttft) -> str:  # 0282 Q4
    if tps is not None and ttft is not None and tps >= 5 and ttft <= 3:
        return "chat"
    if tps is not None and tps >= 1:
        return "background"
    return "unusable"


def judge(cells: dict) -> dict:
    out: dict = {}
    for model in ("S", "L"):
        a, b, c = (cells.get((model, k), {}) for k in ("A", "B", "C"))
        q1 = (a.get("dH") or 0) >= 0.05
        res = {"Q1": q1}
        if q1:
            res["Q2"] = (b.get("dH") is not None and b["dH"] <= 0.5 * a["dH"]
                         and (b.get("Dv") or 0) <= (a.get("Dv") or 0))
            if b.get("dH") is not None and b["dH"] < 0.02:
                res["Q3"] = "B is enough"
            elif b.get("dH") is not None and c.get("dH") is not None:
                res["Q3"] = c["dH"] <= 0.5 * b["dH"]
        res["Q4"] = {k: speed_class(v.get("tok_s"), v.get("ttft_s"))
                     for (m, k), v in cells.items() if m == model}
        out[model] = res
    a0, b4, b = (cells.get(("L", k), {}) for k in ("A0", "B4", "B"))
    if (a0.get("dH") or 0) >= 0.05:
        out["Q5"] = (b4.get("dH") is not None and b4["dH"] <= 0.5 * a0["dH"]
                     and (b4.get("Dv") or 0) <= (a0.get("Dv") or 0))
    else:
        out["Q5"] = "not reproduced on A0"
    out["Q6"] = {k: {x: v.get(x) for x in ("dH", "Dv", "tok_s", "ttft_s", "model_peak_footprint")}
                 for k, v in (("B4", b4), ("B", b))}
    apps = [v.get("app_memory_total") for v in cells.values() if v.get("app_memory_total")]
    if apps and statistics.median(apps) > 3e9:  # 0283 §3
        out["note_0283"] = f"reserved 3 GB < user apps {statistics.median(apps) / 1e9:.1f} GB"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    runs_dir = OUT / ("runs_quick" if a.quick else "runs")
    recs = [json.loads(p.read_text()) for p in sorted(runs_dir.glob("*.json"))]
    bad = validity(recs)
    rows = [per_run(r) | {"invalid": bad.get(r["spec"]["index"])} for r in recs]
    groups: dict[tuple, list] = defaultdict(list)
    for r in rows:
        if not r["invalid"]:
            groups[(r["model"], r["cond"])].append(r)
    keys = ("H0", "H", "dH", "Dv", "tok_s", "ttft_s", "model_peak_footprint", "app_memory_total")
    cells = {k: {x: med(v, x) for x in keys} | {"n": len(v)} for k, v in groups.items()}
    summary = {"runs": rows, "cells": {f"{m} {c}": v for (m, c), v in sorted(cells.items())},
               "judgement": judge(cells)}
    (runs_dir.parent / ("summary_quick.json" if a.quick else "summary.json")).write_text(
        json.dumps(summary, indent=1, default=str)
    )
    print(json.dumps({"cells": summary["cells"], "judgement": summary["judgement"]}, indent=1,
                     default=str))


if __name__ == "__main__":
    main()
