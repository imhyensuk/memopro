"""E014 analysis: medians per part and arm, and the criteria pre-registered in 0062 (C1-C7).

Usage: .venv/bin/python -m experiments.e014_usable.analyze [results.json]
Writes docs/research/data/e014/summary.json and prints a table.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "docs" / "research" / "data" / "e014"
MiB = 2**20

# "comfortable" (0062 C2)
MIN_TOKENS_PER_S = 5.0
MAX_LOAD_S = 120.0
MAX_SWAP_GROWTH = 1 << 30
MAX_PROBE_FACTOR = 2.0  # probe p95 at most twice the idle p95


def med(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def group(cases, part):
    out: dict[str, list[dict]] = {}
    for c in cases:
        if c.get("part") == part and "skipped" not in c:
            out.setdefault(c["label"], []).append(c)
    return out


def summarize(data: dict) -> dict:
    cases = data["cases"]
    idle = med([p.get("p95_ms") for p in data["idle_probe"]])
    table: dict = {}
    for part in data["parts"]:
        rows = {}
        for label, runs in group(cases, part).items():
            ok = [c for c in runs if c["result"].get("ok")]
            r = [c["result"] for c in ok]
            row = {
                "n": len(runs),
                "ok": len(ok),
                "errors": sorted({c["result"].get("error", "")[:160] for c in runs if c not in ok}),
                "peak_footprint_mib": med([c["peak_footprint"] / MiB for c in runs]),
                "swap_growth_mib": med([c["swap_growth"] / MiB for c in runs]),
                "probe_p95_ms": med([c["probe"].get("p95_ms") for c in runs]),
                "probe_max_ms": med([c["probe"].get("max_ms") for c in runs]),
                "wall_s": med([c["wall_s"] for c in runs]),
            }
            for key in (
                "load_s",
                "ttft_s",
                "tokens_per_s",
                "steady_step_s",
                "total_s",
                "turns_after_loading_s",
                "reclaimed_frac",
                "sleep_s",
                "wake_s",
            ):
                values = [x.get(key) for x in r if key in x]
                if values:
                    row[key] = med(values)
            chosen = [tuple(x["chosen"]) for x in r if x.get("chosen")]
            if chosen:
                row["chosen"] = sorted(set(chosen))
            if part == "alternate":
                row["outputs_equal"] = all(x.get("outputs_equal") for x in r) if r else None
            rows[label] = row
        table[part] = rows
    return {"idle_probe_p95_ms": idle, "table": table, "criteria": criteria(table, cases, idle)}


def comfortable(row: dict, idle: float | None) -> bool | None:
    if not row or not row.get("ok"):
        return False
    checks = [
        (row.get("tokens_per_s") or 0) >= MIN_TOKENS_PER_S,
        (row.get("load_s") or 1e9) <= MAX_LOAD_S,
        (row.get("swap_growth_mib") or 0) * MiB <= MAX_SWAP_GROWTH,
        idle is None or (row.get("probe_p95_ms") or 0) <= MAX_PROBE_FACTOR * max(idle, 1.0),
    ]
    return all(checks)


def criteria(table: dict, cases: list[dict], idle: float | None) -> dict:
    out: dict = {}
    infer = table.get("infer", {})
    for label, row in infer.items():
        out.setdefault("C1_C2_infer", {})[label] = {
            "loads": row["ok"] == row["n"],
            "comfortable": comfortable(row, idle),
            "tokens_per_s": row.get("tokens_per_s"),
            "swap_growth_mib": row.get("swap_growth_mib"),
            "probe_p95_ms": row.get("probe_p95_ms"),
        }
    train = table.get("train", {})
    if train:
        naive, session = train.get("naive", {}), train.get("session", {})
        out["C4_train"] = {
            "naive_completes": naive.get("ok") == naive.get("n"),
            "session_completes": session.get("ok") == session.get("n"),
            "overhead": (
                session["steady_step_s"] / naive["steady_step_s"]
                if naive.get("steady_step_s") and session.get("steady_step_s")
                else None
            ),
        }
    alt = table.get("alternate", {})
    if alt:
        base = {k: alt.get(k, {}).get("total_s") for k in ("resident", "reload")}
        out["C5_alternate"] = {
            arm: {
                "total_s": alt.get(arm, {}).get("total_s"),
                "vs_resident": _ratio(alt.get(arm, {}).get("total_s"), base["resident"]),
                "vs_reload": _ratio(alt.get(arm, {}).get("total_s"), base["reload"]),
                "outputs_equal": alt.get(arm, {}).get("outputs_equal"),
            }
            for arm in ("beta", "beta_cache_off")
        }
    rec = table.get("reclaim", {})
    if rec:
        out["C6_reclaim"] = {label: row.get("reclaimed_frac") for label, row in rec.items()}
    f4 = table.get("f4cost", {})
    if f4:
        out["C7_f4_cost"] = _ratio(
            f4.get("off", {}).get("steady_step_s"), f4.get("on", {}).get("steady_step_s")
        )
    return out


def _ratio(a, b):
    return None if a is None or not b else a / b


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / "results.json"
    summary = summarize(json.loads(path.read_text()))
    (OUT / f"summary_{path.stem}.json").write_text(json.dumps(summary, indent=2) + "\n")
    for part, rows in summary["table"].items():
        print(f"\n== {part}")
        for label, row in rows.items():
            keep = {
                k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in row.items()
                if k not in ("errors",)
            }
            print(f"  {label}: {keep}")
            for e in row["errors"]:
                print(f"      error: {e}")
    print("\nidle probe p95 ms:", summary["idle_probe_p95_ms"])
    print(json.dumps(summary["criteria"], indent=2))


if __name__ == "__main__":
    main()
