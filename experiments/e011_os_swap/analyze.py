"""E011 analysis: medians per condition and arm, and the verdicts pre-registered in 0058 (D0-D5).

Usage: .venv/bin/python -m experiments.e011_os_swap.analyze [results.json]
Trials re-run after the sleep interruption (redo.json, data/e011/amendment_sleep.md) take the place
of the trials they replace. Writes docs/research/data/e011/summary.json and prints a table.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "docs" / "research" / "data" / "e011"
MiB = 2**20


def metrics(t: dict) -> dict[str, float]:
    model = t["start"]["model_bytes"]
    pages = t["probe_pressed"]
    return {
        "reclaimed_frac": (t["a_loaded"]["phys_footprint"] - t["a_idle"]["phys_footprint"]) / model,
        "a_pressed_mib": t["a_pressed"]["phys_footprint"] / MiB,
        "a_resident_pressed_mib": t["a_pressed"]["resident_size"] / MiB,
        "weights_resident_pressed": (
            pages["resident_pages"] / pages["total_pages"] if pages["total_pages"] else 0.0
        ),
        "fill_s": t["fill"]["fill_s"],
        "resume_s": t["resume"]["resume_s"],
        "restore_s": t["resume"]["restore_s"],
        "second_forward_s": t["resume"]["second_forward_s"],
        "warm_forward_s": t["start"]["warm_forward_s"],
        "b_pass_s": t["b_pass"]["pass_s"],
        "resume_read_mib": t["a_resume_io"]["diskio_bytesread"] / MiB,
        "exact": float(t["resume"]["exact"]),
    }


def summarize(trials: list[dict]) -> dict:
    groups: dict[tuple[str, str], list[dict]] = {}
    errors = []
    for t in trials:
        if "error" in t:
            errors.append({k: t[k] for k in ("device", "backing", "arm", "error")})
            continue
        groups.setdefault((f"{t['device']}/{t['backing']}", t["arm"]), []).append(metrics(t))
    table: dict[str, dict[str, dict]] = {}
    for (cond, arm), rows in sorted(groups.items()):
        stats = {}
        for key in rows[0]:
            values = [r[key] for r in rows]
            stats[key] = {
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
            }
        stats["n"] = len(rows)
        table.setdefault(cond, {})[arm] = stats
    return {"table": table, "errors": errors, "verdicts": verdicts(table, trials)}


def _med(table, cond, arm, key):
    try:
        return table[cond][arm][key]["median"]
    except KeyError:
        return None


def _compare(value, base):
    if value is None or base is None:
        return "no data"
    ratio = value / base
    if ratio <= 0.8:
        return f"better ({ratio:.2f}x)"
    if ratio >= 1.25:
        return f"worse ({ratio:.2f}x)"
    return f"same ({ratio:.2f}x)"


def verdicts(table: dict, trials: list[dict]) -> dict:
    ok = [t for t in trials if "error" not in t]
    out: dict = {
        "D0_exact": all(t["resume"]["exact"] for t in ok),
        "D0_inexact": [
            (t["device"], t["backing"], t["arm"], t["repeat"])
            for t in ok
            if not t["resume"]["exact"]
        ],
    }
    for cond in ("cpu/anon", "mps/anon", "cpu/mmap"):
        if cond not in table:
            continue
        judged = cond != "cpu/mmap"
        reclaimed = _med(table, cond, "source", "reclaimed_frac")
        out[cond] = {
            "judged": judged,
            "D1_returns_memory": None if reclaimed is None else reclaimed >= 0.8,
            "D1_reclaimed_frac": reclaimed,
            "D2_fill_source_vs_os": _compare(
                _med(table, cond, "source", "fill_s"), _med(table, cond, "os", "fill_s")
            ),
            "D3_resume_source_vs_os": _compare(
                _med(table, cond, "source", "resume_s"), _med(table, cond, "os", "resume_s")
            ),
            "D3_resume_source_vs_reload": _compare(
                _med(table, cond, "source", "resume_s"), _med(table, cond, "reload", "resume_s")
            ),
            "D5_b_pass_source_vs_os": _compare(
                _med(table, cond, "source", "b_pass_s"), _med(table, cond, "os", "b_pass_s")
            ),
        }
    ratio_path = OUT / "ratio.json"
    if ratio_path.exists():
        ratio = json.loads(ratio_path.read_text())
        out["D4"] = {
            dtype: {
                "memopro_zstd1": r["memopro_zstd1"],
                "lz4_16k": r["lz4_16k"],
                "memopro_better": r["memopro_zstd1"] <= r["lz4_16k"] - 0.05,
            }
            for dtype, r in ratio.items()
        }
    return out


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / "results.json"
    data = json.loads(path.read_text())
    trials = data["trials"]
    redo_path = path.with_name("redo.json")
    replaced: list[int] = []
    if redo_path.exists():
        redo = json.loads(redo_path.read_text())["trials"]
        replaced = [t["replaces"] for t in redo]
        trials = [t for i, t in enumerate(trials) if i not in replaced] + redo
    summary = summarize(trials)
    summary["replaced_trials"] = replaced
    summary["pressure_bytes"] = data["pressure_bytes"]
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    keys = (
        "reclaimed_frac",
        "weights_resident_pressed",
        "a_pressed_mib",
        "fill_s",
        "resume_s",
        "b_pass_s",
        "resume_read_mib",
    )
    for cond, arms in summary["table"].items():
        counts = ", ".join(f"{a}={s['n']}" for a, s in arms.items())
        print(f"\n{cond}  (n per arm: {counts})")
        print(f"  {'arm':15}" + "".join(f"{k:>26}" for k in keys))
        for arm, s in arms.items():
            cells = "".join(
                f"{s[k]['median']:>10.2f} [{s[k]['min']:>6.2f},{s[k]['max']:>6.2f}]" for k in keys
            )
            print(f"  {arm:15}{cells}")
    print("\nerrors:", summary["errors"])
    print(json.dumps(summary["verdicts"], indent=2))


if __name__ == "__main__":
    main()
