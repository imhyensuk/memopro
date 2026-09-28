"""E016 analysis: medians per model and arm, and the verdicts pre-registered in 0070 (E1, C2,
H1-H3, P).

Usage: .venv/bin/python -m experiments.e016_rcr.analyze [results.json]
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "docs" / "research" / "data" / "e016"
MiB, GiB = 2**20, 2**30
E014_PROBE_MS = 24.1


def med(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def summarize(d: dict) -> dict:
    idle = med([p.get("p95_ms") for p in d["idle_probe"]])
    probe_limit = max(2 * (idle or 0), E014_PROBE_MS)
    table: dict = {}
    for c in d["cases"]:
        if "skipped" in c:
            table.setdefault(c["model"], {}).setdefault(c["arm"], []).append(None)
            continue
        table.setdefault(c["model"], {}).setdefault(c["arm"], []).append(c)
    rows: dict = {}
    for model, arms in table.items():
        rows[model] = {}
        for arm, cases in arms.items():
            run = [c for c in cases if c is not None]
            ok_a = [c for c in run if c["A"].get("ok")]
            ok_b = [c for c in run if c.get("B", {}).get("ok")]
            rows[model][arm] = {
                "n": len(cases),
                "ok_A": len(ok_a),
                "ok_B": len(ok_b),
                "errors": sorted(
                    {
                        (c["A"].get("error") or c.get("B", {}).get("error") or "")[:120]
                        for c in run
                        if c not in ok_b
                    }
                ),
                "A_tok_s": med([c["A"]["tokens_per_s"] for c in ok_a]),
                "A_load_s": med([c["A"]["load_s"] for c in ok_a]),
                "A_ttft_s": med([c["A"]["ttft_s"] for c in ok_a]),
                "A_peak_mib": med([c["A_mem"]["peak_footprint"] / MiB for c in run]),
                "A_swap_mib": med([c["A_mem"]["swap_growth"] / MiB for c in run]),
                "A_probe_p95": med([c["A_probe"].get("p95_ms") for c in run]),
                "B_tok_s": med([c["B"]["tokens_per_s"] for c in ok_b]),
                "B_peak_mib": med([c["B_mem"]["peak_footprint"] / MiB for c in ok_b]),
                "B_probe_p95": med([c["B_probe"].get("p95_ms") for c in ok_b]),
                "B_same_tokens": all(c["B"].get("same_tokens") for c in ok_b) if ok_b else None,
                "B_resident_before": med(
                    [min(c["B"]["resident_before"]) for c in ok_b if c["B"].get("resident_before")]
                ),
                "tokens": [c["A"]["tokens"] for c in ok_a],
            }
    return {
        "idle_probe_p95": idle,
        "probe_limit_ms": probe_limit,
        "rows": rows,
        "verdicts": verdicts(rows, probe_limit),
    }


def comfortable(r: dict, probe_limit: float) -> bool:
    return bool(
        r
        and r["ok_A"] == r["n"]
        and r["n"] > 0
        and (r["A_tok_s"] or 0) >= 5
        and (r["A_load_s"] or 1e9) <= 120
        and (r["A_swap_mib"] or 0) <= 1024
        and (r["A_probe_p95"] or 0) <= probe_limit
    )


def verdicts(rows: dict, probe_limit: float) -> dict:
    out: dict = {}
    for model, r in rows.items():
        get = lambda arm, key, r=r: (r.get(arm) or {}).get(key)
        naive_t, int4_t = get("naive", "tokens") or [], get("memopro_int4", "tokens") or []
        e1 = {
            "rcr_bf16 = naive": bool(naive_t)
            and all(t == naive_t[0] for t in (get("rcr_bf16", "tokens") or []))
            and all(t == naive_t[0] for t in naive_t),
            "rcr_int4 = memopro_int4": bool(int4_t)
            and all(
                t == int4_t[0]
                for t in (get("rcr_int4", "tokens") or []) + (get("rcr_int4_nopf", "tokens") or [])
            ),
        }
        h1 = {
            "rcr_bf16 <= 0.25 naive": _le(
                get("rcr_bf16", "A_peak_mib"), get("naive", "A_peak_mib"), 0.25
            ),
            "rcr_int4 <= 0.25 memopro_int4": _le(
                get("rcr_int4", "A_peak_mib"), get("memopro_int4", "A_peak_mib"), 0.25
            ),
        }
        h2 = {
            arm: (get(arm, "A_swap_mib") or 0) <= 0.1 * 1024
            for arm in ("rcr_bf16", "rcr_int4", "rcr_int4_nopf")
            if arm in r
        }
        a, b = get("rcr_int4", "A_tok_s"), get("rcr_int4", "B_tok_s")
        m_b = get("memopro_int4", "B_tok_s")
        n_b = get("naive", "B_tok_s")
        h3 = {
            "rcr_int4 graceful": bool(a and b and b >= 0.5 * a and (m_b is None or b >= m_b)),
            "rcr_bf16 >= naive under pressure": bool(
                get("rcr_bf16", "B_tok_s") and (n_b is None or get("rcr_bf16", "B_tok_s") >= n_b)
            ),
        }
        out[model] = {
            "E1": e1,
            "C2": {arm: comfortable(rr, probe_limit) for arm, rr in r.items()},
            "H1": h1,
            "H2": h2,
            "H3": h3,
            "P": {
                arm: {"B_tok_s": get(arm, "B_tok_s"), "B_probe_p95": get(arm, "B_probe_p95")}
                for arm in ("rcr_int4", "rcr_int4_nopf")
            },
        }
    return out


def _le(a, b, factor):
    return None if a is None or b is None else a <= factor * b


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / "results.json"
    s = summarize(json.loads(path.read_text()))
    (OUT / f"summary_{path.stem}.json").write_text(json.dumps(s, indent=2) + "\n")
    for model, arms in s["rows"].items():
        print(f"\n== {model}")
        for arm, r in arms.items():
            keep = {
                k: (round(v, 2) if isinstance(v, float) else v)
                for k, v in r.items()
                if k != "tokens"
            }
            print(f"  {arm}: {keep}")
    print("\nprobe limit ms:", s["probe_limit_ms"], "idle p95:", s["idle_probe_p95"])
    print(json.dumps(s["verdicts"], indent=2))


if __name__ == "__main__":
    main()
