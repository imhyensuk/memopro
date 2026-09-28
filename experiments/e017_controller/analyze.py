"""E017 analysis: medians per model and arm, and the verdicts pre-registered in 0073 (K1-K4, R).

Usage: .venv/bin/python -m experiments.e017_controller.analyze [results.json]
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "docs" / "research" / "data" / "e017"
MiB = 2**20
E014_PROBE_MS = 24.1


def med(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def row(cases: list[dict]) -> dict:
    ok = [c for c in cases if all(c.get(p, {}).get("ok") for p in "ABC")]
    g = lambda f: med([f(c) for c in ok])
    return {
        "n": len(cases),
        "ok": len(ok),
        "A_tok_s": g(lambda c: c["A"]["tokens_per_s"]),
        "B_tok_s": g(lambda c: c["B"]["tokens_per_s"]),
        "C_tok_s": g(lambda c: c["C"]["tokens_per_s"]),
        "A_load_s": g(lambda c: c["A"]["load_s"]),
        "A_peak_mib": g(lambda c: c["A_mem"]["peak_footprint"] / MiB),
        "B_initial_swap_mib": g(lambda c: c["B_swap"]["initial"] / MiB),
        "B_steady_swap_mib": g(lambda c: c["B_swap"]["steady"] / MiB),
        "B_initial_probe_p95": g(lambda c: c["B_initial_probe"].get("p95_ms")),
        "B_steady_probe_p95": g(lambda c: c["B_probe"].get("p95_ms")),
        "A_probe_p95": g(lambda c: c["A_probe"].get("p95_ms")),
        "C_active": [c["C"]["active_end"] for c in ok],
        "A_actions": [sum(bool(l["action"]) for l in c["A"].get("log", [])) for c in ok],
        "actions": [
            [
                (p, l["token"], l["action"])
                for p in "ABC"
                for l in c[p].get("log", [])
                if l["action"]
            ]
            for c in ok
        ],
        "A_tokens": [c["A"]["tokens"] for c in ok],
        "start": [c["A"].get("start") for c in ok],
        "A_active_end": [c["A"]["active_end"] for c in ok],
    }


def verdicts(rows: dict) -> dict:
    out = {}
    for model, r in rows.items():
        get = lambda arm, key, r=r: (r.get(arm) or {}).get(key)
        k1 = {
            "steady swap bf16_ctl <= bf16": _le(
                get("bf16_ctl", "B_steady_swap_mib"), get("bf16", "B_steady_swap_mib")
            ),
            "steady probe bf16_ctl <= max(24.1, bf16)": _le(
                get("bf16_ctl", "B_steady_probe_p95"),
                max(E014_PROBE_MS, get("bf16", "B_steady_probe_p95") or 0),
            ),
        }
        k2 = {"B tok/s bf16_ctl >= bf16": _ge(get("bf16_ctl", "B_tok_s"), get("bf16", "B_tok_s"))}
        if "3B" in model:
            k2["A tok/s bf16_ctl >= 3 x bf16"] = _ge(
                get("bf16_ctl", "A_tok_s"), 3 * (get("bf16", "A_tok_s") or 0)
            )
        k3 = None
        if "1.5B" in model:
            actives = get("bf16_ctl", "C_active") or []
            k3 = {
                "back to bf16 in >= 2 of 3": sum(a == "bf16" for a in actives) >= 2,
                "C >= 0.8 x A": _ge(
                    get("bf16_ctl", "C_tok_s"), 0.8 * (get("bf16_ctl", "A_tok_s") or 0)
                ),
            }
        k4 = {}
        pairs = [("int4_ctl", "int4")] + ([("bf16_ctl", "bf16")] if "1.5B" in model else [])
        for ctl, base in pairs:
            idle = [i for i, n in enumerate(get(ctl, "A_actions") or []) if n == 0]
            base_tokens = get(base, "A_tokens") or []
            ctl_tokens = get(ctl, "A_tokens") or []
            same = (
                bool(idle)
                and bool(base_tokens)
                and all(ctl_tokens[i] == base_tokens[0] for i in idle)
            )
            k4[f"{ctl} == {base} when idle"] = same
            k4[f"{ctl} speed >= 0.9 x {base}"] = _ge(
                get(ctl, "A_tok_s"), 0.9 * (get(base, "A_tok_s") or 0)
            )
        out[model] = {"K1": k1, "K2": k2, "K3": k3, "K4": k4}
    return out


def verdicts_b(rows: dict) -> dict:
    """E017b (0075): the fixed controller. 1.5B bf16 vs bf16_ctl (K0-K4), 3B bf16_ctl (S)."""
    out = {}
    for model, r in rows.items():
        get = lambda arm, key, r=r: (r.get(arm) or {}).get(key)
        if "3B" in model:
            starts = get("bf16_ctl", "start") or []
            out[model] = {
                "S": {
                    "starts on int4 in >= 2 of 3": sum(s == "int4" for s in starts) >= 2,
                    "A tok/s >= 3 x 0.27 (E017 bf16)": _ge(get("bf16_ctl", "A_tok_s"), 3 * 0.27),
                }
            }
            continue
        base = verdicts({model: r})[model]
        idle = [n == 0 for n in get("bf16_ctl", "A_actions") or []]
        ends = get("bf16_ctl", "A_active_end") or []
        k0 = {
            "no A action and A ends on bf16 in >= 2 of 3": sum(
                i and e == "bf16" for i, e in zip(idle, ends, strict=False)
            )
            >= 2
        }
        k4 = {k: v for k, v in base["K4"].items() if k.startswith("bf16_ctl")}
        out[model] = {"K0": k0, "K1": base["K1"], "K2": base["K2"], "K3": base["K3"], "K4": k4}
    return out


def _le(a, b):
    return None if a is None or b is None else a <= b


def _ge(a, b):
    return None if a is None or b is None else a >= b


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / "results.json"
    d = json.loads(path.read_text())
    table: dict = {}
    for c in d["cases"]:
        if "skipped" not in c:
            table.setdefault(c["model"], {}).setdefault(c["arm"], []).append(c)
    rows = {m: {a: row(cs) for a, cs in arms.items()} for m, arms in table.items()}
    judge = verdicts_b if path.parent.name == "e017b" else verdicts
    summary = {"idle_probe": d["idle_probe"], "rows": rows, "verdicts": judge(rows)}
    (path.parent / f"summary_{path.stem}.json").write_text(json.dumps(summary, indent=2) + "\n")
    for model, arms in rows.items():
        print(f"\n== {model}")
        for arm, r in arms.items():
            keep = {
                k: (round(v, 2) if isinstance(v, float) else v)
                for k, v in r.items()
                if k not in ("A_tokens",)
            }
            print(f"  {arm}: {keep}")
    print(json.dumps(summary["verdicts"], indent=2))


if __name__ == "__main__":
    main()
