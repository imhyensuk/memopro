"""E041c (docs/research/0192): E041b's G7 alone (Qwen2.5-7B, `memopro.generate` with the 1.5B
int4 draft at a 3 GiB budget), same code path, measurement and criteria, with nothing else
running on the machine.

    .venv/bin/python -m experiments.e041c_g7.run --all

Results: docs/research/data/e041c/ (cases/G7.json, summary.md, env.json).
"""

from __future__ import annotations

import sys

from experiments.e041b_budget_all import run as base

base.OUT = base.ROOT / "docs" / "research" / "data" / "e041c"
base.CASES = [c for c in base.CASES if c[0] == "G7"]
MIB = base.MIB


def summarize() -> str:
    """E041b's K1 and K3 for G7 only."""
    r = base.json.loads((base.OUT / "cases" / "G7.json").read_text())
    same = [a == b or (len(a) < len(b) and b.startswith(a))
            for a, b in zip(r.get("texts", []), r.get("reference", []), strict=False)]
    k1 = len(same) == len(base.PROMPTS) and all(same)
    k3, detail = False, (r.get("error") or r.get("refused") or "missing")[-200:]
    if "rt" in r:
        growth = r["peak_footprint"] - r["base_footprint"]
        swap = r["swap_after"] - r["swap_before"]
        k3 = growth <= r["budget"] + 256 * MIB and swap <= 64 * MIB and r["rt"]["peak_used"] <= r["budget"]
        detail = (f"+{growth / MIB:.0f} MiB (budget {r['budget'] // MIB}, held {r['held'] / MIB:.0f}), "
                  f"swap {swap / MIB:+.0f} MiB, attempt {r.get('attempt')}")
    rows = ["# E041c summary (0192)", "", f"Gate (G7): **{'pass' if k1 and k3 else 'fail'}**", "",
            "| check | result | detail |", "|---|---|---|",
            f"| K1 G7 text equals plain greedy | {'pass' if k1 else 'fail'} | {sum(same)}/{len(base.PROMPTS)} |",
            f"| K3 footprint <= budget + 256 MiB, swap <= 64 MiB | {'pass' if k3 else 'fail'} | {detail} |"]
    if "seconds" in r:
        rows += ["", f"{sum(r['seconds']) / (base.NEW * len(base.PROMPTS)):.2f} s/token"]
    return "\n".join(rows) + "\n"


base.summarize = summarize

if __name__ == "__main__":
    if "--case" in sys.argv:
        i = sys.argv.index("--case")
        kind, key, budget, seq = sys.argv[i + 1 : i + 5]
        print(base.json.dumps(base.case("x", kind, key, int(budget), int(seq))))
    else:
        base.run_all()  # each case runs in a fresh process through E041b's own --case
