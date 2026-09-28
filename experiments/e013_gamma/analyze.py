"""E013a analysis: sensitivity and specificity of candidate pressure signals (0087 §3).

Ground truth is what a user feels: a second is *harmful* when the responsiveness probe's slowest
read in it exceeded 24.1 ms (E014's comfort threshold). Harmful seconds form episodes (gaps of up
to 2 s merged); a signal detects an episode if it is on anywhere from 2 s before its start to 2 s
after its end. A second is *calm* when no harmful second lies within 5 s of it.

Usage: .venv/bin/python -m experiments.e013_gamma.analyze [docs/research/data/e013/signal.json]
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "docs" / "research" / "data" / "e013"
HARM_MS = 24.1
MERGE, SLACK, CALM = 2, 2, 5
SWAP_STEP = 64e6  # E017c Q1: swap growth over >= 1 s
LOW = 0.10
MIN_EPISODES = 5
SENS, SPEC = 0.8, 0.9


def signals(rows: list[dict]) -> dict[str, list[bool]]:
    out = {
        "macos_level>=warning": [(r["raw_level"] or 0) >= 2 for r in rows],
        "swap_growth>=64MB/1s": [False]
        + [b["swap_used"] - a["swap_used"] >= SWAP_STEP for a, b in itertools.pairwise(rows)],
        "conservative_available<10%": [r["available"] < LOW * r["total"] for r in rows],
        "os_available<10%": [r["kernel_available"] < LOW * r["total"] for r in rows],
    }
    out["swap_or_level"] = [a or b for a, b in zip(out["swap_growth>=64MB/1s"], out["macos_level>=warning"], strict=True)]
    return out


def episodes(harm: list[bool]) -> list[tuple[int, int]]:
    eps: list[list[int]] = []
    for i, h in enumerate(harm):
        if not h:
            continue
        if eps and i - eps[-1][1] <= MERGE + 1:
            eps[-1][1] = i
        else:
            eps.append([i, i])
    return [tuple(e) for e in eps]


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / "signal.json"
    data = json.loads(path.read_text())
    rows = data["rows"]
    harm = [(r["probe_max_ms"] or 0) > HARM_MS for r in rows]
    eps = episodes(harm)
    calm = [not any(harm[max(0, i - CALM) : i + CALM + 1]) for i in range(len(rows))]
    result = {
        "seconds": len(rows),
        "harmful_seconds": sum(harm),
        "episodes": len(eps),
        "calm_seconds": sum(calm),
        "ssd_written_bytes": data.get("ssd_written_bytes"),
        "signals": {},
    }
    phase_names = sorted({r["phase"] for r in rows}, key=lambda p: [r["phase"] for r in rows].index(p))
    for name, on in signals(rows).items():
        detected = sum(any(on[max(0, s - SLACK) : e + SLACK + 1]) for s, e in eps)
        spec_n = sum(1 for c, o in zip(calm, on, strict=True) if c and not o)
        sens = detected / len(eps) if len(eps) >= MIN_EPISODES else None
        spec = spec_n / sum(calm) if sum(calm) else None
        delays = {}
        for ph in phase_names:
            if ":P" not in ph:
                continue
            idx = [i for i, r in enumerate(rows) if r["phase"] == ph]
            first = next((j for j, i in enumerate(idx) if on[i]), None)
            delays[ph] = first
        calm_phase = [i for i, r in enumerate(rows) if "calm" in r["phase"]]
        result["signals"][name] = {
            "sensitivity": sens,
            "specificity": spec,
            "usable": None if sens is None or spec is None else sens >= SENS and spec >= SPEC,
            "on_share_in_calm_phases": sum(on[i] for i in calm_phase) / len(calm_phase)
            if calm_phase
            else None,
            "first_on_s_after_pressure_start": delays,
        }
    by_phase = {}
    for ph in phase_names:
        rs = [r for r in rows if r["phase"] == ph]
        by_phase[ph] = {
            "n": len(rs),
            "harmful": sum((r["probe_max_ms"] or 0) > HARM_MS for r in rs),
            "level_counts": {str(k): sum(r["raw_level"] == k for r in rs) for k in (1, 2, 4)},
            "swap_growth_mb": round((rs[-1]["swap_used"] - rs[0]["swap_used"]) / 1e6),
            "min_available_gb": round(min(r["available"] for r in rs) / 1e9, 2),
        }
    result["by_phase"] = by_phase
    (path.parent / f"summary_{path.stem}.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "by_phase"}, indent=1))
    for ph, v in by_phase.items():
        print(ph, v)


if __name__ == "__main__":
    main()
