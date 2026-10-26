"""E062 (docs/research/0326): ordinary programs at half the memory they need under
``memopro run`` on this Mac, before and after a pager change.

Per workload: plain (no ceiling) twice, then ``memopro run --budget C`` twice with C = half of
the plain peak footprint. Each run is a fresh process under ``/usr/bin/time -l`` (E047's
``once``), with a 15-minute limit; the pager's counters come from ``--report-json``.

    .venv/bin/python -m experiments.e062_pager.run <label>    # docs/research/data/e062/<label>.json
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

from experiments.e047_enable import run as e047

ROOT = e047.ROOT
OUT = ROOT / "docs" / "research" / "data" / "e062"
PY = e047.PY
WORKLOADS = {
    "image": ["experiments/e027_transparent/workload.py", "1024"],
    "dataframe": ["experiments/e040_workloads/dataframe.py"],
    # classify (scikit-learn) is not installed in this venv: not run here (0326)
    "simulate": ["experiments/e040_workloads/simulate.py"],
}
SKIP = {"seconds", "build_seconds", "import_rss_bytes", "maxrss_bytes"}
REPS = 2
LIMIT_S = 15 * 60


def once(cmd: list[str]) -> dict:
    run = e047.subprocess.run

    def limited(*a, **k):  # E047's once, with a time limit
        try:
            return run(*a, **k, timeout=LIMIT_S)
        except subprocess.TimeoutExpired as e:
            return subprocess.CompletedProcess(a[0], -9, e.stdout or "", e.stderr or "")

    e047.subprocess.run = limited
    try:
        return e047.once(cmd)
    finally:
        e047.subprocess.run = run


def same(result: dict | None) -> dict:
    return {k: v for k, v in (result or {}).items() if k not in SKIP}


def main() -> None:
    label = sys.argv[1]
    only = sys.argv[2:] or list(WORKLOADS)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{label}.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    results["commit"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False, cwd=ROOT
    ).stdout.strip()
    for name in only:
        args = WORKLOADS[name]
        plain = [once([PY, *args]) for _ in range(REPS)]
        need = statistics.median(r["peak_footprint"] for r in plain if r["peak_footprint"])
        ceiling = int(need // 2)
        runs = []
        for _ in range(REPS):
            with tempfile.TemporaryDirectory() as d:
                report = Path(d) / "report.json"
                cmd = [e047.MEMOPRO, "run", "--budget", str(ceiling), "--no-torch",
                       "--report-json", str(report), *args]
                r = once(cmd)
                r["report"] = json.loads(report.read_text()) if report.exists() else None
            r["same_as_plain"] = same(r["result"]) == same(plain[0]["result"]) and bool(r["result"])
            runs.append(r)
            t = (r["report"] or {}).get("transparent") or {}
            print(name, r["returncode"], round(r["wall_seconds"], 1), r["peak_footprint"],
                  r["swap_delta"], r["same_as_plain"], "restores", t.get("restores"),
                  round(t.get("restore_seconds", 0), 2), "evict", t.get("evictions"),
                  round(t.get("compress_seconds", 0), 2), "over", t.get("overruns"), flush=True)
        results[name] = {"plain": plain, "ceiling": ceiling, "runs": runs}
        path.write_text(json.dumps(results, indent=1, default=str))
    print("written", path)


if __name__ == "__main__":
    main()
