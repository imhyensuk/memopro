"""E040 (docs/research/0181): three more ordinary programs (pandas, scikit-learn, a NumPy
simulation) unchanged at half the memory they need, with memopro's transparent paging, on a
Linux CI runner (`e040.yml`) — E027's cases and checks for each workload.

    python experiments/e040_workloads/run.py            # everything; results in $E040_OUT

Per workload: 1 plain, no limit; 2 plain at L = peak/2 without swap (must not finish);
3 `memopro run --transparent B` at L without swap, B = L - (RSS after imports + 64 MiB);
4 plain at L with swap (report).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from experiments.e027_transparent import run as e027

OUT = Path(os.environ.get("E040_OUT", str(HERE.parents[1] / "docs" / "research" / "data" / "e040")))
e027.OUT = OUT
MIB = 1 << 20
MARGIN = 64 * MIB
WORKLOADS = ("dataframe", "classify", "simulate")
SKIP = {"seconds", "import_rss_bytes", "maxrss_bytes"}


def workload(name: str) -> dict:
    py, script = sys.executable, str(HERE / f"{name}.py")
    cases = {"1_plain": e027.case(f"{name}_1_plain", [py, script], None, False)}
    plain = cases["1_plain"].get("result") or {}
    peak, base = plain.get("maxrss_bytes", 0), plain.get("import_rss_bytes", 0)
    limit = (peak // 2) // MIB * MIB
    budget = limit - base - MARGIN
    cases["2_plain_limited"] = e027.case(f"{name}_2_plain_limited", [py, script], limit, False)
    report = OUT / "cases" / f"{name}_3_memopro_report.json"
    cmd = [
        py,
        "-m",
        "memopro",
        "run",
        "--transparent",
        f"{budget}B",
        "--report-json",
        str(report),
        script,
    ]
    cases["3_memopro"] = e027.case(f"{name}_3_memopro", cmd, limit, False)
    cases["4_plain_swap"] = e027.case(f"{name}_4_plain_swap", [py, script], limit, True)
    got = cases["3_memopro"].get("result") or {}
    t1 = (
        bool(plain)
        and bool(got)
        and all(got.get(k) == v for k, v in plain.items() if k not in SKIP)
    )
    c2 = cases["2_plain_limited"]
    t2 = c2["returncode"] != 0 and c2["result"] is None
    rep = (json.loads(report.read_text()).get("transparent") or {}) if report.exists() else {}
    c3 = cases["3_memopro"]
    t3 = (
        c3["returncode"] == 0
        and bool(rep)
        and rep.get("overruns") == 0
        and rep.get("peak_used", 1 << 62) <= rep.get("limit", 0)
    )
    return {
        "name": name,
        "peak": peak,
        "import_rss": base,
        "limit": limit,
        "budget": budget,
        "t1": t1,
        "t2": t2,
        "t3": t3,
        "pager": rep,
        "seconds": {k: (c.get("result") or {}).get("seconds") for k, c in cases.items()},
        "returncodes": {k: c["returncode"] for k, c in cases.items()},
    }


def diagnose(name: str) -> dict:
    """After the failure (0184): the same memopro case without a limit, to read the pager's report
    and the process peak (memory the pager does not hold = process peak - pager peak)."""
    py, script = sys.executable, str(HERE / f"{name}.py")
    plain = e027.case(f"diag_{name}_1_plain", [py, script], None, False).get("result") or {}
    limit = (plain.get("maxrss_bytes", 0) // 2) // MIB * MIB
    budget = limit - plain.get("import_rss_bytes", 0) - MARGIN
    report = OUT / "cases" / f"diag_{name}_report.json"
    cmd = [
        py,
        "-m",
        "memopro",
        "run",
        "--transparent",
        f"{budget}B",
        "--report-json",
        str(report),
        script,
    ]
    got = e027.case(f"diag_{name}_3_unlimited", cmd, None, False)
    rep = (json.loads(report.read_text()).get("transparent") or {}) if report.exists() else {}
    res = got.get("result") or {}
    return {
        "name": name,
        "limit": limit,
        "budget": budget,
        "plain_peak": plain.get("maxrss_bytes"),
        "memopro_peak": res.get("maxrss_bytes"),
        "pager": rep,
        "same": all(res.get(k) == v for k, v in plain.items() if k not in SKIP),
    }


def summarize(results: list[dict]) -> str:
    rows = [
        "# E040 summary (0181)",
        "",
        "| workload | T1 same result | T2 plain fails at L | T3 memopro finishes at L | pass |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        ok = r["t1"] and r["t2"] and r["t3"]
        rows.append(
            f"| {r['name']} | {r['t1']} | {r['t2']} | {r['t3']} | **{'pass' if ok else 'fail'}** |"
        )
    rows += [
        "",
        (
            "| workload | peak MiB | after imports MiB | L MiB | budget MiB | plain s | "
            "memopro s | swap s | slowdown | compressed MiB in -> out | overruns |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        s, p = r["seconds"], r["pager"]
        slow = (s["3_memopro"] / s["1_plain"]) if s["3_memopro"] and s["1_plain"] else None
        rows.append(
            f"| {r['name']} | {r['peak'] / MIB:.0f} | {r['import_rss'] / MIB:.0f} | "
            f"{r['limit'] / MIB:.0f} | {r['budget'] / MIB:.0f} | {s['1_plain']} | {s['3_memopro']} | "
            f"{s['4_plain_swap']} | {slow and round(slow, 2)} | "
            f"{p.get('compress_in', 0) / MIB:.0f} -> {p.get('compress_out', 0) / MIB:.0f} | "
            f"{p.get('overruns')} |"
        )
    return "\n".join(rows) + "\n"


def main() -> None:
    (OUT / "cases").mkdir(parents=True, exist_ok=True)
    env = e027.environment()
    for mod in ("pandas", "sklearn", "scipy"):
        env[mod] = __import__(mod).__version__
    (OUT / "env.json").write_text(json.dumps(env, indent=1))
    if os.environ.get("E040_DIAG"):
        diag = [diagnose(w) for w in WORKLOADS]
        (OUT / "diagnosis.json").write_text(json.dumps(diag, indent=1))
        print(json.dumps(diag, indent=1))
        return
    results = [workload(w) for w in WORKLOADS]
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    text = summarize(results)
    (OUT / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
