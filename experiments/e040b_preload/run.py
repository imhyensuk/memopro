"""E040b (docs/research/0190): E040's programs (and E027's image stack) unchanged at half their
memory with every large allocation in memopro's pager (`memopro-preload`, 0189) and the whole
process held to the limit — on a Linux CI runner (`e040b.yml`).

    python experiments/e040b_preload/run.py            # results in $E040B_OUT

Per workload: 1 plain, no limit; 2 plain at L = peak/2 without swap (must not finish);
3 plain under `LD_PRELOAD=libmemopro_preload.so` at L without swap, pager budget L, process
budget L - 64 MiB; 4 plain at L with swap (report). Every case stops after 15 minutes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from experiments.e027_transparent import run as e027

OUT = Path(os.environ.get("E040B_OUT", str(ROOT / "docs" / "research" / "data" / "e040b")))
LIB = os.environ.get("E040B_LIB", str(ROOT / "target" / "release" / "libmemopro_preload.so"))
MIB = 1 << 20
SLACK = 64 * MIB  # what the cgroup counts beyond the process's resident set (page tables etc.)
TIMEOUT = 900
WORKLOADS = {
    "image": ROOT / "experiments" / "e027_transparent" / "workload.py",
    "dataframe": ROOT / "experiments" / "e040_workloads" / "dataframe.py",
    "classify": ROOT / "experiments" / "e040_workloads" / "classify.py",
    "simulate": ROOT / "experiments" / "e040_workloads" / "simulate.py",
}
SKIP = {"seconds", "build_seconds", "import_rss_bytes", "maxrss_bytes"}


def case(name: str, cmd: list[str], limit: int | None, swap: bool) -> dict:
    args = ["sudo", "-E", "systemd-run", "--scope", "--quiet", f"--uid={os.getuid()}",
            f"--gid={os.getgid()}", "-p", "MemoryAccounting=yes"]
    if limit is not None:
        args += ["-p", f"MemoryMax={limit}"] + ([] if swap else ["-p", "MemorySwapMax=0"])
    t = time.perf_counter()
    try:
        p = subprocess.run(args + cmd, capture_output=True, text=True, cwd=ROOT,
                           timeout=TIMEOUT, check=False)
        code, out, err = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        subprocess.run(["sudo", "pkill", "-f", cmd[-1]], check=False)
        code, out, err = "timeout", e.stdout or "", e.stderr or ""
        out = out.decode() if isinstance(out, bytes) else out
        err = err.decode() if isinstance(err, bytes) else err
    rec = {"case": name, "limit": limit, "swap": swap, "returncode": code,
           "wall_seconds": time.perf_counter() - t, "result": e027.last_json(out),
           "stderr_tail": err[-4000:]}
    (OUT / "cases" / f"{name}.json").write_text(json.dumps(rec, indent=1))
    print(f"{name}: returncode {code}, {rec['wall_seconds']:.1f} s", flush=True)
    return rec


def workload(name: str, script: Path) -> dict:
    py = sys.executable
    plain = case(f"{name}_1_plain", [py, str(script)], None, False)
    ref = plain.get("result") or {}
    limit = (ref.get("maxrss_bytes", 0) // 2) // MIB * MIB
    limited = case(f"{name}_2_plain_limited", [py, str(script)], limit, False)
    report = OUT / "cases" / f"{name}_3_preload_report.json"
    env = ["env", f"LD_PRELOAD={LIB}", f"MEMOPRO_PRELOAD_BUDGET={limit}",
           f"MEMOPRO_PRELOAD_PROCESS={limit - SLACK}", f"MEMOPRO_PRELOAD_REPORT={report}"]
    paged = case(f"{name}_3_preload", env + [py, str(script)], limit, False)
    swap = case(f"{name}_4_plain_swap", [py, str(script)], limit, True)
    got = paged.get("result") or {}
    rep = json.loads(report.read_text()) if report.exists() else {}
    t1 = bool(ref) and bool(got) and all(got.get(k) == v for k, v in ref.items() if k not in SKIP)
    t2 = limited["returncode"] != 0 and limited["result"] is None
    t3 = paged["returncode"] == 0 and bool(rep) and rep.get("overruns") == 0
    return {"name": name, "peak": ref.get("maxrss_bytes"), "limit": limit, "t1": t1, "t2": t2,
            "t3": t3, "pager": rep, "maxrss_preload": got.get("maxrss_bytes"),
            "seconds": {"plain": ref.get("seconds"), "preload": got.get("seconds"),
                        "swap": (swap.get("result") or {}).get("seconds")},
            "returncodes": {"limited": limited["returncode"], "preload": paged["returncode"],
                            "swap": swap["returncode"]}}


def summarize(results: list[dict]) -> str:
    rows = ["# E040b summary (0190)", "",
            "| workload | T1 same result | T2 plain fails at L | T3 preload finishes at L | pass |",
            "|---|---|---|---|---|"]
    for r in results:
        ok = r["t1"] and r["t2"] and r["t3"]
        rows.append(f"| {r['name']} | {r['t1']} | {r['t2']} | {r['t3']} | **{'pass' if ok else 'fail'}** |")
    rows += ["", ("| workload | peak MiB | L MiB | plain s | preload s | swap s | slowdown | "
                  "outside pager peak MiB | lowest limit MiB | paged allocations | evictions | "
                  "overruns |"),
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        s, p = r["seconds"], r["pager"]
        slow = round(s["preload"] / s["plain"], 2) if s["preload"] and s["plain"] else None
        rows.append(
            f"| {r['name']} | {(r['peak'] or 0) / MIB:.0f} | {r['limit'] / MIB:.0f} | {s['plain']} | "
            f"{s['preload']} | {s['swap']} | {slow} | {p.get('outside_peak', 0) / MIB:.0f} | "
            f"{p.get('limit_low', 0) / MIB:.0f} | {p.get('paged_allocations')} | "
            f"{p.get('evictions')} | {p.get('overruns')} |")
    return "\n".join(rows) + "\n"


def main() -> None:
    (OUT / "cases").mkdir(parents=True, exist_ok=True)
    env = e027.environment()
    for mod in ("pandas", "sklearn", "scipy"):
        env[mod] = __import__(mod).__version__
    env["preload"] = LIB
    (OUT / "env.json").write_text(json.dumps(env, indent=1))
    only = os.environ.get("E040B_ONLY")
    results = [workload(n, s) for n, s in WORKLOADS.items() if not only or n in only.split(",")]
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    text = summarize(results)
    (OUT / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
