"""E027 (docs/research/0122): gate G-R3 on a Linux CI runner (GitHub Actions, `e027.yml`).

An unchanged NumPy program (`workload.py`) at half the memory it needs, with memopro's
transparent paging (userfaultfd), against the same program alone and against OS swap. Each case
runs in its own cgroup scope (`sudo -E systemd-run --scope -p MemoryMax=...`).

    python experiments/e027_transparent/run.py            # everything; results in $E027_OUT

Cases: 1 plain, no limit; 2 plain at L = peak/2 without swap (must not finish); 3 `memopro run
--transparent` at L without swap; 4 plain at L with swap (report); 5 as 4 with zswap (report).
Also the C ABI smoke test (C1) and the Linux pager tests (U1).
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = Path(os.environ.get("E027_OUT", str(ROOT / "docs" / "research" / "data" / "e027")))
WORKLOAD = HERE / "workload.py"
MIB = 1 << 20
BASE_ALLOWANCE = 192 * MIB  # Python + NumPy outside the pager (0122)
RESULT_KEYS = ("means_sha256", "counts_sha256", "total", "max", "stack_sha256")


def sh(cmd: list[str] | str, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, capture_output=True, text=True, check=False, shell=isinstance(cmd, str), **kw
    )


def scoped(cmd: list[str], limit: int | None, swap: bool) -> subprocess.CompletedProcess:
    args = ["sudo", "-E", "systemd-run", "--scope", "--quiet"]
    args += [f"--uid={os.getuid()}", f"--gid={os.getgid()}", "-p", "MemoryAccounting=yes"]
    if limit is not None:
        args += ["-p", f"MemoryMax={limit}"]
        if not swap:
            args += ["-p", "MemorySwapMax=0"]
    return sh(args + cmd, cwd=ROOT)


def last_json(text: str) -> dict | None:
    for line in reversed(text.strip().splitlines()):
        try:
            return json.loads(line)
        except ValueError:
            continue
    return None


def case(name: str, cmd: list[str], limit: int | None, swap: bool) -> dict:
    t = time.perf_counter()
    proc = scoped(cmd, limit, swap)
    rec = {
        "case": name,
        "limit": limit,
        "swap": swap,
        "returncode": proc.returncode,
        "wall_seconds": time.perf_counter() - t,
        "result": last_json(proc.stdout),
        "stderr_tail": proc.stderr[-4000:],
    }
    (OUT / "cases" / f"{name}.json").write_text(json.dumps(rec, indent=1))
    print(f"{name}: returncode {proc.returncode}, {rec['wall_seconds']:.1f} s", flush=True)
    return rec


def unit_tests() -> dict:
    rust = sh(["cargo", "test", "-p", "memopro", "rt::pager", "--", "--nocapture"], cwd=ROOT)
    py = sh([sys.executable, "-m", "pytest", "-v", "-rs", "tests/test_rt_transparent.py"], cwd=ROOT)
    text = rust.stdout + rust.stderr
    names = (
        "memory_larger_than_the_budget_reads_back_exact",
        "threads_write_and_read_while_chunks_come_and_go",
        "incompressible_chunks_stay_and_are_counted",
        "a_repeated_scan_keeps_a_budget_worth",
    )
    rust_ok = (
        rust.returncode == 0
        and "skipping the pager test" not in text
        and all(f"{n} ... ok" in text for n in names)
    )
    py_passed = py.stdout.count(" PASSED")
    py_ok = py.returncode == 0 and py_passed >= 2
    rec = {
        "rust_returncode": rust.returncode,
        "rust_ok": rust_ok,
        "rust_tail": text[-3000:],
        "python_returncode": py.returncode,
        "python_passed": py_passed,
        "python_ok": py_ok,
        "python_tail": (py.stdout + py.stderr)[-3000:],
    }
    (OUT / "cases" / "unit_tests.json").write_text(json.dumps(rec, indent=1))
    return rec


def c_smoke() -> dict:
    build = sh(["cargo", "build", "-q", "-p", "memopro-c", "--release"], cwd=ROOT)
    lib = ROOT / "target" / "release"
    exe = ROOT / "target" / "mp_smoke"
    cc = sh(
        [
            "cc",
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-o",
            str(exe),
            str(ROOT / "crates" / "memopro-c" / "tests" / "smoke.c"),
            f"-I{ROOT / 'crates' / 'memopro-c' / 'include'}",
            f"-L{lib}",
            "-lmemopro_c",
            f"-Wl,-rpath,{lib}",
        ]
    )
    run = sh([str(exe)]) if cc.returncode == 0 else None
    rec = {
        "build_returncode": build.returncode,
        "cc_returncode": cc.returncode,
        "cc_output": (cc.stdout + cc.stderr)[-2000:],
        "returncode": None if run is None else run.returncode,
        "output": None if run is None else (run.stdout + run.stderr)[-2000:],
    }
    rec["ok"] = run is not None and run.returncode == 0 and run.stdout.strip().endswith("ok")
    (OUT / "cases" / "c_smoke.json").write_text(json.dumps(rec, indent=1))
    return rec


def zswap(on: bool) -> bool:
    value = "Y" if on else "N"
    r = sh(f"echo {value} | sudo tee /sys/module/zswap/parameters/enabled")
    return r.returncode == 0


def environment() -> dict:
    import numpy

    import memopro

    return {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "memopro": memopro.__version__,
        "commit": os.environ.get("GITHUB_SHA", ""),
        "nproc": os.cpu_count(),
        "meminfo": sh("grep -E 'MemTotal|MemAvailable|SwapTotal' /proc/meminfo").stdout,
        "swaps": sh("swapon --show").stdout,
        "zswap_enabled": sh("cat /sys/module/zswap/parameters/enabled").stdout.strip(),
        "unprivileged_userfaultfd": sh("cat /proc/sys/vm/unprivileged_userfaultfd").stdout.strip(),
        "cgroup": sh("stat -fc %T /sys/fs/cgroup").stdout.strip(),
    }


def summarize(cases: dict, units: dict, smoke: dict, plain_peak: int, limit: int) -> str:
    ref = cases["1_plain"].get("result") or {}
    got = cases["3_memopro"].get("result") or {}
    t1 = bool(ref) and bool(got) and all(ref.get(k) == got.get(k) for k in RESULT_KEYS)
    c2 = cases["2_plain_limited"]
    t2 = c2["returncode"] != 0 and c2["result"] is None
    rep = {}
    rj = OUT / "cases" / "3_memopro_report.json"
    if rj.exists():
        rep = json.loads(rj.read_text()).get("transparent") or {}
    c3 = cases["3_memopro"]
    t3 = (
        c3["returncode"] == 0
        and bool(rep)
        and rep.get("overruns") == 0
        and rep.get("peak_used", 1 << 62) <= rep.get("limit", 0)
    )
    c1 = bool(smoke.get("ok"))
    u1 = bool(units.get("rust_ok")) and bool(units.get("python_ok"))
    gate = t1 and t2 and t3 and c1 and u1
    t0 = ref.get("seconds")
    tt = got.get("seconds")
    rows = [
        "# E027 summary (0122)",
        "",
        f"Gate G-R3: **{'pass' if gate else 'fail'}**",
        "",
        (
            f"Plain peak {plain_peak / MIB:.0f} MiB; limit L = {limit / MIB:.0f} MiB; "
            f"pager budget {max(0, limit - BASE_ALLOWANCE) / MIB:.0f} MiB"
        ),
        "",
        "| check | result | detail |",
        "|---|---|---|",
        f"| T1 same result | {'pass' if t1 else 'fail'} | {', '.join(f'{k}={got.get(k)!r:.20}' for k in ('total', 'max'))} |",
        f"| T2 plain does not finish at L | {'pass' if t2 else 'fail'} | returncode {c2['returncode']} |",
        f"| T3 transparent finishes at L | {'pass' if t3 else 'fail'} | returncode {c3['returncode']}, peak {rep.get('peak_used', 0) / MIB:.0f} of {rep.get('limit', 0) / MIB:.0f} MiB, overruns {rep.get('overruns')}, faults {rep.get('faults')}, compressed {rep.get('compress_in', 0) / MIB:.0f} to {rep.get('compress_out', 0) / MIB:.0f} MiB |",
        f"| C1 C ABI | {'pass' if c1 else 'fail'} | {(smoke.get('output') or '').strip().splitlines()[-1:]} |",
        f"| U1 Linux tests | {'pass' if u1 else 'fail'} | rust {units.get('rust_ok')}, python passed {units.get('python_passed')} |",
        "",
        "| case | returncode | seconds (workload) | wall s |",
        "|---|---|---|---|",
    ]
    for name, c in cases.items():
        r = c.get("result") or {}
        rows.append(
            f"| {name} | {c['returncode']} | {r.get('seconds', float('nan')):.1f} | {c['wall_seconds']:.1f} |"
        )
    if t0 and tt:
        rows += ["", f"Slowdown of case 3 over case 1: {tt / t0:.2f}x"]
    return "\n".join(rows) + "\n"


def main() -> None:
    (OUT / "cases").mkdir(parents=True, exist_ok=True)
    (OUT / "env.json").write_text(json.dumps(environment(), indent=1))
    units = unit_tests()
    smoke = c_smoke()
    py = sys.executable
    cases = {}
    cases["1_plain"] = case("1_plain", [py, str(WORKLOAD)], None, False)
    peak = (cases["1_plain"].get("result") or {}).get("maxrss_bytes", 0)
    limit = (peak // 2) // MIB * MIB
    budget = limit - BASE_ALLOWANCE
    cases["2_plain_limited"] = case("2_plain_limited", [py, str(WORKLOAD)], limit, False)
    report = OUT / "cases" / "3_memopro_report.json"
    cases["3_memopro"] = case(
        "3_memopro",
        [
            py,
            "-m",
            "memopro",
            "run",
            "--transparent",
            f"{budget}B",
            "--report-json",
            str(report),
            str(WORKLOAD),
        ],
        limit,
        False,
    )
    cases["4_plain_swap"] = case("4_plain_swap", [py, str(WORKLOAD)], limit, True)
    if zswap(True):
        cases["5_plain_zswap"] = case("5_plain_zswap", [py, str(WORKLOAD)], limit, True)
        zswap(False)
    text = summarize(cases, units, smoke, peak, limit)
    (OUT / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
