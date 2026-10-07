"""E047 (docs/research/0231): memopro.enable / memopro run on the 8 GB M1, pre-registered.

Each case runs in a fresh process under ``/usr/bin/time -l`` (peak memory footprint) with
``MallocLargeCache=0``; machine swap use is read before and after (``sysctl vm.swapusage``).

W1: the E027 image-stack program, unchanged (1,024 images), plain and under ``memopro run
--budget C`` for C = 4GB! (ample), 0.75 and 0.5 of the plain peak footprint.
W2: ``passes.py`` (1 GiB, 4 passes), plain and with ``--ceiling C`` for the same C.
Three repetitions per case; a case whose swap grew more than 64 MiB is run once more and the
second attempt counts (both are kept).

    .venv/bin/python -m experiments.e047_enable.run       # results in docs/research/data/e047/
"""

from __future__ import annotations

import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "research" / "data" / "e047"
PY = sys.executable
MEMOPRO = str(Path(PY).with_name("memopro"))
W1 = ["experiments/e027_transparent/workload.py", "1024"]
W2 = ["-m", "experiments.e047_enable.passes", "--mib", "1024", "--passes", "4"]
REPS = 3
SWAP_LIMIT = 64 << 20
W1_KEYS = ("means_sha256", "counts_sha256", "total", "max", "stack_sha256")


def swap_used() -> int:
    s = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, check=False).stdout
    m = re.search(r"used = ([\d.]+)M", s)
    return int(float(m.group(1)) * (1 << 20)) if m else -1


def last_json(text: str) -> dict | None:
    for line in reversed(text.strip().splitlines()):
        try:
            return json.loads(line)
        except ValueError:
            continue
    return None


def once(cmd: list[str]) -> dict:
    env = {**os.environ, "MallocLargeCache": "0", "PYTHONPATH": str(ROOT)}
    s0 = swap_used()
    t = time.perf_counter()
    p = subprocess.run(["/usr/bin/time", "-l", *cmd], capture_output=True, text=True, cwd=ROOT,
                       env=env, check=False)
    wall = time.perf_counter() - t
    s1 = swap_used()
    m = re.search(r"(\d+)\s+peak memory footprint", p.stderr)
    return {
        "cmd": cmd,
        "returncode": p.returncode,
        "wall_seconds": wall,
        "peak_footprint": int(m.group(1)) if m else None,
        "swap_delta": s1 - s0,
        "result": last_json(p.stdout),
        "stderr_tail": p.stderr[-3000:],
    }


def case(name: str, cmd: list[str]) -> list[dict]:
    runs = []
    for _ in range(REPS):
        r = once(cmd)
        if r["swap_delta"] > SWAP_LIMIT:
            r["contaminated"] = True
            runs.append(r)
            r = once(cmd)
        runs.append(r)
        print(name, r["returncode"], round(r["wall_seconds"], 2), r["peak_footprint"],
              r["swap_delta"], flush=True)
    return runs


def counted(runs: list[dict]) -> list[dict]:
    return [r for r in runs if not r.get("contaminated")]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results: dict = {"env": {
        "platform": platform.platform(), "python": sys.version, "machine": platform.machine(),
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False,
                                 cwd=ROOT).stdout.strip(),
    }}
    for wname, base in (("W1", [PY, *W1]), ("W2", [PY, *W2])):
        plain = case(f"{wname} plain", base)
        need = statistics.median(r["peak_footprint"] for r in counted(plain))
        results[f"{wname}_plain"] = plain
        results[f"{wname}_need"] = need
        for label, ceiling in (("ample", "4GB!"), ("0.75", str(int(0.75 * need))),
                               ("0.5", str(int(0.5 * need)))):
            if wname == "W1":
                cmd = [MEMOPRO, "run", "--budget", ceiling, "--no-torch", *W1]
            else:
                cmd = [PY, *W2, "--ceiling", ceiling]
            results[f"{wname}_{label}"] = {"ceiling": ceiling, "runs": case(f"{wname} {label}", cmd)}
        (OUT / "results.json").write_text(json.dumps(results, indent=1, default=str))
    (OUT / "results.json").write_text(json.dumps(results, indent=1, default=str))
    print("written", OUT / "results.json")


if __name__ == "__main__":
    main()
