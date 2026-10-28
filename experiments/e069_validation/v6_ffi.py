"""E069 V6 (docs/research/0356): the same runtime calls from Rust, C (memopro-c) and Python.

Builds the Rust example and the C program (``cargo``, ``cc``), runs the three and writes one JSON
file. ``--sdk`` passes an SDK to cc on macOS (some Command Line Tools pair a linker with an SDK
it cannot read).

    python -m experiments.e069_validation.v6_ffi OUT.json [--sdk PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GIB = 1 << 30


def per_op(reps: int, f) -> float:
    ns = []
    for _ in range(11):
        t = time.perf_counter_ns()
        for _ in range(reps):
            f()
        ns.append((time.perf_counter_ns() - t) / reps)
    return sorted(ns)[5]


def python_side() -> dict:
    from memopro import rt

    r = rt.Runtime(budget=3 * GIB, process_budget=False)
    small = r.alloc(1 << 20)

    def pin():
        with small.pin():
            pass

    def alloc():
        r.alloc(64 << 10).free()

    out = {"lang": "python", "pin_unpin_ns": per_op(10_000, pin),
           "alloc_free_64k_ns": per_op(2_000, alloc), "stats_ns": per_op(10_000, r.stats)}
    src = np.empty(GIB, np.uint8)  # the bytes the Rust and C sides copy
    step = 64 << 20
    for s in range(0, GIB, step):
        i = np.arange(s, s + step, dtype=np.uint64)
        src[s : s + step] = (i * np.uint64(2654435761)) >> np.uint64(11)
    plain = np.ones(GIB, np.uint8)
    big = r.alloc(GIB)
    g, pg = [], []
    for _ in range(9):
        with big.pin(write=True) as v:
            dst = np.frombuffer(v, np.uint8)
            t = time.perf_counter()
            dst[:] = src
            g.append(GIB / (time.perf_counter() - t) / 1e9)
            out["checksum"] = int(dst.view(np.uint64).sum(dtype=np.uint64))
            del dst
        t = time.perf_counter()
        plain[:] = src
        pg.append(GIB / (time.perf_counter() - t) / 1e9)
    out["copy_gbps"], out["plain_memcpy_gbps"] = sorted(g)[4], sorted(pg)[4]
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--sdk")
    a = p.parse_args()
    import memopro

    run = lambda *c, **k: subprocess.run(c, check=True, cwd=ROOT, **k)
    run("cargo", "build", "-q", "--release", "-p", "memopro-c")  # the C library
    run("cargo", "build", "-q", "--release", "-p", "memopro", "--example", "e069_ffi")
    rust = json.loads(run(str(ROOT / "target/release/examples/e069_ffi"), capture_output=True,
                          text=True).stdout)
    exe = os.path.join(tempfile.mkdtemp(), "v6c")
    sdk = ["-isysroot", a.sdk] if a.sdk else []
    run("cc", "-O2", "-std=c11", *sdk, "-I", "crates/memopro-c/include",
        "experiments/e069_validation/v6_ffi.c", "-L", "target/release", "-lmemopro_c", "-o", exe)
    lib = {"DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH":
           str(ROOT / "target/release")}
    p = subprocess.run([exe], capture_output=True, text=True, env={**os.environ, **lib},
                       check=False)
    if p.returncode:
        raise RuntimeError(f"the C benchmark failed ({p.returncode}): {p.stderr}")
    c = json.loads(p.stdout)
    py = python_side()
    results = [rust, c, py]
    for r in results:
        for k in ("pin_unpin_ns", "alloc_free_64k_ns", "stats_ns"):
            r[k + "_vs_rust"] = r[k] / rust[k]
    env = {"memopro": memopro.__version__, "machine": platform.machine(),
           "platform": platform.platform(), "python": platform.python_version()}
    same = len({r["checksum"] for r in results}) == 1
    with open(a.out, "w") as f:
        json.dump({"env": env, "results": results, "K1_same_bytes": same}, f, indent=1)
    for r in results:
        print(json.dumps(r))
    print("K1 same bytes:", same)


if __name__ == "__main__":
    main()
