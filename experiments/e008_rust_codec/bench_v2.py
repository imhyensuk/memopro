"""E008b (EXPLORATORY, not pre-registered): does using Rust's strengths change E008's outcome?

Rust v2 = specialised byte shuffle + per-thread reused zstd context/scratch + pipelined spill that
streams compressed chunks straight to a file (no Python objects, bounded memory).
Compared, median of 5 after 1 warm-up, level 1, 4 MiB chunks, 8 threads:
  kernel:  B2 Python threads(8) | R1 Rust v1 (E008) | R2 Rust v2 reuse
  spill:   S0 raw write | S1 Python threads compress, then write | S2 Rust pipelined spill
  memory:  peak RSS growth of S2 in a fresh process (vs. output size)
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import memopro._core as core
from experiments._harness.env import capture, save_json
from experiments.e008_rust_codec.bench import (
    CHUNK,
    REPEATS,
    THREADS,
    b2,
    full_fsync_write,
    load_payloads,
    maxrss,
)

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "research" / "data" / "e008"


def median_time(fn, *args) -> tuple[float, object]:
    fn(*args)
    times, out = [], None
    for _ in range(REPEATS):
        t = time.perf_counter()
        out = fn(*args)
        times.append(time.perf_counter() - t)
    return statistics.median(times), out


def mem_probe(path: str, elem: int, target: str) -> None:
    data = Path(path).read_bytes()
    base = maxrss()
    written = core.codec_spill_to_file(data, elem, target, 1, CHUNK, THREADS)
    print(json.dumps({"base_maxrss": base, "after_maxrss": maxrss(), "written": written}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mem-probe", nargs=3, metavar=("PATH", "ELEM", "TARGET"))
    ap.add_argument("--scratch", default=tempfile.gettempdir())
    args = ap.parse_args()
    if args.mem_probe:
        mem_probe(args.mem_probe[0], int(args.mem_probe[1]), args.mem_probe[2])
        return

    t0 = time.time()
    scratch = Path(args.scratch) / "e008b"
    scratch.mkdir(parents=True, exist_ok=True)
    kernel, spill, memory = [], [], []
    try:
        for name, (data, elem) in load_payloads().items():
            mib = len(data) / 2**20
            for vname, fn in {
                "B2 python threads(8)": lambda d, e: b2(d, e, 1),
                "R1 rust v1 (E008)": lambda d, e: core.codec_compress(d, e, 1, CHUNK, THREADS),
                "R2 rust v2 reuse": lambda d, e: core.codec_compress_reuse_size(d, e, 1, CHUNK, THREADS),
            }.items():
                t, _ = median_time(fn, data, elem)
                kernel.append({"payload": name, "variant": vname, "median_s": t, "mib_per_s": mib / t})
                print(f"{name} {vname:24s} {mib / t:7.0f} MiB/s ({time.time() - t0:.0f}s)", flush=True)

            target = scratch / "spill.bin"

            def s0(d=data, tgt=target):
                full_fsync_write(tgt, [d])
                tgt.unlink()

            def s1(d=data, e=elem, tgt=target):
                full_fsync_write(tgt, b2(d, e, 1))
                tgt.unlink()

            def s2(d=data, e=elem, tgt=target):
                n = core.codec_spill_to_file(d, e, str(tgt), 1, CHUNK, THREADS)
                tgt.unlink()
                return n

            for vname, fn in {"S0 raw write": s0, "S1 python threads compress, then write": s1,
                              "S2 rust pipelined spill": s2}.items():
                t, out = median_time(fn)
                spill.append({"payload": name, "variant": vname, "median_s": t,
                              "effective_mib_per_s": mib / t,
                              "bytes_written": out if isinstance(out, int) else None})
                print(f"{name} {vname:40s} {t:.3f}s ({time.time() - t0:.0f}s)", flush=True)

            probe_src = scratch / f"payload_{name}.bin"
            probe_src.write_bytes(data)
            res = subprocess.run(
                [sys.executable, "-m", "experiments.e008_rust_codec.bench_v2", "--mem-probe",
                 str(probe_src), str(elem), str(scratch / "probe_spill.bin")],
                capture_output=True, text=True, cwd=REPO, check=True)
            probe = json.loads(res.stdout.strip().splitlines()[-1])
            memory.append(probe | {"payload": name,
                                   "growth_bytes": probe["after_maxrss"] - probe["base_maxrss"]})
            print("memory", memory[-1], flush=True)
            for p in (probe_src, scratch / "probe_spill.bin"):
                p.unlink(missing_ok=True)
    finally:
        for p in scratch.glob("*"):
            p.unlink()
        scratch.rmdir()

    result = {"experiment": "E008b (exploratory)", "kernel": kernel, "spill": spill,
              "memory": memory, "runtime_s": round(time.time() - t0, 1)}
    save_json(result, OUT / "results_v2.json")
    save_json(capture(__file__, extra={"note": "exploratory, not pre-registered"}),
              OUT / "env_v2.json")


if __name__ == "__main__":
    main()
