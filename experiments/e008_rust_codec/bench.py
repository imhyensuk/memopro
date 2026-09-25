"""E008: Rust parallel spill codec vs Python baselines (pre-registered in docs/research/0023).

Payload: GPT-2 small parameters concatenated (fp32 ~475 MiB, bf16 ~237 MiB).
Codec everywhere: byte shuffle per 4 MiB chunk + zstd (levels 1 and 3).
B0 memcpy | B1 Python single thread | B2 Python ThreadPool(8) | B3 numpy shuffle + zstd threads=8
R1 Rust rayon (8 and 4 threads, GIL released). Plus decompression, a memory-bound probe (C4)
in a subprocess, and spill-to-SSD timing with F_FULLFSYNC (C5).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import fcntl
import json
import os
import resource
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import zstandard as zstd

import memopro._core as core
from experiments._harness.env import capture, save_json, sha256_file

REPO = Path(__file__).resolve().parents[2]
PREREG = REPO / "docs" / "research" / "0023-e007-e008-preregistration.md"
OUT = REPO / "docs" / "research" / "data" / "e008"
CHUNK = 4 << 20
THREADS = 8
REPEATS = 5


def maxrss() -> int:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru if sys.platform == "darwin" else ru * 1024


def load_payloads() -> dict[str, tuple[bytes, int]]:
    import torch

    from experiments.e001_e003_rfc import common as C

    model = C.load_model()
    flat = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    f32 = flat.numpy().tobytes()
    bf16 = flat.to(torch.bfloat16).view(torch.int16).numpy().tobytes()
    return {"fp32": (f32, 4), "bf16": (bf16, 2)}


def shuffle_np(buf, elem: int) -> np.ndarray:
    a = np.frombuffer(buf, dtype=np.uint8).reshape(-1, elem)
    return np.ascontiguousarray(a.T)


def chunks_of(data: bytes):
    mv = memoryview(data)
    return [mv[i : i + CHUNK] for i in range(0, len(data), CHUNK)]


# ------------------------------------------------------------------ compression variants


def b1(data: bytes, elem: int, level: int):
    cctx = zstd.ZstdCompressor(level=level)
    return [cctx.compress(shuffle_np(c, elem)) for c in chunks_of(data)]


def b2(data: bytes, elem: int, level: int):
    local = threading.local()

    def work(c):
        cctx = getattr(local, "cctx", None)
        if cctx is None:
            cctx = local.cctx = zstd.ZstdCompressor(level=level)
        return cctx.compress(shuffle_np(c, elem))

    with cf.ThreadPoolExecutor(THREADS) as ex:
        return list(ex.map(work, chunks_of(data)))


def b3(data: bytes, elem: int, level: int):
    shuffled = np.empty(len(data), dtype=np.uint8)
    for i, c in enumerate(chunks_of(data)):
        shuffled[i * CHUNK : i * CHUNK + len(c)] = shuffle_np(c, elem).reshape(-1)
    return [zstd.ZstdCompressor(level=level, threads=THREADS).compress(shuffled)]


def r1(threads: int):
    def run(data: bytes, elem: int, level: int):
        return core.codec_compress(data, elem, level, CHUNK, threads)

    return run


def py_decompress(chunks, elem: int, total: int) -> bytes:
    dctx = zstd.ZstdDecompressor()
    out = np.empty(total, dtype=np.uint8)
    off = 0
    for c in chunks:
        dec = np.frombuffer(dctx.decompress(c), dtype=np.uint8)
        out[off : off + dec.size] = dec.reshape(elem, -1).T.reshape(-1)
        off += dec.size
    return out.tobytes()


def timed(fn, *args):
    fn(*args)  # warm-up
    times = []
    result = None
    for _ in range(REPEATS):
        t = time.perf_counter()
        result = fn(*args)
        times.append(time.perf_counter() - t)
    return statistics.median(times), times, result


def full_fsync_write(path: Path, parts) -> None:
    with open(path, "wb") as f:
        f.writelines(parts)
        f.flush()
        if hasattr(fcntl, "F_FULLFSYNC"):
            fcntl.fcntl(f.fileno(), fcntl.F_FULLFSYNC)
        else:
            os.fsync(f.fileno())


# ------------------------------------------------------------------ memory probe (subprocess)


def mem_probe(path: str, elem: int, level: int) -> None:
    data = Path(path).read_bytes()
    base = maxrss()
    chunks = core.codec_compress(data, elem, level, CHUNK, THREADS)
    out = sum(len(c) for c in chunks)
    print(json.dumps({"base_maxrss": base, "after_maxrss": maxrss(), "output_bytes": out}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mem-probe", nargs=3, metavar=("PATH", "ELEM", "LEVEL"))
    ap.add_argument("--scratch", default=tempfile.gettempdir())
    args = ap.parse_args()
    if args.mem_probe:
        mem_probe(args.mem_probe[0], int(args.mem_probe[1]), int(args.mem_probe[2]))
        return

    t0 = time.time()
    payloads = load_payloads()
    rows, checks, spill, mem = [], [], [], []
    scratch = Path(args.scratch) / "e008"
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        for name, (data, elem) in payloads.items():
            mib = len(data) / 2**20
            t_copy, _, _ = timed(lambda d: np.frombuffer(d, dtype=np.uint8).copy(), data)
            rows.append({"payload": name, "variant": "B0 memcpy", "level": None,
                         "median_s": t_copy, "mib_per_s": mib / t_copy, "ratio": 1.0})
            for level in (1, 3):
                variants = {"B1 python 1-thread": b1, "B2 python threads(8)": b2,
                            "B3 numpy shuffle + zstd MT(8)": b3,
                            "R1 rust rayon(8)": r1(8), "R1 rust rayon(4)": r1(4)}
                results = {}
                for vname, fn in variants.items():
                    med, all_t, out = timed(fn, data, elem, level)
                    comp = sum(len(c) for c in out)
                    results[vname] = out
                    rows.append({"payload": name, "variant": vname, "level": level,
                                 "median_s": med, "all_s": all_t, "mib_per_s": mib / med,
                                 "ratio": len(data) / comp})
                    print(f"{name} L{level} {vname:30s} {mib / med:8.0f} MiB/s  "
                          f"ratio {len(data) / comp:.4f}  ({time.time() - t0:.0f}s)", flush=True)
                # decompression + correctness
                py_med, _, py_out = timed(py_decompress, results["B1 python 1-thread"], elem, len(data))
                rs_chunks = results["R1 rust rayon(8)"]
                total = len(data)
                rs_med, _, rs_out = timed(
                    lambda ch, e=elem, n=total: core.codec_decompress(ch, e, n, CHUNK, THREADS),
                    rs_chunks)
                rows.append({"payload": name, "variant": "decompress python 1-thread", "level": level,
                             "median_s": py_med, "mib_per_s": mib / py_med})
                rows.append({"payload": name, "variant": "decompress rust rayon(8)", "level": level,
                             "median_s": rs_med, "mib_per_s": mib / rs_med})
                r_py = len(data) / sum(len(c) for c in results["B1 python 1-thread"])
                r_rs = len(data) / sum(len(c) for c in rs_chunks)
                checks.append({"payload": name, "level": level,
                               "rust_roundtrip_bit_exact": rs_out == data,
                               "python_roundtrip_bit_exact": py_out == data,
                               "ratio_python": r_py, "ratio_rust": r_rs,
                               "ratio_rel_diff": abs(r_py - r_rs) / r_py})
            # C4 memory probe in a fresh process
            probe_file = scratch / f"payload_{name}.bin"
            probe_file.write_bytes(data)
            res = subprocess.run([sys.executable, "-m", "experiments.e008_rust_codec.bench",
                                  "--mem-probe", str(probe_file), str(elem), "1"],
                                 capture_output=True, text=True, cwd=REPO, check=True)
            probe = json.loads(res.stdout.strip().splitlines()[-1])
            growth = probe["after_maxrss"] - probe["base_maxrss"]
            bound = probe["output_bytes"] + THREADS * 2 * CHUNK + 64 * 2**20
            mem.append(probe | {"payload": name, "growth_bytes": growth, "bound_bytes": bound,
                                "C4_pass": growth <= bound})
            probe_file.unlink()
            # C5 spill timing (level 1, rust 8 threads)
            raw_t, comp_t = [], []
            for rep in range(REPEATS + 1):
                p = scratch / f"spill_{name}_{rep}.bin"
                t = time.perf_counter()
                full_fsync_write(p, [data])
                dt_raw = time.perf_counter() - t
                p.unlink()
                t = time.perf_counter()
                chunks = core.codec_compress(data, elem, 1, CHUNK, THREADS)
                full_fsync_write(p, chunks)
                dt_comp = time.perf_counter() - t
                p.unlink()
                if rep:  # first repetition is warm-up
                    raw_t.append(dt_raw)
                    comp_t.append(dt_comp)
            spill.append({"payload": name, "raw_write_median_s": statistics.median(raw_t),
                          "compressed_write_median_s": statistics.median(comp_t),
                          "raw_mib_per_s": mib / statistics.median(raw_t),
                          "compressed_effective_mib_per_s": mib / statistics.median(comp_t),
                          "C5_compressed_faster": statistics.median(comp_t) < statistics.median(raw_t)})
            print("spill", spill[-1], flush=True)
    finally:
        for p in scratch.glob("*"):
            p.unlink()
        scratch.rmdir()

    def get(payload, variant, level):
        return next(r for r in rows if r["payload"] == payload and r["variant"] == variant
                    and r["level"] == level)["mib_per_s"]

    criteria = []
    for name in payloads:
        for level in (1, 3):
            rs = get(name, "R1 rust rayon(8)", level)
            b1v = get(name, "B1 python 1-thread", level)
            best_py = max(get(name, "B2 python threads(8)", level),
                          get(name, "B3 numpy shuffle + zstd MT(8)", level))
            criteria.append({"payload": name, "level": level, "rust_vs_b1": rs / b1v,
                             "rust_vs_best_python_parallel": rs / best_py,
                             "C2_pass": rs >= 3 * b1v, "C3_pass": rs >= 1.3 * best_py})
    result = {"experiment": "E008", "config": {"chunk_bytes": CHUNK, "threads": THREADS,
                                               "repeats": REPEATS},
              "rows": rows, "correctness": checks, "memory_probe": mem, "spill": spill,
              "criteria": criteria, "runtime_s": round(time.time() - t0, 1)}
    save_json(result, OUT / "results.json")
    save_json(capture(__file__, extra={"prereg_sha256": sha256_file(PREREG)}), OUT / "env.json")
    print(json.dumps({"criteria": criteria, "correctness": checks, "memory": mem}, indent=1))


if __name__ == "__main__":
    main()
