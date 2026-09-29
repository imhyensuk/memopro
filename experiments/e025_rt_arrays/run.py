"""E025 (docs/research/0111): runtime C-R phase 1 gate G-R1.

A three-pass statistics pipeline over large numeric arrays already on disk (the data regions of
local safetensors files; nothing new is written, 0110 N2), run by several methods, each case in a
fresh process:

    .venv/bin/python -m experiments.e025_rt_arrays.run --all            # every case + summary
    .venv/bin/python -m experiments.e025_rt_arrays.run --case gpt2 memopro 274726912

Results: docs/research/data/e025/ (cases/*.json, summary.md, env.json).
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import resource
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "research" / "data" / "e025"
HUB = ROOT / ".cache" / "hf" / "hub"
BLOCK = 16 << 20  # bytes per block (0111)
SUB = 4 << 20  # elements per compute piece
F_NOCACHE = 48  # macOS fcntl
MIB = 1 << 20

DATASETS = {
    "gpt2": ("models--openai-community--gpt2", ["model.safetensors"], "f32"),
    "q15": ("models--Qwen--Qwen2.5-1.5B-Instruct", ["model.safetensors"], "bf16"),
    "q3": (
        "models--Qwen--Qwen2.5-3B-Instruct",
        ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"],
        "bf16",
    ),
}
DERIVED_SOURCE_BYTES = 256 * MIB  # bf16 from q15, widened to 512 MiB of float32


# ---------------------------------------------------------------- data


def snapshot_file(repo: str, name: str) -> Path:
    found = sorted((HUB / repo / "snapshots").glob(f"*/{name}"))
    if not found:
        raise SystemExit(f"missing {repo}/{name} in {HUB} (E025 uses only local files)")
    return found[0].resolve()


def data_region(path: Path) -> tuple[int, int]:
    """Offset and length of a safetensors file's data region (after its JSON header)."""
    with path.open("rb") as f:
        (header,) = struct.unpack("<Q", f.read(8))
    start = 8 + header
    return start, path.stat().st_size - start


def blocks_of(dataset: str) -> tuple[list[tuple[Path, int, int]], str]:
    repo, names, kind = DATASETS[dataset]
    elem = 4 if kind == "f32" else 2
    out = []
    for name in names:
        path = snapshot_file(repo, name)
        start, length = data_region(path)
        length -= length % elem
        for off in range(0, length, BLOCK):
            out.append((path, start + off, min(BLOCK, length - off)))
    return out, kind


# ---------------------------------------------------------------- pipeline P3


class P3:
    """Three passes of statistics; every method feeds the same blocks in the same order, and
    each piece is copied into the same aligned scratch before computing, so results can be
    compared bit for bit."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.u32 = np.empty(SUB, dtype=np.uint32)
        self.tmp = np.empty(SUB, dtype=np.float32)
        self.tmp2 = np.empty(SUB, dtype=np.float32)
        self.s1 = self.s2 = self.a1 = self.m3 = 0.0
        self.n = self.c3 = 0
        self.mx = 0.0
        self.mu = self.sd = np.float32(0)

    def _pieces(self, raw: np.ndarray):
        vals = raw.view(np.uint16 if self.kind == "bf16" else np.float32)
        for start in range(0, vals.size, SUB):
            part = vals[start : start + SUB]
            k = part.size
            if self.kind == "bf16":
                self.u32[:k] = part
                self.u32[:k] <<= 16
            else:
                self.u32[:k] = part.view(np.uint32)
            yield self.u32[:k].view(np.float32), k

    def run_pass(self, p: int, raw: np.ndarray) -> None:
        for x, k in self._pieces(raw):
            t = self.tmp[:k]
            t2 = self.tmp2[:k]
            if p == 1:
                self.s1 += float(np.sum(x, dtype=np.float64))
                np.square(x, out=t)
                self.s2 += float(np.sum(t, dtype=np.float64))
                self.n += k
            elif p == 2:
                np.subtract(x, self.mu, out=t)
                np.divide(t, self.sd, out=t)
                np.abs(t, out=t2)
                self.c3 += int(np.count_nonzero(t2 > 3.0))
                self.a1 += float(np.sum(t2, dtype=np.float64))
            else:
                np.subtract(x, self.mu, out=t)
                np.divide(t, self.sd, out=t)
                np.power(t, 3, out=t2)
                self.m3 += float(np.sum(t2, dtype=np.float64))
                np.abs(x, out=t)
                self.mx = max(self.mx, float(np.max(t)))

    def end_pass(self, p: int) -> None:
        if p == 1:
            mean = self.s1 / self.n
            var = max(self.s2 / self.n - mean * mean, 1e-30)
            self.mu = np.float32(mean)
            self.sd = np.float32(var**0.5)

    def result(self) -> dict:
        return {
            "n": self.n,
            "mean": float(self.mu),
            "std": float(self.sd),
            "c3": self.c3,
            "a1": self.a1,
            "m3": self.m3,
            "max_abs": self.mx,
        }


# ---------------------------------------------------------------- measurements


def swap_used() -> int:
    out = subprocess.run(
        ["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, check=False
    ).stdout
    # "total = 4096.00M  used = 2439.31M  free = ..."
    for part in out.split("  "):
        if part.strip().startswith("used"):
            return int(float(part.split("=")[1].strip().rstrip("M")) * MIB)
    return -1


def maxrss() -> int:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r if sys.platform == "darwin" else r * 1024


# ---------------------------------------------------------------- methods


def read_block(path: Path, off: int, n: int, buf: np.ndarray, nocache: bool, cache: dict) -> np.ndarray:
    key = (path, nocache)
    f = cache.get(key)
    if f is None:
        f = cache[key] = path.open("rb", buffering=0)
        if nocache and sys.platform == "darwin":
            fcntl.fcntl(f.fileno(), F_NOCACHE, 1)
    view = memoryview(buf)[:n]
    got = os.preadv(f.fileno(), [view], off) if hasattr(os, "preadv") else None
    if got is None:
        f.seek(off)
        got = f.readinto(view)
    if got != n:
        raise OSError(f"short read {got} of {n} at {off} in {path}")
    return buf[:n]


def run_file_case(dataset: str, method: str, budget: int) -> dict:
    blocks, kind = blocks_of(dataset)
    total = sum(n for _, _, n in blocks)
    pipe = P3(kind)
    base = maxrss()
    times, extra = {}, {}
    t_all = time.perf_counter()
    if method == "naive":
        t = time.perf_counter()
        whole = np.empty(total, dtype=np.uint8)
        files: dict = {}
        at = 0
        spans = []
        for path, off, n in blocks:
            read_block(path, off, n, whole[at:], False, files)
            spans.append((at, n))
            at += n
        times["load"] = time.perf_counter() - t
        for p in (1, 2, 3):
            t = time.perf_counter()
            for at, n in spans:
                pipe.run_pass(p, whole[at : at + n])
            pipe.end_pass(p)
            times[f"pass{p}"] = time.perf_counter() - t
    elif method in ("stream", "stream_nc"):
        buf = np.empty(BLOCK, dtype=np.uint8)
        files = {}
        for p in (1, 2, 3):
            t = time.perf_counter()
            for path, off, n in blocks:
                pipe.run_pass(p, read_block(path, off, n, buf, method == "stream_nc", files))
            pipe.end_pass(p)
            times[f"pass{p}"] = time.perf_counter() - t
        extra["read_bytes"] = 3 * total
    elif method == "memmap":
        maps = {}
        for p in (1, 2, 3):
            t = time.perf_counter()
            for path, off, n in blocks:
                m = maps.get(path)
                if m is None:
                    m = maps[path] = np.memmap(path, dtype=np.uint8, mode="r")
                pipe.run_pass(p, m[off : off + n])
            pipe.end_pass(p)
            times[f"pass{p}"] = time.perf_counter() - t
    elif method in ("memopro", "memopro_lru"):
        from memopro import rt

        r = rt.Runtime(budget=budget, policy="lru" if method == "memopro_lru" else "reuse")
        dtype = "float32" if kind == "f32" else "bfloat16"
        bufs = [r.add_file(path, off, n, dtype=dtype) for path, off, n in blocks]
        per_pass = []
        for p in (1, 2, 3):
            before = r.stats()
            t = time.perf_counter()
            for b in bufs:
                b.apply(lambda v, p=p: pipe.run_pass(p, v.view(np.uint8)))
            pipe.end_pass(p)
            times[f"pass{p}"] = time.perf_counter() - t
            after = r.stats()
            per_pass.append(
                {
                    "loaded": after["load_bytes"] - before["load_bytes"],
                    "reread": after["reread_bytes"] - before["reread_bytes"],
                    "read_seconds": after["read_seconds"] - before["read_seconds"],
                }
            )
        extra["per_pass"] = per_pass
        extra["rt"] = r.stats()
        extra["read_bytes"] = extra["rt"]["load_bytes"] + extra["rt"]["reread_bytes"]
    else:
        raise SystemExit(f"unknown method {method}")
    times["total"] = time.perf_counter() - t_all
    return {
        "dataset": dataset,
        "method": method,
        "budget": budget if method.startswith("memopro") else None,
        "W": total,
        "blocks": len(blocks),
        "times": times,
        "result": pipe.result(),
        "rss_base": base,
        "rss_peak": maxrss(),
        **extra,
    }


def run_derived_case(method: str, budget: int) -> dict:
    """512 MiB of float32 made in memory (widened bf16 from q15): no file to re-read."""
    blocks, _ = blocks_of("q15")
    src_path, src_off, _ = blocks[0]
    half = BLOCK // 2
    pieces = DERIVED_SOURCE_BYTES // half
    pipe = P3("f32")
    base = maxrss()
    times, extra = {}, {}
    t_all = time.perf_counter()
    staging = np.empty(half, dtype=np.uint8)
    files: dict = {}
    t = time.perf_counter()
    if method == "naive":
        store = []
        for i in range(pieces):
            raw = read_block(src_path, src_off + i * half, half, staging, False, files)
            store.append((raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32))
        run = lambda p: [pipe.run_pass(p, a.view(np.uint8)) for a in store]
    elif method == "memopro":
        from memopro import rt

        r = rt.Runtime(budget=budget)
        store = []
        for i in range(pieces):
            raw = read_block(src_path, src_off + i * half, half, staging, False, files)
            b = r.alloc(nbytes=BLOCK, dtype="float32")

            def fill(y, raw=raw):
                y.view(np.uint32)[:] = raw.view(np.uint16)
                y.view(np.uint32)[:] <<= 16

            b.apply(fill, write=True)
            store.append(b)

        def run(p):
            for b in store:
                b.apply(lambda v: pipe.run_pass(p, v.view(np.uint8)))

    else:
        raise SystemExit(f"unknown method {method}")
    times["build"] = time.perf_counter() - t
    for p in (1, 2, 3):
        t = time.perf_counter()
        run(p)
        pipe.end_pass(p)
        times[f"pass{p}"] = time.perf_counter() - t
    times["total"] = time.perf_counter() - t_all
    if method == "memopro":
        extra["rt"] = r.stats()
    return {
        "dataset": "derived",
        "method": method,
        "budget": budget if method == "memopro" else None,
        "W": pieces * BLOCK,
        "blocks": pieces,
        "times": times,
        "result": pipe.result(),
        "rss_base": base,
        "rss_peak": maxrss(),
        **extra,
    }


def case(dataset: str, method: str, budget: int) -> dict:
    swap0 = swap_used()
    if dataset == "derived":
        rec = run_derived_case(method, budget)
    else:
        rec = run_file_case(dataset, method, budget)
    rec["swap_before"] = swap0
    rec["swap_after"] = swap_used()
    return rec


# ---------------------------------------------------------------- runner

def plan() -> list[tuple[str, str, int]]:
    w = sum(n for _, _, n in blocks_of("gpt2")[0])
    half, quarter = w // 2, w // 4
    return [
        ("gpt2", "naive", 0),
        ("gpt2", "stream", 0),
        ("gpt2", "stream_nc", 0),
        ("gpt2", "memmap", 0),
        ("gpt2", "memopro", half),
        ("gpt2", "memopro", quarter),
        ("gpt2", "memopro_lru", half),
        ("derived", "naive", 0),
        ("derived", "memopro", 256 * MIB),
        ("q15", "stream_nc", 0),
        ("q15", "memopro", 1 << 30),
        ("q15", "memopro", 512 * MIB),
        ("q15", "memopro_lru", 1 << 30),
        ("q3", "stream_nc", 0),
        ("q3", "memopro", 1 << 30),
    ]


def key(dataset: str, method: str, budget: int) -> str:
    return f"{dataset}__{method}" + (f"__{budget // MIB}MiB" if budget else "")


def run_all() -> None:
    (OUT / "cases").mkdir(parents=True, exist_ok=True)
    env = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
        ).stdout.strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "block_bytes": BLOCK,
        "sub_elements": SUB,
        "swap_at_start": swap_used(),
    }
    import memopro

    env["memopro"] = memopro.__version__
    (OUT / "env.json").write_text(json.dumps(env, indent=1))
    for ds, method, budget in plan():
        name = key(ds, method, budget)
        for attempt in (1, 2):
            print(f"[{time.strftime('%H:%M:%S')}] {name} (attempt {attempt})", flush=True)
            proc = subprocess.run(
                [sys.executable, "-m", "experiments.e025_rt_arrays.run", "--case", ds, method,
                 str(budget)],
                cwd=ROOT,
                capture_output=True,
                text=True,
                env={**os.environ, "MallocLargeCache": "0"},
                check=False,
            )
            if proc.returncode != 0:
                rec = {"dataset": ds, "method": method, "budget": budget, "error": proc.stderr[-3000:]}
                break
            rec = json.loads(proc.stdout.strip().splitlines()[-1])
            rec["attempt"] = attempt
            grew = rec["swap_after"] - rec["swap_before"]
            if grew <= 64 * MIB:
                break
            print(f"   swap grew by {grew / MIB:.0f} MiB: repeating once", flush=True)
            rec["contaminated_first_attempt_swap_growth"] = grew
        (OUT / "cases" / f"{name}.json").write_text(json.dumps(rec, indent=1))
        t = rec.get("times", {}).get("total")
        print(f"   -> {'error' if 'error' in rec else f'{t:.1f} s'}", flush=True)
    summarize()


# ---------------------------------------------------------------- summary


def summarize() -> None:
    recs = {}
    for p in sorted((OUT / "cases").glob("*.json")):
        r = json.loads(p.read_text())
        recs[p.stem] = r
    rows, verdicts = [], []

    def v(name, ok, detail):
        verdicts.append((name, "pass" if ok else ("fail" if ok is False else "n/a"), detail))

    by_ds: dict[str, list] = {}
    for name, r in recs.items():
        by_ds.setdefault(r["dataset"], []).append((name, r))
    # H2: results identical within each dataset
    h2 = []
    for ds, items in by_ds.items():
        results = {json.dumps(r["result"], sort_keys=True) for _, r in items if "result" in r}
        h2.append((ds, len(results) == 1, len(items)))
    v("H2 lossless", all(ok for _, ok, _ in h2), "; ".join(f"{d}: {'same' if ok else 'DIFFERENT'} ({n})" for d, ok, n in h2))
    mem = [(n, r) for n, r in recs.items() if r.get("method", "").startswith("memopro") and "rt" in r]
    h1 = [(n, r["rt"]["peak_used"] <= r["budget"] and r["rss_peak"] - r["rss_base"] <= r["budget"] + 96 * MIB,
           r["rt"]["peak_used"], r["rss_peak"] - r["rss_base"], r["budget"]) for n, r in mem]
    v("H1 ceiling", all(ok for _, ok, *_ in h1) if h1 else None,
      "; ".join(f"{n}: peak {pk / MIB:.0f} / RSS+{g / MIB:.0f} of {b / MIB:.0f} MiB" for n, _, pk, g, b in h1))
    h3 = [(n, r["rt"]["written_bytes"] == 0 and r["swap_after"] - r["swap_before"] <= 64 * MIB,
           r["swap_after"] - r["swap_before"]) for n, r in mem]
    v("H3 no writes", all(ok for _, ok, _ in h3) if h3 else None,
      "; ".join(f"{n}: swap {g / MIB:+.0f} MiB" for n, _, g in h3))
    big = [n for n in recs if n.startswith(("q15__memopro__", "q3__memopro__"))]
    v("H4 runs what did not fit", bool(big) and all("error" not in recs[n] for n in big) and len(big) == 3,
      ", ".join(big))
    h5 = []
    for n, r in mem:
        if r["method"] != "memopro" or r["dataset"] == "derived":
            continue
        ref = recs.get(f"{r['dataset']}__stream_nc")
        if ref and "times" in ref:
            h5.append((n, r["times"]["total"] <= 1.1 * ref["times"]["total"], r["times"]["total"], ref["times"]["total"]))
    v("H5 vs manual streaming", all(ok for _, ok, *_ in h5) if h5 else None,
      "; ".join(f"{n}: {a:.1f} vs {b:.1f} s ({a / b:.2f}x)" for n, _, a, b in h5))
    h7 = []
    for n, r in mem:
        if r["method"] != "memopro" or r["dataset"] == "derived":
            continue
        limit = r["W"] - 0.8 * (r["budget"] - 2 * BLOCK)
        rr = [pp["reread"] for pp in r["per_pass"][1:]]
        h7.append((n, all(x <= limit for x in rr), rr, limit))
    v("H7 policy", all(ok for _, ok, *_ in h7) if h7 else None,
      "; ".join(f"{n}: re-read {[round(x / MIB) for x in rr]} MiB <= {lim / MIB:.0f}" for n, _, rr, lim in h7))
    dm, dn = recs.get("derived__memopro__256MiB"), recs.get("derived__naive")
    ok8 = None
    if dm and dn and "rt" in dm:
        ok8 = (dm["result"] == dn["result"] and dm["rt"]["compressions"] > 0
               and dm["rt"]["peak_used"] <= dm["budget"]
               and dm["rss_peak"] - dm["rss_base"] <= dm["budget"] + 96 * MIB)
    v("H8 compression", ok8, f"compressions {dm['rt']['compressions'] if dm and 'rt' in dm else '-'}, "
      f"ratio {(dm['rt']['compress_in'] / max(1, dm['rt']['compress_out'])) if dm and 'rt' in dm else 0:.2f}")
    naive = recs.get("gpt2__naive")
    h6 = []
    if naive:
        for n, r in mem:
            if r["dataset"] == "gpt2" and r["method"] == "memopro":
                h6.append(f"{n}: {r['times']['total'] / naive['times']['total']:.2f}x naive")
    v("H6 (report) vs naive", None, "; ".join(h6))
    gate = all(s == "pass" for name, s, _ in verdicts if name.split()[0] in ("H1", "H2", "H3", "H4", "H5", "H7", "H8"))
    for name, r in recs.items():
        t = r.get("times", {})
        rows.append(
            f"| {name} | {r.get('W', 0) / MIB:.0f} | {t.get('total', float('nan')):.1f} | "
            + " / ".join(f"{t.get(f'pass{p}', float('nan')):.1f}" for p in (1, 2, 3))
            + f" | {(r.get('read_bytes') or 0) / MIB:.0f} | {(r.get('rss_peak', 0) - r.get('rss_base', 0)) / MIB:.0f} | "
            f"{(r.get('swap_after', 0) - r.get('swap_before', 0)) / MIB:+.0f} | "
            f"{'error' if 'error' in r else 'ok'} |"
        )
    text = ["# E025 summary (0111)", "", f"Gate G-R1: **{'pass' if gate else 'fail'}**", "",
            "| check | result | detail |", "|---|---|---|"]
    text += [f"| {a} | {b} | {c} |" for a, b, c in verdicts]
    text += ["", "| case | W MiB | total s | pass 1/2/3 s | read MiB | RSS growth MiB | swap MiB | status |",
             "|---|---|---|---|---|---|---|---|", *rows]
    (OUT / "summary.md").write_text("\n".join(text) + "\n")
    print("\n".join(text))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--case", nargs=3, metavar=("DATASET", "METHOD", "BUDGET"))
    a = ap.parse_args()
    if a.case:
        ds, method, budget = a.case
        print(json.dumps(case(ds, method, int(budget))))
    elif a.summary:
        summarize()
    elif a.all:
        run_all()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
