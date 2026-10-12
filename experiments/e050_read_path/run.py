"""E050 (docs/research/0246): how close memopro's re-reads come to what the SSD can deliver
(S1 of the speed survey 0245). Three parts on Qwen2.5-7B's safetensors files (14.2 GiB):

- A: the device. Cold reads with F_NOCACHE, piece size x reader threads, no memopro.
- B: memopro's read path alone. A runtime at 1.5 GiB holds every tensor of the model as a
  file-backed buffer and pins them in file order, three passes, nothing computed.
- C: the real work. 7B generation (no draft, 1.5B int4 draft) and 7B LoRA training (129 tokens)
  on the Apple GPU at 1.5 GiB, with the default lookahead (64 MiB) and a quarter of the budget.

    .venv/bin/python -m experiments.e050_read_path.run --all

Results: docs/research/data/e050/ (a.json, cases/*.json, summary.md, env.json).
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import itertools
import json
import mmap
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

from experiments.e028_mps_lora.run import footprint, swap_used

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "research" / "data" / "e050"
MIB, GIB = 1 << 20, 1 << 30
TARGET = "Qwen/Qwen2.5-7B-Instruct"
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
DRAFT = "Qwen/Qwen2.5-1.5B-Instruct"
BUDGET = 1536 * MIB
PIECES = [1 * MIB, 4 * MIB, 16 * MIB, 64 * MIB]
THREADS = [1, 2, 4, 8, 16]
REPS = 3
PROMPT = "Explain how a hash table handles collisions."
NEW_TOKENS = 16
STEPS = 5
# name -> (part, kind, lookahead, prefetch)
CASES = {
    "B-sync": ("b", None, 64 * MIB, False),
    "B-64": ("b", None, 64 * MIB, True),
    "B-384": ("b", None, 384 * MIB, True),
    "G-P-64": ("c", "none", 64 * MIB, True),
    "G-P-384": ("c", "none", 384 * MIB, True),
    "G-S-64": ("c", "draft", 64 * MIB, True),
    "G-S-384": ("c", "draft", 384 * MIB, True),
    "T-64": ("c", "train", 64 * MIB, True),
    "T-384": ("c", "train", 384 * MIB, True),
}

_libc = ctypes.CDLL(None, use_errno=True)
_libc.mmap.restype = ctypes.c_void_p
_libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                       ctypes.c_int, ctypes.c_long]
_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
_libc.msync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
MS_INVALIDATE = 0x2  # macOS <sys/mman.h>
F_NOCACHE = getattr(__import__("fcntl"), "F_NOCACHE", 48)


def shards() -> list[Path]:
    from memopro.access._info import local_safetensors

    return sorted(local_safetensors(TARGET, REVISION))


def _mapped(path: Path, fn):
    size = path.stat().st_size
    fd = os.open(path, os.O_RDONLY)
    try:
        addr = _libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
        if addr in (None, ctypes.c_void_p(-1).value):
            raise OSError(ctypes.get_errno(), "mmap")
        try:
            return fn(addr, size)
        finally:
            _libc.munmap(addr, size)
    finally:
        os.close(fd)


def resident(path: Path, offset: int = 0, nbytes: int | None = None) -> float:
    """Fraction of the region's pages in the page cache (mincore; mapping faults nothing in)."""

    def count(addr: int, size: int) -> float:
        page = mmap.PAGESIZE
        start = offset // page * page
        end = size if nbytes is None else min(size, offset + nbytes)
        n = (end - start + page - 1) // page
        vec = ctypes.create_string_buffer(n)
        if _libc.mincore(addr + start, end - start, vec) != 0:
            raise OSError(ctypes.get_errno(), "mincore")
        return sum(b & 1 for b in vec.raw) / max(n, 1)

    return _mapped(path, count)


def drop_cached(path: Path) -> None:
    """Ask the OS to forget the file's clean cached pages (msync MS_INVALIDATE)."""
    _mapped(path, lambda addr, size: _libc.msync(addr, size, MS_INVALIDATE))


# ---- part A: the device ----------------------------------------------------------------


def read_rate(path: Path, offset: int, total: int, piece: int, threads: int) -> float:
    fd = os.open(path, os.O_RDONLY)
    buf = mmap.mmap(-1, total)
    view = memoryview(buf)
    try:
        import fcntl

        fcntl.fcntl(fd, F_NOCACHE, 1)
        count = itertools.count()
        n = total // piece
        errors: list[BaseException] = []

        def work() -> None:
            try:
                while (i := next(count)) < n:
                    got = os.preadv(fd, [view[i * piece : (i + 1) * piece]], offset + i * piece)
                    if got != piece:
                        raise OSError(f"short read {got} of {piece}")
            except BaseException as e:  # noqa: BLE001 - reported below
                errors.append(e)

        workers = [threading.Thread(target=work) for _ in range(threads)]
        t = time.perf_counter()
        for w in workers:
            w.start()
        for w in workers:
            w.join()
        seconds = time.perf_counter() - t
        if errors:
            raise errors[0]
        return total / seconds
    finally:
        view.release()
        buf.close()
        os.close(fd)


def part_a() -> dict:
    files = shards()
    for f in files:
        drop_cached(f)
    rows = []
    k = 0
    for rep in range(REPS):
        for piece in PIECES:
            for threads in THREADS:
                f = files[k % len(files)]
                offset = (k // len(files)) % 3 * GIB
                k += 1
                before = resident(f, offset, GIB)
                rate = read_rate(f, offset, GIB, piece, threads)
                rows.append({
                    "rep": rep, "piece": piece, "threads": threads, "file": f.name,
                    "offset": offset, "resident_before": before, "bytes_per_s": rate,
                    "resident_after": resident(f, offset, GIB),
                })
                print(f"A piece {piece // MIB} MiB x {threads}: {rate / GIB:.2f} GiB/s "
                      f"(cached before {before:.1%})", flush=True)
    return {"files": [f.name for f in files], "rows": rows}


# ---- parts B and C (one process per case) ---------------------------------------------


def _delta(s0: dict, s1: dict) -> dict:
    keys = ("load_bytes", "reread_bytes", "read_seconds", "restore_seconds", "prefetches",
            "prefetch_bytes", "prefetch_hits", "prefetch_wasted", "prefetch_skipped", "pins")
    return {k: s1[k] - s0[k] for k in keys}


def case_b(lookahead: int, prefetch: bool) -> dict:
    from memopro.hibernate._source import read_header
    from memopro.rt import Runtime

    rt = Runtime(budget=BUDGET, prefetch=prefetch, lookahead=lookahead, process_budget=False)
    bufs = []
    for f in shards():
        drop_cached(f)
        for r in sorted(read_header(f).values(), key=lambda r: r.offset):
            bufs.append(rt.add_file(f, r.offset, r.nbytes))
    passes = []
    for _ in range(3):
        s0 = rt.stats()
        t = time.perf_counter()
        for b in bufs:
            with b.pin():
                pass
        passes.append({"seconds": time.perf_counter() - t, **_delta(s0, rt.stats())})
    return {"buffers": len(bufs), "bytes": sum(b.nbytes for b in bufs), "passes": passes,
            "rt": rt.stats()}


def case_c(kind: str, lookahead: int) -> dict:
    import torch
    import transformers

    import memopro
    import memopro.rt.torch as rtt

    for f in shards():
        drop_cached(f)
    base_fp, _ = footprint()
    swap0 = swap_used()
    model = rtt.stream_model(TARGET, budget=BUDGET, device="mps", revision=REVISION,
                             lookahead=lookahead)
    rt = model.memopro_runtime
    out: dict = {}
    if kind == "train":
        texts = json.loads((ROOT / "docs/research/data/e034/texts.json").read_text())[:STEPS]
        s0, t = rt.stats(), time.perf_counter()
        r = memopro.finetune(model, texts, seq_len=129, epochs=1)
        out = {"seconds": time.perf_counter() - t, "losses": r.losses,
               "step_seconds": r.step_seconds, "work": _delta(s0, rt.stats())}
    else:
        tok = transformers.AutoTokenizer.from_pretrained(TARGET, revision=REVISION)
        draft = rtt.draft_model(DRAFT, target=model) if kind == "draft" else None
        ids = tok.apply_chat_template([{"role": "user", "content": PROMPT}],
                                      add_generation_prompt=True, return_tensors="pt")
        ids = (ids["input_ids"] if hasattr(ids, "keys") else ids).to("mps")
        kw = {"max_new_tokens": NEW_TOKENS, "min_new_tokens": NEW_TOKENS,
              "pad_token_id": tok.eos_token_id}
        s0, t = rt.stats(), time.perf_counter()
        if draft is None:
            with torch.no_grad():
                gen = model.generate(input_ids=ids, do_sample=False, **kw)
        else:
            gen = rtt.generate(model, ids, draft=draft, **kw)
        torch.mps.synchronize()
        out = {"seconds": time.perf_counter() - t, "tokens": gen[0, ids.shape[1]:].tolist(),
               "work": _delta(s0, rt.stats())}
    model.memopro_weights.finish()
    _, peak_fp = footprint()
    return {**out, "kind": kind, "lookahead": lookahead, "budget": BUDGET, "limit": rt.limit,
            "base_footprint": base_fp, "peak_footprint": peak_fp, "swap_before": swap0,
            "swap_after": swap_used()}


# ---- summary ----------------------------------------------------------------------------


def _rate(work: dict, seconds: float) -> float:
    return (work["load_bytes"] + work["reread_bytes"]) / seconds


def summarize() -> str:
    a = json.loads((OUT / "a.json").read_text())
    recs = {p.stem: json.loads(p.read_text()) for p in sorted((OUT / "cases").glob("*.json"))}
    lines = ["# E050 summary (0246)", "", "## A: device (GiB/s, median of 3)", "",
             "| piece \\ threads | " + " | ".join(map(str, THREADS)) + " |",
             "|---|" + "---|" * len(THREADS)]
    med = {}
    for piece in PIECES:
        cells = []
        for th in THREADS:
            v = sorted(r["bytes_per_s"] for r in a["rows"]
                       if r["piece"] == piece and r["threads"] == th)[REPS // 2]
            med[(piece, th)] = v
            cells.append(f"{v / GIB:.2f}")
        lines.append(f"| {piece // MIB} MiB | " + " | ".join(cells) + " |")
    ceiling = max(med.values())
    cached = max(r["resident_before"] for r in a["rows"])
    lines += ["", f"Ceiling {ceiling / GIB:.2f} GiB/s; most cached before a read {cached:.1%}.",
              "", "## B and C", "",
              "| case | seconds | read GiB | GiB/s | of ceiling | pins waited (s) | hits |",
              "|---|---|---|---|---|---|---|"]
    gaps = {}
    for name, r in recs.items():
        if "error" in r:
            lines.append(f"| {name} | error | | | | | |")
            continue
        works = [(p["seconds"], p) for p in r["passes"][1:]] if "passes" in r else \
            [(r["seconds"], r["work"])]
        for sec, w in works:
            rate = _rate(w, sec)
            gaps.setdefault(name, []).append(1 - rate / ceiling)
            lines.append(
                f"| {name} | {sec:.1f} | {(w['load_bytes'] + w['reread_bytes']) / GIB:.1f} | "
                f"{rate / GIB:.2f} | {rate / ceiling:.0%} | {w['restore_seconds']:.1f} | "
                f"{w['prefetch_hits']} |")
    lines += ["", "## Checks", ""]
    same = []
    for a_, b_ in (("G-P-64", "G-P-384"), ("G-S-64", "G-S-384"), ("G-P-64", "G-S-64")):
        if a_ in recs and b_ in recs and "tokens" in recs[a_] and "tokens" in recs[b_]:
            same.append((f"{a_} = {b_}", recs[a_]["tokens"] == recs[b_]["tokens"]))
    if "T-64" in recs and "T-384" in recs and "losses" in recs["T-64"]:
        same.append(("T-64 losses = T-384", recs["T-64"]["losses"] == recs["T-384"]["losses"]))
    for label, ok in same:
        lines.append(f"- {label}: {'yes' if ok else 'NO'}")
    for name, r in recs.items():
        if "swap_after" in r:
            lines.append(f"- {name}: swap {(r['swap_after'] - r['swap_before']) / MIB:+.0f} MiB, "
                         f"footprint +{(r['peak_footprint'] - r['base_footprint']) / MIB:.0f} MiB")
    gp = gaps.get("G-P-64", [1.0])[0]
    lines += ["", f"Decision (0246): gap of G-P-64 to the ceiling = {gp:.0%} -> "
              + ("build S1 (>= 25%)" if gp >= 0.25 else "close S1 (< 25%)")]
    return "\n".join(lines) + "\n"


def run_all() -> None:
    (OUT / "cases").mkdir(parents=True, exist_ok=True)
    env = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                 text=True, check=False).stdout.strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "swap_at_start": swap_used(),
        "ssd": subprocess.run(["system_profiler", "SPNVMeDataType"], capture_output=True,
                              text=True, check=False).stdout,
    }
    (OUT / "env.json").write_text(json.dumps(env, indent=1))
    child = {**os.environ, "MallocLargeCache": "0", "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.1",
             "HF_HOME": str(ROOT / ".cache" / "hf"), "HF_HUB_OFFLINE": "1",
             "TOKENIZERS_PARALLELISM": "false"}
    os.environ.update(child)
    (OUT / "a.json").write_text(json.dumps(part_a(), indent=1))
    for name in CASES:
        print(f"[{time.strftime('%H:%M:%S')}] {name}", flush=True)
        p = subprocess.run([sys.executable, "-m", "experiments.e050_read_path.run", "--case", name],
                           cwd=ROOT, capture_output=True, text=True, env=child, check=False)
        if p.returncode != 0:
            rec = {"case": name, "error": p.stderr[-3000:]}
        else:
            rec = {"case": name, **json.loads(p.stdout.strip().splitlines()[-1])}
        (OUT / "cases" / f"{name}.json").write_text(json.dumps(rec, indent=1))
        print(f"   -> {'error' if 'error' in rec else 'ok'}", flush=True)
    text = summarize()
    (OUT / "summary.md").write_text(text)
    print(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--case")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.case:
        part, kind, lookahead, prefetch = CASES[args.case]
        rec = case_b(lookahead, prefetch) if part == "b" else case_c(kind, lookahead)
        print(json.dumps(rec))
    elif args.summary:
        print(summarize())
    elif args.all:
        run_all()


if __name__ == "__main__":
    main()
