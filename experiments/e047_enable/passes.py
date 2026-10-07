"""E047 W2 (docs/research/0231): repeated passes over an array, with the prediction made first.

Builds N bytes of float32 data (bf16 values widened: low 16 bits zero, like model weights) in
64 MiB pieces, then sums it P times. With ``--ceiling`` it runs inside ``memopro.enable`` and,
before building anything, asks ``Session.estimate`` for the extra seconds of the P passes from a
16 MiB sample; the pager's waiting time during the passes is the measured counterpart.
Prints one JSON line.

    python -m experiments.e047_enable.passes --mib 1024 --passes 4 [--ceiling 900MB]
"""

import argparse
import hashlib
import json
import time

import numpy as np

PIECE = 16 << 20  # float32 elements per 64 MiB piece


def piece(i: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(i)
    bits = rng.standard_normal(n, dtype=np.float32).view(np.uint32) & np.uint32(0xFFFF0000)
    return bits.view(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mib", type=int, default=1024)
    ap.add_argument("--passes", type=int, default=4)
    ap.add_argument("--ceiling", default=None)
    a = ap.parse_args()
    n = a.mib * (1 << 20) // 4
    out: dict = {"mib": a.mib, "passes": a.passes, "ceiling": a.ceiling}
    session = None
    if a.ceiling is not None:
        import memopro

        session = memopro.enable(a.ceiling, torch=False, transformers=False)
        out["ceiling_bytes"] = session.budget
        est = session.estimate(piece(10_000, PIECE // 4), total=n * 4, passes=a.passes)
        out["estimate"] = est
    t0 = time.perf_counter()
    x = np.empty(n, dtype=np.float32)
    for i, start in enumerate(range(0, n, PIECE)):
        x[start : start + PIECE] = piece(i, min(PIECE, n - start))
    out["build_seconds"] = time.perf_counter() - t0
    before = session.pager.stats() if session is not None else None
    sums, times = [], []
    for _ in range(a.passes):
        t = time.perf_counter()
        sums.append(float(x.sum(dtype=np.float64)))
        times.append(time.perf_counter() - t)
    out["pass_seconds"] = times
    out["sums_sha256"] = hashlib.sha256(json.dumps(sums).encode()).hexdigest()
    if session is not None:
        after = session.pager.stats()
        waited = lambda s: s["restore_seconds"] + s["compress_seconds"]
        out["waited_seconds"] = waited(after) - waited(before)
        out["pager"] = dict(after)
        session.disable()
    print(json.dumps(out))


if __name__ == "__main__":
    main()
