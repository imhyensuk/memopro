"""E069 V5 (docs/research/0356): the runtime's lossless compression on six kinds of data.

Each kind is 1 GiB put in a runtime buffer, evicted (compressed in memory) and pinned again
(decompressed); seconds come from the runtime's own counters. Writes one JSON file.

    python -m experiments.e069_validation.v5_codec OUT.json [--weights model.safetensors]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time

import numpy as np

GIB = 1 << 30


def kinds(weights: str | None):
    n = GIB
    rng = np.random.default_rng(0)
    if weights:  # bf16 weights as stored (after the header)
        with open(weights, "rb") as f:
            head = int.from_bytes(f.read(8), "little")
            f.seek(8 + head)
            yield "bf16 weights (7B)", np.frombuffer(f.read(n), np.uint8)
    side = 16384
    x = np.arange(side, dtype=np.float32) / side
    field = np.empty((side, side), np.float32)
    for i in range(0, side, 1024):  # in row blocks: no full-size temporaries
        y = x[i : i + 1024, None]
        field[i : i + 1024] = np.sin(9 * x) * np.cos(7 * y) + 0.01 * x * y
    yield "float32 simulation field", field.ravel()
    del field
    h = w = 512
    yy = np.arange(h, dtype=np.float64)[:, None]
    xx = np.arange(w, dtype=np.float64)[None, :]
    stack = np.empty((n // (2 * h * w), h, w), np.uint16)
    for i in range(len(stack)):  # E027's image generator
        smooth = 2000 + 900 * np.sin(xx / 41 + i / 57) * np.cos(yy / 67 - i / 91)
        stack[i] = smooth.astype(np.uint16) + rng.integers(0, 4, (h, w), dtype=np.uint16)
    yield "uint16 image stack", stack.ravel()
    del stack
    from experiments.e069_validation import csvgen

    text = bytearray()
    for b in csvgen.fact_blocks(n // csvgen.ROW_BYTES + 1, 20_000_000):
        text += b
        if len(text) >= n:
            break
    yield "CSV text", np.frombuffer(bytes(text[:n]), np.uint8)
    del text
    yield "int64 keys", rng.permutation(n // 8).astype(np.int64)
    yield "random bytes", rng.integers(0, 256, n, dtype=np.uint8)


def measure(name: str, data: np.ndarray) -> dict:
    from memopro import rt

    raw = data.view(np.uint8).reshape(-1)
    want = hashlib.sha256(raw).hexdigest()
    # min_saving 0: the codec on every kind; the default policy (0.15) keeps a buffer as is when
    # a probe finds less saving, reported as default_compresses
    r = rt.Runtime(budget=3 * GIB, process_budget=False, min_saving=0.0)
    buf = r.alloc(raw.nbytes)
    with buf.pin(write=True) as v:
        np.frombuffer(v, np.uint8)[:] = raw
    s0 = r.stats()
    t = time.perf_counter()
    evicted = buf.evict()
    wall_c = time.perf_counter() - t
    s1 = r.stats()
    t = time.perf_counter()
    with buf.pin() as v:
        got = hashlib.sha256(np.frombuffer(v, np.uint8)).hexdigest()
    wall_d = time.perf_counter() - t  # includes the hash below, reported apart
    s2 = r.stats()
    t = time.perf_counter()
    hashlib.sha256(raw).hexdigest()
    hash_s = time.perf_counter() - t
    c_s = s1["compress_seconds"] - s0["compress_seconds"]
    d_s = s2["decompress_seconds"] - s1["decompress_seconds"]
    out = s1["compress_out"] - s0["compress_out"]
    rec = {
        "kind": name,
        "bytes": raw.nbytes,
        "identical": got == want,
        "compressed_bytes": out,
        "ratio": raw.nbytes / out if out else None,
        "evicted": evicted,
        "default_compresses": bool(out) and 1 - out / raw.nbytes >= 0.15,
        "compress_s": c_s,
        "decompress_s": d_s,
        "compress_gbps": raw.nbytes / c_s / 1e9 if c_s else None,
        "decompress_gbps": raw.nbytes / d_s / 1e9 if d_s else None,
        "evict_wall_s": wall_c,
        "pin_wall_s_minus_hash": max(0.0, wall_d - hash_s),
    }
    del buf, r
    return rec


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--weights")
    a = p.parse_args()
    import memopro

    recs = []
    for name, data in kinds(a.weights):
        recs.append(measure(name, data))
        print(json.dumps(recs[-1]), flush=True)
        del data
    env = {"memopro": memopro.__version__, "machine": platform.machine(),
           "platform": platform.platform(), "python": platform.python_version()}
    with open(a.out, "w") as f:
        json.dump({"env": env, "results": recs}, f, indent=1)


if __name__ == "__main__":
    main()
