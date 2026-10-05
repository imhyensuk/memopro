"""E040 diagnosis: per 1 MiB chunk, byte shuffle (by element size) + zstd level 1, as the pager does;
share of bytes in chunks that shrink by >= 15% (the pager keeps the others) and the saving."""
import json, sys
import numpy as np
from compression import zstd

def shuffle(a):
    b = np.frombuffer(a.tobytes(), np.uint8)
    e = 4  # the pager default (memopro.rt.transparent elem=4)
    n = len(b) // e * e
    return b[:n].reshape(-1, e).T.copy().tobytes() + b[n:].tobytes()

def chunks(name, a):
    a = np.ascontiguousarray(a)
    raw = a.view(np.uint8).reshape(-1)
    step = (1 << 20) // a.dtype.itemsize
    flat = a.reshape(-1)
    total = good = saved = 0
    for i in range(0, flat.size, step * 4):  # every 4th chunk (sample)
        c = flat[i:i + step]
        out = len(zstd.compress(shuffle(c), level=1))
        total += c.nbytes
        if out <= 0.85 * c.nbytes:
            good += c.nbytes; saved += c.nbytes - out
    return {"array": name, "MiB": round(a.nbytes / 2**20), "compressible_share": round(good / total, 2),
            "saving_share": round(saved / total, 2)}

rows = []
rng = np.random.default_rng(0)
N = 16_000_000
cols = {"customer": rng.integers(0, 1_000_000, N), "store": rng.integers(0, 500, N).astype(np.int32),
        "time": np.sort(rng.integers(1_600_000_000, 1_700_000_000, N)),
        "price": rng.integers(100, 50_000, N) / 100, "qty": rng.integers(1, 10, N).astype(np.int16)}
rows += [chunks("df." + k, v) for k, v in cols.items()]
del cols
D, T, P = 400_000, 200_000, 120
rng = np.random.default_rng(0)
nnz = D * P
c = (rng.zipf(1.3, nnz) - 1) % T
r = np.repeat(np.arange(D), P)
v = np.log1p(rng.integers(1, 5, nnz)).astype(np.float64)
rows += [chunks("clf.cols", c), chunks("clf.rows", r), chunks("clf.vals", v)]
del c, r, v
n = 4096
y, x = np.mgrid[0:n, 0:n].astype(np.float32) / n
u = np.zeros((n, n), np.float32)
for cx, cy, s in ((0.3, 0.3, 0.05), (0.7, 0.6, 0.08), (0.5, 0.8, 0.03)):
    u += np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s * s)).astype(np.float32)
rows.append(chunks("sim.field(step0)", u))
for step in range(100):
    lap = -4 * u; lap[1:] += u[:-1]; lap[:-1] += u[1:]; lap[:, 1:] += u[:, :-1]; lap[:, :-1] += u[:, 1:]
    u += np.float32(0.2) * lap
rows.append(chunks("sim.field(step100)", u))
print(json.dumps(rows, indent=0))
