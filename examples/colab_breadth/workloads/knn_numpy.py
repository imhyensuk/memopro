"""E053 (docs/research/0268): brute-force nearest neighbours with NumPy BLAS — 2.5 M database
vectors x 256 float32 (2.4 GB), three batches of 2,048 queries, each scanning the database in
blocks of 65,536 rows (sequential passes)."""

import numpy as np
from common import done, sha

N, D, Q, K, BLOCK = 2_500_000, 256, 2048, 10, 65_536
rng = np.random.default_rng(0)
db = rng.standard_normal((N, D), dtype=np.float32)
db /= np.linalg.norm(db, axis=1, keepdims=True)
results = []
for b in range(3):
    q = rng.standard_normal((Q, D), dtype=np.float32)
    best_s = np.full((Q, K), -np.inf, np.float32)
    best_i = np.zeros((Q, K), np.int64)
    for start in range(0, N, BLOCK):
        s = q @ db[start : start + BLOCK].T
        idx = np.argpartition(-s, K, axis=1)[:, :K]
        cand_s = np.take_along_axis(s, idx, 1)
        all_s = np.concatenate([best_s, cand_s], 1)
        all_i = np.concatenate([best_i, idx + start], 1)
        keep = np.argsort(-all_s, axis=1, kind="stable")[:, :K]
        best_s, best_i = np.take_along_axis(all_s, keep, 1), np.take_along_axis(all_i, keep, 1)
    results.append(best_i)
done(neighbours=sha(*results))
