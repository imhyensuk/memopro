"""E069 (docs/research/0356) CSV data: deterministic fact and dimension tables.

fact: ``id,dim_id,a,b,text`` with ``id`` a permutation of 0..rows-1 (unique, so a sort has one
answer), ``dim_id`` in 0..dims-1, ``text`` 10-30 words (~140-byte rows). dim: ``dim_id,name,w`` with
every dim_id once, in a shuffled order. Rows are made in blocks of ``BLOCK`` from seeds that depend
only on the block index, so any block can be made again on its own.

    python -m experiments.e069_validation.csvgen fact.csv dim.csv --gib 30
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

BLOCK = 1 << 20
WORDS = np.array(
    [
        w.encode()
        for w in ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi", "rho", "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega", "memory", "budget", "stream", "lossless", "ceiling", "buffer", "page", "chunk", "sort", "join", "merge", "key", "value", "table", "row", "column", "index", "file", "disk"]
    ]
)
ROW_BYTES = 140  # measured mean of a fact row (for --gib)


def _ids(rows: int) -> np.ndarray:
    return np.random.default_rng(12345).permutation(rows).astype(np.int64)


def fact_blocks(rows: int, dims: int):
    """Bytes of the fact table (header first), block by block."""
    yield b"id,dim_id,a,b,text\n"
    ids = _ids(rows)
    for start in range(0, rows, BLOCK):
        n = min(BLOCK, rows - start)
        rng = np.random.default_rng([7, start // BLOCK])
        dim = rng.integers(0, dims, n)
        a = rng.integers(0, 1_000_000, n)
        b = rng.random(n)
        lens = rng.integers(10, 31, n)
        words = WORDS[rng.integers(0, len(WORDS), int(lens.sum()))]
        lines, k = [], 0
        for i in range(n):
            m = int(lens[i])
            text = b" ".join(words[k : k + m])
            k += m
            lines.append(b"%d,%d,%d,%.6f,%s" % (ids[start + i], dim[i], a[i], b[i], text))
        yield b"\n".join(lines) + b"\n"


def dim_blocks(dims: int):
    yield b"dim_id,name,w\n"
    order = np.random.default_rng(999).permutation(dims)
    for start in range(0, dims, BLOCK):
        n = min(BLOCK, dims - start)
        rng = np.random.default_rng([11, start // BLOCK])
        w = rng.random(n)
        names = WORDS[rng.integers(0, len(WORDS), n)]
        yield b"".join(
            b"%d,%s%d,%.6f\n" % (order[start + i], names[i], order[start + i] % 97, w[i])
            for i in range(n)
        )


def write(path: str, blocks) -> int:
    total = 0
    with open(path, "wb") as f:
        for b in blocks:
            f.write(b)
            total += len(b)
    return total


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("fact")
    p.add_argument("dim")
    p.add_argument("--gib", type=float, default=30.0)
    p.add_argument("--dims", type=int, default=20_000_000)
    a = p.parse_args()
    rows = int(a.gib * (1 << 30) / ROW_BYTES)
    n = write(a.fact, fact_blocks(rows, a.dims))
    m = write(a.dim, dim_blocks(a.dims))
    print(f"fact {rows} rows {n} bytes; dim {a.dims} rows {m} bytes", file=sys.stderr)


if __name__ == "__main__":
    main()
