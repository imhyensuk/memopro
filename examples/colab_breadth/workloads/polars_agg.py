"""E053 (docs/research/0268): Polars aggregation over 40 M rows. Polars keeps columns in Arrow
buffers from its own allocator, outside NumPy."""

import numpy as np
import polars as pl
from common import done, sha

N = 40_000_000
rng = np.random.default_rng(0)
df = pl.DataFrame(
    {
        "key": rng.integers(0, 100_000, N),
        "a": rng.standard_normal(N),
        "b": rng.integers(0, 1000, N).astype(np.int32),
    }
)
out = (
    df.with_columns((pl.col("a") * pl.col("b")).alias("ab"))
    .group_by("key")
    .agg(pl.col("ab").sum(), pl.col("a").mean(), pl.len())
    .sort("key")
)
done(agg=sha(np.ascontiguousarray(out.to_numpy().astype(np.float64))), groups=out.height)
