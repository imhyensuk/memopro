"""E040 W1 (docs/research/0181): an ordinary pandas program, run unchanged — a sales table of
16 M rows (customer, store, time, price, quantity), revenue per row, totals per store, the top
100 customers, and the table sorted by store and time."""

import numpy as np
import pandas as pd
from common import done, sha

N = 16_000_000
rng = np.random.default_rng(0)
df = pd.DataFrame(
    {
        "customer": rng.integers(0, 1_000_000, N),
        "store": rng.integers(0, 500, N).astype(np.int32),
        "time": np.sort(rng.integers(1_600_000_000, 1_700_000_000, N)),
        "price": rng.integers(100, 50_000, N) / 100,
        "qty": rng.integers(1, 10, N).astype(np.int16),
    }
)
df["revenue"] = df["price"] * df["qty"]
per_store = df.groupby("store")["revenue"].sum()
top = df.groupby("customer")["revenue"].sum().nlargest(100)
df = df.sort_values(["store", "time"], kind="stable")
done(
    per_store=sha(np.ascontiguousarray(per_store.to_numpy())),
    top=sha(np.ascontiguousarray(top.to_numpy()), np.ascontiguousarray(top.index.to_numpy())),
    sorted_revenue=sha(np.ascontiguousarray(df["revenue"].to_numpy())),
    total=float(df["revenue"].sum()),
)
