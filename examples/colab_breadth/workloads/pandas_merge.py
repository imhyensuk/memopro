"""E053 (docs/research/0268): pandas joins — 15 M orders merged with 2 M customers, then totals
per region and segment and each customer's last order (random access across the tables)."""

import numpy as np
import pandas as pd
from common import done, sha

N, C = 15_000_000, 2_000_000
rng = np.random.default_rng(0)
customers = pd.DataFrame(
    {
        "customer": np.arange(C),
        "region": rng.integers(0, 50, C).astype(np.int16),
        "segment": rng.integers(0, 6, C).astype(np.int8),
    }
)
orders = pd.DataFrame(
    {
        "customer": rng.integers(0, C, N),
        "amount": rng.gamma(2, 30, N).round(2),
        "day": rng.integers(0, 3650, N).astype(np.int32),
    }
)
m = orders.merge(customers, on="customer", how="left")
totals = m.groupby(["region", "segment"])["amount"].sum()
last = m.sort_values("day", kind="stable").groupby("customer")["day"].last()
done(
    totals=sha(np.ascontiguousarray(totals.to_numpy())),
    last=sha(np.ascontiguousarray(last.to_numpy())),
    rows=len(m),
)
