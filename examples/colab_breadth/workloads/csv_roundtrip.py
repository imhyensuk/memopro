"""E053 (docs/research/0268): a pandas table of 12 M rows written to CSV, read back, cleaned and
summarized — the everyday load-transform-save loop."""

import os
import tempfile

import numpy as np
import pandas as pd
from common import done, sha

N = 12_000_000
rng = np.random.default_rng(0)
df = pd.DataFrame(
    {
        "id": np.arange(N),
        "city": rng.integers(0, 2000, N).astype(np.int32),
        "temp": rng.normal(15, 8, N).round(2),
        "rain": rng.gamma(0.5, 3, N).round(2),
        "flag": rng.integers(0, 2, N).astype(np.int8),
    }
)
path = os.path.join(tempfile.gettempdir(), "e053_table.csv")
df.to_csv(path, index=False)
del df
back = pd.read_csv(path)
os.remove(path)
back = back[back["rain"] < 20].copy()
back["feels"] = back["temp"] - 0.1 * back["rain"]
summary = back.groupby("city")[["temp", "rain", "feels"]].mean()
done(
    rows=len(back),
    summary=sha(np.ascontiguousarray(summary.to_numpy())),
    feels=sha(np.ascontiguousarray(back["feels"].to_numpy())),
)
