"""E053 (docs/research/0268): sort 300 M float32 values — argsort (int64), gather, then a
rank-based statistic over the sorted array."""

import numpy as np
from common import done, sha

N = 300_000_000
rng = np.random.default_rng(0)
a = rng.standard_normal(N, dtype=np.float32)
order = np.argsort(a, kind="stable")
s = a[order]
del a
q = s[:: N // 1000].copy()
done(order=sha(order), sorted=sha(s), quantiles=sha(q))
