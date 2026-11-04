"""E053 (docs/research/0268): SciPy — power iteration on a 3 M x 3 M sparse matrix (60 M nonzeros,
20 products), then 2-D FFTs of an 8192 x 8192 float32 field (four rounds of filter and inverse)."""

import numpy as np
import scipy.sparse as sp
from common import done, sha
from scipy import fft

N, NNZ = 3_000_000, 60_000_000
rng = np.random.default_rng(0)
rows = rng.integers(0, N, NNZ, dtype=np.int32)
cols = rng.integers(0, N, NNZ, dtype=np.int32)
A = sp.csr_matrix((rng.random(NNZ, dtype=np.float32), (rows, cols)), shape=(N, N))
del rows, cols
v = np.ones(N, np.float32) / np.sqrt(N)
for _ in range(20):
    v = A @ v
    v /= np.linalg.norm(v)
field = rng.standard_normal((8192, 8192), dtype=np.float32)
k = np.fft.rfftfreq(8192)[None, :] ** 2 + np.fft.fftfreq(8192)[:, None] ** 2
for _ in range(4):
    field = fft.irfft2(fft.rfft2(field) * np.exp(-50 * k).astype(np.float32), s=field.shape)
done(eigvec=sha(v), field=sha(np.ascontiguousarray(field, np.float32)))
