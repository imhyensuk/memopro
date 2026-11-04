"""E053 (docs/research/0268): E027's image-stack program (0122), unchanged in what it computes —
synthetic 16-bit images built one at a time, then whole-array NumPy passes (means, a full-size
boolean temporary, background subtraction in place, statistics, a digest of the stack).

    python image_stack.py [images, default 4096 = 2 GiB]
"""

import sys

import numpy as np
from common import done, sha

H = W = 512
n = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
rng = np.random.default_rng(0)
y = np.arange(H, dtype=np.float64)[:, None]
x = np.arange(W, dtype=np.float64)[None, :]
stack = np.empty((n, H, W), dtype=np.uint16)
for i in range(n):
    smooth = 2000 + 900 * np.sin(x / 41 + i / 57) * np.cos(y / 67 - i / 91)
    stack[i] = smooth.astype(np.uint16) + rng.integers(0, 4, size=(H, W), dtype=np.uint16)
means = stack.mean(axis=(1, 2))
counts = (stack > 2600).sum(axis=(1, 2))
background = stack.min(axis=0)
stack -= background
done(
    means=sha(means),
    counts=sha(counts),
    total=int(stack.sum(dtype=np.uint64)),
    max=int(stack.max()),
    stack=sha(stack),
)
