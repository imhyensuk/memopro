"""E027 workload (docs/research/0122): an ordinary NumPy program, run unchanged.

A stack of synthetic 16-bit grayscale images (default 2,048 of 512x512 = 1 GiB), built one image
at a time, then four passes of plain whole-array NumPy: per-image means, pixels over a threshold
(a full-size boolean temporary), background subtraction in place, and final statistics with a
SHA-256 of the whole stack. Prints one JSON line.

    python workload.py [images]
"""

import hashlib
import json
import resource
import sys
import time

import numpy as np

H = W = 512


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    t0 = time.perf_counter()
    rng = np.random.default_rng(0)
    y = np.arange(H, dtype=np.float64)[:, None]
    x = np.arange(W, dtype=np.float64)[None, :]
    stack = np.empty((n, H, W), dtype=np.uint16)
    for i in range(n):
        smooth = 2000 + 900 * np.sin(x / 41 + i / 57) * np.cos(y / 67 - i / 91)
        noise = rng.integers(0, 4, size=(H, W), dtype=np.uint16)
        stack[i] = smooth.astype(np.uint16) + noise
    built = time.perf_counter() - t0

    means = stack.mean(axis=(1, 2))
    counts = (stack > 2600).sum(axis=(1, 2))
    background = stack.min(axis=0)
    stack -= background
    total = int(stack.sum(dtype=np.uint64))
    peak = int(stack.max())
    digest = hashlib.sha256(stack).hexdigest()
    seconds = time.perf_counter() - t0

    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(
        json.dumps(
            {
                "images": n,
                "means_sha256": hashlib.sha256(means.tobytes()).hexdigest(),
                "counts_sha256": hashlib.sha256(counts.tobytes()).hexdigest(),
                "total": total,
                "max": peak,
                "stack_sha256": digest,
                "build_seconds": built,
                "seconds": seconds,
                "maxrss_bytes": maxrss * (1 if sys.platform == "darwin" else 1024),
            }
        )
    )


if __name__ == "__main__":
    main()
