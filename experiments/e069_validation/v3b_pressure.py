"""E069 V3b pressure (docs/research/0356): keep a cgroup over its memory limit.

Allocates ``--mib`` in 64 MiB blocks and rewrites them in turn until killed, so the kernel keeps
reclaiming (swapping) whatever in the cgroup is not locked.
"""

from __future__ import annotations

import sys

import numpy as np

mib = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
blocks = []
for i in range(mib // 64):
    blocks.append(np.full(64 << 20, i % 251, np.uint8))
print("pressure up", flush=True)
while True:
    for i, b in enumerate(blocks):
        b[::4096] += 1  # one write per page: each block is touched again
