"""E069 V3b (docs/research/0356): an audio renderer whose sample memory must never fault.

A 48 kHz renderer makes 256-frame blocks (deadline 5.33 ms each): 8 voices read 256 frames at
random places of a ``--mib`` int16 sample buffer and are mixed. Block k must be done by
start + (k + 1) * period; a block done later is a glitch (an audible drop-out). ``--buffer``
is ``numpy`` (ordinary memory) or ``locked`` (``memopro.rt`` ``alloc(locked=True)``). Prints one
JSON line.

    python -m experiments.e069_validation.v3b_audio --buffer locked --seconds 60
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import time

import numpy as np

RATE, FRAMES, VOICES = 48_000, 256, 8


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--buffer", choices=("numpy", "locked"), required=True)
    p.add_argument("--mib", type=int, default=256)
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--fifo", type=int, default=50, help="SCHED_FIFO priority (0 = leave it)")
    a = p.parse_args()
    if a.fifo:
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(a.fifo))
    n = a.mib << 19  # int16 samples
    keep = None
    if a.buffer == "locked":
        from memopro import rt

        r = rt.Runtime(budget=(a.mib + 64) << 20, process_budget=False)
        buf = r.alloc(n * 2, dtype="int16", locked=True)
        keep = buf.pin(write=True)
        samples = np.frombuffer(keep.__enter__(), np.int16)
    else:
        samples = np.empty(n, np.int16)
    samples[:] = np.random.default_rng(0).integers(-20000, 20000, n, dtype=np.int16)

    period = FRAMES / RATE
    blocks = int(a.seconds / period)
    rng = np.random.default_rng(1)
    starts = rng.integers(0, n - FRAMES, (blocks, VOICES))
    late = np.empty(blocks)
    render = np.empty(blocks)
    out = np.zeros(FRAMES, np.float32)
    checksum = 0.0
    f0 = resource.getrusage(resource.RUSAGE_SELF).ru_majflt
    t0 = time.perf_counter() + 0.05
    for k in range(blocks):
        begin = t0 + k * period
        wait = begin - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        t = time.perf_counter()
        out[:] = 0
        for s in starts[k]:
            out += samples[s : s + FRAMES]
        checksum += float(out[0])
        done = time.perf_counter()
        render[k] = done - t
        late[k] = done - (begin + period)
    faults = resource.getrusage(resource.RUSAGE_SELF).ru_majflt - f0
    if keep is not None:
        keep.__exit__(None, None, None)
    print(json.dumps({
        "buffer": a.buffer, "mib": a.mib, "seconds": a.seconds, "blocks": blocks,
        "glitches": int((late > 0).sum()),
        "render_ms_p50": float(np.percentile(render, 50) * 1e3),
        "render_ms_p99": float(np.percentile(render, 99) * 1e3),
        "render_ms_max": float(render.max() * 1e3),
        "late_ms_max": float(late.max() * 1e3),
        "major_faults": faults, "checksum": checksum,
        "render_ms_hist": np.histogram(render * 1e3, bins=[0, .05, .1, .2, .5, 1, 2, 5.33, 10,
                                                            50, 1e9])[0].tolist(),
    }))


if __name__ == "__main__":
    main()
