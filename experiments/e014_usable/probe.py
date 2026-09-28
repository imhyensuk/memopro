"""E014 responsiveness probe: how the rest of the system feels while a case runs.

Holds 256 MiB of incompressible data and every 250 ms reads one byte of each page, timing it. On
a comfortable system this takes a few milliseconds; when the OS is compressing or swapping, the
probe's own pages have to come back first and the time jumps. Commands on stdin: "mark" starts a
window, "report" prints the window's statistics as JSON, "quit" exits.
"""

from __future__ import annotations

import json
import resource
import sys
import threading
import time

import numpy as np

SIZE = 256 << 20
PERIOD = 0.25


def quantile(values: list[float], f: float) -> float:
    return values[min(len(values) - 1, int(f * len(values)))]


def main() -> None:
    page = resource.getpagesize()
    buf = np.frombuffer(np.random.default_rng(0).bytes(SIZE), dtype=np.uint8).copy()
    samples: list[float] = []
    lock = threading.Lock()

    def loop() -> None:
        while True:
            t = time.perf_counter()
            int(buf[::page].sum(dtype=np.uint64))
            d = time.perf_counter() - t
            with lock:
                samples.append(d)
            time.sleep(max(0.0, PERIOD - d))

    threading.Thread(target=loop, daemon=True).start()
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        cmd = line.strip()
        if cmd == "mark":
            with lock:
                samples.clear()
            print(json.dumps({"marked": True}), flush=True)
        elif cmd == "report":
            with lock:
                s = sorted(samples)
            if not s:
                print(json.dumps({"n": 0}), flush=True)
                continue
            q = lambda f, s=s: quantile(s, f)
            print(
                json.dumps(
                    {
                        "n": len(s),
                        "p50_ms": q(0.5) * 1e3,
                        "p95_ms": q(0.95) * 1e3,
                        "max_ms": s[-1] * 1e3,
                    }
                ),
                flush=True,
            )
        elif cmd == "quit":
            return


if __name__ == "__main__":
    main()
