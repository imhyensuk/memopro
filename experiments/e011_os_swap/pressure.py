"""E011 pressure (process B): other work that needs memory while A is idle.

Fills ``--bytes`` of incompressible (random) data in 64 MiB chunks and reports the time, then on
{"cmd": "pass"} reads one byte of every page (faulting back whatever the OS moved out) and
reports that time; {"cmd": "quit"} exits.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time

import numpy as np

CHUNK = 64 << 20


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bytes", type=int, required=True)
    size = parser.parse_args().bytes
    page = resource.getpagesize()
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    buf = np.empty(size, dtype=np.uint8)
    words = buf[: size - size % 8].view(np.uint64)
    chunk_s = []
    per = CHUNK // 8
    for start in range(0, len(words), per):
        t1 = time.perf_counter()
        n = min(per, len(words) - start)
        words[start : start + n] = rng.bit_generator.random_raw(n)
        chunk_s.append(time.perf_counter() - t1)
    fill_s = time.perf_counter() - t0
    print(json.dumps({"fill_s": fill_s, "max_chunk_s": max(chunk_s)}), flush=True)
    for line in sys.stdin:
        cmd = json.loads(line)["cmd"]
        if cmd == "pass":
            t0 = time.perf_counter()
            checksum = int(buf[::page].sum(dtype=np.uint64))
            print(
                json.dumps({"pass_s": time.perf_counter() - t0, "checksum": checksum}), flush=True
            )
        elif cmd == "quit":
            return


if __name__ == "__main__":
    main()
