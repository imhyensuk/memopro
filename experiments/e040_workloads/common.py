"""Shared by the E040 workloads: the peak RSS so far and one JSON line at the end."""

import hashlib
import json
import resource
import sys
import time

T0 = time.perf_counter()


def rss() -> int:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r * (1 if sys.platform == "darwin" else 1024)


IMPORT_RSS = rss()  # imported right after the workload's own imports


def sha(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def done(**result) -> None:
    result |= {"seconds": time.perf_counter() - T0, "import_rss_bytes": IMPORT_RSS,
               "maxrss_bytes": rss()}
    print(json.dumps(result))
