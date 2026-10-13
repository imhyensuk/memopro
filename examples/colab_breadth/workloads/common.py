"""Shared by the E053 workloads (docs/research/0268): one JSON line at the end with the result
digests and this process's own peak memory (VmHWM; ru_maxrss survives execve, 0199)."""

import hashlib
import json
import sys
import time

T0 = time.perf_counter()


def peak_bytes() -> int:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    import resource

    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r if sys.platform == "darwin" else r * 1024


def sha(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(memoryview(a).cast("B"))
    return h.hexdigest()[:16]


def done(**result) -> None:
    result |= {"seconds": round(time.perf_counter() - T0, 2), "peak_bytes": peak_bytes()}
    print(json.dumps(result), flush=True)
