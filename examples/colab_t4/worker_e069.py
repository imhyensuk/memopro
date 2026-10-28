"""E069 V2 worker (docs/research/0356): one case in a fresh process, result JSON to $MP_OUT.

    python worker_e069.py gen|s_mp|s_gnu|s_duck|j_mp|j_duck DATA_DIR BUDGET_BYTES

Sorts are compared by the SHA-256 of their output; joins by an order-free hash of their rows
(sum of 64-bit row hashes mod 2**64 and the row count; headers left out). Output is hashed as
it streams, never written. Bytes written: /proc/self/io (this process), wait4 (a child), and
/proc/diskstats (the whole machine).
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback

import numpy as np

KIND, DATA, BUDGET = sys.argv[1], sys.argv[2], int(sys.argv[3])
FACT, DIM, TMP = (os.path.join(DATA, n) for n in ("fact.csv", "dim.csv", "tmp"))
RESULT = {"kind": KIND, "budget": BUDGET}
P = np.uint64(1099511628211)
with np.errstate(over="ignore"):  # P**k mod 2**64, k = 0 .. 65535 (the position in a row)
    POW = np.cumprod(np.concatenate(([1], np.full((1 << 16) - 1, P))).astype(np.uint64))


def vmhwm():
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) * 1024
    return None


def io_written():
    for line in open("/proc/self/io"):
        if line.startswith("write_bytes:"):
            return int(line.split()[1])
    return None


def disk_written():
    disks = set(os.listdir("/sys/block"))
    return sum(int(f[9]) * 512 for f in (line.split() for line in open("/proc/diskstats"))
               if f[2] in disks and not f[2].startswith(("loop", "ram", "zram")))


class RowHash:
    """Order-free hash of the lines it is fed (the first line, a header, left out)."""

    def __init__(self):
        self.carry, self.total, self.rows, self.header = b"", np.uint64(0), 0, None

    def update(self, data):
        data = self.carry + bytes(data)
        cut = data.rfind(b"\n") + 1
        self.carry = data[cut:]
        if self.header is None:
            if cut == 0:
                self.carry = data
                return
            first = data.find(b"\n") + 1
            self.header, data, cut = data[:first], data[first:cut], cut - first
        else:
            data = data[:cut]
        # in pieces of about 1 MiB of whole lines: the hash's temporaries (about 40 bytes per
        # byte) stay small next to the process ceiling being measured
        at = 0
        while at < cut:
            end = data.find(b"\n", min(at + (1 << 20), cut) - 1) + 1 or cut
            self._lines(data[at:end])
            at = end

    def _lines(self, data):
        if not data:
            return
        a = np.frombuffer(data, np.uint8)
        nl = np.flatnonzero(a == 10)
        starts = np.concatenate(([0], nl[:-1] + 1))
        row = np.repeat(np.arange(len(nl)), nl - starts + 1)
        pos = np.arange(len(a)) - starts[row]
        with np.errstate(over="ignore"):
            h = np.add.reduceat((a.astype(np.uint64) + np.uint64(1)) * POW[pos], starts)
            h ^= h >> np.uint64(29)
            h *= np.uint64(0xBF58476D1CE4E5B9)
            h ^= h >> np.uint64(32)
            self.total += h.sum(dtype=np.uint64)
        self.rows += len(nl)

    def value(self):
        return f"{int(self.total):016x}:{self.rows}"


def pump(fd, sink):
    with os.fdopen(fd, "rb", buffering=0) as f:
        while True:
            b = f.read(8 << 20)
            if not b:
                break
            sink(b)


def duck(query):
    """DuckDB (memory limit = the budget, temporary files in TMP) writing CSV into a pipe."""
    import duckdb

    os.makedirs(TMP, exist_ok=True)
    con = duckdb.connect(config={"memory_limit": f"{BUDGET >> 20}MB", "temp_directory": TMP,
                                 "preserve_insertion_order": False})
    r, w = os.pipe()
    sinks = {"s_duck": hashlib.sha256(), "j_duck": RowHash()}
    h = sinks[KIND]
    t = threading.Thread(target=pump, args=(r, h.update))
    t.start()
    try:
        con.execute(f"COPY ({query}) TO '/dev/fd/{w}' (FORMAT csv, HEADER true)")
    finally:
        os.close(w)
        t.join()
    RESULT["temp_peak_note"] = "temporary files in " + TMP
    return h


def main():
    import memopro

    RESULT["memopro"] = memopro.__version__
    d0, w0, t0 = disk_written(), io_written(), time.time()
    if KIND == "gen":
        sys.path.insert(0, os.path.dirname(__file__))
        import csvgen

        os.makedirs(DATA, exist_ok=True)
        rows = int(float(os.environ.get("E069_GIB", "30")) * (1 << 30) / csvgen.ROW_BYTES)
        dims = int(os.environ.get("E069_DIMS", "20000000"))
        RESULT["fact_bytes"] = csvgen.write(FACT, csvgen.fact_blocks(rows, dims))
        RESULT["dim_bytes"] = csvgen.write(DIM, csvgen.dim_blocks(dims))
        RESULT.update(rows=rows, dims=dims)
    elif KIND == "s_mp":
        h = hashlib.sha256()
        st = memopro.sort(FACT, "id", out=h.update, budget=BUDGET)
        RESULT.update(hash=h.hexdigest(), stats=st)
    elif KIND == "j_mp":
        h = RowHash()
        st = memopro.join(FACT, DIM, "dim_id", out=h.update, budget=BUDGET)
        RESULT.update(hash=h.value(), stats=st, header=h.header.decode())
    elif KIND == "s_gnu":
        os.makedirs(TMP, exist_ok=True)
        h = hashlib.sha256()
        with open(FACT, "rb") as f:
            h.update(f.readline())
            start = f.tell()
        env = {**os.environ, "LC_ALL": "C"}
        tail = subprocess.Popen(["tail", "-c", f"+{start + 1}", FACT], stdout=subprocess.PIPE)
        srt = subprocess.Popen(["sort", "-t,", "-k1,1n", "-s", "-S", f"{BUDGET * 3 // 4 >> 20}M",
                                "-T", TMP], stdin=tail.stdout, stdout=subprocess.PIPE, env=env)
        tail.stdout.close()
        while True:
            b = srt.stdout.read(8 << 20)
            if not b:
                break
            h.update(b)
        _, status, ru = os.wait4(srt.pid, 0)
        tail.wait()
        RESULT.update(hash=h.hexdigest(), returncode=status,
                      child_written_bytes=ru.ru_oublock * 512,
                      child_peak_rss=ru.ru_maxrss * 1024)
    elif KIND == "s_duck":
        h = duck(f"SELECT * FROM read_csv('{FACT}', all_varchar=true, header=true) "
                 "ORDER BY CAST(id AS BIGINT)")
        RESULT["hash"] = h.hexdigest()
    elif KIND == "j_duck":
        h = duck(f"SELECT l.id, l.dim_id, l.a, l.b, l.text, r.name, r.w FROM "
                 f"read_csv('{FACT}', all_varchar=true, header=true) l JOIN "
                 f"read_csv('{DIM}', all_varchar=true, header=true) r "
                 "ON CAST(l.dim_id AS BIGINT) = CAST(r.dim_id AS BIGINT)")
        RESULT.update(hash=h.value(), header=h.header.decode())
    RESULT.update(seconds=time.time() - t0, peak_rss_bytes=vmhwm(),
                  written_bytes=io_written() - w0, disk_written_bytes=disk_written() - d0,
                  status="ok")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001 - recorded
        RESULT.update(status="error", error=f"{type(e).__name__}: {e}"[:1500],
                      trace=traceback.format_exc()[-3000:], peak_rss_bytes=vmhwm())
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
        text = json.dumps(RESULT, default=str)
        if os.environ.get("MP_OUT"):
            with open(os.environ["MP_OUT"], "w") as f:
                f.write(text)
        print("RESULT " + text[:2000], flush=True)
