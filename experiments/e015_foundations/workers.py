"""E015 worker (a fresh process per case): one JSON command per line in, one JSON line out.

q1  {"cmd":"q1_prepare","mode":M,"bytes":N,"file":F}  M = anon | mmap_plain | mmap_willneed |
    mmap_readwarm | read_nocache. anon: N bytes of random anonymous memory; mmap_*: the file mapped read-only
    and read once (clean page cache); read_nocache: nothing kept
    {"cmd":"probe"}   share of the region resident
    {"cmd":"q1_reread"}  get the bytes back after pressure and time it
q2  {"cmd":"q2_prepare","bytes":N}  N bytes each of random anonymous memory, a mach PURGABLE
    region and a Metal shared buffer; both purgeable ones set volatile; a sampler records
    every 0.25 s (anon resident share, mach state, Metal state)
    {"cmd":"q2_series"}  the samples so far
q3  {"cmd":"q3_prepare","file":F,"rows":R,"cols":C,"int4_file":F4}  MPS tensors aliasing read-only
    mmaps of the files (no copy) and their results against copied tensors
    {"cmd":"q3_probe"}  resident share of the mapping, footprint
    {"cmd":"q3_recheck"}  compute again after pressure: exactness and time
q4  {"cmd":"q4"}  decode-shaped linear timings on MPS (bf16, fp16, int8pack, int4pack, bnb nf4)
"""

from __future__ import annotations

import fcntl
import json
import os
import statistics
import sys
import threading
import time

from experiments.e011_os_swap.common import rusage
from experiments.e015_foundations import vm

CHUNK = 64 << 20
F_NOCACHE = 48


def footprint() -> int:
    return rusage(os.getpid())["phys_footprint"]


class State:
    def __init__(self) -> None:
        self.region: tuple[int, int] | None = None
        self.mode = ""
        self.file = ""
        self.samples: list = []
        self.q3: dict = {}


def q1_prepare(s: State, cmd: dict) -> dict:
    s.mode, n, s.file = cmd["mode"], cmd["bytes"], cmd["file"]
    if s.mode == "anon":
        addr = vm.anon(n)
        vm.fill_random(addr, n)
        s.region = (addr, n)
    elif s.mode.startswith("mmap"):
        addr, _fd = vm.mmap_file(s.file, n)
        vm.touch(addr, n)  # into the page cache: clean file-backed pages
        s.region = (addr, n)
    return {"resident": vm.resident(*s.region) if s.region else None, "footprint": footprint()}


def q1_reread(s: State) -> dict:
    before = vm.resident(*s.region) if s.region else None
    t = time.perf_counter()
    if s.mode in ("anon", "mmap_plain"):
        vm.touch(*s.region)
    elif s.mode == "mmap_willneed":
        vm.advise(*s.region, "sequential")
        vm.advise(*s.region, "willneed")
        vm.touch(*s.region)
    elif s.mode == "mmap_readwarm":  # large ordinary reads fill the page cache the mapping shares
        fd = os.open(s.file, os.O_RDONLY)
        buf = bytearray(CHUNK)
        view = memoryview(buf)
        while os.readv(fd, [view]) > 0:
            pass
        os.close(fd)
        vm.touch(*s.region)
    else:  # read_nocache: sequential reads that bypass the cache, into one reused buffer
        fd = os.open(s.file, os.O_RDONLY)
        fcntl.fcntl(fd, F_NOCACHE, 1)
        buf = bytearray(CHUNK)
        view = memoryview(buf)
        n = 0
        while True:
            r = os.readv(fd, [view])
            if r <= 0:
                break
            n += r
        os.close(fd)
    return {"seconds": time.perf_counter() - t, "resident_before": before}


def q2_prepare(s: State, cmd: dict) -> dict:
    from experiments.e015_foundations.metal import PURGE, lib

    n = cmd["bytes"]
    a = vm.anon(n)
    vm.fill_random(a, n, 1)
    p = vm.purgeable(n)
    vm.fill_random(p, n, 2)
    vm.set_purgeable(p, 1)
    metal = lib()
    buf = metal.mp_new(n)
    vm.fill_random(metal.mp_contents(buf), n, 3)
    metal.mp_purgeable(buf, PURGE["volatile"])
    t0 = time.perf_counter()

    def sample() -> None:
        while True:
            s.samples.append(
                {
                    "t": time.perf_counter() - t0,
                    "anon": vm.resident(a, n),
                    "mach": vm.STATES.get(vm.purgeable_state(p), "?"),
                    "metal": {2: "nonvolatile", 3: "volatile", 4: "empty"}.get(
                        metal.mp_purgeable(buf, PURGE["keep"]), "?"
                    ),
                    "footprint": footprint(),
                }
            )
            time.sleep(0.25)

    threading.Thread(target=sample, daemon=True).start()
    return {"footprint": footprint()}


def q3_prepare(s: State, cmd: dict) -> dict:
    import torch

    from experiments.e015_foundations.metal import nocopy_tensor

    rows, cols = cmd["rows"], cmd["cols"]
    nbytes = rows * cols * 2
    torch.ones(1, device="mps").sum().item()  # MPS runtime up before measuring
    torch.mps.synchronize()
    x = torch.randn(1, cols, dtype=torch.float16, device="mps", generator=None)
    s.q3["x"] = x
    # reference with a copied tensor, then dropped before measuring the no-copy one
    import numpy as np

    host = np.fromfile(cmd["file"], dtype=np.float16, count=rows * cols).reshape(rows, cols)
    ref = x @ torch.from_numpy(host).to("mps").T
    torch.mps.synchronize()
    s.q3["ref"] = ref.cpu()
    del host, ref
    import gc

    gc.collect()
    torch.mps.empty_cache()
    fp_before = footprint()
    addr, _fd = vm.mmap_file(cmd["file"], nbytes)
    t = time.perf_counter()
    w, _buf = nocopy_tensor(addr, nbytes, (rows, cols), "float16")
    y = x @ w.T
    torch.mps.synchronize()
    first_s = time.perf_counter() - t
    s.q3.update(w=w, region=(addr, nbytes))
    out = {
        "accepted": w.device.type == "mps",
        "exact": bool(torch.equal(y.cpu(), s.q3["ref"])),
        "first_compute_s": first_s,
        "footprint_delta": footprint() - fp_before,
        "resident": vm.resident(addr, nbytes),
        "model_bytes": nbytes,
    }
    # int4 packed weights, also straight from a file
    if cmd.get("int4_file"):
        meta = json.loads(open(cmd["int4_file"] + ".json").read())
        n4 = os.path.getsize(cmd["int4_file"])
        a4, _fd4 = vm.mmap_file(cmd["int4_file"], n4)
        packed, _b4 = nocopy_tensor(a4, n4, tuple(meta["shape"]), "int32")
        scales = torch.tensor(meta["scales"], dtype=torch.bfloat16, device="mps").reshape(
            meta["scales_shape"]
        )
        x4 = torch.tensor(meta["x"], dtype=torch.bfloat16, device="mps").reshape(1, -1)
        y4 = torch._weight_int4pack_mm(x4, packed, meta["group"], scales)
        out["int4_exact"] = bool(
            torch.equal(y4.cpu(), torch.tensor(meta["ref"], dtype=torch.bfloat16).reshape(1, -1))
        )
    return out


def q3_recheck(s: State) -> dict:
    import torch

    addr, n = s.q3["region"]
    before = vm.resident(addr, n)
    t = time.perf_counter()
    y = s.q3["x"] @ s.q3["w"].T
    torch.mps.synchronize()
    return {
        "resident_before": before,
        "seconds": time.perf_counter() - t,
        "exact": bool(torch.equal(y.cpu(), s.q3["ref"])),
        "footprint": footprint(),
    }


def q4() -> dict:
    import torch

    shapes = {
        "1.5B up (8960x1536)": (8960, 1536),
        "1.5B down (1536x8960)": (1536, 8960),
        "3B up (11008x2048)": (11008, 2048),
        "3B down (2048x11008)": (2048, 11008),
    }
    out: dict = {}

    def bench(fn, reps=50):
        for _ in range(5):
            fn()
        torch.mps.synchronize()
        times = []
        for _ in range(reps):
            t = time.perf_counter()
            fn()
            torch.mps.synchronize()
            times.append(time.perf_counter() - t)
        return statistics.median(times)

    for name, (n, k) in shapes.items():
        row: dict = {}
        g = torch.Generator(device="mps").manual_seed(0)
        for dt in (torch.bfloat16, torch.float16):
            w = torch.randn(n, k, dtype=dt, device="mps", generator=g)
            x = torch.randn(1, k, dtype=dt, device="mps", generator=g)
            row[str(dt).removeprefix("torch.")] = bench(
                lambda w=w, x=x: torch.nn.functional.linear(x, w)
            )
        x = torch.randn(1, k, dtype=torch.bfloat16, device="mps")
        w8 = torch.randint(-128, 127, (n, k), dtype=torch.int8, device="mps")
        s8 = torch.rand(n, dtype=torch.bfloat16, device="mps")
        row["int8pack"] = bench(lambda x=x, w8=w8, s8=s8: torch._weight_int8pack_mm(x, w8, s8))
        wq = torch.randint(0, 16, (n, k), dtype=torch.int32, device="mps")
        packed = torch._convert_weight_to_int4pack(
            ((wq[:, ::2] << 4) | wq[:, 1::2]).to(torch.uint8), 8
        )
        for group in (32, 128):
            sz = torch.rand(k // group, n, 2, dtype=torch.bfloat16, device="mps")
            row[f"int4pack_g{group}"] = bench(
                lambda sz=sz, group=group, x=x, packed=packed: torch._weight_int4pack_mm(
                    x, packed, group, sz
                )
            )
        try:
            import bitsandbytes as bnb

            lin = bnb.nn.Linear4bit(
                k, n, bias=False, compute_dtype=torch.bfloat16, quant_type="nf4"
            )
            lin = lin.to("mps")
            row["bnb_nf4"] = bench(lambda lin=lin, x=x: lin(x), reps=20)
        except Exception as e:  # noqa: BLE001 - a missing back end is a result
            row["bnb_nf4_error"] = f"{type(e).__name__}: {e}"[:200]
        row["weight_bytes_bf16"] = n * k * 2
        out[name] = row
    return out


def main() -> None:
    s = State()
    for line in sys.stdin:
        cmd = json.loads(line)
        c = cmd["cmd"]
        try:
            if c == "q1_prepare":
                r = q1_prepare(s, cmd)
            elif c == "probe":
                r = {
                    "resident": vm.resident(*s.region) if s.region else None,
                    "footprint": footprint(),
                }
            elif c == "q1_reread":
                r = q1_reread(s)
            elif c == "q2_prepare":
                r = q2_prepare(s, cmd)
            elif c == "q2_series":
                r = {"samples": list(s.samples)}
            elif c == "q3_prepare":
                r = q3_prepare(s, cmd)
            elif c == "q3_probe":
                addr, n = s.q3["region"]
                r = {"resident": vm.resident(addr, n), "footprint": footprint()}
            elif c == "q3_recheck":
                r = q3_recheck(s)
            elif c == "q4":
                r = q4()
            elif c == "quit":
                print(json.dumps({"bye": True}), flush=True)
                return
            else:
                r = {"error": f"unknown command {c}"}
        except Exception as e:  # noqa: BLE001 - report, keep the protocol alive
            import traceback

            r = {
                "error": f"{type(e).__name__}: {e}"[:1000],
                "trace": traceback.format_exc()[-2000:],
            }
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
