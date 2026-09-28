"""E015: foundations for RCR (0066). Pre-registered in docs/research/0067.

  Q1  cost of getting 1 GiB back after memory pressure: swapped anonymous memory vs a clean
      file mapping (plain, with madvise WILLNEED+SEQUENTIAL, warmed by large ordinary reads) vs
      sequential reads (F_NOCACHE)
  Q2  under pressure, does macOS empty volatile purgeable memory (mach PURGABLE, Metal
      purgeable buffer) before it swaps out anonymous memory of the same size?
  Q3  can a torch MPS tensor alias a read-only file mmap (no copy), compute exactly, not grow
      the footprint, and have its pages dropped and re-read under pressure?
  Q4  MPS decode-shaped linear timings: bf16/fp16 vs int8pack/int4pack vs bitsandbytes nf4

Each case is a fresh worker process (workers.py); pressure comes from the E011 pressure process
(random, incompressible). Data files live in the scratch directory given by --data (not the repo).

Usage: caffeinate -is .venv/bin/python -m experiments.e015_foundations.run --data DIR [--parts ...]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import memopro
from experiments._harness.env import capture, save_json, vm_snapshot
from experiments.e011_os_swap.common import swap_used

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "research" / "data" / "e015"
GiB = 1 << 30
Q1_BYTES = GiB
Q2_BYTES = 768 << 20
Q3_SHAPE = (16384, 32768)  # fp16: 1 GiB
Q1_MODES = ("anon", "mmap_plain", "mmap_willneed", "mmap_readwarm", "read_nocache")
CAP = 4 * GiB


class Proc:
    def __init__(self, *args: str) -> None:
        self.p = subprocess.Popen(
            [sys.executable, "-m", *args],
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )

    def read(self) -> dict:
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError(f"{self.p.args} exited with {self.p.wait()}")
        return json.loads(line)

    def ask(self, **cmd) -> dict:
        self.p.stdin.write(json.dumps(cmd) + "\n")
        self.p.stdin.flush()
        return self.read()

    def close(self) -> None:
        try:
            self.ask(cmd="quit")
        except (BrokenPipeError, RuntimeError, json.JSONDecodeError):
            pass
        try:
            self.p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.p.kill()


def worker() -> Proc:
    return Proc("experiments.e015_foundations.workers")


def pressure(nbytes: int) -> tuple[Proc, dict]:
    b = Proc("experiments.e011_os_swap.pressure", "--bytes", str(nbytes))
    return b, b.read()


def vmdelta(before: dict) -> dict:
    after = vm_snapshot()
    keys = ("Swapouts", "Swapins", "Pageins", "Pageouts", "Pages purged")
    return {k: after.get(k, 0) - before.get(k, 0) for k in keys}


def prepare_data(data: Path) -> dict:
    """1 GiB of fp16 weights, and an int4-packed 4096x4096 weight with its reference result."""
    import numpy as np
    import torch

    data.mkdir(parents=True, exist_ok=True)
    big = data / "weights_fp16.bin"
    rows, cols = Q3_SHAPE
    if not big.exists() or big.stat().st_size != rows * cols * 2:
        rng = np.random.default_rng(0)
        with open(big, "wb") as f:
            f.writelines(
                rng.standard_normal(1024 * cols).astype(np.float16).tobytes()
                for _ in range(rows // 1024)
            )
    int4 = data / "int4_packed.bin"
    if not int4.exists():
        n = k = 4096
        group = 128
        g = torch.Generator().manual_seed(0)
        wq = torch.randint(0, 16, (n, k), dtype=torch.int32, generator=g).to("mps")
        packed = torch._convert_weight_to_int4pack(
            ((wq[:, ::2] << 4) | wq[:, 1::2]).to(torch.uint8), 8
        )
        scales = torch.rand(k // group, n, 2, dtype=torch.bfloat16, generator=g).to("mps")
        x = torch.randn(1, k, dtype=torch.bfloat16, generator=g).to("mps")
        ref = torch._weight_int4pack_mm(x, packed, group, scales)
        packed.cpu().numpy().tofile(int4)
        meta = {
            "shape": list(packed.shape),
            "group": group,
            "scales_shape": list(scales.shape),
            "scales": scales.float().cpu().reshape(-1).tolist(),
            "x": x.float().cpu().reshape(-1).tolist(),
            "ref": ref.float().cpu().reshape(-1).tolist(),
        }
        (data / "int4_packed.bin.json").write_text(json.dumps(meta))
    return {"weights": str(big), "int4": str(int4)}


def q1_case(mode: str, files: dict, p_bytes: int, settle: float) -> dict:
    a = worker()
    b = None
    rec: dict = {"mode": mode, "pressure_bytes": p_bytes}
    try:
        rec["prepare"] = a.ask(cmd="q1_prepare", mode=mode, bytes=Q1_BYTES, file=files["weights"])
        b, rec["fill"] = pressure(p_bytes)
        time.sleep(settle)
        rec["probe"] = a.ask(cmd="probe")
        before, swap0 = vm_snapshot(), swap_used()
        rec["reread"] = a.ask(cmd="q1_reread")
        rec["vm"] = vmdelta(before)
        rec["swap_growth"] = swap_used() - swap0
    finally:
        if b is not None:
            b.close()
        a.close()
    return rec


def q2_case(p_bytes: int, settle: float) -> dict:
    a = worker()
    b = None
    rec: dict = {"bytes": Q2_BYTES, "pressure_bytes": p_bytes}
    try:
        rec["prepare"] = a.ask(cmd="q2_prepare", bytes=Q2_BYTES)
        time.sleep(2)
        before = vm_snapshot()
        b, rec["fill"] = pressure(p_bytes)
        time.sleep(settle)
        rec["series"] = a.ask(cmd="q2_series")["samples"]
        rec["vm"] = vmdelta(before)
    finally:
        if b is not None:
            b.close()
        a.close()
    return rec


def q3_case(files: dict, p_bytes: int, settle: float) -> dict:
    a = worker()
    b = None
    rows, cols = Q3_SHAPE
    rec: dict = {"pressure_bytes": p_bytes}
    try:
        rec["prepare"] = a.ask(
            cmd="q3_prepare", file=files["weights"], rows=rows, cols=cols, int4_file=files["int4"]
        )
        rec["probe_before"] = a.ask(cmd="q3_probe")
        before, swap0 = vm_snapshot(), swap_used()
        b, rec["fill"] = pressure(p_bytes)
        time.sleep(settle)
        rec["probe_pressed"] = a.ask(cmd="q3_probe")
        rec["vm_pressure"] = vmdelta(before)
        rec["swap_growth"] = swap_used() - swap0
        rec["recheck"] = a.ask(cmd="q3_recheck")
    finally:
        if b is not None:
            b.close()
        a.close()
    return rec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="scratch directory for the data files")
    parser.add_argument("--parts", default="q4,q1,q2,q3")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--settle", type=float, default=5.0)
    parser.add_argument("--rest", type=float, default=10.0)
    parser.add_argument("--name", default="results")
    parser.add_argument("--prep", action="store_true", help="only create the data files")
    args = parser.parse_args()
    data = Path(args.data)
    if args.prep:
        print(json.dumps(prepare_data(data)))
        return
    prep = subprocess.run(
        [sys.executable, "-m", "experiments.e015_foundations.run", "--data", str(data), "--prep"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    files = json.loads(prep.stdout.strip().splitlines()[-1])
    host = memopro.doctor(devices=False).env.host
    p_bytes = min(CAP, host.kernel_available_bytes + GiB)
    save_json(
        capture(
            __file__,
            extra={
                "argv": sys.argv,
                "pressure_bytes": p_bytes,
                "available": host.available_bytes,
                "kernel_available": host.kernel_available_bytes,
            },
        ),
        OUT / f"env_{args.name}.json",
    )
    results: dict = {"pressure_bytes": p_bytes}

    def save() -> None:
        save_json(results, OUT / f"{args.name}.json")

    for part in args.parts.split(","):
        if part == "q4":
            a = worker()
            try:
                results["q4"] = [a.ask(cmd="q4") for _ in range(args.repeats)]
            finally:
                a.close()
            print("q4 done", flush=True)
        for rep in range(args.repeats):
            if part == "q1":
                k = rep % len(Q1_MODES)
                for mode in Q1_MODES[k:] + Q1_MODES[:k]:
                    rec = q1_case(mode, files, p_bytes, args.settle)
                    rec["repeat"] = rep
                    results.setdefault("q1", []).append(rec)
                    save()
                    print(
                        f"q1 {rep} {mode:14} {rec.get('reread')} res={rec.get('probe')}", flush=True
                    )
                    time.sleep(args.rest)
            elif part == "q2":
                rec = q2_case(p_bytes, settle=max(args.settle, 8.0))
                rec["repeat"] = rep
                results.setdefault("q2", []).append(rec)
                save()
                last = rec.get("series", [{}])[-1]
                print(f"q2 {rep} last={last} vm={rec.get('vm')}", flush=True)
                time.sleep(args.rest)
            elif part == "q3":
                rec = q3_case(files, p_bytes, args.settle)
                rec["repeat"] = rep
                results.setdefault("q3", []).append(rec)
                save()
                print(
                    f"q3 {rep} prep={rec.get('prepare')} pressed={rec.get('probe_pressed')} "
                    f"recheck={rec.get('recheck')}",
                    flush=True,
                )
                time.sleep(args.rest)
        save()


if __name__ == "__main__":
    main()
