"""E069 V1 and V3a (docs/research/0356) on the 8 GB M1: Qwen2.5-7B 16-bit LoRA at whole-process
ceilings (memory over time) and generation's per-token latency. Every case runs in a fresh
process; a thread samples phys_footprint at 20 Hz.

    HF_HOME=.cache/hf .venv/bin/python -m experiments.e069_validation.v1v3a_mac [case ...]

Results: docs/research/data/e069/mac/ (cases/*.json with timelines, env.json).
"""

from __future__ import annotations

import ctypes
import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "research" / "data" / "e069" / "mac"
MODEL = ROOT / ".cache" / "models" / "Qwen2.5-7B-Instruct"  # revision a09a354
GIB = 1 << 30
CASES = {  # name: (kind, ceiling, seq or new tokens, steps or draft)
    "V1-a": ("train", 4 * GIB, 512, 10),
    "V1-b": ("train", 5 * GIB, 512, 10),
    "V1-c": ("train", int(5.5 * GIB), 2048, 5),
    "V3a-4-auto": ("gen", 4 * GIB, 64, "auto"),
    "V3a-4-none": ("gen", 4 * GIB, 64, None),
    "V3a-5-auto": ("gen", 5 * GIB, 64, "auto"),
    "V3a-5-none": ("gen", 5 * GIB, 64, None),
}
PROMPTS = [
    "Explain how a hash table handles collisions.",
    "Write a Python function that checks whether a string is a palindrome, with a short docstring.",
    "Summarize the causes of the French Revolution in three sentences.",
]


class _Rusage(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f"f{i}", ctypes.c_uint64) for i in range(36)]


def rusage() -> dict:
    """phys_footprint (f7), lifetime max (f28), disk bytes written (f17) of this process."""
    info = _Rusage()
    ctypes.CDLL("/usr/lib/libSystem.B.dylib").proc_pid_rusage(os.getpid(), 4, ctypes.byref(info))
    return {"footprint": info.f7, "max_footprint": info.f28, "disk_written": info.f17}


def swap_used() -> int:
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True,
                         check=False).stdout
    for part in out.split("  "):
        if part.strip().startswith("used"):
            return int(float(part.split("=")[1].strip().rstrip("M")) * (1 << 20))
    return -1


class Sampler(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.t0, self.samples, self.stop = time.perf_counter(), [], threading.Event()
        self.marks: list[tuple[float, str]] = []

    def run(self) -> None:
        while not self.stop.is_set():
            self.samples.append((round(time.perf_counter() - self.t0, 3), rusage()["footprint"]))
            time.sleep(0.05)

    def mark(self, what: str) -> None:
        self.marks.append((round(time.perf_counter() - self.t0, 3), what))


def texts(seq: int, steps: int) -> list[str]:
    import transformers

    from experiments.e026_rt_phase2.run import training_tokens

    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    ids, _ = training_tokens(tok, (seq - 1) * steps)
    return [tok.decode(ids[i * (seq - 1) : (i + 1) * (seq - 1)]) for i in range(steps)]


def child(name: str) -> None:
    kind, ceiling, a, b = CASES[name]
    s = Sampler()
    rec = {"case": name, "kind": kind, "ceiling": ceiling, "swap_before": swap_used(),
           "start": rusage()}
    s.start()
    import memopro

    try:
        if kind == "train":
            data = texts(a, b)
            s.mark("start")
            r = memopro.finetune(str(MODEL), data, budget=ceiling, seq_len=a)
            rec.update(losses=r.losses, loss_bits=[float(x).hex() for x in r.losses],
                       step_s=r.step_seconds, tokens=r.tokens, held_back=r.held_back,
                       status="ok")
        else:
            import transformers

            tok = transformers.AutoTokenizer.from_pretrained(MODEL)

            class Clock:  # times of the tokens a streamer gets (the prompt's put first)
                def __init__(self) -> None:
                    self.events: list[tuple[float, int]] = []

                def put(self, value) -> None:
                    self.events.append((time.perf_counter(), int(value.numel())))

                def end(self) -> None:
                    pass

            outs, runs = [], []
            for p in PROMPTS:
                c = Clock()
                t = time.perf_counter()
                s.mark(f"prompt {len(outs)}")
                outs.append(memopro.generate(str(MODEL), p, tokenizer=tok, budget=ceiling,
                                             draft=b, max_new_tokens=a, streamer=c))
                runs.append({"start": t, "events": c.events})
            rec.update(texts=outs, runs=runs, status="ok")
    except memopro.BudgetExceeded as e:
        rec.update(status="refused", error=str(e))
    s.stop.set()
    s.join()
    rec.update(end=rusage(), swap_after=swap_used(), timeline=s.samples, marks=s.marks)
    print("RESULT " + json.dumps(rec))


def main() -> None:
    if sys.argv[1:2] == ["--child"]:
        child(sys.argv[2])
        return
    (OUT / "cases").mkdir(parents=True, exist_ok=True)
    import memopro

    env = {"memopro": memopro.__version__, "platform": platform.platform(),
           "python": platform.python_version(), "model": str(MODEL),
           "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
           "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                    text=True, cwd=ROOT, check=False).stdout.strip()}
    (OUT / "env.json").write_text(json.dumps(env, indent=1))
    for name in sys.argv[1:] or list(CASES):
        path = OUT / "cases" / f"{name}.json"
        if path.exists():  # first attempt only (0131)
            print(f"{name}: kept from the first attempt")
            continue
        t = time.perf_counter()
        p = subprocess.run([sys.executable, "-m", "experiments.e069_validation.v1v3a_mac",
                            "--child", name], capture_output=True, text=True, cwd=ROOT,
                           check=False)
        rec = next((json.loads(line[7:]) for line in p.stdout.splitlines()
                    if line.startswith("RESULT ")), {"case": name, "status": "crashed"})
        rec.update(returncode=p.returncode, wall_s=time.perf_counter() - t,
                   stderr_tail=p.stderr[-3000:])
        path.write_text(json.dumps(rec))
        print(f"{name}: {rec.get('status')} {rec['wall_s']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
