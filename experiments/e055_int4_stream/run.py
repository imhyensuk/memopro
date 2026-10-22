"""E055 (docs/research/0285): int4 streaming against bf16 streaming on the 8 GB M1 —
Qwen2.5-3B-Instruct, budget 1 GiB, Apple GPU, no draft. Every case runs in its own process
under /usr/bin/time -l; per-token times come from a streamer (the first 4 new tokens are warm-up).

    .venv/bin/python -m experiments.e055_int4_stream.run --cache     # build the int4 file cache
    .venv/bin/python -m experiments.e055_int4_stream.run --cases B16 I4S I4R
    .venv/bin/python -m experiments.e055_int4_stream.run --summary

Results: docs/research/data/e055/ (cases/*.json, cache.json, env.json, summary.md).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "research" / "data" / "e055"
CACHE = ROOT / ".cache"
MIB = 1 << 20
TARGET = "Qwen/Qwen2.5-3B-Instruct"
BUDGET = "1GiB"
NEW = 64
WARMUP = 4
LONG = 512  # prompt tokens for the first-token time (Q3)
PROMPTS = [
    "Explain how a hash table handles collisions.",
    "Write a Python function that checks whether a string is a palindrome, with a short docstring.",
    "Describe the water cycle for a middle-school student.",
    "List five tips for studying for a programming exam and explain each briefly.",
]
# the int4 file cache is written once with the user's consent (0285); the free-space floor is
# lowered for this run because the disk is 13% free (floor 20%) and the cache needs about 2.4 GB
SETTINGS = {"disk_writes": "allow", "min_free_disk_fraction": 0.05}


class Clock:
    """A streamer that keeps the time and ids of every new token (the first put is the prompt)."""

    def __init__(self) -> None:
        self.times: list[float] = []
        self.tokens: list[int] = []
        self.prompt = True

    def put(self, value) -> None:
        if self.prompt:
            self.prompt = False
            return
        self.times.append(time.perf_counter())
        self.tokens += value.reshape(-1).tolist()

    def end(self) -> None:
        pass


def swap_used() -> int:
    out = subprocess.run(
        ["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, check=False
    ).stdout
    return int(float(re.search(r"used = ([\d.]+)M", out).group(1)) * MIB)


def _model(name: str):
    import torch
    import transformers

    import memopro
    import memopro.rt.torch as rtt
    from memopro.techniques.integrations import int4pack

    if name == "I4R":  # int4 in memory: the compute floor (0069's conversion, as draft_model)
        model = transformers.AutoModelForCausalLM.from_pretrained(TARGET, dtype=torch.bfloat16)
        int4pack.convert(model, "mps")
        return model.eval()
    with memopro.using(**SETTINGS):
        return rtt.stream_model(
            TARGET, budget=BUDGET, device="mps", precision="int4" if name == "I4S" else "stored"
        )


def case(name: str) -> dict:
    import torch
    import transformers

    import memopro

    tok = transformers.AutoTokenizer.from_pretrained(TARGET)
    t = time.perf_counter()
    model = _model(name)
    load = time.perf_counter() - t
    streamed = hasattr(model, "memopro_runtime")

    def run(prompt, new: int) -> dict:
        clock = Clock()
        before = model.memopro_runtime.stats() if streamed else {}
        t0 = time.perf_counter()
        if streamed:
            memopro.generate(
                model, prompt, tokenizer=tok, max_new_tokens=new, streamer=clock, chat=False
            )
        else:
            with torch.no_grad():
                model.generate(
                    input_ids=prompt.to("mps"),
                    max_new_tokens=new,
                    do_sample=False,
                    streamer=clock,
                    pad_token_id=tok.eos_token_id,
                )
        torch.mps.synchronize()
        gaps = [b - a for a, b in zip(clock.times, clock.times[1:], strict=False)]
        row = {
            "prompt_tokens": prompt.shape[1],
            "first_token_s": clock.times[0] - t0 if clock.times else None,
            "token_s": gaps,
            "steady_s_per_token": statistics.median(gaps[WARMUP - 1 :]) if len(gaps) >= WARMUP else None,
            "tokens": clock.tokens,
            "text": tok.decode(clock.tokens, skip_special_tokens=True),
        }
        if streamed:
            after = model.memopro_runtime.stats()
            row["reread_bytes"] = after["reread_bytes"] - before["reread_bytes"]
            row["rereads"] = after["rereads"] - before["rereads"]
        return row

    rows = []
    for text in PROMPTS:
        msgs = [{"role": "user", "content": text}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        ids = ids["input_ids"] if hasattr(ids, "keys") else ids
        rows.append(run(ids, NEW))
    filler = " ".join(PROMPTS) * 40
    long_ids = tok(filler, return_tensors="pt").input_ids[:, :LONG]
    long_row = run(long_ids, 8)
    rec = {"load_s": load, "prompts": rows, "long": long_row}
    if streamed:
        model.memopro_weights.finish()
        rec["rt"] = model.memopro_runtime.stats()
        rec["rt_limit"] = model.memopro_runtime.limit
    return rec


def build_cache() -> None:
    """Build (or reuse) the int4 file cache and record its size and the time it took."""
    import memopro
    import memopro.rt.torch as rtt
    from memopro import residency
    from memopro.config import get_config, spill_location

    t = time.perf_counter()
    with memopro.using(**SETTINGS):
        m = rtt.stream_model(TARGET, budget=BUDGET, device="mps", precision="int4")
        path = residency.cache_path(
            spill_location(get_config()) / "fcache", TARGET, None, "stream.int4"
        )
    rec = {
        "seconds": time.perf_counter() - t,
        "path": str(path),
        "bytes": path.stat().st_size,
        "int4_layers": sum(type(x).__name__ == "Int4PackedLinear" for x in m.modules()),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cache.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec))


def _time_l(text: str) -> dict:
    rss = re.search(r"(\d+)\s+maximum resident set size", text)
    fp = re.search(r"(\d+)\s+peak memory footprint", text)
    return {
        "max_rss": int(rss.group(1)) if rss else None,
        "peak_footprint": int(fp.group(1)) if fp else None,
    }


def run_case(name: str) -> None:
    (OUT / "cases").mkdir(parents=True, exist_ok=True)
    swap0, t0 = swap_used(), time.strftime("%Y-%m-%dT%H:%M:%S")
    cmd = ["/usr/bin/time", "-l", sys.executable, "-m", "experiments.e055_int4_stream.run"]
    done = subprocess.run(
        [*cmd, "--inner", name], capture_output=True, text=True, cwd=ROOT, check=False
    )
    lines = [x for x in done.stdout.splitlines() if x.startswith("{")]
    rec = json.loads(lines[-1]) if lines else {"error": done.stderr[-3000:]}
    rec |= {"completed": "prompts" in rec, **_time_l(done.stderr)}
    rec |= {"case": name, "started": t0, "swap_before": swap0, "swap_after": swap_used()}
    (OUT / "cases" / f"{name}.json").write_text(json.dumps(rec, indent=1))
    print(name, "done", rec.get("completed"), flush=True)


def _load(name: str) -> dict | None:
    p = OUT / "cases" / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def summarize() -> str:
    lines = [
        "# E055 summary",
        "",
        (
            "| case | completed | steady s/token (median of prompts) | first token, 512-token "
            "prompt (s) | reread MiB/token | peak footprint MiB | swap Δ MiB | load s |"
        ),
        "|---|---|---|---|---|---|---|---|",
    ]
    recs = {n: _load(n) for n in ("B16", "I4S", "I4R")}
    steady = {}
    for n, r in recs.items():
        if not r or not r.get("completed"):
            lines.append(f"| {n} | {'no' if r else '—'} | | | | | | |")
            continue
        steady[n] = statistics.median(p["steady_s_per_token"] for p in r["prompts"])
        toks = sum(len(p["tokens"]) for p in r["prompts"])
        reread = sum(p.get("reread_bytes", 0) for p in r["prompts"]) / max(toks, 1) / MIB
        lines.append(
            f"| {n} | yes | {steady[n]:.3f} | {r['long']['first_token_s']:.2f} | "
            f"{reread:.0f} | {(r.get('peak_footprint') or 0) / MIB:.0f} | "
            f"{(r['swap_after'] - r['swap_before']) / MIB:+.0f} | {r['load_s']:.1f} |"
        )
    lines.append("")
    if "B16" in steady and "I4S" in steady:
        lines.append(f"- Q1 speed-up I4S over B16: {steady['B16'] / steady['I4S']:.2f}x")
    b, i = recs.get("B16"), recs.get("I4S")
    if b and i and b.get("completed") and i.get("completed"):
        for k, (pb, pi) in enumerate(zip(b["prompts"], i["prompts"], strict=True)):
            same = next(
                (j for j, (x, y) in enumerate(zip(pb["tokens"], pi["tokens"], strict=False)) if x != y),
                None,
            )
            lines.append(
                f"- prompt {k}: I4S first differs from B16 at new token "
                f"{'never' if same is None else same} (of {len(pb['tokens'])})"
            )
        r = recs.get("I4R")
        if r and r.get("completed"):
            eq = sum(p["tokens"] == q["tokens"] for p, q in zip(i["prompts"], r["prompts"], strict=True))
            lines.append(f"- I4S text equal to I4R (same int4 model): {eq}/{len(PROMPTS)}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--cases", nargs="*")
    ap.add_argument("--inner")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "HF_HOME": str(CACHE / "hf"),
            "MallocLargeCache": "0",
        }
    )
    if a.inner:
        print(json.dumps(case(a.inner)))
        return
    if a.cache:
        build_cache()
        return
    if a.cases:
        OUT.mkdir(parents=True, exist_ok=True)
        import torch
        import torchao
        import transformers

        env = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torchao": torchao.__version__,
            "transformers": transformers.__version__,
            "commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, check=False
            ).stdout.strip(),
            "dirty": subprocess.run(
                ["git", "status", "--porcelain", "python"],
                capture_output=True,
                text=True,
                cwd=ROOT,
                check=False,
            ).stdout.strip(),
            "pmset": subprocess.run(
                ["pmset", "-g", "batt"], capture_output=True, text=True, check=False
            ).stdout.strip(),
        }
        (OUT / "env.json").write_text(json.dumps(env, indent=1))
        for name in a.cases:
            run_case(name)
    (OUT / "summary.md").write_text(summarize())
    print((OUT / "summary.md").read_text())


if __name__ == "__main__":
    main()
