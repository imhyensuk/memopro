"""E032 (docs/research/0127, exploratory): can post-training pentanary (5-level) weights serve
memopro, e.g. as a small resident draft for lossless speculative decoding?

Qwen2.5-1.5B-Instruct (local) is streamed through memopro.rt (CPU, 768 MiB budget). For each
weight scheme every nn.Linear (attention, MLP, lm_head) computes with weights quantized on the
fly, row chunk by row chunk; embeddings and norms stay bf16. Teacher-forced over the first
8 x 256 tokens of WikiText-2 test: top-1 agreement with bf16 (the acceptance rate of a greedy
draft) and perplexity. Each scheme in a fresh process:

    .venv/bin/python -m experiments.e032_pentanary.run --all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "research" / "data" / "e032"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"  # 0017
WINDOWS, SEQ, GROUP, ROWS = 8, 256, 64, 4096
SCHEMES = {
    # name: (levels description, bits per weight incl. one fp16 scale per group)
    "bf16": ("reference", 16.0),
    "int4_g32": ("symmetric -7..7, absmax scale, g32", 4.0 + 16 / 32),
    "penta_absmax": ("-2..2, scale absmax/2, g64", math.log2(5) + 16 / GROUP),
    "penta_mse": ("-2..2, MSE-searched scale, g64", math.log2(5) + 16 / GROUP),
    "tern_absmean": ("-1..1, scale mean|w| (BitNet), g64", math.log2(3) + 16 / GROUP),
    "tern_mse": ("-1..1, MSE-searched scale, g64", math.log2(3) + 16 / GROUP),
}


def quantize(w, scheme: str):
    """Dequantized copy (float32) of a [rows, cols] weight chunk."""
    import torch

    rows, cols = w.shape
    g = 32 if scheme == "int4_g32" else GROUP
    x = w.float().reshape(rows, cols // g, g)
    amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    if scheme == "int4_g32":
        s = amax / 7
        q = (x / s).round().clamp(-7, 7)
        return (q * s).reshape(rows, cols)
    top = 2 if scheme.startswith("penta") else 1
    if scheme == "tern_absmean":
        s = x.abs().mean(dim=-1, keepdim=True).clamp_min(1e-12)
        return ((x / s).round().clamp(-1, 1) * s).reshape(rows, cols)
    if scheme == "penta_absmax":
        s = amax / 2
        return ((x / s).round().clamp(-2, 2) * s).reshape(rows, cols)
    # MSE search over scales c * amax / top
    best, best_err = None, None
    for c in torch.linspace(0.3, 1.0, 15):
        s = amax * c / top
        d = (x / s).round().clamp(-top, top) * s
        err = (d - x).pow(2).sum(dim=-1, keepdim=True)
        if best is None:
            best, best_err = d, err
        else:
            better = err < best_err
            best = torch.where(better, d, best)
            best_err = torch.where(better, err, best_err)
    return best.reshape(rows, cols)


def quantized_forward(scheme: str):
    def forward(self, x):
        import torch

        w = self.weight  # pinned bf16 tensor put in place by memopro.rt.torch
        outs = []
        for r in range(0, w.shape[0], ROWS):
            dq = quantize(w[r : r + ROWS], scheme).to(x.dtype)
            outs.append(x @ dq.T)
        y = torch.cat(outs, dim=-1)
        return y + self.bias if self.bias is not None else y

    return forward


def tokens():
    import transformers
    from datasets import load_dataset

    ds = load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1", revision=WIKITEXT_REVISION, split="test"
    )
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    return ids[: WINDOWS * SEQ].view(WINDOWS, SEQ)


def case(scheme: str) -> dict:
    import resource

    import torch

    import memopro.rt.torch as rtt

    torch.manual_seed(0)
    data = tokens()
    model = rtt.stream_model(MODEL, budget=768 << 20)
    if scheme != "bf16":
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                m.forward = types.MethodType(quantized_forward(scheme), m)
    argmax, nll, times = [], [], []
    with torch.no_grad():
        for w in range(WINDOWS):
            t = time.perf_counter()
            logits = model(input_ids=data[w : w + 1], use_cache=False).logits[0].float()
            times.append(time.perf_counter() - t)
            argmax.append(logits.argmax(-1).tolist())
            lp = torch.log_softmax(logits[:-1], dim=-1)
            nll.append(float(-lp.gather(1, data[w, 1:, None]).mean()))
            del logits, lp
    return {
        "scheme": scheme,
        "levels": SCHEMES[scheme][0],
        "bits_per_weight": SCHEMES[scheme][1],
        "argmax": argmax,
        "nll": nll,
        "ppl": math.exp(sum(nll) / len(nll)),
        "seconds": times,
        "maxrss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "rt": model.memopro_runtime.stats(),
    }


def summarize() -> str:
    recs = {p.stem: json.loads(p.read_text()) for p in sorted((OUT / "cases").glob("*.json"))}
    ref = recs.get("bf16")
    rows = [
        "# E032 summary (0127, exploratory)",
        "",
        "| scheme | bits/weight | top-1 agreement with bf16 | expected tokens per verify (k=4) | perplexity |",
        "|---|---|---|---|---|",
    ]
    for name in SCHEMES:
        r = recs.get(name)
        if not r or "argmax" not in r:
            rows.append(
                f"| {name} | - | {r.get('error', 'missing')[-80:] if r else 'missing'} | | |"
            )
            continue
        same = total = 0
        for a, b in zip(r["argmax"], ref["argmax"], strict=True):
            for x, y in zip(a[:-1], b[:-1], strict=True):
                same += x == y
                total += 1
        p = same / total
        k = 4
        tokens_per_verify = (1 - p ** (k + 1)) / (1 - p) if p < 1 else k + 1
        rows.append(
            f"| {name} | {r['bits_per_weight']:.2f} | {p:.3f} | {tokens_per_verify:.2f} | "
            f"{r['ppl']:.2f} |"
        )
    return "\n".join(rows) + "\n"


def run_all() -> None:
    (OUT / "cases").mkdir(parents=True, exist_ok=True)
    env = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
        ).stdout.strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (OUT / "env.json").write_text(json.dumps(env, indent=1))
    child = {
        **os.environ,
        "MallocLargeCache": "0",
        "HF_HOME": str(ROOT / ".cache" / "hf"),
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    for scheme in SCHEMES:
        print(f"[{time.strftime('%H:%M:%S')}] {scheme}", flush=True)
        swap0 = swap_used()
        p = subprocess.run(
            [sys.executable, "-m", "experiments.e032_pentanary.run", "--case", scheme],
            cwd=ROOT,
            capture_output=True,
            text=True,
            env=child,
            check=False,
        )
        rec = (
            json.loads(p.stdout.strip().splitlines()[-1])
            if p.returncode == 0
            else {"scheme": scheme, "error": p.stderr[-3000:]}
        )
        rec["swap_growth"] = swap_used() - swap0
        (OUT / "cases" / f"{scheme}.json").write_text(json.dumps(rec))
        print(f"   -> {'error' if 'error' in rec else 'ok'}", flush=True)
    text = summarize()
    (OUT / "summary.md").write_text(text)
    print(text)


def swap_used() -> int:
    out = subprocess.run(
        ["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, check=False
    ).stdout
    for part in out.split("  "):
        if part.strip().startswith("used"):
            return int(float(part.split("=")[1].strip().rstrip("M")) * (1 << 20))
    return -1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--case")
    a = ap.parse_args()
    if a.case:
        print(json.dumps(case(a.case)))
    elif a.summary:
        print(summarize())
    elif a.all:
        run_all()


if __name__ == "__main__":
    main()
