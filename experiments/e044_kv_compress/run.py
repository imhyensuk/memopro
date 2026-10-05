"""E044 (docs/research/0203): how far does a real KV cache compress losslessly?

Qwen2.5-3B and 7B (bf16, streamed on the Apple GPU) read 2,048 WikiText tokens (E041b's text)
with use_cache; every layer's keys and values are compressed as the runtime compresses buffers
(byte shuffle by element + zstd) and in other layouts, against an entropy bound, with a few
weight matrices of the same model as reference.

    .venv/bin/python -m experiments.e044_kv_compress.run --all

Results: docs/research/data/e044/ (<model>.json, summary.md, env.json).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "research" / "data" / "e044"
TEXT = ROOT / "docs" / "research" / "data" / "e041b" / "texts_2048.json"
MODELS = {
    "q3": ("Qwen/Qwen2.5-3B-Instruct", None, 1536 << 20),
    "q7": ("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28", 2048 << 20),
}
TOKENS = 2048
WEIGHTS = ("q_proj", "k_proj", "v_proj", "down_proj")  # reference matrices of layer 0 and the last


def shuffle(u16: np.ndarray) -> bytes:
    """Bytes of bf16 values grouped by position (all high bytes, then all low bytes)."""
    b = u16.view(np.uint8).reshape(-1, 2)
    return b[:, 1].tobytes() + b[:, 0].tobytes()


def measures(u16: np.ndarray) -> dict:
    """Compressed size / raw size for one bf16 array (row-major as given)."""
    from compression import zstd

    raw = u16.tobytes()
    n = len(raw)
    sym = (u16 >> 7).astype(np.int64)  # sign + 8 exponent bits
    counts = np.bincount(sym, minlength=512)
    p = counts[counts > 0] / sym.size
    h9 = float(-(p * np.log2(p)).sum())
    return {
        "bytes": n,
        "zstd1": len(zstd.compress(raw, level=1)) / n,
        "shuffle_zstd1": len(zstd.compress(shuffle(u16), level=1)) / n,  # the runtime's codec
        "shuffle_zstd9": len(zstd.compress(shuffle(u16), level=9)) / n,
        "entropy_bound": (h9 + 7) / 16,  # sign+exponent at their entropy, mantissa as is
    }


def tensor_measures(t) -> dict:
    """Keys/values [1, kv_heads, tokens, head_dim]: as stored and channel-major."""
    import torch

    u = t.detach().cpu().contiguous().view(torch.int16).numpy().view(np.uint16)
    out = {"stored": measures(u.reshape(-1))}
    out["channel_major"] = measures(np.ascontiguousarray(np.swapaxes(u, -1, -2)).reshape(-1))
    return out


def run_model(key: str) -> dict:
    import torch
    import transformers

    import memopro.rt.torch as rtt
    from memopro.access._info import local_safetensors
    from memopro.hibernate._source import read_header

    name, revision, budget = MODELS[key]
    tok = transformers.AutoTokenizer.from_pretrained(name, revision=revision)
    text = " ".join(json.loads(TEXT.read_text()))
    ids = tok(text, return_tensors="pt").input_ids[:, :TOKENS].to("mps")
    m = rtt.stream_model(name, budget=budget, device="mps", revision=revision)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = m(input_ids=ids, use_cache=True)
    torch.mps.synchronize()
    seconds = time.perf_counter() - t0
    cache = out.past_key_values
    layers = []
    for i, layer in enumerate(cache.layers):
        layers.append({"layer": i, "keys": tensor_measures(layer.keys),
                       "values": tensor_measures(layer.values)})
    ref = {}
    files = local_safetensors(name, revision)
    regions = {}
    for f in files:
        regions.update(read_header(f))
    last = m.config.num_hidden_layers - 1
    for li in (0, last):
        for w in WEIGHTS:
            proj = "self_attn" if w.endswith(("q_proj", "k_proj", "v_proj")) else "mlp"
            r = regions[f"model.layers.{li}.{proj}.{w}.weight"]
            with open(r.path, "rb") as f:
                f.seek(r.offset)
                u = np.frombuffer(f.read(r.nbytes), dtype=np.uint16)
            ref[f"layer{li}.{w}"] = measures(u)
    m.memopro_weights.finish()
    return {"model": name, "tokens": int(ids.shape[1]), "prefill_s": seconds,
            "kv_bytes": sum(L["keys"]["stored"]["bytes"] + L["values"]["stored"]["bytes"]
                            for L in layers),
            "layers": layers, "weights": ref}


def weighted(layers: list[dict], part: str, layout: str, method: str) -> float:
    tot = sum(L[part][layout]["bytes"] for L in layers)
    return sum(L[part][layout][method] * L[part][layout]["bytes"] for L in layers) / tot


def summarize() -> str:
    rows = ["# E044 summary (0203)", "",
            "Compressed size / raw size (lower is better), weighted over all layers.", "",
            "| model | part | layout | zstd1 | shuffle+zstd1 (runtime) | shuffle+zstd9 | entropy bound | worst layer (runtime) |",
            "|---|---|---|---|---|---|---|---|"]
    best = {}
    for key in MODELS:
        path = OUT / f"{key}.json"
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        for part in ("keys", "values"):
            for layout in ("stored", "channel_major"):
                vals = [weighted(r["layers"], part, layout, m)
                        for m in ("zstd1", "shuffle_zstd1", "shuffle_zstd9", "entropy_bound")]
                worst = max(L[part][layout]["shuffle_zstd1"] for L in r["layers"])
                rows.append(f"| {r['model'].split('/')[-1]} | {part} | {layout} | "
                            + " | ".join(f"{v:.3f}" for v in vals) + f" | {worst:.3f} |")
                best[(key, part)] = min(best.get((key, part), 9), vals[1], vals[2])
        w = r["weights"]
        mean = {m: sum(x[m] for x in w.values()) / len(w)
                for m in ("zstd1", "shuffle_zstd1", "shuffle_zstd9", "entropy_bound")}
        rows.append(f"| {r['model'].split('/')[-1]} | weights (ref) | stored | "
                    + " | ".join(f"{mean[m]:.3f}" for m in mean) + " | |")
        rows.append(f"| | KV {r['kv_bytes'] / 2**20:.0f} MiB for {r['tokens']} tokens, "
                    f"prefill {r['prefill_s']:.1f} s | | | | | | |")
    overall = max(best.values()) if best else None
    if overall is not None:
        verdict = ("build (<= 0.6)" if overall <= 0.6 else
                   "modest (0.6-0.8)" if overall <= 0.8 else "little gain (> 0.8)")
        note = (f"Best practical ratio (runtime codec, either layout or level), worst of "
                f"models/parts: **{overall:.3f}** -> pre-registered rule: **{verdict}**")
        rows += ["", note]
    return "\n".join(rows) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--case")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.case:
        (OUT / f"{a.case}.json").write_text(json.dumps(run_model(a.case)))
        return
    if a.all:
        OUT.mkdir(parents=True, exist_ok=True)
        env = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "platform": platform.platform(),
               "python": sys.version.split()[0],
               "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                        capture_output=True, check=False).stdout.strip()}
        (OUT / "env.json").write_text(json.dumps(env, indent=1))
        child = {**os.environ, "HF_HOME": str(ROOT / ".cache" / "hf"), "HF_HUB_OFFLINE": "1",
                 "MallocLargeCache": "0", "TOKENIZERS_PARALLELISM": "false"}
        for key in MODELS:  # each model in a fresh process
            subprocess.run([sys.executable, "-m", "experiments.e044_kv_compress.run", "--case",
                            key], cwd=ROOT, env=child, check=True)
    text = summarize()
    (OUT / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
