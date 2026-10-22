"""E055 Q4 for 7B (docs/research/0285, 0290): quality of int4 streaming against bf16 streaming
on WikiText-2 test. E020's method holds the bf16 model in memory, which an 8 GB machine cannot do
for 7B, so both variants stream (``stream_model``, Apple GPU) and run in their own process:

    bf16  writes the final hidden state of every window (bf16 bits) to --ref
    int4  reads them: KL(bf16 || int4) and top-1 agreement through the same bf16 output head
          (the int4 model keeps the head as stored)

Per window (non-overlapping, --seq tokens): next-token NLL, in chunks of 256 positions as E020.

    E055_MODEL=7b .venv/bin/python -m experiments.e055_int4_stream.quality --variant bf16
    E055_MODEL=7b .venv/bin/python -m experiments.e055_int4_stream.quality --variant int4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

from experiments.e055_int4_stream.run import BUDGET, CACHE, OUT, SETTINGS, TARGET

DATA_ID, DATA_CFG = "Salesforce/wikitext", "wikitext-2-raw-v1"
DATA_REV = "b08601e04326c79dfdd32d625aee71d232d685c3"  # as E020 (0017)
CHUNK = 256


def windows(seq: int, n: int):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    ds = load_dataset(DATA_ID, DATA_CFG, revision=DATA_REV, split="test")
    ids = AutoTokenizer.from_pretrained(TARGET)("\n\n".join(ds["text"]), return_tensors="pt")
    ids = ids.input_ids[0]
    n = min(n, ids.numel() // seq)
    return ids[: n * seq].view(n, seq)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["bf16", "int4"], required=True)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--windows", type=int, default=32)
    a = ap.parse_args()
    os.environ.update({"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})
    os.environ.setdefault("HF_HOME", str(CACHE / "hf"))
    import numpy as np
    import torch

    import memopro
    import memopro.rt.torch as rtt

    with memopro.using(**SETTINGS):
        model = rtt.stream_model(
            TARGET,
            budget=BUDGET,
            device="mps",
            precision="int4" if a.variant == "int4" else "stored",
        )
    data = windows(a.seq, a.windows)
    n, seq = data.shape
    hidden = model.config.hidden_size
    ref_path = OUT / "quality_ref_hidden.bin"
    mode = "w+" if a.variant == "bf16" else "r"
    ref = np.memmap(ref_path, dtype=np.uint16, mode=mode, shape=(n, seq, hidden))
    head = model.get_output_embeddings()
    nll_sum, kl_sum, agree, count, per_window = 0.0, 0.0, 0, 0, []
    t0 = time.perf_counter()
    for w in range(n):
        ids = data[w : w + 1].to("mps")
        with torch.no_grad():
            h = model.model(input_ids=ids).last_hidden_state[0]
            if a.variant == "bf16":
                ref[w] = h.view(torch.int16).cpu().numpy().view(np.uint16)
            else:
                r = torch.from_numpy(np.array(ref[w]).view(np.int16)).view(torch.bfloat16)
                r = r.to("mps")
            target = data[w, 1:].to("mps")
            m = seq - 1
            w_nll, w_kl, w_agree = 0.0, 0.0, 0
            for s in range(0, m, CHUNK):
                e = min(m, s + CHUNK)
                logq = torch.log_softmax(head(h[s:e]).float(), -1)
                w_nll += float(-logq.gather(1, target[s:e, None]).sum())
                if a.variant == "int4":
                    logp = torch.log_softmax(head(r[s:e]).float(), -1)
                    w_kl += float((logp.exp() * (logp - logq)).sum())
                    w_agree += int((logp.argmax(-1) == logq.argmax(-1)).sum())
        nll_sum, kl_sum, agree, count = nll_sum + w_nll, kl_sum + w_kl, agree + w_agree, count + m
        per_window.append({"nll": w_nll / m, "kl": w_kl / m, "agree": w_agree / m})
        print(f"{a.variant} window {w + 1}/{n} ppl {math.exp(nll_sum / count):.3f}", flush=True)
    if a.variant == "bf16":
        ref.flush()
    rec = {
        "model": TARGET,
        "variant": a.variant,
        "budget": BUDGET,
        "seq": seq,
        "windows": n,
        "tokens_scored": count,
        "ppl": math.exp(nll_sum / count),
        "nll": nll_sum / count,
        "kl_vs_bf16": None if a.variant == "bf16" else kl_sum / count,
        "top1_agree_vs_bf16": None if a.variant == "bf16" else agree / count,
        "seconds": time.perf_counter() - t0,
        "per_window": per_window,
        "torch": torch.__version__,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"quality_{a.variant}.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k != "per_window"}))


if __name__ == "__main__":
    main()
