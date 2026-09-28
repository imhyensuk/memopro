"""E020: quality of memopro's MPS int4 (torch `_weight_int4pack_mm`, torchao group-wise affine)
against the bf16 original, on WikiText-2 test (pre-registered in docs/research/0083).

One variant per process:
  bf16        the original, file-backed (`memopro.load(residency="file")`); writes the final
              hidden state of every window to a reference file (bf16 bits, np.memmap)
  int4-gNN    the original loaded on the CPU (memory-mapped), linear layers quantized to int4
              with group NN into MPS with memopro's own `quantize_linear` (as `load` does); the
              output head, embeddings and norms stay bf16 (identical to the original)
  int4deq-gNN the same int4 codes and scales (same torchao function), but each layer dequantizes
              them to bf16 in its forward pass and runs a bf16 matmul: identical quantization,
              faster for long inputs than the int4 kernel (0083 §2), which only computes it
  int4-cache  memopro's int4 file cache (group 64, `load(residency="file", quality="low")`), to
              check that the in-process group-64 variant is what the experiments ran

Per window (non-overlapping, --seq tokens): negative log-likelihood of every next token; for int4
also KL(bf16 || int4) and whether the most likely next token agrees, from the reference hidden
states through the same bf16 output head, in chunks of 256 positions.

Usage: .venv/bin/python -m experiments.e020_int4_quality.evaluate --model M --variant V
         --ref-dir DIR --cache-dir DIR [--seq 2048] [--windows N]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.1")  # 0080/0081, before torch
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[2] / ".cache" / "hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "research" / "data" / "e020"
DATA_ID, DATA_CFG = "Salesforce/wikitext", "wikitext-2-raw-v1"
DATA_REV = "b08601e04326c79dfdd32d625aee71d232d685c3"  # as in 0017
CHUNK = 256


def token_windows(model_id: str, seq: int, windows: int | None) -> torch.Tensor:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    ds = load_dataset(DATA_ID, DATA_CFG, revision=DATA_REV, split="test")
    text = "\n\n".join(ds["text"])  # the GPTQ/AWQ convention
    ids = AutoTokenizer.from_pretrained(model_id)(text, return_tensors="pt").input_ids[0]
    n = ids.numel() // seq
    n = n if windows is None else min(n, windows)
    return ids[: n * seq].view(n, seq)


class DequantLinear(torch.nn.Module):
    """int4 codes (two per byte) and scales as memopro quantizes them; bf16 weight per call."""

    def __init__(self, linear: torch.nn.Linear, group: int) -> None:
        from torchao.quantization.utils import groupwise_affine_quantize_tensor

        super().__init__()
        w = linear.weight.detach().to(torch.bfloat16)
        q, sz = groupwise_affine_quantize_tensor(w, 4, group, dtype=torch.bfloat16)
        self.register_buffer("packed", ((q[:, ::2] << 4) | q[:, 1::2]).to(torch.uint8).to("mps"))
        self.register_buffer("sz", sz.to("mps"))
        self.bias = None if linear.bias is None else linear.bias.detach().to("mps", torch.bfloat16)
        self.out_features, self.in_features, self.group = w.shape[0], w.shape[1], group

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from torchao.quantization.utils import groupwise_affine_dequantize_tensor

        # off the CPU, torchao takes the two-per-byte codes (high nibble first) as they are
        w = groupwise_affine_dequantize_tensor(self.packed, self.sz, 4, self.group)
        return torch.nn.functional.linear(x, w.to(x.dtype), self.bias)


def convert_group(model, group: int, dequant: bool = False) -> int:
    """memopro's `int4pack.convert` with another group size (the module fixes 64)."""
    from memopro.techniques.integrations import int4pack

    done = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if (
                type(child) is torch.nn.Linear
                and not any(part in full.lower() for part in int4pack._SKIP)
                and int4pack._fits(child, group)
            ):
                layer = (
                    DequantLinear(child, group)
                    if dequant
                    else int4pack.quantize_linear(child, "mps", group)
                )
                setattr(module, child_name, layer)
                done += 1
    model.to("mps")
    for p in model.parameters():
        if p.is_floating_point() and p.dtype != torch.bfloat16:
            p.data = p.data.to(torch.bfloat16)
    torch.mps.empty_cache()
    return done


def load_variant(model_id: str, variant: str):
    import memopro

    if variant == "bf16":
        return memopro.load(model_id, residency="file", budget="64GB!"), {}
    if variant == "int4-cache":
        from memopro.access._load import plan_load

        int4 = next(c for c in plan_load(model_id, device="mps").candidates if c.name == "quant.int4")
        need = int4.needs.device + int4.needs.host + (100 << 20)
        model = memopro.load(
            model_id, residency="file", quality="low", budget=f"{need / 1e9:.2f}GB!"
        )
        return model, {"group": 64}
    group = int(variant.split("-g")[1])
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map={"": "cpu"}
    )
    dequant = variant.startswith("int4deq")
    converted = convert_group(model, group, dequant)
    return model, {"group": group, "converted_layers": converted, "dequant": dequant}


def weight_bytes(model) -> int:
    seen, total = set(), 0
    for t in list(model.parameters()) + list(model.buffers()):
        key = (t.untyped_storage().data_ptr(), t.untyped_storage().nbytes())
        if key not in seen:
            seen.add(key)
            total += key[1]
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--ref-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--seq", type=int, default=2048)
    parser.add_argument("--windows", type=int, default=None)
    parser.add_argument("--tag", default="", help="suffix for the result file, e.g. pilot")
    args = parser.parse_args()
    import memopro

    memopro.configure(spill_dir=args.cache_dir, min_free_disk_fraction=0.0)
    windows = token_windows(args.model, args.seq, args.windows)
    n, seq = windows.shape
    t0 = time.perf_counter()
    model, info = load_variant(args.model, args.variant)
    model.eval()
    load_s = time.perf_counter() - t0
    base, head = model.model, model.lm_head
    hidden = model.config.hidden_size
    ref_path = Path(args.ref_dir) / f"{args.model.replace('/', '--')}-s{seq}-n{n}.u16"
    shape = (n, seq, hidden)
    if args.variant == "bf16":
        ref = np.memmap(ref_path, dtype=np.uint16, mode="w+", shape=shape)
    else:
        ref = np.memmap(ref_path, dtype=np.uint16, mode="r", shape=shape)
    nll_sum, kl_sum, agree, count = 0.0, 0.0, 0, 0
    per_window = []
    t1 = time.perf_counter()
    with torch.no_grad():
        for w in range(n):
            ids = windows[w : w + 1].to("mps")
            h = base(input_ids=ids).last_hidden_state[0]  # (seq, hidden), final norm applied
            if args.variant == "bf16":
                ref[w] = h.to(torch.bfloat16).view(torch.uint16).cpu().numpy()
            else:
                r = torch.from_numpy(np.array(ref[w])).view(torch.bfloat16).to("mps")
            w_nll, w_kl, w_agree = 0.0, 0.0, 0
            for s in range(0, seq - 1, CHUNK):
                e = min(s + CHUNK, seq - 1)
                logq = torch.log_softmax(head(h[s:e]).float(), -1)
                target = ids[0, s + 1 : e + 1]
                w_nll += float(-logq.gather(1, target[:, None]).sum())
                if args.variant != "bf16":
                    logp = torch.log_softmax(head(r[s:e]).float(), -1)
                    w_kl += float((logp.exp() * (logp - logq)).sum())
                    w_agree += int((logp.argmax(-1) == logq.argmax(-1)).sum())
                    del logp
                del logq
            m = seq - 1
            nll_sum, kl_sum, agree, count = nll_sum + w_nll, kl_sum + w_kl, agree + w_agree, count + m
            per_window.append({"nll": w_nll / m, "kl": w_kl / m, "agree": w_agree / m})
            if w % 20 == 0:
                print(f"  {args.variant} window {w + 1}/{n} ppl so far {math.exp(nll_sum / count):.3f}",
                      file=sys.stderr, flush=True)
    if args.variant == "bf16":
        ref.flush()
    rec = {
        "model": args.model,
        "variant": args.variant,
        **info,
        "windows": n,
        "seq": seq,
        "tokens_scored": count,
        "ppl": math.exp(nll_sum / count),
        "nll": nll_sum / count,
        "kl_vs_bf16": None if args.variant == "bf16" else kl_sum / count,
        "top1_agree_vs_bf16": None if args.variant == "bf16" else agree / count,
        "weight_bytes": weight_bytes(model),
        "load_s": load_s,
        "eval_s": time.perf_counter() - t1,
        "per_window": per_window,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"{args.model.split('/')[-1]}_{args.variant}{'_' + args.tag if args.tag else ''}.json"
    (OUT / name).write_text(json.dumps(rec, indent=1) + "\n")
    brief = {k: v for k, v in rec.items() if k != "per_window"}
    print(json.dumps(brief), flush=True)


if __name__ == "__main__":
    main()
