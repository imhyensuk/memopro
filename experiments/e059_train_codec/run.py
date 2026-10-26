"""E059: one real training step with saved activations moved into runtime buffers (0319).

Qwen2.5-0.5B-Instruct bf16 on the Apple GPU, PEFT LoRA r 8 on q/k/v/o, AdamW 2e-4, no
gradient checkpointing, WikiText-2 test text in fixed batches. With ``--ceiling`` the run is
under ``memopro.enable(budget=...)``: once the process passes 75% of the ceiling, activations
saved for backward are copied into a runtime buffer, which the runtime compresses when room is
needed. One case per process. Prints one JSON line.

usage: python -m experiments.e059_train_codec.run <model folder> [--ceiling MiB] [--steps N]
"""

import argparse
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", os.path.join(os.getcwd(), ".cache", "hf"))

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--ceiling", type=int, default=0, help="MiB, whole-process ceiling (enable)")
ap.add_argument(
    "--act-budget",
    type=int,
    default=0,
    help="MiB: move every saved activation >= 1 MiB into a runtime of this budget",
)
ap.add_argument("--steps", type=int, default=6)
ap.add_argument("--seq", type=int, default=1024)
ap.add_argument("--batch", type=int, default=2)
a = ap.parse_args()

import memopro
from memopro.rt import process_footprint, process_peak_footprint

session = memopro.enable(budget=f"{a.ceiling}MiB!", numpy=False, transformers=False) if a.ceiling else None

import datasets
import peft
import torch
import transformers


def swap_used() -> int:
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, check=False).stdout
    used = out.split("used = ")[1].split("M")[0]
    return int(float(used))


swap0 = swap_used()
tok = transformers.AutoTokenizer.from_pretrained(a.model)
text = "".join(
    t for t in datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")[
        "text"
    ][:4000]
)
ids = tok(text, return_tensors="pt").input_ids[0]
need = a.steps * a.batch * a.seq
ids = ids[:need].reshape(a.steps, a.batch, a.seq)

torch.manual_seed(0)
model = transformers.AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to("mps")
model = peft.get_peft_model(
    model,
    peft.LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]),
)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
base = process_footprint()
act = None
if a.act_budget:
    from memopro.rt import Runtime
    from memopro.rt.activations import SavedActivations

    act = SavedActivations(Runtime(a.act_budget << 20), ceiling=1, pressure=0.0)
    act.hooks().__enter__()

losses, seconds = [], []
for step in range(a.steps):
    x = ids[step].to("mps")
    torch.mps.synchronize()
    t0 = time.perf_counter()
    loss = model(input_ids=x, labels=x).loss
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    torch.mps.synchronize()
    seconds.append(time.perf_counter() - t0)
    losses.append(loss.item())

out = {
    "ceiling_MiB": a.ceiling,
    "act_budget_MiB": a.act_budget,
    "seq": a.seq,
    "batch": a.batch,
    "losses": [float.hex(x) for x in losses],
    "step_seconds": [round(s, 3) for s in seconds],
    "base_footprint_MiB": round(base / 2**20) if base else None,
    "peak_footprint_MiB": round(process_peak_footprint() / 2**20),
    "swap_delta_MiB": swap_used() - swap0,
}
if session is not None:
    act = session.activations
if act is not None:
    s = act.runtime.stats()
    out |= {
        "moved": act.moved,
        "moved_MiB": round(act.moved_bytes / 2**20),
        "compressions": s["compressions"],
        "compress_in_MiB": round(s["compress_in"] / 2**20),
        "ratio": round(s["compress_out"] / max(1, s["compress_in"]), 4),
        "compress_seconds": round(s["compress_seconds"], 3),
        "decompress_seconds": round(s["decompress_seconds"], 3),
        "decompress_MiB": round(s["decompress_bytes"] / 2**20),
        "runtime_peak_MiB": round(s["peak_used"] / 2**20),
    }
print(json.dumps(out))
sys.stdout.flush()
