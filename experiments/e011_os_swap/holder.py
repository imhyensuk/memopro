"""E011 holder (process A): holds GPT-2, goes idle by one arm, resumes on command.

Reads one JSON command per line on stdin and answers one JSON line on stdout:

  {"cmd": "start"}                load the model, warm forward passes, reference output hash
  {"cmd": "act", "arm": ARM}      os: nothing | source, source_nocache, compress:
                                  memopro.hibernate.now(mode=source|compress) | reload: del model
                                  + gc (+ MPS empty_cache). source_nocache is source in a process
                                  started with MallocLargeCache=0 (set by run.py)
  {"cmd": "probe"}                share of the weights' CPU pages resident in RAM (mincore)
  {"cmd": "resume"}               wake / reload / nothing, then a forward pass; times and exactness
  {"cmd": "quit"}

``--backing mmap`` keeps the weights as transformers 5 loads them on the CPU (fp32 safetensors are
mapped from the file: clean pages the OS can drop). ``--backing anon`` copies them into anonymous
memory first, as after training, a dtype change or any in-place update.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[2] / ".cache" / "hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from transformers import GPT2LMHeadModel

from experiments.e011_os_swap.common import resident_fraction
from memopro import hibernate

MODEL = "openai-community/gpt2"
REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"


def sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def load(device: str, backing: str) -> GPT2LMHeadModel:
    model = GPT2LMHeadModel.from_pretrained(MODEL, revision=REVISION, dtype=torch.float32)
    model = model.to(device).eval()
    if backing == "anon" and device == "cpu":
        for p in model.parameters():
            p.data = p.data.clone()
    return model


def forward_hash(model: GPT2LMHeadModel, x: torch.Tensor) -> str:
    with torch.no_grad():
        out = model(x).logits.float().cpu().numpy().tobytes()
    return hashlib.sha256(out).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "mps"], required=True)
    parser.add_argument("--backing", choices=["mmap", "anon"], default="anon")
    args = parser.parse_args()
    device, backing = args.device, args.backing
    x = torch.randint(0, 50257, (1, 64), generator=torch.Generator().manual_seed(0)).to(device)
    model = handle = arm = ref = None
    params: list = []  # probed without model.parameters(), whose guard would wake the model

    def reply(**fields) -> None:
        print(json.dumps(fields), flush=True)

    for line in sys.stdin:
        cmd = json.loads(line)
        match cmd["cmd"]:
            case "start":
                t0 = time.perf_counter()
                model = load(device, backing)
                params = list(model.parameters())
                sync(device)
                load_s = time.perf_counter() - t0
                ref = forward_hash(model, x)
                warm = []
                for _ in range(3):
                    t0 = time.perf_counter()
                    forward_hash(model, x)
                    warm.append(time.perf_counter() - t0)
                nbytes = sum(p.numel() * p.element_size() for p in model.parameters())
                reply(load_s=load_s, warm_forward_s=statistics.median(warm), model_bytes=nbytes)
            case "act":
                arm = cmd["arm"]
                t0 = time.perf_counter()
                modes = {}
                if arm in ("source", "source_nocache", "compress"):
                    handle = hibernate.now(model, mode=arm.removesuffix("_nocache"))
                    modes = handle.bytes_by_mode()
                elif arm == "reload":
                    del model
                    model, params = None, []
                    gc.collect()
                    if device == "mps":
                        torch.mps.empty_cache()
                elif arm != "os":
                    raise SystemExit(f"unknown arm {arm}")
                sync(device)
                reply(act_s=time.perf_counter() - t0, bytes_by_mode=modes)
            case "probe":
                resident, total = resident_fraction(params)
                reply(resident_pages=resident, total_pages=total)
            case "resume":
                t0 = time.perf_counter()
                if arm == "reload":
                    model = load(device, backing)
                elif handle is not None:
                    handle.wake()
                sync(device)
                restore_s = time.perf_counter() - t0
                first = forward_hash(model, x)
                resume_s = time.perf_counter() - t0
                t1 = time.perf_counter()
                forward_hash(model, x)
                second_s = time.perf_counter() - t1
                reply(
                    restore_s=restore_s,
                    resume_s=resume_s,
                    second_forward_s=second_s,
                    exact=first == ref,
                )
            case "quit":
                reply(bye=True)
                return


if __name__ == "__main__":
    main()
