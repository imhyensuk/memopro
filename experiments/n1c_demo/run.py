"""N1c demo (0032 I3): hibernate real GPT-2 on the dev machine and compare with del + reload.

Demonstration, not a pre-registered experiment. Scenarios run one after another to stay within
this 8 GB machine's memory:

  S1  GPT-2 (fp32) as loaded from the Hugging Face cache -> auto picks ``source`` (no SSD write).
      Measure reclaimed memory, hibernate/wake time and bit-exactness; then the manual baseline
      ``del model; gc.collect(); empty_cache()`` and the time to reload it with from_pretrained.
  S2  the same weights perturbed (as after fine-tuning; no original file matches) in bf16 ->
      auto may only use write-free ``compress``; SSD spill needs consent and the free-space floor.

Usage: .venv/bin/python -m experiments.n1c_demo.run --device mps
"""

from __future__ import annotations

import argparse
import gc
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[2] / ".cache" / "hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch  # noqa: E402
from transformers import GPT2LMHeadModel  # noqa: E402

import memopro  # noqa: E402
from experiments._harness.env import capture, save_json  # noqa: E402
from memopro import _core, hibernate  # noqa: E402

MODEL = "openai-community/gpt2"
REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
OUT = Path(__file__).resolve().parents[2] / "docs" / "research" / "data" / "n1c"


def memory(device: str) -> dict[str, int]:
    gc.collect()
    out = {"rss": _core.hwinfo_process_rss()}
    if device == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()
        out["mps_driver"] = int(torch.mps.driver_allocated_memory())
    return out


def diff(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    return {k: a[k] - b[k] for k in a}


def sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def load(device: str, dtype: torch.dtype = torch.float32) -> GPT2LMHeadModel:
    model = GPT2LMHeadModel.from_pretrained(MODEL, revision=REVISION, dtype=dtype)
    return model.to(device).eval()


def logits(model, x):
    with torch.no_grad():
        return model(x).logits.float().cpu()


def scenario_source(device: str, cycles: int) -> dict:
    x = torch.randint(0, 50257, (1, 64), generator=torch.Generator().manual_seed(0)).to(device)
    base = memory(device)
    t0 = time.perf_counter()
    model = load(device)
    sync(device)
    first_load_s = time.perf_counter() - t0
    ref = logits(model, x)
    loaded = memory(device)
    nbytes = sum(p.numel() * p.element_size() for p in model.parameters())

    sleep_s, wake_s, reclaimed, exact, modes = [], [], [], [], None
    for _ in range(cycles):
        before = memory(device)
        t0 = time.perf_counter()
        h = hibernate.now(model, name="gpt2")
        sync(device)
        sleep_s.append(time.perf_counter() - t0)
        reclaimed.append(diff(before, memory(device)))
        modes = h.bytes_by_mode()
        t0 = time.perf_counter()
        out = logits(model, x)  # forward wakes the model
        wake_s.append(time.perf_counter() - t0)
        exact.append(bool(torch.equal(out, ref)))

    # manual baseline: delete everything, then reload when needed again
    before = memory(device)
    del model, h
    hibernate._handles.clear()
    freed = diff(before, memory(device))
    t0 = time.perf_counter()
    model = load(device)
    sync(device)
    reload_s = time.perf_counter() - t0
    reload_exact = bool(torch.equal(logits(model, x), ref))
    del model
    return {
        "param_bytes": nbytes,
        "loaded_minus_base": diff(loaded, base),
        "first_load_s": first_load_s,
        "modes": modes,
        "hibernate_s": sleep_s,
        "hibernate_s_median": statistics.median(sleep_s),
        "wake_plus_forward_s": wake_s,
        "wake_plus_forward_s_median": statistics.median(wake_s),
        "reclaimed_per_cycle": reclaimed,
        "bit_exact_each_cycle": exact,
        "baseline_del_freed": freed,
        "baseline_reload_s": reload_s,
        "baseline_reload_exact": reload_exact,
    }


def scenario_modified(device: str) -> dict:
    x = torch.randint(0, 50257, (1, 64), generator=torch.Generator().manual_seed(1)).to(device)
    model = load(device, torch.bfloat16)
    with torch.no_grad():
        g = torch.Generator(device="cpu").manual_seed(2)
        for p in model.parameters():
            p.add_((torch.randn(p.shape, generator=g) * 1e-3).to(p.dtype).to(device))
    ref = logits(model, x)
    before = memory(device)
    t0 = time.perf_counter()
    result: dict = {}
    try:
        h = hibernate.now(model, name="gpt2-finetuned")
        sync(device)
        result["hibernate_s"] = time.perf_counter() - t0
        result["modes"] = h.bytes_by_mode()
        result["kept_awake"] = {k: len(v) for k, v in h.kept.items()}
        result["held_bytes"] = sum(r.sleeping.held_bytes for r in h.records)
        result["reclaimed"] = diff(before, memory(device))
        t0 = time.perf_counter()
        out = logits(model, x)
        result["wake_plus_forward_s"] = time.perf_counter() - t0
        result["bit_exact"] = bool(torch.equal(out, ref))
    except memopro.MemoproError as e:
        result["refused"] = str(e)
    spill_plan = [r for r in hibernate.plan(model) if r.mode == "spill"][0]
    result["spill_plan"] = {"available": spill_plan.available, "reason": spill_plan.reason}
    del model
    hibernate._handles.clear()
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--cycles", type=int, default=3)
    args = ap.parse_args()
    env = capture(__file__, extra={"config": vars(args), "model": MODEL, "revision": REVISION})
    doctor = memopro.doctor(devices=False).to_json()
    results = {
        "device": args.device,
        "doctor_before": {"host": doctor["env"]["host"], "budget": doctor["budget"]},
        "S1_source": scenario_source(args.device, args.cycles),
        "S2_modified_bf16": scenario_modified(args.device),
        "ssd_written_today": hibernate.status()["ssd_written_today"],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    save_json(results, OUT / f"demo_{args.device}.json")
    save_json(env, OUT / f"env_{args.device}.json")
    s1, s2 = results["S1_source"], results["S2_modified_bf16"]
    print(f"S1 modes {s1['modes']}, exact {s1['bit_exact_each_cycle']}")
    print(f"   reclaimed {s1['reclaimed_per_cycle'][-1]}  baseline del freed {s1['baseline_del_freed']}")
    print(
        f"   hibernate {s1['hibernate_s_median']:.3f}s  wake+forward {s1['wake_plus_forward_s_median']:.3f}s"
        f"  vs reload {s1['baseline_reload_s']:.3f}s"
    )
    print(f"S2 {s2}")


if __name__ == "__main__":
    main()
