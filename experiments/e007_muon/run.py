"""E007: Muon feasibility on Apple Silicon (MPS), pre-registered in docs/research/0023.

A. optimizer-state bytes after one step: AdamW vs hybrid (Muon on 2D block weights + AdamW rest)
B. full fine-tuning of Adam-pretrained GPT-2 small, 60 steps, lr sweep
C. from-scratch training of a small GPT-2 (4 layers, d=256), 300 steps, lr sweep
Validation loss on fixed WikiText-2 validation sequences; step time excludes the first 5 steps.
"""

from __future__ import annotations

import argparse
import math
import time

import torch

from experiments._harness.env import capture, peak_rss_bytes, save_json, sha256_file
from experiments.e001_e003_rfc import common as C

PREREG = C.REPO / "docs" / "research" / "0023-e007-e008-preregistration.md"
OUT = C.DATA_DIR / "e007"
ANALYTIC_REDUCTION = 0.341


def is_muon_param(name: str, p: torch.Tensor) -> bool:
    return p.ndim == 2 and ".h." in name


def make_optimizers(model, kind: str, lr: float, wd: float = 0.01):
    if kind == "adamw":
        return [torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)]
    named = list(model.named_parameters())
    muon = [p for n, p in named if is_muon_param(n, p)]
    rest = [p for n, p in named if not is_muon_param(n, p)]
    return [
        torch.optim.Muon(muon, lr=lr, weight_decay=wd, momentum=0.95, nesterov=True,
                         adjust_lr_fn="match_rms_adamw"),
        torch.optim.AdamW(rest, lr=lr, weight_decay=wd),
    ]


def state_bytes(opts) -> int:
    total = 0
    for opt in opts:
        for st in opt.state.values():
            for v in st.values():
                if isinstance(v, torch.Tensor):
                    total += v.numel() * v.element_size()
    return total


def sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def val_loss(model, val_ids, device, batch: int = 4) -> float:
    model.eval()
    losses = []
    for i in range(0, val_ids.shape[0], batch):
        x = val_ids[i : i + batch].to(device)
        losses.append(model(x, labels=x).loss.item())
    model.train()
    return sum(losses) / len(losses)


def run(make_model, kind, lr, steps, batch, seq, train_ids, val_ids, device, seed=0):
    torch.manual_seed(seed)
    model = make_model().to(device)
    model.train()
    opts = make_optimizers(model, kind, lr)
    losses, times = [], []
    for step in range(steps):
        ids = C.sample_batch(train_ids, batch, seq, seed=1000 + step).to(device)
        sync(device)
        t = time.perf_counter()
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss = model(ids, labels=ids).loss
        loss.backward()
        for o in opts:
            o.step()
        sync(device)
        times.append(time.perf_counter() - t)
        losses.append(loss.item())
    result = {
        "kind": kind, "lr": lr, "steps": steps,
        "final_train_loss": sum(losses[-10:]) / 10,
        "val_loss": val_loss(model, val_ids, device),
        "finite": all(math.isfinite(x) for x in losses),
        "step_time_s": (sum(times[5:]) / len(times[5:])) if len(times) > 5 else sum(times) / len(times),
        "optimizer_state_bytes": state_bytes(opts),
        "loss_curve": losses,
    }
    if device == "mps":
        result["mps_driver_allocated_bytes"] = torch.mps.driver_allocated_memory()
    del model, opts
    if device == "mps":
        torch.mps.empty_cache()
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device
    t0 = time.time()
    train_ids = C.load_token_ids("train")
    val_all = C.load_token_ids("validation")

    # ---- A. memory accounting after one step (full GPT-2 small)
    mem = {}
    for kind in ("adamw", "hybrid"):
        r = run(C.load_model, kind, 1e-4, 1, 2, 256, train_ids, C.sample_batch(val_all, 4, 256, 7),
                dev)
        mem[kind] = r["optimizer_state_bytes"]
    measured = 1 - mem["hybrid"] / mem["adamw"]
    part_a = {"state_bytes": mem, "measured_reduction": measured,
              "analytic_reduction": ANALYTIC_REDUCTION,
              "M1_pass": abs(measured - ANALYTIC_REDUCTION) <= 0.02}
    print("A:", part_a, flush=True)

    # ---- B. full fine-tuning of the Adam-pretrained GPT-2 small
    val_b = C.sample_batch(val_all, 16, 256, seed=7)
    part_b = []
    for kind in ("adamw", "hybrid"):
        for lr in (3e-5, 1e-4, 3e-4):
            r = run(C.load_model, kind, lr, 60, 2, 256, train_ids, val_b, dev)
            part_b.append(r)
            print(f"B {kind} lr={lr}: val {r['val_loss']:.4f} step {r['step_time_s']:.3f}s "
                  f"({time.time() - t0:.0f}s)", flush=True)

    # ---- C. from-scratch small GPT-2
    from transformers import GPT2Config, GPT2LMHeadModel

    cfg = GPT2Config(n_layer=4, n_embd=256, n_head=4, n_positions=128)

    def tiny():
        torch.manual_seed(0)
        return GPT2LMHeadModel(cfg)

    val_c = C.sample_batch(val_all, 16, 128, seed=7)
    part_c = []
    for kind in ("adamw", "hybrid"):
        for lr in (3e-4, 1e-3, 3e-3):
            r = run(tiny, kind, lr, 300, 8, 128, train_ids, val_c, dev)
            part_c.append(r)
            print(f"C {kind} lr={lr}: val {r['val_loss']:.4f} step {r['step_time_s']:.4f}s "
                  f"({time.time() - t0:.0f}s)", flush=True)

    def best(rows, kind):
        return min((r for r in rows if r["kind"] == kind and r["finite"]), key=lambda r: r["val_loss"])

    b_aw, b_hy = best(part_b, "adamw"), best(part_b, "hybrid")
    c_aw, c_hy = best(part_c, "adamw"), best(part_c, "hybrid")
    criteria = {
        "M1_memory_matches_analytic": part_a["M1_pass"],
        "M2_all_finite_on_device": all(r["finite"] for r in part_b + part_c),
        "M3a_from_scratch_hybrid_le_adamw_x1.02": c_hy["val_loss"] <= c_aw["val_loss"] * 1.02,
        "M3b_full_ft_val": {"adamw": b_aw["val_loss"], "adamw_lr": b_aw["lr"],
                            "hybrid": b_hy["val_loss"], "hybrid_lr": b_hy["lr"]},
        "M3a_values": {"adamw": c_aw["val_loss"], "adamw_lr": c_aw["lr"],
                       "hybrid": c_hy["val_loss"], "hybrid_lr": c_hy["lr"]},
        "M4_step_time_ratio_B": b_hy["step_time_s"] / b_aw["step_time_s"],
        "M4_step_time_ratio_C": c_hy["step_time_s"] / c_aw["step_time_s"],
    }
    criteria["M4_pass"] = max(criteria["M4_step_time_ratio_B"], criteria["M4_step_time_ratio_C"]) <= 1.25
    criteria["decision_add_as_semantics_suggestion"] = (
        criteria["M2_all_finite_on_device"] and criteria["M3a_from_scratch_hybrid_le_adamw_x1.02"]
        and criteria["M4_pass"])
    result = {"experiment": "E007", "device": dev, "part_a": part_a, "part_b": part_b,
              "part_c": part_c, "criteria": criteria, "runtime_s": round(time.time() - t0, 1),
              "peak_rss_bytes": peak_rss_bytes()}
    save_json(result, OUT / "results.json")
    save_json(capture(__file__, extra={"prereg_sha256": sha256_file(PREREG), "device": dev}),
              OUT / "env.json")
    print(criteria)


if __name__ == "__main__":
    main()
