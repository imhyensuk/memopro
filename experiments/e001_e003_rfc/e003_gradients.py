"""E003: gradient accuracy when backprop uses reconstructed block inputs (H3, docs/research/0017).

Reference g*: standard autograd through the full model. RFC path: head backward from the exact x_L,
then block-by-block backward where block l is recomputed from its reconstructed input x_hat_l,
then the embedding backward. The manual path with exact inputs must match g* (cosine > 0.999999),
otherwise the implementation is wrong and the run is invalid.
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from experiments._harness.env import capture, peak_rss_bytes, save_json, sha256_file
from experiments.e001_e003_rfc import common as C

EXPLORE_BITS = [2, 3, 4]
EXPLORE_KS = [0, 1, 2, 3, 5]


def reference_gradients(model, ids):
    model.zero_grad(set_to_none=True)
    loss = model(ids, labels=ids).loss
    loss.backward()
    return loss.item(), grads_of(model)


def grads_of(model) -> dict[str, torch.Tensor]:
    # named_parameters() de-duplicates tied weights (lm_head.weight is transformer.wte.weight)
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


def manual_gradients(model, blocks: C.Blocks, ids, xs, xhat):
    model.zero_grad(set_to_none=True)
    n = blocks.n_layer
    x_last = xs[n].detach().clone().requires_grad_(True)
    loss = blocks.head_loss(x_last, ids)
    loss.backward()
    g = x_last.grad
    for layer in range(n - 1, -1, -1):
        x_in = xhat[layer].detach().clone().requires_grad_(True)
        blocks.block(layer, x_in).backward(g)
        g = x_in.grad
    blocks.embed(ids).backward(g)
    return loss.item(), grads_of(model)


def compare(g: dict, g_ref: dict) -> dict:
    names = sorted(g_ref)
    a = torch.cat([g[n].reshape(-1).double() for n in names])
    b = torch.cat([g_ref[n].reshape(-1).double() for n in names])
    per = []
    for n in names:
        x, y = g[n].reshape(-1).double(), g_ref[n].reshape(-1).double()
        per.append((torch.dot(x, y) / (x.norm() * y.norm()).clamp_min(1e-300)).item())
    per_sorted = sorted(per)
    return {
        "cosine": (torch.dot(a, b) / (a.norm() * b.norm())).item(),
        "rel_l2_error": ((a - b).norm() / b.norm()).item(),
        "per_tensor_cosine_min": per_sorted[0],
        "per_tensor_cosine_median": per_sorted[len(per_sorted) // 2],
        "worst_tensors": [names[i] for i in sorted(range(len(per)), key=per.__getitem__)[:5]],
        "finite": bool(torch.isfinite(a).all().item()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--e002", default=str(C.DATA_DIR / "e002" / "results.json"))
    ap.add_argument("--out", default=str(C.DATA_DIR / "e003"))
    args = ap.parse_args()

    selected = json.loads(open(args.e002).read())["H2"]["selected_accel"]  # pre-registered rule
    torch.manual_seed(0)
    t0 = time.time()
    model = C.load_model()
    ids = C.sample_batch(C.load_token_ids("test"), args.batch, args.seq_len, seed=0)
    blocks = C.Blocks(model, args.seq_len)
    xs = C.exact_activations(blocks, ids)
    validation = C.validate_blocks(model, blocks, ids, xs)
    assert validation["valid"], validation

    loss_ref, g_ref = reference_gradients(model, ids)
    loss_man, g_man = manual_gradients(model, blocks, ids, xs, xs)
    sanity = compare(g_man, g_ref) | {"loss_ref": loss_ref, "loss_manual": loss_man}
    sanity["valid"] = sanity["cosine"] > 0.999999
    print("sanity (exact inputs):", {k: sanity[k] for k in ("cosine", "rel_l2_error", "valid")})
    assert sanity["valid"], sanity

    runs = []
    for bits in EXPLORE_BITS:
        for k in EXPLORE_KS:
            xhat = C.chain_reconstruct(blocks, xs, bits, k, selected)
            _, g = manual_gradients(model, blocks, ids, xs, xhat)
            cmp = compare(g, g_ref)
            runs.append({"bits": bits, "k": k, "accel": selected if k else "none"} | cmp)
            print(f"bits={bits} k={k}: cosine={cmp['cosine']:.6f} rel_err={cmp['rel_l2_error']:.4f}"
                  f" ({time.time() - t0:.0f}s)", flush=True)

    main_run = next(r for r in runs if r["bits"] == 4 and r["k"] == 3)
    control = next(r for r in runs if r["bits"] == 4 and r["k"] == 0)
    h3 = {
        "selected_accel_from_E002": selected,
        "main_cosine_b4_k3": main_run["cosine"],
        "control_cosine_b4_k0": control["cosine"],
        "H3_pass": main_run["cosine"] >= 0.999,
    }
    result = {
        "experiment": "E003",
        "config": vars(args) | {"explore_bits": EXPLORE_BITS, "explore_ks": EXPLORE_KS},
        "validation": validation,
        "sanity_exact_inputs": sanity,
        "H3": h3,
        "runs": runs,
        "runtime_s": round(time.time() - t0, 1),
        "peak_rss_bytes": peak_rss_bytes(),
    }
    save_json(result, f"{args.out}/results.json")
    save_json(
        capture(__file__, extra={"prereg_sha256": sha256_file(C.PREREG), "argv": vars(args)}),
        f"{args.out}/env.json",
    )
    print(h3)


if __name__ == "__main__":
    main()
