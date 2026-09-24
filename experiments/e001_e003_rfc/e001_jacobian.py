"""E001: spectral norm of J_l = d f_l / d x at real activations (hypothesis H1, docs/research/0017).

sigma_l is estimated by 40 power iterations on J^T J. J v is obtained with the double-backward
trick (J v = d/du <J^T u, v>), so no forward-mode AD support is needed. Each of the 2 sequences
(T=512) is measured separately and sigma_l = max over sequences (conservative).
"""

from __future__ import annotations

import argparse
import time

import torch

from experiments._harness.env import capture, peak_rss_bytes, save_json, sha256_file
from experiments.e001_e003_rfc import common as C


def spectral_norm(blocks: C.Blocks, layer: int, x: torch.Tensor, iters: int, seed: int):
    x = x.detach().clone().requires_grad_(True)
    fx = blocks.f(layer, x)
    u = torch.zeros_like(fx, requires_grad=True)
    jt_u = torch.autograd.grad(fx, x, grad_outputs=u, create_graph=True)[0]  # linear in u
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(x.shape, generator=g)
    v /= v.norm()
    history = []
    for _ in range(iters):
        jv = torch.autograd.grad(jt_u, u, grad_outputs=v, retain_graph=True)[0]
        history.append(jv.norm().item())  # ||J v|| with ||v|| = 1 -> converges to sigma_max
        jtjv = torch.autograd.grad(fx, x, grad_outputs=jv, retain_graph=True)[0]
        v = jtjv / jtjv.norm()
    return history


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--out", default=str(C.DATA_DIR / "e001"))
    args = ap.parse_args()

    torch.manual_seed(0)
    t0 = time.time()
    model = C.load_model()
    for p in model.parameters():
        p.requires_grad_(False)
    ids_all = C.load_token_ids("test")
    ids = C.sample_batch(ids_all, args.sequences, args.seq_len, seed=0)
    blocks = C.Blocks(model, args.seq_len)
    xs = C.exact_activations(blocks, ids)
    validation = C.validate_blocks(model, blocks, ids, xs)
    assert validation["valid"], validation

    per_layer = []
    for layer in range(blocks.n_layer):
        seq_results = []
        for s in range(args.sequences):
            hist = spectral_norm(blocks, layer, xs[layer][s : s + 1], args.iters, seed=layer * 100 + s)
            tail = hist[-5:]
            seq_results.append(
                {
                    "sequence": s,
                    "sigma": hist[-1],
                    "history": hist,
                    "tail_rel_change": (max(tail) - min(tail)) / max(tail),
                }
            )
        sigma = max(r["sigma"] for r in seq_results)
        per_layer.append({"layer": layer, "sigma": sigma, "sequences": seq_results})
        print(f"block {layer:2d}: sigma = {sigma:.4f}  ({time.time() - t0:.0f}s)", flush=True)

    n_below = sum(1 for r in per_layer if r["sigma"] < 1)
    result = {
        "experiment": "E001",
        "hypothesis": "H1: sigma_l < 1 for >= 9 of 12 blocks",
        "config": vars(args) | {"token_starts_seed": 0},
        "validation": validation,
        "per_layer": per_layer,
        "summary": {
            "sigmas": [round(r["sigma"], 6) for r in per_layer],
            "n_blocks_sigma_below_1": n_below,
            "H1_pass": n_below >= 9,
        },
        "runtime_s": round(time.time() - t0, 1),
        "peak_rss_bytes": peak_rss_bytes(),
    }
    save_json(result, f"{args.out}/results.json")
    save_json(
        capture(__file__, extra={"prereg_sha256": sha256_file(C.PREREG), "argv": vars(args)}),
        f"{args.out}/env.json",
    )
    print(result["summary"])


if __name__ == "__main__":
    main()
