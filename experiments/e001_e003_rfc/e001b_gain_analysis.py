"""E001b (EXPLORATORY, not pre-registered): why E001 does not settle convergence.

E001 measured the worst-case gain sigma_l = ||J_l||_2. The fixed-point error evolves roughly as
e_{j+1} = -J e_j, so what matters is (a) how much J shrinks the *actual* hint-error direction and
(b) the spectral radius rho(J) <= sigma (plain iteration diverges asymptotically if rho > 1).
Per block and sequence we measure:
- gain ||J r|| for 5 random unit directions r
- gain ||J e|| / ||e|| for the 2/3/4-bit hint error e = Q_b(x) - x
- rho estimate: power iteration on J itself (mean of the last 10 ||J v|| values, 60 iterations)
"""

from __future__ import annotations

import time

import torch

from experiments._harness.env import capture, peak_rss_bytes, save_json, sha256_file
from experiments.e001_e003_rfc import common as C


def analyse(blocks: C.Blocks, layer: int, x: torch.Tensor, seed: int) -> dict:
    x = x.detach().clone().requires_grad_(True)
    fx = blocks.f(layer, x)
    u = torch.zeros_like(fx, requires_grad=True)
    jt_u = torch.autograd.grad(fx, x, grad_outputs=u, create_graph=True)[0]

    def jv(v):
        return torch.autograd.grad(jt_u, u, grad_outputs=v, retain_graph=True)[0]

    g = torch.Generator().manual_seed(seed)
    rand_gains = []
    for _ in range(5):
        r = torch.randn(x.shape, generator=g)
        rand_gains.append(jv(r / r.norm()).norm().item())
    hint_gains = {}
    for bits in (2, 3, 4):
        e = C.quantize_hint(x.detach(), bits) - x.detach()
        hint_gains[bits] = jv(e / e.norm()).norm().item()
    v = torch.randn(x.shape, generator=g)
    v /= v.norm()
    norms = []
    for _ in range(60):
        w = jv(v)
        norms.append(w.norm().item())
        v = w / w.norm()
    return {
        "random_gain_mean": sum(rand_gains) / len(rand_gains),
        "hint_error_gain": hint_gains,
        "rho_estimate": sum(norms[-10:]) / 10,
        "rho_tail_min_max": [min(norms[-10:]), max(norms[-10:])],
    }


def main() -> None:
    torch.manual_seed(0)
    t0 = time.time()
    model = C.load_model()
    for p in model.parameters():
        p.requires_grad_(False)
    ids = C.sample_batch(C.load_token_ids("test"), 2, 512, seed=0)  # same data as E001
    blocks = C.Blocks(model, 512)
    xs = C.exact_activations(blocks, ids)
    assert C.validate_blocks(model, blocks, ids, xs)["valid"]
    per_layer = []
    for layer in range(blocks.n_layer):
        seqs = [analyse(blocks, layer, xs[layer][s : s + 1], seed=layer * 100 + s) for s in range(2)]
        row = {
            "layer": layer,
            "random_gain_mean": max(s["random_gain_mean"] for s in seqs),
            "hint_error_gain": {b: max(s["hint_error_gain"][b] for s in seqs) for b in (2, 3, 4)},
            "rho_estimate": max(s["rho_estimate"] for s in seqs),
            "sequences": seqs,
        }
        per_layer.append(row)
        print(f"block {layer:2d}: random {row['random_gain_mean']:.3f}  hint4 "
              f"{row['hint_error_gain'][4]:.3f}  rho~{row['rho_estimate']:.3f} ({time.time()-t0:.0f}s)",
              flush=True)
    out = C.DATA_DIR / "e001"
    save_json({"experiment": "E001b (exploratory)", "per_layer": per_layer,
               "runtime_s": round(time.time() - t0, 1), "peak_rss_bytes": peak_rss_bytes()},
              out / "gain_analysis.json")
    save_json(capture(__file__, extra={"prereg_sha256": sha256_file(C.PREREG),
                                       "note": "exploratory, not pre-registered"}),
              out / "env_gain_analysis.json")


if __name__ == "__main__":
    main()
