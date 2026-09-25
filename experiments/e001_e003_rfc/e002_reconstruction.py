"""E002: reconstruction of block inputs from low-bit hints (hypothesis H2, docs/research/0017).

Main scenario: chained reconstruction in backward order (x_L exact, x_0 recomputed from ids).
Grid: bits {2,3,4,8} x k {0..5} x accel {plain, anderson}. Diagnostic: local reconstruction with
the exact y = x_{l+1} for k = 0..10, including the self-check residual ||x + f(x) - y|| / ||y||.
"""

from __future__ import annotations

import argparse
import time

import torch

from experiments._harness.env import capture, peak_rss_bytes, save_json, sha256_file
from experiments.e001_e003_rfc import common as C

BITS = [2, 3, 4, 8]
KS = [0, 1, 2, 3, 4, 5]
ACCELS = ["plain", "anderson"]
LOCAL_K = 10


def layer_mean(stats: list[dict]) -> float:
    return sum(s["mean"] for s in stats) / len(stats)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--out", default=str(C.DATA_DIR / "e002"))
    args = ap.parse_args()

    torch.manual_seed(0)
    t0 = time.time()
    model = C.load_model()
    for p in model.parameters():
        p.requires_grad_(False)
    ids = C.sample_batch(C.load_token_ids("test"), args.batch, args.seq_len, seed=0)
    blocks = C.Blocks(model, args.seq_len)
    xs = C.exact_activations(blocks, ids)
    validation = C.validate_blocks(model, blocks, ids, xs)
    assert validation["valid"], validation
    n = blocks.n_layer

    # ---- main: chained reconstruction grid
    chain = []
    for bits in BITS:
        for accel in ACCELS:
            for k in KS:
                if k == 0 and accel == "anderson":
                    continue  # identical to plain k=0
                for recon_x0 in (False, True):
                    xhat = C.chain_reconstruct(blocks, xs, bits, k, accel, reconstruct_x0=recon_x0)
                    layers = range(0 if recon_x0 else 1, n)
                    stats = [C.token_errors(xhat[i], xs[i]) | {"layer": i} for i in layers]
                    finite = all(torch.isfinite(xhat[i]).all().item() for i in layers)
                    chain.append(
                        {
                            "bits": bits,
                            "accel": accel if k > 0 else "none",
                            "k": k,
                            "reconstruct_x0": recon_x0,
                            "finite": finite,
                            "layer_mean_error": layer_mean(stats) if finite else float("inf"),
                            "per_layer": stats,
                        }
                    )
            print(f"chain bits={bits} accel={accel} done ({time.time() - t0:.0f}s)", flush=True)

    # ---- diagnostic: local reconstruction with exact y
    local = []
    for bits in BITS:
        for accel in ACCELS:
            for layer in range(n):
                hint = C.quantize_hint(xs[layer], bits)
                y = xs[layer + 1]
                curve = [C.token_errors(hint, xs[layer])["mean"]]
                res = [C.residual(blocks, layer, hint, y)]

                def record(j, x, _curve=curve, _res=res, _layer=layer, _y=y):
                    _curve.append(C.token_errors(x, xs[_layer])["mean"])
                    _res.append(C.residual(blocks, _layer, x, _y))

                C.solve_fixed_point(
                    lambda z, _l=layer: blocks.f(_l, z), y, hint, LOCAL_K, accel, on_iterate=record
                )
                local.append(
                    {
                        "bits": bits,
                        "accel": accel,
                        "layer": layer,
                        "error_by_k": curve,
                        "residual_by_k": res,
                    }
                )
        print(f"local bits={bits} done ({time.time() - t0:.0f}s)", flush=True)

    # ---- pre-registered H2 evaluation (bits=4, k=3, x_0 recomputed)
    def find(bits, accel, k, recon_x0=False):
        for r in chain:
            if (r["bits"], r["accel"], r["k"], r["reconstruct_x0"]) == (bits, accel, k, recon_x0):
                return r
        raise KeyError((bits, accel, k))

    base = find(4, "none", 0)["layer_mean_error"]
    cand = {a: find(4, a, 3)["layer_mean_error"] for a in ACCELS}
    selected = min(cand, key=cand.get)
    err = cand[selected]
    ratio = err / base
    h2 = {
        "k0_layer_mean_error_b4": base,
        "k3_layer_mean_error_b4": cand,
        "selected_accel": selected,
        "criterion_1_error_le_1e-2": err <= 1e-2,
        "criterion_2_ratio_to_k0_le_0.5": ratio <= 0.5,
        "ratio_to_k0": ratio,
        "H2_pass": err <= 1e-2 and ratio <= 0.5,
    }
    result = {
        "experiment": "E002",
        "config": vars(args) | {"bits": BITS, "ks": KS, "accels": ACCELS, "local_k": LOCAL_K},
        "validation": validation,
        "H2": h2,
        "chain": chain,
        "local": local,
        "runtime_s": round(time.time() - t0, 1),
        "peak_rss_bytes": peak_rss_bytes(),
    }
    save_json(result, f"{args.out}/results.json")
    save_json(
        capture(__file__, extra={"prereg_sha256": sha256_file(C.PREREG), "argv": vars(args)}),
        f"{args.out}/env.json",
    )
    print(h2)


if __name__ == "__main__":
    main()
