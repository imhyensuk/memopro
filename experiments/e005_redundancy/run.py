"""E005: redundancy census pilot and OS-compression baseline (docs/research/0017, 0015 P3).

GPT-2 small in train() mode, AdamW, 20 steps on WikiText-2 train (batch 2 x 256), then one more
forward/backward whose autograd-saved tensors are captured. For each category we measure:
- stored bits/element and order-0 entropy summed over byte planes (bits/element)
- OS-style proxies: LZ4 per 16 KiB page (macOS page size), zstd-3 per 4 KiB page (Linux zram/zswap)
- zstd-3 on the whole chunk, and byte-shuffle + zstd-3 (memopro candidate for hibernate lossless)
- the same for a bf16 cast of float categories (reference)
"""

from __future__ import annotations

import argparse
import math
import time

import lz4.block
import numpy as np
import torch
import zstandard as zstd

from experiments._harness.env import capture, peak_rss_bytes, save_json, sha256_file
from experiments.e001_e003_rfc import common as C

CAP_BYTES = 64 * 2**20
CHUNK = 16 * 2**20


def sample_bytes(tensors: list[torch.Tensor], cap: int = CAP_BYTES) -> tuple[np.ndarray, int]:
    """Concatenate proportional contiguous prefixes of each tensor (keeps local structure)."""
    total = sum(t.numel() * t.element_size() for t in tensors)
    frac = min(1.0, cap / max(total, 1))
    parts = []
    for t in tensors:
        flat = t.detach().reshape(-1).contiguous().cpu()
        n = max(1, math.ceil(flat.numel() * frac))
        parts.append(flat[:n])
    cat = torch.cat([p.view(torch.uint8) if p.dtype == torch.bool else p for p in parts])
    if cat.dtype == torch.bfloat16:
        arr = cat.view(torch.int16).numpy()
    else:
        arr = cat.numpy()
    return arr, total


def byte_plane_entropy(arr: np.ndarray) -> float:
    b = arr.view(np.uint8).reshape(-1, arr.itemsize)
    bits = 0.0
    for j in range(arr.itemsize):
        counts = np.bincount(b[:, j], minlength=256).astype(np.float64)
        p = counts[counts > 0] / counts.sum()
        bits += float(-(p * np.log2(p)).sum())
    return bits


def ratio_whole(buf: bytes, cctx) -> float:
    out = sum(len(cctx.compress(buf[i : i + CHUNK])) for i in range(0, len(buf), CHUNK))
    return len(buf) / out


def ratio_shuffle(arr: np.ndarray, cctx) -> float:
    per_chunk = CHUNK // arr.itemsize
    out = 0
    for i in range(0, arr.size, per_chunk):
        part = arr[i : i + per_chunk]
        planes = part.view(np.uint8).reshape(-1, arr.itemsize).T.copy()  # byte shuffle
        out += len(cctx.compress(planes.tobytes()))
    return arr.nbytes / out


def ratio_pages_lz4(buf: bytes, page: int = 16384) -> float:
    out = 0
    for i in range(0, len(buf), page):
        p = buf[i : i + page]
        out += min(len(lz4.block.compress(p, store_size=False)), len(p))
    return len(buf) / out


def ratio_pages_zstd(buf: bytes, cctx, page: int = 4096) -> float:
    out = 0
    for i in range(0, len(buf), page):
        p = buf[i : i + page]
        out += min(len(cctx.compress(p)), len(p))
    return len(buf) / out


def measure(name: str, tensors: list[torch.Tensor], cctx) -> dict:
    if not tensors:
        return {"category": name, "empty": True}
    arr, total = sample_bytes(tensors)
    buf = arr.tobytes()
    t = time.time()
    shuffled = ratio_shuffle(arr, cctx)
    shuffle_mb_s = arr.nbytes / 2**20 / (time.time() - t)
    entropy = byte_plane_entropy(arr)
    row = {
        "category": name,
        "dtype": str(tensors[0].dtype).replace("torch.", ""),
        "n_tensors": len(tensors),
        "total_bytes": total,
        "sampled_bytes": arr.nbytes,
        "stored_bits_per_element": arr.itemsize * 8,
        "entropy_bits_per_element": entropy,
        "entropy_bound_ratio": arr.itemsize * 8 / entropy if entropy > 0 else float("inf"),
        "ratio_os_lz4_16k_pages": ratio_pages_lz4(buf),
        "ratio_os_zstd_4k_pages": ratio_pages_zstd(buf, cctx),
        "ratio_zstd_whole": ratio_whole(buf, cctx),
        "ratio_shuffle_zstd": shuffled,
        "shuffle_zstd_mb_per_s_python": shuffle_mb_s,
    }
    if tensors[0].is_floating_point() and tensors[0].dtype == torch.float32:
        bf = [x.detach().to(torch.bfloat16) for x in tensors]
        arr16, _ = sample_bytes(bf)
        buf16 = arr16.tobytes()
        row["bf16"] = {
            "entropy_bits_per_element": byte_plane_entropy(arr16),
            "ratio_os_lz4_16k_pages": ratio_pages_lz4(buf16),
            "ratio_shuffle_zstd": ratio_shuffle(arr16, cctx),
            "ratio_vs_fp32_including_cast": 2.0 * ratio_shuffle(arr16, cctx),
        }
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--out", default=str(C.DATA_DIR / "e005"))
    args = ap.parse_args()

    torch.manual_seed(0)
    t0 = time.time()
    model = C.load_model(train=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    ids_all = C.load_token_ids("train")
    losses = []
    for step in range(args.steps):
        ids = C.sample_batch(ids_all, args.batch, args.seq_len, seed=1000 + step)
        opt.zero_grad(set_to_none=True)
        loss = model(ids, labels=ids).loss
        loss.backward()
        opt.step()
        losses.append(loss.item())

    # capture step: saved tensors + block inputs
    params = list(model.parameters())
    param_ptrs = {p.untyped_storage().data_ptr() for p in params}
    saved: list[torch.Tensor] = []
    block_inputs: list[torch.Tensor] = []
    hooks = [
        blk.register_forward_pre_hook(lambda m, a: block_inputs.append(a[0].detach().clone()))
        for blk in model.transformer.h
    ]

    def pack(t):
        saved.append(t)
        return t

    ids = C.sample_batch(ids_all, args.batch, args.seq_len, seed=1000 + args.steps)
    opt.zero_grad(set_to_none=True)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        loss = model(ids, labels=ids).loss
    loss.backward()
    for h in hooks:
        h.remove()

    seen = set()
    act_float, act_other = [], []
    for t in saved:
        if not isinstance(t, torch.Tensor) or t.numel() == 0:
            continue
        ptr = t.untyped_storage().data_ptr()
        if ptr in param_ptrs:
            continue
        key = (t.data_ptr(), t.numel(), t.dtype)
        if key in seen:
            continue
        seen.add(key)
        (act_float if t.is_floating_point() else act_other).append(t)

    states = [opt.state[p] for p in params if p in opt.state]
    categories = {
        "parameters_fp32": [p.detach() for p in params],
        "gradients_fp32": [p.grad for p in params if p.grad is not None],
        "adam_exp_avg_fp32": [s["exp_avg"] for s in states],
        "adam_exp_avg_sq_fp32": [s["exp_avg_sq"] for s in states],
        "saved_activations_float": act_float,
        "saved_activations_nonfloat": act_other,
        "block_inputs_fp32": block_inputs,
        "reference_random_normal_fp32": [torch.randn(4 * 2**20, generator=torch.Generator().manual_seed(0))],
    }
    cctx = zstd.ZstdCompressor(level=3)
    rows = []
    for name, tensors in categories.items():
        rows.append(measure(name, tensors, cctx))
        print(f"{name}: done ({time.time() - t0:.0f}s)", flush=True)

    # pre-registered beta design rules (fp32 parameters and optimizer states)
    rule_rows = [r for r in rows if r["category"] in
                 ("parameters_fp32", "adam_exp_avg_fp32", "adam_exp_avg_sq_fp32")]
    decisions = []
    for r in rule_rows:
        adv = r["ratio_shuffle_zstd"] / r["ratio_os_lz4_16k_pages"]
        decisions.append({
            "category": r["category"],
            "advantage_over_os_lz4": adv,
            "rule1_lossless_beats_os_(>=1.25)": adv >= 1.25,
            "rule2_lossless_small_(<1.3)": r["ratio_shuffle_zstd"] < 1.3,
        })
    result = {
        "experiment": "E005",
        "config": vars(args) | {"cap_bytes": CAP_BYTES, "zstd_level": 3, "lr": 1e-4, "wd": 0.01},
        "train_losses": losses,
        "saved_tensor_counts": {"float": len(act_float), "nonfloat": len(act_other),
                                "block_inputs": len(block_inputs)},
        "rows": rows,
        "beta_rules": decisions,
        "runtime_s": round(time.time() - t0, 1),
        "peak_rss_bytes": peak_rss_bytes(),
    }
    save_json(result, f"{args.out}/results.json")
    save_json(
        capture(__file__, extra={"prereg_sha256": sha256_file(C.PREREG), "argv": vars(args)}),
        f"{args.out}/env.json",
    )
    for r in rows:
        if r.get("empty"):
            continue
        print(f"{r['category']:32s} H={r['entropy_bits_per_element']:5.2f}/{r['stored_bits_per_element']}"
              f"  lz4-16k {r['ratio_os_lz4_16k_pages']:.3f}  zstd-4k {r['ratio_os_zstd_4k_pages']:.3f}"
              f"  zstd {r['ratio_zstd_whole']:.3f}  shuffle+zstd {r['ratio_shuffle_zstd']:.3f}")
    print(decisions)


if __name__ == "__main__":
    main()
