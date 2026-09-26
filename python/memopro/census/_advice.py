"""Turn census findings into actionable advice (0032 P6).

Advice names memopro features or generic technique names only; it never sends users to
external tools (0012 S2). Every estimate says whether it keeps results exact or changes numerics.
"""

from __future__ import annotations

from typing import Any

from memopro._units import format_size

# Share of classified bytes above which a category is worth a suggestion.
_LARGE_SHARE = 0.25
_LOSSLESS_WORTH = 1.3  # below this, lossless compression is not worth suggesting (E005)
_UNCLASSIFIED_SHARE = 0.2


def advise(result: dict[str, Any]) -> list[str]:
    cats = result["categories"]
    total = sum(c["bytes"] for c in cats.values())
    if total == 0:
        return ["nothing was recorded: pass the model/optimizer and run a step inside `with`"]
    out = []

    opt = cats["optimizer_state"]
    if opt["bytes"] / total >= _LARGE_SHARE and opt.get("dtypes", {}).get("float32"):
        saving = int(opt["dtypes"]["float32"] * 0.75)
        fits = opt.get("needed_bits")
        evidence = (
            f"; sampled states fit in {fits} bits by reconstruction error"
            if fits is not None and fits <= 8
            else ""
        )
        out.append(
            f"optimizer state is {opt['bytes'] / total:.0%} of recorded memory: 8-bit optimizer "
            f"states would save about {format_size(saving)} (changes numerics){evidence}; "
            "switching the optimizer is your choice (memopro.train_session only suggests it)"
        )

    act = cats["saved_activations"]
    if act["bytes"] / total >= _LARGE_SHARE:
        out.append(
            f"saved activations are {act['bytes'] / total:.0%} of recorded memory "
            f"({format_size(act['bytes'])}): activation checkpointing trades recomputation for "
            "most of this (exact results) [memopro.train_session applies it when memory runs "
            "out, memopro.optimize(model, goal='train') turns it on]"
        )

    f32 = sum(cats[c].get("dtypes", {}).get("float32", 0) for c in ("parameters", "gradients"))
    if f32 / total >= _LARGE_SHARE:
        out.append(
            f"{format_size(f32)} of parameters and gradients are float32: bf16 mixed precision "
            f"would halve them, about {format_size(f32 // 2)} (changes numerics)"
        )

    compressible = [
        (name, cat["lossless_ratio"], int(cat["bytes"] * (1 - 1 / cat["lossless_ratio"])))
        for name, cat in cats.items()
        if cat.get("lossless_ratio") and cat["lossless_ratio"] >= _LOSSLESS_WORTH and cat["bytes"]
    ]
    if compressible:
        parts = ", ".join(f"{n.replace('_', ' ')} {r:.2f}x" for n, r, _ in compressible)
        saving = sum(s for _, _, s in compressible)
        out.append(
            f"lossless compression ratios: {parts}. Idle copies could give back about "
            f"{format_size(saving)} (exact) [memopro.hibernate mode='compress']"
        )

    massive = [(n, c) for n, c in cats.items() if c.get("massive_tensors")]
    if massive:
        parts = ", ".join(
            f"{n.replace('_', ' ')} ({c['massive_tensors']}, max/median {c['max_peak_to_median']})"
            for n, c in massive
        )
        out.append(
            f"sampled tensors with massive values: {parts}. Lossy formats must treat outliers "
            "separately (K1)"
        )

    sparse = [
        (n, c["zero_fraction"])
        for n, c in cats.items()
        if (c.get("zero_fraction") or 0) >= 0.5 and c["bytes"]
    ]
    if sparse:
        parts = ", ".join(f"{n.replace('_', ' ')} {z:.0%}" for n, z in sparse)
        out.append(f"mostly-zero categories (median share of exact zeros): {parts}")

    for dev, amount in result.get("unclassified_at_end", {}).items():
        allocated = result["allocator_after"][dev]["allocated"]
        if allocated and amount / allocated >= _UNCLASSIFIED_SHARE:
            out.append(
                f"{format_size(amount)} of {dev} memory is not attributable to the model, "
                "gradients or optimizer (other tensors, caches such as a KV cache, or "
                "fragmentation)"
            )
    return out
