"""Human-readable census summary."""

from __future__ import annotations

from typing import Any

from memopro._units import format_size

_LABELS = {
    "parameters": "parameters+buffers",
    "gradients": "gradients",
    "optimizer_state": "optimizer state",
    "saved_activations": "saved activations",
}


def render(result: dict[str, Any]) -> str:
    cats = result["categories"]
    total = sum(c["bytes"] for c in cats.values()) or 1
    light = result["mode"] in ("light", "deep")
    head = f"{'category':20s} {'bytes':>11s} {'share':>6s} {'stored':>7s} {'entropy':>8s} {'lossless':>9s}"
    if light:
        head += f" {'needed':>7s}"
    lines = [f"memopro census ({result['mode']} mode)", "", head]
    for name, c in cats.items():
        stored = f"{c['stored_bits']:.0f}b" if "stored_bits" in c else "-"
        entropy = f"{c['entropy_bits']:.2f}b" if "entropy_bits" in c else "-"
        ratio = f"{c['lossless_ratio']:.2f}x" if c.get("lossless_ratio") else "-"
        line = (
            f"{_LABELS[name]:20s} {format_size(c['bytes']):>11s} {c['bytes'] / total:6.1%} "
            f"{stored:>7s} {entropy:>8s} {ratio:>9s}"
        )
        if light:
            needed = c.get("needed_bits")
            if not c.get("quant_error"):
                shown = "-"
            else:
                shown = ">8b" if needed is None else f"{needed}b"
            line += f" {shown:>7s}"
        lines.append(line)
    lines.append(f"{'total':20s} {format_size(sum(c['bytes'] for c in cats.values())):>11s}")
    for dev, n in result.get("framework_workspace", {}).items():
        lines.append(
            f"{dev} framework workspace (cuBLAS): {format_size(n)} (counted as attributed)"
        )
    for dev, cov in result["coverage_at_end"].items():
        lines.append(
            f"{dev} coverage at end: {cov:.0%} of allocated memory attributed "
            f"({format_size(result['unclassified_at_end'][dev])} unattributed)"
        )
    if light:
        lines.append(
            "needed bits: smallest of 8/4/2 whose worst sampled tensor stays within "
            f"{result['criteria']['tolerance']:g} relative error ({result['criteria']['needed_bits_basis']})"
        )
    deep = result.get("deep")
    if deep:
        tol = deep["tolerance"]
        lines += [
            "",
            (
                "Deep mode: bits the training step needs (loss change <= "
                f"{tol['loss']:g}, cosine >= {tol['cosine']:g})"
            ),
            f"{'category':20s} {'stored':>7s} {'needed':>7s} {'waste':>6s} {'saveable':>11s}",
        ]
        for name in ("parameters", "gradients", "optimizer_state", "saved_activations"):
            e = deep.get(name)
            if not e:
                continue
            needed = e.get("needed_bits")
            shown = f"{needed}b" if needed else ">16b"
            stored = f"{e['stored_bits']:.0f}b" if e.get("stored_bits") else "-"
            waste = f"{e['waste_bits']:.0f}b" if "waste_bits" in e else "-"
            save = format_size(e["saveable_bytes"]) if "saveable_bytes" in e else "-"
            lines.append(f"{_LABELS[name]:20s} {stored:>7s} {shown:>7s} {waste:>6s} {save:>11s}")
    if result["advice"]:
        lines += ["", "Advice"] + [f"  - {a}" for a in result["advice"]]
    return "\n".join(lines)
