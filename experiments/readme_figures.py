"""README figures (docs/images/*.svg, light and dark): measured results only.

Every number comes from a results file under docs/research/data/ or, where the run predates
them, from the research-log entry named next to it. Each figure has a table twin in README.md.
Palette: the dataviz reference instance (categorical slots in fixed order, validated with
validate_palette.js for both modes; aqua is below 3:1 on the light surface, so every mark is
labelled directly). Colour follows the entity: memopro is slot 1 (blue) in every figure.

    .venv/bin/python experiments/readme_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "research" / "data"
OUT = ROOT / "docs" / "images"
MIB, GIB = 1 << 20, 1 << 30

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "slots": ["#2a78d6", "#eb6834", "#1baf7a"],
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "slots": ["#3987e5", "#d95926", "#199e70"],
    },
}
T = THEMES["light"]


def load(path: str) -> dict:
    return json.loads((DATA / path).read_text())


def figure(w: float = 7.2, h: float = 3.0):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(T["surface"])
    ax.set_facecolor(T["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(T["axis"])
    ax.tick_params(colors=T["ink2"], labelsize=9, length=0)
    ax.grid(axis="x", color=T["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def title(ax, text: str, sub: str = "") -> None:
    ax.set_title(
        text, loc="left", fontsize=11, fontweight="bold", color=T["ink"], pad=18 if sub else 8
    )
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=8.5, color=T["ink2"])


def save(fig, name: str, theme: str) -> None:
    fig.tight_layout()
    suffix = "" if theme == "light" else "-dark"
    fig.savefig(OUT / f"{name}{suffix}.svg", facecolor=T["surface"])
    plt.close(fig)


def hbars(ax, labels, values, colors, texts, xmax=None):
    """Horizontal bars, top to bottom in the given order, each labelled at its end."""
    y = list(range(len(labels)))[::-1]
    ax.barh(y, values, height=0.56, color=colors, edgecolor=T["surface"], linewidth=2)
    ax.set_yticks(y, labels)
    span = xmax or max(values)
    for yi, v, t in zip(y, values, texts, strict=True):
        ax.text(v + span * 0.012, yi, t, va="center", fontsize=8.5, color=T["ink"])
    ax.spines["left"].set_visible(False)


# ------------------------------------------------------------------------------------ figures
def fig_fit(theme: str) -> None:
    """16-bit weights of three models against an 8 GB Mac (E034 0151, E036b 0161, E041b 0190)."""
    blue = T["slots"][0]
    fig, ax = figure(h=2.6)
    labels = ["Qwen2.5-1.5B", "Qwen2.5-3B", "Qwen2.5-7B"]
    weights = [3.09, 6.17, 15.23]
    texts = [
        "mlx-tune runs; memopro runs (45 tok/s)",
        "mlx-tune: out of memory; memopro runs (20 tok/s)",
        "too large to load; memopro runs (16 tok/s)",
    ]
    hbars(ax, labels, weights, [blue] * 3, texts, xmax=30)
    ax.axvline(8, color=T["ink2"], linewidth=1, linestyle=(0, (3, 2)))
    ax.text(8.2, -0.62, "8 GB of memory", fontsize=8, color=T["ink2"])
    ax.set_ylim(-0.75, 2.5)
    ax.set_xlim(0, 30)
    ax.set_xlabel("16-bit weights, GB", color=T["ink2"], fontsize=9)
    title(
        ax,
        "16-bit LoRA on an 8 GB MacBook Air M1",
        "weights stay in their files; losses identical whatever the ceiling",
    )
    save(fig, "fit-8gb", theme)


def fig_generation(theme: str) -> None:
    """Seconds per token on the 8 GB M1 (E039 0184, E038b 0174)."""
    blue, orange, aqua = T["slots"]
    fig, ax = figure(h=2.8)
    labels = [
        "3B · llama.cpp (CPU)",
        "3B · memopro + draft",
        "7B · memopro, no draft",
        "7B · memopro + draft",
    ]
    values = [14.75, 1.09, 8.95, 2.37]
    hbars(
        ax,
        labels,
        values,
        [orange, blue, aqua, blue],
        ["14.75 s", "1.09 s", "8.95 s", "2.37 s  (3.8x)"],
        xmax=17,
    )
    ax.set_xlim(0, 17)
    ax.set_xlabel("seconds per token (lower is faster)", color=T["ink2"], fontsize=9)
    title(
        ax,
        "Generation larger than memory, 8 GB M1",
        "same text as the model loaded normally; llama.cpp's Metal offloads ran out of memory",
    )
    save(fig, "generation-8gb", theme)


def fig_ceiling(theme: str) -> None:
    """Process peak as a share of the ceiling set (E068 T4 0.2.0b3, E069 0357)."""
    blue = T["slots"][0]
    t4 = "e068/t4-b3/cases/"
    rows = []
    for name, case in [
        ("3B LoRA, 6 GiB", "Qwen2.5-3B-Instruct__ft_6144"),
        ("3B LoRA, 8 GiB", "Qwen2.5-3B-Instruct__ft_8192"),
        ("7B LoRA, 7 GiB", "Qwen2.5-7B-Instruct__ft_7168"),
        ("7B LoRA, 9 GiB", "Qwen2.5-7B-Instruct__ft_9216"),
    ]:
        r = load(t4 + case + ".json")
        rows.append((name, r["memory_end"]["host_peak_rss_bytes"] / r["budget"]))
    for name, case in [
        ("image stack, half", "D_image_run"),
        ("scikit-learn, half", "D_classify_run"),
    ]:
        r = load(t4 + case + ".json")
        rows.append((name, r["result"]["maxrss_bytes"] / r["budget"]))
    rows.append(("1 GB CSV sort, 313 MiB", 233 / 313))  # dev check, 0357
    fig, ax = figure(h=3.3)
    hbars(
        ax,
        [r[0] for r in rows],
        [100 * r[1] for r in rows],
        [blue] * len(rows),
        [f"{100 * r[1]:.0f}%" for r in rows],
        xmax=115,
    )
    ax.axvline(100, color=T["ink2"], linewidth=1, linestyle=(0, (3, 2)))
    ax.text(100.8, -0.75, "ceiling", fontsize=8, color=T["ink2"])
    ax.set_ylim(-0.9, len(rows) - 0.5)
    ax.set_xlim(0, 115)
    ax.set_xlabel("process peak / ceiling, %", color=T["ink2"], fontsize=9)
    title(
        ax,
        "The whole process stays within the ceiling",
        "Colab T4, 0.2.0b3 (data programs may go 32 MiB over); sort: development check",
    )
    save(fig, "ceiling", theme)


def fig_swap(theme: str) -> None:
    """Bytes written to the disk and seconds at half memory (E069 V4, 0359)."""
    blue, orange, aqua = T["slots"]
    c = "e069/linux/cases/"
    work = [("image stack", "image"), ("scikit-learn", "classify")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for ax in (a1, a2):
        fig.patch.set_facecolor(T["surface"])
        ax.set_facecolor(T["surface"])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(T["axis"])
        ax.tick_params(colors=T["ink2"], labelsize=9, length=0)
        ax.grid(axis="y", color=T["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
    x = [0, 1]
    swap = [load(c + f"v4_{k}_swap.json")["swap_out_bytes"] / MIB for _, k in work]
    mem = [
        (
            load(c + f"v4_{k}_memopro.json")["swap_out_bytes"]
            + (load(c + f"v4_{k}_memopro.json")["write_bytes"] or 0)
        )
        / MIB
        for _, k in work
    ]
    w = 0.34
    a1.bar(
        [i - w / 2 for i in x],
        swap,
        w,
        color=orange,
        edgecolor=T["surface"],
        linewidth=2,
        label="OS swap",
    )
    a1.bar([i + w / 2 for i in x], [max(m, 0) for m in mem], w, color=blue, label="memopro run")
    for i in x:
        a1.text(
            i - w / 2, swap[i] + 40, f"{swap[i]:,.0f} MiB", ha="center", fontsize=8, color=T["ink"]
        )
        a1.text(i + w / 2, mem[i] + 40, "0 B", ha="center", fontsize=8, color=T["ink"])
    a1.set_xticks(x, [w for w, _ in work])
    a1.set_ylabel("written to the SSD, MiB", color=T["ink2"], fontsize=9)
    a1.set_ylim(0, max(swap) * 1.2)
    a1.legend(frameon=False, fontsize=8, labelcolor=T["ink2"], loc="upper right")
    a1.set_title("Bytes written", loc="left", fontsize=10, color=T["ink"])
    plain = [load(c + f"v4_{k}_plain.json")["wall_s"] for _, k in work]
    sw = [load(c + f"v4_{k}_swap.json")["wall_s"] for _, k in work]
    mp = [load(c + f"v4_{k}_memopro.json")["wall_s"] for _, k in work]
    w = 0.26
    for off, vals, col, lab in [
        (-w, plain, aqua, "all memory (no limit)"),
        (0, sw, orange, "OS swap"),
        (w, mp, blue, "memopro run"),
    ]:
        a2.bar(
            [i + off for i in x], vals, w, color=col, edgecolor=T["surface"], linewidth=2, label=lab
        )
        for i in x:
            a2.text(i + off, vals[i] + 1, f"{vals[i]:.0f}", ha="center", fontsize=8, color=T["ink"])
    a2.set_xticks(x, [w for w, _ in work])
    a2.set_ylabel("seconds", color=T["ink2"], fontsize=9)
    a2.set_ylim(0, max(sw) * 1.25)
    a2.legend(frameon=False, fontsize=8, labelcolor=T["ink2"], loc="upper right")
    a2.set_title("Time", loc="left", fontsize=10, color=T["ink"])
    fig.suptitle(
        "Unchanged programs at half their memory (Linux, cgroup)",
        x=0.02,
        ha="left",
        fontsize=11,
        fontweight="bold",
        color=T["ink"],
    )
    save(fig, "half-memory", theme)


def fig_audio(theme: str) -> None:
    """Render time per 5.33 ms audio block while the cgroup swaps (E069 V3b, 0359)."""
    blue, orange, _ = T["slots"]
    c = "e069/linux/cases/"
    r1, r2 = load(c + "v3b_R1.json"), load(c + "v3b_R2.json")
    bins = ["<.05", ".05-.1", ".1-.2", ".2-.5", ".5-1", "1-2", "2-5.3", "5.3-10", "10-50"]
    fig, ax = figure(h=2.9)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color=T["grid"], linewidth=0.8)
    x = list(range(len(bins)))
    w = 0.38
    h1, h2 = r1["render_ms_hist"][: len(bins)], r2["render_ms_hist"][: len(bins)]
    ax.bar(
        [i - w / 2 for i in x],
        [max(v, 0.6) if v else 0 for v in h1],
        w,
        color=orange,
        edgecolor=T["surface"],
        linewidth=2,
        label=f"ordinary buffer: {r1['glitches']} drop-outs, max {r1['render_ms_max']:.1f} ms",
    )
    ax.bar(
        [i + w / 2 for i in x],
        [max(v, 0.6) if v else 0 for v in h2],
        w,
        color=blue,
        edgecolor=T["surface"],
        linewidth=2,
        label=f"locked=True: {r2['glitches']} drop-outs, max {r2['render_ms_max']:.1f} ms",
    )
    ax.set_yscale("log")
    ax.set_xticks(x, bins)
    ax.axvline(6.5, color=T["ink2"], linewidth=1, linestyle=(0, (3, 2)))
    ax.text(6.6, 20, "deadline\n5.33 ms", fontsize=8, color=T["ink2"])
    ax.set_xlabel("render time of one block, ms", color=T["ink2"], fontsize=9)
    ax.set_ylabel("blocks (log)", color=T["ink2"], fontsize=9)
    ax.legend(
        frameon=False,
        fontsize=8,
        labelcolor=T["ink2"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
    )
    title(
        ax,
        "Real-time audio while other memory swaps",
        "60 s at 48 kHz, 256-frame blocks; the cgroup swapped about 8.5 GiB meanwhile",
    )
    save(fig, "audio-locked", theme)


def fig_codec(theme: str) -> None:
    """Lossless compression of 1 GiB of each kind (E069 V5 Linux, 0359)."""
    blue = T["slots"][0]
    rows = load("e069/linux/v5_linux.json")["results"]
    names = {
        "float32 simulation field": "float32 field",
        "uint16 image stack": "uint16 images",
        "CSV text": "CSV text",
        "int64 keys": "int64 keys",
        "random bytes": "random bytes",
    }
    labels = [names[r["kind"]] for r in rows]
    ratios = [r["ratio"] or 1.0 for r in rows]
    texts = [
        f"{r['ratio']:.2f}x, {r['compress_gbps']:.2f} / {r['decompress_gbps']:.2f} GB/s"
        + ("" if r["default_compresses"] else ", kept as is (saves < 15%)")
        if r["ratio"]
        else "kept as is (no saving)"
        for r in rows
    ]
    fig, ax = figure(h=2.8)
    hbars(ax, labels, ratios, [blue] * len(rows), texts, xmax=5.2)
    ax.set_xlim(0, 5.2)
    ax.set_xlabel("size / compressed size", color=T["ink2"], fontsize=9)
    title(
        ax,
        "In-memory lossless compression (byte shuffle + zstd 1)",
        "every buffer comes back bit-identical; GB/s = compress / decompress, Linux runner",
    )
    save(fig, "compression", theme)


def fig_ffi(theme: str) -> None:
    """Runtime calls from Rust, C and Python (E069 V6 Linux, 0359)."""
    rows = {r["lang"]: r for r in load("e069/linux/v6_linux.json")["results"]}
    ops = [
        ("pin + unpin", "pin_unpin_ns"),
        ("alloc + free 64 KiB", "alloc_free_64k_ns"),
        ("stats", "stats_ns"),
    ]
    fig, ax = figure(h=2.9)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color=T["grid"], linewidth=0.8)
    x = list(range(len(ops)))
    w = 0.26
    for off, lang, col, lab in [
        (-w, "rust", T["slots"][0], "Rust"),
        (0, "c", T["slots"][2], "C ABI"),
        (w, "python", T["slots"][1], "Python"),
    ]:
        vals = [rows[lang][k] for _, k in ops]
        ax.bar(
            [i + off for i in x], vals, w, color=col, edgecolor=T["surface"], linewidth=2, label=lab
        )
        for i, v in zip(x, vals, strict=True):
            ax.text(
                i + off,
                v * 1.15,
                f"{v / 1000:.1f} µs" if v >= 1000 else f"{v:.0f} ns",
                ha="center",
                fontsize=7.5,
                color=T["ink"],
            )
    ax.set_yscale("log")
    ax.set_xticks(x, [o for o, _ in ops])
    ax.set_ylabel("ns per call (log)", color=T["ink2"], fontsize=9)
    ax.set_ylim(5, 1e5)
    ax.legend(frameon=False, fontsize=8, labelcolor=T["ink2"], loc="upper right", ncol=3)
    copy = rows["c"]["copy_gbps"]
    title(
        ax,
        "Calling the runtime from other languages",
        f"bulk data runs at memcpy speed in all three ({copy:.1f} GB/s); Linux runner",
    )
    save(fig, "ffi", theme)


def main() -> None:
    global T
    OUT.mkdir(parents=True, exist_ok=True)
    for theme, colors in THEMES.items():
        T = colors
        plt.rcParams.update({"font.family": "DejaVu Sans", "svg.hashsalt": "memopro"})
        for f in (fig_fit, fig_generation, fig_ceiling, fig_swap, fig_audio, fig_codec, fig_ffi):
            f(theme)
    print(sorted(p.name for p in OUT.glob("*.svg")))


if __name__ == "__main__":
    main()
