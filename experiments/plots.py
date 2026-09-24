"""Figures for X1 results (docs/research/data/eNNN/*.png). Static PNGs for the research log.

Style follows the dataviz reference palette (light surface #fcfcfb, categorical slots in fixed
order: blue, orange, aqua; validated with validate_palette.js). Every figure has a table twin in
the corresponding research-log entry, and multi-series figures carry a legend.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter

DATA = Path(__file__).resolve().parents[1] / "docs" / "research" / "data"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua (fixed order)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.labelsize": 9,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
        "font.family": ["Apple SD Gothic Neo", "AppleGothic", "sans-serif"],
        "axes.unicode_minus": False,
    }
)


def _clean(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)


def _log_y(ax):
    """Log y axis with plain decimal tick labels (the Korean font lacks the U+2212 minus)."""
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def _ref_line(ax, y, text):
    ax.axhline(y, color=INK2, linewidth=1, zorder=1)
    ax.annotate(
        text,
        xy=(1, y),
        xycoords=("axes fraction", "data"),
        xytext=(-4, 4),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=8,
        color=INK2,
    )


def fig_e001():
    r = json.loads((DATA / "e001" / "results.json").read_text())
    g = json.loads((DATA / "e001" / "gain_analysis.json").read_text())["per_layer"]
    sig = r["summary"]["sigmas"]
    rho = [x["rho_estimate"] for x in g]
    typ = [x["hint_error_gain"]["4"] for x in g]
    xs = list(range(len(sig)))
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=160)
    series = [
        (sig, "최악 방향 증폭 σ (E001, 사전 등록 H1)"),
        (rho, "스펙트럴 반경 ρ 추정 (E001b, 탐색적) — 반복의 장기 수렴을 결정"),
        (typ, "4비트 힌트 오차 방향의 증폭 (E001b, 탐색적) — 초기 몇 회의 감소를 결정"),
    ]
    for slot, (vals, name) in enumerate(series):
        ax.plot(
            xs,
            vals,
            color=SLOTS[slot],
            marker="o",
            markersize=5,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            linewidth=1.5,
            label=name,
            zorder=3,
        )
    _log_y(ax)
    ax.axhline(1.0, color=INK2, linewidth=1, zorder=1)
    ax.set_xticks(xs)
    ax.set_xlabel("블록 l")
    ax.set_ylabel("증폭률 (로그)")
    ax.set_title("E001 · 블록 야코비안: 모든 층에서 ρ > 1 (GPT-2 small)")
    handles, names = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color=INK2, linewidth=1))
    names.append("증폭률 1 — 이보다 크면 그 방향의 오차가 반복마다 커진다")
    ax.legend(handles, names, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    _clean(ax)
    fig.tight_layout()
    fig.savefig(DATA / "e001" / "gain_by_block.png")
    plt.close(fig)


def fig_e002():
    r = json.loads((DATA / "e002" / "results.json").read_text())
    chain = [c for c in r["chain"] if not c["reconstruct_x0"]]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), dpi=160, sharey=True)
    for ax, bits in zip(axes, (2, 3, 4)):
        base = next(c for c in chain if c["bits"] == bits and c["k"] == 0)["layer_mean_error"]
        for slot, accel in enumerate(("plain", "anderson")):
            pts = [(0, base)] + sorted(
                (c["k"], c["layer_mean_error"])
                for c in chain
                if c["bits"] == bits and c["accel"] == accel
            )
            ks = [p[0] for p in pts]
            ys = [min(p[1], 1e6) for p in pts]
            ax.plot(
                ks,
                ys,
                color=SLOTS[slot],
                marker="o",
                markersize=4,
                markeredgecolor=SURFACE,
                markeredgewidth=1.5,
                label="고정점 반복" if accel == "plain" else "Anderson 가속",
            )
        _log_y(ax)
        _ref_line(ax, 1e-2, "H2 기준 0.01")
        ax.set_title(f"힌트 {bits}비트")
        ax.set_xlabel("반복 횟수 k (0 = 교정 없음)")
        _clean(ax)
    axes[0].set_ylabel("층 평균 토큰별 상대 오차 (로그)")
    axes[0].legend(loc="lower right", bbox_to_anchor=(1.0, 0.12))
    fig.suptitle(
        "E002 · 체인 복원 오차 (x_L 원본, x_0 재계산, 층 1–11 평균)",
        fontsize=11,
        fontweight="semibold",
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(DATA / "e002" / "chain_error_vs_k.png")
    plt.close(fig)


def fig_e003():
    r = json.loads((DATA / "e003" / "results.json").read_text())
    fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=160)
    for slot, bits in enumerate((2, 3, 4)):
        runs = sorted((x["k"], x["cosine"]) for x in r["runs"] if x["bits"] == bits)
        ks = [k for k, _ in runs]
        ys = [1.0 if math.isnan(c) else 1 - c for _, c in runs]  # 1 - cosine, log scale
        ys = [max(y, 1e-9) for y in ys]
        ax.plot(
            ks,
            ys,
            color=SLOTS[slot],
            marker="o",
            markersize=4,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            label=f"{bits}비트",
        )
    _log_y(ax)
    _ref_line(ax, 1e-3, "H3 기준 (코사인 0.999)")
    ax.set_xlabel("반복 횟수 k (0 = 교정 없음)")
    ax.set_ylabel("1 - 코사인 유사도 (로그, 낮을수록 정확)")
    ax.set_title("E003 · 그래디언트 정확도 (정확한 역전파 대비)")
    ax.legend(title="힌트", title_fontsize=8)
    _clean(ax)
    fig.tight_layout()
    fig.savefig(DATA / "e003" / "grad_cosine_vs_k.png")
    plt.close(fig)


def fig_e005():
    r = json.loads((DATA / "e005" / "results.json").read_text())
    rows = [x for x in r["rows"] if not x.get("empty") and x["total_bytes"] >= 2**20]
    labels = {
        "parameters_fp32": "가중치",
        "gradients_fp32": "그래디언트",
        "adam_exp_avg_fp32": "Adam 1차 모멘트",
        "adam_exp_avg_sq_fp32": "Adam 2차 모멘트",
        "saved_activations_float": "저장 활성값",
        "saved_activations_nonfloat": "저장 마스크·인덱스",
        "block_inputs_fp32": "블록 입력 x_l",
        "reference_random_normal_fp32": "참고: 난수 fp32",
    }
    methods = [
        ("ratio_os_lz4_16k_pages", "OS 방식 (LZ4, 16KB 페이지)"),
        ("ratio_os_zstd_4k_pages", "OS 방식 (zstd, 4KB 페이지)"),
        ("ratio_shuffle_zstd", "바이트 셔플 + zstd (memopro 후보)"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    n = len(rows)
    h = 0.24
    for slot, (key, name) in enumerate(methods):
        ys = [i + (slot - 1) * (h + 0.02) for i in range(n)]
        ax.barh(ys, [row[key] for row in rows], height=h, color=SLOTS[slot], label=name, zorder=2)
    for i, row in enumerate(rows):
        ax.annotate(
            f"{row['ratio_shuffle_zstd']:.2f}×",
            (row["ratio_shuffle_zstd"], i + h + 0.02),
            xytext=(3, 0),
            textcoords="offset points",
            va="center",
            fontsize=7,
            color=INK,
        )
    ax.axvline(1.0, color=INK2, linewidth=1, zorder=1)
    ax.set_yticks(range(n))
    ax.set_yticklabels([labels.get(row["category"], row["category"]) for row in rows])
    ax.invert_yaxis()
    ax.set_xlabel("무손실 압축률 (원본 / 압축, 1 = 압축 안 됨)")
    ax.set_title("E005 · 학습 중 텐서의 무손실 압축률 (GPT-2 small, fp32)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=7)
    _clean(ax)
    fig.tight_layout()
    fig.savefig(DATA / "e005" / "compression_ratios.png")
    plt.close(fig)


if __name__ == "__main__":
    targets = sys.argv[1:] or ["e001", "e002", "e003", "e005"]
    for t in targets:
        {"e001": fig_e001, "e002": fig_e002, "e003": fig_e003, "e005": fig_e005}[t]()
        print("figure", t, "ok")
