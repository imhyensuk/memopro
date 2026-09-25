"""Shared pieces for the α experiments (pre-registered in docs/research/0017).

Definitions follow 0017 exactly:
- block function f_l(x) = block_l(x) - x, so x_{l+1} = x_l + f_l(x_l)
- hint = blockwise (64 elements along the hidden dim) symmetric absmax quantization, fp16 scales
- plain fixed point: x <- y - f_l(x); Anderson: type-II mixing (m=3) on the same map
- chained reconstruction: x_L stored exactly, x_0 recomputed from token ids, l = L-1..1 recovered
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(REPO / ".cache" / "hf"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

MODEL_ID = "openai-community/gpt2"
MODEL_REV = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
DATA_ID = "Salesforce/wikitext"
DATA_CFG = "wikitext-2-raw-v1"
DATA_REV = "b08601e04326c79dfdd32d625aee71d232d685c3"
PREREG = REPO / "docs" / "research" / "0017-x1-preregistration.md"
DATA_DIR = REPO / "docs" / "research" / "data"
HINT_BLOCK = 64


# ---------------------------------------------------------------- model and data


def load_model(train: bool = False):
    from transformers import GPT2LMHeadModel

    model = GPT2LMHeadModel.from_pretrained(
        MODEL_ID, revision=MODEL_REV, attn_implementation="eager", dtype=torch.float32
    )
    model.train(train)
    return model


def load_token_ids(split: str = "test") -> torch.Tensor:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REV)
    ds = load_dataset(DATA_ID, DATA_CFG, revision=DATA_REV, split=split)
    text = "".join(ds["text"])
    return tok(text, return_tensors="pt", verbose=False)["input_ids"][0]


def sample_batch(ids: torch.Tensor, batch: int, seq_len: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, ids.numel() - seq_len - 1, (batch,), generator=g).tolist()
    return torch.stack([ids[s : s + seq_len] for s in starts])


def causal_mask(seq_len: int, dtype=torch.float32) -> torch.Tensor:
    """Additive causal mask (1, 1, T, T). transformers 5.x blocks do not mask on their own."""
    m = torch.full((seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype).triu(1)
    return m[None, None]


class Blocks:
    """Direct access to GPT-2 blocks with an explicit causal mask."""

    def __init__(self, model, seq_len: int):
        self.model = model
        self.h = model.transformer.h
        self.n_layer = len(self.h)
        self.mask = causal_mask(seq_len)

    def block(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        out = self.h[layer](x, attention_mask=self.mask)
        return out[0] if isinstance(out, tuple) else out

    def f(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        return self.block(layer, x) - x

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        tr = self.model.transformer
        pos = torch.arange(ids.shape[1])[None]
        return tr.drop(tr.wte(ids) + tr.wpe(pos))

    def head_loss(self, x_last: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        logits = self.model.lm_head(self.model.transformer.ln_f(x_last))
        return F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)).float(), ids[:, 1:].reshape(-1)
        )


@torch.no_grad()
def exact_activations(blocks: Blocks, ids: torch.Tensor) -> list[torch.Tensor]:
    xs = [blocks.embed(ids)]
    for layer in range(blocks.n_layer):
        xs.append(blocks.block(layer, xs[-1]))
    return xs  # x_0 .. x_L (x_L = last block output, before ln_f)


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


@torch.no_grad()
def validate_blocks(model, blocks: Blocks, ids: torch.Tensor, xs: list[torch.Tensor]) -> dict:
    """Block-by-block path must reproduce the full model (0017: otherwise the run is invalid)."""
    out = model(ids, output_hidden_states=True)
    hs = out.hidden_states
    n = blocks.n_layer
    ln_f = model.transformer.ln_f
    hidden_err = [rel(xs[i], hs[i]) for i in range(n)]
    last_raw = rel(xs[n], hs[n])
    last_lnf = rel(ln_f(xs[n]), hs[n])
    logits_err = rel(model.lm_head(ln_f(xs[n])), out.logits)
    return {
        "max_hidden_rel_err_l0_to_l11": max(hidden_err),
        "last_hidden_rel_err_raw": last_raw,
        "last_hidden_rel_err_after_ln_f": last_lnf,
        "logits_rel_err": logits_err,
        "valid": max(hidden_err) < 1e-5 and min(last_raw, last_lnf) < 1e-5 and logits_err < 1e-5,
    }


# ---------------------------------------------------------------- hints (quantization)


def quantize_hint(x: torch.Tensor, bits: int, block: int = HINT_BLOCK) -> torch.Tensor:
    """Dequantized b-bit hint of x: blockwise symmetric absmax, fp16 scales (0017)."""
    shape = x.shape
    xb = x.reshape(-1, block)
    qmax = 2 ** (bits - 1) - 1
    scale = (xb.abs().amax(dim=1, keepdim=True) / qmax).to(torch.float16).to(x.dtype)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.clamp(torch.round(xb / scale), -qmax, qmax)
    return (q * scale).reshape(shape)


def hint_bits_per_element(bits: int, block: int = HINT_BLOCK) -> float:
    return bits + 16 / block


# ---------------------------------------------------------------- fixed-point solvers


@torch.no_grad()
def solve_fixed_point(
    f: Callable[[torch.Tensor], torch.Tensor],
    y: torch.Tensor,
    x0: torch.Tensor,
    k: int,
    accel: str = "plain",
    m: int = 3,
    on_iterate: Callable[[int, torch.Tensor], None] | None = None,
) -> torch.Tensor:
    """Solve x = y - f(x) from x0 using exactly k evaluations of f."""
    x = x0
    if accel == "plain":
        for j in range(k):
            x = y - f(x)
            if on_iterate:
                on_iterate(j + 1, x)
        return x
    if accel != "anderson":
        raise ValueError(accel)
    xs_hist: list[torch.Tensor] = []
    gs_hist: list[torch.Tensor] = []
    for j in range(k):
        g = y - f(x)
        xs_hist.append(x.reshape(-1))
        gs_hist.append(g.reshape(-1))
        if len(xs_hist) > m + 1:
            xs_hist.pop(0)
            gs_hist.pop(0)
        if len(xs_hist) == 1:
            x = g
        else:
            res = [gs_hist[i] - xs_hist[i] for i in range(len(xs_hist))]
            d_res = torch.stack([res[i + 1] - res[i] for i in range(len(res) - 1)], dim=1)
            d_g = torch.stack([gs_hist[i + 1] - gs_hist[i] for i in range(len(gs_hist) - 1)], dim=1)
            a = (d_res.T @ d_res).double()
            b = (d_res.T @ res[-1]).double()
            reg = 1e-10 * a.diagonal().abs().max().clamp_min(1e-300)
            gamma = torch.linalg.solve(a + reg * torch.eye(a.shape[0], dtype=a.dtype), b)
            x = (gs_hist[-1] - d_g @ gamma.to(d_g.dtype)).reshape(x0.shape)
        if on_iterate:
            on_iterate(j + 1, x)
    return x


@torch.no_grad()
def chain_reconstruct(
    blocks: Blocks,
    xs: list[torch.Tensor],
    bits: int,
    k: int,
    accel: str,
    reconstruct_x0: bool = False,
) -> list[torch.Tensor]:
    """Backward-order reconstruction as RFC would do it during backprop (0017 main scenario)."""
    n = blocks.n_layer
    xhat: list[torch.Tensor | None] = [None] * (n + 1)
    xhat[n] = xs[n]
    lowest = 0 if reconstruct_x0 else 1
    for layer in range(n - 1, lowest - 1, -1):
        hint = quantize_hint(xs[layer], bits)
        if k == 0:
            xhat[layer] = hint
        else:
            xhat[layer] = solve_fixed_point(
                lambda z, _l=layer: blocks.f(_l, z), xhat[layer + 1], hint, k, accel
            )
    if not reconstruct_x0:
        xhat[0] = xs[0]
    return xhat  # type: ignore[return-value]


def token_errors(xhat: torch.Tensor, x: torch.Tensor) -> dict[str, float]:
    num = (xhat - x).norm(dim=-1)
    den = x.norm(dim=-1).clamp_min(1e-30)
    e = (num / den).reshape(-1).double()
    return {
        "mean": e.mean().item(),
        "median": e.median().item(),
        "p95": torch.quantile(e, 0.95).item(),
        "max": e.max().item(),
        "frobenius": rel(xhat, x),
    }


def residual(blocks: Blocks, layer: int, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        return rel(x + blocks.f(layer, x), y)
