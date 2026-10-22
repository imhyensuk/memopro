"""int4 weight-only linear layers on Apple GPUs with torch's own kernel (0069).

torch ships ``_weight_int4pack_mm`` for MPS (1.63x a bf16 matmul on the M1, E015 Q4, while
bitsandbytes nf4 ran at 0.35x). torchao's group-wise affine quantization produces the packed
format that kernel expects. memopro only wires the two together: no kernel and no quantization
scheme of its own (0010).

Loading converts one layer at a time from weights loaded on the CPU (safetensors are memory
mapped, so the bf16 original stays clean file-backed memory), so the full half-precision model
never sits on the device.

E020 (0084) changed two things:
- group 32 instead of 64 (Q-b): WikiText-2 perplexity +5.7%/+7.6% instead of +9.2%/+21.1% for
  Qwen2.5-1.5B/3B, for 7-8% more weight bytes;
- long inputs (Q-c): the kernel's time grows linearly with the number of input rows (a 2048-token
  prompt took 38 s on 1.5B, bf16 6.3 s), so from `LONG_INPUT` rows on, a layer dequantizes its
  int4 weight to half precision for that call and uses a plain matmul, then drops it. Only torch
  operations; the packed layout on MPS is the [out, in] codes, eight per int32, low nibble first.
"""

from __future__ import annotations

from typing import Any

import torch

__all__ = ["GROUP", "Int4PackedLinear", "convert", "eligible", "quantize_linear", "works"]

GROUP = 32  # 0084 Q-b (was 64)
INNER_K_TILES = 8
LONG_INPUT = (
    160  # rows from which a call dequantizes instead of the kernel (0085: 1.5B ~180, 3B ~150)
)
# a long input unpacks this many bytes of float32 codes at a time (a slice of output rows): on
# MPS an allocation of 10-512 MiB makes torch reserve a 1 GiB heap (0136), and unpacking a whole
# 11008x2048 weight at once kept about 1 GiB more GPU memory (0286)
UNPACK_BYTES = 8 << 20
# output rows per matrix product of a long input: with fewer (e.g. 190 of a 2048x11008 weight)
# MPS rounded 160-300-row products differently from the whole weight; 512 matched (0288)
UNPACK_ROWS = 512
# weights that stay in half precision (embeddings, the output head, norms), as in `_info`
_SKIP = ("embed", "wte", "wpe", "lm_head", "shared", "position", "norm", "ln_")


class Int4PackedLinear(torch.nn.Module):
    """A linear layer whose weight is int4, group-wise, packed for ``_weight_int4pack_mm``."""

    def __init__(
        self,
        packed: torch.Tensor,
        scales_and_zeros: torch.Tensor,
        in_features: int,
        out_features: int,
        bias: torch.Tensor | None,
        group: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.register_buffer("packed", packed)
        self.register_buffer("scales_and_zeros", scales_and_zeros)
        self.bias = None if bias is None else torch.nn.Parameter(bias, requires_grad=False)
        self.in_features, self.out_features = in_features, out_features
        self.group, self.compute_dtype = group, dtype

    def dequantized(self, start: int = 0, stop: int | None = None) -> torch.Tensor:
        """Output rows ``start:stop`` of the weight in ``compute_dtype``, [rows, in], from the
        packed codes (a temporary)."""
        n, k, g = self.out_features, self.in_features, self.group
        stop = n if stop is None else stop
        words = self.packed.reshape(n, k // 8)[start:stop]
        shifts = torch.arange(0, 32, 4, device=words.device, dtype=words.dtype)
        codes = ((words.unsqueeze(-1) >> shifts) & 15).reshape(stop - start, k // g, g)
        scales = self.scales_and_zeros[:, start:stop, 0].t().unsqueeze(-1).float()
        zeros = self.scales_and_zeros[:, start:stop, 1].t().unsqueeze(-1).float()
        # torchao's tinygemm convention: w = (q - 8) * scale + zero
        w = (codes.float() - 8) * scales + zeros
        return w.reshape(stop - start, k).to(self.compute_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        rows = x.reshape(-1, shape[-1]).to(self.compute_dtype)
        if rows.shape[0] >= LONG_INPUT:  # unpack a slice of output rows at a time (0288)
            n, k = self.out_features, self.in_features
            piece = max(1, UNPACK_BYTES // (4 * k))
            y = rows.new_empty(rows.shape[0], n)
            for a in range(0, n, UNPACK_ROWS):
                b = min(n, a + UNPACK_ROWS)
                w = rows.new_empty(b - a, k)
                for c in range(a, b, piece):
                    w[c - a : min(b, c + piece) - a] = self.dequantized(c, min(b, c + piece))
                y[:, a:b] = torch.nn.functional.linear(rows, w)
        else:
            y = torch._weight_int4pack_mm(rows, self.packed, self.group, self.scales_and_zeros)
        y = y.reshape(*shape[:-1], self.out_features)
        if self.bias is not None:
            y = y + self.bias
        return y.to(x.dtype)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, int4 group {self.group}"


def _fits(linear: torch.nn.Linear, group: int) -> bool:
    k = linear.in_features
    return k % group == 0 and k % (INNER_K_TILES * 16) == 0 and linear.out_features % 8 == 0


def quantize_linear(
    linear: torch.nn.Linear, device: str, group: int = GROUP, dtype: torch.dtype = torch.bfloat16
) -> Int4PackedLinear:
    from torchao.quantization.utils import groupwise_affine_quantize_tensor

    w = linear.weight.detach().to(dtype)
    q, scales_and_zeros = groupwise_affine_quantize_tensor(w, 4, group, dtype=dtype)
    # two nibbles per byte on the CPU, so only the small uint8 tensor goes to the device
    nibbles = ((q[:, ::2] << 4) | q[:, 1::2]).to(torch.uint8)
    del q
    packed = torch._convert_weight_to_int4pack(nibbles.to(device), INNER_K_TILES)
    bias = None if linear.bias is None else linear.bias.detach().to(device, dtype)
    return Int4PackedLinear(
        packed,
        scales_and_zeros.to(device),
        linear.in_features,
        linear.out_features,
        bias,
        group,
        dtype,
    )


def eligible(model: Any) -> list[tuple[Any, str, torch.nn.Linear]]:
    """(parent, attribute, layer) of every ``nn.Linear`` that becomes int4: not named like an
    embedding, the output head or a norm, and of a shape the kernel takes."""
    return [
        (module, child_name, child)
        for name, module in list(model.named_modules())
        for child_name, child in list(module.named_children())
        if type(child) is torch.nn.Linear
        and not any(
            part in (f"{name}.{child_name}" if name else child_name).lower() for part in _SKIP
        )
        and _fits(child, GROUP)
    ]


def convert(model: Any, device: str, dtype: torch.dtype = torch.bfloat16) -> int:
    """Replace eligible ``nn.Linear`` layers by int4 ones on ``device``, then move the rest.

    Returns the number of layers converted. Layers named like embeddings, the output head or
    norms, and layers whose shapes the kernel does not take, stay in ``dtype``.
    """
    layers = eligible(model)
    for module, child_name, child in layers:
        setattr(module, child_name, quantize_linear(child, device, GROUP, dtype))
    converted = len(layers)
    model.to(device)
    for p in model.parameters():  # the rest (embeddings, norms, head) in half precision
        if p.is_floating_point() and p.dtype != dtype:
            p.data = p.data.to(dtype)
    import gc

    gc.collect()  # the per-layer temporaries of the conversion
    if device == "mps":
        torch.mps.empty_cache()
    return converted


def works(device: str) -> str:
    """Empty if a tiny int4 layer runs on ``device`` with torch's kernel, else the reason."""
    if device != "mps":
        return "torch int4pack is used on MPS only (CUDA and CPU have other back ends)"
    try:
        layer = torch.nn.Linear(256, 128)
        q = quantize_linear(layer, device)
        q(torch.randn(2, 256, device=device, dtype=torch.bfloat16))
        return ""
    except Exception as e:  # noqa: BLE001 - any failure means "not here"
        return f"torch int4pack does not run on {device}: {type(e).__name__}: {e}"[:200]
