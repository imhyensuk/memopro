"""Statistics on sampled tensors (CPU, small): entropy, outliers, quantization error.

All functions take a 1-D CPU sample. Definitions follow E005 (byte-plane order-0 entropy) and
0036 B4 (needed bits from blockwise-absmax reconstruction error, not gradient impact).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

BITS = (8, 4, 2)
BLOCK = 64
# Relative L2 error under which a bit width counts as "enough" (0036 B4). bf16 rounding alone is
# about 2**-9 relative, so 1e-2 is a lenient, reconstruction-only criterion.
TOLERANCE = 1e-2
# A tensor has "massive" values when max|x| exceeds this many times the median of its non-zero
# |x| (heuristic; exact zeros are excluded so sparse tensors do not read as infinite).
MASSIVE_RATIO = 100.0


@dataclass(frozen=True)
class SampleStats:
    stored_bits: int
    entropy_bits: float  # order-0 entropy summed over byte planes, bits per element
    peak_to_median: float | None  # floats only; over non-zero magnitudes
    zero_fraction: float | None  # floats only
    quant_rel_error: dict[int, float] | None  # bits -> relative L2 error (light mode)


def byte_plane_entropy(sample: torch.Tensor) -> float:
    itemsize = sample.element_size()
    planes = sample.contiguous().view(torch.uint8).reshape(-1, itemsize)
    total = 0.0
    n = planes.shape[0]
    for b in range(itemsize):
        counts = torch.bincount(planes[:, b].to(torch.int64), minlength=256).double()
        p = counts[counts > 0] / n
        total += float(-(p * torch.log2(p)).sum())
    return total


def peak_to_median(x: torch.Tensor) -> float | None:
    a = x.abs()
    a = a[a > 0]
    if a.numel() == 0:
        return None
    return float(a.max()) / float(a.median())


def quant_rel_error(x: torch.Tensor, bits: int, block: int = BLOCK) -> float:
    """Relative L2 error of symmetric blockwise-absmax quantization to ``bits`` bits."""
    n = x.numel()
    pad = (-n) % block
    xb = torch.nn.functional.pad(x, (0, pad)).reshape(-1, block)
    levels = 2 ** (bits - 1) - 1
    scale = xb.abs().amax(dim=1, keepdim=True) / levels
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    xq = (xb / scale).round().clamp(-levels, levels) * scale
    err = (xq.reshape(-1)[:n] - x).norm()
    ref = x.norm()
    return float(err / ref) if float(ref) > 0 else 0.0


def analyse(sample: torch.Tensor, light: bool) -> SampleStats:
    stored = sample.element_size() * 8
    entropy = byte_plane_entropy(sample) if sample.numel() else 0.0
    if not sample.is_floating_point() or sample.numel() == 0:
        return SampleStats(stored, entropy, None, None, None)
    x = sample.float()
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        return SampleStats(stored, entropy, None, None, None)
    zeros = float((x == 0).float().mean())
    errors = {b: quant_rel_error(x, b) for b in BITS} if light else None
    return SampleStats(stored, entropy, peak_to_median(x), zeros, errors)


def needed_bits(errors: list[dict[int, float]]) -> int | None:
    """Smallest tested bit width whose worst-case sampled error is within TOLERANCE (K3).

    None means even 8 bits exceed the tolerance for some sampled tensor.
    """
    if not errors:
        return None
    for bits in sorted(BITS):
        if max(e[bits] for e in errors) <= TOLERANCE:
            return bits
    return None
