"""EDR v1 (0308): bf16 weights as a 1-byte base and a 4-bit residual, lossless.

- base byte: sign (bit 7) | exponent index (bits 3-6; 15 = escape) | top 3 mantissa bits
- residual: the low 4 mantissa bits, two per byte (element ``2k`` in the low nibble)
- table: the tensor's 15 most frequent exponents; other exponents are escapes, kept as
  (position, original bits)

Base and residual give back every bit (12 bits per weight instead of 16). The base alone is a
draft: the same bf16 values with their low 4 mantissa bits cleared. :func:`encode` and
:func:`decode` run in the Rust core on the CPU; :func:`decode_on` rebuilds on any torch device
(Apple GPU, CUDA) with integer tensor operations only, a slice at a time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from memopro import _core

# elements decoded per slice on a device: its int64 gather indices take 8 bytes each
DEVICE_SLICE = 1 << 22


@dataclass
class Parts:
    """One bf16 tensor in EDR v1 form (NumPy arrays and bytes, on the CPU)."""

    shape: tuple[int, ...]
    table: bytes  # 15 exponents
    base: Any  # uint8, one per element
    resid: Any  # uint8, ceil(n / 2)
    esc_index: bytes  # little-endian u64 positions
    esc_bits: bytes  # little-endian u16 original bits

    @property
    def nbytes(self) -> int:
        return (
            len(self.table)
            + self.base.nbytes
            + self.resid.nbytes
            + len(self.esc_index)
            + len(self.esc_bits)
        )


def _bits(t: Any) -> Any:
    """A bf16 tensor's bits as a contiguous NumPy uint16 array on the CPU (no copy if it can)."""
    import torch

    if t.dtype != torch.bfloat16:
        raise ValueError(f"EDR v1 encodes bfloat16 tensors, not {t.dtype}")
    return t.detach().cpu().contiguous().reshape(-1).view(torch.int16).numpy().view("uint16")


def encode(t: Any) -> Parts:
    """EDR v1 form of a bfloat16 tensor."""
    import numpy as np

    src = _bits(t)
    raw = src.view(np.uint8)  # the core takes byte buffers (RS2)
    table = _core.edr_table(raw)
    base = np.empty(src.size, np.uint8)
    resid = np.empty((src.size + 1) // 2, np.uint8)
    esc_index, esc_bits = _core.edr_encode(raw, table, base, resid)
    return Parts(tuple(t.shape), table, base, resid, esc_index, esc_bits)


def decode(p: Parts, draft: bool = False) -> Any:
    """The bf16 tensor back on the CPU: every bit, or the draft (low 4 mantissa bits cleared)."""
    import numpy as np
    import torch

    n = p.base.size
    out = np.empty(n, np.uint16)
    resid = None if draft else p.resid
    _core.edr_decode(p.base, resid, p.table, p.esc_index, p.esc_bits, out.view(np.uint8))
    return torch.from_numpy(out.view(np.int16)).view(torch.bfloat16).reshape(p.shape)


def device_parts(p: Parts, device: Any) -> dict[str, Any]:
    """The parts as tensors on `device`, ready for :func:`decode_on`."""
    import numpy as np
    import torch

    # what each base byte stands for (sign | exponent | top 3 mantissa bits), as the Rust core
    lut = [
        ((v >> 7) << 15)
        | ((p.table[(v >> 3) & 15] if (v >> 3) & 15 < 15 else 0) << 7)
        | ((v & 7) << 4)
        for v in range(256)
    ]
    table = torch.tensor([x - 65536 if x >= 32768 else x for x in lut], dtype=torch.int16)
    index = torch.from_numpy(np.frombuffer(p.esc_index, np.int64).copy())
    bits = torch.from_numpy(np.frombuffer(p.esc_bits, np.int16).copy())
    return {
        "shape": p.shape,
        "table": table.to(device),
        "base": torch.from_numpy(p.base).to(device),
        "resid": torch.from_numpy(p.resid).to(device),
        "esc_index": index.to(device),
        "esc_bits": bits.to(device),
    }


def decode_on(d: dict[str, Any], draft: bool = False, out: Any = None) -> Any:
    """Rebuild a bf16 tensor where its parts are (any torch device), with integer operations
    only: a 256-entry lookup per base byte and an or with the residual nibble, a slice at a
    time."""
    import torch

    base, table = d["base"], d["table"]
    n = base.numel()
    if out is None:
        out = torch.empty(n, dtype=torch.int16, device=base.device)
    else:
        out = out.reshape(-1).view(torch.int16)
    for start in range(0, n, DEVICE_SLICE):
        stop = min(n, start + DEVICE_SLICE)
        x = table[base[start:stop].int()]  # one lookup per element
        if not draft:
            r = d["resid"][start // 2 : (stop + 1) // 2].to(torch.int16)
            lo = torch.stack((r & 15, r >> 4), dim=1).reshape(-1)[: stop - start]
            x |= lo
        out[start:stop] = x
    index, bits = d["esc_index"], d["esc_bits"]
    if index.numel():
        out[index] = bits if not draft else bits & -16  # -16 = 0xFFF0
    return out.view(torch.bfloat16).reshape(d["shape"])
