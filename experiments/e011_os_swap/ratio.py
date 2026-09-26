"""E011-C: how well GPT-2 weights compress page by page (kernel compressor proxy) vs memopro.

The macOS compressor works on single pages; libcompression LZ4/LZFSE on 16 KiB (and 4 KiB)
pages approximates it. memopro's ``compress`` method compresses whole tensors (byte shuffle +
zstd). Both are measured on the same fp32 weights and on a bf16 copy.

Usage: .venv/bin/python -m experiments.e011_os_swap.ratio
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[2] / ".cache" / "hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from experiments._harness.env import capture, save_json
from experiments.e011_os_swap.common import apple_ratio
from experiments.e011_os_swap.holder import MODEL, REVISION
from memopro import _core

OUT = Path(__file__).resolve().parents[2] / "docs" / "research" / "data" / "e011"


def measure(tensors: dict[str, torch.Tensor]) -> dict[str, float]:
    raw = sum(t.numel() * t.element_size() for t in tensors.values())
    blob = b"".join(t.contiguous().view(torch.uint8).numpy().tobytes() for t in tensors.values())
    out = {"raw_bytes": raw}
    for alg in ("lz4", "lzfse"):
        for page in (16384, 4096):
            out[f"{alg}_{page // 1024}k"] = apple_ratio(blob, alg, page)
    for level in (1, 3):
        stored = sum(
            _core.codec_pack(
                t.contiguous().reshape(-1).view(torch.uint8).numpy(), t.element_size(), level
            ).stored_bytes
            for t in tensors.values()
        )
        out[f"memopro_zstd{level}"] = stored / raw
    return out


def main() -> None:
    path = hf_hub_download(MODEL, "model.safetensors", revision=REVISION)
    weights = {k: v for k, v in load_file(path).items() if v.is_floating_point()}
    result = {
        "fp32": measure({k: v.float() for k, v in weights.items()}),
        "bf16": measure({k: v.to(torch.bfloat16) for k, v in weights.items()}),
    }
    save_json(capture(__file__), OUT / "env_ratio.json")
    save_json(result, OUT / "ratio.json")
    for dtype, r in result.items():
        print(dtype, {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})


if __name__ == "__main__":
    main()
