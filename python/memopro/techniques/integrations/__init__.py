"""Adapters for existing, proven techniques used as optional backends (never reimplemented, 0010).

Planned for v0.2 (A2): safetensors lazy loading, accelerate offload, torch checkpointing, torchao,
bitsandbytes, 8-bit optimizers, KV-cache quantization. Each adapter imports its backend lazily
and reports why it is unavailable instead of failing (U3).
"""
