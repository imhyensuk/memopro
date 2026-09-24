# memopro (Rust core)

Framework-agnostic memory toolkit for deep learning workloads. This crate is the Rust core of
[memopro](https://pypi.org/project/memopro/), a library that helps PyTorch developers work with
models that need more GPU/RAM than they have.

**Status: early development (0.0.x).** The crate currently exposes only version information.
Planned modules for 0.1: `hwinfo` (per-pool memory budgets, including container/cgroup limits),
`codec` (byte-shuffle + zstd lossless compression for floating-point tensors), `ledger` and `spill`
(tensor bookkeeping and chunked spill to disk).

"Memory" here means hardware memory (GPU/RAM), not agent or conversation memory.

License: MIT OR Apache-2.0.
