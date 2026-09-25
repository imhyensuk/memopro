# memopro (Rust core)

Framework-agnostic memory toolkit for deep learning workloads. This crate is the Rust core of
[memopro](https://pypi.org/project/memopro/), a library that helps PyTorch developers reclaim idle
memory and measure how much memory is wasted.

**Status: early development (0.0.x).** Skeleton modules: `hwinfo` (host memory, container limit,
disk capacity), `spill` (write-free restore from original files with digest verification, and
bounded spill files as a last resort), `ledger` (bookkeeping of hibernated buffers and SSD bytes
written), `pressure` (OS memory-pressure signal, later). `codec` (byte shuffle + zstd) is an
experiment prototype. Unbuilt functions return `Error::NotImplemented` naming the planned milestone.

"Memory" here means hardware memory (GPU/RAM), not agent or conversation memory.

License: MIT OR Apache-2.0.
