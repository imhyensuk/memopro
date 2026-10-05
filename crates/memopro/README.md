# memopro (Rust core)

The Rust core of [memopro](https://github.com/imhyensuk/memopro): run work that exceeds a
machine's memory losslessly, within a guaranteed memory ceiling. "Memory" means hardware memory
(GPU/RAM), not agent or conversation memory.

**Status: alpha (0.1.0-alpha.1).** Main pieces:

- `rt`: the budgeted runtime. Each large buffer has a recipe (a verified region of a source file,
  or a computation from other buffers) and a state; under the budget the runtime keeps it,
  compresses it losslessly in memory, drops it and re-reads its source (bypassing the page cache,
  digest-checked) or recomputes it, choosing by measured cost and predicted reuse. A service
  thread prefetches in the learned order; `predict` estimates the slowdown before running.
  Nothing is written to disk.
- `rt::pager` (Linux): transparent paging of anonymous memory with userfaultfd, compressed in
  memory, for unchanged programs.
- `hwinfo`, `pressure`, `residency`, `spill`, `codec`: host memory and limits, OS memory pressure,
  file-backed and GPU no-copy mappings, verified re-reads, and the compression codec.

The C ABI (`mp_*`, `include/memopro.h`) lives in `memopro-c`; Python bindings in `memopro-py`
(PyPI `memopro`).

License: MIT OR Apache-2.0.
