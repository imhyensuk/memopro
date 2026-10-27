# memopro (Rust core)

The Rust core of [memopro](https://github.com/imhyensuk/memopro): run work that needs more memory
than the machine has, losslessly, within a guaranteed memory ceiling. "Memory" means hardware
memory (GPU/RAM), not agent or conversation memory.

**Status: beta (0.2.0-beta.1); the API may still change.** Verified on 8-16 GB machines (Apple
silicon, Linux, Colab T4); the design does not depend on the memory size.

```rust
use memopro::rt::{Config, Runtime};

let rt = Runtime::new(Config::new(256 << 20))?;   // 256 MiB ceiling for managed buffers
let id = rt.alloc(64 << 20, 4)?;                  // 64 MiB of float32-sized elements
rt.pin(id, true)?.as_mut_slice()?.fill(0);        // in memory only while pinned
println!("{:?}", rt.stats());                     // what was kept, compressed, re-read
# Ok::<(), memopro::Error>(())
```

## Modules

- `rt`: the budgeted runtime. Each large buffer has a recipe (a verified region of a source file,
  or a computation from other buffers) and a state; under the budget the runtime keeps it,
  compresses it losslessly in memory, drops it and re-reads its source (bypassing the page cache,
  digest-checked) or recomputes it, choosing by measured cost and predicted reuse. A service
  thread prefetches in the learned order; `predict` estimates the slowdown before running. The
  runtime never writes to disk.
- `rt::pager` (Linux, macOS): transparent paging of anonymous memory for unchanged programs,
  compressed in memory. Linux serves the page faults with userfaultfd, macOS with signals; other
  systems get `Error::Unsupported` and use `rt::Runtime` buffers.
- `codec`, `edr`: byte shuffle + zstd, and EDR for bf16, the lossless in-memory compression.
- `verify`: re-reads of unchanged data from its original files, checked by digest.
- `residency` (Unix): no-copy Apple GPU buffers over runtime memory (macOS).
- `hwinfo`: host memory, container limits and disk capacity.

The C ABI (`mp_*`, `include/memopro.h`) lives in `memopro-c`; the Python package is
[`memopro` on PyPI](https://pypi.org/project/memopro/), built from `memopro-py`.

License: MIT OR Apache-2.0.
