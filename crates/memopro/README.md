# memopro

**Keep large buffers within a hard memory ceiling — losslessly, without writing to disk.**

The Rust core of [memopro](https://github.com/imhyensuk/memopro). Use it to run work whose data is
larger than the memory you want it to use: you set a budget, register your large buffers, and pin
them while you use them. Buffers that do not fit come back bit for bit when you need them again.

**Status: alpha (0.1.0); the API may change.**

```rust
use memopro::rt::{Config, Runtime};

let rt = Runtime::new(Config::new(256 << 20))?;   // 256 MiB ceiling for managed buffers
let id = rt.alloc(64 << 20, 4)?;                  // 64 MiB of float32-sized elements
rt.pin(id, true)?.as_mut_slice()?.fill(0);        // in memory only while pinned
println!("{:?}", rt.stats());                     // what was kept, compressed, re-read
# Ok::<(), memopro::Error>(())
```

## Features

- **Budgeted runtime (`rt::Runtime`).** Allocate buffers, map regions of existing files
  (`add_file`) or define buffers computed from others (`derive`). Total memory stays within the
  budget; a request that cannot fit is refused instead of overrunning.
- **Lossless.** Data that does not fit is compressed in memory, re-read from its file (checked
  against a digest) or recomputed. What you read back is always exactly what you wrote.
- **No disk writes.** The runtime never creates swap or cache files.
- **Prefetching and prediction.** Buffers used in a repeating order are brought back ahead of
  time; `predict` estimates the extra work of a pass before you run it.
- **Transparent paging (`rt::Pager`, Linux and macOS).** Hand out ordinary memory that plain loads
  and stores can use, kept within the budget without any changes to the code that touches it.
- **Utilities.** Fast lossless compression for numeric data (`codec`), host memory, container
  limits and disk capacity (`hwinfo`), the OS memory-pressure signal (`pressure`), and read-only
  file mappings (`residency`, Unix).

## Also available

- **Python:** [`pip install memopro`](https://pypi.org/project/memopro/) — LLM fine-tuning and
  generation beyond device memory, NumPy and PyTorch integration, and a CLI.
- **C:** a C ABI (`mp_*` functions, `include/memopro.h`) in the
  [repository](https://github.com/imhyensuk/memopro).

## Platforms

Linux and macOS; Windows for the runtime without transparent paging. Rust 1.85+.

## License

MIT OR Apache-2.0.
