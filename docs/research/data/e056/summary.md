# E056 summary: int4 file cache build, Qwen2.5-7B-Instruct, 8 GB M1

Command (from the repo root, `.cache` holding the model; before = `main` d55cbc3 code via
`PYTHONPATH=<checkout of d55cbc3>/python`, after = this branch via `PYTHONPATH=python`):

    E055_MODEL=7b /usr/bin/time -l .venv/bin/python -m experiments.e055_int4_stream.run --cache

The cache was deleted before each run (both runs built it from scratch).

| build | wall s | max RSS | peak memory footprint | cache bytes | sha256 (.f / .json) |
|---|---|---|---|---|---|
| before: convert all, then `_write_cache` | 103.5 | 1.69 GB | **7.83 GB** | 6,260,637,696 | before.sha256 |
| after: quantize + write one layer at a time | 59.5 | 3.14 GB | **2.16 GB** | 6,260,637,696 | after.sha256 (identical) |

Raw: before.time / after.time (`/usr/bin/time -l`), before_cache.json / after_cache.json
(the experiment's own record), before.sha256 / after.sha256.
Environment: macOS 27.0.1 arm64, Python 3.14.0, torch 2.14.0, torchao 0.18.0, transformers 5.17.0.
