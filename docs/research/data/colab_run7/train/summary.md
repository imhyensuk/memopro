# memopro Colab T4: training and development

run `20260929-023029`, cell `train`

| model | cap | scenario | status | steps | batch used | tokens/s | median step s | peak alloc | peak GPU used (nvidia-smi) | final loss | max |loss-plain| | param diff vs plain | memopro techniques | check predicted | measured | check error | error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | full | accel_find_batch | ok | 30 | 2 (accelerate) | 874 | 1.172 | 14.03 GiB | 14.53 GiB | 2.1403 | - | - | - | - | - | - |  |
| Qwen2.5-0.5B | full | check | ok | - | 8 | - | - | 14.03 GiB | 14.45 GiB | - | - | - | - | 25.55 GiB | oom | - |  |
| Qwen2.5-0.5B | full | hf_ckpt | oom | 1 | 8 | 805 | 5.089 | 10.55 GiB | 14.52 GiB | 3.3152 | - | - | - | - | - | - | OutOfMemoryError: CUDA out of memory. Tried to allocate 2.32 GiB. GPU 0 has a to |
| Qwen2.5-0.5B | full | memopro | ok | 30 | 8 (micro 1) | 889 | 4.606 | 9.95 GiB | 11.43 GiB | 2.2221 | - | - | micro-batch 1 | - | - | - |  |
| Qwen2.5-0.5B | full | memopro_lossless | ok | 30 | 8 (micro 1) | 890 | 4.604 | 9.95 GiB | 11.43 GiB | 2.2221 | - | - | micro-batch 1 | - | - | - |  |
| Qwen2.5-0.5B | full | plain | oom | 0 | 8 | - | - | 14.03 GiB | 14.45 GiB | - | - | - | - | - | - | - | OutOfMemoryError: CUDA out of memory. Tried to allocate 112.00 MiB. GPU 0 has a  |
| Qwen2.5-0.5B | full | plain_amp | oom | 0 | 8 | - | - | 13.48 GiB | 14.04 GiB | - | - | - | - | - | - | - | OutOfMemoryError: CUDA out of memory. Tried to allocate 2.32 GiB. GPU 0 has a to |
| Qwen2.5-0.5B | 50% | accel_find_batch | error | 0 | 1 (accelerate) | - | - | 7.08 GiB | 7.42 GiB | - | - | - | - | - | - | - | RuntimeError: No executable batch size found, reached zero. |
| Qwen2.5-0.5B | 50% | check | ok | - | 8 | - | - | 7.08 GiB | 7.40 GiB | - | - | - | - | 25.55 GiB | oom | - |  |
| Qwen2.5-0.5B | 50% | hf_ckpt | oom | 0 | 8 | - | - | 6.86 GiB | 7.17 GiB | - | - | - | - | - | - | - | OutOfMemoryError: CUDA out of memory. Tried to allocate 2.32 GiB. GPU 0 has a to |
| Qwen2.5-0.5B | 50% | memopro | error | - | 8 | - | - | 7.06 GiB | 7.41 GiB | - | - | - | - | - | - | - | BudgetExceeded: training does not fit even with micro-batch 1, checkpointing and |
| Qwen2.5-0.5B | 50% | plain | oom | 0 | 8 | - | - | 7.08 GiB | 7.40 GiB | - | - | - | - | - | - | - | OutOfMemoryError: CUDA out of memory. Tried to allocate 28.00 MiB. GPU 0 has a t |
| Qwen2.5-1.5B | full | accel_find_batch | error | 0 | 1 (accelerate) | - | - | 14.32 GiB | 14.56 GiB | - | - | - | - | - | - | - | RuntimeError: No executable batch size found, reached zero. |
| Qwen2.5-1.5B | full | check | ok | - | 4 | - | - | 14.31 GiB | 14.54 GiB | - | - | - | - | 33.02 GiB | oom | - |  |
| Qwen2.5-1.5B | full | hf_ckpt | oom | 0 | 4 | - | - | 13.71 GiB | 14.55 GiB | - | - | - | - | - | - | - | OutOfMemoryError: CUDA out of memory. Tried to allocate 54.00 MiB. GPU 0 has a t |
| Qwen2.5-1.5B | full | memopro | error | - | 4 | - | - | 14.27 GiB | 14.55 GiB | - | - | - | - | - | - | - | BudgetExceeded: training does not fit even with micro-batch 1, checkpointing and |
| Qwen2.5-1.5B | full | memopro_lossless | error | - | 4 | - | - | 14.27 GiB | 14.55 GiB | - | - | - | - | - | - | - | BudgetExceeded: training does not fit even with micro-batch 1, checkpointing and |
| Qwen2.5-1.5B | full | plain | oom | 0 | 4 | - | - | 14.31 GiB | 14.54 GiB | - | - | - | - | - | - | - | OutOfMemoryError: CUDA out of memory. Tried to allocate 70.00 MiB. GPU 0 has a t |
| Qwen2.5-1.5B | full | plain_amp | oom | 0 | 4 | - | - | 14.11 GiB | 14.55 GiB | - | - | - | - | - | - | - | OutOfMemoryError: CUDA out of memory. Tried to allocate 12.00 MiB. GPU 0 has a t |
| Qwen2.5-7B-Instruct | full | qlora_hf | ok | 20 | 4 | 237 | 8.637 | 11.96 GiB | 13.30 GiB | 2.1807 | - | - | - | - | - | - |  |
| Qwen2.5-7B-Instruct | full | qlora_memopro | ok | 20 | 4 (micro 2) | 131 | 15.637 | 13.68 GiB | 14.51 GiB | 2.1387 | - | - | micro-batch 2 | - | - | - |  |
| gpt2 | full | accel_find_batch | ok | 30 | 14 (accelerate) | 3604 | 1.989 | 13.00 GiB | 14.36 GiB | 3.0943 | - | - | - | - | - | - |  |
| gpt2 | full | check | ok | - | 16 | - | - | 12.40 GiB | 13.24 GiB | - | - | - | - | 14.63 GiB | oom | - |  |
| gpt2 | full | hf_ckpt | ok | 30 | 16 | 3320 | 2.467 | 6.34 GiB | 7.64 GiB | 3.0860 | - | - | - | - | - | - |  |
| gpt2 | full | memopro | ok | 30 | 16 (micro 15) | 3632 | 2.256 | 13.16 GiB | 14.34 GiB | 3.0860 | - | - | micro-batch 15 | - | - | - |  |
| gpt2 | full | memopro_lossless | ok | 30 | 16 (micro 15) | 3638 | 2.252 | 13.16 GiB | 14.34 GiB | 3.0860 | - | - | micro-batch 15 | - | - | - |  |
| gpt2 | full | plain | oom | 1 | 16 | 3726 | 2.199 | 13.00 GiB | 13.24 GiB | 3.6548 | - | - | - | - | - | - | OutOfMemoryError: CUDA out of memory. Tried to allocate 1.54 GiB. GPU 0 has a to |
| gpt2 | full | plain_amp | ok | 30 | 16 | 11210 | 0.731 | 12.20 GiB | 13.89 GiB | 3.0946 | - | - | - | - | - | - |  |
| gpt2 | 30% | accel_find_batch | ok | 30 | 2 (accelerate) | 3196 | 0.320 | 4.28 GiB | 4.49 GiB | 2.9890 | - | - | - | - | - | - |  |
| gpt2 | 30% | check | ok | - | 16 | - | - | 4.28 GiB | 4.45 GiB | - | - | - | - | 14.63 GiB | oom | - |  |
| gpt2 | 30% | hf_ckpt | oom | 0 | 16 | - | - | 3.87 GiB | 4.11 GiB | - | - | - | - | - | - | - | OutOfMemoryError: CUDA out of memory. Tried to allocate 1.54 GiB. GPU 0 has a to |
| gpt2 | 30% | memopro | ok | 30 | 16 (micro 2) | 3558 | 2.302 | 3.94 GiB | 4.16 GiB | 3.0860 | - | - | micro-batch 2 | - | - | - |  |
| gpt2 | 30% | plain | oom | 0 | 16 | - | - | 4.28 GiB | 4.45 GiB | - | - | - | - | - | - | - | OutOfMemoryError: CUDA out of memory. Tried to allocate 96.00 MiB. GPU 0 has a t |

Reading guide (objective comparison):
- `plain` on the full GPU is the reference; `max |loss-plain|` and `param diff vs plain` show whether a method trains the same model (memopro's micro-batches are exact up to float rounding; AMP and accelerate's smaller batch are not).
- `cap` limits the process to that fraction of the T4 (emulating a smaller GPU); memopro gets the same size as its budget.
- `check` rows: memopro.check's predicted training peak vs the measured steady-state peak (fp32 AdamW); `oom` means the plain step does not fit.
- tokens/s uses the batch actually trained (accelerate may shrink it).
