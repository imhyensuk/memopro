# memopro on a Colab T4: current code (E048)

run `20261007-145257`, cell `full`

| check | criterion | result | detail |
|---|---|---|---|
| T1 | repository tests on the T4: no failures | fail | {'failed': 1, 'passed': 33, 'skipped': 10} ['FAILED tests/test_rt_torch.py::test_rows_after_the_prompt_are_computed_as_in_plain_generation[cpu]'] |
| L-R1 | W1/W2 results equal the plain run at every ceiling | pass | [True, True, True, True, True, True] |
| L-B1 | peak RSS <= ceiling + 32 MiB at 0.75 and 0.5; W2 overruns 0 | pass | W1 075: 0.59 GiB vs 0.59 GiB; W2 075: 0.82 GiB vs 0.82 GiB; W1 05: 0.40 GiB vs 0.39 GiB; W2 05: 0.55 GiB vs 0.55 GiB |
| L-O1 | ample ceiling: W1 time <= 1.15x plain; W2 waited <= 0.2 s | fail | W1 5.01 -> 5.99 s; W2 waited 0.00 s |
| L-P1 | W2 estimate within 0.25 x waited + 0.2 s at 0.75 and 0.5 | pass | 075: estimate 7.93 vs waited 7.20 s; 05: estimate 12.60 vs waited 12.15 s |
| U1 | 7B 16-bit: Unsloth does not complete; memopro completes at both budgets | fail | Unsloth ok ; memopro ['ok', 'ok'] |
| U2 | memopro losses bit-identical across budgets (3B, 7B) | pass | [True, True] |
| U3 | 3B 16-bit: step s, GPU peak, host peak (PEFT / memopro / Unsloth) | n/a | step s 5.12 / 5.74 / 0.59; GPU 6.43 GiB / 6.41 GiB / 6.41 GiB; host 4.69 GiB / 3.78 GiB / 4.40 GiB; Unsloth dtype torch.float16 |
| U4 | 7B: Unsloth 4-bit vs memopro 16-bit (step s, GPU, first loss) | n/a | step s 1.74 / 42.37; GPU 5.84 GiB / 14.31 GiB; first loss 2.5641 / 2.5411 |
| U5 | 3B loss difference from PEFT (max abs) | n/a | memopro 0.00666, Unsloth 0.00657 |
| V | DINOv2 streamed inference output = loaded normally (both budgets) | pass | plain ok 1.57 s; memopro [('ok', '1.65'), ('ok', '1.69')] |
| VL | DINOv2 LoRA losses bit-identical across budgets | fail | ['ok', 'ok'] [2.4394, 2.1968, 1.8823, 1.5135, 1.3343] |
| S | DINOv2 streamed inference time <= 2x plain (0202) | pass | plain 1.57, memopro 1.69 s; copies {'copies': 807, 'copy_bytes': 4545923072, 'hits': 807, 'prefetched': 0, 'gpu_budget': 13181553869, 'gpu_cache_bytes': 4545923072} |
| D-image | finished cases give the plain result (report: peak, time, overruns) | pass | ok; peak 0.77 GiB vs ceiling 0.77 GiB (plain 1.54 GiB); 12.0 -> 32.2 s; overruns 0 |
| D-classify | finished cases give the plain result (report: peak, time, overruns) | pass | ok; peak 1.19 GiB vs ceiling 1.19 GiB (plain 2.38 GiB); 17.6 -> 61.7 s; overruns 0 |
| D-dataframe | finished cases give the plain result (report: peak, time, overruns) | n/a | timeout; peak 0.00 GiB vs ceiling - (plain 1.66 GiB); 17.6 -> - s; overruns None |
| D-simulate | finished cases give the plain result (report: peak, time, overruns) | pass | ok; peak 1.28 GiB vs ceiling 1.08 GiB (plain 2.16 GiB); 50.1 -> 451.7 s; overruns 60871 |

Pre-registered: docs/research/0237 (and 0227, 0197, 0202, 0231).
Elapsed 69 min of 135.
Judged checks: T1 fail, L-R1 pass, L-B1 pass, U1 fail, U2 pass, V pass, VL fail

## All cases

| case | status | wall s | step s | GPU peak | host/RSS peak |
|---|---|---|---|---|---|
| D_classify_enable | ok | 62.4 | - | 0.00 GiB | 1.19 GiB |
| D_classify_plain | ok | 18.2 | - | 0.00 GiB | 2.38 GiB |
| D_dataframe_enable | timeout | 600.1 | - | 0.00 GiB | - |
| D_dataframe_plain | ok | 18.3 | - | 0.00 GiB | 1.66 GiB |
| D_image_enable | ok | 32.7 | - | 0.00 GiB | 0.77 GiB |
| D_image_plain | ok | 12.3 | - | 0.00 GiB | 1.54 GiB |
| D_simulate_enable | ok | 452.2 | - | 0.00 GiB | 1.28 GiB |
| D_simulate_plain | ok | 50.4 | - | 0.00 GiB | 2.16 GiB |
| L_W1_05 | ok | 17.0 | - | 0.00 GiB | 0.40 GiB |
| L_W1_075 | ok | 11.3 | - | 0.00 GiB | 0.59 GiB |
| L_W1_ample | ok | 6.3 | - | 0.00 GiB | 0.79 GiB |
| L_W1_plain | ok | 5.2 | - | 0.00 GiB | 0.79 GiB |
| L_W2_05 | ok | 22.1 | - | 0.00 GiB | 0.55 GiB |
| L_W2_075 | ok | 16.9 | - | 0.00 GiB | 0.82 GiB |
| L_W2_ample | ok | 8.3 | - | 0.00 GiB | 1.11 GiB |
| L_W2_plain | ok | 7.4 | - | 0.00 GiB | 1.10 GiB |
| Qwen2.5-3B-Instruct__M_2048 | ok | 114.9 | 5.74 | 6.41 GiB | 3.78 GiB |
| Qwen2.5-3B-Instruct__M_3072 | ok | 118.1 | 5.81 | 6.41 GiB | 4.74 GiB |
| Qwen2.5-3B-Instruct__P | ok | 112.9 | 5.12 | 6.43 GiB | 4.69 GiB |
| Qwen2.5-3B-Instruct__U16 | ok | 89.6 | 0.59 | 6.41 GiB | 4.40 GiB |
| Qwen2.5-7B-Instruct__M_4096 | ok | 528.2 | 42.37 | 14.31 GiB | 5.64 GiB |
| Qwen2.5-7B-Instruct__M_6144 | ok | 536.1 | 43.53 | 14.31 GiB | 7.66 GiB |
| Qwen2.5-7B-Instruct__P | oom | 95.7 | - | 14.56 GiB | 4.17 GiB |
| Qwen2.5-7B-Instruct__U16 | ok | 145.3 | 1.28 | 14.37 GiB | 6.28 GiB |
| Qwen2.5-7B-Instruct__U4 | ok | 129.2 | 1.74 | 5.84 GiB | 6.17 GiB |
| VL_dinov2-giant__memopro_1100 | ok | 39.1 | 3.24 | 12.65 GiB | 2.54 GiB |
| VL_dinov2-giant__memopro_2200 | ok | 38.2 | 3.27 | 12.65 GiB | 3.62 GiB |
| VL_dinov2-giant__plain | ok | 36.2 | 3.16 | 12.88 GiB | 4.58 GiB |
| V_dinov2-giant__memopro_1100 | ok | 33.4 | - | 4.81 GiB | 2.36 GiB |
| V_dinov2-giant__memopro_2200 | ok | 24.0 | - | 4.81 GiB | 3.44 GiB |
| V_dinov2-giant__plain | ok | 38.1 | - | 4.74 GiB | 5.13 GiB |
| tests | failed | 39.2 | - | 0.10 GiB | - |

## Not ok

- `D_dataframe_enable` (timeout): timeout
- `Qwen2.5-7B-Instruct__P` (oom): OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 14.56 GiB of which 7.81 MiB is free. Including non-PyTorch memory, this process has 14.55 GiB memory in use. Of the allocated memory 14.29 GiB is allocated by PyTorch, and 119.36 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=
- `tests` (failed): ['FAILED tests/test_rt_torch.py::test_rows_after_the_prompt_are_computed_as_in_plain_generation[cpu]']
