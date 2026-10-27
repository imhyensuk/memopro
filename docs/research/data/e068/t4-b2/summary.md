# memopro 0.2.0b2 on Colab T4 (E068)

run `20261027-110822`, cell `beta_t4`

| check | criterion | result | detail |
|---|---|---|---|
| B1 | repository tests at the tag: no failures | pass | {'passed': 161, 'skipped': 23}  |
| B2 Qwen2.5-3B-Instruct | finetune completes at both ceilings, losses bit-identical | pass | ['ok', 'ok'] [] |
| B3 Qwen2.5-3B-Instruct__ft_6144 | process peak (VmHWM) <= ceiling | pass | ok: peak 5.57 GiB vs ceiling 6.00 GiB |
| B3 Qwen2.5-3B-Instruct__ft_8192 | process peak (VmHWM) <= ceiling | pass | ok: peak 7.33 GiB vs ceiling 8.00 GiB |
| B3 Qwen2.5-3B-Instruct__gen_6144 | process peak (VmHWM) <= ceiling | fail | ok: peak 7.19 GiB vs ceiling 6.00 GiB |
| B4 Qwen2.5-3B-Instruct | generate's text = the model loaded normally | pass | plain ok 0.09 s/token, memopro ok 2.05 s/token |
| R1 Qwen2.5-3B-Instruct | report: step s plain / memopro at both ceilings; first loss | n/a | 5.08 / ['5.53', '5.90']; 2.6548 / 2.6548 |
| B2 Qwen2.5-7B-Instruct | finetune completes at both ceilings, losses bit-identical | pass | ['ok', 'ok'] [] |
| B3 Qwen2.5-7B-Instruct__ft_7168 | process peak (VmHWM) <= ceiling | pass | ok: peak 6.37 GiB vs ceiling 7.00 GiB |
| B3 Qwen2.5-7B-Instruct__ft_9216 | process peak (VmHWM) <= ceiling | pass | ok: peak 8.36 GiB vs ceiling 9.00 GiB |
| B3 Qwen2.5-7B-Instruct__gen_7168 | process peak (VmHWM) <= ceiling | fail | oom: peak 6.83 GiB vs ceiling 7.00 GiB |
| R1 Qwen2.5-7B-Instruct | report: step s plain / memopro at both ceilings; first loss | n/a | - / ['42.23', '43.11']; - / 2.5411 |
| B5 image | memopro run at half: plain result, peak <= ceiling + 32 MiB | pass | ok; peak 0.40 GiB vs 0.39 GiB; 5.0 -> 14.5 s |
| B5 classify | memopro run at half: plain result, peak <= ceiling + 32 MiB | pass | ok; peak 1.19 GiB vs 1.19 GiB; 16.4 -> 49.5 s |
| R2 | report: Unsloth 16-bit 3B step s (memopro at the larger ceiling) | n/a | ok 0.48 / 5.90 |

Pre-registered: docs/research/0346 (E068).
T4 runtime; elapsed 52 min of 120.
Judged: B1 pass, B2 Qwen2.5-3B-Instruct pass, B3 Qwen2.5-3B-Instruct__ft_6144 pass, B3 Qwen2.5-3B-Instruct__ft_8192 pass, B3 Qwen2.5-3B-Instruct__gen_6144 fail, B4 Qwen2.5-3B-Instruct pass, B2 Qwen2.5-7B-Instruct pass, B3 Qwen2.5-7B-Instruct__ft_7168 pass, B3 Qwen2.5-7B-Instruct__ft_9216 pass, B3 Qwen2.5-7B-Instruct__gen_7168 fail, B5 image pass, B5 classify pass

## All cases

| case | status | wall s | step s | host peak |
|---|---|---|---|---|
| D_classify_plain | ok | 17.0 | - | 2.38 GiB |
| D_classify_run | ok | 50.1 | - | 1.19 GiB |
| D_image_plain | ok | 5.4 | - | 0.79 GiB |
| D_image_run | ok | 14.8 | - | 0.40 GiB |
| Qwen2.5-3B-Instruct__U16 | ok | 95.2 | 0.48 | 5.82 GiB |
| Qwen2.5-3B-Instruct__ft_6144 | ok | 102.8 | 5.53 | 5.57 GiB |
| Qwen2.5-3B-Instruct__ft_8192 | ok | 111.4 | 5.90 | 7.33 GiB |
| Qwen2.5-3B-Instruct__gen_6144 | ok | 148.9 | - | 7.19 GiB |
| Qwen2.5-3B-Instruct__plain_gen | ok | 40.6 | - | 4.63 GiB |
| Qwen2.5-3B-Instruct__plain_lora | ok | 100.1 | 5.08 | 4.69 GiB |
| Qwen2.5-7B-Instruct__ft_7168 | ok | 516.0 | 42.23 | 6.37 GiB |
| Qwen2.5-7B-Instruct__ft_9216 | ok | 521.6 | 43.11 | 8.36 GiB |
| Qwen2.5-7B-Instruct__gen_7168 | oom | 220.3 | - | 6.83 GiB |
| tests | ok | 71.1 | - | - |

## Not ok

- `Qwen2.5-7B-Instruct__gen_7168` (oom): OutOfMemoryError: CUDA out of memory. Tried to allocate 1.02 GiB. GPU 0 has a total capacity of 14.56 GiB of which 799.81 MiB is free. Including non-PyTorch memory, this process has 13.78 GiB memory in use. Of the allocated memory 13.33 GiB is allocated by PyTorch, and 328.44 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF
