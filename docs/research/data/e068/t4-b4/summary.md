# memopro main@e29d0a2f65f8 on Colab T4 (E068)

run `20261029-042544`, cell `beta_t4`

| check | criterion | result | detail |
|---|---|---|---|
| B1 | repository tests at the tag: no failures | pass | {'passed': 173, 'skipped': 23}  |
| B2 Qwen2.5-3B-Instruct | finetune completes at both ceilings, losses bit-identical | pass | ['ok', 'ok'] [] |
| B3 Qwen2.5-3B-Instruct__ft_6144 | process peak (VmHWM) <= ceiling | pass | ok: peak 2.05 GiB vs ceiling 6.00 GiB |
| B3 Qwen2.5-3B-Instruct__ft_8192 | process peak (VmHWM) <= ceiling | pass | ok: peak 2.05 GiB vs ceiling 8.00 GiB |
| B3 Qwen2.5-3B-Instruct__gen_6144 | process peak (VmHWM) <= ceiling | pass | ok: peak 2.07 GiB vs ceiling 6.00 GiB |
| B4 Qwen2.5-3B-Instruct | generate's text = the model loaded normally | pass | plain ok 0.08 s/token, memopro ok 0.36 s/token |
| R1 Qwen2.5-3B-Instruct | report: step s plain / memopro at both ceilings; first loss | n/a | 5.14 / ['5.78', '5.89']; 2.6548 / 2.6548 |
| B2 Qwen2.5-7B-Instruct | finetune completes at both ceilings, losses bit-identical | pass | ['ok', 'ok'] [] |
| B3 Qwen2.5-7B-Instruct__ft_7168 | process peak (VmHWM) <= ceiling | pass | ok: peak 3.61 GiB vs ceiling 7.00 GiB |
| B3 Qwen2.5-7B-Instruct__ft_9216 | process peak (VmHWM) <= ceiling | pass | ok: peak 3.61 GiB vs ceiling 9.00 GiB |
| B3 Qwen2.5-7B-Instruct__gen_7168 | process peak (VmHWM) <= ceiling | fail | oom: peak 6.40 GiB vs ceiling 7.00 GiB |
| R1 Qwen2.5-7B-Instruct | report: step s plain / memopro at both ceilings; first loss | n/a | - / ['41.55', '41.70']; - / 2.5411 |
| B5 image | memopro run at half: plain result, peak <= ceiling + 32 MiB | pass | ok; peak 0.40 GiB vs 0.39 GiB; 4.8 -> 14.1 s |
| B5 classify | memopro run at half: plain result, peak <= ceiling + 32 MiB | pass | ok; peak 1.19 GiB vs 1.19 GiB; 16.3 -> 48.6 s |
| R2 | report: Unsloth 16-bit 3B step s (memopro at the larger ceiling) | n/a | ok 0.47 / 5.89 |
| R3 | report (0363): memopro.finetune on the Unsloth model = Unsloth's loop | n/a | ok; bits equal: False; step s 0.49 vs 0.47; first loss 2.6621 / 2.6621 |
| R4 | report (0363): memopro.generate on the Unsloth model = Unsloth's | n/a | ok; same text: True |

Pre-registered: docs/research/0346 (E068).
T4 runtime; elapsed 44 min of 120.
Judged: B1 pass, B2 Qwen2.5-3B-Instruct pass, B3 Qwen2.5-3B-Instruct__ft_6144 pass, B3 Qwen2.5-3B-Instruct__ft_8192 pass, B3 Qwen2.5-3B-Instruct__gen_6144 pass, B4 Qwen2.5-3B-Instruct pass, B2 Qwen2.5-7B-Instruct pass, B3 Qwen2.5-7B-Instruct__ft_7168 pass, B3 Qwen2.5-7B-Instruct__ft_9216 pass, B3 Qwen2.5-7B-Instruct__gen_7168 fail, B5 image pass, B5 classify pass

## All cases

| case | status | wall s | step s | host peak |
|---|---|---|---|---|
| D_classify_plain | ok | 16.9 | - | 2.38 GiB |
| D_classify_run | ok | 49.3 | - | 1.19 GiB |
| D_image_plain | ok | 5.2 | - | 0.79 GiB |
| D_image_run | ok | 14.4 | - | 0.40 GiB |
| Qwen2.5-3B-Instruct__U16 | ok | 91.3 | 0.47 | 4.24 GiB |
| Qwen2.5-3B-Instruct__U16_gen | ok | 49.3 | - | 4.68 GiB |
| Qwen2.5-3B-Instruct__U16_memopro | ok | 72.8 | 0.49 | 4.02 GiB |
| Qwen2.5-3B-Instruct__ft_6144 | ok | 95.1 | 5.78 | 2.05 GiB |
| Qwen2.5-3B-Instruct__ft_8192 | ok | 95.3 | 5.89 | 2.05 GiB |
| Qwen2.5-3B-Instruct__gen_6144 | ok | 32.1 | - | 2.07 GiB |
| Qwen2.5-3B-Instruct__plain_gen | ok | 33.5 | - | 4.63 GiB |
| Qwen2.5-3B-Instruct__plain_lora | ok | 100.9 | 5.14 | 4.69 GiB |
| Qwen2.5-7B-Instruct__ft_7168 | ok | 505.1 | 41.55 | 3.61 GiB |
| Qwen2.5-7B-Instruct__ft_9216 | ok | 500.0 | 41.70 | 3.61 GiB |
| Qwen2.5-7B-Instruct__gen_7168 | oom | 185.7 | - | 6.40 GiB |
| tests | ok | 73.6 | - | - |

## Not ok

- `Qwen2.5-7B-Instruct__gen_7168` (oom): OutOfMemoryError: CUDA out of memory. Tried to allocate 1.02 GiB. GPU 0 has a total capacity of 14.56 GiB of which 979.81 MiB is free. Including non-PyTorch memory, this process has 13.60 GiB memory in use. Of the allocated memory 13.32 GiB is allocated by PyTorch, and 159.81 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF
