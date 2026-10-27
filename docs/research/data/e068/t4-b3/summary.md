# memopro 0.2.0b3 on Colab T4 (E068)

run `20261027-125240`, cell `beta_t4`

| check | criterion | result | detail |
|---|---|---|---|
| B1 | repository tests at the tag: no failures | pass | {'passed': 161, 'skipped': 23}  |
| B2 Qwen2.5-3B-Instruct | finetune completes at both ceilings, losses bit-identical | pass | ['ok', 'ok'] [] |
| B3 Qwen2.5-3B-Instruct__ft_6144 | process peak (VmHWM) <= ceiling | pass | ok: peak 5.56 GiB vs ceiling 6.00 GiB |
| B3 Qwen2.5-3B-Instruct__ft_8192 | process peak (VmHWM) <= ceiling | pass | ok: peak 7.34 GiB vs ceiling 8.00 GiB |
| B3 Qwen2.5-3B-Instruct__gen_6144 | process peak (VmHWM) <= ceiling | fail | error: peak 5.94 GiB vs ceiling 6.00 GiB |
| B4 Qwen2.5-3B-Instruct | generate's text = the model loaded normally | fail | plain ok 0.08 s/token, memopro error - s/token |
| R1 Qwen2.5-3B-Instruct | report: step s plain / memopro at both ceilings; first loss | n/a | 4.91 / ['5.04', '5.12']; 2.6548 / 2.6548 |
| B2 Qwen2.5-7B-Instruct | finetune completes at both ceilings, losses bit-identical | pass | ['ok', 'ok'] [] |
| B3 Qwen2.5-7B-Instruct__ft_7168 | process peak (VmHWM) <= ceiling | pass | ok: peak 6.36 GiB vs ceiling 7.00 GiB |
| B3 Qwen2.5-7B-Instruct__ft_9216 | process peak (VmHWM) <= ceiling | pass | ok: peak 8.37 GiB vs ceiling 9.00 GiB |
| B3 Qwen2.5-7B-Instruct__gen_7168 | process peak (VmHWM) <= ceiling | fail | error: peak 6.92 GiB vs ceiling 7.00 GiB |
| R1 Qwen2.5-7B-Instruct | report: step s plain / memopro at both ceilings; first loss | n/a | - / ['40.31', '40.29']; - / 2.5411 |
| B5 image | memopro run at half: plain result, peak <= ceiling + 32 MiB | pass | ok; peak 0.40 GiB vs 0.39 GiB; 5.1 -> 14.8 s |
| B5 classify | memopro run at half: plain result, peak <= ceiling + 32 MiB | pass | ok; peak 1.19 GiB vs 1.19 GiB; 16.8 -> 51.4 s |
| R2 | report: Unsloth 16-bit 3B step s (memopro at the larger ceiling) | n/a | ok 0.45 / 5.12 |

Pre-registered: docs/research/0346 (E068).
T4 runtime; elapsed 46 min of 120.
Judged: B1 pass, B2 Qwen2.5-3B-Instruct pass, B3 Qwen2.5-3B-Instruct__ft_6144 pass, B3 Qwen2.5-3B-Instruct__ft_8192 pass, B3 Qwen2.5-3B-Instruct__gen_6144 fail, B4 Qwen2.5-3B-Instruct fail, B2 Qwen2.5-7B-Instruct pass, B3 Qwen2.5-7B-Instruct__ft_7168 pass, B3 Qwen2.5-7B-Instruct__ft_9216 pass, B3 Qwen2.5-7B-Instruct__gen_7168 fail, B5 image pass, B5 classify pass

## All cases

| case | status | wall s | step s | host peak |
|---|---|---|---|---|
| D_classify_plain | ok | 17.5 | - | 2.38 GiB |
| D_classify_run | ok | 52.2 | - | 1.19 GiB |
| D_image_plain | ok | 5.5 | - | 0.79 GiB |
| D_image_run | ok | 15.1 | - | 0.40 GiB |
| Qwen2.5-3B-Instruct__U16 | ok | 98.7 | 0.45 | 5.89 GiB |
| Qwen2.5-3B-Instruct__ft_6144 | ok | 99.9 | 5.04 | 5.56 GiB |
| Qwen2.5-3B-Instruct__ft_8192 | ok | 106.0 | 5.12 | 7.34 GiB |
| Qwen2.5-3B-Instruct__gen_6144 | error | 43.6 | - | 5.94 GiB |
| Qwen2.5-3B-Instruct__plain_gen | ok | 40.4 | - | 4.63 GiB |
| Qwen2.5-3B-Instruct__plain_lora | ok | 105.0 | 4.91 | 4.69 GiB |
| Qwen2.5-7B-Instruct__ft_7168 | ok | 506.1 | 40.31 | 6.36 GiB |
| Qwen2.5-7B-Instruct__ft_9216 | ok | 501.6 | 40.29 | 8.37 GiB |
| Qwen2.5-7B-Instruct__gen_7168 | error | 95.2 | - | 6.92 GiB |
| tests | ok | 72.1 | - | - |

## Not ok

- `Qwen2.5-3B-Instruct__gen_6144` (error): BudgetExceeded: a 6144 MiB ceiling cannot hold the largest weight (593 MiB), the cache and intermediates (about 2429 MiB) and the 6138 MiB the process holds besides; raise the budget or lower max_new_tokens
- `Qwen2.5-7B-Instruct__gen_7168` (error): BudgetExceeded: a 7168 MiB ceiling cannot hold the largest weight (1039 MiB), the cache and intermediates (about 2808 MiB) and the 7085 MiB the process holds besides; raise the budget or lower max_new_tokens
