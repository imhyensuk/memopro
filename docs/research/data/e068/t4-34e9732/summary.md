# memopro main@34e9732e7224 on Colab T4 (E068)

run `20261030-032803`, cell `beta_t4`

| check | criterion | result | detail |
|---|---|---|---|
| B1 | repository tests at the tag: no failures | pass | {'passed': 174, 'skipped': 23}  |
| B2 Qwen2.5-3B-Instruct | finetune completes at both ceilings, losses bit-identical | pass | ['ok', 'ok'] [] |
| B3 Qwen2.5-3B-Instruct__ft_6144 | process peak (VmHWM) <= ceiling | pass | ok: peak 2.06 GiB vs ceiling 6.00 GiB |
| B3 Qwen2.5-3B-Instruct__ft_8192 | process peak (VmHWM) <= ceiling | pass | ok: peak 2.06 GiB vs ceiling 8.00 GiB |
| B3 Qwen2.5-3B-Instruct__gen_6144 | process peak (VmHWM) <= ceiling | pass | ok: peak 2.07 GiB vs ceiling 6.00 GiB |
| B4 Qwen2.5-3B-Instruct | generate's text = the model loaded normally | pass | plain ok 0.08 s/token, memopro ok 0.38 s/token |
| R1 Qwen2.5-3B-Instruct | report: step s plain / memopro at both ceilings; first loss | n/a | 5.21 / ['5.86', '5.93']; 2.6548 / 2.6548 |
| B2 Qwen2.5-7B-Instruct | finetune completes at both ceilings, losses bit-identical | pass | ['ok', 'ok'] [] |
| B3 Qwen2.5-7B-Instruct__ft_7168 | process peak (VmHWM) <= ceiling | pass | ok: peak 3.61 GiB vs ceiling 7.00 GiB |
| B3 Qwen2.5-7B-Instruct__ft_9216 | process peak (VmHWM) <= ceiling | pass | ok: peak 3.62 GiB vs ceiling 9.00 GiB |
| B3 Qwen2.5-7B-Instruct__gen_7168 | process peak (VmHWM) <= ceiling | fail | ok: peak 7.00 GiB vs ceiling 7.00 GiB |
| R1 Qwen2.5-7B-Instruct | report: step s plain / memopro at both ceilings; first loss | n/a | - / ['41.71', '42.23']; - / 2.5411 |
| B5 image | memopro run at half: plain result, peak <= ceiling + 32 MiB | pass | ok; peak 0.40 GiB vs 0.39 GiB; 5.4 -> 14.8 s |
| B5 classify | memopro run at half: plain result, peak <= ceiling + 32 MiB | pass | ok; peak 1.19 GiB vs 1.19 GiB; 16.7 -> 49.4 s |
| R2 | report: Unsloth 16-bit 3B step s (memopro at the larger ceiling) | n/a | ok 0.49 / 5.93 |
| R3 | report (0363): memopro.finetune on the Unsloth model = Unsloth's loop | n/a | ok; bits equal: False; step s 0.50 vs 0.49; first loss 2.6621 / 2.6621 |
| R4 | report (0363): memopro.generate on the Unsloth model = Unsloth's | n/a | ok; same text: True |

Pre-registered: docs/research/0346 (E068).
T4 runtime; elapsed 50 min of 120.
Judged: B1 pass, B2 Qwen2.5-3B-Instruct pass, B3 Qwen2.5-3B-Instruct__ft_6144 pass, B3 Qwen2.5-3B-Instruct__ft_8192 pass, B3 Qwen2.5-3B-Instruct__gen_6144 pass, B4 Qwen2.5-3B-Instruct pass, B2 Qwen2.5-7B-Instruct pass, B3 Qwen2.5-7B-Instruct__ft_7168 pass, B3 Qwen2.5-7B-Instruct__ft_9216 pass, B3 Qwen2.5-7B-Instruct__gen_7168 fail, B5 image pass, B5 classify pass

## All cases

| case | status | wall s | step s | host peak |
|---|---|---|---|---|
| D_classify_plain | ok | 17.4 | - | 2.38 GiB |
| D_classify_run | ok | 50.1 | - | 1.19 GiB |
| D_image_plain | ok | 5.9 | - | 0.79 GiB |
| D_image_run | ok | 15.2 | - | 0.40 GiB |
| Qwen2.5-3B-Instruct__U16 | ok | 91.7 | 0.49 | 3.96 GiB |
| Qwen2.5-3B-Instruct__U16_gen | ok | 50.8 | - | 4.08 GiB |
| Qwen2.5-3B-Instruct__U16_memopro | ok | 73.0 | 0.50 | 4.01 GiB |
| Qwen2.5-3B-Instruct__ft_6144 | ok | 98.0 | 5.86 | 2.06 GiB |
| Qwen2.5-3B-Instruct__ft_8192 | ok | 96.1 | 5.93 | 2.06 GiB |
| Qwen2.5-3B-Instruct__gen_6144 | ok | 33.6 | - | 2.07 GiB |
| Qwen2.5-3B-Instruct__plain_gen | ok | 34.3 | - | 4.64 GiB |
| Qwen2.5-3B-Instruct__plain_lora | ok | 101.7 | 5.21 | 4.70 GiB |
| Qwen2.5-7B-Instruct__ft_7168 | ok | 507.6 | 41.71 | 3.61 GiB |
| Qwen2.5-7B-Instruct__ft_9216 | ok | 506.2 | 42.23 | 3.62 GiB |
| Qwen2.5-7B-Instruct__gen_7168 | ok | 487.4 | - | 7.00 GiB |
| tests | ok | 75.4 | - | - |
