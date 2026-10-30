# memopro main@edfb27494d8b on Colab T4 (E068)

run `20261030-224546`, cell `beta_t4`

| check | criterion | result | detail |
|---|---|---|---|
| B1 | repository tests at the tag: no failures | pass | {'passed': 177, 'skipped': 23}  |
| B2 Qwen2.5-3B-Instruct | finetune completes at both ceilings, losses bit-identical | pass | ['ok', 'ok'] [] |
| B3 Qwen2.5-3B-Instruct__ft_6144 | process peak (VmHWM) <= ceiling | pass | ok: peak 2.01 GiB vs ceiling 6.00 GiB |
| B3 Qwen2.5-3B-Instruct__ft_8192 | process peak (VmHWM) <= ceiling | pass | ok: peak 2.01 GiB vs ceiling 8.00 GiB |
| B3 Qwen2.5-3B-Instruct__gen_6144 | process peak (VmHWM) <= ceiling | pass | ok: peak 1.97 GiB vs ceiling 6.00 GiB |
| B4 Qwen2.5-3B-Instruct | generate's text = the model loaded normally | pass | after the first token plain ok 0.06 s/token, memopro ok 0.07; first token (memopro opens the folder each call) ['11.26', '9.18'] s |
| R1 Qwen2.5-3B-Instruct | report: step s plain / memopro at both ceilings; first loss | n/a | 5.18 / ['5.48', '5.74']; 2.6548 / 2.6548 |
| B2 Qwen2.5-7B-Instruct | finetune completes at both ceilings, losses bit-identical | pass | ['ok', 'ok'] [] |
| B3 Qwen2.5-7B-Instruct__ft_7168 | process peak (VmHWM) <= ceiling | pass | ok: peak 3.58 GiB vs ceiling 7.00 GiB |
| B3 Qwen2.5-7B-Instruct__ft_9216 | process peak (VmHWM) <= ceiling | pass | ok: peak 3.58 GiB vs ceiling 9.00 GiB |
| B3 Qwen2.5-7B-Instruct__gen_7168 | process peak (VmHWM) <= ceiling | pass | ok: peak 2.32 GiB vs ceiling 7.00 GiB |
| B4 Qwen2.5-7B-Instruct | generate's text = the model loaded normally | pass | after the first token plain ok 0.06 s/token, memopro ok 0.07; first token (memopro opens the folder each call) ['56.35', '53.45'] s |
| R1 Qwen2.5-7B-Instruct | report: step s plain / memopro at both ceilings; first loss | n/a | - / ['41.72', '42.16']; - / 2.5411 |
| B5 image | memopro run at half: plain result, peak <= ceiling + 32 MiB | pass | ok; peak 0.40 GiB vs 0.39 GiB; 5.3 -> 14.6 s |
| B5 classify | memopro run at half: plain result, peak <= ceiling + 32 MiB | pass | ok; peak 1.19 GiB vs 1.19 GiB; 16.9 -> 48.5 s |
| R2 | report: Unsloth 16-bit 3B step s (memopro at the larger ceiling) | n/a | ok 0.49 / 5.74 |
| R3 | report (0363): memopro.finetune on the Unsloth model = Unsloth's loop | n/a | ok; bits equal: False; step s 0.50 vs 0.49; first loss 2.6621 / 2.6621 |
| R4 | report (0363): memopro.generate on the Unsloth model = Unsloth's | n/a | ok; same text: True |

Pre-registered: docs/research/0346 (E068).
T4 runtime; elapsed 49 min of 120.
Judged: B1 pass, B2 Qwen2.5-3B-Instruct pass, B3 Qwen2.5-3B-Instruct__ft_6144 pass, B3 Qwen2.5-3B-Instruct__ft_8192 pass, B3 Qwen2.5-3B-Instruct__gen_6144 pass, B4 Qwen2.5-3B-Instruct pass, B2 Qwen2.5-7B-Instruct pass, B3 Qwen2.5-7B-Instruct__ft_7168 pass, B3 Qwen2.5-7B-Instruct__ft_9216 pass, B3 Qwen2.5-7B-Instruct__gen_7168 pass, B4 Qwen2.5-7B-Instruct pass, B5 image pass, B5 classify pass

## All cases

| case | status | wall s | step s | host peak |
|---|---|---|---|---|
| D_classify_plain | ok | 17.6 | - | 2.39 GiB |
| D_classify_run | ok | 49.2 | - | 1.19 GiB |
| D_image_plain | ok | 5.6 | - | 0.79 GiB |
| D_image_run | ok | 14.9 | - | 0.40 GiB |
| Qwen2.5-3B-Instruct__U16 | ok | 87.2 | 0.49 | 3.96 GiB |
| Qwen2.5-3B-Instruct__U16_gen | ok | 49.9 | - | 4.95 GiB |
| Qwen2.5-3B-Instruct__U16_memopro | ok | 72.3 | 0.50 | 4.39 GiB |
| Qwen2.5-3B-Instruct__ft_6144 | ok | 92.2 | 5.48 | 2.01 GiB |
| Qwen2.5-3B-Instruct__ft_8192 | ok | 95.6 | 5.74 | 2.01 GiB |
| Qwen2.5-3B-Instruct__gen_6144 | ok | 31.9 | - | 1.97 GiB |
| Qwen2.5-3B-Instruct__plain_gen | ok | 32.5 | - | 4.54 GiB |
| Qwen2.5-3B-Instruct__plain_lora | ok | 101.9 | 5.18 | 4.60 GiB |
| Qwen2.5-7B-Instruct__ft_7168 | ok | 506.6 | 41.72 | 3.58 GiB |
| Qwen2.5-7B-Instruct__ft_9216 | ok | 503.9 | 42.16 | 3.58 GiB |
| Qwen2.5-7B-Instruct__gen_7168 | ok | 123.2 | - | 2.32 GiB |
| Qwen2.5-7B-Instruct__plain_gen | ok | 75.0 | - | 3.58 GiB |
| tests | ok | 71.4 | - | - |
