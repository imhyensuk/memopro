# memopro main@e29d0a2f65f8 on Colab CPU (E068)

run `20261029-065945`, cell `beta_cpu`

| check | criterion | result | detail |
|---|---|---|---|
| B1 | repository tests at the tag: no failures | pass | {'passed': 173, 'skipped': 23}  |
| B2 Qwen2.5-0.5B-Instruct | finetune completes at both ceilings, losses bit-identical | pass | ['ok', 'ok'] [] |
| B3 Qwen2.5-0.5B-Instruct__ft_2560 | process peak (VmHWM) <= ceiling | pass | ok: peak 1.98 GiB vs ceiling 2.50 GiB |
| B3 Qwen2.5-0.5B-Instruct__ft_3072 | process peak (VmHWM) <= ceiling | pass | ok: peak 1.98 GiB vs ceiling 3.00 GiB |
| B3 Qwen2.5-0.5B-Instruct__gen_2560 | process peak (VmHWM) <= ceiling | pass | ok: peak 1.51 GiB vs ceiling 2.50 GiB |
| B4 Qwen2.5-0.5B-Instruct | generate's text = the model loaded normally | pass | plain ok 0.24 s/token, memopro ok 0.41 s/token |
| R1 Qwen2.5-0.5B-Instruct | report: step s plain / memopro at both ceilings; first loss | n/a | 189.97 / ['207.40', '231.26']; 3.3363 / 3.3363 |
| B5 image | memopro run at half: plain result, peak <= ceiling + 32 MiB | pass | ok; peak 0.40 GiB vs 0.39 GiB; 3.2 -> 11.0 s |
| B5 classify | memopro run at half: plain result, peak <= ceiling + 32 MiB | pass | ok; peak 1.19 GiB vs 1.19 GiB; 14.7 -> 41.9 s |

Pre-registered: docs/research/0346 (E068).
CPU runtime; elapsed 59 min of 120.
Judged: B1 pass, B2 Qwen2.5-0.5B-Instruct pass, B3 Qwen2.5-0.5B-Instruct__ft_2560 pass, B3 Qwen2.5-0.5B-Instruct__ft_3072 pass, B3 Qwen2.5-0.5B-Instruct__gen_2560 pass, B4 Qwen2.5-0.5B-Instruct pass, B5 image pass, B5 classify pass

## All cases

| case | status | wall s | step s | host peak |
|---|---|---|---|---|
| D_classify_plain | ok | 15.3 | - | 2.39 GiB |
| D_classify_run | ok | 42.5 | - | 1.19 GiB |
| D_image_plain | ok | 3.5 | - | 0.79 GiB |
| D_image_run | ok | 11.3 | - | 0.40 GiB |
| Qwen2.5-0.5B-Instruct__ft_2560 | ok | 1076.6 | 207.40 | 1.98 GiB |
| Qwen2.5-0.5B-Instruct__ft_3072 | ok | 1176.7 | 231.26 | 1.98 GiB |
| Qwen2.5-0.5B-Instruct__gen_2560 | ok | 35.1 | - | 1.51 GiB |
| Qwen2.5-0.5B-Instruct__plain_gen | ok | 25.8 | - | 1.50 GiB |
| Qwen2.5-0.5B-Instruct__plain_lora | ok | 953.7 | 189.97 | 2.22 GiB |
| tests | ok | 127.9 | - | - |
