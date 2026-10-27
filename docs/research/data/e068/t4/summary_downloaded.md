# memopro 0.2.0b1 on Colab T4 (E068)

run `20261027-083215`, cell `beta_t4`

| check | criterion | result | detail |
|---|---|---|---|
| B1 | repository tests at the tag: no failures | fail | {'failed': 1, 'passed': 160, 'skipped': 23} ['FAILED tests/test_run.py::test_run_applies_the_loading_policy_only_where_the_script_did_not_choose'] |
| B2 Qwen2.5-3B-Instruct | finetune completes at both ceilings, losses bit-identical | pass | ['ok', 'ok'] [] |
| B3 Qwen2.5-3B-Instruct__ft_6144 | process peak (VmHWM) <= ceiling | pass | ok: peak 5.57 GiB vs ceiling 6.00 GiB |
| B3 Qwen2.5-3B-Instruct__ft_8192 | process peak (VmHWM) <= ceiling | pass | ok: peak 7.33 GiB vs ceiling 8.00 GiB |
| B3 Qwen2.5-3B-Instruct__gen_6144 | process peak (VmHWM) <= ceiling | fail | error: peak 1.11 GiB vs ceiling 6.00 GiB |
| B4 Qwen2.5-3B-Instruct | generate's text = the model loaded normally | fail | plain ok 0.08 s/token, memopro error - s/token |
| R1 Qwen2.5-3B-Instruct | report: step s plain / memopro at both ceilings; first loss | n/a | 5.12 / ['5.42', '5.45']; 2.6548 / 2.6548 |
| B2 Qwen2.5-7B-Instruct | finetune completes at both ceilings, losses bit-identical | pass | ['ok', 'ok'] [] |
| B3 Qwen2.5-7B-Instruct__ft_7168 | process peak (VmHWM) <= ceiling | pass | ok: peak 6.53 GiB vs ceiling 7.00 GiB |
| B3 Qwen2.5-7B-Instruct__ft_9216 | process peak (VmHWM) <= ceiling | pass | ok: peak 8.36 GiB vs ceiling 9.00 GiB |
| B3 Qwen2.5-7B-Instruct__gen_7168 | process peak (VmHWM) <= ceiling | fail | error: peak 1.11 GiB vs ceiling 7.00 GiB |
| R1 Qwen2.5-7B-Instruct | report: step s plain / memopro at both ceilings; first loss | n/a | - / ['44.35', '44.29']; - / 2.5411 |
| B5 image | memopro run at half: plain result, peak <= ceiling + 32 MiB | fail | ok; peak 0.68 GiB vs 0.39 GiB; 8.0 -> 17.2 s |
| B5 classify | memopro run at half: plain result, peak <= ceiling + 32 MiB | pass | ok; peak 1.19 GiB vs 1.19 GiB; 19.7 -> 66.6 s |
| R2 | report: Unsloth 16-bit 3B step s (memopro at the larger ceiling) | n/a | ok 0.46 / 5.45 |

Pre-registered: docs/research/0346 (E068).
T4 runtime; elapsed 43 min of 120.
Judged: B1 fail, B2 Qwen2.5-3B-Instruct pass, B3 Qwen2.5-3B-Instruct__ft_6144 pass, B3 Qwen2.5-3B-Instruct__ft_8192 pass, B3 Qwen2.5-3B-Instruct__gen_6144 fail, B4 Qwen2.5-3B-Instruct fail, B2 Qwen2.5-7B-Instruct pass, B3 Qwen2.5-7B-Instruct__ft_7168 pass, B3 Qwen2.5-7B-Instruct__ft_9216 pass, B3 Qwen2.5-7B-Instruct__gen_7168 fail, B5 image fail, B5 classify pass

## All cases

| case | status | wall s | step s | host peak |
|---|---|---|---|---|
| D_classify_plain | ok | 25.6 | - | 2.39 GiB |
| D_classify_run | ok | 80.7 | - | 1.19 GiB |
| D_image_plain | ok | 9.4 | - | 0.79 GiB |
| D_image_run | ok | 21.4 | - | 0.68 GiB |
| Qwen2.5-3B-Instruct__U16 | ok | 178.8 | 0.46 | 4.78 GiB |
| Qwen2.5-3B-Instruct__ft_6144 | ok | 99.3 | 5.42 | 5.57 GiB |
| Qwen2.5-3B-Instruct__ft_8192 | ok | 105.2 | 5.45 | 7.33 GiB |
| Qwen2.5-3B-Instruct__gen_6144 | error | 20.8 | - | 1.11 GiB |
| Qwen2.5-3B-Instruct__plain_gen | ok | 35.9 | - | 4.64 GiB |
| Qwen2.5-3B-Instruct__plain_lora | ok | 113.1 | 5.12 | 4.70 GiB |
| Qwen2.5-7B-Instruct__ft_7168 | ok | 590.6 | 44.35 | 6.53 GiB |
| Qwen2.5-7B-Instruct__ft_9216 | ok | 592.4 | 44.29 | 8.36 GiB |
| Qwen2.5-7B-Instruct__gen_7168 | error | 25.0 | - | 1.11 GiB |
| tests | failed | 70.4 | - | - |

## Not ok

- `Qwen2.5-3B-Instruct__gen_6144` (error): InvalidArgument: /content/drive/MyDrive/memopro_colab/hf_cache/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1/model-00001-of-00002.safetensors: not a safetensors file (header length 7809856579203706414 > file size 79)
- `Qwen2.5-7B-Instruct__gen_7168` (error): InvalidArgument: /content/drive/MyDrive/memopro_colab/hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28/model-00001-of-00004.safetensors: not a safetensors file (header length 7809856579203706414 > file size 79)
- `tests` (failed): ['FAILED tests/test_run.py::test_run_applies_the_loading_policy_only_where_the_script_did_not_choose']
