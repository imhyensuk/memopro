# memopro Colab T4 suite (E042)

run `20261005-134520`, cell `suite`

| check | criterion | result | detail |
|---|---|---|---|
| G-Qwen2.5-3B-Instruct | streamed generation text = the model loaded normally (both budgets) | pass | s/token plain 0.08, memopro ['6.25', '4.95'] |
| L-Qwen2.5-3B-Instruct | 16-bit LoRA completes; losses bit-identical across budgets | fail | plain error []; memopro [], step s ['0.0', '0.0'] |
| G-Qwen2.5-7B-Instruct | streamed generation text = the model loaded normally (both budgets) | pass | s/token plain 0.10, memopro ['47.49', '40.92'] |
| L-Qwen2.5-7B-Instruct | 16-bit LoRA completes; losses bit-identical across budgets | fail | plain error []; memopro [], step s ['0.0', '0.0'] |
| V-resnet-152 | streamed inference output = the model loaded normally (both budgets) | pass | plain ok 0.16 s; memopro [('ok', '0.55'), ('ok', '0.35')] |
| V-dinov2-giant | streamed inference output = the model loaded normally (both budgets) | pass | plain ok 1.66 s; memopro [('ok', '8.61'), ('ok', '4.09')] |
| VL-dinov2-giant | vision LoRA completes; losses bit-identical across budgets | fail | plain error []; memopro [] |
| D-image | same result; process peak <= half of the plain peak (+64 MiB); no overruns | fail | peak 5.19 GiB vs limit 2.59 GiB (plain 5.19 GiB); 11.0 -> 11.1 s; overruns 0 |
| D-dataframe | same result; process peak <= half of the plain peak (+64 MiB); no overruns | fail | peak 5.19 GiB vs limit 2.59 GiB (plain 5.19 GiB); 14.2 -> 17.1 s; overruns 0 |
| D-classify | same result; process peak <= half of the plain peak (+64 MiB); no overruns | fail | peak 5.19 GiB vs limit 2.59 GiB (plain 5.19 GiB); 16.3 -> 17.0 s; overruns 0 |
| D-simulate | same result; process peak <= half of the plain peak (+64 MiB); no overruns | fail | peak 5.19 GiB vs limit 2.59 GiB (plain 5.19 GiB); 46.3 -> 56.7 s; overruns 0 |

Pre-registered criteria: docs/research/0197.

## All cases

| case | status | load s | host peak RSS | GPU peak used |
|---|---|---|---|---|
| data__classify__plain | ok | - | - | 0.00 GiB |
| data__classify__preload | ok | - | - | 0.00 GiB |
| data__dataframe__plain | ok | - | - | 0.00 GiB |
| data__dataframe__preload | ok | - | - | 0.00 GiB |
| data__image__plain | ok | - | - | 0.00 GiB |
| data__image__preload | ok | - | - | 0.00 GiB |
| data__probe | ok | - | - | - |
| data__simulate__plain | ok | - | - | 0.00 GiB |
| data__simulate__preload | ok | - | - | 0.00 GiB |
| lm_gen__Qwen2.5-3B-Instruct__memopro_2048 | ok | 0.7 | 3.74 GiB | 1.30 GiB |
| lm_gen__Qwen2.5-3B-Instruct__memopro_3072 | ok | 0.7 | 4.38 GiB | 1.30 GiB |
| lm_gen__Qwen2.5-3B-Instruct__plain | ok | 25.7 | 4.63 GiB | 6.15 GiB |
| lm_gen__Qwen2.5-7B-Instruct__memopro_4096 | ok | 0.9 | 5.39 GiB | 2.17 GiB |
| lm_gen__Qwen2.5-7B-Instruct__memopro_6144 | ok | 0.8 | 7.33 GiB | 2.17 GiB |
| lm_gen__Qwen2.5-7B-Instruct__plain | ok | 62.0 | 5.19 GiB | 14.49 GiB |
| lm_lora__Qwen2.5-3B-Instruct__memopro_2048 | error | 3.9 | 3.74 GiB | 0.12 GiB |
| lm_lora__Qwen2.5-3B-Instruct__memopro_3072 | error | 3.5 | 3.74 GiB | 0.12 GiB |
| lm_lora__Qwen2.5-3B-Instruct__plain | error | 9.0 | 4.68 GiB | 6.11 GiB |
| lm_lora__Qwen2.5-7B-Instruct__memopro_4096 | error | 9.1 | 5.19 GiB | 0.12 GiB |
| lm_lora__Qwen2.5-7B-Instruct__memopro_6144 | error | 8.6 | 5.19 GiB | 0.12 GiB |
| lm_lora__Qwen2.5-7B-Instruct__plain | error | 56.5 | 5.19 GiB | 14.47 GiB |
| vis_infer__dinov2-giant__memopro_1100 | ok | 1.0 | 5.19 GiB | 0.43 GiB |
| vis_infer__dinov2-giant__memopro_2200 | ok | 0.7 | 5.19 GiB | 0.43 GiB |
| vis_infer__dinov2-giant__plain | ok | 16.2 | 5.19 GiB | 4.79 GiB |
| vis_infer__resnet-152__memopro_120 | ok | 0.4 | 5.19 GiB | 0.31 GiB |
| vis_infer__resnet-152__memopro_60 | ok | 0.4 | 5.19 GiB | 0.32 GiB |
| vis_infer__resnet-152__plain | ok | 0.7 | 5.19 GiB | 0.55 GiB |
| vis_lora__dinov2-giant__memopro_1100 | error | 2.7 | 5.19 GiB | 0.10 GiB |
| vis_lora__dinov2-giant__memopro_2200 | error | 2.4 | 5.19 GiB | 0.10 GiB |
| vis_lora__dinov2-giant__plain | error | 4.8 | 5.19 GiB | 4.52 GiB |
