# memopro Colab T4 suite (E042)

run `20261006-052622`, cell `suite`

| check | criterion | result | detail |
|---|---|---|---|
| G-Qwen2.5-3B-Instruct | streamed generation text = the model loaded normally (both budgets) | pass | s/token plain 0.10, memopro ['0.47', '0.45'] |
| L-Qwen2.5-3B-Instruct | 16-bit LoRA completes; losses bit-identical across budgets | pass | plain ok [2.6548, 3.0293, 3.1205, 2.915, 2.9474]; memopro [2.6548, 3.0282, 3.1224, 2.9169, 2.9425], step s ['9.2', '8.1'] |
| G-Qwen2.5-7B-Instruct | streamed generation text = the model loaded normally (both budgets) | pass | s/token plain 0.08, memopro ['1.47', '1.51'] |
| L-Qwen2.5-7B-Instruct | 16-bit LoRA completes; losses bit-identical across budgets | pass | plain oom []; memopro [2.5411, 3.0155, 3.0426, 2.8188, 2.9649], step s ['54.1', '54.5'] |
| V-resnet-152 | streamed inference output = the model loaded normally (both budgets) | pass | plain ok 0.13 s; memopro [('ok', '0.14'), ('ok', '0.14')] |
| V-dinov2-giant | streamed inference output = the model loaded normally (both budgets) | fail | plain ok 1.60 s; memopro [('error', '-'), ('error', '-')] |
| VL-dinov2-giant | vision LoRA completes; losses bit-identical across budgets | fail | plain ok [2.4394, 2.194, 1.8742, 1.4957, 1.333]; memopro [] |
| S-Qwen2.5-3B-Instruct | same output as plain; time <= the pre-registered limit (0202) | fail | plain 0.10, no cache 0.45, cache 0.46 (limit 0.20); copies {'copies': 434, 'copy_bytes': 6171877376, 'hits': 27406, 'prefetched': 0, 'gpu_budget': 13179456717, 'gpu_cache_bytes': 6171877376} |
| S-Qwen2.5-7B-Instruct | same output as plain; time <= the pre-registered limit (0202) | pass | plain 0.08, no cache 1.51, cache 1.55 (limit 2.00); copies {'copies': 906, 'copy_bytes': 145403030528, 'hits': 20790, 'prefetched': 567, 'gpu_budget': 13179456717, 'gpu_cache_bytes': 13165014016} |
| S-dinov2-giant | same output as plain; time <= the pre-registered limit (0202) | fail | plain 1.60, no cache -, cache - (limit 3.21); copies - |
| D-image | same result; process peak <= half of the plain peak (+64 MiB); no overruns | pass | peak 0.71 GiB vs limit 0.77 GiB (plain 1.54 GiB); 10.5 -> 33.6 s; overruns 0 |
| D-dataframe | same result; process peak <= half of the plain peak (+64 MiB); no overruns | fail | peak - vs limit 0.83 GiB (plain 1.66 GiB); 16.0 -> - s; overruns 0 |
| D-classify | same result; process peak <= half of the plain peak (+64 MiB); no overruns | pass | peak 1.13 GiB vs limit 1.19 GiB (plain 2.38 GiB); 16.8 -> 60.7 s; overruns 0 |
| D-simulate | same result; process peak <= half of the plain peak (+64 MiB); no overruns | fail | peak - vs limit 0.00 GiB (plain 2.16 GiB); 45.9 -> - s; overruns None |

Pre-registered criteria: docs/research/0197.

## All cases

| case | status | load s | host peak RSS | GPU peak used |
|---|---|---|---|---|
| data__classify__plain | ok | - | - | 0.00 GiB |
| data__classify__preload | ok | - | - | 0.00 GiB |
| data__dataframe__plain | ok | - | - | 0.00 GiB |
| data__dataframe__preload | timeout | - | - | 0.00 GiB |
| data__image__plain | ok | - | - | 0.00 GiB |
| data__image__preload | ok | - | - | 0.00 GiB |
| data__probe | ok | - | - | - |
| data__simulate__plain | ok | - | - | 0.00 GiB |
| lm_gen__Qwen2.5-3B-Instruct__memopro_2048 | ok | 0.7 | 3.47 GiB | 6.15 GiB |
| lm_gen__Qwen2.5-3B-Instruct__memopro_3072 | ok | 0.7 | 4.36 GiB | 6.15 GiB |
| lm_gen__Qwen2.5-3B-Instruct__memopro_3072__cache | ok | 0.6 | 4.35 GiB | 6.15 GiB |
| lm_gen__Qwen2.5-3B-Instruct__plain | ok | 22.4 | 4.63 GiB | 6.15 GiB |
| lm_gen__Qwen2.5-7B-Instruct__memopro_4096 | ok | 0.6 | 5.32 GiB | 13.73 GiB |
| lm_gen__Qwen2.5-7B-Instruct__memopro_6144 | ok | 0.6 | 7.31 GiB | 14.01 GiB |
| lm_gen__Qwen2.5-7B-Instruct__memopro_6144__cache | ok | 1.0 | 7.26 GiB | 13.73 GiB |
| lm_gen__Qwen2.5-7B-Instruct__plain | ok | 57.8 | 4.92 GiB | 14.49 GiB |
| lm_lora__Qwen2.5-3B-Instruct__memopro_2048 | ok | 3.8 | 3.76 GiB | 6.41 GiB |
| lm_lora__Qwen2.5-3B-Instruct__memopro_3072 | ok | 4.8 | 4.74 GiB | 6.41 GiB |
| lm_lora__Qwen2.5-3B-Instruct__plain | ok | 7.0 | 3.78 GiB | 6.43 GiB |
| lm_lora__Qwen2.5-7B-Instruct__memopro_4096 | ok | 9.0 | 5.74 GiB | 14.31 GiB |
| lm_lora__Qwen2.5-7B-Instruct__memopro_6144 | ok | 8.9 | 7.67 GiB | 14.31 GiB |
| lm_lora__Qwen2.5-7B-Instruct__plain | oom | 58.0 | 4.99 GiB | 14.56 GiB |
| vis_infer__dinov2-giant__memopro_1100 | error | - | 0.89 GiB | 0.10 GiB |
| vis_infer__dinov2-giant__memopro_2200 | error | - | 0.89 GiB | 0.10 GiB |
| vis_infer__dinov2-giant__memopro_2200__cache | error | - | 0.87 GiB | 0.10 GiB |
| vis_infer__dinov2-giant__plain | ok | 19.7 | 4.55 GiB | 4.74 GiB |
| vis_infer__resnet-152__memopro_120 | ok | 0.4 | 1.42 GiB | 0.58 GiB |
| vis_infer__resnet-152__memopro_60 | ok | 0.4 | 1.36 GiB | 0.58 GiB |
| vis_infer__resnet-152__plain | ok | 0.7 | 1.30 GiB | 0.54 GiB |
| vis_lora__dinov2-giant__memopro_1100 | error | - | 0.94 GiB | 0.10 GiB |
| vis_lora__dinov2-giant__memopro_2200 | error | - | 0.94 GiB | 0.10 GiB |
| vis_lora__dinov2-giant__plain | ok | 14.1 | 4.84 GiB | 12.88 GiB |

## Errors

- `data__dataframe__preload` (timeout): timeout
- `lm_lora__Qwen2.5-7B-Instruct__plain` (oom): OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 14.56 GiB of which 7.81 MiB is free. Including non-PyTorch memory, this process has 14.55 GiB memory in use. Of the allocated memory 14.29 GiB is allocated by PyTorch, and 119.36 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=
- `vis_infer__dinov2-giant__memopro_1100` (error): MemoryError: 
- `vis_infer__dinov2-giant__memopro_2200` (error): MemoryError: 
- `vis_infer__dinov2-giant__memopro_2200__cache` (error): MemoryError: 
- `vis_lora__dinov2-giant__memopro_1100` (error): MemoryError: 
- `vis_lora__dinov2-giant__memopro_2200` (error): MemoryError: 
