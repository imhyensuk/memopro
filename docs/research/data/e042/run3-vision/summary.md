# memopro Colab T4 suite (E042)

run `20261007-021718`, cell `suite`

| check | criterion | result | detail |
|---|---|---|---|
| G-Qwen2.5-3B-Instruct | plain does not fit the T4; streamed completes, same text at both budgets | n/a | not run yet (its notebook part has not been run) |
| L-Qwen2.5-3B-Instruct | 16-bit LoRA completes; losses bit-identical across budgets | n/a | not run yet (its notebook part has not been run) |
| G-Qwen2.5-7B-Instruct | plain does not fit the T4; streamed completes, same text at both budgets | n/a | not run yet (its notebook part has not been run) |
| L-Qwen2.5-7B-Instruct | 16-bit LoRA completes; losses bit-identical across budgets | n/a | not run yet (its notebook part has not been run) |
| V-resnet-152 | streamed inference output = the model loaded normally (both budgets) | pass | plain ok 0.19 s; memopro [('ok', '0.17'), ('ok', '0.14')] |
| V-dinov2-giant | streamed inference output = the model loaded normally (both budgets) | fail | plain ok 1.53 s; memopro [('error', '-'), ('error', '-')] |
| VL-dinov2-giant | vision LoRA completes; losses bit-identical across budgets | fail | plain ok [2.4394, 2.194, 1.8742, 1.4957, 1.333]; memopro [] |
| S-Qwen2.5-3B-Instruct | same output as plain; time <= the pre-registered limit (0202) | n/a | not run yet (its notebook part has not been run) |
| S-Qwen2.5-7B-Instruct | same output as plain; time <= the pre-registered limit (0202) | n/a | not run yet (its notebook part has not been run) |
| S-dinov2-giant | same output as plain; time <= the pre-registered limit (0202) | n/a | not run yet (its notebook part has not been run) |
| D-image | unchanged program at half memory under memopro-preload | n/a | not run yet (its notebook part has not been run) |
| D-dataframe | unchanged program at half memory under memopro-preload | n/a | not run yet (its notebook part has not been run) |
| D-classify | unchanged program at half memory under memopro-preload | n/a | not run yet (its notebook part has not been run) |
| D-simulate | unchanged program at half memory under memopro-preload | n/a | not run yet (its notebook part has not been run) |

Pre-registered criteria: docs/research/0197.

## All cases

| case | status | load s | host peak RSS | GPU peak used |
|---|---|---|---|---|
| vis_infer__dinov2-giant__memopro_1100 | error | - | 0.89 GiB | 0.10 GiB |
| vis_infer__dinov2-giant__memopro_2200 | error | - | 0.90 GiB | 0.10 GiB |
| vis_infer__dinov2-giant__plain | ok | 20.1 | 4.77 GiB | 4.74 GiB |
| vis_infer__resnet-152__memopro_120 | ok | 0.4 | 1.43 GiB | 0.58 GiB |
| vis_infer__resnet-152__memopro_60 | ok | 0.6 | 1.37 GiB | 0.58 GiB |
| vis_infer__resnet-152__plain | ok | 0.8 | 1.32 GiB | 0.55 GiB |
| vis_lora__dinov2-giant__memopro_1100 | error | - | 0.97 GiB | 0.10 GiB |
| vis_lora__dinov2-giant__memopro_2200 | error | - | 0.99 GiB | 0.10 GiB |
| vis_lora__dinov2-giant__plain | ok | 18.9 | 5.18 GiB | 12.88 GiB |

## Errors

- `vis_infer__dinov2-giant__memopro_1100` (error): InvalidArgument: no weights in the files for ['encoder.layer.0.mlp.gate_proj.weight', 'encoder.layer.0.mlp.gate_proj.bias', 'encoder.layer.0.mlp.up_proj.weight', 'encoder.layer.0.mlp.up_proj.bias', 'encoder.layer.1.mlp.gate_proj.weight']
- `vis_infer__dinov2-giant__memopro_2200` (error): InvalidArgument: no weights in the files for ['encoder.layer.0.mlp.gate_proj.weight', 'encoder.layer.0.mlp.gate_proj.bias', 'encoder.layer.0.mlp.up_proj.weight', 'encoder.layer.0.mlp.up_proj.bias', 'encoder.layer.1.mlp.gate_proj.weight']
- `vis_lora__dinov2-giant__memopro_1100` (error): InvalidArgument: no weights in the files for ['encoder.layer.0.mlp.gate_proj.weight', 'encoder.layer.0.mlp.gate_proj.bias', 'encoder.layer.0.mlp.up_proj.weight', 'encoder.layer.0.mlp.up_proj.bias', 'encoder.layer.1.mlp.gate_proj.weight']
- `vis_lora__dinov2-giant__memopro_2200` (error): InvalidArgument: no weights in the files for ['encoder.layer.0.mlp.gate_proj.weight', 'encoder.layer.0.mlp.gate_proj.bias', 'encoder.layer.0.mlp.up_proj.weight', 'encoder.layer.0.mlp.up_proj.bias', 'encoder.layer.1.mlp.gate_proj.weight']
