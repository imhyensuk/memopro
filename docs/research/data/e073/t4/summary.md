# E073: model families on a T4

run `20261102-091733`, cell `families`

| family | class | text | gen_ceiling | losses | ft_ceiling | steady | ratio | peak | note |
|---|---|---|---|---|---|---|---|---|---|
| Llama | none | None | None | None | None | - / - | - | 0.00 GiB | GatedRepoError: 401 Client Error. (Request ID: Root=1-6ae85532-1ac427934a7f9ce401dfec63;66a20fd5-0640-42dc-a281-ffb5a12e3f13)

Cannot access gated repo for url https://huggingf |
| Gemma 3 | none | None | None | None | None | - / - | - | 0.00 GiB | GatedRepoError: 401 Client Error. (Request ID: Root=1-6ae85533-42c5ef29478777046f4503fd;8d97e440-b64f-48da-ab9c-d32c0c5c7c2f)

Cannot access gated repo for url https://huggingf |
| Qwen3 | full | True | True | True | True | 0.059 / 0.083 | 1.40 | 2.05 GiB |  |
| DeepSeek (Qwen2 architecture) | full | True | True | True | True | 0.044 / 0.068 | 1.53 | 2.04 GiB |  |
| Granite | full | True | True | True | True | 0.073 / 0.091 | 1.24 | 1.70 GiB |  |
| Phi | WRONG | True | True | False | True | 0.038 / 0.060 | 1.60 | 1.79 GiB |  |
| SmolLM3 | full | True | True | True | True | 0.055 / 0.079 | 1.44 | 1.87 GiB |  |
| Phi-3 | full | True | True | True | True | 0.060 / 0.060 | 1.01 | 1.61 GiB |  |
| Qwen3.5 | none | None | None | None | None | 0.081 / - | - | 2.68 GiB |  |
| Mistral | full | True | True | True | True | 0.065 / 0.083 | 1.28 | 2.79 GiB |  |

Classes: {'full': 6, 'partial': 0, 'none': 3, 'WRONG': 1}. `WRONG` is a result that differs from the plain run.
steady = s/token after the first token, plain / memopro.
Pre-registered: docs/research/0375 (E073). main@c00eb589638c; elapsed 49 min.

- `Qwen3.5-4B__ft_6144` (error): InvalidArgument: model.layers.0.linear_attn.A_log: dtype torch.bfloat16 in the model, torch.float32 in the file (streaming keeps the stored dtype)
- `Qwen3.5-4B__ft_8192` (error): InvalidArgument: model.layers.0.linear_attn.A_log: dtype torch.bfloat16 in the model, torch.float32 in the file (streaming keeps the stored dtype)
- `Qwen3.5-4B__gen` (error): InvalidArgument: model.layers.0.linear_attn.A_log: dtype torch.bfloat16 in the model, torch.float32 in the file (streaming keeps the stored dtype)
