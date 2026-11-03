# E073b: model families on a T4

run `20261103-022115`, cell `families`

| family | class | text | gen_ceiling | losses | ft_ceiling | losses_default | repeat_same | steady | ratio | peak | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Llama | full | True | True | True | True | True | None | 0.030 / 0.038 | 1.26 | 1.89 GiB |  |
| Gemma 3 | full | True | True | True | True | True | None | 0.060 / 0.079 | 1.32 | 2.07 GiB |  |
| Phi | full | True | True | True | True | False | False | 0.043 / 0.045 | 1.04 | 1.79 GiB |  |
| Qwen3.5 | WRONG | False | True | True | True | True | None | 0.077 / 0.106 | 1.37 | 2.69 GiB |  |

Classes: {'full': 3, 'partial': 0, 'none': 0, 'WRONG': 1}. `WRONG` is a result that differs from the plain run.
steady = s/token after the first token, plain / memopro.
losses = strict runs where there are any; losses_default = default runs across ceilings; repeat_same = the first ceiling's default run against its repeat.
Pre-registered: docs/research/0376 (E073b). e073-families@0df206899b57; elapsed 24 min.

