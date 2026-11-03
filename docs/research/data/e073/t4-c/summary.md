# E073b: model families on a T4

run `20261103-061354`, cell `families`

| family | class | text | gen_ceiling | losses | ft_ceiling | losses_default | repeat_same | steady | ratio | peak | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5 | full | True | True | True | True | True | None | 0.075 / 0.103 | 1.37 | 2.70 GiB |  |

Classes: {'full': 1, 'partial': 0, 'none': 0, 'WRONG': 0}. `WRONG` is a result that differs from the plain run.
steady = s/token after the first token, plain / memopro.
losses = strict runs where there are any; losses_default = default runs across ceilings; repeat_same = the first ceiling's default run against its repeat.
Pre-registered: docs/research/0376 (E073b). e073b-diag@17e2fbbbbeeb; elapsed 12 min.

