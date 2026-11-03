# 0377: where streamed Qwen3.5 differs

run `20261103-050203`, cell `diag`

| model | prompt | status | first_layer | logits_max_abs | argmax | margin | plain_dtypes | streamed_dtypes |
|---|---|---|---|---|---|---|---|---|
| Qwen3.5-4B | 1 | ok | None | 0.0000 | [1919, 1919] | 3.1250 | [] | [] |
| Qwen3.5-4B | 2 | ok | None | 0.0000 | [760, 760] | 0.6250 | [] | [] |
| Qwen3-1.7B | 1 | ok | None | 0.0000 | [151667, 151667] | 12.2500 | [] | [] |
| Qwen3-1.7B | 2 | ok | None | 0.0000 | [151667, 151667] | 11.6250 | [] | [] |

first_layer = the first decoder layer whose output differs (None: none does); margin = plain top-1 minus top-2 logit; *_dtypes = parameters not in the model's dtype. Per-layer differences are in each case's JSON.
e073b-diag@d6585964c2e8; elapsed 6 min.
