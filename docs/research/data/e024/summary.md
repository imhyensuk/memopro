# memopro Colab T4: 7B QLoRA after 0103 (E024)

run `20260929-095510`, cell `qlora`

| check | criterion | result | detail |
|---|---|---|---|
| Q1 | memopro finishes all steps with no out-of-memory retry | pass | status ok, steps 20, micro 2, retries 0 |
| Q2 | peak reserved memory <= memopro's own budget | pass | peak reserved 11.00 GiB, budget 11.96 GiB (E023: 13.90 > 13.75 GiB) |
| Q3 | speed >= 0.9 x the standard recipe, loss within 5e-3 | pass | 229 vs 235 tok/s (0.97x; E023 1.00x at micro 4); loss diff 1.2e-03 |

Pre-registered criteria: docs/research/0104.

## Cache at session start (P-b)

| moment | free | allocated | reserved | cached (reserved - allocated) |
|---|---|---|---|---|
| before train_session | 4.80 GiB | 7.64 GiB | 9.63 GiB | 1.99 GiB |
| after it measured | 4.81 GiB | 7.64 GiB | 9.63 GiB | 1.99 GiB |

- memopro: returned 4.00 MiB of torch's unused CUDA cache before measuring (E023 F2: cache left by loading was mostly not reused)
- plan: MemTracker unavailable here (e.g. frozen parameters): measured one sample instead; one sample needs 949.39 MiB of activations (x1.3 for the allocator); budget 11.96 GiB, model and optimizer state 7.63 GiB -> micro-batch 2 of 4

## Both cases

| case | status | micro | tok/s | peak alloc | peak reserved | peak GPU used | final loss |
|---|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct__cap0__qlora_hf | ok | - | 235 | 11.96 GiB | 13.16 GiB | 13.30 GiB | 2.1807 |
| Qwen2.5-7B-Instruct__cap0__qlora_memopro | ok | 2 | 229 | 10.25 GiB | 11.00 GiB | 11.15 GiB | 2.1807 |
