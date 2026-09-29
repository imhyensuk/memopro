# memopro Colab T4 re-measurement after 0099/0100 (E023)

run `20260929-074132`, cell `remeasure2`

| check | criterion | result | detail |
|---|---|---|---|
| R1 | regression checks all pass | pass | 7/7; failed: none |
| D9 | CUDA int4 load reports the bitsandbytes quality note | pass | chosen load.quant.int4, note in report: True |
| T1 | GPT-2 batch 16: even micro-batch, no retry, loss within 1e-5 of HF ckpt | pass | micro 8 of 16, retries 0; loss diff 1.7e-07, params 1.4e-09; 3491 tok/s (E021: 3632, micro 15) |
| T2 | GPT-2 at a 30% cap: finishes with no out-of-memory retry, loss within 1e-5 | pass | status ok, micro 2, retries 0; loss diff 5.2e-06; 3468 tok/s (E021: micro 3 -> OOM -> 2, 3558) |
| T3 | 0.5B batch 8: micro >= 2, no retry, loss within 1e-5 of fixed micro 1 | pass | micro 2, retries 0; loss diff 1.0e-06, params 6.0e-09; 960 vs 881 tok/s (1.09x; E021: 889 at micro 1) |
| T4 | 7B QLoRA batch 4: micro >= 2, no retry, speed >= 0.95 x standard, loss within 5e-3 | pass | micro 4, retries 0; 227 vs 227 tok/s (1.00x; E022: 194 vs 221 at micro 1); loss diff 1.3e-03 |
| O1 load | 7B default: still int8, report names quality='low' -> int4 (2.9x) | pass | chose load.quant.int8; 5.2 tok/s; hint: quality='low' would load int4 (bitsandbytes nf4) instead: Qwen2.5-7B decoded 2.9x faster o |
| O1 run | `memopro run` 7B: policy applied, report names `--quality low` | pass | APP loaded in 84.5s on cuda:0 | APP generated in 7.0s: GPUs run out of memory when the amount of data they need to process exceeds the capacity of their VRAM (v |

Pre-registered criteria: docs/research/0101. Loss diff = largest relative difference of the per-step losses; params = relative L2 difference of the per-parameter norms. Full details: cases/*.json, logs/, orchestrator.log.

## Training cases

| case | status | steps | micro | retries | tok/s | peak alloc | peak reserved | peak GPU used | at start: free / cached | final loss | plan |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B__cap0__memopro | ok | 30 | 2 | 0 | 960 | 12.47 GiB | 12.96 GiB | 13.10 GiB | 12.51 GiB / 0.11 GiB | 2.2221 | one sample needs 2.21 GiB of activations (x1.25 for the allocator); budget 13.20 GiB, model and optimizer state 1.84 GiB -> micro-batch 2 of 8 |
| Qwen2.5-0.5B__cap0__memopro_micro1 | ok | 30 | 1 | 0 | 881 | 9.95 GiB | 11.38 GiB | 11.52 GiB | 12.51 GiB / 0.11 GiB | 2.2221 | - |
| Qwen2.5-7B-Instruct__cap0__qlora_hf | ok | 20 | - | - | 227 | 11.96 GiB | 13.16 GiB | 13.30 GiB | - | 2.1807 | - |
| Qwen2.5-7B-Instruct__cap0__qlora_memopro | ok | 20 | 4 | 0 | 227 | 12.23 GiB | 13.90 GiB | 14.05 GiB | 4.80 GiB / 1.99 GiB | 2.1812 | MemTracker unavailable here (e.g. frozen parameters): measured one sample instead; one sample needs 949.39 MiB of activations (x1.25 for the allocator); budget  |
| gpt2__cap0__hf_ckpt | ok | 30 | - | - | 3044 | 6.34 GiB | 7.51 GiB | 7.64 GiB | - | 3.0860 | - |
| gpt2__cap0__memopro | ok | 30 | 8 | 0 | 3491 | 8.16 GiB | 8.94 GiB | 9.08 GiB | 13.94 GiB / 0.05 GiB | 3.0860 | one sample needs 703.77 MiB of activations (x1.25 for the allocator); budget 13.06 GiB, model and optimizer state 474.70 MiB -> micro-batch 8 of 16 |
| gpt2__cap30__memopro | ok | 30 | 2 | 0 | 3468 | 3.46 GiB | 3.61 GiB | 3.75 GiB | 13.94 GiB / 0.05 GiB | 3.0860 | one sample needs 703.77 MiB of activations (x1.25 for the allocator); budget 4.37 GiB, model and optimizer state 474.70 MiB -> micro-batch 2 of 16 |
