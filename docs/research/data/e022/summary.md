# memopro Colab T4 re-measurement after 0094 (E022)

run `20260929-060206`, cell `remeasure`

| check | criterion | result | detail |
|---|---|---|---|
| R1 | regression checks all pass | pass | 7/7; failed: none |
| D1 | QLoRA: train_session plans (no 'skipped'), finishes | pass | plan: ['applied', 'applied']; session {'micro': 1, 'active': ['micro-batch 1'], 'retries': 0}; 194 tok/s vs standard 221; layers ['Linear', 'Linear4bit'] |
| D2 Qwen2.5-1.5B-Instruct | memopro picks fp16; TTFT(2048) <= 1.2 x HF fp16 | pass | chose load.half; TTFT 0.70 s vs 0.67 s; decode 24.5 vs 22.9 tok/s; ppl 12.619 vs 12.619 |
| D2 Qwen2.5-3B-Instruct | memopro picks fp16; TTFT(2048) <= 1.2 x HF fp16 | pass | chose load.half; TTFT 1.30 s vs 1.21 s; decode 18.8 vs 19.6 tok/s; ppl 11.169 vs 11.169 |
| D3 low | 7B quality='low' picks int4; decode >= 0.9 x HF bnb4 | pass | chose load.quant.int4; 15.8 vs 14.0 tok/s; ppl 9.051 vs 9.047 |
| D3 default | 7B default quality still picks int8 | pass | chose load.quant.int8; 5.5 tok/s; ppl 8.473 |
| D4 7B | `memopro run` applies its loading policy; no γ entries | pass | APP loaded in 80.9s on cuda:0 | APP generated in 6.9s: GPUs run out of memory when the amount of data they need to process exceeds the capacity of their VRAM (video random-access memory). | APP peak_a |
| D4 1.5B | `memopro run` stands aside for a model that fits as stored | pass | APP loaded in 13.9s on cuda:0 | APP generated in 2.8s: GPUs run out of memory due to their limited on-chip VRAM capacity compared to the large amounts of RAM used in CPUs for managing system-wide data |

Pre-registered criteria: docs/research/0096. Full details: cases/*.json, logs/, orchestrator.log (the cell's own progress, e.g. why a model was not staged).

## Inference cases

| case | status | memopro chose | load s | TTFT 512 | TTFT 2048 | decode tok/s | ppl | peak GPU |
|---|---|---|---|---|---|---|---|---|
| infer__Qwen2.5-1.5B-Instruct__hf_fp16_auto | ok | - | 11.4 | 0.17 | 0.67 | 22.9 | 12.619 | 5.11 GiB |
| infer__Qwen2.5-1.5B-Instruct__memopro | ok | half | 3.4 | 0.15 | 0.70 | 24.5 | 12.619 | 5.12 GiB |
| infer__Qwen2.5-3B-Instruct__hf_fp16_auto | ok | - | 23.7 | 0.25 | 1.21 | 19.6 | 11.169 | 8.12 GiB |
| infer__Qwen2.5-3B-Instruct__memopro | ok | half | 24.2 | 0.24 | 1.30 | 18.8 | 11.169 | 8.13 GiB |
| infer__Qwen2.5-7B-Instruct__hf_bnb4 | ok | - | 62.4 | 0.62 | 2.27 | 14.0 | 9.047 | 7.91 GiB |
| infer__Qwen2.5-7B-Instruct__hf_bnb8 | ok | - | 85.7 | 0.65 | 3.12 | 3.9 | 8.489 | 11.19 GiB |
| infer__Qwen2.5-7B-Instruct__memopro | ok | quant.int8 | 76.1 | 0.50 | 2.45 | 5.5 | 8.473 | 11.19 GiB |
| infer__Qwen2.5-7B-Instruct__memopro_low | ok | quant.int4 | 66.4 | 0.60 | 2.69 | 15.8 | 9.051 | 8.21 GiB |
