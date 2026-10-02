# E026c summary (0120)

E026c checks: **pass** (with E026's passed checks: gate G-R2 pass)

| check | result | detail |
|---|---|---|
| P1 prefetch (repetition 1) | pass | q15: 29.5 vs 31.2 s (0.945x), same as E025 True, re-read per pass [2080, 2016] MiB (limit 2202), waited 1.8 s, prefetched 257 (used 253, wasted 0); q3: 59.2 vs 62.6 s (0.947x), same as E025 True, re-read per pass [5022, 4958] MiB (limit 5144), waited 3.8 s, prefetched 625 (used 621, wasted 0) |
| P2 kept set (repetition 1) | pass | see P1 row |
| (report) repetition 2 | n/a | q15: 30.1 vs 31.2 s (0.963x), same as E025 True, re-read per pass [2080, 2016] MiB (limit 2202), waited 1.8 s, prefetched 257 (used 253, wasted 0); q3: 60.2 vs 62.9 s (0.958x), same as E025 True, re-read per pass [5022, 4958] MiB (limit 5144), waited 3.8 s, prefetched 625 (used 621, wasted 0); P2 True |
| C0r GPT-2 exact | pass | streamed = aligned reference: True |
| D0r GPT-2 LoRA exact | pass | same losses and adapters: True; prefetch wasted 106 of 2162 (E026: 56 of 1316) |
| C1r 1.5B 768 MiB | pass | same tokens and logits as E026: True; decode 1.51 s/token (E026 1.34), RSS+594 MiB, swap -16 MiB |
| R1r prediction | pass | predicted 1.43 vs 1.51 s/token (-5%) |
| (report) 1.5B LoRA prefetching | n/a | prefetched 876, used 565, wasted 311 in 2 steps (E026, 5 steps: 1269, 461, 658); losses [2.6643, 3.7763] (E026 starts [2.6643, 3.7763]) |

| case | status | swap MiB | attempt |
|---|---|---|---|
| llm__gpt2__reference | ok | +0 | 1 |
| llm__gpt2__stream__261MiB | ok | +0 | 1 |
| llm__q15__stream__768MiB | ok | -16 | 2 |
| lora2__q15__stream__768MiB | ok | +33 | 1 |
| lora__gpt2__reference | ok | +0 | 1 |
| lora__gpt2__stream__261MiB | ok | -16 | 1 |
| pipeline__q15__memopro__1024MiB__rep1 | ok | +0 | 1 |
| pipeline__q15__memopro__1024MiB__rep2 | ok | +0 | 1 |
| pipeline__q15__stream_nc__rep1 | ok | -8 | 1 |
| pipeline__q15__stream_nc__rep2 | ok | -56 | 1 |
| pipeline__q3__memopro__1024MiB__rep1 | ok | +0 | 1 |
| pipeline__q3__memopro__1024MiB__rep2 | ok | -8 | 1 |
| pipeline__q3__stream_nc__rep1 | ok | -8 | 1 |
| pipeline__q3__stream_nc__rep2 | ok | -24 | 1 |
