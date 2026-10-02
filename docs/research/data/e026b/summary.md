# E026b summary (0117)

E026b checks: **fail** (with E026's passed checks: gate G-R2 fail)

| check | result | detail |
|---|---|---|
| P1 prefetch (repetition 1) | fail | q15: 30.7 vs 31.5 s (0.974x), same as E025 True, re-read per pass [3936, 2944] MiB (limit 2202), waited 1.8 s, prefetched 431 (used 369, wasted 0); q3: 61.4 vs 62.4 s (0.983x), same as E025 True, re-read per pass [6878, 5902] MiB (limit 5144), waited 3.8 s, prefetched 800 (used 737, wasted 0) |
| P2 kept set (repetition 1) | fail | see P1 row |
| (report) repetition 2 | n/a | q15: 31.1 vs 31.2 s (0.997x), same as E025 True, re-read per pass [3936, 2944] MiB (limit 2202), waited 1.8 s, prefetched 431 (used 369, wasted 0); q3: 62.0 vs 62.8 s (0.988x), same as E025 True, re-read per pass [6878, 5902] MiB (limit 5144), waited 3.8 s, prefetched 800 (used 737, wasted 0); P2 False |
| C0r GPT-2 exact | pass | streamed = aligned reference: True |
| D0r GPT-2 LoRA exact | pass | same losses and adapters: True; prefetch wasted 323 of 2278 (E026: 56 of 1316) |
| C1r 1.5B 768 MiB | pass | same tokens and logits as E026: True; decode 1.40 s/token (E026 1.34), RSS+619 MiB, swap +0 MiB |
| R1r prediction | pass | predicted 1.34 vs 1.40 s/token (-4%) |
| (report) 1.5B LoRA prefetching | n/a | prefetched 1239, used 591, wasted 641 in 2 steps (E026, 5 steps: 1269, 461, 658); losses [2.6643, 3.7763] (E026 starts [2.6643, 3.7763]) |

| case | status | swap MiB | attempt |
|---|---|---|---|
| llm__gpt2__reference | ok | +0 | 1 |
| llm__gpt2__stream__261MiB | ok | +0 | 1 |
| llm__q15__stream__768MiB | ok | +0 | 1 |
| lora2__q15__stream__768MiB | ok | -294 | 2 |
| lora__gpt2__reference | ok | +0 | 1 |
| lora__gpt2__stream__261MiB | ok | +0 | 1 |
| pipeline__q15__memopro__1024MiB__rep1 | ok | +0 | 1 |
| pipeline__q15__memopro__1024MiB__rep2 | ok | +0 | 1 |
| pipeline__q15__stream_nc__rep1 | ok | +0 | 1 |
| pipeline__q15__stream_nc__rep2 | ok | -8 | 1 |
| pipeline__q3__memopro__1024MiB__rep1 | ok | +0 | 1 |
| pipeline__q3__memopro__1024MiB__rep2 | ok | +0 | 1 |
| pipeline__q3__stream_nc__rep1 | ok | +0 | 1 |
| pipeline__q3__stream_nc__rep2 | ok | +0 | 1 |
