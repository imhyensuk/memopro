# E026 summary (0114)

Gate G-R2: **fail**

| check | result | detail |
|---|---|---|
| P1 prefetch | fail | q15: 21.6 vs 22.8 s (0.95x), same as E025 True, prefetched 429 (used 367, wasted 0); q3: 42.2 vs 43.5 s (0.97x), same as E025 True, prefetched 800 (used 737, wasted 0) |
| L1 recompute | pass | 2.3 s (naive 1.7 s), re-computed 95 times (1520 MiB, 0.5 s), peak 120 MiB, swap +0 MiB, same result True |
| C0 GPT-2 exact | pass | streamed = aligned reference: True; file-mapped HF vs aligned: tokens same True, max |diff| prompt 0, last step 9.2e-05 |
| C1 1.5B | pass | 768 MiB: decode 1.34 s/token, first token 3.4 s, RSS+590 MiB, swap +0 MiB, peak 764 MiB; 1024 MiB: decode 1.21 s/token, first token 3.5 s, RSS+674 MiB, swap +0 MiB, peak 1019 MiB; same results: True |
| C2 3B | pass | 1536 MiB: decode 2.72 s/token, first token 6.5 s, RSS+1188 MiB, swap +10 MiB, peak 1532 MiB; 1024 MiB: decode 2.99 s/token, first token 6.4 s, RSS+905 MiB, swap -8 MiB, peak 1020 MiB; same results: True |
| R1 prediction | pass | llm__q15__stream__768MiB: predicted 1.31 vs 1.34 s/token (-2%); llm__q3__stream__1536MiB: predicted 2.57 vs 2.72 s/token (-6%) |
| D0 GPT-2 LoRA exact | pass | losses [4.0068, 3.901, 3.9875, 3.8187, 4.2482, 4.0557, 4.5669, 3.0674, 3.4509, 3.8462]; 0.59 vs 0.40 s/step; data: wikitext-2 train |
| D1 1.5B LoRA | pass | 768 MiB: 257.2 s/step, RSS+352 MiB, swap -224 MiB, peak 764 MiB; 1024 MiB: 281.9 s/step, RSS+527 MiB, swap -207 MiB, peak 1020 MiB; same results: True, same adapters True |
| S1 (report) OS paging | n/a | q15: OS paging 7.38 vs memopro 1.34 s/token, same tokens True, swap +0 MiB; q3: OS paging 13.59 vs memopro 2.72 s/token, same tokens True, swap -16 MiB |

| case | status | time | RSS growth MiB | swap MiB | attempt |
|---|---|---|---|---|---|
| lineage__derived__128MiB | ok | 2.3 s | 187 | +0 | 1 |
| llm__gpt2__mmap | ok | 0.021 s/token | 972 | +0 | 1 |
| llm__gpt2__reference | ok | 0.018 s/token | 530 | +0 | 1 |
| llm__gpt2__stream__261MiB | ok | 0.066 s/token | 327 | +0 | 1 |
| llm__q15__mmap | ok | 7.381 s/token | 2017 | +0 | 1 |
| llm__q15__stream__1024MiB | ok | 1.211 s/token | 674 | +0 | 1 |
| llm__q15__stream__768MiB | ok | 1.335 s/token | 590 | +0 | 1 |
| llm__q3__mmap | ok | 13.586 s/token | 2033 | -16 | 1 |
| llm__q3__stream__1024MiB | ok | 2.986 s/token | 905 | -8 | 1 |
| llm__q3__stream__1536MiB | ok | 2.724 s/token | 1188 | +10 | 2 |
| lora__gpt2__reference | ok | 0.40 s/step | 723 | +0 | 1 |
| lora__gpt2__stream__261MiB | ok | 0.59 s/step | 550 | +0 | 1 |
| lora__q15__stream__1024MiB | ok | 281.91 s/step | 527 | -207 | 1 |
| lora__q15__stream__768MiB | ok | 257.17 s/step | 352 | -224 | 1 |
| pipeline__derived__naive | ok | 1.7 s | 573 | +0 | 1 |
| pipeline__q15__memopro__1024MiB | ok | 21.6 s | 794 | -16 | 1 |
| pipeline__q15__stream_nc | ok | 22.8 s | 53 | -32 | 1 |
| pipeline__q3__memopro__1024MiB | ok | 42.2 s | 899 | +0 | 1 |
| pipeline__q3__stream_nc | ok | 43.5 s | 56 | -8 | 1 |
