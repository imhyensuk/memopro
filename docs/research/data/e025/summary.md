# E025 summary (0111)

Gate G-R1: **pass**

| check | result | detail |
|---|---|---|
| H2 lossless | pass | derived: same (2); gpt2: same (7); q15: same (4); q3: same (2) |
| H1 ceiling | pass | derived__memopro__256MiB: peak 252 / RSS+309 of 256 MiB; gpt2__memopro__130MiB: peak 123 / RSS+199 of 131 MiB; gpt2__memopro__261MiB: peak 256 / RSS+326 of 261 MiB; gpt2__memopro_lru__261MiB: peak 256 / RSS+321 of 261 MiB; q15__memopro__1024MiB: peak 1008 / RSS+1045 of 1024 MiB; q15__memopro__512MiB: peak 496 / RSS+533 of 512 MiB; q15__memopro_lru__1024MiB: peak 1008 / RSS+858 of 1024 MiB; q3__memopro__1024MiB: peak 1017 / RSS+1048 of 1024 MiB |
| H3 no writes | pass | derived__memopro__256MiB: swap +0 MiB; gpt2__memopro__130MiB: swap +0 MiB; gpt2__memopro__261MiB: swap +0 MiB; gpt2__memopro_lru__261MiB: swap +0 MiB; q15__memopro__1024MiB: swap +0 MiB; q15__memopro__512MiB: swap +0 MiB; q15__memopro_lru__1024MiB: swap +0 MiB; q3__memopro__1024MiB: swap +0 MiB |
| H4 runs what did not fit | pass | q15__memopro__1024MiB, q15__memopro__512MiB, q3__memopro__1024MiB |
| H5 vs manual streaming | pass | gpt2__memopro__130MiB: 1.5 vs 1.5 s (0.97x); gpt2__memopro__261MiB: 1.5 vs 1.5 s (0.95x); q15__memopro__1024MiB: 21.7 vs 21.5 s (1.01x); q15__memopro__512MiB: 21.9 vs 21.5 s (1.02x); q3__memopro__1024MiB: 44.7 vs 43.6 s (1.03x) |
| H7 policy | pass | gpt2__memopro__130MiB: re-read [416, 427] MiB <= 444; gpt2__memopro__261MiB: re-read [288, 283] MiB <= 339; q15__memopro__1024MiB: re-read [1952, 1952] MiB <= 2151; q15__memopro__512MiB: re-read [2464, 2464] MiB <= 2560; q3__memopro__1024MiB: re-read [4889, 4894] MiB <= 5092 |
| H8 compression | pass | compressions 107, ratio 2.91 |
| H6 (report) vs naive | n/a | gpt2__memopro__130MiB: 0.90x naive; gpt2__memopro__261MiB: 0.89x naive |

| case | W MiB | total s | pass 1/2/3 s | read MiB | RSS growth MiB | swap MiB | status |
|---|---|---|---|---|---|---|---|
| derived__memopro__256MiB | 512 | 3.4 | 0.6 / 0.7 / 1.7 | 0 | 309 | +0 | ok |
| derived__naive | 512 | 1.7 | 0.1 / 0.2 / 1.2 | 0 | 573 | +0 | ok |
| gpt2__memmap | 523 | 1.4 | 0.2 / 0.2 / 1.0 | 0 | 575 | +0 | ok |
| gpt2__memopro__130MiB | 523 | 1.5 | 0.2 / 0.2 / 1.1 | 1365 | 199 | +0 | ok |
| gpt2__memopro__261MiB | 523 | 1.5 | 0.2 / 0.2 / 1.1 | 1093 | 326 | +0 | ok |
| gpt2__memopro_lru__261MiB | 523 | 1.5 | 0.2 / 0.2 / 1.1 | 1568 | 321 | +0 | ok |
| gpt2__naive | 523 | 1.7 | 0.2 / 0.2 / 1.1 | 0 | 566 | +0 | ok |
| gpt2__stream | 523 | 1.6 | 0.2 / 0.2 / 1.1 | 1568 | 68 | +0 | ok |
| gpt2__stream_nc | 523 | 1.5 | 0.2 / 0.2 / 1.1 | 1568 | 68 | +0 | ok |
| q15__memopro__1024MiB | 2944 | 21.7 | 3.2 / 3.4 / 15.0 | 6849 | 1045 | +0 | ok |
| q15__memopro__512MiB | 2944 | 21.9 | 3.1 / 3.6 / 15.2 | 7873 | 533 | +0 | ok |
| q15__memopro_lru__1024MiB | 2944 | 22.9 | 3.3 / 4.0 / 15.6 | 8833 | 858 | +0 | ok |
| q15__stream_nc | 2944 | 21.5 | 2.9 / 3.5 / 15.0 | 8833 | 56 | +0 | ok |
| q3__memopro__1024MiB | 5886 | 44.7 | 6.7 / 7.4 / 30.6 | 15669 | 1048 | +0 | ok |
| q3__stream_nc | 5886 | 43.6 | 5.8 / 7.1 / 30.6 | 17658 | 67 | -16 | ok |
