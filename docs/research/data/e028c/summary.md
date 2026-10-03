# E028c summary (0137)

Gate G4-B1: **pass**

| check | result | detail |
|---|---|---|
| B1 speed | pass | 1.5B steady 5.6 s/step vs CPU 257.2 s (46x faster) |
| B2 3B completes, same losses | pass | [2.426, 3.5749, 3.287, 3.2601, 3.4832] |
| B3 memory | pass | L15: footprint +997 MiB (budget 768), swap -168 MiB, peak 764 MiB; L3a: footprint +1266 MiB (budget 1024), swap -104 MiB, peak 1020 MiB; L3b: footprint +1011 MiB (budget 768), swap -24 MiB, peak 764 MiB |

| case | step s | tokens/s (steady) | re-read per step MiB |
|---|---|---|---|
| L15 | [8.9, 5.4, 5.4, 6.1, 5.4] | 22.9 | [3995, 6249, 6624, 6210, 6312] |
| L3a | [11.6, 10.1, 10.1, 11.0, 11.8] | 11.9 | [5748, 12935, 12641, 12768, 12454] |
| L3b | [12.1, 10.6, 10.3, 10.3, 11.1] | 12.1 | [5982, 12328, 12844, 12830, 12669] |
