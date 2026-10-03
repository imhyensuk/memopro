# E028 summary (0131)

Gate G4-B1: **fail**

| check | result | detail |
|---|---|---|
| B1 speed | pass | 1.5B steady 5.6 s/step vs CPU 257.2 s (46x faster) |
| B2 3B completes, same losses | pass | [2.4207, 3.5692, 3.2889, 3.2658, 3.4831] |
| B3 memory | fail | L15: footprint +1723 MiB (budget 768), swap +42 MiB, peak 764 MiB; L3a: footprint +2303 MiB (budget 1024), swap +54 MiB, peak 1020 MiB; L3b: footprint +2038 MiB (budget 768), swap +199 MiB, peak 764 MiB |

| case | step s | tokens/s (steady) | re-read per step MiB |
|---|---|---|---|
| L15 | [6.7, 5.0, 6.6, 5.7, 5.1] | 22.9 | [4240, 5560, 5161, 5070, 4737] |
| L3a | [14.2, 9.9, 10.4, 12.4, 14.1] | 10.9 | [7463, 11353, 10264, 11441, 10839] |
| L3b | [13.0, 10.8, 10.8, 9.7, 9.8] | 12.5 | [7744, 11719, 11057, 11762, 11770] |
