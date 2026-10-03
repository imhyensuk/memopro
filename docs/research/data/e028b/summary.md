# E028b summary (0134)

Gate G4-B1: **fail**

| check | result | detail |
|---|---|---|
| B1 speed | pass | 1.5B steady 5.7 s/step vs CPU 257.2 s (45x faster) |
| B2 3B completes, same losses | pass | [2.426, 3.5672, 3.2829, 3.2632, 3.4844] |
| B3 memory | fail | L15: footprint +1980 MiB (budget 768), swap -7 MiB, peak 764 MiB; L3a: footprint +1370 MiB (budget 1024), swap -40 MiB, peak 1020 MiB; L3b: footprint +1111 MiB (budget 768), swap -8 MiB, peak 763 MiB |

| case | step s | tokens/s (steady) | re-read per step MiB |
|---|---|---|---|
| L15 | [7.3, 5.6, 6.0, 5.6, 5.8] | 22.3 | [4779, 5505, 6121, 4947, 5515] |
| L3a | [14.1, 11.4, 11.2, 11.0, 11.0] | 11.5 | [9163, 11979, 10984, 11030, 10525] |
| L3b | [12.4, 10.9, 11.0, 10.8, 10.8] | 11.8 | [5972, 11609, 11649, 11503, 11477] |
