# E051B summary (0252, 0254)

| case | batch | budget MiB | steps | step s (median) | tok/s (steps 2+) | re-read GiB | footprint +MiB | swap MiB |
|---|---|---|---|---|---|---|---|---|
| b1-1 | 1 | 2560 | 16 | 15.4 | 8.3 | 394 | 2720 | +362 |
| b1-2 | 1 | 2560 | 16 | 15.2 | 8.4 | 398 | 2703 | +0 |
| b2-1 | 2 | 2560 | 8 | 19.9 | 12.9 | 208 | 2699 | +0 |
| b2-2 | 2 | 2560 | 8 | 19.6 | 13.2 | 207 | 2699 | +0 |
| b4-1 | 4 | 2560 | 4 | 33.1 | 15.6 | 92 | 2631 | +0 |
| b4-2 | 4 | 2560 | 4 | 33.5 | 15.4 | 94 | 2614 | -8 |
| b4-small | 4 | 2048 | 4 | 34.5 | 14.9 | 102 | 2113 | +0 |

- K1 repeated runs identical: yes
- K1 same batch, another budget: yes
- K2 footprint within budget + 512 MiB: yes
- tok/s median: batch 1 8.4 (1.00x), batch 2 13.0 (1.56x), batch 4 15.5 (1.86x)
- K3 batch 4 >= 1.8x batch 1: yes

Decision (0252/0254): PASS
