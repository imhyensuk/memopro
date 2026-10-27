# E067 summary (0341)

| case | tool | model | finished | steady tok/s | footprint growth MiB | MLX peak MiB | swap MiB |
|---|---|---|---|---|---|---|---|
| M15 | memopro | 15 | 10 steps | 52.8 | 1784 | 1488 | -40 |
| M3 | memopro | 3 | 10 steps | 22.8 | 2743 | 2419 | +18 |
| O15 | original | 15 | 10 steps | 106.5 | 3829 | 3515 | +1034 |
| O3 | original | 3 | 10 steps | 1.8 | 6919 | 6715 | +872 |
| T15 | mlx-tune | 15 | 10 steps | 66.3 | 3722 |  | -212 |
| T3 | mlx-tune | 3 | no: exit 1 /data/e067/mlx-tune/T3/lora_config.yaml', '--save-every', '100', '--grad-checkpoint']' returned non-zero exit status 1.  | | | | +5668 |

| check | held | detail |
|---|---|---|
| K1 1.5B identical to the original | yes | losses and adapter digests |
| K2 3B identical to the original | yes | original finished |
| K3 M15 within the ceiling | yes | MLX peak 1488, footprint +1784, ceiling 1536 MiB |
| K4 M15 swap <= 512 MiB | yes | -40 MiB |
| K3 M3 within the ceiling | yes | MLX peak 2419, footprint +2743, ceiling 2560 MiB |
| K4 M3 swap <= 512 MiB | yes | +18 MiB |
| K5 M3 finishes | yes | 10 steps |
| P1 M15 >= 0.6x O15 | no | 52.8 vs 106.5 |
| P2 T15 >= 1.5x M15 | no | 66.3 vs 52.8 |
| P3 M3 >= 24 tok/s | no | 22.8 |
| P4 O3 fails or swaps > 1 GiB | no | finished, swap +872 MiB |
| P4 T3 fails or swaps > 1 GiB | yes | did not finish, swap +5668 MiB |
| P5 1.5B memopro swaps least | no | memopro -40 MiB |
| P5 3B memopro swaps least | yes | memopro +18 MiB |
