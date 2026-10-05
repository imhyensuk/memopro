# E034 summary (0150)

| case | tool | model | finished | steady tok/s | peak footprint growth MiB | swap MiB | losses |
|---|---|---|---|---|---|---|---|
| P15 | memopro | q15 | 10 steps | 45.4 | 1411 | -16 | [2.802, 3.915, 3.349, 3.553, 3.631, 3.567, 3.456, 3.875, 3.852, 3.929] |
| P3 | memopro | q3 | 10 steps | 20.0 | 1680 | -376 | [2.569, 3.72, 3.448, 3.444, 3.665, 3.385, 3.288, 3.83, 3.804, 3.809] |
| X15 | mlx | q15 | 10 steps | 118.9 | 3632 | +227 | [4.031, 3.482, 3.601, 3.64, 3.846, 3.81, 2.658, 3.494, 3.709, 3.324] |
| X3 | | | no: filename)
FileNotFoundError: [Errno 2] No such file or directory: 'mlx_lm.lora'
 | | | +1442 | |

| prediction | held | detail |
|---|---|---|
| Q1 3B memory | yes | memopro +1680 MiB; mlx-tune did not finish |
| Q4 3B speed | yes | memopro 20.0 tok/s; mlx-tune did not finish |
| Q3 1.5B speed (mlx-tune >= 2x) | yes | mlx-tune 118.9, memopro 45.4 tok/s |
| Q2 3B mlx-tune swap > 1 GiB or unfinished | yes | +1442 MiB |
