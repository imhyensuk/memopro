Dev check after 0343 (whole-process ceiling for MLX), not part of the judged E067 run.
Same worker: `.venv/bin/python -m experiments.e067_mlx_compare.run --case M15|M3`, run alone
after the judged run (system swap already 3.6 GB in use). Losses and adapter digests equal
cases/O15.json and cases/O3.json. Peak footprint: M15 1,461 MiB (ceiling 1,536), M3 2,432 MiB
(ceiling 2,560).
