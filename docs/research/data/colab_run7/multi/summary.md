# memopro Colab T4: multiple models and process-level features

run `20260929-043901`, cell `multi`

| test | status | result |
|---|---|---|
| script Qwen2.5-1.5B-Instruct via memopro_run | ok | APP loaded in 4.6s on cuda:0 | APP generated in 2.2s: GPUs run out of memory due to their limited on-chip VRAM capacity compared to the large amounts of RAM used in CPUs for managing system-wide data and | APP peak_allocated 3102708736 |
| script Qwen2.5-1.5B-Instruct via python | ok | APP loaded in 11.4s on cuda:0 | APP generated in 2.3s: GPUs run out of memory due to their limited on-chip VRAM capacity compared to the large amounts of RAM used in CPUs for managing system-wide data and | APP peak_allocated 3102707712 |
| script Qwen2.5-7B-Instruct via memopro_run | ok | APP loaded in 63.8s on cuda:0 | APP generated in 2.8s: GPUs run out of memory when the amount of data they need to process exceeds the capacity of their VRAM (video random-access memory). | APP peak_allocated 15294390784 |
| script Qwen2.5-7B-Instruct via python | skipped | plain loading needs ~14.2 GiB host RAM, 10.4 GiB available: not run, the OOM killer could take down the notebook kernel |
| doctor | ok | device budget / CUDA free = 0.90; host budget / MemAvailable = 0.90 |
| regression checks | ok | pass 7/7; failed: none |
| rotate: hibernate | ok | turns 6; total 377.5 s; switch (after first round) median 59.42 s; answers repeat: True; peak GPU 7.29 GiB;  |
| rotate: keep_all | oom | turns 2; total - s; switch (after first round) median - s; answers repeat: None; peak GPU 13.29 GiB; OutOfMemoryError: CUDA out of memory. Tried to allocate 28.00 MiB. GPU 0 has a t |
| rotate: reload | ok | turns 6; total 177.4 s; switch (after first round) median 28.71 s; answers repeat: True; peak GPU 13.29 GiB;  |

Reading guide:
- rotate: the same prompt per model each turn; `answers repeat` checks that a model answers identically after being reloaded or woken (hibernate must be exact).
- script: `python` is the unmodified script; `memopro run` applies memopro's loading policy only when the script chose nothing and the model does not fit as stored.
- full details (per turn, per check) are in cases/*.json; logs/ has each process's output.
