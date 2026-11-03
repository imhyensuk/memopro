# E074: a 14B model on a T4

run `20261103-113859`, cell `extreme`

| case | status | first_s | steady | step_s | host_peak | gpu_peak | written | offload_folder | device_map |
|---|---|---|---|---|---|---|---|---|---|
| plain | not run | - | - | - | - | - | - | - |  |
| accelerate | not run | - | - | - | - | - | - | - |  |
| memopro_gen | not run | - | - | - | - | - | - | - |  |
| ft_6144 | ok | - | - | 190.2 | 5.90 GiB | 14.36 GiB | 0.00 GiB | - |  |
| ft_9216 | ok | - | - | 182.5 | 8.92 GiB | 14.36 GiB | 0.00 GiB | - |  |

- E3 memopro trains at both ceilings within them, writes <= 64 MiB: PASS
- E4 losses bit-identical across ceilings: PASS
Reported: memopro text == accelerate text: None.
first_s = first token (pays for loading); steady = s/token after it; step_s = mean training step after the first; written = /proc/self/io write_bytes.
Pre-registered: docs/research/0379 (E074), 0380 (E074b). e074-extreme@6de680e70999; elapsed 32 min.

