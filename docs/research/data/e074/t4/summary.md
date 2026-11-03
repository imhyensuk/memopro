# E074: a 14B model on a T4

run `20261103-070335`, cell `extreme`

| case | status | first_s | steady | step_s | host_peak | gpu_peak | written | offload_folder | device_map |
|---|---|---|---|---|---|---|---|---|---|
| plain | oom | - | - | - | 4.83 GiB | 14.41 GiB | 0.00 GiB | - |  |
| accelerate | ok | 72.2 | 71.26 | - | 7.05 GiB | 12.68 GiB | 0.00 GiB | 0.00 GiB | {'0': 20, 'cpu': 8, 'disk': 24} |
| memopro_gen | ok | 199.3 | 55.95 | - | 5.88 GiB | 14.33 GiB | 0.00 GiB | - |  |
| ft_6144 | error | - | - | - | 5.44 GiB | 14.39 GiB | 0.00 GiB | - |  |
| ft_9216 | error | - | - | - | 8.40 GiB | 14.39 GiB | 0.00 GiB | - |  |

- E1 plain does not complete: PASS
- E2 memopro generates within 6 GiB, writes <= 64 MiB: PASS
- E3 memopro trains at both ceilings within them, writes <= 64 MiB: FAIL
- E4 losses bit-identical across ceilings: FAIL
Reported: memopro text == accelerate text: True.
first_s = first token (pays for loading); steady = s/token after it; step_s = mean training step after the first; written = /proc/self/io write_bytes.
Pre-registered: docs/research/0379 (E074). main@f4f3d1e4815c; elapsed 68 min.

- `ft_6144` (error): RuntimeError: CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`
- `ft_9216` (error): RuntimeError: CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`
- `plain` (oom): OutOfMemoryError: CUDA out of memory. Tried to allocate 50.00 MiB. GPU 0 has a total capacity of 14.56 GiB of which 49.81 MiB is free. Including non-PyTorch memory, this process has 14.51 GiB memory in use. Of the allocated memory 14.41 GiB is allocated by PyTorch, and 1.16 MiB is reserved by PyTorc
