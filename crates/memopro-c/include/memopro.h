/*
 * memopro C ABI (docs/research/0124): a runtime that keeps large buffers under a hard memory
 * budget, losslessly, without writing to disk.
 *
 * Buffers are registered from a file region (dropped and re-read, verified by digest, when
 * memory is short), made in memory (compressed when memory is short), or derived from other
 * buffers by a deterministic function (re-computed when needed, checked against the first
 * result). Use a buffer by pinning it: a pinned buffer stays in memory at the same address until
 * it is unpinned. What the budget cannot hold is refused with MP_ERR_BUDGET, never swapped.
 *
 * Every function returning int returns MP_OK (0) or a negative error code; mp_last_error()
 * describes the last error on the calling thread. All functions are thread-safe.
 *
 *     mp_config cfg;
 *     mp_config_default(&cfg, (uint64_t)2 << 30);        // 2 GiB
 *     mp_runtime *rt;
 *     mp_runtime_new(&cfg, &rt);
 *     mp_buffer b;
 *     mp_add_file(rt, "big.bin", 0, n, sizeof(float), &b);
 *     mp_pin *p;
 *     mp_pin_acquire(rt, b, 0, &p);
 *     const float *x = (const float *)mp_pin_data(p);    // valid until mp_unpin
 *     ...
 *     mp_unpin(p);
 *     mp_runtime_free(rt);
 */
#ifndef MEMOPRO_H
#define MEMOPRO_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MP_ABI_VERSION 1

#define MP_OK 0
#define MP_NO_DATA 1                /* mp_predict: no repeating cycle recorded yet */
#define MP_ERR_INVALID (-1)         /* invalid argument (unknown buffer, bad size, null pointer) */
#define MP_ERR_BUDGET (-2)          /* the budget cannot hold what was asked */
#define MP_ERR_INTEGRITY (-3)       /* data would not come back as it was (changed file, ...) */
#define MP_ERR_IO (-4)              /* an operating-system I/O error */
#define MP_ERR_UNSUPPORTED (-5)     /* not supported on this system */
#define MP_ERR_NOT_IMPLEMENTED (-6) /* not built yet */
#define MP_ERR_PANIC (-99)          /* a bug in memopro; do not use this runtime any more */

typedef struct mp_runtime mp_runtime;
typedef struct mp_pin mp_pin;
typedef uint64_t mp_buffer;

enum { MP_POLICY_REUSE = 0, MP_POLICY_LRU = 1 };
enum { MP_RESIDENT = 0, MP_COMPRESSED = 1, MP_DROPPED = 2, MP_UNLOADED = 3 };

typedef struct {
    uint64_t budget;        /* bytes the runtime's buffers may occupy */
    int32_t compress_level; /* zstd level (1 = fast) */
    double min_saving;      /* smallest fraction compression must save to be used */
    int32_t policy;         /* MP_POLICY_* */
    int32_t prefetch;       /* 1: bring the next buffers back in the background */
    uint64_t lookahead;     /* bytes to bring back ahead of use */
} mp_config;

typedef struct {
    size_t size; /* set to sizeof(mp_stats) by the caller */
    uint64_t budget, reserve, used, peak_used, buffers;
    uint64_t resident_bytes, compressed_bytes, pinned_bytes, pins;
    uint64_t loads, load_bytes, rereads, reread_bytes;
    uint64_t drops, drop_bytes, compressions, compress_in, compress_out;
    uint64_t decompressions, recomputes, recompute_bytes;
    uint64_t prefetches, prefetch_hits, prefetch_wasted, refusals, written_bytes;
    double read_seconds, compress_seconds, decompress_seconds, recompute_seconds;
    double restore_seconds;
} mp_stats;

typedef struct {
    size_t size; /* set to sizeof(mp_prediction) by the caller */
    uint64_t cycle_pins, cycle_bytes, restore_bytes;
    double restore_seconds, compute_seconds, seconds, last_seconds;
    int32_t prefetch;
} mp_prediction;

/*
 * The recipe of a derived buffer: fill `out` (out_size bytes) from the inputs. It must be
 * deterministic and thread-safe, must not keep the input pointers, and returns 0 on success.
 * It runs on the thread that needs the buffer; `user` must stay valid until the buffer is freed.
 */
typedef int (*mp_compute_fn)(void *user, const void *const *inputs, const size_t *input_sizes,
                             size_t n_inputs, void *out, size_t out_size);

uint32_t mp_abi_version(void);
const char *mp_version(void);
/* Message of the last error on this thread ("" if none); valid until the next call. */
const char *mp_last_error(void);

void mp_config_default(mp_config *config, uint64_t budget);
int mp_runtime_new(const mp_config *config, mp_runtime **out);
/* Stops the runtime; pins still held keep their memory until they are unpinned. */
void mp_runtime_free(mp_runtime *rt);
uint64_t mp_limit(const mp_runtime *rt);

int mp_alloc(mp_runtime *rt, size_t nbytes, size_t elem, mp_buffer *out);
int mp_add_file(mp_runtime *rt, const char *path, uint64_t offset, size_t nbytes, size_t elem,
                mp_buffer *out);
int mp_derive(mp_runtime *rt, const mp_buffer *inputs, size_t n_inputs, size_t nbytes,
              size_t elem, mp_compute_fn fn, void *user, mp_buffer *out);

/* write = 1: the only pin of the buffer, writable; the buffer then has no file to re-read. */
int mp_pin_acquire(mp_runtime *rt, mp_buffer id, int write, mp_pin **out);
void *mp_pin_data(const mp_pin *pin);
size_t mp_pin_size(const mp_pin *pin);
void mp_unpin(mp_pin *pin);

int mp_prefetch(mp_runtime *rt, mp_buffer id);
int mp_evict(mp_runtime *rt, mp_buffer id, int *evicted);
int mp_free(mp_runtime *rt, mp_buffer id);
int mp_state(const mp_runtime *rt, mp_buffer id, int *state);
int mp_nbytes(const mp_runtime *rt, mp_buffer id, size_t *out);
int mp_stats_get(const mp_runtime *rt, mp_stats *out);
/* MP_NO_DATA until a buffer has been used twice. */
int mp_predict(const mp_runtime *rt, mp_prediction *out);

#ifdef __cplusplus
}
#endif

#endif /* MEMOPRO_H */
