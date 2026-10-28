/* E069 V6 (docs/research/0356): the same runtime calls as crates/memopro/examples/e069_ffi.rs
 * through the C ABI (memopro-c). Prints one JSON line. */
#define _POSIX_C_SOURCE 200809L
#include "memopro.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define GIB ((size_t)1 << 30)

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static int cmp(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median(double *v, int n) {
    qsort(v, n, sizeof *v, cmp);
    return v[n / 2];
}

static void fail(const char *what) {
    fprintf(stderr, "%s: %s\n", what, mp_last_error());
    exit(2);
}

static mp_runtime *rt;
static mp_buffer small;

static void op_pin(void) {
    mp_pin *p = NULL;
    if (mp_pin_acquire(rt, small, 0, &p) != MP_OK) fail("mp_pin_acquire");
    mp_unpin(p);
}

static void op_alloc(void) {
    mp_buffer id;
    if (mp_alloc(rt, 64 << 10, 1, &id) != MP_OK) fail("mp_alloc");
    if (mp_free(rt, id) != MP_OK) fail("mp_free");
}

static void op_stats(void) {
    mp_stats s;
    s.size = sizeof s;
    if (mp_stats_get(rt, &s) != MP_OK) fail("mp_stats_get");
}

static double per_op(int reps, void (*f)(void)) {
    double ns[11];
    for (int r = 0; r < 11; r++) {
        double t = now();
        for (int i = 0; i < reps; i++) f();
        ns[r] = (now() - t) * 1e9 / reps;
    }
    return median(ns, 11);
}

int main(void) {
    mp_config cfg;
    mp_config_default(&cfg, 3 * GIB);
    if (mp_runtime_new(&cfg, &rt) != MP_OK) fail("mp_runtime_new");
    if (mp_alloc(rt, 1 << 20, 1, &small) != MP_OK) fail("mp_alloc");
    double pin_ns = per_op(100000, op_pin);
    double alloc_ns = per_op(10000, op_alloc);
    double stats_ns = per_op(100000, op_stats);

    unsigned char *src = malloc(GIB), *plain = malloc(GIB);
    for (size_t i = 0; i < GIB; i++) src[i] = (unsigned char)((i * 2654435761u) >> 11);
    memset(plain, 1, GIB);
    mp_buffer big;
    if (mp_alloc(rt, GIB, 1, &big) != MP_OK) fail("mp_alloc 1 GiB");
    double g[9], pg[9];
    uint64_t sum = 0;
    for (int r = 0; r < 9; r++) {
        mp_pin *p = NULL;
        if (mp_pin_acquire(rt, big, 1, &p) != MP_OK) fail("mp_pin_acquire 1 GiB");
        unsigned char *dst = mp_pin_data(p);
        double t = now();
        memcpy(dst, src, GIB);
        g[r] = GIB / (now() - t) / 1e9;
        sum = 0;
        for (size_t i = 0; i < GIB; i += 8) {
            uint64_t w;
            memcpy(&w, dst + i, 8);
            sum += w;
        }
        mp_unpin(p);
        t = now();
        memcpy(plain, src, GIB);
        pg[r] = GIB / (now() - t) / 1e9;
    }
    printf("{\"lang\": \"c\", \"pin_unpin_ns\": %.1f, \"alloc_free_64k_ns\": %.1f, \"stats_ns\": %.1f, "
           "\"copy_gbps\": %.2f, \"plain_memcpy_gbps\": %.2f, \"checksum\": %llu, \"sink\": %d}\n",
           pin_ns, alloc_ns, stats_ns, median(g, 9), median(pg, 9), (unsigned long long)sum,
           plain[GIB / 3]);
    mp_runtime_free(rt);
    return 0;
}
