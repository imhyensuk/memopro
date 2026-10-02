/* C smoke test of include/memopro.h (built and run by CI after `cargo build -p memopro-c`):
 * a file larger than the budget is read back exactly through pins, a derived buffer comes back
 * by re-computation, and errors come back as codes with a message. */
#define _POSIX_C_SOURCE 200809L
#include "memopro.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MIB ((size_t)1 << 20)
#define BLOCKS 16

#define CHECK(expr)                                                                       \
    do {                                                                                  \
        int rc_ = (expr);                                                                 \
        if (rc_ != MP_OK) {                                                               \
            fprintf(stderr, "%s:%d: %s -> %d (%s)\n", __FILE__, __LINE__, #expr, rc_,     \
                    mp_last_error());                                                     \
            return 1;                                                                     \
        }                                                                                 \
    } while (0)

static unsigned char byte_at(size_t i) { return (unsigned char)((i * 2654435761u) >> 11); }

static int widen(void *user, const void *const *inputs, const size_t *sizes, size_t n, void *out,
                 size_t out_size) {
    (void)user;
    if (n != 1 || out_size != sizes[0] * 2) return 1;
    const unsigned short *src = (const unsigned short *)inputs[0];
    unsigned int *dst = (unsigned int *)out;
    for (size_t i = 0; i < sizes[0] / 2; i++) dst[i] = (unsigned int)src[i] << 16;
    return 0;
}

int main(void) {
    char path[] = "/tmp/memopro-c-smoke-XXXXXX";
    FILE *f = NULL;
    int fd = mkstemp(path);
    if (fd < 0 || !(f = fdopen(fd, "wb"))) return 1;
    unsigned char *block = malloc(MIB);
    for (size_t b = 0; b < BLOCKS; b++) {
        for (size_t i = 0; i < MIB; i++) block[i] = byte_at(b * MIB + i);
        fwrite(block, 1, MIB, f);
    }
    fclose(f);

    printf("memopro %s, C ABI %u\n", mp_version(), mp_abi_version());
    mp_config cfg;
    mp_config_default(&cfg, 12 * MIB);
    mp_runtime *rt = NULL;
    CHECK(mp_runtime_new(&cfg, &rt));

    mp_buffer ids[BLOCKS];
    for (size_t b = 0; b < BLOCKS; b++) CHECK(mp_add_file(rt, path, b * MIB, MIB, 1, &ids[b]));
    for (int pass = 0; pass < 3; pass++) {
        for (size_t b = 0; b < BLOCKS; b++) {
            mp_pin *p = NULL;
            CHECK(mp_pin_acquire(rt, ids[b], 0, &p));
            const unsigned char *x = mp_pin_data(p);
            if (mp_pin_size(p) != MIB) return 2;
            for (size_t i = 0; i < MIB; i += 4097)
                if (x[i] != byte_at(b * MIB + i)) return 3;
            mp_unpin(p);
        }
    }

    mp_buffer wide;
    CHECK(mp_derive(rt, &ids[0], 1, 2 * MIB, 4, widen, NULL, &wide));
    CHECK(mp_evict(rt, wide, NULL));
    mp_pin *p = NULL;
    CHECK(mp_pin_acquire(rt, wide, 0, &p));
    const unsigned int *w = mp_pin_data(p);
    for (size_t i = 0; i < MIB / 2; i += 1001) {
        unsigned int lo = byte_at(2 * i), hi = byte_at(2 * i + 1);
        if (w[i] != ((lo | (hi << 8)) << 16)) return 4;
    }
    mp_unpin(p);

    mp_buffer too_big;
    if (mp_alloc(rt, 64 * MIB, 4, &too_big) != MP_ERR_BUDGET) return 5;
    if (strstr(mp_last_error(), "budget") == NULL) return 6;

    mp_stats s;
    memset(&s, 0, sizeof s);
    s.size = sizeof s;
    CHECK(mp_stats_get(rt, &s));
    printf("peak %llu of limit %llu, re-read %llu times, re-computed %llu, written %llu\n",
           (unsigned long long)s.peak_used, (unsigned long long)mp_limit(rt),
           (unsigned long long)s.rereads, (unsigned long long)s.recomputes,
           (unsigned long long)s.written_bytes);
    if (s.peak_used > mp_limit(rt) || s.rereads == 0 || s.recomputes == 0 || s.written_bytes != 0)
        return 7;

    mp_runtime_free(rt);
    remove(path);
    free(block);
    puts("ok");
    return 0;
}
