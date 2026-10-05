/* memopro-preload smoke test (0189): run with
 *   LD_PRELOAD=libmemopro_preload.so MEMOPRO_PRELOAD_BUDGET=67108864 \
 *   MEMOPRO_PRELOAD_REPORT=report.json ./smoke
 * 24 blocks of 8 MiB (3x the budget) through malloc, calloc, realloc and posix_memalign, each
 * filled with a compressible pattern and read back exactly; then small blocks. Prints "ok". */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

#define BLOCKS 24
#define WORDS ((8u << 20) / 4)

static uint32_t pattern(size_t b, size_t i) { return (uint32_t)(b * 7919u + i / 1024u); }

int main(void) {
    uint32_t *blocks[BLOCKS];
    for (size_t b = 0; b < BLOCKS; b++) {
        void *p = NULL;
        if (b % 4 == 0) p = malloc(WORDS * 4);
        else if (b % 4 == 1) {
            p = calloc(WORDS, 4);
            for (size_t i = 0; p && i < WORDS; i += 4096)
                if (((uint32_t *)p)[i]) { puts("calloc not zero"); return 1; }
        } else if (b % 4 == 2) {
            p = malloc(1 << 20);
            p = p ? realloc(p, WORDS * 4) : NULL;
        } else if (posix_memalign(&p, 4096, WORDS * 4) != 0) p = NULL;
        if (!p) { printf("allocation %zu failed\n", b); return 1; }
        if (malloc_usable_size(p) < WORDS * 4) { puts("usable size too small"); return 1; }
        blocks[b] = p;
        for (size_t i = 0; i < WORDS; i++) blocks[b][i] = pattern(b, i);
    }
    for (int pass = 0; pass < 2; pass++)
        for (size_t b = 0; b < BLOCKS; b++)
            for (size_t i = 0; i < WORDS; i++)
                if (blocks[b][i] != pattern(b, i)) {
                    printf("block %zu word %zu differs\n", b, i);
                    return 1;
                }
    blocks[0] = realloc(blocks[0], WORDS * 8);
    for (size_t i = 0; i < WORDS; i++)
        if (blocks[0][i] != pattern(0, i)) { puts("realloc lost data"); return 1; }
    for (size_t b = 0; b < BLOCKS; b++) free(blocks[b]);
    for (int i = 0; i < 100000; i++) free(malloc((size_t)(i % 4000) + 1));
    puts("ok");
    return 0;
}
