// E015 helper: Metal buffers for the experiments (built at run time by metal.py with clang).
// - mp_nocopy: an MTLBuffer over existing page-aligned memory (e.g. a read-only mmap of a file)
// - mp_new: an ordinary shared MTLBuffer; mp_contents: its CPU address
// - mp_purgeable: set (or, with 1 = KeepCurrent, read) the buffer's purgeable state
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

static id<MTLDevice> device(void) {
    static id<MTLDevice> d = nil;
    if (d == nil) d = MTLCreateSystemDefaultDevice();
    return d;
}

void *mp_nocopy(void *ptr, size_t length) {
    id<MTLBuffer> b = [device() newBufferWithBytesNoCopy:ptr
                                                  length:length
                                                 options:MTLResourceStorageModeShared
                                             deallocator:nil];
    return b == nil ? NULL : (__bridge_retained void *)b;
}

void *mp_new(size_t length) {
    id<MTLBuffer> b = [device() newBufferWithLength:length options:MTLResourceStorageModeShared];
    return b == nil ? NULL : (__bridge_retained void *)b;
}

void *mp_contents(void *buffer) { return [(__bridge id<MTLBuffer>)buffer contents]; }

int mp_purgeable(void *buffer, int state) {
    return (int)[(__bridge id<MTLBuffer>)buffer setPurgeableState:(MTLPurgeableState)state];
}

void mp_release(void *buffer) { CFRelease(buffer); }
