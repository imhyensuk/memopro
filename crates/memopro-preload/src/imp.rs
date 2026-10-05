//! The interposed allocation functions; see the crate documentation.

use libc::{c_char, c_int, c_void, size_t};
use memopro::rt::{Pager, PagerConfig, in_pager_thread};
use std::alloc::{GlobalAlloc, Layout};
use std::cell::Cell;
use std::sync::atomic::{AtomicPtr, AtomicU8, AtomicU64, AtomicUsize, Ordering};

type MallocFn = unsafe extern "C" fn(size_t) -> *mut c_void;
type FreeFn = unsafe extern "C" fn(*mut c_void);
type CallocFn = unsafe extern "C" fn(size_t, size_t) -> *mut c_void;
type ReallocFn = unsafe extern "C" fn(*mut c_void, size_t) -> *mut c_void;
type PosixMemalignFn = unsafe extern "C" fn(*mut *mut c_void, size_t, size_t) -> c_int;
type AlignedFn = unsafe extern "C" fn(size_t, size_t) -> *mut c_void;
type UsableFn = unsafe extern "C" fn(*mut c_void) -> size_t;
type MadviseFn = unsafe extern "C" fn(*mut c_void, size_t, c_int) -> c_int;

static REAL_MALLOC: AtomicPtr<c_void> = AtomicPtr::new(std::ptr::null_mut());
static REAL_FREE: AtomicPtr<c_void> = AtomicPtr::new(std::ptr::null_mut());
static REAL_CALLOC: AtomicPtr<c_void> = AtomicPtr::new(std::ptr::null_mut());
static REAL_REALLOC: AtomicPtr<c_void> = AtomicPtr::new(std::ptr::null_mut());
static REAL_POSIX_MEMALIGN: AtomicPtr<c_void> = AtomicPtr::new(std::ptr::null_mut());
static REAL_ALIGNED_ALLOC: AtomicPtr<c_void> = AtomicPtr::new(std::ptr::null_mut());
static REAL_MEMALIGN: AtomicPtr<c_void> = AtomicPtr::new(std::ptr::null_mut());
static REAL_USABLE: AtomicPtr<c_void> = AtomicPtr::new(std::ptr::null_mut());
static REAL_MADVISE: AtomicPtr<c_void> = AtomicPtr::new(std::ptr::null_mut());

// `dlsym` may allocate while the real functions are being looked up: those few bytes come from
// here and are never given back
const ARENA_BYTES: usize = 1 << 16;
struct Arena(std::cell::UnsafeCell<[u8; ARENA_BYTES]>);
// SAFETY: handed out in disjoint pieces through the atomic offset below.
unsafe impl Sync for Arena {}
static ARENA: Arena = Arena(std::cell::UnsafeCell::new([0; ARENA_BYTES]));
static ARENA_USED: AtomicUsize = AtomicUsize::new(0);

const UNSET: u8 = 0;
const STARTING: u8 = 1;
const READY: u8 = 2;
const OFF: u8 = 3;
static STATE: AtomicU8 = AtomicU8::new(UNSET);
static PAGER: AtomicPtr<Pager> = AtomicPtr::new(std::ptr::null_mut());
static THRESHOLD: AtomicUsize = AtomicUsize::new(4 << 20);
static PAGE: AtomicUsize = AtomicUsize::new(0);
static REPORT: AtomicPtr<c_char> = AtomicPtr::new(std::ptr::null_mut());
static PAGED: AtomicU64 = AtomicU64::new(0);
static PAGED_BYTES: AtomicU64 = AtomicU64::new(0);
static STARTED: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();

thread_local! {
    /// Inside our own work (setting up, a pager call, the lookups): allocate from the system.
    static BUSY: Cell<u32> = const { Cell::new(0) };
    static RESOLVING: Cell<bool> = const { Cell::new(false) };
}

/// Rust code in this library (the pager included) allocates from the system directly.
struct System;

// SAFETY: forwards to the system allocator, which honours the layout as below.
unsafe impl GlobalAlloc for System {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.align() <= 16 {
            // SAFETY: plain allocation.
            unsafe { real_malloc(layout.size()).cast() }
        } else {
            let mut out = std::ptr::null_mut();
            let align = layout.align().max(std::mem::size_of::<usize>());
            // SAFETY: a power-of-two alignment that is a multiple of the pointer size.
            let r = unsafe { real_posix_memalign(&mut out, align, layout.size()) };
            if r == 0 {
                out.cast()
            } else {
                std::ptr::null_mut()
            }
        }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
        // SAFETY: allocated by the system above.
        unsafe { real_free(ptr.cast()) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if layout.align() <= 16 {
            // SAFETY: allocated by the system malloc above.
            unsafe { real_realloc(ptr.cast(), new_size).cast() }
        } else {
            // SAFETY: the default reallocation through alloc/copy/dealloc.
            unsafe {
                let new = self.alloc(Layout::from_size_align_unchecked(new_size, layout.align()));
                if !new.is_null() {
                    std::ptr::copy_nonoverlapping(ptr, new, layout.size().min(new_size));
                    self.dealloc(ptr, layout);
                }
                new
            }
        }
    }
}

#[global_allocator]
static GLOBAL: System = System;

fn lookup(name: &[u8]) -> *mut c_void {
    // SAFETY: a NUL-terminated symbol name; RTLD_NEXT finds the next definition (libc's).
    unsafe { libc::dlsym(libc::RTLD_NEXT, name.as_ptr().cast()) }
}

fn resolve() {
    RESOLVING.with(|r| r.set(true));
    REAL_MALLOC.store(lookup(b"malloc\0"), Ordering::Release);
    REAL_FREE.store(lookup(b"free\0"), Ordering::Release);
    REAL_CALLOC.store(lookup(b"calloc\0"), Ordering::Release);
    REAL_REALLOC.store(lookup(b"realloc\0"), Ordering::Release);
    REAL_POSIX_MEMALIGN.store(lookup(b"posix_memalign\0"), Ordering::Release);
    REAL_ALIGNED_ALLOC.store(lookup(b"aligned_alloc\0"), Ordering::Release);
    REAL_MEMALIGN.store(lookup(b"memalign\0"), Ordering::Release);
    REAL_USABLE.store(lookup(b"malloc_usable_size\0"), Ordering::Release);
    REAL_MADVISE.store(lookup(b"madvise\0"), Ordering::Release);
    RESOLVING.with(|r| r.set(false));
}

fn real(slot: &AtomicPtr<c_void>) -> *mut c_void {
    let mut f = slot.load(Ordering::Acquire);
    if f.is_null() && !RESOLVING.with(|r| r.get()) {
        resolve();
        f = slot.load(Ordering::Acquire);
    }
    f
}

fn arena_alloc(size: usize) -> *mut c_void {
    let size = size.div_ceil(16) * 16;
    let start = ARENA_USED.fetch_add(size, Ordering::AcqRel);
    if start + size > ARENA_BYTES {
        return std::ptr::null_mut();
    }
    // SAFETY: a fresh piece inside the arena.
    unsafe { ARENA.0.get().cast::<u8>().add(start).cast() }
}

fn in_arena(p: *mut c_void) -> bool {
    let base = ARENA.0.get() as usize;
    (base..base + ARENA_BYTES).contains(&(p as usize))
}

unsafe fn real_malloc(size: size_t) -> *mut c_void {
    let f = real(&REAL_MALLOC);
    if f.is_null() {
        return arena_alloc(size);
    }
    // SAFETY: libc's malloc.
    unsafe { std::mem::transmute::<*mut c_void, MallocFn>(f)(size) }
}

unsafe fn real_free(p: *mut c_void) {
    if p.is_null() || in_arena(p) {
        return;
    }
    let f = real(&REAL_FREE);
    if !f.is_null() {
        // SAFETY: libc's free of libc's memory.
        unsafe { std::mem::transmute::<*mut c_void, FreeFn>(f)(p) }
    }
}

unsafe fn real_calloc(n: size_t, m: size_t) -> *mut c_void {
    let f = real(&REAL_CALLOC);
    if f.is_null() {
        return n.checked_mul(m).map_or(std::ptr::null_mut(), arena_alloc); // zeroed static memory
    }
    // SAFETY: libc's calloc.
    unsafe { std::mem::transmute::<*mut c_void, CallocFn>(f)(n, m) }
}

unsafe fn real_realloc(p: *mut c_void, size: size_t) -> *mut c_void {
    // SAFETY: libc's realloc of libc's memory.
    unsafe { std::mem::transmute::<*mut c_void, ReallocFn>(real(&REAL_REALLOC))(p, size) }
}

unsafe fn real_posix_memalign(out: *mut *mut c_void, align: size_t, size: size_t) -> c_int {
    // SAFETY: libc's posix_memalign.
    unsafe {
        std::mem::transmute::<*mut c_void, PosixMemalignFn>(real(&REAL_POSIX_MEMALIGN))(
            out, align, size,
        )
    }
}

fn page() -> usize {
    let p = PAGE.load(Ordering::Relaxed);
    if p != 0 {
        return p;
    }
    // SAFETY: a plain query.
    let p = unsafe { libc::sysconf(libc::_SC_PAGESIZE) }.max(4096) as usize;
    PAGE.store(p, Ordering::Relaxed);
    p
}

fn env_u64(name: &[u8]) -> Option<u64> {
    // SAFETY: a NUL-terminated name; getenv returns a C string or null and allocates nothing.
    let v = unsafe { libc::getenv(name.as_ptr().cast()) };
    if v.is_null() {
        return None;
    }
    // SAFETY: getenv's string stays valid while the environment is not changed.
    unsafe { std::ffi::CStr::from_ptr(v) }
        .to_str()
        .ok()?
        .trim()
        .parse()
        .ok()
}

struct Busy;

impl Busy {
    fn enter() -> Busy {
        BUSY.with(|b| b.set(b.get() + 1));
        Busy
    }
}

impl Drop for Busy {
    fn drop(&mut self) {
        BUSY.with(|b| b.set(b.get() - 1));
    }
}

/// The pager, set up at the first large allocation; `None` while it starts, when it could not
/// start, inside our own work and on the pager's thread.
fn pager() -> Option<&'static Pager> {
    if BUSY.with(|b| b.get()) > 0 || in_pager_thread() {
        return None;
    }
    match STATE.load(Ordering::Acquire) {
        READY => {
            // SAFETY: published once below and never freed.
            return Some(unsafe { &*PAGER.load(Ordering::Acquire) });
        }
        UNSET => {}
        _ => return None,
    }
    if STATE
        .compare_exchange(UNSET, STARTING, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return None; // another thread is starting it: system memory meanwhile
    }
    let _busy = Busy::enter();
    let Some(budget) = env_u64(b"MEMOPRO_PRELOAD_BUDGET\0") else {
        STATE.store(OFF, Ordering::Release);
        return None;
    };
    if let Some(t) = env_u64(b"MEMOPRO_PRELOAD_THRESHOLD\0") {
        THRESHOLD.store((t as usize).max(page()), Ordering::Relaxed);
    }
    // SAFETY: as in env_u64.
    REPORT.store(
        unsafe { libc::getenv(c"MEMOPRO_PRELOAD_REPORT".as_ptr()) },
        Ordering::Relaxed,
    );
    let config = PagerConfig {
        process_budget: env_u64(b"MEMOPRO_PRELOAD_PROCESS\0"),
        ..PagerConfig::new(budget)
    };
    match Pager::new(config) {
        Ok(p) => {
            PAGER.store(Box::into_raw(Box::new(p)), Ordering::Release);
            // SAFETY: registering a plain function to run at exit.
            unsafe { libc::atexit(report) };
            let _ = STARTED.set(std::time::Instant::now());
            // a run killed at a time limit leaves no exit report: rewrite it every few seconds
            if let Some(every) = env_u64(b"MEMOPRO_PRELOAD_REPORT_EVERY\0").filter(|s| *s > 0) {
                let _ = std::thread::Builder::new()
                    .name("memopro-report".into())
                    .spawn(move || {
                        loop {
                            std::thread::sleep(std::time::Duration::from_secs(every));
                            report();
                        }
                    });
            }
            STATE.store(READY, Ordering::Release);
            // SAFETY: just published.
            Some(unsafe { &*PAGER.load(Ordering::Acquire) })
        }
        Err(e) => {
            eprintln!("memopro-preload: not paging ({e})");
            STATE.store(OFF, Ordering::Release);
            None
        }
    }
}

/// A pager region for a large allocation, or null to use the system allocator.
fn paged(size: usize) -> *mut c_void {
    if size < THRESHOLD.load(Ordering::Relaxed) {
        return std::ptr::null_mut();
    }
    let Some(p) = pager() else {
        return std::ptr::null_mut();
    };
    let _busy = Busy::enter();
    match p.map(size) {
        Ok(ptr) => {
            PAGED.fetch_add(1, Ordering::Relaxed);
            PAGED_BYTES.fetch_add(size as u64, Ordering::Relaxed);
            ptr.cast()
        }
        Err(_) => std::ptr::null_mut(),
    }
}

/// The length of the pager region at `p`, if it is one (regions start on a page).
fn region(p: *mut c_void) -> Option<usize> {
    if p.is_null() || (p as usize) % page() != 0 || STATE.load(Ordering::Acquire) != READY {
        return None;
    }
    if in_pager_thread() {
        return None; // the pager thread holds its lock and never frees our regions
    }
    // SAFETY: published once and never freed.
    let pager = unsafe { &*PAGER.load(Ordering::Acquire) };
    let _busy = Busy::enter();
    pager.owns(p.cast())
}

extern "C" fn report() {
    let path = REPORT.load(Ordering::Relaxed);
    if path.is_null() || STATE.load(Ordering::Acquire) != READY {
        return;
    }
    let _busy = Busy::enter();
    // SAFETY: published once and never freed.
    let s = unsafe { &*PAGER.load(Ordering::Acquire) }.stats();
    let json = format!(
        "{{\"budget\": {}, \"limit\": {}, \"limit_low\": {}, \"peak_used\": {}, \
         \"outside_peak\": {}, \"faults\": {}, \"zero_fills\": {}, \"restores\": {}, \
         \"restore_seconds\": {}, \"evictions\": {}, \"compress_in\": {}, \"compress_out\": {}, \
         \"compress_seconds\": {}, \"incompressible\": {}, \"overruns\": {}, \"spurious\": {}, \
         \"paged_allocations\": {}, \"paged_bytes\": {}, \"threshold\": {}, \"seconds\": {}}}\n",
        s.budget,
        s.limit,
        s.limit_low,
        s.peak_used,
        s.outside_peak,
        s.faults,
        s.zero_fills,
        s.restores,
        s.restore_seconds,
        s.evictions,
        s.compress_in,
        s.compress_out,
        s.compress_seconds,
        s.incompressible,
        s.overruns,
        s.spurious,
        PAGED.load(Ordering::Relaxed),
        PAGED_BYTES.load(Ordering::Relaxed),
        THRESHOLD.load(Ordering::Relaxed),
        STARTED.get().map_or(0.0, |t| t.elapsed().as_secs_f64()),
    );
    // SAFETY: getenv's string, still valid at exit.
    if let Ok(p) = unsafe { std::ffi::CStr::from_ptr(path) }.to_str() {
        let _ = std::fs::write(p, json);
    }
}

/// # Safety
/// The C `malloc` contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn malloc(size: size_t) -> *mut c_void {
    let p = paged(size);
    if !p.is_null() {
        return p;
    }
    // SAFETY: forwarding.
    unsafe { real_malloc(size) }
}

/// # Safety
/// The C `calloc` contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn calloc(n: size_t, m: size_t) -> *mut c_void {
    if let Some(size) = n.checked_mul(m) {
        let p = paged(size); // pager memory starts as zeros
        if !p.is_null() {
            return p;
        }
    }
    // SAFETY: forwarding.
    unsafe { real_calloc(n, m) }
}

/// # Safety
/// The C `free` contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn free(p: *mut c_void) {
    if region(p).is_some() {
        // SAFETY: published once and never freed; the caller gives the region up.
        let pager = unsafe { &*PAGER.load(Ordering::Acquire) };
        let _busy = Busy::enter();
        // SAFETY: one of this pager's regions, no longer used (the free contract).
        let _ = unsafe { pager.unmap(p.cast()) };
        return;
    }
    // SAFETY: forwarding.
    unsafe { real_free(p) }
}

/// # Safety
/// The C `realloc` contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn realloc(p: *mut c_void, size: size_t) -> *mut c_void {
    if p.is_null() {
        // SAFETY: as malloc.
        return unsafe { malloc(size) };
    }
    if let Some(len) = region(p) {
        if size <= len && size >= THRESHOLD.load(Ordering::Relaxed) {
            return p;
        }
        // SAFETY: as malloc.
        let new = unsafe { malloc(size.max(1)) };
        if !new.is_null() {
            // SAFETY: both blocks hold at least the bytes copied.
            unsafe { std::ptr::copy_nonoverlapping(p.cast::<u8>(), new.cast(), len.min(size)) };
            // SAFETY: as free.
            unsafe { free(p) };
        }
        return new;
    }
    if !in_arena(p) {
        let new = paged(size);
        if !new.is_null() {
            // SAFETY: libc's block holds its usable size; the new one holds `size`.
            unsafe {
                let old = std::mem::transmute::<*mut c_void, UsableFn>(real(&REAL_USABLE))(p);
                std::ptr::copy_nonoverlapping(p.cast::<u8>(), new.cast(), old.min(size));
                real_free(p);
            }
            return new;
        }
        // SAFETY: forwarding.
        return unsafe { real_realloc(p, size) };
    }
    // an arena block (from the lookups): copy out of it
    // SAFETY: as malloc; the arena block is at most the arena.
    let new = unsafe { real_malloc(size) };
    if !new.is_null() {
        let room = ARENA_BYTES - (p as usize - ARENA.0.get() as usize);
        // SAFETY: both ranges are valid for the bytes copied.
        unsafe { std::ptr::copy_nonoverlapping(p.cast::<u8>(), new.cast(), room.min(size)) };
    }
    new
}

fn aligned(align: size_t, size: size_t) -> *mut c_void {
    if align.is_power_of_two() && align <= page() {
        paged(size) // regions start on a page
    } else {
        std::ptr::null_mut()
    }
}

/// # Safety
/// The C `posix_memalign` contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn posix_memalign(
    out: *mut *mut c_void,
    align: size_t,
    size: size_t,
) -> c_int {
    let p = aligned(align, size);
    if !p.is_null() {
        // SAFETY: the caller's out pointer.
        unsafe { *out = p };
        return 0;
    }
    // SAFETY: forwarding.
    unsafe { real_posix_memalign(out, align, size) }
}

/// # Safety
/// The C `aligned_alloc` contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn aligned_alloc(align: size_t, size: size_t) -> *mut c_void {
    let p = aligned(align, size);
    if !p.is_null() {
        return p;
    }
    // SAFETY: libc's aligned_alloc.
    unsafe { std::mem::transmute::<*mut c_void, AlignedFn>(real(&REAL_ALIGNED_ALLOC))(align, size) }
}

/// # Safety
/// The C `memalign` contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn memalign(align: size_t, size: size_t) -> *mut c_void {
    let p = aligned(align, size);
    if !p.is_null() {
        return p;
    }
    // SAFETY: libc's memalign.
    unsafe { std::mem::transmute::<*mut c_void, AlignedFn>(real(&REAL_MEMALIGN))(align, size) }
}

/// # Safety
/// The C `malloc_usable_size` contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn malloc_usable_size(p: *mut c_void) -> size_t {
    if let Some(len) = region(p) {
        return len;
    }
    if p.is_null() || in_arena(p) {
        return 0;
    }
    // SAFETY: libc's malloc_usable_size of libc's block.
    unsafe { std::mem::transmute::<*mut c_void, UsableFn>(real(&REAL_USABLE))(p) }
}

/// # Safety
/// The C `madvise` contract. Huge pages are refused for pager regions: a transparent huge page
/// fault in a userfaultfd range never reached the pager and the toucher spun (NumPy advises
/// `MADV_HUGEPAGE` for large arrays from `malloc`; 0191).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn madvise(addr: *mut c_void, len: size_t, advice: c_int) -> c_int {
    if advice == libc::MADV_HUGEPAGE
        && STATE.load(Ordering::Acquire) == READY
        && !in_pager_thread()
        && BUSY.with(|b| b.get()) == 0
    {
        // SAFETY: published once and never freed.
        let pager = unsafe { &*PAGER.load(Ordering::Acquire) };
        let _busy = Busy::enter();
        if pager.contains(addr.cast()) {
            return 0;
        }
    }
    // SAFETY: libc's madvise.
    unsafe { std::mem::transmute::<*mut c_void, MadviseFn>(real(&REAL_MADVISE))(addr, len, advice) }
}
