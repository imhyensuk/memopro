//! macOS implementation of [`Pager`] (0229): page faults arrive as signals.
//!
//! macOS has no userfaultfd. Each pager reserves one large address range (the arena) with no
//! access and hands out regions from it. A chunk that is not in memory has no access; touching
//! it raises SIGBUS (or SIGSEGV). The signal handler does no paging itself: it reads the chunk's
//! state (an atomic per chunk), asks the pager thread for the chunk by writing its index to a
//! pipe, and sleeps until the state says the chunk is back. Everything else (choosing,
//! compressing, decompressing) runs on the pager thread, outside any signal handler.
//!
//! A chunk comes back whole: it is decompressed into a fresh mapping that is then moved into
//! place with `mach_vm_remap(VM_FLAGS_OVERWRITE)`, so no thread sees it half filled. Eviction
//! write-protects the chunk (readers go on, writers wait), compresses it, then replaces it with
//! a fresh mapping without access, which gives its pages back to the OS at once.
//!
//! Faults of other origin (addresses outside every arena, freed regions) go to the handler that
//! was installed before ours. Kernel accesses to a chunk that is not in memory (`read(2)` into
//! it, `write(2)` from it) are not faults but fail with `EFAULT`, as with Linux user-mode-only
//! userfaultfd.

use super::{PagerConfig, PagerStats};
use crate::codec;
use crate::error::{Error, Result};
use crate::rt::page_size;
use std::collections::{BTreeMap, HashMap};
use std::io;
use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd, OwnedFd, RawFd};
use std::ptr::null_mut;
use std::sync::atomic::{AtomicPtr, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::thread::JoinHandle;
use std::time::Instant;

/// At most this many most recently faulted chunks count as in use and are not evicted.
const HOT_CHUNKS: u64 = 16;
/// Chunks the pager always keeps room for in memory, whatever the budget, a process budget and
/// the compressed copies leave: with room for fewer, the two operands of one copy evicted each
/// other forever (0230). Going over the budget to keep them is counted in `overruns`.
const MIN_CHUNKS: u64 = 8;
/// Address space each pager reserves for its regions (no memory until used).
const ARENA: usize = 64 << 30;
/// Pagers that can exist at once in a process.
const SLOTS: usize = 64;

// chunk states, read by the signal handler
const FREE: u8 = 0; // not part of a region
const ABSENT: u8 = 1; // never touched: no access, comes back as zeros
const OUT: u8 = 2; // compressed: no access
const EVICTING: u8 = 3; // write-protected while it is compressed
const RESIDENT: u8 = 4; // readable and writable

unsafe extern "C" {
    fn mach_vm_remap(
        target_task: libc::mach_port_t,
        target_address: *mut u64,
        size: u64,
        mask: u64,
        flags: libc::c_int,
        src_task: libc::mach_port_t,
        src_address: u64,
        copy: libc::c_int,
        cur_protection: *mut libc::c_int,
        max_protection: *mut libc::c_int,
        inheritance: libc::c_uint,
    ) -> libc::c_int;
}
// <mach/vm_statistics.h>, <mach/vm_inherit.h>
const VM_FLAGS_FIXED: libc::c_int = 0;
const VM_FLAGS_OVERWRITE: libc::c_int = 0x4000;
const VM_INHERIT_COPY: libc::c_uint = 1;

unsafe extern "C" {
    static mach_task_self_: libc::mach_port_t;
}

fn task() -> libc::mach_port_t {
    // SAFETY: set by the system before any code runs; never changes.
    unsafe { mach_task_self_ }
}

thread_local! {
    static PAGER_THREAD: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether the calling thread is a pager's own.
pub fn in_pager_thread() -> bool {
    PAGER_THREAD.with(|c| c.get())
}

/// What the signal handler needs of a pager. Never freed (a handler may still read it after the
/// pager is gone; its states then say FREE).
struct Arena {
    base: usize,
    len: usize,
    chunk: usize,
    states: Box<[AtomicU8]>,
    /// Write end of the request pipe (non-blocking).
    request: RawFd,
    /// Requests served so far (a waiting handler asks again once it moved on).
    served: AtomicU64,
}

impl Arena {
    fn index(&self, addr: usize) -> Option<usize> {
        (addr >= self.base && addr < self.base + self.len).then(|| (addr - self.base) / self.chunk)
    }

    /// Wait in a signal handler until chunk `i` is accessible; false if it is not ours.
    fn wait(&self, i: usize) -> bool {
        let mut asked: Option<u64> = None;
        loop {
            match self.states[i].load(Ordering::Acquire) {
                RESIDENT => return true,
                FREE => return false,
                EVICTING => {}
                _ => {
                    let served = self.served.load(Ordering::Acquire);
                    if asked.is_none_or(|a| served > a) {
                        let msg = i as u64;
                        // SAFETY: write(2) is async-signal-safe; 8 bytes from a local.
                        unsafe {
                            libc::write(self.request, (&msg as *const u64).cast(), 8);
                        }
                        asked = Some(served);
                    }
                }
            }
            let ts = libc::timespec {
                tv_sec: 0,
                tv_nsec: 20_000,
            };
            // SAFETY: nanosleep(2) is async-signal-safe.
            unsafe { libc::nanosleep(&ts, null_mut()) };
        }
    }
}

static ARENAS: [AtomicPtr<Arena>; SLOTS] = [const { AtomicPtr::new(null_mut()) }; SLOTS];
/// The handlers installed before ours: SIGBUS, SIGSEGV.
static PREVIOUS: OnceLock<[libc::sigaction; 2]> = OnceLock::new();

extern "C" fn on_fault(sig: libc::c_int, info: *mut libc::siginfo_t, ctx: *mut libc::c_void) {
    // SAFETY: the kernel passes a valid siginfo for SA_SIGINFO handlers.
    let addr = unsafe { (*info).si_addr } as usize;
    for slot in &ARENAS {
        let a = slot.load(Ordering::Acquire);
        if a.is_null() {
            continue;
        }
        // SAFETY: arenas are never freed.
        let arena = unsafe { &*a };
        if let Some(i) = arena.index(addr) {
            if arena.wait(i) {
                return; // the access runs again and finds the chunk
            }
            break;
        }
    }
    chain(sig, info, ctx);
}

/// Hand a fault that is not ours to the handler installed before ours.
fn chain(sig: libc::c_int, info: *mut libc::siginfo_t, ctx: *mut libc::c_void) {
    let Some(previous) = PREVIOUS.get() else {
        return;
    };
    let old = &previous[if sig == libc::SIGBUS { 0 } else { 1 }];
    let handler = old.sa_sigaction;
    if handler == libc::SIG_DFL || handler == libc::SIG_IGN {
        // the default action happens when the access runs again
        // SAFETY: sigaction(2) is async-signal-safe.
        unsafe { libc::sigaction(sig, old, null_mut()) };
    } else if old.sa_flags & libc::SA_SIGINFO != 0 {
        // SAFETY: the previous handler, called the way it was installed.
        let f: extern "C" fn(libc::c_int, *mut libc::siginfo_t, *mut libc::c_void) =
            unsafe { std::mem::transmute(handler) };
        f(sig, info, ctx);
    } else {
        // SAFETY: as above.
        let f: extern "C" fn(libc::c_int) = unsafe { std::mem::transmute(handler) };
        f(sig);
    }
}

fn install_handler() -> Result<()> {
    static DONE: OnceLock<std::result::Result<(), String>> = OnceLock::new();
    DONE.get_or_init(|| {
        // SAFETY: plain sigaction calls with zeroed, then filled, structures.
        unsafe {
            let mut old: [libc::sigaction; 2] = std::mem::zeroed();
            let mut new: libc::sigaction = std::mem::zeroed();
            new.sa_sigaction = on_fault as *const () as usize;
            new.sa_flags = libc::SA_SIGINFO | libc::SA_ONSTACK;
            libc::sigemptyset(&mut new.sa_mask);
            if libc::sigaction(libc::SIGBUS, null_mut(), &mut old[0]) != 0
                || libc::sigaction(libc::SIGSEGV, null_mut(), &mut old[1]) != 0
            {
                return Err(io::Error::last_os_error().to_string());
            }
            let _ = PREVIOUS.set(old);
            if libc::sigaction(libc::SIGBUS, &new, null_mut()) != 0
                || libc::sigaction(libc::SIGSEGV, &new, null_mut()) != 0
            {
                return Err(io::Error::last_os_error().to_string());
            }
        }
        Ok(())
    })
    .clone()
    .map_err(|e| Error::Unsupported(format!("cannot install the page fault handler: {e}")))
}

fn pipe() -> io::Result<(OwnedFd, OwnedFd)> {
    let mut fds = [0 as RawFd; 2];
    // SAFETY: pipe(2) fills two descriptors we then own.
    if unsafe { libc::pipe(fds.as_mut_ptr()) } != 0 {
        return Err(io::Error::last_os_error());
    }
    for fd in fds {
        // SAFETY: fcntl on descriptors we own.
        unsafe {
            libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC);
            libc::fcntl(fd, libc::F_SETFL, libc::O_NONBLOCK);
        }
    }
    // SAFETY: fresh descriptors.
    Ok(unsafe { (OwnedFd::from_raw_fd(fds[0]), OwnedFd::from_raw_fd(fds[1])) })
}

/// Replace `len` bytes at `addr` with a fresh mapping that has no access (pages go back).
fn blank(addr: usize, len: usize) -> io::Result<()> {
    // SAFETY: `addr..addr+len` lies in our arena; MAP_FIXED replaces only that range.
    let p = unsafe {
        libc::mmap(
            addr as *mut libc::c_void,
            len,
            libc::PROT_NONE,
            libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_FIXED,
            -1,
            0,
        )
    };
    if p == libc::MAP_FAILED {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

fn protect(addr: usize, len: usize, prot: libc::c_int) -> io::Result<()> {
    // SAFETY: a range of our arena.
    if unsafe { libc::mprotect(addr as *mut libc::c_void, len, prot) } != 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

enum Data {
    Absent,
    Resident,
    Compressed(Vec<u8>),
}

struct Chunk {
    data: Data,
    last: u64,
    period: f64,
    faults: u64,
    incompressible: bool,
}

struct State {
    /// First chunk index of each region -> its chunk count.
    regions: BTreeMap<usize, usize>,
    /// Free runs of chunks: first index -> count.
    free: BTreeMap<usize, usize>,
    chunks: HashMap<usize, Chunk>,
    clock: u64,
    resident: u64,
    compressed: u64,
    stats: PagerStats,
    packed: Vec<u8>,
}

struct Shared {
    config: PagerConfig,
    limit: u64,
    arena: &'static Arena,
    slot: usize,
    requests: OwnedFd,
    stop_r: OwnedFd,
    stop_w: OwnedFd,
    state: Mutex<State>,
}

/// Transparent paging under a budget; see the [module documentation](super).
pub struct Pager {
    shared: Arc<Shared>,
    thread: Option<JoinHandle<()>>,
}

impl Pager {
    pub fn new(config: PagerConfig) -> Result<Pager> {
        let page = page_size();
        let chunk = config.chunk;
        if !matches!(config.elem, 1 | 2 | 4 | 8 | 16)
            || chunk == 0
            || chunk % page != 0
            || chunk > codec::CHUNK
            || chunk % config.elem != 0
        {
            return Err(Error::InvalidArgument(format!(
                "chunk {chunk} must be a multiple of the page size ({page}) and of the element \
                 size {}, at most {}",
                config.elem,
                codec::CHUNK
            )));
        }
        let limit = config
            .budget
            .checked_sub(chunk as u64)
            .filter(|l| *l >= chunk as u64)
            .ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "a budget of {} bytes cannot hold two chunks of {chunk}",
                    config.budget
                ))
            })?;
        install_handler()?;
        let slot = ARENAS
            .iter()
            .position(|s| s.load(Ordering::Acquire).is_null())
            .ok_or_else(|| Error::Unsupported(format!("at most {SLOTS} pagers at once")))?;
        let mut len = ARENA - ARENA % chunk;
        let base = loop {
            // SAFETY: a new reservation without access; nothing refers to it.
            let p = unsafe {
                libc::mmap(
                    null_mut(),
                    len,
                    libc::PROT_NONE,
                    libc::MAP_PRIVATE | libc::MAP_ANON,
                    -1,
                    0,
                )
            };
            if p != libc::MAP_FAILED {
                break p as usize;
            }
            if len <= 1 << 30 {
                return Err(Error::Io(io::Error::last_os_error()));
            }
            len /= 2;
            len -= len % chunk;
        };
        let (requests, request_w) = pipe().map_err(Error::Io)?;
        let (stop_r, stop_w) = pipe().map_err(Error::Io)?;
        let n = len / chunk;
        let arena: &'static Arena = Box::leak(Box::new(Arena {
            base,
            len,
            chunk,
            states: (0..n).map(|_| AtomicU8::new(FREE)).collect(),
            // the handler may write to it after the pager is gone: never closed
            request: request_w.into_raw_fd(),
            served: AtomicU64::new(0),
        }));
        if ARENAS[slot]
            .compare_exchange(
                null_mut(),
                (arena as *const Arena).cast_mut(),
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            // SAFETY: our reservation, not handed out.
            unsafe { libc::munmap(base as *mut libc::c_void, len) };
            return Err(Error::Unsupported(
                "pager slots were taken concurrently".into(),
            ));
        }
        let stats = PagerStats {
            budget: config.budget,
            limit,
            limit_low: limit,
            ..PagerStats::default()
        };
        let shared = Arc::new(Shared {
            config,
            limit,
            arena,
            slot,
            requests,
            stop_r,
            stop_w,
            state: Mutex::new(State {
                regions: BTreeMap::new(),
                free: BTreeMap::from([(0, n)]),
                chunks: HashMap::new(),
                clock: 0,
                resident: 0,
                compressed: 0,
                stats,
                packed: Vec::new(),
            }),
        });
        let serving = shared.clone();
        let thread = std::thread::Builder::new()
            .name("memopro-pager".into())
            .spawn(move || serving.serve())
            .map_err(Error::Io)?;
        Ok(Pager {
            shared,
            thread: Some(thread),
        })
    }

    /// A zero-filled region of at least `len` bytes (whole chunks), paged within the budget.
    /// It stays valid until [`Pager::unmap`].
    pub fn map(&self, len: usize) -> Result<*mut u8> {
        if len == 0 {
            return Err(Error::InvalidArgument(
                "a region holds at least one byte".into(),
            ));
        }
        let arena = self.shared.arena;
        let n = len.div_ceil(arena.chunk);
        let mut st = self.shared.lock();
        let Some((&start, &run)) = st.free.iter().find(|(_, run)| **run >= n) else {
            return Err(Error::Budget(format!(
                "the pager's {} GiB of address space is used up",
                arena.len >> 30
            )));
        };
        st.free.remove(&start);
        if run > n {
            st.free.insert(start + n, run - n);
        }
        st.regions.insert(start, n);
        for i in start..start + n {
            st.chunks.insert(
                i,
                Chunk {
                    data: Data::Absent,
                    last: 0,
                    period: 0.0,
                    faults: 0,
                    incompressible: false,
                },
            );
            arena.states[i].store(ABSENT, Ordering::Release);
        }
        st.stats.regions += 1;
        st.stats.mapped_bytes += (n * arena.chunk) as u64;
        Ok((arena.base + start * arena.chunk) as *mut u8)
    }

    /// Give a region back; returns its length.
    ///
    /// # Safety
    /// `addr` came from [`Pager::map`] of this pager, and no one uses the region any more.
    pub unsafe fn unmap(&self, addr: *mut u8) -> Result<usize> {
        let arena = self.shared.arena;
        let chunk = arena.chunk;
        let not_ours = || Error::InvalidArgument(format!("{addr:p} is not a region of this pager"));
        let a = addr as usize;
        let start = arena
            .index(a)
            .filter(|i| arena.base + i * chunk == a)
            .ok_or_else(not_ours)?;
        let mut st = self.shared.lock();
        let n = st.regions.remove(&start).ok_or_else(not_ours)?;
        for i in start..start + n {
            arena.states[i].store(FREE, Ordering::Release);
            match st.chunks.remove(&i).map(|c| c.data) {
                Some(Data::Resident) => st.resident -= chunk as u64,
                Some(Data::Compressed(p)) => st.compressed -= p.len() as u64,
                _ => {}
            }
        }
        blank(a, n * chunk).map_err(Error::Io)?;
        // give the run back, merged with free neighbours
        let (mut first, mut count) = (start, n);
        if let Some((&prev, &len)) = st.free.range(..start).next_back()
            && prev + len == start
        {
            st.free.remove(&prev);
            first = prev;
            count += len;
        }
        if let Some(len) = st.free.remove(&(start + n)) {
            count += len;
        }
        st.free.insert(first, count);
        st.stats.regions -= 1;
        st.stats.mapped_bytes -= (n * chunk) as u64;
        Ok(n * chunk)
    }

    /// The length of the region starting at `addr`, if it is one of ours.
    pub fn owns(&self, addr: *const u8) -> Option<usize> {
        let arena = self.shared.arena;
        let a = addr as usize;
        let i = arena
            .index(a)
            .filter(|i| arena.base + i * arena.chunk == a)?;
        self.shared.lock().regions.get(&i).map(|n| n * arena.chunk)
    }

    /// Whether `addr` lies inside one of this pager's regions.
    pub fn contains(&self, addr: *const u8) -> bool {
        let arena = self.shared.arena;
        arena
            .index(addr as usize)
            .is_some_and(|i| arena.states[i].load(Ordering::Acquire) != FREE)
    }

    pub fn limit(&self) -> u64 {
        self.shared.limit
    }

    pub fn stats(&self) -> PagerStats {
        let st = self.shared.lock();
        let mut s = st.stats.clone();
        s.resident_bytes = st.resident;
        s.compressed_bytes = st.compressed;
        s.used = st.resident + st.compressed;
        s
    }
}

impl Drop for Pager {
    fn drop(&mut self) {
        let one = 1u8;
        // SAFETY: one byte to our stop pipe wakes the pager thread.
        unsafe {
            libc::write(
                self.shared.stop_w.as_raw_fd(),
                (&one as *const u8).cast(),
                1,
            )
        };
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        let arena = self.shared.arena;
        ARENAS[self.shared.slot].store(null_mut(), Ordering::Release);
        for s in arena.states.iter() {
            s.store(FREE, Ordering::Release);
        }
        // SAFETY: our reservation; users must be done with its regions (as with Linux).
        unsafe { libc::munmap(arena.base as *mut libc::c_void, arena.len) };
    }
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The pager thread: serve requests until stopped.
    fn serve(&self) {
        PAGER_THREAD.with(|c| c.set(true));
        let mut msgs = [0u64; 64];
        loop {
            let mut fds = [
                libc::pollfd {
                    fd: self.requests.as_raw_fd(),
                    events: libc::POLLIN,
                    revents: 0,
                },
                libc::pollfd {
                    fd: self.stop_r.as_raw_fd(),
                    events: libc::POLLIN,
                    revents: 0,
                },
            ];
            // SAFETY: two valid pollfd entries.
            if unsafe { libc::poll(fds.as_mut_ptr(), 2, -1) } < 0 {
                continue; // EINTR
            }
            if fds[1].revents != 0 {
                return;
            }
            // SAFETY: reading whole u64 messages into a buffer of that many bytes.
            let n = unsafe {
                libc::read(
                    self.requests.as_raw_fd(),
                    msgs.as_mut_ptr().cast(),
                    std::mem::size_of_val(&msgs),
                )
            };
            if n <= 0 {
                continue;
            }
            let mut st = self.lock();
            for &i in &msgs[..n as usize / 8] {
                self.fault(&mut st, i as usize);
            }
            drop(st);
            self.arena.served.fetch_add(1, Ordering::AcqRel);
        }
    }

    fn fault(&self, st: &mut State, i: usize) {
        st.stats.faults += 1;
        let chunk = self.config.chunk;
        let cstart = self.arena.base + i * chunk;
        st.clock += 1;
        let now = st.clock;
        match st.chunks.get(&i).map(|c| &c.data) {
            None => return, // freed meanwhile: the handler sees FREE
            Some(Data::Resident) => {
                st.stats.spurious += 1;
                return;
            }
            _ => {}
        }
        self.make_room(st, chunk as u64, i);
        let t0 = Instant::now();
        let c = st.chunks.get_mut(&i).expect("checked above");
        let was = std::mem::replace(&mut c.data, Data::Resident);
        if c.faults > 0 {
            let gap = (now - c.last) as f64;
            c.period = if c.period == 0.0 {
                gap
            } else {
                0.5 * c.period + 0.5 * gap
            };
        }
        c.last = now;
        c.faults += 1;
        match was {
            Data::Absent => {
                // fresh pages are zeros, which is what the chunk holds: no one can see it
                // half filled
                if let Err(e) = protect(cstart, chunk, libc::PROT_READ | libc::PROT_WRITE) {
                    eprintln!("memopro pager: a chunk could not be made accessible ({e})");
                    std::process::abort();
                }
                st.stats.zero_fills += 1;
            }
            Data::Compressed(p) => {
                if let Err(e) = self.restore(cstart, &p) {
                    // the only copy of this memory cannot come back: stopping is the only
                    // honest thing
                    eprintln!("memopro pager: a compressed chunk could not be restored ({e})");
                    std::process::abort();
                }
                st.compressed -= p.len() as u64;
                st.stats.restores += 1;
            }
            Data::Resident => {}
        }
        self.arena.states[i].store(RESIDENT, Ordering::Release);
        st.resident += chunk as u64;
        let used = st.resident + st.compressed;
        st.stats.peak_used = st.stats.peak_used.max(used);
        st.stats.restore_seconds += t0.elapsed().as_secs_f64();
    }

    /// Decompress into a fresh mapping and move it over the chunk in one step.
    fn restore(&self, cstart: usize, packed: &[u8]) -> Result<()> {
        let chunk = self.config.chunk;
        // SAFETY: a new private mapping only this thread uses.
        let tmp = unsafe {
            libc::mmap(
                null_mut(),
                chunk,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        if tmp == libc::MAP_FAILED {
            return Err(Error::Io(io::Error::last_os_error()));
        }
        // SAFETY: the mapping made above, `chunk` bytes.
        let out = unsafe { std::slice::from_raw_parts_mut(tmp.cast::<u8>(), chunk) };
        let unpacked = codec::unpack_chunk(packed, self.config.elem, out).map_err(Error::Io);
        let moved = unpacked.and_then(|()| {
            let (mut target, mut cur, mut max) = (cstart as u64, 0, 0);
            // SAFETY: moves our own pages over our own chunk (both in this task).
            let kr = unsafe {
                mach_vm_remap(
                    task(),
                    &mut target,
                    chunk as u64,
                    0,
                    VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE,
                    task(),
                    tmp as u64,
                    0,
                    &mut cur,
                    &mut max,
                    VM_INHERIT_COPY,
                )
            };
            if kr == 0 && target == cstart as u64 {
                Ok(())
            } else {
                Err(Error::Io(io::Error::other(format!("mach_vm_remap: {kr}"))))
            }
        });
        // SAFETY: the temporary mapping; the chunk keeps its pages.
        unsafe { libc::munmap(tmp, chunk) };
        moved
    }

    fn make_room(&self, st: &mut State, need: u64, exclude: usize) {
        let mut limit = self.limit;
        if let Some(total) = self.config.process_budget {
            let outside = crate::rt::process_footprint()
                .unwrap_or(0)
                .saturating_sub(st.resident + st.compressed);
            limit = limit.min(total.saturating_sub(outside + self.config.chunk as u64));
            st.stats.outside_peak = st.stats.outside_peak.max(outside);
            st.stats.limit_low = st.stats.limit_low.min(limit);
        }
        let hard = limit;
        let limit = limit.max(st.compressed + MIN_CHUNKS * self.config.chunk as u64);
        while st.resident + st.compressed + need > limit {
            let fits = (limit - st.compressed) / self.config.chunk as u64;
            let Some(i) = self.choose(st, exclude, fits) else {
                break;
            };
            self.evict(st, i);
        }
        if st.resident + st.compressed + need > hard {
            st.stats.overruns += 1;
        }
    }

    /// As on Linux: the resident chunk whose next fault is predicted farthest, never one of the
    /// most recently faulted (code works on them now); if only those are left, the oldest.
    fn choose(&self, st: &State, exclude: usize, fits: u64) -> Option<usize> {
        let now = st.clock as f64;
        let hot = (fits / 4).clamp(2, HOT_CHUNKS) as f64;
        let (sum, count) = st
            .chunks
            .values()
            .filter(|c| c.period > 0.0)
            .fold((0.0, 0u64), |(s, n), c| (s + c.period, n + 1));
        let typical = if count > 0 { sum / count as f64 } else { 0.0 };
        let mut best: Option<(f64, usize)> = None;
        let mut oldest_hot: Option<(f64, usize)> = None;
        for (&i, c) in &st.chunks {
            if !matches!(c.data, Data::Resident) || c.incompressible || i == exclude {
                continue;
            }
            let age = now - c.last as f64;
            if age < hot {
                if oldest_hot.is_none_or(|(a, _)| age > a) {
                    oldest_hot = Some((age, i));
                }
                continue;
            }
            let period = if c.period > 0.0 { c.period } else { typical };
            let distance = if period > 0.0 {
                (c.last as f64 + period - now).max(0.0) + 1.0
            } else {
                age + 1.0
            };
            if best.is_none_or(|(d, _)| distance > d) {
                best = Some((distance, i));
            }
        }
        best.or(oldest_hot).map(|(_, i)| i)
    }

    fn evict(&self, st: &mut State, i: usize) {
        let chunk = self.config.chunk;
        let cstart = self.arena.base + i * chunk;
        let t0 = Instant::now();
        // from here on writers wait (the handler sees EVICTING); readers go on
        self.arena.states[i].store(EVICTING, Ordering::Release);
        let protected = protect(cstart, chunk, libc::PROT_READ).is_ok();
        let mut packed = std::mem::take(&mut st.packed);
        let saved = protected && {
            // SAFETY: the chunk is resident and write-protected: its bytes cannot change while
            // they are read, and reading them does not fault.
            let src = unsafe { std::slice::from_raw_parts(cstart as *const u8, chunk) };
            codec::pack_chunk(
                src,
                self.config.elem,
                self.config.compress_level,
                &mut packed,
            )
            .is_ok()
                && (packed.len() as f64) < chunk as f64 * (1.0 - self.config.min_saving)
        };
        st.stats.compress_seconds += t0.elapsed().as_secs_f64();
        let stored = if saved { Some(packed.clone()) } else { None };
        let blanked = stored.is_some() && blank(cstart, chunk).is_ok();
        st.packed = packed;
        let c = st.chunks.get_mut(&i).expect("chosen from the table");
        if !blanked {
            if protected {
                let _ = protect(cstart, chunk, libc::PROT_READ | libc::PROT_WRITE);
            }
            self.arena.states[i].store(RESIDENT, Ordering::Release);
            c.incompressible = true;
            st.stats.incompressible += 1;
            return;
        }
        let stored = stored.expect("blanked only when saved");
        let n = stored.len() as u64;
        c.data = Data::Compressed(stored);
        self.arena.states[i].store(OUT, Ordering::Release);
        st.resident -= chunk as u64;
        st.compressed += n;
        st.stats.evictions += 1;
        st.stats.compress_in += chunk as u64;
        st.stats.compress_out += n;
    }
}
