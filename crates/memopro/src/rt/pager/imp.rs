//! Linux implementation of [`Pager`]: userfaultfd with write-protection (kernel 5.7+).

use super::{PagerConfig, PagerStats};
use crate::codec;
use crate::error::{Error, Result};
use crate::rt::page_size;
use std::collections::BTreeMap;
use std::io;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::Instant;

// <linux/userfaultfd.h>
const UFFD_API: u64 = 0xAA;
const UFFD_USER_MODE_ONLY: libc::c_int = 1;
const UFFDIO_API: u64 = 0xC018_AA3F;
const UFFDIO_REGISTER: u64 = 0xC020_AA00;
const UFFDIO_UNREGISTER: u64 = 0x8010_AA01;
const UFFDIO_WAKE: u64 = 0x8010_AA02;
const UFFDIO_COPY: u64 = 0xC028_AA03;
const UFFDIO_WRITEPROTECT: u64 = 0xC018_AA06;
const UFFDIO_REGISTER_MODE_MISSING: u64 = 1 << 0;
const UFFDIO_REGISTER_MODE_WP: u64 = 1 << 1;
const UFFDIO_WRITEPROTECT_MODE_WP: u64 = 1 << 0;
const UFFD_EVENT_PAGEFAULT: u8 = 0x12;
const UFFD_PAGEFAULT_FLAG_WP: u64 = 1 << 1;
const UFFD_FEATURE_PAGEFAULT_FLAG_WP: u64 = 1 << 0;
/// Bits of `_UFFDIO_COPY` and `_UFFDIO_WRITEPROTECT` in the `ioctls` a registration reports.
const HAS_COPY: u64 = 1 << 0x03;
const HAS_WRITEPROTECT: u64 = 1 << 0x06;
/// At most this many most recently faulted chunks count as in use and are not evicted.
const HOT_CHUNKS: u64 = 16;

#[repr(C)]
struct UffdioApi {
    api: u64,
    features: u64,
    ioctls: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct UffdioRange {
    start: u64,
    len: u64,
}

#[repr(C)]
struct UffdioRegister {
    range: UffdioRange,
    mode: u64,
    ioctls: u64,
}

#[repr(C)]
struct UffdioCopy {
    dst: u64,
    src: u64,
    len: u64,
    mode: u64,
    copy: i64,
}

#[repr(C)]
struct UffdioWriteprotect {
    range: UffdioRange,
    mode: u64,
}

/// `struct uffd_msg` as it arrives for a page fault (32 bytes, packed).
#[repr(C, packed)]
#[derive(Clone, Copy, Default)]
struct UffdMsg {
    event: u8,
    _reserved1: u8,
    _reserved2: u16,
    _reserved3: u32,
    flags: u64,
    address: u64,
    _ptid: u32,
    _pad: u32,
}

const _: () = assert!(std::mem::size_of::<UffdMsg>() == 32);

fn ioctl<T>(fd: RawFd, request: u64, arg: &mut T) -> io::Result<()> {
    // SAFETY: `arg` is the structure `request` expects, valid for the call.
    let r = unsafe { libc::ioctl(fd, request as libc::Ioctl, arg as *mut T) };
    if r < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

fn open_uffd() -> Result<OwnedFd> {
    let base = libc::O_CLOEXEC | libc::O_NONBLOCK;
    // user-mode-only faults are allowed unprivileged since Linux 5.11; older kernels reject the
    // flag, and then only a permissive `vm.unprivileged_userfaultfd` lets us in
    for extra in [UFFD_USER_MODE_ONLY, 0] {
        // SAFETY: a plain system call with integer flags.
        let fd = unsafe { libc::syscall(libc::SYS_userfaultfd, base | extra) };
        if fd >= 0 {
            // SAFETY: a fresh descriptor we own.
            return Ok(unsafe { OwnedFd::from_raw_fd(fd as RawFd) });
        }
        let e = io::Error::last_os_error();
        match e.raw_os_error() {
            Some(libc::EINVAL) if extra != 0 => continue,
            Some(libc::EPERM) | Some(libc::EACCES) => {
                return Err(Error::Unsupported(format!(
                    "userfaultfd is not permitted here ({e}); allow it with `sysctl \
                     vm.unprivileged_userfaultfd=1`, or in a container with a seccomp profile \
                     that allows it"
                )));
            }
            Some(libc::ENOSYS) => {
                return Err(Error::Unsupported("this kernel has no userfaultfd".into()));
            }
            _ => return Err(Error::Io(e)),
        }
    }
    Err(Error::Unsupported("userfaultfd could not be opened".into()))
}

enum Data {
    /// Never touched: comes back as zeros.
    Absent,
    Resident,
    Compressed(Vec<u8>),
}

struct Chunk {
    data: Data,
    /// Fault count at the last fault on this chunk.
    last: u64,
    /// Smoothed faults between two faults on this chunk (0 = not known yet).
    period: f64,
    faults: u64,
    /// Compression would not save enough: stays in memory.
    incompressible: bool,
}

struct Region {
    len: usize,
    chunks: Vec<Chunk>,
}

struct State {
    regions: BTreeMap<usize, Region>,
    clock: u64,
    resident: u64,
    compressed: u64,
    stats: PagerStats,
    scratch: Vec<u8>,
    packed: Vec<u8>,
}

struct Shared {
    config: PagerConfig,
    limit: u64,
    uffd: OwnedFd,
    stop: OwnedFd,
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
        let uffd = open_uffd()?;
        let mut api = UffdioApi {
            api: UFFD_API,
            features: UFFD_FEATURE_PAGEFAULT_FLAG_WP,
            ioctls: 0,
        };
        ioctl(uffd.as_raw_fd(), UFFDIO_API, &mut api).map_err(|e| {
            Error::Unsupported(format!(
                "userfaultfd refused write-protect faults ({e}); Linux 5.7 or newer is needed"
            ))
        })?;
        // SAFETY: a plain system call.
        let stop = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC | libc::EFD_NONBLOCK) };
        if stop < 0 {
            return Err(Error::Io(io::Error::last_os_error()));
        }
        // SAFETY: a fresh descriptor we own.
        let stop = unsafe { OwnedFd::from_raw_fd(stop) };
        let stats = PagerStats {
            budget: config.budget,
            limit,
            ..PagerStats::default()
        };
        let shared = Arc::new(Shared {
            config,
            limit,
            uffd,
            stop,
            state: Mutex::new(State {
                regions: BTreeMap::new(),
                clock: 0,
                resident: 0,
                compressed: 0,
                stats,
                scratch: Vec::new(),
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
        let chunk = self.shared.config.chunk;
        let size = len.div_ceil(chunk) * chunk;
        // SAFETY: a new private anonymous mapping; nothing else refers to it.
        let p = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        if p == libc::MAP_FAILED {
            return Err(Error::Io(io::Error::last_os_error()));
        }
        let mut reg = UffdioRegister {
            range: UffdioRange {
                start: p as u64,
                len: size as u64,
            },
            mode: UFFDIO_REGISTER_MODE_MISSING | UFFDIO_REGISTER_MODE_WP,
            ioctls: 0,
        };
        let registered = ioctl(self.shared.uffd.as_raw_fd(), UFFDIO_REGISTER, &mut reg);
        let usable = registered.is_ok()
            && reg.ioctls & (HAS_COPY | HAS_WRITEPROTECT) == (HAS_COPY | HAS_WRITEPROTECT);
        if !usable {
            if registered.is_ok() {
                let mut range = reg.range;
                let _ = ioctl(self.shared.uffd.as_raw_fd(), UFFDIO_UNREGISTER, &mut range);
            }
            // SAFETY: the mapping made above, not handed out.
            unsafe { libc::munmap(p, size) };
            return Err(Error::Unsupported(match registered {
                Err(e) => format!("registering memory with userfaultfd failed: {e}"),
                Ok(()) => "userfaultfd cannot copy into or write-protect this memory".into(),
            }));
        }
        let mut st = self.shared.lock();
        st.regions.insert(
            p as usize,
            Region {
                len: size,
                chunks: (0..size / chunk)
                    .map(|_| Chunk {
                        data: Data::Absent,
                        last: 0,
                        period: 0.0,
                        faults: 0,
                        incompressible: false,
                    })
                    .collect(),
            },
        );
        st.stats.regions += 1;
        st.stats.mapped_bytes += size as u64;
        Ok(p.cast())
    }

    /// Give a region back; returns its length.
    ///
    /// # Safety
    /// `addr` came from [`Pager::map`] of this pager, and no one uses the region any more.
    pub unsafe fn unmap(&self, addr: *mut u8) -> Result<usize> {
        let chunk = self.shared.config.chunk as u64;
        let mut st = self.shared.lock();
        let region = st.regions.remove(&(addr as usize)).ok_or_else(|| {
            Error::InvalidArgument(format!("{addr:p} is not a region of this pager"))
        })?;
        let mut range = UffdioRange {
            start: addr as u64,
            len: region.len as u64,
        };
        let _ = ioctl(self.shared.uffd.as_raw_fd(), UFFDIO_UNREGISTER, &mut range);
        // SAFETY: the caller's contract: our mapping, no longer in use.
        unsafe { libc::munmap(addr.cast(), region.len) };
        for c in &region.chunks {
            match &c.data {
                Data::Resident => st.resident -= chunk,
                Data::Compressed(p) => st.compressed -= p.len() as u64,
                Data::Absent => {}
            }
        }
        st.stats.regions -= 1;
        st.stats.mapped_bytes -= region.len as u64;
        Ok(region.len)
    }

    /// The length of the region starting at `addr`, if it is one of ours.
    pub fn owns(&self, addr: *const u8) -> Option<usize> {
        self.shared
            .lock()
            .regions
            .get(&(addr as usize))
            .map(|r| r.len)
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
        let one: u64 = 1;
        // SAFETY: writing 8 bytes to our eventfd wakes the pager thread.
        unsafe {
            libc::write(
                self.shared.stop.as_raw_fd(),
                (&one as *const u64).cast(),
                std::mem::size_of::<u64>(),
            )
        };
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        let mut st = self.shared.lock();
        let regions = std::mem::take(&mut st.regions);
        for (start, r) in regions {
            let mut range = UffdioRange {
                start: start as u64,
                len: r.len as u64,
            };
            let _ = ioctl(self.shared.uffd.as_raw_fd(), UFFDIO_UNREGISTER, &mut range);
            // SAFETY: our mapping; the pager goes away with it (users must be done with it).
            unsafe { libc::munmap(start as *mut libc::c_void, r.len) };
        }
    }
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The pager thread: serve page faults until stopped.
    fn serve(&self) {
        let uffd = self.uffd.as_raw_fd();
        let mut msgs = [UffdMsg::default(); 64];
        loop {
            let mut fds = [
                libc::pollfd {
                    fd: uffd,
                    events: libc::POLLIN,
                    revents: 0,
                },
                libc::pollfd {
                    fd: self.stop.as_raw_fd(),
                    events: libc::POLLIN,
                    revents: 0,
                },
            ];
            // SAFETY: two valid pollfd entries.
            let r = unsafe { libc::poll(fds.as_mut_ptr(), 2, -1) };
            if r < 0 {
                continue; // EINTR
            }
            if fds[1].revents != 0 {
                return;
            }
            if fds[0].revents & libc::POLLIN == 0 {
                continue;
            }
            // SAFETY: reading whole messages into a buffer of that many bytes.
            let n =
                unsafe { libc::read(uffd, msgs.as_mut_ptr().cast(), std::mem::size_of_val(&msgs)) };
            if n <= 0 {
                continue; // EAGAIN: another wake-up took them
            }
            let count = n as usize / std::mem::size_of::<UffdMsg>();
            let mut st = self.lock();
            for m in &msgs[..count] {
                let (event, address, flags) = (m.event, m.address, m.flags);
                if event == UFFD_EVENT_PAGEFAULT {
                    self.fault(&mut st, address as usize, flags);
                }
            }
        }
    }

    fn fault(&self, st: &mut State, addr: usize, flags: u64) {
        st.stats.faults += 1;
        let chunk = self.config.chunk;
        let page = page_size();
        let found = st
            .regions
            .range(..=addr)
            .next_back()
            .filter(|(start, r)| addr < **start + r.len)
            .map(|(start, _)| *start);
        let Some(start) = found else {
            // not ours (any more): let the thread retry and see what is there
            self.wake(addr & !(page - 1), page);
            return;
        };
        let ci = (addr - start) / chunk;
        let cstart = start + ci * chunk;
        st.clock += 1;
        let now = st.clock;
        let resident = matches!(
            st.regions.get(&start).expect("found above").chunks[ci].data,
            Data::Resident
        );
        if resident {
            // a write that raced an eviction which has not happened, or a second waiter
            st.stats.spurious += 1;
            if flags & UFFD_PAGEFAULT_FLAG_WP != 0 {
                let _ = self.protect(cstart, chunk, false);
            } else {
                self.wake(cstart, chunk);
            }
            return;
        }
        self.make_room(st, chunk as u64, (start, ci));
        let t0 = Instant::now();
        let mut scratch = std::mem::take(&mut st.scratch);
        scratch.resize(chunk, 0);
        let c = &mut st.regions.get_mut(&start).expect("found above").chunks[ci];
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
                scratch.fill(0);
                st.stats.zero_fills += 1;
            }
            Data::Compressed(p) => {
                if let Err(e) = codec::unpack_chunk(&p, self.config.elem, &mut scratch) {
                    // the only copy of this memory is damaged: stopping is the only honest thing
                    eprintln!("memopro pager: a compressed chunk could not be restored ({e})");
                    std::process::abort();
                }
                st.compressed -= p.len() as u64;
                st.stats.restores += 1;
            }
            Data::Resident => {}
        }
        self.copy(cstart, scratch.as_ptr(), chunk);
        st.scratch = scratch;
        st.resident += chunk as u64;
        let used = st.resident + st.compressed;
        st.stats.peak_used = st.stats.peak_used.max(used);
        st.stats.restore_seconds += t0.elapsed().as_secs_f64();
    }

    fn make_room(&self, st: &mut State, need: u64, exclude: (usize, usize)) {
        while st.resident + st.compressed + need > self.limit {
            let Some((start, ci)) = self.choose(st, exclude) else {
                st.stats.overruns += 1;
                return;
            };
            self.evict(st, start, ci);
        }
    }

    /// The chunk in memory whose next fault is predicted farthest, never one of the chunks
    /// faulted in most recently: code works on them right now (two arrays read and written
    /// in step ping-ponged chunk for chunk otherwise, and CI tests took hours, 0128). If only
    /// such chunks are left, the one faulted in longest ago.
    fn choose(&self, st: &State, exclude: (usize, usize)) -> Option<(usize, usize)> {
        let now = st.clock as f64;
        let fits = (self.limit / self.config.chunk as u64).max(1);
        let hot = (fits / 4).clamp(1, HOT_CHUNKS) as f64;
        let (sum, count) = st
            .regions
            .values()
            .flat_map(|r| r.chunks.iter())
            .filter(|c| c.period > 0.0)
            .fold((0.0, 0u64), |(s, n), c| (s + c.period, n + 1));
        let typical = if count > 0 { sum / count as f64 } else { 0.0 };
        let mut best: Option<(f64, usize, usize)> = None;
        let mut oldest_hot: Option<(f64, usize, usize)> = None;
        for (&start, r) in &st.regions {
            for (ci, c) in r.chunks.iter().enumerate() {
                if !matches!(c.data, Data::Resident) || c.incompressible || (start, ci) == exclude {
                    continue;
                }
                let age = now - c.last as f64;
                if age < hot {
                    if oldest_hot.is_none_or(|(a, _, _)| age > a) {
                        oldest_hot = Some((age, start, ci));
                    }
                    continue;
                }
                let period = if c.period > 0.0 { c.period } else { typical };
                let distance = if period > 0.0 {
                    (c.last as f64 + period - now).max(0.0) + 1.0
                } else {
                    age + 1.0
                };
                if best.is_none_or(|(d, _, _)| distance > d) {
                    best = Some((distance, start, ci));
                }
            }
        }
        best.or(oldest_hot).map(|(_, s, c)| (s, c))
    }

    fn evict(&self, st: &mut State, start: usize, ci: usize) {
        let chunk = self.config.chunk;
        let cstart = start + ci * chunk;
        let t0 = Instant::now();
        // from here on writers wait; readers go on
        let protected = self.protect(cstart, chunk, true).is_ok();
        let mut packed = std::mem::take(&mut st.packed);
        let saved = protected && {
            // SAFETY: the chunk is in memory and write-protected, so its bytes cannot change
            // while they are read; the pager thread never faults on its own reads of it.
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
        let c = &mut st
            .regions
            .get_mut(&start)
            .expect("chosen from the table")
            .chunks[ci];
        if !saved {
            if protected {
                let _ = self.protect(cstart, chunk, false);
            }
            c.incompressible = true;
            st.packed = packed;
            st.stats.incompressible += 1;
            return;
        }
        let stored = packed.clone();
        // SAFETY: our own mapping; its content is saved in `stored`.
        unsafe { libc::madvise(cstart as *mut libc::c_void, chunk, libc::MADV_DONTNEED) };
        let n = stored.len() as u64;
        c.data = Data::Compressed(stored);
        st.packed = packed;
        st.resident -= chunk as u64;
        st.compressed += n;
        st.stats.evictions += 1;
        st.stats.compress_in += chunk as u64;
        st.stats.compress_out += n;
        // threads that tried to write meanwhile wait on the write-protection; their faults are
        // queued and bring the chunk back
    }

    /// Fill `len` bytes at `dst` (all absent) from `src` and wake whoever waits on them.
    fn copy(&self, dst: usize, src: *const u8, len: usize) {
        let mut c = UffdioCopy {
            dst: dst as u64,
            src: src as u64,
            len: len as u64,
            mode: 0,
            copy: 0,
        };
        loop {
            match ioctl(self.uffd.as_raw_fd(), UFFDIO_COPY, &mut c) {
                Ok(()) => return,
                Err(e) if e.raw_os_error() == Some(libc::EAGAIN) && c.copy > 0 => {
                    let done = c.copy as u64;
                    c.dst += done;
                    c.src += done;
                    c.len -= done;
                    c.copy = 0;
                    if c.len == 0 {
                        return;
                    }
                }
                Err(_) => {
                    // already there (EEXIST) or gone: whoever waits retries the access
                    self.wake(dst, len);
                    return;
                }
            }
        }
    }

    fn protect(&self, start: usize, len: usize, on: bool) -> io::Result<()> {
        let mut wp = UffdioWriteprotect {
            range: UffdioRange {
                start: start as u64,
                len: len as u64,
            },
            mode: if on { UFFDIO_WRITEPROTECT_MODE_WP } else { 0 },
        };
        ioctl(self.uffd.as_raw_fd(), UFFDIO_WRITEPROTECT, &mut wp)
    }

    fn wake(&self, start: usize, len: usize) {
        let mut range = UffdioRange {
            start: start as u64,
            len: len as u64,
        };
        let _ = ioctl(self.uffd.as_raw_fd(), UFFDIO_WAKE, &mut range);
    }
}
