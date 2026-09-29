//! Runtime C-R, phase 1 (design `docs/design/runtime.md`, 0109, 0110, 0112).
//!
//! Large buffers are held as a recipe plus a state and kept under a hard memory budget. When a
//! buffer has to come in and the budget is full, the runtime makes room by
//!
//! - **dropping** a buffer that can be re-read from its original file (verified by digest,
//!   read without the page cache), or
//! - **compressing** one that cannot (byte shuffle + zstd, lossless, piece by piece),
//!
//! choosing per buffer the action with the smallest expected restore cost per byte freed,
//! weighted by how soon the buffer will be used again (its measured reuse period, so a buffer
//! used in a repeating scan is kept while the one just finished goes first). Everything is
//! lossless (R1), nothing is ever written to disk (0110 N1), and the accounted memory never goes
//! over the budget (R2): room is made before memory is taken, and one compression piece of
//! headroom is kept for the transient piece while compressing.
//!
//! Buffers are used through [`Pin`]s: a pinned buffer is resident and never moved (I2). Slow
//! work (reading, compressing, decompressing) runs outside the lock with the buffer marked busy,
//! so other threads keep going; they wait only for that buffer.

mod region;
mod source;

pub use region::{Region, page_size, round_to_pages};
pub use source::SourceFile;

use crate::codec;
use crate::error::{Error, Result};
use crate::spill::Digest;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::Instant;

/// Identifier of a managed buffer.
pub type BufferId = u64;

/// How the runtime picks what to give up when it needs room.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Policy {
    /// Smallest restore cost per byte freed, divided by the predicted distance to the next use
    /// (from each buffer's measured reuse period; its staleness while no period is known).
    ReuseDistance,
    /// Least recently used first (kept for comparison, E025 H7).
    Lru,
}

/// Runtime settings.
#[derive(Debug, Clone)]
pub struct Config {
    /// Bytes the runtime's buffers may occupy, including compressed pieces.
    pub budget: u64,
    /// zstd level for compression (1 = fast).
    pub compress_level: i32,
    /// Smallest fraction of a buffer that compression must save to be worth doing.
    pub min_saving: f64,
    pub policy: Policy,
}

impl Config {
    pub fn new(budget: u64) -> Self {
        Config {
            budget,
            compress_level: 1,
            min_saving: 0.15,
            policy: Policy::ReuseDistance,
        }
    }
}

/// Where a buffer's data is now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferState {
    /// In memory, usable at once.
    Resident,
    /// Compressed in memory.
    Compressed,
    /// Not in memory; re-read from its original file when needed.
    Dropped,
    /// Registered from a file and never read yet.
    Unloaded,
}

/// Counters and gauges; `used` and `peak_used` are the accounted bytes the budget limits.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Stats {
    pub budget: u64,
    /// Headroom kept for one compression piece (part of the budget).
    pub reserve: u64,
    pub used: u64,
    pub peak_used: u64,
    pub buffers: u64,
    pub resident_bytes: u64,
    pub compressed_bytes: u64,
    pub pinned_bytes: u64,
    /// First reads of file-backed buffers.
    pub loads: u64,
    pub load_bytes: u64,
    /// Reads of buffers that had been dropped.
    pub rereads: u64,
    pub reread_bytes: u64,
    pub read_seconds: f64,
    pub drops: u64,
    pub drop_bytes: u64,
    pub compressions: u64,
    pub compress_in: u64,
    pub compress_out: u64,
    pub compress_seconds: f64,
    pub decompressions: u64,
    pub decompress_bytes: u64,
    pub decompress_seconds: f64,
    /// Buffers found not worth compressing.
    pub incompressible: u64,
    /// Room-making actions (drops + compressions).
    pub evictions: u64,
    /// Requests refused because the budget could not hold them.
    pub refusals: u64,
    /// Time pins spent bringing buffers back (reads + decompression).
    pub restore_seconds: f64,
    /// Always 0: the runtime never writes to disk (0110 N1).
    pub written_bytes: u64,
}

/// The runtime. Cloning shares it.
#[derive(Clone)]
pub struct Runtime {
    shared: Arc<Shared>,
}

struct Shared {
    config: Config,
    reserve: u64,
    state: Mutex<Inner>,
    cond: Condvar,
}

struct Inner {
    bufs: HashMap<BufferId, Buf>,
    next_id: BufferId,
    clock: u64,
    used: u64,
    files: HashMap<PathBuf, Arc<SourceFile>>,
    stats: Stats,
    cost: Cost,
}

struct Buf {
    nbytes: usize,
    elem: usize,
    data: Data,
    source: Option<Src>,
    /// Written through a writable pin: the original file no longer describes it.
    dirty: bool,
    pins: u32,
    writers: u32,
    /// Being read, compressed or decompressed outside the lock.
    busy: bool,
    incompressible: bool,
    ratio: Option<f64>,
    /// Logical time of the last pin.
    last: u64,
    /// Smoothed number of pins between two uses of this buffer (0 = not known yet).
    period: f64,
    uses: u64,
}

struct Src {
    file: Arc<SourceFile>,
    offset: u64,
    /// Known after the first read; required before the buffer may be dropped (R5).
    digest: Option<Digest>,
}

enum Data {
    Empty,
    Resident(Region),
    Compressed(Packed),
}

struct Packed {
    pieces: Vec<Piece>,
    stored: u64,
}

struct Piece {
    region: Region,
    /// Original bytes this piece holds.
    len: usize,
    /// Stored as is (compression would not have made it smaller).
    raw: bool,
}

/// Measured throughputs (bytes per second), smoothed; initial values from E015/E008 (0109 §7).
struct Cost {
    read: f64,
    compress: f64,
    decompress: f64,
}

const SMOOTHING: f64 = 0.3;

fn smooth(old: f64, bytes: usize, seconds: f64) -> f64 {
    if seconds <= 0.0 || bytes == 0 {
        return old;
    }
    (1.0 - SMOOTHING) * old + SMOOTHING * (bytes as f64 / seconds)
}

enum Action {
    Drop,
    Compress,
}

impl Buf {
    fn accounted(&self) -> u64 {
        match &self.data {
            Data::Empty => 0,
            Data::Resident(r) => r.capacity() as u64,
            Data::Compressed(p) => p.stored,
        }
    }

    fn source_valid(&self) -> bool {
        !self.dirty && self.source.as_ref().is_some_and(|s| s.digest.is_some())
    }

    fn state(&self) -> BufferState {
        match &self.data {
            Data::Resident(_) => BufferState::Resident,
            Data::Compressed(_) => BufferState::Compressed,
            Data::Empty if self.source.as_ref().is_some_and(|s| s.digest.is_some()) => {
                BufferState::Dropped
            }
            Data::Empty => BufferState::Unloaded,
        }
    }
}

fn elem_ok(elem: usize) -> bool {
    matches!(elem, 1 | 2 | 4 | 8 | 16)
}

fn mib(n: u64) -> String {
    format!("{:.1} MiB", n as f64 / (1u64 << 20) as f64)
}

impl Runtime {
    /// A runtime whose buffers may occupy at most `config.budget` bytes.
    pub fn new(config: Config) -> Result<Runtime> {
        let reserve = round_to_pages(codec::chunk_bound(codec::CHUNK)) as u64;
        if config.budget < 2 * reserve {
            return Err(Error::InvalidArgument(format!(
                "a budget of {} is too small: the runtime needs at least {}",
                mib(config.budget),
                mib(2 * reserve)
            )));
        }
        if !(0.0..1.0).contains(&config.min_saving) {
            return Err(Error::InvalidArgument(
                "min_saving must be in [0, 1)".into(),
            ));
        }
        let stats = Stats {
            budget: config.budget,
            reserve,
            ..Stats::default()
        };
        Ok(Runtime {
            shared: Arc::new(Shared {
                config,
                reserve,
                state: Mutex::new(Inner {
                    bufs: HashMap::new(),
                    next_id: 1,
                    clock: 0,
                    used: 0,
                    files: HashMap::new(),
                    stats,
                    cost: Cost {
                        read: 1.5e9,
                        compress: 4e8,
                        decompress: 1.2e9,
                    },
                }),
                cond: Condvar::new(),
            }),
        })
    }

    pub fn config(&self) -> &Config {
        &self.shared.config
    }

    /// Bytes buffers may occupy: the budget minus the compression headroom.
    pub fn limit(&self) -> u64 {
        self.shared.config.budget - self.shared.reserve
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.shared.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn wait<'a>(&'a self, st: MutexGuard<'a, Inner>) -> MutexGuard<'a, Inner> {
        self.shared.cond.wait(st).unwrap_or_else(|e| e.into_inner())
    }

    /// A new zero-filled buffer of `nbytes` in memory (no original file: it can only be
    /// compressed to make room). `elem` is the element size used to shuffle bytes before
    /// compression (1, 2, 4, 8 or 16).
    pub fn alloc(&self, nbytes: usize, elem: usize) -> Result<BufferId> {
        check_size(nbytes, elem)?;
        let cap = round_to_pages(nbytes) as u64;
        let st = self.lock();
        let mut st = self.make_room(st, cap, None)?;
        let region = Region::new(nbytes)?;
        st.used += cap;
        st.stats.peak_used = st.stats.peak_used.max(st.used);
        let id = st.next_id;
        st.next_id += 1;
        let clock = st.clock;
        st.bufs.insert(
            id,
            Buf {
                nbytes,
                elem,
                data: Data::Resident(region),
                source: None,
                dirty: false,
                pins: 0,
                writers: 0,
                busy: false,
                incompressible: false,
                ratio: None,
                last: clock,
                period: 0.0,
                uses: 0,
            },
        );
        Ok(id)
    }

    /// Register `nbytes` of `path` from `offset` as a buffer. Nothing is read until it is first
    /// pinned; after that it can be dropped and re-read (the file must not change).
    pub fn add_file(
        &self,
        path: &Path,
        offset: u64,
        nbytes: usize,
        elem: usize,
    ) -> Result<BufferId> {
        check_size(nbytes, elem)?;
        let mut st = self.lock();
        let file = match st.files.get(path) {
            Some(f) => f.clone(),
            None => {
                let f = Arc::new(SourceFile::open(path)?);
                st.files.insert(path.to_path_buf(), f.clone());
                f
            }
        };
        if offset.saturating_add(nbytes as u64) > file.len() {
            return Err(Error::InvalidArgument(format!(
                "{} bytes from offset {offset} run past the end of {} ({} bytes)",
                nbytes,
                path.display(),
                file.len()
            )));
        }
        let id = st.next_id;
        st.next_id += 1;
        let clock = st.clock;
        st.bufs.insert(
            id,
            Buf {
                nbytes,
                elem,
                data: Data::Empty,
                source: Some(Src {
                    file,
                    offset,
                    digest: None,
                }),
                dirty: false,
                pins: 0,
                writers: 0,
                busy: false,
                incompressible: false,
                ratio: None,
                last: clock,
                period: 0.0,
                uses: 0,
            },
        );
        Ok(id)
    }

    /// Make the buffer resident and keep it there until the returned [`Pin`] is dropped. A
    /// writable pin needs the buffer unpinned and makes its original file unusable for it;
    /// while it lives no other pin of the buffer can be taken (so its slices never alias).
    pub fn pin(&self, id: BufferId, write: bool) -> Result<Pin> {
        self.pin_with(id, write, true)
    }

    /// Like [`Runtime::pin`] but without the exclusivity between readers and writers: the pin
    /// only guarantees the memory stays put. For callers that reach the data through the raw
    /// address (the Python buffer protocol, the C ABI), not through [`Pin::as_slice`].
    ///
    /// # Safety
    ///
    /// Do not call [`Pin::as_slice`] or [`Pin::as_mut_slice`] on a shared pin while anything
    /// may write the buffer through another pin: that would alias a Rust reference.
    pub unsafe fn pin_shared(&self, id: BufferId, write: bool) -> Result<Pin> {
        self.pin_with(id, write, false)
    }

    fn pin_with(&self, id: BufferId, write: bool, exclusive: bool) -> Result<Pin> {
        let mut st = self.lock();
        loop {
            let b = st
                .bufs
                .get(&id)
                .ok_or_else(|| Error::InvalidArgument(format!("unknown buffer {id}")))?;
            if b.busy {
                st = self.wait(st);
                continue;
            }
            if exclusive && write && b.pins > 0 {
                return Err(Error::InvalidArgument(format!(
                    "buffer {id} is pinned; a writable pin needs it unpinned"
                )));
            }
            if exclusive && !write && b.writers > 0 {
                return Err(Error::InvalidArgument(format!(
                    "buffer {id} is pinned for writing"
                )));
            }
            let state = b.state();
            match state {
                BufferState::Resident => break,
                BufferState::Compressed => st = self.restore_packed(st, id)?,
                BufferState::Dropped | BufferState::Unloaded => st = self.restore_source(st, id)?,
            }
        }
        st.clock += 1;
        let now = st.clock;
        let b = st.bufs.get_mut(&id).expect("checked above");
        if b.uses > 0 {
            let gap = (now - b.last) as f64;
            b.period = if b.period == 0.0 {
                gap
            } else {
                0.5 * b.period + 0.5 * gap
            };
        }
        b.last = now;
        b.uses += 1;
        b.pins += 1;
        if write {
            if exclusive {
                b.writers += 1;
            }
            b.dirty = true;
            b.ratio = None;
            b.incompressible = false;
        }
        let Data::Resident(region) = &b.data else {
            unreachable!("restored above")
        };
        Ok(Pin {
            shared: self.shared.clone(),
            id,
            ptr: region.as_ptr() as usize,
            len: b.nbytes,
            write,
            exclusive,
        })
    }

    /// Give up the buffer and its memory. Fails while it is pinned.
    pub fn free(&self, id: BufferId) -> Result<()> {
        let mut st = self.lock();
        loop {
            let b = st
                .bufs
                .get(&id)
                .ok_or_else(|| Error::InvalidArgument(format!("unknown buffer {id}")))?;
            if b.busy {
                st = self.wait(st);
                continue;
            }
            if b.pins > 0 {
                return Err(Error::InvalidArgument(format!("buffer {id} is pinned")));
            }
            break;
        }
        let b = st.bufs.remove(&id).expect("checked above");
        st.used -= b.accounted();
        drop(b);
        self.shared.cond.notify_all();
        Ok(())
    }

    /// Give up the buffer's memory now if that is possible without losing it (drop when it can
    /// be re-read, else compress). Returns whether anything was freed.
    pub fn evict(&self, id: BufferId) -> Result<bool> {
        let mut st = self.lock();
        loop {
            let b = st
                .bufs
                .get(&id)
                .ok_or_else(|| Error::InvalidArgument(format!("unknown buffer {id}")))?;
            if b.busy {
                st = self.wait(st);
                continue;
            }
            if b.pins > 0 {
                return Ok(false);
            }
            break;
        }
        let level = self.shared.config.compress_level;
        let min_saving = self.shared.config.min_saving;
        let b = st.bufs.get_mut(&id).expect("checked above");
        let action = match (&b.data, b.source_valid()) {
            (Data::Resident(_) | Data::Compressed(_), true) => Some(Action::Drop),
            (Data::Resident(r), false) if !b.incompressible => {
                let ratio = *b.ratio.get_or_insert_with(|| probe_ratio(r, b.elem, level));
                if 1.0 - 1.0 / ratio < min_saving {
                    b.incompressible = true;
                    None
                } else {
                    Some(Action::Compress)
                }
            }
            _ => None,
        };
        match action {
            Some(Action::Drop) => {
                self.drop_data(&mut st, id);
                Ok(true)
            }
            Some(Action::Compress) => {
                let before = st.used;
                let st = self.compress(st, id)?;
                Ok(st.used < before)
            }
            None => Ok(false),
        }
    }

    pub fn state(&self, id: BufferId) -> Result<BufferState> {
        let st = self.lock();
        st.bufs
            .get(&id)
            .map(Buf::state)
            .ok_or_else(|| Error::InvalidArgument(format!("unknown buffer {id}")))
    }

    /// Size in bytes of a buffer.
    pub fn nbytes(&self, id: BufferId) -> Result<usize> {
        let st = self.lock();
        st.bufs
            .get(&id)
            .map(|b| b.nbytes)
            .ok_or_else(|| Error::InvalidArgument(format!("unknown buffer {id}")))
    }

    pub fn stats(&self) -> Stats {
        let st = self.lock();
        let mut s = st.stats.clone();
        s.used = st.used;
        s.buffers = st.bufs.len() as u64;
        for b in st.bufs.values() {
            match &b.data {
                Data::Resident(r) => {
                    s.resident_bytes += r.capacity() as u64;
                    if b.pins > 0 {
                        s.pinned_bytes += r.capacity() as u64;
                    }
                }
                Data::Compressed(p) => s.compressed_bytes += p.stored,
                Data::Empty => {}
            }
        }
        s
    }

    // ------------------------------------------------------------------ room

    /// Make room for `need` more accounted bytes, giving up other buffers (never `exclude`,
    /// never pinned or busy ones). Fails with [`Error::Budget`] when nothing more can go.
    fn make_room<'a>(
        &'a self,
        mut st: MutexGuard<'a, Inner>,
        need: u64,
        exclude: Option<BufferId>,
    ) -> Result<MutexGuard<'a, Inner>> {
        let limit = self.limit();
        if need > limit {
            st.stats.refusals += 1;
            return Err(Error::Budget(format!(
                "one buffer needs {}, more than the budget {} minus the runtime's {} headroom",
                mib(need),
                mib(self.shared.config.budget),
                mib(self.shared.reserve)
            )));
        }
        loop {
            if st.used + need <= limit {
                return Ok(st);
            }
            match self.choose(&mut st, exclude) {
                Some((id, Action::Drop)) => self.drop_data(&mut st, id),
                Some((id, Action::Compress)) => st = self.compress(st, id)?,
                None if st.bufs.values().any(|b| b.busy) => st = self.wait(st),
                None => {
                    st.stats.refusals += 1;
                    let pinned: u64 = st
                        .bufs
                        .values()
                        .filter(|b| b.pins > 0)
                        .map(Buf::accounted)
                        .sum();
                    let kept: u64 = st.used - pinned;
                    return Err(Error::Budget(format!(
                        "need {} more, {} of {} in use ({} pinned, {} that cannot be dropped or \
                         compressed further); unpin buffers or raise the budget",
                        mib(need),
                        mib(st.used),
                        mib(limit),
                        mib(pinned),
                        mib(kept)
                    )));
                }
            }
        }
    }

    /// The buffer and action with the lowest restore cost per byte freed, per distance to the
    /// next use; `None` when nothing can be given up.
    fn choose(&self, st: &mut Inner, exclude: Option<BufferId>) -> Option<(BufferId, Action)> {
        let now = st.clock as f64;
        let policy = self.shared.config.policy;
        let level = self.shared.config.compress_level;
        let min_saving = self.shared.config.min_saving;
        let (read, compress, decompress) = (st.cost.read, st.cost.compress, st.cost.decompress);
        // A buffer whose period is not measured yet is assumed to recur like the others do on
        // average; with no period known at all, staleness (LRU) is the only guide.
        let (sum, count) = st
            .bufs
            .values()
            .filter(|b| b.period > 0.0)
            .fold((0.0, 0u64), |(s, c), b| (s + b.period, c + 1));
        let typical = if count > 0 { sum / count as f64 } else { 0.0 };
        let mut best: Option<(f64, BufferId, Action)> = None;
        let mut newly_incompressible = 0;
        for (&id, b) in st.bufs.iter_mut() {
            if Some(id) == exclude || b.pins > 0 || b.busy {
                continue;
            }
            let staleness = now - b.last as f64 + 1.0;
            let period = if b.period > 0.0 { b.period } else { typical };
            let distance = match policy {
                Policy::ReuseDistance if period > 0.0 => {
                    (b.last as f64 + period - now).max(0.0) + 1.0
                }
                Policy::Lru | Policy::ReuseDistance => staleness,
            };
            let n = b.nbytes as f64;
            let valid = b.source_valid();
            let mut consider = |cost: f64, freed: f64, action: Action| {
                if freed <= 0.0 {
                    return;
                }
                let score = cost / freed / distance;
                if best.as_ref().is_none_or(|(s, _, _)| score < *s) {
                    best = Some((score, id, action));
                }
            };
            match &b.data {
                Data::Resident(r) => {
                    let cap = r.capacity() as f64;
                    if valid {
                        consider(n / read, cap, Action::Drop);
                    } else if !b.incompressible {
                        let ratio = match b.ratio {
                            Some(x) => x,
                            None => {
                                let x = probe_ratio(r, b.elem, level);
                                b.ratio = Some(x);
                                x
                            }
                        };
                        if 1.0 - 1.0 / ratio < min_saving {
                            b.incompressible = true;
                            newly_incompressible += 1;
                        } else {
                            consider(
                                n / compress + n / decompress,
                                cap * (1.0 - 1.0 / ratio),
                                Action::Compress,
                            );
                        }
                    }
                }
                Data::Compressed(p) if valid => {
                    let extra = (n / read - n / decompress).max(0.1 * n / read);
                    consider(extra, p.stored as f64, Action::Drop);
                }
                _ => {}
            }
        }
        st.stats.incompressible += newly_incompressible;
        best.map(|(_, id, action)| (id, action))
    }

    /// Forget the data of a buffer that can be re-read from its original file.
    fn drop_data(&self, st: &mut Inner, id: BufferId) {
        let b = st.bufs.get_mut(&id).expect("chosen from the table");
        let freed = b.accounted();
        let old = std::mem::replace(&mut b.data, Data::Empty);
        st.used -= freed;
        st.stats.drops += 1;
        st.stats.drop_bytes += freed;
        st.stats.evictions += 1;
        drop(old);
    }

    /// Compress a resident buffer piece by piece outside the lock. Memory stays within the
    /// budget: each piece's pages are given back before the next piece is stored, so at most one
    /// stored piece (the reserve) is extra at any time.
    fn compress<'a>(
        &'a self,
        mut st: MutexGuard<'a, Inner>,
        id: BufferId,
    ) -> Result<MutexGuard<'a, Inner>> {
        let level = self.shared.config.compress_level;
        let b = st.bufs.get_mut(&id).expect("chosen from the table");
        let Data::Resident(region) = std::mem::replace(&mut b.data, Data::Empty) else {
            unreachable!("only resident buffers are compressed")
        };
        b.busy = true;
        let (nbytes, elem) = (b.nbytes, b.elem);
        let cap = region.capacity() as u64;
        drop(st);
        let t0 = Instant::now();
        let result = pack_region(region, nbytes, elem, level);
        let seconds = t0.elapsed().as_secs_f64();
        let mut st = self.lock();
        let b = st.bufs.get_mut(&id).expect("busy buffers are not freed");
        b.busy = false;
        match result {
            Ok(packed) => {
                let stored = packed.stored;
                b.ratio = Some(nbytes as f64 / stored.max(1) as f64);
                b.data = Data::Compressed(packed);
                st.used = st.used - cap + stored;
                st.stats.compressions += 1;
                st.stats.compress_in += nbytes as u64;
                st.stats.compress_out += stored;
                st.stats.compress_seconds += seconds;
                st.stats.evictions += 1;
                st.cost.compress = smooth(st.cost.compress, nbytes, seconds);
            }
            Err((region, _)) => {
                // the data is whole again in `region`; never try this buffer again
                b.data = Data::Resident(region);
                b.incompressible = true;
                st.stats.incompressible += 1;
            }
        }
        self.shared.cond.notify_all();
        Ok(st)
    }

    /// Bring a dropped or never-read buffer back from its file (outside the lock).
    fn restore_source<'a>(
        &'a self,
        st: MutexGuard<'a, Inner>,
        id: BufferId,
    ) -> Result<MutexGuard<'a, Inner>> {
        let (nbytes, file, offset, expected) = {
            let b = st.bufs.get(&id).expect("checked by the caller");
            let Some(src) = &b.source else {
                return Err(Error::Integrity(format!(
                    "buffer {id} has no data and no way back (this is a bug)"
                )));
            };
            (b.nbytes, src.file.clone(), src.offset, src.digest.clone())
        };
        let cap = round_to_pages(nbytes) as u64;
        let mut st = self.make_room(st, cap, Some(id))?;
        st.used += cap;
        st.stats.peak_used = st.stats.peak_used.max(st.used);
        st.bufs.get_mut(&id).expect("still there").busy = true;
        drop(st);
        let t0 = Instant::now();
        let result = Region::new(nbytes).and_then(|mut region| {
            let digest = file.read_into(
                offset,
                &mut region.as_mut_slice()[..nbytes],
                expected.as_ref(),
            )?;
            Ok((region, digest))
        });
        let seconds = t0.elapsed().as_secs_f64();
        let mut st = self.lock();
        let b = st.bufs.get_mut(&id).expect("busy buffers are not freed");
        b.busy = false;
        let outcome = match result {
            Ok((region, digest)) => {
                b.data = Data::Resident(region);
                let first = expected.is_none();
                if first {
                    b.source.as_mut().expect("read from it").digest = Some(digest);
                }
                let s = &mut st.stats;
                if first {
                    s.loads += 1;
                    s.load_bytes += nbytes as u64;
                } else {
                    s.rereads += 1;
                    s.reread_bytes += nbytes as u64;
                }
                s.read_seconds += seconds;
                s.restore_seconds += seconds;
                st.cost.read = smooth(st.cost.read, nbytes, seconds);
                Ok(())
            }
            Err(e) => {
                st.used -= cap;
                Err(e)
            }
        };
        self.shared.cond.notify_all();
        outcome.map(|()| st)
    }

    /// Decompress a compressed buffer (outside the lock); its pieces go when it is whole.
    fn restore_packed<'a>(
        &'a self,
        st: MutexGuard<'a, Inner>,
        id: BufferId,
    ) -> Result<MutexGuard<'a, Inner>> {
        let nbytes = st.bufs.get(&id).expect("checked by the caller").nbytes;
        let cap = round_to_pages(nbytes) as u64;
        let mut st = self.make_room(st, cap, Some(id))?;
        let b = st.bufs.get_mut(&id).expect("still there");
        let Data::Compressed(packed) = std::mem::replace(&mut b.data, Data::Empty) else {
            unreachable!("checked by the caller")
        };
        b.busy = true;
        let elem = b.elem;
        st.used += cap;
        st.stats.peak_used = st.stats.peak_used.max(st.used);
        drop(st);
        let t0 = Instant::now();
        let result = unpack_region(&packed, nbytes, elem);
        let seconds = t0.elapsed().as_secs_f64();
        let mut st = self.lock();
        let b = st.bufs.get_mut(&id).expect("busy buffers are not freed");
        b.busy = false;
        let outcome = match result {
            Ok(region) => {
                b.data = Data::Resident(region);
                st.used -= packed.stored;
                let s = &mut st.stats;
                s.decompressions += 1;
                s.decompress_bytes += nbytes as u64;
                s.decompress_seconds += seconds;
                s.restore_seconds += seconds;
                st.cost.decompress = smooth(st.cost.decompress, nbytes, seconds);
                drop(packed);
                Ok(())
            }
            Err(e) => {
                b.data = Data::Compressed(packed);
                st.used -= cap;
                Err(e)
            }
        };
        self.shared.cond.notify_all();
        outcome.map(|()| st)
    }
}

fn check_size(nbytes: usize, elem: usize) -> Result<()> {
    if nbytes == 0 {
        return Err(Error::InvalidArgument(
            "a buffer holds at least one byte".into(),
        ));
    }
    if !elem_ok(elem) || nbytes % elem != 0 {
        return Err(Error::InvalidArgument(format!(
            "element size {elem} must be 1, 2, 4, 8 or 16 and divide {nbytes} bytes"
        )));
    }
    Ok(())
}

/// Compression ratio of the first piece of a buffer (at most 1 MiB), without changing it.
fn probe_ratio(region: &Region, elem: usize, level: i32) -> f64 {
    let n = region.len().min(1 << 20) / elem * elem;
    if n == 0 {
        return 1.0;
    }
    let mut out = Vec::new();
    match codec::pack_chunk(&region.as_slice()[..n], elem, level, &mut out) {
        Ok(()) if !out.is_empty() => n as f64 / out.len() as f64,
        _ => 1.0,
    }
}

thread_local! {
    static OUT: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Compress `region` piece by piece, giving each piece's pages back once it is stored. On
/// failure the data is put back whole into the region, which is returned with the error.
fn pack_region(
    mut region: Region,
    nbytes: usize,
    elem: usize,
    level: i32,
) -> std::result::Result<Packed, (Region, Error)> {
    let mut pieces: Vec<Piece> = Vec::new();
    let mut stored = 0u64;
    let mut off = 0usize;
    while off < nbytes {
        let len = codec::CHUNK.min(nbytes - off);
        let piece = OUT.with(|cell| -> Result<Piece> {
            let mut out = cell.borrow_mut();
            let src = &region.as_slice()[off..off + len];
            codec::pack_chunk(src, elem, level, &mut out)?;
            if out.len() < len {
                Ok(Piece {
                    region: Region::from_bytes(&out)?,
                    len,
                    raw: false,
                })
            } else {
                Ok(Piece {
                    region: Region::from_bytes(src)?,
                    len,
                    raw: true,
                })
            }
        });
        match piece {
            Ok(p) => {
                stored += p.region.capacity() as u64;
                pieces.push(p);
                region.release(off, len);
                off += len;
            }
            Err(e) => {
                // put back what was already given up
                let mut at = 0;
                for p in &pieces {
                    let dst = &mut region.as_mut_slice()[at..at + p.len];
                    if p.raw {
                        dst.copy_from_slice(&p.region.as_slice()[..p.len]);
                    } else if let Err(e2) = codec::unpack_chunk(p.region.as_slice(), elem, dst) {
                        return Err((
                            region,
                            Error::Integrity(format!("could not undo compression: {e2}")),
                        ));
                    }
                    at += p.len;
                }
                return Err((region, e));
            }
        }
    }
    Ok(Packed { pieces, stored })
}

fn unpack_region(packed: &Packed, nbytes: usize, elem: usize) -> Result<Region> {
    let mut region = Region::new(nbytes)?;
    let mut at = 0;
    for p in &packed.pieces {
        let dst = &mut region.as_mut_slice()[at..at + p.len];
        if p.raw {
            dst.copy_from_slice(&p.region.as_slice()[..p.len]);
        } else {
            codec::unpack_chunk(p.region.as_slice(), elem, dst)
                .map_err(|e| Error::Integrity(format!("compressed data is damaged: {e}")))?;
        }
        at += p.len;
    }
    if at != nbytes {
        return Err(Error::Integrity("compressed pieces do not add up".into()));
    }
    Ok(region)
}

/// A buffer held resident; unpins when dropped.
pub struct Pin {
    shared: Arc<Shared>,
    id: BufferId,
    ptr: usize,
    len: usize,
    write: bool,
    exclusive: bool,
}

impl Pin {
    pub fn id(&self) -> BufferId {
        self.id
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn is_writable(&self) -> bool {
        self.write
    }

    /// Address of the data; valid until the pin is dropped.
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr as *const u8
    }

    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: a pinned buffer is resident and never moved or freed (I2); an exclusive pin
        // never coexists with a writer, and shared pins are documented not to alias this way.
        unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.len) }
    }

    /// The data, writable; only for pins taken with `write = true`.
    pub fn as_mut_slice(&mut self) -> Result<&mut [u8]> {
        if !self.write {
            return Err(Error::InvalidArgument("pin is read-only".into()));
        }
        // SAFETY: as above, and a writable pin is the only pin of its buffer.
        Ok(unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut u8, self.len) })
    }
}

impl Drop for Pin {
    fn drop(&mut self) {
        let mut st = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(b) = st.bufs.get_mut(&self.id) {
            b.pins -= 1;
            if self.write && self.exclusive {
                b.writers -= 1;
            }
        }
        drop(st);
        self.shared.cond.notify_all();
    }
}

#[cfg(test)]
mod tests;
