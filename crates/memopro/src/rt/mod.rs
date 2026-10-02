//! Runtime C-R (design `docs/design/runtime.md`; phase 1 0112, phase 2 0115).
//!
//! Large buffers are held as a recipe plus a state and kept under a hard memory budget. When a
//! buffer has to come in and the budget is full, the runtime makes room by
//!
//! - **dropping** a buffer that can be re-read from its original file (verified by digest,
//!   read without the page cache) or **re-computed** from the buffers it was derived from
//!   (verified against the digest of its first result), or
//! - **compressing** one that has neither (byte shuffle + zstd, lossless, piece by piece),
//!
//! choosing per buffer the action with the smallest expected restore cost per byte freed,
//! weighted by how soon the buffer will be used again (its measured reuse period, so a buffer
//! used in a repeating scan is kept while the one just finished goes first). Everything is
//! lossless (R1), nothing is ever written to disk (0110 N1), and the accounted memory never goes
//! over the budget (R2): room is made before memory is taken, and one compression piece of
//! headroom is kept for the transient piece while compressing.
//!
//! A service thread (RS7, lifted in 0110 N3) brings buffers back before they are needed: it
//! learns which buffer follows which from the order of pins and restores the next ones while the
//! caller computes, without evicting anything needed sooner. It only touches memory the runtime
//! owns (R4, I6); re-computation runs on the caller's thread.
//!
//! Buffers are used through [`Pin`]s: a pinned buffer is resident and never moved (I2). Slow
//! work (reading, compressing, decompressing, re-computing) runs outside the lock with the buffer
//! marked busy, so other threads keep going; they wait only for that buffer.

mod predict;
mod region;
mod source;

pub use predict::Prediction;
pub use region::{Region, page_size, round_to_pages};
pub use source::SourceFile;

use crate::codec;
use crate::error::{Error, Result};
use crate::spill::{Digest, digest};
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::Instant;

/// Identifier of a managed buffer.
pub type BufferId = u64;

/// A deterministic computation that fills its output from its inputs (for [`Runtime::derive`]).
pub type Compute = Arc<dyn Fn(&[&[u8]], &mut [u8]) -> Result<()> + Send + Sync>;

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
    /// Bring the next buffers back in the background while the caller computes.
    pub prefetch: bool,
    /// Bytes to bring back ahead of use (at least one buffer).
    pub lookahead: u64,
}

impl Config {
    pub fn new(budget: u64) -> Self {
        Config {
            budget,
            compress_level: 1,
            min_saving: 0.15,
            policy: Policy::ReuseDistance,
            prefetch: true,
            lookahead: 64 << 20,
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
    /// Not in memory; re-read from its original file or re-computed when needed.
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
    pub pins: u64,
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
    /// Buffers re-computed from the buffers they were derived from.
    pub recomputes: u64,
    pub recompute_bytes: u64,
    pub recompute_seconds: f64,
    /// Buffers found not worth compressing.
    pub incompressible: u64,
    /// Room-making actions (drops + compressions).
    pub evictions: u64,
    /// Requests refused because the budget could not hold them.
    pub refusals: u64,
    /// Time pins spent waiting for buffers to come back.
    pub restore_seconds: f64,
    /// Buffers brought back ahead of use by the service thread.
    pub prefetches: u64,
    pub prefetch_bytes: u64,
    /// Pins that found a prefetched buffer ready.
    pub prefetch_hits: u64,
    /// Prefetched buffers given up again before they were used.
    pub prefetch_wasted: u64,
    /// Prefetches not done because only buffers needed sooner could have made room.
    pub prefetch_skipped: u64,
    /// Always 0: the runtime never writes to disk (0110 N1).
    pub written_bytes: u64,
}

/// The runtime. Cloning shares it; the service thread stops when the last clone is dropped
/// (pins keep the memory they hold valid until they are dropped too).
#[derive(Clone)]
pub struct Runtime {
    shared: Arc<Shared>,
    _owner: Arc<Owner>,
}

struct Owner {
    shared: Arc<Shared>,
    service: Mutex<Option<JoinHandle<()>>>,
}

impl Drop for Owner {
    fn drop(&mut self) {
        self.shared.lock().stop = true;
        self.shared.cond.notify_all();
        if let Some(handle) = self
            .service
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
        {
            let _ = handle.join();
        }
    }
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
    last_pinned: Option<BufferId>,
    /// The buffer used before `last_pinned` (context for `pairs`).
    prev_pinned: Option<BufferId>,
    /// Second-order learned order: (previous, current) -> next. Tells a forward pass from a
    /// backward pass over the same buffers, which a single successor per buffer cannot (0117 F2).
    pairs: HashMap<(BufferId, BufferId), BufferId>,
    queue: VecDeque<(BufferId, f64)>,
    queued: HashSet<BufferId>,
    trace: VecDeque<Trace>,
    stop: bool,
}

/// One pin, for prediction: which buffer, how big, when, and how long it waited.
#[derive(Debug, Clone, Copy)]
struct Trace {
    id: BufferId,
    at: Instant,
    waited: f64,
}

const TRACE_LEN: usize = 1 << 16;

struct Buf {
    nbytes: usize,
    elem: usize,
    data: Data,
    source: Option<Src>,
    lineage: Option<Lineage>,
    /// Buffers derived from this one.
    dependents: Vec<BufferId>,
    /// Written through a writable pin: the original file no longer describes it.
    dirty: bool,
    pins: u32,
    writers: u32,
    /// Being read, compressed, decompressed or re-computed outside the lock.
    busy: bool,
    incompressible: bool,
    ratio: Option<f64>,
    /// Logical time of the last pin.
    last: u64,
    /// Smoothed number of pins between two uses of this buffer (0 = not known yet).
    period: f64,
    uses: u64,
    /// The buffer pinned right after this one last time (learned order, for prefetching).
    next: Option<BufferId>,
    /// Brought back by the service thread and not used since.
    prefetched: bool,
    /// At its last use it had stayed in memory since the use before: part of the set the demand
    /// policy keeps. Prefetching leaves these alone while they fit in the budget minus the
    /// prefetch window (0117 F1).
    kept: bool,
    /// In memory, uncompressed, since its last use.
    stayed: bool,
}

struct Src {
    file: Arc<SourceFile>,
    offset: u64,
    /// Known after the first read; required before the buffer may be dropped (R5).
    digest: Option<Digest>,
}

struct Lineage {
    inputs: Vec<BufferId>,
    compute: Compute,
    /// Digest of the first result: every re-computation must reproduce it (R1).
    digest: Digest,
    /// Measured seconds of one computation.
    seconds: f64,
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

/// Measured throughputs; initial values from E015/E008 (0109 §7).
struct Cost {
    read: Rate,
    compress: Rate,
    decompress: Rate,
}

/// Throughput of one kind of transfer as decayed sums of bytes and seconds, so a large transfer
/// weighs by its size and a small one adds its fixed cost. (A smoothed average of per-transfer
/// rates swung with the last transfer's size: with many small weights next to large ones it
/// was off by up to 5x, 0115.)
#[derive(Clone, Copy)]
struct Rate {
    bytes: f64,
    seconds: f64,
}

/// Weight kept per new transfer (about the last 50 transfers count).
const DECAY: f64 = 0.98;

impl Rate {
    /// A prior worth one 16 MiB transfer at `bytes_per_second`.
    fn assumed(bytes_per_second: f64) -> Rate {
        let bytes = 16.0 * 1024.0 * 1024.0;
        Rate {
            bytes,
            seconds: bytes / bytes_per_second,
        }
    }

    fn per_second(&self) -> f64 {
        self.bytes / self.seconds
    }

    fn add(&mut self, bytes: usize, seconds: f64) {
        if seconds <= 0.0 || bytes == 0 {
            return;
        }
        self.bytes = DECAY * self.bytes + bytes as f64;
        self.seconds = DECAY * self.seconds + seconds;
    }
}

enum Action {
    Drop,
    Compress,
}

impl Buf {
    fn new(nbytes: usize, elem: usize, data: Data, clock: u64) -> Buf {
        Buf {
            nbytes,
            elem,
            data,
            source: None,
            lineage: None,
            dependents: Vec::new(),
            dirty: false,
            pins: 0,
            writers: 0,
            busy: false,
            incompressible: false,
            ratio: None,
            last: clock,
            period: 0.0,
            uses: 0,
            next: None,
            prefetched: false,
            kept: false,
            stayed: false,
        }
    }

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

    /// Can be dropped and brought back without loss.
    fn recoverable(&self) -> bool {
        self.source_valid() || self.lineage.is_some()
    }

    fn state(&self) -> BufferState {
        match &self.data {
            Data::Resident(_) => BufferState::Resident,
            Data::Compressed(_) => BufferState::Compressed,
            Data::Empty if self.recoverable() => BufferState::Dropped,
            Data::Empty => BufferState::Unloaded,
        }
    }
}

fn elem_ok(elem: usize) -> bool {
    matches!(elem, 1 | 2 | 4 | 8 | 16)
}

fn mib(n: u64) -> String {
    format!("{:.2} MiB", n as f64 / (1u64 << 20) as f64)
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

fn unknown(id: BufferId) -> Error {
    Error::InvalidArgument(format!("unknown buffer {id}"))
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
        let prefetch = config.prefetch;
        let shared = Arc::new(Shared {
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
                    read: Rate::assumed(1.5e9),
                    compress: Rate::assumed(4e8),
                    decompress: Rate::assumed(1.2e9),
                },
                last_pinned: None,
                prev_pinned: None,
                pairs: HashMap::new(),
                queue: VecDeque::new(),
                queued: HashSet::new(),
                trace: VecDeque::new(),
                stop: false,
            }),
            cond: Condvar::new(),
        });
        let service = if prefetch {
            let s = shared.clone();
            Some(
                std::thread::Builder::new()
                    .name("memopro-rt".into())
                    .spawn(move || s.serve())
                    .map_err(Error::Io)?,
            )
        } else {
            None
        };
        Ok(Runtime {
            _owner: Arc::new(Owner {
                shared: shared.clone(),
                service: Mutex::new(service),
            }),
            shared,
        })
    }

    pub fn config(&self) -> &Config {
        &self.shared.config
    }

    /// Bytes buffers may occupy: the budget minus the compression headroom.
    pub fn limit(&self) -> u64 {
        self.shared.limit()
    }

    /// A new zero-filled buffer of `nbytes` in memory (no original file: it can only be
    /// compressed to make room). `elem` is the element size used to shuffle bytes before
    /// compression (1, 2, 4, 8 or 16).
    pub fn alloc(&self, nbytes: usize, elem: usize) -> Result<BufferId> {
        check_size(nbytes, elem)?;
        let cap = round_to_pages(nbytes) as u64;
        let sh = &self.shared;
        let st = sh.lock();
        let mut st = sh.make_room(st, cap, None, None)?;
        let region = Region::new(nbytes)?;
        st.used += cap;
        st.stats.peak_used = st.stats.peak_used.max(st.used);
        let id = st.next_id;
        st.next_id += 1;
        let clock = st.clock;
        st.bufs
            .insert(id, Buf::new(nbytes, elem, Data::Resident(region), clock));
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
        let mut st = self.shared.lock();
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
        let mut b = Buf::new(nbytes, elem, Data::Empty, clock);
        b.source = Some(Src {
            file,
            offset,
            digest: None,
        });
        st.bufs.insert(id, b);
        Ok(id)
    }

    /// A buffer of `nbytes` computed by `compute` from `inputs`, now. Afterwards it can be
    /// dropped and re-computed (as long as the inputs are unchanged); every re-computation is
    /// checked against the digest of this first result, so a non-deterministic `compute` makes
    /// the re-computation fail with [`Error::Integrity`] rather than return other data.
    pub fn derive(
        &self,
        inputs: &[BufferId],
        nbytes: usize,
        elem: usize,
        compute: Compute,
    ) -> Result<BufferId> {
        check_size(nbytes, elem)?;
        {
            let st = self.shared.lock();
            if let Some(&missing) = inputs.iter().find(|i| !st.bufs.contains_key(i)) {
                return Err(unknown(missing));
            }
        }
        let id = self.alloc(nbytes, elem)?;
        let made = (|| {
            let pins = inputs
                .iter()
                .map(|&i| self.shared.pin_with(&self.shared, i, false, false, false))
                .collect::<Result<Vec<Pin>>>()?;
            let mut out = self.shared.pin_with(&self.shared, id, true, true, false)?;
            let slices: Vec<&[u8]> = pins.iter().map(Pin::as_slice).collect();
            let t0 = Instant::now();
            compute(&slices, out.as_mut_slice()?)?;
            let seconds = t0.elapsed().as_secs_f64();
            Ok((digest(out.as_slice()), seconds))
        })();
        let (result_digest, seconds) = match made {
            Ok(x) => x,
            Err(e) => {
                let _ = self.free(id);
                return Err(e);
            }
        };
        let mut st = self.shared.lock();
        let b = st.bufs.get_mut(&id).expect("just made");
        b.dirty = false;
        b.lineage = Some(Lineage {
            inputs: inputs.to_vec(),
            compute,
            digest: result_digest,
            seconds,
        });
        for i in inputs {
            if let Some(ib) = st.bufs.get_mut(i) {
                ib.dependents.push(id);
            }
        }
        Ok(id)
    }

    /// Make the buffer resident and keep it there until the returned [`Pin`] is dropped. A
    /// writable pin needs the buffer unpinned and makes its original file unusable for it;
    /// while it lives no other pin of the buffer can be taken (so its slices never alias).
    pub fn pin(&self, id: BufferId, write: bool) -> Result<Pin> {
        self.shared.pin_with(&self.shared, id, write, true, true)
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
        self.shared.pin_with(&self.shared, id, write, false, true)
    }

    /// Like [`Runtime::pin_shared`], but not counted as a use: no effect on reuse periods, the
    /// learned order, the prediction trace or prefetching. For views of a recipe's inputs.
    ///
    /// # Safety
    ///
    /// As for [`Runtime::pin_shared`].
    pub unsafe fn pin_untracked(&self, id: BufferId, write: bool) -> Result<Pin> {
        self.shared.pin_with(&self.shared, id, write, false, false)
    }

    /// Ask the service thread to bring the buffer back now (a hint; no effect without prefetch).
    pub fn prefetch(&self, id: BufferId) -> Result<()> {
        let mut st = self.shared.lock();
        if !st.bufs.contains_key(&id) {
            return Err(unknown(id));
        }
        if self.shared.config.prefetch && st.queued.insert(id) {
            st.queue.push_back((id, 1.0));
            drop(st);
            self.shared.cond.notify_all();
        }
        Ok(())
    }

    /// Give up the buffer and its memory. Fails while it is pinned, and while a buffer derived
    /// from it is not in memory (it would have no way back).
    pub fn free(&self, id: BufferId) -> Result<()> {
        let sh = &self.shared;
        let mut st = sh.lock();
        loop {
            let b = st.bufs.get(&id).ok_or_else(|| unknown(id))?;
            if b.busy {
                st = sh.wait(st);
                continue;
            }
            if b.pins > 0 {
                return Err(Error::InvalidArgument(format!("buffer {id} is pinned")));
            }
            break;
        }
        sh.release_dependents(&mut st, id)?;
        let b = st.bufs.remove(&id).expect("checked above");
        st.used -= b.accounted();
        if let Some(l) = &b.lineage {
            for i in &l.inputs {
                if let Some(ib) = st.bufs.get_mut(i) {
                    ib.dependents.retain(|d| *d != id);
                }
            }
        }
        st.queued.remove(&id);
        st.pairs.retain(|&(a, c), n| a != id && c != id && *n != id);
        drop(b);
        sh.cond.notify_all();
        Ok(())
    }

    /// Give up the buffer's memory now if that is possible without losing it (drop when it can
    /// be re-read or re-computed, else compress). Returns whether anything was freed.
    pub fn evict(&self, id: BufferId) -> Result<bool> {
        let sh = &self.shared;
        let mut st = sh.lock();
        loop {
            let b = st.bufs.get(&id).ok_or_else(|| unknown(id))?;
            if b.busy {
                st = sh.wait(st);
                continue;
            }
            if b.pins > 0 {
                return Ok(false);
            }
            break;
        }
        let level = sh.config.compress_level;
        let min_saving = sh.config.min_saving;
        let b = st.bufs.get_mut(&id).expect("checked above");
        let recoverable = b.recoverable();
        let action = match (&b.data, recoverable) {
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
                sh.drop_data(&mut st, id);
                Ok(true)
            }
            Some(Action::Compress) => {
                let before = st.used;
                let st = sh.compress(st, id)?;
                Ok(st.used < before)
            }
            None => Ok(false),
        }
    }

    pub fn state(&self, id: BufferId) -> Result<BufferState> {
        let st = self.shared.lock();
        st.bufs.get(&id).map(Buf::state).ok_or_else(|| unknown(id))
    }

    /// Size in bytes of a buffer.
    pub fn nbytes(&self, id: BufferId) -> Result<usize> {
        let st = self.shared.lock();
        st.bufs
            .get(&id)
            .map(|b| b.nbytes)
            .ok_or_else(|| unknown(id))
    }

    pub fn stats(&self) -> Stats {
        let st = self.shared.lock();
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

    /// Predicted cost of repeating the last recorded cycle of pins (from one use of the most
    /// recently pinned buffer to the next): bytes that must come back per cycle under this
    /// budget and the time that takes. `None` until a buffer has been pinned twice.
    pub fn predict(&self) -> Option<Prediction> {
        let st = self.shared.lock();
        let trace = &st.trace;
        let last = trace.back()?;
        let start = trace
            .iter()
            .rev()
            .skip(1)
            .position(|t| t.id == last.id)
            .map(|k| trace.len() - 2 - k)?;
        let cycle: Vec<&Trace> = trace.iter().skip(start + 1).collect();
        let last_seconds = last.at.duration_since(trace[start].at).as_secs_f64();
        let waited: f64 = cycle.iter().map(|t| t.waited).sum();
        let mut items = Vec::with_capacity(cycle.len());
        let mut distinct = HashSet::new();
        let mut cycle_bytes = 0u64;
        for t in &cycle {
            let Some(b) = st.bufs.get(&t.id) else {
                continue;
            };
            let bytes = round_to_pages(b.nbytes) as u64;
            if distinct.insert(t.id) {
                cycle_bytes += bytes;
            }
            let n = b.nbytes as f64;
            let restore_seconds = if b.source.is_some() && !b.dirty {
                n / st.cost.read.per_second()
            } else if let Some(l) = &b.lineage {
                l.seconds
            } else {
                n / st.cost.decompress.per_second()
            };
            items.push(predict::Item {
                id: t.id,
                bytes,
                restore_seconds,
            });
        }
        let prefetch = self.shared.config.prefetch;
        let lookahead = if prefetch { self.shared.window() } else { 0 };
        let capacity = self.shared.limit().saturating_sub(lookahead);
        let (restore_bytes, restore_seconds) = predict::steady_misses(&items, capacity);
        let compute_seconds = (last_seconds - waited).max(0.0);
        let seconds = if prefetch {
            compute_seconds.max(restore_seconds)
        } else {
            compute_seconds + restore_seconds
        };
        Some(Prediction {
            cycle_pins: items.len() as u64,
            cycle_bytes,
            restore_bytes,
            restore_seconds,
            compute_seconds,
            seconds,
            last_seconds,
            prefetch,
        })
    }
}

impl Shared {
    fn limit(&self) -> u64 {
        self.config.budget - self.reserve
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn wait<'a>(&'a self, st: MutexGuard<'a, Inner>) -> MutexGuard<'a, Inner> {
        self.cond.wait(st).unwrap_or_else(|e| e.into_inner())
    }

    /// Pin a buffer; `record` counts it as a use (reuse periods, learned order, prediction
    /// trace, prefetching). Internal pins (inputs of a re-computation, views handed to a
    /// recipe) are not uses of their own.
    fn pin_with(
        &self,
        me: &Arc<Shared>,
        id: BufferId,
        write: bool,
        exclusive: bool,
        record: bool,
    ) -> Result<Pin> {
        let t0 = Instant::now();
        let mut st = self.lock();
        loop {
            let b = st.bufs.get(&id).ok_or_else(|| unknown(id))?;
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
            let has_lineage = b.lineage.is_some();
            match b.state() {
                BufferState::Resident => break,
                BufferState::Compressed => st = self.restore_packed(st, id, None)?,
                BufferState::Dropped if has_lineage => st = self.restore_lineage(me, st, id)?,
                BufferState::Dropped | BufferState::Unloaded => {
                    st = self.restore_source(st, id, None)?
                }
            }
        }
        if write {
            self.release_dependents(&mut st, id)?;
            self.forget_lineage(&mut st, id);
        }
        let waited = t0.elapsed().as_secs_f64();
        if record {
            st.clock += 1;
            st.stats.pins += 1;
            st.stats.restore_seconds += waited;
            let now = st.clock;
            if st.last_pinned != Some(id) {
                if let Some(prev) = st.last_pinned {
                    if let Some(pb) = st.bufs.get_mut(&prev) {
                        pb.next = Some(id);
                    }
                    if let Some(before) = st.prev_pinned {
                        if st.pairs.len() >= 4 * st.bufs.len() + 1024 {
                            st.pairs.clear(); // no stable order: do not grow without bound
                        }
                        st.pairs.insert((before, prev), id);
                    }
                }
                st.prev_pinned = st.last_pinned;
                st.last_pinned = Some(id);
            }
            if st.trace.len() == TRACE_LEN {
                st.trace.pop_front();
            }
            st.trace.push_back(Trace {
                id,
                at: Instant::now(),
                waited,
            });
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
            b.kept = b.stayed;
            b.stayed = true;
            if std::mem::take(&mut b.prefetched) {
                st.stats.prefetch_hits += 1;
            }
        }
        let b = st.bufs.get_mut(&id).expect("checked above");
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
        let pin = Pin {
            shared: me.clone(),
            id,
            ptr: region.as_ptr() as usize,
            len: b.nbytes,
            write,
            exclusive,
        };
        if record && self.config.prefetch {
            self.schedule(&mut st, id);
        }
        Ok(pin)
    }

    /// Bytes the service thread may bring in ahead of use: `lookahead`, at most a quarter of
    /// the budget (a larger window only evicts what it brought in).
    fn window(&self) -> u64 {
        self.config.lookahead.min(self.limit() / 4).max(1)
    }

    /// Queue the buffers expected after `from` (learned order: by the pair of the previous and
    /// the current buffer when known, else by the current one) that are not in memory, within
    /// the next `window` bytes of use. Every buffer ahead counts toward the window, in memory or
    /// not: counting only the missing ones let the prefetcher run up to 64 buffers ahead and fill
    /// the budget with future buffers, evicting the ones kept for the next pass (E026b, 0119).
    fn schedule(&self, st: &mut Inner, from: BufferId) {
        let window = self.window();
        let mut prev = st.prev_pinned;
        let mut cur = from;
        let mut bytes = 0u64;
        let mut steps = 0.0;
        let mut seen = HashSet::new();
        let mut pushed = false;
        loop {
            let by_pair = prev.and_then(|p| st.pairs.get(&(p, cur)).copied());
            let Some(n) = by_pair.or_else(|| st.bufs.get(&cur).and_then(|b| b.next)) else {
                break;
            };
            if n == from || !seen.insert(n) || steps >= 64.0 {
                break;
            }
            steps += 1.0;
            let Some(b) = st.bufs.get(&n) else {
                break;
            };
            bytes += round_to_pages(b.nbytes) as u64;
            let wanted = !matches!(b.state(), BufferState::Resident) && b.lineage.is_none();
            if wanted && !b.busy && st.queued.insert(n) {
                st.queue.push_back((n, steps));
                pushed = true;
            }
            if bytes >= window {
                break;
            }
            prev = Some(cur);
            cur = n;
        }
        if pushed {
            self.cond.notify_all();
        }
    }

    /// The service thread: bring queued buffers back until the runtime is dropped.
    fn serve(self: Arc<Self>) {
        let mut st = self.lock();
        loop {
            if st.stop {
                return;
            }
            let Some((id, distance)) = st.queue.pop_front() else {
                st = self.wait(st);
                continue;
            };
            st.queued.remove(&id);
            st = self.prefetch_one(st, id, distance);
        }
    }

    fn prefetch_one<'a>(
        &'a self,
        st: MutexGuard<'a, Inner>,
        id: BufferId,
        distance: f64,
    ) -> MutexGuard<'a, Inner> {
        let Some(b) = st.bufs.get(&id) else {
            return st;
        };
        if b.busy || b.pins > 0 || b.lineage.is_some() {
            return st;
        }
        let nbytes = b.nbytes as u64;
        let result = match b.state() {
            BufferState::Resident => return st,
            BufferState::Compressed => self.restore_packed(st, id, Some(distance)),
            BufferState::Dropped | BufferState::Unloaded => {
                self.restore_source(st, id, Some(distance))
            }
        };
        match result {
            Ok(mut st) => {
                if let Some(b) = st.bufs.get_mut(&id) {
                    b.prefetched = true;
                }
                st.stats.prefetches += 1;
                st.stats.prefetch_bytes += nbytes;
                st
            }
            Err(e) => {
                let mut st = self.lock();
                if matches!(e, Error::Budget(_)) {
                    st.stats.prefetch_skipped += 1;
                }
                st
            }
        }
    }

    /// Before a buffer changes (writable pin) or goes away (free): buffers derived from it must
    /// not depend on it any more. Fails if one of them is not in memory (it would be lost).
    fn release_dependents(&self, st: &mut Inner, id: BufferId) -> Result<()> {
        let deps = st
            .bufs
            .get(&id)
            .map(|b| b.dependents.clone())
            .unwrap_or_default();
        let dropped: Vec<BufferId> = deps
            .iter()
            .copied()
            .filter(|d| {
                st.bufs
                    .get(d)
                    .is_some_and(|b| b.lineage.is_some() && matches!(b.data, Data::Empty))
            })
            .collect();
        if !dropped.is_empty() {
            return Err(Error::InvalidArgument(format!(
                "buffer {id} is needed to re-compute buffers {dropped:?}, which are not in \
                 memory; use them (pin) or free them first"
            )));
        }
        for d in deps {
            if let Some(b) = st.bufs.get_mut(&d) {
                b.lineage = None;
            }
        }
        if let Some(b) = st.bufs.get_mut(&id) {
            b.dependents.clear();
        }
        Ok(())
    }

    /// The buffer no longer is what its recipe computes (it is being written).
    fn forget_lineage(&self, st: &mut Inner, id: BufferId) {
        let Some(l) = st.bufs.get_mut(&id).and_then(|b| b.lineage.take()) else {
            return;
        };
        for i in l.inputs {
            if let Some(ib) = st.bufs.get_mut(&i) {
                ib.dependents.retain(|d| *d != id);
            }
        }
    }

    // ------------------------------------------------------------------ room

    /// Make room for `need` more accounted bytes, giving up other buffers (never `exclude`,
    /// never pinned or busy ones). With `bound`, only buffers predicted to be used later than
    /// `bound` pins from now may go, and failing costs nothing (prefetching). Fails with
    /// [`Error::Budget`] when nothing more can go.
    fn make_room<'a>(
        &'a self,
        mut st: MutexGuard<'a, Inner>,
        need: u64,
        exclude: Option<BufferId>,
        bound: Option<f64>,
    ) -> Result<MutexGuard<'a, Inner>> {
        let limit = self.limit();
        if need > limit {
            if bound.is_none() {
                st.stats.refusals += 1;
            }
            return Err(Error::Budget(format!(
                "one buffer needs {}, more than the budget {} minus the runtime's {} headroom",
                mib(need),
                mib(self.config.budget),
                mib(self.reserve)
            )));
        }
        loop {
            if st.used + need <= limit {
                return Ok(st);
            }
            match self.choose(&mut st, exclude, bound) {
                Some((id, Action::Drop)) => self.drop_data(&mut st, id),
                Some((id, Action::Compress)) => st = self.compress(st, id)?,
                None if bound.is_some() => {
                    return Err(Error::Budget("prefetch would evict sooner needs".into()));
                }
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
    fn choose(
        &self,
        st: &mut Inner,
        exclude: Option<BufferId>,
        bound: Option<f64>,
    ) -> Option<(BufferId, Action)> {
        let now = st.clock as f64;
        let policy = self.config.policy;
        let level = self.config.compress_level;
        let min_saving = self.config.min_saving;
        let (read, compress, decompress) = (
            st.cost.read.per_second(),
            st.cost.compress.per_second(),
            st.cost.decompress.per_second(),
        );
        // A buffer whose period is not measured yet is assumed to recur like the others do on
        // average; with no period known at all, staleness (LRU) is the only guide.
        let (sum, count) = st
            .bufs
            .values()
            .filter(|b| b.period > 0.0)
            .fold((0.0, 0u64), |(s, c), b| (s + b.period, c + 1));
        let typical = if count > 0 { sum / count as f64 } else { 0.0 };
        // restore cost of a buffer that is not resident now (for lineage inputs)
        let restore_cost: HashMap<BufferId, f64> = st
            .bufs
            .iter()
            .filter(|(_, b)| !matches!(b.data, Data::Resident(_)))
            .map(|(&id, b)| {
                let n = b.nbytes as f64;
                let c = match (&b.data, &b.lineage) {
                    (Data::Compressed(_), _) => n / decompress,
                    (_, Some(l)) => l.seconds,
                    _ => n / read,
                };
                (id, c)
            })
            .collect();
        // Prefetching takes room from the buffers the demand policy keeps only when they hold
        // more than the budget minus the window. Otherwise a prefetcher running ahead of the
        // computation evicts what would have been hits, and the kept set wears away pass after
        // pass (E026 P1, 0116); this splits memory into a kept part and a prefetch window.
        let protect_kept = bound.is_some() && {
            let kept: u64 = st
                .bufs
                .values()
                .filter(|b| b.kept)
                .map(Buf::accounted)
                .sum();
            kept + self.window() <= self.limit()
        };
        let mut best: Option<(f64, BufferId, Action)> = None;
        let mut newly_incompressible = 0;
        for (&id, b) in st.bufs.iter_mut() {
            if Some(id) == exclude || b.pins > 0 || b.busy || (protect_kept && b.kept) {
                continue;
            }
            let staleness = now - b.last as f64 + 1.0;
            let period = if b.period > 0.0 { b.period } else { typical };
            let distance = match policy {
                // brought in for a use that has not come yet: about to be used
                Policy::ReuseDistance if b.prefetched => 1.0,
                Policy::ReuseDistance if period > 0.0 => {
                    (b.last as f64 + period - now).max(0.0) + 1.0
                }
                Policy::Lru | Policy::ReuseDistance => staleness,
            };
            // a prefetch never gives up what is needed sooner, nor another prefetch not used yet
            if bound.is_some_and(|d| distance <= d || b.prefetched) {
                continue;
            }
            let n = b.nbytes as f64;
            let source_ok = b.source_valid();
            let recompute_cost = b.lineage.as_ref().map(|l| {
                l.seconds
                    + l.inputs
                        .iter()
                        .map(|i| restore_cost.get(i).copied().unwrap_or(0.0))
                        .sum::<f64>()
            });
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
                    if source_ok {
                        consider(n / read, cap, Action::Drop);
                    }
                    if let Some(c) = recompute_cost {
                        consider(c, cap, Action::Drop);
                    }
                    if !source_ok && !b.incompressible {
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
                Data::Compressed(p) => {
                    if source_ok {
                        let extra = (n / read - n / decompress).max(0.1 * n / read);
                        consider(extra, p.stored as f64, Action::Drop);
                    } else if let Some(c) = recompute_cost {
                        let extra = (c - n / decompress).max(0.1 * c);
                        consider(extra, p.stored as f64, Action::Drop);
                    }
                }
                Data::Empty => {}
            }
        }
        st.stats.incompressible += newly_incompressible;
        best.map(|(_, id, action)| (id, action))
    }

    /// Forget the data of a buffer that can be re-read or re-computed.
    fn drop_data(&self, st: &mut Inner, id: BufferId) {
        let b = st.bufs.get_mut(&id).expect("chosen from the table");
        let freed = b.accounted();
        let old = std::mem::replace(&mut b.data, Data::Empty);
        let wasted = std::mem::take(&mut b.prefetched);
        b.kept = false;
        b.stayed = false;
        st.used -= freed;
        st.stats.drops += 1;
        st.stats.drop_bytes += freed;
        st.stats.evictions += 1;
        if wasted {
            st.stats.prefetch_wasted += 1;
        }
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
        let level = self.config.compress_level;
        let b = st.bufs.get_mut(&id).expect("chosen from the table");
        let Data::Resident(region) = std::mem::replace(&mut b.data, Data::Empty) else {
            unreachable!("only resident buffers are compressed")
        };
        b.busy = true;
        let wasted = std::mem::take(&mut b.prefetched);
        b.kept = false;
        b.stayed = false;
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
                if wasted {
                    st.stats.prefetch_wasted += 1;
                }
                st.cost.compress.add(nbytes, seconds);
            }
            Err((region, _)) => {
                // the data is whole again in `region`; never try this buffer again
                b.data = Data::Resident(region);
                b.incompressible = true;
                st.stats.incompressible += 1;
            }
        }
        self.cond.notify_all();
        Ok(st)
    }

    /// Bring a dropped or never-read buffer back from its file (outside the lock).
    fn restore_source<'a>(
        &'a self,
        st: MutexGuard<'a, Inner>,
        id: BufferId,
        bound: Option<f64>,
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
        let mut st = self.make_room(st, cap, Some(id), bound)?;
        st.used += cap;
        st.stats.peak_used = st.stats.peak_used.max(st.used);
        st.bufs.get_mut(&id).expect("still there").busy = true;
        drop(st);
        let t0 = Instant::now();
        let result = Region::new(nbytes).and_then(|mut region| {
            let got = file.read_into(
                offset,
                &mut region.as_mut_slice()[..nbytes],
                expected.as_ref(),
            )?;
            Ok((region, got))
        });
        let seconds = t0.elapsed().as_secs_f64();
        let mut st = self.lock();
        let b = st.bufs.get_mut(&id).expect("busy buffers are not freed");
        b.busy = false;
        let outcome = match result {
            Ok((region, got)) => {
                b.data = Data::Resident(region);
                let first = expected.is_none();
                if first {
                    b.source.as_mut().expect("read from it").digest = Some(got);
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
                st.cost.read.add(nbytes, seconds);
                Ok(())
            }
            Err(e) => {
                st.used -= cap;
                Err(e)
            }
        };
        self.cond.notify_all();
        outcome.map(|()| st)
    }

    /// Decompress a compressed buffer (outside the lock); its pieces go when it is whole.
    fn restore_packed<'a>(
        &'a self,
        st: MutexGuard<'a, Inner>,
        id: BufferId,
        bound: Option<f64>,
    ) -> Result<MutexGuard<'a, Inner>> {
        let nbytes = st.bufs.get(&id).expect("checked by the caller").nbytes;
        let cap = round_to_pages(nbytes) as u64;
        let mut st = self.make_room(st, cap, Some(id), bound)?;
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
                st.cost.decompress.add(nbytes, seconds);
                drop(packed);
                Ok(())
            }
            Err(e) => {
                b.data = Data::Compressed(packed);
                st.used -= cap;
                Err(e)
            }
        };
        self.cond.notify_all();
        outcome.map(|()| st)
    }

    /// Re-compute a dropped derived buffer from its inputs (on the caller's thread: `compute`
    /// may be Python), and check the result against the digest of the first computation.
    fn restore_lineage<'a>(
        &'a self,
        me: &Arc<Shared>,
        st: MutexGuard<'a, Inner>,
        id: BufferId,
    ) -> Result<MutexGuard<'a, Inner>> {
        let mut st = st;
        let b = st.bufs.get_mut(&id).expect("checked by the caller");
        let l = b.lineage.as_ref().expect("checked by the caller");
        let (inputs, compute, want) = (l.inputs.clone(), l.compute.clone(), l.digest.clone());
        let nbytes = b.nbytes;
        b.busy = true;
        drop(st);
        let cap = round_to_pages(nbytes) as u64;
        let mut reserved = false;
        let result = (|| {
            let pins = inputs
                .iter()
                .map(|&i| self.pin_with(me, i, false, false, false))
                .collect::<Result<Vec<Pin>>>()?;
            let st = self.lock();
            let mut st = self.make_room(st, cap, Some(id), None)?;
            st.used += cap;
            st.stats.peak_used = st.stats.peak_used.max(st.used);
            reserved = true;
            drop(st);
            let mut region = Region::new(nbytes)?;
            let slices: Vec<&[u8]> = pins.iter().map(Pin::as_slice).collect();
            let t0 = Instant::now();
            compute(&slices, &mut region.as_mut_slice()[..nbytes])?;
            let seconds = t0.elapsed().as_secs_f64();
            if digest(&region.as_slice()[..nbytes]) != want {
                return Err(Error::Integrity(format!(
                    "re-computing buffer {id} gave different data: its computation is not \
                     deterministic or its inputs changed"
                )));
            }
            Ok((region, seconds))
        })();
        let mut st = self.lock();
        let b = st.bufs.get_mut(&id).expect("busy buffers are not freed");
        b.busy = false;
        let outcome = match result {
            Ok((region, seconds)) => {
                b.data = Data::Resident(region);
                if let Some(l) = &mut b.lineage {
                    l.seconds = 0.7 * l.seconds + 0.3 * seconds;
                }
                st.stats.recomputes += 1;
                st.stats.recompute_bytes += nbytes as u64;
                st.stats.recompute_seconds += seconds;
                Ok(())
            }
            Err(e) => {
                if reserved {
                    st.used -= cap;
                }
                Err(e)
            }
        };
        self.cond.notify_all();
        outcome.map(|()| st)
    }
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
        let mut st = self.shared.lock();
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
