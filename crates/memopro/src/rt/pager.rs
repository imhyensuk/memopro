//! Transparent paging of anonymous memory under a budget (runtime phase 3, 0124; Linux userfaultfd).
//!
//! A [`Pager`] hands out ordinary memory ([`Pager::map`]): code reads and writes it with plain
//! loads and stores and does not know it is managed. Its pages start absent. The first touch of
//! a chunk raises a page fault that the pager's thread serves through `userfaultfd`, filling the
//! chunk with zeros or with its compressed copy. When the chunks in memory would go over the
//! budget, the pager evicts one: it write-protects the chunk (writers wait), compresses it
//! losslessly in memory (byte shuffle + zstd, as the runtime does), and gives its pages back to the
//! OS; the next touch brings it back bit for bit. Nothing is written to disk (0110 N1).
//!
//! The victim is chosen as in the explicit runtime: the chunk whose next fault is predicted
//! farthest, from each chunk's measured period between faults (faults are the clock). Chunks that
//! stay in memory never fault, so they look overdue and stay; the chunk just brought in by a scan
//! is the one that goes (MRU), which keeps a budget's worth of a repeated scan.
//!
//! Limits: Linux uses userfaultfd (kernel 5.7 or newer for write-protecting anonymous memory;
//! unprivileged use needs `UFFD_USER_MODE_ONLY`, 5.11, or `vm.unprivileged_userfaultfd`); macOS
//! serves the same faults through signals (0229, `pager/macos.rs`). Elsewhere [`Pager::new`]
//! returns [`Error::Unsupported`] and the explicit [`Runtime`](super::Runtime) is the way. Kernel accesses (e.g. `read(2)` into a managed buffer)
//! are not served in user-mode-only mode and fail with `EFAULT`; incompressible data cannot be
//! evicted, so a budget full of it is overrun and counted (`overruns`), not refused.

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
use crate::error::{Error, Result};

/// Settings of a [`Pager`].
#[derive(Debug, Clone)]
pub struct PagerConfig {
    /// Bytes the chunks in memory and their compressed copies may occupy.
    pub budget: u64,
    /// Bytes per chunk (a multiple of the page size, at most 4 MiB).
    pub chunk: usize,
    /// Element size the compressor shuffles by (4 suits float32/int32 arrays).
    pub elem: usize,
    /// zstd level (1 = fast).
    pub compress_level: i32,
    /// Smallest fraction compression must save for a chunk to be evicted.
    pub min_saving: f64,
    /// Bytes the whole process may occupy (0189): when set, the chunks' limit shrinks by the
    /// memory the process holds outside the pager (resident set minus the pager's own bytes),
    /// measured whenever room is made.
    pub process_budget: Option<u64>,
}

impl PagerConfig {
    pub fn new(budget: u64) -> Self {
        PagerConfig {
            budget,
            chunk: 1 << 20,
            elem: 4,
            compress_level: 1,
            min_saving: 0.15,
            process_budget: None,
        }
    }
}

/// Counters of a [`Pager`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PagerStats {
    pub budget: u64,
    /// Bytes chunks may occupy: the budget minus one chunk of scratch.
    pub limit: u64,
    pub used: u64,
    pub peak_used: u64,
    pub regions: u64,
    pub mapped_bytes: u64,
    pub resident_bytes: u64,
    pub compressed_bytes: u64,
    /// Page faults served.
    pub faults: u64,
    /// Chunks filled with zeros (first touch).
    pub zero_fills: u64,
    /// Chunks brought back from their compressed copy.
    pub restores: u64,
    pub restore_seconds: f64,
    pub evictions: u64,
    pub compress_in: u64,
    pub compress_out: u64,
    pub compress_seconds: f64,
    /// Chunks found not worth compressing (kept in memory for good).
    pub incompressible: u64,
    /// Fills done although nothing could be evicted (the budget was overrun).
    pub overruns: u64,
    /// Faults on chunks already in memory (a write racing an eviction, or a second waiter).
    pub spurious: u64,
    /// With a process budget: the most memory seen outside the pager, and the lowest limit
    /// the chunks were held to.
    pub outside_peak: u64,
    pub limit_low: u64,
    /// Chunks given up while nothing faulted because memory outside the pager grew (0230).
    pub trims: u64,
}

#[cfg(target_os = "linux")]
mod imp;

#[cfg(target_os = "linux")]
pub use imp::{Pager, in_pager_thread};

#[cfg(target_os = "macos")]
mod macos;

#[cfg(target_os = "macos")]
pub use macos::{Pager, in_pager_thread};

/// Whether the calling thread is a pager's own (always false where there is no pager).
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub fn in_pager_thread() -> bool {
    false
}

/// Transparent paging is only built on Linux (userfaultfd).
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub struct Pager {
    _never: std::convert::Infallible,
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn unsupported() -> Error {
    Error::Unsupported(
        "transparent paging needs Linux userfaultfd; on this system use rt::Runtime buffers \
         (explicit pins)"
            .into(),
    )
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
impl Pager {
    pub fn new(_config: PagerConfig) -> Result<Pager> {
        Err(unsupported())
    }

    pub fn map(&self, _len: usize) -> Result<*mut u8> {
        match self._never {}
    }

    /// # Safety
    /// See the Linux implementation.
    pub unsafe fn unmap(&self, _addr: *mut u8) -> Result<usize> {
        match self._never {}
    }

    pub fn owns(&self, _addr: *const u8) -> Option<usize> {
        match self._never {}
    }

    pub fn contains(&self, _addr: *const u8) -> bool {
        match self._never {}
    }

    pub fn limit(&self) -> u64 {
        match self._never {}
    }

    pub fn stats(&self) -> PagerStats {
        match self._never {}
    }
}

#[cfg(test)]
#[cfg(any(target_os = "linux", target_os = "macos"))]
mod tests;

#[cfg(test)]
mod portable_tests {
    use super::*;
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    use crate::error::Error;

    #[test]
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    fn other_systems_say_why() {
        let e = Pager::new(PagerConfig::new(64 << 20)).err().unwrap();
        assert!(e.to_string().contains("userfaultfd"), "{e}");
    }

    #[test]
    fn defaults_are_sane() {
        let c = PagerConfig::new(1 << 30);
        assert_eq!(c.chunk % 4096, 0);
        assert!(c.chunk <= crate::codec::CHUNK);
        let _ = Error::Unsupported(String::new());
    }
}
