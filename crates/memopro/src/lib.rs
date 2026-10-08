//! memopro core: framework-agnostic memory toolkit for deep learning workloads.
//!
//! The crate knows nothing about PyTorch or any other framework; it works on byte buffers,
//! dtypes and shapes. Python bindings live in the separate `memopro-py` crate.
//!
//! Modules:
//! - [`hwinfo`]: host memory, container limit and disk capacity for the per-pool budget (v0.1)
//! - [`spill`]: write-free restore from original files with digest checks, and spill files (v0.1)
//! - [`ledger`]: bookkeeping of hibernated buffers and SSD bytes written (v0.1)
//! - [`codec`]: byte shuffle + zstd for lossless in-RAM compression (v0.1)
//! - [`pressure`]: OS memory-pressure signal for γ elastic (v0.3)
//! - [`rt`]: runtime C-R: large buffers under a hard memory budget, dropped and re-read from
//!   their original files or compressed, never written to disk (0109, 0112)
//!
//! Unbuilt functions return [`Error::NotImplemented`] with the planned milestone; nothing panics.

pub mod codec;
pub mod error;
pub mod hwinfo;
pub mod ledger;
pub mod pressure;
#[cfg(unix)]
pub mod residency;
pub mod rt;
pub mod spill;

pub use error::{Error, Result};

/// The README's example compiles and runs (`cargo test --doc`), so the crates.io page stays true.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;

/// Version of the memopro core, taken from the crate manifest.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Returns the version of the memopro core.
pub fn version() -> &'static str {
    VERSION
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_matches_manifest() {
        assert_eq!(version(), env!("CARGO_PKG_VERSION"));
        assert!(!version().is_empty());
    }
}
