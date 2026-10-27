//! memopro core: framework-agnostic memory toolkit for deep learning workloads.
//!
//! The crate knows nothing about PyTorch or any other framework; it works on byte buffers,
//! dtypes and shapes. Python bindings live in the separate `memopro-py` crate.
//!
//! Modules:
//! - [`hwinfo`]: host memory, container limit and disk capacity for the per-pool budget (v0.1)
//! - [`verify`]: reads of original file regions checked by digest (v0.1)
//! - [`codec`]: byte shuffle + zstd for lossless in-RAM compression (v0.1)
//! - [`csv`]: row and key kernels for `memopro.sort` / `memopro.join` (E069)
//! - [`edr`]: bf16 weights as a 1-byte base (a draft) and a 4-bit residual, lossless (0308)
//! - [`rt`]: runtime C-R: large buffers under a hard memory budget, dropped and re-read from
//!   their original files or compressed, never written to disk (0109, 0112)
//!
//! Unbuilt functions return [`Error::NotImplemented`] with the planned milestone; nothing panics.

pub mod codec;
pub mod csv;
pub mod edr;
pub mod error;
pub mod hwinfo;
#[cfg(unix)]
pub mod residency;
pub mod rt;
pub mod verify;

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
