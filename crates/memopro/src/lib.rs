//! memopro core: framework-agnostic memory toolkit for deep learning workloads.
//!
//! The crate knows nothing about PyTorch or any other framework; it works on byte buffers,
//! dtypes and shapes. Python bindings live in the separate `memopro-py` crate.
//!
//! Status: early development. Only version information is exposed so far.

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
