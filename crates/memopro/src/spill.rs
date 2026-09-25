//! Spill engine (0027 RS1–RS5, 0032 H1–H4, P5): move tensor bytes between memory and files.
//!
//! Main path (H1): re-read an unchanged tensor from its original file (e.g. safetensors) and
//! verify it with a parallel digest; no SSD write. Last resort: write a spill file, only when the
//! caller's policy allows it (the policy itself lives in Python).
//!
//! Design rules:
//! - RS1: one call does the whole transfer; no bulk-size intermediate objects.
//! - RS2: inputs are borrowed buffers (the bindings take the buffer protocol, never `bytes`).
//! - RS3: one process-wide worker pool with per-thread state.
//! - RS4: extra memory <= `buffers x chunk_bytes` + a constant, whatever the transfer size.
//! - RS5: compute overlaps I/O.
//! - P5: spill files are created 0600 and hold raw bytes plus metadata (never pickle).
//!
//! Status: skeleton. The engine is built after gate Gβ (0032).

use crate::error::{Error, Result};
use std::path::{Path, PathBuf};

const PLANNED: &str = "v0.1 A1b, after gate Gβ";

/// Digest of a buffer. The algorithm (fast, parallel) is chosen in A1b.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Digest(pub Vec<u8>);

/// A byte range in an existing file that holds the original data of a tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceRef {
    pub path: PathBuf,
    pub offset: u64,
    pub len: u64,
}

/// A spill file written by the engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpillFile {
    pub path: PathBuf,
    pub len: u64,
    pub digest: Digest,
}

/// Engine tuning. `threads = 0` means all logical CPUs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineConfig {
    pub threads: usize,
    pub chunk_bytes: usize,
    /// Fixed number of chunk buffers; `0` means `2 x threads`.
    pub buffers: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        EngineConfig {
            threads: 0,
            chunk_bytes: 4 << 20,
            buffers: 0,
        }
    }
}

impl EngineConfig {
    fn resolved_threads(&self) -> usize {
        if self.threads == 0 {
            std::thread::available_parallelism().map_or(1, |n| n.get())
        } else {
            self.threads
        }
    }

    fn resolved_buffers(&self) -> usize {
        if self.buffers == 0 {
            2 * self.resolved_threads()
        } else {
            self.buffers
        }
    }

    /// Upper bound of the extra memory held by transfer buffers (RS4), independent of the
    /// transfer size. Tests compare measured RSS growth against this bound.
    pub fn buffer_bound_bytes(&self) -> usize {
        self.resolved_buffers() * self.chunk_bytes
    }

    fn validate(&self) -> Result<()> {
        if self.chunk_bytes == 0 {
            return Err(Error::InvalidArgument("chunk_bytes must be > 0".into()));
        }
        Ok(())
    }
}

/// The spill engine.
#[derive(Debug)]
pub struct Engine {
    config: EngineConfig,
}

impl Engine {
    pub fn new(config: EngineConfig) -> Result<Self> {
        config.validate()?;
        Ok(Engine { config })
    }

    pub fn config(&self) -> &EngineConfig {
        &self.config
    }

    /// Digest `data` in parallel (used to prove a tensor still equals its source).
    pub fn digest(&self, _data: &[u8]) -> Result<Digest> {
        Err(Error::not_implemented("spill::Engine::digest", PLANNED))
    }

    /// Fill `dst` from an original file region (write-free restore, H1).
    pub fn read_source_into(&self, _source: &SourceRef, _dst: &mut [u8]) -> Result<()> {
        Err(Error::not_implemented(
            "spill::Engine::read_source_into",
            PLANNED,
        ))
    }

    /// Write `src` to a new spill file in `dir` (last resort; the caller checked the policy).
    pub fn write(&self, _src: &[u8], _dir: &Path) -> Result<SpillFile> {
        Err(Error::not_implemented("spill::Engine::write", PLANNED))
    }

    /// Restore a spill file into `dst`, verifying its digest.
    pub fn read_into(&self, _file: &SpillFile, _dst: &mut [u8]) -> Result<()> {
        Err(Error::not_implemented("spill::Engine::read_into", PLANNED))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffer_bound_does_not_depend_on_transfer_size() {
        let cfg = EngineConfig {
            threads: 4,
            chunk_bytes: 1 << 20,
            buffers: 0,
        };
        assert_eq!(cfg.buffer_bound_bytes(), 8 << 20);
        let fixed = EngineConfig { buffers: 3, ..cfg };
        assert_eq!(fixed.buffer_bound_bytes(), 3 << 20);
    }

    #[test]
    fn zero_chunk_is_rejected() {
        let cfg = EngineConfig {
            chunk_bytes: 0,
            ..EngineConfig::default()
        };
        assert!(matches!(Engine::new(cfg), Err(Error::InvalidArgument(_))));
    }

    #[test]
    fn skeleton_reports_not_implemented() {
        let engine = Engine::new(EngineConfig::default()).unwrap();
        assert!(matches!(
            engine.digest(b"abc"),
            Err(Error::NotImplemented { .. })
        ));
        assert!(matches!(
            engine.write(b"abc", Path::new(".")),
            Err(Error::NotImplemented { .. })
        ));
    }
}
