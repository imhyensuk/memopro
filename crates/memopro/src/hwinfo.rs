//! Host facts for the per-pool budget (architecture §3.1, U7): RAM, swap, container limit, disk.
//!
//! Planned implementation (A1a): the `sysinfo` crate (which reads cgroup v1/v2 limits) plus the
//! missing pieces only (0011 L13). Accelerator facts (CUDA, MPS) come from torch on the Python
//! side, not from here.
//!
//! Status: skeleton.

use crate::error::{Error, Result};
use std::path::{Path, PathBuf};

/// Host memory as the process can really use it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryInfo {
    pub total_bytes: u64,
    pub available_bytes: u64,
    pub swap_total_bytes: u64,
    pub swap_free_bytes: u64,
    /// Container (cgroup) memory limit, if the process runs under one.
    pub cgroup_limit_bytes: Option<u64>,
}

impl MemoryInfo {
    /// Memory the process may use: available RAM, capped by the container limit.
    pub fn usable_bytes(&self) -> u64 {
        match self.cgroup_limit_bytes {
            Some(limit) => self.available_bytes.min(limit),
            None => self.available_bytes,
        }
    }
}

/// Capacity of the file system that holds `path`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiskInfo {
    pub path: PathBuf,
    pub total_bytes: u64,
    pub available_bytes: u64,
}

impl DiskInfo {
    /// Fraction of the disk that is free, in `[0, 1]`.
    pub fn free_fraction(&self) -> f64 {
        if self.total_bytes == 0 {
            return 0.0;
        }
        self.available_bytes as f64 / self.total_bytes as f64
    }
}

/// Snapshot of host memory.
pub fn memory() -> Result<MemoryInfo> {
    Err(Error::not_implemented("hwinfo::memory", "v0.1 A1a"))
}

/// Capacity of the disk that holds `path`.
pub fn disk(_path: &Path) -> Result<DiskInfo> {
    Err(Error::not_implemented("hwinfo::disk", "v0.1 A1a"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usable_memory_respects_container_limit() {
        let mut m = MemoryInfo {
            total_bytes: 16 << 30,
            available_bytes: 10 << 30,
            swap_total_bytes: 0,
            swap_free_bytes: 0,
            cgroup_limit_bytes: None,
        };
        assert_eq!(m.usable_bytes(), 10 << 30);
        m.cgroup_limit_bytes = Some(4 << 30);
        assert_eq!(m.usable_bytes(), 4 << 30);
    }

    #[test]
    fn free_fraction_handles_empty_disk() {
        let d = DiskInfo {
            path: PathBuf::from("/"),
            total_bytes: 0,
            available_bytes: 0,
        };
        assert_eq!(d.free_fraction(), 0.0);
    }

    #[test]
    fn skeleton_reports_not_implemented() {
        assert!(matches!(memory(), Err(Error::NotImplemented { .. })));
        assert!(matches!(
            disk(Path::new("/")),
            Err(Error::NotImplemented { .. })
        ));
    }
}
