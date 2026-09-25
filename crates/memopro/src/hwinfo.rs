//! Host facts for the per-pool budget (architecture §3.1, U7): RAM, swap, container limit, disk.
//!
//! Memory comes from the `sysinfo` crate, which also reads cgroup v1/v2 limits (0011 L13); disk
//! capacity from `statvfs`. Accelerator facts (CUDA, MPS) come from torch on the Python side.
//!
//! **Available memory is conservative (0035):** `total - used`, where "used" is memory the OS
//! cannot hand out without compressing or swapping someone: app (anonymous) memory, wired memory
//! and the compressor on macOS (as Activity Monitor counts it), `MemTotal - MemAvailable` on Linux.
//! macOS' own estimate also counts active anonymous pages of other apps as available; it is kept
//! as `kernel_available_bytes` for reference but never used for budgets.

use crate::error::{Error, Result};
use std::path::{Path, PathBuf};
use sysinfo::{
    CpuRefreshKind, MemoryRefreshKind, ProcessRefreshKind, ProcessesToUpdate, RefreshKind, System,
};

/// Memory limit of the container (cgroup) the process runs in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CgroupMemory {
    pub limit_bytes: u64,
    /// `limit - usage`, i.e. what the process may still allocate inside the container.
    pub free_bytes: u64,
}

/// Host memory as the process can really use it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryInfo {
    pub total_bytes: u64,
    /// Conservative: memory obtainable without compressing or swapping anything.
    pub available_bytes: u64,
    /// The operating system's own estimate (optimistic on macOS); reference only.
    pub kernel_available_bytes: u64,
    pub swap_total_bytes: u64,
    pub swap_free_bytes: u64,
    /// Present only when a container limit below physical memory applies.
    pub cgroup: Option<CgroupMemory>,
}

impl MemoryInfo {
    /// Memory the process may use: available RAM, capped by what the container still allows.
    pub fn usable_bytes(&self) -> u64 {
        match self.cgroup {
            Some(c) => self.available_bytes.min(c.free_bytes),
            None => self.available_bytes,
        }
    }

    pub fn swap_used_bytes(&self) -> u64 {
        self.swap_total_bytes.saturating_sub(self.swap_free_bytes)
    }
}

/// CPU facts shown by `doctor`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuInfo {
    pub brand: String,
    pub logical_cpus: usize,
}

/// Capacity of the file system that holds `path`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiskInfo {
    pub path: PathBuf,
    pub total_bytes: u64,
    /// Space available to unprivileged users (`f_bavail`); purgeable space is not counted.
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

fn cgroup_from(total: u64, limit: u64, free: u64) -> Option<CgroupMemory> {
    // sysinfo reports the host size when the hierarchy has no real limit.
    (limit < total).then_some(CgroupMemory {
        limit_bytes: limit,
        free_bytes: free.min(limit),
    })
}

/// Snapshot of host memory.
pub fn memory() -> Result<MemoryInfo> {
    let sys = System::new_with_specifics(
        RefreshKind::nothing().with_memory(MemoryRefreshKind::everything()),
    );
    let total = sys.total_memory();
    if total == 0 {
        return Err(Error::InvalidArgument(
            "the operating system reported 0 bytes of memory".into(),
        ));
    }
    let cgroup = sys
        .cgroup_limits()
        .and_then(|c| cgroup_from(total, c.total_memory, c.free_memory));
    Ok(MemoryInfo {
        total_bytes: total,
        available_bytes: total.saturating_sub(sys.used_memory()),
        kernel_available_bytes: sys.available_memory(),
        swap_total_bytes: sys.total_swap(),
        swap_free_bytes: sys.free_swap(),
        cgroup,
    })
}

/// Resident set size of the current process, in bytes (used to measure reclaimed memory).
pub fn process_rss() -> Result<u64> {
    let pid = sysinfo::get_current_pid().map_err(|e| Error::InvalidArgument(e.to_string()))?;
    let mut sys = System::new();
    sys.refresh_processes_specifics(
        ProcessesToUpdate::Some(&[pid]),
        false,
        ProcessRefreshKind::nothing().with_memory(),
    );
    sys.process(pid)
        .map(|p| p.memory())
        .ok_or_else(|| Error::InvalidArgument("current process not found".into()))
}

/// CPU brand and logical CPU count.
pub fn cpu() -> CpuInfo {
    let sys =
        System::new_with_specifics(RefreshKind::nothing().with_cpu(CpuRefreshKind::nothing()));
    CpuInfo {
        brand: sys
            .cpus()
            .first()
            .map(|c| c.brand().trim().to_string())
            .unwrap_or_default(),
        logical_cpus: sys.cpus().len(),
    }
}

/// Capacity of the file system that holds `path` (which must exist).
#[cfg(unix)]
#[allow(clippy::useless_conversion)] // statvfs field widths differ between platforms
pub fn disk(path: &Path) -> Result<DiskInfo> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let c_path = CString::new(path.as_os_str().as_bytes())
        .map_err(|_| Error::InvalidArgument(format!("path contains a NUL byte: {path:?}")))?;
    let mut st = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    // SAFETY: `c_path` is a valid NUL-terminated string and `st` points to writable memory of
    // the right type; statvfs fully initialises it when it returns 0.
    let rc = unsafe { libc::statvfs(c_path.as_ptr(), st.as_mut_ptr()) };
    if rc != 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    // SAFETY: rc == 0, so the struct was initialised.
    let st = unsafe { st.assume_init() };
    let block = u64::from(st.f_frsize);
    Ok(DiskInfo {
        path: path.to_path_buf(),
        total_bytes: u64::from(st.f_blocks).saturating_mul(block),
        available_bytes: u64::from(st.f_bavail).saturating_mul(block),
    })
}

/// Capacity of the file system that holds `path` (not implemented on this platform yet).
#[cfg(not(unix))]
pub fn disk(_path: &Path) -> Result<DiskInfo> {
    Err(Error::not_implemented("hwinfo::disk on this OS", "v0.1.x"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(available: u64, cgroup: Option<CgroupMemory>) -> MemoryInfo {
        MemoryInfo {
            total_bytes: 16 << 30,
            available_bytes: available,
            kernel_available_bytes: available,
            swap_total_bytes: 4 << 30,
            swap_free_bytes: 3 << 30,
            cgroup,
        }
    }

    #[test]
    fn usable_memory_respects_container_headroom() {
        assert_eq!(info(10 << 30, None).usable_bytes(), 10 << 30);
        let limited = info(
            10 << 30,
            Some(CgroupMemory {
                limit_bytes: 4 << 30,
                free_bytes: 1 << 30,
            }),
        );
        assert_eq!(limited.usable_bytes(), 1 << 30);
        assert_eq!(limited.swap_used_bytes(), 1 << 30);
    }

    #[test]
    fn host_sized_cgroup_is_not_a_limit() {
        assert_eq!(cgroup_from(8 << 30, 8 << 30, 5 << 30), None);
        assert_eq!(
            cgroup_from(8 << 30, 512 << 20, 600 << 20),
            Some(CgroupMemory {
                limit_bytes: 512 << 20,
                free_bytes: 512 << 20
            })
        );
    }

    #[test]
    fn memory_snapshot_is_consistent() {
        let m = memory().unwrap();
        assert!(m.total_bytes > 0);
        assert!(m.available_bytes <= m.total_bytes);
        assert!(m.kernel_available_bytes <= m.total_bytes);
        assert!(m.swap_free_bytes <= m.swap_total_bytes);
        assert!(m.usable_bytes() <= m.available_bytes);
    }

    #[test]
    fn process_rss_grows_with_touched_memory() {
        let before = process_rss().unwrap();
        let block = vec![1u8; 64 << 20];
        let after = process_rss().unwrap();
        assert!(before > 0);
        assert!(after >= before + (32 << 20), "{before} -> {after}");
        drop(block);
    }

    #[test]
    fn cpu_count_is_positive() {
        assert!(cpu().logical_cpus > 0);
    }

    #[cfg(unix)]
    #[test]
    fn disk_of_root_and_missing_path() {
        let d = disk(Path::new("/")).unwrap();
        assert!(d.total_bytes > 0 && d.available_bytes <= d.total_bytes);
        assert!((0.0..=1.0).contains(&d.free_fraction()));
        assert!(matches!(
            disk(Path::new("/definitely/missing/path")),
            Err(Error::Io(_))
        ));
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

    /// Run inside a memory-limited Linux container, e.g.
    /// `docker run --memory=512m -e MEMOPRO_EXPECT_CGROUP_LIMIT=536870912 ... cargo test -- --ignored`.
    #[test]
    #[ignore = "needs a memory-limited container"]
    fn cgroup_limit_matches_container() {
        let expected: u64 = std::env::var("MEMOPRO_EXPECT_CGROUP_LIMIT")
            .expect("set MEMOPRO_EXPECT_CGROUP_LIMIT")
            .parse()
            .unwrap();
        let m = memory().unwrap();
        let c = m.cgroup.expect("no cgroup limit detected");
        assert_eq!(c.limit_bytes, expected);
        assert!(m.usable_bytes() <= expected);
    }
}
