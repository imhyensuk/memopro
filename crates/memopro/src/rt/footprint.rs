//! Memory this process occupies, for budgets over the whole process (0189, 0229).

/// Bytes of memory this process occupies: the physical footprint on macOS (what Activity Monitor
/// shows, including compressed memory and Apple GPU memory), the resident set on Linux. `None`
/// where it cannot be measured.
pub fn process_footprint() -> Option<u64> {
    imp()
}

#[cfg(target_os = "macos")]
fn imp() -> Option<u64> {
    // SAFETY: rusage_info_v2 is plain data; proc_pid_rusage fills it for our own pid.
    unsafe {
        let mut info: libc::rusage_info_v2 = std::mem::zeroed();
        let r = libc::proc_pid_rusage(
            libc::getpid(),
            libc::RUSAGE_INFO_V2,
            (&mut info as *mut libc::rusage_info_v2).cast(),
        );
        (r == 0).then_some(info.ri_phys_footprint)
    }
}

#[cfg(target_os = "linux")]
fn imp() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/statm").ok()?;
    let pages: u64 = s.split_whitespace().nth(1)?.parse().ok()?;
    Some(pages * super::page_size() as u64)
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn imp() -> Option<u64> {
    None
}

#[cfg(test)]
mod tests {
    #[test]
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[ignore = "measures the whole process: run with --ignored --test-threads=1"]
    fn footprint_grows_with_touched_memory() {
        let before = super::process_footprint().unwrap();
        let v = vec![1u8; 64 << 20];
        let after = super::process_footprint().unwrap();
        assert!(after >= before + (48 << 20), "{before} -> {after}");
        drop(v);
    }
}
