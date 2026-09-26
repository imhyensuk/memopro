//! OS memory-pressure signal for γ elastic (v0.3, 0052 E6).
//!
//! - macOS: `kern.memorystatus_vm_pressure_level` (1 normal, 2 warning, 4 critical), the level
//!   the kernel itself uses for its memory-pressure notifications.
//! - Linux: pressure stall information (PSI) of this process's cgroup v2 (`memory.pressure`), or
//!   the whole system (`/proc/pressure/memory`). `some`/`full` avg10 are the percentage of the
//!   last 10 s in which some/all tasks stalled on memory. Levels use initial thresholds
//!   ([`WARN_SOME`] …) that are to be calibrated by measurement.
//! - Other systems: [`Error::Unsupported`]; callers fail open.
//!
//! Reading is cheap (one `sysctl` or one small file) and never blocks on memory itself.

#[cfg(not(target_os = "macos"))]
use crate::error::Error;
use crate::error::Result;

/// Coarse pressure level, comparable across operating systems.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    Normal,
    Warning,
    Critical,
}

impl Level {
    pub fn as_str(self) -> &'static str {
        match self {
            Level::Normal => "normal",
            Level::Warning => "warning",
            Level::Critical => "critical",
        }
    }
}

/// One reading of the system's memory pressure.
#[derive(Debug, Clone, PartialEq)]
pub struct Pressure {
    pub level: Level,
    /// Where the reading came from: `"macos"`, `"cgroup"` or `"system"`.
    pub source: &'static str,
    /// Linux PSI: % of the last 10 s in which at least one task stalled on memory.
    pub some_avg10: Option<f64>,
    /// Linux PSI: % of the last 10 s in which all non-idle tasks stalled on memory.
    pub full_avg10: Option<f64>,
    /// macOS: the raw kernel level (1, 2 or 4).
    pub raw_level: Option<i64>,
}

/// PSI thresholds (percent of the last 10 s). Initial values, calibrated later (0052 E6).
pub const WARN_SOME: f64 = 10.0;
pub const WARN_FULL: f64 = 1.0;
pub const CRIT_SOME: f64 = 40.0;
pub const CRIT_FULL: f64 = 10.0;

/// Level from PSI averages.
pub fn level_from_psi(some: f64, full: f64) -> Level {
    if some >= CRIT_SOME || full >= CRIT_FULL {
        Level::Critical
    } else if some >= WARN_SOME || full >= WARN_FULL {
        Level::Warning
    } else {
        Level::Normal
    }
}

/// `(some avg10, full avg10)` from the text of a PSI file.
pub fn parse_psi(text: &str) -> Option<(f64, f64)> {
    let avg10 = |kind: &str| {
        text.lines()
            .find(|l| l.starts_with(kind))?
            .split_whitespace()
            .find_map(|field| field.strip_prefix("avg10="))?
            .parse::<f64>()
            .ok()
    };
    let some = avg10("some ")?;
    Some((some, avg10("full ").unwrap_or(0.0)))
}

/// Level from the macOS kernel value.
pub fn level_from_macos(raw: i64) -> Level {
    match raw {
        r if r >= 4 => Level::Critical,
        2 | 3 => Level::Warning,
        _ => Level::Normal,
    }
}

/// Current memory pressure of the system (or of this process's container on Linux).
pub fn current() -> Result<Pressure> {
    imp::current()
}

#[cfg(target_os = "macos")]
mod imp {
    use super::*;

    pub fn current() -> Result<Pressure> {
        let name = c"kern.memorystatus_vm_pressure_level";
        let mut value: libc::c_int = 0;
        let mut len = std::mem::size_of::<libc::c_int>();
        // SAFETY: a valid NUL-terminated name, a buffer of `len` bytes and no new value.
        let rc = unsafe {
            libc::sysctlbyname(
                name.as_ptr(),
                (&mut value as *mut libc::c_int).cast(),
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        if rc != 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        let raw = i64::from(value);
        Ok(Pressure {
            level: level_from_macos(raw),
            source: "macos",
            some_avg10: None,
            full_avg10: None,
            raw_level: Some(raw),
        })
    }
}

#[cfg(target_os = "linux")]
mod imp {
    use super::*;
    use std::path::PathBuf;

    fn cgroup_pressure_file() -> Option<PathBuf> {
        let text = std::fs::read_to_string("/proc/self/cgroup").ok()?;
        let path = text.lines().find_map(|l| l.strip_prefix("0::"))?;
        let file = PathBuf::from("/sys/fs/cgroup")
            .join(path.trim_start_matches('/'))
            .join("memory.pressure");
        file.is_file().then_some(file)
    }

    pub fn current() -> Result<Pressure> {
        let (file, source) = match cgroup_pressure_file() {
            Some(f) => (f, "cgroup"),
            None => (PathBuf::from("/proc/pressure/memory"), "system"),
        };
        let text = std::fs::read_to_string(&file).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                Error::Unsupported("the kernel does not expose PSI (CONFIG_PSI off)".into())
            } else {
                Error::Io(e)
            }
        })?;
        let (some, full) = parse_psi(&text)
            .ok_or_else(|| Error::Unsupported(format!("unexpected PSI format in {file:?}")))?;
        Ok(Pressure {
            level: level_from_psi(some, full),
            source,
            some_avg10: Some(some),
            full_avg10: Some(full),
            raw_level: None,
        })
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
mod imp {
    use super::*;

    pub fn current() -> Result<Pressure> {
        Err(Error::Unsupported(
            "memory-pressure signal on this operating system".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;

    #[test]
    fn levels_are_ordered() {
        assert!(Level::Normal < Level::Warning && Level::Warning < Level::Critical);
    }

    #[test]
    fn psi_is_parsed_and_mapped() {
        let text = "some avg10=12.50 avg60=3.00 avg300=1.00 total=123\n\
                    full avg10=0.40 avg60=0.10 avg300=0.00 total=45\n";
        assert_eq!(parse_psi(text), Some((12.5, 0.4)));
        assert_eq!(level_from_psi(12.5, 0.4), Level::Warning);
        assert_eq!(level_from_psi(0.0, 0.0), Level::Normal);
        assert_eq!(level_from_psi(5.0, 12.0), Level::Critical);
        assert_eq!(
            parse_psi("some avg10=1.00 avg60=0 avg300=0 total=0\n"),
            Some((1.0, 0.0))
        );
        assert_eq!(parse_psi("garbage"), None);
    }

    #[test]
    fn macos_levels_are_mapped() {
        assert_eq!(level_from_macos(1), Level::Normal);
        assert_eq!(level_from_macos(2), Level::Warning);
        assert_eq!(level_from_macos(4), Level::Critical);
    }

    #[test]
    fn current_reads_or_says_unsupported() {
        match current() {
            Ok(p) => {
                assert!(matches!(p.source, "macos" | "cgroup" | "system"));
                if p.source != "macos" {
                    assert!(p.some_avg10.is_some());
                }
            }
            Err(Error::Unsupported(_)) => {}
            Err(e) => panic!("unexpected error: {e}"),
        }
    }
}
