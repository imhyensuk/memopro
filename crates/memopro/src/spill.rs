//! Spill engine (0027 RS1–RS5, 0032 H1–H4, P5, 0038): move tensor bytes between memory and files.
//!
//! Main path (H1): re-read an unchanged tensor from its original file (e.g. safetensors) straight
//! into the destination buffer and verify it with a digest; no SSD write. Last resort: write a
//! spill file, only when the caller's policy allows it (the policy itself lives in Python).
//!
//! - RS1: each call does the whole transfer; no intermediate copies (reads land in `dst`, writes
//!   come from `src`).
//! - RS2: buffers are borrowed (the Python bindings take the buffer protocol, never `bytes`).
//! - RS3: reads and digests run on the process-wide rayon pool; writes on one process-wide,
//!   low-I/O-priority pool (H4).
//! - RS4: no transfer buffers at all, so extra memory does not grow with the transfer size.
//! - RS5: digests are computed chunk by chunk while other chunks are being read or written.
//! - P5: spill files are created `0600` with `create_new` and hold raw bytes only; metadata and
//!   the digest stay with the caller (never pickle).
//!
//! Digest: xxh3-128 of every [`DIGEST_CHUNK`], combined with the length (0038). It detects
//! corruption and changed files; it is not a cryptographic hash. The value does not depend on
//! the number of threads.

use crate::error::{Error, Result};
use rayon::prelude::*;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use xxhash_rust::xxh3::xxh3_128;

/// Chunk size for digests and parallel I/O.
pub const DIGEST_CHUNK: usize = 4 << 20;

/// 128-bit digest of a buffer (16 bytes, little-endian).
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

fn combine(len: usize, parts: &[u128]) -> Digest {
    let mut buf = Vec::with_capacity(8 + 16 * parts.len());
    buf.extend_from_slice(&(len as u64).to_le_bytes());
    for p in parts {
        buf.extend_from_slice(&p.to_le_bytes());
    }
    Digest(xxh3_128(&buf).to_le_bytes().to_vec())
}

/// Digest of `data`, computed in parallel.
pub fn digest(data: &[u8]) -> Digest {
    let parts: Vec<u128> = data.par_chunks(DIGEST_CHUNK).map(xxh3_128).collect();
    combine(data.len(), &parts)
}

#[cfg(target_os = "macos")]
fn lower_io_priority() {
    unsafe extern "C" {
        fn setiopolicy_np(
            iotype: libc::c_int,
            scope: libc::c_int,
            policy: libc::c_int,
        ) -> libc::c_int;
    }
    const IOPOL_TYPE_DISK: libc::c_int = 0;
    const IOPOL_SCOPE_THREAD: libc::c_int = 1;
    const IOPOL_UTILITY: libc::c_int = 4;
    // SAFETY: plain syscall wrapper with constant arguments; failure only keeps the default.
    unsafe {
        setiopolicy_np(IOPOL_TYPE_DISK, IOPOL_SCOPE_THREAD, IOPOL_UTILITY);
    }
}

#[cfg(target_os = "linux")]
fn lower_io_priority() {
    const IOPRIO_WHO_PROCESS: libc::c_long = 1; // with id 0: the calling thread
    const IOPRIO_CLASS_BE: libc::c_long = 2;
    const LOWEST_BE_LEVEL: libc::c_long = 7;
    // SAFETY: ioprio_set with constant arguments; failure only keeps the default priority.
    unsafe {
        libc::syscall(
            libc::SYS_ioprio_set,
            IOPRIO_WHO_PROCESS,
            0,
            (IOPRIO_CLASS_BE << 13) | LOWEST_BE_LEVEL,
        );
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn lower_io_priority() {}

/// Process-wide pool for spill writes, with lowered I/O priority (H4, RS3).
fn write_pool() -> Option<&'static rayon::ThreadPool> {
    static POOL: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(std::thread::available_parallelism().map_or(2, |n| n.get().min(4)))
            .thread_name(|i| format!("memopro-spill-{i}"))
            .start_handler(|_| lower_io_priority())
            .build()
            .ok()
    })
    .as_ref()
}

#[cfg(unix)]
fn read_at(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(buf, offset)
}

#[cfg(windows)]
fn read_at(file: &File, mut buf: &mut [u8], mut offset: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        match file.seek_read(buf, offset)? {
            0 => return Err(std::io::ErrorKind::UnexpectedEof.into()),
            n => {
                buf = &mut buf[n..];
                offset += n as u64;
            }
        }
    }
    Ok(())
}

fn read_region(
    file: &File,
    offset: u64,
    dst: &mut [u8],
    expected: Option<&Digest>,
) -> Result<Digest> {
    let parts: std::io::Result<Vec<u128>> = dst
        .par_chunks_mut(DIGEST_CHUNK)
        .enumerate()
        .map(|(i, chunk)| {
            read_at(file, chunk, offset + (i * DIGEST_CHUNK) as u64)?;
            Ok(xxh3_128(chunk))
        })
        .collect();
    let got = combine(dst.len(), &parts?);
    match expected {
        Some(want) if *want != got => Err(Error::Integrity(
            "data read back does not match its digest".into(),
        )),
        _ => Ok(got),
    }
}

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// The spill engine. Stateless apart from the process-wide pools.
#[derive(Debug, Default, Clone, Copy)]
pub struct Engine;

impl Engine {
    pub fn new() -> Self {
        Engine
    }

    /// Digest `data` in parallel.
    pub fn digest(&self, data: &[u8]) -> Digest {
        digest(data)
    }

    /// Fill `dst` from an original file region (write-free restore, H1) and return its digest.
    /// With `expected`, fail with [`Error::Integrity`] if the data differs.
    pub fn read_source_into(
        &self,
        source: &SourceRef,
        dst: &mut [u8],
        expected: Option<&Digest>,
    ) -> Result<Digest> {
        if dst.len() as u64 != source.len {
            return Err(Error::InvalidArgument(format!(
                "destination holds {} bytes, source region {}",
                dst.len(),
                source.len
            )));
        }
        let file = File::open(&source.path)?;
        let size = file.metadata()?.len();
        if source.offset.saturating_add(source.len) > size {
            return Err(Error::Integrity(format!(
                "{} is {size} bytes, shorter than the recorded region",
                source.path.display()
            )));
        }
        read_region(&file, source.offset, dst, expected)
    }

    /// Write `src` to a new spill file in `dir` (last resort; the caller checked the policy).
    #[cfg(unix)]
    pub fn write(&self, src: &[u8], dir: &Path) -> Result<SpillFile> {
        use std::fs::OpenOptions;
        use std::os::unix::fs::{FileExt, OpenOptionsExt};
        let name = format!(
            "{}-{}-{}.mpspill",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos()),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        );
        let path = dir.join(name);
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&path)?;
        let job = || -> std::io::Result<Vec<u128>> {
            file.set_len(src.len() as u64)?;
            src.par_chunks(DIGEST_CHUNK)
                .enumerate()
                .map(|(i, chunk)| {
                    file.write_all_at(chunk, (i * DIGEST_CHUNK) as u64)?;
                    Ok(xxh3_128(chunk))
                })
                .collect()
        };
        let parts = match write_pool() {
            Some(pool) => pool.install(job),
            None => job(),
        };
        match parts {
            Ok(parts) => Ok(SpillFile {
                path,
                len: src.len() as u64,
                digest: combine(src.len(), &parts),
            }),
            Err(e) => {
                drop(file);
                let _ = std::fs::remove_file(&path);
                Err(e.into())
            }
        }
    }

    /// Write `src` to a new spill file (not implemented on this platform yet).
    #[cfg(not(unix))]
    pub fn write(&self, _src: &[u8], _dir: &Path) -> Result<SpillFile> {
        Err(Error::not_implemented(
            "spill::Engine::write on this OS",
            "v0.1.x",
        ))
    }

    /// Restore a spill file into `dst`, verifying its digest.
    pub fn read_into(&self, file: &SpillFile, dst: &mut [u8]) -> Result<()> {
        let source = SourceRef {
            path: file.path.clone(),
            offset: 0,
            len: file.len,
        };
        self.read_source_into(&source, dst, Some(&file.digest))
            .map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(n: usize) -> Vec<u8> {
        (0..n).map(|i| (i * 31 + i / 7) as u8).collect()
    }

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("memopro-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn digest_is_deterministic_and_sensitive() {
        let a = data(DIGEST_CHUNK * 2 + 5);
        let d1 = digest(&a);
        let d2 = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| digest(&a));
        assert_eq!(d1, d2, "independent of thread count");
        assert_eq!(d1.0.len(), 16);
        let mut b = a.clone();
        b[DIGEST_CHUNK + 3] ^= 1;
        assert_ne!(d1, digest(&b));
        assert_ne!(digest(&a[..10]), digest(&a[..11]));
    }

    #[test]
    fn spill_roundtrip_is_bit_exact_and_private() {
        let dir = tmpdir("spill");
        let src = data(DIGEST_CHUNK * 2 + 123);
        let engine = Engine::new();
        let file = engine.write(&src, &dir).unwrap();
        assert_eq!(file.len, src.len() as u64);
        assert_eq!(file.digest, digest(&src));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&file.path).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600);
        }
        let mut out = vec![0u8; src.len()];
        engine.read_into(&file, &mut out).unwrap();
        assert_eq!(out, src);
        std::fs::remove_file(&file.path).unwrap();
    }

    #[test]
    fn corrupted_or_changed_files_are_detected() {
        let dir = tmpdir("corrupt");
        let src = data(100_000);
        let engine = Engine::new();
        let file = engine.write(&src, &dir).unwrap();
        let mut bytes = std::fs::read(&file.path).unwrap();
        bytes[5000] ^= 0xff;
        std::fs::write(&file.path, &bytes).unwrap();
        let mut out = vec![0u8; src.len()];
        assert!(matches!(
            engine.read_into(&file, &mut out),
            Err(Error::Integrity(_))
        ));
        std::fs::write(&file.path, &bytes[..1000]).unwrap(); // truncated
        assert!(matches!(
            engine.read_into(&file, &mut out),
            Err(Error::Integrity(_))
        ));
        std::fs::remove_file(&file.path).unwrap();
    }

    #[test]
    fn source_region_read_lands_in_destination() {
        let dir = tmpdir("source");
        let path = dir.join("weights.bin");
        let blob = data(DIGEST_CHUNK + 4096);
        std::fs::write(&path, &blob).unwrap();
        let region = SourceRef {
            path: path.clone(),
            offset: 100,
            len: (DIGEST_CHUNK + 1000) as u64,
        };
        let mut dst = vec![0u8; region.len as usize];
        let d = Engine::new()
            .read_source_into(&region, &mut dst, None)
            .unwrap();
        assert_eq!(&dst[..], &blob[100..100 + region.len as usize]);
        assert_eq!(d, digest(&dst));
        let mut short = vec![0u8; 10];
        assert!(matches!(
            Engine::new().read_source_into(&region, &mut short, None),
            Err(Error::InvalidArgument(_))
        ));
        std::fs::remove_file(&path).unwrap();
    }
}
