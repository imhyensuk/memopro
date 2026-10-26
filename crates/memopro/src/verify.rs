//! Verified reads of file regions (0032 H1, 0038): read bytes of an unchanged original file (e.g.
//! safetensors) straight into a destination buffer and check them with a digest. memopro never
//! writes these files (0328); the runtime drops buffers it can read back this way.
//!
//! Reads and digests run on one process-wide pool of [`readers`] threads, chunk by chunk, with no
//! transfer buffers (0027 RS1-RS5). Digest: xxh3-128 of every [`DIGEST_CHUNK`], combined with the
//! length (0038). It detects corruption and changed files; it is not a cryptographic hash. The
//! value does not depend on the number of threads.

use crate::error::{Error, Result};
use rayon::prelude::*;
use std::fs::File;
use std::sync::OnceLock;
use xxhash_rust::xxh3::xxh3_128;

/// Chunk size for digests and parallel I/O.
pub const DIGEST_CHUNK: usize = 4 << 20;

/// 128-bit digest of a buffer (16 bytes, little-endian).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Digest(pub Vec<u8>);

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

/// Per-[`DIGEST_CHUNK`] hashes of `data`. Hashing consecutive pieces whose lengths are multiples
/// of `DIGEST_CHUNK` and combining all parts with [`digest_combine`] gives [`digest`] of the whole,
/// so callers can verify large data piece by piece with bounded buffers (0061 F1).
/// Reads in flight at once: `MEMOPRO_READ_THREADS` when set (a calibration knob for other
/// devices), else 4: more did not read faster on an M1 (E050, 0247) and each reader keeps one
/// [`DIGEST_CHUNK`]-sized scratch buffer for unaligned reads.
pub fn readers() -> usize {
    std::env::var("MEMOPRO_READ_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or(4)
}

/// Process-wide pool for reads (RS3).
fn read_pool() -> Option<&'static rayon::ThreadPool> {
    static POOL: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(readers())
            .thread_name(|i| format!("memopro-read-{i}"))
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

/// File offsets of uncached reads are aligned to this. macOS reads an `F_NOCACHE` file at an
/// unaligned offset through its cache: slower (1.4 vs 2.0 GiB/s) and the pages stay cached
/// (E050 follow-up, 0248). Safetensors data starts at any byte (0115).
const ALIGN: u64 = 16 << 10;

std::thread_local! {
    static SCRATCH: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Fill `dst` from `offset`: directly when the offset is aligned, else through an aligned
/// per-thread scratch buffer covering the aligned pages around it (same bytes, one copy).
fn read_aligned(file: &File, file_len: u64, dst: &mut [u8], offset: u64) -> std::io::Result<()> {
    let skew = offset % ALIGN;
    if skew == 0 {
        return read_at(file, dst, offset);
    }
    let start = offset - skew;
    let end = (offset + dst.len() as u64)
        .next_multiple_of(ALIGN)
        .min(file_len);
    let len = (end - start) as usize;
    SCRATCH.with(|s| {
        let mut s = s.borrow_mut();
        if s.len() < len + ALIGN as usize {
            *s = vec![0; len + ALIGN as usize];
        }
        let at = s.as_ptr().align_offset(ALIGN as usize);
        let buf = &mut s[at..at + len];
        read_at(file, buf, start)?;
        let from = skew as usize;
        dst.copy_from_slice(&buf[from..from + dst.len()]);
        Ok(())
    })
}

pub(crate) fn read_region(
    file: &File,
    offset: u64,
    dst: &mut [u8],
    expected: Option<&Digest>,
) -> Result<Digest> {
    let len = dst.len();
    let file_len = file.metadata()?.len();
    let mut job = move || -> std::io::Result<Vec<u128>> {
        dst.par_chunks_mut(DIGEST_CHUNK)
            .enumerate()
            .map(|(i, chunk)| {
                read_aligned(file, file_len, chunk, offset + (i * DIGEST_CHUNK) as u64)?;
                Ok(xxh3_128(chunk))
            })
            .collect()
    };
    let parts = match read_pool() {
        Some(pool) => pool.install(job),
        None => job(),
    };
    let got = combine(len, &parts?);
    match expected {
        Some(want) if *want != got => Err(Error::Integrity(
            "data read back does not match its digest".into(),
        )),
        _ => Ok(got),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(n: usize) -> Vec<u8> {
        (0..n).map(|i| (i * 31 + i / 7) as u8).collect()
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
    fn a_region_read_lands_in_the_destination_and_is_checked() {
        let path = std::env::temp_dir().join(format!("memopro-verify-{}", std::process::id()));
        let blob = data(DIGEST_CHUNK + 4096);
        std::fs::write(&path, &blob).unwrap();
        let file = File::open(&path).unwrap();
        let mut dst = vec![0u8; DIGEST_CHUNK + 1000];
        let d = read_region(&file, 100, &mut dst, None).unwrap();
        assert_eq!(&dst[..], &blob[100..100 + dst.len()]);
        assert_eq!(d, digest(&dst));
        let wrong = digest(&blob[..10]);
        assert!(matches!(
            read_region(&file, 100, &mut dst, Some(&wrong)),
            Err(Error::Integrity(_))
        ));
        std::fs::remove_file(&path).unwrap();
    }
}
