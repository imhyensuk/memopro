//! Lossless codec for floating-point buffers: byte shuffle + zstd, in independent chunks.
//!
//! Chunks are compressed and decompressed in parallel (rayon). Every worker only touches its own
//! chunk and one scratch buffer, so extra memory is bounded by `threads x chunk` plus the output.
//! Status: prototype for experiment E008 (docs/research/0023). No novelty is claimed: this is the
//! well-known byte-shuffle idea (as in blosc) on top of the zstd library.

use rayon::prelude::*;
use std::io;

/// Group byte `b` of every element together (byte planes). `src.len()` must be a multiple of `elem`.
pub fn shuffle(src: &[u8], elem: usize, dst: &mut [u8]) {
    assert_eq!(src.len(), dst.len());
    assert!(elem > 0 && src.len() % elem == 0);
    let n = src.len() / elem;
    match elem {
        2 => {
            let (p0, p1) = dst.split_at_mut(n);
            for ((e, a), b) in src.chunks_exact(2).zip(p0).zip(p1) {
                *a = e[0];
                *b = e[1];
            }
            return;
        }
        4 => {
            let (p0, rest) = dst.split_at_mut(n);
            let (p1, rest) = rest.split_at_mut(n);
            let (p2, p3) = rest.split_at_mut(n);
            for ((((e, a), b), c), d) in src.chunks_exact(4).zip(p0).zip(p1).zip(p2).zip(p3) {
                *a = e[0];
                *b = e[1];
                *c = e[2];
                *d = e[3];
            }
            return;
        }
        _ => {}
    }
    for (i, e) in src.chunks_exact(elem).enumerate() {
        for (b, &byte) in e.iter().enumerate() {
            dst[b * n + i] = byte;
        }
    }
}

/// Inverse of [`shuffle`].
pub fn unshuffle(src: &[u8], elem: usize, dst: &mut [u8]) {
    assert_eq!(src.len(), dst.len());
    assert!(elem > 0 && src.len() % elem == 0);
    let n = src.len() / elem;
    match elem {
        2 => {
            let (p0, p1) = src.split_at(n);
            for ((e, a), b) in dst.chunks_exact_mut(2).zip(p0).zip(p1) {
                e[0] = *a;
                e[1] = *b;
            }
            return;
        }
        4 => {
            let (p0, rest) = src.split_at(n);
            let (p1, rest) = rest.split_at(n);
            let (p2, p3) = rest.split_at(n);
            for ((((e, a), b), c), d) in dst.chunks_exact_mut(4).zip(p0).zip(p1).zip(p2).zip(p3) {
                e[0] = *a;
                e[1] = *b;
                e[2] = *c;
                e[3] = *d;
            }
            return;
        }
        _ => {}
    }
    for (i, e) in dst.chunks_exact_mut(elem).enumerate() {
        for (b, byte) in e.iter_mut().enumerate() {
            *byte = src[b * n + i];
        }
    }
}

fn chunk_len(chunk_bytes: usize, elem: usize) -> usize {
    (chunk_bytes - chunk_bytes % elem).max(elem)
}

fn pool(threads: usize) -> io::Result<rayon::ThreadPool> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads) // 0 = rayon default (all logical CPUs)
        .build()
        .map_err(io::Error::other)
}

/// Shuffle + zstd-compress `src` in independent chunks, in parallel.
pub fn compress(
    src: &[u8],
    elem: usize,
    chunk_bytes: usize,
    level: i32,
    threads: usize,
) -> io::Result<Vec<Vec<u8>>> {
    let chunk = chunk_len(chunk_bytes, elem);
    pool(threads)?.install(|| {
        src.par_chunks(chunk)
            .map(|c| {
                let mut shuffled = vec![0u8; c.len()];
                shuffle(c, elem, &mut shuffled);
                zstd::bulk::compress(&shuffled, level)
            })
            .collect()
    })
}

/// Per-worker state reused across chunks: zstd context, shuffle scratch and output buffer.
struct Worker {
    zstd: zstd::bulk::Compressor<'static>,
    scratch: Vec<u8>,
}

fn worker(level: i32) -> io::Result<Worker> {
    Ok(Worker {
        zstd: zstd::bulk::Compressor::new(level)?,
        scratch: Vec::new(),
    })
}

fn compress_chunk(w: &mut io::Result<Worker>, c: &[u8], elem: usize) -> io::Result<Vec<u8>> {
    let w = w
        .as_mut()
        .map_err(|e| io::Error::new(e.kind(), e.to_string()))?;
    w.scratch.resize(c.len(), 0);
    shuffle(c, elem, &mut w.scratch);
    w.zstd.compress(&w.scratch)
}

/// Like [`compress`] but reuses per-thread zstd contexts and scratch buffers (E008b, exploratory).
pub fn compress_reuse(
    src: &[u8],
    elem: usize,
    chunk_bytes: usize,
    level: i32,
    threads: usize,
) -> io::Result<Vec<Vec<u8>>> {
    let chunk = chunk_len(chunk_bytes, elem);
    pool(threads)?.install(|| {
        src.par_chunks(chunk)
            .map_init(|| worker(level), |w, c| compress_chunk(w, c, elem))
            .collect()
    })
}

/// Compress in parallel and stream length-prefixed chunks to `path`, overlapping compression of
/// the next window with writing of the previous one. Extra memory is bounded by roughly
/// `5 x threads x chunk` regardless of `src.len()`; the file is fully synced (F_FULLFSYNC on
/// macOS) before returning. Returns the number of bytes written. (E008b, exploratory)
pub fn spill_to_file(
    src: &[u8],
    elem: usize,
    chunk_bytes: usize,
    level: i32,
    threads: usize,
    path: &std::path::Path,
) -> io::Result<u64> {
    use std::io::Write;
    let chunk = chunk_len(chunk_bytes, elem);
    let pool = pool(threads)?;
    let window = pool.current_num_threads() * 2;
    let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<u8>>(window);
    let file = std::fs::File::create(path)?;
    let writer = std::thread::spawn(move || -> io::Result<u64> {
        let mut f = io::BufWriter::with_capacity(1 << 20, file);
        let mut written = 0u64;
        for buf in rx {
            f.write_all(&(buf.len() as u64).to_le_bytes())?;
            f.write_all(&buf)?;
            written += 8 + buf.len() as u64;
        }
        f.into_inner().map_err(|e| e.into_error())?.sync_all()?;
        Ok(written)
    });
    let chunks: Vec<&[u8]> = src.chunks(chunk).collect();
    let mut result = Ok(());
    for win in chunks.chunks(window) {
        let outs: io::Result<Vec<Vec<u8>>> = pool.install(|| {
            win.par_iter()
                .map_init(|| worker(level), |w, c| compress_chunk(w, c, elem))
                .collect()
        });
        match outs {
            Ok(v) => {
                if v.into_iter().any(|o| tx.send(o).is_err()) {
                    break; // writer failed; its error is reported below
                }
            }
            Err(e) => {
                result = Err(e);
                break;
            }
        }
    }
    drop(tx);
    let written = writer
        .join()
        .map_err(|_| io::Error::other("writer thread panicked"))??;
    result.map(|()| written)
}

/// Decompress chunks produced by [`compress`] directly into `dst` (length = original length).
pub fn decompress_into(
    chunks: &[&[u8]],
    elem: usize,
    chunk_bytes: usize,
    dst: &mut [u8],
    threads: usize,
) -> io::Result<()> {
    let chunk = chunk_len(chunk_bytes, elem);
    let expected = dst.len().div_ceil(chunk);
    if chunks.len() != expected {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("expected {expected} chunks, got {}", chunks.len()),
        ));
    }
    pool(threads)?.install(|| {
        dst.par_chunks_mut(chunk)
            .zip(chunks.par_iter())
            .try_for_each(|(out, comp)| {
                let shuffled = zstd::bulk::decompress(comp, out.len())?;
                if shuffled.len() != out.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "chunk length mismatch",
                    ));
                }
                unshuffle(&shuffled, elem, out);
                Ok(())
            })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(n: usize) -> Vec<u8> {
        // deterministic pseudo-random f32 values incl. special values
        let mut v = Vec::with_capacity(n * 4);
        let mut x: u32 = 0x1234_5678;
        for i in 0..n {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            let f = match i % 97 {
                0 => f32::NAN,
                1 => f32::INFINITY,
                2 => f32::MIN_POSITIVE / 3.0, // subnormal
                _ => (x as f32 / u32::MAX as f32 - 0.5) * 3.0,
            };
            v.extend_from_slice(&f.to_le_bytes());
        }
        v
    }

    #[test]
    fn shuffle_roundtrip() {
        let src = sample(1001);
        let mut s = vec![0; src.len()];
        let mut back = vec![0; src.len()];
        shuffle(&src, 4, &mut s);
        unshuffle(&s, 4, &mut back);
        assert_eq!(src, back);
    }

    #[test]
    fn compress_roundtrip_bit_exact_with_ragged_last_chunk() {
        let src = sample(300_001);
        for threads in [1, 4] {
            let chunks = compress(&src, 4, 64 * 1024, 1, threads).unwrap();
            let refs: Vec<&[u8]> = chunks.iter().map(|c| c.as_slice()).collect();
            let mut out = vec![0u8; src.len()];
            decompress_into(&refs, 4, 64 * 1024, &mut out, threads).unwrap();
            assert_eq!(src, out);
        }
    }

    #[test]
    fn fast_shuffle_matches_generic_for_2_and_4() {
        let src = sample(10_001);
        for elem in [2usize, 4] {
            let n = src.len() / elem * elem;
            let s = &src[..n];
            let mut fast = vec![0; n];
            shuffle(s, elem, &mut fast);
            let m = n / elem;
            let mut generic = vec![0; n];
            for (i, e) in s.chunks_exact(elem).enumerate() {
                for (b, &byte) in e.iter().enumerate() {
                    generic[b * m + i] = byte;
                }
            }
            assert_eq!(fast, generic);
            let mut back = vec![0; n];
            unshuffle(&fast, elem, &mut back);
            assert_eq!(back, s);
        }
    }

    #[test]
    fn compress_reuse_matches_compress() {
        let src = sample(200_003);
        let a = compress(&src, 4, 64 * 1024, 1, 3).unwrap();
        let b = compress_reuse(&src, 4, 64 * 1024, 1, 3).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn spill_to_file_roundtrip() {
        let src = sample(250_001);
        let path =
            std::env::temp_dir().join(format!("memopro_spill_test_{}.bin", std::process::id()));
        let written = spill_to_file(&src, 4, 64 * 1024, 1, 3, &path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(bytes.len() as u64, written);
        let mut chunks = Vec::new();
        let mut off = 0;
        while off < bytes.len() {
            let len = u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap()) as usize;
            chunks.push(&bytes[off + 8..off + 8 + len]);
            off += 8 + len;
        }
        let mut out = vec![0u8; src.len()];
        decompress_into(&chunks, 4, 64 * 1024, &mut out, 2).unwrap();
        assert_eq!(out, src);
    }

    #[test]
    fn wrong_chunk_count_is_an_error() {
        let src = sample(1000);
        let chunks = compress(&src, 4, 1024, 1, 1).unwrap();
        let refs: Vec<&[u8]> = chunks.iter().take(1).map(|c| c.as_slice()).collect();
        let mut out = vec![0u8; src.len()];
        assert!(decompress_into(&refs, 4, 1024, &mut out, 1).is_err());
    }
}
