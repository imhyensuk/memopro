//! Lossless codec for floating-point buffers: byte shuffle + zstd, in independent chunks.
//!
//! Used by `hibernate(mode="compress")` to keep idle tensors compressed in RAM. Chunks are
//! processed in parallel on the process-wide rayon pool; every worker thread keeps one zstd
//! context and one scratch buffer for the life of the process (RS3), so extra memory is bounded
//! by `threads x CHUNK` plus the output. No novelty is claimed: this is the well-known
//! byte-shuffle idea (as in blosc) on top of the zstd library.
//!
//! The E008 prototypes (a thread pool per call, `bytes` inputs) were removed in 0038; E008 is
//! reproduced from commit 9fd9bd3.

use rayon::prelude::*;
use std::cell::RefCell;
use std::io;

/// Uncompressed bytes per chunk (a multiple of every element size up to 16).
pub const CHUNK: usize = 4 << 20;

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

struct Worker {
    level: i32,
    cctx: zstd::bulk::Compressor<'static>,
    dctx: zstd::bulk::Decompressor<'static>,
    scratch: Vec<u8>,
}

thread_local! {
    static WORKER: RefCell<Option<Worker>> = const { RefCell::new(None) };
}

/// Run `f` with this thread's reusable zstd contexts and scratch buffer (RS3).
fn with_worker<R>(level: i32, f: impl FnOnce(&mut Worker) -> io::Result<R>) -> io::Result<R> {
    WORKER.with(|cell| {
        let mut slot = cell.borrow_mut();
        let fresh = !matches!(slot.as_ref(), Some(w) if w.level == level);
        if fresh {
            *slot = Some(Worker {
                level,
                cctx: zstd::bulk::Compressor::new(level)?,
                dctx: zstd::bulk::Decompressor::new()?,
                scratch: Vec::new(),
            });
        }
        f(slot.as_mut().expect("worker initialised above"))
    })
}

fn check_elem(len: usize, elem: usize) -> io::Result<()> {
    if elem == 0 || elem > 16 || CHUNK % elem != 0 || len % elem != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("element size {elem} does not divide the buffer ({len} bytes)"),
        ));
    }
    Ok(())
}

/// Shuffle + zstd-compress `src` in independent [`CHUNK`]-sized chunks, in parallel.
pub fn pack(src: &[u8], elem: usize, level: i32) -> io::Result<Vec<Vec<u8>>> {
    check_elem(src.len(), elem)?;
    src.par_chunks(CHUNK)
        .map(|c| {
            with_worker(level, |w| {
                w.scratch.resize(c.len(), 0);
                shuffle(c, elem, &mut w.scratch);
                w.cctx.compress(&w.scratch)
            })
        })
        .collect()
}

/// Decompress chunks from [`pack`] directly into `dst` (length = original length).
pub fn unpack_into(chunks: &[Vec<u8>], elem: usize, dst: &mut [u8]) -> io::Result<()> {
    check_elem(dst.len(), elem)?;
    let expected = dst.len().div_ceil(CHUNK);
    if chunks.len() != expected {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("expected {expected} chunks, got {}", chunks.len()),
        ));
    }
    dst.par_chunks_mut(CHUNK)
        .zip(chunks.par_iter())
        .try_for_each(|(out, comp)| {
            with_worker(0, |w| {
                w.scratch.resize(out.len(), 0);
                let n = w.dctx.decompress_to_buffer(comp, &mut w.scratch)?;
                if n != out.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "chunk length mismatch",
                    ));
                }
                unshuffle(&w.scratch, elem, out);
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
    fn pack_roundtrip_is_bit_exact_across_chunks_and_levels() {
        // 2.5 chunks: exercises the ragged last chunk; NaN/Inf/subnormal included
        let src = sample(CHUNK * 5 / 8 + 7);
        for (elem, level) in [(4usize, 1), (2, 3), (1, 1)] {
            let n = src.len() / elem * elem;
            let chunks = pack(&src[..n], elem, level).unwrap();
            assert_eq!(chunks.len(), n.div_ceil(CHUNK));
            let mut out = vec![0u8; n];
            unpack_into(&chunks, elem, &mut out).unwrap();
            assert_eq!(&src[..n], &out[..]);
        }
    }

    #[test]
    fn compressible_data_shrinks() {
        let src = vec![7u8; CHUNK + 100];
        let chunks = pack(&src, 1, 1).unwrap();
        assert!(chunks.iter().map(Vec::len).sum::<usize>() < src.len() / 100);
    }

    #[test]
    fn bad_element_size_and_chunk_count_are_rejected() {
        assert!(pack(&[0u8; 10], 4, 1).is_err());
        assert!(pack(&[0u8; 12], 0, 1).is_err());
        let chunks = pack(&[1u8; 16], 4, 1).unwrap();
        let mut out = vec![0u8; CHUNK + 16];
        assert!(unpack_into(&chunks, 4, &mut out).is_err());
    }
}
