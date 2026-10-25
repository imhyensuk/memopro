//! EDR v1 (0308): bf16 weights split losslessly into a 1-byte base and a 4-bit residual.
//!
//! - base byte: sign (bit 7) | exponent index (bits 3-6; 15 = escape) | top 3 mantissa bits
//! - residual: the low 4 mantissa bits, two per byte (element `2k` in the low nibble)
//! - table: the 15 most frequent exponents of the tensor; an exponent outside it is an escape,
//!   whose original bits are kept in a short list (about 0.04% of Qwen2.5 weights)
//!
//! Base and residual together give back every bit; the base alone is a draft: the same bf16
//! value with its low 4 mantissa bits cleared (Qwen2.5-0.5B: 87% greedy agreement). 12 bits per
//! weight instead of 16, decodable anywhere with a table lookup, shifts and ors.

use crate::{Error, Result};
use rayon::prelude::*;

/// Exponent index that marks an escape.
pub const ESCAPE: u8 = 15;
/// Elements per parallel piece (even, so residual bytes split cleanly).
const CHUNK: usize = 1 << 20;

/// Escaped elements: their positions and original bits, in increasing position order.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Escapes {
    pub index: Vec<u64>,
    pub bits: Vec<u16>,
}

/// The 15 most frequent exponents of `src` (ties: the lower exponent first).
pub fn table(src: &[u16]) -> [u8; 15] {
    let counts = src
        .par_chunks(CHUNK)
        .map(|c| {
            let mut h = [0u64; 256];
            for &x in c {
                h[((x >> 7) & 0xFF) as usize] += 1;
            }
            h
        })
        .reduce(
            || [0u64; 256],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b) {
                    *x += y;
                }
                a
            },
        );
    let mut order: Vec<usize> = (0..256).collect();
    order.sort_by(|&a, &b| counts[b].cmp(&counts[a]).then(a.cmp(&b)));
    let mut t = [0u8; 15];
    for (slot, &e) in t.iter_mut().zip(&order) {
        *slot = e as u8;
    }
    t
}

/// What each base byte stands for: sign | exponent | top 3 mantissa bits (escapes: exponent 0,
/// overwritten afterwards).
pub fn lookup(table: &[u8; 15]) -> [u16; 256] {
    let mut lut = [0u16; 256];
    for (v, x) in lut.iter_mut().enumerate() {
        let idx = (v >> 3) & 0xF;
        let exp = if idx < 15 { table[idx] as u16 } else { 0 };
        *x = (((v >> 7) as u16) << 15) | (exp << 7) | (((v & 7) as u16) << 4);
    }
    lut
}

fn inverse(table: &[u8; 15]) -> [u8; 256] {
    let mut inv = [ESCAPE; 256];
    for (i, &e) in table.iter().enumerate().rev() {
        inv[e as usize] = i as u8;
    }
    inv
}

/// Encode `src` into `base` (one byte per element) and `resid` (`src.len().div_ceil(2)` bytes).
pub fn encode(src: &[u16], table: &[u8; 15], base: &mut [u8], resid: &mut [u8]) -> Result<Escapes> {
    if base.len() != src.len() || resid.len() != src.len().div_ceil(2) {
        return Err(Error::InvalidArgument(format!(
            "edr::encode: {} elements need a base of {} and a residual of {} bytes, got {} and {}",
            src.len(),
            src.len(),
            src.len().div_ceil(2),
            base.len(),
            resid.len()
        )));
    }
    let inv = inverse(table);
    let pieces: Vec<Vec<(u64, u16)>> = src
        .par_chunks(CHUNK)
        .zip(base.par_chunks_mut(CHUNK))
        .zip(resid.par_chunks_mut(CHUNK / 2))
        .enumerate()
        .map(|(p, ((s, b), r))| {
            let mut esc = Vec::new();
            for (i, (&x, out)) in s.iter().zip(b.iter_mut()).enumerate() {
                let idx = inv[((x >> 7) & 0xFF) as usize];
                if idx == ESCAPE {
                    esc.push(((p * CHUNK + i) as u64, x));
                }
                *out = (((x >> 15) as u8) << 7) | (idx << 3) | ((x >> 4) & 7) as u8;
            }
            for (j, out) in r.iter_mut().enumerate() {
                let lo = s[2 * j] & 0xF;
                let hi = s.get(2 * j + 1).map_or(0, |&x| x & 0xF);
                *out = (lo | (hi << 4)) as u8;
            }
            esc
        })
        .collect();
    let mut esc = Escapes::default();
    for (i, x) in pieces.into_iter().flatten() {
        esc.index.push(i);
        esc.bits.push(x);
    }
    Ok(esc)
}

/// Decode into `out`: every bit back with `resid`, or the draft (low 4 mantissa bits cleared)
/// with `resid = None`.
pub fn decode(
    base: &[u8],
    resid: Option<&[u8]>,
    table: &[u8; 15],
    esc: (&[u64], &[u16]),
    out: &mut [u16],
) -> Result<()> {
    let n = out.len();
    if base.len() != n || resid.is_some_and(|r| r.len() != n.div_ceil(2)) {
        return Err(Error::InvalidArgument(format!(
            "edr::decode: {n} elements need a base of {n} and a residual of {} bytes",
            n.div_ceil(2)
        )));
    }
    if esc.0.len() != esc.1.len() || esc.0.iter().any(|&i| i as usize >= n) {
        return Err(Error::InvalidArgument(
            "edr::decode: bad escape list".into(),
        ));
    }
    let lut = lookup(table);
    out.par_chunks_mut(CHUNK)
        .zip(base.par_chunks(CHUNK))
        .enumerate()
        .for_each(|(p, (o, b))| match resid {
            // two elements per residual byte: one table lookup and one or each
            Some(r) => {
                let r = &r[p * CHUNK / 2..p * CHUNK / 2 + o.len().div_ceil(2)];
                let mut pairs = o.chunks_exact_mut(2);
                let mut bases = b.chunks_exact(2);
                for ((x, v), &rb) in (&mut pairs).zip(&mut bases).zip(r) {
                    x[0] = lut[v[0] as usize] | (rb & 0xF) as u16;
                    x[1] = lut[v[1] as usize] | (rb >> 4) as u16;
                }
                if let ([x], [v]) = (pairs.into_remainder(), bases.remainder()) {
                    *x = lut[*v as usize] | (r[r.len() - 1] & 0xF) as u16;
                }
            }
            None => {
                for (x, &v) in o.iter_mut().zip(b) {
                    *x = lut[v as usize];
                }
            }
        });
    let keep = if resid.is_some() { 0xFFFF } else { 0xFFF0 };
    for (&i, &x) in esc.0.iter().zip(esc.1) {
        out[i as usize] = x & keep;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(src: &[u16]) -> (Vec<u16>, Vec<u16>, Escapes) {
        let t = table(src);
        let mut base = vec![0u8; src.len()];
        let mut resid = vec![0u8; src.len().div_ceil(2)];
        let esc = encode(src, &t, &mut base, &mut resid).unwrap();
        let mut full = vec![0u16; src.len()];
        let mut draft = vec![0u16; src.len()];
        decode(&base, Some(&resid), &t, (&esc.index, &esc.bits), &mut full).unwrap();
        decode(&base, None, &t, (&esc.index, &esc.bits), &mut draft).unwrap();
        (full, draft, esc)
    }

    #[test]
    fn every_bf16_bit_pattern_comes_back_and_the_draft_clears_four_bits() {
        let src: Vec<u16> = (0..=u16::MAX).collect(); // NaNs, infinities, subnormals, both zeros
        let (full, draft, esc) = roundtrip(&src);
        assert_eq!(full, src);
        assert!(draft.iter().zip(&src).all(|(&d, &s)| d == s & 0xFFF0));
        // 256 exponents, 15 in the table: the rest escapes
        assert_eq!(esc.index.len(), 65536 - 15 * 256);
    }

    #[test]
    fn odd_lengths_and_sizes_across_pieces_round_trip() {
        let mut x = 0x9E37_79B9u32;
        let src: Vec<u16> = (0..(CHUNK * 2 + 7))
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                // weight-like: a few exponents around 2^-6, random sign and mantissa
                (((x >> 16) & 0x8000) | ((0x76 + (x % 6)) << 7) | (x >> 9 & 0x7F)) as u16
            })
            .collect();
        let (full, _, esc) = roundtrip(&src);
        assert_eq!(full, src);
        assert!(esc.index.is_empty());
    }

    #[test]
    fn wrong_sizes_are_refused() {
        let t = table(&[0; 4]);
        let mut base = vec![0u8; 3];
        let mut resid = vec![0u8; 2];
        assert!(encode(&[0; 4], &t, &mut base, &mut resid).is_err());
        let mut out = vec![0u16; 4];
        assert!(decode(&[0; 4], Some(&[0; 1]), &t, (&[], &[]), &mut out).is_err());
        assert!(decode(&[0; 4], None, &t, (&[9], &[0]), &mut out).is_err());
    }
}
