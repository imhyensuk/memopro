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

/// EDR v2 (0315): the exponents entropy-coded, everything else as in v1.
///
/// - `exps`: canonical Huffman codes of the exponents (at most 15 bits), in blocks of [`BLOCK`]
///   elements that each start on a byte; `offsets[b]` is where block `b` starts, so blocks
///   decode in parallel. Every exponent has a code: no escapes.
/// - `hi`: sign and top 3 mantissa bits, a nibble per element (two per byte, low first)
/// - `lo`: the low 4 mantissa bits, a nibble per element
///
/// About 10.7 bits per weight for bf16 LLM weights (v1: 12). `exps` + `hi` is the draft.
pub mod v2 {
    use crate::{Error, Result};
    use rayon::prelude::*;

    /// Elements per independently decodable block of exponent codes.
    pub const BLOCK: usize = 4096;
    /// Longest code; the decoding table has 2^MAX_LEN entries.
    pub const MAX_LEN: u32 = 15;

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct Encoded {
        pub lens: [u8; 256],
        pub offsets: Vec<u64>,
        pub exps: Vec<u8>,
        pub hi: Vec<u8>,
        pub lo: Vec<u8>,
    }

    impl Encoded {
        pub fn nbytes(&self) -> usize {
            256 + self.offsets.len() * 8 + self.exps.len() + self.hi.len() + self.lo.len()
        }
    }

    /// Code lengths (at most `MAX_LEN`) of a Huffman code for `counts`.
    pub fn lengths(counts: &[u64; 256]) -> [u8; 256] {
        let mut lens = [0u8; 256];
        let used: Vec<usize> = (0..256).filter(|&s| counts[s] > 0).collect();
        if used.len() == 1 {
            lens[used[0]] = 1;
            return lens;
        }
        if used.is_empty() {
            return lens;
        }
        // Huffman by repeated merging of the two lightest nodes; a node is a list of symbols
        let mut nodes: Vec<(u64, Vec<usize>)> =
            used.iter().map(|&s| (counts[s], vec![s])).collect();
        while nodes.len() > 1 {
            nodes.sort_by_key(|a| std::cmp::Reverse(a.0));
            let (wa, a) = nodes.pop().expect("two nodes");
            let (wb, b) = nodes.pop().expect("two nodes");
            for &s in a.iter().chain(&b) {
                lens[s] += 1;
            }
            nodes.push((wa + wb, a.into_iter().chain(b).collect()));
        }
        // limit to MAX_LEN, then lengthen the rarest short codes until the Kraft sum fits
        for &s in &used {
            lens[s] = lens[s].min(MAX_LEN as u8);
        }
        let kraft =
            |l: &[u8; 256]| -> u64 { used.iter().map(|&s| 1u64 << (MAX_LEN - l[s] as u32)).sum() };
        let mut by_rarity = used.clone();
        by_rarity.sort_by_key(|&s| counts[s]);
        while kraft(&lens) > 1u64 << MAX_LEN {
            let s = *by_rarity
                .iter()
                .find(|&&s| (lens[s] as u32) < MAX_LEN)
                .expect("a code shorter than the limit");
            lens[s] += 1;
        }
        lens
    }

    /// Canonical codes for `lens` (codes in increasing (length, symbol) order).
    fn codes(lens: &[u8; 256]) -> [u16; 256] {
        let mut order: Vec<usize> = (0..256).filter(|&s| lens[s] > 0).collect();
        order.sort_by_key(|&s| (lens[s], s));
        let mut codes = [0u16; 256];
        let (mut code, mut len) = (0u32, 0u8);
        for s in order {
            code <<= lens[s] - len;
            len = lens[s];
            codes[s] = code as u16;
            code += 1;
        }
        codes
    }

    /// Bits looked at by the first-level decoding table (16 KiB: fits the L1 cache).
    const FAST: u32 = 12;

    /// Decoding tables. First level: the next `FAST` bits -> up to two exponents (bits 0-7, 8-15),
    /// the first code's length (16-19), both codes' length (20-24) and how many (25-26; 0 = a
    /// code longer than `FAST`). Second level: the next `MAX_LEN` bits -> exponent << 4 | length.
    fn tables(lens: &[u8; 256]) -> (Vec<u32>, Vec<u16>) {
        let codes = codes(lens);
        let mut full = vec![0u16; 1 << MAX_LEN];
        for s in 0..256 {
            let l = lens[s] as u32;
            if l > 0 {
                let first = (codes[s] as usize) << (MAX_LEN - l);
                full[first..first + (1 << (MAX_LEN - l))].fill(((s as u16) << 4) | l as u16);
            }
        }
        // one code at the top of `FAST` bits, from the full table
        let one = |i: usize| full[i << (MAX_LEN - FAST)];
        let fast = (0..1usize << FAST)
            .map(|i| {
                let a = one(i);
                let la = (a & 0xF) as u32;
                if a == 0 || la > FAST {
                    return 0;
                }
                let b = one((i << la) & ((1 << FAST) - 1));
                let lb = (b & 0xF) as u32;
                let (s1, s2) = ((a >> 4) as u32, (b >> 4) as u32);
                if b != 0 && la + lb <= FAST {
                    s1 | s2 << 8 | la << 16 | (la + lb) << 20 | 2 << 25
                } else {
                    s1 | la << 16 | la << 20 | 1 << 25
                }
            })
            .collect();
        (fast, full)
    }

    fn nibbles(src: &[u16], f: impl Fn(u16) -> u8 + Sync) -> Vec<u8> {
        let mut out = vec![0u8; src.len().div_ceil(2)];
        out.par_iter_mut().enumerate().for_each(|(j, o)| {
            let a = f(src[2 * j]);
            let b = src.get(2 * j + 1).map_or(0, |&x| f(x));
            *o = a | (b << 4);
        });
        out
    }

    pub fn encode(src: &[u16]) -> Encoded {
        let counts = src
            .par_chunks(1 << 20)
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
        let lens = lengths(&counts);
        let codes = codes(&lens);
        let blocks: Vec<Vec<u8>> = src
            .par_chunks(BLOCK)
            .map(|c| {
                let mut out = Vec::with_capacity(c.len() / 2);
                let (mut acc, mut n) = (0u64, 0u32);
                for &x in c {
                    let e = ((x >> 7) & 0xFF) as usize;
                    acc = (acc << lens[e]) | codes[e] as u64;
                    n += lens[e] as u32;
                    while n >= 8 {
                        n -= 8;
                        out.push((acc >> n) as u8);
                    }
                }
                if n > 0 {
                    out.push((acc << (8 - n)) as u8);
                }
                out
            })
            .collect();
        let mut offsets = Vec::with_capacity(blocks.len());
        let mut exps = Vec::with_capacity(blocks.iter().map(Vec::len).sum());
        for b in blocks {
            offsets.push(exps.len() as u64);
            exps.extend_from_slice(&b);
        }
        let hi = nibbles(src, |x| (((x >> 15) as u8) << 3) | ((x >> 4) & 7) as u8);
        let lo = nibbles(src, |x| (x & 0xF) as u8);
        Encoded {
            lens,
            offsets,
            exps,
            hi,
            lo,
        }
    }

    /// Decode into `out` (`out.len()` elements): every bit with `lo`, the draft (low 4 mantissa
    /// bits cleared) without.
    pub fn decode(
        lens: &[u8; 256],
        offsets: &[u64],
        exps: &[u8],
        hi: &[u8],
        lo: Option<&[u8]>,
        out: &mut [u16],
    ) -> Result<()> {
        let n = out.len();
        if offsets.len() != n.div_ceil(BLOCK)
            || hi.len() != n.div_ceil(2)
            || lo.is_some_and(|l| l.len() != n.div_ceil(2))
        {
            return Err(Error::InvalidArgument(format!(
                "edr::v2::decode: parts do not describe {n} elements"
            )));
        }
        let kraft: u64 = lens
            .iter()
            .filter(|&&l| l > 0)
            .map(|&l| 1u64 << (MAX_LEN.saturating_sub(l as u32)))
            .sum();
        if lens.iter().any(|&l| l as u32 > MAX_LEN) || kraft > 1 << MAX_LEN {
            return Err(Error::InvalidArgument(
                "edr::v2::decode: code lengths do not form a prefix code".into(),
            ));
        }
        let (fast, full) = tables(lens);
        // a nibble byte -> both elements' bits in place (element 2k low, 2k + 1 high half)
        let both = |f: fn(u32) -> u32| -> [u32; 256] {
            std::array::from_fn(|b| f(b as u32 & 0xF) | f(b as u32 >> 4) << 16)
        };
        let his = both(|h| (h >> 3) << 15 | (h & 7) << 4);
        let los = both(|l| l);
        out.par_chunks_mut(BLOCK * LANES)
            .enumerate()
            .try_for_each(|(g, o)| {
                // the exponents of up to LANES blocks, decoded in lockstep: one block's codes
                // form a chain of dependent look-ups, several blocks keep the core busy
                let mut ex = [[0u8; BLOCK + 1]; LANES];
                let mut lanes = Vec::with_capacity(LANES);
                for (k, part) in o.chunks(BLOCK).enumerate() {
                    let b = g * LANES + k;
                    let start = offsets[b] as usize;
                    let end = offsets.get(b + 1).map_or(exps.len(), |&x| x as usize);
                    if start > end || end > exps.len() {
                        return Err(Error::InvalidArgument(
                            "edr::v2::decode: bad block offsets".into(),
                        ));
                    }
                    lanes.push(Lane::new(&exps[start..end], part.len()));
                }
                if let ([a, b, c, d], [la, lb, lc, ld]) = (&mut ex, &mut lanes[..]) {
                    while !(la.done() || lb.done() || lc.done() || ld.done()) {
                        la.step(&fast, &full, a)?;
                        lb.step(&fast, &full, b)?;
                        lc.step(&fast, &full, c)?;
                        ld.step(&fast, &full, d)?;
                    }
                }
                for (lane, ex) in lanes.iter_mut().zip(&mut ex) {
                    while !lane.done() {
                        lane.step(&fast, &full, ex)?;
                    }
                }
                // then sign and mantissa nibbles around them
                for (k, (o, ex)) in o.chunks_mut(BLOCK).zip(&ex).enumerate() {
                    let half = (g * LANES + k) * BLOCK / 2;
                    let m = o.len();
                    let hi = &hi[half..half + m.div_ceil(2)];
                    let lo = lo.map(|lo| &lo[half..half + m.div_ceil(2)]);
                    let pairs = m / 2;
                    let (ox, ot) = o.split_at_mut(2 * pairs);
                    let pair = |h: u8, l: u8, e: &[u8]| {
                        his[h as usize] | los[l as usize] | (e[0] as u32) << 7 | (e[1] as u32) << 23
                    };
                    match lo {
                        Some(lo) => {
                            for (((x, e), &h), &l) in ox
                                .chunks_exact_mut(2)
                                .zip(ex.chunks_exact(2))
                                .zip(hi)
                                .zip(lo)
                            {
                                let v = pair(h, l, e);
                                (x[0], x[1]) = (v as u16, (v >> 16) as u16);
                            }
                        }
                        None => {
                            for ((x, e), &h) in
                                ox.chunks_exact_mut(2).zip(ex.chunks_exact(2)).zip(hi)
                            {
                                let v = pair(h, 0, e);
                                (x[0], x[1]) = (v as u16, (v >> 16) as u16);
                            }
                        }
                    }
                    if let Some(x) = ot.first_mut() {
                        let l = lo.map_or(0, |lo| lo[pairs]);
                        *x = pair(hi[pairs], l, &ex[2 * pairs..]) as u16;
                    }
                }
                Ok(())
            })
    }

    /// Blocks decoded together by one thread.
    const LANES: usize = 4;

    /// Where the decoding of one block's exponent codes is.
    struct Lane<'a> {
        bytes: &'a [u8],
        len: usize,
        acc: u64,
        n: u32,
        pos: usize,
        i: usize,
    }

    impl<'a> Lane<'a> {
        fn new(bytes: &'a [u8], len: usize) -> Self {
            Lane {
                bytes,
                len,
                acc: 0,
                n: 0,
                pos: 0,
                i: 0,
            }
        }

        fn done(&self) -> bool {
            self.i >= self.len
        }

        /// One look-up: one or two exponents into `ex`.
        #[inline(always)]
        fn step(&mut self, fast: &[u32], full: &[u16], ex: &mut [u8; BLOCK + 1]) -> Result<()> {
            if self.n < MAX_LEN {
                // 32 more bits, zeros past the end of the block
                let (b, pos) = (self.bytes, self.pos);
                let v = match b.get(pos..pos + 4) {
                    Some(w) => u32::from_be_bytes(w.try_into().expect("4 bytes")),
                    None => {
                        (0..4).fold(0u32, |v, k| (v << 8) | *b.get(pos + k).unwrap_or(&0) as u32)
                    }
                };
                self.acc = (self.acc << 32) | v as u64;
                self.n += 32;
                self.pos += 4;
            }
            let i = self.i.min(BLOCK - 1);
            let e = fast[((self.acc >> (self.n - FAST)) & ((1 << FAST) - 1)) as usize];
            if e != 0 {
                // no branch on one or two: both slots written, `i` moves by the count (a second
                // code past the block's end only reads padding)
                ex[i] = e as u8;
                ex[i + 1] = (e >> 8) as u8;
                self.n -= (e >> 20) & 0x1F;
                self.i += (e >> 25) as usize;
            } else {
                let e = full[((self.acc >> (self.n - MAX_LEN)) & ((1 << MAX_LEN) - 1)) as usize];
                if e == 0 {
                    return Err(Error::InvalidArgument(
                        "edr::v2::decode: corrupt exponent codes".into(),
                    ));
                }
                ex[i] = (e >> 4) as u8;
                self.n -= (e & 0xF) as u32;
                self.i += 1;
            }
            Ok(())
        }
    }
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

    #[test]
    fn v2_round_trips_every_pattern_and_weight_like_data() {
        let all: Vec<u16> = (0..=u16::MAX).collect();
        let e = v2::encode(&all);
        let mut out = vec![0u16; all.len()];
        v2::decode(&e.lens, &e.offsets, &e.exps, &e.hi, Some(&e.lo), &mut out).unwrap();
        assert_eq!(out, all);
        v2::decode(&e.lens, &e.offsets, &e.exps, &e.hi, None, &mut out).unwrap();
        assert!(out.iter().zip(&all).all(|(&d, &s)| d == s & 0xFFF0));
        let mut x = 0x2545_F491u32;
        let w: Vec<u16> = (0..(v2::BLOCK * 3 + 5))
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                let e = 0x78 - (x % 7).min(x % 5); // skewed exponents, as in weights
                (((x >> 16) & 0x8000) | (e << 7) | (x >> 9 & 0x7F)) as u16
            })
            .collect();
        let e = v2::encode(&w);
        let mut out = vec![0u16; w.len()];
        v2::decode(&e.lens, &e.offsets, &e.exps, &e.hi, Some(&e.lo), &mut out).unwrap();
        assert_eq!(out, w);
        assert!((e.nbytes() as f64) * 8.0 / (w.len() as f64) < 11.5);
    }

    #[test]
    fn v2_refuses_bad_parts() {
        let e = v2::encode(&[0x3F80, 0x4000, 0xBF80]);
        let mut out = vec![0u16; 3];
        let mut lens = e.lens;
        lens[0] = 16;
        assert!(v2::decode(&lens, &e.offsets, &e.exps, &e.hi, None, &mut out).is_err());
        lens = [1; 256];
        assert!(v2::decode(&lens, &e.offsets, &e.exps, &e.hi, None, &mut out).is_err());
        assert!(v2::decode(&e.lens, &[], &e.exps, &e.hi, None, &mut out).is_err());
        assert!(v2::decode(&e.lens, &[9], &e.exps, &e.hi, None, &mut out).is_err());
        assert!(v2::decode(&e.lens, &e.offsets, &e.exps, &e.hi[..1], None, &mut out).is_err());
    }

    #[test]
    fn v2_code_lengths_stay_within_the_limit() {
        let mut counts = [0u64; 256];
        for (i, c) in counts.iter_mut().enumerate() {
            *c = 1u64 << (i % 40); // a skew that would give codes far longer than 15 bits
        }
        let l = v2::lengths(&counts);
        assert!(l.iter().all(|&x| (1..=v2::MAX_LEN as u8).contains(&x)));
        let kraft: u64 = l.iter().map(|&x| 1u64 << (v2::MAX_LEN - x as u32)).sum();
        assert!(kraft <= 1 << v2::MAX_LEN);
    }
}
