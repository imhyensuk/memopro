//! CSV kernels for `memopro.sort` / `memopro.join` (E069 V2, 0356; 0361): row positions, one
//! key per row, and byte gathers, written into the caller's buffers so that no index temporary
//! is allocated per block (NumPy's took 16 bytes per gathered byte, and the allocator kept them:
//! macOS large cache 0060, glibc heaps 0349).
//!
//! CSV as RFC 4180 writes it: lines end in `\n` or `\r\n`, fields are separated by `,`, and a
//! field may be enclosed in `"` (then it may hold `,`, line breaks and `""` for a quote). A row
//! ends at a line break outside quotes.

use crate::error::{Error, Result};

/// How the key field is read by [`rows`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum KeyMode {
    /// Every key must be an integer (an error otherwise).
    Int,
    /// Keys are text: only their positions are given.
    Text,
    /// Integers where they parse; whether all did is returned.
    Detect,
}

/// End of the last complete row of `block` (just after its line break outside quotes); 0 when
/// the block holds no complete row.
pub fn cut(block: &[u8]) -> usize {
    let (mut quoted, mut end) = (false, 0);
    for (i, &b) in block.iter().enumerate() {
        match b {
            b'"' => quoted = !quoted,
            b'\n' if !quoted => end = i + 1,
            _ => {}
        }
    }
    end
}

/// Number of rows (line breaks outside quotes) in `block`.
pub fn count_rows(block: &[u8]) -> usize {
    let mut quoted = false;
    let mut n = 0;
    for &b in block {
        match b {
            b'"' => quoted = !quoted,
            b'\n' if !quoted => n += 1,
            _ => {}
        }
    }
    n
}

/// The bytes of a field without the quotes that enclose it.
fn unquoted(f: &[u8]) -> &[u8] {
    if f.len() >= 2 && f[0] == b'"' && f[f.len() - 1] == b'"' {
        &f[1..f.len() - 1]
    } else {
        f
    }
}

/// Rows of `block` (whole rows, each ending in a line break outside quotes): for row `i`, its
/// start, its length with the line ending, its length without it (`\n` or `\r\n`), the start
/// and end of field `col` (with any enclosing quotes), and that field as an `i64` key when
/// `mode` asks for one. Every row must have `fields` fields. Each output must hold at least the
/// number of rows; returns that number and whether every key was an integer.
#[allow(clippy::too_many_arguments)]
pub fn rows(
    block: &[u8],
    col: usize,
    fields: usize,
    mode: KeyMode,
    starts: &mut [i64],
    lens: &mut [i64],
    body: &mut [i64],
    keys: &mut [i64],
    key_start: &mut [i64],
    key_end: &mut [i64],
) -> Result<(usize, bool)> {
    if col >= fields {
        return Err(Error::InvalidArgument(format!(
            "column {col} of {fields} fields"
        )));
    }
    let cap = [
        starts.len(),
        lens.len(),
        body.len(),
        keys.len(),
        key_start.len(),
        key_end.len(),
    ]
    .into_iter()
    .min()
    .unwrap_or(0);
    let (mut n, mut at, mut all_int) = (0, 0, true);
    let (mut field, mut fs, mut fe, mut quoted) = (0, 0, None, false);
    for (i, &b) in block.iter().enumerate() {
        match b {
            b'"' => quoted = !quoted,
            b',' if !quoted => {
                if field == col {
                    fe = Some(i);
                }
                field += 1;
                if field == col {
                    fs = i + 1;
                }
            }
            b'\n' if !quoted => {
                let end = if i > at && block[i - 1] == b'\r' {
                    i - 1
                } else {
                    i
                };
                if field + 1 != fields {
                    return Err(Error::InvalidArgument(format!(
                        "a row has {} fields, not {fields}",
                        field + 1
                    )));
                }
                if n == cap {
                    return Err(Error::InvalidArgument(format!(
                        "more than {cap} rows for the outputs"
                    )));
                }
                let (ks, ke) = (if col == 0 { at } else { fs }, fe.unwrap_or(end));
                keys[n] = match mode {
                    KeyMode::Text => 0,
                    KeyMode::Int => parse_key(unquoted(&block[ks..ke]))?,
                    KeyMode::Detect => match parse_key(unquoted(&block[ks..ke])) {
                        Ok(v) => v,
                        Err(_) => {
                            all_int = false;
                            0
                        }
                    },
                };
                starts[n] = at as i64;
                lens[n] = (i + 1 - at) as i64;
                body[n] = (end - at) as i64;
                key_start[n] = ks as i64;
                key_end[n] = ke as i64;
                n += 1;
                at = i + 1;
                (field, fs, fe) = (0, 0, None);
            }
            _ => {}
        }
    }
    if at != block.len() {
        return Err(Error::InvalidArgument(
            "a block must end with a whole row (a quote may be left open)".into(),
        ));
    }
    Ok((n, all_int && mode != KeyMode::Text))
}

/// Text keys (`src[starts[i]..ends[i]]`, enclosing quotes removed) as `width`-byte records into
/// `out`, cut or padded with zeros; with `width` 0 nothing is written. Returns the longest key.
pub fn fixed(
    src: &[u8],
    starts: &[i64],
    ends: &[i64],
    width: usize,
    out: &mut [u8],
) -> Result<usize> {
    if starts.len() != ends.len() || out.len() < width * starts.len() {
        return Err(Error::InvalidArgument(
            "key arrays and output do not match".into(),
        ));
    }
    let mut longest = 0;
    for (i, (&s, &e)) in starts.iter().zip(ends).enumerate() {
        let r = range(src.len(), s, e - s)?;
        let k = unquoted(&src[r]);
        longest = longest.max(k.len());
        if width > 0 {
            let rec = &mut out[i * width..(i + 1) * width];
            let m = k.len().min(width);
            rec[..m].copy_from_slice(&k[..m]);
            rec[m..].fill(0);
        }
    }
    Ok(longest)
}

fn parse_key(s: &[u8]) -> Result<i64> {
    let bad = || {
        Error::InvalidArgument(format!(
            "keys must be integers, not {:?}",
            String::from_utf8_lossy(s)
        ))
    };
    let (neg, digits) = match s.first() {
        Some(b'-') => (true, &s[1..]),
        _ => (false, s),
    };
    if digits.is_empty() {
        return Err(bad());
    }
    let mut v: i64 = 0;
    for &d in digits {
        if !d.is_ascii_digit() {
            return Err(bad());
        }
        v = v
            .checked_mul(10)
            .and_then(|v| v.checked_add(i64::from(d - b'0')))
            .ok_or_else(bad)?;
    }
    Ok(if neg { -v } else { v })
}

fn range(len: usize, start: i64, n: i64) -> Result<std::ops::Range<usize>> {
    let (s, e) = (start as usize, (start + n) as usize);
    if start < 0 || n < 0 || e > len {
        return Err(Error::InvalidArgument(format!(
            "segment {start}+{n} outside {len} bytes"
        )));
    }
    Ok(s..e)
}

/// The segments (`starts[i]`, `lens[i]`) of `src`, one after another, into `out`; returns the
/// bytes written.
pub fn gather(src: &[u8], starts: &[i64], lens: &[i64], out: &mut [u8]) -> Result<usize> {
    if starts.len() != lens.len() {
        return Err(Error::InvalidArgument(
            "starts and lengths differ in length".into(),
        ));
    }
    let mut at = 0;
    for (&s, &n) in starts.iter().zip(lens) {
        let r = range(src.len(), s, n)?;
        let end = at + r.len();
        if end > out.len() {
            return Err(Error::InvalidArgument("output too small".into()));
        }
        out[at..end].copy_from_slice(&src[r]);
        at = end;
    }
    Ok(at)
}

/// Joined rows into `out`: for each `i`, `left[ls[i]..][..ll[i]]`, a comma when `comma`,
/// `right[rs[i]..][..rl[i]]` and a line ending (`\r\n` when `crlf`); returns the bytes written.
#[allow(clippy::too_many_arguments)]
pub fn join_rows(
    left: &[u8],
    ls: &[i64],
    ll: &[i64],
    right: &[u8],
    rs: &[i64],
    rl: &[i64],
    comma: bool,
    crlf: bool,
    out: &mut [u8],
) -> Result<usize> {
    let n = ls.len();
    if ll.len() != n || rs.len() != n || rl.len() != n {
        return Err(Error::InvalidArgument("row arrays differ in length".into()));
    }
    let mut at = 0;
    for i in 0..n {
        let a = range(left.len(), ls[i], ll[i])?;
        let b = range(right.len(), rs[i], rl[i])?;
        let end = at + a.len() + usize::from(comma) + b.len() + 1 + usize::from(crlf);
        if end > out.len() {
            return Err(Error::InvalidArgument("output too small".into()));
        }
        out[at..at + a.len()].copy_from_slice(&left[a.clone()]);
        at += a.len();
        if comma {
            out[at] = b',';
            at += 1;
        }
        let blen = b.len();
        out[at..at + blen].copy_from_slice(&right[b]);
        at += blen;
        if crlf {
            out[at] = b'\r';
            at += 1;
        }
        out[at] = b'\n';
        at += 1;
    }
    Ok(at)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(
        block: &[u8],
        col: usize,
        fields: usize,
        mode: KeyMode,
    ) -> Result<(usize, bool, [[i64; 4]; 6])> {
        let mut v = [[0i64; 4]; 6];
        let [s, l, b, k, ks, ke] = &mut v;
        let (n, all) = rows(block, col, fields, mode, s, l, b, k, ks, ke)?;
        Ok((n, all, v))
    }

    #[test]
    fn rows_keys_and_gathers() {
        let block = b"3,x,7\n-12,yy,8\n5,z,9\n";
        let (n, all, [s, l, b, k, _, _]) = run(block, 0, 3, KeyMode::Int).unwrap();
        assert_eq!((n, all), (3, true));
        assert_eq!(
            (&k[..3], &l[..3], &b[..3]),
            (&[3, -12, 5][..], &[6, 9, 6][..], &[5, 8, 5][..])
        );
        let mut out = vec![0u8; 21];
        let n = gather(block, &[s[2], s[0]], &[l[2], l[0]], &mut out).unwrap();
        assert_eq!(&out[..n], b"5,z,9\n3,x,7\n");
        // the last field before \r\n is the key: the \r is the line ending, not the field
        let (n, _, [_, l, b, k, _, _]) = run(b"a,7\r\nb,8\r\n", 1, 2, KeyMode::Int).unwrap();
        assert_eq!(
            (n, &k[..2], &l[..2], &b[..2]),
            (2, &[7, 8][..], &[5, 5][..], &[3, 3][..])
        );
        assert!(run(b"1,2\n", 0, 3, KeyMode::Int).is_err());
        assert!(run(b"x,2\n", 0, 2, KeyMode::Int).is_err());
        assert!(run(b"99999999999999999999,2\n", 0, 2, KeyMode::Int).is_err());
        assert!(!run(b"1,a\nx,b\n", 0, 2, KeyMode::Detect).unwrap().1);
        let mut j = vec![0u8; 32];
        let n = join_rows(b"1,a", &[0], &[3], b"b,c", &[0], &[3], true, false, &mut j).unwrap();
        assert_eq!(&j[..n], b"1,a,b,c\n");
    }

    #[test]
    fn quoted_fields_and_text_keys() {
        // a quoted field holding a comma, a line break and an escaped quote; a quoted key
        let block = b"\"b,1\",\"x\ny\"\"z\"\n\"a\",2\n";
        assert_eq!(count_rows(block), 2);
        assert_eq!(cut(&block[..block.len() - 3]), 15);
        let (n, all, [_, l, _, _, ks, ke]) = run(block, 0, 2, KeyMode::Detect).unwrap();
        assert_eq!((n, all, l[0], l[1]), (2, false, 15, 6));
        let mut out = [0u8; 8];
        let longest = fixed(block, &ks[..2], &ke[..2], 4, &mut out).unwrap();
        assert_eq!((longest, &out), (3, b"b,1\0a\0\0\0"));
        let (_, all, [_, _, _, k, _, _]) = run(b"\"7\",a\n8,b\n", 0, 2, KeyMode::Detect).unwrap();
        assert!(all && k[..2] == [7, 8]);
        assert!(run(b"\"open,1\n", 0, 2, KeyMode::Text).is_err());
    }
}
