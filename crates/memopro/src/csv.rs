//! Simple CSV kernels for `memopro.sort` / `memopro.join` (E069 V2, 0356): row positions, one
//! integer key per row, and byte gathers, written into the caller's buffers so that no index
//! temporary is allocated per block (NumPy's took 16 bytes per gathered byte, and the
//! allocator kept them: macOS large cache 0060, glibc heaps 0349).
//!
//! Simple CSV only: lines end in `\n`, fields are separated by `,`, and no field is quoted.

use crate::error::{Error, Result};

/// Number of lines (`\n` bytes) in `block`.
pub fn count_lines(block: &[u8]) -> usize {
    block.iter().filter(|&&b| b == b'\n').count()
}

/// Rows of `block` (whole lines, each ending in `\n`): for row `i`, its start, its length with
/// the newline, field `col` parsed as an `i64` key, and that field's start and end. Every row
/// must have `fields` fields and no quote. Each output must hold at least the number of rows;
/// returns that number.
#[allow(clippy::too_many_arguments)]
pub fn rows(
    block: &[u8],
    col: usize,
    fields: usize,
    starts: &mut [i64],
    lens: &mut [i64],
    keys: &mut [i64],
    key_start: &mut [i64],
    key_end: &mut [i64],
) -> Result<usize> {
    if col >= fields {
        return Err(Error::InvalidArgument(format!(
            "column {col} of {fields} fields"
        )));
    }
    let cap = [
        starts.len(),
        lens.len(),
        keys.len(),
        key_start.len(),
        key_end.len(),
    ]
    .into_iter()
    .min()
    .unwrap_or(0);
    let (mut n, mut at) = (0, 0);
    while at < block.len() {
        let Some(nl) = block[at..].iter().position(|&b| b == b'\n') else {
            return Err(Error::InvalidArgument(
                "a block must end with a whole line".into(),
            ));
        };
        let line = &block[at..at + nl];
        if n == cap {
            return Err(Error::InvalidArgument(format!(
                "more than {cap} rows for the outputs"
            )));
        }
        let (mut field, mut fs, mut fe) = (0, 0, line.len());
        for (i, &b) in line.iter().enumerate() {
            match b {
                b',' => {
                    if field == col {
                        fe = i;
                    }
                    field += 1;
                    if field == col {
                        fs = i + 1;
                    }
                }
                b'"' => {
                    return Err(Error::InvalidArgument(
                        "quoted CSV fields are not supported".into(),
                    ));
                }
                _ => {}
            }
        }
        if field + 1 != fields {
            return Err(Error::InvalidArgument(format!(
                "a row has {} fields, not {fields}",
                field + 1
            )));
        }
        if field == col {
            fe = line.len();
        }
        keys[n] = parse_key(&line[fs..fe])?;
        starts[n] = at as i64;
        lens[n] = nl as i64 + 1;
        key_start[n] = (at + fs) as i64;
        key_end[n] = (at + fe) as i64;
        n += 1;
        at += nl + 1;
    }
    Ok(n)
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
/// `right[rs[i]..][..rl[i]]` and a newline; returns the bytes written.
#[allow(clippy::too_many_arguments)]
pub fn join_rows(
    left: &[u8],
    ls: &[i64],
    ll: &[i64],
    right: &[u8],
    rs: &[i64],
    rl: &[i64],
    comma: bool,
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
        let end = at + a.len() + usize::from(comma) + b.len() + 1;
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
        out[at] = b'\n';
        at += 1;
    }
    Ok(at)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_keys_and_gathers() {
        let block = b"3,x,7\n-12,yy,8\n5,z,9\n";
        let mut v = [[0i64; 3]; 5];
        let [s, l, k, ks, ke] = &mut v;
        assert_eq!(rows(block, 0, 3, s, l, k, ks, ke).unwrap(), 3);
        assert_eq!(*k, [3, -12, 5]);
        assert_eq!(*l, [6, 9, 6]);
        let mut out = vec![0u8; 21];
        let n = gather(block, &[s[2], s[0]], &[l[2], l[0]], &mut out).unwrap();
        assert_eq!(&out[..n], b"5,z,9\n3,x,7\n");
        let mut v2 = [[0i64; 3]; 5];
        let [s2, l2, k2, ks2, ke2] = &mut v2;
        assert_eq!(rows(block, 2, 3, s2, l2, k2, ks2, ke2).unwrap(), 3);
        assert_eq!(*k2, [7, 8, 9]);
        assert!(rows(b"1,\"a\"\n", 0, 2, s, l, k, ks, ke).is_err());
        assert!(rows(b"1,2\n", 0, 3, s, l, k, ks, ke).is_err());
        assert!(rows(b"x,2\n", 0, 2, s, l, k, ks, ke).is_err());
        assert!(rows(b"99999999999999999999,2\n", 0, 2, s, l, k, ks, ke).is_err());
        let mut j = vec![0u8; 32];
        let n = join_rows(b"1,a", &[0], &[3], b"b,c", &[0], &[3], true, &mut j).unwrap();
        assert_eq!(&j[..n], b"1,a,b,c\n");
    }
}
