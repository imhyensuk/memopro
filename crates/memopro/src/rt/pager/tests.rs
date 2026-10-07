//! Pager tests (Linux, macOS): data larger than the budget comes back bit for bit through plain loads
//! and stores, from one thread or several, and incompressible data is kept (and counted).

use super::*;
use crate::error::Error;

const MIB: usize = 1 << 20;

/// A pager with 256 KiB chunks, or `None` where userfaultfd is not permitted (said on stderr).
fn pager(budget_mib: u64) -> Option<Pager> {
    let config = PagerConfig {
        chunk: 256 * 1024,
        ..PagerConfig::new(budget_mib << 20)
    };
    match Pager::new(config) {
        Ok(p) => Some(p),
        Err(Error::Unsupported(why)) => {
            eprintln!("skipping the pager test: {why}");
            None
        }
        Err(e) => panic!("{e}"),
    }
}

fn pattern(i: usize) -> u32 {
    ((i / 7) % 1000) as u32 // compresses well, like widened bf16 or small integers
}

#[test]
fn memory_larger_than_the_budget_reads_back_exact() {
    let Some(p) = pager(8) else { return };
    let len = 32 * MIB;
    let ptr = p.map(len).unwrap();
    // SAFETY: a fresh region of `len` bytes, used only here.
    let words = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u32, len / 4) };
    assert!(words.iter().step_by(4099).all(|&w| w == 0), "starts zeroed");
    for (i, w) in words.iter_mut().enumerate() {
        *w = pattern(i);
    }
    let expect: u64 = (0..len / 4).map(|i| u64::from(pattern(i))).sum();
    for _ in 0..3 {
        let sum: u64 = words.iter().map(|&w| u64::from(w)).sum();
        assert_eq!(sum, expect);
    }
    let s = p.stats();
    assert!(s.evictions > 0 && s.restores > 0, "{s:?}");
    assert_eq!(s.overruns, 0, "{s:?}");
    assert!(s.peak_used <= s.limit, "{s:?}");
    // SAFETY: the region is no longer used.
    assert_eq!(unsafe { p.unmap(ptr) }.unwrap(), len);
    assert_eq!(p.stats().used, 0);
}

#[test]
fn threads_write_and_read_while_chunks_come_and_go() {
    let Some(p) = pager(8) else { return };
    let len = 24 * MIB;
    let ptr = p.map(len).unwrap() as usize;
    let quarter = len / 4 / 4; // words per thread
    std::thread::scope(|s| {
        for t in 0..4 {
            s.spawn(move || {
                // SAFETY: each thread writes its own quarter of the region.
                let words = unsafe {
                    std::slice::from_raw_parts_mut((ptr as *mut u32).add(t * quarter), quarter)
                };
                for round in 0..2u32 {
                    for (i, w) in words.iter_mut().enumerate() {
                        *w = pattern(t * quarter + i) + round;
                    }
                }
            });
        }
    });
    // SAFETY: the region, all threads done.
    let words = unsafe { std::slice::from_raw_parts(ptr as *const u32, len / 4) };
    for (i, &w) in words.iter().enumerate() {
        assert_eq!(w, pattern(i) + 1, "word {i}");
    }
    assert!(p.stats().evictions > 0);
    // SAFETY: done with it.
    unsafe { p.unmap(ptr as *mut u8) }.unwrap();
}

#[test]
fn incompressible_chunks_stay_and_are_counted() {
    let Some(p) = pager(4) else { return };
    let len = 8 * MIB;
    let ptr = p.map(len).unwrap();
    // SAFETY: a fresh region.
    let bytes = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    for b in bytes.iter_mut() {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *b = x as u8;
    }
    let mut y = 0x9E37_79B9_7F4A_7C15u64;
    for &b in bytes.iter() {
        y ^= y << 13;
        y ^= y >> 7;
        y ^= y << 17;
        assert_eq!(b, y as u8);
    }
    let s = p.stats();
    assert!(s.incompressible > 0 && s.overruns > 0, "{s:?}");
    // SAFETY: done with it.
    unsafe { p.unmap(ptr) }.unwrap();
}

#[test]
fn a_repeated_scan_keeps_a_budget_worth() {
    let Some(p) = pager(8) else { return };
    let len = 32 * MIB;
    let ptr = p.map(len).unwrap();
    // SAFETY: a fresh region.
    let words = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u32, len / 4) };
    for (i, w) in words.iter_mut().enumerate() {
        *w = pattern(i);
    }
    let mut restores = Vec::new();
    for _ in 0..3 {
        let before = p.stats().restores;
        std::hint::black_box(words.iter().map(|&w| u64::from(w)).sum::<u64>());
        restores.push(p.stats().restores - before);
    }
    let chunks = (len / (256 * 1024)) as u64;
    let fits = p.limit() / (256 * 1024) as u64;
    // LRU would bring every chunk back each pass; the reuse-distance choice keeps most of what fits
    assert!(
        restores[2] <= chunks - fits / 2,
        "{restores:?} of {chunks} chunks, room for {fits}"
    );
    // SAFETY: done with it.
    unsafe { p.unmap(ptr) }.unwrap();
}

#[test]
fn two_arrays_used_in_step_do_not_ping_pong() {
    // `dst[i] = src[i] + 1` over two regions larger than the budget: evicting the chunk just
    // faulted in (the other operand) made every few elements fault (CI took hours, 0128)
    let Some(p) = pager(8) else { return };
    let len = 16 * MIB;
    let (src, dst) = (p.map(len).unwrap(), p.map(len).unwrap());
    // SAFETY: two fresh regions of `len` bytes, used only here.
    let (a, b) = unsafe {
        (
            std::slice::from_raw_parts_mut(src as *mut u32, len / 4),
            std::slice::from_raw_parts_mut(dst as *mut u32, len / 4),
        )
    };
    for (i, w) in a.iter_mut().enumerate() {
        *w = pattern(i);
    }
    let before = p.stats().faults;
    for (x, y) in b.iter_mut().zip(a.iter()) {
        *x = *y + 1;
    }
    let faults = p.stats().faults - before;
    let chunks = (2 * len / (256 * 1024)) as u64;
    assert!(faults <= 4 * chunks, "{faults} faults for {chunks} chunks");
    assert!(
        b.iter()
            .enumerate()
            .step_by(1009)
            .all(|(i, &w)| w == pattern(i) + 1)
    );
    // SAFETY: done with both.
    unsafe {
        p.unmap(src).unwrap();
        p.unmap(dst).unwrap();
    }
}

/// The process footprint, or `None` (said on stderr) where it cannot be measured.
fn footprint() -> Option<u64> {
    let f = crate::rt::process_footprint();
    if f.is_none() {
        eprintln!("skipping: no process footprint here");
    }
    f
}

#[test]
#[ignore = "measures the whole process: run with --ignored --test-threads=1"]
fn evicted_chunks_leave_the_process() {
    // tests run in parallel threads of one process: allow other tests' memory as slack
    let Some(p) = pager(8) else { return };
    let Some(start) = footprint() else { return };
    let len = 96 * MIB;
    let ptr = p.map(len).unwrap();
    // SAFETY: a fresh region.
    let words = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u32, len / 4) };
    for (i, w) in words.iter_mut().enumerate() {
        *w = pattern(i);
    }
    let grown = footprint().unwrap().saturating_sub(start);
    assert!(
        grown < 48 * MIB as u64,
        "footprint grew {grown} for 96 MiB under 8 MiB"
    );
    assert!(
        words
            .iter()
            .enumerate()
            .step_by(997)
            .all(|(i, &w)| w == pattern(i))
    );
    // SAFETY: done with it.
    unsafe { p.unmap(ptr) }.unwrap();
}

#[test]
#[ignore = "measures the whole process: run with --ignored --test-threads=1"]
fn a_process_budget_counts_memory_outside_the_pager() {
    let Some(now) = footprint() else { return };
    let config = PagerConfig {
        chunk: 256 * 1024,
        process_budget: Some(now + 24 * MIB as u64),
        ..PagerConfig::new(64 << 20)
    };
    let p = match Pager::new(config) {
        Ok(p) => p,
        Err(Error::Unsupported(why)) => return eprintln!("skipping: {why}"),
        Err(e) => panic!("{e}"),
    };
    let outside = vec![7u8; 16 * MIB]; // memory the pager does not own, touched
    let len = 32 * MIB;
    let ptr = p.map(len).unwrap();
    // SAFETY: a fresh region.
    let words = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u32, len / 4) };
    for (i, w) in words.iter_mut().enumerate() {
        *w = pattern(i);
    }
    assert!(
        words
            .iter()
            .enumerate()
            .step_by(991)
            .all(|(i, &w)| w == pattern(i))
    );
    let s = p.stats();
    // a 64 MiB budget alone would have kept all 32 MiB; the process budget left far less
    assert!(s.evictions > 0, "{s:?}");
    assert!(s.limit_low < 24 * MIB as u64, "{s:?}");
    assert!(s.outside_peak >= 16 * MIB as u64, "{s:?}");
    std::hint::black_box(&outside);
    // SAFETY: done with it.
    unsafe { p.unmap(ptr) }.unwrap();
}

#[test]
fn several_pagers_serve_their_own_faults() {
    let (Some(a), Some(b)) = (pager(4), pager(4)) else {
        return;
    };
    let len = 12 * MIB;
    let (pa, pb) = (a.map(len).unwrap() as usize, b.map(len).unwrap() as usize);
    std::thread::scope(|s| {
        for (ptr, add) in [(pa, 1u32), (pb, 2u32)] {
            s.spawn(move || {
                // SAFETY: each thread uses its own region.
                let w = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u32, len / 4) };
                for (i, x) in w.iter_mut().enumerate() {
                    *x = pattern(i) + add;
                }
                for _ in 0..2 {
                    assert!(w.iter().enumerate().all(|(i, &x)| x == pattern(i) + add));
                }
            });
        }
    });
    assert!(a.stats().restores > 0 && b.stats().restores > 0);
    // SAFETY: done with both.
    unsafe {
        a.unmap(pa as *mut u8).unwrap();
        b.unmap(pb as *mut u8).unwrap();
    }
}
