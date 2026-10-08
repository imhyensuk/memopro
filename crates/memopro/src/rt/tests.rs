//! Runtime tests: lossless round trips (I4), the budget ceiling (I1), pins (I2), ways back (I3),
//! the reuse-distance policy on repeating scans (E025 H7), and concurrent use.

use super::*;
use std::io::Write;
use std::sync::OnceLock;

const MIB: usize = 1 << 20;

/// A shared 24 MiB test file with deterministic content (written once per test run).
fn data_file() -> &'static (PathBuf, Vec<u8>) {
    static FILE: OnceLock<(PathBuf, Vec<u8>)> = OnceLock::new();
    FILE.get_or_init(|| {
        let data: Vec<u8> = (0..24 * MIB as u64)
            .map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8)
            .collect();
        let dir = std::env::temp_dir().join(format!("memopro-rt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("data.bin");
        std::fs::File::create(&path)
            .unwrap()
            .write_all(&data)
            .unwrap();
        (path, data)
    })
}

/// A runtime without prefetching, so counts are deterministic.
fn runtime(budget_mib: u64, policy: Policy) -> Runtime {
    let mut c = Config::new(budget_mib * MIB as u64);
    c.policy = policy;
    c.prefetch = false;
    Runtime::new(c).unwrap()
}

fn prefetching(budget_mib: u64, lookahead_mib: u64) -> Runtime {
    let mut c = Config::new(budget_mib * MIB as u64);
    c.lookahead = lookahead_mib * MIB as u64;
    Runtime::new(c).unwrap()
}

/// f32 values whose low 16 bits are zero (like bf16 widened): compress well.
fn widened(n_floats: usize, seed: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(n_floats * 4);
    let mut x = seed | 1;
    for _ in 0..n_floats {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        let hi = (0x3c00 + (x % 0x0800)) as u16; // small positive floats
        out.extend_from_slice(&((hi as u32) << 16).to_le_bytes());
    }
    out
}

fn noise(n: usize, seed: u64) -> Vec<u8> {
    let mut x = seed | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 32) as u8
        })
        .collect()
}

#[test]
fn budget_must_leave_room_for_the_reserve() {
    assert!(Runtime::new(Config::new(MIB as u64)).is_err());
    let rt = runtime(32, Policy::ReuseDistance);
    assert!(rt.limit() < 32 * MIB as u64);
    assert!(rt.alloc(0, 1).is_err());
    assert!(rt.alloc(10, 4).is_err()); // 10 is not a multiple of 4
    assert!(rt.alloc(12, 3).is_err()); // 3 is not an element size
    assert!(rt.pin(999, false).is_err());
}

#[test]
fn alloc_write_read_round_trip() {
    let rt = runtime(32, Policy::ReuseDistance);
    let id = rt.alloc(3 * MIB, 4).unwrap();
    {
        let mut p = rt.pin(id, true).unwrap();
        for (i, b) in p.as_mut_slice().unwrap().iter_mut().enumerate() {
            *b = (i % 253) as u8;
        }
    }
    let p = rt.pin(id, false).unwrap();
    assert!(
        p.as_slice()
            .iter()
            .enumerate()
            .all(|(i, &b)| b == (i % 253) as u8)
    );
    assert_eq!(rt.state(id).unwrap(), BufferState::Resident);
    assert_eq!(rt.nbytes(id).unwrap(), 3 * MIB);
}

#[test]
fn file_buffers_load_on_first_pin_drop_and_reread_bit_exact() {
    let (path, data) = data_file();
    let rt = runtime(16, Policy::ReuseDistance);
    let a = rt.add_file(path, 100, 4 * MIB, 1).unwrap();
    assert_eq!(rt.state(a).unwrap(), BufferState::Unloaded);
    assert_eq!(
        rt.pin(a, false).unwrap().as_slice(),
        &data[100..100 + 4 * MIB]
    );
    assert!(rt.evict(a).unwrap());
    assert_eq!(rt.state(a).unwrap(), BufferState::Dropped);
    assert_eq!(
        rt.pin(a, false).unwrap().as_slice(),
        &data[100..100 + 4 * MIB]
    );
    let s = rt.stats();
    assert_eq!((s.loads, s.rereads, s.drops), (1, 1, 1));
    assert_eq!(s.written_bytes, 0);
    assert!(rt.add_file(path, (24 * MIB - 10) as u64, 11, 1).is_err());
}

#[test]
fn the_ceiling_holds_and_every_buffer_comes_back_exact() {
    let (path, data) = data_file();
    let rt = runtime(12, Policy::ReuseDistance);
    // 24 MiB of file buffers of mixed sizes in a budget of about 8 MiB for buffers
    let mut ids = Vec::new();
    let mut off = 0usize;
    let mut size = 700 * 1024;
    while off + size <= 24 * MIB {
        ids.push((rt.add_file(path, off as u64, size, 1).unwrap(), off, size));
        off += size;
        size = 300 * 1024 + (size * 7) % (1500 * 1024);
    }
    let mut x: u64 = 12345;
    for _ in 0..400 {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        let (id, off, size) = ids[(x % ids.len() as u64) as usize];
        let p = rt.pin(id, false).unwrap();
        assert_eq!(p.as_slice(), &data[off..off + size]);
        let s = rt.stats();
        assert!(
            s.used <= rt.limit(),
            "used {} over limit {}",
            s.used,
            rt.limit()
        );
    }
    let s = rt.stats();
    assert!(s.peak_used <= rt.limit());
    assert!(s.rereads > 0 && s.drops > 0);
}

#[test]
fn pinned_buffers_stay_and_too_many_pins_are_refused() {
    let (path, _) = data_file();
    let rt = runtime(12, Policy::ReuseDistance);
    let ids: Vec<_> = (0..6)
        .map(|i| rt.add_file(path, (i * 2 * MIB) as u64, 2 * MIB, 1).unwrap())
        .collect();
    let first = rt.pin(ids[0], false).unwrap();
    let second = rt.pin(ids[1], false).unwrap();
    for &id in &ids[2..] {
        drop(rt.pin(id, false).unwrap());
    }
    assert_eq!(rt.state(ids[0]).unwrap(), BufferState::Resident);
    assert_eq!(rt.state(ids[1]).unwrap(), BufferState::Resident);
    // keep pinning until the budget is full of pins: then the runtime must refuse, not overrun
    let mut held = vec![first, second];
    let mut refused = false;
    for &id in &ids[2..] {
        match rt.pin(id, false) {
            Ok(p) => held.push(p),
            Err(Error::Budget(_)) => {
                refused = true;
                break;
            }
            Err(e) => panic!("unexpected error {e}"),
        }
    }
    assert!(refused);
    assert!(rt.stats().refusals >= 1);
    assert!(rt.free(ids[0]).is_err()); // pinned
    drop(held);
    assert!(rt.free(ids[0]).is_ok());
}

#[test]
fn buffers_without_a_file_are_compressed_and_come_back_exact() {
    let rt = runtime(16, Policy::ReuseDistance);
    let mut ids = Vec::new();
    for k in 0..8 {
        let content = widened(MIB / 2, 7 + k); // 2 MiB each, 16 MiB in total
        let id = rt.alloc(content.len(), 4).unwrap();
        rt.pin(id, true)
            .unwrap()
            .as_mut_slice()
            .unwrap()
            .copy_from_slice(&content);
        ids.push((id, content));
        assert!(rt.stats().used <= rt.limit());
    }
    let s = rt.stats();
    assert!(s.compressions > 0, "nothing was compressed: {s:?}");
    for (id, content) in &ids {
        assert_eq!(rt.pin(*id, false).unwrap().as_slice(), &content[..]);
    }
    let s = rt.stats();
    assert!(s.decompressions > 0);
    assert!(s.peak_used <= rt.limit());
    assert_eq!(s.written_bytes, 0);
}

#[test]
fn incompressible_data_without_a_file_is_refused_not_overrun() {
    let rt = runtime(12, Policy::ReuseDistance);
    let mut refused = false;
    for k in 0..10 {
        let content = noise(2 * MIB, 99 + k);
        match rt.alloc(content.len(), 1) {
            Ok(id) => rt
                .pin(id, true)
                .unwrap()
                .as_mut_slice()
                .unwrap()
                .copy_from_slice(&content),
            Err(Error::Budget(_)) => {
                refused = true;
                break;
            }
            Err(e) => panic!("unexpected error {e}"),
        }
    }
    assert!(refused);
    let s = rt.stats();
    assert!(s.incompressible > 0);
    assert!(s.peak_used <= rt.limit());
}

#[test]
fn writing_makes_the_file_unusable_as_a_way_back() {
    let (path, data) = data_file();
    let rt = runtime(16, Policy::ReuseDistance);
    let id = rt.add_file(path, 0, 2 * MIB, 4).unwrap();
    drop(rt.pin(id, false).unwrap());
    assert!(rt.pin(id, true).is_ok()); // written: the file no longer matches
    // evicting now has to keep the data: dropping would lose the write
    let _ = rt.evict(id).unwrap();
    assert_ne!(rt.state(id).unwrap(), BufferState::Dropped);
    assert_eq!(rt.pin(id, false).unwrap().as_slice(), &data[..2 * MIB]);
    // a read-only pin and a writable pin never coexist
    let r = rt.pin(id, false).unwrap();
    assert!(rt.pin(id, true).is_err());
    drop(r);
}

#[test]
fn a_changed_file_fails_the_reread_instead_of_giving_wrong_data() {
    let dir = std::env::temp_dir().join(format!("memopro-rt-chg-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("c.bin");
    std::fs::write(&path, vec![5u8; 2 * MIB]).unwrap();
    let rt = runtime(16, Policy::ReuseDistance);
    let id = rt.add_file(&path, 0, 2 * MIB, 1).unwrap();
    drop(rt.pin(id, false).unwrap());
    assert!(rt.evict(id).unwrap());
    std::thread::sleep(std::time::Duration::from_millis(20));
    std::fs::write(&path, vec![6u8; 2 * MIB]).unwrap();
    assert!(matches!(rt.pin(id, false), Err(Error::Integrity(_))));
    std::fs::remove_file(&path).ok();
}

/// Repeating scans over more data than fits: LRU re-reads everything each pass; the
/// reuse-distance policy keeps about one budget's worth (Belady's optimum for cycles).
#[test]
fn repeating_scans_keep_a_budget_worth_instead_of_thrashing() {
    let (path, _) = data_file();
    let rereads_in_last_pass = |policy| {
        let rt = runtime(12, policy);
        let ids: Vec<_> = (0..20)
            .map(|i| rt.add_file(path, (i * MIB) as u64, MIB, 1).unwrap())
            .collect();
        for _ in 0..3 {
            for &id in &ids {
                drop(rt.pin(id, false).unwrap());
            }
        }
        let before = rt.stats().rereads;
        for &id in &ids {
            drop(rt.pin(id, false).unwrap());
        }
        (rt.stats().rereads - before, rt.limit() as usize / MIB)
    };
    let (lru, _) = rereads_in_last_pass(Policy::Lru);
    let (reuse, fits) = rereads_in_last_pass(Policy::ReuseDistance);
    assert_eq!(lru, 20, "LRU thrashes on a cycle");
    assert!(
        reuse <= 20 - (fits - 2) as u64,
        "reuse-distance re-read {reuse} of 20 with room for {fits}"
    );
}

#[test]
fn threads_can_share_a_runtime() {
    let (path, data) = data_file();
    let rt = runtime(12, Policy::ReuseDistance);
    let ids: Vec<_> = (0..12)
        .map(|i| {
            (
                rt.add_file(path, (i * 2 * MIB) as u64, 2 * MIB, 1).unwrap(),
                i * 2 * MIB,
            )
        })
        .collect();
    std::thread::scope(|s| {
        for t in 0..3u64 {
            let rt = rt.clone();
            let ids = &ids;
            s.spawn(move || {
                let mut x = 17 + t;
                for _ in 0..60 {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    let (id, off) = ids[(x % ids.len() as u64) as usize];
                    let p = rt.pin(id, false).unwrap();
                    assert_eq!(p.as_slice(), &data[off..off + 2 * MIB]);
                }
            });
        }
    });
    let s = rt.stats();
    assert!(s.peak_used <= rt.limit());
    assert_eq!(s.pinned_bytes, 0);
}

// ---------------------------------------------------------------- phase 2 (0115)

#[test]
fn prefetching_brings_the_next_buffer_back_while_the_caller_works() {
    let (path, data) = data_file();
    let rt = prefetching(12, 2);
    let ids: Vec<_> = (0..20)
        .map(|i| rt.add_file(path, (i * MIB) as u64, MIB, 1).unwrap())
        .collect();
    for _ in 0..4 {
        for (i, &id) in ids.iter().enumerate() {
            let p = rt.pin(id, false).unwrap();
            assert_eq!(p.as_slice(), &data[i * MIB..(i + 1) * MIB]);
            std::thread::sleep(std::time::Duration::from_millis(3)); // "compute"
        }
    }
    let s = rt.stats();
    assert!(s.prefetches > 0, "{s:?}");
    assert!(s.prefetch_hits > 0, "{s:?}");
    assert!(s.peak_used <= rt.limit());
}

#[test]
fn prefetch_hints_and_dropping_the_runtime_stop_the_service_thread() {
    let (path, _) = data_file();
    let rt = prefetching(16, 4);
    let id = rt.add_file(path, 0, MIB, 1).unwrap();
    rt.prefetch(id).unwrap();
    for _ in 0..200 {
        if rt.state(id).unwrap() == BufferState::Resident {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert_eq!(rt.state(id).unwrap(), BufferState::Resident);
    assert!(rt.prefetch(999).is_err());
    let pin = rt.pin(id, false).unwrap();
    drop(rt); // joins the service thread; the pin stays valid
    assert_eq!(pin.len(), MIB);
}

/// Rereads per pass of a repeating scan over 60 blocks of 256 KiB (room for 31) with "compute"
/// between pins, so the service thread runs ahead of the caller as in E026 P1.
fn rereads_per_pass(prefetch: bool) -> (Vec<u64>, Stats) {
    let (path, _) = data_file();
    let mut c = Config::new(12 * MIB as u64);
    c.prefetch = prefetch;
    c.lookahead = MIB as u64; // four blocks
    let rt = Runtime::new(c).unwrap();
    let block = MIB / 4;
    let ids: Vec<_> = (0..60)
        .map(|i| rt.add_file(path, (i * block) as u64, block, 1).unwrap())
        .collect();
    let mut per_pass = Vec::new();
    for _ in 0..5 {
        let before = rt.stats().rereads;
        for &id in &ids {
            let p = rt.pin(id, false).unwrap();
            std::hint::black_box(p.as_slice()[0]);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        std::thread::sleep(std::time::Duration::from_millis(20)); // let the service thread settle
        per_pass.push(rt.stats().rereads - before);
    }
    (per_pass, rt.stats())
}

#[test]
fn prefetching_keeps_what_the_policy_keeps() {
    let (without, _) = rereads_per_pass(false);
    let (with, s) = rereads_per_pass(true);
    assert!(s.prefetch_hits > 0, "{s:?}");
    // Before 0117 F1 the prefetcher evicted the kept blocks one by one and every pass re-read
    // all 60; now the window (four blocks) is the only extra cost.
    for pass in 2..5 {
        assert!(
            with[pass] <= without[pass] + 5,
            "pass {pass}: {} blocks re-read with prefetching, {} without",
            with[pass],
            without[pass]
        );
    }
}

/// E026's pipeline in small: 184 blocks, room for 72 (the least the reserve allows), a
/// four-block window, and more compute than reading. Rereads per pass, with and without
/// prefetching.
fn pipeline_rereads(prefetch: bool) -> (Vec<u64>, Stats) {
    let (path, _) = data_file();
    let block = 64 * 1024;
    let probe = Runtime::new(Config::new(64 * MIB as u64)).unwrap();
    let reserve = probe.stats().reserve;
    drop(probe);
    let mut c = Config::new(reserve + 72 * block as u64);
    c.prefetch = prefetch;
    c.lookahead = 4 * block as u64;
    let rt = Runtime::new(c).unwrap();
    let ids: Vec<_> = (0..184)
        .map(|i| rt.add_file(path, (i * block) as u64, block, 1).unwrap())
        .collect();
    let mut per_pass = Vec::new();
    for _ in 0..4 {
        let before = rt.stats().rereads;
        for &id in &ids {
            let p = rt.pin(id, false).unwrap();
            std::hint::black_box(p.as_slice()[0]);
            std::thread::sleep(std::time::Duration::from_micros(400));
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
        per_pass.push(rt.stats().rereads - before);
    }
    (per_pass, rt.stats())
}

#[test]
fn prefetching_looks_ahead_only_its_window() {
    let (without, _) = pipeline_rereads(false);
    let (with, s) = pipeline_rereads(true);
    assert!(s.prefetch_hits > 0, "{s:?}");
    for pass in 1..4 {
        assert!(
            with[pass] <= without[pass] + 8,
            "pass {pass}: {} of 184 blocks re-read with prefetching, {} without ({s:?})",
            with[pass],
            without[pass]
        );
    }
}

#[test]
fn prefetching_and_pins_race_on_compressed_buffers_safely() {
    // while a pin makes room (compressing, outside the lock) the service thread may bring the
    // same buffer back first; the pin must notice instead of assuming it is still compressed
    let rt = prefetching(24, 8);
    let bufs: Vec<_> = (0..8)
        .map(|k| {
            let id = rt.alloc(4 * MIB, 4).unwrap();
            let data = widened(MIB, 7 + k);
            rt.pin(id, true)
                .unwrap()
                .as_mut_slice()
                .unwrap()
                .copy_from_slice(&data);
            (id, data)
        })
        .collect();
    for _ in 0..6 {
        for (id, data) in &bufs {
            assert_eq!(rt.pin(*id, false).unwrap().as_slice(), &data[..]);
        }
    }
    let s = rt.stats();
    assert!(s.compressions > 0 && s.peak_used <= rt.limit(), "{s:?}");
}

#[test]
fn prefetching_follows_a_forward_and_backward_order() {
    let (path, _) = data_file();
    let rt = prefetching(12, 2);
    let ids: Vec<_> = (0..14)
        .map(|i| rt.add_file(path, (i * MIB) as u64, MIB, 1).unwrap())
        .collect();
    let mut order: Vec<_> = ids.clone();
    order.extend(ids.iter().rev().copied());
    for step in 0..8 {
        if step == 3 {
            let s = rt.stats();
            assert!(s.prefetches > 0, "{s:?}");
        }
        for &id in &order {
            drop(rt.pin(id, false).unwrap());
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    }
    let s = rt.stats();
    // a single successor per buffer points backwards in the forward pass and forwards in the
    // backward pass; pairs tell the two apart
    assert!(
        s.prefetch_wasted * 4 <= s.prefetches,
        "wasted {} of {} prefetches",
        s.prefetch_wasted,
        s.prefetches
    );
}

fn widen(inputs: &[&[u8]], out: &mut [u8]) -> Result<()> {
    for (o, i) in out.chunks_exact_mut(4).zip(inputs[0].chunks_exact(2)) {
        o[..2].fill(0);
        o[2..].copy_from_slice(i);
    }
    Ok(())
}

#[test]
fn derived_buffers_are_recomputed_exactly_when_dropped() {
    let (path, data) = data_file();
    let rt = runtime(16, Policy::ReuseDistance);
    let src = rt.add_file(path, 0, 2 * MIB, 2).unwrap();
    let wide = rt.derive(&[src], 4 * MIB, 4, Arc::new(widen)).unwrap();
    let expect: Vec<u8> = data[..2 * MIB]
        .chunks_exact(2)
        .flat_map(|c| [0, 0, c[0], c[1]])
        .collect();
    assert_eq!(rt.pin(wide, false).unwrap().as_slice(), &expect[..]);
    assert!(rt.evict(wide).unwrap());
    assert_eq!(rt.state(wide).unwrap(), BufferState::Dropped);
    assert!(rt.evict(src).unwrap());
    assert_eq!(rt.pin(wide, false).unwrap().as_slice(), &expect[..]);
    let s = rt.stats();
    assert_eq!(s.recomputes, 1);
    assert_eq!(s.rereads, 1); // the input came back from its file first
}

#[test]
fn a_nondeterministic_recipe_fails_instead_of_returning_other_data() {
    use std::sync::atomic::{AtomicU8, Ordering};
    let rt = runtime(16, Policy::ReuseDistance);
    let base = rt.alloc(MIB, 1).unwrap();
    let calls = Arc::new(AtomicU8::new(0));
    let c = calls.clone();
    let noisy: Compute = Arc::new(move |_: &[&[u8]], out: &mut [u8]| {
        out.fill(c.fetch_add(1, Ordering::SeqCst));
        Ok(())
    });
    let d = rt.derive(&[base], MIB, 1, noisy).unwrap();
    assert!(rt.evict(d).unwrap());
    assert!(matches!(rt.pin(d, false), Err(Error::Integrity(_))));
}

#[test]
fn inputs_of_dropped_buffers_cannot_change_or_go() {
    let (path, _) = data_file();
    let rt = runtime(16, Policy::ReuseDistance);
    let src = rt.add_file(path, 0, 2 * MIB, 2).unwrap();
    let wide = rt.derive(&[src], 4 * MIB, 4, Arc::new(widen)).unwrap();
    assert!(rt.evict(wide).unwrap());
    assert!(rt.free(src).is_err());
    assert!(rt.pin(src, true).is_err());
    // once the derived buffer is back, the input may change: the derived one keeps its data
    // and no longer claims a recipe
    drop(rt.pin(wide, false).unwrap());
    drop(rt.pin(src, true).unwrap());
    assert!(rt.evict(wide).is_ok());
    assert_ne!(rt.state(wide).unwrap(), BufferState::Dropped);
    assert!(rt.free(src).is_ok());
}

#[test]
fn prediction_matches_the_rereads_of_a_repeating_scan() {
    let (path, _) = data_file();
    let rt = runtime(12, Policy::ReuseDistance);
    let ids: Vec<_> = (0..20)
        .map(|i| rt.add_file(path, (i * MIB) as u64, MIB, 1).unwrap())
        .collect();
    assert!(rt.predict().is_none());
    for _ in 0..3 {
        for &id in &ids {
            drop(rt.pin(id, false).unwrap());
        }
    }
    let p = rt.predict().expect("a full cycle was recorded");
    assert_eq!(p.cycle_pins, 20);
    assert_eq!(p.cycle_bytes, 20 * MIB as u64);
    let before = rt.stats().reread_bytes;
    for &id in &ids {
        drop(rt.pin(id, false).unwrap());
    }
    let actual = rt.stats().reread_bytes - before;
    let predicted = p.restore_bytes;
    assert!(
        actual.abs_diff(predicted) <= MIB as u64,
        "predicted {predicted}, re-read {actual}"
    );
}

#[test]
fn holding_back_shrinks_the_limit_and_gives_up_unpinned_buffers() {
    // 0165: memory the runtime does not own (activations) counts against the same budget
    let (path, data) = data_file();
    let rt = runtime(32, Policy::ReuseDistance);
    let full = rt.limit();
    let ids: Vec<_> = (0..4)
        .map(|i| rt.add_file(path, (i * 4 * MIB) as u64, 4 * MIB, 1).unwrap())
        .collect();
    for &id in &ids {
        drop(rt.pin(id, false).unwrap()); // all four in memory: 16 MiB
    }
    assert_eq!(rt.stats().used, 16 * MIB as u64);
    let keep = rt.pin(ids[0], false).unwrap();
    rt.hold_back(full - 8 * MIB as u64).unwrap(); // room for two buffers
    assert_eq!(rt.limit(), 8 * MIB as u64);
    assert!(rt.stats().used <= rt.limit());
    assert_eq!(keep.as_slice(), &data[..4 * MIB]); // the pinned one stayed
    // the pinned buffer cannot go: holding back more fails and changes nothing
    assert!(rt.hold_back(full - 2 * MIB as u64).is_err());
    assert_eq!(rt.limit(), 8 * MIB as u64);
    drop(keep);
    rt.hold_back(0).unwrap();
    assert_eq!(rt.limit(), full);
    // what was given up comes back from the file, unchanged
    for (i, &id) in ids.iter().enumerate() {
        assert_eq!(
            rt.pin(id, false).unwrap().as_slice(),
            &data[i * 4 * MIB..(i + 1) * 4 * MIB]
        );
    }
    // never more than the budget minus the runtime's own headroom
    assert!(rt.hold_back(32 * MIB as u64).is_err());
}

#[test]
#[ignore = "measures the whole process: run with --ignored --test-threads=1"]
#[cfg(any(target_os = "macos", target_os = "linux"))]
fn a_process_budget_counts_memory_outside_the_runtime() {
    let outside = vec![7u8; 16 * MIB]; // memory the runtime does not own, touched
    let data = widened(MIB, 3); // 4 MiB that compress well
    let now = process_footprint().unwrap();
    let mut c = Config::new(128 * MIB as u64);
    c.prefetch = false;
    c.process_budget = Some(now + 40 * MIB as u64);
    let rt = Runtime::new(c).unwrap();
    let ids: Vec<_> = (0..16)
        .map(|_| {
            let id = rt.alloc(4 * MIB, 4).unwrap();
            rt.pin(id, true)
                .unwrap()
                .as_mut_slice()
                .unwrap()
                .copy_from_slice(&data);
            id
        })
        .collect();
    for &id in &ids {
        assert_eq!(rt.pin(id, false).unwrap().as_slice(), &data[..]);
    }
    let s = rt.stats();
    // 64 MiB of buffers fit the 128 MiB budget alone; 40 MiB left in the process made them
    // compress (allocator memory the codec leaves behind counts as outside too: keep slack)
    assert!(s.compressions > 0, "{s:?}");
    assert!(s.outside_peak >= 16 * MIB as u64, "{s:?}"); // the 16 MiB vector
    assert!(s.limit_low < 40 * MIB as u64, "{s:?}");
    std::hint::black_box(&outside);
}

/// One repeating scan at a fixed ratio of data to budget, at a given absolute scale: 16 buffers
/// of `scale` MiB without a file, a budget for about 60% of them. Returns the decompressions of
/// the last pass and the stats.
fn scan_at_scale(scale: usize) -> (u64, Stats) {
    let reserve = round_to_pages(codec::chunk_bound(codec::CHUNK)) as u64;
    let total = 16 * scale * MIB;
    let mut c = Config::new(total as u64 * 6 / 10 + reserve);
    c.prefetch = false;
    let rt = Runtime::new(c).unwrap();
    let contents: Vec<_> = (0..16)
        .map(|k| widened(scale * MIB / 4, 11 + k as u32))
        .collect();
    let ids: Vec<_> = contents
        .iter()
        .map(|c| {
            let id = rt.alloc(c.len(), 4).unwrap();
            rt.pin(id, true)
                .unwrap()
                .as_mut_slice()
                .unwrap()
                .copy_from_slice(c);
            id
        })
        .collect();
    let mut last = 0;
    for _ in 0..3 {
        let before = rt.stats().decompressions;
        for (id, c) in ids.iter().zip(&contents) {
            assert_eq!(rt.pin(*id, false).unwrap().as_slice(), &c[..]);
        }
        last = rt.stats().decompressions - before;
    }
    let s = rt.stats();
    assert!(s.peak_used <= rt.limit(), "scale {scale}: {s:?}");
    assert_eq!(s.written_bytes, 0);
    (last, s)
}

#[test]
fn the_same_ratio_behaves_the_same_at_every_scale() {
    // scale.md S1: what the runtime does depends on the ratio of data to budget, not on its
    // absolute size; a constant that does not scale shows up as a difference here
    let (base, _) = scan_at_scale(1);
    assert!(base > 0, "nothing was restored at a 60% budget");
    for scale in [2, 4] {
        let (n, s) = scan_at_scale(scale);
        assert!(
            n.abs_diff(base) <= 1,
            "scale {scale}: {n} restores per pass, {base} at scale 1 ({s:?})"
        );
    }
}
