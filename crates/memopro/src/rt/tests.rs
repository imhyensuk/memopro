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

fn runtime(budget_mib: u64, policy: Policy) -> Runtime {
    let mut c = Config::new(budget_mib * MIB as u64);
    c.policy = policy;
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
