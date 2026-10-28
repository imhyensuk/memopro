//! E069 V6 (docs/research/0356): runtime calls made straight from Rust, the baseline for the C
//! ABI and Python. Prints one JSON line: median ns per operation and bulk copy GB/s.
//!
//!     cargo run --release -p memopro --example e069_ffi

use std::time::Instant;

use memopro::rt::{Config, Runtime};

const GIB: usize = 1 << 30;

fn per_op(reps: usize, rounds: usize, mut f: impl FnMut()) -> f64 {
    let mut ns: Vec<f64> = (0..rounds)
        .map(|_| {
            let t = Instant::now();
            for _ in 0..reps {
                f();
            }
            t.elapsed().as_nanos() as f64 / reps as f64
        })
        .collect();
    ns.sort_by(f64::total_cmp);
    ns[rounds / 2]
}

fn checksum(bytes: &[u8]) -> u64 {
    bytes.chunks_exact(8).fold(0u64, |s, w| {
        s.wrapping_add(u64::from_le_bytes(w.try_into().unwrap()))
    })
}

fn main() {
    let rt = Runtime::new(Config::new(3 * GIB as u64)).unwrap();
    let small = rt.alloc(1 << 20, 1).unwrap();
    let pin_ns = per_op(100_000, 11, || drop(rt.pin(small, false).unwrap()));
    let alloc_ns = per_op(10_000, 11, || {
        let id = rt.alloc(64 << 10, 1).unwrap();
        rt.free(id).unwrap();
    });
    let stats_ns = per_op(100_000, 11, || {
        std::hint::black_box(rt.stats());
    });

    let src: Vec<u8> = (0..GIB)
        .map(|i| (i.wrapping_mul(2654435761) >> 11) as u8)
        .collect();
    let big = rt.alloc(GIB, 1).unwrap();
    // written, so its pages exist as the pinned buffer's do; the two copies alternate
    let mut plain = vec![1u8; GIB];
    let (mut gbps, mut plain_gbps, mut sum) = (Vec::new(), Vec::new(), 0);
    for _ in 0..9 {
        let mut p = rt.pin(big, true).unwrap();
        let dst = p.as_mut_slice().unwrap();
        let t = Instant::now();
        dst.copy_from_slice(&src);
        gbps.push(GIB as f64 / t.elapsed().as_secs_f64() / 1e9);
        sum = checksum(dst);
        drop(p);
        let t = Instant::now();
        std::hint::black_box(&mut plain).copy_from_slice(std::hint::black_box(&src));
        std::hint::black_box(&plain);
        plain_gbps.push(GIB as f64 / t.elapsed().as_secs_f64() / 1e9);
    }
    gbps.sort_by(f64::total_cmp);
    plain_gbps.sort_by(f64::total_cmp);
    println!(
        "{{\"lang\": \"rust\", \"pin_unpin_ns\": {pin_ns:.1}, \"alloc_free_64k_ns\": {alloc_ns:.1}, \
         \"stats_ns\": {stats_ns:.1}, \"copy_gbps\": {:.2}, \"plain_memcpy_gbps\": {:.2}, \
         \"checksum\": {sum}}}",
        gbps[4], plain_gbps[4]
    );
}
