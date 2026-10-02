//! Slowdown prediction (0109 R6, 0115): replay the last recorded cycle of pins against the budget
//! with Belady's rule (the reuse-distance policy approximates it on repeating patterns) and turn
//! the bytes that would have to come back into time with the measured throughputs.

use std::collections::HashMap;

/// Predicted cost of repeating the last recorded cycle of buffer uses.
#[derive(Debug, Clone, PartialEq)]
pub struct Prediction {
    /// Pins in one cycle (from one use of a buffer to its next use).
    pub cycle_pins: u64,
    /// Distinct bytes the cycle touches.
    pub cycle_bytes: u64,
    /// Bytes expected to come back per cycle in steady state.
    pub restore_bytes: u64,
    /// Time to bring them back at the measured throughputs.
    pub restore_seconds: f64,
    /// Time the last cycle spent outside the runtime (the caller's own work).
    pub compute_seconds: f64,
    /// Predicted seconds per cycle: restore time hidden behind compute when prefetching,
    /// added to it when not.
    pub seconds: f64,
    /// Measured seconds of the last recorded cycle.
    pub last_seconds: f64,
    pub prefetch: bool,
}

/// One buffer in a recorded cycle: its accounted size and the seconds it takes to come back.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Item {
    pub id: u64,
    pub bytes: u64,
    pub restore_seconds: f64,
}

/// Bytes and seconds that must come back per repetition of `cycle` with `capacity` bytes kept,
/// by Belady's rule (evict what is used farthest in the future), in steady state.
pub(crate) fn steady_misses(cycle: &[Item], capacity: u64) -> (u64, f64) {
    const REPEATS: usize = 3;
    let n = cycle.len();
    if n == 0 {
        return (0, 0.0);
    }
    let seq: Vec<Item> = (0..REPEATS * n).map(|i| cycle[i % n]).collect();
    // next occurrence of the same buffer after position i
    let mut next = vec![usize::MAX; seq.len()];
    let mut seen: HashMap<u64, usize> = HashMap::new();
    for i in (0..seq.len()).rev() {
        if let Some(&j) = seen.get(&seq[i].id) {
            next[i] = j;
        }
        seen.insert(seq[i].id, i);
    }
    // resident buffers: id -> position of their next use
    let mut cache: HashMap<u64, (u64, usize)> = HashMap::new();
    let mut used = 0u64;
    let (mut bytes, mut seconds) = (0u64, 0.0f64);
    for (i, item) in seq.iter().enumerate() {
        let last_round = i >= (REPEATS - 1) * n;
        if let Some(entry) = cache.get_mut(&item.id) {
            entry.1 = next[i];
            continue;
        }
        if last_round {
            bytes += item.bytes;
            seconds += item.restore_seconds;
        }
        cache.insert(item.id, (item.bytes, next[i]));
        used += item.bytes;
        while used > capacity {
            let victim = cache
                .iter()
                .filter(|(id, _)| **id != item.id)
                .max_by_key(|(_, (_, nx))| *nx)
                .map(|(id, (b, _))| (*id, *b));
            match victim {
                Some((id, b)) => {
                    cache.remove(&id);
                    used -= b;
                }
                None => break,
            }
        }
    }
    (bytes, seconds)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cycle(n: u64, bytes: u64) -> Vec<Item> {
        (0..n)
            .map(|id| Item {
                id,
                bytes,
                restore_seconds: 1.0,
            })
            .collect()
    }

    #[test]
    fn a_cycle_that_fits_needs_nothing_back() {
        assert_eq!(steady_misses(&cycle(5, 10), 50), (0, 0.0));
    }

    #[test]
    fn a_cycle_larger_than_capacity_misses_about_n_minus_capacity() {
        // 20 buffers, room for 8: Belady misses (20 - 8) * 20 / 19 = 12.6 per cycle on average
        // (it rotates which buffers it keeps), so 12 or 13 in any one cycle
        let (bytes, seconds) = steady_misses(&cycle(20, 1), 8);
        assert!((12..=13).contains(&bytes), "{bytes}");
        assert_eq!(seconds, bytes as f64);
    }

    #[test]
    fn repeated_use_inside_a_cycle_counts_once_while_resident() {
        let mut c = cycle(4, 1);
        c.push(c[0]); // buffer 0 used twice per cycle (like tied embeddings)
        let (bytes, _) = steady_misses(&c, 100);
        assert_eq!(bytes, 0);
    }
}
