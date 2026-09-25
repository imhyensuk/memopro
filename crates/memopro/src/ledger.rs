//! Ledger of hibernated buffers: where each one lives now and how many bytes it holds.
//!
//! The ledger is the single source of truth for `status()` and the report: reclaimed bytes per
//! method and SSD bytes written (0032 H5). It is plain bookkeeping and holds no tensor data.

use crate::spill::{Digest, SourceRef, SpillFile};
use std::collections::BTreeMap;

/// Where a hibernated buffer lives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Location {
    /// Dropped from memory; restore re-reads the original file (no SSD write).
    Source(SourceRef),
    /// Moved from the accelerator to host RAM.
    Host,
    /// Compressed in RAM; `stored_bytes` is the compressed size.
    Compressed { stored_bytes: u64 },
    /// Written to a spill file.
    Spilled(SpillFile),
}

impl Location {
    /// Short method name, matching the Python modes.
    pub fn mode(&self) -> &'static str {
        match self {
            Location::Source(_) => "source",
            Location::Host => "host",
            Location::Compressed { .. } => "compress",
            Location::Spilled(_) => "spill",
        }
    }
}

/// One hibernated buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    pub nbytes: u64,
    pub location: Location,
    pub digest: Option<Digest>,
}

/// Registry of hibernated buffers, keyed by an id the caller receives on insert.
#[derive(Debug, Default)]
pub struct Ledger {
    entries: BTreeMap<u64, Entry>,
    next_id: u64,
    ssd_bytes_written: u64,
}

impl Ledger {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a hibernated buffer and return its id. Spill files count as SSD writes.
    pub fn insert(&mut self, entry: Entry) -> u64 {
        if let Location::Spilled(file) = &entry.location {
            self.ssd_bytes_written += file.len;
        }
        let id = self.next_id;
        self.next_id += 1;
        self.entries.insert(id, entry);
        id
    }

    pub fn get(&self, id: u64) -> Option<&Entry> {
        self.entries.get(&id)
    }

    /// Forget a buffer after it has been restored.
    pub fn remove(&mut self, id: u64) -> Option<Entry> {
        self.entries.remove(&id)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Bytes held by hibernated buffers per method name.
    pub fn bytes_by_mode(&self) -> BTreeMap<&'static str, u64> {
        let mut out = BTreeMap::new();
        for e in self.entries.values() {
            *out.entry(e.location.mode()).or_insert(0) += e.nbytes;
        }
        out
    }

    /// Total bytes written to SSD since the ledger was created (restores do not reduce it).
    pub fn ssd_bytes_written(&self) -> u64 {
        self.ssd_bytes_written
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn spilled(len: u64) -> Entry {
        Entry {
            nbytes: len,
            location: Location::Spilled(SpillFile {
                path: PathBuf::from("/tmp/x"),
                len,
                digest: Digest(vec![0]),
            }),
            digest: None,
        }
    }

    #[test]
    fn tracks_bytes_per_mode_and_ssd_writes() {
        let mut ledger = Ledger::new();
        let a = ledger.insert(Entry {
            nbytes: 100,
            location: Location::Host,
            digest: None,
        });
        let b = ledger.insert(spilled(40));
        assert_ne!(a, b);
        assert_eq!(ledger.len(), 2);
        let by_mode = ledger.bytes_by_mode();
        assert_eq!(by_mode["host"], 100);
        assert_eq!(by_mode["spill"], 40);
        assert_eq!(ledger.ssd_bytes_written(), 40);

        ledger.remove(b).unwrap();
        assert_eq!(
            ledger.ssd_bytes_written(),
            40,
            "restores do not undo SSD wear"
        );
        assert!(!ledger.bytes_by_mode().contains_key("spill"));
    }
}
