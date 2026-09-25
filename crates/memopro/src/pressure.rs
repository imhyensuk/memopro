//! OS memory-pressure signal for γ elastic (v0.3): macOS memory pressure, Linux PSI.
//!
//! Status: skeleton. Redundancy is re-checked before v0.3 starts (0030 C5).

use crate::error::{Error, Result};

/// Coarse pressure level, comparable across operating systems.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    Normal,
    Warning,
    Critical,
}

/// Current memory-pressure level of the system.
pub fn current() -> Result<Level> {
    Err(Error::not_implemented("pressure::current", "v0.3 N3"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_are_ordered() {
        assert!(Level::Normal < Level::Warning && Level::Warning < Level::Critical);
        assert!(matches!(current(), Err(Error::NotImplemented { .. })));
    }
}
