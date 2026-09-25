//! Error type shared by all modules.

use std::fmt;
use std::io;

/// Errors returned by the memopro core.
#[derive(Debug)]
pub enum Error {
    /// Part of the designed API that is not built yet; names the planned milestone.
    NotImplemented {
        feature: &'static str,
        planned: &'static str,
    },
    /// An argument is outside its valid range.
    InvalidArgument(String),
    /// An operating-system I/O error.
    Io(io::Error),
}

/// Result alias used across the crate.
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub(crate) fn not_implemented(feature: &'static str, planned: &'static str) -> Self {
        Error::NotImplemented { feature, planned }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NotImplemented { feature, planned } => {
                write!(f, "{feature} is not implemented yet (planned: {planned})")
            }
            Error::InvalidArgument(msg) => write!(f, "invalid argument: {msg}"),
            Error::Io(e) => write!(f, "I/O error: {e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Error::Io(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn not_implemented_names_the_plan() {
        let e = Error::not_implemented("hwinfo::memory", "v0.1 A1a");
        assert_eq!(
            e.to_string(),
            "hwinfo::memory is not implemented yet (planned: v0.1 A1a)"
        );
    }

    #[test]
    fn io_errors_keep_their_source() {
        let e: Error = io::Error::other("disk gone").into();
        assert!(std::error::Error::source(&e).is_some());
    }
}
