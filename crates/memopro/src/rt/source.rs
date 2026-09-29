//! Original files the runtime re-reads buffers from (0109 `Source`, 0110).
//!
//! Reads bypass the page cache (macOS `F_NOCACHE`; on Linux the pages read are dropped from the
//! cache afterwards with `POSIX_FADV_DONTNEED`): the page cache is physical memory too, and the
//! runtime's budget has to cover all memory it causes (0110). Reads run in parallel pieces
//! straight into the destination and hash each piece while reading, like the spill engine's
//! write-free restore (0038, RS1, RS4, RS5). The file is never written.

use crate::error::{Error, Result};
use crate::spill::{Digest, read_region};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// An open original file, with the size and modification time it had when registered.
#[derive(Debug)]
pub struct SourceFile {
    path: PathBuf,
    file: File,
    len: u64,
    modified: Option<SystemTime>,
}

impl SourceFile {
    /// Open `path` read-only, without page caching where the OS allows it.
    pub fn open(path: &Path) -> Result<SourceFile> {
        let file = File::open(path)?;
        let meta = file.metadata()?;
        if !meta.is_file() {
            return Err(Error::InvalidArgument(format!(
                "{} is not a regular file",
                path.display()
            )));
        }
        bypass_cache(&file);
        Ok(SourceFile {
            path: path.to_path_buf(),
            file,
            len: meta.len(),
            modified: meta.modified().ok(),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// File size when registered.
    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Fail if the file's size or modification time changed since it was registered.
    pub fn check_unchanged(&self) -> Result<()> {
        let meta = std::fs::metadata(&self.path)?;
        if meta.len() != self.len || meta.modified().ok() != self.modified {
            return Err(Error::Integrity(format!(
                "{} changed since it was registered (size or modification time)",
                self.path.display()
            )));
        }
        Ok(())
    }

    /// Fill `dst` from `offset` and return the digest of what was read; with `expected`, fail
    /// with [`Error::Integrity`] if the data differs.
    pub fn read_into(
        &self,
        offset: u64,
        dst: &mut [u8],
        expected: Option<&Digest>,
    ) -> Result<Digest> {
        self.check_unchanged()?;
        let end = offset
            .checked_add(dst.len() as u64)
            .ok_or_else(|| Error::InvalidArgument("region end overflows".into()))?;
        if end > self.len {
            return Err(Error::InvalidArgument(format!(
                "region {offset}..{end} is outside {} ({} bytes)",
                self.path.display(),
                self.len
            )));
        }
        let digest = read_region(&self.file, offset, dst, expected)?;
        forget_cached(&self.file, offset, dst.len());
        Ok(digest)
    }
}

#[cfg(target_os = "macos")]
fn bypass_cache(file: &File) {
    use std::os::fd::AsRawFd;
    // SAFETY: fcntl on an open descriptor; failure only keeps the default caching.
    unsafe {
        libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1);
    }
}

#[cfg(not(target_os = "macos"))]
fn bypass_cache(_file: &File) {}

#[cfg(target_os = "linux")]
fn forget_cached(file: &File, offset: u64, len: usize) {
    use std::os::fd::AsRawFd;
    // SAFETY: advice on an open descriptor; failure only leaves the pages cached.
    unsafe {
        libc::posix_fadvise(
            file.as_raw_fd(),
            offset as libc::off_t,
            len as libc::off_t,
            libc::POSIX_FADV_DONTNEED,
        );
    }
}

#[cfg(not(target_os = "linux"))]
fn forget_cached(_file: &File, _offset: u64, _len: usize) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spill::digest;
    use std::io::Write;

    fn temp_file(name: &str, data: &[u8]) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("memopro-rt-src-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(name);
        let mut f = File::create(&path).unwrap();
        f.write_all(data).unwrap();
        path
    }

    #[test]
    fn reads_regions_and_checks_digests() {
        let data: Vec<u8> = (0..(9 << 20)).map(|i: u32| (i * 7 % 256) as u8).collect();
        let path = temp_file("a.bin", &data);
        let src = SourceFile::open(&path).unwrap();
        let mut dst = vec![0u8; 5 << 20];
        let d = src.read_into(1234, &mut dst, None).unwrap();
        assert_eq!(&dst[..], &data[1234..1234 + (5 << 20)]);
        assert_eq!(d, digest(&dst));
        assert!(src.read_into(1234, &mut dst, Some(&d)).is_ok());
        let wrong = digest(b"other");
        assert!(matches!(
            src.read_into(1234, &mut dst, Some(&wrong)),
            Err(Error::Integrity(_))
        ));
        let mut too_long = vec![0u8; 10 << 20];
        assert!(matches!(
            src.read_into(0, &mut too_long, None),
            Err(Error::InvalidArgument(_))
        ));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_changed_file_is_refused() {
        let path = temp_file("b.bin", &[1u8; 4096]);
        let src = SourceFile::open(&path).unwrap();
        std::fs::write(&path, [2u8; 5000]).unwrap();
        let mut dst = vec![0u8; 100];
        assert!(matches!(
            src.read_into(0, &mut dst, None),
            Err(Error::Integrity(_))
        ));
        std::fs::remove_file(&path).ok();
    }
}
