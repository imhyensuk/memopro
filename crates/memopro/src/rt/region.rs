//! Memory the runtime owns (0109 `rt::arena`, 0112).
//!
//! Every managed buffer, and every compressed piece of one, lives in its own page-aligned
//! anonymous mapping. Dropping a [`Region`] unmaps it, so the memory goes straight back to the
//! operating system instead of an allocator cache: macOS libmalloc kept freed memory until
//! pressure (0060), PyTorch's MPS allocator reserved whole heaps (0080) and CUDA's cache kept
//! pieces it never reused (0105). [`Region::release`] gives back part of a region while the rest
//! stays in use, which lets the runtime compress a buffer piece by piece without holding the
//! whole buffer twice.

use crate::error::{Error, Result};
use std::sync::OnceLock;

/// The operating system's page size.
pub fn page_size() -> usize {
    static SIZE: OnceLock<usize> = OnceLock::new();
    *SIZE.get_or_init(|| {
        #[cfg(unix)]
        {
            // SAFETY: sysconf with a constant name has no preconditions.
            let n = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
            if n > 0 { n as usize } else { 4096 }
        }
        #[cfg(not(unix))]
        {
            4096
        }
    })
}

/// `n` rounded up to whole pages.
pub fn round_to_pages(n: usize) -> usize {
    let p = page_size();
    n.div_ceil(p) * p
}

/// A page-aligned block of memory owned by the runtime.
#[derive(Debug)]
pub struct Region {
    ptr: *mut u8,
    len: usize,
    cap: usize,
}

// SAFETY: a region is plain memory with a single owner; access goes through `&`/`&mut`.
unsafe impl Send for Region {}
unsafe impl Sync for Region {}

impl Region {
    /// A new zero-filled region of `len` bytes (rounded up to whole pages).
    pub fn new(len: usize) -> Result<Region> {
        if len == 0 {
            return Err(Error::InvalidArgument(
                "a region holds at least one byte".into(),
            ));
        }
        let cap = round_to_pages(len);
        let ptr = alloc(cap)?;
        Ok(Region { ptr, len, cap })
    }

    /// A new region holding a copy of `data`.
    pub fn from_bytes(data: &[u8]) -> Result<Region> {
        let mut region = Region::new(data.len())?;
        region.as_mut_slice().copy_from_slice(data);
        Ok(region)
    }

    /// Bytes asked for.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Always false: regions hold at least one byte.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Bytes of memory the region occupies (whole pages).
    pub fn capacity(&self) -> usize {
        self.cap
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }

    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: ptr is valid for len bytes while self lives.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as above, and `&mut self` gives exclusive access.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// Give the whole pages inside `offset .. offset + len` back to the operating system; they
    /// read as zero afterwards (and take memory again only when written). A range that reaches
    /// the end of the data also releases the rest of the last page. Returns the bytes released.
    pub fn release(&mut self, offset: usize, len: usize) -> usize {
        let p = page_size();
        let end = if offset.saturating_add(len) >= self.len {
            self.cap
        } else {
            (offset + len) / p * p
        };
        let start = offset.div_ceil(p) * p;
        if start >= end {
            return 0;
        }
        if release_pages(self.ptr, start, end - start) {
            end - start
        } else {
            0
        }
    }
}

impl Drop for Region {
    fn drop(&mut self) {
        free(self.ptr, self.cap);
    }
}

#[cfg(unix)]
fn alloc(cap: usize) -> Result<*mut u8> {
    // SAFETY: anonymous private mapping; the kernel picks the address.
    let ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            cap,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANON,
            -1,
            0,
        )
    };
    if ptr == libc::MAP_FAILED {
        return Err(Error::Io(std::io::Error::last_os_error()));
    }
    Ok(ptr as *mut u8)
}

#[cfg(unix)]
fn free(ptr: *mut u8, cap: usize) {
    // SAFETY: ptr/cap came from `alloc`; nothing refers to the region any more.
    unsafe {
        libc::munmap(ptr as *mut libc::c_void, cap);
    }
}

/// Replace the pages with fresh zero pages: the old ones are freed at once on every Unix.
#[cfg(unix)]
fn release_pages(ptr: *mut u8, start: usize, len: usize) -> bool {
    // SAFETY: the range lies inside a mapping we own; MAP_FIXED replaces exactly that range.
    let got = unsafe {
        libc::mmap(
            ptr.add(start) as *mut libc::c_void,
            len,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_FIXED,
            -1,
            0,
        )
    };
    got != libc::MAP_FAILED
}

#[cfg(not(unix))]
fn layout(cap: usize) -> std::alloc::Layout {
    std::alloc::Layout::from_size_align(cap, page_size()).expect("page-aligned layout")
}

#[cfg(not(unix))]
fn alloc(cap: usize) -> Result<*mut u8> {
    // SAFETY: non-zero size, valid alignment.
    let ptr = unsafe { std::alloc::alloc_zeroed(layout(cap)) };
    if ptr.is_null() {
        return Err(Error::Io(std::io::Error::from(
            std::io::ErrorKind::OutOfMemory,
        )));
    }
    Ok(ptr)
}

#[cfg(not(unix))]
fn free(ptr: *mut u8, cap: usize) {
    // SAFETY: allocated by `alloc` with the same layout.
    unsafe { std::alloc::dealloc(ptr, layout(cap)) }
}

/// Without page mappings a part of a region cannot be given back; the caller keeps counting it.
#[cfg(not(unix))]
fn release_pages(_ptr: *mut u8, _start: usize, _len: usize) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regions_are_zeroed_page_aligned_and_whole_pages() {
        let r = Region::new(10).unwrap();
        assert_eq!(r.len(), 10);
        assert_eq!(r.capacity(), page_size());
        assert_eq!(r.as_ptr() as usize % page_size(), 0);
        assert!(r.as_slice().iter().all(|&b| b == 0));
        assert!(Region::new(0).is_err());
    }

    #[test]
    fn from_bytes_copies() {
        let data: Vec<u8> = (0..5000u32).map(|i| (i % 251) as u8).collect();
        let r = Region::from_bytes(&data).unwrap();
        assert_eq!(r.as_slice(), &data[..]);
    }

    #[test]
    fn release_gives_back_whole_pages_and_zeroes_them() {
        let p = page_size();
        let mut r = Region::new(4 * p).unwrap();
        r.as_mut_slice().fill(7);
        // an unaligned range releases only the whole pages inside it
        let got = r.release(p / 2, 2 * p);
        if cfg!(unix) {
            assert_eq!(got, p);
            assert!(r.as_slice()[p..2 * p].iter().all(|&b| b == 0));
            assert!(r.as_slice()[..p].iter().all(|&b| b == 7));
            assert!(r.as_slice()[2 * p..].iter().all(|&b| b == 7));
            // a range reaching the end releases the rest
            assert_eq!(r.release(3 * p, p), p);
        } else {
            assert_eq!(got, 0);
        }
    }
}
