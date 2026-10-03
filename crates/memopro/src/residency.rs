//! RCR F class (0066, 0072): weights kept as clean file-backed pages.
//!
//! [`FileMap`] maps a whole file read-only. The pages are clean and file-backed: under memory
//! pressure the OS drops them without writing anything and reads them back on the next touch
//! (E015 G-F). On macOS, [`FileMap::metal_buffer`] wraps the mapping as an `MTLBuffer` without
//! copying (`newBufferWithBytesNoCopy`), which the Python side hands to torch as an MPS tensor
//! through DLPack. [`FileMap::prefetch`] warms the page cache with large sequential reads, which
//! reread far faster than page faults (E015 Q1: 0.74 vs 0.28 GB/s).
//!
//! No new crates: libc for mapping, the Objective-C runtime for the one Metal call.

use crate::error::{Error, Result};
use std::fs::File;
use std::os::fd::AsRawFd;
use std::path::Path;

/// A read-only mapping of a whole file, rounded up to whole pages.
#[derive(Debug)]
pub struct FileMap {
    ptr: *mut libc::c_void,
    len: usize,
    file_len: usize,
    file: File,
    #[cfg(target_os = "macos")]
    metal: std::sync::Mutex<usize>,
}

// SAFETY: the mapping is read-only and lives until drop; the raw pointer is only an address.
unsafe impl Send for FileMap {}
unsafe impl Sync for FileMap {}

fn page_size() -> usize {
    // SAFETY: sysconf has no preconditions.
    let p = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if p > 0 { p as usize } else { 4096 }
}

impl FileMap {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let file_len = file.metadata()?.len() as usize;
        if file_len == 0 {
            return Err(Error::InvalidArgument(format!(
                "{} is empty",
                path.display()
            )));
        }
        let page = page_size();
        let len = file_len.div_ceil(page) * page;
        // SAFETY: a fresh read-only shared mapping of an open file; checked against MAP_FAILED.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error().into());
        }
        Ok(FileMap {
            ptr,
            len,
            file_len,
            file,
            #[cfg(target_os = "macos")]
            metal: std::sync::Mutex::new(0),
        })
    }

    /// Address of the first byte.
    pub fn addr(&self) -> usize {
        self.ptr as usize
    }

    /// Mapped length (whole pages, at least the file length).
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn file_len(&self) -> usize {
        self.file_len
    }

    /// Share of the mapping's pages in RAM (`mincore`; never touches the pages).
    pub fn resident(&self) -> Result<f64> {
        let page = page_size();
        let pages = self.len / page;
        let mut vec = vec![0u8; pages];
        // SAFETY: vec holds one byte per page of our own mapping.
        let rc = unsafe { libc::mincore(self.ptr, self.len, vec.as_mut_ptr().cast()) };
        if rc != 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        Ok(vec.iter().filter(|b| **b & 1 != 0).count() as f64 / pages.max(1) as f64)
    }

    /// Read ``len`` bytes at ``offset`` with large sequential reads, so the page cache (which the
    /// mapping shares) holds them. Returns the bytes read.
    pub fn prefetch(&self, offset: usize, len: usize) -> Result<usize> {
        use std::os::unix::fs::FileExt;
        const CHUNK: usize = 8 << 20;
        let end = offset.saturating_add(len).min(self.file_len);
        let mut buf = vec![0u8; CHUNK.min(end.saturating_sub(offset)).max(1)];
        let mut at = offset;
        while at < end {
            let n = (end - at).min(buf.len());
            let got = self.file.read_at(&mut buf[..n], at as u64)?;
            if got == 0 {
                break;
            }
            at += got;
        }
        Ok(at - offset)
    }

    /// A retained `MTLBuffer` over the whole mapping, without copying (created once).
    #[cfg(target_os = "macos")]
    pub fn metal_buffer(&self) -> Result<usize> {
        let mut cached = self.metal.lock().expect("metal lock");
        if *cached == 0 {
            // SAFETY: page-aligned address and length of a live mapping; see metal::no_copy.
            *cached = unsafe { metal::no_copy(self.ptr, self.len)? };
        }
        Ok(*cached)
    }

    #[cfg(not(target_os = "macos"))]
    pub fn metal_buffer(&self) -> Result<usize> {
        Err(Error::Unsupported(
            "Metal buffers exist on macOS only".into(),
        ))
    }
}

impl Drop for FileMap {
    fn drop(&mut self) {
        #[cfg(target_os = "macos")]
        if let Ok(b) = self.metal.lock() {
            if *b != 0 {
                // SAFETY: we own one retain of this buffer.
                unsafe { metal::release(*b) };
            }
        }
        // SAFETY: unmapping exactly what open() mapped.
        unsafe { libc::munmap(self.ptr, self.len) };
    }
}

/// An `MTLBuffer` over `len` bytes at `ptr`, without copying (macOS; G4 E1: runtime buffers used
/// by the GPU in place). The caller owns one retain and gives it back with [`metal_release`].
///
/// # Safety
/// `ptr` must stay valid (mapped, not reused) for `len` bytes until the buffer is released and
/// no GPU work uses it any more.
pub unsafe fn metal_wrap(ptr: *mut u8, len: usize) -> Result<usize> {
    let page = page_size();
    if ptr.is_null() || (ptr as usize) % page != 0 || len == 0 || len % page != 0 {
        return Err(Error::InvalidArgument(format!(
            "a no-copy Metal buffer needs a page-aligned address and a length in whole pages \
             ({page} bytes); got {ptr:p}, {len}"
        )));
    }
    #[cfg(target_os = "macos")]
    {
        // SAFETY: checked alignment; validity is the caller's contract.
        unsafe { metal::no_copy(ptr.cast(), len) }
    }
    #[cfg(not(target_os = "macos"))]
    {
        Err(Error::Unsupported(
            "Metal buffers exist on macOS only".into(),
        ))
    }
}

/// Give back a buffer from [`metal_wrap`].
///
/// # Safety
/// `buffer` came from [`metal_wrap`] and is released once.
pub unsafe fn metal_release(buffer: usize) {
    #[cfg(target_os = "macos")]
    // SAFETY: the caller's contract: one retain we own.
    unsafe {
        metal::release(buffer)
    };
    #[cfg(not(target_os = "macos"))]
    let _ = buffer; // metal_wrap never succeeds here, so there is nothing to give back
}

#[cfg(target_os = "macos")]
mod metal {
    use super::{Error, Result};
    use std::ffi::{CStr, c_void};

    #[link(name = "Metal", kind = "framework")]
    unsafe extern "C" {
        fn MTLCreateSystemDefaultDevice() -> *mut c_void;
    }
    #[link(name = "objc")]
    unsafe extern "C" {
        fn sel_registerName(name: *const std::ffi::c_char) -> *mut c_void;
        fn objc_msgSend();
    }

    fn sel(name: &CStr) -> *mut c_void {
        // SAFETY: a NUL-terminated selector name.
        unsafe { sel_registerName(name.as_ptr()) }
    }

    /// `[device newBufferWithBytesNoCopy:ptr length:len options:Shared deallocator:nil]`.
    ///
    /// # Safety
    /// `ptr` must be page-aligned and valid for `len` bytes (a multiple of the page size) for
    /// the buffer's lifetime.
    pub unsafe fn no_copy(ptr: *mut c_void, len: usize) -> Result<usize> {
        // SAFETY: plain C function; may return null (no Metal device).
        let device = unsafe { MTLCreateSystemDefaultDevice() };
        if device.is_null() {
            return Err(Error::Unsupported("no Metal device".into()));
        }
        type NoCopy = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            *mut c_void,
            usize,
            usize,
            *mut c_void,
        ) -> *mut c_void;
        // SAFETY: objc_msgSend called with the exact prototype of this selector (arm64 ABI).
        let send: NoCopy = unsafe { std::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
        let options = 0usize; // MTLResourceStorageModeShared | CPU cache mode default
        let buffer = unsafe {
            send(
                device,
                sel(c"newBufferWithBytesNoCopy:length:options:deallocator:"),
                ptr,
                len,
                options,
                std::ptr::null_mut(),
            )
        };
        unsafe { release(device as usize) };
        if buffer.is_null() {
            return Err(Error::Unsupported(
                "newBufferWithBytesNoCopy returned nil (alignment or length)".into(),
            ));
        }
        Ok(buffer as usize)
    }

    /// `[object release]`.
    ///
    /// # Safety
    /// `object` must be a live Objective-C object we own a retain of.
    pub unsafe fn release(object: usize) {
        type Release = unsafe extern "C" fn(*mut c_void, *mut c_void);
        // SAFETY: -release takes no arguments and returns void.
        let send: Release = unsafe { std::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
        unsafe { send(object as *mut c_void, sel(c"release")) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn file(bytes: &[u8]) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!(
            "memopro-fmap-{}-{}",
            std::process::id(),
            bytes.len()
        ));
        let mut f = File::create(&p).unwrap();
        f.write_all(bytes).unwrap();
        p
    }

    #[test]
    fn maps_reads_and_prefetches_a_file() {
        let data: Vec<u8> = (0..300_000u32).map(|i| (i % 251) as u8).collect();
        let p = file(&data);
        let m = FileMap::open(&p).unwrap();
        assert_eq!(m.file_len(), data.len());
        assert!(m.len() >= data.len() && m.len() % page_size() == 0);
        // SAFETY: reading within the file length of a live read-only mapping.
        let view = unsafe { std::slice::from_raw_parts(m.addr() as *const u8, data.len()) };
        assert_eq!(view, &data[..]);
        assert!(m.resident().unwrap() > 0.0);
        assert_eq!(m.prefetch(1000, 5000).unwrap(), 5000);
        assert_eq!(m.prefetch(data.len() - 10, 1000).unwrap(), 10); // clipped at the end
        drop(m);
        std::fs::remove_file(p).unwrap();
    }

    #[test]
    fn empty_files_are_rejected() {
        let p = file(&[]);
        assert!(matches!(FileMap::open(&p), Err(Error::InvalidArgument(_))));
        std::fs::remove_file(p).unwrap();
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_buffer_is_created_once_or_unsupported() {
        let p = file(&vec![7u8; 100_000]);
        let m = FileMap::open(&p).unwrap();
        match m.metal_buffer() {
            Ok(b) => assert_eq!(m.metal_buffer().unwrap(), b),
            Err(Error::Unsupported(_)) => {} // e.g. a CI VM without a usable GPU
            Err(e) => panic!("{e}"),
        }
        drop(m);
        std::fs::remove_file(p).unwrap();
    }
}
