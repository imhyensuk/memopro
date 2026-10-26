//! No-copy Metal buffers over runtime memory (macOS, G4 E1, 0130): the Apple GPU reads and
//! writes page-aligned runtime regions in place; the Python side hands each buffer to torch as an
//! MPS tensor through DLPack.
//!
//! No new crates: the Objective-C runtime for the Metal calls.

use crate::error::{Error, Result};

fn page_size() -> usize {
    // SAFETY: sysconf has no preconditions.
    let p = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if p > 0 { p as usize } else { 4096 }
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
