//! C ABI of the memopro runtime (`include/memopro.h`, docs/research 0124).
//!
//! Thin wrappers over [`memopro::rt`]: opaque `mp_runtime` and `mp_pin` handles, integer status
//! codes, a per-thread error message, and no panic ever crossing the boundary (a panic becomes
//! `MP_ERR_PANIC`). Pins are exclusive in the Rust sense: a writable pin is the only pin of its
//! buffer, so C code cannot race a writer against readers through this API.

#![allow(non_camel_case_types)]

use memopro::Error;
use memopro::rt::{BufferId, BufferState, Compute, Config, Pin, Policy, Runtime};
use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::sync::Arc;

pub const MP_ABI_VERSION: u32 = 1;
pub const MP_OK: c_int = 0;
pub const MP_NO_DATA: c_int = 1;
pub const MP_ERR_INVALID: c_int = -1;
pub const MP_ERR_BUDGET: c_int = -2;
pub const MP_ERR_INTEGRITY: c_int = -3;
pub const MP_ERR_IO: c_int = -4;
pub const MP_ERR_UNSUPPORTED: c_int = -5;
pub const MP_ERR_NOT_IMPLEMENTED: c_int = -6;
pub const MP_ERR_PANIC: c_int = -99;

/// Opaque runtime handle.
pub struct mp_runtime {
    rt: Runtime,
}

/// Opaque pin handle.
pub struct mp_pin {
    pin: Pin,
}

pub type mp_buffer = u64;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct mp_config {
    pub budget: u64,
    pub compress_level: i32,
    pub min_saving: f64,
    pub policy: i32,
    pub prefetch: i32,
    pub lookahead: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct mp_stats {
    pub size: usize,
    pub budget: u64,
    pub reserve: u64,
    pub used: u64,
    pub peak_used: u64,
    pub buffers: u64,
    pub resident_bytes: u64,
    pub compressed_bytes: u64,
    pub pinned_bytes: u64,
    pub pins: u64,
    pub loads: u64,
    pub load_bytes: u64,
    pub rereads: u64,
    pub reread_bytes: u64,
    pub drops: u64,
    pub drop_bytes: u64,
    pub compressions: u64,
    pub compress_in: u64,
    pub compress_out: u64,
    pub decompressions: u64,
    pub recomputes: u64,
    pub recompute_bytes: u64,
    pub prefetches: u64,
    pub prefetch_hits: u64,
    pub prefetch_wasted: u64,
    pub refusals: u64,
    pub written_bytes: u64,
    pub read_seconds: f64,
    pub compress_seconds: f64,
    pub decompress_seconds: f64,
    pub recompute_seconds: f64,
    pub restore_seconds: f64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct mp_prediction {
    pub size: usize,
    pub cycle_pins: u64,
    pub cycle_bytes: u64,
    pub restore_bytes: u64,
    pub restore_seconds: f64,
    pub compute_seconds: f64,
    pub seconds: f64,
    pub last_seconds: f64,
    pub prefetch: i32,
}

pub type mp_compute_fn = Option<
    unsafe extern "C" fn(
        user: *mut c_void,
        inputs: *const *const c_void,
        input_sizes: *const usize,
        n_inputs: usize,
        out: *mut c_void,
        out_size: usize,
    ) -> c_int,
>;

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn set_error(message: &str) {
    let text = CString::new(message.replace('\0', " ")).unwrap_or_default();
    LAST_ERROR.with(|e| *e.borrow_mut() = text);
}

fn code(e: &Error) -> c_int {
    match e {
        Error::InvalidArgument(_) => MP_ERR_INVALID,
        Error::Budget(_) => MP_ERR_BUDGET,
        Error::Integrity(_) => MP_ERR_INTEGRITY,
        Error::Io(_) => MP_ERR_IO,
        Error::Unsupported(_) => MP_ERR_UNSUPPORTED,
        Error::NotImplemented { .. } => MP_ERR_NOT_IMPLEMENTED,
    }
}

/// Run `f`, turning errors into codes (and the thread's message) and panics into
/// `MP_ERR_PANIC`: no Rust panic may unwind into C.
fn guard(f: impl FnOnce() -> Result<c_int, Error>) -> c_int {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(c)) => c,
        Ok(Err(e)) => {
            set_error(&e.to_string());
            code(&e)
        }
        Err(p) => {
            let what = p
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| p.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown".into());
            set_error(&format!("internal error in memopro (a bug): {what}"));
            MP_ERR_PANIC
        }
    }
}

fn invalid(what: &str) -> Error {
    Error::InvalidArgument(what.into())
}

/// # Safety
/// `rt` is null or a pointer from `mp_runtime_new` not yet freed.
unsafe fn runtime<'a>(rt: *const mp_runtime) -> Result<&'a Runtime, Error> {
    // SAFETY: the caller's contract.
    unsafe { rt.as_ref() }
        .map(|r| &r.rt)
        .ok_or_else(|| invalid("runtime is null"))
}

/// # Safety
/// `out` is null or valid for a write of `T`.
unsafe fn put<T>(out: *mut T, value: T) -> Result<(), Error> {
    if out.is_null() {
        return Err(invalid("output pointer is null"));
    }
    // SAFETY: checked non-null; the caller's contract for validity.
    unsafe { out.write(value) };
    Ok(())
}

#[unsafe(no_mangle)]
pub extern "C" fn mp_abi_version() -> u32 {
    MP_ABI_VERSION
}

#[unsafe(no_mangle)]
pub extern "C" fn mp_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast()
}

#[unsafe(no_mangle)]
pub extern "C" fn mp_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}

/// # Safety
/// `config` is null or valid for a write of `mp_config`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mp_config_default(config: *mut mp_config, budget: u64) {
    let c = Config::new(budget);
    let value = mp_config {
        budget,
        compress_level: c.compress_level,
        min_saving: c.min_saving,
        policy: 0,
        prefetch: 1,
        lookahead: c.lookahead,
    };
    // SAFETY: the caller's contract; a null pointer is ignored.
    let _ = unsafe { put(config, value) };
}

/// # Safety
/// `config` points to an `mp_config`; `out` is valid for a write of a pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mp_runtime_new(
    config: *const mp_config,
    out: *mut *mut mp_runtime,
) -> c_int {
    guard(|| {
        // SAFETY: the caller's contract.
        let c = unsafe { config.as_ref() }.ok_or_else(|| invalid("config is null"))?;
        if out.is_null() {
            return Err(invalid("output pointer is null"));
        }
        let mut cfg = Config::new(c.budget);
        cfg.compress_level = c.compress_level;
        cfg.min_saving = c.min_saving;
        cfg.policy = match c.policy {
            0 => Policy::ReuseDistance,
            1 => Policy::Lru,
            other => return Err(invalid(&format!("unknown policy {other}"))),
        };
        cfg.prefetch = c.prefetch != 0;
        cfg.lookahead = c.lookahead;
        let rt = Runtime::new(cfg)?;
        let handle = Box::into_raw(Box::new(mp_runtime { rt }));
        // SAFETY: checked non-null above.
        unsafe { put(out, handle) }?;
        Ok(MP_OK)
    })
}

/// # Safety
/// `rt` is null or a pointer from `mp_runtime_new`, freed only once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mp_runtime_free(rt: *mut mp_runtime) {
    if rt.is_null() {
        return;
    }
    let _ = guard(|| {
        // SAFETY: the caller's contract: this handle came from Box::into_raw and is freed once.
        drop(unsafe { Box::from_raw(rt) });
        Ok(MP_OK)
    });
}

/// # Safety
/// `rt` is null or a live runtime.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mp_limit(rt: *const mp_runtime) -> u64 {
    // SAFETY: the caller's contract.
    unsafe { runtime(rt) }.map(|r| r.limit()).unwrap_or(0)
}

/// # Safety
/// `rt` is a live runtime; `out` is valid for a write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mp_alloc(
    rt: *mut mp_runtime,
    nbytes: usize,
    elem: usize,
    out: *mut mp_buffer,
) -> c_int {
    guard(|| {
        // SAFETY: the caller's contract.
        let r = unsafe { runtime(rt) }?;
        let id = r.alloc(nbytes, elem)?;
        // SAFETY: the caller's contract.
        unsafe { put(out, id) }?;
        Ok(MP_OK)
    })
}

fn path_of(path: &CStr) -> Result<PathBuf, Error> {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        Ok(PathBuf::from(std::ffi::OsStr::from_bytes(path.to_bytes())))
    }
    #[cfg(not(unix))]
    {
        path.to_str()
            .map(PathBuf::from)
            .map_err(|_| invalid("path is not UTF-8"))
    }
}

/// # Safety
/// `rt` is a live runtime; `path` is a NUL-terminated string; `out` is valid for a write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mp_add_file(
    rt: *mut mp_runtime,
    path: *const c_char,
    offset: u64,
    nbytes: usize,
    elem: usize,
    out: *mut mp_buffer,
) -> c_int {
    guard(|| {
        // SAFETY: the caller's contract.
        let r = unsafe { runtime(rt) }?;
        if path.is_null() {
            return Err(invalid("path is null"));
        }
        // SAFETY: non-null, NUL-terminated by the caller's contract.
        let p = path_of(unsafe { CStr::from_ptr(path) })?;
        let id = r.add_file(&p, offset, nbytes, elem)?;
        // SAFETY: the caller's contract.
        unsafe { put(out, id) }?;
        Ok(MP_OK)
    })
}

/// # Safety
/// `rt` is a live runtime; `inputs` points to `n_inputs` buffer ids (or is null when
/// `n_inputs` is 0); `fn_` is a thread-safe deterministic function and `user` stays valid until
/// the derived buffer is freed; `out` is valid for a write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mp_derive(
    rt: *mut mp_runtime,
    inputs: *const mp_buffer,
    n_inputs: usize,
    nbytes: usize,
    elem: usize,
    fn_: mp_compute_fn,
    user: *mut c_void,
    out: *mut mp_buffer,
) -> c_int {
    guard(|| {
        // SAFETY: the caller's contract.
        let r = unsafe { runtime(rt) }?;
        let f = fn_.ok_or_else(|| invalid("compute function is null"))?;
        let ids: Vec<BufferId> = if n_inputs == 0 {
            Vec::new()
        } else if inputs.is_null() {
            return Err(invalid("inputs is null"));
        } else {
            // SAFETY: the caller's contract: `n_inputs` ids at `inputs`.
            unsafe { std::slice::from_raw_parts(inputs, n_inputs) }.to_vec()
        };
        let user = user as usize; // the caller promises the function is thread-safe
        let compute: Compute = Arc::new(move |ins: &[&[u8]], dst: &mut [u8]| {
            let ptrs: Vec<*const c_void> = ins.iter().map(|s| s.as_ptr().cast()).collect();
            let sizes: Vec<usize> = ins.iter().map(|s| s.len()).collect();
            // SAFETY: the pointers are valid for the duration of the call (the inputs are
            // pinned); the function's contract is the caller's.
            let rc = unsafe {
                f(
                    user as *mut c_void,
                    ptrs.as_ptr(),
                    sizes.as_ptr(),
                    ptrs.len(),
                    dst.as_mut_ptr().cast(),
                    dst.len(),
                )
            };
            if rc == 0 {
                Ok(())
            } else {
                Err(Error::InvalidArgument(format!(
                    "the compute function returned {rc}"
                )))
            }
        });
        let id = r.derive(&ids, nbytes, elem, compute)?;
        // SAFETY: the caller's contract.
        unsafe { put(out, id) }?;
        Ok(MP_OK)
    })
}

/// # Safety
/// `rt` is a live runtime; `out` is valid for a write of a pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mp_pin_acquire(
    rt: *mut mp_runtime,
    id: mp_buffer,
    write: c_int,
    out: *mut *mut mp_pin,
) -> c_int {
    guard(|| {
        // SAFETY: the caller's contract.
        let r = unsafe { runtime(rt) }?;
        if out.is_null() {
            return Err(invalid("output pointer is null"));
        }
        let pin = r.pin(id, write != 0)?;
        let handle = Box::into_raw(Box::new(mp_pin { pin }));
        // SAFETY: checked non-null.
        unsafe { put(out, handle) }?;
        Ok(MP_OK)
    })
}

/// # Safety
/// `pin` is null or a live pin.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mp_pin_data(pin: *const mp_pin) -> *mut c_void {
    // SAFETY: the caller's contract.
    match unsafe { pin.as_ref() } {
        Some(p) => p.pin.as_ptr() as *mut c_void,
        None => std::ptr::null_mut(),
    }
}

/// # Safety
/// `pin` is null or a live pin.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mp_pin_size(pin: *const mp_pin) -> usize {
    // SAFETY: the caller's contract.
    unsafe { pin.as_ref() }.map_or(0, |p| p.pin.len())
}

/// # Safety
/// `pin` is null or a pin from `mp_pin_acquire`, released once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mp_unpin(pin: *mut mp_pin) {
    if pin.is_null() {
        return;
    }
    let _ = guard(|| {
        // SAFETY: the caller's contract: from Box::into_raw, released once.
        drop(unsafe { Box::from_raw(pin) });
        Ok(MP_OK)
    });
}

/// # Safety
/// `rt` is a live runtime.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mp_prefetch(rt: *mut mp_runtime, id: mp_buffer) -> c_int {
    guard(|| {
        // SAFETY: the caller's contract.
        unsafe { runtime(rt) }?.prefetch(id)?;
        Ok(MP_OK)
    })
}

/// # Safety
/// `rt` is a live runtime; `evicted` is null or valid for a write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mp_evict(
    rt: *mut mp_runtime,
    id: mp_buffer,
    evicted: *mut c_int,
) -> c_int {
    guard(|| {
        // SAFETY: the caller's contract.
        let done = unsafe { runtime(rt) }?.evict(id)?;
        if !evicted.is_null() {
            // SAFETY: non-null, the caller's contract.
            unsafe { put(evicted, c_int::from(done)) }?;
        }
        Ok(MP_OK)
    })
}

/// # Safety
/// `rt` is a live runtime.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mp_free(rt: *mut mp_runtime, id: mp_buffer) -> c_int {
    guard(|| {
        // SAFETY: the caller's contract.
        unsafe { runtime(rt) }?.free(id)?;
        Ok(MP_OK)
    })
}

/// # Safety
/// `rt` is a live runtime; `state` is valid for a write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mp_state(
    rt: *const mp_runtime,
    id: mp_buffer,
    state: *mut c_int,
) -> c_int {
    guard(|| {
        // SAFETY: the caller's contract.
        let s = unsafe { runtime(rt) }?.state(id)?;
        let value = match s {
            BufferState::Resident => 0,
            BufferState::Compressed => 1,
            BufferState::Dropped => 2,
            BufferState::Unloaded => 3,
        };
        // SAFETY: the caller's contract.
        unsafe { put(state, value) }?;
        Ok(MP_OK)
    })
}

/// # Safety
/// `rt` is a live runtime; `out` is valid for a write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mp_nbytes(rt: *const mp_runtime, id: mp_buffer, out: *mut usize) -> c_int {
    guard(|| {
        // SAFETY: the caller's contract.
        let n = unsafe { runtime(rt) }?.nbytes(id)?;
        // SAFETY: the caller's contract.
        unsafe { put(out, n) }?;
        Ok(MP_OK)
    })
}

/// # Safety
/// `rt` is a live runtime; `out` points to an `mp_stats` whose `size` is set.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mp_stats_get(rt: *const mp_runtime, out: *mut mp_stats) -> c_int {
    guard(|| {
        // SAFETY: the caller's contract.
        let r = unsafe { runtime(rt) }?;
        // SAFETY: the caller's contract.
        let o = unsafe { out.as_mut() }.ok_or_else(|| invalid("output pointer is null"))?;
        if o.size < std::mem::size_of::<mp_stats>() {
            return Err(invalid(
                "mp_stats.size is smaller than this library's mp_stats",
            ));
        }
        let s = r.stats();
        *o = mp_stats {
            size: std::mem::size_of::<mp_stats>(),
            budget: s.budget,
            reserve: s.reserve,
            used: s.used,
            peak_used: s.peak_used,
            buffers: s.buffers,
            resident_bytes: s.resident_bytes,
            compressed_bytes: s.compressed_bytes,
            pinned_bytes: s.pinned_bytes,
            pins: s.pins,
            loads: s.loads,
            load_bytes: s.load_bytes,
            rereads: s.rereads,
            reread_bytes: s.reread_bytes,
            drops: s.drops,
            drop_bytes: s.drop_bytes,
            compressions: s.compressions,
            compress_in: s.compress_in,
            compress_out: s.compress_out,
            decompressions: s.decompressions,
            recomputes: s.recomputes,
            recompute_bytes: s.recompute_bytes,
            prefetches: s.prefetches,
            prefetch_hits: s.prefetch_hits,
            prefetch_wasted: s.prefetch_wasted,
            refusals: s.refusals,
            written_bytes: s.written_bytes,
            read_seconds: s.read_seconds,
            compress_seconds: s.compress_seconds,
            decompress_seconds: s.decompress_seconds,
            recompute_seconds: s.recompute_seconds,
            restore_seconds: s.restore_seconds,
        };
        Ok(MP_OK)
    })
}

/// # Safety
/// `rt` is a live runtime; `out` points to an `mp_prediction` whose `size` is set.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mp_predict(rt: *const mp_runtime, out: *mut mp_prediction) -> c_int {
    guard(|| {
        // SAFETY: the caller's contract.
        let r = unsafe { runtime(rt) }?;
        // SAFETY: the caller's contract.
        let o = unsafe { out.as_mut() }.ok_or_else(|| invalid("output pointer is null"))?;
        if o.size < std::mem::size_of::<mp_prediction>() {
            return Err(invalid(
                "mp_prediction.size is smaller than this library's mp_prediction",
            ));
        }
        let Some(p) = r.predict() else {
            return Ok(MP_NO_DATA);
        };
        *o = mp_prediction {
            size: std::mem::size_of::<mp_prediction>(),
            cycle_pins: p.cycle_pins,
            cycle_bytes: p.cycle_bytes,
            restore_bytes: p.restore_bytes,
            restore_seconds: p.restore_seconds,
            compute_seconds: p.compute_seconds,
            seconds: p.seconds,
            last_seconds: p.last_seconds,
            prefetch: i32::from(p.prefetch),
        };
        Ok(MP_OK)
    })
}

#[cfg(test)]
mod tests;
