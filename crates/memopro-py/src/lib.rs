//! Python bindings for the memopro core, exposed as `memopro._core`.
//!
//! Bulk data is passed through the buffer protocol (RS2): the bindings borrow the caller's memory
//! (e.g. `tensor.view(torch.uint8).numpy()`) and release the GIL while working on it. The caller
//! must keep the buffer alive and unchanged for the duration of the call, which the Python layer
//! does by holding the object.

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{
    PyBufferError, PyNotImplementedError, PyOSError, PyRuntimeError, PyValueError,
};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyType};
use std::os::raw::{c_char, c_int, c_void};
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

fn to_py_err(e: memopro::Error) -> PyErr {
    match e {
        memopro::Error::NotImplemented { .. } => PyNotImplementedError::new_err(e.to_string()),
        memopro::Error::InvalidArgument(_) => PyValueError::new_err(e.to_string()),
        memopro::Error::Integrity(_) => PyRuntimeError::new_err(e.to_string()),
        memopro::Error::Unsupported(_) => PyNotImplementedError::new_err(e.to_string()),
        memopro::Error::Budget(_) => memopro_error("BudgetExceeded", e.to_string()),
        memopro::Error::Io(io) => PyOSError::new_err(io.to_string()),
    }
}

/// An exception of the Python package's own hierarchy (`memopro._errors`), or RuntimeError if
/// that cannot be imported.
fn memopro_error(name: &str, msg: String) -> PyErr {
    Python::attach(|py| {
        match py
            .import("memopro._errors")
            .and_then(|m| m.getattr(name))
            .map(|c| c.cast_into::<PyType>())
        {
            Ok(Ok(cls)) => PyErr::from_type(cls, msg),
            _ => PyRuntimeError::new_err(msg),
        }
    })
}

/// Errors of the runtime (`memopro.rt`) as memopro exceptions.
fn rt_err(e: memopro::Error) -> PyErr {
    match e {
        memopro::Error::Budget(_) => memopro_error("BudgetExceeded", e.to_string()),
        memopro::Error::Integrity(_) => memopro_error("IntegrityError", e.to_string()),
        memopro::Error::InvalidArgument(_) => memopro_error("InvalidArgument", e.to_string()),
        other => to_py_err(other),
    }
}

/// Address and length of a borrowed buffer, sendable into `py.detach`.
#[derive(Clone, Copy)]
struct Raw {
    ptr: usize,
    len: usize,
}

fn borrow(obj: &Bound<'_, PyAny>, writable: bool) -> PyResult<(PyBuffer<u8>, Raw)> {
    let buf = PyBuffer::<u8>::get(obj)?;
    if !buf.is_c_contiguous() {
        return Err(PyValueError::new_err("buffer must be C-contiguous"));
    }
    if writable && buf.readonly() {
        return Err(PyValueError::new_err("destination buffer is read-only"));
    }
    let raw = Raw {
        ptr: buf.buf_ptr() as usize,
        len: buf.len_bytes(),
    };
    Ok((buf, raw))
}

impl Raw {
    /// SAFETY: the `PyBuffer` this came from must outlive the returned slice.
    unsafe fn slice<'a>(self) -> &'a [u8] {
        unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.len) }
    }

    /// SAFETY: as `slice`, and nobody else may access the memory meanwhile.
    unsafe fn slice_mut<'a>(self) -> &'a mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut u8, self.len) }
    }
}

/// Returns the version of the Rust core linked into this extension module.
#[pyfunction]
fn core_version() -> &'static str {
    memopro::version()
}

// ------------------------------------------------------------------ hwinfo

/// Host memory snapshot (conservative `available_bytes`, see `memopro::hwinfo`).
#[pyfunction]
fn hwinfo_memory(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let m = py.detach(memopro::hwinfo::memory).map_err(to_py_err)?;
    let d = PyDict::new(py);
    d.set_item("total_bytes", m.total_bytes)?;
    d.set_item("available_bytes", m.available_bytes)?;
    d.set_item("kernel_available_bytes", m.kernel_available_bytes)?;
    d.set_item("swap_total_bytes", m.swap_total_bytes)?;
    d.set_item("swap_free_bytes", m.swap_free_bytes)?;
    d.set_item("cgroup_limit_bytes", m.cgroup.map(|c| c.limit_bytes))?;
    d.set_item("cgroup_free_bytes", m.cgroup.map(|c| c.free_bytes))?;
    d.set_item("usable_bytes", m.usable_bytes())?;
    Ok(d)
}

/// Resident set size of this process in bytes.
/// Current OS memory pressure: ``{"level", "source", "some_avg10", "full_avg10", "raw_level"}``.
#[pyfunction]
fn pressure_current(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let p = py.detach(memopro::pressure::current).map_err(to_py_err)?;
    let d = PyDict::new(py);
    d.set_item("level", p.level.as_str())?;
    d.set_item("source", p.source)?;
    d.set_item("some_avg10", p.some_avg10)?;
    d.set_item("full_avg10", p.full_avg10)?;
    d.set_item("raw_level", p.raw_level)?;
    Ok(d)
}

#[pyfunction]
fn hwinfo_process_rss(py: Python<'_>) -> PyResult<u64> {
    py.detach(memopro::hwinfo::process_rss).map_err(to_py_err)
}

/// CPU brand and logical CPU count.
#[pyfunction]
fn hwinfo_cpu(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let c = py.detach(memopro::hwinfo::cpu);
    let d = PyDict::new(py);
    d.set_item("brand", c.brand)?;
    d.set_item("logical_cpus", c.logical_cpus)?;
    Ok(d)
}

/// Capacity of the file system holding `path` (must exist).
#[pyfunction]
fn hwinfo_disk(py: Python<'_>, path: PathBuf) -> PyResult<Bound<'_, PyDict>> {
    let disk = py
        .detach(|| memopro::hwinfo::disk(&path))
        .map_err(to_py_err)?;
    let d = PyDict::new(py);
    d.set_item("path", disk.path)?;
    d.set_item("total_bytes", disk.total_bytes)?;
    d.set_item("available_bytes", disk.available_bytes)?;
    Ok(d)
}

// ------------------------------------------------------------------ spill engine

/// Digest (16 bytes) of a buffer.
#[pyfunction]
fn engine_digest<'py>(py: Python<'py>, data: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBytes>> {
    let (_keep, raw) = borrow(data, false)?;
    // SAFETY: `_keep` holds the buffer for the whole call.
    let d = py.detach(|| memopro::spill::digest(unsafe { raw.slice() }));
    Ok(PyBytes::new(py, &d.0))
}

/// Per-chunk hashes of a buffer (16 bytes per `DIGEST_CHUNK`, concatenated); see
/// `engine_digest_combine`. Lets Python verify large tensors piece by piece (0061 F1).
#[pyfunction]
fn engine_digest_parts<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyBytes>> {
    let (_keep, raw) = borrow(data, false)?;
    // SAFETY: `_keep` holds the buffer for the whole call.
    let parts = py.detach(|| memopro::spill::digest_parts(unsafe { raw.slice() }));
    let out: Vec<u8> = parts.iter().flat_map(|p| p.to_le_bytes()).collect();
    Ok(PyBytes::new(py, &out))
}

/// Digest (16 bytes) of `length` bytes whose chunk hashes are `parts` (from `engine_digest_parts`
/// over consecutive pieces whose lengths are multiples of `DIGEST_CHUNK`).
#[pyfunction]
fn engine_digest_combine<'py>(
    py: Python<'py>,
    length: usize,
    parts: &[u8],
) -> PyResult<Bound<'py, PyBytes>> {
    if parts.len() % 16 != 0 {
        return Err(PyValueError::new_err("parts must be 16 bytes each"));
    }
    let hashes: Vec<u128> = parts
        .chunks_exact(16)
        .map(|c| u128::from_le_bytes(c.try_into().expect("16 bytes")))
        .collect();
    let d = memopro::spill::digest_combine(length, &hashes);
    Ok(PyBytes::new(py, &d.0))
}

/// Fill `dst` from `len(dst)` bytes of `path` at `offset`; verify against `expected` if given.
/// Returns the digest of what was read.
#[pyfunction]
#[pyo3(signature = (path, offset, dst, expected = None))]
fn engine_read_source_into<'py>(
    py: Python<'py>,
    path: PathBuf,
    offset: u64,
    dst: &Bound<'py, PyAny>,
    expected: Option<Vec<u8>>,
) -> PyResult<Bound<'py, PyBytes>> {
    let (_keep, raw) = borrow(dst, true)?;
    let source = memopro::spill::SourceRef {
        path,
        offset,
        len: raw.len as u64,
    };
    let expected = expected.map(memopro::spill::Digest);
    let d = py
        .detach(|| {
            // SAFETY: `_keep` holds the writable buffer for the whole call.
            memopro::spill::Engine::new().read_source_into(
                &source,
                unsafe { raw.slice_mut() },
                expected.as_ref(),
            )
        })
        .map_err(to_py_err)?;
    Ok(PyBytes::new(py, &d.0))
}

// ------------------------------------------------------------------ RCR F class (0072)

/// A read-only mapping of a whole file (clean, file-backed pages; 0072). On macOS
/// `metal_buffer()` wraps it as an MTLBuffer without copying.
#[cfg(unix)]
#[pyclass(frozen)]
struct FileMap {
    inner: memopro::residency::FileMap,
}

#[cfg(unix)]
#[pymethods]
impl FileMap {
    #[new]
    fn new(path: PathBuf) -> PyResult<Self> {
        let inner = memopro::residency::FileMap::open(&path).map_err(to_py_err)?;
        Ok(FileMap { inner })
    }

    /// Address of the first mapped byte.
    #[getter]
    fn addr(&self) -> usize {
        self.inner.addr()
    }

    /// Mapped length in bytes (whole pages).
    #[getter]
    fn length(&self) -> usize {
        self.inner.len()
    }

    /// File length in bytes.
    #[getter]
    fn file_length(&self) -> usize {
        self.inner.file_len()
    }

    /// Share of the mapping's pages in RAM.
    fn resident(&self) -> PyResult<f64> {
        self.inner.resident().map_err(to_py_err)
    }

    /// Warm the page cache for `length` bytes at `offset` with large reads (GIL released).
    fn prefetch(&self, py: Python<'_>, offset: usize, length: usize) -> PyResult<usize> {
        py.detach(|| self.inner.prefetch(offset, length))
            .map_err(to_py_err)
    }

    /// The retained MTLBuffer over the whole mapping (macOS; created once, released on drop).
    fn metal_buffer(&self) -> PyResult<usize> {
        self.inner.metal_buffer().map_err(to_py_err)
    }
}

/// Write a buffer to a new `0600` spill file in `directory`; returns `(path, length, digest)`.
#[pyfunction]
fn engine_write<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    directory: PathBuf,
) -> PyResult<(PathBuf, u64, Bound<'py, PyBytes>)> {
    let (_keep, raw) = borrow(data, false)?;
    let file = py
        .detach(|| {
            // SAFETY: `_keep` holds the buffer for the whole call.
            memopro::spill::Engine::new().write(unsafe { raw.slice() }, &directory)
        })
        .map_err(to_py_err)?;
    Ok((file.path, file.len, PyBytes::new(py, &file.digest.0)))
}

// ------------------------------------------------------------------ in-RAM codec

/// Losslessly compressed copy of a buffer (byte shuffle + zstd), owned by Rust.
#[pyclass(module = "memopro._core", frozen)]
struct Compressed {
    chunks: Vec<Vec<u8>>,
    raw_len: usize,
    itemsize: usize,
}

#[pymethods]
impl Compressed {
    /// Size of the original data.
    #[getter]
    fn raw_bytes(&self) -> usize {
        self.raw_len
    }

    /// Size held in memory by the compressed chunks.
    #[getter]
    fn stored_bytes(&self) -> usize {
        self.chunks.iter().map(Vec::len).sum()
    }

    /// Decompress into `dst` (a writable buffer of exactly `raw_bytes`).
    fn unpack_into(&self, py: Python<'_>, dst: &Bound<'_, PyAny>) -> PyResult<()> {
        let (_keep, raw) = borrow(dst, true)?;
        if raw.len != self.raw_len {
            return Err(PyValueError::new_err(format!(
                "destination holds {} bytes, compressed data expands to {}",
                raw.len, self.raw_len
            )));
        }
        py.detach(|| {
            // SAFETY: `_keep` holds the writable buffer for the whole call.
            memopro::codec::unpack_into(&self.chunks, self.itemsize, unsafe { raw.slice_mut() })
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

/// Compress a buffer of `itemsize`-byte elements; returns a `Compressed` object.
#[pyfunction]
#[pyo3(signature = (data, itemsize, level = 1))]
fn codec_pack(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    itemsize: usize,
    level: i32,
) -> PyResult<Compressed> {
    let (_keep, raw) = borrow(data, false)?;
    let chunks = py
        .detach(|| {
            // SAFETY: `_keep` holds the buffer for the whole call.
            memopro::codec::pack(unsafe { raw.slice() }, itemsize, level)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(Compressed {
        chunks,
        raw_len: raw.len,
        itemsize,
    })
}

// ---------------------------------------------------------------- runtime C-R (0112)

/// The runtime (`memopro.rt.Runtime` wraps it).
#[pyclass(module = "memopro._core", name = "RtRuntime", frozen)]
struct RtRuntime {
    rt: memopro::rt::Runtime,
}

fn state_name(s: memopro::rt::BufferState) -> &'static str {
    use memopro::rt::BufferState as S;
    match s {
        S::Resident => "resident",
        S::Compressed => "compressed",
        S::Dropped => "dropped",
        S::Unloaded => "unloaded",
    }
}

#[pymethods]
impl RtRuntime {
    #[new]
    #[pyo3(signature = (
        budget, compress_level=1, min_saving=0.15, policy="reuse", prefetch=true,
        lookahead=64 << 20
    ))]
    fn new(
        budget: u64,
        compress_level: i32,
        min_saving: f64,
        policy: &str,
        prefetch: bool,
        lookahead: u64,
    ) -> PyResult<Self> {
        let mut config = memopro::rt::Config::new(budget);
        config.compress_level = compress_level;
        config.min_saving = min_saving;
        config.prefetch = prefetch;
        config.lookahead = lookahead;
        config.policy = match policy {
            "reuse" => memopro::rt::Policy::ReuseDistance,
            "lru" => memopro::rt::Policy::Lru,
            other => {
                return Err(memopro_error(
                    "InvalidArgument",
                    format!("policy must be 'reuse' or 'lru', not {other:?}"),
                ));
            }
        };
        Ok(RtRuntime {
            rt: memopro::rt::Runtime::new(config).map_err(rt_err)?,
        })
    }

    /// Bytes buffers may occupy (budget minus the compression headroom and what is held back).
    fn limit(&self) -> u64 {
        self.rt.limit()
    }

    /// Keep `nbytes` of the budget for memory the runtime does not own (0165).
    fn hold_back(&self, py: Python<'_>, nbytes: u64) -> PyResult<()> {
        let rt = self.rt.clone();
        py.detach(move || rt.hold_back(nbytes)).map_err(rt_err)
    }

    fn alloc(&self, py: Python<'_>, nbytes: usize, elem: usize) -> PyResult<u64> {
        let rt = self.rt.clone();
        py.detach(move || rt.alloc(nbytes, elem)).map_err(rt_err)
    }

    fn add_file(&self, path: PathBuf, offset: u64, nbytes: usize, elem: usize) -> PyResult<u64> {
        self.rt
            .add_file(&path, offset, nbytes, elem)
            .map_err(rt_err)
    }

    #[pyo3(signature = (id, write=false))]
    fn pin(&self, py: Python<'_>, id: u64, write: bool) -> PyResult<RtPin> {
        let rt = self.rt.clone();
        // SAFETY: Python reaches the data only through the raw address (buffer protocol),
        // never through Rust slices, so shared pins cannot alias a Rust reference.
        let pin = py
            .detach(move || unsafe { rt.pin_shared(id, write) })
            .map_err(rt_err)?;
        Ok(RtPin {
            ptr: pin.as_ptr() as usize,
            len: pin.len(),
            write,
            pin: Mutex::new(Some(pin)),
            exports: AtomicUsize::new(0),
            released: AtomicBool::new(false),
        })
    }

    fn free(&self, py: Python<'_>, id: u64) -> PyResult<()> {
        let rt = self.rt.clone();
        py.detach(move || rt.free(id)).map_err(rt_err)
    }

    /// Ask the service thread to bring the buffer back now.
    fn prefetch(&self, id: u64) -> PyResult<()> {
        self.rt.prefetch(id).map_err(rt_err)
    }

    /// A buffer computed by `func(out, inputs)` now and re-computed when needed. `out` is a
    /// memoryview valid only during the call (the Python layer copies the result into it);
    /// `inputs` are pins of the input buffers (buffer protocol), so arrays made from them stay
    /// valid as long as they live.
    fn derive(
        &self,
        py: Python<'_>,
        inputs: Vec<u64>,
        nbytes: usize,
        elem: usize,
        func: Py<PyAny>,
    ) -> PyResult<u64> {
        let views_rt = self.rt.clone();
        let ids = inputs.clone();
        let compute: memopro::rt::Compute = std::sync::Arc::new(
            move |_ins: &[&[u8]], out: &mut [u8]| -> memopro::Result<()> {
                // input pins for Python (not counted as uses); taken before the GIL
                let pins = ids
                    .iter()
                    // SAFETY: Python reaches them only through the buffer protocol
                    .map(|&id| unsafe { views_rt.pin_untracked(id, false) })
                    .collect::<memopro::Result<Vec<_>>>()?;
                Python::attach(|py| -> PyResult<()> {
                    // SAFETY: `out` stays valid for this call and the view is released below;
                    // only the Python layer's own code touches it.
                    let out_view = unsafe {
                        Bound::from_owned_ptr_or_err(
                            py,
                            ffi::PyMemoryView_FromMemory(
                                out.as_mut_ptr() as *mut c_char,
                                out.len() as isize,
                                ffi::PyBUF_WRITE,
                            ),
                        )?
                    };
                    let in_pins = pins
                        .into_iter()
                        .map(|pin| {
                            Bound::new(
                                py,
                                RtPin {
                                    ptr: pin.as_ptr() as usize,
                                    len: pin.len(),
                                    write: false,
                                    pin: Mutex::new(Some(pin)),
                                    exports: AtomicUsize::new(0),
                                    released: AtomicBool::new(false),
                                },
                            )
                        })
                        .collect::<PyResult<Vec<_>>>()?;
                    let result = func
                        .bind(py)
                        .call1((out_view.clone(), PyList::new(py, &in_pins)?));
                    let _ = out_view.call_method0("release");
                    for p in &in_pins {
                        p.get().release();
                    }
                    result.map(|_| ())
                })
                .map_err(|e| {
                    memopro::Error::InvalidArgument(format!("the derive function failed: {e}"))
                })
            },
        );
        let rt = self.rt.clone();
        py.detach(move || rt.derive(&inputs, nbytes, elem, compute))
            .map_err(rt_err)
    }

    /// Predicted cost of repeating the last recorded cycle of pins (None until a buffer was
    /// pinned twice).
    fn predict<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(p) = self.rt.predict() else {
            return Ok(None);
        };
        let d = PyDict::new(py);
        d.set_item("cycle_pins", p.cycle_pins)?;
        d.set_item("cycle_bytes", p.cycle_bytes)?;
        d.set_item("restore_bytes", p.restore_bytes)?;
        d.set_item("restore_seconds", p.restore_seconds)?;
        d.set_item("compute_seconds", p.compute_seconds)?;
        d.set_item("seconds", p.seconds)?;
        d.set_item("last_seconds", p.last_seconds)?;
        d.set_item("prefetch", p.prefetch)?;
        Ok(Some(d))
    }

    fn evict(&self, py: Python<'_>, id: u64) -> PyResult<bool> {
        let rt = self.rt.clone();
        py.detach(move || rt.evict(id)).map_err(rt_err)
    }

    fn state(&self, id: u64) -> PyResult<&'static str> {
        self.rt.state(id).map(state_name).map_err(rt_err)
    }

    fn nbytes(&self, id: u64) -> PyResult<usize> {
        self.rt.nbytes(id).map_err(rt_err)
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = self.rt.stats();
        let d = PyDict::new(py);
        d.set_item("budget", s.budget)?;
        d.set_item("reserve", s.reserve)?;
        d.set_item("used", s.used)?;
        d.set_item("peak_used", s.peak_used)?;
        d.set_item("buffers", s.buffers)?;
        d.set_item("resident_bytes", s.resident_bytes)?;
        d.set_item("compressed_bytes", s.compressed_bytes)?;
        d.set_item("pinned_bytes", s.pinned_bytes)?;
        d.set_item("loads", s.loads)?;
        d.set_item("load_bytes", s.load_bytes)?;
        d.set_item("rereads", s.rereads)?;
        d.set_item("reread_bytes", s.reread_bytes)?;
        d.set_item("read_seconds", s.read_seconds)?;
        d.set_item("drops", s.drops)?;
        d.set_item("drop_bytes", s.drop_bytes)?;
        d.set_item("compressions", s.compressions)?;
        d.set_item("compress_in", s.compress_in)?;
        d.set_item("compress_out", s.compress_out)?;
        d.set_item("compress_seconds", s.compress_seconds)?;
        d.set_item("decompressions", s.decompressions)?;
        d.set_item("decompress_bytes", s.decompress_bytes)?;
        d.set_item("decompress_seconds", s.decompress_seconds)?;
        d.set_item("incompressible", s.incompressible)?;
        d.set_item("evictions", s.evictions)?;
        d.set_item("refusals", s.refusals)?;
        d.set_item("restore_seconds", s.restore_seconds)?;
        d.set_item("pins", s.pins)?;
        d.set_item("recomputes", s.recomputes)?;
        d.set_item("recompute_bytes", s.recompute_bytes)?;
        d.set_item("recompute_seconds", s.recompute_seconds)?;
        d.set_item("prefetches", s.prefetches)?;
        d.set_item("prefetch_bytes", s.prefetch_bytes)?;
        d.set_item("prefetch_hits", s.prefetch_hits)?;
        d.set_item("prefetch_wasted", s.prefetch_wasted)?;
        d.set_item("prefetch_skipped", s.prefetch_skipped)?;
        d.set_item("written_bytes", s.written_bytes)?;
        Ok(d)
    }
}

/// A pinned buffer. Exposes the buffer protocol (so `numpy.frombuffer(pin, dtype)` views it
/// without copying); the buffer stays pinned while any such view exists, even after
/// `release()` (which only stops new views).
#[pyclass(module = "memopro._core", name = "RtPin", frozen)]
struct RtPin {
    pin: Mutex<Option<memopro::rt::Pin>>,
    ptr: usize,
    len: usize,
    write: bool,
    exports: AtomicUsize,
    released: AtomicBool,
}

impl RtPin {
    fn unpin_if_unused(&self) {
        if self.released.load(Ordering::SeqCst) && self.exports.load(Ordering::SeqCst) == 0 {
            self.pin.lock().unwrap_or_else(|e| e.into_inner()).take();
        }
    }
}

#[pymethods]
impl RtPin {
    #[getter]
    fn nbytes(&self) -> usize {
        self.len
    }

    /// Address of the pinned data (page-aligned; valid while the pin is held).
    #[getter]
    fn address(&self) -> usize {
        self.ptr
    }

    #[getter]
    fn writable(&self) -> bool {
        self.write
    }

    /// Views still alive (numpy arrays or memoryviews of this pin).
    #[getter]
    fn views(&self) -> usize {
        self.exports.load(Ordering::SeqCst)
    }

    /// True once the buffer is unpinned (released and no view left).
    #[getter]
    fn unpinned(&self) -> bool {
        self.pin.lock().unwrap_or_else(|e| e.into_inner()).is_none()
    }

    /// Stop handing out views; the buffer is unpinned as soon as no view is left.
    fn release(&self) {
        self.released.store(true, Ordering::SeqCst);
        self.unpin_if_unused();
    }

    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        let this = slf.get();
        if view.is_null() {
            return Err(PyBufferError::new_err("view is null"));
        }
        if this.released.load(Ordering::SeqCst) {
            return Err(PyBufferError::new_err("this pin was released"));
        }
        if (flags & ffi::PyBUF_WRITABLE) == ffi::PyBUF_WRITABLE && !this.write {
            return Err(PyBufferError::new_err(
                "the buffer is pinned read-only; pin it with write=True",
            ));
        }
        this.exports.fetch_add(1, Ordering::SeqCst);
        // SAFETY: `view` is a valid Py_buffer to fill; the memory stays valid while this pin
        // object lives, and `obj` keeps it alive until the view is released.
        unsafe {
            (*view).obj = slf.clone().into_any().into_ptr();
            (*view).buf = this.ptr as *mut c_void;
            (*view).len = this.len as isize;
            (*view).readonly = if this.write { 0 } else { 1 };
            (*view).itemsize = 1;
            (*view).format = if (flags & ffi::PyBUF_FORMAT) == ffi::PyBUF_FORMAT {
                c"B".as_ptr() as *mut c_char
            } else {
                std::ptr::null_mut()
            };
            (*view).ndim = 1;
            (*view).shape = if (flags & ffi::PyBUF_ND) == ffi::PyBUF_ND {
                &mut (*view).len
            } else {
                std::ptr::null_mut()
            };
            (*view).strides = if (flags & ffi::PyBUF_STRIDES) == ffi::PyBUF_STRIDES {
                &mut (*view).itemsize
            } else {
                std::ptr::null_mut()
            };
            (*view).suboffsets = std::ptr::null_mut();
            (*view).internal = std::ptr::null_mut();
        }
        Ok(())
    }

    unsafe fn __releasebuffer__(&self, _view: *mut ffi::Py_buffer) {
        self.exports.fetch_sub(1, Ordering::SeqCst);
        self.unpin_if_unused();
    }
}

// ---------------------------------------------------------------- transparent paging (0124)

/// NumPy's `PyDataMemAllocator` (numpy/ndarraytypes.h, NumPy 1.22 or newer).
#[repr(C)]
#[derive(Clone, Copy)]
struct NpyAllocator {
    ctx: *mut c_void,
    malloc: unsafe extern "C" fn(*mut c_void, usize) -> *mut c_void,
    calloc: unsafe extern "C" fn(*mut c_void, usize, usize) -> *mut c_void,
    realloc: unsafe extern "C" fn(*mut c_void, *mut c_void, usize) -> *mut c_void,
    free: unsafe extern "C" fn(*mut c_void, *mut c_void, usize),
}

/// NumPy's `PyDataMem_Handler` (version 1).
#[repr(C)]
struct NpyHandler {
    name: [c_char; 127],
    version: u8,
    allocator: NpyAllocator,
}

/// What the NumPy handler functions reach through `ctx`: the pager for large arrays and NumPy's
/// default allocator for the rest.
struct HandlerCtx {
    pager: std::sync::Arc<memopro::rt::Pager>,
    threshold: usize,
    default: NpyAllocator,
}

const MEM_HANDLER: &std::ffi::CStr = c"mem_handler";

/// NumPy's C API table (`_ARRAY_API`), from NumPy 2 or 1.x.
fn numpy_api(py: Python<'_>) -> PyResult<*mut *mut c_void> {
    let module = py
        .import("numpy._core._multiarray_umath")
        .or_else(|_| py.import("numpy.core._multiarray_umath"))?;
    let capsule = module.getattr("_ARRAY_API")?;
    // SAFETY: `_ARRAY_API` is NumPy's unnamed capsule of its function table.
    let table = unsafe { ffi::PyCapsule_GetPointer(capsule.as_ptr(), std::ptr::null()) };
    if table.is_null() {
        return Err(PyErr::fetch(py));
    }
    Ok(table.cast())
}

// Indices in NumPy's C API table (numpy/__multiarray_api.h, stable since NumPy 1.22).
const PYDATAMEM_SETHANDLER: usize = 304;
const PYDATAMEM_DEFAULTHANDLER: usize = 306;

unsafe extern "C" fn np_malloc(ctx: *mut c_void, size: usize) -> *mut c_void {
    // SAFETY: `ctx` is the HandlerCtx this handler was made with; it lives as long as the capsule.
    let c = unsafe { &*(ctx as *const HandlerCtx) };
    if size >= c.threshold
        && let Ok(p) = c.pager.map(size)
    {
        return p.cast();
    }
    // SAFETY: NumPy's own default allocator.
    unsafe { (c.default.malloc)(c.default.ctx, size) }
}

unsafe extern "C" fn np_calloc(ctx: *mut c_void, n: usize, elsize: usize) -> *mut c_void {
    // SAFETY: as in np_malloc.
    let c = unsafe { &*(ctx as *const HandlerCtx) };
    if let Some(size) = n.checked_mul(elsize)
        && size >= c.threshold
        && let Ok(p) = c.pager.map(size)
    {
        return p.cast(); // pager memory starts zeroed
    }
    // SAFETY: NumPy's own default allocator.
    unsafe { (c.default.calloc)(c.default.ctx, n, elsize) }
}

unsafe extern "C" fn np_realloc(
    ctx: *mut c_void,
    ptr: *mut c_void,
    new_size: usize,
) -> *mut c_void {
    // SAFETY: as in np_malloc.
    let c = unsafe { &*(ctx as *const HandlerCtx) };
    if ptr.is_null() {
        // SAFETY: same contract.
        return unsafe { np_malloc(ctx, new_size) };
    }
    let Some(old) = c.pager.owns(ptr.cast()) else {
        // SAFETY: NumPy's own allocation.
        return unsafe { (c.default.realloc)(c.default.ctx, ptr, new_size) };
    };
    // SAFETY: same contract.
    let q = unsafe { np_malloc(ctx, new_size) };
    if !q.is_null() {
        // SAFETY: both regions hold at least min(old, new) bytes; the pager serves the faults.
        unsafe { std::ptr::copy_nonoverlapping(ptr as *const u8, q as *mut u8, old.min(new_size)) };
        // SAFETY: NumPy hands the old block back to us with realloc.
        let _ = unsafe { c.pager.unmap(ptr.cast()) };
    }
    q
}

unsafe extern "C" fn np_free(ctx: *mut c_void, ptr: *mut c_void, size: usize) {
    if ptr.is_null() {
        return;
    }
    // SAFETY: as in np_malloc.
    let c = unsafe { &*(ctx as *const HandlerCtx) };
    if c.pager.owns(ptr.cast()).is_some() {
        // SAFETY: NumPy frees the block it got from np_malloc/np_calloc/np_realloc.
        let _ = unsafe { c.pager.unmap(ptr.cast()) };
        return;
    }
    // SAFETY: NumPy's own allocation.
    unsafe { (c.default.free)(c.default.ctx, ptr, size) }
}

unsafe extern "C" fn free_handler(capsule: *mut ffi::PyObject) {
    // SAFETY: called once by Python when the capsule dies; it holds the handler made below.
    unsafe {
        let h = ffi::PyCapsule_GetPointer(capsule, MEM_HANDLER.as_ptr()) as *mut NpyHandler;
        if !h.is_null() {
            let handler = Box::from_raw(h);
            drop(Box::from_raw(handler.allocator.ctx as *mut HandlerCtx));
        }
    }
}

/// Transparent paging of large NumPy arrays (Linux userfaultfd, `memopro.rt.transparent`).
#[pyclass(module = "memopro._core", name = "RtPager", frozen)]
struct RtPager {
    pager: std::sync::Arc<memopro::rt::Pager>,
    threshold: usize,
}

#[pymethods]
impl RtPager {
    #[new]
    #[pyo3(signature = (budget, chunk=1 << 20, elem=4, compress_level=1, min_saving=0.15, threshold=16 << 20))]
    fn new(
        budget: u64,
        chunk: usize,
        elem: usize,
        compress_level: i32,
        min_saving: f64,
        threshold: usize,
    ) -> PyResult<Self> {
        let config = memopro::rt::PagerConfig {
            budget,
            chunk,
            elem,
            compress_level,
            min_saving,
        };
        let pager = memopro::rt::Pager::new(config).map_err(rt_err)?;
        Ok(RtPager {
            pager: std::sync::Arc::new(pager),
            threshold: threshold.max(1),
        })
    }

    fn limit(&self) -> u64 {
        self.pager.limit()
    }

    /// A zero-filled region of at least `nbytes`; returns its address (for tests and ctypes).
    fn map(&self, py: Python<'_>, nbytes: usize) -> PyResult<usize> {
        let pager = &self.pager;
        py.detach(|| pager.map(nbytes).map(|p| p as usize))
            .map_err(rt_err)
    }

    /// Give back a region from `map`; returns its length. Nothing may use it afterwards.
    fn unmap(&self, py: Python<'_>, address: usize) -> PyResult<usize> {
        let pager = &self.pager;
        // SAFETY: the caller's contract (documented): the region is no longer used.
        py.detach(|| unsafe { pager.unmap(address as *mut u8) })
            .map_err(rt_err)
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = self.pager.stats();
        let d = PyDict::new(py);
        d.set_item("budget", s.budget)?;
        d.set_item("limit", s.limit)?;
        d.set_item("used", s.used)?;
        d.set_item("peak_used", s.peak_used)?;
        d.set_item("regions", s.regions)?;
        d.set_item("mapped_bytes", s.mapped_bytes)?;
        d.set_item("resident_bytes", s.resident_bytes)?;
        d.set_item("compressed_bytes", s.compressed_bytes)?;
        d.set_item("faults", s.faults)?;
        d.set_item("zero_fills", s.zero_fills)?;
        d.set_item("restores", s.restores)?;
        d.set_item("restore_seconds", s.restore_seconds)?;
        d.set_item("evictions", s.evictions)?;
        d.set_item("compress_in", s.compress_in)?;
        d.set_item("compress_out", s.compress_out)?;
        d.set_item("compress_seconds", s.compress_seconds)?;
        d.set_item("incompressible", s.incompressible)?;
        d.set_item("overruns", s.overruns)?;
        d.set_item("spurious", s.spurious)?;
        d.set_item("written_bytes", 0u64)?;
        Ok(d)
    }

    /// A NumPy memory handler (`mem_handler` capsule) that puts arrays of `threshold` bytes or
    /// more in this pager and the rest where NumPy would.
    fn numpy_handler(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let api = numpy_api(py)?;
        // SAFETY: entry 306 is `PyObject *PyDataMem_DefaultHandler` (a capsule NumPy keeps).
        let default_capsule =
            unsafe { *(*api.add(PYDATAMEM_DEFAULTHANDLER) as *mut *mut ffi::PyObject) };
        // SAFETY: NumPy's default handler capsule, named "mem_handler".
        let default = unsafe { ffi::PyCapsule_GetPointer(default_capsule, MEM_HANDLER.as_ptr()) }
            as *const NpyHandler;
        if default.is_null() {
            return Err(PyErr::fetch(py));
        }
        let ctx = Box::into_raw(Box::new(HandlerCtx {
            pager: self.pager.clone(),
            threshold: self.threshold,
            // SAFETY: checked non-null; NumPy's default handler lives as long as NumPy.
            default: unsafe { (*default).allocator },
        }));
        let mut name = [0 as c_char; 127];
        for (d, s) in name.iter_mut().zip(b"memopro_pager") {
            *d = *s as c_char;
        }
        let handler = Box::into_raw(Box::new(NpyHandler {
            name,
            version: 1,
            allocator: NpyAllocator {
                ctx: ctx.cast(),
                malloc: np_malloc,
                calloc: np_calloc,
                realloc: np_realloc,
                free: np_free,
            },
        }));
        // SAFETY: the capsule owns `handler` (and its ctx) and frees them in `free_handler`.
        let capsule =
            unsafe { ffi::PyCapsule_New(handler.cast(), MEM_HANDLER.as_ptr(), Some(free_handler)) };
        if capsule.is_null() {
            // SAFETY: not handed to Python; free what we made.
            unsafe {
                drop(Box::from_raw(handler));
                drop(Box::from_raw(ctx));
            }
            return Err(PyErr::fetch(py));
        }
        // SAFETY: a new reference we own.
        Ok(unsafe { Bound::from_owned_ptr(py, capsule) }.unbind())
    }
}

/// Make `handler` (a `mem_handler` capsule, or None for NumPy's default) NumPy's memory
/// handler in the current context; returns the previous one.
#[pyfunction]
fn numpy_set_handler(py: Python<'_>, handler: Option<Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
    let api = numpy_api(py)?;
    // SAFETY: entry 304 is `PyObject *PyDataMem_SetHandler(PyObject *handler)`.
    let set: unsafe extern "C" fn(*mut ffi::PyObject) -> *mut ffi::PyObject =
        unsafe { std::mem::transmute(*api.add(PYDATAMEM_SETHANDLER)) };
    let arg = handler
        .as_ref()
        .map_or(std::ptr::null_mut(), |h| h.as_ptr());
    // SAFETY: a capsule named "mem_handler" or NULL, as NumPy expects; returns a new reference.
    let old = unsafe { set(arg) };
    if old.is_null() {
        return Err(PyErr::fetch(py));
    }
    // SAFETY: a new reference we own.
    Ok(unsafe { Bound::from_owned_ptr(py, old) }.unbind())
}

/// A no-copy `MTLBuffer` over pinned runtime memory (macOS, G4 E1); give it back with
/// `metal_release`. The caller keeps the memory pinned until the GPU no longer uses it.
#[cfg(unix)]
#[pyfunction]
fn metal_wrap(address: usize, length: usize) -> PyResult<usize> {
    // SAFETY: the caller's contract (documented): pinned memory outlives the buffer's use.
    unsafe { memopro::residency::metal_wrap(address as *mut u8, length) }.map_err(rt_err)
}

#[cfg(unix)]
#[pyfunction]
fn metal_release(buffer: usize) {
    // SAFETY: a handle from metal_wrap, released once by the Python side.
    unsafe { memopro::residency::metal_release(buffer) }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", memopro::VERSION)?;
    m.add("DIGEST_CHUNK", memopro::spill::DIGEST_CHUNK)?;
    m.add_function(wrap_pyfunction!(core_version, m)?)?;
    m.add_function(wrap_pyfunction!(hwinfo_memory, m)?)?;
    m.add_function(wrap_pyfunction!(hwinfo_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(hwinfo_process_rss, m)?)?;
    m.add_function(wrap_pyfunction!(hwinfo_disk, m)?)?;
    m.add_function(wrap_pyfunction!(pressure_current, m)?)?;
    m.add_function(wrap_pyfunction!(engine_digest, m)?)?;
    m.add_function(wrap_pyfunction!(engine_digest_parts, m)?)?;
    m.add_function(wrap_pyfunction!(engine_digest_combine, m)?)?;
    m.add_function(wrap_pyfunction!(engine_read_source_into, m)?)?;
    m.add_function(wrap_pyfunction!(engine_write, m)?)?;
    #[cfg(unix)]
    m.add_class::<FileMap>()?;
    #[cfg(unix)]
    m.add_function(wrap_pyfunction!(metal_wrap, m)?)?;
    #[cfg(unix)]
    m.add_function(wrap_pyfunction!(metal_release, m)?)?;
    m.add_class::<Compressed>()?;
    m.add_function(wrap_pyfunction!(codec_pack, m)?)?;
    m.add_class::<RtRuntime>()?;
    m.add_class::<RtPin>()?;
    m.add_class::<RtPager>()?;
    m.add_function(wrap_pyfunction!(numpy_set_handler, m)?)?;
    Ok(())
}
