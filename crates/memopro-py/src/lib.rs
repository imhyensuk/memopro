//! Python bindings for the memopro core, exposed as `memopro._core`.
//!
//! Bulk data is passed through the buffer protocol (RS2): the bindings borrow the caller's memory
//! (e.g. `tensor.view(torch.uint8).numpy()`) and release the GIL while working on it. The caller
//! must keep the buffer alive and unchanged for the duration of the call, which the Python layer
//! does by holding the object.

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyNotImplementedError, PyOSError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use std::path::PathBuf;

fn to_py_err(e: memopro::Error) -> PyErr {
    match e {
        memopro::Error::NotImplemented { .. } => PyNotImplementedError::new_err(e.to_string()),
        memopro::Error::InvalidArgument(_) => PyValueError::new_err(e.to_string()),
        memopro::Error::Integrity(_) => PyRuntimeError::new_err(e.to_string()),
        memopro::Error::Unsupported(_) => PyNotImplementedError::new_err(e.to_string()),
        memopro::Error::Io(io) => PyOSError::new_err(io.to_string()),
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
    m.add_function(wrap_pyfunction!(engine_read_source_into, m)?)?;
    m.add_function(wrap_pyfunction!(engine_write, m)?)?;
    m.add_class::<Compressed>()?;
    m.add_function(wrap_pyfunction!(codec_pack, m)?)?;
    Ok(())
}
