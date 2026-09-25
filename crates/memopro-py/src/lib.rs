//! Python bindings for the memopro core, exposed as `memopro._core`.

use pyo3::exceptions::{PyNotImplementedError, PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

/// Returns the version of the Rust core linked into this extension module.
#[pyfunction]
fn core_version() -> &'static str {
    memopro::version()
}

/// Byte-shuffle + zstd compress `data` in independent chunks, in parallel, without the GIL.
///
/// `threads = 0` uses all logical CPUs. Returns one `bytes` object per chunk.
#[pyfunction]
#[pyo3(signature = (data, itemsize, level = 1, chunk_bytes = 4 << 20, threads = 0))]
fn codec_compress<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyBytes>,
    itemsize: usize,
    level: i32,
    chunk_bytes: usize,
    threads: usize,
) -> PyResult<Vec<Bound<'py, PyBytes>>> {
    let src: &[u8] = data.as_bytes(); // immutable bytes object, kept alive by `data`
    let chunks = py
        .detach(|| memopro::codec::compress(src, itemsize, chunk_bytes, level, threads))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(chunks.iter().map(|c| PyBytes::new(py, c)).collect())
}

/// Decompress chunks from `codec_compress` into a new `bytes` of `total_len`, without the GIL.
#[pyfunction]
#[pyo3(signature = (chunks, itemsize, total_len, chunk_bytes = 4 << 20, threads = 0))]
fn codec_decompress<'py>(
    py: Python<'py>,
    chunks: Vec<Bound<'py, PyBytes>>,
    itemsize: usize,
    total_len: usize,
    chunk_bytes: usize,
    threads: usize,
) -> PyResult<Bound<'py, PyBytes>> {
    let refs: Vec<&[u8]> = chunks.iter().map(|c| c.as_bytes()).collect();
    PyBytes::new_with(py, total_len, |out: &mut [u8]| {
        // `out` belongs to a bytes object not yet visible to Python code: safe to fill without GIL
        py.detach(|| memopro::codec::decompress_into(&refs, itemsize, chunk_bytes, out, threads))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })
}

/// E008b (exploratory): like `codec_compress` but with per-thread reused zstd contexts and
/// scratch buffers. Returns only the total compressed size (kernel throughput measurement).
#[pyfunction]
#[pyo3(signature = (data, itemsize, level = 1, chunk_bytes = 4 << 20, threads = 0))]
fn codec_compress_reuse_size(
    py: Python<'_>,
    data: &Bound<'_, PyBytes>,
    itemsize: usize,
    level: i32,
    chunk_bytes: usize,
    threads: usize,
) -> PyResult<usize> {
    let src: &[u8] = data.as_bytes();
    py.detach(|| memopro::codec::compress_reuse(src, itemsize, chunk_bytes, level, threads))
        .map(|chunks| chunks.iter().map(Vec::len).sum())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// E008b (exploratory): pipelined parallel compression streamed straight to a file (fully
/// synced), without creating Python objects for the compressed chunks. Returns bytes written.
#[pyfunction]
#[pyo3(signature = (data, itemsize, path, level = 1, chunk_bytes = 4 << 20, threads = 0))]
fn codec_spill_to_file(
    py: Python<'_>,
    data: &Bound<'_, PyBytes>,
    itemsize: usize,
    path: std::path::PathBuf,
    level: i32,
    chunk_bytes: usize,
    threads: usize,
) -> PyResult<u64> {
    let src: &[u8] = data.as_bytes();
    py.detach(|| memopro::codec::spill_to_file(src, itemsize, chunk_bytes, level, threads, &path))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

fn to_py_err(e: memopro::Error) -> PyErr {
    match e {
        memopro::Error::NotImplemented { .. } => PyNotImplementedError::new_err(e.to_string()),
        memopro::Error::InvalidArgument(_) => PyValueError::new_err(e.to_string()),
        memopro::Error::Io(io) => PyOSError::new_err(io.to_string()),
    }
}

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
fn hwinfo_disk(py: Python<'_>, path: std::path::PathBuf) -> PyResult<Bound<'_, PyDict>> {
    let disk = py
        .detach(|| memopro::hwinfo::disk(&path))
        .map_err(to_py_err)?;
    let d = PyDict::new(py);
    d.set_item("path", disk.path)?;
    d.set_item("total_bytes", disk.total_bytes)?;
    d.set_item("available_bytes", disk.available_bytes)?;
    Ok(d)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", memopro::VERSION)?;
    m.add_function(wrap_pyfunction!(core_version, m)?)?;
    m.add_function(wrap_pyfunction!(hwinfo_memory, m)?)?;
    m.add_function(wrap_pyfunction!(hwinfo_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(hwinfo_disk, m)?)?;
    m.add_function(wrap_pyfunction!(codec_compress, m)?)?;
    m.add_function(wrap_pyfunction!(codec_decompress, m)?)?;
    m.add_function(wrap_pyfunction!(codec_compress_reuse_size, m)?)?;
    m.add_function(wrap_pyfunction!(codec_spill_to_file, m)?)?;
    Ok(())
}
