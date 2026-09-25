//! Python bindings for the memopro core, exposed as `memopro._core`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

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

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", memopro::VERSION)?;
    m.add_function(wrap_pyfunction!(core_version, m)?)?;
    m.add_function(wrap_pyfunction!(codec_compress, m)?)?;
    m.add_function(wrap_pyfunction!(codec_decompress, m)?)?;
    m.add_function(wrap_pyfunction!(codec_compress_reuse_size, m)?)?;
    m.add_function(wrap_pyfunction!(codec_spill_to_file, m)?)?;
    Ok(())
}
