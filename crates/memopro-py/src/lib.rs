//! Python bindings for the memopro core, exposed as `memopro._core`.

use pyo3::prelude::*;

/// Returns the version of the Rust core linked into this extension module.
#[pyfunction]
fn core_version() -> &'static str {
    memopro::version()
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", memopro::VERSION)?;
    m.add_function(wrap_pyfunction!(core_version, m)?)?;
    Ok(())
}
