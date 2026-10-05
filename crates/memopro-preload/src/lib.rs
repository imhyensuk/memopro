//! `LD_PRELOAD` library for Linux (0189): allocations of `MEMOPRO_PRELOAD_THRESHOLD` bytes or
//! more (default 4 MiB) made by an unchanged program — C extensions included — come from
//! memopro's transparent pager ([`memopro::rt::Pager`]), which keeps them within a budget by
//! compressing chunks losslessly in memory. Smaller allocations go to the system `malloc`.
//!
//! Settings (environment, read at the first large allocation):
//! - `MEMOPRO_PRELOAD_BUDGET` bytes for the pager's chunks and their compressed copies
//!   (required; without it everything goes to the system allocator);
//! - `MEMOPRO_PRELOAD_PROCESS` bytes for the whole process: the pager's limit shrinks by what
//!   the process holds outside it;
//! - `MEMOPRO_PRELOAD_THRESHOLD` bytes;
//! - `MEMOPRO_PRELOAD_REPORT` a file for the pager's counters at exit (JSON).
//!
//! Prior art: ExtMEM (USENIX ATC 2024) preloads a library over userfaultfd, libvmmalloc
//! interposes `malloc`; nothing here is claimed as new. Limits: a child made by `fork` without
//! `exec` must not touch the parent's paged memory (the pager thread is not there).
//! Elsewhere than Linux the library is empty.

#[cfg(target_os = "linux")]
mod imp;
