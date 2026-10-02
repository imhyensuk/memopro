//! The C ABI called as C would call it (the C header is exercised by `tests/smoke.c` in CI).

use super::*;
use std::io::Write;

const MIB: usize = 1 << 20;

fn data_file(n: usize) -> (PathBuf, Vec<u8>) {
    let data: Vec<u8> = (0..n as u64)
        .map(|i| (i.wrapping_mul(2_654_435_761) >> 11) as u8)
        .collect();
    let dir = std::env::temp_dir().join(format!("memopro-c-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("data-{n}.bin"));
    std::fs::File::create(&path)
        .unwrap()
        .write_all(&data)
        .unwrap();
    (path, data)
}

fn cstr(p: &std::path::Path) -> CString {
    CString::new(p.to_str().unwrap()).unwrap()
}

fn new_runtime(budget: u64) -> *mut mp_runtime {
    let mut cfg = std::mem::MaybeUninit::<mp_config>::uninit();
    unsafe { mp_config_default(cfg.as_mut_ptr(), budget) };
    let cfg = unsafe { cfg.assume_init() };
    let mut rt = std::ptr::null_mut();
    assert_eq!(unsafe { mp_runtime_new(&cfg, &mut rt) }, MP_OK);
    rt
}

fn last_error() -> String {
    unsafe { CStr::from_ptr(mp_last_error()) }
        .to_string_lossy()
        .into_owned()
}

#[test]
fn file_buffers_come_back_exact_under_the_budget() {
    let (path, data) = data_file(16 * MIB);
    let rt = new_runtime(12 * MIB as u64);
    let p = cstr(&path);
    let ids: Vec<mp_buffer> = (0..16)
        .map(|i| {
            let mut id = 0;
            let rc = unsafe { mp_add_file(rt, p.as_ptr(), (i * MIB) as u64, MIB, 1, &mut id) };
            assert_eq!(rc, MP_OK, "{}", last_error());
            id
        })
        .collect();
    for _ in 0..3 {
        for (i, &id) in ids.iter().enumerate() {
            let mut pin = std::ptr::null_mut();
            assert_eq!(unsafe { mp_pin_acquire(rt, id, 0, &mut pin) }, MP_OK);
            let n = unsafe { mp_pin_size(pin) };
            let bytes = unsafe { std::slice::from_raw_parts(mp_pin_data(pin) as *const u8, n) };
            assert_eq!(bytes, &data[i * MIB..(i + 1) * MIB]);
            unsafe { mp_unpin(pin) };
        }
    }
    let mut s = mp_stats {
        size: std::mem::size_of::<mp_stats>(),
        ..Default::default()
    };
    assert_eq!(unsafe { mp_stats_get(rt, &mut s) }, MP_OK);
    assert!(s.peak_used <= unsafe { mp_limit(rt) });
    assert!(s.rereads > 0 && s.written_bytes == 0 && s.pinned_bytes == 0);
    let mut pr = mp_prediction {
        size: std::mem::size_of::<mp_prediction>(),
        ..Default::default()
    };
    assert_eq!(unsafe { mp_predict(rt, &mut pr) }, MP_OK);
    assert_eq!(pr.cycle_pins, 16);
    unsafe { mp_runtime_free(rt) };
}

#[test]
fn errors_are_codes_with_a_message() {
    let rt = new_runtime(12 * MIB as u64);
    let mut id = 0;
    // a buffer larger than the budget can ever hold is refused, not swapped
    assert_eq!(unsafe { mp_alloc(rt, 64 * MIB, 4, &mut id) }, MP_ERR_BUDGET);
    assert!(last_error().contains("budget"), "{}", last_error());
    assert_eq!(unsafe { mp_alloc(rt, MIB, 4, &mut id) }, MP_OK);
    let mut pin = std::ptr::null_mut();
    assert_eq!(unsafe { mp_pin_acquire(rt, id, 1, &mut pin) }, MP_OK);
    let mut other = std::ptr::null_mut();
    // a writable pin is the only pin of its buffer
    assert_eq!(
        unsafe { mp_pin_acquire(rt, id, 0, &mut other) },
        MP_ERR_INVALID
    );
    unsafe { mp_unpin(pin) };
    assert_eq!(unsafe { mp_free(rt, 999) }, MP_ERR_INVALID);
    assert!(last_error().contains("unknown buffer"));
    assert_eq!(unsafe { mp_free(std::ptr::null_mut(), 1) }, MP_ERR_INVALID);
    let mut state = -1;
    assert_eq!(unsafe { mp_state(rt, id, &mut state) }, MP_OK);
    let mut short = mp_stats::default(); // size 0: refused, not overrun
    assert_eq!(unsafe { mp_stats_get(rt, &mut short) }, MP_ERR_INVALID);
    let mut pr = mp_prediction {
        size: std::mem::size_of::<mp_prediction>(),
        ..Default::default()
    };
    assert_eq!(unsafe { mp_predict(rt, &mut pr) }, MP_NO_DATA);
    unsafe { mp_runtime_free(rt) };
    unsafe { mp_runtime_free(std::ptr::null_mut()) };
}

unsafe extern "C" fn widen(
    _user: *mut c_void,
    inputs: *const *const c_void,
    sizes: *const usize,
    n: usize,
    out: *mut c_void,
    out_size: usize,
) -> c_int {
    if n != 1 {
        return 1;
    }
    let (src, len) = unsafe { (*inputs as *const u16, *sizes / 2) };
    if out_size != len * 4 {
        return 2;
    }
    let dst = out as *mut u32;
    for i in 0..len {
        unsafe { *dst.add(i) = u32::from(*src.add(i)) << 16 };
    }
    0
}

#[test]
fn derived_buffers_are_recomputed_through_the_c_function() {
    let (path, data) = data_file(8 * MIB);
    let rt = new_runtime(12 * MIB as u64);
    let p = cstr(&path);
    let mut wide = Vec::new();
    for i in 0..8 {
        let mut src = 0;
        let rc = unsafe { mp_add_file(rt, p.as_ptr(), (i * MIB) as u64, MIB, 2, &mut src) };
        assert_eq!(rc, MP_OK);
        let mut d = 0;
        let rc = unsafe {
            mp_derive(
                rt,
                &src,
                1,
                2 * MIB,
                4,
                Some(widen),
                std::ptr::null_mut(),
                &mut d,
            )
        };
        assert_eq!(rc, MP_OK, "{}", last_error());
        wide.push(d);
    }
    for _ in 0..2 {
        for (i, &d) in wide.iter().enumerate() {
            let mut pin = std::ptr::null_mut();
            assert_eq!(
                unsafe { mp_pin_acquire(rt, d, 0, &mut pin) },
                MP_OK,
                "{}",
                last_error()
            );
            let got =
                unsafe { std::slice::from_raw_parts(mp_pin_data(pin) as *const u32, MIB / 2) };
            let src = &data[i * MIB..(i + 1) * MIB];
            for (k, &w) in got.iter().enumerate().step_by(4099) {
                let x = u16::from_le_bytes([src[2 * k], src[2 * k + 1]]);
                assert_eq!(w, u32::from(x) << 16);
            }
            unsafe { mp_unpin(pin) };
        }
    }
    let mut s = mp_stats {
        size: std::mem::size_of::<mp_stats>(),
        ..Default::default()
    };
    assert_eq!(unsafe { mp_stats_get(rt, &mut s) }, MP_OK);
    assert!(s.recomputes > 0, "{s:?}");
    unsafe { mp_runtime_free(rt) };
}

#[test]
fn version_and_abi() {
    assert_eq!(mp_abi_version(), MP_ABI_VERSION);
    let v = unsafe { CStr::from_ptr(mp_version()) }.to_str().unwrap();
    assert_eq!(v, memopro::VERSION);
}
