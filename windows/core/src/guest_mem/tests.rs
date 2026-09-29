//! The memory routines against byte-by-byte references, and the route over mock CRT routines.
//!
//! Every size from 0 to 1024 bytes runs through each routine, at every source
//! and destination offset in a 16-byte line for the sizes up to 200 and at a
//! few for the rest, with guard bytes on both sides of the destination.
//! `memmove` runs every size inside one buffer at shifts of up to 70 bytes in
//! both directions, every shift for the sizes up to 200 and a set around the
//! block and line sizes for the rest. The route tests stand in for the CRT with routines that
//! count their calls and delegate to the standard library, and check which
//! side each call lands on under an emulator, natively, and with no CRT;
//! that the host is asked once, also when threads race the first call; and
//! that a host query which itself copies memory neither recurses nor waits.

use std::{
    cell::Cell,
    ffi::c_void,
    ptr::NonNull,
    sync::atomic::{AtomicUsize, Ordering},
};

use super::*;

/// Largest size the tests move.
const MAX: usize = 1024;
/// Guard bytes kept on both sides of every destination.
const GUARD: usize = 32;
/// The value the guard bytes and untouched destinations hold.
const SENTINEL: u8 = 0xa5;

thread_local! {
    /// Calls the mock CRT routines took on this thread, by slot.
    static CRT_CALLS: Cell<[usize; 4]> = const { Cell::new([0; 4]) };
}

/// Counts a call of the mock in `slot` on this thread.
fn count(slot: usize) {
    CRT_CALLS.with(|calls| {
        let mut all = calls.get();
        all[slot] += 1;
        calls.set(all);
    });
}

/// The calls the mock CRT routines took on this thread so far.
fn crt_calls() -> [usize; 4] {
    CRT_CALLS.with(Cell::get)
}

/// The CRT's `memcpy` for the tests: counts, then copies through the standard library.
unsafe extern "C" fn mock_memcpy(dst: *mut c_void, src: *const c_void, n: usize) -> *mut c_void {
    count(MEMCPY);
    // SAFETY: the route forwards `memcpy`'s contract.
    unsafe { std::ptr::copy_nonoverlapping(src.cast::<u8>(), dst.cast::<u8>(), n) };
    dst
}

/// The CRT's `memmove` for the tests.
unsafe extern "C" fn mock_memmove(dst: *mut c_void, src: *const c_void, n: usize) -> *mut c_void {
    count(MEMMOVE);
    // SAFETY: the route forwards `memmove`'s contract.
    unsafe { std::ptr::copy(src.cast::<u8>(), dst.cast::<u8>(), n) };
    dst
}

/// The CRT's `memset` for the tests.
unsafe extern "C" fn mock_memset(dst: *mut c_void, value: i32, n: usize) -> *mut c_void {
    count(MEMSET);
    let [byte, ..] = value.to_le_bytes();
    // SAFETY: the route forwards `memset`'s contract.
    unsafe { std::ptr::write_bytes(dst.cast::<u8>(), byte, n) };
    dst
}

/// The CRT's `memcmp` for the tests.
unsafe extern "C" fn mock_memcmp(a: *const c_void, b: *const c_void, n: usize) -> i32 {
    count(MEMCMP);
    // SAFETY: the route forwards `memcmp`'s contract, `n` readable bytes at `a`.
    let a = unsafe { std::slice::from_raw_parts(a.cast::<u8>(), n) };
    // SAFETY: and at `b`.
    let b = unsafe { std::slice::from_raw_parts(b.cast::<u8>(), n) };
    reference_compare(a, b)
}

/// The mock CRT routines, as the host would report them.
fn mock_crt() -> CrtMem {
    let address = |f: *const ()| NonNull::new(f.cast_mut().cast::<c_void>()).expect("fn address");
    // SAFETY: each address is a mock with the C signature of the routine it
    // is passed as, and the mocks live for the whole test binary.
    unsafe {
        CrtMem::new(
            address(mock_memcpy as *const ()),
            address(mock_memmove as *const ()),
            address(mock_memset as *const ()),
            address(mock_memcmp as *const ()),
        )
    }
}

/// A host under an x86 emulator, with the mock CRT.
fn emulated() -> MemHost {
    MemHost::new(true, Some(mock_crt()))
}

/// A host that runs the image natively or translated, with the mock CRT.
fn native() -> MemHost {
    MemHost::new(false, Some(mock_crt()))
}

/// A host whose CRT routines were not found.
const fn without_crt() -> MemHost {
    MemHost::new(false, None)
}

/// `memcmp`'s answer as the sign of the first differing byte, from slice order.
fn reference_compare(a: &[u8], b: &[u8]) -> i32 {
    match a.iter().zip(b).find(|(x, y)| x != y) {
        Some((x, y)) => i32::from(*x) - i32::from(*y),
        None => 0,
    }
}

/// Bytes that vary from position to position, so a misplaced byte shows.
fn pattern(len: usize, seed: u8) -> Vec<u8> {
    (0..len)
        .map(|i| {
            let [low, ..] = i.to_le_bytes();
            low.wrapping_mul(31).wrapping_add(seed) ^ (low >> 3)
        })
        .collect()
}

/// Source and destination offsets to try for a copy of `n` bytes.
fn offsets(n: usize) -> Vec<(usize, usize)> {
    if n <= 200 {
        (0..16).flat_map(|s| (0..16).map(move |d| (s, d))).collect()
    } else {
        vec![(0, 0), (1, 0), (0, 1), (3, 7), (15, 8), (8, 15)]
    }
}

/// Runs a copy of `n` bytes into a guarded buffer and checks the copy and both guards.
fn check_copy(n: usize, route: &str, copy_fn: impl Fn(*mut u8, *const u8, usize)) {
    for (src_off, dst_off) in offsets(n) {
        let src = pattern(n + src_off, 7);
        let mut dst = vec![SENTINEL; GUARD + dst_off + n + GUARD];
        let start = GUARD + dst_off;
        copy_fn(dst[start..].as_mut_ptr(), src[src_off..].as_ptr(), n);
        assert_eq!(
            &dst[start..start + n],
            &src[src_off..],
            "{route}: {n} bytes from offset {src_off} to {dst_off}"
        );
        assert!(
            dst[..start]
                .iter()
                .chain(&dst[start + n..])
                .all(|&b| b == SENTINEL),
            "{route}: {n} bytes from offset {src_off} to {dst_off} wrote outside the range"
        );
    }
}

#[test]
fn copy_matches_the_source_at_every_size_and_offset() {
    for n in 0..=MAX {
        // SAFETY: the check hands over ranges of `n` bytes that do not overlap.
        check_copy(n, "copy", |d, s, n| unsafe { copy(d, s, n) });
    }
}

#[test]
fn copy_overlapping_matches_the_source_for_disjoint_ranges() {
    for n in 0..=MAX {
        // SAFETY: the check hands over ranges of `n` bytes.
        check_copy(n, "copy_overlapping", |d, s, n| unsafe {
            copy_overlapping(d, s, n);
        });
    }
}

#[test]
fn copy_overlapping_moves_as_if_through_a_buffer_in_both_directions() {
    const SHIFT: usize = 70;
    const SOME_SHIFTS: [usize; 17] = [
        0, 1, 6, 7, 37, 53, 54, 55, 69, 70, 71, 85, 86, 102, 133, 139, 140,
    ];
    for n in 0..=MAX {
        let original = pattern(SHIFT + n + SHIFT + 1, 11);
        // The source sits in the middle, at an alignment that changes with `n`.
        let src_at = SHIFT + n % 2;
        let dst_ats: Vec<usize> = if n <= 200 {
            (0..=2 * SHIFT).collect()
        } else {
            SOME_SHIFTS.to_vec()
        };
        for dst_at in dst_ats {
            let mut buffer = original.clone();
            let mut expected = original.clone();
            expected.copy_within(src_at..src_at + n, dst_at);
            let base = buffer.as_mut_ptr();
            // SAFETY: both ranges lie inside `buffer`.
            unsafe { copy_overlapping(base.wrapping_add(dst_at), base.wrapping_add(src_at), n) };
            assert_eq!(buffer, expected, "{n} bytes from {src_at} to {dst_at}");
        }
    }
}

#[test]
fn fill_sets_every_size_and_offset_and_nothing_else() {
    for n in 0..=MAX {
        let offsets = if n <= 200 { 0..16 } else { 0..3 };
        for off in offsets {
            for byte in [0, 1, 0x7f, 0x80, 0xff] {
                let mut dst = vec![SENTINEL; GUARD + off + n + GUARD];
                let start = GUARD + off;
                // SAFETY: `n` bytes from `start` lie inside `dst`.
                unsafe { fill(dst[start..].as_mut_ptr(), byte, n) };
                assert!(
                    dst[start..start + n].iter().all(|&b| b == byte),
                    "{n} bytes at {off}"
                );
                assert!(
                    dst[..start]
                        .iter()
                        .chain(&dst[start + n..])
                        .all(|&b| b == SENTINEL),
                    "{n} bytes at {off} wrote outside the range"
                );
            }
        }
    }
}

#[test]
fn compare_finds_equal_ranges_equal_at_every_size_and_offset() {
    for n in 0..=MAX {
        for (a_off, b_off) in offsets(n) {
            let a = pattern(n + a_off, 3);
            let mut b = vec![0; b_off];
            b.extend_from_slice(&a[a_off..]);
            // SAFETY: both ranges hold `n` bytes.
            let order = unsafe { compare(a[a_off..].as_ptr(), b[b_off..].as_ptr(), n) };
            assert_eq!(order, 0, "{n} bytes at {a_off} and {b_off}");
        }
    }
}

#[test]
fn compare_orders_by_the_first_difference_as_unsigned() {
    for n in (1..=MAX).filter(|n| *n <= 70 || n % 37 == 0 || *n == MAX) {
        let a = pattern(n, 5);
        for at in 0..n {
            for (x, y) in [
                (0x00, 0xff),
                (0xff, 0x00),
                (0x7f, 0x80),
                (0x80, 0x7f),
                (0x41, 0x42),
            ] {
                let mut left = a.clone();
                let mut right = a.clone();
                left[at] = x;
                right[at] = y;
                // A later difference the other way round must not decide.
                if at + 1 < n {
                    left[n - 1] = 0;
                    right[n - 1] = 0xff;
                }
                let expected = reference_compare(&left, &right).signum();
                // SAFETY: both ranges hold `n` bytes.
                let order = unsafe { compare(left.as_ptr(), right.as_ptr(), n) };
                assert_eq!(
                    order.signum(),
                    expected,
                    "{n} bytes, first difference at {at}"
                );
                assert_eq!(order, i32::from(x) - i32::from(y), "{n} bytes at {at}");
            }
        }
    }
}

/// Copies, moves, fills and compares `n` bytes through `route`, checking every result.
fn exercise(route: &MemRoute, host: fn() -> MemHost, n: usize) {
    let src = pattern(n, 9);
    let mut dst = vec![SENTINEL; n];
    // SAFETY: `n` bytes at both ends, which do not overlap.
    unsafe { route.memcpy(host, dst.as_mut_ptr().cast(), src.as_ptr().cast(), n) };
    assert_eq!(dst, src, "memcpy of {n} bytes");

    let mut buffer = pattern(n + 8, 13);
    let mut expected = buffer.clone();
    expected.copy_within(0..n, 8);
    let base = buffer.as_mut_ptr();
    // SAFETY: both ranges lie inside `buffer`.
    unsafe { route.memmove(host, base.wrapping_add(8).cast(), base.cast(), n) };
    assert_eq!(buffer, expected, "memmove of {n} bytes");

    // SAFETY: `n` writable bytes.
    unsafe { route.memset(host, dst.as_mut_ptr().cast(), 0x1c3, n) };
    assert!(
        dst.iter().all(|&b| b == 0xc3),
        "memset of {n} bytes uses the low byte"
    );

    let mut other = src.clone();
    if let Some(last) = other.last_mut() {
        *last ^= 1;
    }
    // SAFETY: `n` readable bytes on both sides.
    let order = unsafe { route.memcmp(host, src.as_ptr().cast(), other.as_ptr().cast(), n) };
    assert_eq!(
        order.signum(),
        reference_compare(&src, &other).signum(),
        "memcmp of {n} bytes"
    );
}

#[test]
fn latching_an_undecided_route_decides_it_and_later_calls_keep_it() {
    let route = MemRoute::new();
    assert_eq!(route.latch(emulated), LatchedRoute::Emulated);
    exercise(
        &route,
        || panic!("a decided route asks the host again"),
        MAX,
    );
    assert_eq!(crt_calls(), [1, 1, 0, 0]);
}

#[test]
fn under_an_emulator_only_long_copies_reach_the_crt() {
    let route = MemRoute::new();
    for n in [0, 1, 64, 65, 256, LOCAL_COPY_MAX] {
        exercise(&route, emulated, n);
    }
    assert_eq!(
        crt_calls(),
        [0; 4],
        "copies up to the limit, fills and compares stay local"
    );
    assert_eq!(route.latch(emulated), LatchedRoute::Emulated);

    for n in [LOCAL_COPY_MAX + 1, MAX] {
        exercise(&route, emulated, n);
    }
    assert_eq!(
        crt_calls(),
        [2, 2, 0, 0],
        "each longer memcpy and memmove calls the CRT once"
    );
}

#[test]
fn natively_every_call_reaches_the_crt() {
    let route = MemRoute::new();
    for n in [0, 1, 64, LOCAL_COPY_MAX, MAX] {
        exercise(&route, native, n);
    }
    assert_eq!(crt_calls(), [5; 4]);
    assert_eq!(route.latch(native), LatchedRoute::Crt);
}

#[test]
fn with_no_crt_every_call_stays_local() {
    let route = MemRoute::new();
    for n in [0, 1, 64, LOCAL_COPY_MAX, LOCAL_COPY_MAX + 1, MAX] {
        exercise(&route, without_crt, n);
    }
    assert_eq!(crt_calls(), [0; 4]);
    assert_eq!(route.latch(without_crt), LatchedRoute::NoCrt);
}

#[test]
fn an_emulator_with_no_crt_keeps_long_copies_local() {
    let route = MemRoute::new();
    exercise(&route, || MemHost::new(true, None), MAX);
    assert_eq!(crt_calls(), [0; 4]);
    assert_eq!(route.latch(emulated), LatchedRoute::NoCrt);
}

#[test]
fn the_host_is_asked_once() {
    let route = MemRoute::new();
    let asked = Cell::new(0);
    let host = || {
        asked.set(asked.get() + 1);
        native()
    };
    let (src, mut dst) = ([1u8; 8], [0u8; 8]);
    for _ in 0..3 {
        // SAFETY: 8 bytes at both ends, which do not overlap.
        unsafe { route.memcpy(host, dst.as_mut_ptr().cast(), src.as_ptr().cast(), 8) };
    }
    assert_eq!(route.latch(host), LatchedRoute::Crt);
    assert_eq!(asked.get(), 1);
}

#[test]
fn a_host_query_that_copies_memory_runs_the_local_routines_meanwhile() {
    let route = MemRoute::new();
    let host = || {
        let (src, mut dst) = (pattern(100, 1), vec![0u8; 100]);
        // SAFETY: 100 bytes at both ends, which do not overlap.
        unsafe { route.memcpy(native, dst.as_mut_ptr().cast(), src.as_ptr().cast(), 100) };
        assert_eq!(dst, src, "a copy made while deciding");
        assert_eq!(route.latch(native), LatchedRoute::Undecided);
        native()
    };
    let (src, mut dst) = (pattern(40, 2), vec![0u8; 40]);
    // SAFETY: 40 bytes at both ends, which do not overlap.
    unsafe { route.memcpy(host, dst.as_mut_ptr().cast(), src.as_ptr().cast(), 40) };
    assert_eq!(dst, src);
    assert_eq!(
        crt_calls(),
        [1, 0, 0, 0],
        "only the first call, after the decision, reaches the CRT"
    );
    assert_eq!(route.latch(native), LatchedRoute::Crt);
}

#[test]
fn threads_racing_the_first_call_copy_correctly_and_ask_the_host_once() {
    let route = MemRoute::new();
    let asked = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for thread in 0..8u8 {
            let (route, asked) = (&route, &asked);
            scope.spawn(move || {
                for n in [0, 3, 33, 100, LOCAL_COPY_MAX + 7] {
                    let src = pattern(n, thread);
                    let mut dst = vec![0u8; n];
                    let host = || {
                        asked.fetch_add(1, Ordering::Relaxed);
                        native()
                    };
                    // SAFETY: `n` bytes at both ends, which do not overlap.
                    unsafe { route.memcpy(host, dst.as_mut_ptr().cast(), src.as_ptr().cast(), n) };
                    assert_eq!(dst, src, "thread {thread}, {n} bytes");
                }
            });
        }
    });
    assert_eq!(asked.load(Ordering::Relaxed), 1);
    assert_eq!(route.latch(native), LatchedRoute::Crt);
}
