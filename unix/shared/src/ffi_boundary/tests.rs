//! Unit tests for the typed FFI-boundary pointer wrappers.
//!
//! A local `#[repr(C)]` struct stands in for a C in-param: the tests pin that `opt` filters null
//! on `InPtr` and `OutPtr`, that reads and writes through `InPtr`, `InPtrMut`, `ValueIn`,
//! `OutPtr` and `VtableThis` land on the original storage, and that `InPtr`, `Option<InPtr>`
//! (which relies on the null-pointer niche), `OutPtr` and `VtableThis` are still exactly
//! pointer-sized, so those four cost nothing at the call seam.

use super::*;

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct Point {
    x: i32,
    y: i32,
}

#[test]
fn in_ptr_opt_filters_null() {
    // SAFETY: passing literal null is sound; opt filters it.
    let opt: Option<InPtr<'_, Point>> = unsafe { InPtr::opt(core::ptr::null()) };
    assert!(opt.is_none());
}

#[test]
fn in_ptr_round_trip() {
    let p = Point { x: 7, y: -3 };
    let raw: *const c_void = (&raw const p).cast();
    // SAFETY: `raw` points to a live local `Point` for the call frame.
    let wrap: InPtr<'_, Point> = unsafe { InPtr::opt(raw) }.unwrap();
    assert_eq!(*wrap, p);
}

#[test]
fn in_ptr_mut_round_trip() {
    let mut p = Point { x: 1, y: 2 };
    let raw: *mut c_void = (&raw mut p).cast();
    // SAFETY: exclusive access — local `p` not aliased.
    let mut wrap: InPtrMut<'_, Point> = unsafe { InPtrMut::opt(raw) }.unwrap();
    wrap.x = 99;
    assert_eq!(p.x, 99);
}

#[test]
fn value_in_reads_by_value() {
    let p = Point { x: 5, y: 6 };
    let raw: *const c_void = (&raw const p).cast();
    // SAFETY: `raw` points to a live local `Point`.
    let v: ValueIn<'_, Point> = unsafe { ValueIn::opt(raw) }.unwrap();
    assert_eq!(v.read(), p);
}

#[test]
fn out_ptr_writes_through_pointer() {
    let mut p = Point { x: 0, y: 0 };
    let raw: *mut Point = &raw mut p;
    // SAFETY: `raw` points to a writable local.
    let o: OutPtr<'_, Point> = unsafe { OutPtr::opt(raw) }.unwrap();
    o.write(Point { x: 11, y: 22 });
    assert_eq!(p, Point { x: 11, y: 22 });
}

#[test]
fn out_ptr_opt_filters_null() {
    // SAFETY: null is sound; opt filters it.
    let opt: Option<OutPtr<'_, Point>> = unsafe { OutPtr::opt(core::ptr::null_mut()) };
    assert!(opt.is_none());
}

#[test]
fn vtable_this_round_trip() {
    let mut p = Point { x: 10, y: 20 };
    let raw: *mut c_void = (&raw mut p).cast();
    // SAFETY: simulating an IUnknown thunk entry with a live local.
    let wrap: VtableThis<'_, Point> = unsafe { VtableThis::new(raw) };
    assert_eq!(*wrap, Point { x: 10, y: 20 });
}

#[test]
fn types_are_zero_cost() {
    assert_eq!(
        core::mem::size_of::<InPtr<'_, Point>>(),
        core::mem::size_of::<*const Point>(),
    );
    assert_eq!(
        core::mem::size_of::<Option<InPtr<'_, Point>>>(),
        core::mem::size_of::<*const Point>(),
    );
    assert_eq!(
        core::mem::size_of::<OutPtr<'_, Point>>(),
        core::mem::size_of::<*mut Point>(),
    );
    assert_eq!(
        core::mem::size_of::<VtableThis<'_, Point>>(),
        core::mem::size_of::<*mut Point>(),
    );
}

/// An aligned caller array is borrowed, not copied.
#[test]
fn an_aligned_caller_array_is_borrowed() {
    let src = [[1.0f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];
    // SAFETY: `src` holds the two rows the count names.
    let got = unsafe { super::slice_from_caller(src.as_ptr(), src.len()) };
    assert!(matches!(got, std::borrow::Cow::Borrowed(_)));
    assert_eq!(
        got.as_ptr(),
        src.as_ptr(),
        "the borrow is the caller's memory"
    );
}

/// A misaligned caller array is copied, with its contents intact.
#[test]
fn a_misaligned_caller_array_is_copied() {
    let want = [1.0f32, 2.0, 3.0, 4.0];
    let mut buf = [0.0f32; 5];
    // SAFETY: `buf` is 20 bytes, so byte offset 1 is in range.
    let dst = unsafe { buf.as_mut_ptr().cast::<u8>().add(1) };
    // SAFETY: 16 bytes from byte offset 1 stay inside the 20, no overlap.
    unsafe {
        core::ptr::copy_nonoverlapping(want.as_ptr().cast::<u8>(), dst, size_of::<[f32; 4]>());
    };
    // SAFETY: one byte into a 20-byte buffer, leaving the 16 written above.
    let unaligned = unsafe { buf.as_ptr().byte_add(1) };
    assert!(
        !unaligned.is_aligned(),
        "the test needs a misaligned pointer"
    );

    // SAFETY: `unaligned` addresses the 16 bytes written above.
    let got = unsafe { super::slice_from_caller(unaligned.cast::<[f32; 4]>(), 1) };
    assert!(matches!(got, std::borrow::Cow::Owned(_)));
    let bits = |v: [f32; 4]| v.map(f32::to_bits);
    assert_eq!(
        bits(got[0]),
        bits(want),
        "the copy carries the caller's bytes",
    );
}

/// A zero count borrows nothing and reads nothing.
#[test]
fn a_zero_count_array_is_empty() {
    let src = [[0.0f32; 4]; 1];
    // SAFETY: a zero-length read of a live allocation.
    let got = unsafe { super::slice_from_caller(src.as_ptr(), 0) };
    assert!(got.is_empty());
}
