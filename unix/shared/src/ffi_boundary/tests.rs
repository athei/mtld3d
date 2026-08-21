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
