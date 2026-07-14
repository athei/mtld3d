//! Typed FFI-boundary references.
//!
//! Every untyped pointer that crosses an FFI seam (COM vtable `this` on the
//! PE side, unix-call handler params, typed in/out C struct pointers) is
//! wrapped in a zero-cost newtype defined here. The newtype constructor is
//! `unsafe fn` — that is the **one** place a caller asserts the ABI contract
//! (alignment, validity, lifetime). Methods on the newtype are safe because
//! the invariant is type-encoded and the abstract lifetime `'a` is bound to
//! the call frame via `PhantomData<&'a …>`, preventing escape.
//!
//! A safe `pub fn` that accepts an arbitrary raw pointer and hands back a
//! reference synthesises an unbounded lifetime, which is unsound by strict
//! Rust convention; the constructors here are `unsafe fn` for that reason.
//! The analogous stdlib primitives — [`core::slice::from_raw_parts`],
//! [`Box::from_raw`](std::boxed::Box::from_raw),
//! [`core::pin::Pin::new_unchecked`] — are all `unsafe fn` on the same
//! grounds.
//!
//! Each call site writes its `// SAFETY:` justification at the constructor,
//! derived from the calling ABI (e.g. "`IDirect3DDevice9` vtable guarantees
//! `this` is `*mut Direct3DDevice9` for the call duration"). See
//! `docs/CONVENTIONS.md` §"Unsafe is a last resort" for the policy.

use core::{
    ffi::c_void,
    marker::PhantomData,
    ops::{Deref, DerefMut},
    ptr::NonNull,
};

/// Borrowed input pointer at an FFI entry. Filters null via `opt`.
///
/// The lifetime `'a` is the call-frame scope during which the caller
/// guarantees the pointee outlives. Implements [`Deref`] to `T` so callers
/// invoke methods directly (`wrap.method()`).
pub struct InPtr<'a, T>(NonNull<T>, PhantomData<&'a T>);

impl<T> InPtr<'_, T> {
    /// Wrap `p` as a borrowed `T`, filtering null to `None`.
    ///
    /// # Safety
    ///
    /// Caller asserts that any non-null `p` is a valid, properly-aligned,
    /// initialised `T*` whose underlying storage outlives the call frame.
    /// Common ABIs that satisfy this: COM vtable methods (the
    /// `IDirect3DXxx9` ABI guarantees the typed pointer is valid for the
    /// duration of the call) and typed FFI in-params from C.
    #[must_use]
    pub const unsafe fn opt(p: *const c_void) -> Option<Self> {
        match NonNull::new(p.cast::<T>().cast_mut()) {
            Some(nn) => Some(Self(nn, PhantomData)),
            None => None,
        }
    }

    /// [`Self::opt`] that panics on null.
    ///
    /// # Safety
    /// As `opt`, plus caller asserts non-null.
    ///
    /// # Panics
    /// Panics if `p` is null.
    #[must_use]
    pub const unsafe fn new(p: *const c_void) -> Self {
        // SAFETY: forwarded from this fn's unsafe contract.
        match unsafe { Self::opt(p) } {
            Some(v) => v,
            None => panic!("FFI in-ptr null"),
        }
    }
}

impl<T> Deref for InPtr<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        // SAFETY: invariant carried from construction.
        unsafe { self.0.as_ref() }
    }
}

/// Exclusively-borrowed input pointer (handler params written back to the caller).
///
/// Same shape as [`InPtr`] but yields `&mut T` via [`DerefMut`].
pub struct InPtrMut<'a, T>(NonNull<T>, PhantomData<&'a mut T>);

impl<T> InPtrMut<'_, T> {
    /// Wrap `p` as an exclusively-borrowed `T`, filtering null to `None`.
    ///
    /// # Safety
    ///
    /// As [`InPtr::opt`], plus the caller asserts exclusive access for the
    /// call frame. For unix-call handler params: the PE side blocks on
    /// `unix_call` while the unix side holds the only reference.
    #[must_use]
    pub const unsafe fn opt(p: *mut c_void) -> Option<Self> {
        match NonNull::new(p.cast::<T>()) {
            Some(nn) => Some(Self(nn, PhantomData)),
            None => None,
        }
    }

    /// [`Self::opt`] that panics on null.
    ///
    /// # Safety
    /// As `opt`, plus caller asserts non-null.
    ///
    /// # Panics
    /// Panics if `p` is null.
    #[must_use]
    pub const unsafe fn new(p: *mut c_void) -> Self {
        // SAFETY: forwarded from this fn's unsafe contract.
        match unsafe { Self::opt(p) } {
            Some(v) => v,
            None => panic!("FFI in-ptr-mut null"),
        }
    }
}

impl<T> Deref for InPtrMut<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        // SAFETY: invariant carried from construction.
        unsafe { self.0.as_ref() }
    }
}

impl<T> DerefMut for InPtrMut<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: invariant carried from construction.
        unsafe { self.0.as_mut() }
    }
}

/// Read-by-value typed FFI in-param (`D3DRECT`, `D3DMATRIX`, etc.).
///
/// Filters null at `opt`; `read` consumes `self` and returns the value.
pub struct ValueIn<'a, T: Copy>(NonNull<T>, PhantomData<&'a T>);

impl<T: Copy> ValueIn<'_, T> {
    /// Wrap `p` for a by-value read, filtering null to `None`.
    ///
    /// # Safety
    ///
    /// Caller asserts that any non-null `p` is a properly-aligned pointer
    /// to an initialised `T` (the D3D9 ABI for typed-struct in-params).
    #[must_use]
    pub const unsafe fn opt(p: *const c_void) -> Option<Self> {
        match NonNull::new(p.cast::<T>().cast_mut()) {
            Some(nn) => Some(Self(nn, PhantomData)),
            None => None,
        }
    }

    /// Consume `self` and return the pointee by value.
    #[must_use]
    pub const fn read(self) -> T {
        // SAFETY: invariant carried from construction.
        unsafe { self.0.as_ptr().read() }
    }

    /// One-shot null-filtered read. Convenience for the `opt + read` pair.
    ///
    /// # Safety
    /// As [`Self::opt`].
    #[must_use]
    pub const unsafe fn read_opt(p: *const c_void) -> Option<T> {
        // SAFETY: forwarded from this fn's unsafe contract.
        match unsafe { Self::opt(p) } {
            Some(v) => Some(v.read()),
            None => None,
        }
    }
}

/// Null-guarded FFI out-param. `write` consumes `self`.
///
/// Construction is `unsafe` (the caller asserts writeability and alignment);
/// `write` is safe because the invariant is type-encoded.
pub struct OutPtr<'a, T>(NonNull<T>, PhantomData<&'a mut T>);

impl<T> OutPtr<'_, T> {
    /// Wrap `out` as an out-param, filtering null to `None`.
    ///
    /// # Safety
    ///
    /// Caller asserts that any non-null `out` is a writable, properly-aligned
    /// `T*` valid for the call frame (the D3D9 ABI for typed out-params).
    #[must_use]
    pub const unsafe fn opt(out: *mut T) -> Option<Self> {
        match NonNull::new(out) {
            Some(nn) => Some(Self(nn, PhantomData)),
            None => None,
        }
    }

    /// Consume `self` and write `val` through the pointer.
    pub const fn write(self, val: T) {
        // SAFETY: invariant carried from construction.
        unsafe { self.0.as_ptr().write(val) };
    }

    /// One-shot null-guarded write. Convenience for the `opt + write` pair.
    ///
    /// Silent no-op on null `out`.
    ///
    /// # Safety
    /// As [`Self::opt`].
    pub unsafe fn write_opt(out: *mut T, val: T) {
        // SAFETY: forwarded from this fn's unsafe contract.
        if let Some(o) = unsafe { Self::opt(out) } {
            o.write(val);
        }
    }
}

/// `IUnknown` vtable `this`.
///
/// Crashes on null per D3D9 spec (the spec leaves null-`this` UB; silently
/// filtering would mask refcount underflow).
///
/// Construction is `unsafe`; [`Deref`] and [`DerefMut`] are safe.
pub struct VtableThis<'a, T>(NonNull<T>, PhantomData<&'a mut T>);

impl<T> VtableThis<'_, T> {
    /// Wrap a vtable `this`, panicking on null.
    ///
    /// # Safety
    ///
    /// Caller is invoked from an `IUnknown` vtable thunk (`AddRef`,
    /// `Release`, `QueryInterface`, or any method that wants crash-on-bug
    /// rather than silent recovery). The D3D9 ABI guarantees `this` is
    /// `*mut T` for the call duration; null-`this` is UB per spec and is
    /// deliberately not filtered.
    ///
    /// # Panics
    ///
    /// Panics on null `this` (preserves crash-on-refcount-miscount).
    #[must_use]
    pub const unsafe fn new(this: *mut c_void) -> Self {
        match NonNull::new(this.cast::<T>()) {
            Some(nn) => Self(nn, PhantomData),
            None => panic!("D3D9 vtable this=null"),
        }
    }
}

impl<T> Deref for VtableThis<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        // SAFETY: invariant carried from construction.
        unsafe { self.0.as_ref() }
    }
}

impl<T> DerefMut for VtableThis<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: invariant carried from construction.
        unsafe { self.0.as_mut() }
    }
}

#[cfg(test)]
mod tests {
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
}
