//! Owning cached COM pointer (`CachedComPtr<T, K>`) with RAII refcount.
//!
//! D3D9 makes us hold `AddRef`'d pointers to external COM objects we don't
//! own (the currently-bound render-target surface, vertex buffer at
//! stream 0, etc.). The bookkeeping shape is always the same:
//!
//! - On swap: bump the incoming pointer (if non-null), drop the old one.
//! - On teardown: drop whatever is currently cached.
//!
//! Inline `(*p).vtbl().{add_ref,release}(...)` pairs replicated this at
//! every bound-slot site; refcount correctness leaked to call sites. The
//! type below encodes the invariant: construction (`adopt`) bumps the
//! refcount; `Drop` decrements it. Assignment Drops the old value, so
//! `field = unsafe { CachedComPtr::adopt(new) }` is the swap idiom.
//!
//! The `K: Ownership` type parameter selects the bookkeeping path:
//!
//! - `Owned` (default) — bumps the public `IUnknown` refcount through the
//!   COM vtable (`AddRef`/`Release` thunks). Used for state-block captures
//!   that can outlive the live binding.
//! - `Bound` — bumps the wrapper's device-internal `private_refcount`
//!   inline (no vtable indirection, no `ApiTimer` instrumentation). Used
//!   for the per-draw bind hot path (texture stages, bound VB/IB,
//!   render targets, shader slots, vertex declaration). Keeps a dual
//!   public/private refcount split: the public `IUnknown` count and a
//!   device-internal binding count are tracked separately.

use core::{ffi::c_void, marker::PhantomData, ptr::null_mut};

use mtld3d_shared::VtableThis;

use crate::device::{device_wrapper_add_ref, device_wrapper_release};

/// COM types whose vtable starts with the `IUnknown` head.
///
/// Exposes callable `AddRef`/`Release` thunks, plus a device-internal
/// "bound slot" refcount that swap-by-bind paths use to keep the
/// object alive across game-side `Release`. Implemented by every
/// `IDirect3DXxx9` wrapper in this crate.
pub trait ComUnknown {
    fn vtbl_add_ref(&self) -> unsafe extern "system" fn(*mut c_void) -> u32;
    fn vtbl_release(&self) -> unsafe extern "system" fn(*mut c_void) -> u32;
    /// Increment the device-internal "bound slot" refcount.
    fn private_refcount_inc(&mut self);
    /// Decrement the device-internal "bound slot" refcount.
    ///
    /// If both `refcount` and `private_refcount` reach zero, finalize
    /// the wrapper (free allocations and run any encoder-thread
    /// cleanup).
    ///
    /// # Safety
    /// Caller asserts: the wrapper is live with at least one private
    /// refcount outstanding; after this returns, the wrapper may be
    /// freed and the caller must not access `*this`.
    unsafe fn private_refcount_dec_maybe_finalize(this: *mut Self);
}

/// Marker selecting how [`CachedComPtr`] manages the refcount on its slot.
///
/// # Safety
/// `on_drop` may free the wrapper at `p`; callers of [`CachedComPtr`]
/// rely on this trait being implemented only for the marker types
/// declared in this module.
pub unsafe trait Ownership {
    /// Adjust the refcount on `p` to claim a slot reference.
    ///
    /// # Safety
    /// `p` is non-null and points to a live `T`.
    unsafe fn on_adopt<T: ComUnknown>(p: *mut T);
    /// Release the slot reference on `p`; the wrapper may be freed.
    ///
    /// # Safety
    /// `p` is non-null and was previously claimed by [`Self::on_adopt`].
    unsafe fn on_drop<T: ComUnknown>(p: *mut T);
}

/// Public-refcount ownership: bumps/decrements via the COM vtable's `AddRef`/`Release` thunks.
///
/// The slot participates in the public `IUnknown` refcount the game can
/// observe via `QueryInterface` etc. Used for state-block captures
/// (`StateOp::*` variants) that may outlive the live binding.
pub struct Owned;

// SAFETY: `on_adopt`/`on_drop` only call vtable thunks; correctness
// relies on the same invariants as direct `(*p).vtbl().add_ref(...)`.
unsafe impl Ownership for Owned {
    unsafe fn on_adopt<T: ComUnknown>(p: *mut T) {
        // SAFETY: caller asserts `p` non-null and points to a live `T`.
        let f = unsafe { (*p).vtbl_add_ref() };
        // SAFETY: `f` is the AddRef thunk for the same vtable; passing
        // `p` as IUnknown `this` matches the D3D9 ABI.
        unsafe { f(p.cast::<c_void>()) };
    }
    unsafe fn on_drop<T: ComUnknown>(p: *mut T) {
        // SAFETY: caller asserts `p` non-null and points to a live `T`.
        let f = unsafe { (*p).vtbl_release() };
        // SAFETY: `f` is the Release thunk for the same vtable; passing
        // `p` as IUnknown `this` matches the D3D9 ABI.
        unsafe { f(p.cast::<c_void>()) };
    }
}

/// Private-refcount ownership: bumps/decrements the wrapper's `private_refcount` field directly.
///
/// Via [`ComUnknown::private_refcount_inc`] and
/// [`ComUnknown::private_refcount_dec_maybe_finalize`]. No vtable
/// indirection, no `ApiTimer` instrumentation. Invisible to external COM
/// callers. Used for device-internal bind slots (texture stages, bound
/// VB/IB, render targets, shader slots, vertex declaration).
pub struct Bound;

// SAFETY: `on_adopt` only increments a `u32`; `on_drop` calls the
// wrapper's destruction predicate, which may free the wrapper iff
// both counters reach zero.
unsafe impl Ownership for Bound {
    unsafe fn on_adopt<T: ComUnknown>(p: *mut T) {
        // SAFETY: caller asserts `p` non-null and points to a live `T`.
        unsafe { (*p).private_refcount_inc() };
    }
    unsafe fn on_drop<T: ComUnknown>(p: *mut T) {
        // SAFETY: caller asserts `p` non-null and was previously
        // claimed by `on_adopt`; the trait method may free the wrapper.
        unsafe { T::private_refcount_dec_maybe_finalize(p) };
    }
}

/// Cached pointer to an external COM object.
///
/// The contained pointer is either null or addresses a `T` we hold one
/// refcount on (public or private, selected by [`Ownership`] marker `K`).
/// Constructed via [`Self::adopt`] (which bumps the matching refcount);
/// released via `Drop`. Assignment runs `Drop` on the old value, so the
/// swap idiom is `field = unsafe { CachedComPtr::adopt(new) };`.
pub struct CachedComPtr<T: ComUnknown, K: Ownership = Owned>(*mut T, PhantomData<K>);

impl<T: ComUnknown, K: Ownership> CachedComPtr<T, K> {
    /// Null pointer; safe to construct without owning any retain.
    #[must_use]
    pub const fn null() -> Self {
        Self(null_mut(), PhantomData)
    }

    /// Adopt a refcount on `p`.
    ///
    /// Calls the `K`-selected `on_adopt` on construction (if non-null);
    /// the returned [`CachedComPtr`] will call `K::on_drop` on `Drop`.
    ///
    /// # Safety
    /// Caller asserts: `p` is null OR a valid `*mut T` that remains
    /// callable per `K`'s contract for the lifetime of the returned
    /// [`CachedComPtr`].
    #[must_use]
    pub unsafe fn adopt(p: *mut T) -> Self {
        if !p.is_null() {
            // SAFETY: non-null verified above; caller asserts `p` is a
            // valid `*mut T`.
            unsafe { K::on_adopt(p) };
        }
        Self(p, PhantomData)
    }

    /// Raw pointer for callers that already understand the COM invariant.
    ///
    /// E.g. passing to handler closures, comparing identity.
    #[must_use]
    pub const fn raw(&self) -> *mut T {
        self.0
    }

    /// Safe reference to the pointed-to `T`, or `None` if the slot is null.
    ///
    /// The returned borrow is bound to `&self`, so the slot cannot be
    /// swapped (and the refcount cannot be released) while the borrow is
    /// live.
    #[must_use]
    pub const fn as_ref(&self) -> Option<&T> {
        // SAFETY: a non-null slot holds one refcount on a live `T` (per
        // `adopt`'s invariant). Lifetime elision ties the returned `&T`
        // to `&self`, preventing slot replacement during the borrow.
        unsafe { self.0.as_ref() }
    }
}

impl<T: ComUnknown, K: Ownership> Default for CachedComPtr<T, K> {
    fn default() -> Self {
        Self::null()
    }
}

impl<T: ComUnknown, K: Ownership> Drop for CachedComPtr<T, K> {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: invariant from `adopt` — non-null `self.0` is a
            // valid `*mut T` and the slot holds one refcount on it.
            unsafe { K::on_drop(self.0) };
            self.0 = null_mut();
        }
    }
}

// ── Central public-refcount engine for D3D9 child COM objects ──
//
// Every child wrapper (buffers, shaders, declarations, textures, surfaces,
// queries, state blocks, swapchains) shares one public-`IUnknown` refcount
// lifecycle, so the bookkeeping lives here once instead of being re-derived
// (slightly differently) in each per-type `AddRef`/`Release` thunk:
//
//   - `AddRef` bumps the public refcount; on the 0→1 edge it forwards one
//     reference to the owning device (a child keeps its parent device alive
//     for as long as the app holds a public reference).
//   - `Release` drops the public refcount, tolerating release-past-zero
//     (D3D9 permits it); on the 1→0 edge it finalizes the wrapper (unless the
//     object is device- or container-owned and cached), then releases the
//     forwarded device reference *last* — that release may destroy the device.
//
// Per the D3D9 object model, each child holds one reference on its owning
// device for its public lifetime and releases that device reference last,
// since doing so may cause the device to be destroyed.

/// A D3D9 child COM object driven by the central [`com_add_ref`] / [`com_release`] engine.
///
/// Each `IDirect3DXxx9` wrapper (except the device itself, which owns the
/// teardown sequence) implements it.
///
/// # Safety
/// Implementers must expose the wrapper's own public refcount via
/// [`refcount_mut`](Self::refcount_mut), report a truthful
/// [`private_refcount`](Self::private_refcount), and provide a
/// [`finalize`](Self::finalize) that frees the wrapper exactly once when both
/// counters reach zero. [`device_forward_target`](Self::device_forward_target)
/// must name the owning device iff this object forwards its public refcount to
/// it (so every forwarded `AddRef` is balanced by a forwarded `Release`).
pub unsafe trait ComChild: Sized {
    /// The wrapper's public `IUnknown` refcount field.
    fn refcount_mut(&mut self) -> &mut u32;

    /// The device-internal "bound slot" refcount.
    ///
    /// `0` for wrappers that have none (queries, state blocks, swapchains).
    /// The wrapper is finalized only once both this and the public refcount
    /// reach zero.
    fn private_refcount(&self) -> u32 {
        0
    }

    /// The owning device wrapper this object forwards its public 0↔1 transitions to.
    ///
    /// A `Direct3DDevice9`*, or null to not forward (the default).
    fn device_forward_target(&self) -> *mut c_void {
        null_mut()
    }

    /// Whether a public 1→0 transition finalizes (frees) the wrapper now.
    ///
    /// Device- or container-owned cached objects (the implicit swapchain and
    /// the implicit render-target / depth-stencil surfaces) return `false`:
    /// they are never freed by `Release`, only at their owner's teardown.
    fn finalizes_on_zero(&self) -> bool {
        true
    }

    /// Free the wrapper and all backing allocations.
    ///
    /// Called at most once, when both the public and private refcounts have
    /// reached zero.
    ///
    /// # Safety
    /// `this` is a live wrapper with both counters at zero; the caller must not
    /// access it afterwards.
    unsafe fn finalize(this: *mut Self);
}

/// `IUnknown::AddRef` for a [`ComChild`]: bump the public refcount.
///
/// On the 0→1 edge, forward one reference to the owning device.
///
/// # Safety
/// `this` is a live `*mut T` obtained from a `T` vtable `AddRef` thunk.
pub unsafe fn com_add_ref<T: ComChild>(this: *mut c_void) -> u32 {
    let (rc, forward) = {
        // SAFETY: IUnknown AddRef thunk — D3D9 ABI guarantees `this` is the
        // live `*mut T` for the call; null `this` is UB per spec.
        let mut wrap = unsafe { VtableThis::<T>::new(this) };
        let obj: &mut T = &mut wrap;
        let rc = *obj.refcount_mut() + 1;
        *obj.refcount_mut() = rc;
        let forward = if rc == 1 {
            obj.device_forward_target()
        } else {
            null_mut()
        };
        (rc, forward)
    };
    // No-op when `forward` is null (object does not forward to the device).
    device_wrapper_add_ref(forward);
    rc
}

/// Register a freshly created [`ComChild`] with its owning device.
///
/// Born at public refcount 1, it thereby bypasses the 0→1 edge in
/// [`com_add_ref`], so registration takes the one device reference its public
/// refcount holds. No-op for objects whose
/// [`device_forward_target`](ComChild::device_forward_target) is null.
///
/// # Safety
/// `this` is a freshly created, live `*mut T` at public refcount 1 that has not
/// yet been handed to the app.
pub unsafe fn com_register_child<T: ComChild>(this: *mut T) {
    // SAFETY: caller passes a live, freshly created wrapper.
    let obj = unsafe { &*this };
    device_wrapper_add_ref(obj.device_forward_target());
}

/// `IUnknown::Release` for a [`ComChild`]: drop the public refcount.
///
/// Tolerating release-past-zero, finalize on the 1→0 edge unless the object is
/// cached, and forward the device release *last* (it may destroy the device).
///
/// # Safety
/// `this` is a live `*mut T` obtained from a `T` vtable `Release` thunk.
pub unsafe fn com_release<T: ComChild>(this: *mut c_void) -> u32 {
    let (rc, forward, finalize_now) = {
        // SAFETY: IUnknown Release thunk — D3D9 ABI guarantees `this` is the
        // live `*mut T` for the call; null `this` is UB per spec.
        let mut wrap = unsafe { VtableThis::<T>::new(this) };
        let obj: &mut T = &mut wrap;
        // Tolerate Release-past-zero (D3D9 permits it). The implicit objects
        // are released past their app reference, expecting the device to hold
        // the base ref; without this the public refcount underflows.
        if *obj.refcount_mut() == 0 {
            return 0;
        }
        let rc = *obj.refcount_mut() - 1;
        *obj.refcount_mut() = rc;
        if rc != 0 {
            return rc;
        }
        // Public refcount hit zero. Capture the device forward target *before*
        // any finalize frees the wrapper.
        let forward = obj.device_forward_target();
        let finalize_now = obj.finalizes_on_zero() && obj.private_refcount() == 0;
        (rc, forward, finalize_now)
    };
    if finalize_now {
        // SAFETY: both counters are zero (`finalizes_on_zero()` true and
        // `private_refcount()` zero) — no other reference can survive.
        unsafe { T::finalize(this.cast::<T>()) };
    }
    // Forward the device release last: it may run `device_release` and tear the
    // device down. No-op when `forward` is null. The wrapper may already be
    // freed (finalized above), so `forward` must have been read before that.
    device_wrapper_release(forward);
    rc
}
