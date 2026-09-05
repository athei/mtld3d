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

use mtld3d_core::api_lock::ApiGuard;
use mtld3d_shared::{InPtr, VtableThis};
use mtld3d_types::{D3D_OK, D3DERR_INVALIDCALL, E_NOINTERFACE, Guid};

use crate::device::{
    device_api_lock, device_wrapper_add_ref, device_wrapper_note_reset_blocker,
    device_wrapper_release,
};

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
/// it (so every forwarded `AddRef` is balanced by a forwarded `Release`), and
/// [`owning_device`](Self::owning_device) must name the device this object was
/// created from, and null once that device is gone for a type that does not
/// pin it: the engine takes a reference on whatever it returns and hands it to
/// the application. A type that pins its device answers unconditionally, since
/// the device cannot go while this object holds a reference on it.
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

    /// The device that created this object, and what `GetDevice` answers.
    ///
    /// Every child names one, so there is no default: a `Direct3DDevice9`*, or
    /// null only once that device is gone.
    fn owning_device(&self) -> *mut c_void;

    /// The owning device wrapper this object forwards its public 0↔1 transitions to.
    ///
    /// The creating device, which is the answer wherever a public reference
    /// keeps that device alive. A type that deliberately does not pin its
    /// device overrides this and returns null for the cases that opt out;
    /// having a device and pinning one are separate questions.
    fn device_forward_target(&self) -> *mut c_void {
        self.owning_device()
    }

    /// Whether a public 1→0 transition finalizes (frees) the wrapper now.
    ///
    /// Device- or container-owned cached objects (the implicit swapchain and
    /// the implicit render-target / depth-stencil surfaces) return `false`:
    /// they are never freed by `Release`, only at their owner's teardown.
    fn finalizes_on_zero(&self) -> bool {
        true
    }

    /// Whether an outstanding public reference to this object blocks `Reset`.
    ///
    /// True for a `D3DPOOL_DEFAULT` resource and for the device's implicit
    /// surfaces: D3D9 rejects `Reset` while the app still holds any of them.
    /// The engine counts such objects on the owning device across their
    /// public 0↔1 edges (`AddRef` / `Release` / creation), so only
    /// app-visible references count, never the device's own bind slots.
    /// Requires a non-null [`device_forward_target`](Self::device_forward_target).
    fn blocks_reset_while_referenced(&self) -> bool {
        false
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

/// `GetDevice` for any [`ComChild`] that names an owning device.
///
/// Every child object knows the device it was created from, so the answer is
/// always available; the call fails only for a caller that passed no
/// out-param, or for a child whose device has already gone. The returned
/// pointer carries one reference of its own, which the caller releases.
///
/// # Safety
/// `this` is a `*mut T` from a `T` vtable thunk, and `device` is the caller's
/// out-param: either null or a writable `*mut c_void` slot. A null `this` is
/// filtered rather than fatal, unlike the `IUnknown` thunks, where `VtableThis`
/// crashes on purpose so that a refcount miscount surfaces where it happened.
pub unsafe fn com_get_device<T: ComChild>(this: *mut c_void, device: *mut *mut c_void) -> i32 {
    if device.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is `*mut T` per the object's ABI.
    let Some(obj) = (unsafe { InPtr::<T>::opt(this) }) else {
        crate::null_out(device);
        return D3DERR_INVALIDCALL;
    };
    let wrapper = obj.owning_device();
    if wrapper.is_null() {
        crate::null_out(device);
        return D3DERR_INVALIDCALL;
    }
    device_wrapper_add_ref(wrapper);
    // SAFETY: non-null (checked) and writable per the ABI.
    unsafe { *device = wrapper };
    D3D_OK
}

/// Hold the owning device's API lock for a child thunk; see [`device_api_lock`].
///
/// Every entry point of every child object binds this as its first
/// statement, the child-local getters and setters included, so two threads
/// on one object are serialised the way the device's own entry points are.
/// A null `this`, or a child with no device (a managed texture between
/// devices), gets the no-op guard.
#[inline]
pub fn com_api_lock<T: ComChild>(this: *mut c_void) -> ApiGuard {
    // SAFETY: vtable thunk; `this` is `*mut T` per the object's ABI.
    let Some(obj) = (unsafe { InPtr::<T>::opt(this) }) else {
        return ApiGuard::NOOP;
    };
    device_api_lock(obj.owning_device())
}

// ── Shader bytecode read-back ──
//
// Not part of the refcount engine above: no COM object, no vtable, no
// `ComChild`. It lives here because both shader wrappers answer `GetFunction`
// from a token stream and nothing else does.

/// `GetFunction` for the two shader objects.
///
/// Hands the app back the token stream it created the shader from.
/// `pSizeOfData` is required and is both in and out; `pData` is optional, and
/// a null one asks for the size alone, so a caller can size its buffer and ask
/// again. On the copy path `*pSizeOfData` is left exactly as the caller set
/// it: the value is the buffer's size going in, apps do not expect it back
/// changed, and one that reads it afterwards would see its own number.
///
/// # Safety
/// `data` and `size_of_data` are the caller's out-params per the D3D9 ABI:
/// `size_of_data` is either null or points to a writable `u32`, and a non-null `data` points to
/// at least `*size_of_data` writable bytes.
pub unsafe fn com_get_function(bytecode: &[u32], data: *mut c_void, size_of_data: *mut u32) -> i32 {
    if size_of_data.is_null() {
        return D3DERR_INVALIDCALL;
    }
    let need = core::mem::size_of_val(bytecode);
    let Ok(need_u32) = u32::try_from(need) else {
        return D3DERR_INVALIDCALL;
    };
    if data.is_null() {
        // SAFETY: non-null (checked) and writable per the ABI.
        unsafe { *size_of_data = need_u32 };
        return D3D_OK;
    }
    // SAFETY: non-null (checked) and readable per the ABI.
    if unsafe { *size_of_data } < need_u32 {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: the caller's buffer holds at least `need` bytes per the check
    // above, and the token stream is a live `[u32]` for the length copied.
    unsafe {
        core::ptr::copy_nonoverlapping(bytecode.as_ptr().cast::<u8>(), data.cast::<u8>(), need);
    }
    D3D_OK
}

/// `IUnknown::AddRef` for a [`ComChild`]: bump the public refcount.
///
/// On the 0→1 edge, forward one reference to the owning device.
///
/// # Safety
/// `this` is a live `*mut T` obtained from a `T` vtable `AddRef` thunk.
pub unsafe fn com_add_ref<T: ComChild>(this: *mut c_void) -> u32 {
    let (rc, forward, blocks_reset) = {
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
        (rc, forward, obj.blocks_reset_while_referenced())
    };
    // No-op when `forward` is null (object does not forward to the device).
    device_wrapper_add_ref(forward);
    if blocks_reset {
        device_wrapper_note_reset_blocker(forward, true);
    }
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
    if obj.blocks_reset_while_referenced() {
        device_wrapper_note_reset_blocker(obj.device_forward_target(), true);
    }
}

/// `IUnknown::Release` for a [`ComChild`]: drop the public refcount.
///
/// Tolerating release-past-zero, finalize on the 1→0 edge unless the object is
/// cached, and forward the device release *last* (it may destroy the device).
///
/// # Safety
/// `this` is a live `*mut T` obtained from a `T` vtable `Release` thunk.
pub unsafe fn com_release<T: ComChild>(this: *mut c_void) -> u32 {
    let (rc, forward, finalize_now, blocks_reset) = {
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
        (
            rc,
            forward,
            finalize_now,
            obj.blocks_reset_while_referenced(),
        )
    };
    if finalize_now {
        // SAFETY: both counters are zero (`finalizes_on_zero()` true and
        // `private_refcount()` zero) — no other reference can survive.
        unsafe { T::finalize(this.cast::<T>()) };
    }
    if blocks_reset {
        device_wrapper_note_reset_blocker(forward, false);
    }
    // Forward the device release last: it may run `device_release` and tear the
    // device down. No-op when `forward` is null. The wrapper may already be
    // freed (finalized above), so `forward` must have been read before that.
    device_wrapper_release(forward);
    rc
}

/// `IUnknown::QueryInterface` for a wrapper that is exactly one COM object.
///
/// Every `IDirect3DXxx9` wrapper here answers for `IUnknown` and the
/// interfaces it implements with itself: `*ppv = this` plus one `AddRef`
/// through `add_ref`, the wrapper's own thunk, so the reference is counted
/// the way the wrapper counts. Anything else is `E_NOINTERFACE` with `*ppv`
/// nulled, logged once per IID: an SDK probing for an interface the object
/// does not have (`IDirect3DDevice9Ex`, say) is a port candidate worth seeing.
///
/// # Safety
/// `this` is the live wrapper for the vtable call, `riid` is the caller's
/// read-only GUID pointer (or null) and `ppv` is a writable out slot (or null).
pub unsafe fn com_query_interface(
    this: *mut c_void,
    riid: *const Guid,
    ppv: *mut *mut c_void,
    accepted: &[Guid],
    add_ref: extern "system" fn(*mut c_void) -> u32,
    name: &'static str,
) -> i32 {
    // SAFETY: `riid` is the caller's read-only GUID pointer per the contract.
    let iid = unsafe { mtld3d_shared::InPtr::<Guid>::opt(riid.cast()) };
    let riid_lo = iid.as_ref().map_or(0, |g| g.data1);
    log::trace!(target: crate::LOG_TARGET, "{name}::QueryInterface(riid_lo={riid_lo:#010x})");
    let matched = iid.is_some_and(|iid| accepted.contains(&iid));
    if matched && !ppv.is_null() {
        // SAFETY: validated writable out pointer per the contract.
        unsafe { *ppv = this };
        add_ref(this);
        return D3D_OK;
    }
    mtld3d_shared::log_once_warn_by!(
        target: crate::LOG_TARGET,
        key: u64::from(riid_lo),
        "{name}::QueryInterface(riid_lo={riid_lo:#010x}) → E_NOINTERFACE"
    );
    crate::null_out(ppv);
    E_NOINTERFACE
}
