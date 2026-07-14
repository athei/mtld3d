//! `IDirect3DResource9::{Set,Get,Free}PrivateData` storage.
//!
//! Every D3D9 resource can hold arbitrary application data keyed by GUID:
//! either a raw byte blob or an `IUnknown*` the runtime `AddRef`s on store and
//! `Release`s on overwrite/free/destroy. A resource embeds one
//! [`PrivateDataStore`] and forwards its three vtable thunks to it; the
//! `IUnknown` lifecycle (and the `AddRef`/`Release` callbacks games hang off
//! it) falls out of the store's own `Drop`.

use core::ffi::c_void;
use std::collections::HashMap;

use mtld3d_types::{
    D3D_OK, D3DERR_INVALIDCALL, D3DERR_MOREDATA, D3DERR_NOTFOUND, D3DSPD_IUNKNOWN, Guid,
};

/// `IUnknown` vtable head, enough to call `AddRef`/`Release` on an external COM pointer.
///
/// Such a pointer is handed to `SetPrivateData(..., D3DSPD_IUNKNOWN)`.
#[repr(C)]
struct IUnknownVtbl {
    query_interface: unsafe extern "system" fn(*mut c_void, *const Guid, *mut *mut c_void) -> i32,
    add_ref: unsafe extern "system" fn(*mut c_void) -> u32,
    release: unsafe extern "system" fn(*mut c_void) -> u32,
}

#[repr(C)]
struct IUnknownHead {
    vtbl: *const IUnknownVtbl,
}

/// Call `AddRef` on a non-null external `IUnknown*`.
///
/// # Safety
/// `punk` must be a live COM object whose first field is a pointer to an
/// `IUnknown`-compatible vtable (its `AddRef`/`Release` slots are callable).
unsafe fn iunknown_add_ref(punk: *mut c_void) {
    // SAFETY: caller asserts `punk` heads a live IUnknownHead.
    let vtbl = unsafe { (*punk.cast::<IUnknownHead>()).vtbl };
    // SAFETY: `vtbl` is the object's installed IUnknown vtable.
    let add_ref = unsafe { (*vtbl).add_ref };
    // SAFETY: `add_ref` is the IUnknown `AddRef` slot; `punk` is its `this`.
    unsafe { add_ref(punk) };
}

/// Call `Release` on a non-null external `IUnknown*`.
///
/// # Safety
/// As [`iunknown_add_ref`]; additionally a matching `AddRef` must have been
/// issued (the store only releases what it `AddRef`'d).
unsafe fn iunknown_release(punk: *mut c_void) {
    // SAFETY: caller asserts `punk` heads a live IUnknownHead.
    let vtbl = unsafe { (*punk.cast::<IUnknownHead>()).vtbl };
    // SAFETY: `vtbl` is the object's installed IUnknown vtable.
    let release = unsafe { (*vtbl).release };
    // SAFETY: `release` is the IUnknown `Release` slot; `punk` is its `this`.
    unsafe { release(punk) };
}

enum Entry {
    Blob(Vec<u8>),
    Unknown(*mut c_void),
}

impl Entry {
    /// Release the held `IUnknown`, if any. Consumes the entry.
    fn release(self) {
        if let Self::Unknown(punk) = self
            && !punk.is_null()
        {
            // SAFETY: an `Unknown` entry was `AddRef`'d in `set`; release the
            // matching reference exactly once as the entry is dropped.
            unsafe { iunknown_release(punk) };
        }
    }
}

/// Per-resource GUID-keyed private-data table.
#[derive(Default)]
pub struct PrivateDataStore {
    entries: HashMap<Guid, Entry>,
}

impl PrivateDataStore {
    /// `IDirect3DResource9::SetPrivateData`.
    ///
    /// `D3DSPD_IUNKNOWN` stores the `IUnknown*` at `data` (`AddRef`'d);
    /// otherwise `size` bytes at `data` are copied into a blob. Replacing a
    /// key releases the previous entry.
    ///
    /// # Safety
    /// `data` must be valid for `size` bytes (or, for `D3DSPD_IUNKNOWN`, a live
    /// `IUnknown*` of pointer width), per the D3D9 ABI.
    pub unsafe fn set(&mut self, guid: &Guid, data: *const c_void, size: u32, flags: u32) -> i32 {
        if data.is_null() {
            return D3DERR_INVALIDCALL;
        }
        let entry = if flags & D3DSPD_IUNKNOWN != 0 {
            if size as usize != core::mem::size_of::<*mut c_void>() {
                return D3DERR_INVALIDCALL;
            }
            // For D3DSPD_IUNKNOWN, `data` *is* the IUnknown interface pointer.
            let punk = data.cast_mut();
            // SAFETY: caller asserts `data` is a live IUnknown for this flag.
            unsafe { iunknown_add_ref(punk) };
            Entry::Unknown(punk)
        } else {
            let len = size as usize;
            // SAFETY: caller asserts `data` is readable for `size` bytes.
            let bytes = unsafe { core::slice::from_raw_parts(data.cast::<u8>(), len) };
            Entry::Blob(bytes.to_vec())
        };
        if let Some(old) = self.entries.insert(*guid, entry) {
            old.release();
        }
        D3D_OK
    }

    /// `IDirect3DResource9::GetPrivateData`.
    ///
    /// Copies a stored blob into `data` (or `AddRef`s + writes a stored
    /// `IUnknown*`), writing the byte size to `*size_inout`. Returns
    /// `D3DERR_MOREDATA` (and the needed size) when the caller's buffer is
    /// too small, `D3DERR_NOTFOUND` when the key is absent.
    ///
    /// # Safety
    /// `size_inout` is a writable `u32`; `data` (when the supplied size is
    /// adequate) is writable for that many bytes, per the D3D9 ABI.
    pub unsafe fn get(&self, guid: &Guid, data: *mut c_void, size_inout: *mut u32) -> i32 {
        // Match the D3D9 GetPrivateData HRESULT ordering:
        // (1) GUID lookup FIRST — an absent key is NOTFOUND and never dereferences
        //     `size_inout` (so an absent key with a NULL size pointer still
        //     returns NOTFOUND, not INVALIDCALL).
        let Some(entry) = self.entries.get(guid) else {
            return D3DERR_NOTFOUND;
        };
        // (2) only an existing entry deref's the size pointer.
        if size_inout.is_null() {
            return D3DERR_INVALIDCALL;
        }
        let ptr_size =
            u32::try_from(core::mem::size_of::<*mut c_void>()).expect("pointer size fits u32");
        let needed = match entry {
            Entry::Blob(bytes) => u32::try_from(bytes.len()).unwrap_or(u32::MAX),
            Entry::Unknown(_) => ptr_size,
        };
        // SAFETY: caller asserts `size_inout` is a readable `u32`.
        let avail = unsafe { *size_inout };
        // (3) report the needed size.
        // SAFETY: caller asserts `size_inout` is writable.
        unsafe { *size_inout = needed };
        // (4) a NULL data buffer is a pure size query — succeeds regardless of the
        //     supplied size (BEFORE the MOREDATA check).
        if data.is_null() {
            return D3D_OK;
        }
        // (5) a non-NULL buffer that is too small is MOREDATA.
        if avail < needed {
            return D3DERR_MOREDATA;
        }
        match entry {
            Entry::Blob(bytes) => {
                // SAFETY: `data` has room for `needed` bytes (checked above) and
                // does not alias `bytes` (caller-owned out buffer).
                unsafe {
                    core::ptr::copy_nonoverlapping(bytes.as_ptr(), data.cast::<u8>(), bytes.len());
                }
            }
            Entry::Unknown(punk) => {
                let punk = *punk;
                if !punk.is_null() {
                    // SAFETY: the stored IUnknown is live; `AddRef` for caller.
                    unsafe { iunknown_add_ref(punk) };
                }
                // SAFETY: `data` is an `IUnknown**` of pointer width here.
                unsafe { *data.cast::<*mut c_void>() = punk };
            }
        }
        D3D_OK
    }

    /// `IDirect3DResource9::FreePrivateData`.
    ///
    /// Releases the entry for `guid`, or `D3DERR_NOTFOUND` if absent.
    pub fn free(&mut self, guid: &Guid) -> i32 {
        self.entries.remove(guid).map_or(D3DERR_NOTFOUND, |entry| {
            entry.release();
            D3D_OK
        })
    }
}

impl Drop for PrivateDataStore {
    fn drop(&mut self) {
        for (_, entry) in self.entries.drain() {
            entry.release();
        }
    }
}
