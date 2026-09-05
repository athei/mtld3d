//! Cross-thread handles for a `D3DCREATE_MULTITHREADED` device.
//!
//! D3D9 promises that a device created with the flag, and every object it
//! creates, may be called from any thread. [`Harness::shared`] hands out a
//! [`SharedDevice`] only for such a device, so the safe dispatch methods here
//! are sound by that promise, and the borrow of the harness keeps the device
//! alive for as long as a handle exists. Pointers travel as `usize`, so the
//! handles are `Send + Sync` without an `unsafe impl`, and `std::thread::scope`
//! is the way to use them: it joins every worker before the harness can drop.

use core::{ffi::c_void, marker::PhantomData};

use mtld3d_types::{
    D3D_OK, D3DCREATE_MULTITHREADED, IDirect3DDevice9Vtbl, IDirect3DQuery9Vtbl,
    IDirect3DVertexBuffer9Vtbl,
};

use crate::{
    harness::Harness,
    resource::{Query, VertexBuffer},
    vtbl::deref_vtbl,
};

/// A device handle any thread may call.
pub struct SharedDevice<'h> {
    device: usize,
    _marker: PhantomData<&'h ()>,
}

impl Harness {
    /// A handle to this device for another thread.
    ///
    /// # Panics
    /// Panics unless the device was created with `D3DCREATE_MULTITHREADED`:
    /// without the flag D3D9 leaves a call from a second thread undefined.
    #[must_use]
    pub fn shared(&self) -> SharedDevice<'_> {
        assert!(
            self.behavior_flags() & D3DCREATE_MULTITHREADED != 0,
            "only a D3DCREATE_MULTITHREADED device may be shared between threads"
        );
        SharedDevice {
            device: self.device() as usize,
            _marker: PhantomData,
        }
    }
}

impl SharedDevice<'_> {
    const fn device(&self) -> *mut c_void {
        self.device as *mut c_void
    }

    fn vtbl(&self) -> &'static IDirect3DDevice9Vtbl {
        // SAFETY: the device is live for the harness borrow this handle carries.
        unsafe { deref_vtbl::<IDirect3DDevice9Vtbl>(self.device()) }
    }

    /// `SetRenderState`.
    #[must_use]
    pub fn set_render_state(&self, state: u32, value: u32) -> i32 {
        // SAFETY: vtable thunk; the device is live and callable from any thread.
        unsafe { (self.vtbl().set_render_state)(self.device(), state, value) }
    }

    /// `Present` to the whole backbuffer.
    #[must_use]
    pub fn present(&self) -> i32 {
        // SAFETY: vtable thunk; all-null args present the entire backbuffer.
        unsafe {
            (self.vtbl().present)(
                self.device(),
                core::ptr::null(),
                core::ptr::null(),
                core::ptr::null_mut(),
                core::ptr::null(),
            )
        }
    }

    /// A handle to `vb` for another thread; the buffer outlives the handle.
    #[must_use]
    pub fn share_vertex_buffer<'a>(&self, vb: &'a VertexBuffer<'_>) -> SharedVertexBuffer<'a> {
        SharedVertexBuffer {
            vb: vb.as_ptr() as usize,
            _marker: PhantomData,
        }
    }

    /// A handle to `query` for another thread; the query outlives the handle.
    #[must_use]
    pub fn share_query<'a>(&self, query: &'a Query<'_>) -> SharedQuery<'a> {
        SharedQuery {
            query: query.as_ptr() as usize,
            _marker: PhantomData,
        }
    }
}

/// A vertex-buffer handle any thread may call.
pub struct SharedVertexBuffer<'a> {
    vb: usize,
    _marker: PhantomData<&'a ()>,
}

impl SharedVertexBuffer<'_> {
    const fn vb(&self) -> *mut c_void {
        self.vb as *mut c_void
    }

    fn vtbl(&self) -> &'static IDirect3DVertexBuffer9Vtbl {
        // SAFETY: the buffer is live for the borrow this handle carries.
        unsafe { deref_vtbl::<IDirect3DVertexBuffer9Vtbl>(self.vb()) }
    }

    /// Lock the whole buffer with `flags`, write `words` from offset 0, unlock.
    ///
    /// Returns the first failing `HRESULT`, or `D3D_OK` once both calls
    /// succeeded. The buffer must hold at least `words`.
    #[must_use]
    pub fn fill_u32(&self, words: &[u32], flags: u32) -> i32 {
        let mut bits: *mut c_void = core::ptr::null_mut();
        // SAFETY: vtable thunk; the buffer is live and `&mut bits` is writable.
        let hr = unsafe { (self.vtbl().lock)(self.vb(), 0, 0, &raw mut bits, flags) };
        if hr != D3D_OK {
            return hr;
        }
        // SAFETY: a whole-buffer lock maps at least `words` (the caller's
        // contract) and the mapping is ours until the unlock below.
        unsafe { core::ptr::copy_nonoverlapping(words.as_ptr(), bits.cast::<u32>(), words.len()) };
        // SAFETY: vtable thunk; balances the lock above.
        unsafe { (self.vtbl().unlock)(self.vb()) }
    }
}

/// A query handle any thread may call.
pub struct SharedQuery<'a> {
    query: usize,
    _marker: PhantomData<&'a ()>,
}

impl SharedQuery<'_> {
    const fn query(&self) -> *mut c_void {
        self.query as *mut c_void
    }

    fn vtbl(&self) -> &'static IDirect3DQuery9Vtbl {
        // SAFETY: the query is live for the borrow this handle carries.
        unsafe { deref_vtbl::<IDirect3DQuery9Vtbl>(self.query()) }
    }

    /// `Issue` (`D3DISSUE_END` / `D3DISSUE_BEGIN`). Returns the hr.
    #[must_use]
    pub fn issue(&self, flags: u32) -> i32 {
        // SAFETY: vtable thunk; the query is live.
        unsafe { (self.vtbl().issue)(self.query(), flags) }
    }

    /// Read a 4-byte result. Returns `(hr, value)`.
    #[must_use]
    pub fn data_u32(&self, flags: u32) -> (i32, u32) {
        let mut value = 0u32;
        // SAFETY: vtable thunk; `&mut value` covers the 4-byte result.
        let hr = unsafe {
            (self.vtbl().get_data)(self.query(), (&raw mut value).cast::<c_void>(), 4, flags)
        };
        (hr, value)
    }
}
