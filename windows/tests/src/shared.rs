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
    D3D_OK, D3DCLEAR_TARGET, D3DCREATE_MULTITHREADED, D3DFMT_A8R8G8B8, D3DLOCK_READONLY,
    D3DLOCKED_RECT, D3DPOOL_SYSTEMMEM, D3DSURFACE_DESC, IDirect3DDevice9Vtbl, IDirect3DQuery9Vtbl,
    IDirect3DSurface9Vtbl, IDirect3DVertexBuffer9Vtbl,
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

    /// `Clear(D3DCLEAR_TARGET)` of the whole target to `color`.
    #[must_use]
    pub fn clear_target(&self, color: u32) -> i32 {
        // SAFETY: vtable thunk; a zero count with a null rect array clears the
        // whole target.
        unsafe {
            (self.vtbl().clear)(
                self.device(),
                0,
                core::ptr::null(),
                D3DCLEAR_TARGET,
                color,
                1.0,
                0,
            )
        }
    }

    /// `BeginScene`.
    #[must_use]
    pub fn begin_scene(&self) -> i32 {
        // SAFETY: vtable thunk; the device is live and callable from any thread.
        unsafe { (self.vtbl().begin_scene)(self.device()) }
    }

    /// `EndScene`.
    #[must_use]
    pub fn end_scene(&self) -> i32 {
        // SAFETY: vtable thunk; the device is live and callable from any thread.
        unsafe { (self.vtbl().end_scene)(self.device()) }
    }

    /// `DrawPrimitiveUP` from a slice of vertices matching the bound FVF.
    ///
    /// The stride is `V`'s size, so the caller's vertex type has to be the
    /// layout the FVF names.
    ///
    /// # Panics
    /// Panics if `V` is larger than a `u32` stride can name.
    #[must_use]
    pub fn draw_primitive_up<V>(&self, prim: u32, prim_count: u32, verts: &[V]) -> i32 {
        let stride = u32::try_from(core::mem::size_of::<V>()).expect("vertex stride fits u32");
        // SAFETY: vtable thunk; `verts` is read-only for the call and the
        // device is callable from any thread.
        unsafe {
            (self.vtbl().draw_primitive_up)(
                self.device(),
                prim,
                prim_count,
                verts.as_ptr().cast::<c_void>(),
                stride,
            )
        }
    }

    /// Read a backbuffer pixel as `0xAARRGGBB` through the D3D9 read-back chain.
    ///
    /// The chain `Harness::read_pixel` runs, callable from a worker thread:
    /// `GetRenderTarget(0)`, `CreateOffscreenPlainSurface(D3DPOOL_SYSTEMMEM)`,
    /// `GetRenderTargetData`, `LockRect(READONLY)`. Both surfaces are released
    /// before the return. A failing call comes back as `Err((name, hr))` so a
    /// worker can report it instead of panicking on a thread the test then
    /// has to join.
    ///
    /// # Errors
    /// The first call of the chain that fails, by name, with its hr.
    pub fn read_pixel(&self, x: u32, y: u32) -> Result<u32, (&'static str, i32)> {
        let mut rt: *mut c_void = core::ptr::null_mut();
        // SAFETY: vtable thunk; `&mut rt` is writable.
        let hr = unsafe { (self.vtbl().get_render_target)(self.device(), 0, &raw mut rt) };
        if hr < 0 || rt.is_null() {
            return Err(("GetRenderTarget", hr));
        }
        let rt = OwnedSurface(rt);
        let mut desc = D3DSURFACE_DESC {
            format: 0,
            resource_type: 0,
            usage: 0,
            pool: 0,
            multi_sample_type: 0,
            multi_sample_quality: 0,
            width: 0,
            height: 0,
        };
        // SAFETY: vtable thunk; the surface is live and `&mut desc` is writable.
        let hr = unsafe { (rt.vtbl().get_desc)(rt.0, &raw mut desc) };
        if hr < 0 {
            return Err(("GetDesc", hr));
        }
        let mut sysmem: *mut c_void = core::ptr::null_mut();
        // SAFETY: vtable thunk; `&mut sysmem` is writable, a null shared handle
        // is permitted.
        let hr = unsafe {
            (self.vtbl().create_offscreen_plain_surface)(
                self.device(),
                desc.width,
                desc.height,
                D3DFMT_A8R8G8B8,
                D3DPOOL_SYSTEMMEM,
                &raw mut sysmem,
                core::ptr::null_mut(),
            )
        };
        if hr < 0 || sysmem.is_null() {
            return Err(("CreateOffscreenPlainSurface", hr));
        }
        let sysmem = OwnedSurface(sysmem);
        // SAFETY: vtable thunk; both surfaces are live.
        let hr = unsafe { (self.vtbl().get_render_target_data)(self.device(), rt.0, sysmem.0) };
        if hr < 0 {
            return Err(("GetRenderTargetData", hr));
        }
        let mut locked = D3DLOCKED_RECT {
            pitch: 0,
            bits: core::ptr::null_mut(),
        };
        // SAFETY: vtable thunk; the surface is live, `&mut locked` is writable,
        // a null rect locks the whole surface.
        let hr = unsafe {
            (sysmem.vtbl().lock_rect)(
                sysmem.0,
                &raw mut locked,
                core::ptr::null(),
                D3DLOCK_READONLY,
            )
        };
        if hr < 0 || locked.bits.is_null() {
            return Err(("LockRect", hr));
        }
        let pitch = usize::try_from(locked.pitch).map_err(|_| ("LockRect pitch", hr))?;
        let offset = y as usize * pitch + x as usize * 4;
        // SAFETY: the lock covers `desc.height` rows of `pitch` bytes and
        // `(x, y)` is inside the surface the caller asked about, so the offset
        // stays within the mapping.
        let pixel_ptr = unsafe { locked.bits.cast::<u8>().add(offset) };
        // SAFETY: the pointer is inside the locked mapping, four bytes of it
        // remain, and the read happens before the unlock below.
        let pixel = unsafe { pixel_ptr.cast::<u32>().read_unaligned() };
        // SAFETY: vtable thunk; the surface is locked and live.
        let hr = unsafe { (sysmem.vtbl().unlock_rect)(sysmem.0) };
        if hr < 0 {
            return Err(("UnlockRect", hr));
        }
        Ok(pixel)
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

/// A surface reference released when it goes out of scope.
///
/// `read_pixel` takes two of them and returns early on any failing call, so
/// the release rides a drop rather than every exit.
struct OwnedSurface(*mut c_void);

impl OwnedSurface {
    fn vtbl(&self) -> &'static IDirect3DSurface9Vtbl {
        // SAFETY: the surface is live until the drop below releases it.
        unsafe { deref_vtbl::<IDirect3DSurface9Vtbl>(self.0) }
    }
}

impl Drop for OwnedSurface {
    fn drop(&mut self) {
        // SAFETY: vtable thunk; this is the one reference the chain holds.
        unsafe { (self.vtbl().release)(self.0) };
    }
}
