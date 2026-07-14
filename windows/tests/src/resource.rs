//! RAII wrappers over the D3D9 COM resources a test creates.
//!
//! Each owns one reference and releases it on `Drop`; each borrows the
//! [`Harness`](crate::Harness) for `'h` so a resource can never outlive its
//! device. All `unsafe` vtable dispatch for resources lives here, so test
//! files stay `unsafe`-free.

use core::{ffi::c_void, marker::PhantomData};

use mtld3d_types::{
    D3DINDEXBUFFER_DESC, D3DLOCKED_RECT, D3DSURFACE_DESC, D3DVERTEXBUFFER_DESC,
    IDirect3DIndexBuffer9Vtbl, IDirect3DPixelShader9Vtbl, IDirect3DQuery9Vtbl,
    IDirect3DStateBlock9Vtbl, IDirect3DSurface9Vtbl, IDirect3DTexture9Vtbl,
    IDirect3DVertexBuffer9Vtbl, IDirect3DVertexDeclaration9Vtbl, IDirect3DVertexShader9Vtbl,
};

use crate::{
    check::{expect_created, expect_ok},
    vtbl::deref_vtbl,
};

// ── Texture ──

/// An `IDirect3DTexture9`.
pub struct Texture<'h> {
    ptr: *mut c_void,
    _marker: PhantomData<&'h ()>,
}

impl<'h> Texture<'h> {
    pub const fn from_raw(ptr: *mut c_void) -> Self {
        Self {
            ptr,
            _marker: PhantomData,
        }
    }

    /// The raw COM `this` pointer (for binding via `Harness::set_texture`).
    #[must_use]
    pub const fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    fn vtbl(&self) -> &'static IDirect3DTexture9Vtbl {
        // SAFETY: `self.ptr` is a live texture for the wrapper's lifetime.
        unsafe { deref_vtbl::<IDirect3DTexture9Vtbl>(self.ptr) }
    }

    /// Lock the whole of mip `level`. The returned guard unlocks on drop.
    #[must_use]
    pub fn lock_rect(&self, level: u32, flags: u32) -> LockedRect<'_> {
        self.lock_inner(level, core::ptr::null(), flags)
    }

    /// Lock a sub-rectangle of mip `level`.
    ///
    /// `rect` is a `D3DRECT`-style `[left, top, right, bottom]`.
    #[must_use]
    pub fn lock_rect_partial(&self, level: u32, rect: &[i32; 4], flags: u32) -> LockedRect<'_> {
        self.lock_inner(level, rect.as_ptr().cast::<c_void>(), flags)
    }

    fn lock_inner(&self, level: u32, rect: *const c_void, flags: u32) -> LockedRect<'_> {
        let mut locked = D3DLOCKED_RECT {
            pitch: 0,
            bits: core::ptr::null_mut(),
        };
        // SAFETY: vtable thunk; `self.ptr` is live and `&mut locked` is writable.
        let hr = unsafe { (self.vtbl().lock_rect)(self.ptr, level, &raw mut locked, rect, flags) };
        expect_ok(hr, "Texture LockRect");
        LockedRect {
            owner: LockOwner::Texture {
                this: self.ptr,
                level,
            },
            pitch: locked.pitch,
            bits: locked.bits,
            _marker: PhantomData,
        }
    }

    /// Get mip `level` as a [`Surface`] (`AddRef`'d; released on drop).
    ///
    /// # Panics
    /// Panics if the call fails.
    #[must_use]
    pub fn surface_level(&self, level: u32) -> Surface<'h> {
        let mut surface: *mut c_void = core::ptr::null_mut();
        // SAFETY: vtable thunk; `self.ptr` is live and `&mut surface` is writable.
        let hr = unsafe { (self.vtbl().get_surface_level)(self.ptr, level, &raw mut surface) };
        expect_created(hr, surface, "GetSurfaceLevel");
        Surface::from_raw(surface)
    }

    /// Mip-chain length.
    #[must_use]
    pub fn level_count(&self) -> u32 {
        // SAFETY: vtable thunk; `self.ptr` is live.
        unsafe { (self.vtbl().get_level_count)(self.ptr) }
    }

    /// Describe mip `level`. Returns `(hr, desc)`.
    #[must_use]
    pub fn level_desc(&self, level: u32) -> (i32, D3DSURFACE_DESC) {
        let mut desc = zeroed_surface_desc();
        // SAFETY: vtable thunk; `self.ptr` is live and `&mut desc` is writable.
        let hr = unsafe { (self.vtbl().get_level_desc)(self.ptr, level, &raw mut desc) };
        (hr, desc)
    }

    /// `SetLOD` — returns the previous LOD.
    #[must_use]
    pub fn set_lod(&self, lod: u32) -> u32 {
        // SAFETY: vtable thunk; `self.ptr` is live.
        unsafe { (self.vtbl().set_lod)(self.ptr, lod) }
    }

    /// Current LOD.
    #[must_use]
    pub fn lod(&self) -> u32 {
        // SAFETY: vtable thunk; `self.ptr` is live.
        unsafe { (self.vtbl().get_lod)(self.ptr) }
    }

    /// `SetAutoGenFilterType` — returns the hr.
    #[must_use]
    pub fn set_auto_gen_filter_type(&self, filter: u32) -> i32 {
        // SAFETY: vtable thunk; `self.ptr` is live.
        unsafe { (self.vtbl().set_auto_gen_filter_type)(self.ptr, filter) }
    }

    /// `GetAutoGenFilterType`.
    #[must_use]
    pub fn auto_gen_filter_type(&self) -> u32 {
        // SAFETY: vtable thunk; `self.ptr` is live.
        unsafe { (self.vtbl().get_auto_gen_filter_type)(self.ptr) }
    }

    /// `AddDirtyRect(null)` — flag the whole texture dirty. Returns the hr.
    #[must_use]
    pub fn add_dirty_rect(&self) -> i32 {
        // SAFETY: vtable thunk; `self.ptr` is live, null rect = whole surface.
        unsafe { (self.vtbl().add_dirty_rect)(self.ptr, core::ptr::null()) }
    }

    /// `GetType` (`D3DRTYPE_*`).
    #[must_use]
    pub fn resource_type(&self) -> u32 {
        // SAFETY: vtable thunk; `self.ptr` is live.
        unsafe { (self.vtbl().get_type)(self.ptr) }
    }

    /// `SetPrivateData` — store a small blob under a test GUID. Returns the hr.
    #[must_use]
    pub fn set_private_data_hr(&self) -> i32 {
        let guid = mtld3d_types::Guid {
            data1: 1,
            data2: 2,
            data3: 3,
            data4: [4; 8],
        };
        let data = [0u8; 4];
        // SAFETY: vtable thunk; `&guid` and `data` are read-only for the call.
        unsafe {
            (self.vtbl().set_private_data)(
                self.ptr,
                &raw const guid,
                data.as_ptr().cast::<c_void>(),
                4,
                0,
            )
        }
    }

    /// `GetDevice` (a documented stub today). Returns the hr.
    #[must_use]
    pub fn get_device_hr(&self) -> i32 {
        let mut out: *mut c_void = core::ptr::null_mut();
        // SAFETY: vtable thunk; `&mut out` is writable.
        unsafe { (self.vtbl().get_device)(self.ptr, &raw mut out) }
    }

    /// `PreLoad` — a no-op that must not crash.
    pub fn pre_load(&self) {
        // SAFETY: vtable thunk; `self.ptr` is live.
        unsafe { (self.vtbl().pre_load)(self.ptr) };
    }

    /// `SetPriority` — returns the previous priority.
    #[must_use]
    pub fn set_priority(&self, priority: u32) -> u32 {
        // SAFETY: vtable thunk; `self.ptr` is live.
        unsafe { (self.vtbl().set_priority)(self.ptr, priority) }
    }

    /// `GetPriority`.
    #[must_use]
    pub fn priority(&self) -> u32 {
        // SAFETY: vtable thunk; `self.ptr` is live.
        unsafe { (self.vtbl().get_priority)(self.ptr) }
    }
}

impl Drop for Texture<'_> {
    fn drop(&mut self) {
        // SAFETY: vtable thunk; `self.ptr` is live and this is its last use.
        unsafe { (self.vtbl().release)(self.ptr) };
    }
}

// ── Surface ──

/// An `IDirect3DSurface9`.
pub struct Surface<'h> {
    ptr: *mut c_void,
    _marker: PhantomData<&'h ()>,
}

impl Surface<'_> {
    pub const fn from_raw(ptr: *mut c_void) -> Self {
        Self {
            ptr,
            _marker: PhantomData,
        }
    }

    /// The raw COM `this` pointer (for `SetRenderTarget` etc.).
    #[must_use]
    pub const fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    fn vtbl(&self) -> &'static IDirect3DSurface9Vtbl {
        // SAFETY: `self.ptr` is a live surface for the wrapper's lifetime.
        unsafe { deref_vtbl::<IDirect3DSurface9Vtbl>(self.ptr) }
    }

    /// Lock the whole surface. The returned guard unlocks on drop.
    #[must_use]
    pub fn lock_rect(&self, flags: u32) -> LockedRect<'_> {
        let mut locked = D3DLOCKED_RECT {
            pitch: 0,
            bits: core::ptr::null_mut(),
        };
        // SAFETY: vtable thunk; `self.ptr` is live and `&mut locked` is writable.
        let hr =
            unsafe { (self.vtbl().lock_rect)(self.ptr, &raw mut locked, core::ptr::null(), flags) };
        expect_ok(hr, "Surface LockRect");
        LockedRect {
            owner: LockOwner::Surface { this: self.ptr },
            pitch: locked.pitch,
            bits: locked.bits,
            _marker: PhantomData,
        }
    }

    /// Describe the surface. Returns `(hr, desc)`.
    #[must_use]
    pub fn desc(&self) -> (i32, D3DSURFACE_DESC) {
        let mut desc = zeroed_surface_desc();
        // SAFETY: vtable thunk; `self.ptr` is live and `&mut desc` is writable.
        let hr = unsafe { (self.vtbl().get_desc)(self.ptr, &raw mut desc) };
        (hr, desc)
    }

    /// Call `GetDC` with the out slot pre-seeded to `sentinel`.
    ///
    /// Returns `(hr, out)`: on a rejected call the out slot must be left
    /// untouched, so `out == sentinel` proves the implementation did not write
    /// through it.
    #[must_use]
    pub fn get_dc(&self, sentinel: *mut c_void) -> (i32, *mut c_void) {
        let mut out = sentinel;
        // SAFETY: vtable thunk; `self.ptr` is live and `&mut out` is writable.
        let hr = unsafe { (self.vtbl().get_dc)(self.ptr, &raw mut out) };
        (hr, out)
    }
}

impl Drop for Surface<'_> {
    fn drop(&mut self) {
        // SAFETY: vtable thunk; `self.ptr` is live and this is its last use.
        unsafe { (self.vtbl().release)(self.ptr) };
    }
}

// ── Vertex / index buffers ──

/// An `IDirect3DVertexBuffer9`.
pub struct VertexBuffer<'h> {
    ptr: *mut c_void,
    _marker: PhantomData<&'h ()>,
}

impl VertexBuffer<'_> {
    pub const fn from_raw(ptr: *mut c_void) -> Self {
        Self {
            ptr,
            _marker: PhantomData,
        }
    }

    /// The raw COM `this` pointer (for `SetStreamSource`).
    #[must_use]
    pub const fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    fn vtbl(&self) -> &'static IDirect3DVertexBuffer9Vtbl {
        // SAFETY: `self.ptr` is a live vertex buffer for the wrapper's lifetime.
        unsafe { deref_vtbl::<IDirect3DVertexBuffer9Vtbl>(self.ptr) }
    }

    /// Lock `[offset, offset+size)` bytes (`size == 0` locks the whole buffer).
    ///
    /// The returned guard unlocks on drop.
    #[must_use]
    pub fn lock(&self, offset: u32, size: u32, flags: u32) -> BufferLock<'_> {
        let mut bits: *mut c_void = core::ptr::null_mut();
        // SAFETY: vtable thunk; `self.ptr` is live and `&mut bits` is writable.
        let hr = unsafe { (self.vtbl().lock)(self.ptr, offset, size, &raw mut bits, flags) };
        expect_ok(hr, "VertexBuffer Lock");
        // SAFETY: the unlock thunk has a stable ABI; copied out so the guard
        // need not reborrow the vtable.
        let unlock = self.vtbl().unlock;
        BufferLock {
            this: self.ptr,
            bits,
            unlock,
            _marker: PhantomData,
        }
    }

    /// `SetPriority` — returns the previous priority.
    #[must_use]
    pub fn set_priority(&self, priority: u32) -> u32 {
        // SAFETY: vtable thunk; `self.ptr` is live.
        unsafe { (self.vtbl().set_priority)(self.ptr, priority) }
    }

    /// `GetPriority`.
    #[must_use]
    pub fn priority(&self) -> u32 {
        // SAFETY: vtable thunk; `self.ptr` is live.
        unsafe { (self.vtbl().get_priority)(self.ptr) }
    }

    /// Describe the buffer. Returns `(hr, desc)`.
    #[must_use]
    pub fn desc(&self) -> (i32, D3DVERTEXBUFFER_DESC) {
        let mut desc = D3DVERTEXBUFFER_DESC {
            format: 0,
            resource_type: 0,
            usage: 0,
            pool: 0,
            size: 0,
            fvf: 0,
        };
        // SAFETY: vtable thunk; `self.ptr` is live and `&mut desc` is writable.
        let hr = unsafe { (self.vtbl().get_desc)(self.ptr, &raw mut desc) };
        (hr, desc)
    }
}

impl Drop for VertexBuffer<'_> {
    fn drop(&mut self) {
        // SAFETY: vtable thunk; `self.ptr` is live and this is its last use.
        unsafe { (self.vtbl().release)(self.ptr) };
    }
}

/// An `IDirect3DIndexBuffer9`.
pub struct IndexBuffer<'h> {
    ptr: *mut c_void,
    _marker: PhantomData<&'h ()>,
}

impl IndexBuffer<'_> {
    pub const fn from_raw(ptr: *mut c_void) -> Self {
        Self {
            ptr,
            _marker: PhantomData,
        }
    }

    /// The raw COM `this` pointer (for `SetIndices`).
    #[must_use]
    pub const fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    fn vtbl(&self) -> &'static IDirect3DIndexBuffer9Vtbl {
        // SAFETY: `self.ptr` is a live index buffer for the wrapper's lifetime.
        unsafe { deref_vtbl::<IDirect3DIndexBuffer9Vtbl>(self.ptr) }
    }

    /// Lock `[offset, offset+size)` bytes (`size == 0` locks the whole buffer).
    #[must_use]
    pub fn lock(&self, offset: u32, size: u32, flags: u32) -> BufferLock<'_> {
        let mut bits: *mut c_void = core::ptr::null_mut();
        // SAFETY: vtable thunk; `self.ptr` is live and `&mut bits` is writable.
        let hr = unsafe { (self.vtbl().lock)(self.ptr, offset, size, &raw mut bits, flags) };
        expect_ok(hr, "IndexBuffer Lock");
        // SAFETY: the unlock thunk has a stable ABI; copied out for the guard.
        let unlock = self.vtbl().unlock;
        BufferLock {
            this: self.ptr,
            bits,
            unlock,
            _marker: PhantomData,
        }
    }

    /// Describe the buffer. Returns `(hr, desc)`.
    #[must_use]
    pub fn desc(&self) -> (i32, D3DINDEXBUFFER_DESC) {
        let mut desc = D3DINDEXBUFFER_DESC {
            format: 0,
            resource_type: 0,
            usage: 0,
            pool: 0,
            size: 0,
        };
        // SAFETY: vtable thunk; `self.ptr` is live and `&mut desc` is writable.
        let hr = unsafe { (self.vtbl().get_desc)(self.ptr, &raw mut desc) };
        (hr, desc)
    }
}

impl Drop for IndexBuffer<'_> {
    fn drop(&mut self) {
        // SAFETY: vtable thunk; `self.ptr` is live and this is its last use.
        unsafe { (self.vtbl().release)(self.ptr) };
    }
}

// ── Shaders ──

/// An `IDirect3DVertexShader9`.
pub struct VertexShader<'h> {
    ptr: *mut c_void,
    _marker: PhantomData<&'h ()>,
}

impl VertexShader<'_> {
    pub const fn from_raw(ptr: *mut c_void) -> Self {
        Self {
            ptr,
            _marker: PhantomData,
        }
    }

    /// The raw COM `this` pointer (for `SetVertexShader`).
    #[must_use]
    pub const fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }
}

impl Drop for VertexShader<'_> {
    fn drop(&mut self) {
        // SAFETY: `self.ptr` is a live vertex shader; this is its last use.
        let vtbl = unsafe { deref_vtbl::<IDirect3DVertexShader9Vtbl>(self.ptr) };
        // SAFETY: vtable thunk; `self.ptr` is the matching live shader.
        unsafe { (vtbl.release)(self.ptr) };
    }
}

/// An `IDirect3DPixelShader9`.
pub struct PixelShader<'h> {
    ptr: *mut c_void,
    _marker: PhantomData<&'h ()>,
}

impl PixelShader<'_> {
    pub const fn from_raw(ptr: *mut c_void) -> Self {
        Self {
            ptr,
            _marker: PhantomData,
        }
    }

    /// The raw COM `this` pointer (for `SetPixelShader`).
    #[must_use]
    pub const fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }
}

impl Drop for PixelShader<'_> {
    fn drop(&mut self) {
        // SAFETY: `self.ptr` is a live pixel shader; this is its last use.
        let vtbl = unsafe { deref_vtbl::<IDirect3DPixelShader9Vtbl>(self.ptr) };
        // SAFETY: vtable thunk; `self.ptr` is the matching live shader.
        unsafe { (vtbl.release)(self.ptr) };
    }
}

// ── State block ──

/// An `IDirect3DStateBlock9`.
pub struct StateBlock<'h> {
    ptr: *mut c_void,
    _marker: PhantomData<&'h ()>,
}

impl StateBlock<'_> {
    pub const fn from_raw(ptr: *mut c_void) -> Self {
        Self {
            ptr,
            _marker: PhantomData,
        }
    }

    fn vtbl(&self) -> &'static IDirect3DStateBlock9Vtbl {
        // SAFETY: `self.ptr` is a live state block for the wrapper's lifetime.
        unsafe { deref_vtbl::<IDirect3DStateBlock9Vtbl>(self.ptr) }
    }

    /// Re-snapshot the device's current state into this block. Returns the hr.
    #[must_use]
    pub fn capture(&self) -> i32 {
        // SAFETY: vtable thunk; `self.ptr` is live.
        unsafe { (self.vtbl().capture)(self.ptr) }
    }

    /// Replay the captured state onto the device. Returns the hr.
    #[must_use]
    pub fn apply(&self) -> i32 {
        // SAFETY: vtable thunk; `self.ptr` is live.
        unsafe { (self.vtbl().apply)(self.ptr) }
    }
}

impl Drop for StateBlock<'_> {
    fn drop(&mut self) {
        // SAFETY: vtable thunk; `self.ptr` is live and this is its last use.
        unsafe { (self.vtbl().release)(self.ptr) };
    }
}

// ── Query ──

/// An `IDirect3DQuery9`.
pub struct Query<'h> {
    ptr: *mut c_void,
    _marker: PhantomData<&'h ()>,
}

impl Query<'_> {
    pub const fn from_raw(ptr: *mut c_void) -> Self {
        Self {
            ptr,
            _marker: PhantomData,
        }
    }

    fn vtbl(&self) -> &'static IDirect3DQuery9Vtbl {
        // SAFETY: `self.ptr` is a live query for the wrapper's lifetime.
        unsafe { deref_vtbl::<IDirect3DQuery9Vtbl>(self.ptr) }
    }

    /// `GetType` (`D3DQUERYTYPE_*`).
    #[must_use]
    pub fn query_type(&self) -> u32 {
        // SAFETY: vtable thunk; `self.ptr` is live.
        unsafe { (self.vtbl().get_type)(self.ptr) }
    }

    /// Byte size of the result `GetData` writes.
    #[must_use]
    pub fn data_size(&self) -> u32 {
        // SAFETY: vtable thunk; `self.ptr` is live.
        unsafe { (self.vtbl().get_data_size)(self.ptr) }
    }

    /// `Issue` (`D3DISSUE_END` / `D3DISSUE_BEGIN`). Returns the hr.
    #[must_use]
    pub fn issue(&self, flags: u32) -> i32 {
        // SAFETY: vtable thunk; `self.ptr` is live.
        unsafe { (self.vtbl().issue)(self.ptr, flags) }
    }

    /// Read a 4-byte result. Returns `(hr, value)`.
    #[must_use]
    pub fn data_u32(&self, flags: u32) -> (i32, u32) {
        let mut value = 0u32;
        // SAFETY: vtable thunk; `&mut value` covers the 4-byte EVENT/OCCLUSION result.
        let hr = unsafe {
            (self.vtbl().get_data)(self.ptr, (&raw mut value).cast::<c_void>(), 4, flags)
        };
        (hr, value)
    }
}

impl Drop for Query<'_> {
    fn drop(&mut self) {
        // SAFETY: vtable thunk; `self.ptr` is live and this is its last use.
        unsafe { (self.vtbl().release)(self.ptr) };
    }
}

// ── Vertex declaration ──

/// An `IDirect3DVertexDeclaration9`.
pub struct VertexDeclaration<'h> {
    ptr: *mut c_void,
    _marker: PhantomData<&'h ()>,
}

impl VertexDeclaration<'_> {
    pub const fn from_raw(ptr: *mut c_void) -> Self {
        Self {
            ptr,
            _marker: PhantomData,
        }
    }

    /// The raw COM `this` pointer (for `SetVertexDeclaration`).
    #[must_use]
    pub const fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }
}

impl Drop for VertexDeclaration<'_> {
    fn drop(&mut self) {
        // SAFETY: `self.ptr` is a live vertex declaration; this is its last use.
        let vtbl = unsafe { deref_vtbl::<IDirect3DVertexDeclaration9Vtbl>(self.ptr) };
        // SAFETY: vtable thunk; `self.ptr` is the matching live declaration.
        unsafe { (vtbl.release)(self.ptr) };
    }
}

// ── Lock guards ──

enum LockOwner {
    Texture { this: *mut c_void, level: u32 },
    Surface { this: *mut c_void },
}

/// A held texture/surface lock. Exposes the mapped span and unlocks on drop.
pub struct LockedRect<'a> {
    owner: LockOwner,
    pitch: i32,
    bits: *mut c_void,
    _marker: PhantomData<&'a ()>,
}

impl LockedRect<'_> {
    /// Row pitch in bytes.
    #[must_use]
    pub const fn pitch(&self) -> i32 {
        self.pitch
    }

    /// Raw pointer to the mapped span.
    ///
    /// For tests that fill multiple rows honouring [`Self::pitch`] (the row
    /// stride may exceed `width * bpp`).
    #[must_use]
    pub const fn bits_ptr(&self) -> *mut u8 {
        self.bits.cast::<u8>()
    }

    /// Copy `data` into the mapped span as contiguous `u32` pixels.
    ///
    /// # Panics
    /// The caller must ensure `data` fits within the locked region.
    pub const fn write_u32(&mut self, data: &[u32]) {
        // SAFETY: `bits` maps at least `data.len()` u32s of the locked region
        // (caller's contract); the &mut borrow makes the write exclusive.
        unsafe {
            core::ptr::copy_nonoverlapping(data.as_ptr(), self.bits.cast::<u32>(), data.len());
        }
    }

    /// Copy `data` into the mapped span at offset 0 — for sub-32-bit and compressed formats.
    ///
    /// `data` is any `Copy` POD: `u8`/`u16`/`u32` pixels or block bytes.
    ///
    /// # Panics
    /// The caller must ensure `data` fits within the locked region.
    pub const fn write<T: Copy>(&mut self, data: &[T]) {
        // SAFETY: `bits` maps at least `size_of_val(data)` bytes of the locked
        // region (caller's contract); the &mut borrow makes the write exclusive.
        unsafe {
            core::ptr::copy_nonoverlapping(data.as_ptr(), self.bits.cast::<T>(), data.len());
        }
    }

    /// View the first `count` `u32` pixels of the mapped span.
    #[must_use]
    pub const fn as_u32(&self, count: usize) -> &[u32] {
        // SAFETY: `bits` is valid for `count` u32s within the locked region
        // (caller's contract) and lives until this guard drops.
        unsafe { core::slice::from_raw_parts(self.bits.cast::<u32>(), count) }
    }
}

impl Drop for LockedRect<'_> {
    fn drop(&mut self) {
        match self.owner {
            LockOwner::Texture { this, level } => {
                // SAFETY: `this` is the live texture this guard locked.
                let vtbl = unsafe { deref_vtbl::<IDirect3DTexture9Vtbl>(this) };
                // SAFETY: vtable thunk; `this` is the matching live texture.
                unsafe { (vtbl.unlock_rect)(this, level) };
            }
            LockOwner::Surface { this } => {
                // SAFETY: `this` is the live surface this guard locked.
                let vtbl = unsafe { deref_vtbl::<IDirect3DSurface9Vtbl>(this) };
                // SAFETY: vtable thunk; `this` is the matching live surface.
                unsafe { (vtbl.unlock_rect)(this) };
            }
        }
    }
}

/// A held vertex/index-buffer lock. Exposes the mapped span and unlocks on drop.
pub struct BufferLock<'a> {
    this: *mut c_void,
    bits: *mut c_void,
    unlock: unsafe extern "system" fn(*mut c_void) -> i32,
    _marker: PhantomData<&'a ()>,
}

impl BufferLock<'_> {
    /// Copy `data` (any `Copy` POD) into the mapped span at byte offset 0.
    ///
    /// # Panics
    /// The caller must ensure `data` fits within the locked region.
    pub const fn write<T: Copy>(&mut self, data: &[T]) {
        // SAFETY: `bits` maps at least `size_of_val(data)` bytes of the locked
        // region (caller's contract); the &mut borrow makes the write exclusive.
        unsafe {
            core::ptr::copy_nonoverlapping(data.as_ptr(), self.bits.cast::<T>(), data.len());
        }
    }
}

impl Drop for BufferLock<'_> {
    fn drop(&mut self) {
        // SAFETY: `self.unlock` is this buffer's unlock thunk and `self.this` is
        // the live buffer it came from.
        unsafe { (self.unlock)(self.this) };
    }
}

const fn zeroed_surface_desc() -> D3DSURFACE_DESC {
    D3DSURFACE_DESC {
        format: 0,
        resource_type: 0,
        usage: 0,
        pool: 0,
        multi_sample_type: 0,
        multi_sample_quality: 0,
        width: 0,
        height: 0,
    }
}
