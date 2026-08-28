//! RAII wrappers over the D3D9 COM resources a test creates.
//!
//! Each owns one reference and releases it on `Drop`; each borrows the
//! [`Harness`](crate::Harness) for `'h` so a resource can never outlive its
//! device. All `unsafe` vtable dispatch for resources lives here, so test
//! files stay `unsafe`-free.

use core::{ffi::c_void, marker::PhantomData};

use mtld3d_types::{
    D3DINDEXBUFFER_DESC, D3DLOCKED_BOX, D3DLOCKED_RECT, D3DSURFACE_DESC, D3DVERTEXBUFFER_DESC,
    D3DVOLUME_DESC, Guid, IDirect3DCubeTexture9Vtbl, IDirect3DIndexBuffer9Vtbl,
    IDirect3DPixelShader9Vtbl, IDirect3DQuery9Vtbl, IDirect3DStateBlock9Vtbl,
    IDirect3DSurface9Vtbl, IDirect3DTexture9Vtbl, IDirect3DVertexBuffer9Vtbl,
    IDirect3DVertexDeclaration9Vtbl, IDirect3DVertexShader9Vtbl, IDirect3DVolume9Vtbl,
    IDirect3DVolumeTexture9Vtbl,
};

use crate::{
    check::{expect_created, expect_ok},
    vtbl::deref_vtbl,
};

// ── Private data ──

/// `SetPrivateData(guid, blob, len, 0)` through a resource's own thunk.
fn set_private_data(
    set: unsafe extern "system" fn(*mut c_void, *const Guid, *const c_void, u32, u32) -> i32,
    this: *mut c_void,
    guid: &Guid,
    blob: &[u8],
) -> i32 {
    let len = u32::try_from(blob.len()).expect("blob fits u32");
    // SAFETY: vtable thunk; `blob` is readable for `len`.
    unsafe {
        set(
            this,
            &raw const *guid,
            blob.as_ptr().cast::<c_void>(),
            len,
            0,
        )
    }
}

/// `GetPrivateData(guid, out, &mut size)`; a null `out` asks for the size alone.
fn get_private_data(
    get: unsafe extern "system" fn(*mut c_void, *const Guid, *mut c_void, *mut u32) -> i32,
    this: *mut c_void,
    guid: &Guid,
    out: Option<&mut [u8]>,
) -> (i32, u32) {
    let (ptr, mut size) = out.map_or((core::ptr::null_mut(), 0), |b| {
        let len = u32::try_from(b.len()).expect("buffer fits u32");
        (b.as_mut_ptr().cast::<c_void>(), len)
    });
    // SAFETY: vtable thunk; `ptr` is null or writable for `size` bytes.
    let hr = unsafe { get(this, &raw const *guid, ptr, &raw mut size) };
    (hr, size)
}

// ── Volume texture ──

/// An `IDirect3DVolumeTexture9`.
pub struct VolumeTexture<'h> {
    ptr: *mut c_void,
    _marker: PhantomData<&'h ()>,
}

impl VolumeTexture<'_> {
    pub const fn from_raw(ptr: *mut c_void) -> Self {
        Self {
            ptr,
            _marker: PhantomData,
        }
    }

    /// `GetDevice`, returning the hr and the device it wrote.
    ///
    /// The pointer carries a reference of its own; the caller releases it.
    #[must_use]
    pub fn get_device(&self) -> (i32, *mut c_void) {
        let mut out: *mut c_void = core::ptr::null_mut();
        // SAFETY: vtable thunk; `&mut out` is writable.
        let hr = unsafe { (self.vtbl().get_device)(self.ptr, &raw mut out) };
        (hr, out)
    }

    fn vtbl(&self) -> &'static IDirect3DVolumeTexture9Vtbl {
        // SAFETY: `self.ptr` is a live volume texture for the wrapper's lifetime.
        unsafe { deref_vtbl::<IDirect3DVolumeTexture9Vtbl>(self.ptr) }
    }

    #[must_use]
    pub const fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// Mip-chain length.
    #[must_use]
    pub fn level_count(&self) -> u32 {
        // SAFETY: vtable thunk; `self.ptr` is live.
        unsafe { (self.vtbl().get_level_count)(self.ptr) }
    }

    /// Describe mip `level`. Returns `(hr, desc)`.
    #[must_use]
    pub fn level_desc(&self, level: u32) -> (i32, D3DVOLUME_DESC) {
        let mut desc = D3DVOLUME_DESC {
            format: 0,
            resource_type: 0,
            usage: 0,
            pool: 0,
            width: 0,
            height: 0,
            depth: 0,
        };
        // SAFETY: vtable thunk; `self.ptr` is live and `&mut desc` is writable.
        let hr = unsafe {
            (self.vtbl().get_level_desc)(self.ptr, level, (&raw mut desc).cast::<c_void>())
        };
        (hr, desc)
    }

    /// `LockBox` over the whole of mip `level`. Returns the hr and whether `pBits` came back null.
    ///
    /// The struct is seeded with a garbage pointer first, so a rejected lock
    /// that leaves it untouched reads as non-null.
    #[must_use]
    pub fn lock_box_probe(&self, level: u32, flags: u32) -> (i32, bool) {
        let mut locked = D3DLOCKED_BOX {
            row_pitch: 0,
            slice_pitch: 0,
            bits: core::ptr::without_provenance_mut(0xdead_beef),
        };
        // SAFETY: vtable thunk; `self.ptr` is live, `&mut locked` is writable,
        // a null box locks the whole level.
        let hr = unsafe {
            (self.vtbl().lock_box)(self.ptr, level, &raw mut locked, core::ptr::null(), flags)
        };
        (hr, locked.bits.is_null())
    }

    /// `UnlockBox` for mip `level`. Returns the hr.
    #[must_use]
    pub fn unlock_box(&self, level: u32) -> i32 {
        // SAFETY: vtable thunk; `self.ptr` is live.
        unsafe { (self.vtbl().unlock_box)(self.ptr, level) }
    }

    /// Fill mip `level` of a 32-bit-per-texel volume through `LockBox` / `UnlockBox`.
    ///
    /// `texels` is the whole level, tightly packed slice by slice, row by
    /// row; the write honours the row and slice pitches the lock reports.
    ///
    /// # Panics
    /// Panics if the lock fails or `texels` is not exactly one level's worth.
    pub fn write_u32(&self, level: u32, texels: &[u32]) {
        self.write_texels(level, texels);
    }

    /// [`Self::write_u32`] for 16-bit-per-texel formats (R5G6B5, A4R4G4B4, ...).
    ///
    /// # Panics
    /// Panics if the lock fails or `texels` is not exactly one level's worth.
    pub fn write_u16(&self, level: u32, texels: &[u16]) {
        self.write_texels(level, texels);
    }

    fn write_texels<T: Copy>(&self, level: u32, texels: &[T]) {
        let (hr, desc) = self.level_desc(level);
        expect_ok(hr, "VolumeTexture GetLevelDesc");
        let (width, height, depth) = (
            desc.width as usize,
            desc.height as usize,
            desc.depth as usize,
        );
        assert_eq!(texels.len(), width * height * depth, "one level of texels");
        let mut locked = D3DLOCKED_BOX {
            row_pitch: 0,
            slice_pitch: 0,
            bits: core::ptr::null_mut(),
        };
        // SAFETY: vtable thunk; `self.ptr` is live, `&mut locked` is writable,
        // a null box locks the whole level.
        let hr = unsafe {
            (self.vtbl().lock_box)(self.ptr, level, &raw mut locked, core::ptr::null(), 0)
        };
        expect_ok(hr, "VolumeTexture LockBox");
        assert!(!locked.bits.is_null(), "LockBox handed out a pointer");
        let row_pitch = usize::try_from(locked.row_pitch).expect("row pitch is positive");
        let slice_pitch = usize::try_from(locked.slice_pitch).expect("slice pitch is positive");
        for z in 0..depth {
            for y in 0..height {
                let row = &texels[(z * height + y) * width..][..width];
                // SAFETY: `LockBox` mapped `slice_pitch * depth` writable bytes
                // at `bits`, so the row start lands inside its own slice.
                let dst = unsafe {
                    locked
                        .bits
                        .cast::<u8>()
                        .add(z * slice_pitch + y * row_pitch)
                };
                // SAFETY: `width * size_of::<T>()` bytes from the row start
                // never exceed `row_pitch`, so the copy stays inside the
                // mapping.
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        row.as_ptr().cast::<u8>(),
                        dst,
                        width * core::mem::size_of::<T>(),
                    );
                }
            }
        }
        expect_ok(self.unlock_box(level), "VolumeTexture UnlockBox");
    }

    /// `GetVolumeLevel`, handing back the sub-resource it wrote.
    #[must_use]
    pub fn get_volume_level(&self, level: u32) -> (i32, Option<Volume<'_>>) {
        let mut out: *mut c_void = core::ptr::null_mut();
        // SAFETY: vtable thunk; `&mut out` is writable.
        let hr = unsafe { (self.vtbl().get_volume_level)(self.ptr, level, &raw mut out) };
        (hr, (!out.is_null()).then(|| Volume::from_raw(out)))
    }
}

impl Drop for VolumeTexture<'_> {
    fn drop(&mut self) {
        // SAFETY: vtable thunk; `self.ptr` is live and this is its last use.
        unsafe { (self.vtbl().release)(self.ptr) };
    }
}

// ── Volume ──

/// An `IDirect3DVolume9`, a level of a volume texture.
pub struct Volume<'h> {
    ptr: *mut c_void,
    _marker: PhantomData<&'h ()>,
}

impl Volume<'_> {
    pub const fn from_raw(ptr: *mut c_void) -> Self {
        Self {
            ptr,
            _marker: PhantomData,
        }
    }

    /// `GetDevice`, returning the hr and the device it wrote.
    ///
    /// The pointer carries a reference of its own; the caller releases it.
    #[must_use]
    pub fn get_device(&self) -> (i32, *mut c_void) {
        let mut out: *mut c_void = core::ptr::null_mut();
        // SAFETY: vtable thunk; `&mut out` is writable.
        let hr = unsafe { (self.vtbl().get_device)(self.ptr, &raw mut out) };
        (hr, out)
    }

    fn vtbl(&self) -> &'static IDirect3DVolume9Vtbl {
        // SAFETY: `self.ptr` is a live volume for the wrapper's lifetime.
        unsafe { deref_vtbl::<IDirect3DVolume9Vtbl>(self.ptr) }
    }

    /// The raw COM `this` pointer (for asserting sub-resource identity).
    #[must_use]
    pub const fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// `SetPrivateData(guid, blob, len, 0)`.
    #[must_use]
    pub fn set_private_data_hr(&self, guid: &Guid, blob: &[u8]) -> i32 {
        set_private_data(self.vtbl().set_private_data, self.ptr, guid, blob)
    }

    /// `GetPrivateData(guid, out, &mut size)`, returning the hr and the size.
    ///
    /// A null `out` asks for the size alone.
    #[must_use]
    pub fn get_private_data(&self, guid: &Guid, out: Option<&mut [u8]>) -> (i32, u32) {
        get_private_data(self.vtbl().get_private_data, self.ptr, guid, out)
    }

    /// `FreePrivateData(guid)`.
    #[must_use]
    pub fn free_private_data_hr(&self, guid: &Guid) -> i32 {
        // SAFETY: vtable thunk; `self.ptr` is live.
        unsafe { (self.vtbl().free_private_data)(self.ptr, &raw const *guid) }
    }

    #[must_use]
    pub fn desc(&self) -> (i32, D3DVOLUME_DESC) {
        let mut desc = D3DVOLUME_DESC {
            format: 0,
            resource_type: 0,
            usage: 0,
            pool: 0,
            width: 0,
            height: 0,
            depth: 0,
        };
        // SAFETY: vtable thunk; `self.ptr` is live and `&mut desc` is writable.
        let hr = unsafe { (self.vtbl().get_desc)(self.ptr, &raw mut desc) };
        (hr, desc)
    }

    /// Fill the level through `IDirect3DVolume9::LockBox` / `UnlockBox`.
    ///
    /// `texels` is the whole level, tightly packed slice by slice, row by
    /// row; the write honours the row and slice pitches the lock reports.
    ///
    /// # Panics
    /// Panics if the lock fails or `texels` is not exactly one level's worth.
    pub fn write_u32(&self, texels: &[u32]) {
        self.write_texels(texels);
    }

    /// [`Self::write_u32`] for 16-bit-per-texel formats (R5G6B5, A4R4G4B4, ...).
    ///
    /// # Panics
    /// Panics if the lock fails or `texels` is not exactly one level's worth.
    pub fn write_u16(&self, texels: &[u16]) {
        self.write_texels(texels);
    }

    fn write_texels<T: Copy>(&self, texels: &[T]) {
        let (hr, desc) = self.desc();
        expect_ok(hr, "Volume GetDesc");
        let (width, height, depth) = (
            desc.width as usize,
            desc.height as usize,
            desc.depth as usize,
        );
        assert_eq!(texels.len(), width * height * depth, "one level of texels");
        let mut locked = D3DLOCKED_BOX {
            row_pitch: 0,
            slice_pitch: 0,
            bits: core::ptr::null_mut(),
        };
        // SAFETY: vtable thunk; `self.ptr` is live, `&mut locked` is writable,
        // a null box locks the whole level.
        let hr = unsafe { (self.vtbl().lock_box)(self.ptr, &raw mut locked, core::ptr::null(), 0) };
        expect_ok(hr, "Volume LockBox");
        assert!(!locked.bits.is_null(), "LockBox handed out a pointer");
        let row_pitch = usize::try_from(locked.row_pitch).expect("row pitch is positive");
        let slice_pitch = usize::try_from(locked.slice_pitch).expect("slice pitch is positive");
        for z in 0..depth {
            for y in 0..height {
                let row = &texels[(z * height + y) * width..][..width];
                // SAFETY: `LockBox` mapped `slice_pitch * depth` writable bytes
                // at `bits`, so the row start lands inside its own slice.
                let dst = unsafe {
                    locked
                        .bits
                        .cast::<u8>()
                        .add(z * slice_pitch + y * row_pitch)
                };
                // SAFETY: `width * size_of::<T>()` bytes from the row start
                // never exceed `row_pitch`, so the copy stays inside the
                // mapping.
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        row.as_ptr().cast::<u8>(),
                        dst,
                        width * core::mem::size_of::<T>(),
                    );
                }
            }
        }
        // SAFETY: vtable thunk; `self.ptr` is live and the level is locked.
        let hr = unsafe { (self.vtbl().unlock_box)(self.ptr) };
        expect_ok(hr, "Volume UnlockBox");
    }
}

impl Drop for Volume<'_> {
    fn drop(&mut self) {
        // SAFETY: vtable thunk; `self.ptr` is live and this is its last use.
        unsafe { (self.vtbl().release)(self.ptr) };
    }
}

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

    /// `LockRect` over the whole of mip `level`. Returns the hr and whether `pBits` came back null.
    ///
    /// The struct is seeded with a garbage pointer first, so a rejected lock
    /// that leaves it untouched reads as non-null. For a test that expects the
    /// lock to fail; a successful one leaves the level mapped until
    /// [`Self::unlock_rect`].
    #[must_use]
    pub fn lock_rect_probe(&self, level: u32, flags: u32) -> (i32, bool) {
        let mut locked = D3DLOCKED_RECT {
            pitch: 0,
            bits: core::ptr::without_provenance_mut(0xdead_beef),
        };
        // SAFETY: vtable thunk; `self.ptr` is live, `&mut locked` is writable,
        // a null rect locks the whole level.
        let hr = unsafe {
            (self.vtbl().lock_rect)(self.ptr, level, &raw mut locked, core::ptr::null(), flags)
        };
        (hr, locked.bits.is_null())
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

    /// `UnlockRect` for mip `level`. Returns the hr.
    #[must_use]
    pub fn unlock_rect(&self, level: u32) -> i32 {
        // SAFETY: vtable thunk; `self.ptr` is live.
        unsafe { (self.vtbl().unlock_rect)(self.ptr, level) }
    }

    /// `AddDirtyRect(null)` — flag the whole texture dirty. Returns the hr.
    #[must_use]
    pub fn add_dirty_rect(&self) -> i32 {
        // SAFETY: vtable thunk; `self.ptr` is live, null rect = whole surface.
        unsafe { (self.vtbl().add_dirty_rect)(self.ptr, core::ptr::null()) }
    }

    /// `AddDirtyRect` over one sub-rectangle. Returns the hr.
    ///
    /// `rect` is a `D3DRECT`-style `[left, top, right, bottom]`.
    #[must_use]
    pub fn add_dirty_rect_partial(&self, rect: &[i32; 4]) -> i32 {
        // SAFETY: vtable thunk; `self.ptr` is live and `rect` is four `LONG`s,
        // the `RECT` layout the call reads.
        unsafe { (self.vtbl().add_dirty_rect)(self.ptr, rect.as_ptr().cast::<c_void>()) }
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

    /// `GetDevice`, returning the hr and the device it wrote.
    ///
    /// The pointer carries a reference of its own; the caller releases it.
    #[must_use]
    pub fn get_device(&self) -> (i32, *mut c_void) {
        let mut out: *mut c_void = core::ptr::null_mut();
        // SAFETY: vtable thunk; `&mut out` is writable.
        let hr = unsafe { (self.vtbl().get_device)(self.ptr, &raw mut out) };
        (hr, out)
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

    /// Current public refcount, read through a balanced `AddRef`/`Release` pair.
    ///
    /// `AddRef` answers the count it just produced, so the standing count is one
    /// less. A sub-resource forwards its own references here, which is what makes
    /// this the count a `GetSurfaceLevel` test watches.
    #[must_use]
    pub fn refcount(&self) -> u32 {
        // SAFETY: vtable thunk; `self.ptr` is live.
        let bumped = unsafe { (self.vtbl().add_ref)(self.ptr) };
        // SAFETY: balances the AddRef above; this wrapper keeps its own reference.
        unsafe { (self.vtbl().release)(self.ptr) };
        bumped - 1
    }
}

impl Drop for Texture<'_> {
    fn drop(&mut self) {
        // SAFETY: vtable thunk; `self.ptr` is live and this is its last use.
        unsafe { (self.vtbl().release)(self.ptr) };
    }
}

// ── Cube texture ──

/// An `IDirect3DCubeTexture9`.
pub struct CubeTexture<'h> {
    ptr: *mut c_void,
    _marker: PhantomData<&'h ()>,
}

impl CubeTexture<'_> {
    pub const fn from_raw(ptr: *mut c_void) -> Self {
        Self {
            ptr,
            _marker: PhantomData,
        }
    }

    /// `GetDevice`, returning the hr and the device it wrote.
    ///
    /// The pointer carries a reference of its own; the caller releases it.
    #[must_use]
    pub fn get_device(&self) -> (i32, *mut c_void) {
        let mut out: *mut c_void = core::ptr::null_mut();
        // SAFETY: vtable thunk; `&mut out` is writable.
        let hr = unsafe { (self.vtbl().get_device)(self.ptr, &raw mut out) };
        (hr, out)
    }

    /// The raw COM pointer used for base-texture binding.
    #[must_use]
    pub const fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    fn vtbl(&self) -> &'static IDirect3DCubeTexture9Vtbl {
        // SAFETY: `self.ptr` is a live cube texture for the wrapper lifetime.
        unsafe { deref_vtbl::<IDirect3DCubeTexture9Vtbl>(self.ptr) }
    }

    /// Lock one cube face and mip level.
    ///
    /// # Panics
    /// Panics if `LockRect` fails.
    #[must_use]
    pub fn lock_rect(&self, face: u32, level: u32, flags: u32) -> LockedRect<'_> {
        let mut locked = D3DLOCKED_RECT {
            pitch: 0,
            bits: core::ptr::null_mut(),
        };
        // SAFETY: live cube texture and writable lock out-param.
        let hr = unsafe {
            (self.vtbl().lock_rect)(
                self.ptr,
                face,
                level,
                &raw mut locked,
                core::ptr::null(),
                flags,
            )
        };
        expect_ok(hr, "CubeTexture LockRect");
        LockedRect {
            owner: LockOwner::Cube {
                this: self.ptr,
                face,
                level,
            },
            pitch: locked.pitch,
            bits: locked.bits,
            _marker: PhantomData,
        }
    }

    /// Get a parent-backed face surface.
    ///
    /// # Panics
    /// Panics if `GetCubeMapSurface` fails.
    #[must_use]
    pub fn surface(&self, face: u32, level: u32) -> Surface<'_> {
        let mut surface = core::ptr::null_mut();
        // SAFETY: live cube texture and writable surface out-param.
        let hr =
            unsafe { (self.vtbl().get_cube_map_surface)(self.ptr, face, level, &raw mut surface) };
        expect_created(hr, surface, "GetCubeMapSurface");
        Surface::from_raw(surface)
    }

    /// `GetCubeMapSurface` returning `(hr, this)` for error-path tests.
    ///
    /// The unchecked form of [`Self::surface`]: a caller that expects a
    /// rejection reads both the hr and the untouched out-param, and a caller
    /// that gets a surface owns the reference it was handed.
    #[must_use]
    pub fn try_surface(&self, face: u32, level: u32) -> (i32, *mut c_void) {
        let mut surface = core::ptr::null_mut();
        // SAFETY: live cube texture and writable surface out-param.
        let hr =
            unsafe { (self.vtbl().get_cube_map_surface)(self.ptr, face, level, &raw mut surface) };
        (hr, surface)
    }

    /// Mip-chain length.
    #[must_use]
    pub fn level_count(&self) -> u32 {
        // SAFETY: live cube texture.
        unsafe { (self.vtbl().get_level_count)(self.ptr) }
    }
}

impl Drop for CubeTexture<'_> {
    fn drop(&mut self) {
        // SAFETY: live cube texture and this wrapper owns one reference.
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

    /// `GetDevice`, returning the hr and the device it wrote.
    ///
    /// The pointer carries a reference of its own; the caller releases it.
    #[must_use]
    pub fn get_device(&self) -> (i32, *mut c_void) {
        let mut out: *mut c_void = core::ptr::null_mut();
        // SAFETY: vtable thunk; `&mut out` is writable.
        let hr = unsafe { (self.vtbl().get_device)(self.ptr, &raw mut out) };
        (hr, out)
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

    /// `LockRect` over the whole surface. Returns the hr and whether `pBits` came back null.
    ///
    /// The struct is seeded with a garbage pointer first, so a rejected lock
    /// that leaves it untouched reads as non-null. For a test that expects the
    /// lock to fail; a successful one leaves the surface mapped until
    /// [`Self::unlock_rect`].
    #[must_use]
    pub fn lock_rect_probe(&self, flags: u32) -> (i32, bool) {
        let mut locked = D3DLOCKED_RECT {
            pitch: 0,
            bits: core::ptr::without_provenance_mut(0xdead_beef),
        };
        // SAFETY: vtable thunk; `self.ptr` is live, `&mut locked` is writable,
        // a null rect locks the whole surface.
        let hr =
            unsafe { (self.vtbl().lock_rect)(self.ptr, &raw mut locked, core::ptr::null(), flags) };
        (hr, locked.bits.is_null())
    }

    /// `UnlockRect`. Returns the hr.
    #[must_use]
    pub fn unlock_rect(&self) -> i32 {
        // SAFETY: vtable thunk; `self.ptr` is live.
        unsafe { (self.vtbl().unlock_rect)(self.ptr) }
    }

    /// `GetType` (`D3DRTYPE_*`).
    #[must_use]
    pub fn resource_type(&self) -> u32 {
        // SAFETY: vtable thunk; `self.ptr` is live.
        unsafe { (self.vtbl().get_type)(self.ptr) }
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

    /// `GetDC`, asserting success and returning a guard over the memory DC.
    ///
    /// The guard reads and writes pixels through GDI and releases the DC on
    /// request; use [`Self::get_dc`] instead to observe the raw hr.
    ///
    /// # Panics
    /// Panics if the call fails or hands back a null `HDC`.
    #[must_use]
    pub fn dc(&self) -> SurfaceDc<'_> {
        let (hr, hdc) = self.get_dc(core::ptr::null_mut());
        expect_ok(hr, "Surface GetDC");
        assert!(!hdc.is_null(), "GetDC returned a null HDC");
        SurfaceDc {
            surface: self.ptr,
            hdc,
            _marker: PhantomData,
        }
    }

    /// Give up this wrapper's reference without releasing it.
    ///
    /// For a test that hands a surface's last reference to the device (a bound
    /// render target) and then reads the object back through a non-owning view.
    #[must_use]
    pub const fn into_raw(self) -> *mut c_void {
        let ptr = self.ptr;
        core::mem::forget(self);
        ptr
    }

    /// `SetPrivateData(guid, blob, len, 0)`.
    #[must_use]
    pub fn set_private_data_hr(&self, guid: &Guid, blob: &[u8]) -> i32 {
        set_private_data(self.vtbl().set_private_data, self.ptr, guid, blob)
    }

    /// `GetPrivateData(guid, out, &mut size)`, returning the hr and the size.
    ///
    /// A null `out` asks for the size alone.
    #[must_use]
    pub fn get_private_data(&self, guid: &Guid, out: Option<&mut [u8]>) -> (i32, u32) {
        get_private_data(self.vtbl().get_private_data, self.ptr, guid, out)
    }

    /// `FreePrivateData(guid)`.
    #[must_use]
    pub fn free_private_data_hr(&self, guid: &Guid) -> i32 {
        // SAFETY: vtable thunk; `self.ptr` is live.
        unsafe { (self.vtbl().free_private_data)(self.ptr, &raw const *guid) }
    }
}

impl Drop for Surface<'_> {
    fn drop(&mut self) {
        // SAFETY: vtable thunk; `self.ptr` is live and this is its last use.
        unsafe { (self.vtbl().release)(self.ptr) };
    }
}

/// A held `IDirect3DSurface9::GetDC` memory DC.
///
/// Reads and writes pixels through GDI, which is how a test observes the
/// surface exactly as a game's GDI drawing does. [`Self::release`] hands back
/// the `ReleaseDC` hr; a dropped guard leaves the DC held, which the next
/// `LockRect` / `GetDC` on the surface then rejects.
pub struct SurfaceDc<'a> {
    surface: *mut c_void,
    hdc: *mut c_void,
    _marker: PhantomData<&'a ()>,
}

impl SurfaceDc<'_> {
    /// Read one pixel as a `COLORREF` (`0x00BBGGRR`).
    #[must_use]
    pub fn get_pixel(&self, x: i32, y: i32) -> u32 {
        crate::win32::dc_get_pixel(self.hdc.addr(), x, y)
    }

    /// Paint one pixel; `color` is a `COLORREF` (`0x00BBGGRR`).
    ///
    /// Returns the colour GDI stored, which for a DIB of a lower-precision
    /// format is the nearest representable one.
    #[must_use]
    pub fn set_pixel(&self, x: i32, y: i32, color: u32) -> u32 {
        crate::win32::dc_set_pixel(self.hdc.addr(), x, y, color)
    }

    /// `ReleaseDC`, returning the hr.
    #[must_use]
    pub fn release(self) -> i32 {
        // SAFETY: `self.surface` is the live surface the DC was taken from.
        let vtbl = unsafe { deref_vtbl::<IDirect3DSurface9Vtbl>(self.surface) };
        // SAFETY: vtable thunk; `self.hdc` is the handle the surface's own
        // `GetDC` returned and has not been released yet.
        unsafe { (vtbl.release_dc)(self.surface, self.hdc) }
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

    /// `GetDevice`, returning the hr and the device it wrote.
    ///
    /// The pointer carries a reference of its own; the caller releases it.
    #[must_use]
    pub fn get_device(&self) -> (i32, *mut c_void) {
        let mut out: *mut c_void = core::ptr::null_mut();
        // SAFETY: vtable thunk; `&mut out` is writable.
        let hr = unsafe { (self.vtbl().get_device)(self.ptr, &raw mut out) };
        (hr, out)
    }

    /// `GetDevice` through this buffer's vtable, on a caller-chosen `this`.
    ///
    /// A null `this` and a null out-param are contract cases: the thunk
    /// answers rather than faulting, which a wrapper method cannot express.
    ///
    /// # Safety
    /// `this` is null or a live vertex buffer; `device` is null or a writable
    /// `*mut c_void` slot.
    #[must_use]
    pub unsafe fn get_device_raw(&self, this: *mut c_void, device: *mut *mut c_void) -> i32 {
        // SAFETY: vtable thunk; the caller states what `this` and `device` are.
        unsafe { (self.vtbl().get_device)(this, device) }
    }

    /// `SetPrivateData(guid, blob, len, 0)`.
    #[must_use]
    pub fn set_private_data_hr(&self, guid: &Guid, blob: &[u8]) -> i32 {
        set_private_data(self.vtbl().set_private_data, self.ptr, guid, blob)
    }

    /// `GetPrivateData(guid, out, &mut size)`, returning the hr and the size.
    ///
    /// A null `out` asks for the size alone.
    #[must_use]
    pub fn get_private_data(&self, guid: &Guid, out: Option<&mut [u8]>) -> (i32, u32) {
        get_private_data(self.vtbl().get_private_data, self.ptr, guid, out)
    }

    /// `FreePrivateData(guid)`.
    #[must_use]
    pub fn free_private_data_hr(&self, guid: &Guid) -> i32 {
        // SAFETY: vtable thunk; `self.ptr` is live.
        unsafe { (self.vtbl().free_private_data)(self.ptr, &raw const *guid) }
    }

    /// `SetPrivateData(guid, punk, sizeof(ptr), D3DSPD_IUNKNOWN)`.
    ///
    /// The runtime holds a reference on `punk` until the key is overwritten,
    /// freed, or the resource dies.
    ///
    /// # Panics
    /// Never in practice: only if a pointer does not fit `u32`.
    #[must_use]
    pub fn set_private_data_unknown(&self, guid: &Guid, punk: *mut c_void) -> i32 {
        let size = u32::try_from(size_of::<*mut c_void>()).expect("pointer size fits u32");
        // SAFETY: vtable thunk; for `D3DSPD_IUNKNOWN` the data pointer *is*
        // the interface pointer, and `punk` is a live COM object.
        unsafe {
            (self.vtbl().set_private_data)(
                self.ptr,
                &raw const *guid,
                punk.cast_const(),
                size,
                mtld3d_types::D3DSPD_IUNKNOWN,
            )
        }
    }

    /// `GetPrivateData` for a stored `IUnknown`.
    ///
    /// Returns the hr, the pointer it wrote, and the size it reported.
    ///
    /// The pointer carries a reference of its own; the caller releases it.
    ///
    /// # Panics
    /// Never in practice: only if a pointer does not fit `u32`.
    #[must_use]
    pub fn get_private_data_unknown(&self, guid: &Guid) -> (i32, *mut c_void, u32) {
        let mut punk: *mut c_void = core::ptr::null_mut();
        let mut size = u32::try_from(size_of::<*mut c_void>()).expect("pointer size fits u32");
        // SAFETY: vtable thunk; `&mut punk` is a writable pointer slot of the
        // width `size` names.
        let hr = unsafe {
            (self.vtbl().get_private_data)(
                self.ptr,
                &raw const *guid,
                (&raw mut punk).cast::<c_void>(),
                &raw mut size,
            )
        };
        (hr, punk, size)
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

    /// `GetDevice`, returning the hr and the device it wrote.
    ///
    /// The pointer carries a reference of its own; the caller releases it.
    #[must_use]
    pub fn get_device(&self) -> (i32, *mut c_void) {
        let mut out: *mut c_void = core::ptr::null_mut();
        // SAFETY: vtable thunk; `&mut out` is writable.
        let hr = unsafe { (self.vtbl().get_device)(self.ptr, &raw mut out) };
        (hr, out)
    }

    /// `SetPrivateData(guid, blob, len, 0)`.
    #[must_use]
    pub fn set_private_data_hr(&self, guid: &Guid, blob: &[u8]) -> i32 {
        set_private_data(self.vtbl().set_private_data, self.ptr, guid, blob)
    }

    /// `GetPrivateData(guid, out, &mut size)`, returning the hr and the size.
    ///
    /// A null `out` asks for the size alone.
    #[must_use]
    pub fn get_private_data(&self, guid: &Guid, out: Option<&mut [u8]>) -> (i32, u32) {
        get_private_data(self.vtbl().get_private_data, self.ptr, guid, out)
    }

    /// `FreePrivateData(guid)`.
    #[must_use]
    pub fn free_private_data_hr(&self, guid: &Guid) -> i32 {
        // SAFETY: vtable thunk; `self.ptr` is live.
        unsafe { (self.vtbl().free_private_data)(self.ptr, &raw const *guid) }
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

    /// `GetDevice`, returning the hr and the device it wrote.
    ///
    /// The pointer carries a reference of its own; the caller releases it.
    #[must_use]
    pub fn get_device(&self) -> (i32, *mut c_void) {
        let mut out: *mut c_void = core::ptr::null_mut();
        // SAFETY: `self.ptr` is a live vertex shader.
        let vtbl = unsafe { deref_vtbl::<IDirect3DVertexShader9Vtbl>(self.ptr) };
        // SAFETY: vtable thunk; `&mut out` is writable.
        let hr = unsafe { (vtbl.get_device)(self.ptr, &raw mut out) };
        (hr, out)
    }

    /// The raw COM `this` pointer (for `SetVertexShader`).
    #[must_use]
    pub const fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// `GetFunction(data, &mut size)`.
    ///
    /// Both out-params pass through as given, so a test can exercise the size
    /// query and the error paths rather than only the happy one.
    ///
    /// # Safety
    /// `data` is either null or points to at least `*size` writable bytes;
    /// `size` is either null or a writable `u32`. Both nulls are contract
    /// cases a test may pass deliberately.
    pub unsafe fn get_function(&self, data: *mut c_void, size: *mut u32) -> i32 {
        // SAFETY: `self.ptr` is a live vertex shader.
        let vtbl = unsafe { deref_vtbl::<IDirect3DVertexShader9Vtbl>(self.ptr) };
        // SAFETY: vtable thunk; `data`/`size` are the caller's out-params.
        unsafe { (vtbl.get_function)(self.ptr, data, size) }
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

    /// `GetDevice`, returning the hr and the device it wrote.
    ///
    /// The pointer carries a reference of its own; the caller releases it.
    #[must_use]
    pub fn get_device(&self) -> (i32, *mut c_void) {
        let mut out: *mut c_void = core::ptr::null_mut();
        // SAFETY: `self.ptr` is a live pixel shader.
        let vtbl = unsafe { deref_vtbl::<IDirect3DPixelShader9Vtbl>(self.ptr) };
        // SAFETY: vtable thunk; `&mut out` is writable.
        let hr = unsafe { (vtbl.get_device)(self.ptr, &raw mut out) };
        (hr, out)
    }

    /// The raw COM `this` pointer (for `SetPixelShader`).
    #[must_use]
    pub const fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// `GetFunction(data, &mut size)`.
    ///
    /// Both out-params pass through as given, so a test can exercise the size
    /// query and the error paths rather than only the happy one.
    ///
    /// # Safety
    /// `data` is either null or points to at least `*size` writable bytes;
    /// `size` is either null or a writable `u32`. Both nulls are contract
    /// cases a test may pass deliberately.
    pub unsafe fn get_function(&self, data: *mut c_void, size: *mut u32) -> i32 {
        // SAFETY: `self.ptr` is a live pixel shader.
        let vtbl = unsafe { deref_vtbl::<IDirect3DPixelShader9Vtbl>(self.ptr) };
        // SAFETY: vtable thunk; `data`/`size` are the caller's out-params.
        unsafe { (vtbl.get_function)(self.ptr, data, size) }
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

    /// `GetDevice`, returning the hr and the device it wrote.
    ///
    /// The pointer carries a reference of its own; the caller releases it.
    #[must_use]
    pub fn get_device(&self) -> (i32, *mut c_void) {
        let mut out: *mut c_void = core::ptr::null_mut();
        // SAFETY: vtable thunk; `&mut out` is writable.
        let hr = unsafe { (self.vtbl().get_device)(self.ptr, &raw mut out) };
        (hr, out)
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

    /// `GetDevice`, returning the hr and the device it wrote.
    ///
    /// The pointer carries a reference of its own; the caller releases it.
    #[must_use]
    pub fn get_device(&self) -> (i32, *mut c_void) {
        let mut out: *mut c_void = core::ptr::null_mut();
        // SAFETY: vtable thunk; `&mut out` is writable.
        let hr = unsafe { (self.vtbl().get_device)(self.ptr, &raw mut out) };
        (hr, out)
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

    /// `GetDevice`, returning the hr and the device it wrote.
    ///
    /// The pointer carries a reference of its own; the caller releases it.
    #[must_use]
    pub fn get_device(&self) -> (i32, *mut c_void) {
        let mut out: *mut c_void = core::ptr::null_mut();
        // SAFETY: `self.ptr` is a live vertex declaration.
        let vtbl = unsafe { deref_vtbl::<IDirect3DVertexDeclaration9Vtbl>(self.ptr) };
        // SAFETY: vtable thunk; `&mut out` is writable.
        let hr = unsafe { (vtbl.get_device)(self.ptr, &raw mut out) };
        (hr, out)
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
    Texture {
        this: *mut c_void,
        level: u32,
    },
    Cube {
        this: *mut c_void,
        face: u32,
        level: u32,
    },
    Surface {
        this: *mut c_void,
    },
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

    /// View the first `count` `u16` lanes of the mapped span.
    ///
    /// For the 16-bit-per-channel formats, where one texel spans several
    /// lanes (a half-float RGBA texel is four of them).
    #[must_use]
    pub const fn as_u16(&self, count: usize) -> &[u16] {
        // SAFETY: `bits` is valid for `count` u16s within the locked region
        // (caller's contract) and lives until this guard drops.
        unsafe { core::slice::from_raw_parts(self.bits.cast::<u16>(), count) }
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
            LockOwner::Cube { this, face, level } => {
                // SAFETY: `this` is the live cube texture this guard locked.
                let vtbl = unsafe { deref_vtbl::<IDirect3DCubeTexture9Vtbl>(this) };
                // SAFETY: vtable thunk; face and level match the lock.
                unsafe { (vtbl.unlock_rect)(this, face, level) };
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

    /// Read `count` `Copy` POD values from the mapped span at byte offset 0.
    ///
    /// # Panics
    /// The caller must ensure `count` values fit within the locked region.
    #[must_use]
    pub fn read<T: Copy>(&self, count: usize) -> Vec<T> {
        let mut out = Vec::with_capacity(count);
        // SAFETY: `bits` maps at least `count * size_of::<T>()` bytes of the
        // locked region (caller's contract); `out` has `count` capacity and the
        // values are `Copy` POD, so the bytes are a valid `T` sequence.
        unsafe { core::ptr::copy_nonoverlapping(self.bits.cast::<T>(), out.as_mut_ptr(), count) };
        // SAFETY: the copy above initialised `count` elements.
        unsafe { out.set_len(count) };
        out
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
