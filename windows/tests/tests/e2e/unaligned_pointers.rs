//! Application pointers at addresses their types are not aligned for.
//!
//! An application keeps the structs and arrays it hands D3D9 wherever it likes, a packed record or
//! an odd stack slot included, and the D3D9 runtime copies through them. Every call here gets a
//! pointer one byte past an aligned address, so a layer that formed a reference to it, or read or
//! wrote through it as an aligned value, would trip the `misaligned pointer dereference` check of
//! the debug-assertion build these tests run against. Each test then checks that the values land.

use core::ffi::c_void;

use mtld3d_tests::{Factory, Harness, HarnessConfig, RhwVertex, assert_pixel_eq, deref_vtbl};
use mtld3d_types::{
    D3D_OK, D3DADAPTER_IDENTIFIER9, D3DBOX, D3DCAPS9, D3DCLEAR_TARGET, D3DCOLORVALUE,
    D3DCREATE_HARDWARE_VERTEXPROCESSING, D3DDECL_END_STREAM, D3DDECLTYPE_FLOAT3,
    D3DDECLTYPE_UNUSED, D3DDECLUSAGE_POSITION, D3DDEVICE_CREATION_PARAMETERS, D3DDEVTYPE_HAL,
    D3DDISPLAYMODE, D3DDISPLAYMODEEX, D3DDISPLAYMODEEX_SIZE, D3DDISPLAYROTATION_IDENTITY,
    D3DFMT_A8R8G8B8, D3DFMT_INDEX16, D3DFVF_DIFFUSE, D3DFVF_XYZRHW, D3DLIGHT_POINT, D3DLIGHT9,
    D3DLOCK_READONLY, D3DLOCKED_BOX, D3DLOCKED_RECT, D3DMATERIAL9, D3DMATRIX, D3DPOOL_MANAGED,
    D3DPOOL_SYSTEMMEM, D3DPRESENT_INTERVAL_IMMEDIATE, D3DPRESENT_PARAMETERS, D3DPT_TRIANGLELIST,
    D3DRECT, D3DRS_AMBIENT, D3DRTYPE_SURFACE, D3DRTYPE_VOLUME, D3DSAMP_ADDRESSU,
    D3DSCANLINEORDERING_PROGRESSIVE, D3DSURFACE_DESC, D3DSWAPEFFECT_DISCARD, D3DTADDRESS_CLAMP,
    D3DTOP_SELECTARG1, D3DTS_WORLD, D3DTSS_COLOROP, D3DVECTOR, D3DVERTEXELEMENT9, D3DVIEWPORT9,
    D3DVOLUME_DESC, Guid, IDirect3D9Vtbl, IDirect3DCubeTexture9Vtbl, IDirect3DDevice9ExVtbl,
    IDirect3DDevice9Vtbl, IDirect3DIndexBuffer9Vtbl, IDirect3DSurface9Vtbl,
    IDirect3DSwapChain9Vtbl, IDirect3DTexture9Vtbl, IDirect3DVertexBuffer9Vtbl,
    IDirect3DVertexDeclaration9Vtbl, IDirect3DVertexShader9Vtbl, IDirect3DVolumeTexture9Vtbl,
    IID_IDIRECT3DTEXTURE9,
};

/// The texel value the lock tests write and read back.
const MARK: u32 = 0x1122_3344;

/// `vs_2_0 { dcl_position v0; mov oPos, v0 }`.
const VS_PASSTHROUGH: [u32; 8] = [
    0xFFFE_0200,
    0x0200_001F,
    0x8000_0000,
    0x900F_0000,
    0x0200_0001,
    0xC00F_0000,
    0x90E4_0000,
    0x0000_FFFF,
];

/// Scratch memory whose usable bytes start one byte past a 16-byte boundary.
#[repr(C, align(16))]
struct Odd {
    bytes: [u8; 2048],
}

impl Odd {
    const fn new() -> Self {
        Self { bytes: [0; 2048] }
    }

    /// The misaligned address, typed as `T`.
    fn ptr<T>(&mut self) -> *mut T {
        let p = self.bytes[1..].as_mut_ptr().cast::<T>();
        assert!(
            !p.is_aligned(),
            "the slot must be misaligned for the type under test"
        );
        p
    }

    /// Store `values` back to back from the misaligned address and return it.
    fn put<T: Copy>(&mut self, values: &[T]) -> *mut T {
        let len = size_of_val(values);
        assert!(len < self.bytes.len(), "the values fit the scratch");
        self.bytes[1..=len].copy_from_slice(
            // SAFETY: `values` is a live slice of `len` bytes, viewed as bytes.
            unsafe { core::slice::from_raw_parts(values.as_ptr().cast::<u8>(), len) },
        );
        self.ptr::<T>()
    }

    /// Read the `index`th `T` stored from the misaligned address.
    ///
    /// # Safety
    /// The bytes there must be a valid `T`: the D3D9 structs this file reads are
    /// integers, floats and raw pointers, for which any bytes are.
    unsafe fn get<T>(&self, index: usize) -> T {
        assert!(
            (index + 1) * size_of::<T>() < self.bytes.len(),
            "the read fits the scratch"
        );
        let p = self.bytes[1..].as_ptr().cast::<T>().wrapping_add(index);
        // SAFETY: in bounds per the assert, valid per the caller; no alignment is assumed.
        unsafe { p.read_unaligned() }
    }
}

fn device_vtbl(h: &Harness) -> &'static IDirect3DDevice9Vtbl {
    // SAFETY: the harness device is live for the harness lifetime.
    unsafe { deref_vtbl::<IDirect3DDevice9Vtbl>(h.device()) }
}

/// Byte offset of texel (1, 1) in a 4-byte-per-texel level of `pitch`.
fn texel_one_one(pitch: i32) -> usize {
    usize::try_from(pitch).expect("a lock pitch is positive") + 4
}

/// `IDirect3DTexture9::LockRect` writes a misaligned `D3DLOCKED_RECT` and reads a misaligned rect.
///
/// The whole-level lock hands back a usable pitch and pointer, a texel written through it reads
/// back through a partial lock whose `RECT` sits at an odd address, and `GetLevelDesc` fills a
/// misaligned `D3DSURFACE_DESC`.
#[test]
fn texture_lock_rect_and_level_desc_through_misaligned_structs() {
    let h = Harness::new();
    let texture = h.create_texture(4, 4, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
    // SAFETY: the texture is live for the test.
    let vt = unsafe { deref_vtbl::<IDirect3DTexture9Vtbl>(texture.as_ptr()) };

    let mut out = Odd::new();
    // SAFETY: vtable thunk on a live texture; the out struct has room, the rect is null.
    let hr = unsafe { (vt.lock_rect)(texture.as_ptr(), 0, out.ptr(), core::ptr::null(), 0) };
    assert_eq!(hr, D3D_OK, "LockRect into a misaligned D3DLOCKED_RECT");
    // SAFETY: LockRect wrote the struct: an integer and a pointer.
    let whole: D3DLOCKED_RECT = unsafe { out.get(0) };
    assert!(
        whole.pitch >= 16,
        "pitch {} covers a 4-texel row",
        whole.pitch
    );
    assert!(!whole.bits.is_null(), "LockRect returned a pointer");
    let texel = whole
        .bits
        .cast::<u8>()
        .wrapping_add(texel_one_one(whole.pitch));
    // SAFETY: texel (1, 1) of the locked 4x4 level.
    unsafe { texel.cast::<u32>().write_unaligned(MARK) };
    assert_eq!(texture.unlock_rect(0), D3D_OK, "UnlockRect");

    let mut rect = Odd::new();
    let corner = rect.put(&[D3DRECT {
        x1: 1,
        y1: 1,
        x2: 3,
        y2: 3,
    }]);
    // SAFETY: vtable thunk on a live texture; out and rect point into live scratch.
    let hr = unsafe {
        (vt.lock_rect)(
            texture.as_ptr(),
            0,
            out.ptr(),
            corner.cast(),
            D3DLOCK_READONLY,
        )
    };
    assert_eq!(hr, D3D_OK, "LockRect of a misaligned RECT");
    // SAFETY: LockRect wrote the struct.
    let part: D3DLOCKED_RECT = unsafe { out.get(0) };
    // SAFETY: a partial lock's pointer addresses the rect's top-left texel, (1, 1).
    let read = unsafe { part.bits.cast::<u32>().read_unaligned() };
    assert_eq!(
        read, MARK,
        "the partial lock starts at the misaligned rect's corner"
    );
    assert_eq!(texture.unlock_rect(0), D3D_OK, "UnlockRect");

    let mut desc = Odd::new();
    // SAFETY: vtable thunk on a live texture; the out struct has room.
    let hr = unsafe { (vt.get_level_desc)(texture.as_ptr(), 0, desc.ptr()) };
    assert_eq!(hr, D3D_OK, "GetLevelDesc into a misaligned D3DSURFACE_DESC");
    // SAFETY: GetLevelDesc wrote the struct: integers only.
    let d: D3DSURFACE_DESC = unsafe { desc.get(0) };
    assert_eq!(
        (d.format, d.resource_type, d.width, d.height),
        (D3DFMT_A8R8G8B8, D3DRTYPE_SURFACE, 4, 4),
        "the level's description",
    );
}

/// A system-memory surface and a cube face lock into misaligned `D3DLOCKED_RECT`s.
///
/// The surface's `GetDesc` fills a misaligned `D3DSURFACE_DESC`, and a texel written through
/// each lock reads back through a second one.
#[test]
fn surface_and_cube_lock_rect_through_misaligned_structs() {
    let h = Harness::new();
    let surface = h.create_offscreen_plain_surface(4, 4, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    // SAFETY: the surface is live for the test.
    let svt = unsafe { deref_vtbl::<IDirect3DSurface9Vtbl>(surface.as_ptr()) };
    let mut out = Odd::new();
    for (flags, write) in [(0, true), (D3DLOCK_READONLY, false)] {
        // SAFETY: vtable thunk on a live surface; the out struct has room, the rect is null.
        let hr = unsafe { (svt.lock_rect)(surface.as_ptr(), out.ptr(), core::ptr::null(), flags) };
        assert_eq!(
            hr, D3D_OK,
            "surface LockRect into a misaligned D3DLOCKED_RECT"
        );
        // SAFETY: LockRect wrote the struct.
        let locked: D3DLOCKED_RECT = unsafe { out.get(0) };
        let texel = locked
            .bits
            .cast::<u8>()
            .wrapping_add(texel_one_one(locked.pitch));
        if write {
            // SAFETY: texel (1, 1) of the locked 4x4 surface.
            unsafe { texel.cast::<u32>().write_unaligned(MARK) };
        } else {
            // SAFETY: as above.
            let read = unsafe { texel.cast::<u32>().read_unaligned() };
            assert_eq!(
                read, MARK,
                "the surface texel written through the first lock"
            );
        }
        assert_eq!(surface.unlock_rect(), D3D_OK, "surface UnlockRect");
    }
    let mut desc = Odd::new();
    // SAFETY: vtable thunk on a live surface; the out struct has room.
    let hr = unsafe { (svt.get_desc)(surface.as_ptr(), desc.ptr()) };
    assert_eq!(hr, D3D_OK, "GetDesc into a misaligned D3DSURFACE_DESC");
    // SAFETY: GetDesc wrote the struct: integers only.
    let d: D3DSURFACE_DESC = unsafe { desc.get(0) };
    assert_eq!(
        (d.format, d.pool, d.width, d.height),
        (D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM, 4, 4),
        "the surface's description",
    );

    let cube = h.create_cube_texture_owned(4, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
    // SAFETY: the cube texture is live for the test.
    let cvt = unsafe { deref_vtbl::<IDirect3DCubeTexture9Vtbl>(cube.as_ptr()) };
    for (flags, write) in [(0, true), (D3DLOCK_READONLY, false)] {
        // SAFETY: vtable thunk on a live cube; the out struct has room, the rect is null.
        let hr =
            unsafe { (cvt.lock_rect)(cube.as_ptr(), 2, 0, out.ptr(), core::ptr::null(), flags) };
        assert_eq!(hr, D3D_OK, "cube LockRect into a misaligned D3DLOCKED_RECT");
        // SAFETY: LockRect wrote the struct.
        let locked: D3DLOCKED_RECT = unsafe { out.get(0) };
        let texel = locked
            .bits
            .cast::<u8>()
            .wrapping_add(texel_one_one(locked.pitch));
        if write {
            // SAFETY: texel (1, 1) of the locked face.
            unsafe { texel.cast::<u32>().write_unaligned(MARK) };
        } else {
            // SAFETY: as above.
            let read = unsafe { texel.cast::<u32>().read_unaligned() };
            assert_eq!(read, MARK, "the cube texel written through the first lock");
        }
        assert_eq!(cube.unlock_rect(2, 0), D3D_OK, "cube UnlockRect");
    }
}

/// `LockBox` writes a misaligned `D3DLOCKED_BOX` and reads a misaligned `D3DBOX`.
///
/// `GetLevelDesc` fills a misaligned `D3DVOLUME_DESC` too.
#[test]
fn volume_lock_box_and_level_desc_through_misaligned_structs() {
    let h = Harness::new();
    let (hr, volume) =
        h.try_create_volume_texture([4, 4, 4], 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
    assert_eq!(hr, D3D_OK, "CreateVolumeTexture");
    let volume = volume.expect("a created volume texture");
    // SAFETY: the volume texture is live for the test.
    let vt = unsafe { deref_vtbl::<IDirect3DVolumeTexture9Vtbl>(volume.as_ptr()) };

    let mut out = Odd::new();
    let mut region = Odd::new();
    let corner = region.put(&[D3DBOX {
        left: 1,
        top: 1,
        right: 3,
        bottom: 3,
        front: 1,
        back: 3,
    }]);
    // SAFETY: vtable thunk on a live volume; out and box point into live scratch.
    let hr = unsafe { (vt.lock_box)(volume.as_ptr(), 0, out.ptr(), corner.cast(), 0) };
    assert_eq!(
        hr, D3D_OK,
        "LockBox of a misaligned D3DBOX into a misaligned D3DLOCKED_BOX"
    );
    // SAFETY: LockBox wrote the struct: integers and a pointer.
    let part: D3DLOCKED_BOX = unsafe { out.get(0) };
    assert!(
        part.row_pitch >= 16,
        "row pitch {} covers a 4-texel row",
        part.row_pitch
    );
    assert!(
        part.slice_pitch >= 4 * part.row_pitch,
        "slice pitch covers four rows"
    );
    // SAFETY: a partial lock's pointer addresses the box's first texel, (1, 1, 1).
    unsafe { part.bits.cast::<u32>().write_unaligned(MARK) };
    assert_eq!(volume.unlock_box(0), D3D_OK, "UnlockBox");

    // SAFETY: vtable thunk on a live volume; the out struct has room, the box is null.
    let hr = unsafe {
        (vt.lock_box)(
            volume.as_ptr(),
            0,
            out.ptr(),
            core::ptr::null(),
            D3DLOCK_READONLY,
        )
    };
    assert_eq!(hr, D3D_OK, "LockBox into a misaligned D3DLOCKED_BOX");
    // SAFETY: LockBox wrote the struct.
    let whole: D3DLOCKED_BOX = unsafe { out.get(0) };
    let offset = usize::try_from(whole.slice_pitch + whole.row_pitch + 4).expect("positive");
    let texel = whole.bits.cast::<u8>().wrapping_add(offset);
    // SAFETY: texel (1, 1, 1) of the locked 4x4x4 level.
    let read = unsafe { texel.cast::<u32>().read_unaligned() };
    assert_eq!(read, MARK, "the texel written through the box lock");
    assert_eq!(volume.unlock_box(0), D3D_OK, "UnlockBox");

    let mut desc = Odd::new();
    // SAFETY: vtable thunk on a live volume; the out struct has room.
    let hr =
        unsafe { (vt.get_level_desc)(volume.as_ptr(), 0, desc.ptr::<D3DVOLUME_DESC>().cast()) };
    assert_eq!(hr, D3D_OK, "GetLevelDesc into a misaligned D3DVOLUME_DESC");
    // SAFETY: GetLevelDesc wrote the struct: integers only.
    let d: D3DVOLUME_DESC = unsafe { desc.get(0) };
    assert_eq!(
        (d.format, d.resource_type, d.width, d.height, d.depth),
        (D3DFMT_A8R8G8B8, D3DRTYPE_VOLUME, 4, 4, 4),
        "the volume level's description",
    );
}

/// The device's state getters and setters read and write misaligned structs and values.
///
/// Viewport, transform, material, light, clip plane, render, texture-stage and sampler state,
/// FVF, and the device caps, which must match what an aligned call reports.
#[test]
fn device_state_round_trips_through_misaligned_structs() {
    let h = Harness::new();
    let vt = device_vtbl(&h);
    let dev = h.device();

    let viewport = D3DVIEWPORT9 {
        x: 3,
        y: 5,
        width: 100,
        height: 60,
        min_z: 0.25,
        max_z: 0.75,
    };
    let mut set = Odd::new();
    let mut get = Odd::new();
    // SAFETY: vtable thunk; the in struct is live scratch.
    let hr = unsafe { (vt.set_viewport)(dev, set.put(&[viewport]).cast()) };
    assert_eq!(hr, D3D_OK, "SetViewport from a misaligned D3DVIEWPORT9");
    // SAFETY: vtable thunk; the out struct has room.
    let hr = unsafe { (vt.get_viewport)(dev, get.ptr::<D3DVIEWPORT9>().cast()) };
    assert_eq!(hr, D3D_OK, "GetViewport into a misaligned D3DVIEWPORT9");
    // SAFETY: GetViewport wrote the struct: integers and floats.
    let v: D3DVIEWPORT9 = unsafe { get.get(0) };
    assert_eq!(
        (
            v.x,
            v.y,
            v.width,
            v.height,
            v.min_z.to_bits(),
            v.max_z.to_bits()
        ),
        (3, 5, 100, 60, 0.25f32.to_bits(), 0.75f32.to_bits()),
        "the viewport round-trips",
    );

    let mut m = [0.0f32; 16];
    for (i, value) in m.iter_mut().enumerate() {
        *value = f32::from(u8::try_from(i).expect("16 entries")) + 0.5;
    }
    // SAFETY: vtable thunk; the in struct is live scratch.
    let hr = unsafe { (vt.set_transform)(dev, D3DTS_WORLD, set.put(&[D3DMATRIX { m }]).cast()) };
    assert_eq!(hr, D3D_OK, "SetTransform from a misaligned D3DMATRIX");
    // SAFETY: vtable thunk; the out struct has room.
    let hr = unsafe { (vt.get_transform)(dev, D3DTS_WORLD, get.ptr::<D3DMATRIX>().cast()) };
    assert_eq!(hr, D3D_OK, "GetTransform into a misaligned D3DMATRIX");
    // SAFETY: GetTransform wrote the struct: floats.
    let got: D3DMATRIX = unsafe { get.get(0) };
    assert_eq!(
        got.m.map(f32::to_bits),
        m.map(f32::to_bits),
        "the world matrix round-trips"
    );

    let colour = |r: f32| D3DCOLORVALUE {
        r,
        g: 0.5,
        b: 0.25,
        a: 1.0,
    };
    let material = D3DMATERIAL9 {
        diffuse: colour(0.1),
        ambient: colour(0.2),
        specular: colour(0.3),
        emissive: colour(0.4),
        power: 7.0,
    };
    // SAFETY: vtable thunk; the in struct is live scratch.
    let hr = unsafe { (vt.set_material)(dev, set.put(&[material]).cast()) };
    assert_eq!(hr, D3D_OK, "SetMaterial from a misaligned D3DMATERIAL9");
    // SAFETY: vtable thunk; the out struct has room.
    let hr = unsafe { (vt.get_material)(dev, get.ptr::<D3DMATERIAL9>().cast()) };
    assert_eq!(hr, D3D_OK, "GetMaterial into a misaligned D3DMATERIAL9");
    // SAFETY: GetMaterial wrote the struct: floats.
    let got: D3DMATERIAL9 = unsafe { get.get(0) };
    assert_eq!(
        (got.emissive.r.to_bits(), got.power.to_bits()),
        (0.4f32.to_bits(), 7.0f32.to_bits()),
        "the material round-trips",
    );

    let light = D3DLIGHT9 {
        type_: D3DLIGHT_POINT,
        diffuse: colour(0.9),
        specular: colour(0.0),
        ambient: colour(0.0),
        position: D3DVECTOR {
            x: 1.0,
            y: 2.0,
            z: 3.0,
        },
        direction: D3DVECTOR::default(),
        range: 50.0,
        falloff: 1.0,
        attenuation0: 1.0,
        attenuation1: 0.0,
        attenuation2: 0.0,
        theta: 0.0,
        phi: 0.0,
    };
    // SAFETY: vtable thunk; the in struct is live scratch.
    let hr = unsafe { (vt.set_light)(dev, 0, set.put(&[light]).cast()) };
    assert_eq!(hr, D3D_OK, "SetLight from a misaligned D3DLIGHT9");
    // SAFETY: vtable thunk; the out struct has room.
    let hr = unsafe { (vt.get_light)(dev, 0, get.ptr::<D3DLIGHT9>().cast()) };
    assert_eq!(hr, D3D_OK, "GetLight into a misaligned D3DLIGHT9");
    // SAFETY: GetLight wrote the struct: integers and floats.
    let got: D3DLIGHT9 = unsafe { get.get(0) };
    assert_eq!(
        (got.type_, got.position.z.to_bits(), got.range.to_bits()),
        (D3DLIGHT_POINT, 3.0f32.to_bits(), 50.0f32.to_bits()),
        "the light round-trips",
    );
    // SAFETY: vtable thunk on a live device.
    let got = unsafe { (vt.light_enable)(dev, 0, 1) };
    assert_eq!(got, D3D_OK, "LightEnable");
    // SAFETY: vtable thunk; the out BOOL has room.
    let hr = unsafe { (vt.get_light_enable)(dev, 0, get.ptr()) };
    assert_eq!(hr, D3D_OK, "GetLightEnable into a misaligned BOOL");
    // SAFETY: GetLightEnable wrote an integer.
    assert_ne!(unsafe { get.get::<i32>(0) }, 0, "the light reads enabled");

    let plane = [0.5f32, -1.0, 2.0, 4.0];
    // SAFETY: vtable thunk; the four floats are live scratch.
    let hr = unsafe { (vt.set_clip_plane)(dev, 0, set.put(&plane)) };
    assert_eq!(hr, D3D_OK, "SetClipPlane from misaligned floats");
    // SAFETY: vtable thunk; four floats of room.
    let hr = unsafe { (vt.get_clip_plane)(dev, 0, get.ptr()) };
    assert_eq!(hr, D3D_OK, "GetClipPlane into misaligned floats");
    // SAFETY: GetClipPlane wrote four floats.
    let got: [f32; 4] = unsafe { get.get(0) };
    assert_eq!(
        got.map(f32::to_bits),
        plane.map(f32::to_bits),
        "the clip plane round-trips"
    );

    assert_eq!(
        h.set_render_state(D3DRS_AMBIENT, 0x0012_3456),
        D3D_OK,
        "SetRenderState"
    );
    assert_eq!(
        h.set_texture_stage_state(0, D3DTSS_COLOROP, D3DTOP_SELECTARG1),
        D3D_OK,
        "TSS"
    );
    assert_eq!(
        h.set_sampler_state(0, D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP),
        D3D_OK,
        "sampler"
    );
    assert_eq!(h.set_fvf(D3DFVF_XYZRHW | D3DFVF_DIFFUSE), D3D_OK, "SetFVF");
    // SAFETY: vtable thunk; the out DWORD has room.
    let hr = unsafe { (vt.get_render_state)(dev, D3DRS_AMBIENT, get.ptr()) };
    assert_eq!(hr, D3D_OK, "GetRenderState into a misaligned DWORD");
    // SAFETY: the getter wrote an integer.
    let got = unsafe { get.get::<u32>(0) };
    assert_eq!(got, 0x0012_3456, "the render state round-trips");
    // SAFETY: vtable thunk; the out DWORD has room.
    let hr = unsafe { (vt.get_texture_stage_state)(dev, 0, D3DTSS_COLOROP, get.ptr()) };
    assert_eq!(hr, D3D_OK, "GetTextureStageState into a misaligned DWORD");
    // SAFETY: the getter wrote an integer.
    let got = unsafe { get.get::<u32>(0) };
    assert_eq!(got, D3DTOP_SELECTARG1, "the stage state round-trips");
    // SAFETY: vtable thunk; the out DWORD has room.
    let hr = unsafe { (vt.get_sampler_state)(dev, 0, D3DSAMP_ADDRESSU, get.ptr()) };
    assert_eq!(hr, D3D_OK, "GetSamplerState into a misaligned DWORD");
    // SAFETY: the getter wrote an integer.
    let got = unsafe { get.get::<u32>(0) };
    assert_eq!(got, D3DTADDRESS_CLAMP, "the sampler state round-trips");
    // SAFETY: vtable thunk; the out DWORD has room.
    let hr = unsafe { (vt.get_fvf)(dev, get.ptr()) };
    assert_eq!(hr, D3D_OK, "GetFVF into a misaligned DWORD");
    // SAFETY: the getter wrote an integer.
    let got = unsafe { get.get::<u32>(0) };
    assert_eq!(got, D3DFVF_XYZRHW | D3DFVF_DIFFUSE, "the FVF round-trips");

    let mut caps = Odd::new();
    // SAFETY: vtable thunk; the out struct has room.
    let hr = unsafe { (vt.get_device_caps)(dev, caps.ptr()) };
    assert_eq!(hr, D3D_OK, "GetDeviceCaps into a misaligned D3DCAPS9");
    let aligned = h.device_caps();
    // SAFETY: the aligned caps are a live value, viewed as its bytes.
    let want = unsafe {
        core::slice::from_raw_parts(
            core::ptr::from_ref(&aligned).cast::<u8>(),
            size_of::<D3DCAPS9>(),
        )
    };
    assert_eq!(
        &caps.bytes[1..=size_of::<D3DCAPS9>()],
        want,
        "the device caps land whole"
    );

    let mut mode = Odd::new();
    // SAFETY: vtable thunk; the out struct has room.
    let hr = unsafe { (vt.get_display_mode)(dev, 0, mode.ptr::<D3DDISPLAYMODE>().cast()) };
    assert_eq!(
        hr, D3D_OK,
        "GetDisplayMode into a misaligned D3DDISPLAYMODE"
    );
    // SAFETY: GetDisplayMode wrote the struct: integers only.
    let got: D3DDISPLAYMODE = unsafe { mode.get(0) };
    assert!(
        got.width > 0 && got.height > 0,
        "the display mode has a size"
    );
}

/// Shader constants, vertex elements and shader tokens move through misaligned arrays.
#[test]
fn arrays_round_trip_through_misaligned_pointers() {
    let h = Harness::new();
    let vt = device_vtbl(&h);
    let dev = h.device();
    let mut set = Odd::new();
    let mut get = Odd::new();

    let rows = [[1.0f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];
    // SAFETY: vtable thunk; two rows of live scratch.
    let hr = unsafe { (vt.set_vertex_shader_constant_f)(dev, 3, set.put(&rows).cast(), 2) };
    assert_eq!(hr, D3D_OK, "SetVertexShaderConstantF from misaligned rows");
    // SAFETY: vtable thunk; two rows of room.
    let hr = unsafe { (vt.get_vertex_shader_constant_f)(dev, 3, get.ptr::<[f32; 4]>().cast(), 2) };
    assert_eq!(hr, D3D_OK, "GetVertexShaderConstantF into misaligned rows");
    for (i, row) in rows.iter().enumerate() {
        // SAFETY: the getter wrote two rows of floats.
        let got: [f32; 4] = unsafe { get.get(i) };
        assert_eq!(
            got.map(f32::to_bits),
            row.map(f32::to_bits),
            "float row {i}"
        );
    }

    let ints = [[1i32, -2, 3, -4]];
    // SAFETY: vtable thunk; one row of live scratch.
    let hr = unsafe { (vt.set_pixel_shader_constant_i)(dev, 1, set.put(&ints).cast(), 1) };
    assert_eq!(hr, D3D_OK, "SetPixelShaderConstantI from a misaligned row");
    // SAFETY: vtable thunk; one row of room.
    let hr = unsafe { (vt.get_pixel_shader_constant_i)(dev, 1, get.ptr::<[i32; 4]>().cast(), 1) };
    assert_eq!(hr, D3D_OK, "GetPixelShaderConstantI into a misaligned row");
    // SAFETY: the getter wrote one row of integers.
    let got = unsafe { get.get::<[i32; 4]>(0) };
    assert_eq!(got, ints[0], "the integer row round-trips");

    let bools = [1i32, 0, 1];
    // SAFETY: vtable thunk; three BOOLs of live scratch.
    let hr = unsafe { (vt.set_pixel_shader_constant_b)(dev, 2, set.put(&bools), 3) };
    assert_eq!(hr, D3D_OK, "SetPixelShaderConstantB from misaligned BOOLs");
    // SAFETY: vtable thunk; three BOOLs of room.
    let hr = unsafe { (vt.get_pixel_shader_constant_b)(dev, 2, get.ptr(), 3) };
    assert_eq!(hr, D3D_OK, "GetPixelShaderConstantB into misaligned BOOLs");
    // SAFETY: the getter wrote three integers.
    let got: [i32; 3] = unsafe { get.get(0) };
    assert_eq!(
        got.map(|b| b != 0),
        [true, false, true],
        "the BOOLs round-trip"
    );

    let elements = [
        D3DVERTEXELEMENT9 {
            stream: 0,
            offset: 0,
            type_: D3DDECLTYPE_FLOAT3,
            method: 0,
            usage: D3DDECLUSAGE_POSITION,
            usage_index: 0,
        },
        D3DVERTEXELEMENT9 {
            stream: D3DDECL_END_STREAM,
            offset: 0,
            type_: D3DDECLTYPE_UNUSED,
            method: 0,
            usage: 0,
            usage_index: 0,
        },
    ];
    let mut decl: *mut c_void = core::ptr::null_mut();
    // SAFETY: vtable thunk; the element array is live scratch, `decl` a writable slot.
    let hr =
        unsafe { (vt.create_vertex_declaration)(dev, set.put(&elements).cast(), &raw mut decl) };
    assert_eq!(
        hr, D3D_OK,
        "CreateVertexDeclaration from a misaligned element array"
    );
    // SAFETY: the declaration was just created.
    let dvt = unsafe { deref_vtbl::<IDirect3DVertexDeclaration9Vtbl>(decl) };
    let mut count = Odd::new();
    // SAFETY: vtable thunk; room for the elements and the count.
    let hr = unsafe { (dvt.get_declaration)(decl, get.ptr(), count.ptr()) };
    assert_eq!(hr, D3D_OK, "GetDeclaration into misaligned arrays");
    // SAFETY: the getter wrote an integer.
    assert_eq!(unsafe { count.get::<u32>(0) }, 2, "the element count");
    for (i, want) in elements.iter().enumerate() {
        // SAFETY: the getter wrote two elements of integers.
        let got: D3DVERTEXELEMENT9 = unsafe { get.get(i) };
        assert_eq!(&got, want, "element {i}");
    }
    // SAFETY: balances the reference CreateVertexDeclaration returned.
    unsafe { (dvt.release)(decl) };

    let mut shader: *mut c_void = core::ptr::null_mut();
    // SAFETY: vtable thunk; the tokens are live scratch, `shader` a writable slot.
    let hr = unsafe { (vt.create_vertex_shader)(dev, set.put(&VS_PASSTHROUGH), &raw mut shader) };
    assert_eq!(hr, D3D_OK, "CreateVertexShader from misaligned tokens");
    // SAFETY: the shader was just created.
    let svt = unsafe { deref_vtbl::<IDirect3DVertexShader9Vtbl>(shader) };
    // SAFETY: vtable thunk; a null buffer asks for the size alone.
    let hr = unsafe { (svt.get_function)(shader, core::ptr::null_mut(), count.ptr()) };
    assert_eq!(hr, D3D_OK, "GetFunction size into a misaligned UINT");
    // SAFETY: the getter wrote an integer.
    let size = unsafe { count.get::<u32>(0) };
    assert_eq!(
        usize::try_from(size).expect("fits"),
        size_of_val(&VS_PASSTHROUGH),
        "token bytes"
    );
    // SAFETY: vtable thunk; `get` has room for the tokens, `count` holds their size.
    let hr = unsafe { (svt.get_function)(shader, get.ptr::<u32>().cast(), count.ptr()) };
    assert_eq!(hr, D3D_OK, "GetFunction into a misaligned buffer");
    for (i, want) in VS_PASSTHROUGH.iter().enumerate() {
        // SAFETY: the getter wrote the tokens.
        assert_eq!(unsafe { get.get::<u32>(i) }, *want, "token {i}");
    }
    // SAFETY: balances the reference CreateVertexShader returned.
    unsafe { (svt.release)(shader) };
}

/// GUIDs and interface out-slots at misaligned addresses.
///
/// `QueryInterface` reads a misaligned IID and writes a misaligned interface slot, `GetTexture`
/// writes a misaligned slot, and the private-data calls take a misaligned GUID and size.
#[test]
fn guids_and_interface_slots_through_misaligned_pointers() {
    let h = Harness::new();
    let vt = device_vtbl(&h);
    let texture = h.create_texture(4, 4, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
    // SAFETY: the texture is live for the test.
    let tvt = unsafe { deref_vtbl::<IDirect3DTexture9Vtbl>(texture.as_ptr()) };
    let mut iid = Odd::new();
    let mut slot = Odd::new();

    // SAFETY: vtable thunk; the IID is live scratch and the slot has room.
    let hr = unsafe {
        (tvt.query_interface)(
            texture.as_ptr(),
            iid.put(&[IID_IDIRECT3DTEXTURE9]),
            slot.ptr(),
        )
    };
    assert_eq!(hr, D3D_OK, "QueryInterface of a misaligned IID");
    // SAFETY: QueryInterface wrote a pointer.
    let same = unsafe { slot.get::<*mut c_void>(0) };
    assert_eq!(
        same,
        texture.as_ptr(),
        "QueryInterface hands back the texture"
    );
    // SAFETY: balances the reference QueryInterface added.
    unsafe { (tvt.release)(same) };

    assert_eq!(h.set_texture(0, &texture), D3D_OK, "SetTexture");
    // SAFETY: vtable thunk; the slot has room.
    let hr = unsafe { (vt.get_texture)(h.device(), 0, slot.ptr()) };
    assert_eq!(hr, D3D_OK, "GetTexture into a misaligned slot");
    // SAFETY: GetTexture wrote a pointer.
    let bound = unsafe { slot.get::<*mut c_void>(0) };
    assert_eq!(
        bound,
        texture.as_ptr(),
        "GetTexture hands back the bound texture"
    );
    // SAFETY: balances the reference GetTexture added.
    unsafe { (tvt.release)(bound) };

    let guid = Guid {
        data1: 0x0BAD_F00D,
        data2: 1,
        data3: 2,
        data4: [3, 4, 5, 6, 7, 8, 9, 10],
    };
    let blob = [0xA5u8; 12];
    let key = iid.put(&[guid]);
    // SAFETY: vtable thunk; the GUID is live scratch and `blob` twelve readable bytes.
    let hr = unsafe { (tvt.set_private_data)(texture.as_ptr(), key, blob.as_ptr().cast(), 12, 0) };
    assert_eq!(hr, D3D_OK, "SetPrivateData under a misaligned GUID");
    let mut size = Odd::new();
    let size_slot = size.put(&[12u32]);
    let mut read = [0u8; 12];
    // SAFETY: vtable thunk; the GUID and size are live scratch, `read` twelve writable bytes.
    let hr = unsafe {
        (tvt.get_private_data)(texture.as_ptr(), key, read.as_mut_ptr().cast(), size_slot)
    };
    assert_eq!(
        hr, D3D_OK,
        "GetPrivateData under a misaligned GUID and size"
    );
    assert_eq!(read, blob, "the private data round-trips");
    // SAFETY: GetPrivateData wrote an integer.
    assert_eq!(unsafe { size.get::<u32>(0) }, 12, "the size it reported");
}

/// `DrawPrimitiveUP` reads a misaligned vertex array, and `Clear` a misaligned rect array.
#[test]
fn draws_read_misaligned_vertex_and_rect_arrays() {
    const BLUE: u32 = 0xFF00_00FF;
    const RED: u32 = 0xFFFF_0000;
    const GREEN: u32 = 0xFF00_FF00;

    let h = Harness::new();
    let vt = device_vtbl(&h);
    assert_eq!(h.set_fvf(D3DFVF_XYZRHW | D3DFVF_DIFFUSE), D3D_OK, "SetFVF");
    // One triangle that covers the whole back buffer.
    let corner = |x: f32, y: f32| RhwVertex {
        x,
        y,
        z: 0.5,
        rhw: 1.0,
        color: RED,
    };
    let triangle = [
        corner(-8.0, -8.0),
        corner(4096.0, -8.0),
        corner(-8.0, 4096.0),
    ];
    let mut vertices = Odd::new();
    let mut rects = Odd::new();
    let stride = u32::try_from(size_of::<RhwVertex>()).expect("a vertex fits u32");

    h.render_once(BLUE, |_| {
        // SAFETY: vtable thunk; three vertices of live scratch at `stride`.
        let hr = unsafe {
            (vt.draw_primitive_up)(
                h.device(),
                D3DPT_TRIANGLELIST,
                1,
                vertices.put(&triangle).cast(),
                stride,
            )
        };
        assert_eq!(hr, D3D_OK, "DrawPrimitiveUP from a misaligned vertex array");
        let rect = D3DRECT {
            x1: 10,
            y1: 10,
            x2: 20,
            y2: 20,
        };
        // SAFETY: vtable thunk; one rect of live scratch.
        let hr = unsafe {
            (vt.clear)(
                h.device(),
                1,
                rects.put(&[rect]).cast(),
                D3DCLEAR_TARGET,
                GREEN,
                1.0,
                0,
            )
        };
        assert_eq!(hr, D3D_OK, "Clear of a misaligned rect array");
    });

    assert_pixel_eq(
        h.read_pixel(320, 240),
        RED,
        "the triangle drawn from misaligned vertices",
    );
    assert_pixel_eq(
        h.read_pixel(15, 15),
        GREEN,
        "inside the misaligned clear rect",
    );
    assert_pixel_eq(h.read_pixel(30, 30), RED, "outside the clear rect");
}

/// `CreateAdditionalSwapChain` writes four resolved fields into misaligned present parameters.
///
/// A windowed request with zero width, height, count and format comes back with the resolved
/// four and every other field as the caller set it, and the chain's `GetPresentParameters`
/// fills a misaligned struct with the same geometry.
#[test]
fn additional_swap_chain_resolves_misaligned_present_parameters() {
    let h = Harness::new();
    let vt = device_vtbl(&h);
    let request = D3DPRESENT_PARAMETERS {
        back_buffer_width: 0,
        back_buffer_height: 0,
        back_buffer_format: 0,
        back_buffer_count: 0,
        multi_sample_type: 0,
        multi_sample_quality: 0,
        swap_effect: D3DSWAPEFFECT_DISCARD,
        device_window: 0,
        windowed: 1,
        enable_auto_depth_stencil: 0,
        auto_depth_stencil_format: 0,
        flags: 0,
        full_screen_refresh_rate_in_hz: 0,
        presentation_interval: D3DPRESENT_INTERVAL_IMMEDIATE,
    };
    let mut params = Odd::new();
    let mut chain: *mut c_void = core::ptr::null_mut();
    // SAFETY: vtable thunk; the struct is live scratch and `chain` a writable slot.
    let hr = unsafe {
        (vt.create_additional_swap_chain)(h.device(), params.put(&[request]).cast(), &raw mut chain)
    };
    assert_eq!(
        hr, D3D_OK,
        "CreateAdditionalSwapChain from a misaligned D3DPRESENT_PARAMETERS"
    );
    // SAFETY: the call wrote the struct: integers only.
    let got: D3DPRESENT_PARAMETERS = unsafe { params.get(0) };
    assert!(
        got.back_buffer_width > 0 && got.back_buffer_height > 0,
        "the zero size resolves to the window's client area, got {}x{}",
        got.back_buffer_width,
        got.back_buffer_height,
    );
    assert_ne!(
        got.back_buffer_format, 0,
        "the unknown format resolves to the display's"
    );
    assert_eq!(got.back_buffer_count, 1, "a zero count resolves to one");
    assert_eq!(
        (
            got.swap_effect,
            got.device_window,
            got.windowed,
            got.presentation_interval
        ),
        (D3DSWAPEFFECT_DISCARD, 0, 1, D3DPRESENT_INTERVAL_IMMEDIATE),
        "the fields D3D9 does not resolve stay as the caller set them",
    );

    // SAFETY: the swap chain was just created.
    let svt = unsafe { deref_vtbl::<IDirect3DSwapChain9Vtbl>(chain) };
    let mut stored = Odd::new();
    // SAFETY: vtable thunk on a live swap chain; the out struct has room.
    let hr = unsafe {
        (svt.get_present_parameters)(chain, stored.ptr::<D3DPRESENT_PARAMETERS>().cast())
    };
    assert_eq!(
        hr, D3D_OK,
        "GetPresentParameters into a misaligned D3DPRESENT_PARAMETERS"
    );
    // SAFETY: the getter wrote the struct: integers only.
    let kept: D3DPRESENT_PARAMETERS = unsafe { stored.get(0) };
    assert_eq!(
        (
            kept.back_buffer_width,
            kept.back_buffer_height,
            kept.back_buffer_count
        ),
        (got.back_buffer_width, got.back_buffer_height, 1),
        "the chain reports the geometry it resolved",
    );
    // SAFETY: balances the reference CreateAdditionalSwapChain returned.
    unsafe { (svt.release)(chain) };
}

/// The factory and device queries fill misaligned structs.
///
/// `IDirect3D9::GetDeviceCaps` byte for byte against the aligned call, `GetAdapterIdentifier`,
/// `GetCreationParameters`, the gamma ramp both ways, and the vertex and index buffer `Lock`
/// out-slot.
#[test]
fn factory_and_device_queries_through_misaligned_structs() {
    let h = Harness::new();
    let vt = device_vtbl(&h);
    let dev = h.device();
    let mut slot = Odd::new();
    // SAFETY: vtable thunk; the slot has room.
    let hr = unsafe { (vt.get_direct3d)(dev, slot.ptr()) };
    assert_eq!(hr, D3D_OK, "GetDirect3D into a misaligned slot");
    // SAFETY: GetDirect3D wrote a pointer.
    let d3d = unsafe { slot.get::<*mut c_void>(0) };
    // SAFETY: GetDirect3D returned a live interface with a reference of its own.
    let fvt = unsafe { deref_vtbl::<IDirect3D9Vtbl>(d3d) };

    let mut caps = Odd::new();
    // SAFETY: vtable thunk on a live interface; the out struct has room.
    let hr = unsafe { (fvt.get_device_caps)(d3d, 0, D3DDEVTYPE_HAL, caps.ptr()) };
    assert_eq!(
        hr, D3D_OK,
        "IDirect3D9::GetDeviceCaps into a misaligned D3DCAPS9"
    );
    let aligned = h.device_caps();
    let start = core::ptr::from_ref(&aligned).cast::<u8>();
    // SAFETY: the aligned caps are a live value, viewed as its bytes.
    let want = unsafe { core::slice::from_raw_parts(start, size_of::<D3DCAPS9>()) };
    assert_eq!(
        &caps.bytes[1..=size_of::<D3DCAPS9>()],
        want,
        "the factory caps land whole"
    );

    let mut ident = Odd::new();
    // SAFETY: vtable thunk on a live interface; the out struct has room.
    let hr = unsafe { (fvt.get_adapter_identifier)(d3d, 0, 0, ident.ptr()) };
    assert_eq!(
        hr, D3D_OK,
        "GetAdapterIdentifier into a misaligned D3DADAPTER_IDENTIFIER9"
    );
    // SAFETY: the getter wrote the struct: byte arrays and integers.
    let id: D3DADAPTER_IDENTIFIER9 = unsafe { ident.get(0) };
    assert_ne!(id.vendor_id, 0, "the identifier names a vendor");
    assert_ne!(
        id.device_name[0], 0,
        "the identifier names the display device"
    );
    // SAFETY: balances the reference GetDirect3D added.
    unsafe { (fvt.release)(d3d) };

    let mut creation = Odd::new();
    // SAFETY: vtable thunk; the out struct has room.
    let hr = unsafe {
        (vt.get_creation_parameters)(dev, creation.ptr::<D3DDEVICE_CREATION_PARAMETERS>().cast())
    };
    assert_eq!(hr, D3D_OK, "GetCreationParameters into a misaligned struct");
    // SAFETY: the getter wrote the struct: integers only.
    let c: D3DDEVICE_CREATION_PARAMETERS = unsafe { creation.get(0) };
    assert_eq!(
        (c.adapter_ordinal, c.device_type),
        (0, D3DDEVTYPE_HAL),
        "the creation adapter and type",
    );
    assert_ne!(
        c.behavior_flags & D3DCREATE_HARDWARE_VERTEXPROCESSING,
        0,
        "the behaviour flags"
    );

    // Red, green and blue each rise end to end, blue less steeply, so the channels differ.
    let mut ramp = [0u16; 768];
    for (i, entry) in ramp.iter_mut().enumerate() {
        let step = u16::try_from(i % 256).expect("a ramp index fits u16");
        *entry = if i < 512 { step * 256 } else { step * 200 };
    }
    let mut set = Odd::new();
    let mut get = Odd::new();
    // SAFETY: vtable thunk; the 768 entries are live scratch.
    unsafe { (vt.set_gamma_ramp)(dev, 0, 0, set.put(&ramp).cast()) };
    // SAFETY: vtable thunk; room for 768 entries.
    unsafe { (vt.get_gamma_ramp)(dev, 0, get.ptr::<u16>().cast()) };
    for (i, want) in ramp.iter().enumerate() {
        // SAFETY: GetGammaRamp wrote 768 integers.
        let got = unsafe { get.get::<u16>(i) };
        assert_eq!(got, *want, "gamma ramp entry {i}");
    }

    let vb = h.create_vertex_buffer(64, 0, 0, D3DPOOL_MANAGED);
    // SAFETY: the buffer is live for the test.
    let bvt = unsafe { deref_vtbl::<IDirect3DVertexBuffer9Vtbl>(vb.as_ptr()) };
    let ib = h.create_index_buffer(64, 0, D3DFMT_INDEX16, D3DPOOL_MANAGED);
    // SAFETY: the buffer is live for the test.
    let ivt = unsafe { deref_vtbl::<IDirect3DIndexBuffer9Vtbl>(ib.as_ptr()) };
    for (flags, write) in [(0, true), (D3DLOCK_READONLY, false)] {
        // SAFETY: vtable thunk on a live buffer; the slot has room.
        let hr = unsafe { (bvt.lock)(vb.as_ptr(), 8, 4, slot.ptr(), flags) };
        assert_eq!(hr, D3D_OK, "vertex buffer Lock into a misaligned slot");
        // SAFETY: Lock wrote a pointer.
        let data = unsafe { slot.get::<*mut c_void>(0) }.cast::<u32>();
        // SAFETY: vtable thunk on a live buffer; the slot has room.
        let hr = unsafe { (ivt.lock)(ib.as_ptr(), 8, 4, slot.ptr(), flags) };
        assert_eq!(hr, D3D_OK, "index buffer Lock into a misaligned slot");
        // SAFETY: Lock wrote a pointer.
        let indices = unsafe { slot.get::<*mut c_void>(0) }.cast::<u32>();
        if write {
            // SAFETY: four bytes of each locked range.
            unsafe { data.write_unaligned(MARK) };
            // SAFETY: as above.
            unsafe { indices.write_unaligned(!MARK) };
        } else {
            // SAFETY: four bytes of each locked range.
            let read = (unsafe { data.read_unaligned() }, unsafe {
                indices.read_unaligned()
            });
            assert_eq!(
                read,
                (MARK, !MARK),
                "the buffers keep what the first locks wrote"
            );
        }
        // SAFETY: vtable thunk on a live, locked buffer.
        let hr = unsafe { (bvt.unlock)(vb.as_ptr()) };
        assert_eq!(hr, D3D_OK, "vertex buffer Unlock");
        // SAFETY: as above.
        let hr = unsafe { (ivt.unlock)(ib.as_ptr()) };
        assert_eq!(hr, D3D_OK, "index buffer Unlock");
    }
}

/// An extended device's `GetDisplayModeEx` reads a misaligned `Size` and fills the struct.
#[test]
fn display_mode_ex_through_misaligned_structs() {
    let h = Harness::create(&HarnessConfig {
        factory: Factory::Extended,
        ..HarnessConfig::default()
    });
    // SAFETY: the harness made an extended device, live for the harness lifetime.
    let vt = unsafe { deref_vtbl::<IDirect3DDevice9ExVtbl>(h.device()) };
    let mut mode = Odd::new();
    let mut rotation = Odd::new();
    let seeded = mode.put(&[D3DDISPLAYMODEEX_SIZE, 0, 0, 0, 0, 0]);
    // SAFETY: vtable thunk; the mode and rotation are live scratch.
    let hr = unsafe { (vt.get_display_mode_ex)(h.device(), 0, seeded.cast(), rotation.ptr()) };
    assert_eq!(
        hr, D3D_OK,
        "GetDisplayModeEx into a misaligned D3DDISPLAYMODEEX"
    );
    // SAFETY: the getter wrote the struct: integers only.
    let got: D3DDISPLAYMODEEX = unsafe { mode.get(0) };
    assert_eq!(got.size, D3DDISPLAYMODEEX_SIZE, "the size field");
    assert!(got.width > 0 && got.height > 0, "the mode has a size");
    assert_eq!(
        got.scan_line_ordering, D3DSCANLINEORDERING_PROGRESSIVE,
        "progressive"
    );
    // SAFETY: the getter wrote an integer.
    let rot = unsafe { rotation.get::<u32>(0) };
    assert_eq!(rot, D3DDISPLAYROTATION_IDENTITY, "the rotation");
}
