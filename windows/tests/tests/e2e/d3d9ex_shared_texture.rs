//! Shared `D3DPOOL_DEFAULT` textures of an extended device.
//!
//! `CreateTexture` with a `pSharedHandle` on an extended device's default
//! pool opens the shared texture whose handle the slot holds. A slot holding
//! null asks for a new shared texture, which is not available, and so are the
//! shapes no shared texture has; a slot that names no shared resource is an
//! invalid call. Where the Wine's Direct3D 11 is DXMT, the runtime whose
//! handles this layer opens, a Direct3D 11 device creates the textures to
//! open: what it draws an extended device samples and reads back through the
//! texture it opened. Elsewhere those tests say on stdout that they did not
//! run.

use core::ffi::c_void;

use mtld3d_tests::{D3d11Device, D3d11Texture, Harness, SharedHandle, Texture};
use mtld3d_types::{
    D3D_OK, D3DERR_INVALIDCALL, D3DERR_NOTAVAILABLE, D3DFMT_A8R8G8B8, D3DFMT_A16B16G16R16F,
    D3DFMT_DXT1, D3DFMT_L8, D3DFMT_X8R8G8B8, D3DKMT_GLOBAL_HANDLE_MASK, D3DLOCK_READONLY,
    D3DPOOL_DEFAULT, D3DPOOL_SYSTEMMEM, D3DUSAGE_DYNAMIC, D3DUSAGE_RENDERTARGET,
};

use super::d3d9ex_resources::{extended, sample_center};

const SIZE: u32 = 64;
const RED: u32 = 0xFFFF_0000;
const GREEN: u32 = 0xFF00_FF00;
const BLUE: u32 = 0xFF00_00FF;
/// A global handle value no resource was created with.
const UNKNOWN_HANDLE: usize = 0x7fff_fff0;

/// `CreateTexture` on `h` with a slot that holds `handle`.
fn try_open(
    h: &Harness,
    size: (u32, u32),
    levels: u32,
    usage: u32,
    format: u32,
    handle: *mut c_void,
) -> (i32, Option<Texture<'_>>) {
    let mut slot = handle;
    let answer = {
        // SAFETY: on a default-pool create the slot's value is a handle, which
        // the create looks up; nothing is read through it.
        let shared = unsafe { SharedHandle::to(&mut slot) };
        h.try_create_texture_shared_usage(size, levels, usage, format, D3DPOOL_DEFAULT, &shared)
    };
    assert_eq!(slot, handle, "the create leaves the slot as it was");
    answer
}

/// Open the shared texture `handle` names on `h`.
fn open_shared(h: &Harness, usage: u32, format: u32, handle: *mut c_void) -> Texture<'_> {
    let (hr, texture) = try_open(h, (SIZE, SIZE), 1, usage, format, handle);
    assert_eq!(hr, D3D_OK, "an open of a shared handle");
    texture.expect("an open hands out a texture")
}

/// The centre texel of `texture`'s level, read back through `h`.
fn read_back(h: &Harness, texture: &Texture<'_>) -> u32 {
    let level = texture.surface_level(0);
    let sysmem = h.create_offscreen_plain_surface(SIZE, SIZE, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    assert_eq!(
        h.get_render_target_data_hr(&level, &sysmem),
        D3D_OK,
        "GetRenderTargetData of a shared render target"
    );
    let lock = sysmem.lock_rect(D3DLOCK_READONLY);
    let row = usize::try_from(SIZE).expect("u32 fits usize");
    let center = row * (row / 2) + row / 2;
    lock.as_u32(center + 1)[center]
}

/// Clear `texture`, a render target of `h`, to `color` and wait for the GPU.
fn fill(h: &Harness, texture: &Texture<'_>, color: u32) {
    let level = texture.surface_level(0);
    let back_buffer = h.render_target(0);
    assert_eq!(h.set_render_target(0, &level), D3D_OK, "bind the texture");
    assert_eq!(h.clear_target(color), D3D_OK, "clear the texture");
    assert_eq!(
        h.set_render_target(0, &back_buffer),
        D3D_OK,
        "bind the back buffer again"
    );
    assert_eq!(read_back(h, texture), color, "the device's own read-back");
}

/// Sample `texture` on `h` until its centre reads `color`, for a writer nothing waits on.
///
/// A Direct3D 11 clear is submitted when its flush returns, not finished, and
/// nothing orders it before this device's commands. So the sample is repeated
/// for a bounded number of frames; the last reading is returned either way.
fn sample_until(h: &Harness, texture: &Texture<'_>, color: u32) -> u32 {
    let mut seen = 0;
    for _ in 0..200 {
        seen = sample_center(h, texture);
        if seen == color {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    seen
}

/// The process's Direct3D 11 device when it is DXMT's, with a note when it is not.
fn dxmt_device() -> Option<D3d11Device> {
    let Some(device) = D3d11Device::create() else {
        println!("[e2e] no Direct3D 11 hardware device here; the shared open is not run");
        return None;
    };
    if !device.is_dxmt() {
        println!("[e2e] the Direct3D 11 here is not DXMT; the shared open is not run");
        return None;
    }
    Some(device)
}

/// A shared Direct3D 11 render target of `SIZE` and the handle `GetSharedHandle` gives for it.
fn shared_d3d11_texture(d3d11: &D3d11Device) -> (D3d11Texture, *mut c_void) {
    let (texture, handle) = d3d11
        .create_shared_texture(SIZE, SIZE)
        .expect("a Direct3D 11 shared texture and its handle");
    assert!(
        u32::try_from(handle).is_ok_and(|handle| handle & D3DKMT_GLOBAL_HANDLE_MASK != 0),
        "GetSharedHandle hands out a D3DKMT global handle, got {handle:#x}"
    );
    (texture, core::ptr::without_provenance_mut::<c_void>(handle))
}

/// A slot holding null asks for a new shared texture, which only the other runtime creates.
#[test]
fn a_null_handle_slot_is_not_available() {
    let h = extended();
    for (levels, usage, format, what) in [
        (1, D3DUSAGE_RENDERTARGET, D3DFMT_A8R8G8B8, "a render target"),
        (1, 0, D3DFMT_X8R8G8B8, "a plain texture"),
        (0, 0, D3DFMT_A8R8G8B8, "a full mip chain"),
        (1, D3DUSAGE_DYNAMIC, D3DFMT_A8R8G8B8, "a dynamic texture"),
        (1, 0, D3DFMT_L8, "L8"),
    ] {
        let (hr, texture) = try_open(
            &h,
            (SIZE, SIZE),
            levels,
            usage,
            format,
            core::ptr::null_mut(),
        );
        assert_eq!(
            (hr, texture.is_none()),
            (D3DERR_NOTAVAILABLE, true),
            "{what}"
        );
    }
    h.render_once(GREEN, |_| {});
    assert_eq!(
        h.read_pixel(1, 1),
        GREEN,
        "the device renders after the refusals"
    );
}

/// The shapes no shared texture has are refused before the handle is looked at.
#[test]
fn an_open_of_another_shape_is_not_available() {
    let h = extended();
    let handle = core::ptr::without_provenance_mut::<c_void>(UNKNOWN_HANDLE);
    for (levels, usage, format, what) in [
        (0, 0, D3DFMT_A8R8G8B8, "a full mip chain"),
        (3, 0, D3DFMT_A8R8G8B8, "three levels"),
        (1, D3DUSAGE_DYNAMIC, D3DFMT_A8R8G8B8, "a dynamic texture"),
        (1, 0, D3DFMT_L8, "L8"),
        (1, 0, D3DFMT_DXT1, "DXT1"),
        (
            1,
            D3DUSAGE_RENDERTARGET,
            D3DFMT_A16B16G16R16F,
            "A16B16G16R16F",
        ),
    ] {
        let (hr, texture) = try_open(&h, (SIZE, SIZE), levels, usage, format, handle);
        assert_eq!(
            (hr, texture.is_none()),
            (D3DERR_NOTAVAILABLE, true),
            "{what}"
        );
    }
}

/// A slot that names no shared resource is an invalid call.
#[test]
fn an_open_of_no_shared_resource_is_an_invalid_call() {
    let h = extended();
    for (stray, what) in [
        (0x1f4_usize, "an NT handle value"),
        (0x3fff_fff0, "a value with neither top bit"),
        (
            UNKNOWN_HANDLE,
            "a global handle value nothing was created with",
        ),
    ] {
        let stray = core::ptr::without_provenance_mut::<c_void>(stray);
        let (hr, texture) = try_open(&h, (SIZE, SIZE), 1, 0, D3DFMT_A8R8G8B8, stray);
        assert_eq!(
            (hr, texture.is_none()),
            (D3DERR_INVALIDCALL, true),
            "{what}"
        );
    }
    h.render_once(BLUE, |_| {});
    assert_eq!(
        h.read_pixel(1, 1),
        BLUE,
        "the device renders after the refusals"
    );
}

/// A Direct3D 11 texture's legacy handle opens on an extended device as the same pixels.
#[test]
fn a_direct3d_11_shared_texture_opens_on_an_extended_device() {
    let h = extended();
    let Some(d3d11) = dxmt_device() else {
        return;
    };
    let (texture, handle) = shared_d3d11_texture(&d3d11);
    let opened = open_shared(&h, 0, D3DFMT_A8R8G8B8, handle);

    assert!(
        d3d11.clear(&texture, [1.0, 0.0, 0.0, 1.0]),
        "a render-target view"
    );
    assert_eq!(sample_until(&h, &opened, RED), RED, "what Direct3D 11 drew");
    assert!(
        d3d11.clear(&texture, [0.0, 0.0, 1.0, 1.0]),
        "a render-target view"
    );
    assert_eq!(
        sample_until(&h, &opened, BLUE),
        BLUE,
        "its next frame, through the same open"
    );
}

/// A texture opened as a render target is drawn into and read back by the extended device too.
#[test]
fn an_opened_render_target_takes_what_either_runtime_draws() {
    let h = extended();
    let Some(d3d11) = dxmt_device() else {
        return;
    };
    let (texture, handle) = shared_d3d11_texture(&d3d11);
    let opened = open_shared(&h, D3DUSAGE_RENDERTARGET, D3DFMT_A8R8G8B8, handle);

    fill(&h, &opened, GREEN);
    assert_eq!(sample_center(&h, &opened), GREEN, "what the device drew");
    assert!(
        d3d11.clear(&texture, [1.0, 0.0, 0.0, 1.0]),
        "a render-target view"
    );
    assert_eq!(
        sample_until(&h, &opened, RED),
        RED,
        "what Direct3D 11 drew over it"
    );
    assert_eq!(read_back(&h, &opened), RED, "the same, read back");
}

/// Several opens of one handle, on one device and on another, are one storage.
#[test]
fn one_handle_opens_more_than_once() {
    let first_device = extended();
    let second_device = extended();
    let Some(d3d11) = dxmt_device() else {
        return;
    };
    let (texture, handle) = shared_d3d11_texture(&d3d11);
    let first = open_shared(&first_device, 0, D3DFMT_A8R8G8B8, handle);
    let second = open_shared(&first_device, 0, D3DFMT_X8R8G8B8, handle);
    let painter = open_shared(
        &second_device,
        D3DUSAGE_RENDERTARGET,
        D3DFMT_A8R8G8B8,
        handle,
    );

    assert!(
        d3d11.clear(&texture, [0.0, 0.0, 1.0, 1.0]),
        "a render-target view"
    );
    assert_eq!(
        sample_until(&first_device, &first, BLUE),
        BLUE,
        "the first open"
    );
    assert_eq!(
        sample_until(&first_device, &second, BLUE),
        BLUE,
        "the second open"
    );
    // The other device's read-back in `fill` has its clear finished on the
    // GPU before the first device samples.
    fill(&second_device, &painter, GREEN);
    assert_eq!(
        sample_center(&first_device, &first),
        GREEN,
        "what another extended device rendered through its own open"
    );
}

/// An open released before any frame carried it leaves its device working.
#[test]
fn an_opened_texture_released_before_its_first_frame_leaves_the_device_working() {
    let h = extended();
    let Some(d3d11) = dxmt_device() else {
        return;
    };
    let (texture, handle) = shared_d3d11_texture(&d3d11);
    drop(open_shared(
        &h,
        D3DUSAGE_RENDERTARGET,
        D3DFMT_A8R8G8B8,
        handle,
    ));
    h.render_once(GREEN, |_| {});
    assert_eq!(h.read_pixel(1, 1), GREEN);

    // The handle still opens, and the open sees what is drawn after it.
    let opened = open_shared(&h, 0, D3DFMT_A8R8G8B8, handle);
    assert!(
        d3d11.clear(&texture, [1.0, 0.0, 0.0, 1.0]),
        "a render-target view"
    );
    assert_eq!(sample_until(&h, &opened, RED), RED);
}

/// An open at another extent than the shared texture's is an invalid call.
#[test]
fn an_open_at_another_extent_is_an_invalid_call() {
    let h = extended();
    let Some(d3d11) = dxmt_device() else {
        return;
    };
    let (texture, handle) = shared_d3d11_texture(&d3d11);
    for size in [(SIZE / 2, SIZE), (SIZE, SIZE * 2)] {
        let (hr, opened) = try_open(&h, size, 1, 0, D3DFMT_A8R8G8B8, handle);
        assert_eq!(
            (hr, opened.is_none()),
            (D3DERR_INVALIDCALL, true),
            "{}x{}",
            size.0,
            size.1
        );
    }

    // The refusals left the handle as it was.
    let opened = open_shared(&h, 0, D3DFMT_A8R8G8B8, handle);
    assert!(
        d3d11.clear(&texture, [0.0, 1.0, 0.0, 1.0]),
        "a render-target view"
    );
    assert_eq!(sample_until(&h, &opened, GREEN), GREEN);
}
