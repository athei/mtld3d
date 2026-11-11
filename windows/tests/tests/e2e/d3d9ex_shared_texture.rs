//! Shared `D3DPOOL_DEFAULT` textures of an extended device.
//!
//! `CreateTexture` with a `pSharedHandle` on an extended device's default
//! pool shares the texture: a slot holding null receives the handle of a new
//! shared texture, and a slot holding a handle opens the texture it names.
//! Two extended devices stand in for the two runtimes that share: what one
//! renders into the texture it created, the other samples and reads back
//! through the texture it opened, and the other way round. Where the Wine's
//! Direct3D 11 is DXMT, the same holds between an extended device and a
//! Direct3D 11 device, in both directions. The shapes no shared texture has
//! are refused, and so is a slot that names no shared texture.

use core::ffi::c_void;

use mtld3d_tests::{D3d11Device, Harness, SharedHandle, Texture};
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
/// Whether `value` is a D3DKMT global handle: 32 bits wide with one of its two top bits set.
fn is_global_handle(value: usize) -> bool {
    u32::try_from(value).is_ok_and(|handle| handle & D3DKMT_GLOBAL_HANDLE_MASK != 0)
}

/// Create a shared texture on `h`, returning it and the handle the create wrote.
fn create_shared(h: &Harness, usage: u32, format: u32) -> (Texture<'_>, *mut c_void) {
    let mut slot: *mut c_void = core::ptr::null_mut();
    let (hr, texture) = {
        // SAFETY: the slot holds null, a request to share, so nothing is read through it.
        let handle = unsafe { SharedHandle::to(&mut slot) };
        h.try_create_texture_shared_usage((SIZE, SIZE), 1, usage, format, D3DPOOL_DEFAULT, &handle)
    };
    assert_eq!(hr, D3D_OK, "a shared create");
    let texture = texture.expect("a shared create hands out a texture");
    assert!(
        is_global_handle(slot.addr()),
        "the create writes a D3DKMT global handle, got {slot:p}"
    );
    (texture, slot)
}

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
    assert_eq!(slot, handle, "an open leaves the handle in its slot");
    answer
}

/// Open the shared texture `handle` names on `h`.
fn open_shared(h: &Harness, usage: u32, format: u32, handle: *mut c_void) -> Texture<'_> {
    let (hr, texture) = try_open(h, (SIZE, SIZE), 1, usage, format, handle);
    assert_eq!(hr, D3D_OK, "an open of a shared handle");
    texture.expect("an open hands out a texture")
}

/// The centre texel of `texture`'s level, read back through `h`.
///
/// The read-back waits for the device's submitted work, so everything `h`
/// rendered into the texture before it has finished on the GPU when this
/// returns, which is what orders it before the other device's commands.
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

/// What one device renders into a texture it created, another reads through the one it opened.
#[test]
fn a_render_target_created_on_one_device_opens_on_another_as_the_same_pixels() {
    let producer = extended();
    let consumer = extended();
    let (created, handle) = create_shared(&producer, D3DUSAGE_RENDERTARGET, D3DFMT_A8R8G8B8);
    let opened = open_shared(&consumer, D3DUSAGE_RENDERTARGET, D3DFMT_A8R8G8B8, handle);

    fill(&producer, &created, RED);
    assert_eq!(
        read_back(&consumer, &opened),
        RED,
        "the opener reads back what the creator rendered"
    );
    assert_eq!(
        sample_center(&consumer, &opened),
        RED,
        "the opener samples what the creator rendered"
    );

    fill(&consumer, &opened, BLUE);
    assert_eq!(
        read_back(&producer, &created),
        BLUE,
        "the creator reads back what the opener rendered"
    );
    assert_eq!(
        sample_center(&producer, &created),
        BLUE,
        "the creator samples what the opener rendered"
    );
}

/// A texture created without a usage is still the storage another device renders into.
#[test]
fn a_plain_shared_texture_samples_what_the_opening_device_rendered() {
    let owner = extended();
    let painter = extended();
    let (created, handle) = create_shared(&owner, 0, D3DFMT_X8R8G8B8);
    let opened = open_shared(&painter, D3DUSAGE_RENDERTARGET, D3DFMT_X8R8G8B8, handle);

    fill(&painter, &opened, GREEN);
    assert_eq!(sample_center(&owner, &created), GREEN);
    fill(&painter, &opened, RED);
    assert_eq!(
        sample_center(&owner, &created),
        RED,
        "a second frame of the other device reaches the same texture"
    );
}

/// Several opens of one handle are one storage.
#[test]
fn one_handle_opens_more_than_once() {
    let producer = extended();
    let consumer = extended();
    let (created, handle) = create_shared(&producer, D3DUSAGE_RENDERTARGET, D3DFMT_A8R8G8B8);
    let first = open_shared(&consumer, 0, D3DFMT_A8R8G8B8, handle);
    let second = open_shared(&consumer, 0, D3DFMT_A8R8G8B8, handle);

    fill(&producer, &created, BLUE);
    assert_eq!(sample_center(&consumer, &first), BLUE, "the first open");
    assert_eq!(sample_center(&consumer, &second), BLUE, "the second open");
}

/// A texture released before any frame carried it leaves its device working.
#[test]
fn a_shared_texture_released_before_its_first_frame_leaves_the_device_working() {
    let h = extended();
    let consumer = extended();
    let (created, handle) = create_shared(&h, D3DUSAGE_RENDERTARGET, D3DFMT_A8R8G8B8);
    let opened = open_shared(&consumer, 0, D3DFMT_A8R8G8B8, handle);
    drop(opened);
    drop(created);
    h.render_once(GREEN, |_| {});
    assert_eq!(h.read_pixel(1, 1), GREEN);
    consumer.render_once(BLUE, |_| {});
    assert_eq!(consumer.read_pixel(1, 1), BLUE);
}

/// The creator's release retires the handle; a texture opened before it keeps its storage.
#[test]
fn a_handle_stops_opening_once_its_creator_released_the_texture() {
    let producer = extended();
    let consumer = extended();
    let (created, handle) = create_shared(&producer, D3DUSAGE_RENDERTARGET, D3DFMT_A8R8G8B8);
    let opened = open_shared(&consumer, D3DUSAGE_RENDERTARGET, D3DFMT_A8R8G8B8, handle);
    fill(&producer, &created, RED);
    drop(created);

    let (hr, late) = try_open(&consumer, (SIZE, SIZE), 1, 0, D3DFMT_A8R8G8B8, handle);
    assert_eq!(
        (hr, late.is_none()),
        (D3DERR_INVALIDCALL, true),
        "an open after the creator's release"
    );
    assert_eq!(
        sample_center(&consumer, &opened),
        RED,
        "the opened texture keeps what was rendered"
    );
    fill(&consumer, &opened, GREEN);
    assert_eq!(sample_center(&consumer, &opened), GREEN);
}

/// The shapes no shared texture has are refused before anything is shared.
#[test]
fn a_shared_create_of_another_shape_is_not_available() {
    let h = extended();
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
        let mut slot: *mut c_void = core::ptr::null_mut();
        let (hr, texture) = {
            // SAFETY: the slot holds null, so nothing is read through it.
            let handle = unsafe { SharedHandle::to(&mut slot) };
            h.try_create_texture_shared_usage(
                (SIZE, SIZE),
                levels,
                usage,
                format,
                D3DPOOL_DEFAULT,
                &handle,
            )
        };
        assert_eq!(
            (hr, texture.is_none()),
            (D3DERR_NOTAVAILABLE, true),
            "{what}"
        );
        assert!(slot.is_null(), "{what}: a refused create writes no handle");
    }
}

/// A slot that names no shared texture of the requested shape is an invalid call.
#[test]
fn an_open_of_no_shared_texture_is_an_invalid_call() {
    let producer = extended();
    let consumer = extended();
    let (created, handle) = create_shared(&producer, D3DUSAGE_RENDERTARGET, D3DFMT_A8R8G8B8);

    for (stray, what) in [
        (0x1f4_usize, "an NT handle value"),
        (0x3fff_fff0, "a value with neither top bit"),
        (
            0x7fff_fff0,
            "a global handle value nothing was created with",
        ),
    ] {
        let stray = core::ptr::without_provenance_mut::<c_void>(stray);
        let (hr, texture) = try_open(&consumer, (SIZE, SIZE), 1, 0, D3DFMT_A8R8G8B8, stray);
        assert_eq!(
            (hr, texture.is_none()),
            (D3DERR_INVALIDCALL, true),
            "{what}"
        );
    }

    let (hr, texture) = try_open(&consumer, (SIZE / 2, SIZE), 1, 0, D3DFMT_A8R8G8B8, handle);
    assert_eq!(
        (hr, texture.is_none()),
        (D3DERR_INVALIDCALL, true),
        "another extent than the shared texture's"
    );
    let (hr, texture) = try_open(&consumer, (SIZE, SIZE), 2, 0, D3DFMT_A8R8G8B8, handle);
    assert_eq!(
        (hr, texture.is_none()),
        (D3DERR_NOTAVAILABLE, true),
        "two levels"
    );
    let (hr, texture) = try_open(&consumer, (SIZE, SIZE), 1, 0, D3DFMT_L8, handle);
    assert_eq!(
        (hr, texture.is_none()),
        (D3DERR_NOTAVAILABLE, true),
        "a format that is not shared"
    );

    // The refusals left the handle as it was.
    let opened = open_shared(&consumer, 0, D3DFMT_A8R8G8B8, handle);
    fill(&producer, &created, BLUE);
    assert_eq!(sample_center(&consumer, &opened), BLUE);
}

/// A Direct3D 11 texture's legacy handle opens on an extended device as the same pixels.
///
/// Runs where the Wine's Direct3D 11 is DXMT, the runtime whose shared
/// handles this layer opens, and says so on stdout where it is not.
#[test]
fn a_direct3d_11_shared_texture_opens_on_an_extended_device() {
    let h = extended();
    let Some(d3d11) = dxmt_device() else {
        return;
    };
    let (texture, handle) = d3d11
        .create_shared_texture(SIZE, SIZE)
        .expect("a Direct3D 11 shared texture and its handle");
    assert!(
        is_global_handle(handle),
        "GetSharedHandle hands out a D3DKMT global handle, got {handle:#x}"
    );
    let handle = core::ptr::without_provenance_mut::<c_void>(handle);
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

/// A texture an extended device created shared opens with `OpenSharedResource`.
///
/// Runs where the Wine's Direct3D 11 is DXMT, as the test above.
#[test]
fn a_shared_texture_of_an_extended_device_opens_in_direct3d_11() {
    let h = extended();
    let Some(d3d11) = dxmt_device() else {
        return;
    };
    let (created, handle) = create_shared(&h, D3DUSAGE_RENDERTARGET, D3DFMT_A8R8G8B8);
    let opened = d3d11
        .open_shared_texture(handle.addr())
        .expect("OpenSharedResource takes the handle CreateTexture wrote");

    assert!(
        d3d11.clear(&opened, [0.0, 1.0, 0.0, 1.0]),
        "a render-target view of the opened texture"
    );
    assert_eq!(
        sample_until(&h, &created, GREEN),
        GREEN,
        "what Direct3D 11 drew"
    );
}

/// The process's Direct3D 11 device when it is DXMT's, with a note when it is not.
fn dxmt_device() -> Option<D3d11Device> {
    let Some(device) = D3d11Device::create() else {
        println!("[e2e] no Direct3D 11 hardware device here; the cross-API share is not run");
        return None;
    };
    if !device.is_dxmt() {
        println!("[e2e] the Direct3D 11 here is not DXMT; the cross-API share is not run");
        return None;
    }
    Some(device)
}
