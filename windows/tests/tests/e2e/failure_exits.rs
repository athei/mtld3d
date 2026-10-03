//! Requests the device has to refuse or survive, and the device after them.
//!
//! Each test makes a request that used to end the process, leak, or reach
//! Metal with a descriptor it aborts on: a texture or surface past the extent
//! the device reports, a lock rect whose offset overflows a 32-bit pointer, an
//! additional swap chain with nowhere to go. It checks the `HRESULT`, then has
//! the device clear, present and read a pixel back, so a refusal that left
//! something half made, or let the request through to Metal, fails here.

use mtld3d_tests::{Harness, Texture, TexturedVertex};
use mtld3d_types::{
    D3D_OK, D3DERR_INVALIDCALL, D3DERR_NOTAVAILABLE, D3DFMT_A8R8G8B8, D3DFMT_D24S8, D3DFVF_DIFFUSE,
    D3DFVF_TEX1, D3DFVF_XYZ, D3DPOOL_DEFAULT, D3DPOOL_MANAGED, D3DPOOL_SCRATCH, D3DPOOL_SYSTEMMEM,
    D3DPT_TRIANGLESTRIP, D3DRS_LIGHTING, D3DUSAGE_RENDERTARGET,
};

const GREEN: u32 = 0xFF00_FF00;

/// One texel past the 2D extent the device reports in `MaxTextureWidth`.
const PAST_2D_LIMIT: u32 = 16385;
/// Past the 2D extent Metal itself accepts on any supported GPU.
const PAST_METAL_2D_LIMIT: u32 = 65535;
/// One texel past the volume extent the device reports in `MaxVolumeExtent`.
const PAST_VOLUME_LIMIT: u32 = 2049;

/// The device still clears, presents and reads back after a refused request.
fn assert_device_draws(h: &Harness, case: &str) {
    h.render_once(GREEN, |_| {});
    assert_eq!(
        h.read_pixel(4, 4),
        GREEN,
        "{case}: the device draws after it"
    );
}

/// A 2D texture past the device's extent is refused in the pools Metal backs.
///
/// The device reports its limit in `MaxTextureWidth` and `MaxTextureHeight`.
/// A request past it in `D3DPOOL_DEFAULT` or `D3DPOOL_MANAGED` answers
/// `D3DERR_NOTAVAILABLE`, on either axis, for a render target too, and for
/// one past what Metal could make at all, and the device draws on. A
/// `D3DPOOL_SYSTEMMEM` or `D3DPOOL_SCRATCH` texture has no Metal texture to
/// size and creates.
#[test]
fn a_texture_past_the_extent_limit_is_refused_in_the_gpu_pools() {
    let h = Harness::new();
    for (width, height, usage, pool, case) in [
        (PAST_2D_LIMIT, 16, 0, D3DPOOL_DEFAULT, "wide DEFAULT"),
        (16, PAST_2D_LIMIT, 0, D3DPOOL_MANAGED, "tall MANAGED"),
        (
            PAST_METAL_2D_LIMIT,
            16,
            0,
            D3DPOOL_DEFAULT,
            "past Metal's limit",
        ),
        (
            PAST_METAL_2D_LIMIT,
            16,
            D3DUSAGE_RENDERTARGET,
            D3DPOOL_DEFAULT,
            "render target",
        ),
    ] {
        let (hr, texture) = h.try_create_texture(width, height, 1, usage, D3DFMT_A8R8G8B8, pool);
        assert_eq!(hr, D3DERR_NOTAVAILABLE, "{case}: CreateTexture is refused");
        assert!(texture.is_null(), "{case}: no texture is handed out");
        assert_device_draws(&h, case);
    }
    for (pool, case) in [
        (D3DPOOL_SYSTEMMEM, "SYSTEMMEM"),
        (D3DPOOL_SCRATCH, "SCRATCH"),
    ] {
        let (hr, texture) = h.try_create_texture(PAST_2D_LIMIT, 4, 1, 0, D3DFMT_A8R8G8B8, pool);
        assert_eq!(
            hr, D3D_OK,
            "{case}: a CPU-only texture past the limit creates"
        );
        drop(Texture::from_raw(texture));
        assert_device_draws(&h, case);
    }
}

/// A cube texture whose edge is past the device's extent is refused before its faces are made.
#[test]
fn a_cube_texture_past_the_extent_limit_is_refused() {
    let h = Harness::new();
    for pool in [D3DPOOL_DEFAULT, D3DPOOL_MANAGED] {
        let (hr, texture) = h.try_create_cube_texture(PAST_2D_LIMIT, 1, 0, D3DFMT_A8R8G8B8, pool);
        assert_eq!(hr, D3DERR_NOTAVAILABLE, "pool {pool}: CreateCubeTexture");
        assert!(
            texture.is_null(),
            "pool {pool}: no cube texture is handed out"
        );
        assert_device_draws(&h, "cube");
    }
}

/// A volume texture past the device's volume extent is refused.
///
/// A volume more than one slice deep is held to `MaxVolumeExtent` on every
/// axis, the depth as well as the width and the height. One slice deep it
/// is a 2D texture and takes the 2D limit.
#[test]
fn a_volume_texture_past_the_extent_limit_is_refused() {
    let h = Harness::new();
    for (extent, case) in [
        ([PAST_VOLUME_LIMIT, 4, 4], "width"),
        ([4, PAST_VOLUME_LIMIT, 4], "height"),
        ([4, 4, PAST_VOLUME_LIMIT], "depth"),
        ([PAST_2D_LIMIT, 4, 1], "one slice past the 2D limit"),
    ] {
        let (hr, texture) =
            h.try_create_volume_texture(extent, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
        assert_eq!(hr, D3DERR_NOTAVAILABLE, "{case}: CreateVolumeTexture");
        assert!(texture.is_none(), "{case}: no volume texture is handed out");
        assert_device_draws(&h, case);
    }
    let (hr, texture) =
        h.try_create_volume_texture([4096, 4, 1], 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
    assert_eq!(hr, D3D_OK, "one slice within the 2D limit creates");
    drop(texture);
    assert_device_draws(&h, "one slice within the 2D limit");
}

/// A standalone surface past the extent Metal accepts is refused with `D3DERR_INVALIDCALL`.
#[test]
fn a_surface_past_the_extent_limit_is_refused() {
    let h = Harness::new();
    let (hr, surface) =
        h.create_render_target_ms_hr((PAST_METAL_2D_LIMIT, 16), D3DFMT_A8R8G8B8, (0, 0), 0);
    assert_eq!(hr, D3DERR_INVALIDCALL, "CreateRenderTarget");
    assert!(surface.is_none());
    assert_device_draws(&h, "render target");
    let (hr, surface) =
        h.create_depth_stencil_surface_ms_hr((16, PAST_METAL_2D_LIMIT), D3DFMT_D24S8, (0, 0));
    assert_eq!(hr, D3DERR_INVALIDCALL, "CreateDepthStencilSurface");
    assert!(surface.is_none());
    assert_device_draws(&h, "depth stencil");
    let (hr, surface) = h.create_offscreen_plain_surface_seeded(
        PAST_METAL_2D_LIMIT,
        16,
        D3DFMT_A8R8G8B8,
        D3DPOOL_DEFAULT,
    );
    assert_eq!(
        hr, D3DERR_INVALIDCALL,
        "DEFAULT CreateOffscreenPlainSurface"
    );
    assert!(surface.is_null());
    assert_device_draws(&h, "offscreen plain");
}

/// A system-memory texture past Metal's extent, bound for sampling, does not end the process.
///
/// It creates, since its pool needs no Metal texture, but a sampling bind
/// asks for one. That request is refused, the stage samples nothing, and the
/// device draws on.
#[test]
fn sampling_a_system_memory_texture_past_the_extent_limit_draws_on() {
    let h = Harness::new();
    let texture = h.create_texture(
        PAST_METAL_2D_LIMIT,
        4,
        1,
        0,
        D3DFMT_A8R8G8B8,
        D3DPOOL_SYSTEMMEM,
    );
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), D3D_OK);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1), D3D_OK);
    assert_eq!(h.set_texture(0, &texture), D3D_OK);
    let quad = [(-1.0, 1.0), (1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)].map(|(x, y)| TexturedVertex {
        x,
        y,
        z: 0.5,
        color: 0xFFFF_FFFF,
        u: 0.0,
        v: 0.0,
    });
    h.render_once(GREEN, |d| {
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLESTRIP, 2, &quad), D3D_OK);
    });
    assert_eq!(h.clear_texture(0), D3D_OK);
    assert_device_draws(&h, "after sampling");
}

/// A system-memory lock rect far past the surface returns a pointer rather than ending the process.
///
/// A `D3DPOOL_SYSTEMMEM` surface accepts any lock rect and hands back the
/// pointer at its unclamped origin, which the caller does not dereference. An
/// origin 16M rows down is past what a 32-bit pointer offset holds; it wraps
/// as the pointer arithmetic does, and the surface unlocks and is usable.
#[test]
fn a_far_off_system_memory_lock_rect_wraps_instead_of_overflowing() {
    let h = Harness::new();
    let surface = h.create_offscreen_plain_surface(64, 64, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    let (hr, null_bits) = surface.lock_rect_partial_probe(&[0, 0x0100_0000, 4, 0x0100_0004], 0);
    assert_eq!(hr, D3D_OK, "the far-off rect locks");
    assert!(!null_bits, "a pointer is handed back");
    assert_eq!(surface.unlock_rect(), D3D_OK);
    let locked = surface.lock_rect(0);
    assert_eq!(locked.pitch(), 256, "a whole-surface lock after it");
    drop(locked);
    assert_device_draws(&h, "lock");
}

/// `CreateAdditionalSwapChain` with nowhere to put the chain makes none.
///
/// The call answers `D3DERR_INVALIDCALL` and takes no device reference, so
/// the device's count is where it was and its final `Release` still tears it
/// down.
#[test]
fn an_additional_swap_chain_with_a_null_output_is_refused() {
    let h = Harness::new();
    let before = h.device_refcount();
    assert_eq!(h.additional_swapchain_null_output_hr(), D3DERR_INVALIDCALL);
    assert_eq!(h.device_refcount(), before, "no device reference is taken");
    assert_device_draws(&h, "swap chain");
}
