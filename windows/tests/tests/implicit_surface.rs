//! Device-owned implicit render-target / backbuffer / depth-stencil surfaces.
//!
//! `GetRenderTarget(0)`, `GetBackBuffer(0)` and `GetDepthStencilSurface` each
//! return a single cached, device-owned object: the same pointer every call,
//! `GetRenderTarget(0) == GetBackBuffer(0)`, surviving its refcount reaching
//! zero (destroyed only at device teardown), and resolving its extent live from
//! the device so a `Reset` that recreates the backbuffer is reflected without
//! re-allocating the surface.

use mtld3d_tests::Harness;
use mtld3d_types::D3DERR_INVALIDCALL;

#[test]
fn implicit_render_target_is_cached_and_aliases_backbuffer() {
    let h = Harness::new();

    let rt1 = h.render_target(0);
    let rt2 = h.render_target(0);
    assert_eq!(
        rt1.as_ptr(),
        rt2.as_ptr(),
        "GetRenderTarget(0) must return the one cached implicit surface every call"
    );

    let bb = h.back_buffer(0);
    assert_eq!(
        rt1.as_ptr(),
        bb.as_ptr(),
        "GetRenderTarget(0) and GetBackBuffer(0) are the same device-owned object"
    );
}

#[test]
fn implicit_render_target_survives_refcount_zero() {
    let h = Harness::new();

    // Take the cached pointer, then release every reference to it.
    let cached = {
        let rt = h.render_target(0);
        rt.as_ptr()
    };

    // Device-owned: it is NOT freed at refcount 0, so re-acquiring returns the
    // very same object (D3D9 never re-allocates the implicit render target).
    let rt_again = h.render_target(0);
    assert_eq!(
        rt_again.as_ptr(),
        cached,
        "the implicit render target must persist past refcount 0"
    );

    // Still live + usable: its description resolves the current backbuffer size.
    let (hr, desc) = rt_again.desc();
    assert_eq!(hr, 0, "GetDesc on the re-acquired implicit RT");
    assert_eq!((desc.width, desc.height), (640, 480), "live extent");
}

#[test]
fn implicit_render_target_extent_tracks_reset_live() {
    let h = Harness::new();

    let before = h.render_target(0).as_ptr();

    let hr = h.reset(320, 240);
    assert_eq!(hr, 0, "Reset(320x240) failed: 0x{hr:08X}");

    // Identity is stable across Reset (the cached surface is never re-allocated),
    // while its extent resolves LIVE from the recreated backbuffer — proving the
    // surface does not snapshot a now-freed Metal handle.
    let rt = h.render_target(0);
    assert_eq!(
        rt.as_ptr(),
        before,
        "implicit RT identity must survive Reset"
    );
    let (hr, desc) = rt.desc();
    assert_eq!(hr, 0, "GetDesc after Reset");
    assert_eq!(
        (desc.width, desc.height),
        (320, 240),
        "implicit RT extent must track the post-Reset backbuffer (live resolution)"
    );
}

#[test]
fn get_dc_on_non_lockable_backbuffer_rejects_and_preserves_out() {
    let h = Harness::new();

    // The default backbuffer is non-lockable, so `GetDC` rejects with
    // `INVALIDCALL` and must leave the caller's out `HDC` untouched. Seed the
    // out slot with a sentinel and assert it survives the rejected call.
    let sentinel = 0xdead_beef_usize as *mut core::ffi::c_void;
    let (hr, out) = h.back_buffer(0).get_dc(sentinel);
    assert_eq!(
        hr, D3DERR_INVALIDCALL,
        "GetDC on a non-lockable backbuffer must return INVALIDCALL"
    );
    assert_eq!(
        out, sentinel,
        "a rejected GetDC must not write through the out HDC"
    );
}

#[test]
fn implicit_depth_stencil_is_cached() {
    let h = Harness::with_depth();

    let ds1 = h
        .depth_stencil_surface()
        .expect("auto depth-stencil present");
    let ds2 = h
        .depth_stencil_surface()
        .expect("auto depth-stencil present");
    assert_eq!(
        ds1.as_ptr(),
        ds2.as_ptr(),
        "GetDepthStencilSurface must return the one cached implicit surface"
    );
}
