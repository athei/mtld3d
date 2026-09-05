//! Device-owned implicit render-target / backbuffer / depth-stencil surfaces.
//!
//! `GetRenderTarget(0)`, `GetBackBuffer(0)` and `GetDepthStencilSurface` each
//! return a single cached, device-owned object: the same pointer every call,
//! `GetRenderTarget(0) == GetBackBuffer(0)`, surviving its refcount reaching
//! zero (destroyed only at device teardown), and resolving its extent live from
//! the device so a `Reset` that recreates the backbuffer is reflected without
//! re-allocating the surface.

use mtld3d_tests::{Harness, HarnessConfig};
use mtld3d_types::{
    D3D_OK, D3DERR_INVALIDCALL, D3DLOCK_READONLY, D3DPRESENTFLAG_LOCKABLE_BACKBUFFER,
};

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
fn release_dc_on_a_lockable_backbuffer_reaches_the_back_buffer() {
    // The DC over a lockable back buffer wraps a read-back snapshot rather than
    // the back buffer's own pixels, so it owes the surface coherence in both
    // directions: it shows what the GPU painted before it, and what GDI draws
    // into it reaches the back buffer at `ReleaseDC`, with no Present in
    // between. Every coordinate here is the reported one, so the test also
    // stands under `make test SCALE=<n>`.
    const GREEN: u32 = 0xFF00_FF00;
    const RED: u32 = 0xFFFF_0000;
    const GREEN_COLORREF: u32 = 0x0000_FF00;
    const RED_COLORREF: u32 = 0x0000_00FF;
    let h = Harness::with_lockable_back_buffer();

    assert_eq!(h.clear_target(GREEN), D3D_OK, "clear the back buffer green");
    let bb = h.back_buffer(0);
    let dc = bb.dc();
    assert_eq!(
        dc.get_pixel(320, 240),
        GREEN_COLORREF,
        "the DC reads the colour the Clear painted",
    );
    dc.fill_block(64, RED_COLORREF);
    assert_eq!(dc.release(), D3D_OK, "ReleaseDC");

    // Alpha is masked off: GDI leaves the fourth byte at zero, but a
    // `render.scale` below 100% returns the frame through the MetalFX resolve,
    // which hands back an opaque one whatever the surface holds. The claim
    // here is about the colour GDI drew, not about the byte it did not write.
    assert_eq!(
        h.read_pixel(16, 16) | 0xFF00_0000,
        RED,
        "what GDI drew into the DC reaches the back buffer",
    );
    assert_eq!(
        h.read_pixel(320, 240),
        GREEN,
        "the pixels GDI left alone still hold the clear colour",
    );
}

#[test]
fn release_dc_on_a_lockable_backbuffer_resamples_under_a_render_scale() {
    // `render.scale` rasterizes the back buffer smaller than the extent `GetDC`
    // hands the DIB out at, so the write-back has to resample on the way in.
    // Pinning the key here runs that path in every test run rather than only in
    // the scaled sweep; a machine without MetalFX holds the scale at 1.0 and
    // takes the direct upload, which the same assertions cover.
    const GREEN: u32 = 0xFF00_FF00;
    const RED: u32 = 0xFFFF_0000;
    const RED_COLORREF: u32 = 0x0000_00FF;
    let h = Harness::create(&HarnessConfig {
        present_flags: D3DPRESENTFLAG_LOCKABLE_BACKBUFFER,
        config_entries: "render.scale=0.75",
        ..HarnessConfig::default()
    });

    assert_eq!(h.clear_target(GREEN), D3D_OK, "clear the back buffer green");
    let bb = h.back_buffer(0);
    let dc = bb.dc();
    dc.fill_block(64, RED_COLORREF);
    assert_eq!(dc.release(), D3D_OK, "ReleaseDC");

    // Deep inside the block on both sides of the round trip, so the linear
    // downscale and the resolve back up both read only red neighbours.
    assert_eq!(
        h.read_pixel(16, 16) | 0xFF00_0000,
        RED,
        "the write-back resamples GDI's drawing into the scaled back buffer",
    );
    assert_eq!(
        h.read_pixel(320, 240),
        GREEN,
        "the pixels GDI left alone still hold the clear colour",
    );
}

#[test]
fn read_only_lock_rect_on_a_non_lockable_backbuffer_reads_the_rendered_pixels() {
    // D3D9 gives a backbuffer created without `D3DPRESENTFLAG_LOCKABLE_BACKBUFFER`
    // no CPU access at all and rejects every `LockRect` of it. A read-only lock
    // is accepted here and served by a GPU read-back instead, because that is
    // the shape of the screenshot and character-portrait paths titles drive
    // through the backbuffer. A lock that asks to write is still rejected, and
    // so is a lock of any other non-lockable render target.
    const WIDTH: u32 = 640;
    const FILL: u32 = 0xFF20_4080;
    let h = Harness::new();
    assert_eq!(h.clear_target(FILL), 0, "clear the backbuffer");
    let backbuffer = h.back_buffer(0);

    let (hr, bits_null) = backbuffer.lock_rect_probe(0);
    assert_eq!(
        hr, D3DERR_INVALIDCALL,
        "a writable lock of a non-lockable backbuffer must return INVALIDCALL"
    );
    assert!(
        !bits_null,
        "a rejected LockRect must leave the caller's D3DLOCKED_RECT untouched"
    );
    assert_eq!(
        backbuffer.unlock_rect(),
        D3DERR_INVALIDCALL,
        "UnlockRect without a lock held must return INVALIDCALL"
    );

    let locked = backbuffer.lock_rect(D3DLOCK_READONLY);
    assert_eq!(
        locked.pitch().cast_unsigned(),
        WIDTH * 4,
        "the read-back page steps by the backbuffer format's row pitch"
    );
    assert_eq!(
        locked.as_u32(1)[0],
        FILL,
        "the read-back must show the cleared backbuffer"
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
