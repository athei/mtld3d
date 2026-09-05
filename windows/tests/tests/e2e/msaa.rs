//! Multisample anti-aliasing: the swap chain, standalone surfaces and the resolve.
//!
//! The observable property throughout is coverage. A diagonal edge drawn into a
//! multisampled target resolves to intermediate pixels along the edge; the same
//! draw into a single-sampled target has none, every pixel being fully inside
//! or fully outside. Each test that renders counts those intermediate pixels
//! along one scanline, so nothing depends on where exactly the rasterizer puts
//! the edge or on the device's sample positions.

use mtld3d_tests::{
    Harness, HarnessConfig, Rgba8, RhwVertex, Surface, Texture, TexturedVertex, assert_pixel_eq,
};
use mtld3d_types::{
    D3D_OK, D3DCLEAR_TARGET, D3DCLEAR_ZBUFFER, D3DCMP_ALWAYS, D3DCMP_LESS, D3DERR_INVALIDCALL,
    D3DERR_NOTAVAILABLE, D3DFMT_A8R8G8B8, D3DFMT_D16, D3DFMT_D24S8, D3DFMT_DXT1, D3DFMT_INTZ,
    D3DFMT_X8R8G8B8, D3DFVF_DIFFUSE, D3DFVF_TEX1, D3DFVF_XYZ, D3DFVF_XYZRHW, D3DLOCK_READONLY,
    D3DMULTISAMPLE_2_SAMPLES, D3DMULTISAMPLE_4_SAMPLES, D3DMULTISAMPLE_NONE,
    D3DMULTISAMPLE_NONMASKABLE, D3DPOOL_DEFAULT, D3DPOOL_SYSTEMMEM, D3DPT_TRIANGLELIST,
    D3DRS_LIGHTING, D3DRS_MULTISAMPLEMASK, D3DRS_POINTSIZE, D3DRS_ZENABLE, D3DRS_ZFUNC,
    D3DRS_ZWRITEENABLE, D3DSAMP_ADDRESSU, D3DSAMP_ADDRESSV, D3DSAMP_MAGFILTER, D3DSAMP_MINFILTER,
    D3DTADDRESS_CLAMP, D3DTEXF_NONE, D3DTEXF_POINT, D3DUSAGE_DEPTHSTENCIL,
};

/// Edge of the standalone render targets, small enough to keep the readback cheap.
const RT_SIZE: u32 = 64;
/// [`RT_SIZE`] as the vertex positions state it.
const RT_SIZE_F: f32 = 64.0;

const BLACK: u32 = 0xFF00_0000;
const WHITE: u32 = 0xFFFF_FFFF;
const BLUE: u32 = 0xFF00_00FF;

/// A windowed device with the given swap-chain multisample type and depth format.
fn harness(multi_sample_type: u32, depth_format: Option<u32>) -> Harness {
    Harness::create(&HarnessConfig {
        depth_format,
        multi_sample_type,
        ..HarnessConfig::default()
    })
}

/// The lower-left half of a `width`×`height` target, in `color`, at depth `z`.
///
/// The hypotenuse runs corner to corner, so the band of pixels it crosses is
/// partially covered: exactly the band [`count_intermediate`] counts.
const fn diagonal(width: f32, height: f32, z: f32, color: u32) -> [RhwVertex; 3] {
    let (w, h) = (width, height);
    [
        RhwVertex {
            x: 0.0,
            y: 0.0,
            z,
            rhw: 1.0,
            color,
        },
        RhwVertex {
            x: w,
            y: 0.0,
            z,
            rhw: 1.0,
            color,
        },
        RhwVertex {
            x: 0.0,
            y: h,
            z,
            rhw: 1.0,
            color,
        },
    ]
}

/// Arm the fixed-function pipeline for an unlit pre-transformed draw.
fn arm(h: &Harness) {
    assert_eq!(h.set_fvf(D3DFVF_XYZRHW | D3DFVF_DIFFUSE), 0, "SetFVF");
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0, "lighting off");
}

/// A pixel that is well inside the triangle, clear of all three of its edges.
///
/// Not pixel zero: the triangle's own left edge runs down the target's edge, so
/// the leftmost column is half covered and resolves to half intensity.
const INSIDE_X: u32 = 4;

/// Pixels that are neither the background nor the fill.
///
/// A single-sampled rasterizer writes every pixel either fully or not at all,
/// so the count is zero; a multisampled one resolves the partially covered
/// pixels along the edge to something in between. Reads only the red channel,
/// which the black-to-white contrast makes the whole signal.
fn count_intermediate(pixels: &[u32]) -> usize {
    pixels
        .iter()
        .filter(|&&p| {
            let r = Rgba8::from_pixel(p).r;
            r > 16 && r < 239
        })
        .count()
}

/// The back buffer's extent as the pre-transformed vertex positions state it.
///
/// `f32::from` rather than an `as` cast: the dimensions are u16-range in
/// practice, and the conversion has to stay exact for the triangle to land on
/// the target's corners.
fn back_buffer_extent(h: &Harness) -> (f32, f32) {
    let (width, height) = h.dims();
    (
        f32::from(u16::try_from(width).expect("back-buffer width fits u16")),
        f32::from(u16::try_from(height).expect("back-buffer height fits u16")),
    )
}

/// Read the back buffer's middle scanline.
///
/// `StretchRect` into a single-sampled staging target first: D3D9 rejects
/// `GetRenderTargetData` on a multisampled surface, and the resolve is the
/// step the application is expected to take instead. Doing it unconditionally
/// keeps the multisampled and the single-sampled reads on one path.
fn back_buffer_row(h: &Harness) -> Vec<u32> {
    let (width, height) = h.dims();
    let staging = h.create_render_target(width, height, D3DFMT_X8R8G8B8);
    let back = h.back_buffer(0);
    assert_eq!(
        h.stretch_rect(&back, &staging, D3DTEXF_NONE),
        0,
        "StretchRect the back buffer into the staging target"
    );
    surface_row(h, &staging, (width, height))
}

/// Read the middle scanline of an `RT_SIZE`-square render target.
///
/// The surface must be single-sampled; a multisampled one is resolved with
/// `StretchRect` first, as D3D9 requires.
fn render_target_row(h: &Harness, rt: &Surface<'_>) -> Vec<u32> {
    surface_row(h, rt, (RT_SIZE, RT_SIZE))
}

/// Read the middle scanline of a single-sampled render target of `size`.
fn surface_row(h: &Harness, rt: &Surface<'_>, size: (u32, u32)) -> Vec<u32> {
    let (width, height) = size;
    let sysmem =
        h.create_offscreen_plain_surface(width, height, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    assert_eq!(
        h.get_render_target_data_hr(rt, &sysmem),
        0,
        "GetRenderTargetData"
    );
    let locked = sysmem.lock_rect(D3DLOCK_READONLY);
    let pitch_px = locked.pitch().cast_unsigned() / 4;
    let base = ((height / 2) * pitch_px) as usize;
    let row = locked.as_u32(base + width as usize);
    row[base..base + width as usize].to_vec()
}

/// Render the diagonal into `rt`, which must be `RT_SIZE` square.
fn draw_diagonal_into(h: &Harness, rt: &Surface<'_>) {
    arm(h);
    assert_eq!(h.set_render_target(0, rt), 0, "SetRenderTarget");
    assert_eq!(h.begin_scene(), 0, "BeginScene");
    assert_eq!(h.clear_target(BLACK), 0, "Clear");
    assert_eq!(
        h.draw_primitive_up(
            D3DPT_TRIANGLELIST,
            1,
            &diagonal(RT_SIZE_F, RT_SIZE_F, 0.5, WHITE)
        ),
        0,
        "DrawPrimitiveUP",
    );
    assert_eq!(h.end_scene(), 0, "EndScene");
}

// ── CheckDeviceMultiSampleType ──

#[test]
fn check_device_multi_sample_type_answers_the_device() {
    let h = Harness::factory_only();

    let (hr, levels) = h.check_device_multi_sample_type(D3DFMT_X8R8G8B8, 1, D3DMULTISAMPLE_NONE);
    assert_eq!(hr, D3D_OK, "NONE is always available");
    assert_eq!(levels, 1, "a maskable level has exactly one quality level");

    // Metal guarantees a sample count of 4 on every GPU family mtld3d runs on,
    // so this is an unconditional answer rather than a device-dependent one.
    let (hr, levels) =
        h.check_device_multi_sample_type(D3DFMT_X8R8G8B8, 1, D3DMULTISAMPLE_4_SAMPLES);
    assert_eq!(hr, D3D_OK, "4x colour");
    assert_eq!(levels, 1, "a maskable level has exactly one quality level");
    assert_eq!(
        h.check_device_multi_sample_type(D3DFMT_D24S8, 1, D3DMULTISAMPLE_4_SAMPLES)
            .0,
        D3D_OK,
        "4x depth"
    );

    // NONMASKABLE reports how many rungs its quality ladder has; quality `q`
    // means `1 << q` samples, so 4x support alone puts the count at three.
    let (hr, levels) =
        h.check_device_multi_sample_type(D3DFMT_X8R8G8B8, 1, D3DMULTISAMPLE_NONMASKABLE);
    assert_eq!(hr, D3D_OK, "NONMASKABLE");
    assert!(
        levels >= 3,
        "at least quality 0..2 (1x, 2x, 4x), got {levels}"
    );
}

#[test]
fn check_device_multi_sample_type_rejects_malformed_and_unsupported() {
    let h = Harness::factory_only();

    // 3 and 15 are inside `D3DMULTISAMPLE_TYPE` but name counts no hardware
    // offers, so they are merely unavailable; 17 is outside the enumeration
    // and malformed, as is `D3DFMT_UNKNOWN`.
    assert_eq!(
        h.check_device_multi_sample_type(D3DFMT_X8R8G8B8, 1, 3).0,
        D3DERR_NOTAVAILABLE,
        "3 samples"
    );
    assert_eq!(
        h.check_device_multi_sample_type(D3DFMT_X8R8G8B8, 1, 15).0,
        D3DERR_NOTAVAILABLE,
        "15 samples"
    );
    assert_eq!(
        h.check_device_multi_sample_type(D3DFMT_X8R8G8B8, 1, 17).0,
        D3DERR_INVALIDCALL,
        "17 samples"
    );
    assert_eq!(
        h.check_device_multi_sample_type(0, 1, D3DMULTISAMPLE_NONE)
            .0,
        D3DERR_INVALIDCALL,
        "D3DFMT_UNKNOWN"
    );

    // A format whose whole point is per-sample readback, and a
    // block-compressed one that is not renderable at all.
    assert_eq!(
        h.check_device_multi_sample_type(D3DFMT_INTZ, 1, D3DMULTISAMPLE_4_SAMPLES)
            .0,
        D3DERR_NOTAVAILABLE,
        "multisampled INTZ"
    );
    assert_eq!(
        h.check_device_multi_sample_type(D3DFMT_DXT1, 1, D3DMULTISAMPLE_2_SAMPLES)
            .0,
        D3DERR_NOTAVAILABLE,
        "multisampled DXT1"
    );
    // The same format still answers the "is this usable at all" probe.
    assert_eq!(
        h.check_device_multi_sample_type(D3DFMT_INTZ, 1, D3DMULTISAMPLE_NONE)
            .0,
        D3D_OK,
        "single-sampled INTZ"
    );
}

// ── Surface creation ──

#[test]
fn create_render_target_honours_the_multisample_type() {
    let h = harness(D3DMULTISAMPLE_NONE, None);
    let rt = h.create_render_target_ms(
        (RT_SIZE, RT_SIZE),
        D3DFMT_X8R8G8B8,
        (D3DMULTISAMPLE_4_SAMPLES, 0),
    );
    let (hr, desc) = rt.desc();
    assert_eq!(hr, 0, "GetDesc on the multisampled render target");
    assert_eq!(
        desc.multi_sample_type, D3DMULTISAMPLE_4_SAMPLES,
        "GetDesc reports the type the surface was created with"
    );
    assert_eq!(desc.multi_sample_quality, 0, "quality round-trips");

    // A lockable multisampled render target has no meaning: the lock is
    // defined against a single-sample surface.
    assert_eq!(
        h.create_render_target_ms_hr(
            (RT_SIZE, RT_SIZE),
            D3DFMT_X8R8G8B8,
            (D3DMULTISAMPLE_4_SAMPLES, 0),
            1,
        )
        .0,
        D3DERR_INVALIDCALL,
        "lockable + multisampled"
    );
    // A sample count no device can serve is rejected at create time too.
    assert_eq!(
        h.create_render_target_ms_hr((RT_SIZE, RT_SIZE), D3DFMT_X8R8G8B8, (3, 0), 0)
            .0,
        D3DERR_INVALIDCALL,
        "3 samples"
    );
}

#[test]
fn create_depth_stencil_surface_honours_the_multisample_type() {
    let h = harness(D3DMULTISAMPLE_NONE, None);
    let (hr, ds) = h.create_depth_stencil_surface_ms_hr(
        (RT_SIZE, RT_SIZE),
        D3DFMT_D24S8,
        (D3DMULTISAMPLE_4_SAMPLES, 0),
    );
    assert_eq!(hr, 0, "CreateDepthStencilSurface(4x)");
    let ds = ds.expect("multisampled depth surface");
    let (hr, desc) = ds.desc();
    assert_eq!(hr, 0, "GetDesc on the multisampled depth surface");
    assert_eq!(
        desc.multi_sample_type, D3DMULTISAMPLE_4_SAMPLES,
        "GetDesc reports the type"
    );
}

// ── The resolve ──

#[test]
fn a_single_sampled_render_target_has_no_partial_coverage() {
    // The control for `multisampled_render_target_resolves_the_edge`: the same
    // draw with no multisampling writes only fully-covered pixels.
    let h = harness(D3DMULTISAMPLE_NONE, None);
    let rt = h.create_render_target(RT_SIZE, RT_SIZE, D3DFMT_X8R8G8B8);
    draw_diagonal_into(&h, &rt);
    let row = render_target_row(&h, &rt);
    assert_eq!(
        count_intermediate(&row),
        0,
        "a single-sampled edge is hard: {row:02X?}"
    );
}

#[test]
fn multisampled_render_target_resolves_the_edge() {
    let h = harness(D3DMULTISAMPLE_NONE, None);
    let rt = h.create_render_target_ms(
        (RT_SIZE, RT_SIZE),
        D3DFMT_X8R8G8B8,
        (D3DMULTISAMPLE_4_SAMPLES, 0),
    );
    draw_diagonal_into(&h, &rt);

    // D3D9 makes the application resolve a multisampled surface with
    // `StretchRect` before reading it back, which is what the resolve fills.
    let plain = h.create_render_target(RT_SIZE, RT_SIZE, D3DFMT_X8R8G8B8);
    assert_eq!(
        h.get_render_target_data_hr(
            &rt,
            &h.create_offscreen_plain_surface(RT_SIZE, RT_SIZE, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM)
        ),
        D3DERR_INVALIDCALL,
        "GetRenderTargetData rejects a multisampled source"
    );
    assert_eq!(h.stretch_rect(&rt, &plain, D3DTEXF_NONE), 0, "resolve");
    let row = render_target_row(&h, &plain);
    assert!(
        count_intermediate(&row) > 0,
        "a 4x edge resolves to partial coverage: {row:02X?}"
    );
    assert_pixel_eq(
        row[INSIDE_X as usize],
        WHITE,
        "a pixel clear of every edge is fully covered",
    );
    assert_pixel_eq(row[RT_SIZE as usize - 1], BLACK, "and the last one is not");
}

#[test]
fn multisampled_back_buffer_presents_a_resolved_edge() {
    let h = harness(D3DMULTISAMPLE_4_SAMPLES, None);
    let bb = h.back_buffer(0);
    let (hr, desc) = bb.desc();
    assert_eq!(hr, 0, "GetDesc on the multisampled back buffer");
    assert_eq!(
        desc.multi_sample_type, D3DMULTISAMPLE_4_SAMPLES,
        "the back buffer reports the swap chain's type"
    );
    drop(bb);

    let (width_f, height_f) = back_buffer_extent(&h);
    arm(&h);
    h.render_once(BLACK, |d| {
        assert_eq!(
            d.draw_primitive_up(
                D3DPT_TRIANGLELIST,
                1,
                &diagonal(width_f, height_f, 0.5, WHITE)
            ),
            0,
            "DrawPrimitiveUP",
        );
    });
    let row = back_buffer_row(&h);
    assert!(
        count_intermediate(&row) > 0,
        "the presented back buffer carries the resolved edge"
    );
}

#[test]
fn stretch_rect_resolves_a_multisampled_source() {
    let h = harness(D3DMULTISAMPLE_NONE, None);
    let msaa = h.create_render_target_ms(
        (RT_SIZE, RT_SIZE),
        D3DFMT_X8R8G8B8,
        (D3DMULTISAMPLE_4_SAMPLES, 0),
    );
    let plain = h.create_render_target(RT_SIZE, RT_SIZE, D3DFMT_X8R8G8B8);
    draw_diagonal_into(&h, &msaa);
    assert_eq!(
        h.stretch_rect(&msaa, &plain, D3DTEXF_NONE),
        0,
        "StretchRect from a multisampled source"
    );

    let row = render_target_row(&h, &plain);
    assert!(
        count_intermediate(&row) > 0,
        "the copy carries the resolved edge: {row:02X?}"
    );

    // The reverse direction spreads each source pixel over every sample, so
    // the round trip comes back exactly as it went in.
    let back = h.create_render_target(RT_SIZE, RT_SIZE, D3DFMT_X8R8G8B8);
    assert_eq!(
        h.stretch_rect(&plain, &msaa, D3DTEXF_NONE),
        0,
        "StretchRect into a multisampled destination"
    );
    assert_eq!(
        h.stretch_rect(&msaa, &back, D3DTEXF_NONE),
        0,
        "and back out"
    );
    let round_trip = render_target_row(&h, &back);
    assert_eq!(
        round_trip, row,
        "a copy through a multisampled surface changes nothing"
    );
}

#[test]
fn depth_test_holds_on_a_multisampled_target() {
    let h = harness(D3DMULTISAMPLE_4_SAMPLES, Some(D3DFMT_D24S8));
    let (width_f, height_f) = back_buffer_extent(&h);
    arm(&h);
    assert_eq!(h.set_render_state(D3DRS_ZENABLE, 1), 0, "depth test on");

    // A near white triangle, then the same triangle further away: the second
    // draw must fail the depth test and leave the first one's pixels, edge
    // included.
    assert!(h.pump(), "WM_QUIT before render");
    assert_eq!(h.begin_scene(), 0, "BeginScene");
    assert_eq!(
        h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, BLACK, 1.0, 0),
        0,
        "clear colour + depth"
    );
    assert_eq!(
        h.draw_primitive_up(
            D3DPT_TRIANGLELIST,
            1,
            &diagonal(width_f, height_f, 0.2, WHITE)
        ),
        0,
        "near draw",
    );
    assert_eq!(
        h.draw_primitive_up(
            D3DPT_TRIANGLELIST,
            1,
            &diagonal(width_f, height_f, 0.8, BLUE)
        ),
        0,
        "far draw",
    );
    assert_eq!(h.end_scene(), 0, "EndScene");
    assert_eq!(h.present(), 0, "Present");

    let row = back_buffer_row(&h);
    assert!(
        count_intermediate(&row) > 0,
        "the depth-tested 4x edge still resolves"
    );
    assert_pixel_eq(
        row[INSIDE_X as usize],
        WHITE,
        "the near draw's white survives the occluded blue one",
    );
}

#[test]
fn a_multisampled_depth_surface_binds_beside_a_multisampled_target() {
    // A depth surface created at the target's sample count is the pairing
    // Metal accepts; the draw below would be dropped if the pass had been
    // rejected for disagreeing attachments.
    let h = harness(D3DMULTISAMPLE_4_SAMPLES, None);
    let (width, height) = h.dims();
    let (width_f, height_f) = back_buffer_extent(&h);
    let ds = h
        .create_depth_stencil_surface_ms_hr(
            (width, height),
            D3DFMT_D16,
            (D3DMULTISAMPLE_4_SAMPLES, 0),
        )
        .1
        .expect("multisampled depth surface");
    assert_eq!(
        h.set_depth_stencil_surface(&ds),
        0,
        "SetDepthStencilSurface(4x)"
    );
    arm(&h);
    assert_eq!(h.set_render_state(D3DRS_ZENABLE, 1), 0, "depth test on");
    assert!(h.pump(), "WM_QUIT before render");
    assert_eq!(h.begin_scene(), 0, "BeginScene");
    assert_eq!(
        h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, BLACK, 1.0, 0),
        0,
        "clear colour + depth"
    );
    assert_eq!(
        h.draw_primitive_up(
            D3DPT_TRIANGLELIST,
            1,
            &diagonal(width_f, height_f, 0.5, WHITE)
        ),
        0,
        "DrawPrimitiveUP",
    );
    assert_eq!(h.end_scene(), 0, "EndScene");
    assert_eq!(h.present(), 0, "Present");

    let row = back_buffer_row(&h);
    assert!(
        count_intermediate(&row) > 0,
        "the draw reached a 4x colour + 4x depth pass"
    );
}

#[test]
fn a_single_sampled_depth_surface_drops_beside_a_multisampled_target() {
    // D3D9 leaves the depth-stencil surface bound across `SetRenderTarget`, so
    // a multisampled target can land beside the single-sampled depth the
    // device was created with. Metal takes a pass's sample count from its
    // attachments and rejects one whose attachments disagree, so the pass
    // drops the depth attachment. Everything else built for that pass has to
    // agree: a pipeline that still declared a depth format would be rejected
    // at the draw ("For depth attachment, the renderPipelineState pixelFormat
    // must be MTLPixelFormatInvalid, as no texture is set"), and the draw
    // would never reach the target.
    let h = harness(D3DMULTISAMPLE_NONE, Some(D3DFMT_D24S8));
    let (width, height) = h.dims();
    let (width_f, height_f) = back_buffer_extent(&h);
    let rt = h.create_render_target_ms(
        (width, height),
        D3DFMT_X8R8G8B8,
        (D3DMULTISAMPLE_4_SAMPLES, 0),
    );
    assert_eq!(h.set_render_target(0, &rt), 0, "SetRenderTarget(4x)");
    arm(&h);
    assert_eq!(h.set_render_state(D3DRS_ZENABLE, 1), 0, "depth test on");

    assert!(h.pump(), "WM_QUIT before render");
    assert_eq!(h.begin_scene(), 0, "BeginScene");
    assert_eq!(
        h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, BLACK, 1.0, 0),
        0,
        "clear colour + depth"
    );
    assert_eq!(
        h.draw_primitive_up(
            D3DPT_TRIANGLELIST,
            1,
            &diagonal(width_f, height_f, 0.5, WHITE)
        ),
        0,
        "DrawPrimitiveUP",
    );
    assert_eq!(h.end_scene(), 0, "EndScene");

    let plain = h.create_render_target(width, height, D3DFMT_X8R8G8B8);
    assert_eq!(h.stretch_rect(&rt, &plain, D3DTEXF_NONE), 0, "resolve");
    let row = surface_row(&h, &plain, (width, height));
    assert!(
        count_intermediate(&row) > 0,
        "the draw reached the 4x target with depth dropped: {row:02X?}"
    );
    assert_pixel_eq(
        row[INSIDE_X as usize],
        WHITE,
        "a pixel clear of every edge carries the draw",
    );
}

// ── D3DRS_MULTISAMPLEMASK ──

#[test]
fn multisample_mask_selects_the_samples_a_draw_writes() {
    let h = harness(D3DMULTISAMPLE_4_SAMPLES, None);
    let (width_f, height_f) = back_buffer_extent(&h);
    arm(&h);

    // Every sample masked out: the draw covers the target but writes nothing.
    h.render_once(BLACK, |d| {
        assert_eq!(d.set_render_state(D3DRS_MULTISAMPLEMASK, 0), 0, "mask 0");
        assert_eq!(
            d.draw_primitive_up(
                D3DPT_TRIANGLELIST,
                1,
                &diagonal(width_f, height_f, 0.5, WHITE)
            ),
            0,
            "masked draw",
        );
    });
    assert_pixel_eq(
        back_buffer_row(&h)[INSIDE_X as usize],
        BLACK,
        "a fully masked draw writes no sample",
    );

    // Half the samples: a fully covered pixel resolves to half intensity.
    h.render_once(BLACK, |d| {
        assert_eq!(
            d.set_render_state(D3DRS_MULTISAMPLEMASK, 0b0011),
            0,
            "mask 3"
        );
        assert_eq!(
            d.draw_primitive_up(
                D3DPT_TRIANGLELIST,
                1,
                &diagonal(width_f, height_f, 0.5, WHITE)
            ),
            0,
            "half-masked draw",
        );
    });
    let half = Rgba8::from_pixel(back_buffer_row(&h)[INSIDE_X as usize]);
    assert!(
        half.r > 16 && half.r < 239,
        "two of four samples resolve to a partial value, got {half:?}"
    );

    // The default mask restores the full write.
    h.render_once(BLACK, |d| {
        assert_eq!(
            d.set_render_state(D3DRS_MULTISAMPLEMASK, 0xFFFF_FFFF),
            0,
            "default mask"
        );
        assert_eq!(
            d.draw_primitive_up(
                D3DPT_TRIANGLELIST,
                1,
                &diagonal(width_f, height_f, 0.5, WHITE)
            ),
            0,
            "unmasked draw",
        );
    });
    assert_pixel_eq(
        back_buffer_row(&h)[INSIDE_X as usize],
        WHITE,
        "the default mask writes every sample",
    );
}

// ── RESZ from a multisampled depth surface ──

/// The depth the RESZ scene writes, and the grey it samples back as.
const RESZ_DEPTH: f32 = 0.25;

/// One screen-space triangle covering the whole `RT_SIZE` target at depth `z`.
///
/// Twice the target's edge in both directions, so every sample of every pixel
/// is inside it and the depth written is `z` at each of them.
const fn covering_triangle(z: f32) -> [RhwVertex; 3] {
    [
        RhwVertex {
            x: 0.0,
            y: 0.0,
            z,
            rhw: 1.0,
            color: WHITE,
        },
        RhwVertex {
            x: RT_SIZE_F * 2.0,
            y: 0.0,
            z,
            rhw: 1.0,
            color: WHITE,
        },
        RhwVertex {
            x: 0.0,
            y: RT_SIZE_F * 2.0,
            z,
            rhw: 1.0,
            color: WHITE,
        },
    ]
}

/// A clip-space quad over the whole target with the texture mapped onto it.
const fn textured_quad() -> [TexturedVertex; 6] {
    const fn v(x: f32, y: f32, u: f32, tv: f32) -> TexturedVertex {
        TexturedVertex {
            x,
            y,
            z: 0.5,
            color: WHITE,
            u,
            v: tv,
        }
    }
    [
        v(-1.0, 1.0, 0.0, 0.0),
        v(1.0, 1.0, 1.0, 0.0),
        v(-1.0, -1.0, 0.0, 1.0),
        v(1.0, 1.0, 1.0, 0.0),
        v(1.0, -1.0, 1.0, 1.0),
        v(-1.0, -1.0, 0.0, 1.0),
    ]
}

/// Fill `intz` with depth 1.0, so a resolve that never runs is visible.
fn prime_intz(h: &Harness, rt: &Surface<'_>, intz: &Texture<'_>) {
    assert_eq!(h.set_render_target(0, rt), 0, "SetRenderTarget(prime)");
    assert_eq!(
        h.set_depth_stencil_surface(&intz.surface_level(0)),
        0,
        "bind the INTZ level as depth"
    );
    assert_eq!(
        h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, BLACK, 1.0, 0),
        0,
        "clear the INTZ to the far plane"
    );
}

/// Render depth `RESZ_DEPTH` into `ds` beside `rt`, then RESZ it into `intz`.
fn resz_into(h: &Harness, rt: &Surface<'_>, ds: &Surface<'_>, intz: &Texture<'_>) {
    arm(h);
    h.select_diffuse_stage(0);
    assert_eq!(h.set_render_target(0, rt), 0, "SetRenderTarget(scene)");
    assert_eq!(h.set_depth_stencil_surface(ds), 0, "SetDepthStencilSurface");
    assert_eq!(h.set_render_state(D3DRS_ZENABLE, 1), 0, "depth test on");
    assert_eq!(h.set_render_state(D3DRS_ZWRITEENABLE, 1), 0, "depth writes");
    assert_eq!(
        h.set_render_state(D3DRS_ZFUNC, D3DCMP_ALWAYS),
        0,
        "depth func"
    );
    assert_eq!(h.begin_scene(), 0, "BeginScene");
    assert_eq!(
        h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, BLACK, 1.0, 0),
        0,
        "clear colour + depth"
    );
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &covering_triangle(RESZ_DEPTH)),
        0,
        "depth-writing draw",
    );
    assert_eq!(h.end_scene(), 0, "EndScene");

    assert_eq!(h.set_texture(0, intz), 0, "bind the RESZ destination");
    assert_eq!(
        h.set_render_state(D3DRS_POINTSIZE, 0x7fa0_5000),
        0,
        "the RESZ magic value"
    );
}

/// Sample `intz` over the whole of `rt` and read the middle pixel back.
///
/// An INTZ texture answers a fixed-function fetch with the raw stored depth
/// broadcast to every channel, so the quad reads back as the depth value.
fn sample_intz(h: &Harness, rt: &Surface<'_>, intz: &Texture<'_>) -> u32 {
    h.select_texture_stage(0);
    assert_eq!(h.set_render_target(0, rt), 0, "SetRenderTarget(sample)");
    assert_eq!(
        h.clear_depth_stencil_surface(),
        0,
        "unbind depth so the INTZ is a sampler, not an attachment"
    );
    assert_eq!(h.set_render_state(D3DRS_ZENABLE, 0), 0, "depth test off");
    assert_eq!(h.set_texture(0, intz), 0, "bind the INTZ as a sampler");
    for (state, value) in [
        (D3DSAMP_MINFILTER, D3DTEXF_POINT),
        (D3DSAMP_MAGFILTER, D3DTEXF_POINT),
        (D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP),
        (D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP),
    ] {
        assert_eq!(h.set_sampler_state(0, state, value), 0, "sampler state");
    }
    assert_eq!(
        h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1),
        0,
        "SetFVF"
    );
    assert_eq!(h.begin_scene(), 0, "BeginScene");
    assert_eq!(h.clear_target(BLUE), 0, "clear the sample target");
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &textured_quad()),
        0,
        "sample the resolved depth",
    );
    assert_eq!(h.end_scene(), 0, "EndScene");
    surface_row(h, rt, (RT_SIZE, RT_SIZE))[(RT_SIZE / 2) as usize]
}

#[test]
fn resz_resolves_a_multisampled_depth_surface() {
    // The RESZ hack hands the bound depth-stencil to a single-sampled INTZ
    // texture. From a multisampled surface that is a resolve, not a copy:
    // Metal's blit encoder refuses the sample-count change, so the pass
    // machinery has to take the resolve instead. The scene writes one constant
    // depth, so the multisampled answer and the single-sampled one are the
    // same value and can be compared directly.
    let h = harness(D3DMULTISAMPLE_NONE, None);
    if h.device_is_paravirtual() {
        // The paravirtual device hands a later encoder of the same command
        // buffer the content a depth resolve target held before the resolve,
        // so a resolved depth cannot be read back through it there.
        return;
    }
    let size = (RT_SIZE, RT_SIZE);
    let ms = (D3DMULTISAMPLE_4_SAMPLES, 0);

    let ms_rt = h.create_render_target_ms(size, D3DFMT_A8R8G8B8, ms);
    let ms_ds = h
        .create_depth_stencil_surface_ms_hr(size, D3DFMT_D24S8, ms)
        .1
        .expect("4x depth surface");
    let ss_rt = h.create_render_target(RT_SIZE, RT_SIZE, D3DFMT_A8R8G8B8);
    let ss_ds = h.create_depth_stencil_surface(RT_SIZE, RT_SIZE, D3DFMT_D24S8);
    let resolved_depth = h.create_texture(
        RT_SIZE,
        RT_SIZE,
        1,
        D3DUSAGE_DEPTHSTENCIL,
        D3DFMT_INTZ,
        D3DPOOL_DEFAULT,
    );
    let copied_depth = h.create_texture(
        RT_SIZE,
        RT_SIZE,
        1,
        D3DUSAGE_DEPTHSTENCIL,
        D3DFMT_INTZ,
        D3DPOOL_DEFAULT,
    );

    // Both destinations start at the far plane, so a resolve that never
    // happens reads back white instead of the scene's depth.
    prime_intz(&h, &ss_rt, &resolved_depth);
    prime_intz(&h, &ss_rt, &copied_depth);

    resz_into(&h, &ms_rt, &ms_ds, &resolved_depth);
    let from_multisampled = Rgba8::from_pixel(sample_intz(&h, &ss_rt, &resolved_depth));

    resz_into(&h, &ss_rt, &ss_ds, &copied_depth);
    let from_single_sampled = Rgba8::from_pixel(sample_intz(&h, &ss_rt, &copied_depth));

    assert!(
        from_multisampled.r < 200,
        "the multisampled resolve ran at all, got {from_multisampled:?}"
    );
    assert!(
        from_multisampled.r.abs_diff(from_single_sampled.r) <= 2,
        "the resolved depth matches the single-sampled RESZ: \
         multisampled {from_multisampled:?} vs single-sampled {from_single_sampled:?}"
    );

    assert_eq!(h.clear_texture(0), 0, "clear the sampler bind");
}

// ── StretchRect from a multisampled depth surface ──

#[test]
fn stretch_rect_resolves_a_multisampled_depth_surface() {
    // A depth-to-depth StretchRect out of a multisampled surface is a resolve,
    // not a copy: Metal's blit encoder refuses the sample-count change, so the
    // pass machinery has to take it. The destination is observed through the
    // depth test rather than read back, which D3D9 does not allow on a depth
    // surface.
    let h = harness(D3DMULTISAMPLE_NONE, None);
    if h.device_is_paravirtual() {
        // The paravirtual device hands a later encoder of the same command
        // buffer the content a depth resolve target held before the resolve,
        // so a resolved depth cannot be read back through it there.
        return;
    }
    let size = (RT_SIZE, RT_SIZE);
    let ms = (D3DMULTISAMPLE_4_SAMPLES, 0);

    let ms_rt = h.create_render_target_ms(size, D3DFMT_A8R8G8B8, ms);
    let ms_ds = h
        .create_depth_stencil_surface_ms_hr(size, D3DFMT_D24S8, ms)
        .1
        .expect("4x depth surface");
    let ss_rt = h.create_render_target(RT_SIZE, RT_SIZE, D3DFMT_A8R8G8B8);
    let ss_ds = h.create_depth_stencil_surface(RT_SIZE, RT_SIZE, D3DFMT_D24S8);

    // The destination starts at the far plane, so a resolve that never runs
    // lets the probe draw through and paints the target white.
    assert_eq!(h.set_render_target(0, &ss_rt), 0, "SetRenderTarget(prime)");
    assert_eq!(
        h.set_depth_stencil_surface(&ss_ds),
        0,
        "SetDepthStencilSurface(prime)"
    );
    assert_eq!(
        h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, BLACK, 1.0, 0),
        0,
        "clear the destination to the far plane"
    );

    // Write one constant depth into the multisampled surface.
    arm(&h);
    h.select_diffuse_stage(0);
    assert_eq!(h.set_render_target(0, &ms_rt), 0, "SetRenderTarget(scene)");
    assert_eq!(
        h.set_depth_stencil_surface(&ms_ds),
        0,
        "SetDepthStencilSurface(scene)"
    );
    assert_eq!(h.set_render_state(D3DRS_ZENABLE, 1), 0, "depth test on");
    assert_eq!(h.set_render_state(D3DRS_ZWRITEENABLE, 1), 0, "depth writes");
    assert_eq!(
        h.set_render_state(D3DRS_ZFUNC, D3DCMP_ALWAYS),
        0,
        "depth func"
    );
    assert_eq!(h.begin_scene(), 0, "BeginScene");
    assert_eq!(
        h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, BLACK, 1.0, 0),
        0,
        "clear colour + depth"
    );
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &covering_triangle(RESZ_DEPTH)),
        0,
        "depth-writing draw",
    );
    assert_eq!(h.end_scene(), 0, "EndScene");

    assert_eq!(
        h.stretch_rect(&ms_ds, &ss_ds, D3DTEXF_NONE),
        D3D_OK,
        "StretchRect the multisampled depth into the single-sampled surface"
    );

    // Probe the resolved depth: a draw behind it is rejected, so the target
    // keeps its clear colour. Without the resolve the destination still holds
    // the far plane and the probe paints over it.
    assert_eq!(h.set_render_target(0, &ss_rt), 0, "SetRenderTarget(probe)");
    assert_eq!(
        h.set_depth_stencil_surface(&ss_ds),
        0,
        "SetDepthStencilSurface(probe)"
    );
    assert_eq!(
        h.set_render_state(D3DRS_ZFUNC, D3DCMP_LESS),
        0,
        "depth func"
    );
    assert_eq!(h.begin_scene(), 0, "BeginScene");
    assert_eq!(h.clear_target(BLUE), 0, "clear the probe target");
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &covering_triangle(RESZ_DEPTH + 0.25)),
        0,
        "probe draw behind the resolved depth",
    );
    assert_eq!(h.end_scene(), 0, "EndScene");

    assert_pixel_eq(
        surface_row(&h, &ss_rt, (RT_SIZE, RT_SIZE))[(RT_SIZE / 2) as usize],
        BLUE,
        "the resolved depth rejects the draw behind it",
    );
}

// ── A clear ordered before a StretchRect resolve ──

/// A triangle over the left quarter of the target, in `color`.
///
/// At the middle scanline it reaches a quarter of the way across, so one probe
/// lands inside it and one well clear of it.
const fn corner_triangle(color: u32) -> [RhwVertex; 3] {
    [
        RhwVertex {
            x: 0.0,
            y: 0.0,
            z: 0.5,
            rhw: 1.0,
            color,
        },
        RhwVertex {
            x: RT_SIZE_F / 4.0,
            y: 0.0,
            z: 0.5,
            rhw: 1.0,
            color,
        },
        RhwVertex {
            x: 0.0,
            y: RT_SIZE_F,
            z: 0.5,
            rhw: 1.0,
            color,
        },
    ]
}

#[test]
fn a_clear_before_a_stretch_rect_resolve_does_not_wipe_it() {
    // Clear(target), render into a multisampled surface, StretchRect it into
    // the target, then draw over a corner of the target. The clear is ordered
    // first, so the copy has to survive it: what the last draw does not cover
    // reads back as the resolved image.
    let h = harness(D3DMULTISAMPLE_NONE, None);
    let ms = h.create_render_target_ms(
        (RT_SIZE, RT_SIZE),
        D3DFMT_A8R8G8B8,
        (D3DMULTISAMPLE_4_SAMPLES, 0),
    );
    let target = h.create_render_target(RT_SIZE, RT_SIZE, D3DFMT_A8R8G8B8);
    arm(&h);
    h.select_diffuse_stage(0);

    assert_eq!(
        h.set_render_target(0, &target),
        0,
        "SetRenderTarget(target)"
    );
    assert_eq!(h.clear_target(BLUE), 0, "clear the target");

    assert_eq!(h.set_render_target(0, &ms), 0, "SetRenderTarget(scene)");
    assert_eq!(h.begin_scene(), 0, "BeginScene(scene)");
    assert_eq!(h.clear_target(BLACK), 0, "clear the multisampled surface");
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &covering_triangle(0.5)),
        0,
        "covering draw",
    );
    assert_eq!(h.end_scene(), 0, "EndScene(scene)");

    assert_eq!(
        h.stretch_rect(&ms, &target, D3DTEXF_NONE),
        D3D_OK,
        "StretchRect the multisampled surface into the target"
    );

    assert_eq!(h.set_render_target(0, &target), 0, "SetRenderTarget(over)");
    assert_eq!(h.begin_scene(), 0, "BeginScene(over)");
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &corner_triangle(BLACK)),
        0,
        "corner draw",
    );
    assert_eq!(h.end_scene(), 0, "EndScene(over)");

    let row = surface_row(&h, &target, (RT_SIZE, RT_SIZE));
    assert_pixel_eq(row[INSIDE_X as usize], BLACK, "the corner draw landed");
    assert_pixel_eq(
        row[(RT_SIZE - 4) as usize],
        WHITE,
        "the resolved image survives the clear that was ordered before it",
    );
}
