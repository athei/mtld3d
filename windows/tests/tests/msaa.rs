//! Multisample anti-aliasing: the swap chain, standalone surfaces and the resolve.
//!
//! The observable property throughout is coverage. A diagonal edge drawn into a
//! multisampled target resolves to intermediate pixels along the edge; the same
//! draw into a single-sampled target has none, every pixel being fully inside
//! or fully outside. Each test that renders counts those intermediate pixels
//! along one scanline, so nothing depends on where exactly the rasterizer puts
//! the edge or on the device's sample positions.

use mtld3d_tests::{Harness, HarnessConfig, Rgba8, RhwVertex, Surface, assert_pixel_eq};
use mtld3d_types::{
    D3D_OK, D3DCLEAR_TARGET, D3DCLEAR_ZBUFFER, D3DERR_INVALIDCALL, D3DERR_NOTAVAILABLE,
    D3DFMT_A8R8G8B8, D3DFMT_D16, D3DFMT_D24S8, D3DFMT_DXT1, D3DFMT_INTZ, D3DFMT_X8R8G8B8,
    D3DFVF_DIFFUSE, D3DFVF_XYZRHW, D3DLOCK_READONLY, D3DMULTISAMPLE_2_SAMPLES,
    D3DMULTISAMPLE_4_SAMPLES, D3DMULTISAMPLE_NONE, D3DMULTISAMPLE_NONMASKABLE, D3DPOOL_SYSTEMMEM,
    D3DPT_TRIANGLELIST, D3DRS_LIGHTING, D3DRS_MULTISAMPLEMASK, D3DRS_ZENABLE, D3DTEXF_NONE,
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
