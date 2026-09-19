//! Wide-format DEFAULT offscreen conversion and its API boundary.

use mtld3d_tests::{Harness, Surface};
use mtld3d_types::{
    D3D_OK, D3DERR_INVALIDCALL, D3DFMT_A8B8G8R8, D3DFMT_A8R8G8B8, D3DFMT_A16B16G16R16,
    D3DFMT_A32B32G32R32F, D3DFMT_L16, D3DLOCK_READONLY, D3DPOOL_DEFAULT, D3DPOOL_SYSTEMMEM,
    D3DTEXF_NONE,
};

const WIDTH: usize = 6;
const HEIGHT: usize = 2;
const SENTINEL: u32 = 0x7f12_3456;

fn source_bytes(format: u32) -> Vec<u8> {
    let texels: Vec<Vec<u8>> = if format == D3DFMT_A16B16G16R16 {
        [
            [0_u16, 0x3333, 0x6666, 0x9999],
            [0x3333, 0x6666, 0x9999, 0xcccc],
            [0xffff, 0, 0xcccc, 0x6666],
            [128, 129, 32768, 65535],
        ]
        .map(|rgba| rgba.into_iter().flat_map(u16::to_le_bytes).collect())
        .into()
    } else if format == D3DFMT_A8B8G8R8 {
        [
            [0, 0x33, 0x66, 0x99],
            [0x33, 0x66, 0x99, 0xcc],
            [0xff, 0, 0xcc, 0x66],
            [0, 0xff, 0x80, 0xff],
        ]
        .map(Vec::from)
        .into()
    } else {
        [
            [0_f32, 0.2, 0.4, 0.6],
            [0.2, 0.4, 0.6, 0.8],
            [1.0, 0.0, 0.8, 0.4],
            [-0.25, 1.25, 0.5, 1.0],
        ]
        .map(|rgba| rgba.into_iter().flat_map(f32::to_le_bytes).collect())
        .into()
    };
    (0..WIDTH * HEIGHT)
        .flat_map(|i| texels[i % 4].iter().copied())
        .collect()
}

fn assert_pixels(surface: &Surface<'_>, expected: &[u32]) {
    let locked = surface.lock_rect(D3DLOCK_READONLY);
    let pitch = usize::try_from(locked.pitch()).expect("positive pitch") / 4;
    let pixels = locked.as_u32(pitch * HEIGHT);
    for y in 0..HEIGHT {
        assert_eq!(
            &pixels[y * pitch..y * pitch + WIDTH],
            &expected[y * WIDTH..(y + 1) * WIDTH]
        );
    }
}

fn check_conversion(format: u32, gpu_source: bool, gpu_destination: bool) {
    let h = Harness::new();
    let source = h.create_offscreen_plain_surface(6, 2, format, D3DPOOL_DEFAULT);
    let writer = h.create_offscreen_plain_surface(6, 2, format, D3DPOOL_DEFAULT);
    let destination = h.create_offscreen_plain_surface(6, 2, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    let primer = h.create_offscreen_plain_surface(6, 2, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    let target = h.create_render_target(6, 2, D3DFMT_A8R8G8B8);
    let readback = h.create_offscreen_plain_surface(6, 2, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    let previous = h.create_offscreen_plain_surface(6, 2, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    let bytes = source_bytes(format);
    let row_bytes = bytes.len() / HEIGHT;
    let fourth = if format == D3DFMT_A16B16G16R16 {
        0xff00_0180
    } else {
        0xff00_ff80
    };
    let full: Vec<_> = [0x9900_3366, 0xcc33_6699, 0x66ff_00cc, fourth].repeat(3);
    for partial in [false, true] {
        source
            .lock_rect(0)
            .write_u8_rect(row_bytes, HEIGHT, &vec![0; bytes.len()]);
        writer.lock_rect(0).write_u8_rect(row_bytes, HEIGHT, &bytes);
        if gpu_source {
            assert_eq!(h.stretch_rect(&writer, &source, D3DTEXF_NONE), D3D_OK);
        } else {
            source.lock_rect(0).write_u8_rect(row_bytes, HEIGHT, &bytes);
        }
        // Give the destination stale CPU pixels, then replace it on the GPU.
        // The partial conversion must preserve the GPU-owned sentinel outside.
        let initial = if gpu_destination { 0 } else { SENTINEL };
        destination
            .lock_rect(0)
            .write_u32_rect(WIDTH, HEIGHT, &[initial; WIDTH * HEIGHT]);
        assert_eq!(
            h.stretch_rect(&destination, &previous, D3DTEXF_NONE),
            D3D_OK
        );
        primer
            .lock_rect(0)
            .write_u32_rect(WIDTH, HEIGHT, &[SENTINEL; WIDTH * HEIGHT]);
        if gpu_destination {
            assert_eq!(h.stretch_rect(&primer, &destination, D3DTEXF_NONE), D3D_OK);
        }
        let mut expected = full.clone();
        let hr = if partial {
            expected.fill(SENTINEL);
            expected[8..10].copy_from_slice(&full[1..3]);
            h.stretch_rect_rects(
                &source,
                (1, 0, 3, 1),
                &destination,
                (2, 1, 4, 2),
                D3DTEXF_NONE,
            )
        } else {
            h.stretch_rect(&source, &destination, D3DTEXF_NONE)
        };
        assert_eq!(hr, D3D_OK);
        // Read the GPU copy before mapping the converted destination.
        assert_eq!(h.stretch_rect(&destination, &target, D3DTEXF_NONE), D3D_OK);
        assert_eq!(h.get_render_target_data_hr(&target, &readback), D3D_OK);
        assert_pixels(&readback, &expected);
        assert_pixels(&destination, &expected);
        assert_pixels(&previous, &[initial; WIDTH * HEIGHT]);
        let locked = source.lock_rect(D3DLOCK_READONLY);
        let pitch = usize::try_from(locked.pitch()).expect("positive pitch");
        let actual = locked.as_u8(pitch * HEIGHT);
        for y in 0..HEIGHT {
            assert_eq!(
                &actual[y * pitch..y * pitch + row_bytes],
                &bytes[y * row_bytes..(y + 1) * row_bytes]
            );
        }
    }
}

#[test]
fn wide_stretch_unorm16_cpu_source() {
    check_conversion(D3DFMT_A16B16G16R16, false, true);
}
#[test]
fn wide_stretch_unorm16_gpu_source() {
    check_conversion(D3DFMT_A16B16G16R16, true, true);
}
#[test]
fn wide_stretch_float32_cpu_source() {
    check_conversion(D3DFMT_A32B32G32R32F, false, true);
}
#[test]
fn wide_stretch_float32_gpu_source() {
    check_conversion(D3DFMT_A32B32G32R32F, true, true);
}

#[test]
fn wide_stretch_rejects_an_uncovered_pair_without_writing() {
    let h = Harness::new();
    let source = h.create_offscreen_plain_surface(6, 2, D3DFMT_L16, D3DPOOL_DEFAULT);
    let destination = h.create_offscreen_plain_surface(6, 2, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    source
        .lock_rect(0)
        .write_u8_rect(WIDTH * 2, HEIGHT, &[0xff; WIDTH * HEIGHT * 2]);
    destination
        .lock_rect(0)
        .write_u32_rect(WIDTH, HEIGHT, &[SENTINEL; WIDTH * HEIGHT]);
    assert_eq!(
        h.stretch_rect(&source, &destination, D3DTEXF_NONE),
        D3DERR_INVALIDCALL
    );
    assert_pixels(&destination, &[SENTINEL; WIDTH * HEIGHT]);
}

#[test]
fn wide_stretch_codec_does_not_expand_update_api_acceptance() {
    let h = Harness::new();
    for format in [D3DFMT_A16B16G16R16, D3DFMT_A32B32G32R32F] {
        let source = h.create_texture(6, 2, 1, 0, format, D3DPOOL_SYSTEMMEM);
        let standalone = h.create_offscreen_plain_surface(6, 2, format, D3DPOOL_SYSTEMMEM);
        let destination = h.create_texture(6, 2, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
        let bytes = source_bytes(format);
        source
            .lock_rect(0, 0)
            .write_u8_rect(bytes.len() / HEIGHT, HEIGHT, &bytes);
        standalone
            .lock_rect(0)
            .write_u8_rect(bytes.len() / HEIGHT, HEIGHT, &bytes);
        destination
            .lock_rect(0, 0)
            .write_u32_rect(WIDTH, HEIGHT, &[SENTINEL; WIDTH * HEIGHT]);
        assert_eq!(
            h.update_texture_hr(&source, &destination),
            D3DERR_INVALIDCALL
        );
        let level = destination.surface_level(0);
        assert_eq!(
            h.update_surface_hr(&source.surface_level(0), &level),
            D3DERR_INVALIDCALL
        );
        assert_eq!(h.update_surface_hr(&standalone, &level), D3DERR_INVALIDCALL);
        let locked = destination.lock_rect(0, D3DLOCK_READONLY);
        let pitch = usize::try_from(locked.pitch()).expect("positive pitch") / 4;
        let pixels = locked.as_u32(pitch * HEIGHT);
        for y in 0..HEIGHT {
            assert_eq!(&pixels[y * pitch..y * pitch + WIDTH], &[SENTINEL; WIDTH]);
        }
    }
}

#[test]
fn wide_stretch_narrow_codec_cpu_source_ordering() {
    check_conversion(D3DFMT_A8B8G8R8, false, true);
}

#[test]
fn wide_stretch_narrow_codec_gpu_source_ordering() {
    check_conversion(D3DFMT_A8B8G8R8, true, true);
}

#[test]
fn wide_stretch_preserves_an_earlier_gpu_read_of_its_destination() {
    check_conversion(D3DFMT_A16B16G16R16, false, false);
}

fn check_later_lock(partial: bool, render_reader: bool) {
    let h = Harness::new();
    let source = h.create_offscreen_plain_surface(6, 2, D3DFMT_A16B16G16R16, D3DPOOL_DEFAULT);
    let destination = h.create_offscreen_plain_surface(6, 2, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    let previous = h.create_offscreen_plain_surface(6, 2, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    let target = h.create_render_target(6, 2, D3DFMT_A8R8G8B8);
    let readback = h.create_offscreen_plain_surface(6, 2, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    let rendered = h.create_render_target(12, 4, D3DFMT_A8R8G8B8);
    let rendered_readback =
        h.create_offscreen_plain_surface(12, 4, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    let bytes = source_bytes(D3DFMT_A16B16G16R16);
    source
        .lock_rect(0)
        .write_u8_rect(bytes.len() / HEIGHT, HEIGHT, &bytes);
    let converted = [0x9900_3366, 0xcc33_6699, 0x66ff_00cc, 0xff00_0180].repeat(3);
    assert_eq!(h.stretch_rect(&source, &destination, D3DTEXF_NONE), D3D_OK);
    assert_eq!(
        h.stretch_rect(&destination, &previous, D3DTEXF_NONE),
        D3D_OK
    );
    if render_reader {
        assert_eq!(
            h.stretch_rect_rects(
                &destination,
                (0, 0, 6, 2),
                &rendered,
                (0, 0, 12, 4),
                D3DTEXF_NONE
            ),
            D3D_OK
        );
    }
    // Without render_reader the destination is never sampled. Neither form
    // performs a readback between conversion, GPU read and the later lock.
    let mut expected = converted.clone();
    if partial {
        destination
            .lock_rect_partial(&[2, 1, 4, 2], 0)
            .write_u32_rect(2, 1, &[SENTINEL; 2]);
        expected[8..10].fill(SENTINEL);
    } else {
        destination
            .lock_rect(0)
            .write_u32_rect(WIDTH, HEIGHT, &[SENTINEL; WIDTH * HEIGHT]);
        expected.fill(SENTINEL);
    }
    assert_eq!(h.stretch_rect(&destination, &target, D3DTEXF_NONE), D3D_OK);
    assert_eq!(h.get_render_target_data_hr(&target, &readback), D3D_OK);
    assert_pixels(&readback, &expected);
    if !partial {
        assert_pixels(&previous, &converted);
    }
    if render_reader {
        assert_eq!(
            h.get_render_target_data_hr(&rendered, &rendered_readback),
            D3D_OK
        );
        let locked = rendered_readback.lock_rect(D3DLOCK_READONLY);
        let pitch = usize::try_from(locked.pitch()).expect("positive pitch") / 4;
        let pixels = locked.as_u32(pitch * 4);
        for y in 0..4 {
            for x in 0..12 {
                assert_eq!(pixels[y * pitch + x], converted[(y / 2) * WIDTH + x / 2]);
            }
        }
    }
    // An overlapping contended partial lock retains the documented in-place
    // policy. Only its final pixels and untouched region are defined here.
}

#[test]
fn wide_stretch_later_whole_lock_upload_stays_after_conversion() {
    check_later_lock(false, false);
}

#[test]
fn wide_stretch_later_partial_lock_preserves_converted_region() {
    check_later_lock(true, false);
}

#[test]
fn wide_stretch_first_ordered_upload_is_traceable() {
    let h = Harness::new();
    let source = h.create_offscreen_plain_surface(6, 2, D3DFMT_A16B16G16R16, D3DPOOL_DEFAULT);
    let destination = h.create_offscreen_plain_surface(6, 2, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    let bytes = source_bytes(D3DFMT_A16B16G16R16);
    source
        .lock_rect(0)
        .write_u8_rect(bytes.len() / HEIGHT, HEIGHT, &bytes);
    assert_eq!(h.stretch_rect(&source, &destination, D3DTEXF_NONE), D3D_OK);
    let target = h.create_render_target(6, 2, D3DFMT_A8R8G8B8);
    let readback = h.create_offscreen_plain_surface(6, 2, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    assert_eq!(h.stretch_rect(&destination, &target, D3DTEXF_NONE), D3D_OK);
    assert_eq!(h.get_render_target_data_hr(&target, &readback), D3D_OK);
    assert_pixels(
        &readback,
        &[0x9900_3366, 0xcc33_6699, 0x66ff_00cc, 0xff00_0180].repeat(3),
    );
}

#[test]
fn wide_stretch_later_lock_keeps_an_earlier_render_reader() {
    check_later_lock(false, true);
}

#[test]
fn wide_stretch_repeated_whole_and_partial_uploads_keep_order() {
    let h = Harness::new();
    let source = h.create_offscreen_plain_surface(6, 2, D3DFMT_A16B16G16R16, D3DPOOL_DEFAULT);
    let destination = h.create_offscreen_plain_surface(6, 2, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    let previous = h.create_offscreen_plain_surface(6, 2, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    let first = h.create_render_target(6, 2, D3DFMT_A8R8G8B8);
    let target = h.create_render_target(6, 2, D3DFMT_A8R8G8B8);
    let readback = h.create_offscreen_plain_surface(6, 2, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    let bytes = source_bytes(D3DFMT_A16B16G16R16);
    source
        .lock_rect(0)
        .write_u8_rect(bytes.len() / HEIGHT, HEIGHT, &bytes);
    assert_eq!(h.stretch_rect(&source, &destination, D3DTEXF_NONE), D3D_OK);
    assert_eq!(
        h.stretch_rect(&destination, &previous, D3DTEXF_NONE),
        D3D_OK
    );
    destination
        .lock_rect(0)
        .write_u32_rect(WIDTH, HEIGHT, &[SENTINEL; WIDTH * HEIGHT]);
    assert_eq!(h.stretch_rect(&destination, &first, D3DTEXF_NONE), D3D_OK);
    let second = 0xaabb_ccdd;
    destination
        .lock_rect(0)
        .write_u32_rect(WIDTH, HEIGHT, &[second; WIDTH * HEIGHT]);
    assert_eq!(h.stretch_rect(&destination, &target, D3DTEXF_NONE), D3D_OK);
    destination
        .lock_rect_partial(&[2, 1, 4, 2], 0)
        .write_u32_rect(2, 1, &[SENTINEL; 2]);
    assert_eq!(h.stretch_rect(&destination, &target, D3DTEXF_NONE), D3D_OK);
    // No submission or readback separates the conversion and three later
    // writes. The overlapping partial-lock policy leaves the intermediate
    // second snapshot unspecified; the final copy must preserve its other
    // pixels, and earlier whole-level snapshots keep their retained staging.
    let mut expected = [second; WIDTH * HEIGHT];
    expected[8..10].fill(SENTINEL);
    assert_eq!(h.get_render_target_data_hr(&target, &readback), D3D_OK);
    assert_pixels(&readback, &expected);
    assert_eq!(h.get_render_target_data_hr(&first, &readback), D3D_OK);
    assert_pixels(&readback, &[SENTINEL; WIDTH * HEIGHT]);
    assert_pixels(
        &previous,
        &[0x9900_3366, 0xcc33_6699, 0x66ff_00cc, 0xff00_0180].repeat(3),
    );
}
