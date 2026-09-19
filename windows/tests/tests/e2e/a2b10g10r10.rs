//! Native packed ten-bit A2B10G10R10 texture regressions.

use mtld3d_tests::Harness;
use mtld3d_types::{D3DFMT_A2B10G10R10, D3DPOOL_MANAGED};

#[test]
fn a2b10g10r10_creation_is_not_capability_gated() {
    let h = Harness::new();
    let texture = h.create_texture(3, 2, 1, 0, D3DFMT_A2B10G10R10, D3DPOOL_MANAGED);
    assert_eq!(texture.level_desc(0).1.format, D3DFMT_A2B10G10R10);
}

#[test]
fn a2b10g10r10_channels_alpha_and_lower_rgb_bits() {
    super::packed10::channels_alpha_and_lower_rgb_bits(D3DFMT_A2B10G10R10);
}

#[test]
fn a2b10g10r10_native_words_pools_mips_and_updates() {
    super::packed10::native_words_pools_mips_and_updates(D3DFMT_A2B10G10R10);
}

#[test]
fn a2b10g10r10_queries_and_noautogen() {
    super::packed10::queries_and_noautogen(D3DFMT_A2B10G10R10);
}

#[test]
fn a2b10g10r10_cube_volume_filter_and_mips() {
    super::packed10::cube_volume_filter_and_mips(D3DFMT_A2B10G10R10);
}

#[test]
fn a2b10g10r10_colorfill_and_gpu_authority() {
    super::packed10::colorfill_and_gpu_authority(D3DFMT_A2B10G10R10);
}

#[test]
fn a2b10g10r10_rejects_mixed_raw_copies() {
    super::packed10::rejects_mixed_raw_copies(D3DFMT_A2B10G10R10);
}

#[test]
fn a2b10g10r10_and_a2r10g10b10_have_distinct_raw_lane_order() {
    let h = Harness::new();
    let rgb = h.create_texture(1, 1, 1, 0, D3DFMT_A2B10G10R10, D3DPOOL_MANAGED);
    let bgr = h.create_texture(
        1,
        1,
        1,
        0,
        mtld3d_types::D3DFMT_A2R10G10B10,
        D3DPOOL_MANAGED,
    );
    for alpha in 0..4 {
        let same_word = (alpha << 30) | (1022 << 20) | (341 << 10) | 5;
        rgb.lock_rect(0, 0).write_u32(&[same_word]);
        bgr.lock_rect(0, 0).write_u32(&[same_word]);
        let expected_rgb = ((alpha * 85) << 24) | 0x0001_55ff;
        let expected_bgr = ((alpha * 85) << 24) | 0x00ff_5501;
        mtld3d_tests::assert_pixel_approx(
            super::packed10::sample(&h, &rgb, [0.5; 2]),
            expected_rgb,
            1,
            "RGB low red",
        );
        mtld3d_tests::assert_pixel_approx(
            super::packed10::sample(&h, &bgr, [0.5; 2]),
            expected_bgr,
            1,
            "same word BGR low blue",
        );
        bgr.lock_rect(0, 0)
            .write_u32(&[(alpha << 30) | (5 << 20) | (341 << 10) | 0x03fe]);
        mtld3d_tests::assert_pixel_approx(
            super::packed10::sample(&h, &bgr, [0.5; 2]),
            expected_rgb,
            1,
            "different words same color",
        );
    }
}
