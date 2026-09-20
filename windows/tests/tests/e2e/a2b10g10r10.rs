//! Native packed ten-bit A2B10G10R10 texture regressions.

use mtld3d_tests::{Harness, assert_pixel_approx};
use mtld3d_types::{D3DFMT_A2B10G10R10, D3DFMT_A2R10G10B10, D3DPOOL_MANAGED};

use super::packed10::{packed, sample};

/// Creation needs no configuration key and reports the requested format.
#[test]
fn a2b10g10r10_creation_is_not_capability_gated() {
    let h = Harness::new();
    let texture = h.create_texture(3, 2, 1, 0, D3DFMT_A2B10G10R10, D3DPOOL_MANAGED);
    assert_eq!(texture.level_desc(0).1.format, D3DFMT_A2B10G10R10);
}

/// Red is the low lane and alpha the top two bits, with all ten bits of each colour lane kept.
#[test]
fn a2b10g10r10_channels_alpha_and_lower_rgb_bits() {
    super::packed10::channels_alpha_and_lower_rgb_bits(D3DFMT_A2B10G10R10);
}

/// Locks expose the native words in every pool, and copies and updates move them unchanged.
#[test]
fn a2b10g10r10_native_words_pools_mips_and_updates() {
    super::packed10::native_words_pools_mips_and_updates(D3DFMT_A2B10G10R10);
}

/// Queries agree with creation, with one level and the usage kept for AUTOGEN requests.
#[test]
fn a2b10g10r10_queries_and_noautogen() {
    super::packed10::queries_and_noautogen(D3DFMT_A2B10G10R10);
}

/// Cube faces, volume slices and their mips stay apart, and filtering and wrapping are native.
#[test]
fn a2b10g10r10_cube_volume_filter_and_mips() {
    super::packed10::cube_volume_filter_and_mips(D3DFMT_A2B10G10R10);
}

/// `ColorFill` writes the nearest codes, and a lock after a GPU copy reads the copied words.
#[test]
fn a2b10g10r10_colorfill_and_gpu_authority() {
    super::packed10::colorfill_and_gpu_authority(D3DFMT_A2B10G10R10);
}

/// Copies to and from A2R10G10B10 and unrelated formats are rejected with no write.
#[test]
fn a2b10g10r10_rejects_mixed_raw_copies() {
    super::packed10::rejects_mixed_raw_copies(D3DFMT_A2B10G10R10);
}

/// One word samples with red and blue exchanged between the two ten-bit formats.
#[test]
fn a2b10g10r10_and_a2r10g10b10_have_distinct_raw_lane_order() {
    let h = Harness::new();
    let rgb = h.create_texture(1, 1, 1, 0, D3DFMT_A2B10G10R10, D3DPOOL_MANAGED);
    let bgr = h.create_texture(1, 1, 1, 0, D3DFMT_A2R10G10B10, D3DPOOL_MANAGED);
    for alpha in 0..4 {
        let same_word = (alpha << 30) | (1022 << 20) | (341 << 10) | 5;
        rgb.lock_rect(0, 0).write_u32(&[same_word]);
        bgr.lock_rect(0, 0).write_u32(&[same_word]);
        let expected_rgb = ((alpha * 85) << 24) | 0x0001_55ff;
        let expected_bgr = ((alpha * 85) << 24) | 0x00ff_5501;
        assert_pixel_approx(
            sample(&h, &rgb, [0.5; 2]),
            expected_rgb,
            1,
            "red in the low lane",
        );
        assert_pixel_approx(
            sample(&h, &bgr, [0.5; 2]),
            expected_bgr,
            1,
            "blue in the low lane",
        );
        bgr.lock_rect(0, 0)
            .write_u32(&[packed(D3DFMT_A2R10G10B10, 5, 341, 1022, alpha)]);
        assert_pixel_approx(
            sample(&h, &bgr, [0.5; 2]),
            expected_rgb,
            1,
            "outer lanes exchanged, same colour",
        );
    }
}
