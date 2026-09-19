//! Native packed ten-bit A2R10G10B10 texture regressions.

use mtld3d_tests::Harness;
use mtld3d_types::{D3DFMT_A2R10G10B10, D3DPOOL_MANAGED};

/// Creation needs no configuration key and reports the requested format.
#[test]
fn a2r10g10b10_creation_is_not_capability_gated() {
    let h = Harness::new();
    let texture = h.create_texture(3, 2, 1, 0, D3DFMT_A2R10G10B10, D3DPOOL_MANAGED);
    assert_eq!(texture.level_desc(0).1.format, D3DFMT_A2R10G10B10);
}

/// Blue is the low lane and alpha the top two bits, with all ten bits of each colour lane kept.
#[test]
fn a2r10g10b10_channels_alpha_and_lower_rgb_bits() {
    super::packed10::channels_alpha_and_lower_rgb_bits(D3DFMT_A2R10G10B10);
}

/// Locks expose the native words in every pool, and copies and updates move them unchanged.
#[test]
fn a2r10g10b10_native_words_pools_mips_and_updates() {
    super::packed10::native_words_pools_mips_and_updates(D3DFMT_A2R10G10B10);
}

/// Queries agree with creation, with one level and the usage kept for AUTOGEN requests.
#[test]
fn a2r10g10b10_queries_and_noautogen() {
    super::packed10::queries_and_noautogen(D3DFMT_A2R10G10B10);
}

/// Cube faces, volume slices and their mips stay apart, and filtering and wrapping are native.
#[test]
fn a2r10g10b10_cube_volume_filter_and_mips() {
    super::packed10::cube_volume_filter_and_mips(D3DFMT_A2R10G10B10);
}

/// `ColorFill` writes the nearest codes, and a lock after a GPU copy reads the copied words.
#[test]
fn a2r10g10b10_colorfill_and_gpu_authority() {
    super::packed10::colorfill_and_gpu_authority(D3DFMT_A2R10G10B10);
}

/// Copies to and from A2B10G10R10 and unrelated formats are rejected with no write.
#[test]
fn a2r10g10b10_rejects_mixed_raw_copies() {
    super::packed10::rejects_mixed_raw_copies(D3DFMT_A2R10G10B10);
}
