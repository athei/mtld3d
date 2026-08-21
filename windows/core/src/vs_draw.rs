//! Per-draw vertex-stage uniform shared by the fixed-function and DXSO vertex shaders.
//!
//! Bound as inline bytes at `VS_DRAW_SLOT` (see `mtld3d_shared::mtl`) and
//! deduplicated per pass by the encoder, so a draw only pays for it when one
//! of its inputs changed. It carries the render states a vertex shader reads at
//! runtime without forking a shader variant: the point size with its clamp
//! range and the point scale factors. [`VS_DRAW_MSL`] is the MSL view of the
//! same bytes; the two move in lock-step.

use mtld3d_types::{
    D3DRS_POINTSCALE_A, D3DRS_POINTSCALE_B, D3DRS_POINTSCALE_C, D3DRS_POINTSIZE,
    D3DRS_POINTSIZE_MAX, D3DRS_POINTSIZE_MIN, RENDER_STATE_COUNT,
};

/// Size of the uniform in bytes: two `float4` rows.
pub const VS_DRAW_BYTES: usize = 32;

/// MSL declaration of the uniform, emitted ahead of every vertex function.
///
/// `point = (D3DRS_POINTSIZE, D3DRS_POINTSIZE_MIN, D3DRS_POINTSIZE_MAX, 0)`
/// and `point_scale = (D3DRS_POINTSCALE_A, _B, _C, 0)`.
pub const VS_DRAW_MSL: &str = "struct VsDraw {\n    float4 point;\n    float4 point_scale;\n};\n\n";

/// Serialise the point render states into the uniform's byte layout.
///
/// Each state DWORD already holds an f32 bit pattern, so the lanes are copied
/// through verbatim; the fourth lane of each row is padding.
#[must_use]
pub fn build_vs_draw_bytes(rs: &[u32; RENDER_STATE_COUNT]) -> [u8; VS_DRAW_BYTES] {
    let lanes = [
        rs[D3DRS_POINTSIZE as usize],
        rs[D3DRS_POINTSIZE_MIN as usize],
        rs[D3DRS_POINTSIZE_MAX as usize],
        0,
        rs[D3DRS_POINTSCALE_A as usize],
        rs[D3DRS_POINTSCALE_B as usize],
        rs[D3DRS_POINTSCALE_C as usize],
        0,
    ];
    let mut out = [0u8; VS_DRAW_BYTES];
    for (chunk, lane) in out.chunks_exact_mut(4).zip(lanes) {
        chunk.copy_from_slice(&lane.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use mtld3d_types::render_state_defaults;

    use super::*;

    fn lane(bytes: &[u8; VS_DRAW_BYTES], i: usize) -> f32 {
        f32::from_le_bytes([
            bytes[i * 4],
            bytes[i * 4 + 1],
            bytes[i * 4 + 2],
            bytes[i * 4 + 3],
        ])
    }

    #[test]
    fn defaults_pack_size_one_clamped_to_the_cap_and_identity_scale() {
        let bytes = build_vs_draw_bytes(&render_state_defaults());
        assert_eq!(lane(&bytes, 0), 1.0, "POINTSIZE");
        assert_eq!(lane(&bytes, 1), 1.0, "POINTSIZE_MIN");
        assert_eq!(
            lane(&bytes, 2),
            mtld3d_types::MAX_POINT_SIZE,
            "POINTSIZE_MAX"
        );
        assert_eq!(lane(&bytes, 4), 1.0, "POINTSCALE_A");
        assert_eq!(lane(&bytes, 5), 0.0, "POINTSCALE_B");
        assert_eq!(lane(&bytes, 6), 0.0, "POINTSCALE_C");
    }

    #[test]
    fn every_point_state_lands_in_its_lane() {
        let mut rs = render_state_defaults();
        rs[D3DRS_POINTSIZE as usize] = 32.0f32.to_bits();
        rs[D3DRS_POINTSIZE_MIN as usize] = 2.0f32.to_bits();
        rs[D3DRS_POINTSIZE_MAX as usize] = 48.0f32.to_bits();
        rs[D3DRS_POINTSCALE_A as usize] = 0.5f32.to_bits();
        rs[D3DRS_POINTSCALE_B as usize] = 0.25f32.to_bits();
        rs[D3DRS_POINTSCALE_C as usize] = 0.125f32.to_bits();
        let bytes = build_vs_draw_bytes(&rs);
        assert_eq!(
            [0, 1, 2, 4, 5, 6].map(|i| lane(&bytes, i)),
            [32.0, 2.0, 48.0, 0.5, 0.25, 0.125]
        );
        assert_eq!(lane(&bytes, 3), 0.0);
        assert_eq!(lane(&bytes, 7), 0.0);
    }

    #[test]
    fn msl_struct_matches_the_byte_layout() {
        // Two float4 rows, named as the emitters read them.
        assert!(VS_DRAW_MSL.contains("float4 point;"));
        assert!(VS_DRAW_MSL.contains("float4 point_scale;"));
        assert_eq!(VS_DRAW_MSL.matches("float4").count() * 16, VS_DRAW_BYTES);
    }
}
