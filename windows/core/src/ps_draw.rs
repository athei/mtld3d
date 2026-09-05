//! Per-draw fragment-stage uniform for the pixel-shader reads that depend on the render scale.
//!
//! Bound as inline bytes at `PS_DRAW_SLOT` (see `mtld3d_shared::mtl`) for a
//! draw whose pixel shader reads `vPos` into a target rasterized below the
//! resolution D3D9 reports, and deduplicated per pass by the encoder. It
//! carries the logical pixels per render pixel, which the emitter applies to
//! the rasterized position so the register reads in the reported space.
//! [`PS_DRAW_MSL`] is the MSL view of the same bytes; the two move in
//! lock-step.

use crate::render_scale::RenderScale;

/// Size of the uniform in bytes: one `float4` row.
pub const PS_DRAW_BYTES: usize = 16;

/// MSL declaration of the uniform, emitted ahead of a pixel function that reads it.
///
/// `vpos_scale = (1 / scale, 1 / scale, 0, 0)`: logical pixels per render
/// pixel along each axis.
pub const PS_DRAW_MSL: &str = "struct PsDraw {\n    float4 vpos_scale;\n};\n\n";

/// The uniform's bytes for a target rasterized at `scale`.
#[must_use]
pub fn build_ps_draw_bytes(scale: RenderScale) -> [u8; PS_DRAW_BYTES] {
    let per_render_pixel = 1.0 / scale.factor();
    let lanes = [per_render_pixel, per_render_pixel, 0.0, 0.0];
    let mut out = [0u8; PS_DRAW_BYTES];
    for (chunk, lane) in out.chunks_exact_mut(4).zip(lanes) {
        chunk.copy_from_slice(&lane.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests;
