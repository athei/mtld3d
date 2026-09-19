//! Per-draw vertex-stage uniform shared by the fixed-function and DXSO vertex shaders.
//!
//! Bound as inline bytes at `VS_DRAW_SLOT` (see `mtld3d_shared::mtl`) and
//! deduplicated per pass by the encoder, so a draw only pays for it when one
//! of its inputs changed. It carries the state a vertex shader reads at
//! runtime without forking a shader variant: the point size with its clamp
//! range and the point scale factors, the inverse view matrix the
//! fixed-function shader needs to get from eye space back to world space, and
//! the user clip planes. [`VS_DRAW_MSL`] is the MSL view of the same bytes;
//! the two move in lock-step.

use mtld3d_types::{
    D3DMATRIX, D3DRS_CLIPPING, D3DRS_CLIPPLANEENABLE, D3DRS_POINTSCALE_A, D3DRS_POINTSCALE_B,
    D3DRS_POINTSCALE_C, D3DRS_POINTSIZE_MAX, D3DRS_POINTSIZE_MIN, RENDER_STATE_COUNT,
};

use crate::ff_state::FfState;

/// User clip planes a draw can apply at once: `D3DCAPS9::MaxUserClipPlanes`.
///
/// Six is what every D3D9-era GPU offered; the
/// vertex shader emits one `[[clip_distance]]` lane per enabled plane.
pub const MAX_CLIP_PLANES: usize = 6;

/// Size of the uniform in bytes: twelve `float4` rows.
pub const VS_DRAW_BYTES: usize = 16 * (2 + 4 + MAX_CLIP_PLANES);

/// MSL declaration of the uniform, emitted ahead of every vertex function.
///
/// `point = (numeric_point_size, D3DRS_POINTSIZE_MIN, D3DRS_POINTSIZE_MAX, 0)`,
/// `point_scale = (D3DRS_POINTSCALE_A, _B, _C, 0)`, `inv_view` the rows of
/// `transpose(inverse(D3DTS_VIEW))` so `dot(pos_view, inv_view[i])` is lane
/// `i` of the world-space position, and `clip` the enabled user clip planes
/// packed from index 0 (zero past the count).
pub const VS_DRAW_MSL: &str = "struct VsDraw {\n    float4 point;\n    float4 point_scale;\n    float4 inv_view[4];\n    float4 clip[6];\n};\n\n";

/// Device-local inverse view, independent of transient draw snapshots.
///
/// The key is checked when packing a draw, so bulk state restoration and
/// transform multiplication cannot leave a stale inverse behind. Point-state
/// and plane changes reuse it, as do new frames with an unchanged view.
pub struct VsDrawState {
    inverse: Option<InverseView>,
    #[cfg(test)]
    inversions: usize,
}

impl Default for VsDrawState {
    fn default() -> Self {
        Self::new()
    }
}

impl VsDrawState {
    /// Create a device's empty inverse-view cache.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            inverse: None,
            #[cfg(test)]
            inversions: 0,
        }
    }

    /// Pack draw constants, computing an inverse only for enabled clip planes.
    ///
    /// `point_size` is the last numeric POINTSIZE, excluding driver controls.
    /// Unused inverse rows are identity. Active planes retain the general
    /// inverse's arithmetic and its identity fallback for a singular view.
    #[must_use]
    pub fn build_bytes(
        &mut self,
        rs: &[u32; RENDER_STATE_COUNT],
        point_size: u32,
        view: &D3DMATRIX,
        planes: &[[f32; 4]],
    ) -> [u8; VS_DRAW_BYTES] {
        let rows = if clip_plane_count(rs) == 0 {
            &D3DMATRIX::IDENTITY
        } else {
            self.inverse_rows(view)
        };
        pack_bytes(rs, point_size, rows, planes)
    }

    fn inverse_rows(&mut self, view: &D3DMATRIX) -> &D3DMATRIX {
        let view_bits = view.m.map(f32::to_bits);
        if self
            .inverse
            .as_ref()
            .is_none_or(|entry| entry.view_bits != view_bits)
        {
            let inverse = FfState::inverse(view).unwrap_or_else(|| {
                mtld3d_shared::log_once_warn!(
                    target: crate::LOG_TARGET,
                    "D3DTS_VIEW is singular; user clip planes use the identity view"
                );
                D3DMATRIX::IDENTITY
            });
            self.inverse = Some(InverseView {
                view_bits,
                rows: FfState::transpose(&inverse),
            });
            #[cfg(test)]
            {
                self.inversions += 1;
            }
        }
        &self
            .inverse
            .as_ref()
            .expect("inverse was populated for this view")
            .rows
    }
}

/// Number of user clip planes a draw applies.
///
/// The enabled planes among the first [`MAX_CLIP_PLANES`] of
/// `D3DRS_CLIPPLANEENABLE`, or none while `D3DRS_CLIPPING`, the master
/// clipping switch, is off. Folded into the vertex-shader keys, which is
/// why it lives next to the byte layout the keyed shader reads.
#[must_use]
pub fn clip_plane_count(rs: &[u32; RENDER_STATE_COUNT]) -> u8 {
    if rs[D3DRS_CLIPPING as usize] == 0 {
        return 0;
    }
    let mask = rs[D3DRS_CLIPPLANEENABLE as usize] & ((1u32 << MAX_CLIP_PLANES) - 1);
    // At most six bits survive the mask, so the conversion cannot fail.
    u8::try_from(mask.count_ones()).unwrap_or(6)
}

struct InverseView {
    view_bits: [u32; 16],
    rows: D3DMATRIX,
}

/// Serialise point states, transposed inverse rows and enabled planes.
fn pack_bytes(
    rs: &[u32; RENDER_STATE_COUNT],
    point_size: u32,
    inverse_rows: &D3DMATRIX,
    planes: &[[f32; 4]],
) -> [u8; VS_DRAW_BYTES] {
    let mut lanes = [0u32; VS_DRAW_BYTES / 4];
    lanes[0] = point_size;
    lanes[1] = rs[D3DRS_POINTSIZE_MIN as usize];
    lanes[2] = rs[D3DRS_POINTSIZE_MAX as usize];
    lanes[4] = rs[D3DRS_POINTSCALE_A as usize];
    lanes[5] = rs[D3DRS_POINTSCALE_B as usize];
    lanes[6] = rs[D3DRS_POINTSCALE_C as usize];
    for (lane, v) in lanes[8..24].iter_mut().zip(&inverse_rows.m) {
        *lane = v.to_bits();
    }
    let mut next = 24;
    if rs[D3DRS_CLIPPING as usize] != 0 {
        let mask = rs[D3DRS_CLIPPLANEENABLE as usize];
        for (i, plane) in planes.iter().take(MAX_CLIP_PLANES).enumerate() {
            if mask & (1 << i) == 0 {
                continue;
            }
            for (lane, v) in lanes[next..next + 4].iter_mut().zip(plane) {
                *lane = v.to_bits();
            }
            next += 4;
        }
    }
    let mut out = [0u8; VS_DRAW_BYTES];
    for (chunk, lane) in out.as_chunks_mut::<4>().0.iter_mut().zip(lanes) {
        *chunk = lane.to_le_bytes();
    }
    out
}

#[cfg(test)]
mod tests;
