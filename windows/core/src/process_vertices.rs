//! Software vertex processing for `IDirect3DDevice9::ProcessVertices`.
//!
//! `ProcessVertices` runs the fixed-function transform on the CPU and writes
//! the result to a destination vertex buffer, rather than to the screen. The
//! source is read through the current FVF from stream 0; the destination
//! format is the destination buffer's own FVF. The position is transformed by
//! the world-view-projection matrix, perspective-divided, and mapped through
//! the viewport into screen coordinates plus `rhw = 1/w` (a `D3DFVF_XYZRHW`
//! output). Any other component the destination FVF names is copied from the
//! matching source component.
//!
//! This is the CPU path only: it does not run per-vertex lighting, so a
//! destination that asks for a lit colour under `D3DRS_LIGHTING` gets the
//! source colour passed through. Callers that care log the divergence; no
//! shipped target uses software vertex processing.

use mtld3d_types::{
    D3DDECLUSAGE_COLOR, D3DDECLUSAGE_POSITION, D3DDECLUSAGE_POSITIONT, D3DDECLUSAGE_TEXCOORD,
    D3DMATRIX, D3DVIEWPORT9,
};

use crate::convert::{decl_type_to_metal_format, fvf_to_elements};

/// One source-and-destination request for [`process_vertices`].
pub struct ProcessVerticesRequest<'a> {
    /// Source vertex bytes, starting at the first vertex to process.
    pub src: &'a [u8],
    /// Source vertex stride in bytes (the bound stream-0 stride).
    pub src_stride: u32,
    /// Source FVF (the device's current FVF).
    pub src_fvf: u32,
    /// Destination FVF (the destination buffer's creation FVF).
    pub dst_fvf: u32,
    /// Number of vertices to transform.
    pub count: u32,
    /// The composite world-view-projection matrix in D3D9 row-vector order.
    pub wvp: D3DMATRIX,
    /// The device viewport the transformed positions map through.
    pub viewport: D3DVIEWPORT9,
}

/// Transform `req.count` vertices and return the destination bytes.
///
/// The returned buffer is `count * dst_stride` bytes, ready to copy into the
/// destination vertex buffer at its start offset. Returns `None` when the
/// destination FVF carries no position (nothing to transform into) or the
/// source is too short for the requested count.
#[must_use]
pub fn process_vertices(req: &ProcessVerticesRequest) -> Option<Vec<u8>> {
    let (src_elems, src_stride) = fvf_to_elements(req.src_fvf);
    let (dst_elems, dst_stride) = fvf_to_elements(req.dst_fvf);
    let src_stride = if req.src_stride != 0 {
        req.src_stride
    } else {
        src_stride
    };
    let count = req.count as usize;
    let src_stride = src_stride as usize;
    let dst_stride = dst_stride as usize;
    if src_stride == 0 || dst_stride == 0 {
        return None;
    }
    if req.src.len() < count.saturating_mul(src_stride) {
        return None;
    }
    // The destination position: a transformed `POSITIONT` (XYZRHW) is the whole
    // point of ProcessVertices. A destination without one has nothing to
    // transform into.
    let dst_pos = dst_elems
        .iter()
        .find(|e| e.usage == D3DDECLUSAGE_POSITIONT)?;
    let src_pos = src_elems
        .iter()
        .find(|e| e.usage == D3DDECLUSAGE_POSITION || e.usage == D3DDECLUSAGE_POSITIONT)?;

    let mut out = vec![0u8; count * dst_stride];
    for i in 0..count {
        let src_base = i * src_stride;
        let dst_base = i * dst_stride;
        let pos = read_vec3(&req.src[src_base + src_pos.offset as usize..]);
        let screen = transform_to_screen(pos, &req.wvp, &req.viewport);
        write_vec4(&mut out[dst_base + dst_pos.offset as usize..], screen);

        // Every other destination component is copied from the source
        // component with the same usage and index, truncated to the smaller
        // of the two byte sizes. An absent source component stays zero.
        for de in &dst_elems {
            if de.usage == D3DDECLUSAGE_POSITIONT {
                continue;
            }
            let Some(se) = src_elems
                .iter()
                .find(|se| se.usage == de.usage && se.usage_index == de.usage_index)
            else {
                continue;
            };
            let n = decl_type_to_metal_format(de.type_)
                .1
                .min(decl_type_to_metal_format(se.type_).1) as usize;
            let (so, dsto) = (se.offset as usize, de.offset as usize);
            out[dst_base + dsto..dst_base + dsto + n]
                .copy_from_slice(&req.src[src_base + so..src_base + so + n]);
        }
    }
    Some(out)
}

/// Whether the destination FVF asks for a shaded output the CPU path cannot produce.
///
/// Used by the caller to decide whether to warn that software lighting is not
/// run (a lit destination colour is passed through from the source instead).
#[must_use]
pub fn dst_wants_shaded_output(dst_fvf: u32) -> bool {
    let (elems, _) = fvf_to_elements(dst_fvf);
    elems
        .iter()
        .any(|e| e.usage == D3DDECLUSAGE_COLOR || e.usage == D3DDECLUSAGE_TEXCOORD)
}

fn read_vec3(b: &[u8]) -> [f32; 3] {
    [
        f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        f32::from_le_bytes([b[4], b[5], b[6], b[7]]),
        f32::from_le_bytes([b[8], b[9], b[10], b[11]]),
    ]
}

fn write_vec4(b: &mut [u8], v: [f32; 4]) {
    for (i, c) in v.iter().enumerate() {
        b[i * 4..i * 4 + 4].copy_from_slice(&c.to_le_bytes());
    }
}

/// `pos * WVP`, perspective divide, then the D3D9 viewport map to screen space.
///
/// Returns `(screen_x, screen_y, screen_z, rhw)` — the `D3DFVF_XYZRHW` result
/// with `rhw = 1/w`.
fn transform_to_screen(pos: [f32; 3], wvp: &D3DMATRIX, vp: &D3DVIEWPORT9) -> [f32; 4] {
    let m = &wvp.m;
    // D3D9 row-vector transform: clip[j] = Σ_k v[k]·M[k*4+j], v = (x, y, z, 1).
    let v = [pos[0], pos[1], pos[2], 1.0];
    // Array-sum, not a fused polynomial: keep the products unfused so a host
    // and a device build agree bit-for-bit (matches the FF VS constant math).
    let mut clip = [0.0f32; 4];
    for (j, c) in clip.iter_mut().enumerate() {
        *c = [
            v[0] * m[j],
            v[1] * m[4 + j],
            v[2] * m[8 + j],
            v[3] * m[12 + j],
        ]
        .iter()
        .sum();
    }
    let w = clip[3];
    let inv_w = if w.abs() > 0.0 { w.recip() } else { 0.0 };
    let ndc = [clip[0] * inv_w, clip[1] * inv_w, clip[2] * inv_w];
    // Viewport dims/origin fit u16 in practice (D3D9 max RT dim is 16384);
    // convert without an `as`-cast precision-loss lint (the draw-path idiom).
    let to_f = |v: u32| f32::from(u16::try_from(v).unwrap_or(u16::MAX));
    let unit_x = ndc[0].mul_add(0.5, 0.5);
    let unit_y = ndc[1].mul_add(-0.5, 0.5);
    [
        unit_x.mul_add(to_f(vp.width), to_f(vp.x)),
        unit_y.mul_add(to_f(vp.height), to_f(vp.y)),
        ndc[2].mul_add(vp.max_z - vp.min_z, vp.min_z),
        inv_w,
    ]
}

#[cfg(test)]
mod tests;
