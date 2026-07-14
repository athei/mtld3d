//! Vertex/pixel shader bindings and constant buffers owned by `DeviceInner`.
//!
//! Shader pointer slots own one refcount each via [`CachedComPtr`];
//! `replace_*` setters transfer one refcount into the slot and release
//! the prior slot value via auto-`Drop` on assignment.

use super::{
    com_ref::{Bound, CachedComPtr},
    pixel_shader::Direct3DPixelShader9,
    vertex_shader::Direct3DVertexShader9,
};

/// Float constant registers (`c0..c255`), each a `float4`.
///
/// Doubles as the `vs_3_0` `Set*ShaderConstantF` register-window ceiling:
/// a vertex write whose `[start, start + count)` exceeds this returns
/// `D3DERR_INVALIDCALL` rather than clamping silently into the mirror.
pub const CONSTANT_ROWS: usize = 256;
/// `ps_3_0` `SetPixelShaderConstantF` register-window ceiling.
///
/// The pixel pipeline exposes fewer float constants than the vertex one
/// (224 vs 256), so a pixel write past row 223 is `D3DERR_INVALIDCALL`.
/// Fits within the shared [`CONSTANT_ROWS`] mirror.
pub const PS_FLOAT_CONSTANT_LIMIT: usize = 224;
/// Integer constant registers (`i0..i15`), each an `int4`.
///
/// The SM2/SM3 ceiling — D3DCAPS9 has no per-device override for this
/// count.
pub const INT_CONSTANT_ROWS: usize = 16;
/// Boolean constant registers (`b0..b15`), each a single D3D9 `BOOL`.
///
/// Stored as `i32`, matching the `Set*ShaderConstantB(const BOOL*)` ABI.
pub const BOOL_CONSTANT_COUNT: usize = 16;

pub struct ShaderBindings {
    /// Currently bound vertex shader slot.
    ///
    /// Uses the `Bound` ownership marker — swaps bump the wrapper's
    /// `private_refcount` inline.
    vertex_shader: CachedComPtr<Direct3DVertexShader9, Bound>,
    /// Currently bound pixel shader slot. Same `Bound` semantics.
    pixel_shader: CachedComPtr<Direct3DPixelShader9, Bound>,
    vs_constants: [[f32; 4]; CONSTANT_ROWS],
    ps_constants: [[f32; 4]; CONSTANT_ROWS],
    vs_constants_i: [[i32; 4]; INT_CONSTANT_ROWS],
    vs_constants_b: [i32; BOOL_CONSTANT_COUNT],
    ps_constants_i: [[i32; 4]; INT_CONSTANT_ROWS],
    ps_constants_b: [i32; BOOL_CONSTANT_COUNT],
}

impl ShaderBindings {
    pub const fn new() -> Self {
        Self {
            vertex_shader: CachedComPtr::null(),
            pixel_shader: CachedComPtr::null(),
            vs_constants: [[0.0; 4]; CONSTANT_ROWS],
            ps_constants: [[0.0; 4]; CONSTANT_ROWS],
            vs_constants_i: [[0; 4]; INT_CONSTANT_ROWS],
            vs_constants_b: [0; BOOL_CONSTANT_COUNT],
            ps_constants_i: [[0; 4]; INT_CONSTANT_ROWS],
            ps_constants_b: [0; BOOL_CONSTANT_COUNT],
        }
    }

    pub const fn vertex_shader(&self) -> *mut Direct3DVertexShader9 {
        self.vertex_shader.raw()
    }

    pub const fn pixel_shader(&self) -> *mut Direct3DPixelShader9 {
        self.pixel_shader.raw()
    }

    /// Bind `new` as the current vertex shader, transferring one refcount to the slot.
    ///
    /// The transfer runs via [`CachedComPtr::adopt`], releasing the prior
    /// slot value via auto-`Drop` on assignment.
    ///
    /// Returns whether the bound pointer changed. A bound shader is kept
    /// alive by its slot refcount, so identical pointers mean the same
    /// immutable object (same id / input semantics / `max_const_used`) —
    /// callers gate the VDECL / VS-source re-resolve on this.
    pub fn replace_vertex_shader(&mut self, new: *mut Direct3DVertexShader9) -> bool {
        let changed = self.vertex_shader.raw() != new;
        // SAFETY: `new` is null or a live IDirect3DVertexShader9 supplied
        // by the calling D3D9 vtable thunk; AddRef/Release thunks valid
        // for our lifetime.
        self.vertex_shader = unsafe { CachedComPtr::adopt(new) };
        changed
    }

    /// Bind `new` as the current pixel shader, transferring one refcount to the slot.
    ///
    /// The transfer runs via [`CachedComPtr::adopt`], releasing the prior
    /// slot value via auto-`Drop` on assignment.
    ///
    /// Returns whether the bound pointer changed — same soundness as
    /// [`Self::replace_vertex_shader`]: an equal pointer is the same
    /// immutable shader, so callers gate the PS-source re-resolve on it.
    pub fn replace_pixel_shader(&mut self, new: *mut Direct3DPixelShader9) -> bool {
        let changed = self.pixel_shader.raw() != new;
        // SAFETY: see [`Self::replace_vertex_shader`].
        self.pixel_shader = unsafe { CachedComPtr::adopt(new) };
        changed
    }

    /// Write `data` into the VS constant mirror starting at row `start`.
    ///
    /// Returns whether any row actually changed.
    ///
    /// Redundant-set elimination: a same-value constant write produces a
    /// byte-identical mirror (and so a byte-identical encoder delta), so
    /// callers gate the encoder propagation + snapshot dirty-mark on the
    /// returned bool. The compare is per-`f32` via `to_bits`, exact and
    /// NaN-safe.
    pub fn write_vs_constants(&mut self, start: u32, data: &[[f32; 4]]) -> bool {
        let start = start as usize;
        let end = (start + data.len()).min(CONSTANT_ROWS);
        if start >= end {
            return false;
        }
        let dst = &mut self.vs_constants[start..end];
        let src = &data[..end - start];
        let changed = rows_differ(dst, src);
        if changed {
            dst.copy_from_slice(src);
        }
        changed
    }

    /// Write `data` into the PS constant mirror starting at row `start`.
    ///
    /// Returns whether any row actually changed. Same redundant-set
    /// semantics as [`Self::write_vs_constants`].
    pub fn write_ps_constants(&mut self, start: u32, data: &[[f32; 4]]) -> bool {
        let start = start as usize;
        let end = (start + data.len()).min(CONSTANT_ROWS);
        if start >= end {
            return false;
        }
        let dst = &mut self.ps_constants[start..end];
        let src = &data[..end - start];
        let changed = rows_differ(dst, src);
        if changed {
            dst.copy_from_slice(src);
        }
        changed
    }

    pub const fn vs_constants_copy(&self) -> [[f32; 4]; CONSTANT_ROWS] {
        self.vs_constants
    }

    pub const fn ps_constants_copy(&self) -> [[f32; 4]; CONSTANT_ROWS] {
        self.ps_constants
    }

    /// Write `data` into the VS integer-constant mirror starting at row `start`.
    ///
    /// Returns whether any row changed. Same redundant-set semantics as
    /// [`Self::write_vs_constants`]; `i32` rows compare directly (no
    /// `to_bits` — integers have no NaN aliasing).
    pub fn write_vs_constants_i(&mut self, start: u32, data: &[[i32; 4]]) -> bool {
        write_rows(&mut self.vs_constants_i, start, data)
    }

    /// Write `data` into the VS boolean-constant mirror starting at row `start`.
    ///
    /// Returns whether any value changed.
    pub fn write_vs_constants_b(&mut self, start: u32, data: &[i32]) -> bool {
        write_rows(&mut self.vs_constants_b, start, data)
    }

    /// PS integer-constant analogue of [`Self::write_vs_constants_i`].
    pub fn write_ps_constants_i(&mut self, start: u32, data: &[[i32; 4]]) -> bool {
        write_rows(&mut self.ps_constants_i, start, data)
    }

    /// PS boolean-constant analogue of [`Self::write_vs_constants_b`].
    pub fn write_ps_constants_b(&mut self, start: u32, data: &[i32]) -> bool {
        write_rows(&mut self.ps_constants_b, start, data)
    }

    pub const fn vs_constants_i_copy(&self) -> [[i32; 4]; INT_CONSTANT_ROWS] {
        self.vs_constants_i
    }

    /// The VS integer-constant file as native-endian bytes.
    ///
    /// `INT_CONSTANT_ROWS` × `int4` = 256 B, ready to bind as the shader's
    /// `vs_i` buffer (slot 14).
    pub fn vs_constants_i_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(INT_CONSTANT_ROWS * 4 * 4);
        for row in &self.vs_constants_i {
            for &v in row {
                out.extend_from_slice(&v.to_ne_bytes());
            }
        }
        out
    }

    pub const fn vs_constants_b_copy(&self) -> [i32; BOOL_CONSTANT_COUNT] {
        self.vs_constants_b
    }

    pub const fn ps_constants_i_copy(&self) -> [[i32; 4]; INT_CONSTANT_ROWS] {
        self.ps_constants_i
    }

    pub const fn ps_constants_b_copy(&self) -> [i32; BOOL_CONSTANT_COUNT] {
        self.ps_constants_b
    }
}

/// Copy `data` into `dst_all` starting at row `start`, clamping to the register file's length.
///
/// Returns whether any row actually changed. Shared by the integer and
/// boolean constant writers — both back a fixed-size mirror and want the
/// same redundant-set gate as the float path, but compare by value
/// (`PartialEq`) rather than `f32::to_bits`.
fn write_rows<T: Copy + PartialEq>(dst_all: &mut [T], start: u32, data: &[T]) -> bool {
    let start = start as usize;
    let end = (start + data.len()).min(dst_all.len());
    if start >= end {
        return false;
    }
    let dst = &mut dst_all[start..end];
    let src = &data[..end - start];
    let changed = dst != src;
    if changed {
        dst.copy_from_slice(src);
    }
    changed
}

/// Whether the existing rows `cur` differ from the incoming rows `new`.
///
/// Compared per-`f32` via `to_bits` (exact, NaN-safe — a same-value write
/// must read as unchanged so the redundant-set gate can fire). The two
/// slices are the same length by construction at every call site.
fn rows_differ(cur: &[[f32; 4]], new: &[[f32; 4]]) -> bool {
    cur.as_flattened()
        .iter()
        .zip(new.as_flattened())
        .any(|(a, b)| a.to_bits() != b.to_bits())
}
