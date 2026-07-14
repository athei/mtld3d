//! Fixed-Function MSL emitter.
//!
//! Converts a pair of `FFVSKey` / `FFPSKey` (summaries of the current D3D9
//! fixed-function state) into Metal Shading Language text that plugs into the
//! same `mtld3d_vs` / `mtld3d_ps` entry-point contract used by the DXSO
//! translator in `emit.rs`:
//!
//! - Vertex attributes at `[[attribute(N)]]` with N matching
//!   `fvf_to_vertex_attrs` (0=position, 1=normal, 2=diffuse, 3=specular,
//!   4..11=texcoord0..7).
//! - Varyings struct: `position` first, then `texcoord0..7`, then
//!   `color0..1` (texcoord-before-color workaround for a Metal
//!   shader-compiler crash).
//! - Constants on buffer slot 15 as `float4 *vs_c` / `ps_c` with layout
//!   documented inline where they are referenced.
//!
//! Feature scope: World/View/Projection transform, directional/point
//! lighting with Blinn-Phong specular (spot collapses to directional), the
//! common texture-stage combiner ops, the end-of-cascade specular add, and
//! alpha test. Remaining advanced FF features are deferred.
//!
//! `FfVsKey` / `FfStage` / `FfPsKey` fields are part of the crate's public
//! cache-key contract with `d3d9::ff_state` — callers construct them
//! field-by-field from D3D9 state and use them as hash-map keys, so every
//! field is deliberately `pub`. Unlike the (private) IR operand types, the
//! transparency here is intentional.

use std::fmt::Write;

use mtld3d_types::{
    D3DCMP_ALWAYS, D3DCMP_EQUAL, D3DCMP_GREATER, D3DCMP_GREATEREQUAL, D3DCMP_LESS,
    D3DCMP_LESSEQUAL, D3DCMP_NEVER, D3DCMP_NOTEQUAL, D3DDECLUSAGE_BLENDINDICES,
    D3DDECLUSAGE_BLENDWEIGHT, D3DDECLUSAGE_COLOR, D3DDECLUSAGE_NORMAL, D3DDECLUSAGE_POSITION,
    D3DDECLUSAGE_POSITIONT, D3DDECLUSAGE_TEXCOORD, D3DTA_ALPHAREPLICATE, D3DTA_COMPLEMENT,
    D3DTA_CURRENT, D3DTA_DIFFUSE, D3DTA_SPECULAR, D3DTA_TEXTURE, D3DTA_TFACTOR, D3DTOP_ADD,
    D3DTOP_ADDSIGNED, D3DTOP_ADDSIGNED2X, D3DTOP_ADDSMOOTH, D3DTOP_BLENDCURRENTALPHA,
    D3DTOP_BLENDDIFFUSEALPHA, D3DTOP_BLENDFACTORALPHA, D3DTOP_BLENDTEXTUREALPHA, D3DTOP_DISABLE,
    D3DTOP_DOTPRODUCT3, D3DTOP_MODULATE, D3DTOP_MODULATE2X, D3DTOP_MODULATE4X, D3DTOP_SELECTARG1,
    D3DTOP_SELECTARG2, D3DTOP_SUBTRACT,
};

use super::emit::{VariantFlags, VariantKey, fog_blend_active, write_fog_blend};

// The FF emitter stores D3D9 texture-op / texture-arg / compare-func codes in
// `u8` cache-key fields and matches on them; the canonical `mtld3d_types`
// constants are `u32`, so each scrutinee is widened via `u32::from(...)` at the
// match — Rust match patterns require the pattern and scrutinee to share a type.

bitflags::bitflags! {
    /// Boolean predicates for `FfVsKey`.
    ///
    /// Packs ten 1-bit fields into a `u16` so the cache key stays compact and
    /// `Hash` walks one word instead of ten. Bit layout is stable —
    /// `SHADER_CACHE_SCHEMA_VERSION` must bump on any reorder or addition.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct FfVsFlags: u16 {
        /// Vertex declaration has a NORMAL element.
        const HAS_NORMAL = 1 << 0;
        /// Vertex declaration has a COLOR0 element (diffuse).
        const HAS_COLOR0 = 1 << 1;
        /// Vertex declaration has a COLOR1 element (specular).
        const HAS_COLOR1 = 1 << 2;
        /// `D3DRS_LIGHTING` enabled.
        ///
        /// XYZRHW bypasses per-vertex lighting regardless of the RS, so this
        /// is always false when `HAS_RHW`.
        const LIGHTING_ENABLED = 1 << 3;
        /// `D3DFVF_XYZRHW`: position is pre-transformed (screen-space with 1/w).
        ///
        /// `emit_vs` takes a different code path that skips WVP + lighting and
        /// maps `(screen_x, screen_y)` through the viewport dimensions into
        /// clip space.
        const HAS_RHW = 1 << 4;
        /// `D3DRS_COLORVERTEX` — gates the material-source override.
        ///
        /// When clear, the resolver ignores `*_source` and always reads
        /// from the material constant.
        const COLOR_VERTEX = 1 << 5;
        /// `D3DRS_SPECULARENABLE`.
        ///
        /// Gates per-light Blinn-Phong specular emission into `color1`; when
        /// clear, `color1` receives `saturate(float4(0.0))`.
        const SPECULAR_ENABLE = 1 << 6;
        /// `D3DRS_INDEXEDVERTEXBLENDENABLE`: the world-matrix index source.
        ///
        /// When set, per-vertex BLENDINDICES select world matrices from
        /// `world_palette[idx[i]]`; when clear, sequential matrices
        /// `world_palette[0..count]` are used. Indexed mode also requires
        /// `DECLARED_INDICES` to be set.
        const VERTEX_BLEND_INDEXED = 1 << 7;
        /// Vertex declaration has a BLENDINDICES element.
        ///
        /// Mirrored into the key so adding the indices attribute triggers a
        /// fresh shader variant.
        const DECLARED_INDICES = 1 << 8;
        /// `D3DRS_LOCALVIEWER` — selects the specular view-vector model.
        ///
        /// Per-vertex `normalize(-posEye)` when set (the D3D9 default), the
        /// constant infinite-viewer `(0, 0, -1)` when clear. Canonicalized at
        /// key build: only set when lighting + specular are both on, so draws
        /// that never read V don't fork variants.
        const LOCAL_VIEWER = 1 << 9;
        /// A COLOR0 (diffuse) element is declared on a non-zero, *unbound* stream.
        ///
        /// The descriptor drops it. The vertex reads 0 from the unbound
        /// stream, so the unlit FF VS outputs black for diffuse — NOT the
        /// white material default it uses when diffuse is entirely absent.
        const DIFFUSE_DECLARED_UNBOUND = 1 << 10;
        /// The vertex format came from `SetVertexDeclaration`, not `SetFVF`.
        ///
        /// A COLORVERTEX material source pointing at a vertex colour the
        /// declaration omits reads 0, whereas FVF falls back to the material
        /// colour.
        const USES_VERTEX_DECL = 1 << 11;
        /// `D3DRS_NORMALIZENORMALS` is enabled.
        ///
        /// The FF VS then renormalizes the eye-space normal after the
        /// inverse-transpose transform; when clear (the D3D9 default) the
        /// transformed normal keeps its magnitude, so a non-unit model normal
        /// scales the lighting.
        const NORMALIZE_NORMALS = 1 << 12;
    }
}

/// Summary of the current D3D9 Fixed-Function vertex state.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FfVsKey {
    /// Boolean predicates. See `FfVsFlags` for individual bit semantics.
    pub flags: FfVsFlags,
    /// Number of TEXCOORD input attributes the VS declares.
    ///
    /// I.e. the number of `float4 v{4+i} [[attribute(4+i)]]` entries in
    /// `VertexIn`. Equal to the vertex-stream texcoord count derived from the
    /// FVF / `VertexDeclaration` (`FfVsLayout::tex_coord_count`). Must agree
    /// with the `MTLVertexDescriptor` built by `convert::resolve_attrs_for_ff`.
    pub input_tex_coord_count: u8,
    /// Number of per-stage texcoord *varyings* the VS emits for the PS to sample.
    ///
    /// Inflated above `input_tex_coord_count` when PS texture stages are
    /// active without matching vertex-stream texcoords — the VS still needs to
    /// emit an output for each active stage so the PS can read it, but the
    /// passthru reads fall back to `float4(0,0,0,1)` for coord-set indices
    /// outside `input_tex_coord_count`.
    pub tex_coord_count: u8,
    /// Bit `i` set iff slot `i`'s `D3DLIGHT9` contributes to FF VS shading.
    ///
    /// I.e. it has non-zero `Type` AND is enabled via `LightEnable(i, TRUE)`.
    /// The emitter needs only per-slot activity plus the type masks below, so
    /// the light type is carried as bitmasks rather than a per-slot array.
    pub light_active_mask: u8,
    /// Bit `i` set iff active light `i` is DIRECTIONAL.
    ///
    /// Only meaningful when the corresponding `light_active_mask` bit is set.
    pub light_directional_mask: u8,
    /// Bit `i` set iff active light `i` is SPOT.
    ///
    /// Only meaningful when the corresponding `light_active_mask` bit is set;
    /// a slot with neither type bit takes the POINT branch.
    pub light_spot_mask: u8,
    /// `D3DRS_DIFFUSEMATERIALSOURCE` (0 = `MCS_MATERIAL`, 1 = `MCS_COLOR1`, 2 = `MCS_COLOR2`).
    ///
    /// Routed through `resolve_mat` at the diffuse modulation site.
    pub diffuse_source: u8,
    /// `D3DRS_AMBIENTMATERIALSOURCE`.
    ///
    /// Routed through `resolve_mat` at the ambient accumulation site.
    pub ambient_source: u8,
    /// `D3DRS_SPECULARMATERIALSOURCE`.
    ///
    /// Routed through `resolve_mat` at the specular modulation site in the
    /// light loop.
    pub specular_source: u8,
    /// `D3DRS_EMISSIVEMATERIALSOURCE`.
    ///
    /// Routed through `resolve_mat` at the initial `diffuseAccum` emissive
    /// term.
    pub emissive_source: u8,
    /// Vertex fog mode, resolved from the fog render states and `HAS_RHW`.
    ///
    /// 0 = off, 1 = EXP, 2 = EXP2, 3 = LINEAR, 4 = factor taken straight from
    /// the specular (COLOR1) alpha, which is what D3D9 does when fog is enabled
    /// and both `D3DRS_FOGVERTEXMODE` and `D3DRS_FOGTABLEMODE` are
    /// `D3DFOG_NONE`. A non-`NONE` table mode fogs per-pixel and holds this at
    /// 0. XYZRHW vertices bypass the vertex-fog computation, so a `HAS_RHW`
    /// draw carries either 0 (fog off, or table fog) or 4.
    pub fog_mode: u8,
    /// Per-stage TCI (texture coordinate index) mode.
    ///
    /// Decoded from the high byte of `D3DTSS_TEXCOORDINDEX[i]`. 0 = PASSTHRU,
    /// 1 = CAMERASPACENORMAL, 2 = CAMERASPACEPOSITION,
    /// 3 = CAMERASPACEREFLECTIONVECTOR. Mode 4 (SPHEREMAP) and unknown modes
    /// fall back to passthru with a one-shot warn.
    pub tci_modes: [u8; 8],
    /// Per-stage input coord-set index for passthru mode.
    ///
    /// Decoded from the low byte of `D3DTSS_TEXCOORDINDEX[i]` (0..7).
    pub tci_coord_indices: [u8; 8],
    /// Declared component count (1..=4) of each *input* TEXCOORD set.
    ///
    /// Indexed by coord-set (= `usage_index`); `0` if the set is absent.
    /// Mirrors `FfVsLayout::tex_coord_dims`. The texcoord-transform emission
    /// uses this to expand a `FLOATn` coordinate per the D3D9 fixed-function
    /// rule before the per-stage texture matrix multiply.
    pub tex_coord_dims: [u8; 8],
    /// Per-stage texture-transform flags packed: low 3 bits = count, bit 4 = `D3DTTFF_PROJECTED`.
    pub tt_flags: [u8; 8],
    /// Number of world matrices blended per vertex. `0` disables blending.
    ///
    /// Derived from `D3DRS_VERTEXBLEND`.
    pub vertex_blend_count: u8,
    /// Number of weight lanes the vertex decl declares.
    pub declared_weights_count: u8,
}

impl FfVsKey {
    #[inline]
    #[must_use]
    pub const fn has_normal(&self) -> bool {
        self.flags.contains(FfVsFlags::HAS_NORMAL)
    }
    #[inline]
    #[must_use]
    pub const fn has_color0(&self) -> bool {
        self.flags.contains(FfVsFlags::HAS_COLOR0)
    }
    #[inline]
    #[must_use]
    pub const fn has_color1(&self) -> bool {
        self.flags.contains(FfVsFlags::HAS_COLOR1)
    }
    #[inline]
    #[must_use]
    pub const fn has_diffuse_declared_unbound(&self) -> bool {
        self.flags.contains(FfVsFlags::DIFFUSE_DECLARED_UNBOUND)
    }
    #[inline]
    #[must_use]
    pub const fn uses_vertex_decl(&self) -> bool {
        self.flags.contains(FfVsFlags::USES_VERTEX_DECL)
    }
    #[inline]
    #[must_use]
    pub const fn lighting_enabled(&self) -> bool {
        self.flags.contains(FfVsFlags::LIGHTING_ENABLED)
    }
    #[inline]
    #[must_use]
    pub const fn has_rhw(&self) -> bool {
        self.flags.contains(FfVsFlags::HAS_RHW)
    }
    #[inline]
    #[must_use]
    pub const fn color_vertex(&self) -> bool {
        self.flags.contains(FfVsFlags::COLOR_VERTEX)
    }
    #[inline]
    #[must_use]
    pub const fn specular_enable(&self) -> bool {
        self.flags.contains(FfVsFlags::SPECULAR_ENABLE)
    }
    #[inline]
    #[must_use]
    pub const fn local_viewer(&self) -> bool {
        self.flags.contains(FfVsFlags::LOCAL_VIEWER)
    }
    #[inline]
    #[must_use]
    pub const fn normalize_normals(&self) -> bool {
        self.flags.contains(FfVsFlags::NORMALIZE_NORMALS)
    }
    #[inline]
    #[must_use]
    pub const fn vertex_blend_indexed(&self) -> bool {
        self.flags.contains(FfVsFlags::VERTEX_BLEND_INDEXED)
    }
    #[inline]
    #[must_use]
    pub const fn declared_indices(&self) -> bool {
        self.flags.contains(FfVsFlags::DECLARED_INDICES)
    }
}

/// Texture-transform flag accessor helpers.
///
/// Kept as free functions rather than methods to avoid pulling `impl FfVsKey`
/// into the public surface.
pub const fn tt_count(flags: u8) -> u8 {
    flags & 0x07
}

pub const fn tt_projected(flags: u8) -> bool {
    (flags & 0x10) != 0
}

/// Per-stage texture combiner state derived from `texture_stage_states[stage]`.
///
/// Note: D3D9's `D3DTSS_TEXCOORDINDEX` controls *both* the VS (TCI mode +
/// input coord-set selection) and the PS (which varying to sample from).
/// Both concerns are handled on the VS side
/// (`FfVsKey::tci_modes` + `FfVsKey::tci_coord_indices`, one entry per
/// stage) so the VS emits the correct coord for each stage into
/// `Varyings.texcoord[stage]`. The PS then samples stage `N` using
/// `Varyings.texcoord[N]` — no per-stage indirection needed here.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct FfStage {
    pub color_op: u8,
    pub color_arg1: u8,
    pub color_arg2: u8,
    pub alpha_op: u8,
    pub alpha_arg1: u8,
    pub alpha_arg2: u8,
    pub has_texture: bool,
}

/// Summary of the current D3D9 Fixed-Function pixel state.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FfPsKey {
    pub stages: [FfStage; 8],
    /// `D3DRS_SPECULARENABLE`.
    ///
    /// When set, the interpolated specular color (`color1`) is added to the
    /// cascade result after the last texture stage and before fog — the D3D9
    /// fixed-function "specular add" stage. FF-only: a bound programmable PS
    /// never applies it.
    pub specular_add: bool,
    /// Bit `i` set iff stage `i` has `D3DTTFF_PROJECTED`.
    ///
    /// The FF sampler then divides the texture coordinate by its `.w`
    /// component before sampling (the projective divide), matching the FF /
    /// `ps_1_x` `tex` semantics. The VS leaves the divisor in `.w`
    /// (`tex_coord_dims` transform), so this is a per-pixel divide. `.w == 0`
    /// samples at the origin (D3D9 returns the `(0,0)` texel there).
    pub tt_projected_mask: u8,
}

/// Fixed-function attribute convention used by `emit_vertex_in`.
///
/// Returns the `[[attribute(N)]]` index that a vertex element with the given
/// `(usage, usage_index)` lands on in the FF VS. `None` means the FF VS does
/// not consume this semantic — callers should skip the element.
///
/// This is the single source of truth for the FF input layout: the vertex
/// descriptor built by the pipeline and the `struct VertexIn` emitted here
/// must agree. Whenever `emit_vertex_in` changes, this function changes
/// with it.
///
/// `usage` / `usage_index` use the D3DDECLUSAGE_* byte values (0=POSITION,
/// 3=NORMAL, 5=TEXCOORD, 9=POSITIONT, 10=COLOR, …).
#[must_use]
pub const fn ff_attr_index_for_semantic(usage: u8, usage_index: u8) -> Option<u16> {
    match (usage, usage_index) {
        (D3DDECLUSAGE_POSITION | D3DDECLUSAGE_POSITIONT, 0) => Some(0),
        (D3DDECLUSAGE_NORMAL, 0) => Some(1),
        (D3DDECLUSAGE_COLOR, 0) => Some(2),
        (D3DDECLUSAGE_COLOR, 1) => Some(3),
        (D3DDECLUSAGE_TEXCOORD, i) if i < 8 => Some(4 + i as u16),
        (D3DDECLUSAGE_BLENDWEIGHT, 0) => Some(12),
        (D3DDECLUSAGE_BLENDINDICES, 0) => Some(13),
        _ => None,
    }
}

#[must_use]
pub fn emit_vs_ff(vs_key: &FfVsKey) -> String {
    emit_vs_ff_named(vs_key, super::emit::DEFAULT_VS_ENTRY)
}

#[must_use]
pub fn emit_ps_ff(ps_key: &FfPsKey, variant: VariantKey) -> String {
    emit_ps_ff_named(ps_key, variant, super::emit::DEFAULT_PS_ENTRY)
}

#[must_use]
pub fn emit_vs_ff_named(vs_key: &FfVsKey, entry: &str) -> String {
    let mut out = String::new();
    out.push_str("#include <metal_stdlib>\n");
    out.push_str("using namespace metal;\n\n");
    emit_vertex_in(&mut out, vs_key);
    emit_varyings(&mut out, false);
    emit_vs(&mut out, vs_key, entry);
    out
}

#[must_use]
pub fn emit_ps_ff_named(ps_key: &FfPsKey, variant: VariantKey, entry: &str) -> String {
    let mut out = String::new();
    out.push_str("#include <metal_stdlib>\n");
    out.push_str("using namespace metal;\n\n");
    emit_varyings(&mut out, variant.flags.contains(VariantFlags::FLAT_SHADE));
    if variant.flags.contains(VariantFlags::SRGB_WRITE) {
        super::emit::emit_srgb_write_helper(&mut out);
    }
    emit_ps(&mut out, ps_key, variant, entry);
    out
}

// ── Vertex input ──
// Declare each FVF-derived attribute as float4 at the matching index; Metal
// fills missing lanes per spec (xyz → w=1; xy → z=0,w=1).

fn emit_vertex_in(out: &mut String, vs: &FfVsKey) {
    out.push_str("struct VertexIn {\n");
    // Position: XYZ path is float3-padded (Metal zero-fills w), XYZRHW path
    // needs all four lanes (screen_x, screen_y, z, rhw).
    out.push_str("    float4 v0 [[attribute(0)]];\n");
    if vs.has_normal() {
        out.push_str("    float4 v1 [[attribute(1)]];\n");
    }
    if vs.has_color0() {
        out.push_str("    float4 v2 [[attribute(2)]];\n");
    }
    if vs.has_color1() {
        out.push_str("    float4 v3 [[attribute(3)]];\n");
    }
    for i in 0..vs.input_tex_coord_count {
        let _ = writeln!(out, "    float4 v{idx} [[attribute({idx})]];", idx = 4 + i);
    }
    // Vertex blending inputs. Only declared when the resolved blend mode
    // will use them — keeps the MTLVertexDescriptor and VertexIn in lock-
    // step with what the game actually wired up.
    if vs.vertex_blend_count > 0 && vs.declared_weights_count > 0 {
        out.push_str("    float4 blend_weight [[attribute(12)]];\n");
    }
    if vs.vertex_blend_count > 0 && vs.declared_indices() {
        out.push_str("    uint4 blend_indices [[attribute(13)]];\n");
    }
    out.push_str("};\n\n");
}

/// Declared dimension (1..=4) of the input texcoord set `src`.
///
/// 0 when the set is not present in the vertex stream. Guards against reading
/// a `VertexIn` attribute the vertex declaration never provided.
fn input_dim(vs: &FfVsKey, src: u32) -> u8 {
    if src < u32::from(vs.input_tex_coord_count) {
        vs.tex_coord_dims[src as usize]
    } else {
        0
    }
}

/// MSL `float4` expression for the raw vertex texcoord at coord-set `src`.
///
/// Components beyond the declared dimension are forced to 0.0. D3D9 fills
/// unbacked texcoord components with 0 (Metal would fill `.w` with 1.0), so
/// the masking is explicit. Returns the zero coordinate when the set is absent
/// (warned once per site).
fn masked_input_rhs(vs: &FfVsKey, stage: usize, src: u32) -> String {
    let attr = 4 + src;
    match input_dim(vs, src) {
        1 => format!("float4(in.v{attr}.x, 0.0, 0.0, 0.0)"),
        2 => format!("float4(in.v{attr}.xy, 0.0, 0.0)"),
        3 => format!("float4(in.v{attr}.xyz, 0.0)"),
        4 => format!("in.v{attr}"),
        _ => {
            mtld3d_shared::log_once_warn!(target: super::LOG_TARGET,
                "dxso FF: stage {stage} passthru coord-set {src} not in vertex stream (input_tex_coord_count={}) → float4(0)",
                vs.input_tex_coord_count
            );
            "float4(0.0)".to_string()
        }
    }
}

fn emit_varyings(out: &mut String, flat: bool) {
    out.push_str("struct Varyings {\n");
    // Must match `dxso::emit::emit_varyings` byte-for-byte — see the
    // invariance comment there. Analog of an `Invariant` decoration on
    // a SPIR-V `gl_Position` output.
    out.push_str("    float4 position [[position, invariant]];\n");
    // Secondary POSITION user varying — must match `dxso::emit::emit_varyings`
    // at the same declaration index (bare positional varying) for FF↔
    // programmable stage-in linkage. The FF VS never writes it.
    out.push_str("    float4 position1;\n");
    for i in 0..8 {
        let _ = writeln!(out, "    float4 texcoord{i};");
    }
    // `[[flat]]` (D3DSHADE_FLAT) on the PS struct only — matches
    // `dxso::emit::emit_varyings`; the qualifier is ignored on VS output, and
    // `flat` is false for the FF VS, so field-index stage-in linkage holds.
    let q = if flat { " [[flat]]" } else { "" };
    for i in 0..2 {
        let _ = writeln!(out, "    float4 color{i}{q};");
    }
    // Fog factor carried in .x (1.0 = unfogged pixel, 0.0 = fog color).
    // Always declared so VS and PS agree on the layout regardless of fog key.
    out.push_str("    float4 fog;\n");
    // NDC depth (`clip.z / clip.w`) for the table-fog Z source, interpolated
    // like the rasterizer's own depth. Deliberately NOT `in.position.z`:
    // Metal folds the encoder `setDepthBias` into the fragment `[[position]]`
    // depth (scaled to float-buffer ulps), while D3D9 pixel fog wants the
    // RAW `D3DRS_DEPTHBIAS` added to the unbiased depth.
    // Matches `dxso::emit::emit_varyings` byte-for-byte.
    out.push_str("    float fog_z [[center_no_perspective]];\n");
    // Match `dxso::emit::emit_varyings` — a programmable VS that writes
    // oPts / dcl_psize must link to an FF PS, and vice versa, so the
    // layout has to stay identical.
    out.push_str("    float point_size [[point_size]];\n");
    out.push_str("};\n\n");
}

/// Emit the vertex-blending block that computes `pos_view` and (when the VS reads normals) `n`.
///
/// Sourced from the per-bone pre-multiplied
/// `transpose(world_palette[i] × view)` matrices packed at
/// `vs_c[95 + bone*4 .. 95 + bone*4 + 4]` by `ff_state::build_vs_constants`.
///
/// Position formula (K = `vertex_blend_count`):
///
/// ```text
///   pos_view = Σ_{i=0..K-1} w[i] · M[idx[i]] · pos + (1 − Σ w[i]) · M[idx[K-1]] · pos
/// ```
///
/// Normal uses the top-3x3 of the same per-bone matrix (no inverse-
/// transpose — matches D3D9 spec).
///
/// Index source:
/// - Indexed mode (`vertex_blend_indexed = true`): `idx[i] = in.blend_indices[i]`.
/// - Sequential mode: `idx[i] = i` (matrices come from `world_palette[0..K]`).
///
/// Special case K=1 with `vertex_blend_indexed = true` (`D3DVBF_0WEIGHTS`):
/// no explicit weights, single matrix at `in.blend_indices[0]` with weight 1.
fn emit_vertex_blend(out: &mut String, vs: &FfVsKey) {
    debug_assert!(vs.vertex_blend_count > 0);
    let k = vs.vertex_blend_count as usize;
    let indexed = vs.vertex_blend_indexed();
    let has_normal = vs.has_normal();
    out.push_str("    float4 pos_view = float4(0.0);\n");
    if has_normal {
        out.push_str("    float3 n_blend = float3(0.0);\n");
    }

    // D3DVBF_0WEIGHTS indexed-only path: K=1, weight = 1.0, single matrix.
    if k == 1 && indexed {
        out.push_str("    {\n");
        out.push_str("        uint idx = in.blend_indices[0];\n");
        out.push_str("        constant float4 *m = vs_c + 95 + idx * 4u;\n");
        out.push_str(
            "        pos_view = float4(dot(pos, m[0]), dot(pos, m[1]), dot(pos, m[2]), dot(pos, m[3]));\n",
        );
        if has_normal {
            out.push_str("        n_blend = float3(dot(in.v1.xyz, m[0].xyz), dot(in.v1.xyz, m[1].xyz), dot(in.v1.xyz, m[2].xyz));\n");
        }
        out.push_str("    }\n");
        return;
    }

    // Normal mode (D3DVBF_kWEIGHTS): K = (k-1) explicit weights + 1 implicit.
    // K==1 here means no explicit weights — implicit single-bone (rare via
    // sequential mode; weight collapses to 1.0).
    let explicit = k.saturating_sub(1);
    out.push_str("    float weight_sum = 0.0;\n");
    for i in 0..explicit {
        out.push_str("    {\n");
        let _ = writeln!(out, "        float w = in.blend_weight[{i}];");
        if indexed {
            let _ = writeln!(out, "        uint idx = in.blend_indices[{i}];");
        } else {
            let _ = writeln!(out, "        uint idx = {i}u;");
        }
        out.push_str("        constant float4 *m = vs_c + 95 + idx * 4u;\n");
        out.push_str(
            "        pos_view += w * float4(dot(pos, m[0]), dot(pos, m[1]), dot(pos, m[2]), dot(pos, m[3]));\n",
        );
        if has_normal {
            out.push_str("        n_blend += w * float3(dot(in.v1.xyz, m[0].xyz), dot(in.v1.xyz, m[1].xyz), dot(in.v1.xyz, m[2].xyz));\n");
        }
        out.push_str("        weight_sum += w;\n");
        out.push_str("    }\n");
    }
    // Implicit last-weight contribution.
    out.push_str("    {\n");
    out.push_str("        float w = 1.0 - weight_sum;\n");
    let last = explicit;
    if indexed {
        let _ = writeln!(out, "        uint idx = in.blend_indices[{last}];");
    } else {
        let _ = writeln!(out, "        uint idx = {last}u;");
    }
    // World-matrix palette base is row 95 (same as the explicit-weight loop and
    // the K=1 indexed path, and the encoder's upload base in `ff_state`); the
    // implicit last-weight contribution reads from the same row-95 base.
    out.push_str("        constant float4 *m = vs_c + 95 + idx * 4u;\n");
    out.push_str(
        "        pos_view += w * float4(dot(pos, m[0]), dot(pos, m[1]), dot(pos, m[2]), dot(pos, m[3]));\n",
    );
    if has_normal {
        out.push_str("        n_blend += w * float3(dot(in.v1.xyz, m[0].xyz), dot(in.v1.xyz, m[1].xyz), dot(in.v1.xyz, m[2].xyz));\n");
    }
    out.push_str("    }\n");
}

// ── VS ──
// Constant buffer layout (float4 rows at `vs_c[i]`):
//   [0..3]    transpose(WorldView) full 4 rows — clip-position step 1
//                                              (full row dot), plus
//                                              eye-space normal (.xyz lanes
//                                              of rows 0..2), eye-space
//                                              position (full 4-component
//                                              dot of rows 0..2), and
//                                              eye-space Z (row 2) for fog.
//   [4..7]    transpose(Projection) — clip-position step 2 (Proj·(WV·pos)).
//                                     Two-step decomposition is load-bearing:
//                                     pre-multiplying WVP CPU-side produces
//                                     a different FP-rounding shape than
//                                     any programmable shader, causing
//                                     FF↔programmable z-fight on AMD-Mac.
//   [8]       fog params: (fogStart, fogEnd, fogDensity, 0). Kept low in the
//             layout so enabling fog on an otherwise-light draw does not force
//             uploading rows 9..53. Read only when `fog_mode != 0`;
//             zero-filled otherwise.
//   [9]       global ambient RGBA
//   [10]      material.diffuse RGBA
//   [11]      material.ambient RGBA
//   [12]      material.specular RGBA
//   [13]      material.emissive RGBA
//   [14]      material.power in .x (.yzw unused)
//   [15..62]  per-light × 8 slots, 6 float4s each (light i at base 15+i*6):
//     +0  position.xyz, light_type (.w; 0 = disabled)
//     +1  direction.xyz (eye space, normalized), spot falloff (.w)
//     +2  diffuse RGBA
//     +3  ambient.rgb, spot_offset (.w)
//     +4  attenuation0/1/2 (.xyz), range (.w)
//     +5  specular.rgb, spot_scale (.w)
//   [63..94]  per-stage texture-transform matrices (8 stages × 4 rows each)
//             stored as `transpose(D3DTS_TEXTUREn)` so the VS applies them
//             via `dot(tc, vs_c[63+stage*4 + i])`. Reserved even when
//             `tt_flags[stage] == 0`; the VS only reads slots it needs.
//
// XYZRHW path — the transform constants are unused; the *same* slot 0 holds
// `(viewport_width, viewport_height, 0, 0)` instead. The VS skips lighting
// because D3D9 disables per-vertex lighting for XYZRHW regardless of the
// D3DRS_LIGHTING state, and bypasses vertex fog for the same reason.
//
// The encoder-side builder in `windows/d3d9/src/ff_state.rs` packs this layout;
// changes here must stay in sync.
fn emit_vs(out: &mut String, vs: &FfVsKey, entry: &str) {
    let _ = writeln!(out, "vertex Varyings {entry}(");
    out.push_str("    VertexIn in [[stage_in]],\n");
    out.push_str("    constant float4 *vs_c [[buffer(15)]],\n");
    // Half-pixel rasterization fixup uniform (VS buffer 13): `(1/vp_w,
    // -1/vp_h, 0, 0)`, supplied per-draw by the encoder from the live
    // viewport. Read by the transformed-position epilogue below (the XYZRHW
    // branch folds the same offset into `ndc_x`/`ndc_y` directly). Vertex
    // buffer 13 is free — the FF VS uses only slots 0 (stream) and 15 (vs_c).
    out.push_str("    constant float4 &pos_fixup [[buffer(13)]]\n");
    out.push_str(") {\n");
    out.push_str("    Varyings out;\n");
    // FF VS doesn't currently expose PSIZE in the FVF decoder (caps
    // skip D3DFVFCAPS_PSIZE), so just default to 1.0 — matches D3D9
    // spec for point primitives without an explicit size.
    out.push_str("    out.point_size = 1.0;\n");

    if vs.has_rhw() {
        // Pre-transformed screen-space vertices: `(in.v0.x, in.v0.y)` are
        // pixel coordinates in render-target space. Metal's `setViewport`
        // maps NDC back to the viewport's pixel rect, so the VS must
        // subtract the viewport origin before normalizing — otherwise an
        // XYZRHW vertex at RT pixel (100, 100) with viewport origin
        // (0, 200) ends up at RT pixel (100, −100+200) instead of
        // (100, 100). `vs_c[0] = (vp_w, vp_h, vp_x, vp_y)` from
        // `ff_state::build_vs_constants`. Y is flipped (D3D9 top-down →
        // Metal bottom-up NDC). `in.v0.w` is the reciprocal of the
        // homogeneous w from the original transform; clip space wants us
        // to undo it so perspective-correct interpolation behaves
        // (multiply all lanes by w so `pos / pos.w` lands back in NDC).
        //
        // Half-pixel rasterization fixup: D3D9's window→NDC mapping is shifted
        // half a pixel from Metal's, so on-boundary geometry lands one pixel
        // up-left of the D3D9 reference. Add
        // half a pixel right (`+ 1/vp_w` in NDC) and down (`- 1/vp_h` in NDC,
        // since Metal NDC is +y-up so framebuffer-down is −y). Half a pixel is
        // `1/vp` in NDC because NDC spans 2.0 across `vp` pixels. Applied to
        // `ndc_x`/`ndc_y` before the `* w` so the offset survives the divide.
        out.push_str("    float2 vp = vs_c[0].xy;\n");
        out.push_str("    float2 vp_origin = vs_c[0].zw;\n");
        out.push_str("    float rhw = in.v0.w;\n");
        out.push_str("    float w = 1.0 / max(rhw, 1e-30);\n");
        out.push_str(
            "    float ndc_x = ((in.v0.x - vp_origin.x) / vp.x) * 2.0 - 1.0 + 1.0 / vp.x;\n",
        );
        out.push_str(
            "    float ndc_y = 1.0 - ((in.v0.y - vp_origin.y) / vp.y) * 2.0 - 1.0 / vp.y;\n",
        );
        out.push_str("    out.position = float4(ndc_x * w, ndc_y * w, in.v0.z * w, w);\n");

        if vs.has_color0() {
            out.push_str("    out.color0 = in.v2;\n");
        } else if vs.has_diffuse_declared_unbound() {
            out.push_str("    out.color0 = float4(0.0);\n");
        } else {
            out.push_str("    out.color0 = float4(1.0);\n");
        }
        // Pre-transformed geometry carries its specular (oD1) in the
        // COLOR1 attribute; pass it through for the PS specular add /
        // D3DTA_SPECULAR consumers.
        if vs.has_color1() {
            out.push_str("    out.color1 = in.v3;\n");
        } else {
            out.push_str("    out.color1 = float4(0.0);\n");
        }

        // XYZRHW (pre-transformed) geometry bypasses the texture-coordinate
        // transform entirely: D3D9 passes the declared texcoord components
        // straight through with the standard `.w = 1.0` fill, ignoring
        // D3DTSS_TEXTURETRANSFORMFLAGS and D3DTSS_TEXCOORDINDEX texgen modes
        // (eye-space `n` / `posEye` are undefined for pre-transformed verts).
        // We still honour the coord-set selector so a stage reading a
        // non-default set picks the right attribute.
        for i in 0..vs.tex_coord_count as usize {
            let mode = vs.tci_modes[i];
            if matches!(mode, 1..=3) {
                mtld3d_shared::log_once_warn!(target: super::LOG_TARGET,
                    "dxso FF: TCI mode {mode} on XYZRHW stage {i} — eye-space undefined, falling back to passthru"
                );
            }
            let src = u32::from(vs.tci_coord_indices[i].min(7));
            let n = input_dim(vs, src);
            let raw = masked_input_rhs(vs, i, src);
            if (1..4).contains(&n) {
                // FLOATn (n<4): pre-transformed texcoords fill `.w` with 1.0.
                let _ = writeln!(out, "    out.texcoord{i} = float4(({raw}).xyz, 1.0);");
            } else {
                let _ = writeln!(out, "    out.texcoord{i} = {raw};");
            }
        }
        for i in vs.tex_coord_count as usize..8 {
            let _ = writeln!(out, "    out.texcoord{i} = float4(0.0);");
        }

        // XYZRHW bypasses vertex fog COMPUTATION, but D3D9 still fogs
        // pre-transformed geometry from the specular (COLOR1) alpha when table
        // mode is NONE (fog_mode 4). The PS reads out.fog.x as the fog factor.
        if vs.fog_mode == 4 && vs.has_color1() {
            out.push_str("    out.fog = float4(in.v3.w, 0.0, 0.0, 0.0);\n");
        } else {
            out.push_str("    out.fog = float4(1.0);\n");
        }
        // NDC depth for the table-fog Z source (see the Varyings decl). The
        // clip-space round trip (`z*w / w`) keeps the FP rounding shape the
        // rasterizer's own depth uses.
        out.push_str("    out.fog_z = out.position.z / out.position.w;\n");

        out.push_str("    return out;\n");
        out.push_str("}\n");
        return;
    }

    // Clip-space position via two-step `Proj · (WV · pos)`. The
    // two-step decomposition is load-bearing for FF↔programmable z
    // parity at the math level: pre-multiplying WVP CPU-side and
    // emitting one matmul produces a different FP-rounding shape than
    // any programmable shader. Plain MSL `dot()` — Apple Silicon has
    // hardware dot-product. Cross-shader bit-invariance is not the
    // goal (the implicit decal depth bias in
    // `windows/d3d9/src/draw.rs` handles z-fight from genuinely
    // different per-pipeline transforms); `[[position, invariant]]` +
    // `setPreserveInvariance(true)` + `setMathMode(Safe)` on the VS
    // compile keep clip-position bit-stable WITHIN a shader for
    // reflection-style same-shader-twice scenarios.
    out.push_str("    float4 pos = float4(in.v0.xyz, 1.0);\n");
    if vs.vertex_blend_count > 0 {
        emit_vertex_blend(out, vs);
    } else {
        out.push_str(
            "    float4 pos_view = float4(dot(pos, vs_c[0]), dot(pos, vs_c[1]), dot(pos, vs_c[2]), dot(pos, vs_c[3]));\n",
        );
    }
    out.push_str(
        "    out.position = float4(dot(pos_view, vs_c[4]), dot(pos_view, vs_c[5]), dot(pos_view, vs_c[6]), dot(pos_view, vs_c[7]));\n",
    );
    // Half-pixel rasterization fixup: shift the clip-space position half a
    // pixel right (+x) and down (−y in Metal's +y-up NDC) so on-boundary
    // geometry matches the D3D9 reference.
    // `pos_fixup.xy = (1/vp_w, -1/vp_h)`; scale by `.w` so the offset holds
    // after the perspective divide.
    out.push_str("    out.position.x += pos_fixup.x * out.position.w;\n");
    out.push_str("    out.position.y += pos_fixup.y * out.position.w;\n");

    // TCI pre-scan: if any active stage needs eye-space normal / position
    // but the lighting branch below won't declare them, emit them here.
    let active = vs.tex_coord_count as usize;
    let need_eye_normal = vs.tci_modes[..active].iter().any(|&m| m == 1 || m == 3);
    let need_eye_pos = vs.tci_modes[..active].iter().any(|&m| m == 2 || m == 3);
    let will_emit_in_lighting = vs.lighting_enabled() && vs.has_normal();
    let blended = vs.vertex_blend_count > 0;
    if !will_emit_in_lighting {
        if need_eye_normal && vs.has_normal() {
            if blended {
                out.push_str("    float3 n = normalize(n_blend);\n");
            } else {
                out.push_str("    float3 n = normalize(float3(dot(in.v1.xyz, vs_c[0].xyz), dot(in.v1.xyz, vs_c[1].xyz), dot(in.v1.xyz, vs_c[2].xyz)));\n");
            }
        }
        if need_eye_pos {
            if blended {
                out.push_str("    float3 posEye = pos_view.xyz;\n");
            } else {
                out.push_str("    float3 posEye = float3(dot(pos, vs_c[0]), dot(pos, vs_c[1]), dot(pos, vs_c[2]));\n");
            }
        }
    }

    // Diffuse / specular color.
    if vs.lighting_enabled() {
        // Ambient and emissive contributions are normal-independent, so FF
        // lighting still runs without a vertex normal — D3D9 zeroes only the
        // per-light N·L diffuse/specular terms. `has_n` gates the normal-only
        // math; everything else (material resolution, emissive, global and
        // per-light ambient) applies regardless.
        let has_n = vs.has_normal();
        let mut mat_flags = MatColorFlags::empty();
        mat_flags.set(MatColorFlags::COLOR_VERTEX, vs.color_vertex());
        mat_flags.set(MatColorFlags::HAS_COLOR0, vs.has_color0());
        mat_flags.set(MatColorFlags::HAS_COLOR1, vs.has_color1());
        mat_flags.set(MatColorFlags::USES_DECL, vs.uses_vertex_decl());
        let mat_diffuse = resolve_mat(vs.diffuse_source, 10, mat_flags);
        let mat_ambient = resolve_mat(vs.ambient_source, 11, mat_flags);
        let mat_specular = resolve_mat(vs.specular_source, 12, mat_flags);
        let mat_emissive = resolve_mat(vs.emissive_source, 13, mat_flags);
        if blended {
            // Blended-WV path: `n_blend` and `pos_view` are accumulated in
            // `emit_vertex_blend` from the per-bone palette × view matrices.
            out.push_str("    float3 posEye = pos_view.xyz;\n");
            if has_n {
                out.push_str("    float3 n = normalize(n_blend);\n");
            }
        } else {
            // Single-WV path: top-3x3 of transposed WorldView at vs_c[0..2].
            // Eye-space position (needs only `pos`) for point-light vectors and
            // specular half-angle. `vs_c[0..2]` hold full rows of transpose(WV)
            // including translation.
            out.push_str("    float3 posEye = float3(dot(pos, vs_c[0]), dot(pos, vs_c[1]), dot(pos, vs_c[2]));\n");
            if has_n {
                // The eye normal is transformed by the D3D9 normal matrix so a
                // non-orthonormal world (scale/shear) scales the normal magnitude
                // correctly. `vs_c[0..2].xyz` are the WV columns (rows of
                // transpose(WV)); the cofactor cross-products of THOSE vectors over
                // the determinant form the normal matrix — for a pure rotation R it
                // reduces to R·n (the same direction the old normalize(WV·n) gave),
                // and for a uniform scale s to R·n / s. Feeding the transposed
                // components (vs_c[0].x, vs_c[1].x, …) instead would transpose the
                // matrix and apply the inverse rotation, making lighting swim as the
                // camera turns. Computed inline to avoid a separate constant slot.
                // D3D9 only renormalizes when D3DRS_NORMALIZENORMALS is set, so an
                // un-renormalized non-unit model normal otherwise scales the lighting.
                out.push_str("    float3 wvr0 = vs_c[0].xyz;\n");
                out.push_str("    float3 wvr1 = vs_c[1].xyz;\n");
                out.push_str("    float3 wvr2 = vs_c[2].xyz;\n");
                out.push_str("    float3 ncof0 = cross(wvr1, wvr2);\n");
                out.push_str("    float3 ncof1 = cross(wvr2, wvr0);\n");
                out.push_str("    float3 ncof2 = cross(wvr0, wvr1);\n");
                out.push_str("    float nwvdet = dot(wvr0, ncof0);\n");
                // normal matrix has rows cof_i / det ⇒ (N × n)_i = dot(cof_i, n) / det.
                out.push_str("    float3 n = (abs(nwvdet) > 1e-12)\n");
                out.push_str("        ? float3(dot(ncof0, in.v1.xyz), dot(ncof1, in.v1.xyz), dot(ncof2, in.v1.xyz)) / nwvdet\n");
                out.push_str("        : float3(dot(in.v1.xyz, vs_c[0].xyz), dot(in.v1.xyz, vs_c[1].xyz), dot(in.v1.xyz, vs_c[2].xyz));\n");
                if vs.normalize_normals() {
                    out.push_str("    n = normalize(n);\n");
                }
            }
        }
        if has_n && vs.specular_enable() {
            out.push_str("    float mat_power = vs_c[14].x;\n");
            if vs.local_viewer() {
                // Local viewer: per-vertex direction to the eye (the eye
                // sits at the origin of eye space).
                out.push_str("    float3 V = normalize(-posEye);\n");
            } else {
                // Infinite viewer: constant view vector along the view
                // axis (LH eye space looks down +z, so toward the eye
                // is -z).
                out.push_str("    float3 V = float3(0.0, 0.0, -1.0);\n");
            }
        }
        let _ = writeln!(
            out,
            "    float4 diffuseAccum = {mat_emissive} + vs_c[9] * {mat_ambient};"
        );
        out.push_str("    float3 specAccum = float3(0.0);\n");
        // Walk active light slots via the bitmask (1 bit per slot, MSB→LSB
        // order is irrelevant since each iteration emits an independent
        // accumulation block). The per-light DIRECTIONAL/SPOT/POINT branch
        // reads the two type masks; a slot with neither bit is POINT.
        let mut active = vs.light_active_mask;
        while active != 0 {
            let i = active.trailing_zeros() as usize;
            active &= active - 1;
            let base = 15 + i * 6;
            let is_directional = (vs.light_directional_mask & (1u8 << i)) != 0;
            let is_spot = (vs.light_spot_mask & (1u8 << i)) != 0;
            let _ = writeln!(out, "    {{");
            if is_directional {
                // DIRECTIONAL — constant direction, no attenuation.
                let _ = writeln!(out, "        float3 L = -vs_c[{b}].xyz;", b = base + 1);
                out.push_str("        float atten = 1.0;\n");
            } else {
                // POINT / SPOT — eye-space vector from vertex to light.
                let _ = writeln!(out, "        float3 toL = vs_c[{base}].xyz - posEye;");
                out.push_str("        float dist = length(toL);\n");
                out.push_str("        float3 L = toL / max(dist, 1e-30);\n");
                // Attenuation: 1 / (a0 + a1*d + a2*d²). Clamp to zero when
                // beyond range (.w of the attenuation slot) per D3D9 spec —
                // `step(dist, range)` is 1.0 when dist <= range, else 0.
                let _ = writeln!(out, "        float4 atten_k = vs_c[{b}];", b = base + 4);
                out.push_str(
                    "        float atten = 1.0 / max(atten_k.x + atten_k.y * dist + atten_k.z * dist * dist, 1e-30);\n",
                );
                out.push_str("        atten *= step(dist, atten_k.w);\n");
                if is_spot {
                    // Spot cone factor: rho is the cosine of the angle
                    // between the light→vertex ray and the spot axis;
                    // saturate(rho·scale + offset) is 1 inside theta, 0
                    // outside phi, the penumbra fraction between (scale /
                    // offset are precomputed at pack time on the specular /
                    // ambient rows' .w; falloff rides the direction row's
                    // .w).
                    let _ = writeln!(
                        out,
                        "        float rho = dot(-L, vs_c[{b1}].xyz);",
                        b1 = base + 1
                    );
                    let _ = writeln!(
                        out,
                        "        atten *= pow(saturate(rho * vs_c[{b5}].w + vs_c[{b3}].w), vs_c[{b1}].w);",
                        b5 = base + 5,
                        b3 = base + 3,
                        b1 = base + 1
                    );
                }
            }
            // The diffuse N·L term needs the surface normal; a normal-less
            // vertex contributes no diffuse/specular light (D3D9 behaviour). The
            // per-light ambient term below is normal-independent and always
            // applies, both modulated by attenuation.
            if has_n {
                // Clamp to [0,1]: the eye normal is now un-renormalized (unless
                // D3DRS_NORMALIZENORMALS), so a magnitude > 1 can push the dot
                // past 1 — D3D9 clamps the diffuse N·L term.
                out.push_str("        float ndotl = clamp(dot(n, L), 0.0, 1.0);\n");
                let _ = writeln!(
                    out,
                    "        diffuseAccum += atten * ndotl * (vs_c[{d}] * {mat_diffuse});",
                    d = base + 2
                );
            }
            let _ = writeln!(
                out,
                "        diffuseAccum += atten * (vs_c[{a}] * {mat_ambient});",
                a = base + 3
            );
            if has_n && vs.specular_enable() {
                // Blinn-Phong specular: H = normalize(L + V), NdotH = max(0, n·H),
                // specFactor = NdotH^power (zero when ndotl <= 0). Weighted by
                // lightSpecular × matSpecular per the D3D9 lighting equation;
                // rgb only — FF lighting defines no specular alpha.
                out.push_str("        float3 H = normalize(L + V);\n");
                out.push_str("        float ndoth = max(0.0, dot(n, H));\n");
                out.push_str(
                    "        float specFactor = (ndotl > 0.0) ? pow(ndoth, mat_power) : 0.0;\n",
                );
                let _ = writeln!(
                    out,
                    "        specAccum += atten * specFactor * (vs_c[{s}].rgb * {mat_specular}.rgb);",
                    s = base + 5
                );
            }
            let _ = writeln!(out, "    }}");
        }
        // Saturate and preserve material-diffuse alpha on color0.
        out.push_str("    float4 lit = saturate(diffuseAccum);\n");
        let _ = writeln!(out, "    lit.a = {mat_diffuse}.a;");
        out.push_str("    out.color0 = lit;\n");
        out.push_str("    out.color1 = float4(saturate(specAccum), 0.0);\n");
    } else {
        if vs.has_color0() {
            out.push_str("    out.color0 = in.v2;\n");
        } else if vs.has_diffuse_declared_unbound() {
            // DIFFUSE declared on an unbound (non-stream-0) source reads 0,
            // not the white material default used when diffuse is absent.
            out.push_str("    out.color0 = float4(0.0);\n");
        } else {
            // A missing DIFFUSE stream reads opaque white in FF — the
            // same default the XYZRHW branch uses. Material diffuse is
            // NOT the unlit fallback.
            out.push_str("    out.color0 = float4(1.0);\n");
        }
        // Unlit oD1 is the vertex COLOR1 attribute (specular) when declared.
        if vs.has_color1() {
            out.push_str("    out.color1 = in.v3;\n");
        } else {
            out.push_str("    out.color1 = float4(0.0);\n");
        }
    }

    // Un-normalized eye-space normal for CAMERASPACENORMAL texgen. D3D9 does
    // NOT normalize generated texture coordinates (that is only done for
    // lighting, and only under D3DRS_NORMALIZENORMALS), so this is distinct
    // from the normalized `n` the lighting branch declares.
    let need_texgen_normal =
        vs.has_normal() && vs.tci_modes[..vs.tex_coord_count as usize].contains(&1);
    if need_texgen_normal {
        if vs.vertex_blend_count > 0 {
            out.push_str("    float3 n_texgen = n_blend;\n");
        } else {
            out.push_str("    float3 n_texgen = float3(dot(in.v1.xyz, vs_c[0].xyz), dot(in.v1.xyz, vs_c[1].xyz), dot(in.v1.xyz, vs_c[2].xyz));\n");
        }
    }

    // Per-stage texcoord emission — the D3D9 fixed-function texture-coordinate
    // transform. The TCI mode picks the raw source coordinate; D3DTTFF then
    // optionally multiplies it by the per-stage texture matrix at
    // `vs_c[63 + i*4]` (pre-transposed) and selects how many components
    // survive. PROJECTED stashes the projective divisor in `.w` for the PS to
    // divide by (the PS reads the varying un-divided, so the divide
    // must happen at sample time, not here). Implements the D3D9 D3DTTFF
    // texture-coordinate transform per the spec.
    for i in 0..vs.tex_coord_count as usize {
        let mode = vs.tci_modes[i];
        let src = u32::from(vs.tci_coord_indices[i].min(7));
        let tt = vs.tt_flags[i];
        let count = tt_count(tt); // 0 (passthru) or 2/3/4 (matrix transform)
        let projected = tt_projected(tt);

        // Raw source coordinate `raw{i}` (components beyond its dimension `n`
        // already zeroed) and its dimension `n`.
        let n: u8;
        match mode {
            0 => {
                n = input_dim(vs, src);
                let _ = writeln!(out, "    float4 raw{i} = {};", masked_input_rhs(vs, i, src));
            }
            1 if vs.has_normal() => {
                n = 3;
                let _ = writeln!(out, "    float4 raw{i} = float4(n_texgen, 0.0);");
            }
            2 => {
                n = 3;
                let _ = writeln!(out, "    float4 raw{i} = float4(posEye, 0.0);");
            }
            3 if vs.has_normal() => {
                // R = 2 * N * dot(N, E) - E, where E = normalize(posEye).
                n = 3;
                let _ = writeln!(out, "    float4 raw{i};");
                let _ = writeln!(out, "    {{");
                out.push_str("        float3 E_tci = normalize(posEye);\n");
                out.push_str("        float3 R_tci = 2.0 * n * dot(n, E_tci) - E_tci;\n");
                let _ = writeln!(out, "        raw{i} = float4(R_tci, 0.0);");
                let _ = writeln!(out, "    }}");
            }
            1 | 3 => {
                mtld3d_shared::log_once_warn!(target: super::LOG_TARGET,
                    "dxso FF: TCI mode {mode} needs a vertex normal but none declared → passthru"
                );
                n = input_dim(vs, src);
                let _ = writeln!(out, "    float4 raw{i} = {};", masked_input_rhs(vs, i, src));
            }
            _ => {
                // 4 = SPHEREMAP; higher values are undefined. SPHEREMAP
                // is not implemented.
                mtld3d_shared::log_once_warn!(target: super::LOG_TARGET, "dxso FF: TCI mode {mode} not implemented → passthru");
                n = input_dim(vs, src);
                let _ = writeln!(out, "    float4 raw{i} = {};", masked_input_rhs(vs, i, src));
            }
        }

        if (2..=4).contains(&count) {
            // D3DTTFF_COUNT2..4: pad the first unbacked component to 1.0,
            // multiply by the texture matrix, keep `count` components (zero
            // the rest), and — when PROJECTED — copy the last kept component
            // into `.w` so the PS divides by it.
            let base = 63 + i * 4;
            if (1..4).contains(&n) {
                let _ = writeln!(out, "    raw{i}[{n}] = 1.0;");
            }
            let _ = writeln!(
                out,
                "    float4 r{i} = float4(dot(raw{i}, vs_c[{r0}]), dot(raw{i}, vs_c[{r1}]), dot(raw{i}, vs_c[{r2}]), dot(raw{i}, vs_c[{r3}]));",
                r0 = base,
                r1 = base + 1,
                r2 = base + 2,
                r3 = base + 3,
            );
            let tc_expr = match (count, projected) {
                (2, false) => format!("float4(r{i}.x, r{i}.y, 0.0, 0.0)"),
                (2, true) => format!("float4(r{i}.x, r{i}.y, 0.0, r{i}.y)"),
                (3, false) => format!("float4(r{i}.x, r{i}.y, r{i}.z, 0.0)"),
                (3, true) => format!("float4(r{i}.x, r{i}.y, r{i}.z, r{i}.z)"),
                // COUNT4 keeps all four; `.w` is already the projective divisor.
                _ => format!("r{i}"),
            };
            let _ = writeln!(out, "    out.texcoord{i} = {tc_expr};");
        } else {
            // Passthru (DISABLE / COUNT1 / count > COUNT4): the masked raw
            // coordinate unchanged. PROJECTED copies the last provided
            // component (index n-1) into `.w` as the divisor.
            if projected && n >= 1 {
                let comp = ['x', 'y', 'z', 'w'][(n - 1) as usize];
                let _ = writeln!(
                    out,
                    "    out.texcoord{i} = float4(raw{i}.xyz, raw{i}.{comp});"
                );
            } else {
                let _ = writeln!(out, "    out.texcoord{i} = raw{i};");
            }
        }
    }
    for i in vs.tex_coord_count as usize..8 {
        let _ = writeln!(out, "    out.texcoord{i} = float4(0.0);");
    }

    // Vertex fog factor (linear / exp / exp2). Eye-space Z from vs_c[2]
    // (row 2 of transpose(WV), full 4 components including translation) so
    // `dot(pos, vs_c[2])` is the eye-space Z coordinate of the vertex. Fog
    // params live at vs_c[8] (see the layout comment above).
    match vs.fog_mode {
        1 => {
            // EXP: exp(-density * z) = exp2(-1.442695 * density * z)
            out.push_str("    {\n");
            out.push_str("        float eyeZ = abs(dot(pos, vs_c[2]));\n");
            out.push_str("        float fogDensity = vs_c[8].z;\n");
            out.push_str("        float f = saturate(exp2(-1.442695 * fogDensity * eyeZ));\n");
            out.push_str("        out.fog = float4(f, 0.0, 0.0, 0.0);\n");
            out.push_str("    }\n");
        }
        2 => {
            // EXP2: exp(-(density*z)^2) = exp2(-1.442695 * (density*z)^2)
            out.push_str("    {\n");
            out.push_str("        float eyeZ = abs(dot(pos, vs_c[2]));\n");
            out.push_str("        float fogDensity = vs_c[8].z;\n");
            out.push_str("        float dz = fogDensity * eyeZ;\n");
            out.push_str("        float f = saturate(exp2(-1.442695 * dz * dz));\n");
            out.push_str("        out.fog = float4(f, 0.0, 0.0, 0.0);\n");
            out.push_str("    }\n");
        }
        3 => {
            // LINEAR: (end - z) / (end - start)
            out.push_str("    {\n");
            out.push_str("        float eyeZ = abs(dot(pos, vs_c[2]));\n");
            out.push_str("        float fogStart = vs_c[8].x;\n");
            out.push_str("        float fogEnd = vs_c[8].y;\n");
            // Signed denominator (no `max(..,eps)`): reversed fog (start>end)
            // needs the negative range, and start==end is fully fogged (f=0)
            // per D3D9 — a clamped denominator would instead yield +inf. For
            // the normal start<end case the denominator is exactly `end-start`.
            out.push_str("        float fogRange = fogEnd - fogStart;\n");
            out.push_str(
                "        float f = fogRange == 0.0 ? 0.0 : saturate((fogEnd - eyeZ) / fogRange);\n",
            );
            out.push_str("        out.fog = float4(f, 0.0, 0.0, 0.0);\n");
            out.push_str("    }\n");
        }
        // Per-vertex fog factor sourced from the specular (COLOR1) alpha — the
        // D3D9 path when fog is enabled but both vertex and table fog modes are
        // D3DFOG_NONE. No declared specular falls through to the unfogged
        // default below (`float4(1.0)`), matching the D3D9 oFog default.
        4 if vs.has_color1() => {
            out.push_str("    out.fog = float4(in.v3.w, 0.0, 0.0, 0.0);\n");
        }
        _ => {
            out.push_str("    out.fog = float4(1.0);\n");
        }
    }
    // NDC depth for the table-fog Z source (see the Varyings decl).
    out.push_str("    out.fog_z = out.position.z / out.position.w;\n");

    out.push_str("    return out;\n");
    out.push_str("}\n");
}

bitflags::bitflags! {
    /// Vertex-colour availability that gates FF material-source resolution.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct MatColorFlags: u8 {
        /// `D3DRS_COLORVERTEX` is on — per-vertex colour may override the material constant.
        const COLOR_VERTEX = 1 << 0;
        /// The vertex carries a `COLOR0` (diffuse) channel.
        const HAS_COLOR0 = 1 << 1;
        /// The vertex carries a `COLOR1` (specular) channel.
        const HAS_COLOR1 = 1 << 2;
        /// The draw uses a vertex declaration (not a legacy FVF).
        ///
        /// An absent colour reads 0 rather than falling back to the material
        /// constant.
        const USES_DECL = 1 << 3;
    }
}

/// Resolves a D3D9 D3DMATERIALCOLORSOURCE field to its MSL expression.
///
/// The field (0 = `MCS_MATERIAL`, 1 = `MCS_COLOR1`, 2 = `MCS_COLOR2`) selects
/// the expression that feeds the FF lighting math. When `D3DRS_COLORVERTEX` is
/// false, the override is ignored and the material constant is always used.
fn resolve_mat(source: u8, mat_slot: u32, flags: MatColorFlags) -> String {
    if flags.contains(MatColorFlags::COLOR_VERTEX) {
        if source == 1 && flags.contains(MatColorFlags::HAS_COLOR0) {
            return "in.v2".to_string();
        }
        if source == 2 && flags.contains(MatColorFlags::HAS_COLOR1) {
            return "in.v3".to_string();
        }
        if (source == 1 || source == 2) && flags.contains(MatColorFlags::USES_DECL) {
            // A vertex declaration that omits the requested COLOR reads 0 — only
            // the legacy FVF path falls back to the material colour for an
            // absent vertex colour source.
            return "float4(0.0)".to_string();
        }
        if source > 2 {
            mtld3d_shared::log_once_warn!(target: super::LOG_TARGET, "dxso FF: unknown material source {source} → MCS_MATERIAL");
        }
    }
    format!("vs_c[{mat_slot}]")
}

// ── PS ──
// Constant buffer layout (float4 rows at `ps_c[i]`):
//   [0]  texture factor RGBA
// Alpha reference lives in a dedicated scalar buffer on slot 14 and fog
// color in a dedicated float4 on slot 13 — same contract as the
// user-shader path in `emit.rs::emit_ps_function`.
fn emit_ps(out: &mut String, ps: &FfPsKey, variant: VariantKey, entry: &str) {
    // Discover which stages sample a texture so we know which textures+samplers
    // to declare in the entry-point signature.
    let _ = writeln!(out, "fragment float4 {entry}(");
    out.push_str("    Varyings in [[stage_in]],\n");
    out.push_str("    constant float4 *ps_c [[buffer(15)]]");
    let alpha_test_active =
        variant.alpha_func != 0 && u32::from(variant.alpha_func) != D3DCMP_ALWAYS;
    if alpha_test_active {
        out.push_str(",\n    constant float &alpha_ref [[buffer(14)]]");
    }
    if fog_blend_active(variant) {
        out.push_str(",\n    constant float4 *fog_data [[buffer(13)]]");
    }
    for (i, stage) in ps.stages.iter().enumerate() {
        if u32::from(stage.color_op) == D3DTOP_DISABLE {
            break;
        }
        if stage.has_texture {
            // A depth-format texture bound to this slot (sampleable shadow
            // map) must be declared `depth2d<float>` — Metal rejects binding a
            // `Depth32Float` texture to a `texture2d<float>` slot. Mirrors the
            // programmable emitter's depth branch (`emit.rs`).
            let tex_ty = if (variant.depth_sampler_mask & (1u16 << i)) != 0 {
                "depth2d<float>"
            } else if (variant.volume_sampler_mask & (1u16 << i)) != 0 {
                // Volume (3D) texture bound: the underlying MTLTexture is
                // `MTLTextureType3D`; binding it via `texture2d<float>` fails
                // Metal's type-check and samples black.
                "texture3d<float>"
            } else {
                "texture2d<float>"
            };
            let _ = write!(
                out,
                ",\n    {tex_ty} s{i} [[texture({i})]],\n    sampler samp{i} [[sampler({i})]]"
            );
        }
    }
    out.push_str("\n) {\n");

    // current = diffuse by default (CURRENT at stage 0 resolves to DIFFUSE
    // since no previous stage contributed).
    out.push_str("    float4 current = in.color0;\n");

    for (i, stage) in ps.stages.iter().enumerate() {
        if u32::from(stage.color_op) == D3DTOP_DISABLE {
            break;
        }
        if stage.has_texture {
            // VS emits the TCI-resolved coord for stage i into
            // `Varyings.texcoord[i]`, so PS stage i samples slot i directly.
            // Depth slots are comparison samplers (`compareFunction =
            // LessEqual`); a `depth2d` texture must be read with
            // `sample_compare` (reference = the texcoord's z), matching the
            // programmable emitter's `sample_or_compare` depth branch.
            if (variant.depth_fetch_mask & (1u16 << i)) != 0 {
                // INTZ/DF24/DF16: read the RAW stored depth (broadcast) via a
                // plain `.sample()` on the depth2d binding — not a shadow
                // comparison — honouring D3DTTFF_PROJECTED like the colour path.
                let uv = if (ps.tt_projected_mask & (1u8 << i)) != 0 {
                    format!(
                        "(in.texcoord{i}.w != 0.0 ? in.texcoord{i}.xy / in.texcoord{i}.w : float2(0.0))"
                    )
                } else {
                    format!("in.texcoord{i}.xy")
                };
                let _ = writeln!(
                    out,
                    "    float4 t{i} = float4(s{i}.sample(samp{i}, {uv}, level(0)));"
                );
            } else if (variant.depth_sampler_mask & (1u16 << i)) != 0 {
                let _ = writeln!(
                    out,
                    "    float4 t{i} = float4(s{i}.sample_compare(samp{i}, in.texcoord{i}.xy, saturate(in.texcoord{i}.z), level(0)));",
                );
            } else {
                // D3DTTFF_PROJECTED: divide the coordinate by `.w` before
                // sampling (the FF / ps_1_x projective sample). `.w == 0`
                // samples at the origin, matching native (D3D9 returns the
                // (0,0) texel there). A volume-bound slot samples with the
                // full `.xyz` coordinate (texture3d takes a float3).
                let volume = (variant.volume_sampler_mask & (1u16 << i)) != 0;
                let (sw, zero) = if volume {
                    ("xyz", "float3(0.0)")
                } else {
                    ("xy", "float2(0.0)")
                };
                let uv = if (ps.tt_projected_mask & (1u8 << i)) != 0 {
                    format!(
                        "(in.texcoord{i}.w != 0.0 ? in.texcoord{i}.{sw} / in.texcoord{i}.w : {zero})"
                    )
                } else {
                    format!("in.texcoord{i}.{sw}")
                };
                let _ = writeln!(out, "    float4 t{i} = s{i}.sample(samp{i}, {uv});");
            }
        }
        // Unbound-texture "invalid op" handling: a stage with NO bound texture
        // whose op consumes a D3DTA_TEXTURE argument resolves to
        // SELECTARG1(CURRENT) — the unbound-texture default is CURRENT, not
        // opaque white. SELECTARG1 of CURRENT is just `current`, so short-circuit
        // to it. Colour and alpha are tested independently; textured stages take
        // the byte-identical path.
        let color_expr = if !stage.has_texture
            && op_reads_texture(stage.color_op, stage.color_arg1, stage.color_arg2)
        {
            "current".to_string()
        } else {
            let c1 = resolve_arg(stage.color_arg1, i, stage.has_texture);
            let c2 = resolve_arg(stage.color_arg2, i, stage.has_texture);
            apply_op(stage.color_op, &c1, &c2, i, stage.has_texture)
        };
        let alpha_expr = if !stage.has_texture
            && op_reads_texture(stage.alpha_op, stage.alpha_arg1, stage.alpha_arg2)
        {
            "current".to_string()
        } else {
            let a1 = resolve_arg(stage.alpha_arg1, i, stage.has_texture);
            let a2 = resolve_arg(stage.alpha_arg2, i, stage.has_texture);
            apply_op_scalar(stage.alpha_op, &a1, &a2, i, stage.has_texture)
        };
        let _ = writeln!(
            out,
            "    current = float4(({color_expr}).rgb, ({alpha_expr}).a);",
        );
    }

    // End-of-cascade specular add: oD1 joins the cascade result after the
    // last stage and before fog, rgb only (alpha untouched), clamped like
    // every cascade op.
    if ps.specular_add {
        out.push_str("    current = float4(saturate(current.rgb + in.color1.rgb), current.a);\n");
    }

    out.push_str("    float4 oC0 = current;\n");

    // Fog blend (vertex fog from `in.fog.x`, or per-pixel table fog from the
    // rasterizer position — see `emit::write_fog_blend`). Alpha untouched per
    // D3D9 spec. Fog data binds on slot 13, shared with the programmable PS
    // emitter so both paths stay in lockstep.
    write_fog_blend(out, variant, "oC0");

    // Alpha test from VariantKey (packs D3DCMP_* in .alpha_func; 0 or
    // D3DCMP_ALWAYS → no discard). Ref value read from scalar buffer slot 14.
    let af = u32::from(variant.alpha_func);
    if af != 0 && af != D3DCMP_ALWAYS {
        let cmp = match af {
            D3DCMP_NEVER => "false",
            D3DCMP_LESS => "oC0.a < alpha_ref",
            D3DCMP_EQUAL => "oC0.a == alpha_ref",
            D3DCMP_LESSEQUAL => "oC0.a <= alpha_ref",
            D3DCMP_GREATER => "oC0.a > alpha_ref",
            D3DCMP_NOTEQUAL => "oC0.a != alpha_ref",
            D3DCMP_GREATEREQUAL => "oC0.a >= alpha_ref",
            other => {
                mtld3d_shared::log_once_warn!(target: super::LOG_TARGET, "ff-fallback alpha_func unhandled={other} → always-pass");
                "true"
            }
        };
        let _ = writeln!(out, "    if (!({cmp})) discard_fragment();");
    }
    // D3DRS_SRGBWRITEENABLE: in-shader linear→sRGB encode on the final colour,
    // after fog/specular/alpha-test, alpha left linear. Shared helper with the
    // programmable PS emitter (`emit::emit_srgb_write_helper`).
    if variant.flags.contains(VariantFlags::SRGB_WRITE) {
        out.push_str("    oC0.rgb = mtld3d_linear_to_srgb(oC0.rgb);\n");
    }
    out.push_str("    return oC0;\n");
    out.push_str("}\n");
}

/// Does `op` read a `D3DTA_TEXTURE` argument?
///
/// An op that selects only the other slot (SELECTARG1 ignores arg2, SELECTARG2
/// ignores arg1) does not consume the unread one. Used to detect a texture
/// stage referencing an unbound texture so it can be rewritten to
/// SELECTARG1(CURRENT) — the D3D9 default for an unbound-texture stage.
fn op_reads_texture(op: u8, arg1: u8, arg2: u8) -> bool {
    let op = u32::from(op);
    let is_tex = |a: u8| u32::from(a & 0x0f) == D3DTA_TEXTURE;
    (is_tex(arg1) && op != D3DTOP_SELECTARG2) || (is_tex(arg2) && op != D3DTOP_SELECTARG1)
}

fn resolve_arg(arg: u8, stage: usize, has_texture: bool) -> String {
    let selector = u32::from(arg & 0x0f);
    let arg32 = u32::from(arg);
    let resolved = match selector {
        D3DTA_DIFFUSE => "in.color0".to_string(),
        D3DTA_CURRENT => "current".to_string(),
        D3DTA_SPECULAR => "in.color1".to_string(),
        D3DTA_TEXTURE => {
            if has_texture {
                format!("t{stage}")
            } else {
                "float4(1.0)".to_string()
            }
        }
        D3DTA_TFACTOR => "ps_c[0]".to_string(),
        other => {
            mtld3d_shared::log_once_warn!(target: super::LOG_TARGET,
                "ff-fallback texture-arg unhandled={other} (stage={stage}) → float4(1.0)"
            );
            "float4(1.0)".to_string()
        }
    };
    // D3DTA_ALPHAREPLICATE broadcasts the alpha channel across RGBA, then
    // D3DTA_COMPLEMENT inverts (1 - x) — applied in that order per D3D9.
    let mut expr = resolved;
    if arg32 & D3DTA_ALPHAREPLICATE != 0 {
        expr = format!("{expr}.aaaa");
    }
    if arg32 & D3DTA_COMPLEMENT != 0 {
        expr = format!("(1.0 - {expr})");
    }
    let unhandled = arg32 & 0xf0 & !(D3DTA_ALPHAREPLICATE | D3DTA_COMPLEMENT);
    if unhandled != 0 {
        mtld3d_shared::log_once_warn!(target: super::LOG_TARGET,
            "ff-fallback texture-arg modifier unhandled={unhandled:#x} (arg={arg:#x})"
        );
    }
    expr
}

fn apply_op(op: u8, a: &str, b: &str, stage: usize, has_texture: bool) -> String {
    match u32::from(op) {
        D3DTOP_SELECTARG1 => a.to_string(),
        D3DTOP_SELECTARG2 => b.to_string(),
        D3DTOP_MODULATE => format!("({a} * {b})"),
        D3DTOP_MODULATE2X => format!("saturate(2.0 * {a} * {b})"),
        D3DTOP_MODULATE4X => format!("saturate(4.0 * {a} * {b})"),
        D3DTOP_ADD => format!("saturate({a} + {b})"),
        D3DTOP_ADDSIGNED => format!("saturate({a} + {b} - 0.5)"),
        D3DTOP_ADDSIGNED2X => format!("saturate(2.0 * ({a} + {b} - 0.5))"),
        D3DTOP_SUBTRACT => format!("saturate({a} - {b})"),
        D3DTOP_ADDSMOOTH => format!("saturate({a} + {b} * (1.0 - {a}))"),
        D3DTOP_BLENDDIFFUSEALPHA => {
            format!("({a} * in.color0.a + {b} * (1.0 - in.color0.a))")
        }
        D3DTOP_BLENDTEXTUREALPHA => {
            let tex = if has_texture {
                format!("t{stage}")
            } else {
                "current".to_string()
            };
            format!("({a} * {tex}.a + {b} * (1.0 - {tex}.a))")
        }
        D3DTOP_BLENDFACTORALPHA => format!("({a} * ps_c[0].a + {b} * (1.0 - ps_c[0].a))"),
        D3DTOP_BLENDCURRENTALPHA => format!("({a} * current.a + {b} * (1.0 - current.a))"),
        // result = saturate(4 * dot(Arg1.rgb - 0.5, Arg2.rgb - 0.5)), broadcast
        // to RGBA (including alpha) — the D3D9 signed dot-product for tangent-
        // space normal lighting.
        D3DTOP_DOTPRODUCT3 => {
            format!("float4(saturate(4.0 * dot(({a}).rgb - 0.5, ({b}).rgb - 0.5)))")
        }
        other => {
            mtld3d_shared::log_once_warn!(target: super::LOG_TARGET, "ff-fallback texture-op unhandled={other} → SELECTARG1");
            a.to_string()
        }
    }
}

fn apply_op_scalar(op: u8, a: &str, b: &str, stage: usize, has_texture: bool) -> String {
    // Same algebra — MSL handles float4 and scalar uniformly; we take .a below.
    apply_op(op, a, b, stage, has_texture)
}
