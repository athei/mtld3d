//! DXSO → MSL text emitter.
//!
//! Emits each stage (VS or PS) as its own MSL translation unit with a single
//! entry point (`mtld3d_vs` or `mtld3d_ps`), so a render pipeline can freely
//! mix custom-VS + FF-PS or vice versa.
//!
//! Layout conventions:
//! - Float constants live in buffer slot 15 as `float4 *vs_c` / `ps_c`.
//! - Dynamic VS integer constants (a `loop`/`rep` counter fed by
//!   `SetVertexShaderConstantI`) live in vertex buffer slot 14 as `int4 *vs_i`,
//!   declared only when the shader reads a non-`defi` integer constant.
//! - Vertex attributes are declared at `[[attribute(N)]]` where N is the
//!   D3D9 input register index (v0 → attribute 0). The pipeline's vertex
//!   descriptor must place matching data at those slots.
//! - Varyings use `position` first, then texcoord0..7, then color0..1.
//!   Texcoord-before-color ordering works around a Metal shader-compiler
//!   crash. The same struct shape is emitted by every per-stage
//!   function — VS and PS from independent libraries still link at
//!   pipeline-creation time.

use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    fmt::Write,
};

use super::{
    ir::{
        DeclUsage, Declaration, DstMods, DstOperand, DxsoProgram, InstrFlags, Instruction, RegKind,
        Register, ShaderType, SrcModifier, SrcOperand, Swizzle, TextureType, WriteMask,
    },
    opcode::Opcode,
};

bitflags::bitflags! {
    /// Per-variant boolean features folded into the PS shader cache key.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
    pub struct VariantFlags: u8 {
        /// Table-fog source select, only meaningful when `fog_table_mode != 0`.
        ///
        /// Clear otherwise, to avoid variant churn: clear = pixel depth
        /// (`in.position.z` + `D3DRS_DEPTHBIAS`), set = eye W
        /// (`1 / in.position.w`). D3D9 picks by the projection matrix's 4th
        /// column — (0,0,0,1) means W==1 everywhere (orthographic) so fog reads
        /// Z; anything else reads W (the D3D9 pixel-fog source rule).
        const FOG_SOURCE_W = 1 << 0;
        /// `D3DRS_SHADEMODE == D3DSHADE_FLAT`.
        ///
        /// When set, the PS `Varyings` struct declares `color0`/`color1` with
        /// the `[[flat]]` interpolation qualifier so the diffuse/specular
        /// colours come from the provoking (first) vertex instead of being
        /// perspective-interpolated. Folded into the PS cache key so FLAT and
        /// GOURAUD draws get distinct libraries.
        const FLAT_SHADE = 1 << 1;
        /// `D3DRS_SRGBWRITEENABLE != 0`.
        ///
        /// When set, the PS applies the linear→sRGB OETF to the final colour
        /// rgb (alpha untouched) just before output, after
        /// fog/specular/alpha-test. D3D9 semantics are "encode the shader
        /// output linear→sRGB on write to an sRGB-capable render target"; the
        /// Metal RT/pipeline format stays PLAIN (so formats without an sRGB
        /// twin, e.g. R5G6B5, still work) and the encode happens in-shader.
        /// Folded into the PS cache key so sRGB-write and plain draws get
        /// distinct libraries.
        const SRGB_WRITE = 1 << 2;
    }
}

/// Shader specialization key carried through PS emission.
///
/// Covers the features that aren't part of the per-stage pixel combiner state.
///
/// `alpha_func` is the D3DCMP_* value for the alpha test; `fog_mode` mirrors
/// `FfVsKey::fog_mode` and gates the PS-side fog blend (0 = no blend
/// emitted).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct VariantKey {
    pub alpha_func: u8,
    pub fog_mode: u8,
    /// Per-pixel ("table") fog — `D3DRS_FOGTABLEMODE`.
    ///
    /// Set when fog is enabled and the table mode isn't NONE: 1 = EXP,
    /// 2 = EXP2, 3 = LINEAR (`D3DFOG_*`). Non-zero makes the PS compute the fog
    /// factor per-pixel from the rasterizer position (the interpolated
    /// `in.fog.x` vertex factor is ignored); `fog_mode` is 0 in that case. Fog
    /// params arrive in `fog_data[1]` = (start, end, density, depth-bias) on
    /// buffer 13.
    pub fog_table_mode: u8,
    /// Bit `i` set ⇒ sampler slot `i` is bound to a depth-format texture.
    ///
    /// The PS emitter outputs `depth2d<float>` for that slot and wraps
    /// `s{i}.sample(...)` in `float4(...)` (the depth sampler returns a
    /// single-channel scalar; downstream code reads `.x/.y/.z/.w` and would
    /// otherwise miscompile). Folded into the PS shader cache key so different
    /// bind shapes get distinct compiled libraries.
    pub depth_sampler_mask: u16,
    /// Bit `i` set ⇒ sampler slot `i` is a "readable raw depth" FOURCC texture.
    ///
    /// INTZ/DF24/DF16 — a SUBSET of `depth_sampler_mask`. The slot still binds
    /// as `depth2d<float>`, but is read with a plain `.sample()` returning the
    /// RAW stored depth (broadcast to `float4`) instead of `sample_compare`,
    /// and gets a non-comparison sampler — per D3D9, raw-depth FOURCC
    /// formats are read as raw values rather than through a hardware shadow
    /// comparison. Part of the PS cache key.
    pub depth_fetch_mask: u16,
    /// Bit `i` set ⇒ sampler slot `i` is bound to a volume (3D) texture.
    ///
    /// A `CreateVolumeTexture` resource with `depth > 1`, which the unix
    /// side creates as `MTLTextureType3D`. The FIXED-FUNCTION PS declares
    /// `texture3d<float>` and samples with the texcoord's `.xyz` for such
    /// slots; declaring `texture2d<float>` would fail Metal's binding
    /// type-check and sample black.
    /// The programmable path ignores this — SM2/SM3 carry the sampler
    /// dimensionality in their `dcl` tokens. Part of the PS cache key.
    pub volume_sampler_mask: u16,
    /// Bit `i` set ⇒ texture stage `i` has `D3DTTFF_PROJECTED`.
    ///
    /// The fixed-function and `ps_1_0`..`ps_1_3` pixel pipelines apply an
    /// IMPLICIT per-pixel projective divide (coord ÷ its `.w`) before sampling
    /// that stage; `ps_1_4` projects via DZ/DW modifiers and `ps_2_0`+ ignore
    /// TTFF, so the SM1 emitter consumes this only for `shader_minor < 4`.
    /// Folded in at draw time (the FF PS uses its own `FfPsKey` mask). Part of
    /// the PS cache key.
    pub tt_projected_mask: u8,
    /// Packed boolean features — pixel-fog source, flat shade, sRGB write.
    ///
    /// See [`VariantFlags`].
    pub flags: VariantFlags,
}

#[derive(Debug)]
pub enum EmitError {
    WrongShaderType,
    UnsupportedInstruction(String),
    UnsupportedRegisterKind(String),
}

/// Default VS / PS entry-point names used when callers don't supply one.
///
/// Those callers are the tests and the offline `disasm` tool.
///
/// Production paths route through the `_named` variants with stable
/// per-shader-id strings so each Metal pipeline state in Xcode's capture
/// inspector shows a distinct function name (`mtld3d_vs_ff_5f3a0001`,
/// `mtld3d_ps_sm3_a2b1c4d8`, …).
pub const DEFAULT_VS_ENTRY: &str = "mtld3d_vs";
pub const DEFAULT_PS_ENTRY: &str = "mtld3d_ps";

/// Emit MSL for a vertex shader using the default `mtld3d_vs` entry name.
///
/// # Errors
///
/// See [`emit_vs_programmable_named`].
pub fn emit_vs_programmable(vs: &DxsoProgram) -> Result<String, EmitError> {
    emit_vs_programmable_named(vs, DEFAULT_VS_ENTRY, u16::MAX)
}

/// Emit MSL for a pixel shader using the default `mtld3d_ps` entry name.
///
/// # Errors
///
/// See [`emit_ps_programmable_named`].
pub fn emit_ps_programmable(ps: &DxsoProgram, variant: VariantKey) -> Result<String, EmitError> {
    emit_ps_programmable_named(ps, variant, DEFAULT_PS_ENTRY)
}

/// Emit MSL for a vertex shader, naming the function `entry`.
///
/// # Errors
///
/// [`EmitError::WrongShaderType`] if `vs.shader_type` isn't [`ShaderType::Vertex`];
/// [`EmitError::UnsupportedInstruction`] for opcodes not yet lowered;
/// [`EmitError::UnsupportedRegisterKind`] for register classes outside the
/// SM2/SM3 set we model.
pub fn emit_vs_programmable_named(
    vs: &DxsoProgram,
    entry: &str,
    provided_mask: u16,
) -> Result<String, EmitError> {
    if vs.shader_type != ShaderType::Vertex {
        return Err(EmitError::WrongShaderType);
    }
    let mut out = String::new();
    w(&mut out, "#include <metal_stdlib>\n");
    w(&mut out, "using namespace metal;\n\n");
    emit_vertex_in(&mut out, vs, provided_mask);
    emit_varyings(&mut out, false);
    emit_const_rel_helper(&mut out, vs);
    emit_vs_function(&mut out, vs, entry, provided_mask)?;
    Ok(out)
}

/// Relative constant addressing (`c[a0.<comp> + N]`) must index the FULL constant space.
///
/// That includes `def`-declared constants. Those are emitted as scalar `c{n}`
/// locals (cheap direct reads), so the dynamic index can't reach them through
/// the uniform buffer alone — slots a shader `def`s but the app never uploads
/// would read zero. Emit a lookup helper that returns the `def` value for any
/// matching register index and falls back to the uniform buffer (`cb`)
/// otherwise. Only emitted when the shader both uses relative addressing and
/// declares at least one `def` constant; the `load_src` rel-addr path routes
/// through it under the same condition.
fn emit_const_rel_helper(out: &mut String, prog: &DxsoProgram) {
    if !prog.uses_relative_const_addressing() || prog.def_constants.is_empty() {
        return;
    }
    w(
        out,
        "inline float4 mtld3d_const_rel(int idx, constant float4 *cb) {\n",
    );
    w(out, "    switch (idx) {\n");
    for def in &prog.def_constants {
        let v = def.value;
        let _ = writeln!(
            out,
            "        case {idx}: return float4({x}, {y}, {z}, {w});",
            idx = def.reg.index,
            x = fmt_float(v[0]),
            y = fmt_float(v[1]),
            z = fmt_float(v[2]),
            w = fmt_float(v[3])
        );
    }
    w(out, "        default: break;\n");
    w(out, "    }\n");
    // Clamp the fallback index into the D3D9 const-register range so a negative
    // or out-of-range `a0` can never index the uniform buffer out of bounds.
    w(out, "    return cb[clamp(idx, 0, 255)];\n");
    w(out, "}\n\n");
}

/// Emit MSL for a pixel shader, naming the function `entry`.
///
/// # Errors
///
/// [`EmitError::WrongShaderType`] if `ps.shader_type` isn't [`ShaderType::Pixel`];
/// [`EmitError::UnsupportedInstruction`] / [`EmitError::UnsupportedRegisterKind`]
/// per [`emit_vs_programmable_named`].
pub fn emit_ps_programmable_named(
    ps: &DxsoProgram,
    variant: VariantKey,
    entry: &str,
) -> Result<String, EmitError> {
    if ps.shader_type != ShaderType::Pixel {
        return Err(EmitError::WrongShaderType);
    }
    let mut out = String::new();
    w(&mut out, "#include <metal_stdlib>\n");
    w(&mut out, "using namespace metal;\n\n");
    emit_varyings(&mut out, variant.flags.contains(VariantFlags::FLAT_SHADE));
    emit_const_rel_helper(&mut out, ps);
    if variant.flags.contains(VariantFlags::SRGB_WRITE) {
        emit_srgb_write_helper(&mut out);
    }
    emit_ps_function(&mut out, ps, variant, entry)?;
    Ok(out)
}

fn w(out: &mut String, s: &str) {
    out.push_str(s);
}

/// Emit the linear→sRGB OETF helper for `D3DRS_SRGBWRITEENABLE`.
///
/// The RT is kept at its plain (non-`_srgb`) Metal pixel format — required for
/// formats without an sRGB twin such as `R5G6B5` — so the encode happens in the
/// pixel shader on the final colour rgb (alpha is left linear). Per channel this
/// is the standard sRGB transfer function:
/// `c <= 0.0031308 ? 12.92*c : 1.055*c^(1/2.4) - 0.055`.
/// `select(hi, lo, cond)` returns `cond ? lo : hi` per component; the `pow`
/// domain is guarded with `max(c, 0)` so a negative lane never yields a NaN
/// (that lane always selects the linear branch anyway). Shared by the FF PS
/// emitter (`ff::emit_ps_ff_named`).
pub fn emit_srgb_write_helper(out: &mut String) {
    w(
        out,
        "static inline float3 mtld3d_linear_to_srgb(float3 c) {\n",
    );
    w(out, "    float3 lo = 12.92 * c;\n");
    w(
        out,
        "    float3 hi = 1.055 * pow(max(c, float3(0.0)), 1.0 / 2.4) - 0.055;\n",
    );
    w(out, "    return select(hi, lo, c <= 0.0031308);\n");
    w(out, "}\n\n");
}

// ── Vertex input struct ──
// One attribute per semantic declaration on an input register. Width is always
// float4 in MSL; Metal fills missing components from the attribute format.

fn emit_vertex_in(out: &mut String, vs: &DxsoProgram, provided_mask: u16) {
    w(out, "struct VertexIn {\n");
    for decl in &vs.declarations {
        if let Declaration::Semantic { reg, .. } = decl
            && reg.kind == RegKind::Input
        {
            // Only declare attributes the vertex declaration provides; an omitted
            // input is read as float4(0) in the body, so there is no descriptor
            // slot to back it.
            if reg.index >= 16 || (provided_mask & (1u16 << reg.index)) != 0 {
                let _ = writeln!(
                    out,
                    "    float4 v{idx} [[attribute({idx})]];",
                    idx = reg.index
                );
            }
        }
    }
    w(out, "};\n\n");
}

// ── Varyings ──
// Always emits a fixed shape. Unwritten fields have undefined value in the VS
// return; Metal's pipeline validation doesn't complain, and unused inputs get
// dead-code-eliminated by the MSL compiler.

fn emit_varyings(out: &mut String, flat: bool) {
    w(out, "struct Varyings {\n");
    // `invariant` — the analog of an `Invariant` decoration on a SPIR-V
    // `gl_Position` output — keeps the clip-space position bit-stable WITHIN a
    // single shader across draws (with `setPreserveInvariance(true)` at VS
    // compile time). It does not make the FF and programmable pipelines agree
    // bit-for-bit; see the `Dp3`/`Dp4` emission for why cross-shader
    // bit-invariance is not the goal.
    w(out, "    float4 position [[position, invariant]];\n");
    // Secondary POSITION semantic (`dcl_positionN`, N>=1) as an interpolated
    // user varying — a VS may output it and a PS read it, distinct from the
    // clip-space `[[position]]`. Bare field (no `[[user(N)]]`), so it MUST stay
    // at the same declaration index in `ff::emit_varyings` for FF↔programmable
    // stage-in linkage.
    w(out, "    float4 position1;\n");
    for i in 0..8 {
        let _ = writeln!(out, "    float4 texcoord{i};");
    }
    // `[[flat]]` on the PS input takes diffuse/specular from the provoking
    // (first) vertex (D3DSHADE_FLAT). Only emitted on the PS struct — the
    // VS-output qualifier is ignored by Metal (the fragment-input qualifier
    // governs), and `flat` is always false for VS emission, so FF↔programmable
    // stage-in linkage by field index is preserved.
    let q = if flat { " [[flat]]" } else { "" };
    for i in 0..2 {
        let _ = writeln!(out, "    float4 color{i}{q};");
    }
    // Fog varying — must match `ff::emit_varyings` so FF / programmable
    // pipelines are swappable at the VS↔PS boundary.
    w(out, "    float4 fog;\n");
    // NDC depth for the table-fog Z source — see `ff::emit_varyings` for why
    // this is not `in.position.z` (Metal folds the rasterizer depth bias into
    // the fragment `[[position]]`).
    w(out, "    float fog_z [[center_no_perspective]];\n");
    // VS point-size output for point-sprite primitives. `[[point_size]]`
    // is a VS-only attribute Metal consumes during rasterization for
    // point primitives; it's ignored on triangles and not propagated
    // into fragment input. Always-present so VS and PS struct layouts
    // stay aligned (FF + programmable mix-and-match).
    w(out, "    float point_size [[point_size]];\n");
    w(out, "};\n\n");
}

// ── VS function ──

fn emit_vs_function(
    out: &mut String,
    vs: &DxsoProgram,
    entry: &str,
    provided_mask: u16,
) -> Result<(), EmitError> {
    let _ = writeln!(out, "vertex Varyings {entry}(");
    w(out, "    VertexIn in [[stage_in]],\n");
    w(out, "    constant float4 *vs_c [[buffer(15)]]");
    // Half-pixel rasterization fixup uniform (VS buffer 13): `(1/vp_w,
    // -1/vp_h, 0, 0)`, supplied per-draw by the encoder from the live
    // viewport. Read by the position epilogue below. Buffer 13 is free on the
    // VS side (app float constants live in vs_c/buffer 15, int constants in
    // vs_i/buffer 14, the vertex stream in buffer 0), so it never collides
    // with an app-uploaded constant — D3D9 float consts are 0..255 in vs_c.
    w(out, ",\n    constant float4 &pos_fixup [[buffer(13)]]");
    // A dynamic integer constant (typically a `loop aL, iN` / `rep iN` counter
    // fed by SetVertexShaderConstantI) reads the runtime int4 buffer at slot 14.
    if vs.uses_dynamic_int_constants() {
        w(out, ",\n    constant int4 *vs_i [[buffer(14)]]");
    }
    w(out, "\n) {\n");
    w(out, "    float4 r[32];\n");
    // D3D9 address register `a0`. SM2 has exactly one int4 a0; the emitter
    // allocates one int4 local so `mova` / `c[a0.x + N]` / reading `a0`
    // directly all go through the same identifier. Zero-init so early
    // reads before any `mova` produce a defined value.
    w(out, "    int4 a = int4(0);\n");
    // SM3 predicate register p0 — written by `setp_<cmp>`, read by
    // predicated instructions and `breakp`. Default false so reads
    // before any setp produce a defined value (predicated writes are
    // skipped, predicated reads return 0).
    w(out, "    bool4 p0 = bool4(false);\n");
    w(out, "    Varyings out;\n");
    w(out, "    out.position = float4(0.0);\n");
    // Secondary POSITION varying — defined default so a `dcl_position1` read
    // by a paired PS is never garbage when the VS leaves it unwritten.
    w(out, "    out.position1 = float4(0.0);\n");
    // D3D9 default vertex colours when the VS omits oD0/oD1: diffuse = opaque
    // white, specular = black. A shader that writes
    // oD0/oD1 overwrites these via register_write_target, so any shader emitting
    // diffuse/specular is unchanged.
    w(out, "    out.color0 = float4(1.0);\n");
    w(out, "    out.color1 = float4(0.0);\n");
    // Default-initialize fog to 1.0 (unfogged) so shaders that never write
    // oFog pair safely with the FF PS fog-blend (variant.fog_mode != 0
    // would read garbage otherwise). Writes to oFog land here too, via
    // register_write_target's RastOut 1 → "out.fog" mapping.
    w(out, "    out.fog = float4(1.0);\n");
    // VS oPts / `dcl_psize` writes route through `_psize_storage` so
    // `store_dst`'s write_mask path applies cleanly even though the
    // Varyings field itself is scalar. Default 1.0 mirrors D3D9 spec
    // for point primitives without an explicit size; extracted to
    // `out.point_size` at return.
    w(out, "    float4 _psize_storage = float4(1.0);\n");
    // Discard sink for RastOut indices we don't route to a varying.
    // Writing to this local instead of `out.position` keeps the
    // position computation from being clobbered on the following line.
    w(out, "    float4 _rastout_discard = float4(0.0);\n");

    for def in &vs.def_constants {
        let v = def.value;
        let _ = writeln!(
            out,
            "    float4 c{idx} = float4({x}, {y}, {z}, {w});",
            idx = def.reg.index,
            x = fmt_float(v[0]),
            y = fmt_float(v[1]),
            z = fmt_float(v[2]),
            w = fmt_float(v[3])
        );
    }
    for def in &vs.def_int_constants {
        let _ = writeln!(
            out,
            "    int4 i{idx} = int4({}, {}, {}, {});",
            def.value[0],
            def.value[1],
            def.value[2],
            def.value[3],
            idx = def.reg.index
        );
    }

    let def_consts: BTreeSet<u16> = vs.def_constants.iter().map(|d| d.reg.index).collect();
    let def_int_consts: BTreeSet<u16> = vs.def_int_constants.iter().map(|d| d.reg.index).collect();
    let vs_output_map = build_vs_output_map(vs);
    let subs = (!vs.subroutines.is_empty()).then_some(&vs.subroutines);
    let ctx = EmitContext::vs(
        vs.major,
        vs.minor,
        &def_consts,
        &def_int_consts,
        vs_output_map.as_ref(),
        subs,
        provided_mask,
    );
    for inst in &vs.instructions {
        translate_instruction(out, inst, &ctx)?;
    }

    // SM1/SM2 clamp the vertex colour outputs (oD0/oD1) to [0,1] before
    // interpolation; SM3 does not clamp them (per the D3D9 shader-model
    // rules). Apply at the epilogue so per-instruction `out.color0 = …`
    // writes stay verbatim.
    if vs.major < 3 {
        w(out, "    out.color0 = saturate(out.color0);\n");
        w(out, "    out.color1 = saturate(out.color1);\n");
    }
    // D3D9 vertex fog with a programmable VS that never writes oFog takes the
    // per-vertex fog factor from the OUTPUT specular alpha instead. Only
    // statically fog-writing shaders keep the prologue's unfogged 1.0 default
    // (which then also covers a dynamically-skipped predicated oFog write).
    // `out.color1` defaults to 0.0, so a shader writing neither oFog nor oD1
    // is fully fogged — matching the D3D9 zero specular-alpha default.
    if !vs_writes_fog(vs, vs_output_map.as_ref()) {
        w(out, "    out.fog = float4(out.color1.w);\n");
    }
    // Half-pixel rasterization fixup (see the buffer-13 `pos_fixup` arg): shift
    // the clip-space position half a pixel right (+x) and down (−y in Metal's
    // +y-up NDC) so on-boundary geometry matches the D3D9 reference.
    // `pos_fixup.xy = (1/vp_w, -1/vp_h)`; scaled by `.w` so the offset
    // survives the perspective divide. Applied after every instruction so
    // per-op `oPos` writes stay verbatim.
    w(out, "    out.position.x += pos_fixup.x * out.position.w;\n");
    w(out, "    out.position.y += pos_fixup.y * out.position.w;\n");
    // NDC depth for the table-fog Z source (see the Varyings decl).
    w(out, "    out.fog_z = out.position.z / out.position.w;\n");
    w(out, "    out.point_size = _psize_storage.x;\n");
    w(out, "    return out;\n");
    w(out, "}\n");
    Ok(())
}

/// SM3 VS routes every output through a `dcl_<usage> <reg>` declaration.
///
/// The dcl carries the semantic regardless of which output-flavor register
/// kind the HLSL compiler picked. Some compilers ship SM3 outputs as
/// `RegKind::TexcoordOut` (`D3DSPR_TEXCRDOUT`, type 6, aliased as
/// `D3DSPR_OUTPUT` in SM3); others emit `RegKind::Output` (type 11),
/// `RegKind::RastOut`, or `RegKind::AttrOut`. Walk all of them and key
/// on `(kind, index)` so SM3-aware lookup overrides the SM2
/// register-kind defaults in `register_write_target`.
fn build_vs_output_map(vs: &DxsoProgram) -> Option<BTreeMap<(RegKind, u16), String>> {
    if vs.major != 3 {
        return None;
    }
    let mut map: BTreeMap<(RegKind, u16), String> = BTreeMap::new();
    for decl in &vs.declarations {
        if let Declaration::Semantic {
            usage,
            usage_index,
            reg,
        } = decl
            && is_output_reg_kind(reg.kind)
        {
            let target = match usage {
                // POSITION0 is clip space ([[position]]); POSITION1+ are
                // interpolated user varyings (out.position1, …).
                DeclUsage::Position if *usage_index == 0 => "out.position".to_string(),
                DeclUsage::Position => format!("out.position{usage_index}"),
                DeclUsage::Fog => "out.fog".to_string(),
                DeclUsage::Color => format!("out.color{usage_index}"),
                DeclUsage::Texcoord => format!("out.texcoord{usage_index}"),
                // SM3 `dcl_psize oN` — point-size output. Same
                // float4-storage indirection as SM2 `oPts` so the
                // write_mask path stays uniform.
                DeclUsage::PSize => "_psize_storage".to_string(),
                other => {
                    mtld3d_shared::log_once_warn_by!(
                        target: super::LOG_TARGET,
                        key: u64::from(*usage_index),
                        "dxso: VS3 output usage {other:?}{usage_index} unmapped → write sunk"
                    );
                    "_rastout_discard".to_string()
                }
            };
            map.insert((reg.kind, reg.index), target);
        }
    }
    Some(map)
}

const fn is_output_reg_kind(kind: RegKind) -> bool {
    matches!(
        kind,
        RegKind::RastOut | RegKind::AttrOut | RegKind::TexcoordOut | RegKind::Output
    )
}

/// Whether any instruction (main body or subroutine) writes the fog output.
///
/// `oFog` (`RastOut` index 1) in SM1/SM2, or an output register `dcl`-ed with
/// `DeclUsage::Fog` in SM3. Mirrors `register_write_target`'s resolution: an
/// SM3 output write without a matching dcl is sunk, so it does not count.
fn vs_writes_fog(vs: &DxsoProgram, output_map: Option<&BTreeMap<(RegKind, u16), String>>) -> bool {
    vs.instructions
        .iter()
        .chain(vs.subroutines.values().flatten())
        .filter_map(|inst| inst.dst.as_ref())
        .any(|dst| {
            let reg = &dst.reg;
            if let Some(map) = output_map {
                return map
                    .get(&(reg.kind, reg.index))
                    .is_some_and(|target| target == "out.fog");
            }
            reg.kind == RegKind::RastOut && reg.index == 1
        })
}

// ── PS function ──

fn emit_ps_function(
    out: &mut String,
    ps: &DxsoProgram,
    variant: VariantKey,
    entry: &str,
) -> Result<(), EmitError> {
    // PS 2.0 DCL usage is structural only — the register kind fixes the
    // semantic: `RegKind::Input` is v0..vN (COLOR0..COLORN per D3D9 SM2),
    // `RegKind::Addr` is t0..tN (read via `in.texcoord{index}` in
    // `register_read_expr`, no map entry needed).
    //
    // PS 3.0 unifies inputs under `RegKind::Input` and the semantic comes
    // from the matching `dcl_<usage><index> vN` — the varying slot is the
    // declared `usage_index`, not `reg.index`. Walk the dcls and pick
    // color/texcoord based on the declared usage when major == 3.
    let mut ps_input_map: BTreeMap<u16, String> = BTreeMap::new();
    let mut samplers: BTreeMap<u16, TextureType> = BTreeMap::new();
    for decl in &ps.declarations {
        match decl {
            Declaration::Semantic {
                reg,
                usage,
                usage_index,
            } => {
                if reg.kind == RegKind::Input {
                    let mapped = if ps.major == 3 {
                        match usage {
                            DeclUsage::Color => format!("color{usage_index}"),
                            DeclUsage::Texcoord => format!("texcoord{usage_index}"),
                            // Fog gets its own varying that mirrors VS3
                            // `dcl_fog oN` writes; the FF PS expects the
                            // same `in.fog.x` channel so links stay clean.
                            DeclUsage::Fog => "fog".to_string(),
                            // PS3 `dcl_position0 vN` is the varying-syntax form
                            // of vPos — clip-space position from VS post-
                            // rasterizer becomes screen-space pixel coords on
                            // read via the [[position]] field. POSITION1+ read
                            // the matching interpolated user varying instead.
                            DeclUsage::Position if *usage_index == 0 => "position".to_string(),
                            DeclUsage::Position => format!("position{usage_index}"),
                            other => {
                                mtld3d_shared::log_once_warn_by!(
                                    target: super::LOG_TARGET,
                                    key: u64::from(*usage_index),
                                    "dxso: PS3 input usage {other:?}{usage_index} unmapped → reads return color{usage_index}"
                                );
                                format!("color{usage_index}")
                            }
                        }
                    } else {
                        format!("color{}", reg.index)
                    };
                    ps_input_map.insert(reg.index, mapped);
                }
            }
            Declaration::Sampler { texture_type, reg } => {
                samplers.insert(reg.index, *texture_type);
            }
        }
    }

    // SM1 has no `dcl_<dim> sN`: the sampler bound to a stage is implicit
    // (stage N → sampler N). Synthesize a 2D sampler entry for every stage a
    // sampling op references (the dst register number) so the texture/sampler
    // function params + coord swizzle get emitted. Dimensionality isn't
    // encoded in SM1 bytecode; 2D matches every conformance case (cube
    // environment mapping via texm3x3spec/vspec is best-effort).
    if ps.major == 1 {
        for inst in &ps.instructions {
            if sm1_op_samples(inst.opcode)
                && let Some(d) = inst.dst.as_ref()
            {
                samplers
                    .entry(d.reg.index)
                    .or_insert(TextureType::Texture2D);
            }
        }
    }

    // SM3 PS dedicated registers: `MiscType` reg 0 = vPos (screen-space
    // pixel coord), reg 1 = vFace (front-facing flag). Identified by
    // register index, not the dcl usage — the D3D9 spec allows the
    // dcl token's usage to be Position/Face but the index is the
    // canonical disambiguator.
    let mut has_vpos = false;
    let mut has_vface = false;
    for decl in &ps.declarations {
        if let Declaration::Semantic { reg, .. } = decl
            && reg.kind == RegKind::MiscType
        {
            match reg.index {
                0 => has_vpos = true,
                1 => has_vface = true,
                other => {
                    mtld3d_shared::log_once_warn_by!(
                        target: super::LOG_TARGET,
                        key: u64::from(other),
                        "dxso: PS3 MiscType reg {other} unrecognized (only 0=vPos, 1=vFace) → reads return zero"
                    );
                }
            }
        }
    }
    // SM2/SM3 oDepth: the PS function must return a struct binding
    // both `oC0 [[color(0)]]` and `oDepth [[depth(any)]]` instead of
    // a bare float4. Pre-scan so we only pay the struct cost when the
    // shader actually writes oDepth — most don't.
    let has_depth_out = ps.instructions.iter().any(|i| {
        // Explicit `oDepth` write, or an SM1 op that writes fragment depth
        // as a side effect (`texdepth`, `texm3x2depth`) — both route through
        // the `_depth_storage` / `PsOut` path below.
        matches!(i.opcode, Opcode::TexDepth | Opcode::TexM3x2Depth)
            || i.dst
                .as_ref()
                .is_some_and(|d| d.reg.kind == RegKind::DepthOut)
    });
    if has_depth_out {
        w(out, "struct PsOut {\n");
        w(out, "    float4 oC0 [[color(0)]];\n");
        w(out, "    float oDepth [[depth(any)]];\n");
        w(out, "};\n\n");
    }

    // texbem/texbeml/bem read the per-stage bump-environment matrix +
    // luminance scale/offset from a dedicated PS uniform (buffer 12), packed
    // d3d9-side from the texture-stage state. Emitted only when such an op is
    // present so non-bump shaders carry no extra binding.
    let has_bump_env = ps
        .instructions
        .iter()
        .any(|i| matches!(i.opcode, Opcode::TexBem | Opcode::TexBemL | Opcode::Bem));

    if has_depth_out {
        let _ = writeln!(out, "fragment PsOut {entry}(");
    } else {
        let _ = writeln!(out, "fragment float4 {entry}(");
    }
    w(out, "    Varyings in [[stage_in]],\n");
    w(out, "    constant float4 *ps_c [[buffer(15)]]");
    let alpha_test_active = variant.alpha_func != 0 && variant.alpha_func != 8;
    if alpha_test_active {
        w(out, ",\n    constant float &alpha_ref [[buffer(14)]]");
    }
    if fog_blend_active(variant) {
        w(out, ",\n    constant float4 *fog_data [[buffer(13)]]");
    }
    if has_bump_env {
        w(out, ",\n    constant float4 *bump_env [[buffer(12)]]");
    }
    for (idx, ty) in &samplers {
        // Depth-format binding (sampleable shadow map): the texture
        // slot must be `depth2d<float>` even if the shader's `dcl_2d`
        // declared a regular 2D sampler — the underlying MTLTexture
        // is `Depth32Float` and binding via `texture2d<float>` would
        // fail Metal validation. Cube/3D depth maps aren't a thing
        // mtld3d routes today; if `depth_sampler_mask` is set on a
        // non-2D slot the type stays as the declared one and Metal
        // validation will surface the mismatch.
        let depth_bound = (variant.depth_sampler_mask & (1u16 << *idx)) != 0;
        let tex_ty = match ty {
            TextureType::TextureCube => "texturecube<float>",
            TextureType::Texture3D => "texture3d<float>",
            TextureType::Texture2D => {
                if depth_bound {
                    "depth2d<float>"
                } else {
                    "texture2d<float>"
                }
            }
            other @ TextureType::Unknown => {
                mtld3d_shared::log_once_warn!(target: super::LOG_TARGET,
                    "dxso: unknown sampler TextureType={other:?} → defaulting to texture2d<float>"
                );
                if depth_bound {
                    "depth2d<float>"
                } else {
                    "texture2d<float>"
                }
            }
        };
        let _ = write!(
            out,
            ",\n    {tex_ty} s{idx} [[texture({idx})]],\n    sampler samp{idx} [[sampler({idx})]]"
        );
    }
    // SM3 vPos reads the post-rasterizer screen-space pixel coord — i.e. the
    // `[[position]]` value, which the `Varyings` struct ALREADY declares as
    // `position [[position, invariant]]`, so the vPos read uses `in.position`
    // (see the MiscType read path). Declaring a SECOND `float4 v_pos
    // [[position]]` fragment arg is a duplicate-`[[position]]` MSL error that
    // fails the whole shader to compile (it then silently never renders — the
    // clear shows through), so do NOT emit one.
    if has_vface {
        // MSL gives front-facing as a bool; SM3 vFace is +1.0 / -1.0.
        // Conversion happens in the prologue so reads can use the
        // `v_face` float directly.
        w(out, ",\n    bool v_face_in [[front_facing]]");
    }
    w(out, "\n) {\n");
    w(out, "    float4 r[32];\n");
    // SM1 pixel shaders: `tN` (RegKind::Addr) is a read-write register that
    // holds the iterated texture coordinate AND receives `tex`/`texcoord`/
    // `texbem`/… results. Back it with a mutable array seeded from the
    // texcoord varyings; SM1 register read/write routes here (see
    // `register_read_expr` / `register_write_target`). SM2+ leaves `tN`
    // read-only (`in.texcoordN`) so this array isn't emitted.
    if ps.major == 1 {
        w(out, "    float4 t[8];\n");
        for i in 0..8 {
            let _ = writeln!(out, "    t[{i}] = in.texcoord{i};");
        }
    }
    if has_vface {
        w(out, "    float v_face = v_face_in ? 1.0 : -1.0;\n");
    }
    // PS rel-addr (`c[a0.x + N]`) flows through the same `load_src` path as
    // VS, which emits `ps_c[a.<comp> + N]`. Declare `a` so the emitted MSL
    // compiles even though SM2.x PS doesn't write `a0` itself — relevant
    // for SM3 PS, which does support relative const addressing.
    w(out, "    int4 a = int4(0);\n");
    // See `emit_vs_function` for the rationale — same SM3 predicate
    // register, same default.
    w(out, "    bool4 p0 = bool4(false);\n");
    w(out, "    float4 oC0 = float4(0.0);\n");
    if has_depth_out {
        // DepthOut writes go through `store_dst`, which expects a
        // float4 target so the write_mask path is uniform. Stash the
        // value in a float4 local; extract `.x` at return.
        w(out, "    float4 _depth_storage = float4(0.0);\n");
    }

    // ps_1_x clamps its float constants to [-1, 1] (the fixed-point range the
    // hardware held them in). ps_2_0+ keep full range.
    // Clamp at the definition so every `cN` read observes the clamped value.
    let clamp_ps1 = |x: f32| if ps.major == 1 { x.clamp(-1.0, 1.0) } else { x };
    for def in &ps.def_constants {
        let v = def.value;
        let _ = writeln!(
            out,
            "    float4 c{idx} = float4({x}, {y}, {z}, {w});",
            idx = def.reg.index,
            x = fmt_float(clamp_ps1(v[0])),
            y = fmt_float(clamp_ps1(v[1])),
            z = fmt_float(clamp_ps1(v[2])),
            w = fmt_float(clamp_ps1(v[3]))
        );
    }
    for def in &ps.def_int_constants {
        let _ = writeln!(
            out,
            "    int4 i{idx} = int4({}, {}, {}, {});",
            def.value[0],
            def.value[1],
            def.value[2],
            def.value[3],
            idx = def.reg.index
        );
    }

    let def_consts: BTreeSet<u16> = ps.def_constants.iter().map(|d| d.reg.index).collect();
    let def_int_consts: BTreeSet<u16> = ps.def_int_constants.iter().map(|d| d.reg.index).collect();
    let subs = (!ps.subroutines.is_empty()).then_some(&ps.subroutines);
    let ctx = EmitContext::ps(&PsInit {
        major: ps.major,
        minor: ps.minor,
        map: &ps_input_map,
        samplers: &samplers,
        depth_sampler_mask: variant.depth_sampler_mask,
        depth_fetch_mask: variant.depth_fetch_mask,
        tt_projected_mask: variant.tt_projected_mask,
        def_consts: &def_consts,
        def_int_consts: &def_int_consts,
        has_vpos,
        has_vface,
        has_depth_out,
        subroutines: subs,
    });
    for inst in &ps.instructions {
        translate_instruction(out, inst, &ctx)?;
    }

    // SM1 pixel shaders have no explicit colour-output register: the final
    // pixel colour is whatever the shader left in `r0` (`D3DSPR_COLOROUT` was
    // introduced in ps_2_0). SM2+ writes `oC0` directly, so this bridge is
    // SM1-only. Emitted before alpha-test/fog because both operate on `oC0`.
    if ps.major == 1 {
        w(out, "    oC0 = r[0];\n");
    }

    // Alpha test uses a dedicated slot-14 constant buffer so it doesn't
    // collide with the user shader's own PS constants on slot 15.
    emit_alpha_test(out, variant.alpha_func);
    // Fog blend mirrors the FF PS path in `ff.rs::emit_ps` — shared helper,
    // shared slot-13 `fog_data` invariant. For vertex fog the VS-side oFog
    // default/fallback (see `emit_vs_function`) supplies `in.fog.x`.
    write_fog_blend(out, variant, "oC0");
    // D3DRS_SRGBWRITEENABLE: in-shader linear→sRGB encode on the final colour,
    // after fog/specular/alpha-test, alpha left linear. See the helper doc.
    if variant.flags.contains(VariantFlags::SRGB_WRITE) {
        w(out, "    oC0.rgb = mtld3d_linear_to_srgb(oC0.rgb);\n");
    }
    if has_depth_out {
        w(out, "    PsOut _ps_out;\n");
        w(out, "    _ps_out.oC0 = oC0;\n");
        w(out, "    _ps_out.oDepth = _depth_storage.x;\n");
        w(out, "    return _ps_out;\n");
    } else {
        w(out, "    return oC0;\n");
    }
    w(out, "}\n");
    Ok(())
}

/// Whether this variant blends fog at the end of the PS.
///
/// Either vertex fog (interpolated `in.fog.x`) or per-pixel table fog. Gates
/// the `fog_data` buffer-13 binding in both the FF and programmable PS
/// emitters.
pub const fn fog_blend_active(variant: VariantKey) -> bool {
    variant.fog_mode != 0 || variant.fog_table_mode != 0
}

/// Emit the end-of-PS fog blend into `target` (an lvalue such as `oC0` or `current`).
///
/// Shared by the FF (`ff.rs::emit_ps`) and programmable PS emitters so both
/// read the same slot-13 `fog_data` invariant: `fog_data[0]` = fog colour
/// RGBA, `fog_data[1]` = (start, end, density, depth-bias).
///
/// Vertex fog blends with the VS-interpolated factor `in.fog.x`. Table fog
/// (`fog_table_mode != 0`) computes the factor per-pixel: the source is the
/// unbiased NDC depth `in.fog_z` plus the RAW `D3DRS_DEPTHBIAS` (real
/// hardware fogs the post-bias fragment depth) for an orthographic
/// projection, or the eye W `1/in.position.w` for a perspective one
/// (`fog_source_w`). The Z source is
/// the dedicated `fog_z` varying, NOT `in.position.z`: Metal folds the
/// encoder `setDepthBias` into the fragment `[[position]]` depth scaled to
/// float-buffer ulps (2^exponent(z)-sized steps), which is neither the raw
/// D3D bias nor absent. `precise::exp` keeps EXP/EXP2 inside the test's
/// 48-ulp window regardless of the compiler's fast-math mode. LINEAR shares
/// the VS path's zero-range rule: start == end is fully fogged (f = 0), and
/// the signed denominator keeps reversed fog (start > end) working.
pub fn write_fog_blend(out: &mut String, variant: VariantKey, target: &str) {
    if !fog_blend_active(variant) {
        return;
    }
    if variant.fog_table_mode != 0 {
        if variant.flags.contains(VariantFlags::FOG_SOURCE_W) {
            w(out, "    float fog_c = 1.0 / in.position.w;\n");
        } else {
            w(out, "    float fog_c = in.fog_z + fog_data[1].w;\n");
        }
        match variant.fog_table_mode {
            1 => {
                // D3DFOG_EXP: f = 1 / e^(d·density)
                w(
                    out,
                    "    float fog_f = saturate(precise::exp(-fog_data[1].z * fog_c));\n",
                );
            }
            2 => {
                // D3DFOG_EXP2: f = 1 / e^((d·density)²)
                w(out, "    float fog_dz = fog_data[1].z * fog_c;\n");
                w(
                    out,
                    "    float fog_f = saturate(precise::exp(-fog_dz * fog_dz));\n",
                );
            }
            _ => {
                // D3DFOG_LINEAR: f = (end - d) / (end - start)
                w(
                    out,
                    "    float fog_range = fog_data[1].y - fog_data[1].x;\n",
                );
                w(
                    out,
                    "    float fog_f = fog_range == 0.0 ? 0.0 : saturate((fog_data[1].y - fog_c) / fog_range);\n",
                );
            }
        }
        let _ = writeln!(
            out,
            "    {target} = float4(mix(fog_data[0].rgb, {target}.rgb, fog_f), {target}.a);"
        );
    } else {
        let _ = writeln!(
            out,
            "    {target} = float4(mix(fog_data[0].rgb, {target}.rgb, saturate(in.fog.x)), {target}.a);"
        );
    }
}

/// Append a `discard_fragment()` to the PS body for all alpha-test functions other than ALWAYS.
///
/// The reference value is read from a dedicated `alpha_ref` buffer on slot 14 —
/// see `FF_ALPHA_REF_BUFFER_SLOT` in `windows/d3d9/src/device.rs`.
fn emit_alpha_test(out: &mut String, alpha_func: u8) {
    // 0 and D3DCMP_ALWAYS (=8) both mean "no discard".
    if alpha_func == 0 || alpha_func == 8 {
        return;
    }
    let cmp = match alpha_func {
        1 => "false",              // D3DCMP_NEVER
        2 => "oC0.a < alpha_ref",  // D3DCMP_LESS
        3 => "oC0.a == alpha_ref", // D3DCMP_EQUAL
        4 => "oC0.a <= alpha_ref", // D3DCMP_LESSEQUAL
        5 => "oC0.a > alpha_ref",  // D3DCMP_GREATER
        6 => "oC0.a != alpha_ref", // D3DCMP_NOTEQUAL
        7 => "oC0.a >= alpha_ref", // D3DCMP_GREATEREQUAL
        other => {
            mtld3d_shared::log_once_warn!(target: super::LOG_TARGET, "dxso: alpha_func unhandled={other} → always-pass");
            "true"
        }
    };
    let _ = writeln!(out, "    if (!({cmp})) discard_fragment();");
}

// ── Emit context ──
// Tells the register-expression helpers whether we're in a VS or PS and how
// to map PS input registers to varying fields.

bitflags::bitflags! {
    /// Boolean predicates carried in `EmitContext`.
    ///
    /// Ephemeral per-compile state — never serialised, never hashed.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct EmitContextFlags: u8 {
        /// Stage discriminator: set for vertex shader emission, clear for pixel shader.
        ///
        /// Drives register-expression helpers' VS / PS branching.
        const IS_VERTEX = 1 << 0;
        /// PS only: SM3 vPos register (screen-space pixel coord) is declared and read.
        const PS_HAS_VPOS = 1 << 1;
        /// PS only: SM3 vFace register (front-facing as ±1.0) is declared and read.
        const PS_HAS_VFACE = 1 << 2;
        /// PS only: shader writes `oDepth` somewhere.
        ///
        /// Set → return type is the `PsOut` struct, `DepthOut` writes route to
        /// `_depth_storage`. Clear → `DepthOut` writes warn-and-sink (legacy
        /// SM2 path).
        const PS_HAS_DEPTH_OUT = 1 << 3;
    }
}

struct EmitContext<'a> {
    flags: EmitContextFlags,
    /// Shader-model major.minor from the program header.
    ///
    /// Texkill emit branches on this: SM2+/SM1.4 take `inst.dst` at face
    /// value and honor its `write_mask`; SM1.0-1.3 reads the matching
    /// texcoord input by index.
    shader_major: u8,
    shader_minor: u8,
    ps_input_map: Option<&'a BTreeMap<u16, String>>,
    /// PS only: per-sampler texture type from the `dcl_*` declarations.
    ///
    /// Picks the coord-swizzle for `texld` (`.xy` for 2D, `.xyz` for
    /// cube and 3D). VS leaves this `None`.
    ps_samplers: Option<&'a BTreeMap<u16, TextureType>>,
    /// PS only: bit `i` set ⇒ sampler slot `i` is bound to a depth-format texture.
    ///
    /// So `s{i}.sample(...)` returns a scalar `float` (not `float4`) and the
    /// binding must be `depth2d<float>` rather than `texture2d<float>`. The
    /// sample call sites wrap the result in `float4(...)` to keep downstream
    /// `.x/.y/.z/.w` reads typed.
    ps_depth_sampler_mask: u16,
    /// PS only: subset of `ps_depth_sampler_mask` that are "readable raw depth" FOURCC textures.
    ///
    /// INTZ/DF24/DF16: for these the sample site emits a plain `.sample()`
    /// (raw stored depth) instead of `sample_compare`.
    ps_depth_fetch_mask: u16,
    /// PS only: bit `i` set ⇒ stage `i` has `D3DTTFF_PROJECTED`.
    ///
    /// The SM1 (`ps_1_0`..`ps_1_3`) `tex`/`texld` emit divides the texcoord
    /// by its `.w` before sampling. See `VariantKey::tt_projected_mask`.
    ps_tt_projected_mask: u8,
    /// Stack of open loop scopes.
    ///
    /// Each entry is `Some(<aL_index>)` for a `loop aL, iN` block (the index
    /// identifies which `aL_<n>` local was declared) or `None` for a `rep iN`
    /// block (no aL allocated). Pushed by Loop / Rep, popped by `EndLoop` /
    /// `EndRep`, scanned by `RegKind::Loop` reads to find the most recent
    /// enclosing `Some` entry. `RefCell` to keep `EmitContext` immutable
    /// from `translate_instruction`'s POV.
    loop_stack: RefCell<Vec<Option<usize>>>,
    /// Number of `loop` blocks emitted so far.
    ///
    /// Each open allocates a fresh `aL_<count>` local so nested `loop` blocks
    /// don't collide. Monotonic — popped frames don't reuse names.
    loop_al_count: RefCell<usize>,
    /// Parsed `Label N` / `Ret` blocks keyed by label id.
    ///
    /// `Call sN` and `CallNz` look up the body and inline-expand it. None for
    /// shaders without subroutines (subroutines map empty); the borrow saves
    /// a pointer-chase through `DxsoProgram` during instruction translation.
    subroutines: Option<&'a std::collections::BTreeMap<u32, Vec<Instruction>>>,
    /// Stack of currently-expanding label ids.
    ///
    /// `Call sN` pushes on entry, pops on return. Cycle detection: a recursive
    /// call (label already on the stack) emits a once-per-label warn and skips
    /// the expansion rather than blowing the host stack.
    expansion_stack: RefCell<Vec<u32>>,
    /// SM3 VS only: maps `(reg.kind, reg.index)` → `out.<semantic>` MSL field.
    ///
    /// Populated from the `dcl_*` declarations. SM2 VS uses register-kind
    /// dispatch in `register_write_target` and leaves this `None`. Keying on
    /// the kind too is load-bearing — different HLSL backends emit SM3 outputs
    /// as `TexcoordOut`, `AttrOut`, `RastOut`, or Output, sometimes with
    /// overlapping indices.
    vs_output_map: Option<&'a BTreeMap<(RegKind, u16), String>>,
    /// Const-register indices defined by a `def` instruction.
    ///
    /// They exist as local variables `cN` — reads for these registers reference
    /// the local instead of the runtime buffer.
    def_consts: &'a BTreeSet<u16>,
    /// Integer-const indices defined by a `defi` instruction (baked into MSL as `int4 iN` locals).
    ///
    /// A `ConstInt` read NOT in this set is a *dynamic* integer constant: in a
    /// VS it reads the runtime `vs_i` buffer (slot 14).
    def_int_consts: &'a BTreeSet<u16>,
    /// VS only: bit `i` set ⇒ input register `vi` is provided by the vertex declaration.
    ///
    /// A clear bit means the declaration omits that attribute, so the VS reads
    /// `float4(0)` for it and `VertexIn` does not declare it. All-ones for PS
    /// and for fully-provided decls.
    vs_provided_mask: u16,
}

/// Init bundle for `EmitContext::ps`.
///
/// The PS construction shape doesn't fit clippy's argument-count threshold,
/// and the fields are exactly the PS-specific subset of `EmitContext`.
struct PsInit<'a> {
    major: u8,
    minor: u8,
    map: &'a BTreeMap<u16, String>,
    samplers: &'a BTreeMap<u16, TextureType>,
    depth_sampler_mask: u16,
    depth_fetch_mask: u16,
    tt_projected_mask: u8,
    def_consts: &'a BTreeSet<u16>,
    def_int_consts: &'a BTreeSet<u16>,
    has_vpos: bool,
    has_vface: bool,
    has_depth_out: bool,
    subroutines: Option<&'a std::collections::BTreeMap<u32, Vec<Instruction>>>,
}

impl<'a> EmitContext<'a> {
    const fn vs(
        major: u8,
        minor: u8,
        def_consts: &'a BTreeSet<u16>,
        def_int_consts: &'a BTreeSet<u16>,
        vs_output_map: Option<&'a BTreeMap<(RegKind, u16), String>>,
        subroutines: Option<&'a std::collections::BTreeMap<u32, Vec<Instruction>>>,
        vs_provided_mask: u16,
    ) -> Self {
        Self {
            flags: EmitContextFlags::IS_VERTEX,
            shader_major: major,
            shader_minor: minor,
            ps_input_map: None,
            ps_samplers: None,
            ps_depth_sampler_mask: 0,
            ps_depth_fetch_mask: 0,
            ps_tt_projected_mask: 0,
            vs_output_map,
            def_consts,
            def_int_consts,
            vs_provided_mask,
            loop_stack: RefCell::new(Vec::new()),
            loop_al_count: RefCell::new(0),
            subroutines,
            expansion_stack: RefCell::new(Vec::new()),
        }
    }

    fn ps(init: &PsInit<'a>) -> Self {
        let mut flags = EmitContextFlags::empty();
        flags.set(EmitContextFlags::PS_HAS_VPOS, init.has_vpos);
        flags.set(EmitContextFlags::PS_HAS_VFACE, init.has_vface);
        flags.set(EmitContextFlags::PS_HAS_DEPTH_OUT, init.has_depth_out);
        Self {
            flags,
            shader_major: init.major,
            shader_minor: init.minor,
            ps_input_map: Some(init.map),
            ps_samplers: Some(init.samplers),
            ps_depth_sampler_mask: init.depth_sampler_mask,
            ps_depth_fetch_mask: init.depth_fetch_mask,
            ps_tt_projected_mask: init.tt_projected_mask,
            vs_output_map: None,
            def_consts: init.def_consts,
            def_int_consts: init.def_int_consts,
            vs_provided_mask: u16::MAX,
            loop_stack: RefCell::new(Vec::new()),
            loop_al_count: RefCell::new(0),
            subroutines: init.subroutines,
            expansion_stack: RefCell::new(Vec::new()),
        }
    }

    #[inline]
    const fn is_vertex(&self) -> bool {
        self.flags.contains(EmitContextFlags::IS_VERTEX)
    }

    /// Whether VS input register `v{reg}` is backed by the vertex declaration.
    ///
    /// A register the declaration omits reads `float4(0)`. Registers ≥ 16 are
    /// outside the mask and treated as provided (no D3D9 VS declares them).
    #[inline]
    const fn vs_input_provided(&self, reg: u16) -> bool {
        reg >= 16 || (self.vs_provided_mask & (1u16 << reg)) != 0
    }

    #[inline]
    const fn ps_has_vpos(&self) -> bool {
        self.flags.contains(EmitContextFlags::PS_HAS_VPOS)
    }

    #[inline]
    const fn ps_has_vface(&self) -> bool {
        self.flags.contains(EmitContextFlags::PS_HAS_VFACE)
    }

    #[inline]
    const fn ps_has_depth_out(&self) -> bool {
        self.flags.contains(EmitContextFlags::PS_HAS_DEPTH_OUT)
    }

    /// True when sampler slot `idx` is bound to a depth-format texture.
    const fn is_depth_sampler(&self, idx: u16) -> bool {
        idx < 16 && (self.ps_depth_sampler_mask & (1u16 << idx)) != 0
    }

    /// True when sampler slot `idx` is a "readable raw depth" FOURCC texture (INTZ/DF24/DF16).
    ///
    /// Sample the RAW stored depth, not a shadow comparison.
    const fn is_raw_depth_sampler(&self, idx: u16) -> bool {
        idx < 16 && (self.ps_depth_fetch_mask & (1u16 << idx)) != 0
    }

    /// True when texture stage `idx` has `D3DTTFF_PROJECTED`.
    ///
    /// The SM1 emitter should apply the implicit per-pixel projective divide
    /// before sampling.
    fn is_tt_projected(&self, idx: u16) -> bool {
        idx < 8
            && (self.ps_tt_projected_mask & (1u8 << idx)) != 0
            && !matches!(
                self.ps_samplers.and_then(|m| m.get(&idx)),
                Some(TextureType::TextureCube)
            )
    }

    /// True for Shader Model 1 (`vs_1_1` / `ps_1_0`..`1_4`).
    ///
    /// Gates the legacy register/texture semantics that differ from SM2+: the
    /// read-write PS `tN` register file, the `tex`/`texcoord` opcode meanings,
    /// and the plain-`mov` address-register write.
    #[inline]
    const fn is_sm1(&self) -> bool {
        self.shader_major == 1
    }
}

// ── Instruction translation ──

fn translate_instruction(
    out: &mut String,
    inst: &Instruction,
    ctx: &EmitContext,
) -> Result<(), EmitError> {
    // Sampler sources carry only a register index (consumed directly by the
    // opcode handler — e.g. `TexLd` uses `inst.srcs[1].reg.index`), never a
    // cooked data value. Skip the generic load for them so `register_read_expr`
    // stays honest (no placeholder for a register that has no readable value).
    let srcs: Vec<String> = inst
        .srcs
        .iter()
        .map(|s| {
            if s.reg.kind == RegKind::Sampler {
                Ok(String::new())
            } else {
                load_src(s, ctx)
            }
        })
        .collect::<Result<_, _>>()?;

    let expr = match inst.opcode {
        Opcode::Mov => srcs[0].clone(),
        Opcode::Add => format!("({} + {})", srcs[0], srcs[1]),
        Opcode::Sub => format!("({} - {})", srcs[0], srcs[1]),
        Opcode::Mul => format!("({} * {})", srcs[0], srcs[1]),
        Opcode::Mad => format!("fma({}, {}, {})", srcs[0], srcs[1], srcs[2]),
        // Plain MSL `dot()` — Apple Silicon has hardware dot-product;
        // let the compiler use it. Cross-shader bit-invariance is not
        // achievable in general (per-shader matrices and vertex inputs
        // genuinely differ between FF and programmable paths), so the
        // implicit decal depth bias in `windows/d3d9/src/draw.rs`
        // handles the visible symptom. `[[position, invariant]]` +
        // `setPreserveInvariance(true)` + `setMathMode(Safe)` on VS
        // remain in place as cheap defense — they keep clip-position
        // bit-stable WITHIN a single shader across frames, which
        // helps reflection passes and the like.
        Opcode::Dp3 => format!("float4(dot(({}).xyz, ({}).xyz))", srcs[0], srcs[1]),
        Opcode::Dp4 => format!("float4(dot({}, {}))", srcs[0], srcs[1]),
        Opcode::Rcp => format!("float4(1.0 / ({}).x)", srcs[0]),
        Opcode::Rsq => format!("float4(rsqrt(abs(({}).x)))", srcs[0]),
        Opcode::Min => format!("min({}, {})", srcs[0], srcs[1]),
        Opcode::Max => format!("max({}, {})", srcs[0], srcs[1]),
        Opcode::Frc => format!("fract({})", srcs[0]),
        // D3D9: lrp dst, s0, s1, s2 → dst = s0*s1 + (1-s0)*s2 = mix(s2, s1, s0)
        Opcode::Lrp => format!("mix({}, {}, {})", srcs[2], srcs[1], srcs[0]),
        // D3D9 `nrm` scales EVERY written component (incl. w) by
        // 1/length(src.xyz): dst = src * rsqrt(dot(src.xyz, src.xyz)). The
        // write-mask is applied by store_dst (so `nrm r.xyz` leaves w intact).
        Opcode::Nrm => format!("(({s}) * rsqrt(dot(({s}).xyz, ({s}).xyz)))", s = srcs[0]),
        Opcode::Abs => format!("abs({})", srcs[0]),
        // D3D9: pow(base, exp) = base <= 0 ? 0 : pow(base, exp); uses abs(base).
        Opcode::Pow => format!("float4(pow(abs(({}).x), ({}).x))", srcs[0], srcs[1]),
        // D3D9 ExpP/LogP are the partial-precision twins of Exp/Log
        // (PS 1.x targets that lacked full fp32 in the ALU). Modern
        // hardware runs both at full precision, so the lowering is
        // identical to Exp/Log.
        Opcode::Exp | Opcode::ExpP => format!("float4(exp2(({}).x))", srcs[0]),
        Opcode::Log | Opcode::LogP => format!("float4(log2(abs(({}).x)))", srcs[0]),
        Opcode::Crs => format!("float4(cross(({}).xyz, ({}).xyz), 0.0)", srcs[0], srcs[1]),
        Opcode::Sgn => format!("sign({})", srcs[0]),
        // D3D9: dp2add dst, s0, s1, s2 — dst.* = (s0.x*s1.x + s0.y*s1.y) + s2.x.
        // Lowered to a 4-wide broadcast so the standard `store_dst`
        // write-mask path applies cleanly.
        Opcode::Dp2Add => format!(
            "float4(dot(({s0}).xy, ({s1}).xy) + ({s2}).x)",
            s0 = srcs[0],
            s1 = srcs[1],
            s2 = srcs[2]
        ),
        // SM3 screen-space derivatives. Metal's `dfdx` / `dfdy` match
        // D3D9 semantics: per-component partial derivative across the
        // 2x2 pixel quad.
        Opcode::Dsx => format!("dfdx({})", srcs[0]),
        Opcode::Dsy => format!("dfdy({})", srcs[0]),
        // D3D9 lit src — fixed-function lighting coefficients:
        //   dst.x = 1
        //   dst.y = max(src.x, 0)
        //   dst.z = src.x > 0 ? pow(max(src.y, 0), src.w) : 0
        //   dst.w = 1
        // The src.x > 0 gate avoids `pow(0, w)` blowing up when the
        // diffuse term is zero. `clamp` mirrors the D3D9 exponent
        // range — Metal's pow has the same well-behavedness so the
        // gate alone is enough.
        Opcode::Lit => format!(
            "float4(1.0, max(({s}).x, 0.0), \
             (({s}).x > 0.0) ? pow(max(({s}).y, 0.0), ({s}).w) : 0.0, \
             1.0)",
            s = srcs[0]
        ),
        // D3D9 dst src0, src1 — distance vector for attenuation:
        //   dst.x = 1
        //   dst.y = src0.y * src1.y
        //   dst.z = src0.z
        //   dst.w = src1.w
        Opcode::Dst => format!(
            "float4(1.0, ({s0}).y * ({s1}).y, ({s0}).z, ({s1}).w)",
            s0 = srcs[0],
            s1 = srcs[1]
        ),
        // `cnd dst, src0, src1, src2` — conditional move (D3D9 ps_1_x).
        // ps_1_4 compares every component (`src0 > 0.5 ? src1 : src2`
        // per-lane); ps_1_1..1_3 test only the scalar `.x` lane and broadcast.
        // A ps_1_1..1_3 *co-issued* `cnd` whose dst is not alpha-only bypasses
        // the compare entirely and selects src1 (per the D3D9 ps_1_1..1_3
        // co-issued-`cnd` rule).
        Opcode::Cnd => {
            let (s0, s1, s2) = (&srcs[0], &srcs[1], &srcs[2]);
            if ctx.shader_major == 1 && ctx.shader_minor >= 4 {
                format!("select({s2}, {s1}, {s0} > float4(0.5))")
            } else {
                let alpha_only = inst.dst.as_ref().is_some_and(|d| d.write_mask.0 == 0b1000);
                if inst.flags.contains(InstrFlags::COISSUE) && !alpha_only {
                    s1.clone()
                } else {
                    format!("select({s2}, {s1}, ({s0}).x > 0.5)")
                }
            }
        }
        // D3D9: `sincos dst, src` — dst.x = cos(src.x), dst.y = sin(src.x).
        // The destination write_mask filters which lanes actually land via
        // `store_dst` (typically .xy). The SM2 three-source variant carries
        // approximation-coefficient registers in srcs[1..2] which are used
        // by hardware that lacks native sincos — Metal has both, so we
        // ignore them and emit the precise builtins.
        Opcode::SinCos => format!("float4(cos(({s}).x), sin(({s}).x), 0.0, 0.0)", s = srcs[0]),
        Opcode::M4x4 => emit_matmul(&srcs[0], &inst.srcs[1], 4, 4, ctx)?,
        Opcode::M4x3 => emit_matmul(&srcs[0], &inst.srcs[1], 4, 3, ctx)?,
        Opcode::M3x4 => emit_matmul(&srcs[0], &inst.srcs[1], 3, 4, ctx)?,
        Opcode::M3x3 => emit_matmul(&srcs[0], &inst.srcs[1], 3, 3, ctx)?,
        Opcode::M3x2 => emit_matmul(&srcs[0], &inst.srcs[1], 3, 2, ctx)?,
        // D3D9: `cmp dst, s0, s1, s2` — per-component `(s0 >= 0) ? s1 : s2`.
        // MSL `select(a, b, cond)` is `cond ? b : a`, so the arg order is
        // (s2, s1, cond). The float4 comparison yields bool4 which `select`
        // applies componentwise.
        Opcode::Cmp => format!(
            "select({s2}, {s1}, {s0} >= float4(0.0))",
            s0 = srcs[0],
            s1 = srcs[1],
            s2 = srcs[2]
        ),
        // D3D9: `slt dst, s0, s1` — componentwise `(s0 < s1) ? 1 : 0`.
        // `step(edge, x)` returns `x >= edge ? 1 : 0`, so `1 - step(s1, s0)`
        // gives the strict-less-than semantics.
        Opcode::Slt => format!("(float4(1.0) - step({}, {}))", srcs[1], srcs[0]),
        // D3D9: `sge dst, s0, s1` — componentwise `(s0 >= s1) ? 1 : 0`.
        // That's exactly `step(s1, s0)`.
        Opcode::Sge => format!("step({}, {})", srcs[1], srcs[0]),
        // ps_1_0..1_3 `texcoord tN`: copy iterated texcoord set N into the
        // register as colour data, clamped to [0,1]. The set index is the dst
        // register number (no source operand).
        // ps_1_4 `texcrd rN, tM`: copy texcoord set M into rN, NOT clamped.
        Opcode::TexCoord => {
            let dst = inst.dst.as_ref().ok_or_else(|| {
                EmitError::UnsupportedInstruction("texcoord/texcrd missing dst".into())
            })?;
            if ctx.shader_minor >= 4 {
                srcs[0].clone()
            } else {
                format!("saturate(in.texcoord{})", dst.reg.index)
            }
        }
        Opcode::TexLd if ctx.is_sm1() => {
            // SM1 reuses opcode 66 with different operand semantics from SM2
            // `texld`: the sampler index is the DESTINATION register number,
            // not a separate operand.
            //   ps_1_0..1_3 `tex tN`     → sample stage N at the coord already
            //                              in tN (dst), write back to tN.
            //   ps_1_4      `texld rN,s` → sample stage N at coord `s`, write rN.
            let dst = inst.dst.as_ref().ok_or_else(|| {
                EmitError::UnsupportedInstruction("SM1 tex/texld missing dst".into())
            })?;
            let sampler_idx = dst.reg.index;
            let coord = if ctx.shader_minor >= 4 {
                // ps_1_4 projects via the explicit DZ/DW source modifiers, not
                // the texture-stage TTFF flags — never apply the implicit divide.
                srcs[0].clone()
            } else {
                // ps_1_0..1_3 read the texcoord from register tN. When the stage
                // has D3DTTFF_PROJECTED, D3D9 applies an implicit per-pixel
                // projective divide (coord ÷ its `.w`, the divisor the FF VS
                // stashed there) before sampling. A `.w` of 0 samples the
                // origin (matching the FF PS path + the reference's divisor-0
                // → black).
                let raw = register_read_expr(dst.reg, ctx)?;
                if ctx.is_tt_projected(sampler_idx) {
                    format!(
                        "((({raw}).w != 0.0) ? float4(({raw}).xyz / ({raw}).w, 1.0) \
                         : float4(0.0, 0.0, 0.0, 1.0))"
                    )
                } else {
                    raw
                }
            };
            sample_or_compare(ctx, sampler_idx, &coord, None)
        }
        Opcode::TexLd => {
            // SM2/SM3 `texld dst, coord, sampler` — srcs[1] is the sampler
            // register ref. The coord swizzle depends on the sampler's declared
            // dimensionality: 2D → `.xy` (float2), Cube/3D → `.xyz` (float3,
            // used as both a 3D coord and a cube direction vector).
            let sampler_idx = inst.srcs[1].reg.index;
            // `texldp` (D3DSI_TEXLD_PROJECT) divides the coordinate by its `.w`
            // before sampling. The divisor is always `.w` for SM2+ (unlike the
            // fixed-function projective divide whose divisor depends on the
            // texcoord count). The downstream swizzle in `sample_or_compare`
            // then takes `.xy` / `.xyz` of the already-projected coordinate.
            if inst.flags.contains(InstrFlags::TEX_PROJECTED) {
                let coord = &srcs[0];
                let projected = format!("(({coord}) / ({coord}).w)");
                sample_or_compare(ctx, sampler_idx, &projected, None)
            } else {
                sample_or_compare(ctx, sampler_idx, &srcs[0], None)
            }
        }
        // SM3 texldl — sample with explicit LOD in coord.w.
        // `s.sample(samp, coord, level(lod))` is the MSL form.
        Opcode::TexLdL => {
            let sampler_idx = inst.srcs[1].reg.index;
            let suffix = format!(", level(({coord}).w)", coord = srcs[0]);
            sample_or_compare(ctx, sampler_idx, &srcs[0], Some(&suffix))
        }
        // SM3 texldd — sample with explicit gradients in srcs[2]/srcs[3].
        // 2D samplers use `gradient2d(ddx.xy, ddy.xy)`. The swizzle
        // helper picks `.xy` for 2D and `.xyz` for 3D / cube — Metal
        // has matching `gradientcube` / `gradient3d` overloads.
        Opcode::TexLdD => {
            let sampler_idx = inst.srcs[1].reg.index;
            let gradient = match ctx.ps_samplers.and_then(|m| m.get(&sampler_idx)) {
                Some(TextureType::TextureCube) => format!(
                    "gradientcube(({ddx}).xyz, ({ddy}).xyz)",
                    ddx = srcs[2],
                    ddy = srcs[3]
                ),
                Some(TextureType::Texture3D) => format!(
                    "gradient3d(({ddx}).xyz, ({ddy}).xyz)",
                    ddx = srcs[2],
                    ddy = srcs[3]
                ),
                _ => format!(
                    "gradient2d(({ddx}).xy, ({ddy}).xy)",
                    ddx = srcs[2],
                    ddy = srcs[3]
                ),
            };
            let suffix = format!(", {gradient}");
            sample_or_compare(ctx, sampler_idx, &srcs[0], Some(&suffix))
        }
        Opcode::TexKill => {
            // D3D9 texkill encodes its single operand in DST-form (write
            // mask in token bits 16-19), not SRC-form (swizzle in bits
            // 16-23). The parser routes it through `decode_dst` because
            // `Opcode::has_destination()` lists TexKill, so `inst.srcs`
            // is empty here.
            //
            // Semantics by shader model:
            //   SM2+ / SM1.4 → read dst register directly, honor write_mask.
            //   SM1.0-1.3    → read pixel texcoord N (= dst.id.num); always .xyz.
            let dst = inst.dst.as_ref().ok_or_else(|| {
                EmitError::UnsupportedInstruction("texkill missing dst operand".into())
            })?;
            let load = if ctx.shader_major >= 2 || (ctx.shader_major == 1 && ctx.shader_minor == 4)
            {
                load_src(
                    &SrcOperand {
                        reg: dst.reg,
                        swizzle: Swizzle::IDENTITY,
                        modifier: SrcModifier::None,
                        rel_addr: None,
                    },
                    ctx,
                )?
            } else {
                format!("in.texcoord{}", dst.reg.index)
            };
            let mask = if ctx.shader_major >= 2 {
                let chars = write_mask_chars(dst.write_mask);
                if chars.is_empty() {
                    return Ok(());
                }
                chars
            } else {
                String::from("xyz")
            };
            let _ = writeln!(
                out,
                "    if (any(({load}).{mask} < 0.0)) discard_fragment();"
            );
            return Ok(());
        }
        // ── SM1 (ps_1_x) legacy texture-addressing ops ──
        // These ride the read-write `t[]` register file. The conformance
        // suite exercises texbem/texcoord/texkill; the rest are the full
        // DX8-era bump/environment-mapping family, lowered best-effort to MSL
        // (the suite does not test them). Each carries its D3D9 formula inline.

        // `texbem tN, tM` perturbs stage N's texcoord by the 2×2 bump matrix
        // applied to the signed du/dv in tM (a previously-sampled bump map),
        // then samples stage N. `texbeml` additionally modulates rgb by the
        // luminance `tM.z * lscale + loffset`. Matrix/luminance come from the
        // per-stage bump-env uniform (buffer 12).
        Opcode::TexBem | Opcode::TexBemL => {
            let dst = inst
                .dst
                .as_ref()
                .ok_or_else(|| EmitError::UnsupportedInstruction("texbem missing dst".into()))?;
            let n = dst.reg.index;
            let coord = register_read_expr(dst.reg, ctx)?;
            let bump = &srcs[0];
            let (m00, m01, m10, m11) = bump_matrix_exprs(n);
            // Base texcoord (`tN`). When stage N has D3DTTFF_PROJECTED the FF VS
            // stashed the projective divisor in `.w`, so divide `.xy` by it before
            // perturbing (a `.w` of 0 reads the origin), matching the implicit
            // divide the plain `tex`/`texld` path applies.
            let (bx, by) = if ctx.is_tt_projected(n) {
                (
                    format!("((({coord}).w != 0.0) ? ({coord}).x / ({coord}).w : 0.0)"),
                    format!("((({coord}).w != 0.0) ? ({coord}).y / ({coord}).w : 0.0)"),
                )
            } else {
                (format!("({coord}).x"), format!("({coord}).y"))
            };
            let u = format!("{bx} + {m00} * ({bump}).x + {m10} * ({bump}).y");
            let v = format!("{by} + {m01} * ({bump}).x + {m11} * ({bump}).y");
            let coord4 = format!("float4({u}, {v}, 0.0, 0.0)");
            let sampled = sample_or_compare(ctx, n, &coord4, None);
            store_dst(out, *dst, &sampled, ctx);
            if matches!(inst.opcode, Opcode::TexBemL) {
                let target = register_write_target(dst.reg, ctx);
                let (lscale, loffset) = bump_lum_exprs(n);
                let lum = format!("saturate(({bump}).z * {lscale} + {loffset})");
                let _ = writeln!(
                    out,
                    "    {target} = float4({target}.rgb * {lum}, {target}.a);"
                );
            }
            return Ok(());
        }
        // `bem rN, src0, src1` (ps_1_4) — apply the 2×2 bump matrix to src1 and
        // add src0, writing the .xy result (no sampling). Stage = dst number.
        Opcode::Bem => {
            let dst = inst
                .dst
                .as_ref()
                .ok_or_else(|| EmitError::UnsupportedInstruction("bem missing dst".into()))?;
            let (m00, m01, m10, m11) = bump_matrix_exprs(dst.reg.index);
            let x = format!(
                "({s0}).x + {m00} * ({s1}).x + {m10} * ({s1}).y",
                s0 = srcs[0],
                s1 = srcs[1]
            );
            let y = format!(
                "({s0}).y + {m01} * ({s1}).x + {m11} * ({s1}).y",
                s0 = srcs[0],
                s1 = srcs[1]
            );
            format!("float4({x}, {y}, 0.0, 0.0)")
        }
        // 3×2 / 3×3 matrix-multiply texture addressing. Each `pad` computes one
        // row (a 3-component dot of the register's iterated texcoord with the
        // source) and stashes it in the register's .x; the closing
        // `tex`/`m3x3`/`spec` reads the prior rows from the preceding
        // registers' .x and assembles the sample coordinate.
        Opcode::TexM3x2Pad | Opcode::TexM3x3Pad => {
            let dst = inst.dst.as_ref().ok_or_else(|| {
                EmitError::UnsupportedInstruction("texm3xNpad missing dst".into())
            })?;
            let coord = register_read_expr(dst.reg, ctx)?;
            format!(
                "float4(dot(({coord}).xyz, ({s}).xyz), 0.0, 0.0, 0.0)",
                s = srcs[0]
            )
        }
        // `texm3x2tex tM, src` — second row v = dot(coord_m, src); u = the pad
        // result in t[M-1].x; sample stage M at (u, v).
        Opcode::TexM3x2Tex => {
            let dst = inst.dst.as_ref().ok_or_else(|| {
                EmitError::UnsupportedInstruction("texm3x2tex missing dst".into())
            })?;
            let m = dst.reg.index;
            let coord = register_read_expr(dst.reg, ctx)?;
            let v = format!("dot(({coord}).xyz, ({s}).xyz)", s = srcs[0]);
            let u = format!("t[{}].x", m.saturating_sub(1));
            let coord4 = format!("float4({u}, {v}, 0.0, 0.0)");
            sample_or_compare(ctx, m, &coord4, None)
        }
        // `texm3x2depth tM, src` (ps_1_3) — z = pad result (t[M-1].x),
        // w = dot(coord_m, src); write fragment depth = z / w.
        Opcode::TexM3x2Depth => {
            let dst = inst.dst.as_ref().ok_or_else(|| {
                EmitError::UnsupportedInstruction("texm3x2depth missing dst".into())
            })?;
            let m = dst.reg.index;
            let coord = register_read_expr(dst.reg, ctx)?;
            let w = format!("dot(({coord}).xyz, ({s}).xyz)", s = srcs[0]);
            let z = format!("t[{}].x", m.saturating_sub(1));
            let _ = writeln!(
                out,
                "    _depth_storage = float4(({w}) != 0.0 ? saturate(({z}) / ({w})) : 0.0);"
            );
            return Ok(());
        }
        // `texm3x3 / texm3x3tex / texm3x3spec / texm3x3vspec tM, src[, eye]` —
        // assemble the 3-vector normal (u, v, w) from the two preceding pads
        // plus this row's dot. `m3x3` writes it; `tex` samples stage M with it;
        // `spec`/`vspec` reflect an eye vector about it and sample (cube map).
        Opcode::TexM3x3Tex | Opcode::TexM3x3 | Opcode::TexM3x3Spec | Opcode::TexM3x3VSpec => {
            let dst = inst
                .dst
                .as_ref()
                .ok_or_else(|| EmitError::UnsupportedInstruction("texm3x3* missing dst".into()))?;
            let m = dst.reg.index;
            let coord = register_read_expr(dst.reg, ctx)?;
            let u = format!("t[{}].x", m.saturating_sub(2));
            let v = format!("t[{}].x", m.saturating_sub(1));
            let w = format!("dot(({coord}).xyz, ({s}).xyz)", s = srcs[0]);
            let normal = format!("float3({u}, {v}, {w})");
            match inst.opcode {
                Opcode::TexM3x3 => format!("float4({normal}, 1.0)"),
                Opcode::TexM3x3Tex => {
                    let coord4 = format!("float4({normal}, 0.0)");
                    sample_or_compare(ctx, m, &coord4, None)
                }
                _ => {
                    let eye = if matches!(inst.opcode, Opcode::TexM3x3Spec) {
                        format!("({e}).xyz", e = srcs[1])
                    } else {
                        // vspec: eye vector from the .w of the three coord regs.
                        format!(
                            "float3(t[{a}].w, t[{b}].w, ({coord}).w)",
                            a = m.saturating_sub(2),
                            b = m.saturating_sub(1)
                        )
                    };
                    let refl = format!(
                        "(2.0 * dot({normal}, {eye}) / dot({normal}, {normal}) * {normal} - {eye})"
                    );
                    let coord4 = format!("float4({refl}, 0.0)");
                    sample_or_compare(ctx, m, &coord4, None)
                }
            }
        }
        // `texreg2{ar,gb,rgb} tN, tM` — sample stage N using components of the
        // source register as coordinates: ar = (alpha, red), gb = (green,
        // blue), rgb = (red, green, blue) as a 3D coord.
        Opcode::TexReg2Ar => {
            let dst = inst
                .dst
                .as_ref()
                .ok_or_else(|| EmitError::UnsupportedInstruction("texreg2ar missing dst".into()))?;
            let coord4 = format!("float4(({s}).w, ({s}).x, 0.0, 0.0)", s = srcs[0]);
            sample_or_compare(ctx, dst.reg.index, &coord4, None)
        }
        Opcode::TexReg2Gb => {
            let dst = inst
                .dst
                .as_ref()
                .ok_or_else(|| EmitError::UnsupportedInstruction("texreg2gb missing dst".into()))?;
            let coord4 = format!("float4(({s}).y, ({s}).z, 0.0, 0.0)", s = srcs[0]);
            sample_or_compare(ctx, dst.reg.index, &coord4, None)
        }
        Opcode::TexReg2Rgb => {
            let dst = inst.dst.as_ref().ok_or_else(|| {
                EmitError::UnsupportedInstruction("texreg2rgb missing dst".into())
            })?;
            let coord4 = format!("float4(({s}).xyz, 0.0)", s = srcs[0]);
            sample_or_compare(ctx, dst.reg.index, &coord4, None)
        }
        // `texdp3 tN, src` — 3-component dot product, broadcast to all lanes.
        Opcode::TexDp3 => {
            let dst = inst
                .dst
                .as_ref()
                .ok_or_else(|| EmitError::UnsupportedInstruction("texdp3 missing dst".into()))?;
            let coord = register_read_expr(dst.reg, ctx)?;
            format!("float4(dot(({coord}).xyz, ({s}).xyz))", s = srcs[0])
        }
        // `texdp3tex tN, src` — 3-component dot u, then sample stage N as a 1D
        // texture at (u, 0).
        Opcode::TexDp3Tex => {
            let dst = inst
                .dst
                .as_ref()
                .ok_or_else(|| EmitError::UnsupportedInstruction("texdp3tex missing dst".into()))?;
            let coord = register_read_expr(dst.reg, ctx)?;
            let u = format!("dot(({coord}).xyz, ({s}).xyz)", s = srcs[0]);
            let coord4 = format!("float4({u}, 0.0, 0.0, 0.0)");
            sample_or_compare(ctx, dst.reg.index, &coord4, None)
        }
        // `texdepth rN` (ps_1_4) — interpret r.x as z, r.y as w; write fragment
        // depth = z / w. Per the D3D9 `texdepth` reference behavior, the
        // divisor is clamped to at most 1.0 *before* the divide and the
        // result rides through a [0,1] clamp; there is no special case for
        // `r.y == 0` — `saturate` turns the resulting ±inf into 1.0 / 0.0, which
        // is the observed D3D9 behaviour. MSL `saturate` == `clamp(.,0,1)`.
        Opcode::TexDepth => {
            let dst = inst
                .dst
                .as_ref()
                .ok_or_else(|| EmitError::UnsupportedInstruction("texdepth missing dst".into()))?;
            let src = register_read_expr(dst.reg, ctx)?;
            let _ = writeln!(
                out,
                "    _depth_storage = float4(saturate(({src}).x / min(({src}).y, 1.0)));"
            );
            return Ok(());
        }
        // Nop has no semantic effect; Ret outside subroutines is invalid
        // in Metal but the parser already strips ret from subroutine bodies,
        // so a stray ret in the main stream emits nothing.
        Opcode::Nop | Opcode::Ret => return Ok(()),
        // SM2.x / SM3 conditional control flow.
        // `if src` is true when src.x is nonzero per the D3D9 spec.
        // No dst, srcs[0] holds the condition operand.
        Opcode::If => {
            let _ = writeln!(out, "    if (({}).x != 0.0) {{", srcs[0]);
            return Ok(());
        }
        // `ifc_<cmp> s0, s1` — comparison flavour rides in the
        // instruction-token control bits, decoded by the parser into
        // `inst.cmp_func`. Always present here; default to `==` if a
        // future bytecode arrives without one (logged by the parser
        // path that produced None).
        Opcode::Ifc => {
            let op_str = inst.cmp_func.map_or("==", super::ir::CmpFunc::op);
            let _ = writeln!(
                out,
                "    if (({}).x {} ({}).x) {{",
                srcs[0], op_str, srcs[1]
            );
            return Ok(());
        }
        Opcode::Else => {
            w(out, "    } else {\n");
            return Ok(());
        }
        Opcode::EndIf => {
            w(out, "    }\n");
            return Ok(());
        }
        // SM2.x / SM3 `loop aL, iN` — iN.x = iteration count,
        // iN.y = initial aL value, iN.z = step. Allocate a fresh
        // `aL_<n>` local so nested loops don't collide; the loop
        // stack tracks the index so RegKind::Loop reads inside the
        // body resolve to the right name.
        Opcode::Loop => {
            let counter_idx = {
                let mut count = ctx.loop_al_count.borrow_mut();
                let idx = *count;
                *count += 1;
                idx
            };
            ctx.loop_stack.borrow_mut().push(Some(counter_idx));
            // iN is in srcs[1] for `loop aL, iN`. srcs[0] is the
            // aL register reference (a `RegKind::Loop` operand the
            // parser materializes); we don't need its expression
            // since aL is locally named.
            let counter = &srcs[1];
            let _ = writeln!(
                out,
                "    for (int aL_{counter_idx} = ({counter}).y, _aL_step_{counter_idx} = ({counter}).z, _aL_end_{counter_idx} = ({counter}).x; \
                 _aL_end_{counter_idx}-- > 0; aL_{counter_idx} += _aL_step_{counter_idx}) {{"
            );
            return Ok(());
        }
        // `rep iN` — iterate iN.x times. No aL allocated.
        Opcode::Rep => {
            ctx.loop_stack.borrow_mut().push(None);
            let depth = ctx.loop_stack.borrow().len() - 1;
            let _ = writeln!(
                out,
                "    for (int _rep_{depth} = 0; _rep_{depth} < ({c}).x; ++_rep_{depth}) {{",
                c = srcs[0]
            );
            return Ok(());
        }
        Opcode::EndLoop | Opcode::EndRep => {
            ctx.loop_stack.borrow_mut().pop();
            w(out, "    }\n");
            return Ok(());
        }
        Opcode::Break => {
            w(out, "    break;\n");
            return Ok(());
        }
        Opcode::BreakC => {
            let op_str = inst.cmp_func.map_or("==", super::ir::CmpFunc::op);
            let _ = writeln!(
                out,
                "    if (({}).x {} ({}).x) break;",
                srcs[0], op_str, srcs[1]
            );
            return Ok(());
        }
        // SM3 `setp_<cmp> p0, s0, s1` — componentwise predicate set.
        // Bypass `store_dst`: p0 is bool4, the standard write-mask
        // path expects float4 lvalues. `inst.cmp_func` is decoded by
        // the parser; default to `==` if absent.
        Opcode::SetP => {
            let op_str = inst.cmp_func.map_or("==", super::ir::CmpFunc::op);
            let _ = writeln!(out, "    p0 = ({} {} {});", srcs[0], op_str, srcs[1]);
            return Ok(());
        }
        // `breakp pN` — predicate-gated break. The predicate operand
        // is stored in `inst.predicate` (since `predicated` is set on
        // the instruction); break when `any` of the gated lanes is
        // true. `any(p0)` is conservative and matches D3D9 PS
        // semantics where the swizzle picks specific lanes.
        Opcode::BreakP => {
            let pred = inst.predicate.as_ref().ok_or_else(|| {
                EmitError::UnsupportedInstruction("breakp without predicate operand".into())
            })?;
            let pred_expr = predicate_gate_expr(pred);
            let _ = writeln!(out, "    if ({pred_expr}) break;");
            return Ok(());
        }
        // `call sN` — inline-expand the labelled subroutine. The
        // src operand carries the label index in `reg.index`. The
        // expansion stack rejects cycles with a once-per-label warn.
        Opcode::Call => {
            let label_id = u32::from(inst.srcs[0].reg.index);
            expand_subroutine(out, label_id, ctx)?;
            return Ok(());
        }
        // `callnz sN, src` — Call gated on `src.x != 0`. src0 is the
        // label, src1 is the condition.
        Opcode::CallNz => {
            let label_id = u32::from(inst.srcs[0].reg.index);
            let _ = writeln!(out, "    if (({}).x != 0.0) {{", srcs[1]);
            expand_subroutine(out, label_id, ctx)?;
            w(out, "    }\n");
            return Ok(());
        }
        Opcode::MovA => {
            // `mova a0, src` writes the VS-local int4 address register. See
            // `write_address_register` for why this bypasses `store_dst`.
            let Some(dst) = inst.dst else {
                return Err(EmitError::UnsupportedInstruction(
                    "MovA without dst".to_string(),
                ));
            };
            write_address_register(out, dst, &srcs[0], /* use_floor */ false);
            return Ok(());
        }
        op => return Err(EmitError::UnsupportedInstruction(format!("{op:?}"))),
    };

    if let Some(dst) = inst.dst {
        // vs_1_1 writes the address register with a plain `mov a0, …`
        // (the `mova` opcode arrived in SM2). Route any VS write to
        // `RegKind::Addr` through the int4 `a` register, bypassing the
        // float `store_dst` path the same way the `MovA` arm does.
        if dst.reg.kind == RegKind::Addr && ctx.is_vertex() {
            write_address_register(out, dst, &expr, /* use_floor */ true);
            return Ok(());
        }
        // Predicated execution: wrap the dst write in a conditional
        // gated by the predicate operand the parser split off. The
        // gate uses the operand's swizzle to pick a p0 lane and its
        // SrcModifier::Not to negate. Unpredicated → straight write.
        if let Some(pred) = &inst.predicate {
            let gate = predicate_gate_expr(pred);
            let _ = writeln!(out, "    if ({gate}) {{");
            store_dst(out, dst, &expr, ctx);
            w(out, "    }\n");
        } else {
            store_dst(out, dst, &expr, ctx);
        }
    }
    Ok(())
}

/// Inline-expand a subroutine body at a `Call` / `CallNz` site.
///
/// Cycle detection: if the label is already on the expansion stack,
/// warn once and skip. Missing labels (call to an undefined id)
/// also warn-and-skip rather than failing the emit — best-effort
/// translation, surface the issue.
fn expand_subroutine(out: &mut String, label: u32, ctx: &EmitContext) -> Result<(), EmitError> {
    let Some(subs) = ctx.subroutines else {
        mtld3d_shared::log_once_warn_by!(
            target: super::LOG_TARGET,
            key: u64::from(label),
            "dxso: call sN with no subroutine map → skipped (label {label})"
        );
        return Ok(());
    };
    let Some(body) = subs.get(&label) else {
        mtld3d_shared::log_once_warn_by!(
            target: super::LOG_TARGET,
            key: u64::from(label),
            "dxso: call to undefined subroutine label {label} → skipped"
        );
        return Ok(());
    };
    {
        let stack = ctx.expansion_stack.borrow();
        if stack.contains(&label) {
            drop(stack);
            mtld3d_shared::log_once_warn_by!(
                target: super::LOG_TARGET,
                key: u64::from(label),
                "dxso: recursive call detected at label {label} → expansion skipped"
            );
            return Ok(());
        }
    }
    ctx.expansion_stack.borrow_mut().push(label);
    for sub_inst in body {
        translate_instruction(out, sub_inst, ctx)?;
    }
    ctx.expansion_stack.borrow_mut().pop();
    Ok(())
}

/// Build the boolean MSL expression that gates a predicated instruction (or `breakp`).
///
/// The predicate operand carries a swizzle picking which p0 lane is the gate,
/// and a `Not` modifier inverts the test. `any(p0)` is the fallback for an
/// `xyzw` swizzle.
fn predicate_gate_expr(pred: &SrcOperand) -> String {
    let swiz = pred.swizzle.0;
    let comp = b"xyzw"[swiz[0] as usize] as char;
    // SM3 typically replicates a single component (e.g. .xxxx); if all
    // four lanes are identical, use the scalar lane directly. Otherwise
    // collapse to `any(p0)` — the D3D9 spec says lanes act
    // independently, but the most conservative gate is "any lane true".
    let scalar = swiz.iter().all(|&c| c == swiz[0]);
    let base = if scalar {
        format!("p0.{comp}")
    } else {
        "any(p0)".to_string()
    };
    if pred.modifier == SrcModifier::Not {
        format!("!({base})")
    } else {
        base
    }
}

// ── Matrix multiply ──
// m*x* dst, s0, c_base: expands to `rows` dot products across `cols`-wide
// vectors. `mat_start` is the first matrix row; subsequent rows are at
// index+1..index+rows-1.

fn emit_matmul(
    row_expr: &str,
    mat_start: &SrcOperand,
    cols: u8,
    rows: u8,
    ctx: &EmitContext,
) -> Result<String, EmitError> {
    let mut components = Vec::with_capacity(rows as usize);
    for i in 0..rows {
        let mat_row = SrcOperand {
            reg: Register {
                kind: mat_start.reg.kind,
                index: mat_start.reg.index + u16::from(i),
            },
            ..*mat_start
        };
        let mat_expr = load_src(&mat_row, ctx)?;
        let inner = if cols == 3 {
            format!("dot(({row_expr}).xyz, ({mat_expr}).xyz)")
        } else {
            format!("dot({row_expr}, {mat_expr})")
        };
        components.push(inner);
    }
    // Pad to float4 so the expression type matches dst. Unused lanes are 0.
    while components.len() < 4 {
        components.push("0.0".to_string());
    }
    Ok(format!(
        "float4({}, {}, {}, {})",
        components[0], components[1], components[2], components[3]
    ))
}

// ── Source operand expression ──

/// Resolve the innermost open `loop aL, iN` frame to the index `n` of its `aL_<n>` local.
///
/// Returns `None` when there is no enclosing `loop` (an `aL` reference inside
/// only `rep iN` blocks, or outside any loop) — malformed bytecode the caller
/// handles by warning and falling back.
fn current_loop_al(ctx: &EmitContext) -> Option<usize> {
    ctx.loop_stack.borrow().iter().rev().find_map(|f| *f)
}

fn load_src(src: &SrcOperand, ctx: &EmitContext) -> Result<String, EmitError> {
    let base = if let Some(rel) = src.rel_addr {
        // Relative addressing: `c[<index> + N]`. D3D9 allows this only on the
        // constant buffer. The dynamic index is either the address register
        // (`c[a0.<swiz> + N]`, SM1/SM2) or the loop counter inside a
        // `loop aL, iN` block (`c[aL + N]`, SM2+); both resolve to a scalar
        // int. Anything else is not something the parser should produce — if
        // it ever does, fail loudly rather than emit garbage.
        if src.reg.kind != RegKind::Const {
            return Err(EmitError::UnsupportedRegisterKind(format!(
                "rel_addr on non-const register {:?}",
                src.reg.kind
            )));
        }
        let rel_index = match rel.reg.kind {
            // `a0` — scalar-sourced from the first swizzle component.
            RegKind::Addr => {
                let rel_component = b"xyzw"[rel.swizzle.0[0] as usize] as char;
                format!("a.{rel_component}")
            }
            // `aL` — the int loop counter of the innermost enclosing loop.
            RegKind::Loop => current_loop_al(ctx).map_or_else(
                || {
                    mtld3d_shared::log_once_warn!(target: super::LOG_TARGET,
                        "dxso: c[aL + N] outside a loop scope → indexes from 0"
                    );
                    "0".to_string()
                },
                |idx| format!("aL_{idx}"),
            ),
            other => {
                return Err(EmitError::UnsupportedRegisterKind(format!(
                    "rel_addr using unsupported register {other:?}"
                )));
            }
        };
        let buf = if ctx.is_vertex() { "vs_c" } else { "ps_c" };
        if ctx.def_consts.is_empty() {
            format!("{buf}[{rel_index} + {}]", src.reg.index)
        } else {
            // `def`-declared constants live in scalar locals, invisible to the
            // uniform buffer; the helper overlays them onto the dynamic index.
            format!("mtld3d_const_rel({rel_index} + {}, {buf})", src.reg.index)
        }
    } else {
        register_read_expr(src.reg, ctx)?
    };
    let swizzled = apply_swizzle(&base, src.swizzle);
    let modified = apply_src_modifier(&swizzled, src.modifier);
    Ok(modified)
}

fn register_read_expr(reg: Register, ctx: &EmitContext) -> Result<String, EmitError> {
    Ok(match reg.kind {
        RegKind::Temp => format!("r[{}]", reg.index),
        RegKind::Const => {
            if ctx.def_consts.contains(&reg.index) {
                // `def` constants are already clamped at their definition for
                // ps_1_x (see `emit_ps_function`), so the read is the bare local.
                format!("c{}", reg.index)
            } else if !ctx.is_vertex() && ctx.is_sm1() {
                // ps_1_x clamps float constant register reads to [-1, 1] (the
                // fixed-point hardware range). The backing buffer holds the raw
                // value, so clamp at the read.
                format!("clamp(ps_c[{}], -1.0, 1.0)", reg.index)
            } else {
                let buf = if ctx.is_vertex() { "vs_c" } else { "ps_c" };
                format!("{buf}[{}]", reg.index)
            }
        }
        RegKind::Input => {
            if ctx.is_vertex() {
                if ctx.vs_input_provided(reg.index) {
                    format!("in.v{}", reg.index)
                } else {
                    // The vertex declaration omits this input — D3D9 reads zero.
                    "float4(0.0)".to_owned()
                }
            } else {
                let name = ctx
                    .ps_input_map
                    .and_then(|m| m.get(&reg.index))
                    .cloned()
                    .unwrap_or_else(|| format!("color{}", reg.index));
                format!("in.{name}")
            }
        }
        // Direct named constants we allow reading back (rare — mostly for
        // instructions that reference their own dst as a src, e.g. Cmp).
        RegKind::ColorOut => format!("oC{}", reg.index),
        // Reg type 3 is context-dependent in DXSO: VS `a0` address
        // register, PS `tN` texture-coordinate input register (SM1.x/SM2.0).
        // For VS it widens from int4 → float4 so the standard swizzle /
        // modifier pipeline keeps working; most uses go through the
        // `rel_addr` path above. For PS it is implicitly bound to the Nth
        // texcoord varying — PS SM2 never writes to `tN`, so we only need
        // the read path here.
        RegKind::Addr => {
            if ctx.is_vertex() {
                "float4(a)".to_string()
            } else if ctx.is_sm1() {
                // SM1 PS: read-write `tN` register backed by the `t[]` array
                // (seeded from the texcoord varyings in `emit_ps_function`).
                format!("t[{}]", reg.index)
            } else {
                format!("in.texcoord{}", reg.index)
            }
        }
        // `aL` reads inside a `loop aL, iN` block — scan back through
        // the open scopes for the most recent Loop frame (a `Some`
        // entry) and use its `aL_<n>` local. Reads outside any loop,
        // or inside only `rep` blocks (which have no aL), warn and
        // return zero so the surface stays diagnosable.
        RegKind::Loop => current_loop_al(ctx).map_or_else(
            || {
                mtld3d_shared::log_once_warn!(target: super::LOG_TARGET,
                    "dxso: aL read outside a loop scope → returns float4(0)"
                );
                "float4(0)".to_string()
            },
            |idx| format!("float4(aL_{idx})"),
        ),
        // `iN` integer-constant reads. A `defi`-declared constant is a baked
        // `int4 iN` local; a dynamic one (fed by SetVertexShaderConstantI) in a
        // VS reads the runtime `vs_i` buffer (slot 14). Cast to float4 so the
        // standard swizzle / modifier pipeline applies.
        RegKind::ConstInt => {
            if ctx.is_vertex() && !ctx.def_int_consts.contains(&reg.index) {
                format!("float4(vs_i[{}])", reg.index)
            } else {
                format!("float4(i{})", reg.index)
            }
        }
        // Label register reads only land here as the operand of
        // `Call sN` / `CallNz` — the emit arm pulls `reg.index`
        // directly off the parsed `SrcOperand` rather than this
        // string, so the placeholder is never used. Emitting one
        // keeps `load_src` from failing on the label argument.
        RegKind::Label => format!("/* label {} */", reg.index),
        // SM3 PS dedicated registers. Reg index disambiguates:
        //   0 = vPos  → screen-space pixel coord (`v_pos` is the
        //               [[position]] fragment-function arg).
        //   1 = vFace → ±1.0 float (`v_face` is computed from
        //               `front_facing` in the function prologue).
        // The reads only land here when a matching `dcl_*` was seen
        // by `emit_ps_function` (the bool flags gate emission of the
        // corresponding fragment-function arg). A read without a
        // declaration would compile-fail; warn-and-zero so the
        // surface is at least diagnosable.
        RegKind::MiscType => match reg.index {
            // vPos = the `[[position]]` from the `Varyings` (`in.position`).
            // D3D9 vPos is the INTEGER pixel coord (so `frc(vPos)` is 0 at a
            // pixel), but Metal `[[position]]` is the pixel CENTRE (x+0.5) —
            // subtract 0.5 to match D3D9.
            0 if ctx.ps_has_vpos() => "(in.position - 0.5)".to_string(),
            1 if ctx.ps_has_vface() => "float4(v_face)".to_string(),
            other => {
                mtld3d_shared::log_once_warn_by!(
                    target: super::LOG_TARGET,
                    key: u64::from(other),
                    "dxso: PS3 MiscType reg {other} read without matching dcl → returns float4(0)"
                );
                "float4(0.0)".to_string()
            }
        },
        kind => return Err(EmitError::UnsupportedRegisterKind(format!("{kind:?}"))),
    })
}

fn apply_swizzle(base: &str, swiz: Swizzle) -> String {
    if swiz == Swizzle::IDENTITY {
        return base.to_string();
    }
    let chars: String = swiz
        .0
        .iter()
        .map(|&c| b"xyzw"[c as usize] as char)
        .collect();
    format!("({base}).{chars}")
}

fn apply_src_modifier(expr: &str, m: SrcModifier) -> String {
    match m {
        SrcModifier::None => expr.to_string(),
        SrcModifier::Neg => format!("(-{expr})"),
        SrcModifier::Bias => format!("({expr} - 0.5)"),
        SrcModifier::BiasNeg => format!("-({expr} - 0.5)"),
        SrcModifier::Sign => format!("(2.0 * {expr} - 1.0)"),
        SrcModifier::SignNeg => format!("(1.0 - 2.0 * {expr})"),
        SrcModifier::Comp => format!("(1.0 - {expr})"),
        SrcModifier::X2 => format!("(2.0 * {expr})"),
        SrcModifier::X2Neg => format!("(-2.0 * {expr})"),
        SrcModifier::Abs => format!("abs({expr})"),
        SrcModifier::AbsNeg => format!("(-abs({expr}))"),
        // Dz/Dw: perspective divide for `tex2Dproj` (HLSL) / `texldp`
        // (DXSO). FXC encodes the projective form as `texld` with this
        // modifier on the coord source; divide the whole coord vector
        // by `.z` (Dz) or `.w` (Dw) before sampling. Required for
        // projective shadow sampling (e.g. CSM foliage receivers) and
        // any other projective texture op.
        SrcModifier::Dz | SrcModifier::Dw => {
            let component = if matches!(m, SrcModifier::Dz) {
                'z'
            } else {
                'w'
            };
            format!("({expr} / ({expr}).{component})")
        }
        SrcModifier::Not => {
            // Logical NOT — for float operands the D3D9 semantic is
            // component-wise "non-zero → false, zero → true". MSL
            // `bool4(v)` evaluates each lane as non-zero; `!bool4(v)`
            // negates; cast back to float4 keeps downstream `.xyzw`
            // reads typed. Predicate-side Not is handled separately in
            // `predicate_gate_expr` and never reaches this arm.
            format!("float4(!bool4({expr}))")
        }
    }
}

// ── Destination storage ──

/// Write the VS int4 address register `a`.
///
/// Rounds each component to the nearest integer per the D3D9 spec. Shared by
/// `mova` (SM2+) and a plain `mov a0, …` (`vs_1_1`, which predates `mova`).
/// Bypasses `store_dst`'s `saturate()` / float-swizzle path — neither makes
/// sense for the int4 `a`.
fn write_address_register(out: &mut String, dst: DstOperand, value: &str, use_floor: bool) {
    // D3D9: `mova` (SM2+) rounds the float to the nearest integer; a plain
    // `mov` to the address register (vs_1_1, which has no `mova`) FLOORS it
    // (mova -2.4 → -2, but mov -2.4 → -3).
    let conv = if use_floor { "floor" } else { "round" };
    let rounded = format!("int4({conv}({value}))");
    if dst.write_mask == WriteMask::ALL {
        let _ = writeln!(out, "    a = {rounded};");
    } else {
        let chars = write_mask_chars(dst.write_mask);
        let _ = writeln!(out, "    a.{chars} = ({rounded}).{chars};");
    }
}

fn store_dst(out: &mut String, dst: DstOperand, value: &str, ctx: &EmitContext) {
    let target = register_write_target(dst.reg, ctx);
    // PS 1.x result shift modifier (`_x2`/`_x4`/`_x8`, `_d2`/`_d4`/`_d8`):
    // multiply the result by 2^shift_scale, applied before `_sat`. The field
    // is zero for every SM2+ shader, so SM2/SM3 emit is unchanged.
    let scaled = if dst.shift_scale == 0 {
        value.to_string()
    } else {
        format!(
            "({value} * {})",
            fmt_float(2.0_f32.powi(i32::from(dst.shift_scale)))
        )
    };
    let value = if dst.mods.contains(DstMods::SATURATE) {
        format!("saturate({scaled})")
    } else {
        scaled
    };

    let mask = dst.write_mask;
    if mask == WriteMask::ALL {
        let _ = writeln!(out, "    {target} = {value};");
    } else {
        let chars = write_mask_chars(mask);
        let _ = writeln!(out, "    {target}.{chars} = ({value}).{chars};");
    }
}

fn register_write_target(reg: Register, ctx: &EmitContext) -> String {
    // SM3 VS: the dcl semantic is the source of truth for any
    // output-flavor write, regardless of which output reg kind the
    // compiler picked. Consult the dcl-derived map first; fall through
    // to SM2 register-kind defaults only when the map is absent (SM2)
    // or has no entry (a missing-dcl bug — sink-and-warn).
    if let Some(map) = ctx.vs_output_map
        && is_output_reg_kind(reg.kind)
    {
        return map.get(&(reg.kind, reg.index)).map_or_else(
            || {
                mtld3d_shared::log_once_warn_by!(
                    target: super::LOG_TARGET,
                    key: u64::from(reg.index),
                    "dxso: VS3 output {:?}[{}] written without matching dcl → write sunk",
                    reg.kind,
                    reg.index
                );
                "_rastout_discard".to_string()
            },
            std::clone::Clone::clone,
        );
    }

    match reg.kind {
        RegKind::Temp => format!("r[{}]", reg.index),
        RegKind::RastOut => match reg.index {
            0 => "out.position".to_string(),
            1 => "out.fog".to_string(),
            // SM2 `oPts` — point-size output. Routed through the
            // float4 `_psize_storage` local so `store_dst`'s
            // write_mask path stays uniform; .x is extracted to the
            // scalar `out.point_size` field at return.
            2 => "_psize_storage".to_string(),
            other => {
                mtld3d_shared::log_once_warn!(target: super::LOG_TARGET,
                    "dxso: RastOut index {other} dropped (write sunk to scratch local)"
                );
                "_rastout_discard".to_string()
            }
        },
        RegKind::AttrOut => format!("out.color{}", reg.index),
        RegKind::TexcoordOut => format!("out.texcoord{}", reg.index),
        // SM1 PS `tN` write target — the read-write texcoord/sample register
        // (`tex`/`texcoord`/`texbem` results). VS address-register writes are
        // intercepted before `store_dst` (see `translate_instruction`), so
        // this arm is only reached for SM1 pixel shaders.
        RegKind::Addr => format!("t[{}]", reg.index),
        RegKind::ColorOut => format!("oC{}", reg.index),
        RegKind::DepthOut => {
            if ctx.ps_has_depth_out() {
                "_depth_storage".to_string()
            } else {
                mtld3d_shared::log_once_warn!(target: super::LOG_TARGET,
                    "dxso: DepthOut write hit a PS without a depth-out scan flag → write sunk"
                );
                "/* oDepth */".to_string()
            }
        }
        // SM3 predicate register `p0`. SetP bypasses `store_dst`
        // (p0 is bool4, write-mask path expects float4), so a write
        // here only fires for unusual bytecode; warn rather than
        // emit garbage.
        RegKind::Predicate => {
            mtld3d_shared::log_once_warn!(target: super::LOG_TARGET,
                "dxso: Predicate (p0) dst hit `store_dst` — only setp should write p0; check parser"
            );
            "/* p0 */".to_string()
        }
        other => {
            mtld3d_shared::log_once_warn!(target: super::LOG_TARGET, "dxso: unsupported dst RegKind={other:?} → write discarded");
            format!("/* unsupported reg {other:?} */")
        }
    }
}

/// Picks the coord swizzle for a sample call based on the sampler's declared texture type.
///
/// 2D → `xy`, Cube / 3D → `xyz`. Shared by the plain `texld`, `texldl`, and
/// `texldd` translations so future texture-dimensionality work has one site
/// to update.
fn sampler_coord_swizzle(ctx: &EmitContext, sampler_idx: u16) -> &'static str {
    match ctx.ps_samplers.and_then(|m| m.get(&sampler_idx)) {
        Some(TextureType::TextureCube | TextureType::Texture3D) => "xyz",
        _ => "xy",
    }
}

/// True for SM1 texture ops that issue a `sample` call.
///
/// So the destination register's stage needs an implicit sampler binding. SM1
/// bytecode has no `dcl_<dim> sN`; the sampler bound to a stage is implicit
/// (stage N → sampler N). Excludes the pad / dot-only / depth-only ops.
const fn sm1_op_samples(op: Opcode) -> bool {
    matches!(
        op,
        Opcode::TexLd
            | Opcode::TexBem
            | Opcode::TexBemL
            | Opcode::TexM3x2Tex
            | Opcode::TexM3x3Tex
            | Opcode::TexM3x3Spec
            | Opcode::TexM3x3VSpec
            | Opcode::TexReg2Ar
            | Opcode::TexReg2Gb
            | Opcode::TexReg2Rgb
            | Opcode::TexDp3Tex
    )
}

/// MSL expressions for texture stage `stage`'s 2×2 bump-environment matrix.
///
/// `D3DTSS_BUMPENVMAT00/01/10/11`, packed by the d3d9 side as a `float4`
/// `(m00, m01, m10, m11)` at `bump_env[stage*2]`. Returns
/// `(m00, m01, m10, m11)`.
fn bump_matrix_exprs(stage: u16) -> (String, String, String, String) {
    let i = stage * 2;
    (
        format!("bump_env[{i}].x"),
        format!("bump_env[{i}].y"),
        format!("bump_env[{i}].z"),
        format!("bump_env[{i}].w"),
    )
}

/// MSL expressions for stage `stage`'s bump luminance scale/offset.
///
/// `D3DTSS_BUMPENVLSCALE`/`LOFFSET`, packed as `(lscale, loffset, 0, 0)` at
/// `bump_env[stage*2+1]`. Returns `(lscale, loffset)`.
fn bump_lum_exprs(stage: u16) -> (String, String) {
    let i = stage * 2 + 1;
    (format!("bump_env[{i}].x"), format!("bump_env[{i}].y"))
}

/// Emit a sampling expression honouring D3D9's hardware shadow filter:
///
/// * Color slot → `s.sample(samp, coord.<sw>[, suffix])` returning float4.
/// * Depth slot → `s.sample_compare(samp, coord.xy, saturate(coord.z)[, suffix])`
///   returning a single PCF result; broadcast to `float4(...)` so the
///   register stays a `float4` for downstream `.xyzw` reads.
///
/// The D3D9 shadow-mapping idiom is `tex2D(s, float3(uv, z_ref))` against
/// a depth-format texture: hardware returns "is `z_ref` ≤ stored depth?"
/// as a 0..1 PCF value. Metal exposes the same primitive as
/// `depth2d<float>::sample_compare(sampler, coord, ref)`, gated on the
/// sampler's `compareFunction` being non-`Never` — that bit is set on
/// the bind side via the encoder's `is_compare` sampler-key flag.
///
/// The reference is `saturate`d to [0,1]. D3D9 authors `z_ref` for a D24
/// UNORM depth target, but Apple Silicon has no UNORM depth format — every
/// depth texture is promoted to `Depth32Float` (`format.rs`), which does
/// not clamp the comparison reference the way fixed-point UNORM does. A
/// `z_ref` slightly past 1.0 (routine at shadow-cascade edges) would then
/// read as fully occluded.
///
/// `suffix` is the trailing `, level(...)` (texldl) or `, gradientNN(...)`
/// (texldd) text — `None` for plain `texld`. `sample_compare` accepts the
/// same `level(...)` / `gradient*(...)` overloads, so the suffix passes
/// through unchanged.
fn sample_or_compare(
    ctx: &EmitContext,
    sampler_idx: u16,
    coord_expr: &str,
    suffix: Option<&str>,
) -> String {
    let coord_swizzle = sampler_coord_swizzle(ctx, sampler_idx);
    let suffix_str = suffix.unwrap_or("");
    if ctx.is_raw_depth_sampler(sampler_idx) {
        // INTZ/DF24/DF16: read the RAW stored normalized depth (broadcast to
        // float4) via a plain `.sample()` on the `depth2d<float>` binding — NOT
        // a hardware shadow comparison (per D3D9, raw-depth FOURCC formats are
        // excluded from shadow sampling). The projective `.q` divide, if any,
        // was already folded into `coord_expr` by the texldp caller. Pin
        // `level(0)` for the same no-mip / discard-derivative
        // reason as the compare path; texldl/texldd override via their suffix.
        let lod_suffix = if suffix_str.is_empty() {
            ", level(0)"
        } else {
            suffix_str
        };
        return format!(
            "float4(s{sampler_idx}.sample(samp{sampler_idx}, ({coord_expr}).xy{lod_suffix}))"
        );
    }
    if ctx.is_depth_sampler(sampler_idx) {
        // Plain `texld` against a depth sampler has no explicit LOD
        // suffix; Metal would then use implicit gradients to pick the
        // mip level. Cascade shadow maps have no mips, and implicit
        // gradients in a 2×2 quad where a neighbour ran
        // `discard_fragment` are *undefined* per the Metal spec —
        // exactly the case for alpha-cut foliage shadow receivers.
        // Force `level(0)` to pin the mip and eliminate the
        // discard-driven derivative dependency. `texldl` / `texldd`
        // already pass their own suffix and override this default.
        let lod_suffix = if suffix_str.is_empty() {
            ", level(0)"
        } else {
            suffix_str
        };
        format!(
            "float4(s{sampler_idx}.sample_compare(samp{sampler_idx}, ({coord_expr}).xy, saturate(({coord_expr}).z){lod_suffix}))"
        )
    } else {
        format!(
            "s{sampler_idx}.sample(samp{sampler_idx}, ({coord_expr}).{coord_swizzle}{suffix_str})"
        )
    }
}

fn write_mask_chars(m: WriteMask) -> String {
    let mut s = String::new();
    for i in 0..4 {
        if m.covers(i) {
            s.push(b"xyzw"[i as usize] as char);
        }
    }
    s
}

fn fmt_float(v: f32) -> String {
    // Special IEEE values need their MSL macro spellings — Rust's `{}` prints
    // `NaN` / `inf` / `-inf`, none of which are valid MSL float literals
    // (`NaN` would even get a spurious `.0` appended below). `metal_stdlib`
    // provides `NAN` and `INFINITY`.
    if v.is_nan() {
        return "NAN".to_string();
    }
    if v.is_infinite() {
        return if v < 0.0 {
            "-INFINITY".to_string()
        } else {
            "INFINITY".to_string()
        };
    }
    // Always emit with a decimal point so the literal parses as float in MSL.
    let s = format!("{v}");
    if s.contains('.') || s.contains('e') || s.contains('E') {
        s
    } else {
        format!("{s}.0")
    }
}
