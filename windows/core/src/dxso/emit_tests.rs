//! Emitter tests.
//!
//! We don't pin the exact output (formatting churn would break tests
//! without catching real bugs) — instead check that the generated MSL
//! contains the structural elements each instruction should produce.
//!
//! Each test emits VS and PS independently (matching the per-stage API) and
//! concatenates the two strings into one check target.

use super::{
    emit::{VariantFlags, VariantKey, emit_ps_programmable, emit_vs_programmable},
    parser::parse,
};

const VS_HEADER: u32 = 0xFFFE_0200;
const PS_HEADER: u32 = 0xFFFF_0200;
const VS3_HEADER: u32 = 0xFFFE_0300;
const PS3_HEADER: u32 = 0xFFFF_0300;
const END_TOKEN: u32 = 0x0000_FFFF;
const SWIZ_IDENTITY: u8 = 0xE4;

const TYPE_TEMP: u32 = 0;
const TYPE_INPUT: u32 = 1;
const TYPE_CONST: u32 = 2;
const TYPE_ADDR: u32 = 3;
const TYPE_RASTOUT: u32 = 4;
const TYPE_TEXCOORDOUT: u32 = 6;
const TYPE_COLOROUT: u32 = 8;
const TYPE_OUTPUT: u32 = 11;

/// `dcl_<usage>_<index>` token: bit 31 set + usage bits 0..4 + index bits 16..19.
fn dcl_usage_token(usage: u8, index: u8) -> u32 {
    0x8000_0000 | (u32::from(usage) & 0x1F) | ((u32::from(index) & 0xF) << 16)
}

const DCL_POSITION: u8 = 0;
const DCL_TEXCOORD: u8 = 5;
const DCL_COLOR: u8 = 10;

const OP_MOV: u16 = 1;
const OP_ADD: u16 = 2;
const OP_DP3: u16 = 8;
const OP_DP4: u16 = 9;
const OP_SLT: u16 = 12;
const OP_SGE: u16 = 13;
const OP_M3X4: u16 = 22;
const OP_M3X2: u16 = 24;
const OP_DCL: u16 = 31;
const OP_CRS: u16 = 33;
const OP_SGN: u16 = 34;
const OP_MOVA: u16 = 46;
const OP_EXPP: u16 = 78;
const OP_LOGP: u16 = 79;
const OP_DEF: u16 = 81;
const OP_CMP: u16 = 88;
const OP_LIT: u16 = 16;
const OP_DST: u16 = 17;
const OP_CND: u16 = 80;
const OP_LOOP: u16 = 27;
const OP_ENDLOOP: u16 = 29;
const OP_REP: u16 = 38;
const OP_ENDREP: u16 = 39;
const OP_IF: u16 = 40;
const OP_IFC: u16 = 41;
const OP_ELSE: u16 = 42;
const OP_ENDIF: u16 = 43;
const OP_BREAK: u16 = 44;
const OP_BREAKC: u16 = 45;
const OP_DEFI: u16 = 48;
const OP_BREAKP: u16 = 96;
const OP_SETP: u16 = 94;

const TYPE_PREDICATE: u32 = 19;
const TYPE_LABEL: u32 = 18;
const OP_CALL: u16 = 25;
const OP_CALLNZ: u16 = 26;
const OP_RET: u16 = 28;
const OP_LABEL: u16 = 30;

const TYPE_CONSTINT: u32 = 7;
const TYPE_LOOP: u32 = 15;
const OP_DP2ADD: u16 = 90;
const OP_DSX: u16 = 91;
const OP_DSY: u16 = 92;
const OP_TEXLDD: u16 = 93;
const OP_TEXLDL: u16 = 95;
const OP_TEXKILL: u16 = 65;

/// `.xxxx` swizzle (replicate component 0).
const SWIZ_XXXX: u8 = 0x00;

fn reg_bits(reg_type: u32, index: u16) -> u32 {
    let low = reg_type & 0x7;
    let high = (reg_type >> 3) & 0x3;
    // Bit 31 marks every dst/src parameter token in real D3D9 bytecode (all
    // shader models). SM1 operand counting keys off it, so the helpers must
    // set it to model the stream a shader compiler actually emits.
    0x8000_0000 | (low << 28) | (high << 11) | u32::from(index)
}

fn opcode_token(opcode: u16, token_count: u32) -> u32 {
    u32::from(opcode) | (token_count << 24)
}

fn dst_token(reg_type: u32, index: u16, write_mask: u8, saturate: bool) -> u32 {
    let mut t = reg_bits(reg_type, index);
    t |= (u32::from(write_mask) & 0xF) << 16;
    if saturate {
        t |= 1 << 20;
    }
    t
}

fn src_token(reg_type: u32, index: u16, swizzle: u8, modifier: u8) -> u32 {
    let mut t = reg_bits(reg_type, index);
    t |= u32::from(swizzle) << 16;
    t |= (u32::from(modifier) & 0xF) << 24;
    t
}

fn trivial_passthrough_vs() -> Vec<u32> {
    // dcl_position v0; mov oPos, v0;
    vec![
        VS_HEADER,
        opcode_token(OP_DCL, 2),
        0x0000_0000,
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ]
}

fn red_constant_ps() -> Vec<u32> {
    // def c0, 1, 0, 0, 1; mov oC0, c0;
    vec![
        PS_HEADER,
        opcode_token(OP_DEF, 5),
        dst_token(TYPE_CONST, 0, 0xF, false),
        f32::to_bits(1.0),
        f32::to_bits(0.0),
        f32::to_bits(0.0),
        f32::to_bits(1.0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ]
}

fn emit_pair_for_tests(vs_bc: &[u32], ps_bc: &[u32], variant: VariantKey) -> String {
    let vs = parse(vs_bc).expect("VS parse");
    let ps = parse(ps_bc).expect("PS parse");
    let vs_msl = emit_vs_programmable(&vs).expect("emit VS");
    let ps_msl = emit_ps_programmable(&ps, variant).expect("emit PS");
    format!("{vs_msl}\n{ps_msl}")
}

#[test]
fn minimal_vs_plus_ps_emits_valid_msl_skeleton() {
    let msl = emit_pair_for_tests(
        &trivial_passthrough_vs(),
        &red_constant_ps(),
        VariantKey::default(),
    );

    assert!(
        msl.contains("#include <metal_stdlib>"),
        "missing stdlib:\n{msl}"
    );
    assert!(msl.contains("struct VertexIn"), "no VertexIn:\n{msl}");
    assert!(
        msl.contains("float4 v0 [[attribute(0)]]"),
        "no VertexIn::v0:\n{msl}"
    );
    assert!(msl.contains("struct Varyings"), "no Varyings:\n{msl}");
    assert!(
        msl.contains("float4 position [[position, invariant]]"),
        "no position field:\n{msl}"
    );
    assert!(
        msl.contains("constant float4 *vs_c [[buffer(15)]]"),
        "no VS constants slot:\n{msl}"
    );
    assert!(
        msl.contains("constant float4 *ps_c [[buffer(15)]]"),
        "no PS constants slot:\n{msl}"
    );
    assert!(
        msl.contains("vertex Varyings mtld3d_vs("),
        "no VS entry:\n{msl}"
    );
    assert!(
        msl.contains("fragment float4 mtld3d_ps("),
        "no PS entry:\n{msl}"
    );
    assert!(
        msl.contains("out.position = in.v0;"),
        "mov to oPos missing:\n{msl}"
    );
    assert!(
        msl.contains("float4 c0 = float4(1.0, 0.0, 0.0, 1.0);"),
        "def c0 missing:\n{msl}"
    );
    assert!(msl.contains("oC0 = c0;"), "mov oC0, c0 missing:\n{msl}");
}

#[test]
fn varyings_put_texcoord_before_color() {
    let msl = emit_pair_for_tests(
        &trivial_passthrough_vs(),
        &red_constant_ps(),
        VariantKey::default(),
    );

    let tc0 = msl.find("texcoord0").expect("texcoord0 missing");
    let color0 = msl.find("color0").expect("color0 missing");
    assert!(
        tc0 < color0,
        "texcoord should precede color in Varyings to avoid Metal compiler crash"
    );
}

#[test]
fn emit_vs_1_1_real_bytecode_without_length_field() {
    // A literal vs_1_1 stream with no instruction-length fields. Real SM1
    // opcode tokens carry none, so the parser counts operands by the bit-31
    // run; each `mov` must therefore resolve its sources rather than parse
    // with zero (which would leave the emitter reading a missing `srcs[0]`).
    // This must translate cleanly: oPos → position, oD0 → color varying.
    let bc = vec![
        0xFFFE_0101, // vs_1_1
        0x0000_001F,
        0x8000_0000,
        0x900F_0000, // dcl_position v0
        0x0000_0001,
        0xC00F_0000,
        0x90E4_0000, // mov oPos, v0
        0x0000_0001,
        0xD00F_0000,
        0xA0E4_0000, // mov oD0, c0
        0x0000_FFFF, // end
    ];
    let vs = parse(&bc).expect("vs_1_1 parse");
    let msl = emit_vs_programmable(&vs).expect("vs_1_1 emit");
    assert!(
        msl.contains("out.position = in.v0;"),
        "mov oPos missing:\n{msl}"
    );
    assert!(
        msl.contains("out.color0 = vs_c[0];") || msl.contains("color0 = vs_c[0]"),
        "mov oD0, c0 should write color0 from constant 0:\n{msl}"
    );
}

#[test]
fn swizzle_and_write_mask_emit_correctly() {
    // vs_2_0 { dcl_position v0; mov r0.xy, v0.zwxy; mov oPos, r0; }
    let bc = vec![
        VS_HEADER,
        opcode_token(OP_DCL, 2),
        0x0000_0000,
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEMP, 0, 0b0011, false),     // .xy mask
        src_token(TYPE_INPUT, 0, 0b01_00_11_10, 0), // .zwxy
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let msl = emit_pair_for_tests(&bc, &red_constant_ps(), VariantKey::default());

    // r[0].xy = (in.v0).zwxy.xy;
    assert!(
        msl.contains("r[0].xy = (in.v0).zwxy.xy;") || msl.contains("r[0].xy = ((in.v0).zwxy).xy;"),
        "swizzle + write mask incorrect:\n{msl}"
    );
}

#[test]
fn saturate_wraps_in_saturate_call() {
    // vs_2_0 { dcl_position v0; mov_sat r0, v0; mov oPos, r0; }
    let bc = vec![
        VS_HEADER,
        opcode_token(OP_DCL, 2),
        0x0000_0000,
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEMP, 0, 0xF, true), // saturate
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let msl = emit_pair_for_tests(&bc, &red_constant_ps(), VariantKey::default());

    assert!(
        msl.contains("r[0] = saturate(in.v0);"),
        "saturate missing:\n{msl}"
    );
}

#[test]
fn add_with_negate_modifier() {
    // vs_2_0 { dcl_position v0; add r0, v0, -c0; mov oPos, r0; }
    let bc = vec![
        VS_HEADER,
        opcode_token(OP_DCL, 2),
        0x0000_0000,
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_ADD, 3),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 1), // modifier=neg
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let msl = emit_pair_for_tests(&bc, &red_constant_ps(), VariantKey::default());

    assert!(
        msl.contains("r[0] = (in.v0 + (-vs_c[0]));"),
        "add with neg modifier incorrect:\n{msl}"
    );
}

/// Build a src token that also carries a relative-addressing sub-token in the following u32.
///
/// The returned Vec contains two entries: the src token (with bit 13 set)
/// and the rel-addr token.
fn src_token_rel(
    reg_type: u32,
    index: u16,
    swizzle: u8,
    modifier: u8,
    rel_reg_type: u32,
    rel_index: u16,
    rel_swizzle: u8,
) -> [u32; 2] {
    let mut src = reg_bits(reg_type, index);
    src |= u32::from(swizzle) << 16;
    src |= (u32::from(modifier) & 0xF) << 24;
    src |= 1 << 13; // rel-addr flag
    let mut rel = reg_bits(rel_reg_type, rel_index);
    rel |= u32::from(rel_swizzle) << 16;
    [src, rel]
}

#[test]
fn mova_writes_address_register() {
    // vs_2_0 { dcl_position v0; mova a0, v0; mov oPos, v0; }
    let bc = vec![
        VS_HEADER,
        opcode_token(OP_DCL, 2),
        0x0000_0000,
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_MOVA, 2),
        dst_token(TYPE_ADDR, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let msl = emit_pair_for_tests(&bc, &red_constant_ps(), VariantKey::default());
    assert!(
        msl.contains("int4 a = int4(0);"),
        "address register declaration missing:\n{msl}"
    );
    assert!(
        msl.contains("a = int4(round(in.v0));"),
        "mova didn't emit round-to-int4 write:\n{msl}"
    );
}

#[test]
fn mova_respects_write_mask() {
    // vs_2_0 { dcl_position v0; mova a0.xy, v0; mov oPos, v0; }
    let bc = vec![
        VS_HEADER,
        opcode_token(OP_DCL, 2),
        0x0000_0000,
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_MOVA, 2),
        dst_token(TYPE_ADDR, 0, 0b0011, false), // .xy mask
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let msl = emit_pair_for_tests(&bc, &red_constant_ps(), VariantKey::default());
    assert!(
        msl.contains("a.xy = (int4(round(in.v0))).xy;"),
        "mova write mask not honored:\n{msl}"
    );
}

#[test]
fn reading_addr_register_casts_to_float4() {
    // vs_2_0 { dcl_position v0; mova a0, v0; mov r0, a0; mov oPos, v0; }
    let bc = vec![
        VS_HEADER,
        opcode_token(OP_DCL, 2),
        0x0000_0000,
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_MOVA, 2),
        dst_token(TYPE_ADDR, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_ADDR, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let msl = emit_pair_for_tests(&bc, &red_constant_ps(), VariantKey::default());
    assert!(
        msl.contains("r[0] = float4(a);"),
        "reading a0 didn't widen to float4:\n{msl}"
    );
}

#[test]
fn relative_addressed_const_read_uses_a() {
    // vs_2_0 { dcl_position v0; mova a0, v0; mov r0, c[a0.x + 5]; mov oPos, r0; }
    let mut bc = vec![
        VS_HEADER,
        opcode_token(OP_DCL, 2),
        0x0000_0000,
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_MOVA, 2),
        dst_token(TYPE_ADDR, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        // A rel-addr src is two tokens, so the opcode token-count reflects
        // that: dst (1) + src (2) = 3 operand tokens.
        opcode_token(OP_MOV, 3),
        dst_token(TYPE_TEMP, 0, 0xF, false),
    ];
    bc.extend_from_slice(&src_token_rel(
        TYPE_CONST,
        5,
        SWIZ_IDENTITY,
        0,
        TYPE_ADDR,
        0,
        SWIZ_XXXX,
    ));
    bc.extend_from_slice(&[
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ]);
    let msl = emit_pair_for_tests(&bc, &red_constant_ps(), VariantKey::default());
    assert!(
        msl.contains("r[0] = vs_c[a.x + 5];"),
        "relative-addressed const read not emitted correctly:\n{msl}"
    );
    let vs = parse(&bc).expect("VS parse");
    assert!(
        vs.uses_relative_const_addressing(),
        "rel-addr on const must be detected — the draw path gates \
         the full-constant-buffer upload on this flag"
    );
}

#[test]
fn relative_addressed_const_read_overlays_def_constants() {
    // def c2, 0.25, 0.5, 0.75, 1.0; mova a0, v0; mov r0, c[a0.x + 0]; mov oPos, r0
    // The relative read must see the `def`'d c2 (which the app never uploads),
    // not the empty uniform slot — so it routes through the overlay helper.
    let mut bc = vec![
        VS_HEADER,
        opcode_token(OP_DCL, 2),
        0x0000_0000,
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_DEF, 5),
        dst_token(TYPE_CONST, 2, 0xF, false),
        f32::to_bits(0.25),
        f32::to_bits(0.5),
        f32::to_bits(0.75),
        f32::to_bits(1.0),
        opcode_token(OP_MOVA, 2),
        dst_token(TYPE_ADDR, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 3),
        dst_token(TYPE_TEMP, 0, 0xF, false),
    ];
    bc.extend_from_slice(&src_token_rel(
        TYPE_CONST,
        0,
        SWIZ_IDENTITY,
        0,
        TYPE_ADDR,
        0,
        SWIZ_XXXX,
    ));
    bc.extend_from_slice(&[
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ]);
    let msl = emit_pair_for_tests(&bc, &red_constant_ps(), VariantKey::default());
    assert!(
        msl.contains("mtld3d_const_rel(a.x + 0, vs_c)"),
        "rel-addr read with def constants must route through the overlay:\n{msl}"
    );
    assert!(
        msl.contains("case 2: return float4(0.25"),
        "overlay helper must carry the def constant value:\n{msl}"
    );
    assert!(
        !msl.contains("r[0] = vs_c[a.x + 0];"),
        "rel-addr read must NOT bypass the overlay when defs exist:\n{msl}"
    );
}

#[test]
fn ps_relative_const_addressing_emits_a_indexed_buffer() {
    // `load_src` emits `ps_c[a.<comp> + N]` when a const source carries a
    // rel-addr operand, mirroring the VS path, so the PS prologue must
    // declare `a` or the MSL fails to compile. SM2.x PS does not permit
    // relative const addressing, so this construct is synthetic — but SM3
    // PS does, and the emission shape is pinned here so the SM3 PS path
    // rests on a working SM2 one.
    //
    // ps_2_0 { dcl t0; mov r0, c[t0.x + 5]; mov oC0, r0; }
    let mut bc = vec![
        PS_HEADER,
        opcode_token(OP_DCL, 2),
        0x8000_0000, // POSITION usage (structural only on PS 2.0)
        dst_token(TYPE_ADDR, 0, 0xF, false),
        opcode_token(OP_MOV, 3),
        dst_token(TYPE_TEMP, 0, 0xF, false),
    ];
    bc.extend_from_slice(&src_token_rel(
        TYPE_CONST,
        5,
        SWIZ_IDENTITY,
        0,
        TYPE_ADDR,
        0,
        SWIZ_XXXX,
    ));
    bc.extend_from_slice(&[
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ]);
    let ps = parse(&bc).expect("PS parse");
    let ps_msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS");
    assert!(
        ps_msl.contains("int4 a = int4(0);"),
        "PS prologue must declare `a` so rel-addr `ps_c[a.<comp> + N]` compiles:\n{ps_msl}"
    );
    assert!(
        ps_msl.contains("ps_c[a.x + 5]"),
        "PS rel-addr emission shape:\n{ps_msl}"
    );
}

#[test]
fn ps_rel_addr_emission_has_preceding_a_declaration() {
    // Whenever `ps_c[a.` appears in emitted PS MSL, the `int4 a`
    // declaration must precede it in the same function. Defends against
    // future edits removing the declaration without spotting the rel-addr
    // emit path in `load_src`.
    let mut bc = vec![
        PS_HEADER,
        opcode_token(OP_DCL, 2),
        0x8000_0000,
        dst_token(TYPE_ADDR, 0, 0xF, false),
        opcode_token(OP_MOV, 3),
        dst_token(TYPE_TEMP, 0, 0xF, false),
    ];
    bc.extend_from_slice(&src_token_rel(
        TYPE_CONST,
        7,
        SWIZ_IDENTITY,
        0,
        TYPE_ADDR,
        0,
        SWIZ_XXXX,
    ));
    bc.extend_from_slice(&[
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ]);
    let ps = parse(&bc).expect("PS parse");
    let ps_msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS");
    let rel_pos = ps_msl
        .find("ps_c[a.")
        .expect("test bytecode must produce ps_c[a.<...>]");
    let a_decl_pos = ps_msl
        .find("int4 a")
        .expect("`int4 a` declaration missing in PS function");
    assert!(
        a_decl_pos < rel_pos,
        "`int4 a` must precede ps_c[a.<...>] usage:\n{ps_msl}"
    );
}

#[test]
fn uses_relative_const_addressing_is_false_for_static_reads() {
    // vs_2_0 { dcl_position v0; mov r0, c[5]; mov oPos, r0; }
    // Same program shape as above but with no rel-addr.
    let bc = vec![
        VS_HEADER,
        opcode_token(OP_DCL, 2),
        0x0000_0000,
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_CONST, 5, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let vs = parse(&bc).expect("VS parse");
    assert!(!vs.uses_relative_const_addressing());
    // Plain static read still reports the right max so the fast path
    // keeps the short CB upload.
    assert_eq!(vs.max_const_reg(), Some(5));
}

#[test]
fn cmp_emits_select_on_ge_zero() {
    // vs_2_0 { dcl_position v0; cmp r0, v0, c0, c1; mov oPos, v0; }
    let bc = vec![
        VS_HEADER,
        opcode_token(OP_DCL, 2),
        0x0000_0000,
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_CMP, 4),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONST, 1, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let msl = emit_pair_for_tests(&bc, &red_constant_ps(), VariantKey::default());
    // MSL select(s2, s1, s0 >= 0): cond=true returns s1 (=vs_c[0]).
    assert!(msl.contains("select("), "cmp must emit select(): {msl}");
    assert!(
        msl.contains(">= float4(0.0)"),
        "cmp condition missing: {msl}"
    );
}

#[test]
fn slt_emits_step_complement() {
    // vs_2_0 { dcl_position v0; slt r0, v0, c0; mov oPos, v0; }
    let bc = vec![
        VS_HEADER,
        opcode_token(OP_DCL, 2),
        0x0000_0000,
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_SLT, 3),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let msl = emit_pair_for_tests(&bc, &red_constant_ps(), VariantKey::default());
    // slt s0, s1 → (s0 < s1) ? 1 : 0 = 1 - step(s1, s0).
    assert!(
        msl.contains("(float4(1.0) - step(vs_c[0], in.v0))"),
        "slt emission: {msl}"
    );
}

#[test]
fn sge_emits_step() {
    // vs_2_0 { dcl_position v0; sge r0, v0, c0; mov oPos, v0; }
    let bc = vec![
        VS_HEADER,
        opcode_token(OP_DCL, 2),
        0x0000_0000,
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_SGE, 3),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let msl = emit_pair_for_tests(&bc, &red_constant_ps(), VariantKey::default());
    // sge s0, s1 → (s0 >= s1) ? 1 : 0 = step(s1, s0).
    assert!(msl.contains("step(vs_c[0], in.v0)"), "sge emission: {msl}");
}

#[test]
fn m3x2_emits_two_dp3s() {
    // vs_2_0 { dcl_position v0; m3x2 r0, v0, c0; mov oPos, v0; }
    let bc = vec![
        VS_HEADER,
        opcode_token(OP_DCL, 2),
        0x0000_0000,
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_M3X2, 3),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let msl = emit_pair_for_tests(&bc, &red_constant_ps(), VariantKey::default());
    // Two dp3 calls against vs_c[0] and vs_c[1], remaining lanes zeroed.
    // `.xyz` swizzle on both operands matches the DP3 lowering shape.
    assert!(msl.contains("dot((in.v0).xyz, (vs_c[0]).xyz)"), "{msl}");
    assert!(msl.contains("dot((in.v0).xyz, (vs_c[1]).xyz)"), "{msl}");
    // rows=2 pads with two 0.0 lanes.
    assert!(msl.contains("float4("), "{msl}");
}

#[test]
fn m3x4_emits_four_dp3s() {
    // vs_2_0 { dcl_position v0; m3x4 r0, v0, c0; mov oPos, v0; }
    let bc = vec![
        VS_HEADER,
        opcode_token(OP_DCL, 2),
        0x0000_0000,
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_M3X4, 3),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let msl = emit_pair_for_tests(&bc, &red_constant_ps(), VariantKey::default());
    // Four plain dp3 calls against rows vs_c[0..3], `.xyz` swizzle on both.
    for i in 0..4 {
        let expected = format!("dot((in.v0).xyz, (vs_c[{i}]).xyz)");
        assert!(msl.contains(&expected), "{msl}");
    }
}

#[test]
fn ps2_vreg_input_maps_to_color_not_position() {
    // PS 2.0 `dcl v0` encodes usage=0 (POSITION) in its DCL token — the
    // usage field is structural-only, the real semantic comes from the
    // register kind. A PS 2.0 v-reg read must resolve to `in.color0`
    // (interpolated vertex color), not Metal's `in.position` (fragment
    // screen coord): a composite shader that uses `v0.zzzz` / `v0.wwww`
    // as scene↔bloom LERP weight and bloom-squared gain would otherwise
    // read clip-space Z/W and blend a gradient across the whole screen.
    //
    // Mirrors the structure of a WoW composite PS:
    //   dcl t0; dcl t1; dcl v0;
    //   def c0, 1, 0, 0, 0;
    //   mov oC0, v0;  // write the input color straight through
    let bc = vec![
        PS_HEADER,
        // def c0, 1, 0, 0, 0
        opcode_token(OP_DEF, 5),
        dst_token(TYPE_CONST, 0, 0xF, false),
        f32::to_bits(1.0),
        f32::to_bits(0.0),
        f32::to_bits(0.0),
        f32::to_bits(0.0),
        // dcl t0 (usage token 0x80000000 = POSITION, structural only on PS 2.0)
        opcode_token(OP_DCL, 2),
        0x8000_0000,
        dst_token(TYPE_ADDR, 0, 0xF, false),
        // dcl t1
        opcode_token(OP_DCL, 2),
        0x8000_0000,
        dst_token(TYPE_ADDR, 1, 0xF, false),
        // dcl v0 (same POSITION-encoded usage; should still resolve to color0)
        opcode_token(OP_DCL, 2),
        0x8000_0000,
        dst_token(TYPE_INPUT, 0, 0xF, false),
        // mov oC0, v0
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let msl = emit_pair_for_tests(&trivial_passthrough_vs(), &bc, VariantKey::default());
    assert!(
        msl.contains("in.color0"),
        "PS v0 read should resolve to in.color0:\n{msl}"
    );
    assert!(
        !msl.contains("oC0 = in.position"),
        "PS v0 must map to a color input, not in.position:\n{msl}"
    );
}

#[test]
fn programmable_ps_emits_fog_blend_when_variant_fog_mode_set() {
    let variant = VariantKey {
        alpha_func: 0,
        fog_mode: 3, // D3DFOG_LINEAR
        fog_table_mode: 0,
        depth_sampler_mask: 0,
        depth_fetch_mask: 0,
        volume_sampler_mask: 0,
        tt_projected_mask: 0,
        flags: VariantFlags::empty(),
    };
    let msl = emit_pair_for_tests(&trivial_passthrough_vs(), &red_constant_ps(), variant);
    assert!(
        msl.contains("constant float4 *fog_data [[buffer(13)]]"),
        "programmable PS must bind fog_data on slot 13 when fog_mode != 0:\n{msl}"
    );
    assert!(
        msl.contains("mix(fog_data[0].rgb, oC0.rgb, saturate(in.fog.x))"),
        "programmable PS must blend fog color with oC0:\n{msl}"
    );
}

#[test]
fn programmable_ps_omits_fog_blend_when_variant_fog_mode_zero() {
    let msl = emit_pair_for_tests(
        &trivial_passthrough_vs(),
        &red_constant_ps(),
        VariantKey::default(),
    );
    assert!(!msl.contains("fog_data"), "{msl}");
    assert!(!msl.contains("in.fog.x"), "{msl}");
}

#[test]
fn vs_without_ofog_write_falls_back_to_output_specular_alpha() {
    // A VS that never writes oFog sources the fog factor from the OUTPUT
    // specular alpha — per the D3D9 spec the fallback is 1 - oD1.a.
    let msl = emit_pair_for_tests(
        &trivial_passthrough_vs(),
        &red_constant_ps(),
        VariantKey::default(),
    );
    assert!(
        msl.contains("out.fog = float4(out.color1.w);"),
        "VS without an oFog write must fall back to the output specular alpha:\n{msl}"
    );
}

#[test]
fn vs_writing_ofog_keeps_its_value() {
    // dcl_position v0; mov oPos, v0; mov oFog, v0 — RastOut index 1 is oFog.
    let bc = vec![
        VS_HEADER,
        opcode_token(OP_DCL, 2),
        0x0000_0000,
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 1, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let vs = parse(&bc).expect("VS parse");
    let msl = emit_vs_programmable(&vs).expect("emit");
    assert!(
        !msl.contains("out.fog = float4(out.color1.w);"),
        "a fog-writing VS must not emit the specular-alpha fallback:\n{msl}"
    );
    assert!(msl.contains("out.fog = "), "{msl}");
}

#[test]
fn crs_emits_cross_product() {
    // vs_2_0 { dcl_position v0; crs r0, v0, c0; mov oPos, v0; }
    let bc = vec![
        VS_HEADER,
        opcode_token(OP_DCL, 2),
        0x0000_0000,
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_CRS, 3),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let msl = emit_pair_for_tests(&bc, &red_constant_ps(), VariantKey::default());
    assert!(
        msl.contains("cross((in.v0).xyz, (vs_c[0]).xyz)"),
        "crs must emit cross() on .xyz operands:\n{msl}"
    );
}

#[test]
fn sgn_emits_sign_builtin() {
    // vs_2_0 { dcl_position v0; sgn r0, v0; mov oPos, v0; }
    let bc = vec![
        VS_HEADER,
        opcode_token(OP_DCL, 2),
        0x0000_0000,
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_SGN, 2),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let msl = emit_pair_for_tests(&bc, &red_constant_ps(), VariantKey::default());
    assert!(
        msl.contains("r[0] = sign(in.v0);"),
        "sgn must emit sign() on full float4:\n{msl}"
    );
}

#[test]
fn dp2add_emits_dot2_plus_scalar() {
    // vs_2_0 { dcl_position v0; dp2add r0, v0, c0, c1; mov oPos, v0; }
    let bc = vec![
        VS_HEADER,
        opcode_token(OP_DCL, 2),
        0x0000_0000,
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_DP2ADD, 4),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONST, 1, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let msl = emit_pair_for_tests(&bc, &red_constant_ps(), VariantKey::default());
    assert!(
        msl.contains("dot((in.v0).xy, (vs_c[0]).xy) + (vs_c[1]).x"),
        "dp2add must combine 2-wide dot with scalar add:\n{msl}"
    );
}

#[test]
fn dsx_dsy_emit_metal_derivatives() {
    // ps_2_0 { dcl t0; dsx r0, t0; dsy r1, t0; mov oC0, r0; }
    let bc = vec![
        PS_HEADER,
        opcode_token(OP_DCL, 2),
        0x8000_0000,
        dst_token(TYPE_ADDR, 0, 0xF, false),
        opcode_token(OP_DSX, 2),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_ADDR, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_DSY, 2),
        dst_token(TYPE_TEMP, 1, 0xF, false),
        src_token(TYPE_ADDR, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let msl = emit_pair_for_tests(&trivial_passthrough_vs(), &bc, VariantKey::default());
    assert!(msl.contains("dfdx(in.texcoord0)"), "dsx → dfdx:\n{msl}");
    assert!(msl.contains("dfdy(in.texcoord0)"), "dsy → dfdy:\n{msl}");
}

#[test]
fn lit_emits_lighting_coefficients() {
    // vs_2_0 { dcl_position v0; lit r0, v0; mov oPos, v0; }
    let bc = vec![
        VS_HEADER,
        opcode_token(OP_DCL, 2),
        0x0000_0000,
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_LIT, 2),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let msl = emit_pair_for_tests(&bc, &red_constant_ps(), VariantKey::default());
    assert!(
        msl.contains("max((in.v0).x, 0.0)"),
        "lit must compute max(src.x, 0) for the diffuse term:\n{msl}"
    );
    assert!(
        msl.contains("pow(max((in.v0).y, 0.0), (in.v0).w)"),
        "lit must compute pow(max(src.y,0), src.w) for the specular term:\n{msl}"
    );
    assert!(
        msl.contains("(in.v0).x > 0.0"),
        "lit must gate the specular term on src.x > 0:\n{msl}"
    );
}

#[test]
fn dst_emits_distance_attenuation_vector() {
    // vs_2_0 { dcl_position v0; dst r0, v0, c0; mov oPos, v0; }
    let bc = vec![
        VS_HEADER,
        opcode_token(OP_DCL, 2),
        0x0000_0000,
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_DST, 3),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let msl = emit_pair_for_tests(&bc, &red_constant_ps(), VariantKey::default());
    assert!(
        msl.contains("(in.v0).y * (vs_c[0]).y"),
        "dst[1] = src0.y * src1.y:\n{msl}"
    );
    assert!(msl.contains("(in.v0).z"), "dst[2] = src0.z:\n{msl}");
    assert!(msl.contains("(vs_c[0]).w"), "dst[3] = src1.w:\n{msl}");
}

#[test]
fn cnd_emits_conditional_select_on_half() {
    // ps_1_x style conditional move. Use SM2 PS for the host test.
    // ps_2_0 { dcl t0; cnd r0, t0, c0, c1; mov oC0, r0; }
    let bc = vec![
        PS_HEADER,
        opcode_token(OP_DEF, 5),
        dst_token(TYPE_CONST, 0, 0xF, false),
        f32::to_bits(1.0),
        f32::to_bits(0.0),
        f32::to_bits(0.0),
        f32::to_bits(1.0),
        opcode_token(OP_DEF, 5),
        dst_token(TYPE_CONST, 1, 0xF, false),
        f32::to_bits(0.0),
        f32::to_bits(1.0),
        f32::to_bits(0.0),
        f32::to_bits(1.0),
        opcode_token(OP_DCL, 2),
        0x8000_0000,
        dst_token(TYPE_ADDR, 0, 0xF, false),
        opcode_token(OP_CND, 4),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_ADDR, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONST, 1, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let msl = emit_pair_for_tests(&trivial_passthrough_vs(), &bc, VariantKey::default());
    assert!(
        msl.contains("select(c1, c0, (in.texcoord0).x > 0.5)"),
        "cnd must emit select gated on src0.x > 0.5:\n{msl}"
    );
}

#[test]
fn secondary_position_semantic_routes_through_position1_varying() {
    // vs_3_0 { dcl_position0 v0; dcl_position1 v1; dcl_position0 o0;
    //          dcl_position1 o1; mov o0,v0; mov o1,v1; }
    let vs_bc = vec![
        VS3_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 1),
        dst_token(TYPE_INPUT, 1, 0xF, false),
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_OUTPUT, 0, 0xF, false),
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 1),
        dst_token(TYPE_OUTPUT, 1, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_OUTPUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_OUTPUT, 1, 0xF, false),
        src_token(TYPE_INPUT, 1, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let vs = parse(&vs_bc).expect("VS3 parse");
    let vs_msl = emit_vs_programmable(&vs).expect("emit VS3");
    assert!(
        vs_msl.contains("out.position1 = in.v1;") || vs_msl.contains("out.position1 ="),
        "POSITION1 output must route to out.position1, not clobber out.position:\n{vs_msl}"
    );
    assert!(
        vs_msl.contains("float4 position1;"),
        "Varyings must carry the position1 field:\n{vs_msl}"
    );

    // ps_3_0 { dcl_position1 v0; mov oC0, v0; } — reads the position1 varying.
    let ps_bc = vec![
        PS3_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 1),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&ps_bc).expect("PS3 parse");
    let ps_msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS3");
    assert!(
        ps_msl.contains("in.position1"),
        "PS dcl_position1 must read the position1 varying:\n{ps_msl}"
    );
}

#[test]
fn cnd_ps_1_4_compares_per_component_and_ps_1_1_coissue_selects_src1() {
    // Build `cnd r0, t0, c0, c1; mov r0, r0` (the trailing mov keeps r0 as the
    // ps_1_x colour output) under a given header + control bits.
    const OP_CND: u16 = 80;
    let emit = |header: u32, cnd_extra: u32| {
        let bc = vec![
            header,
            opcode_token(OP_DEF, 5),
            dst_token(TYPE_CONST, 0, 0xF, false),
            f32::to_bits(0.0),
            f32::to_bits(1.0),
            f32::to_bits(0.0),
            f32::to_bits(1.0),
            opcode_token(OP_DEF, 5),
            dst_token(TYPE_CONST, 1, 0xF, false),
            f32::to_bits(1.0),
            f32::to_bits(0.0),
            f32::to_bits(1.0),
            f32::to_bits(1.0),
            opcode_token(OP_CND, 4) | cnd_extra,
            dst_token(TYPE_TEMP, 0, 0xF, false),
            src_token(TYPE_ADDR, 0, SWIZ_IDENTITY, 0), // ps_1_x t0
            src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
            src_token(TYPE_CONST, 1, SWIZ_IDENTITY, 0),
            END_TOKEN,
        ];
        let ps = parse(&bc).expect("PS parse");
        emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS")
    };
    // ps_1_4: per-component compare (whole-vector `> float4(0.5)`).
    let ps14 = emit(0xFFFF_0104, 0);
    assert!(
        ps14.contains("select(c1, c0, t[0] > float4(0.5))"),
        "ps_1_4 cnd must compare per component:\n{ps14}"
    );
    // ps_1_1 plain: scalar `.x > 0.5` broadcast.
    let ps11 = emit(0xFFFF_0101, 0);
    assert!(
        ps11.contains("select(c1, c0, (t[0]).x > 0.5)"),
        "ps_1_1 cnd must test the scalar .x lane:\n{ps11}"
    );
    // ps_1_1 co-issued (D3DSI_COISSUE, RGB write): selects src1 unconditionally.
    let ps11_coissue = emit(0xFFFF_0101, 0x4000_0000);
    assert!(
        !ps11_coissue.contains("> 0.5"),
        "co-issued non-alpha cnd must bypass the compare:\n{ps11_coissue}"
    );
    assert!(
        ps11_coissue.contains("r[0] = c0;"),
        "co-issued cnd must select src1 (c0) unconditionally:\n{ps11_coissue}"
    );
}

#[test]
fn expp_logp_lower_to_full_precision_builtins() {
    // ps 1.x partial-precision exp/log map to full-precision Metal
    // builtins on modern hardware.
    // vs_2_0 { dcl_position v0; expp r0, v0; logp r1, v0; mov oPos, v0; }
    let bc = vec![
        VS_HEADER,
        opcode_token(OP_DCL, 2),
        0x0000_0000,
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_EXPP, 2),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_LOGP, 2),
        dst_token(TYPE_TEMP, 1, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let msl = emit_pair_for_tests(&bc, &red_constant_ps(), VariantKey::default());
    assert!(
        msl.contains("exp2((in.v0).x)"),
        "expp must lower to exp2:\n{msl}"
    );
    assert!(
        msl.contains("log2(abs((in.v0).x))"),
        "logp must lower to log2:\n{msl}"
    );
}

#[test]
fn texldl_emits_sample_with_explicit_lod() {
    // ps_3_0 { dcl_2d s0; dcl t0; texldl r0, t0, s0; mov oC0, r0; }
    // texldl carries the LOD in coord.w.
    let bc = vec![
        PS3_HEADER,
        opcode_token(OP_DCL, 2),
        0x9000_0000, // dcl_2d sampler usage token (texture type 2D in bits 27..30)
        dst_token(10 /* TYPE_SAMPLER */, 0, 0xF, false),
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_TEXCOORD, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_TEXLDL, 3),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        src_token(10 /* TYPE_SAMPLER */, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("PS3 parse");
    let ps_msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS3");
    assert!(
        ps_msl.contains("s0.sample(samp0, (in.texcoord0).xy, level((in.texcoord0).w))"),
        "texldl must pass coord.w as level():\n{ps_msl}"
    );
}

#[test]
fn texldp_divides_coord_by_w_before_sampling() {
    // ps_2_0 { dcl_2d s0; dcl t0; texldp r0, t0, s0; mov oC0, r0; }
    // The D3DSI_TEXLD_PROJECT control bit (0x00010000) makes texld sample at
    // coord.xy / coord.w; plain texld (no bit) samples raw.
    const OP_TEXLD: u16 = 66;
    let proj = |bc_extra: u32| {
        let bc = vec![
            PS_HEADER,
            opcode_token(OP_DCL, 2),
            0x9000_0000, // dcl_2d s0
            dst_token(10 /* TYPE_SAMPLER */, 0, 0xF, false),
            opcode_token(OP_DCL, 2),
            dcl_usage_token(DCL_TEXCOORD, 0),
            dst_token(TYPE_INPUT, 0, 0xF, false),
            opcode_token(OP_TEXLD, 3) | bc_extra,
            dst_token(TYPE_TEMP, 0, 0xF, false),
            src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
            src_token(10 /* TYPE_SAMPLER */, 0, SWIZ_IDENTITY, 0),
            opcode_token(OP_MOV, 2),
            dst_token(TYPE_COLOROUT, 0, 0xF, false),
            src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
            END_TOKEN,
        ];
        let ps = parse(&bc).expect("PS parse");
        emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS")
    };
    // texldp: coordinate divided by its .w before the .xy sampler swizzle.
    let projected = proj(0x0001_0000);
    assert!(
        projected.contains(".w)).xy"),
        "texldp must divide the coord by .w before sampling:\n{projected}"
    );
    // Plain texld: no projective divide.
    let plain = proj(0);
    assert!(
        !plain.contains(".w)).xy"),
        "plain texld must not project:\n{plain}"
    );
}

#[test]
fn texldd_emits_sample_with_gradient2d_for_2d_sampler() {
    // ps_3_0 { dcl_2d s0; dcl t0; texldd r0, t0, s0, r1, r2; mov oC0, r0; }
    let bc = vec![
        PS3_HEADER,
        opcode_token(OP_DCL, 2),
        0x9000_0000,
        dst_token(10 /* TYPE_SAMPLER */, 0, 0xF, false),
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_TEXCOORD, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_TEXLDD, 5),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        src_token(10 /* TYPE_SAMPLER */, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_TEMP, 1, SWIZ_IDENTITY, 0),
        src_token(TYPE_TEMP, 2, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("PS3 parse");
    let ps_msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS3");
    assert!(
        ps_msl.contains("gradient2d((r[1]).xy, (r[2]).xy)"),
        "texldd on a 2D sampler must use gradient2d() with .xy gradients:\n{ps_msl}"
    );
}

#[test]
fn depth_sampler_mask_emits_depth2d_binding_and_widens_sample_result() {
    // ps_3_0 { dcl_2d s0; dcl t0; texld r0, t0, s0; mov oC0, r0; }
    // With depth_sampler_mask bit 0 set, the slot binding must be
    // `depth2d<float>` instead of `texture2d<float>`, and the sample
    // call must be wrapped in `float4(...)` so downstream code reading
    // `.xyzw` keeps compiling. Mirrors how WoW's shadow PS samples
    // a D24X8 texture bound via CreateTexture(DEPTHSTENCIL).
    let bc = vec![
        PS3_HEADER,
        opcode_token(OP_DCL, 2),
        0x9000_0000, // dcl_2d s0
        dst_token(10 /* TYPE_SAMPLER */, 0, 0xF, false),
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_TEXCOORD, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(66 /* OP_TEXLD */, 3),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        src_token(10 /* TYPE_SAMPLER */, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("PS3 parse");

    // Without the mask: standard color path.
    let plain = emit_ps_programmable(&ps, VariantKey::default()).expect("emit plain");
    assert!(
        plain.contains("texture2d<float> s0 [[texture(0)]]"),
        "plain bind must be texture2d<float>:\n{plain}"
    );
    assert!(
        !plain.contains("depth2d<float>"),
        "plain bind must NOT mention depth2d:\n{plain}"
    );
    assert!(
        plain.contains("s0.sample(samp0, (in.texcoord0).xy)"),
        "plain sample must be the bare s0.sample expression:\n{plain}"
    );
    assert!(
        !plain.contains("float4(s0.sample"),
        "plain sample must NOT be wrapped in float4():\n{plain}"
    );
    assert!(
        !plain.contains("saturate"),
        "color path must NOT saturate — the clamp is depth-branch only:\n{plain}"
    );

    // With the mask: depth2d binding + sample_compare with the reference
    // depth saturate()d to [0,1], wrapped in float4 for downstream .xyzw
    // reads. This is the D3D9 hardware-shadow PCF idiom — `tex2D(s_shadow,
    // float3(uv, z_ref))` against a depth-format texture returns the
    // comparison result, not raw depth. The saturate() replicates the
    // clamp a D24 UNORM target gives for free; Depth32Float (Apple
    // Silicon's only depth format) does not. See `sample_or_compare`.
    let depth_variant = VariantKey {
        depth_sampler_mask: 0b0001,
        depth_fetch_mask: 0,
        ..VariantKey::default()
    };
    let depth = emit_ps_programmable(&ps, depth_variant).expect("emit depth");
    assert!(
        depth.contains("depth2d<float> s0 [[texture(0)]]"),
        "depth bind must be depth2d<float>:\n{depth}"
    );
    assert!(
        !depth.contains("texture2d<float>"),
        "depth bind must NOT mention texture2d:\n{depth}"
    );
    assert!(
        depth.contains(
            "float4(s0.sample_compare(samp0, (in.texcoord0).xy, saturate((in.texcoord0).z), level(0)))"
        ),
        "depth sample must be sample_compare with saturate()d ref and level(0):\n{depth}"
    );
    assert!(
        !depth.contains("s0.sample(samp0,"),
        "depth slot must NOT use plain sample():\n{depth}"
    );

    // Smoke-compile under Metal. The structural assertions above don't
    // catch wrong sampler return types — a buggy emitter that drops
    // the `float4(...)` wrap would still match the `s0.sample_compare(...)`
    // substring, but Metal would refuse to compile because
    // sample_compare returns `float`, not `float4`. Run the same compile
    // the production unix .so uses so the bug surfaces here.
    metal_compile_or_fail(&depth);
}

#[test]
fn ps_sampler_index_8_emits_slot_8_binding_and_compiles() {
    // ps_3_0 { dcl_2d s8; dcl t0; texld r0, t0, s8; mov oC0, r0; }
    //
    // PS3.0 allows sampler slots s0–s15. WoW's HD shadow receivers declare
    // `dcl_2d s8` for a 4th cascade-shadow-map tile (the `else`-branch in
    // the cascade-select ladder), so the emitter must carry the
    // SetTexture(8, …) binding through. A cap of 8 PS sampler stages would
    // silently drop it, producing Apple "Missing Fragment Texture s8"
    // warnings and a visibly missing far-cascade shadow. This test locks
    // the emitter path so the s8 binding can never be quietly regressed.
    let bc = vec![
        PS3_HEADER,
        opcode_token(OP_DCL, 2),
        0x9000_0000, // dcl_2d sampler usage token (texture type 2D in bits 27..30)
        dst_token(10 /* TYPE_SAMPLER */, 8, 0xF, false),
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_TEXCOORD, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(66 /* OP_TEXLD */, 3),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        src_token(10 /* TYPE_SAMPLER */, 8, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("PS3 parse");

    // Plain color path — slot 8 must appear in the MSL signature and
    // be sampled by `s8.sample(samp8, …)`.
    let plain = emit_ps_programmable(&ps, VariantKey::default()).expect("emit plain");
    assert!(
        plain.contains("texture2d<float> s8 [[texture(8)]]"),
        "PS sampler 8 must bind at Metal slot 8:\n{plain}"
    );
    assert!(
        plain.contains("sampler samp8 [[sampler(8)]]"),
        "PS sampler-state 8 must bind at Metal sampler slot 8:\n{plain}"
    );
    assert!(
        plain.contains("s8.sample(samp8, (in.texcoord0).xy)"),
        "PS body must sample s8 via samp8:\n{plain}"
    );

    // Depth-mask path with bit 8 set — slot 8 becomes a depth2d binding
    // and the sample turns into sample_compare. Mirrors WoW's shadow
    // receiver path with the cascade-3 depth texture bound on stage 8.
    let depth_variant = VariantKey {
        depth_sampler_mask: 1 << 8,
        depth_fetch_mask: 0,
        ..VariantKey::default()
    };
    let depth = emit_ps_programmable(&ps, depth_variant).expect("emit depth");
    assert!(
        depth.contains("depth2d<float> s8 [[texture(8)]]"),
        "depth_sampler_mask bit 8 must rewrite slot 8 binding to depth2d:\n{depth}"
    );
    assert!(
        depth.contains("s8.sample_compare(samp8,"),
        "depth slot 8 must use sample_compare:\n{depth}"
    );

    // Smoke-compile both variants under Metal so a future emitter
    // refactor that breaks slot-8 codegen surfaces here instead of
    // shipping to the unix .so.
    metal_compile_or_fail(&plain);
    metal_compile_or_fail(&depth);
}

#[test]
fn fog_msl_compiles_under_metal() {
    use super::ff::{FfPsKey, FfStage, emit_ps_ff};
    // Every fog blend shape through a real Metal compile: `precise::exp`,
    // the `in.position` fragcoord read, and the two-row `fog_data` binding
    // must all be valid MSL on both the FF and programmable PS emitters.
    let ps_key = FfPsKey {
        stages: [FfStage {
            color_op: 1, // D3DTOP_DISABLE
            ..FfStage::default()
        }; 8],
        specular_add: false,
        tt_projected_mask: 0,
    };
    let variants = [
        (0u8, 1u8, false), // EXP, Z source
        (0, 2, true),      // EXP2, W source
        (0, 3, false),     // LINEAR, Z source
        (0, 3, true),      // LINEAR, W source
        (3, 0, false),     // vertex fog
    ];
    for (fog_mode, fog_table_mode, fog_source_w) in variants {
        let mut flags = VariantFlags::empty();
        flags.set(VariantFlags::FOG_SOURCE_W, fog_source_w);
        let variant = VariantKey {
            fog_mode,
            fog_table_mode,
            flags,
            ..VariantKey::default()
        };
        metal_compile_or_fail(&emit_ps_ff(&ps_key, variant));
        let ps = parse(&red_constant_ps()).expect("PS parse");
        let msl = emit_ps_programmable(&ps, variant).expect("emit PS");
        metal_compile_or_fail(&msl);
    }
}

/// Compile MSL through `MTLDevice::newLibraryWithSource_options_error`.
///
/// Uses the same options the production unix .so uses (`metal/shader.rs`).
/// Skips if no Metal device is available (headless / no-GPU test runner).
fn metal_compile_or_fail(msl: &str) {
    use objc2_foundation::NSString;
    use objc2_metal::{
        MTLCompileOptions, MTLCreateSystemDefaultDevice, MTLDevice, MTLLanguageVersion,
    };
    let Some(device) = MTLCreateSystemDefaultDevice() else {
        eprintln!("MTLCreateSystemDefaultDevice returned nil — skipping Metal-compile check");
        return;
    };
    let options = MTLCompileOptions::new();
    options.setLanguageVersion(MTLLanguageVersion::Version2_4);
    let source = NSString::from_str(msl);
    if let Err(err) = device.newLibraryWithSource_options_error(&source, Some(&options)) {
        panic!("MSL failed Metal compilation: {err}\n--- MSL ---\n{msl}");
    }
}

#[test]
fn texldd_uses_gradientcube_for_cube_sampler() {
    // ps_3_0 { dcl_cube s0; dcl t0; texldd r0, t0, s0, r1, r2; mov oC0, r0; }
    // Cube samplers consume float3 coord + float3 gradients.
    let bc = vec![
        PS3_HEADER,
        opcode_token(OP_DCL, 2),
        0x9800_0000, // dcl_cube — texture type CUBE = 3 in bits 27..30
        dst_token(10 /* TYPE_SAMPLER */, 0, 0xF, false),
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_TEXCOORD, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_TEXLDD, 5),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        src_token(10 /* TYPE_SAMPLER */, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_TEMP, 1, SWIZ_IDENTITY, 0),
        src_token(TYPE_TEMP, 2, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("PS3 parse");
    let ps_msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS3");
    assert!(
        ps_msl.contains("(in.texcoord0).xyz"),
        "cube samplers need .xyz coord:\n{ps_msl}"
    );
    assert!(
        ps_msl.contains("gradientcube((r[1]).xyz, (r[2]).xyz)"),
        "texldd on cube sampler must use gradientcube() with .xyz gradients:\n{ps_msl}"
    );
}

// ── SM3 ──

#[test]
fn sm3_vs_position_output_resolves_via_dcl() {
    // vs_3_0 { dcl_position o2; mov o2, v0; }
    // SM3 unifies outputs under reg type 11 (RegKind::Output); the
    // `dcl_position` carries the semantic — register index alone is not
    // sufficient (could be o0, o2, o7…).
    let bc = vec![
        VS3_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_OUTPUT, 2, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_OUTPUT, 2, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let vs = parse(&bc).expect("VS3 parse");
    let vs_msl = emit_vs_programmable(&vs).expect("emit VS3");
    assert!(
        vs_msl.contains("out.position = in.v0;"),
        "VS3 oN with dcl_position should resolve to out.position:\n{vs_msl}"
    );
}

#[test]
fn sm3_vs_outputs_via_texcoordout_kind_resolve_through_dcl() {
    // WoW's HLSL compiler ships SM3 outputs as `RegKind::TexcoordOut`
    // (D3DSPR_TEXCRDOUT, type 6) — the type aliases D3DSPR_OUTPUT in SM3,
    // with the dcl carrying the actual semantic. The output map keys on
    // (kind, index) so these resolve through the dcl; matching only
    // `RegKind::Output` (type 11) would leave them to fall through the SM2
    // default that maps TexcoordOut[0] to `out.texcoord0`, so a
    // dcl_position output would never write clip-space position and the
    // geometry would collapse.
    //
    // vs_3_0 { dcl_position oT0; mov oT0, v0; }
    let bc = vec![
        VS3_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_TEXCOORDOUT, 0, 0xF, false),
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEXCOORDOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let vs = parse(&bc).expect("VS3 parse");
    let vs_msl = emit_vs_programmable(&vs).expect("emit VS3");
    assert!(
        vs_msl.contains("out.position = in.v0;"),
        "VS3 dcl_position oT0 must resolve to out.position via dcl, not out.texcoord0:\n{vs_msl}"
    );
    assert!(
        !vs_msl.contains("out.texcoord0 = in.v0;"),
        "VS3 dcl_position oT0 must not fall back to texcoord0:\n{vs_msl}"
    );
}

#[test]
fn sm3_vs_color_and_texcoord_outputs_resolve_via_dcl() {
    // vs_3_0 {
    //     dcl_position o0;
    //     dcl_color0 o3;
    //     dcl_texcoord2 o5;
    //     mov o0, v0;
    //     mov o3, v0;
    //     mov o5, v0;
    // }
    let bc = vec![
        VS3_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_OUTPUT, 0, 0xF, false),
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_COLOR, 0),
        dst_token(TYPE_OUTPUT, 3, 0xF, false),
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_TEXCOORD, 2),
        dst_token(TYPE_OUTPUT, 5, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_OUTPUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_OUTPUT, 3, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_OUTPUT, 5, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let vs = parse(&bc).expect("VS3 parse");
    let vs_msl = emit_vs_programmable(&vs).expect("emit VS3");
    assert!(
        vs_msl.contains("out.position = in.v0;"),
        "VS3 dcl_position o0 → out.position:\n{vs_msl}"
    );
    assert!(
        vs_msl.contains("out.color0 = in.v0;"),
        "VS3 dcl_color0 o3 → out.color0 (varying = usage_index, not reg index):\n{vs_msl}"
    );
    assert!(
        vs_msl.contains("out.texcoord2 = in.v0;"),
        "VS3 dcl_texcoord2 o5 → out.texcoord2:\n{vs_msl}"
    );
}

#[test]
fn sm3_ps_input_texcoord_resolves_via_dcl() {
    // ps_3_0 { dcl_texcoord3 v5; mov oC0, v5; }
    // The PS input map must distinguish color from texcoord by the dcl
    // usage. SM2 always assumed color — that breaks SM3.
    let bc = vec![
        PS3_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_TEXCOORD, 3),
        dst_token(TYPE_INPUT, 5, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 5, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("PS3 parse");
    let ps_msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS3");
    assert!(
        ps_msl.contains("oC0 = in.texcoord3;"),
        "PS3 v5 with dcl_texcoord3 must read from in.texcoord3:\n{ps_msl}"
    );
    assert!(
        !ps_msl.contains("oC0 = in.color5"),
        "PS3 v5 must not fall back to in.color5 (SM2 assumption):\n{ps_msl}"
    );
}

#[test]
fn sm3_ps_input_color_uses_usage_index_not_reg_index() {
    // ps_3_0 { dcl_color2 v7; mov oC0, v7; }
    // Varying slot is `usage_index` (=2), not `reg.index` (=7).
    let bc = vec![
        PS3_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_COLOR, 2),
        dst_token(TYPE_INPUT, 7, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 7, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("PS3 parse");
    let ps_msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS3");
    assert!(
        ps_msl.contains("oC0 = in.color2;"),
        "PS3 dcl_color2 v7 must read from in.color2:\n{ps_msl}"
    );
}

#[test]
fn sincos_emits_cos_sin_pair() {
    // vs_3_0 { dcl_position v0; sincos r1.xy, r0.x; mov oPos, v0; }
    // SM3 single-source form: dst.x = cos(src.x), dst.y = sin(src.x).
    // Write mask 0b0011 picks only .xy from the float4 we synthesize.
    const OP_SINCOS: u16 = 37;
    let bc = vec![
        VS3_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_SINCOS, 2),
        dst_token(TYPE_TEMP, 1, 0b0011, false),
        src_token(TYPE_TEMP, 0, 0x00 /* .xxxx */, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let vs = parse(&bc).expect("VS3 parse");
    let vs_msl = emit_vs_programmable(&vs).expect("emit VS3");
    assert!(
        vs_msl.contains("cos(") && vs_msl.contains("sin("),
        "sincos must emit both cos() and sin() builtins:\n{vs_msl}"
    );
    assert!(
        vs_msl.contains("r[1].xy"),
        "sincos write_mask 0b0011 must land in r[1].xy:\n{vs_msl}"
    );
}

#[test]
fn call_inline_expands_subroutine_body() {
    // vs_3_0 {
    //   dcl_position v0;
    //   dcl_position oT0;
    //   call l0;
    //   ret;
    //   label l0;
    //     mov oT0, v0;
    //   ret;
    // }
    let bc = vec![
        VS3_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_TEXCOORDOUT, 0, 0xF, false),
        opcode_token(OP_CALL, 1),
        src_token(TYPE_LABEL, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_RET, 0),
        opcode_token(OP_LABEL, 1),
        src_token(TYPE_LABEL, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEXCOORDOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_RET, 0),
        END_TOKEN,
    ];
    let vs = parse(&bc).expect("VS3 parse");
    let vs_msl = emit_vs_programmable(&vs).expect("emit VS3");
    assert!(
        vs_msl.contains("out.position = in.v0;"),
        "subroutine body must inline-expand at the call site:\n{vs_msl}"
    );
}

#[test]
fn callnz_wraps_inlined_body_in_conditional() {
    // vs_3_0 {
    //   dcl_position v0;
    //   dcl_position oT0;
    //   callnz l0, c0;
    //   label l0;
    //     mov oT0, v0;
    //   ret;
    // }
    let bc = vec![
        VS3_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_TEXCOORDOUT, 0, 0xF, false),
        opcode_token(OP_CALLNZ, 2),
        src_token(TYPE_LABEL, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_LABEL, 1),
        src_token(TYPE_LABEL, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEXCOORDOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_RET, 0),
        END_TOKEN,
    ];
    let vs = parse(&bc).expect("VS3 parse");
    let vs_msl = emit_vs_programmable(&vs).expect("emit VS3");
    assert!(
        vs_msl.contains("if ((vs_c[0]).x != 0.0) {"),
        "callnz must gate the inlined body on the condition src:\n{vs_msl}"
    );
    assert!(
        vs_msl.contains("out.position = in.v0;"),
        "callnz body must inline-expand:\n{vs_msl}"
    );
}

#[test]
fn setp_lt_emits_componentwise_predicate_assignment() {
    // ps_3_0 { dcl t0; setp_lt p0, t0, t0; mov oC0, t0; }
    let setp_lt_token = u32::from(OP_SETP) | ((4u32) << 16) | (3u32 << 24); // cmp=Lt(4), token_count=3
    let bc = vec![
        PS3_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_TEXCOORD, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        setp_lt_token,
        dst_token(TYPE_PREDICATE, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("PS3 parse");
    let ps_msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS3");
    assert!(
        ps_msl.contains("bool4 p0 = bool4(false);"),
        "PS prologue must declare p0:\n{ps_msl}"
    );
    assert!(
        ps_msl.contains("p0 = (in.texcoord0 < in.texcoord0);"),
        "setp_lt must emit a componentwise bool4 assignment:\n{ps_msl}"
    );
}

#[test]
fn predicated_instruction_wraps_dst_write_in_p0_check() {
    // ps_3_0 {
    //   dcl t0;
    //   setp_lt p0, t0, t0;
    //   (p0) mov oC0, t0;
    // }
    // Token format for the predicated mov: opcode bits, predicate flag
    // (bit 28), token_count covering predicate operand + dst + src.
    let setp_lt_token = u32::from(OP_SETP) | ((4u32) << 16) | (3u32 << 24);
    let predicated_mov_token = u32::from(OP_MOV) | (1u32 << 28) | (3u32 << 24); // predicated, count=3
    let bc = vec![
        PS3_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_TEXCOORD, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        setp_lt_token,
        dst_token(TYPE_PREDICATE, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        predicated_mov_token,
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_PREDICATE, 0, 0x00 /* .xxxx */, 0),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("PS3 parse");
    let ps_msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS3");
    assert!(
        ps_msl.contains("if (p0.x) {"),
        "predicated mov must gate the write on p0.x:\n{ps_msl}"
    );
    assert!(
        ps_msl.contains("oC0 = in.texcoord0;"),
        "the wrapped store_dst must still emit the underlying write:\n{ps_msl}"
    );
}

#[test]
fn breakp_emits_predicate_gated_break() {
    // vs_3_0 { defi i0, 4,0,1,0; loop aL, i0; (p0) breakp p0; endloop; ... }
    let predicated_breakp_token = u32::from(OP_BREAKP) | (1u32 << 28) | (1u32 << 24);
    let bc = vec![
        VS3_HEADER,
        opcode_token(OP_DEFI, 5),
        dst_token(TYPE_CONSTINT, 0, 0xF, false),
        4u32,
        0u32,
        1u32,
        0u32,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_LOOP, 2),
        src_token(TYPE_LOOP, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONSTINT, 0, SWIZ_IDENTITY, 0),
        predicated_breakp_token,
        src_token(TYPE_PREDICATE, 0, 0x00 /* .xxxx */, 0),
        opcode_token(OP_ENDLOOP, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEXCOORDOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let vs = parse(&bc).expect("VS3 parse");
    let vs_msl = emit_vs_programmable(&vs).expect("emit VS3");
    assert!(
        vs_msl.contains("if (p0.x) break;"),
        "breakp must gate break on the predicate operand:\n{vs_msl}"
    );
}

#[test]
fn defi_emits_int4_local() {
    // vs_3_0 { defi i0, 4, 0, 1, 0; dcl_position v0; mov oPos, v0; }
    let bc = vec![
        VS3_HEADER,
        opcode_token(OP_DEFI, 5),
        dst_token(TYPE_CONSTINT, 0, 0xF, false),
        4u32,
        0u32,
        1u32,
        0u32,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEXCOORDOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let vs = parse(&bc).expect("VS3 parse");
    let vs_msl = emit_vs_programmable(&vs).expect("emit VS3");
    assert!(
        vs_msl.contains("int4 i0 = int4(4, 0, 1, 0);"),
        "defi must emit an `int4 iN = int4(...)` local:\n{vs_msl}"
    );
}

#[test]
fn loop_emits_for_with_named_al_counter() {
    // vs_3_0 { defi i0, 4, 0, 1, 0; loop aL, i0; mov r0, c[aL]; endloop; }
    let bc = vec![
        VS3_HEADER,
        opcode_token(OP_DEFI, 5),
        dst_token(TYPE_CONSTINT, 0, 0xF, false),
        4u32,
        0u32,
        1u32,
        0u32,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_LOOP, 2),
        src_token(TYPE_LOOP, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONSTINT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_LOOP, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_ENDLOOP, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEXCOORDOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let vs = parse(&bc).expect("VS3 parse");
    let vs_msl = emit_vs_programmable(&vs).expect("emit VS3");
    assert!(
        vs_msl.contains("for (int aL_0 ="),
        "loop must allocate aL_0:\n{vs_msl}"
    );
    assert!(
        vs_msl.contains("aL_0 += _aL_step_0"),
        "loop step must reference its own counters:\n{vs_msl}"
    );
    assert!(
        vs_msl.contains("float4(aL_0)"),
        "RegKind::Loop reads inside the body must use aL_0:\n{vs_msl}"
    );
}

#[test]
fn loop_relative_const_addressing_indexes_by_al() {
    // vs_3_0 { defi i0, 4, 0, 1, 0; dcl_position v0;
    //          loop aL, i0; mov r0, c[aL + 8]; endloop;
    //          mov oTexcoord0, v0; }
    // `c[aL + N]` carries a relative-addressing token whose index register is
    // the loop counter (TYPE_LOOP), not the address register — it must resolve
    // to the enclosing loop's `aL_<n>` local, indexing the constant buffer.
    let mut bc = vec![
        VS3_HEADER,
        opcode_token(OP_DEFI, 5),
        dst_token(TYPE_CONSTINT, 0, 0xF, false),
        4u32,
        0u32,
        1u32,
        0u32,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_LOOP, 2),
        src_token(TYPE_LOOP, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONSTINT, 0, SWIZ_IDENTITY, 0),
        // mov r0, c[aL + 8] — dst (1) + rel-addr src (2) = 3 operand tokens.
        opcode_token(OP_MOV, 3),
        dst_token(TYPE_TEMP, 0, 0xF, false),
    ];
    bc.extend_from_slice(&src_token_rel(
        TYPE_CONST,
        8,
        SWIZ_IDENTITY,
        0,
        TYPE_LOOP,
        0,
        SWIZ_XXXX,
    ));
    bc.extend_from_slice(&[
        opcode_token(OP_ENDLOOP, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEXCOORDOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ]);
    let vs = parse(&bc).expect("VS3 parse");
    let vs_msl = emit_vs_programmable(&vs).expect("emit VS3");
    assert!(
        vs_msl.contains("r[0] = vs_c[aL_0 + 8];"),
        "c[aL + N] must index the constant buffer by the loop counter:\n{vs_msl}"
    );
    assert!(
        vs.uses_relative_const_addressing(),
        "rel-addr on const must be detected for the full-constant-buffer upload gate"
    );
}

#[test]
fn dynamic_int_constant_reads_the_runtime_vs_i_buffer() {
    // vs_3_0 { dcl_position v0; loop aL, i0; mov r0, c[aL + 8]; endloop;
    //          mov oTexcoord0, v0; }
    // i0 has NO `defi` — it is a dynamic integer constant fed by
    // SetVertexShaderConstantI, so the loop counter must read the runtime
    // `vs_i` buffer (slot 14), not a baked `int4 i0` local.
    let mut bc = vec![
        VS3_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_LOOP, 2),
        src_token(TYPE_LOOP, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONSTINT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 3),
        dst_token(TYPE_TEMP, 0, 0xF, false),
    ];
    bc.extend_from_slice(&src_token_rel(
        TYPE_CONST,
        8,
        SWIZ_IDENTITY,
        0,
        TYPE_LOOP,
        0,
        SWIZ_XXXX,
    ));
    bc.extend_from_slice(&[
        opcode_token(OP_ENDLOOP, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEXCOORDOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ]);
    let vs = parse(&bc).expect("VS3 parse");
    assert!(
        vs.uses_dynamic_int_constants(),
        "a non-defi iN read must be flagged as a dynamic integer constant"
    );
    let vs_msl = emit_vs_programmable(&vs).expect("emit VS3");
    assert!(
        vs_msl.contains("constant int4 *vs_i [[buffer(14)]]"),
        "a dynamic-int-const shader must declare the vs_i buffer at slot 14:\n{vs_msl}"
    );
    assert!(
        vs_msl.contains("vs_i[0]"),
        "the dynamic i0 loop counter must read vs_i[0]:\n{vs_msl}"
    );
    assert!(
        vs_msl.contains("vs_c[aL_0 + 8]"),
        "the aL-relative const read must still index vs_c by the loop counter:\n{vs_msl}"
    );
}

#[test]
fn defi_int_constant_stays_a_baked_local_without_vs_i() {
    // vs_3_0 { defi i0, 4, 0, 1, 0; dcl_position v0; loop aL, i0; mov r0, c0;
    //          endloop; mov oTexcoord0, v0; }
    // A defi'd i0 is a compile-time local; the shader must NOT declare vs_i.
    let bc = vec![
        VS3_HEADER,
        opcode_token(OP_DEFI, 5),
        dst_token(TYPE_CONSTINT, 0, 0xF, false),
        4u32,
        0u32,
        1u32,
        0u32,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_LOOP, 2),
        src_token(TYPE_LOOP, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONSTINT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_ENDLOOP, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEXCOORDOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let vs = parse(&bc).expect("VS3 parse");
    assert!(
        !vs.uses_dynamic_int_constants(),
        "a defi'd iN is static, not a dynamic integer constant"
    );
    let vs_msl = emit_vs_programmable(&vs).expect("emit VS3");
    assert!(
        !vs_msl.contains("vs_i"),
        "a defi-only shader must not declare or read vs_i:\n{vs_msl}"
    );
    assert!(
        vs_msl.contains("int4 i0 = int4(4, 0, 1, 0);"),
        "the defi'd i0 must stay a baked local:\n{vs_msl}"
    );
}

#[test]
fn rep_emits_for_loop_without_al() {
    // vs_3_0 { defi i0, 8, 0, 0, 0; rep i0; mov r0, r1; endrep; ... }
    let bc = vec![
        VS3_HEADER,
        opcode_token(OP_DEFI, 5),
        dst_token(TYPE_CONSTINT, 0, 0xF, false),
        8u32,
        0u32,
        0u32,
        0u32,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_REP, 1),
        src_token(TYPE_CONSTINT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_TEMP, 1, SWIZ_IDENTITY, 0),
        opcode_token(OP_ENDREP, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEXCOORDOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let vs = parse(&bc).expect("VS3 parse");
    let vs_msl = emit_vs_programmable(&vs).expect("emit VS3");
    assert!(
        vs_msl.contains("for (int _rep_0 = 0; _rep_0 < (float4(i0)).x; ++_rep_0) {"),
        "rep must emit a counted for loop:\n{vs_msl}"
    );
}

#[test]
fn break_emits_msl_break_inside_loop() {
    // vs_3_0 { defi i0, 4,0,1,0; loop aL, i0; break; endloop; ... }
    let bc = vec![
        VS3_HEADER,
        opcode_token(OP_DEFI, 5),
        dst_token(TYPE_CONSTINT, 0, 0xF, false),
        4u32,
        0u32,
        1u32,
        0u32,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_LOOP, 2),
        src_token(TYPE_LOOP, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONSTINT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_BREAK, 0),
        opcode_token(OP_ENDLOOP, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEXCOORDOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let vs = parse(&bc).expect("VS3 parse");
    let vs_msl = emit_vs_programmable(&vs).expect("emit VS3");
    assert!(
        vs_msl.contains("    break;\n"),
        "break must emit `break;`:\n{vs_msl}"
    );
}

#[test]
fn breakc_lt_emits_conditional_break() {
    // breakc_lt s0, s1 — opcode 45 with cmp = 4 (Lt) in bits 16-23.
    let breakc_lt_token = u32::from(OP_BREAKC) | ((4u32) << 16) | (2u32 << 24);
    // vs_3_0 { defi i0, 4,0,1,0; loop aL, i0; breakc_lt s0, s1; endloop; ... }
    let bc = vec![
        VS3_HEADER,
        opcode_token(OP_DEFI, 5),
        dst_token(TYPE_CONSTINT, 0, 0xF, false),
        4u32,
        0u32,
        1u32,
        0u32,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_LOOP, 2),
        src_token(TYPE_LOOP, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONSTINT, 0, SWIZ_IDENTITY, 0),
        breakc_lt_token,
        src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_TEMP, 1, SWIZ_IDENTITY, 0),
        opcode_token(OP_ENDLOOP, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEXCOORDOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let vs = parse(&bc).expect("VS3 parse");
    let vs_msl = emit_vs_programmable(&vs).expect("emit VS3");
    assert!(
        vs_msl.contains("if ((r[0]).x < (r[1]).x) break;"),
        "breakc_lt must emit a conditional break:\n{vs_msl}"
    );
}

#[test]
fn nested_loops_get_distinct_al_indices() {
    // Two nested `loop` blocks must allocate aL_0 and aL_1 so reads
    // bind to the correct enclosing scope.
    let bc = vec![
        VS3_HEADER,
        opcode_token(OP_DEFI, 5),
        dst_token(TYPE_CONSTINT, 0, 0xF, false),
        4u32,
        0u32,
        1u32,
        0u32,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_LOOP, 2),
        src_token(TYPE_LOOP, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONSTINT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_LOOP, 2),
        src_token(TYPE_LOOP, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONSTINT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_ENDLOOP, 0),
        opcode_token(OP_ENDLOOP, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEXCOORDOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let vs = parse(&bc).expect("VS3 parse");
    let vs_msl = emit_vs_programmable(&vs).expect("emit VS3");
    assert!(vs_msl.contains("aL_0"), "outer loop missing:\n{vs_msl}");
    assert!(
        vs_msl.contains("aL_1"),
        "inner loop must get distinct aL_1:\n{vs_msl}"
    );
}

#[test]
fn if_emits_msl_branch_on_x_lane() {
    // ps_3_0 { dcl t0; if t0; mov oC0, t0; endif; }
    let bc = vec![
        PS3_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_TEXCOORD, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_IF, 1),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_ENDIF, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("PS3 parse");
    let ps_msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS3");
    assert!(
        ps_msl.contains("if ((in.texcoord0).x != 0.0) {"),
        "`if src` must check src.x != 0:\n{ps_msl}"
    );
    assert!(
        ps_msl.contains("    }\n"),
        "endif must emit a closing brace:\n{ps_msl}"
    );
}

#[test]
fn ifc_lt_emits_msl_strict_less_than() {
    // Build an `ifc_lt` instruction: opcode 41 with cmp = 4 (Lt) in
    // bits 16-23 of the instruction token.
    // ps_3_0 { dcl t0; ifc_lt t0, t0; mov oC0, t0; endif; }
    let ifc_lt_token = u32::from(OP_IFC) | ((4u32) << 16) | (2u32 << 24); // cmp=4 (Lt), token_count=2
    let bc = vec![
        PS3_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_TEXCOORD, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        ifc_lt_token,
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_ENDIF, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("PS3 parse");
    let ps_msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS3");
    assert!(
        ps_msl.contains("if ((in.texcoord0).x < (in.texcoord0).x) {"),
        "ifc_lt must compare src0.x < src1.x:\n{ps_msl}"
    );
}

#[test]
fn if_else_endif_balanced_braces() {
    // ps_3_0 { dcl t0; if t0; mov oC0, t0; else; mov oC0, t0; endif; }
    let bc = vec![
        PS3_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_TEXCOORD, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_IF, 1),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_ELSE, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_ENDIF, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("PS3 parse");
    let ps_msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS3");
    assert!(
        ps_msl.contains("} else {"),
        "else must emit a `}} else {{`:\n{ps_msl}"
    );
    let opens = ps_msl.matches("if (").count();
    let closes = ps_msl.matches("    }\n").count();
    assert!(
        closes >= opens,
        "every `if (` must have a matching `}}`:\n{ps_msl}"
    );
}

#[test]
fn vs_writing_psize_via_opts_routes_through_storage_local() {
    // vs_2_0 { dcl_position v0; mov oPts, c0; mov oPos, v0; }
    // SM2 RastOut[2] is `oPts`. The Varyings field is scalar but the
    // emit goes through a `_psize_storage` float4 so `store_dst`'s
    // write-mask path stays uniform.
    let bc = vec![
        VS_HEADER,
        opcode_token(OP_DCL, 2),
        0x0000_0000,
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 2, 0xF, false),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let vs = parse(&bc).expect("VS parse");
    let vs_msl = emit_vs_programmable(&vs).expect("emit VS");
    assert!(
        vs_msl.contains("float point_size [[point_size]]"),
        "Varyings must declare a [[point_size]] field:\n{vs_msl}"
    );
    assert!(
        vs_msl.contains("float4 _psize_storage = float4(1.0);"),
        "VS prologue must default _psize_storage to 1.0:\n{vs_msl}"
    );
    assert!(
        vs_msl.contains("_psize_storage = vs_c[0];"),
        "oPts write must land in _psize_storage:\n{vs_msl}"
    );
    assert!(
        vs_msl.contains("out.point_size = _psize_storage.x;"),
        "VS epilogue must extract scalar point size from storage:\n{vs_msl}"
    );
}

#[test]
fn sm3_vs_writing_psize_via_dcl_routes_through_storage_local() {
    // vs_3_0 { dcl_position oT0; dcl_psize oT4; mov oT0, v0; mov oT4, c0; }
    let bc = vec![
        VS3_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_TEXCOORDOUT, 0, 0xF, false),
        opcode_token(OP_DCL, 2),
        dcl_usage_token(4 /* DeclUsage::PSize */, 0),
        dst_token(TYPE_TEXCOORDOUT, 4, 0xF, false),
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEXCOORDOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEXCOORDOUT, 4, 0xF, false),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let vs = parse(&bc).expect("VS3 parse");
    let vs_msl = emit_vs_programmable(&vs).expect("emit VS3");
    assert!(
        vs_msl.contains("_psize_storage = vs_c[0];"),
        "SM3 dcl_psize write must land in _psize_storage:\n{vs_msl}"
    );
    assert!(
        vs_msl.contains("out.point_size = _psize_storage.x;"),
        "VS epilogue must extract scalar point size:\n{vs_msl}"
    );
}

#[test]
fn ps_writing_odepth_returns_psout_struct_with_depth_field() {
    // ps_3_0 { dcl t0; mov oDepth, t0.x; mov oC0, t0; }
    // DepthOut writes flip the PS return type to a struct that
    // exposes both `oC0 [[color(0)]]` and `oDepth [[depth(any)]]`.
    const TYPE_DEPTHOUT: u32 = 9;
    let bc = vec![
        PS3_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_TEXCOORD, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_DEPTHOUT, 0, 0x1 /* .x only */, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("PS3 parse");
    let ps_msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS3");
    assert!(
        ps_msl.contains("struct PsOut"),
        "PS writing oDepth must emit a PsOut struct:\n{ps_msl}"
    );
    assert!(
        ps_msl.contains("float4 oC0 [[color(0)]];"),
        "PsOut must bind oC0 to color(0):\n{ps_msl}"
    );
    assert!(
        ps_msl.contains("float oDepth [[depth(any)]];"),
        "PsOut must bind oDepth to depth(any):\n{ps_msl}"
    );
    assert!(
        ps_msl.contains("fragment PsOut mtld3d_ps("),
        "fragment must return PsOut:\n{ps_msl}"
    );
    assert!(
        ps_msl.contains("_depth_storage"),
        "DepthOut writes must route through _depth_storage:\n{ps_msl}"
    );
    assert!(
        ps_msl.contains("_ps_out.oDepth = _depth_storage.x;"),
        "scalar oDepth value extracted from _depth_storage.x at return:\n{ps_msl}"
    );
    assert!(
        ps_msl.contains("return _ps_out;"),
        "fragment must return the struct:\n{ps_msl}"
    );
}

#[test]
fn ps_without_odepth_keeps_float4_return_for_simplicity() {
    // PS without oDepth writes stays on the bare-`float4` return path —
    // no struct, no extra storage local.
    let ps = parse(&red_constant_ps()).expect("PS parse");
    let ps_msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS");
    assert!(
        !ps_msl.contains("struct PsOut"),
        "no PsOut struct unless shader writes oDepth:\n{ps_msl}"
    );
    assert!(
        ps_msl.contains("fragment float4 mtld3d_ps("),
        "fragment stays float4 → return oC0:\n{ps_msl}"
    );
    assert!(
        !ps_msl.contains("_depth_storage"),
        "no _depth_storage local without oDepth:\n{ps_msl}"
    );
}

#[test]
fn sm3_ps_vpos_via_misctype_reads_in_position_no_duplicate_position() {
    // ps_3_0 { dcl_position vPos; mov oC0, vPos; }
    // SM3 dedicated `vPos` register lives on RegKind::MiscType index 0 and
    // reads the screen-space pixel coord — which IS the `[[position]]` the
    // `Varyings` struct already declares, so vPos reads `in.position`. A
    // SECOND `float4 v_pos [[position]]` fragment arg is a duplicate-
    // `[[position]]` MSL error that fails the shader to compile (it then
    // never renders). D3D9 vPos is the integer pixel coord vs Metal's
    // pixel-centre `[[position]]`, hence the `- 0.5`.
    const TYPE_MISCTYPE: u32 = 17;
    let bc = vec![
        PS3_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_MISCTYPE, 0, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_MISCTYPE, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("PS3 parse");
    let ps_msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS3");
    assert!(
        !ps_msl.contains("v_pos [[position]]"),
        "vPos must NOT add a second [[position]] arg (duplicate is an MSL error):\n{ps_msl}"
    );
    assert_eq!(
        ps_msl.matches("[[position").count(),
        1,
        "exactly one [[position...]] (the Varyings field):\n{ps_msl}"
    );
    assert!(
        ps_msl.contains("oC0 = (in.position - 0.5);"),
        "vPos read must resolve to (in.position - 0.5):\n{ps_msl}"
    );
}

#[test]
fn sm3_ps_vface_via_misctype_converts_bool_to_signed_float() {
    // ps_3_0 { dcl_face vFace; mov oC0, vFace; }
    // SM3 vFace lives on RegKind::MiscType index 1; D3D9 convention is
    // +1.0 for front-facing, -1.0 for back. MSL gives a bool through
    // [[front_facing]], so the prologue computes the signed float.
    const TYPE_MISCTYPE: u32 = 17;
    let bc = vec![
        PS3_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_MISCTYPE, 1, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_MISCTYPE, 1, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("PS3 parse");
    let ps_msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS3");
    assert!(
        ps_msl.contains("bool v_face_in [[front_facing]]"),
        "vFace dcl must add a [[front_facing]] fragment-function arg:\n{ps_msl}"
    );
    assert!(
        ps_msl.contains("float v_face = v_face_in ? 1.0 : -1.0;"),
        "vFace prologue must convert bool → ±1.0 float:\n{ps_msl}"
    );
    assert!(
        ps_msl.contains("oC0 = float4(v_face);"),
        "vFace read must broadcast the float to float4:\n{ps_msl}"
    );
}

#[test]
fn sm3_ps_without_misctype_dcl_omits_position_and_face_args() {
    // Shaders that don't use vPos/vFace must not pay the cost of always-on
    // [[position]] / [[front_facing]] args. Metal accepts them either way
    // but the emitted MSL stays minimal.
    let bc = vec![
        PS3_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_COLOR, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("PS3 parse");
    let ps_msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS3");
    assert!(
        !ps_msl.contains("[[position]]") || ps_msl.matches("[[position]]").count() == 1,
        "Varyings already has one [[position]] field; no extra:\n{ps_msl}"
    );
    assert!(
        !ps_msl.contains("[[front_facing]]"),
        "no [[front_facing]] arg unless shader declares vFace:\n{ps_msl}"
    );
    assert!(
        !ps_msl.contains("v_face"),
        "no v_face local unless shader declares vFace:\n{ps_msl}"
    );
}

#[test]
fn sm3_ps_input_fog_resolves_to_fog_varying() {
    // ps_3_0 { dcl_fog v3; mov oC0, v3; }
    // The PS3 fog varying must read `in.fog`, mirroring VS3 `dcl_fog oN`
    // writes. Without a dedicated fog arm the input hits the wildcard and
    // returns `in.color0`, mis-sampling distant fog.
    const DCL_FOG: u8 = 11;
    let bc = vec![
        PS3_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_FOG, 0),
        dst_token(TYPE_INPUT, 3, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 3, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("PS3 parse");
    let ps_msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS3");
    assert!(
        ps_msl.contains("oC0 = in.fog;"),
        "PS3 dcl_fog v3 must read from in.fog:\n{ps_msl}"
    );
}

#[test]
fn sm3_ps_input_position_resolves_to_position_varying() {
    // ps_3_0 { dcl_position v0; mov oC0, v0; }
    let bc = vec![
        PS3_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("PS3 parse");
    let ps_msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS3");
    assert!(
        ps_msl.contains("oC0 = in.position;"),
        "PS3 dcl_position v0 must read from in.position (screen-space coord post-rasterizer):\n{ps_msl}"
    );
}

#[test]
fn sm2_ps_input_mapping_unaffected_by_sm3_changes() {
    // Companion to `ps2_vreg_input_maps_to_color_not_position`: SM2 PS with
    // `dcl v0` (Position usage encoded structurally) must still resolve to
    // in.color0, not in.texcoord0 or in.position.
    let msl = emit_pair_for_tests(
        &trivial_passthrough_vs(),
        &red_constant_ps(),
        VariantKey::default(),
    );
    assert!(
        msl.contains("oC0 = c0;"),
        "SM2 PS must resolve oC0 to c0:\n{msl}"
    );
}

#[test]
fn dp4_emits_plain_dot() {
    // dp4 lowers to plain MSL `dot(a, b)`. Apple Silicon has hardware
    // dot-product; let the compiler use it. Cross-shader bit-invariance
    // is not the goal here — per-pipeline matrix bytes genuinely differ
    // between FF and programmable paths, so no emit-shape trick can
    // bridge it. The implicit decal depth bias in
    // `windows/d3d9/src/draw.rs` handles the actual symptom.
    //
    // vs_2_0 { dcl_position v0; dp4 r0.x, v0, c0; mov oPos, r0; }
    let bc = vec![
        VS_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_DP4, 3),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let msl = emit_pair_for_tests(&bc, &red_constant_ps(), VariantKey::default());
    assert!(
        msl.contains("dot(in.v0, vs_c[0])"),
        "dp4 must lower to plain MSL dot():\n{msl}"
    );
    assert!(
        !msl.contains("fma_dot4_invariant"),
        "fma_dot4_invariant helper should be gone:\n{msl}"
    );
}

#[test]
fn dp3_emits_plain_dot() {
    // Same rationale as `dp4_emits_plain_dot`. Plain `dot()` on the
    // `.xyz` swizzle of both operands.
    //
    // vs_2_0 { dcl_position v0; dp3 r0.x, v0, c0; mov oPos, v0; }
    let bc = vec![
        VS_HEADER,
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_POSITION, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_DP3, 3),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let msl = emit_pair_for_tests(&bc, &red_constant_ps(), VariantKey::default());
    assert!(
        msl.contains("dot((in.v0).xyz, (vs_c[0]).xyz)"),
        "dp3 must lower to plain MSL dot() on .xyz swizzle:\n{msl}"
    );
    assert!(
        !msl.contains("fma_dot3_invariant"),
        "fma_dot3_invariant helper should be gone:\n{msl}"
    );
}

#[test]
fn vertex_blend_msl_compiles_under_metal() {
    use super::ff::{FfVsFlags, FfVsKey, emit_vs_ff};
    // Exercise the three blend shapes through a real Metal compile so the
    // emitted code is syntactically and semantically valid MSL — same
    // discipline as `every_emitted_msl_compiles_under_metal` for the SM3
    // corpus.
    let mut sequential = FfVsKey {
        flags: FfVsFlags::HAS_NORMAL | FfVsFlags::COLOR_VERTEX,
        input_tex_coord_count: 0,
        tex_coord_count: 0,
        light_active_mask: 0,
        light_directional_mask: 0,
        light_spot_mask: 0,
        diffuse_source: 1,
        ambient_source: 0,
        specular_source: 2,
        emissive_source: 0,
        fog_mode: 0,
        tci_modes: [0; 8],
        tci_coord_indices: [0; 8],
        tex_coord_dims: [0; 8],
        tt_flags: [0; 8],
        vertex_blend_count: 3,
        declared_weights_count: 2,
    };
    metal_compile_or_fail(&emit_vs_ff(&sequential));

    sequential.vertex_blend_count = 4;
    sequential
        .flags
        .insert(FfVsFlags::VERTEX_BLEND_INDEXED | FfVsFlags::DECLARED_INDICES);
    sequential.declared_weights_count = 3;
    metal_compile_or_fail(&emit_vs_ff(&sequential));

    let indexed_only = FfVsKey {
        vertex_blend_count: 1,
        declared_weights_count: 0,
        ..sequential
    };
    metal_compile_or_fail(&emit_vs_ff(&indexed_only));
}

#[test]
fn ff_vs_lit_specular_msl_compiles_under_metal() {
    use super::ff::{FfVsFlags, FfVsKey, emit_vs_ff};
    // Lit + specular + one directional, one point, and one spot light,
    // through a real Metal compile — covers the Blinn-Phong block, the
    // per-light specular-row reads, and the spot cone factor.
    let key = FfVsKey {
        flags: FfVsFlags::HAS_NORMAL
            | FfVsFlags::COLOR_VERTEX
            | FfVsFlags::LIGHTING_ENABLED
            | FfVsFlags::SPECULAR_ENABLE,
        input_tex_coord_count: 0,
        tex_coord_count: 0,
        light_active_mask: 0b111,
        light_directional_mask: 0b001,
        light_spot_mask: 0b100,
        diffuse_source: 1,
        ambient_source: 0,
        specular_source: 2,
        emissive_source: 0,
        fog_mode: 0,
        tci_modes: [0; 8],
        tci_coord_indices: [0; 8],
        tex_coord_dims: [0; 8],
        tt_flags: [0; 8],
        vertex_blend_count: 0,
        declared_weights_count: 0,
    };
    metal_compile_or_fail(&emit_vs_ff(&key));
}

#[test]
fn ff_ps_specular_add_msl_compiles_under_metal() {
    use super::ff::{FfPsKey, FfStage, emit_ps_ff};
    // End-of-cascade specular add plus a D3DTA_SPECULAR stage argument,
    // through a real Metal compile.
    let mut stages = [FfStage {
        color_op: 1, // D3DTOP_DISABLE
        ..FfStage::default()
    }; 8];
    stages[0] = FfStage {
        color_op: 2,   // D3DTOP_SELECTARG1
        color_arg1: 4, // D3DTA_SPECULAR
        alpha_op: 2,   // D3DTOP_SELECTARG1
        alpha_arg1: 0, // D3DTA_DIFFUSE
        ..FfStage::default()
    };
    let key = FfPsKey {
        stages,
        specular_add: true,
        tt_projected_mask: 0,
    };
    metal_compile_or_fail(&emit_ps_ff(&key, VariantKey::default()));
}

#[test]
fn texkill_ps20_full_mask_kills_on_all_components() {
    // ps_2_0 { def c0, 1, 0, 0, 1; texkill r0; mov oC0, c0; }
    // r0 is uninitialised — runtime behaviour is not the point here, only
    // that the emitted MSL reads r0 with the full write-mask (`.xyzw`).
    // Decoding the operand in SRC form instead would turn the mask bits
    // into a `.wwxx` swizzle.
    let ps_bc = vec![
        PS_HEADER,
        opcode_token(OP_DEF, 5),
        dst_token(TYPE_CONST, 0, 0xF, false),
        f32::to_bits(1.0),
        f32::to_bits(0.0),
        f32::to_bits(0.0),
        f32::to_bits(1.0),
        opcode_token(OP_TEXKILL, 1),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&ps_bc).expect("PS parse");
    let ps_msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS");
    assert!(
        ps_msl.contains("if (any((r[0]).xyzw < 0.0)) discard_fragment();"),
        "texkill r0 must read r0 with full .xyzw mask:\n{ps_msl}"
    );
    assert!(
        !ps_msl.contains(".wwxx"),
        "texkill must not leak the SRC-form .wwxx swizzle:\n{ps_msl}"
    );
    metal_compile_or_fail(&ps_msl);
}

#[test]
fn texkill_ps20_partial_mask_honored() {
    // ps_2_0 { def c0, 1, 0, 0, 1; texkill r0.xyz; mov oC0, c0; }
    // SM2+ honors the dst write_mask — without it, some
    // post-processing shaders (e.g. ENB-style effect chains) kill
    // every pixel.
    let ps_bc = vec![
        PS_HEADER,
        opcode_token(OP_DEF, 5),
        dst_token(TYPE_CONST, 0, 0xF, false),
        f32::to_bits(1.0),
        f32::to_bits(0.0),
        f32::to_bits(0.0),
        f32::to_bits(1.0),
        opcode_token(OP_TEXKILL, 1),
        dst_token(TYPE_TEMP, 0, 0b0111, false), // .xyz
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&ps_bc).expect("PS parse");
    let ps_msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS");
    assert!(
        ps_msl.contains("if (any((r[0]).xyz < 0.0)) discard_fragment();"),
        "texkill r0.xyz must emit .xyz mask:\n{ps_msl}"
    );
    metal_compile_or_fail(&ps_msl);
}

#[test]
fn texld_dw_modifier_emits_perspective_divide() {
    // ps_2_0 { texld r0, t0_dw, s0; mov oC0, r0; }
    // FXC encodes HLSL `tex2Dproj(s0, t0)` as `texld` with the Dw
    // modifier on the coord source. The emitter must divide the whole
    // texcoord vector by `.w` before sampling — dropping the divide
    // mis-samples shadow cascades (foliage self-shadow flicker).
    // Modifier value 10 = Dw per `parser.rs`.
    const SRC_MOD_DW: u8 = 10;
    let ps_bc = vec![
        PS_HEADER,
        opcode_token(OP_DCL, 2),
        0x9000_0000, // dcl_2d s0
        dst_token(10 /* TYPE_SAMPLER */, 0, 0xF, false),
        opcode_token(66 /* OP_TEXLD */, 3),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_ADDR, 0, SWIZ_IDENTITY, SRC_MOD_DW), // t0 with Dw
        src_token(10 /* TYPE_SAMPLER */, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&ps_bc).expect("PS parse");
    let ps_msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit PS");
    assert!(
        ps_msl.contains("in.texcoord0") && ps_msl.contains("/ ("),
        "Dw modifier must emit a perspective divide:\n{ps_msl}"
    );
    assert!(
        ps_msl.contains(").w)"),
        "Dw modifier must divide by .w (not .z):\n{ps_msl}"
    );
    assert!(
        !ps_msl.contains("not implemented"),
        "Dw must no longer trigger the warn-and-passthrough stub:\n{ps_msl}"
    );
    metal_compile_or_fail(&ps_msl);
}

#[test]
fn depth_sample_compare_gets_level_zero_by_default() {
    // ps_3_0 { dcl_2d s0; dcl t0; texld r0, t0, s0; mov oC0, r0; }
    // Cascade shadow maps have no mips; `sample_compare` with implicit
    // gradients is undefined when neighbour fragments in the 2×2 quad
    // ran `discard_fragment` (alpha-cut foliage receiver). Force
    // `level(0)` so the mip pick doesn't depend on derivatives.
    let bc = vec![
        PS3_HEADER,
        opcode_token(OP_DCL, 2),
        0x9000_0000, // dcl_2d s0
        dst_token(10 /* TYPE_SAMPLER */, 0, 0xF, false),
        opcode_token(OP_DCL, 2),
        dcl_usage_token(DCL_TEXCOORD, 0),
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(66 /* OP_TEXLD */, 3),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        src_token(10 /* TYPE_SAMPLER */, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("PS parse");

    let depth_variant = VariantKey {
        depth_sampler_mask: 0b0001,
        depth_fetch_mask: 0,
        ..VariantKey::default()
    };
    let depth = emit_ps_programmable(&ps, depth_variant).expect("emit depth");
    assert!(
        depth.contains(", level(0)))"),
        "depth sample_compare must pin LOD with level(0):\n{depth}"
    );

    // Non-depth path must NOT acquire a level(0) — the implicit-gradient
    // fix only applies to sample_compare against a depth sampler.
    let plain = emit_ps_programmable(&ps, VariantKey::default()).expect("emit plain");
    assert!(
        !plain.contains("level(0)"),
        "non-depth s0.sample must not be pinned to level(0):\n{plain}"
    );
    metal_compile_or_fail(&depth);
}

// ── Shader Model 1 ──

const OP_TEXCOORD: u16 = 64;
const OP_TEX: u16 = 66;
const OP_TEXBEM: u16 = 67;
const OP_TEXDEPTH: u16 = 87;

#[test]
fn vs_1_1_passthrough_uses_implicit_position_output() {
    // vs_1_1 { dcl_position v0; mov oPos, v0; } — vs_1_1 has no dcl for the
    // implicit oPos output; the RastOut[0] register kind alone routes it.
    let bc = [
        0xFFFE_0101,
        opcode_token(OP_DCL, 2),
        0x0000_0000,
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let vs = parse(&bc).expect("vs_1_1 parse");
    let msl = emit_vs_programmable(&vs).expect("emit vs_1_1");
    assert!(
        msl.contains("out.position = in.v0;"),
        "vs_1_1 oPos write missing:\n{msl}"
    );
    metal_compile_or_fail(&msl);
}

#[test]
fn programmable_vs_emits_half_pixel_pos_fixup() {
    // Every DXSO VS declares the buffer-13 `pos_fixup` uniform and applies a
    // half-pixel window→NDC fixup in the position epilogue (after every
    // instruction, so per-op `oPos` writes stay verbatim) so on-boundary
    // geometry matches the D3D9 reference filling convention.
    let bc = [
        0xFFFE_0101,
        opcode_token(OP_DCL, 2),
        0x0000_0000,
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let vs = parse(&bc).expect("vs_1_1 parse");
    let msl = emit_vs_programmable(&vs).expect("emit vs_1_1");
    assert!(
        msl.contains("constant float4 &pos_fixup [[buffer(13)]]"),
        "VS must declare the pos_fixup uniform at slot 13:\n{msl}"
    );
    assert!(
        msl.contains("out.position.x += pos_fixup.x * out.position.w;")
            && msl.contains("out.position.y += pos_fixup.y * out.position.w;"),
        "VS must apply the half-pixel pos_fixup epilogue:\n{msl}"
    );
    // The `mov oPos, v0` still lands verbatim before the epilogue.
    assert!(
        msl.contains("out.position = in.v0;"),
        "oPos write must survive the epilogue:\n{msl}"
    );
    metal_compile_or_fail(&msl);
}

#[test]
fn ps_1_1_tex_and_texcoord_use_t_register_array() {
    // ps_1_1 { texcoord t0; tex t1; mov oC0, t1; }
    // t0 receives the (clamped) iterated texcoord; t1 samples stage 1 using
    // its own iterated coord and writes the result back to t1.
    let bc = [
        0xFFFF_0101,
        opcode_token(OP_TEXCOORD, 1),
        dst_token(TYPE_ADDR, 0, 0xF, false),
        opcode_token(OP_TEX, 1),
        dst_token(TYPE_ADDR, 1, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_ADDR, 1, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("ps_1_1 parse");
    let msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit ps_1_1");
    assert!(msl.contains("float4 t[8];"), "no t[] register file:\n{msl}");
    assert!(
        msl.contains("t[0] = in.texcoord0;"),
        "t[] not seeded from texcoord varyings:\n{msl}"
    );
    assert!(
        msl.contains("saturate(in.texcoord0)"),
        "texcoord must clamp the iterated coord:\n{msl}"
    );
    // `tex t1` samples stage 1 (implicit sampler) at the coord in t[1].
    assert!(
        msl.contains("s1.sample(samp1, (t[1]).xy)"),
        "tex must sample stage 1 from t[1]:\n{msl}"
    );
    assert!(
        msl.contains("texture2d<float> s1 [[texture(1)]]"),
        "implicit SM1 sampler not synthesized:\n{msl}"
    );
    metal_compile_or_fail(&msl);
}

#[test]
fn ps_1_1_implicit_r0_is_the_colour_output() {
    // ps_1_1 { tex t0; mov r0, t0; }
    // SM1 PS has no D3DSPR_COLOROUT register — the final pixel colour is
    // whatever the shader left in r0. The emitter must bridge `oC0 = r[0]`
    // after the body, or every such shader returns the float4(0.0) `oC0`
    // default (black).
    let bc = [
        0xFFFF_0101,
        opcode_token(OP_TEX, 1),
        dst_token(TYPE_ADDR, 0, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_ADDR, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("ps_1_1 parse");
    let msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit ps_1_1");
    assert!(
        msl.contains("oC0 = r[0];"),
        "SM1 PS must route r0 to the colour output:\n{msl}"
    );
    // The bridge must precede `return oC0;` so the returned colour is r0.
    let bridge = msl.find("oC0 = r[0];").expect("bridge present");
    let ret = msl.find("return oC0;").expect("return present");
    assert!(bridge < ret, "bridge must come before return:\n{msl}");
    metal_compile_or_fail(&msl);
}

#[test]
fn ps_1_4_texld_samples_destination_stage() {
    // ps_1_4 { texcrd r0, t0; phase; texld r1, r0; mov oC0, r1; }
    // ps_1_4 texld has no sampler operand — the sampler index is the dst
    // register number (r1 → sampler 1).
    const OP_PHASE: u16 = 0xFFFD; // D3DSIO_PHASE
    const OP_TEXLD: u16 = 66;
    let bc = [
        0xFFFF_0104,
        opcode_token(OP_TEXCOORD, 2),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_ADDR, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_PHASE, 0),
        opcode_token(OP_TEXLD, 2),
        dst_token(TYPE_TEMP, 1, 0xF, false),
        src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_TEMP, 1, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("ps_1_4 parse");
    let msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit ps_1_4");
    assert!(
        msl.contains("s1.sample(samp1, (r[0]).xy)"),
        "ps_1_4 texld must sample dst-numbered stage from the coord src:\n{msl}"
    );
    metal_compile_or_fail(&msl);
}

#[test]
fn ps_1_1_add_x2_result_modifier_scales() {
    // ps_1_1 { texcoord t0; add_x2 r0, t0, t0; mov oC0, r0; }
    // The `_x2` result modifier (shift_scale = +1) doubles the result.
    let mut add_dst = dst_token(TYPE_TEMP, 0, 0xF, false);
    add_dst |= 1 << 24; // shift_scale = +1 → ×2
    let bc = [
        0xFFFF_0101,
        opcode_token(OP_TEXCOORD, 1),
        dst_token(TYPE_ADDR, 0, 0xF, false),
        opcode_token(OP_ADD, 3),
        add_dst,
        src_token(TYPE_ADDR, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_ADDR, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("ps_1_1 parse");
    let msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit ps_1_1");
    assert!(
        msl.contains("* 2"),
        "add_x2 must scale the result by 2:\n{msl}"
    );
    metal_compile_or_fail(&msl);
}

#[test]
fn ps_1_1_texbem_emits_bump_uniform_and_perturb() {
    // ps_1_1 { tex t0; texbem t1, t0; mov oC0, t1; }
    // texbem perturbs stage 1's coord by the bump matrix applied to t0, then
    // samples stage 1. The per-stage bump matrix comes from buffer(12).
    let bc = [
        0xFFFF_0101,
        opcode_token(OP_TEX, 1),
        dst_token(TYPE_ADDR, 0, 0xF, false),
        opcode_token(OP_TEXBEM, 2),
        dst_token(TYPE_ADDR, 1, 0xF, false),
        src_token(TYPE_ADDR, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_ADDR, 1, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("ps_1_1 parse");
    let msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit ps_1_1");
    assert!(
        msl.contains("constant float4 *bump_env [[buffer(12)]]"),
        "texbem must bind the bump-env uniform on slot 12:\n{msl}"
    );
    assert!(
        msl.contains("bump_env[2]"),
        "texbem on stage 1 must read bump_env[2] (= stage*2):\n{msl}"
    );
    metal_compile_or_fail(&msl);
}

#[test]
fn ps_1_3_texdepth_writes_depth_output() {
    // ps_1_3 { tex t0; texdepth r5; } — texdepth writes fragment depth from
    // r5.x / r5.y, so the function must return the PsOut depth struct.
    let bc = [
        0xFFFF_0103,
        opcode_token(OP_TEX, 1),
        dst_token(TYPE_ADDR, 0, 0xF, false),
        opcode_token(OP_TEXDEPTH, 1),
        dst_token(TYPE_TEMP, 5, 0xF, false),
        END_TOKEN,
    ];
    let ps = parse(&bc).expect("ps_1_3 parse");
    let msl = emit_ps_programmable(&ps, VariantKey::default()).expect("emit ps_1_3");
    assert!(
        msl.contains("oDepth [[depth(any)]]") && msl.contains("_depth_storage"),
        "texdepth must route through the PsOut depth path:\n{msl}"
    );
    metal_compile_or_fail(&msl);
}

#[test]
fn constant_register_limits_reject_out_of_range_files() {
    // Addressing a constant register past the model's file must be caught
    // so CreateShader returns INVALIDCALL. The bytecode is hand-assembled
    // to use the out-of-range indices the native assembler refuses.

    // vs_1_1 { def c255; add r0, v0, c255; mov oPos, r0 } — c255 is the last
    // in-range vertex float constant.
    let vs_float_in_range = [
        0xFFFE_0101_u32,
        0x0000_001F,
        0x8000_0000,
        0x900F_0000,
        0x0000_0051,
        0xA00F_00FF,
        0x3F80_0000,
        0x3F80_0000,
        0x3F80_0000,
        0x3F80_0000,
        0x0000_0002,
        0x800F_0000,
        0x90E4_0000,
        0xA0E4_00FF,
        0x0000_0001,
        0xC00F_0000,
        0x80E4_0000,
        0x0000_FFFF,
    ];
    assert!(
        !parse(&vs_float_in_range)
            .expect("vs_float_in_range parse")
            .violates_constant_register_limits(),
        "c255 is the last in-range vertex float constant"
    );

    // Same shader at c256 — one past the 256-entry vertex float file.
    let vs_float_over = [
        0xFFFE_0101_u32,
        0x0000_001F,
        0x8000_0000,
        0x900F_0000,
        0x0000_0051,
        0xA00F_0100,
        0x3F80_0000,
        0x3F80_0000,
        0x3F80_0000,
        0x3F80_0000,
        0x0000_0002,
        0x800F_0000,
        0x90E4_0000,
        0xA0E4_0100,
        0x0000_0001,
        0xC00F_0000,
        0x80E4_0000,
        0x0000_FFFF,
    ];
    assert!(
        parse(&vs_float_over)
            .expect("vs_float_over parse")
            .violates_constant_register_limits(),
        "c256 overflows the vertex float file"
    );

    // vs_3_0 { defi i16; rep i16; add r0,r0,v0; endrep; mov o0,r0 } — i16 is one
    // past the 16-entry integer file.
    let vs_int_over = [
        0xFFFE_0300_u32,
        0x0200_001F,
        0x8000_0000,
        0x900F_0000,
        0x0200_001F,
        0x8000_0000,
        0xE00F_0000,
        0x0500_0030,
        0xF00F_0010,
        0x0000_0001,
        0x0000_0001,
        0x0000_0001,
        0x0000_0001,
        0x0100_0026,
        0xF0E4_0010,
        0x0300_0002,
        0x800F_0000,
        0x80E4_0000,
        0x90E4_0000,
        0x0000_0027,
        0x0200_0001,
        0xE00F_0000,
        0x80E4_0000,
        0x0000_FFFF,
    ];
    assert!(
        parse(&vs_int_over)
            .expect("vs_int_over parse")
            .violates_constant_register_limits(),
        "i16 overflows the integer constant file"
    );
}
