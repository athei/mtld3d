//! Unit tests for the DXSO parser.
//!
//! Hand-crafted bytecode covers the happy paths (SM2 vs/ps round-trip, def
//! constants, swizzles, modifiers) and the error paths (unsupported
//! versions, control flow, truncation).

use super::{
    ir::{
        DeclUsage, Declaration, DstMods, DxsoError, RegKind, ShaderType, SrcModifier, Swizzle,
        WriteMask,
    },
    opcode::Opcode,
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
const TYPE_COLOROUT: u32 = 8;
const TYPE_OUTPUT: u32 = 11;

const OP_MOV: u16 = 1;
const OP_DCL: u16 = 31;
const OP_DEF: u16 = 81;
const OP_CALL: u16 = 25;
const OP_IF: u16 = 40;
const OP_TEXKILL: u16 = 65;

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

#[test]
fn parse_minimal_vs_passthrough() {
    // vs_2_0 { dcl_position v0; mov oPos, v0; }
    let bc = [
        VS_HEADER,
        opcode_token(OP_DCL, 2),
        0x0000_0000, // usage_token: Position (usage=0, index=0)
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let prog = parse(&bc).expect("vs_2_0 passthrough should parse");

    assert_eq!(prog.shader_type, ShaderType::Vertex);
    assert_eq!(prog.major, 2);
    assert_eq!(prog.minor, 0);
    assert_eq!(prog.declarations.len(), 1);
    assert_eq!(prog.def_constants.len(), 0);
    assert_eq!(prog.instructions.len(), 1);

    match prog.declarations[0] {
        Declaration::Semantic {
            usage,
            usage_index,
            reg,
        } => {
            assert_eq!(usage, DeclUsage::Position);
            assert_eq!(usage_index, 0);
            assert_eq!(reg.kind, RegKind::Input);
            assert_eq!(reg.index, 0);
        }
        Declaration::Sampler { .. } => panic!("expected Semantic declaration"),
    }

    let inst = &prog.instructions[0];
    assert_eq!(inst.opcode, Opcode::Mov);
    let dst = inst.dst.expect("mov has dst");
    assert_eq!(dst.reg.kind, RegKind::RastOut);
    assert_eq!(dst.reg.index, 0);
    assert_eq!(dst.write_mask, WriteMask::ALL);
    assert!(!dst.mods.contains(DstMods::SATURATE));
    assert_eq!(inst.srcs.len(), 1);
    assert_eq!(inst.srcs[0].reg.kind, RegKind::Input);
    assert_eq!(inst.srcs[0].modifier, SrcModifier::None);
}

#[test]
fn parse_ps_def_then_mov_constant() {
    // ps_2_0 { def c0, 1, 0, 0, 1; mov oC0, c0; }
    let bc = [
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
    ];
    let prog = parse(&bc).expect("ps_2_0 def+mov should parse");

    assert_eq!(prog.shader_type, ShaderType::Pixel);
    assert_eq!(prog.def_constants.len(), 1);
    assert_eq!(prog.def_constants[0].reg.kind, RegKind::Const);
    assert_eq!(prog.def_constants[0].reg.index, 0);
    let value_bits = prog.def_constants[0].value.map(f32::to_bits);
    assert_eq!(value_bits, [1.0_f32, 0.0, 0.0, 1.0].map(f32::to_bits),);

    let inst = &prog.instructions[0];
    assert_eq!(inst.opcode, Opcode::Mov);
    assert_eq!(inst.dst.expect("mov has dst").reg.kind, RegKind::ColorOut);
}

#[test]
fn swizzle_and_modifier_decoded() {
    // mov r0, -c0.yyzw
    let bc = [
        VS_HEADER,
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_CONST, 0, 0b11_10_01_01, 1),
        END_TOKEN,
    ];
    let prog = parse(&bc).expect("parse");
    let src = &prog.instructions[0].srcs[0];
    assert_eq!(src.modifier, SrcModifier::Neg);
    assert_eq!(src.swizzle, Swizzle([1, 1, 2, 3]));
}

#[test]
fn saturate_flag_decoded() {
    let bc = [
        VS_HEADER,
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_TEMP, 0, 0xF, true),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let prog = parse(&bc).expect("parse");
    assert!(
        prog.instructions[0]
            .dst
            .expect("has dst")
            .mods
            .contains(DstMods::SATURATE)
    );
}

#[test]
fn comment_tokens_are_skipped() {
    let bc = [
        VS_HEADER,
        0xFFFE_u32 | (2u32 << 16), // comment opcode + 2-token payload
        0xDEAD_BEEF,
        0xCAFE_BABE,
        END_TOKEN,
    ];
    let prog = parse(&bc).expect("comment should parse as no-op");
    assert_eq!(prog.instructions.len(), 0);
    assert_eq!(prog.declarations.len(), 0);
}

#[test]
fn accept_vs_1_1_implicit_output() {
    // vs_1_1 { mov oPos, v0; } — no dcl for oPos (implicit in SM1).
    let bc = [
        0xFFFE_0101,
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let prog = parse(&bc).expect("vs_1_1 should now parse");
    assert_eq!(prog.shader_type, ShaderType::Vertex);
    assert_eq!(prog.major, 1);
    assert_eq!(prog.minor, 1);
    assert_eq!(prog.instructions.len(), 1);
    assert_eq!(prog.instructions[0].opcode, Opcode::Mov);
}

#[test]
fn parse_vs_1_1_real_bytecode_without_length_field() {
    // The exact vs_1_1 stream a real D3D compiler emits: opcode tokens carry
    // NO instruction-length field (bits
    // 24-27 are zero — that field arrived in SM2.0), and every dst/src token
    // sets bit 31. The operand count must come from the bit-31 run, not the
    // (absent) length field; otherwise each `mov` parses with no dst/src and
    // the emitter panics indexing an empty source list.
    let bc = [
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
    let prog = parse(&bc).expect("real vs_1_1 bytecode should parse");
    assert_eq!(prog.major, 1);
    assert_eq!(prog.instructions.len(), 2);

    let mov_pos = &prog.instructions[0];
    assert_eq!(mov_pos.opcode, Opcode::Mov);
    assert_eq!(
        mov_pos.dst.as_ref().expect("oPos dst").reg.kind,
        RegKind::RastOut
    );
    assert_eq!(mov_pos.srcs.len(), 1);
    assert_eq!(mov_pos.srcs[0].reg.kind, RegKind::Input);

    let mov_color = &prog.instructions[1];
    assert_eq!(mov_color.opcode, Opcode::Mov);
    assert_eq!(
        mov_color.dst.as_ref().expect("oD0 dst").reg.kind,
        RegKind::AttrOut
    );
    assert_eq!(mov_color.srcs.len(), 1);
    assert_eq!(mov_color.srcs[0].reg.kind, RegKind::Const);
}

#[test]
fn parse_vs_1_1_relative_addressing_has_no_extra_token() {
    // vs_1_1 { dcl_position v0; mov a0.x, c7.x; mov oD0, c[a0.x + 3]; mov oPos,
    // v0; } — the SM1 relative-addressing form (`c[a0.x + N]`). In SM1
    // `c[a0.x + N]` sets the source's relative bit (1<<13)
    // but carries NO separate relative-address token (a0 is implicit), unlike
    // SM2+. The decoder must consume only one token for that source; otherwise
    // it swallows the next opcode and the stream parses as Truncated.
    let bc = [
        0xFFFE_0101, // vs_1_1
        0x0000_001F,
        0x8000_0000,
        0x900F_0000, // dcl_position v0
        0x0000_0001,
        0xB001_0000,
        0xA000_0007, // mov a0.x, c7.x
        0x0000_0001,
        0xD00F_0000,
        0xA0E4_2003, // mov oD0, c[a0.x + 3]
        0x0000_0001,
        0xC00F_0000,
        0x90E4_0000, // mov oPos, v0
        0x0000_FFFF, // end
    ];
    let prog = parse(&bc).expect("SM1 relative addressing should parse");
    assert_eq!(prog.instructions.len(), 3);

    let rel_mov = &prog.instructions[1];
    assert_eq!(rel_mov.opcode, Opcode::Mov);
    assert_eq!(rel_mov.srcs.len(), 1);
    let rel = rel_mov.srcs[0]
        .rel_addr
        .as_ref()
        .expect("c[a0.x + 3] carries relative addressing");
    assert_eq!(rel.reg.kind, RegKind::Addr);
    assert_eq!(rel.reg.index, 0);
    // The final `mov oPos, v0` must survive intact — proof the relative source
    // did not swallow this instruction's opcode token.
    assert_eq!(prog.instructions[2].opcode, Opcode::Mov);
    assert_eq!(prog.instructions[2].srcs[0].reg.kind, RegKind::Input);
}

#[test]
fn accept_ps_1_4_and_drop_phase() {
    // ps_1_4 { texcrd r0, t0; phase; texld r0, r0; } — `phase` is a no-op
    // boundary and must be dropped, not parsed into an instruction.
    const OP_PHASE: u16 = 0xFFFD; // D3DSIO_PHASE
    const OP_TEXCOORD: u16 = 64;
    const OP_TEXLD: u16 = 66;
    let bc = [
        0xFFFF_0104,
        opcode_token(OP_TEXCOORD, 2),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_ADDR, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_PHASE, 0),
        opcode_token(OP_TEXLD, 2),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        src_token(TYPE_TEMP, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let prog = parse(&bc).expect("ps_1_4 should now parse");
    assert_eq!(prog.major, 1);
    assert_eq!(prog.minor, 4);
    // Two real instructions; `phase` left no trace.
    assert_eq!(prog.instructions.len(), 2);
    assert_eq!(prog.instructions[0].opcode, Opcode::TexCoord);
    assert_eq!(prog.instructions[1].opcode, Opcode::TexLd);
}

#[test]
fn parse_dst_shift_scale() {
    // ps_1_1 { add_x2 r0, t0, t0; } — the `_x2` result modifier rides in the
    // dst token's shift-scale nibble (bits 24-27); +1 == ×2.
    const OP_ADD: u16 = 2;
    let mut dst = dst_token(TYPE_TEMP, 0, 0xF, false);
    dst |= 1 << 24; // shift_scale = +1
    let bc = [
        0xFFFF_0101,
        opcode_token(OP_ADD, 3),
        dst,
        src_token(TYPE_ADDR, 0, SWIZ_IDENTITY, 0),
        src_token(TYPE_ADDR, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let prog = parse(&bc).expect("ps_1_1 should parse");
    assert_eq!(
        prog.instructions[0].dst.expect("add has dst").shift_scale,
        1
    );
}

#[test]
fn reject_unsupported_sm1_minor() {
    // vs_1_0 does not exist (VS starts at 1.1); ps_1_5 does not exist.
    assert_eq!(
        parse(&[0xFFFE_0100, END_TOKEN]).unwrap_err(),
        DxsoError::UnsupportedShaderModel { major: 1, minor: 0 },
    );
    assert_eq!(
        parse(&[0xFFFF_0105, END_TOKEN]).unwrap_err(),
        DxsoError::UnsupportedShaderModel { major: 1, minor: 5 },
    );
}

#[test]
fn accept_sm_3_0_vs_header() {
    // Covers the header gate only: SM3 VS bytecode is accepted and its
    // shader type and version are read back. The structural emit (output
    // decls, `vPos`, `vFace`, …) is covered by the emitter tests.
    let bc = [VS3_HEADER, END_TOKEN];
    let prog = parse(&bc).expect("SM3 VS header should now parse");
    assert_eq!(prog.shader_type, ShaderType::Vertex);
    assert_eq!(prog.major, 3);
    assert_eq!(prog.minor, 0);
}

#[test]
fn accept_sm_3_0_ps_header() {
    let bc = [PS3_HEADER, END_TOKEN];
    let prog = parse(&bc).expect("SM3 PS header should now parse");
    assert_eq!(prog.shader_type, ShaderType::Pixel);
    assert_eq!(prog.major, 3);
    assert_eq!(prog.minor, 0);
}

#[test]
fn parse_sm_3_0_vs_with_named_output() {
    // vs_3_0 { dcl_position o0; mov o0, v0; }
    // SM3 VS unifies its outputs under RegKind::Output (reg type 11);
    // the dcl_position carries the semantic.
    let bc = [
        VS3_HEADER,
        opcode_token(OP_DCL, 2),
        0x8000_0000, // usage_token: Position
        dst_token(TYPE_OUTPUT, 0, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_OUTPUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let prog = parse(&bc).expect("SM3 VS with dcl_position oN should parse");
    let dcl = prog
        .declarations
        .iter()
        .find_map(|d| match d {
            Declaration::Semantic {
                usage,
                usage_index,
                reg,
            } if reg.kind == RegKind::Output => Some((*usage, *usage_index, reg.index)),
            _ => None,
        })
        .expect("expected an Output dcl");
    assert_eq!(dcl, (DeclUsage::Position, 0, 0));
    assert_eq!(prog.instructions.len(), 1);
}

#[test]
fn accept_call_in_sm3() {
    // `call sN` parses as a regular instruction. The src is a
    // Label register pointing at the subroutine; the emitter
    // inline-expands at this site.
    let bc = [
        VS3_HEADER,
        opcode_token(OP_CALL, 1),
        // Label src — reg type 18 (0x12), index 0.
        ((0x12u32 & 0x7) << 28) | (((0x12u32 >> 3) & 0x3) << 11),
        END_TOKEN,
    ];
    let prog = parse(&bc).expect("call sN must parse");
    assert_eq!(prog.instructions.len(), 1);
    assert_eq!(prog.instructions[0].opcode, Opcode::Call);
}

#[test]
fn accept_if_in_sm3() {
    // SM2.x / SM3 `if` parses as a regular instruction; the emitter lowers it.
    let bc = [
        VS3_HEADER,
        opcode_token(OP_IF, 1),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let prog = parse(&bc).expect("`if` must parse without rejection");
    assert_eq!(prog.instructions.len(), 1);
    assert_eq!(prog.instructions[0].opcode, Opcode::If);
}

#[test]
fn reject_invalid_header_magic() {
    let bc = [0x1234_0200, END_TOKEN];
    assert_eq!(parse(&bc).unwrap_err(), DxsoError::InvalidHeader);
}

#[test]
fn reject_missing_end_token() {
    let bc: [u32; 1] = [VS_HEADER];
    assert_eq!(parse(&bc).unwrap_err(), DxsoError::Truncated);
}

#[test]
fn reject_unknown_opcode() {
    let bc = [VS_HEADER, opcode_token(200, 0), END_TOKEN];
    assert_eq!(parse(&bc).unwrap_err(), DxsoError::UnknownOpcode(200));
}

#[test]
fn parse_subroutine_separates_body_from_main() {
    // vs_3_0 { mov oPos, v0; ret; label l0; mov oPos, c0; ret; }
    // Main has the first mov; subroutine 0 owns the second.
    const OP_RET: u16 = 28;
    const OP_LABEL: u16 = 30;
    const TYPE_LABEL: u32 = 18;
    let label_src = (TYPE_LABEL & 0x7) << 28 | ((TYPE_LABEL >> 3) & 0x3) << 11;
    let bc = [
        VS3_HEADER,
        opcode_token(OP_DCL, 2),
        0x8000_0000,
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_RET, 0),
        opcode_token(OP_LABEL, 1),
        label_src,
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_CONST, 0, SWIZ_IDENTITY, 0),
        opcode_token(OP_RET, 0),
        END_TOKEN,
    ];
    let prog = parse(&bc).expect("label/ret subroutine must parse");
    assert_eq!(
        prog.instructions.len(),
        1,
        "main should hold only the pre-label mov, not the labelled body"
    );
    assert_eq!(
        prog.subroutines.len(),
        1,
        "one labelled subroutine collected"
    );
    let sub = prog.subroutines.get(&0).expect("label 0 collected");
    assert_eq!(sub.len(), 1, "subroutine body has one mov");
    assert_eq!(sub[0].opcode, Opcode::Mov);
}

#[test]
fn texkill_decodes_operand_as_dst_full_mask() {
    // ps_2_0 { texkill r0 }
    // The operand sits in DST-form (write mask in bits 16-19), not SRC-form,
    // so it must land in `inst.dst` (r0, mask=ALL) with `inst.srcs` empty.
    // Decoding it as a source would misread the mask bits as a swizzle.
    let bc = [
        PS_HEADER,
        opcode_token(OP_TEXKILL, 1),
        dst_token(TYPE_TEMP, 0, 0xF, false),
        END_TOKEN,
    ];
    let prog = parse(&bc).expect("ps_2_0 texkill r0 should parse");
    let inst = &prog.instructions[0];
    assert_eq!(inst.opcode, Opcode::TexKill);
    let dst = inst.dst.expect("texkill operand parsed as dst");
    assert_eq!(dst.reg.kind, RegKind::Temp);
    assert_eq!(dst.reg.index, 0);
    assert_eq!(dst.write_mask, WriteMask::ALL);
    assert!(inst.srcs.is_empty(), "texkill has no src operands");
}

#[test]
fn texkill_honors_partial_write_mask() {
    // ps_2_0 { texkill r1.xyz }
    let bc = [
        PS_HEADER,
        opcode_token(OP_TEXKILL, 1),
        dst_token(TYPE_TEMP, 1, 0b0111, false),
        END_TOKEN,
    ];
    let prog = parse(&bc).expect("ps_2_0 texkill r1.xyz should parse");
    let dst = prog.instructions[0].dst.expect("texkill dst");
    assert_eq!(dst.reg.index, 1);
    assert_eq!(dst.write_mask, WriteMask(0b0111));
}

#[test]
fn pixel_shader_position_input_decl_is_invalid() {
    // ps_3_0 { dcl_position0 v0; mov oC0, v0; } — the rasterizer position is the
    // special vPos register, so declaring POSITION index 0 on a v# input is
    // invalid and CreatePixelShader rejects it. The
    // bytecode still *parses*; the validity check is a separate post-parse
    // predicate.
    let bc = [
        PS3_HEADER,
        opcode_token(OP_DCL, 2),
        0x0000_0000, // usage_token: Position (usage=0, index=0)
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let prog = parse(&bc).expect("ps_3_0 dcl_position v0 should still parse");
    assert!(
        prog.has_invalid_pixel_input_decl(),
        "POSITION on a pixel-shader input register must be flagged invalid"
    );
}

#[test]
fn pixel_shader_texcoord_input_decl_is_valid() {
    // ps_3_0 { dcl_texcoord v0; mov oC0, v0; } — TEXCOORD is a legal PS input.
    let bc = [
        PS3_HEADER,
        opcode_token(OP_DCL, 2),
        0x0000_0005, // usage_token: Texcoord (usage=5, index=0)
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let prog = parse(&bc).expect("ps_3_0 dcl_texcoord v0 should parse");
    assert!(!prog.has_invalid_pixel_input_decl());
}

#[test]
fn vertex_shader_position_input_decl_is_valid() {
    // vs_2_0 { dcl_position v0; mov oPos, v0; } — POSITION input is legal in a VS.
    let bc = [
        VS_HEADER,
        opcode_token(OP_DCL, 2),
        0x0000_0000, // usage_token: Position (usage=0, index=0)
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_RASTOUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let prog = parse(&bc).expect("vs_2_0 dcl_position v0 should parse");
    assert!(!prog.has_invalid_pixel_input_decl());
}

#[test]
fn ps_2_0_plain_input_decl_is_valid() {
    // ps_2_0 { dcl v0; mov oC0, v0; } — the SM2 dcl token (usage bits 0) is a
    // plain color input, NOT dcl_position. Same token as the ps_3_0 bad shader,
    // but only SM3 reads usage bits as a POSITION semantic, so SM2 must NOT be
    // flagged.
    let bc = [
        PS_HEADER,
        opcode_token(OP_DCL, 2),
        0x0000_0000, // usage_token: usage bits 0 (color input in SM2)
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let prog = parse(&bc).expect("ps_2_0 dcl v0 should parse");
    assert!(
        !prog.has_invalid_pixel_input_decl(),
        "SM2 plain input dcl must not be read as a forbidden POSITION"
    );
}

#[test]
fn ps_3_0_position_index_one_input_is_valid() {
    // ps_3_0 { dcl_position1 v0; mov oC0, v0; } — POSITION *index 1* is an
    // ordinary user semantic and a legal PS input; only POSITION0 (the
    // rasterizer position, which lives in vPos) is forbidden on a v# input.
    let bc = [
        PS3_HEADER,
        opcode_token(OP_DCL, 2),
        0x0001_0000, // usage_token: Position (usage=0), index=1
        dst_token(TYPE_INPUT, 0, 0xF, false),
        opcode_token(OP_MOV, 2),
        dst_token(TYPE_COLOROUT, 0, 0xF, false),
        src_token(TYPE_INPUT, 0, SWIZ_IDENTITY, 0),
        END_TOKEN,
    ];
    let prog = parse(&bc).expect("ps_3_0 dcl_position1 v0 should parse");
    assert!(
        !prog.has_invalid_pixel_input_decl(),
        "POSITION index 1 is a valid PS input semantic"
    );
}

#[test]
fn ps_position_index_1_is_accepted() {
    // A "good" ps_3_0 with dcl_position1 v0 (POSITION index 1) — valid, so
    // parse + validation must accept it. Guards against the usage-index gate
    // regressing back to rejecting all POSITION inputs.
    let bc = [
        0xffff_0300u32,
        0x0200_001f,
        0x8001_0000,
        0x900f_0000, // dcl_position1 v0
        0x0200_0001,
        0x800f_0800,
        0x90e4_0000, // mov oC0, v0
        0x0000_ffff,
    ];
    let prog = parse(&bc).expect("test_position_index ps_code should parse");
    assert!(
        !prog.has_invalid_pixel_input_decl(),
        "the ps_3_0 dcl_position1 ps_code must be accepted"
    );
}

#[test]
fn ps_position_index_0_is_rejected() {
    // A "bad" ps_3_0 with dcl_position0 v0 (POSITION index 0 on a v# input).
    let bc = [
        0xffff_0300u32,
        0x0200_001f,
        0x8000_0000,
        0x900f_0000, // dcl_position0 v0
        0x0200_0001,
        0x800f_0800,
        0x90e4_0000, // mov oC0, v0
        0x0000_ffff,
    ];
    let prog = parse(&bc).expect("ps_code_bad should still parse");
    assert!(
        prog.has_invalid_pixel_input_decl(),
        "the ps_3_0 dcl_position0 input must be rejected"
    );
}
