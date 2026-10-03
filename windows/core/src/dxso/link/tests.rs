//! Unit tests for semantic linkage between SM3 vertex and pixel shaders.

use mtld3d_types::{
    D3DDECLUSAGE_BLENDWEIGHT, D3DDECLUSAGE_COLOR, D3DDECLUSAGE_DEPTH, D3DDECLUSAGE_FOG,
    D3DDECLUSAGE_NORMAL, D3DDECLUSAGE_POSITION, D3DDECLUSAGE_POSITIONT, D3DDECLUSAGE_PSIZE,
    D3DDECLUSAGE_SAMPLE, D3DDECLUSAGE_TANGENT, D3DDECLUSAGE_TEXCOORD,
};

use super::{
    LinkInputs, MAX_LINKED_INPUTS, PsInputs, Semantic, SemanticSet, VsOutputs,
    decl_passthrough_code, passthrough_extras, write_extra_members, write_passthrough_inputs,
    write_passthrough_outputs,
};
use crate::dxso::{
    ir::{DeclUsage, DstMods, DstOperand, RegKind, Register, WriteMask},
    parser::parse,
};

const VS3_HEADER: u32 = 0xFFFE_0300;
const PS3_HEADER: u32 = 0xFFFF_0300;
const END_TOKEN: u32 = 0x0000_FFFF;
const DCL: u32 = 0x0200_001F;
/// `D3DSPR_OUTPUT` (type 6) destination register token without mask or index.
const OUTPUT: u32 = 0xE000_0000;
/// `D3DSPR_INPUT` (type 1) destination register token without mask or index.
const INPUT: u32 = 0x9000_0000;

const POSITION: u32 = 0;
const NORMAL: u32 = 3;
const PSIZE: u32 = 4;
const TEXCOORD: u32 = 5;
const TANGENT: u32 = 6;
const COLOR: u32 = 10;
const FOG: u32 = 11;

/// `dcl_<usage><index> <kind>N.<mask>` as its three tokens.
const fn dcl(usage: u32, index: u32, kind: u32, reg: u32, mask: u32) -> [u32; 3] {
    [
        DCL,
        0x8000_0000 | (index << 16) | usage,
        kind | (mask << 16) | reg,
    ]
}

fn program(header: u32, dcls: &[[u32; 3]]) -> crate::dxso::DxsoProgram {
    let mut tokens = vec![header];
    for d in dcls {
        tokens.extend_from_slice(d);
    }
    tokens.push(END_TOKEN);
    parse(&tokens).expect("dcl-only program parses")
}

const fn dst(kind: RegKind, index: u16, mask: u8) -> DstOperand {
    DstOperand {
        reg: Register { kind, index },
        write_mask: WriteMask(mask),
        mods: DstMods::empty(),
        shift_scale: 0,
    }
}

#[test]
fn members_are_named_after_the_semantic() {
    let cases = [
        (DeclUsage::Position, 0, "position", false),
        (DeclUsage::Position, 1, "position1", false),
        (DeclUsage::Position, 2, "position2", true),
        (DeclUsage::Texcoord, 15, "texcoord15", false),
        (DeclUsage::Color, 1, "color1", false),
        (DeclUsage::Color, 2, "color2", true),
        (DeclUsage::Fog, 0, "fog", false),
        (DeclUsage::Fog, 1, "fog1", true),
        (DeclUsage::PSize, 0, "psize0", false),
        (DeclUsage::Normal, 0, "normal0", true),
        (DeclUsage::BlendIndices, 3, "blendindices3", true),
        (DeclUsage::PositionT, 0, "positiont0", true),
        (DeclUsage::Sample, 15, "sample15", true),
    ];
    for (usage, index, member, extra) in cases {
        let semantic = Semantic::new(usage, index);
        assert_eq!(semantic.member(), member);
        assert_eq!(semantic.is_extra(), extra, "{member}");
        assert_eq!(Semantic::from_code(semantic.code()), semantic, "{member}");
    }
}

#[test]
fn a_vertex_shader_outputs_only_its_declared_extras() {
    let vs = program(
        VS3_HEADER,
        &[
            dcl(POSITION, 0, INPUT, 0, 0xF),
            dcl(POSITION, 0, OUTPUT, 0, 0xF),
            dcl(NORMAL, 0, OUTPUT, 1, 0x7),
            dcl(COLOR, 2, OUTPUT, 2, 0xF),
            dcl(TEXCOORD, 3, OUTPUT, 3, 0x3),
            dcl(PSIZE, 0, OUTPUT, 4, 0x1),
            dcl(FOG, 0, OUTPUT, 5, 0x1),
        ],
    );
    let outputs = SemanticSet::vs_outputs(&vs);
    let listed: Vec<_> = outputs.iter().map(Semantic::member).collect();
    assert_eq!(
        listed,
        ["normal0", "color2"],
        "the vertex input is no output"
    );
    let plan = VsOutputs::build(&vs).expect("SM3 VS");
    let mut struct_text = String::new();
    write_extra_members(&mut struct_text, plan.extras());
    assert_eq!(struct_text, "    float4 normal0;\n    float4 color2;\n");

    let sm2 = parse(&[0xFFFE_0200, END_TOKEN]).expect("vs_2_0");
    assert!(SemanticSet::vs_outputs(&sm2).is_empty());
    assert!(VsOutputs::build(&sm2).is_none());
}

#[test]
fn input_order_fixes_the_mask_bits() {
    let ps = program(
        PS3_HEADER,
        &[
            dcl(TEXCOORD, 0, INPUT, 0, 0xF),
            dcl(TANGENT, 0, INPUT, 1, 0xF),
            dcl(NORMAL, 0, INPUT, 2, 0xF),
            dcl(NORMAL, 0, INPUT, 3, 0xF),
            dcl(COLOR, 2, INPUT, 4, 0xF),
        ],
    );
    let inputs = LinkInputs::ps_inputs(&ps);
    let order: Vec<_> = inputs.iter().map(Semantic::member).collect();
    assert_eq!(order, ["tangent0", "normal0", "color2"]);

    let vs = program(
        VS3_HEADER,
        &[
            dcl(COLOR, 2, OUTPUT, 1, 0xF),
            dcl(TANGENT, 0, OUTPUT, 2, 0xF),
        ],
    );
    assert_eq!(inputs.mask_against(&SemanticSet::vs_outputs(&vs)), 0b101);
    assert_eq!(inputs.mask_against(&SemanticSet::default()), 0);
}

#[test]
fn inputs_past_the_mask_width_are_not_linked() {
    let dcls: Vec<_> = (0..10).map(|i| dcl(NORMAL, i, INPUT, i, 0xF)).collect();
    let inputs = LinkInputs::ps_inputs(&program(PS3_HEADER, &dcls));
    assert_eq!(inputs.iter().count(), MAX_LINKED_INPUTS);
    assert!(
        inputs
            .position(Semantic::new(DeclUsage::Normal, 8))
            .is_none()
    );
}

#[test]
fn a_shared_output_register_is_staged_and_split_by_lane() {
    let vs = program(
        VS3_HEADER,
        &[
            dcl(POSITION, 0, OUTPUT, 0, 0xF),
            dcl(TEXCOORD, 0, OUTPUT, 1, 0x3),
            dcl(TEXCOORD, 1, OUTPUT, 1, 0xC),
            dcl(COLOR, 0, OUTPUT, 2, 0x7),
            dcl(FOG, 0, OUTPUT, 2, 0x8),
            dcl(NORMAL, 0, OUTPUT, 3, 0xF),
        ],
    );
    let plan = VsOutputs::build(&vs).expect("SM3 VS");
    let target = |index| plan.targets()[&(RegKind::TexcoordOut, index)].as_str();
    assert_eq!(target(0), "out.position");
    assert_eq!(target(1), "_o0");
    assert_eq!(target(2), "_o1");
    assert_eq!(target(3), "out.normal0");

    let mut prologue = String::new();
    plan.write_prologue(&mut prologue);
    assert!(
        prologue.contains("out.normal0 = float4(0.0);"),
        "{prologue}"
    );
    assert!(
        prologue.contains("float4 _o0 = float4(0.0, 0.0, 0.0, 0.0);"),
        "{prologue}"
    );
    assert!(
        prologue.contains("float4 _o1 = float4(1.0, 1.0, 1.0, 1.0);"),
        "COLOR0 and FOG0 lanes start unwritten-white and unfogged:\n{prologue}"
    );

    let mut epilogue = String::new();
    plan.write_epilogue(&mut epilogue);
    for line in [
        "out.texcoord0.xy = _o0.xy;",
        "out.texcoord1.zw = _o0.zw;",
        "out.color0.xyz = _o1.xyz;",
        "out.fog = float4(_o1.w);",
    ] {
        assert!(epilogue.contains(line), "{line} missing:\n{epilogue}");
    }

    assert!(plan.writes_fog(dst(RegKind::TexcoordOut, 2, 0x8)));
    assert!(!plan.writes_fog(dst(RegKind::TexcoordOut, 2, 0x7)));
    assert!(!plan.writes_fog(dst(RegKind::TexcoordOut, 3, 0xF)));
}

#[test]
fn a_scalar_output_off_lane_x_is_staged() {
    let vs = program(
        VS3_HEADER,
        &[dcl(PSIZE, 0, OUTPUT, 4, 0x2), dcl(FOG, 0, OUTPUT, 5, 0x1)],
    );
    let plan = VsOutputs::build(&vs).expect("SM3 VS");
    assert_eq!(plan.targets()[&(RegKind::TexcoordOut, 4)], "_o0");
    assert_eq!(plan.targets()[&(RegKind::TexcoordOut, 5)], "out.fog");
    let mut prologue = String::new();
    plan.write_prologue(&mut prologue);
    assert!(
        prologue.contains("float4 _o0 = float4(0.0, vs_draw.point.x, 0.0, 0.0);"),
        "{prologue}"
    );
    let mut epilogue = String::new();
    plan.write_epilogue(&mut epilogue);
    assert_eq!(epilogue, "    _psize_storage.x = _o0.y;\n");
}

#[test]
fn pixel_inputs_read_their_semantic_member() {
    let ps = program(
        PS3_HEADER,
        &[
            dcl(TEXCOORD, 2, INPUT, 0, 0x3),
            dcl(NORMAL, 0, INPUT, 1, 0x7),
            dcl(TANGENT, 0, INPUT, 2, 0xF),
            dcl(COLOR, 1, INPUT, 3, 0xF),
            dcl(PSIZE, 0, INPUT, 4, 0x1),
        ],
    );
    let unlinked = PsInputs::build(&ps, 0);
    assert_eq!(
        unlinked.reads()[&0],
        "in.texcoord2",
        "the whole member is read"
    );
    assert_eq!(unlinked.reads()[&1], "float4(0.0)");
    assert_eq!(unlinked.reads()[&2], "float4(0.0)");
    assert_eq!(unlinked.reads()[&3], "in.color1");
    assert_eq!(
        unlinked.reads()[&4],
        "float4(0.0)",
        "the point size is no varying"
    );
    assert!(unlinked.extras().is_empty());

    let tangent_only = PsInputs::build(&ps, 0b10);
    assert_eq!(tangent_only.reads()[&1], "float4(0.0)");
    assert_eq!(tangent_only.reads()[&2], "in.tangent0");
    let members: Vec<_> = tangent_only.extras().iter().map(|s| s.member()).collect();
    assert_eq!(members, ["tangent0"]);
}

#[test]
fn a_shared_input_register_is_assembled_lane_by_lane() {
    let ps = program(
        PS3_HEADER,
        &[
            dcl(TEXCOORD, 0, INPUT, 0, 0x3),
            dcl(TEXCOORD, 1, INPUT, 0, 0xC),
            dcl(NORMAL, 0, INPUT, 1, 0x7),
            dcl(TEXCOORD, 4, INPUT, 1, 0x8),
        ],
    );
    let plan = PsInputs::build(&ps, 0);
    assert_eq!(plan.reads()[&0], "_v0");
    assert_eq!(plan.reads()[&1], "_v1");
    let mut prologue = String::new();
    plan.write_prologue(&mut prologue);
    assert!(
        prologue.contains(
            "float4 _v0 = float4(in.texcoord0.x, in.texcoord0.y, in.texcoord1.z, in.texcoord1.w);"
        ),
        "{prologue}"
    );
    assert!(
        prologue.contains("float4 _v1 = float4(0.0, 0.0, 0.0, in.texcoord4.w);"),
        "an unlinked NORMAL0 reads zero in its lanes:\n{prologue}"
    );
}

#[test]
fn a_lane_no_dcl_covers_comes_from_the_first_semantic() {
    let ps = program(
        PS3_HEADER,
        &[
            dcl(TEXCOORD, 0, INPUT, 0, 0x1),
            dcl(TEXCOORD, 1, INPUT, 0, 0x2),
        ],
    );
    let mut prologue = String::new();
    PsInputs::build(&ps, 0).write_prologue(&mut prologue);
    assert!(
        prologue.contains(
            "float4 _v0 = float4(in.texcoord0.x, in.texcoord1.y, in.texcoord0.z, in.texcoord0.w);"
        ),
        "{prologue}"
    );
}

#[test]
fn a_pixel_shader_below_sm3_keeps_its_structural_colour_inputs() {
    let ps = parse(&[0xFFFF_0200, DCL, 0x8000_0000, 0x900F_0001, END_TOKEN]).expect("ps_2_0");
    let plan = PsInputs::build(&ps, 0);
    assert_eq!(plan.reads()[&1], "in.color1");
    assert!(LinkInputs::ps_inputs(&ps).is_empty());
}

/// The passthrough list of `codes`, zero-filled past the end.
fn passthrough_of(codes: &[u8]) -> [u8; MAX_LINKED_INPUTS] {
    let mut list = [0; MAX_LINKED_INPUTS];
    list[..codes.len()].copy_from_slice(codes);
    list
}

fn code(usage: u32, index: u32) -> u8 {
    let usage = u8::try_from(usage).expect("usage fits u8");
    let index = u8::try_from(index).expect("index fits u8");
    decl_passthrough_code(usage, index).expect("semantic passes through")
}

#[test]
fn a_pretransformed_layout_passes_through_what_the_ff_stage_does_not_write() {
    // The fixed-function stage writes these itself.
    for (usage, index) in [
        (D3DDECLUSAGE_POSITION, 0),
        (D3DDECLUSAGE_POSITIONT, 0),
        (D3DDECLUSAGE_PSIZE, 0),
        (D3DDECLUSAGE_TEXCOORD, 0),
        (D3DDECLUSAGE_TEXCOORD, 7),
        (D3DDECLUSAGE_COLOR, 0),
        (D3DDECLUSAGE_COLOR, 1),
    ] {
        assert_eq!(
            decl_passthrough_code(usage, index),
            None,
            "usage {usage} index {index}"
        );
    }
    // Everything else a `ps_3_0` can declare reaches it by semantic.
    for (usage, index) in [
        (D3DDECLUSAGE_POSITION, 1),
        (D3DDECLUSAGE_FOG, 0),
        (D3DDECLUSAGE_NORMAL, 0),
        (D3DDECLUSAGE_BLENDWEIGHT, 0),
        (D3DDECLUSAGE_TANGENT, 3),
        (D3DDECLUSAGE_COLOR, 2),
        (D3DDECLUSAGE_DEPTH, 0),
        (D3DDECLUSAGE_SAMPLE, 15),
        (D3DDECLUSAGE_POSITIONT, 1),
    ] {
        assert!(
            decl_passthrough_code(usage, index).is_some(),
            "usage {usage} index {index}"
        );
    }
    // A usage outside D3DDECLUSAGE never does.
    assert_eq!(decl_passthrough_code(D3DDECLUSAGE_SAMPLE + 1, 0), None);
}

#[test]
fn a_pretransformed_vertex_stage_writes_each_entry_into_its_member() {
    let list = passthrough_of(&[
        code(NORMAL, 0),
        code(FOG, 0),
        code(TANGENT, 0),
        code(POSITION, 1),
    ]);
    let mut inputs = String::new();
    write_passthrough_inputs(&mut inputs, list, 15);
    for (k, attr) in [(0, 15), (1, 16), (2, 17), (3, 18)] {
        assert!(
            inputs.contains(&format!("float4 p{k} [[attribute({attr})]];")),
            "{inputs}"
        );
    }
    assert!(!inputs.contains("p4"), "{inputs}");

    let mut outputs = String::new();
    write_passthrough_outputs(&mut outputs, list, true);
    for line in [
        "out.normal0 = in.p0;",
        "out.fog = in.p1;",
        "out.tangent0 = in.p2;",
        "out.position1 = in.p3;",
    ] {
        assert!(outputs.contains(line), "{line} missing from\n{outputs}");
    }
    // Under the specular-alpha fog factor the fog member stays the factor.
    let mut factor = String::new();
    write_passthrough_outputs(&mut factor, list, false);
    assert!(!factor.contains("out.fog"), "{factor}");
    assert!(factor.contains("out.normal0 = in.p0;"), "{factor}");

    // Only the extras get members of their own; fog and the secondary
    // position ride the members every `Varyings` declares.
    let members: Vec<_> = passthrough_extras(list)
        .iter()
        .map(|s| s.member())
        .collect();
    assert_eq!(members, ["normal0", "tangent0"]);
    let outputs = SemanticSet::passthrough_outputs(&list);
    assert!(outputs.contains(Semantic::new(DeclUsage::Normal, 0)));
    assert!(outputs.contains(Semantic::new(DeclUsage::Tangent, 0)));
    assert!(!outputs.contains(Semantic::new(DeclUsage::Fog, 0)));
}

#[test]
fn a_pixel_shader_links_the_extras_a_pretransformed_layout_passes_through() {
    let ps = program(
        PS3_HEADER,
        &[
            dcl(TANGENT, 0, INPUT, 0, 0xF),
            dcl(NORMAL, 0, INPUT, 1, 0xF),
            dcl(COLOR, 2, INPUT, 2, 0xF),
        ],
    );
    let inputs = LinkInputs::ps_inputs(&ps);
    let list = passthrough_of(&[code(NORMAL, 0), code(FOG, 0), code(TANGENT, 0)]);
    let mask = inputs.mask_against(&SemanticSet::passthrough_outputs(&list));
    // TANGENT0 and NORMAL0 link; COLOR2 is not in the declaration.
    assert_eq!(mask, 0b011);
    let plan = PsInputs::build(&ps, mask);
    assert_eq!(plan.reads()[&0], "in.tangent0");
    assert_eq!(plan.reads()[&1], "in.normal0");
    assert_eq!(plan.reads()[&2], "float4(0.0)");
    let untransformed = SemanticSet::passthrough_outputs(&[0; MAX_LINKED_INPUTS]);
    assert_eq!(inputs.mask_against(&untransformed), 0);
}
