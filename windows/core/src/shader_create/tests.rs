use mtld3d_shared::shader_create::{DeclUsage, ShaderStage, ShaderUsage};

use super::{ShaderCreateError, parse_shader};
use crate::ids::ProgramId;

#[test]
fn malformed_and_wrong_stage_are_rejected() {
    assert!(matches!(
        parse_shader(&ShaderStage::Vertex, &[]),
        Err(ShaderCreateError::Parse(_))
    ));
    assert!(matches!(
        parse_shader(&ShaderStage::Vertex, &[0xffff_0300, 0x0000_ffff]),
        Err(ShaderCreateError::WrongStage)
    ));
    assert!(matches!(
        parse_shader(&ShaderStage::Pixel, &[0xfffe_0300, 0x0000_ffff]),
        Err(ShaderCreateError::WrongStage)
    ));
}

#[test]
fn constant_register_boundaries_are_preserved() {
    for (stage, header, last) in [
        (ShaderStage::Vertex, 0xfffe_0200, 255),
        (ShaderStage::Pixel, 0xffff_0200, 31),
        (ShaderStage::Pixel, 0xffff_0300, 223),
    ] {
        let mut tokens = [
            header,
            0x0200_0001,
            0x800f_0000,
            0xa0e4_0000 | last,
            0x0000_ffff,
        ];
        let shader = parse_shader(&stage, &tokens).expect("last constant is valid");
        assert_eq!(shader.max_const_used, last + 1);
        tokens[3] += 1;
        assert!(matches!(
            parse_shader(&stage, &tokens),
            Err(ShaderCreateError::ConstantRegisterLimit)
        ));
    }
}

#[test]
fn pixel_position_zero_is_rejected_but_other_indices_are_valid() {
    let mut tokens = [0xffff_0300, 0x0200_001f, 0, 0x900f_0000, 0x0000_ffff];
    assert!(matches!(
        parse_shader(&ShaderStage::Pixel, &tokens),
        Err(ShaderCreateError::InvalidPixelInput)
    ));
    tokens[2] = 1 << 16;
    assert!(parse_shader(&ShaderStage::Pixel, &tokens).is_ok());
}

#[test]
fn vertex_metadata_keeps_semantics_in_declaration_order() {
    let tokens = [
        0xfffe_0300,
        0x0200_001f,
        0x0002_0005,
        0x900f_0003,
        0x0200_001f,
        0x0000_0000,
        0x900f_0001,
        0x0000_ffff,
    ];
    let shader = parse_shader(&ShaderStage::Vertex, &tokens).expect("valid vertex shader");
    assert_eq!(shader.id, ProgramId::from_tokens(&tokens));
    assert_eq!(shader.program.bytecode().as_ref(), tokens);
    assert_eq!(shader.input_semantics.len(), 2);
    assert_eq!(shader.input_semantics[0].usage, DeclUsage::Texcoord);
    assert_eq!(shader.input_semantics[0].usage_index, 2);
    assert_eq!(shader.input_semantics[0].register_index, 3);
    assert_eq!(shader.input_semantics[1].usage, DeclUsage::Position);
    assert!(!shader.usage.contains(ShaderUsage::AUTOMATIC_FOG));
}

#[test]
fn pixel_metadata_preserves_fog_and_output_mask() {
    let mut tokens = [
        0xffff_0200,
        0x0200_0001,
        0x800f_0802,
        0xa0e4_0000,
        0x0000_ffff,
    ];
    let shader = parse_shader(&ShaderStage::Pixel, &tokens).expect("valid pixel shader");
    assert!(shader.usage.contains(ShaderUsage::AUTOMATIC_FOG));
    assert_eq!(shader.color_out_mask, 4);
    assert!(shader.input_semantics.is_empty());
    tokens[0] = 0xffff_0300;
    assert!(
        !parse_shader(&ShaderStage::Pixel, &tokens)
            .expect("valid SM3 shader")
            .usage
            .contains(ShaderUsage::AUTOMATIC_FOG)
    );
}

#[test]
fn vertex_constant_usage_flags_are_independent() {
    let tokens = [
        0xfffe_0300,
        0x0300_0001,
        0x800f_0000,
        0xa0e4_2005,
        0xb000_0000,
        0x0200_0001,
        0x800f_0000,
        0xf0e4_0000,
        0x0200_0001,
        0x800f_0000,
        0xe0e4_0803,
        0x0000_ffff,
    ];
    let shader = parse_shader(&ShaderStage::Vertex, &tokens).expect("valid constant references");
    assert!(shader.usage.contains(ShaderUsage::RELATIVE_CONST));
    assert!(shader.usage.contains(ShaderUsage::INT_CONST));
    assert!(shader.usage.contains(ShaderUsage::BOOL_CONST));
    assert!(!shader.usage.contains(ShaderUsage::BUMP_ENV));
}

#[test]
fn pixel_relative_constant_read_is_reported() {
    // ps_3_0: defi i0, 4, 0, 1, 0; mov r0, c1; loop aL, i0;
    // add r0, r0, c[aL + 2]; endloop; mov oC0, r0
    let tokens = [
        0xffff_0300,
        0x0500_0030,
        0xf00f_0000,
        4,
        0,
        1,
        0,
        0x0200_0001,
        0x800f_0000,
        0xa0e4_0001,
        0x0200_001b,
        0xf0e4_0800,
        0xf0e4_0000,
        0x0400_0002,
        0x800f_0000,
        0x80e4_0000,
        0xa0e4_2002,
        0xf000_0800,
        0x0000_001d,
        0x0200_0001,
        0x800f_0800,
        0x80e4_0000,
        0x0000_ffff,
    ];
    let shader = parse_shader(&ShaderStage::Pixel, &tokens).expect("valid relative read");
    assert!(shader.usage.contains(ShaderUsage::RELATIVE_CONST));
    assert_eq!(shader.max_const_used, 3);
}

#[test]
fn pixel_integer_and_boolean_files_start_at_ps_2_x() {
    // ps_2_x { defb b0, true; if b0; mov oC0, c0; endif }
    let bool_branch = [
        0xffff_0201,
        0x0200_002f,
        0xe00f_0800,
        1,
        0x0100_0028,
        0xe0e4_0800,
        0x0200_0001,
        0x800f_0800,
        0xa0e4_0000,
        0x0000_002b,
        0x0000_ffff,
    ];
    // ps_2_x { defi i0, 2, 0, 0, 0; mov r0, c0; rep i0; add r0, r0, c1; endrep;
    // mov oC0, r0 }
    let int_loop = [
        0xffff_0201,
        0x0500_0030,
        0xf00f_0000,
        2,
        0,
        0,
        0,
        0x0200_0001,
        0x800f_0000,
        0xa0e4_0000,
        0x0100_0026,
        0xf0e4_0000,
        0x0300_0002,
        0x800f_0000,
        0x80e4_0000,
        0xa0e4_0001,
        0x0000_0027,
        0x0200_0001,
        0x800f_0800,
        0x80e4_0000,
        0x0000_ffff,
    ];
    for mut tokens in [bool_branch.to_vec(), int_loop.to_vec()] {
        assert!(parse_shader(&ShaderStage::Pixel, &tokens).is_ok());
        tokens[0] = 0xffff_0200;
        assert!(matches!(
            parse_shader(&ShaderStage::Pixel, &tokens),
            Err(ShaderCreateError::ConstantRegisterLimit)
        ));
    }
}
