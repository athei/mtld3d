use super::*;
use crate::dxso::DeclUsage;

#[test]
fn triangle_fan_expands_to_list() {
    // 5 fan vertices (1 byte each) → 3 triangles: (0,1,2),(0,2,3),(0,3,4).
    let src = [10u8, 11, 12, 13, 14];
    let out = expand_triangle_fan(&src, 1, 3);
    assert_eq!(out, vec![10, 11, 12, 10, 12, 13, 10, 13, 14]);
}

#[test]
fn triangle_fan_respects_stride() {
    // 4 vertices of 2 bytes → 2 triangles: (0,1,2),(0,2,3).
    let src = [0u8, 0, 1, 1, 2, 2, 3, 3];
    let out = expand_triangle_fan(&src, 2, 2);
    assert_eq!(out, vec![0, 0, 1, 1, 2, 2, 0, 0, 2, 2, 3, 3]);
}

fn u16_indices(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn u32_indices(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn nonindexed_fan_becomes_absolute_u16_triangles() {
    // DrawPrimitive(FAN, start 10, 3 prims): fan vertices 10..=14.
    let fan = triangle_fan_indices(10, 3).expect("fits");
    assert_eq!(fan.index_type, IndexType::UInt16);
    assert_eq!(
        u16_indices(&fan.bytes),
        vec![10, 11, 12, 10, 12, 13, 10, 13, 14]
    );
    assert_eq!((fan.min_vertex, fan.max_vertex), (10, 14));
}

#[test]
fn fan_pattern_is_the_relative_fan_and_clips_to_the_buffer() {
    let mut out = vec![0xAAu8; fan_pattern_bytes(3)];
    fill_fan_pattern_u16(&mut out, 3);
    assert_eq!(u16_indices(&out), vec![0, 1, 2, 0, 2, 3, 0, 3, 4]);
    // Asking for more triangles than the buffer holds writes what fits.
    let mut short = vec![0xAAu8; fan_pattern_bytes(1)];
    fill_fan_pattern_u16(&mut short, 5);
    assert_eq!(u16_indices(&short), vec![0, 1, 2]);
    // The last addressable triangle ends exactly at u16::MAX.
    let mut tail = vec![0u8; fan_pattern_bytes(FAN_PATTERN_MAX_TRIANGLES)];
    fill_fan_pattern_u16(&mut tail, FAN_PATTERN_MAX_TRIANGLES);
    assert_eq!(
        &u16_indices(&tail)[tail.len() / 2 - 3..],
        &[0, u16::MAX - 1, u16::MAX]
    );
}

#[test]
fn fan_widens_to_u32_past_u16_range() {
    let fan = triangle_fan_indices(0xFFFE, 1).expect("fits");
    assert_eq!(fan.index_type, IndexType::UInt32);
    assert_eq!(u32_indices(&fan.bytes), vec![0xFFFE, 0xFFFF, 0x1_0000]);
    assert!(triangle_fan_indices(u32::MAX - 1, 1).is_none());
}

#[test]
fn indexed_fan_folds_the_base_vertex_in() {
    // 16-bit app indices 5,6,7,8 with base vertex 100: triangles over
    // 105..=108.
    let src: Vec<u8> = [5u16, 6, 7, 8]
        .iter()
        .flat_map(|i| i.to_le_bytes())
        .collect();
    let fan = triangle_fan_indices_from(&src, 2, 100, 2).expect("fits");
    assert_eq!(fan.index_type, IndexType::UInt16);
    assert_eq!(u16_indices(&fan.bytes), vec![105, 106, 107, 105, 107, 108]);
    assert_eq!((fan.min_vertex, fan.max_vertex), (105, 108));
    // A negative base is legal as long as no index goes below zero.
    let fan = triangle_fan_indices_from(&src, 2, -5, 2).expect("fits");
    assert_eq!(u16_indices(&fan.bytes), vec![0, 1, 2, 0, 2, 3]);
    assert!(triangle_fan_indices_from(&src, 2, -6, 2).is_none());
}

#[test]
fn indexed_fan_reads_32_bit_indices_and_rejects_short_streams() {
    let src: Vec<u8> = [1u32, 2, 0x2_0000]
        .iter()
        .flat_map(|i| i.to_le_bytes())
        .collect();
    let fan = triangle_fan_indices_from(&src, 4, 0, 1).expect("fits");
    assert_eq!(fan.index_type, IndexType::UInt32);
    assert_eq!(u32_indices(&fan.bytes), vec![1, 2, 0x2_0000]);
    assert!(triangle_fan_indices_from(&src, 4, 0, 2).is_none());
    assert!(triangle_fan_indices_from(&src, 3, 0, 1).is_none());
}

fn pos3() -> D3DVERTEXELEMENT9 {
    D3DVERTEXELEMENT9 {
        stream: 0,
        offset: 0,
        type_: D3DDECLTYPE_FLOAT3,
        method: 0,
        usage: D3DDECLUSAGE_POSITION,
        usage_index: 0,
    }
}

fn tex0(offset: u16) -> D3DVERTEXELEMENT9 {
    D3DVERTEXELEMENT9 {
        stream: 0,
        offset,
        type_: D3DDECLTYPE_FLOAT2,
        method: 0,
        usage: D3DDECLUSAGE_TEXCOORD,
        usage_index: 0,
    }
}

fn to_bits4(arr: [f32; 4]) -> [u32; 4] {
    [
        arr[0].to_bits(),
        arr[1].to_bits(),
        arr[2].to_bits(),
        arr[3].to_bits(),
    ]
}

#[test]
fn d3dcolor_to_rgba_default_is_white() {
    // D3DRS_BLENDFACTOR's default is 0xFFFFFFFF (opaque white).
    let rgba = d3dcolor_to_rgba_f32(0xFFFF_FFFF);
    assert_eq!(to_bits4(rgba), to_bits4([1.0, 1.0, 1.0, 1.0]));
}

#[test]
fn d3dcolor_to_rgba_zero_is_transparent_black() {
    let rgba = d3dcolor_to_rgba_f32(0x0000_0000);
    assert_eq!(to_bits4(rgba), to_bits4([0.0, 0.0, 0.0, 0.0]));
}

#[test]
fn d3dcolor_to_rgba_argb_byte_order() {
    // 0xAARRGGBB. A=0x80, R=0x40, G=0x20, B=0x10. The u8→f32 path
    // is exact (each byte fits f32 mantissa), so bit-equality holds.
    let rgba = d3dcolor_to_rgba_f32(0x8040_2010);
    assert_eq!(rgba[0].to_bits(), (f32::from(0x40u8) / 255.0).to_bits());
    assert_eq!(rgba[1].to_bits(), (f32::from(0x20u8) / 255.0).to_bits());
    assert_eq!(rgba[2].to_bits(), (f32::from(0x10u8) / 255.0).to_bits());
    assert_eq!(rgba[3].to_bits(), (f32::from(0x80u8) / 255.0).to_bits());
}

#[test]
fn linear_to_srgb_encodes_colour_lanes_only() {
    // 0x7f linear stores as 0xbb once sRGB-encoded; alpha passes through.
    let rgba = linear_to_srgb_rgba(d3dcolor_to_rgba_f32(0x407f_7f7f));
    // The stored byte, as the exactly representable float it rounds to.
    let to_byte = |v: f32| (v * 255.0).round().to_bits();
    let byte = |b: u8| f32::from(b).to_bits();
    assert_eq!(to_byte(rgba[0]), byte(0xbb));
    assert_eq!(to_byte(rgba[1]), byte(0xbb));
    assert_eq!(to_byte(rgba[2]), byte(0xbb));
    assert_eq!(rgba[3].to_bits(), (f32::from(0x40u8) / 255.0).to_bits());
    // The end points map onto themselves (to within a ulp at white), and
    // an over-range lane clamps before encoding.
    let ends = linear_to_srgb_rgba([0.0, 1.0, 2.0, 1.0]);
    assert_eq!(ends[0].to_bits(), 0.0f32.to_bits());
    assert_eq!(to_byte(ends[1]), byte(0xff));
    assert_eq!(to_byte(ends[2]), byte(0xff));
}

#[test]
fn color_fill_a8r8g8b8_roundtrips_the_d3dcolor() {
    // BGRA8 bytes read back as the same D3DCOLOR: filling 0xdeadbeef must
    // read back 0xdeadbeef.
    let bytes = d3dcolor_fill_pixel_bytes(0xdead_beef, D3DFMT_A8R8G8B8).unwrap();
    assert_eq!(bytes, vec![0xef, 0xbe, 0xad, 0xde]);
    assert_eq!(
        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        0xdead_beef
    );
}

#[test]
fn color_fill_r32f_is_red_channel_normalized() {
    // R=0xad → 0xad/255.0: ColorFill promotes the red byte to a
    // normalized float.
    let bytes = d3dcolor_fill_pixel_bytes(0x00ad_0000, D3DFMT_R32F).unwrap();
    let f = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    assert_eq!(f.to_bits(), (f32::from(0xadu8) / 255.0).to_bits());
}

#[test]
fn color_fill_r5g6b5_packs_top_bits() {
    // Filling 0xdeadbeef into an R5G6B5 surface packs to the 16-bit value
    // 0xadfd (R=0xad>>3, G=0xbe>>2, B=0xef>>3).
    let bytes = d3dcolor_fill_pixel_bytes(0xdead_beef, D3DFMT_R5G6B5).unwrap();
    assert_eq!(u16::from_le_bytes([bytes[0], bytes[1]]), 0xadfd);
}

#[test]
fn color_fill_unsupported_format_is_none() {
    // Block / expanded / unmapped formats aren't encoded yet.
    assert!(d3dcolor_fill_pixel_bytes(0xffff_ffff, D3DFMT_X8R8G8B8).is_some());
    assert!(d3dcolor_fill_pixel_bytes(0xffff_ffff, 0x0000_0000).is_none());
}

#[test]
fn decl_type_to_metal_format_table() {
    // Each D3DDECLTYPE we support maps to a typed VertexFormat and a
    // size. If anyone flips a mapping here without updating both sides,
    // this catches it.
    assert_eq!(
        decl_type_to_metal_format(D3DDECLTYPE_FLOAT1),
        (VertexFormat::Float, 4)
    );
    assert_eq!(
        decl_type_to_metal_format(D3DDECLTYPE_FLOAT2),
        (VertexFormat::Float2, 8)
    );
    assert_eq!(
        decl_type_to_metal_format(D3DDECLTYPE_FLOAT3),
        (VertexFormat::Float3, 12)
    );
    assert_eq!(
        decl_type_to_metal_format(D3DDECLTYPE_FLOAT4),
        (VertexFormat::Float4, 16)
    );
    assert_eq!(
        decl_type_to_metal_format(D3DDECLTYPE_D3DCOLOR),
        (VertexFormat::UChar4NormalizedBgra, 4)
    );
    assert_eq!(
        decl_type_to_metal_format(D3DDECLTYPE_UBYTE4),
        (VertexFormat::UChar4, 4)
    );
    assert_eq!(
        decl_type_to_metal_format(D3DDECLTYPE_UBYTE4N),
        (VertexFormat::UChar4Normalized, 4)
    );
    assert_eq!(
        decl_type_to_metal_format(D3DDECLTYPE_SHORT2),
        (VertexFormat::Short2, 4)
    );
    assert_eq!(
        decl_type_to_metal_format(D3DDECLTYPE_SHORT4),
        (VertexFormat::Short4, 8)
    );
    assert_eq!(
        decl_type_to_metal_format(D3DDECLTYPE_SHORT2N),
        (VertexFormat::Short2Normalized, 4)
    );
    assert_eq!(
        decl_type_to_metal_format(D3DDECLTYPE_SHORT4N),
        (VertexFormat::Short4Normalized, 8)
    );
    assert_eq!(
        decl_type_to_metal_format(D3DDECLTYPE_USHORT2N),
        (VertexFormat::UShort2Normalized, 4)
    );
    assert_eq!(
        decl_type_to_metal_format(D3DDECLTYPE_USHORT4N),
        (VertexFormat::UShort4Normalized, 8)
    );
    assert_eq!(
        decl_type_to_metal_format(D3DDECLTYPE_FLOAT16_2),
        (VertexFormat::Half2, 4)
    );
    assert_eq!(
        decl_type_to_metal_format(D3DDECLTYPE_FLOAT16_4),
        (VertexFormat::Half4, 8)
    );
    // Unsupported types report INVALID so the caller can skip.
    assert_eq!(
        decl_type_to_metal_format(D3DDECLTYPE_UDEC3),
        (VertexFormat::Invalid, 0)
    );
    assert_eq!(
        decl_type_to_metal_format(D3DDECLTYPE_DEC3N),
        (VertexFormat::Invalid, 0)
    );
}

#[test]
fn fvf_synthesize_elements_position_normal_tex1() {
    let (elems, stride) = fvf_to_elements(D3DFVF_XYZ | D3DFVF_NORMAL | (1 << 8));
    assert_eq!(elems.len(), 3);
    assert_eq!(elems[0].usage, D3DDECLUSAGE_POSITION);
    assert_eq!(elems[0].type_, D3DDECLTYPE_FLOAT3);
    assert_eq!(elems[0].offset, 0);
    assert_eq!(elems[1].usage, D3DDECLUSAGE_NORMAL);
    assert_eq!(elems[1].offset, 12);
    assert_eq!(elems[2].usage, D3DDECLUSAGE_TEXCOORD);
    assert_eq!(elems[2].usage_index, 0);
    assert_eq!(elems[2].offset, 24);
    assert_eq!(stride, 32);
}

#[test]
fn fvf_synthesize_elements_xyzrhw_diffuse_tex1() {
    let (elems, stride) = fvf_to_elements(D3DFVF_XYZRHW | D3DFVF_DIFFUSE | (1 << 8));
    assert_eq!(elems.len(), 3);
    assert_eq!(elems[0].usage, D3DDECLUSAGE_POSITIONT);
    assert_eq!(elems[0].type_, D3DDECLTYPE_FLOAT4);
    assert_eq!(elems[1].usage, D3DDECLUSAGE_COLOR);
    assert_eq!(elems[1].usage_index, 0);
    assert_eq!(elems[1].type_, D3DDECLTYPE_D3DCOLOR);
    assert_eq!(elems[1].offset, 16);
    assert_eq!(elems[2].usage, D3DDECLUSAGE_TEXCOORD);
    assert_eq!(elems[2].offset, 20);
    assert_eq!(stride, 28);
}

#[test]
fn fvf_to_elements_matches_d3d9_blend_matrix() {
    // Each row is (type_, usage, usage_index, offset); the table maps an
    // fvf to its expected element rows via the canonical D3D9 FVF ->
    // declaration conversion. Covers every XYZBn / LASTBETA combination,
    // including the XYZB2|D3DCOLOR quirk (weight = D3DCOLOR, index =
    // UBYTE4).
    type Row = (u8, u8, u8, u16);
    let cases: &[(u32, &[Row])] = &[
        (
            D3DFVF_XYZ,
            &[(D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_POSITION, 0, 0)],
        ),
        (
            D3DFVF_XYZW,
            &[(D3DDECLTYPE_FLOAT4, D3DDECLUSAGE_POSITION, 0, 0)],
        ),
        (
            D3DFVF_XYZRHW,
            &[(D3DDECLTYPE_FLOAT4, D3DDECLUSAGE_POSITIONT, 0, 0)],
        ),
        (
            D3DFVF_XYZB1,
            &[
                (D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_POSITION, 0, 0),
                (D3DDECLTYPE_FLOAT1, D3DDECLUSAGE_BLENDWEIGHT, 0, 12),
            ],
        ),
        (
            D3DFVF_XYZB1 | D3DFVF_LASTBETA_UBYTE4,
            &[
                (D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_POSITION, 0, 0),
                (D3DDECLTYPE_UBYTE4, D3DDECLUSAGE_BLENDINDICES, 0, 12),
            ],
        ),
        (
            D3DFVF_XYZB1 | D3DFVF_LASTBETA_D3DCOLOR,
            &[
                (D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_POSITION, 0, 0),
                (D3DDECLTYPE_D3DCOLOR, D3DDECLUSAGE_BLENDINDICES, 0, 12),
            ],
        ),
        (
            D3DFVF_XYZB2,
            &[
                (D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_POSITION, 0, 0),
                (D3DDECLTYPE_FLOAT2, D3DDECLUSAGE_BLENDWEIGHT, 0, 12),
            ],
        ),
        (
            D3DFVF_XYZB2 | D3DFVF_LASTBETA_UBYTE4,
            &[
                (D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_POSITION, 0, 0),
                (D3DDECLTYPE_FLOAT1, D3DDECLUSAGE_BLENDWEIGHT, 0, 12),
                (D3DDECLTYPE_UBYTE4, D3DDECLUSAGE_BLENDINDICES, 0, 16),
            ],
        ),
        (
            D3DFVF_XYZB2 | D3DFVF_LASTBETA_D3DCOLOR,
            &[
                (D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_POSITION, 0, 0),
                (D3DDECLTYPE_D3DCOLOR, D3DDECLUSAGE_BLENDWEIGHT, 0, 12),
                (D3DDECLTYPE_UBYTE4, D3DDECLUSAGE_BLENDINDICES, 0, 16),
            ],
        ),
        (
            D3DFVF_XYZB3,
            &[
                (D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_POSITION, 0, 0),
                (D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_BLENDWEIGHT, 0, 12),
            ],
        ),
        (
            D3DFVF_XYZB3 | D3DFVF_LASTBETA_UBYTE4,
            &[
                (D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_POSITION, 0, 0),
                (D3DDECLTYPE_FLOAT2, D3DDECLUSAGE_BLENDWEIGHT, 0, 12),
                (D3DDECLTYPE_UBYTE4, D3DDECLUSAGE_BLENDINDICES, 0, 20),
            ],
        ),
        (
            D3DFVF_XYZB3 | D3DFVF_LASTBETA_D3DCOLOR,
            &[
                (D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_POSITION, 0, 0),
                (D3DDECLTYPE_FLOAT2, D3DDECLUSAGE_BLENDWEIGHT, 0, 12),
                (D3DDECLTYPE_D3DCOLOR, D3DDECLUSAGE_BLENDINDICES, 0, 20),
            ],
        ),
        (
            D3DFVF_XYZB4,
            &[
                (D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_POSITION, 0, 0),
                (D3DDECLTYPE_FLOAT4, D3DDECLUSAGE_BLENDWEIGHT, 0, 12),
            ],
        ),
        (
            D3DFVF_XYZB4 | D3DFVF_LASTBETA_UBYTE4,
            &[
                (D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_POSITION, 0, 0),
                (D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_BLENDWEIGHT, 0, 12),
                (D3DDECLTYPE_UBYTE4, D3DDECLUSAGE_BLENDINDICES, 0, 24),
            ],
        ),
        (
            D3DFVF_XYZB4 | D3DFVF_LASTBETA_D3DCOLOR,
            &[
                (D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_POSITION, 0, 0),
                (D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_BLENDWEIGHT, 0, 12),
                (D3DDECLTYPE_D3DCOLOR, D3DDECLUSAGE_BLENDINDICES, 0, 24),
            ],
        ),
        (
            D3DFVF_XYZB5,
            &[
                (D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_POSITION, 0, 0),
                (D3DDECLTYPE_FLOAT4, D3DDECLUSAGE_BLENDWEIGHT, 0, 12),
                (D3DDECLTYPE_FLOAT1, D3DDECLUSAGE_BLENDINDICES, 0, 28),
            ],
        ),
        (
            D3DFVF_XYZB5 | D3DFVF_LASTBETA_UBYTE4,
            &[
                (D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_POSITION, 0, 0),
                (D3DDECLTYPE_FLOAT4, D3DDECLUSAGE_BLENDWEIGHT, 0, 12),
                (D3DDECLTYPE_UBYTE4, D3DDECLUSAGE_BLENDINDICES, 0, 28),
            ],
        ),
        (
            D3DFVF_XYZB5 | D3DFVF_LASTBETA_D3DCOLOR,
            &[
                (D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_POSITION, 0, 0),
                (D3DDECLTYPE_FLOAT4, D3DDECLUSAGE_BLENDWEIGHT, 0, 12),
                (D3DDECLTYPE_D3DCOLOR, D3DDECLUSAGE_BLENDINDICES, 0, 28),
            ],
        ),
    ];
    for (fvf, expected) in cases {
        let (elems, _stride) = fvf_to_elements(*fvf);
        assert_eq!(
            elems.len(),
            expected.len(),
            "element count for fvf {fvf:#x}"
        );
        for (i, (ty, usage, usage_index, offset)) in expected.iter().enumerate() {
            assert_eq!(elems[i].type_, *ty, "type fvf {fvf:#x} elem {i}");
            assert_eq!(elems[i].usage, *usage, "usage fvf {fvf:#x} elem {i}");
            assert_eq!(
                elems[i].usage_index, *usage_index,
                "usage_index fvf {fvf:#x} elem {i}"
            );
            assert_eq!(elems[i].offset, *offset, "offset fvf {fvf:#x} elem {i}");
            assert_eq!(elems[i].stream, 0, "stream fvf {fvf:#x} elem {i}");
            assert_eq!(elems[i].method, 0, "method fvf {fvf:#x} elem {i}");
        }
    }
}

#[test]
fn fvf_synthesize_elements_xyzb3() {
    let (elems, stride) = fvf_to_elements(D3DFVF_XYZB3);
    // XYZB3 with no LASTBETA flag: 3 floats position + 3 blend weights.
    assert_eq!(elems.len(), 2);
    assert_eq!(elems[0].usage, D3DDECLUSAGE_POSITION);
    assert_eq!(elems[1].usage, D3DDECLUSAGE_BLENDWEIGHT);
    assert_eq!(elems[1].type_, D3DDECLTYPE_FLOAT3);
    assert_eq!(stride, 24);
}

#[test]
fn resolve_attrs_for_vs_swaps_register_indices() {
    // VS declares position on v2 and texcoord0 on v7 — the resolved
    // attr_index must match the register, not the FVF convention.
    let semantics = vec![
        InputSemantic {
            usage: DeclUsage::Position,
            usage_index: 0,
            register_index: 2,
        },
        InputSemantic {
            usage: DeclUsage::Texcoord,
            usage_index: 0,
            register_index: 7,
        },
    ];
    let elems = [pos3(), tex0(12)];
    let resolved = resolve_attrs_for_vs(&elems, &semantics);
    assert_eq!(resolved.attrs.len(), 2);
    assert_eq!(resolved.attrs[0].attr_index, 2);
    assert_eq!(resolved.attrs[1].attr_index, 7);
    assert_eq!(resolved.extents[0], 20);
    assert_eq!(resolved.used_streams, 0b1);
}

#[test]
fn resolve_attrs_skips_unused_semantics() {
    // VS declares only POSITION; NORMAL in the decl is silently dropped.
    let semantics = vec![InputSemantic {
        usage: DeclUsage::Position,
        usage_index: 0,
        register_index: 0,
    }];
    let elems = [
        pos3(),
        D3DVERTEXELEMENT9 {
            stream: 0,
            offset: 12,
            type_: D3DDECLTYPE_FLOAT3,
            method: 0,
            usage: D3DDECLUSAGE_NORMAL,
            usage_index: 0,
        },
    ];
    let resolved = resolve_attrs_for_vs(&elems, &semantics);
    assert_eq!(resolved.attrs.len(), 1);
    assert_eq!(resolved.attrs[0].attr_index, 0);
    // The extent still covers the normal element so the vertex buffer
    // layout is correct even with an unused attribute.
    assert_eq!(resolved.extents[0], 24);
}

#[test]
fn resolve_attrs_for_ff_matches_ff_convention() {
    // POSITION → attr(0), TEXCOORD0 → attr(4). Must agree with
    // `crate::dxso::ff_attr_index_for_semantic`.
    let elems = [pos3(), tex0(12)];
    let resolved = resolve_attrs_for_ff(&elems);
    assert_eq!(resolved.attrs.len(), 2);
    assert_eq!(resolved.attrs[0].attr_index, 0);
    assert_eq!(resolved.attrs[1].attr_index, 4);
    assert_eq!(resolved.extents[0], 20);
}

#[test]
fn resolve_attrs_keeps_each_stream_separate() {
    // POSITION on stream 0, COLOR0 on stream 1 at offset 0, an unconsumed
    // NORMAL on stream 1 past it: stream 1's extent covers the normal,
    // the colour attribute points at buffer 1, and both streams are used.
    let elems = [
        pos3(),
        D3DVERTEXELEMENT9 {
            stream: 1,
            offset: 0,
            type_: D3DDECLTYPE_D3DCOLOR,
            method: 0,
            usage: D3DDECLUSAGE_COLOR,
            usage_index: 0,
        },
        D3DVERTEXELEMENT9 {
            stream: 1,
            offset: 4,
            type_: D3DDECLTYPE_FLOAT3,
            method: 0,
            usage: D3DDECLUSAGE_NORMAL,
            usage_index: 0,
        },
    ];
    let semantics = vec![
        InputSemantic {
            usage: DeclUsage::Position,
            usage_index: 0,
            register_index: 0,
        },
        InputSemantic {
            usage: DeclUsage::Color,
            usage_index: 0,
            register_index: 1,
        },
    ];
    let resolved = resolve_attrs_for_vs(&elems, &semantics);
    assert_eq!(resolved.attrs.len(), 2);
    assert_eq!(resolved.attrs[0].buffer_index, 0);
    assert_eq!(resolved.attrs[1].buffer_index, 1);
    assert_eq!(resolved.attrs[1].attr_index, 1);
    assert_eq!(resolved.extents[0], 12);
    assert_eq!(resolved.extents[1], 16);
    assert_eq!(resolved.used_streams, 0b11);

    // A stream that only carries unconsumed elements is not used, but its
    // extent is still reported.
    let resolved = resolve_attrs_for_vs(&elems, &semantics[..1]);
    assert_eq!(resolved.used_streams, 0b1);
    assert_eq!(resolved.extents[1], 16);

    // The FF path maps streams the same way.
    let resolved = resolve_attrs_for_ff(&elems);
    assert_eq!(resolved.attrs.len(), 3);
    assert_eq!(resolved.attrs[1].buffer_index, 1);
    assert_eq!(resolved.used_streams, 0b11);
}

#[test]
fn resolve_attrs_drops_streams_past_the_slot_table() {
    let elems = [
        pos3(),
        D3DVERTEXELEMENT9 {
            stream: 16,
            offset: 0,
            type_: D3DDECLTYPE_D3DCOLOR,
            method: 0,
            usage: D3DDECLUSAGE_COLOR,
            usage_index: 0,
        },
    ];
    let resolved = resolve_attrs_for_ff(&elems);
    assert_eq!(resolved.attrs.len(), 1);
    assert_eq!(resolved.used_streams, 0b1);
    let layout = ff_vs_layout_from_elements(&elems, true);
    assert!(
        !layout.has_color0(),
        "dropped element leaves no flag behind"
    );
}

fn end() -> D3DVERTEXELEMENT9 {
    D3DVERTEXELEMENT9 {
        stream: D3DDECL_END_STREAM,
        offset: 0,
        type_: mtld3d_types::D3DDECLTYPE_UNUSED,
        method: 0,
        usage: 0,
        usage_index: 0,
    }
}

#[test]
fn pack_vertex_decl_hash_stable_across_calls() {
    let elems = [pos3(), tex0(12), end()];
    let h_a = pack_vertex_decl(&elems).expect("pack a").hash;
    let h_b = pack_vertex_decl(&elems).expect("pack b").hash;
    assert_eq!(h_a, h_b);
    let swapped = [pos3(), tex0(16), end()];
    let h_c = pack_vertex_decl(&swapped).expect("pack c").hash;
    assert_ne!(h_a, h_c);
}

#[test]
fn pack_vertex_decl_multi_stream_distinct_hash_and_mask() {
    // Two layouts that differ *only* by stream must hash differently so
    // the pipeline cache keeps them apart, and the stream mask names the
    // streams the draw path has to snapshot.
    let on_stream = |stream| D3DVERTEXELEMENT9 {
        stream,
        offset: 0,
        type_: D3DDECLTYPE_FLOAT3,
        method: 0,
        usage: D3DDECLUSAGE_POSITION,
        usage_index: 0,
    };
    let a = pack_vertex_decl(&[on_stream(0), end()]).expect("stream 0 accepted");
    let b = pack_vertex_decl(&[on_stream(1), end()]).expect("stream 1 accepted");
    assert_ne!(a.hash, b.hash, "stream must participate in the decl hash");
    assert_eq!(a.stream_mask, 0b01);
    assert_eq!(b.stream_mask, 0b10);
    let both = pack_vertex_decl(&[on_stream(0), tex0(0), on_stream(3), end()]).expect("pack");
    assert_eq!(both.stream_mask, 0b1001);
    // A stream past the slot table is accepted (D3D9 validates structure
    // only) but contributes no bit.
    let wide = pack_vertex_decl(&[on_stream(0), on_stream(16), end()]).expect("pack");
    assert_eq!(wide.stream_mask, 0b1);
}

#[test]
fn pack_vertex_decl_requires_terminator() {
    assert!(pack_vertex_decl(&[pos3()]).is_none());
}

#[test]
fn pack_vertex_decl_preserves_terminator_in_output() {
    let elems = [pos3(), tex0(12), end()];
    let packed = pack_vertex_decl(&elems).expect("pack").elements_with_end;
    assert_eq!(packed.len(), 3);
    assert_eq!(packed.last().unwrap().stream, D3DDECL_END_STREAM);
}

#[test]
fn ff_vs_layout_clamps_tex_coord_count_to_8() {
    // A vertex declaration that claims TEXCOORD at usage_index = 12
    // must not produce tex_coord_count > 8 — FfVsKey's per-stage
    // arrays are [u8; 8] and OOB-crashed the encoder thread.
    let elements = [
        pos3(),
        D3DVERTEXELEMENT9 {
            stream: 0,
            offset: 12,
            type_: D3DDECLTYPE_FLOAT2,
            method: 0,
            usage: D3DDECLUSAGE_TEXCOORD,
            usage_index: 12,
        },
    ];
    let layout = ff_vs_layout_from_elements(&elements, false);
    assert_eq!(layout.tex_coord_count, 8);
}

#[test]
fn ff_vs_layout_in_spec_usage_index_7_yields_8() {
    let elements = [
        pos3(),
        D3DVERTEXELEMENT9 {
            stream: 0,
            offset: 12,
            type_: D3DDECLTYPE_FLOAT2,
            method: 0,
            usage: D3DDECLUSAGE_TEXCOORD,
            usage_index: 7,
        },
    ];
    let layout = ff_vs_layout_from_elements(&elements, false);
    assert_eq!(layout.tex_coord_count, 8);
}

#[test]
fn ff_vs_layout_single_tex0_yields_1() {
    let layout = ff_vs_layout_from_elements(&[pos3(), tex0(12)], false);
    assert_eq!(layout.tex_coord_count, 1);
}

#[test]
fn d3d_depth_bias_zero_passes_through() {
    // D3DRS_DEPTHBIAS default is 0.0 (u32 0). Most draws don't touch
    // it — the scaled output must stay exactly zero so games that
    // never write the state see no rasterizer offset.
    assert_eq!(d3d_depth_bias_to_metal(0).to_bits(), 0.0_f32.to_bits());
}

#[test]
fn d3d_depth_bias_scales_by_two_pow_23() {
    // D3D9 spec: 1 ULP at the depth resolution. Metal's setDepthBias
    // takes the value in absolute float units of the depth format.
    // mtld3d's depth always resolves to Depth32Float (mantissa = 23
    // bits), so the scale is 2^23.
    let raw = 1.0f32.to_bits();
    let scaled = d3d_depth_bias_to_metal(raw);
    // 2^23 = 8_388_608.0 is exactly representable in f32; bit-equality holds.
    assert_eq!(scaled.to_bits(), 8_388_608.0_f32.to_bits());
}

#[test]
fn d3d_depth_bias_negative_pushes_toward_camera() {
    // Negative bias is the canonical decal-pull-forward direction.
    // Sign must be preserved through the scale.
    // raw = -1.0 / 2^23 → scale × raw = -1.0
    let raw = (-(1.0_f32 / 8_388_608.0_f32)).to_bits();
    let scaled = d3d_depth_bias_to_metal(raw);
    assert!((scaled - -1.0).abs() < 1e-6);
}

#[test]
fn looks_like_decal_fires_on_alpha_blended_no_bias() {
    // Canonical decal pattern: depth-test on, depth-write off,
    // alpha-blend on, game's DEPTHBIAS + SLOPESCALEDEPTHBIAS both
    // zero. Predicate fires → caller substitutes
    // IMPLICIT_DECAL_BIAS_RAW for the zero game bias.
    let inputs = DecalHeuristicInputs {
        depth_enable: 1,
        depth_write: 0,
        blend_enable: 1,
        raw_depth_bias: 0,
        raw_slope_scale: 0,
    };
    assert!(looks_like_decal(inputs));
}

#[test]
fn looks_like_decal_skips_alpha_blended_depth_writer() {
    // An alpha-blended draw that ALSO writes depth is not a decal:
    // the depth-write prong excludes it, so it keeps the game's own
    // bias. Widening the predicate to such draws would need a
    // different signal (e.g. D3DRS_ALPHATESTENABLE).
    let inputs = DecalHeuristicInputs {
        depth_enable: 1,
        depth_write: 1,
        blend_enable: 1,
        raw_depth_bias: 0,
        raw_slope_scale: 0,
    };
    assert!(!looks_like_decal(inputs));
}

#[test]
fn looks_like_decal_skips_game_supplied_bias() {
    // Alpha-blended decal-shaped draw whose game-side
    // D3DRS_DEPTHBIAS is already non-zero. The predicate declines,
    // so the game's own bias is left alone rather than clobbered.
    let inputs = DecalHeuristicInputs {
        depth_enable: 1,
        depth_write: 0,
        blend_enable: 1,
        raw_depth_bias: 0x3a83_126f, // ~ +1e-3 as f32 bits
        raw_slope_scale: 0,
    };
    assert!(!looks_like_decal(inputs));
}

#[test]
fn looks_like_decal_skips_opaque_draw() {
    // No alpha blend → not a decal pattern. Solid geometry that
    // happens to disable depth-write (e.g. a deferred normals
    // prepass) shouldn't be pulled toward camera.
    let inputs = DecalHeuristicInputs {
        depth_enable: 1,
        depth_write: 0,
        blend_enable: 0,
        raw_depth_bias: 0,
        raw_slope_scale: 0,
    };
    assert!(!looks_like_decal(inputs));
}

#[test]
fn implicit_decal_bias_scales_to_safe_metal_band() {
    // Magnitude band rationale:
    // (a) > ~500 Metal units swamps the depth-buffer's 2^-23
    //     step plus the structural eye-space delta observed
    //     between two SM3 pipelines on Apple Silicon at grazing
    //     angles;
    // (b) < ~5000 keeps flat decals from punching through
    //     adjacent geometry on steep terrain.
    // Tune the constant if a future workload forces it out of
    // this band; the test catches accidental order-of-magnitude
    // changes.
    let metal = d3d_depth_bias_to_metal(IMPLICIT_DECAL_BIAS_RAW);
    assert!(
        metal < 0.0,
        "implicit bias must pull toward camera, got {metal}"
    );
    let mag = -metal;
    assert!(mag > 500.0, "magnitude {mag} too small to swamp ULP noise");
    assert!(
        mag < 5000.0,
        "magnitude {mag} risks punching through terrain"
    );
}

#[test]
fn d3d_to_metal_blend_op_table() {
    assert_eq!(d3d_to_metal_blend_op(1), BlendOperation::Add);
    assert_eq!(d3d_to_metal_blend_op(2), BlendOperation::Subtract);
    assert_eq!(d3d_to_metal_blend_op(3), BlendOperation::ReverseSubtract);
    assert_eq!(d3d_to_metal_blend_op(4), BlendOperation::Min);
    assert_eq!(d3d_to_metal_blend_op(5), BlendOperation::Max);
    // Unknown → Add (with warn).
    assert_eq!(d3d_to_metal_blend_op(0), BlendOperation::Add);
    assert_eq!(d3d_to_metal_blend_op(99), BlendOperation::Add);
}
