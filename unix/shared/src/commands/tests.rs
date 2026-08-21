use super::*;

/// Decode the packing that the unix-side `CommandType::DrawIndexedPrimitives` handler uses.
///
/// Kept in one place so the encoder and decoder stay in sync.
fn decode_draw_indexed(cmd: &Command) -> (u32, u32, u32, u64, u32, i32) {
    let primitive_type = cmd.param_a;
    let index_buffer = cmd.param_b;
    let offset = u32::try_from(cmd.param_c >> 32).expect("high 32 bits fit u32");
    let base_vertex = u32::try_from(cmd.param_c & 0xFFFF_FFFF)
        .expect("low 32 bits fit u32")
        .cast_signed();
    let (index_count, index_type, _) = Command::unpack_indexed_draw_counts(cmd.param_d);
    (
        primitive_type,
        index_count,
        index_type,
        index_buffer,
        offset,
        base_vertex,
    )
}

#[test]
fn draw_indexed_primitives_roundtrip_zero_base() {
    let cmd = Command::draw_indexed_primitives(
        PrimitiveType::Triangle,
        42,
        IndexType::UInt16,
        0xDEAD_BEEF_0000_0000,
        128,
        0,
        1,
    );
    let (prim, cnt, ty, buf, off, base) = decode_draw_indexed(&cmd);
    assert_eq!(prim, PrimitiveType::Triangle as u32);
    assert_eq!(cnt, 42);
    assert_eq!(ty, IndexType::UInt16 as u32);
    assert_eq!(buf, 0xDEAD_BEEF_0000_0000);
    assert_eq!(off, 128);
    assert_eq!(base, 0);
}

#[test]
fn draw_indexed_primitives_roundtrip_positive_base() {
    let cmd = Command::draw_indexed_primitives(
        PrimitiveType::TriangleStrip,
        6,
        IndexType::UInt32,
        1,
        42,
        100_000,
        1,
    );
    let (_, _, _, _, off, base) = decode_draw_indexed(&cmd);
    assert_eq!(off, 42);
    assert_eq!(base, 100_000);
}

#[test]
fn draw_indexed_primitives_roundtrip_negative_base() {
    // Sign-extension guard: `-100_000` must survive as `-100_000` on
    // the other end, not as a large positive u32 pattern.
    let cmd = Command::draw_indexed_primitives(
        PrimitiveType::TriangleStrip,
        6,
        IndexType::UInt32,
        1,
        0,
        -100_000,
        1,
    );
    let (_, _, _, _, off, base) = decode_draw_indexed(&cmd);
    assert_eq!(off, 0);
    assert_eq!(base, -100_000);
}

#[test]
fn indexed_draw_counts_pack_without_overlap() {
    // A full 32-bit index count next to the 23-bit D3D9 instance ceiling
    // and a non-zero index type: each field decodes to what went in.
    let packed = Command::pack_indexed_draw_counts(u32::MAX, IndexType::UInt32, 0x7F_FFFF);
    assert_eq!(
        Command::unpack_indexed_draw_counts(packed),
        (u32::MAX, IndexType::UInt32 as u32, 0x7F_FFFF)
    );
    // Both indexed draw forms carry the same packing.
    let up = Command::draw_indexed_primitives_up(
        PrimitiveType::Triangle,
        6,
        IndexType::UInt16,
        0xA000,
        12,
        4,
    );
    assert_eq!(
        Command::unpack_indexed_draw_counts(up.param_d),
        (6, IndexType::UInt16 as u32, 4)
    );
}

#[test]
fn blit_command_layout_matches_wow64() {
    assert_eq!(core::mem::align_of::<BlitCommand>(), 8);
    // 4 cmd + 4 mip_level + 8 src_handle + 8 dst_handle + 8
    // src_offset + 8 bytes_per_row + 4 origin_x + 4 origin_y +
    // 4 region_w + 4 region_h + 8 dst_offset + 8 byte_size +
    // 4 depth + 4 bytes_per_image + 4 dst_mip_level + 4 pad0 = 88
    assert_eq!(core::mem::size_of::<BlitCommand>(), 88);
}

#[test]
fn copy_buffer_to_buffer_round_trip() {
    let cmd = BlitCommand::copy_buffer_to_buffer(&CopyBufferToBufferInfo {
        src_buffer: 0xAAAA_1111_2222_3333,
        dst_buffer: 0xBBBB_4444_5555_6666,
        src_offset: 0x1000,
        dst_offset: 0x2000,
        byte_size: 0x8000,
    });
    assert_eq!(cmd.cmd, BlitCommandType::CopyBufferToBuffer as u32);
    assert_eq!(cmd.src_handle, 0xAAAA_1111_2222_3333);
    assert_eq!(cmd.dst_handle, 0xBBBB_4444_5555_6666);
    assert_eq!(cmd.src_offset, 0x1000);
    assert_eq!(cmd.dst_offset, 0x2000);
    assert_eq!(cmd.byte_size, 0x8000);
}

#[test]
fn copy_buffer_to_texture_round_trip() {
    let cmd = BlitCommand::copy_buffer_to_texture(&CopyBufferToTextureInfo {
        buffer_handle: 0xAAAA_BBBB_CCCC_DDDD,
        buffer_offset: 256,
        bytes_per_row: 1024,
        texture_handle: 0x1111_2222_3333_4444,
        destination_slice: 0,
        mip_level: 2,
        origin_x: 10,
        origin_y: 20,
        region_w: 128,
        region_h: 64,
        depth: 1,
        bytes_per_image: 1024 * 64,
    });
    assert_eq!(cmd.cmd, BlitCommandType::CopyBufferToTexture as u32);
    assert_eq!(cmd.src_handle, 0xAAAA_BBBB_CCCC_DDDD);
    assert_eq!(cmd.dst_handle, 0x1111_2222_3333_4444);
    assert_eq!(cmd.dst_offset, 0);
    assert_eq!(cmd.src_offset, 256);
    assert_eq!(cmd.bytes_per_row, 1024);
    assert_eq!(cmd.mip_level, 2);
    assert_eq!(cmd.origin_x, 10);
    assert_eq!(cmd.origin_y, 20);
    assert_eq!(cmd.region_w, 128);
    assert_eq!(cmd.region_h, 64);
    assert_eq!(cmd.depth, 1);
    assert_eq!(cmd.bytes_per_image, 1024 * 64);
}

#[test]
fn cube_upload_packs_face_into_existing_destination_offset() {
    let cmd = BlitCommand::copy_buffer_to_texture(&CopyBufferToTextureInfo {
        buffer_handle: 1,
        buffer_offset: 0,
        bytes_per_row: 16,
        texture_handle: 2,
        destination_slice: 5,
        mip_level: 0,
        origin_x: 0,
        origin_y: 0,
        region_w: 4,
        region_h: 4,
        depth: 1,
        bytes_per_image: 64,
    });
    assert_eq!(cmd.dst_offset, 5);
    assert_eq!(core::mem::size_of_val(&cmd), 88);
}

#[test]
fn generate_mipmaps_packs_texture_handle_only() {
    let cmd = BlitCommand::generate_mipmaps(0xCAFE_F00D_0BAD_BEEF);
    assert_eq!(cmd.cmd, BlitCommandType::GenerateMipmaps as u32);
    assert_eq!(cmd.dst_handle, 0xCAFE_F00D_0BAD_BEEF);
    assert_eq!(cmd.src_handle, 0);
    assert_eq!(cmd.mip_level, 0);
    assert_eq!(cmd.src_offset, 0);
    assert_eq!(cmd.dst_offset, 0);
    assert_eq!(cmd.bytes_per_row, 0);
    assert_eq!(cmd.origin_x, 0);
    assert_eq!(cmd.origin_y, 0);
    assert_eq!(cmd.region_w, 0);
    assert_eq!(cmd.region_h, 0);
    assert_eq!(cmd.byte_size, 0);
}

#[test]
fn copy_texture_to_texture_full_mip_zeros_sub_rect_fields() {
    let cmd = BlitCommand::copy_texture_to_texture_full_mip(0xDEAD, 0xBEEF, 1, 512, 256);
    assert_eq!(cmd.cmd, BlitCommandType::CopyTextureToTexture as u32);
    assert_eq!(cmd.src_handle, 0xDEAD);
    assert_eq!(cmd.dst_handle, 0xBEEF);
    assert_eq!(cmd.mip_level, 1);
    assert_eq!(cmd.src_offset, 0);
    assert_eq!(cmd.bytes_per_row, 0);
    assert_eq!(cmd.origin_x, 0);
    assert_eq!(cmd.origin_y, 0);
    assert_eq!(cmd.region_w, 512);
    assert_eq!(cmd.region_h, 256);
    assert_eq!(cmd.dst_offset, 0);
}

#[test]
fn set_stencil_reference_roundtrip() {
    let cmd = Command::set_stencil_reference(0x7F);
    assert_eq!(cmd.cmd, CommandType::SetStencilReference as u32);
    assert_eq!(cmd.param_a, 0x7F);
}

#[test]
fn set_blend_color_roundtrip() {
    let cmd = Command::set_blend_color(0.25, 0.5, 0.75, 1.0);
    assert_eq!(cmd.cmd, CommandType::SetBlendColor as u32);
    let to_u32 = |v: u64| u32::try_from(v).expect("packed f32 bits fit u32");
    assert_eq!(cmd.param_a, 0.25_f32.to_bits());
    assert_eq!(to_u32(cmd.param_b), 0.5_f32.to_bits());
    assert_eq!(to_u32(cmd.param_c), 0.75_f32.to_bits());
    assert_eq!(to_u32(cmd.param_d), 1.0_f32.to_bits());
}

#[test]
fn set_depth_bias_roundtrip() {
    let cmd = Command::set_depth_bias(-1.5, 0.25);
    assert_eq!(cmd.cmd, CommandType::SetDepthBias as u32);
    assert_eq!(cmd.param_a, (-1.5_f32).to_bits());
    assert_eq!(
        u32::try_from(cmd.param_b).expect("packed f32 bits fit u32"),
        0.25_f32.to_bits(),
    );
    assert_eq!(cmd.param_c, 0);
    assert_eq!(cmd.param_d, 0);
}

#[test]
fn copy_texture_to_texture_sub_rect_packs_dst_origin() {
    let cmd = BlitCommand::copy_texture_to_texture_sub_rect(&CopyTextureSubRectInfo {
        src_texture: 0xAAAA,
        dst_texture: 0xBBBB,
        mip_level: 2,
        dst_mip_level: 1,
        src_origin_x: 16,
        src_origin_y: 32,
        dst_origin_x: 100,
        dst_origin_y: 200,
        region_w: 64,
        region_h: 48,
    });
    assert_eq!(cmd.cmd, BlitCommandType::CopyTextureToTexture as u32);
    assert_eq!(cmd.src_handle, 0xAAAA);
    assert_eq!(cmd.dst_handle, 0xBBBB);
    assert_eq!(cmd.mip_level, 2);
    assert_eq!(cmd.dst_mip_level, 1);
    assert_eq!(cmd.origin_x, 16);
    assert_eq!(cmd.origin_y, 32);
    assert_eq!(cmd.region_w, 64);
    assert_eq!(cmd.region_h, 48);
    // dst origin packed as (y << 32) | x — decoder splits the same way.
    assert_eq!(cmd.dst_offset & 0xFFFF_FFFF, 100);
    assert_eq!((cmd.dst_offset >> 32) & 0xFFFF_FFFF, 200);
}

#[test]
fn draw_indexed_primitives_packs_index_count_and_type() {
    let cmd = Command::draw_indexed_primitives(
        PrimitiveType::Triangle,
        u32::MAX,
        IndexType::UInt32,
        0,
        0,
        0,
        1,
    );
    let (_, cnt, ty, _, _, _) = decode_draw_indexed(&cmd);
    assert_eq!(cnt, u32::MAX);
    assert_eq!(ty, IndexType::UInt32 as u32);
}
