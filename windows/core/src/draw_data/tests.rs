use mtld3d_shared::command_header::COMMAND_HEADER_BYTES;

use super::*;
use crate::encoder_draw::{
    draw_payload_size,
    draw_record::{DrawView, VertexView},
    write_draw_into,
};

#[test]
fn decoded_extra_streams_borrow_command_bytes_and_preserve_order() {
    let stream = |index| StreamBinding {
        stream: index,
        buffer_id: BufferId::new_unique(),
        backing_ptr: 0,
        backing_len: 4096,
        backing_generation: u64::from(index),
        offset: u32::from(index) * 16,
        stride: 16,
        freq: 1,
    };
    let draw = DrawOp {
        metal_prim: PrimitiveType::Triangle,
        vertex_source: VertexSource::Bound {
            first: stream(1),
            extra: ExtraStreams::Owned([stream(3), stream(7)].into()),
            stream0_freq: 1,
        },
        index_source: IndexSource::None {
            start_vertex: 2,
            vertex_count: 3,
        },
    };
    let mut arena = ScratchArena::new();
    let size = draw_payload_size(&draw).unwrap();
    let first = arena
        .write_command(4, 0, size, |destination| {
            write_draw_into(&draw, destination, size)?;
            Ok(size)
        })
        .unwrap();
    // SAFETY: the arena retains this initialized fixed payload through every view below.
    let bytes = unsafe {
        core::slice::from_raw_parts(
            (first.address + COMMAND_HEADER_BYTES as u64) as *const u8,
            size,
        )
    };
    let restored = DrawView::new(bytes).unwrap();
    let VertexView::Bound { records, .. } = restored.vertices() else {
        unreachable!()
    };
    assert_eq!(records.as_ptr().cast::<u8>(), bytes[16..].as_ptr());
    assert_eq!(
        records.iter().map(|value| value.stream).collect::<Vec<_>>(),
        [1, 3, 7]
    );
    let second = arena
        .write_command(4, 0, size, |destination| {
            write_draw_into(&draw, destination, size)?;
            Ok(size)
        })
        .unwrap();
    assert_ne!(first.address, second.address);
    assert_eq!(
        records
            .iter()
            .map(|value| value.generation)
            .collect::<Vec<_>>(),
        [1, 3, 7]
    );
}

#[test]
fn a_missing_depth_texture_falls_back_to_the_depth_kind() {
    let variant = VariantKey {
        depth_sampler_mask: 0b0110,
        depth_fetch_mask: 0b0100,
        cube_sampler_mask: 0b1000,
        volume_sampler_mask: 0b1_0000,
        ..VariantKey::default()
    };
    assert_eq!(
        missing_texture_kind(variant, 1),
        NullTextureKind::Depth2D,
        "a comparison slot"
    );
    assert_eq!(
        missing_texture_kind(variant, 2),
        NullTextureKind::Depth2D,
        "a raw-depth slot is depth2d too"
    );
    assert_eq!(
        missing_texture_kind(variant, 3),
        NullTextureKind::TextureCube
    );
    assert_eq!(missing_texture_kind(variant, 4), NullTextureKind::Texture3D);
    assert_eq!(missing_texture_kind(variant, 0), NullTextureKind::Texture2D);
    assert_eq!(
        missing_texture_kind(variant, 16),
        NullTextureKind::Texture2D,
        "a slot past the masks"
    );
}

#[test]
fn a_lod_table_source_keys_apart_and_keeps_every_other_field() {
    let source = ProgrammableVsSource {
        vs_id: ProgramId::from_tokens(&[0xFFFE_0300, 0xFFFF]),
        max_const_used: 12,
        provided_input_mask: 0x3,
        flags: ShaderSourceFlags::RELATIVE | ShaderSourceFlags::INTEGER,
        clip_plane_count: 2,
        sampler_kinds: crate::dxso::VsSamplerKinds {
            volume_mask: 0b0010,
            cube_mask: 0b0100,
            lod_table: false,
        },
        reserved: [0; 7],
    };
    let tabled = source.with_lod_table();
    assert!(tabled.sampler_kinds.lod_table);
    assert_eq!(
        (
            tabled.vs_id,
            tabled.max_const_used,
            tabled.provided_input_mask,
            tabled.flags.bits(),
            tabled.clip_plane_count,
            tabled.sampler_kinds.volume_mask,
            tabled.sampler_kinds.cube_mask,
            tabled.reserved,
        ),
        (
            source.vs_id,
            source.max_const_used,
            source.provided_input_mask,
            source.flags.bits(),
            source.clip_plane_count,
            source.sampler_kinds.volume_mask,
            source.sampler_kinds.cube_mask,
            source.reserved,
        )
    );
    assert_ne!(
        VsSourceView::Programmable(&tabled).disk_key(),
        VsSourceView::Programmable(&source).disk_key(),
        "a shader reading the table is a library of its own"
    );
}

#[test]
fn a_zero_padded_copy_keeps_the_bytes_and_zeroes_the_tail() {
    let mut arena = ScratchArena::new();
    // Dirty the region first: a reused arena chunk carries stale bytes.
    // SAFETY: nothing reads the token after the arena is cleared.
    let _dirty = unsafe { arena_alloc_bytes(&mut arena, &[0xAA; 64]) };
    arena.clear();
    // SAFETY: the arena outlives the token, read before the arena drops.
    let copy = unsafe { arena_alloc_zero_padded(&mut arena, &[1, 2, 3, 4, 5, 6], 12) };
    assert_eq!(copy.as_slice(), [1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0]);
}

#[test]
fn stream_layouts_mark_the_streams_whose_stride_is_short_of_the_extent() {
    let attrs = [VertexAttrDesc {
        attr_index: 0,
        buffer_index: 0,
        offset: 0,
        format: mtld3d_shared::mtl::VertexFormat::Float4,
    }];
    let mut extents = [0; 16];
    extents[0] = 32;
    extents[1] = 16;
    extents[2] = 12;
    let header = DeclarationHeader {
        vdecl_hash: 0,
        extents,
        count: 1,
        used_streams: 0b111,
        reserved: 0,
    };
    // SAFETY: both locals outlive the token, which is only read below.
    let snapshot = unsafe { AttrSnapshot::new(NonNull::from(&attrs[0]), NonNull::from(&header)) };
    let strides = [16, 0, 16];
    let (layouts, crossing) = stream_layouts_with(&snapshot, |stream, extent| StreamLayout {
        stride: layout_stride(strides[stream as usize], extent),
        step: VertexStepFunction::PerVertex,
        step_rate: 1,
    });
    // Stream 0 steps 16 bytes under a 32-byte extent; stream 1's zero stride
    // steps by its extent and stream 2's stride covers its extent.
    assert_eq!(crossing, 0b001);
    assert_eq!(layouts[0].stride, 16);
    assert_eq!(layouts[1].stride, 16);
    assert!(!layouts[3].is_used());
}
