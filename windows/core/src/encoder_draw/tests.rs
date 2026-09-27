use mtld3d_shared::{
    encoder_wire::{FrameSlab, WireError, WireReader},
    mtl::{IndexType, PrimitiveType},
};

use super::{DrawReader, DrawWriter, SnapshotAttributes, SnapshotDelta, store};
use crate::{
    draw_data::{
        CurrentSnapshot, DrawOp, IndexSource, ScratchSlice, StreamBinding, VertexSource, VsSource,
        VsSourcePtr,
    },
    ids::{BufferId, ProgramId},
    scratch::ScratchArena,
};

fn encode_draw(draw: &DrawOp) -> FrameSlab {
    let mut slab = FrameSlab::new();
    let mut writer = DrawWriter::new();
    slab.push_record(1, |output| writer.encode_draw(draw, output))
        .unwrap();
    // SAFETY: each fixture retains the immutable byte ranges it encodes; bound buffers
    // remain numeric identities and are never dereferenced by either parser.
    let mut checked = unsafe { WireReader::new_trusted(&slab.as_bytes()[6..]) };
    // SAFETY: same fixture lifetime applies to the reconstruction comparison.
    let mut wire = unsafe { WireReader::new_trusted(&slab.as_bytes()[6..]) };
    // SAFETY: fixture byte owners outlive this immediately dropped decoded operation.
    let mut decoder = unsafe { DrawReader::new() };
    assert_eq!(
        super::validate_wire_draw(&mut checked, |_, _| Ok(())).is_ok(),
        decoder.decode_draw(&mut wire).is_ok(),
        "stack-only validation must agree with native reconstruction"
    );
    slab
}

#[test]
fn all_index_sources_round_trip_as_explicit_fields() {
    let index_bytes = [0u8, 0, 1, 0, 2, 0];
    // SAFETY: index_bytes stays initialized and immutable for every token in this test.
    let borrowed =
        unsafe { ScratchSlice::from_raw_parts(std::ptr::NonNull::from(&index_bytes[0]), 6) };
    let sources = [
        IndexSource::None {
            start_vertex: 2,
            vertex_count: 3,
        },
        IndexSource::Fan {
            start_vertex: 4,
            primitive_count: 7,
        },
        IndexSource::Up {
            bytes: borrowed,
            index_count: 3,
            index_type: IndexType::UInt16,
        },
        IndexSource::Generated {
            data: borrowed,
            index_count: 3,
            index_type: IndexType::UInt16,
            min_vertex: 0,
            max_vertex: 2,
        },
        IndexSource::Bound {
            buffer_id: BufferId::new_unique(),
            backing_ptr: index_bytes.as_ptr() as usize,
            backing_len: index_bytes.len(),
            backing_generation: 9,
            offset: 0,
            index_count: 3,
            index_type: IndexType::UInt16,
            base_vertex: -4,
        },
    ];
    for index_source in sources {
        let draw = DrawOp {
            metal_prim: PrimitiveType::Triangle,
            vertex_source: VertexSource::Up {
                bytes: captured(&[1, 2, 3, 4]),
                size: 4,
                stride: 4,
            },
            index_source,
        };
        let encoded = encode_draw(&draw);
        // SAFETY: every borrowed range is index_bytes, retained by this test.
        let mut stream = unsafe { WireReader::new_trusted(encoded.as_bytes()) };
        let mut record = stream.next_record().unwrap().unwrap();
        // SAFETY: all byte ranges and decoded values stay live until the loop ends.
        let mut decoder = unsafe { DrawReader::new() };
        let restored = decoder.decode_draw(&mut record.payload).unwrap();
        assert!(record.payload.is_empty());
        assert_eq!(encode_draw(&restored).as_bytes(), encoded.as_bytes());
    }
}

#[test]
fn bound_stream_fields_round_trip_and_reject_untrusted_addresses() {
    let data = [1u8; 16];
    let draw = DrawOp {
        metal_prim: PrimitiveType::Point,
        vertex_source: VertexSource::Bound {
            first: StreamBinding {
                stream: 2,
                buffer_id: BufferId::new_unique(),
                backing_ptr: data.as_ptr() as usize,
                backing_len: data.len(),
                backing_generation: 11,
                offset: 4,
                stride: 4,
                freq: 1,
            },
            extra: Box::new([]),
            stream0_freq: 7,
        },
        index_source: IndexSource::None {
            start_vertex: 0,
            vertex_count: 3,
        },
    };
    let encoded = encode_draw(&draw);
    let mut stream = WireReader::new(encoded.as_bytes());
    let mut record = stream.next_record().unwrap().unwrap();
    // SAFETY: data remains live and unchanged for all decoded values.
    let mut decoder = unsafe { DrawReader::new() };
    assert!(matches!(
        decoder.decode_draw(&mut record.payload),
        Err(WireError::InvalidValue)
    ));
    decoder.clear();
    // SAFETY: this record was built from retained data and contains no other address.
    let mut stream = unsafe { WireReader::new_trusted(encoded.as_bytes()) };
    let mut record = stream.next_record().unwrap().unwrap();
    let restored = decoder.decode_draw(&mut record.payload).unwrap();
    assert_eq!(encode_draw(&restored).as_bytes(), encoded.as_bytes());
}

#[test]
fn partial_deltas_preserve_structural_referents_and_clear_only_changed_bytes() {
    let source = VsSource::Programmable {
        vs_id: ProgramId::from_tokens(&[0xfffe_0300, 0xffff]),
        max_const_used: 19,
        uses_rel_const: true,
        provided_input_mask: 0x99,
        uses_int_const: false,
        uses_bool_const: true,
        clip_plane_count: 3,
        sampler_kinds: crate::dxso::VsSamplerKinds::default(),
    };
    let uniform = captured(&[1, 2, 3, 4]);
    let mut initial = SnapshotDelta {
        vs: Some(&source),
        ..SnapshotDelta::default()
    };
    initial.bytes[1] = Some(Some(uniform));
    initial.bytes[2] = Some(Some(uniform));
    let mut clearing = SnapshotDelta::default();
    clearing.bytes[1] = Some(None);
    let mut writer = DrawWriter::new();
    let mut encoded = FrameSlab::new();
    encoded
        .push_record(1, |out| writer.encode_snapshot_delta(&initial, out))
        .unwrap();
    let first_length = encoded.as_bytes().len();
    encoded
        .push_record(1, |out| writer.encode_snapshot_delta(&clearing, out))
        .unwrap();
    assert_eq!(encoded.as_bytes().len() - first_length, 11);
    let mut scratch = ScratchArena::new();
    // SAFETY: the static uniform range and native scratch remain live through every token use.
    let mut decoder = unsafe { DrawReader::new() };
    // SAFETY: the only encoded addresses point to the immutable static uniform above.
    let mut stream = unsafe { WireReader::new_trusted(encoded.as_bytes()) };
    let mut record = stream.next_record().unwrap().unwrap();
    let first = decoder
        .decode_snapshot_delta(&mut record.payload, &mut scratch)
        .unwrap();
    let mut record = stream.next_record().unwrap().unwrap();
    let second = decoder
        .decode_snapshot_delta(&mut record.payload, &mut scratch)
        .unwrap();
    // SAFETY: scratch retains the two initialized snapshots for both borrows.
    let first = unsafe { &*first.as_ptr() };
    // SAFETY: scratch retains this initialized snapshot through all assertions.
    let second = unsafe { &*second.as_ptr() };
    assert!(std::ptr::eq(
        first.vs.unwrap().as_ref(),
        second.vs.unwrap().as_ref()
    ));
    assert_eq!(first.ps_constants.unwrap().as_raw(), uniform.as_raw());
    assert!(second.ps_constants.is_none());
    assert_eq!(second.alpha_ref_bytes.unwrap().as_raw(), uniform.as_raw());
}

#[test]
fn invalid_delta_mask_poisoning_prevents_partial_replay() {
    let mut scratch = ScratchArena::new();
    // SAFETY: no record in this test contains a borrowed address.
    let mut decoder = unsafe { DrawReader::new() };
    let mut invalid = WireReader::new(&[0, 0, 2, 0]);
    assert!(matches!(
        decoder.decode_snapshot_delta(&mut invalid, &mut scratch),
        Err(WireError::InvalidValue)
    ));
    let mut empty = WireReader::new(&[]);
    assert!(matches!(
        decoder.decode_snapshot_delta(&mut empty, &mut scratch),
        Err(WireError::InvalidValue)
    ));
    decoder.clear();
    assert!(matches!(
        decoder.decode_snapshot_delta(&mut empty, &mut scratch),
        Err(WireError::Truncated)
    ));
}

fn full_delta(snapshot: &CurrentSnapshot) -> SnapshotDelta<'_> {
    let stages = snapshot.stage_bindings.as_ref().map(|value| {
        // SAFETY: fixture bindings occupy exactly the initialized mask-sized scratch prefix.
        let bindings = unsafe {
            std::slice::from_raw_parts(
                std::ptr::from_ref(value.iter().next().unwrap().1),
                value.mask().count_ones() as usize,
            )
        };
        (value.mask(), bindings)
    });
    SnapshotDelta {
        render_state: snapshot
            .render_state
            .as_ref()
            .map(crate::draw_data::RenderStatePtr::as_ref),
        stages,
        attrs: snapshot.attrs.as_ref().map(|value| SnapshotAttributes {
            attrs: value.as_slice(),
            extents: &value.extents,
            used_streams: value.used_streams,
            vdecl_hash: value.vdecl_hash,
        }),
        vs: snapshot.vs.as_ref().map(VsSourcePtr::as_ref),
        ps: snapshot
            .ps
            .as_ref()
            .map(crate::draw_data::PsSourcePtr::as_ref),
        variant: snapshot.variant,
        bytes: [
            snapshot.vs_constants,
            snapshot.ps_constants,
            snapshot.alpha_ref_bytes,
            snapshot.fog_color_bytes,
            snapshot.bump_env_bytes,
            snapshot.vs_int_const_bytes,
            snapshot.vs_bool_const_bytes,
            snapshot.ps_int_const_bytes,
            snapshot.ps_bool_const_bytes,
            snapshot.vs_draw_bytes,
        ]
        .map(Some),
        depth_stencil: Some(snapshot.depth_stencil),
    }
}

#[test]
fn every_truncated_draw_prefix_fails() {
    let draw = DrawOp {
        metal_prim: PrimitiveType::Triangle,
        vertex_source: VertexSource::Up {
            bytes: captured(&[0; 12]),
            size: 12,
            stride: 4,
        },
        index_source: IndexSource::None {
            start_vertex: 0,
            vertex_count: 3,
        },
    };
    let encoded = encode_draw(&draw);
    let payload = &encoded.as_bytes()[6..];
    for count in 0..payload.len() {
        assert_draw_rejected(&payload[..count]);
    }
    let mut invalid = payload.to_vec();
    invalid[..4].copy_from_slice(&u32::MAX.to_le_bytes());
    assert_draw_rejected(&invalid);
    invalid.copy_from_slice(payload);
    invalid[4] = u8::MAX;
    assert_draw_rejected(&invalid);
}

#[test]
fn malformed_draw_extents_and_stream_indices_fail_before_replay() {
    let cases = [
        DrawOp {
            metal_prim: PrimitiveType::Triangle,
            vertex_source: VertexSource::Up {
                bytes: captured(&[0; 4]),
                size: 8,
                stride: 4,
            },
            index_source: IndexSource::None {
                start_vertex: 0,
                vertex_count: 1,
            },
        },
        DrawOp {
            metal_prim: PrimitiveType::Triangle,
            vertex_source: VertexSource::Up {
                bytes: captured(&[0; 4]),
                size: 4,
                stride: 4,
            },
            index_source: IndexSource::Up {
                bytes: captured(&[0; 2]),
                index_count: 2,
                index_type: IndexType::UInt16,
            },
        },
        DrawOp {
            metal_prim: PrimitiveType::Triangle,
            vertex_source: VertexSource::Bound {
                first: StreamBinding {
                    stream: 16,
                    buffer_id: BufferId::new_unique(),
                    backing_ptr: 0,
                    backing_len: 0,
                    backing_generation: 1,
                    offset: 0,
                    stride: 4,
                    freq: 1,
                },
                extra: Box::new([]),
                stream0_freq: 1,
            },
            index_source: IndexSource::None {
                start_vertex: 0,
                vertex_count: 1,
            },
        },
        DrawOp {
            metal_prim: PrimitiveType::Triangle,
            vertex_source: VertexSource::Up {
                bytes: captured(&[0; 4]),
                size: 4,
                stride: 4,
            },
            index_source: IndexSource::Generated {
                data: ScratchSlice::EMPTY,
                index_count: 3,
                index_type: IndexType::UInt16,
                min_vertex: 0,
                max_vertex: 2,
            },
        },
    ];
    for draw in cases {
        let bytes = encode_draw(&draw);
        let mut records = WireReader::new(bytes.as_bytes());
        let mut record = records.next_record().unwrap().unwrap();
        // SAFETY: these records contain no nonempty borrowed ranges.
        let mut decoder = unsafe { DrawReader::new() };
        assert!(matches!(
            decoder.decode_draw(&mut record.payload),
            Err(WireError::InvalidValue)
        ));
    }
}

#[test]
fn complete_snapshot_reconstructs_native_structures_and_borrows_only_bytes() {
    use mtld3d_shared::{VertexAttrDesc, mtl::VertexFormat};

    use crate::{
        depth_stencil_state::DepthStencilSnapshot,
        draw_data::{
            AttrSnapshot, DepthScissorFlags, DepthStencilFlags, PsSource, PsSourcePtr,
            RenderStatePtr, RenderStateSnapshot, StageBinding, bump_packed_stage_bindings,
        },
        dxso::{FfPsKey, FfStage, FfVsFlags, FfVsKey, VariantKey},
        ids::TextureId,
        pipeline_state::PipelineRsBits,
    };

    let mut source = ScratchArena::new();
    let bytes = [1u8, 2, 3, 4];
    // SAFETY: bytes remains immutable and live through both encodings and every decoded token.
    let uniform = unsafe { ScratchSlice::from_raw_parts(std::ptr::NonNull::from(&bytes[0]), 4) };
    let render = store(
        &mut source,
        RenderStateSnapshot {
            pipeline_rs: PipelineRsBits::default(),
            depth_scissor: DepthScissorFlags::DEPTH_ENABLE,
            depth_stencil_state: DepthStencilSnapshot::inert(),
            cull_mode: 2,
            fill_mode: 3,
            scissor_rect: [1, 2, 20, 30],
            blend_factor: 0x1234_5678,
            depth_bias: 17,
            slope_scale_depth_bias: 18,
            stencil_ref: 19,
            sample_mask: 3,
        },
    );
    let vertex = store(
        &mut source,
        VsSource::FixedFunction {
            key: FfVsKey {
                flags: FfVsFlags::HAS_NORMAL,
                input_tex_coord_count: 2,
                tex_coord_count: 3,
                light_active_mask: 3,
                light_directional_mask: 1,
                light_spot_mask: 2,
                diffuse_source: 1,
                ambient_source: 2,
                specular_source: 0,
                emissive_source: 0,
                fog_mode: 3,
                tci_modes: [0; 8],
                tci_coord_indices: [1; 8],
                tex_coord_dims: [2; 8],
                tt_flags: [0; 8],
                vertex_blend_count: 2,
                declared_weights_count: 1,
                clip_plane_count: 0,
            },
            max_row_count: 30,
        },
    );
    let pixel = store(
        &mut source,
        PsSource::FixedFunction {
            key: FfPsKey {
                stages: [FfStage::default(); 8],
                specular_add: true,
                tt_projected_mask: 3,
            },
            sampled_stage_mask: 1,
            constant_rows: 2,
        },
    );
    let packed: [StageBinding; 16] = std::array::from_fn(|_| StageBinding {
        texture_id: TextureId::new_unique(),
        sampler_state: [7; 14],
    });
    // SAFETY: all 16 initialized bindings match the full mask and source survives every token.
    let stages = unsafe { bump_packed_stage_bindings(&mut source, u16::MAX, &packed) };
    let attributes: [VertexAttrDesc; 16] = std::array::from_fn(|index| VertexAttrDesc {
        attr_index: u32::try_from(index).unwrap(),
        buffer_index: u32::try_from(index).unwrap(),
        format: VertexFormat::Float4,
        offset: 0,
    });
    let (pointer, length) = source.alloc_slice(&attributes);
    // SAFETY: the initialized descriptor array remains in source until all tokens are forgotten.
    let attrs = unsafe {
        AttrSnapshot::new(
            std::ptr::NonNull::new(pointer).unwrap(),
            length,
            [16; 16],
            1,
            0xaabb,
        )
    };
    // SAFETY: source retains this render state throughout the test.
    let render = unsafe { RenderStatePtr::new(render) };
    // SAFETY: source retains this vertex source throughout the test.
    let vertex = unsafe { VsSourcePtr::new(vertex) };
    // SAFETY: source retains this pixel source throughout the test.
    let pixel = unsafe { PsSourcePtr::new(pixel) };
    let snapshot = CurrentSnapshot {
        render_state: Some(render),
        stage_bindings: Some(stages),
        attrs: Some(attrs),
        vs: Some(vertex),
        ps: Some(pixel),
        variant: Some(VariantKey::default()),
        vs_constants: Some(uniform),
        ps_constants: Some(uniform),
        alpha_ref_bytes: Some(uniform),
        fog_color_bytes: Some(uniform),
        bump_env_bytes: Some(uniform),
        vs_int_const_bytes: Some(uniform),
        vs_bool_const_bytes: Some(uniform),
        ps_int_const_bytes: Some(uniform),
        ps_bool_const_bytes: Some(uniform),
        vs_draw_bytes: Some(uniform),
        depth_stencil: DepthStencilFlags::HAS_DEPTH,
    };
    let mut first = FrameSlab::new();
    let mut writer = DrawWriter::new();
    first
        .push_record(1, |output| {
            writer.encode_snapshot_delta(&full_delta(&snapshot), output)
        })
        .unwrap();
    let mut direct = ScratchArena::new();
    let (_, direct_length, _) = direct
        .write_record(1, super::SNAPSHOT_DELTA_MAX_BYTES, |output| {
            writer.encode_snapshot_delta(&full_delta(&snapshot), output)
        })
        .unwrap();
    assert_eq!(direct_length, first.as_bytes().len());
    assert!(direct_length <= super::SNAPSHOT_DELTA_MAX_BYTES);
    let payload = &first.as_bytes()[6..];
    for length in 0..payload.len() {
        // SAFETY: this read-only validator borrows the same retained fixture bytes.
        let mut checked = unsafe { WireReader::new_trusted(&payload[..length]) };
        assert!(super::validate_wire_snapshot(&mut checked).is_err());
        let mut truncated_scratch = ScratchArena::new();
        // SAFETY: any complete ranges in this prefix name the live immutable bytes above.
        let mut truncated_decoder = unsafe { DrawReader::new() };
        // SAFETY: the encoded borrowed uniform range stays initialized through this decode.
        let mut truncated = unsafe { WireReader::new_trusted(&payload[..length]) };
        assert!(
            truncated_decoder
                .decode_snapshot_delta(&mut truncated, &mut truncated_scratch)
                .is_err()
        );
    }
    // SAFETY: the fixture retains every initialized byte span for this read-only pass.
    let mut checked = unsafe { WireReader::new_trusted(payload) };
    super::validate_wire_snapshot(&mut checked).unwrap();
    assert!(checked.is_empty());
    let mut native = ScratchArena::new();
    // SAFETY: the only wire addresses name bytes above, alive through all decoded-token uses.
    let mut stream = unsafe { WireReader::new_trusted(first.as_bytes()) };
    let mut record = stream.next_record().unwrap().unwrap();
    // SAFETY: native and bytes remain allocated and immutable throughout the returned token's life.
    let mut decoder = unsafe { DrawReader::new() };
    let restored = decoder
        .decode_snapshot_delta(&mut record.payload, &mut native)
        .unwrap();
    // SAFETY: native retains the decoded snapshot through the second encoding.
    let restored = unsafe { &*restored.as_ptr() };
    assert_ne!(
        std::ptr::from_ref(restored.vs.unwrap().as_ref()),
        std::ptr::from_ref(snapshot.vs.unwrap().as_ref())
    );
    assert!(record.payload.is_empty());
    let mut second = FrameSlab::new();
    writer.clear();
    second
        .push_record(1, |output| {
            writer.encode_snapshot_delta(&full_delta(restored), output)
        })
        .unwrap();
    assert_eq!(first.as_bytes(), second.as_bytes());
}

fn captured(bytes: &'static [u8]) -> ScratchSlice {
    if bytes.is_empty() {
        return ScratchSlice::EMPTY;
    }
    // SAFETY: static arrays remain initialized and immutable for the entire test.
    unsafe {
        ScratchSlice::from_raw_parts(
            std::ptr::NonNull::from(&bytes[0]),
            u32::try_from(bytes.len()).unwrap(),
        )
    }
}

#[test]
fn up_payloads_remain_borrowed_and_do_not_expand_wire_records() {
    static SMALL: [u8; 4] = [7; 4];
    static LARGE: [u8; 4800] = [9; 4800];
    let make = |bytes: &'static [u8]| DrawOp {
        metal_prim: PrimitiveType::Triangle,
        vertex_source: VertexSource::Up {
            bytes: captured(bytes),
            size: u32::try_from(bytes.len()).unwrap(),
            stride: 16,
        },
        index_source: IndexSource::Up {
            bytes: captured(&[0, 0, 1, 0, 2, 0]),
            index_count: 3,
            index_type: IndexType::UInt16,
        },
    };
    let small = encode_draw(&make(&SMALL));
    let large_draw = make(&LARGE);
    let large = encode_draw(&large_draw);
    assert_eq!(small.as_bytes().len(), large.as_bytes().len());
    let IndexSource::Up { bytes: indices, .. } = &large_draw.index_source else {
        unreachable!()
    };
    let (index_ptr, index_len) = indices.as_raw();
    let ranges = [
        (LARGE.as_ptr() as u64, LARGE.len() as u64),
        (index_ptr, u64::from(index_len)),
    ];
    // SAFETY: both immutable static captures remain live through all decoded reads.
    let mut wire = unsafe { WireReader::new_trusted_with_ranges(large.as_bytes(), &ranges) };
    let mut record = wire.next_record().unwrap().unwrap();
    // SAFETY: static capture and all decoder-owned structures outlive the draw.
    let mut decoder = unsafe { DrawReader::new() };
    let restored = decoder.decode_draw(&mut record.payload).unwrap();
    let VertexSource::Up { bytes, .. } = restored.vertex_source else {
        unreachable!()
    };
    assert_eq!(bytes.as_raw().0, LARGE.as_ptr() as u64);
    assert_eq!(bytes.as_slice(), LARGE);
    let IndexSource::Up { bytes, .. } = restored.index_source else {
        unreachable!()
    };
    assert_eq!(bytes.as_raw().0, index_ptr);

    let shortened = [(LARGE.as_ptr() as u64, LARGE.len() as u64 - 1)];
    // SAFETY: range describes only retained static bytes; decoding must reject its escape.
    let mut wire = unsafe { WireReader::new_trusted_with_ranges(large.as_bytes(), &shortened) };
    let mut record = wire.next_record().unwrap().unwrap();
    assert!(matches!(
        decoder.decode_draw(&mut record.payload),
        Err(WireError::InvalidValue)
    ));
}

#[test]
fn released_cpu_backing_keeps_bound_vertex_and_index_buffer_descriptors() {
    let draw = DrawOp {
        metal_prim: PrimitiveType::Triangle,
        vertex_source: VertexSource::Bound {
            first: StreamBinding {
                stream: 0,
                buffer_id: BufferId::new_unique(),
                backing_ptr: 0,
                backing_len: crate::page_box::PAGE_SIZE,
                backing_generation: 0,
                offset: 0,
                stride: 24,
                freq: 1,
            },
            extra: Box::new([]),
            stream0_freq: 1,
        },
        index_source: IndexSource::Bound {
            buffer_id: BufferId::new_unique(),
            backing_ptr: 0,
            backing_len: crate::page_box::PAGE_SIZE,
            backing_generation: 0,
            offset: 0,
            index_count: 3,
            index_type: IndexType::UInt16,
            base_vertex: 0,
        },
    };
    let encoded = encode_draw(&draw);
    let mut stream = WireReader::new(encoded.as_bytes());
    let mut record = stream.next_record().unwrap().unwrap();
    // SAFETY: released backing descriptors contain no borrowed CPU address or payload.
    let mut decoder = unsafe { DrawReader::new() };
    let restored = decoder.decode_draw(&mut record.payload).unwrap();
    assert!(record.payload.is_empty());
    assert_eq!(encode_draw(&restored).as_bytes(), encoded.as_bytes());
}

fn assert_draw_rejected(payload: &[u8]) {
    // SAFETY: callers retain every unmodified fixture byte span; malformed cases
    // alter scalar discriminants or truncate fields, never forge byte addresses.
    let mut checked = unsafe { WireReader::new_trusted(payload) };
    assert!(super::validate_wire_draw(&mut checked, |_, _| Ok(())).is_err());
    // SAFETY: the same fixture storage remains live during reconstruction.
    let mut wire = unsafe { WireReader::new_trusted(payload) };
    // SAFETY: decoded values cannot outlive the retained fixture storage.
    let mut decoder = unsafe { DrawReader::new() };
    assert!(decoder.decode_draw(&mut wire).is_err());
}

#[test]
fn maximum_bound_stream_draw_validation_matches_reconstruction() {
    let stream = |index| StreamBinding {
        stream: index,
        buffer_id: BufferId::new_unique(),
        backing_ptr: 0,
        backing_len: crate::page_box::PAGE_SIZE,
        backing_generation: 0,
        offset: 0,
        stride: 16,
        freq: 1,
    };
    let draw = DrawOp {
        metal_prim: PrimitiveType::Triangle,
        vertex_source: VertexSource::Bound {
            first: stream(0),
            extra: (1..16).map(stream).collect(),
            stream0_freq: 1,
        },
        index_source: IndexSource::None {
            start_vertex: 0,
            vertex_count: 3,
        },
    };
    let encoded = encode_draw(&draw);
    let payload = &encoded.as_bytes()[6..];
    let mut checked = WireReader::new(payload);
    super::validate_wire_draw(&mut checked, |address, length| {
        assert_eq!((address, length), (0, crate::page_box::PAGE_SIZE as u64));
        Ok(())
    })
    .unwrap();
    assert!(checked.is_empty());
    for length in 0..payload.len() {
        assert_draw_rejected(&payload[..length]);
    }
}
