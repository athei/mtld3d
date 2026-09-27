use mtld3d_shared::encoder_wire::{WireError, WireReader};

use super::{DrawReader, DrawWriter, SnapshotAttributes, SnapshotDelta};
use crate::{
    draw_data::{CurrentSnapshot, ScratchSlice, VsSource, VsSourcePtr},
    ids::ProgramId,
    scratch::ScratchArena,
};

const fn vs_address(source: crate::draw_data::VsSourceView<'_>) -> *const u8 {
    match source {
        crate::draw_data::VsSourceView::Programmable(value) => std::ptr::from_ref(value).cast(),
        crate::draw_data::VsSourceView::FixedFunction(value) => std::ptr::from_ref(value).cast(),
    }
}

fn store<T>(scratch: &mut ScratchArena, value: T) -> std::ptr::NonNull<T> {
    const {
        assert!(!std::mem::needs_drop::<T>());
    }
    let ptr = scratch.alloc_uninit::<T>();
    // SAFETY: the arena reserved aligned, exclusive space for one T.
    unsafe { ptr.write(value) };
    std::ptr::NonNull::new(ptr).expect("arena allocation is non-null")
}

fn decode_snapshot(
    decoder: &mut DrawReader,
    reader: &mut WireReader<'_>,
    scratch: &mut ScratchArena,
) -> Result<crate::draw_data::CurrentSnapshotPtr, WireError> {
    let length = u32::try_from(reader.remaining_len()).map_err(|_| WireError::TooLarge)?;
    let payload = reader.bytes(length)?;
    // SAFETY: each fixture retains its typed canonical records and referenced byte ranges.
    unsafe { decoder.decode_snapshot(payload, scratch) }
}

fn encode_snapshot<'a>(arena: &'a mut ScratchArena, delta: &SnapshotDelta<'_>) -> &'a [u8] {
    let mut writer = DrawWriter::new();
    let allocation = arena
        .write_command(
            u16::from(mtld3d_shared::encoder_protocol::EncoderOpcode::SetSnapshot),
            0,
            super::SNAPSHOT_DELTA_MAX_BYTES,
            |destination| writer.capture_snapshot(delta, destination),
        )
        .unwrap();
    let pointer = usize::try_from(allocation.address).unwrap() as *const u8;
    // SAFETY: the arena retains this initialized immutable payload for the returned borrow.
    unsafe { std::slice::from_raw_parts(pointer.wrapping_add(16), allocation.record_bytes - 16) }
}

#[test]
fn partial_deltas_preserve_structural_referents_and_clear_only_changed_bytes() {
    let source = VsSource::Programmable(crate::draw_data::ProgrammableVsSource {
        vs_id: ProgramId::from_tokens(&[0xfffe_0300, 0xffff]),
        max_const_used: 19,

        provided_input_mask: 0x99,

        clip_plane_count: 3,
        sampler_kinds: crate::dxso::VsSamplerKinds::default(),

        flags: crate::draw_data::ShaderSourceFlags::RELATIVE
            | crate::draw_data::ShaderSourceFlags::BOOLEAN,
    });
    let uniform = captured(&[1, 2, 3, 4]);
    let mut initial = SnapshotDelta {
        vs: Some(source.as_view()),
        ..SnapshotDelta::default()
    };
    initial.bytes[1] = Some(Some(uniform));
    initial.bytes[2] = Some(Some(uniform));
    let mut clearing = SnapshotDelta::default();
    clearing.bytes[1] = Some(None);
    let mut initial_arena = ScratchArena::new();
    let initial_bytes = encode_snapshot(&mut initial_arena, &initial);
    let mut clearing_arena = ScratchArena::new();
    let clearing_bytes = encode_snapshot(&mut clearing_arena, &clearing);
    assert_eq!(clearing_bytes.len(), 24);
    let mut scratch = ScratchArena::new();
    // SAFETY: both command arenas, uniforms and native scratch outlive decoded tokens.
    let mut decoder = unsafe { DrawReader::new() };
    // SAFETY: fixture command storage and referenced uniform bytes stay immutable and live.
    let mut first_reader = unsafe { WireReader::new_trusted(initial_bytes) };
    let first = decode_snapshot(&mut decoder, &mut first_reader, &mut scratch).unwrap();
    // SAFETY: the clearing command remains live through all decoded token uses.
    let mut second_reader = unsafe { WireReader::new_trusted(clearing_bytes) };
    let second = decode_snapshot(&mut decoder, &mut second_reader, &mut scratch).unwrap();
    // SAFETY: scratch retains the two initialized snapshots for both borrows.
    let first = unsafe { &*first.as_ptr() };
    // SAFETY: scratch retains this initialized snapshot through all assertions.
    let second = unsafe { &*second.as_ptr() };
    assert!(std::ptr::eq(
        vs_address(first.vs.unwrap().as_ref()),
        vs_address(second.vs.unwrap().as_ref())
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
    // SAFETY: this invalid scalar-only fixture borrows no external data.
    let mut invalid = unsafe { WireReader::new_trusted(&[0, 0, 2, 0, 0, 0, 0, 0]) };
    assert!(matches!(
        decode_snapshot(&mut decoder, &mut invalid, &mut scratch),
        Err(WireError::InvalidValue)
    ));
    // SAFETY: empty bytes contain no referents.
    let mut empty = unsafe { WireReader::new_trusted(&[]) };
    assert!(matches!(
        decode_snapshot(&mut decoder, &mut empty, &mut scratch),
        Err(WireError::InvalidValue)
    ));
    decoder.clear();
    assert!(matches!(
        decode_snapshot(&mut decoder, &mut empty, &mut scratch),
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
            extents: value.extents(),
            used_streams: value.used_streams(),
            vdecl_hash: value.vdecl_hash(),
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
fn complete_snapshot_borrows_canonical_leaves_and_reconstructs_only_native_roots() {
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
            reserved: 0,
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
        VsSource::FixedFunction(crate::draw_data::FixedVsSource {
            key: FfVsKey {
                reserved: 0,
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

            reserved: [0; 6],
        }),
    );
    let pixel = store(
        &mut source,
        PsSource::FixedFunction(crate::draw_data::FixedPsSource {
            key: FfPsKey {
                stages: [FfStage::default(); 8],
                specular_add: true,
                tt_projected_mask: 3,
            },
            sampled_stage_mask: 1,
            constant_rows: 2,

            reserved: [0; 3],
        }),
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
    let header = store(
        &mut source,
        crate::draw_data::DeclarationHeader {
            vdecl_hash: 0xaabb,
            extents: [16; 16],
            count: length,
            used_streams: 1,
            reserved: 0,
        },
    );
    // SAFETY: source retains the initialized descriptor array and header through every token use.
    let attrs = unsafe { AttrSnapshot::new(std::ptr::NonNull::new(pointer).unwrap(), header) };
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
    let mut first = ScratchArena::new();
    let payload = encode_snapshot(&mut first, &full_delta(&snapshot));
    assert!(payload.len() <= super::SNAPSHOT_DELTA_MAX_BYTES);
    for length in 0..payload.len() {
        let mut truncated_scratch = ScratchArena::new();
        // SAFETY: any complete ranges in this prefix name the live immutable bytes above.
        let mut truncated_decoder = unsafe { DrawReader::new() };
        // SAFETY: the encoded borrowed uniform range stays initialized through this decode.
        let mut truncated = unsafe { WireReader::new_trusted(&payload[..length]) };
        assert!(
            decode_snapshot(
                &mut truncated_decoder,
                &mut truncated,
                &mut truncated_scratch
            )
            .is_err()
        );
    }
    let mut native = ScratchArena::new();
    // SAFETY: the only wire addresses name bytes above, alive through all decoded-token uses.
    let mut record = unsafe { WireReader::new_trusted(payload) };
    // SAFETY: native and bytes remain allocated and immutable throughout the returned token's life.
    let mut decoder = unsafe { DrawReader::new() };
    let restored = decode_snapshot(&mut decoder, &mut record, &mut native).unwrap();
    // SAFETY: native retains the decoded snapshot through the second encoding.
    let restored = unsafe { &*restored.as_ptr() };
    assert_ne!(
        vs_address(restored.vs.unwrap().as_ref()),
        vs_address(snapshot.vs.unwrap().as_ref())
    );
    assert!(record.is_empty());
    let leaf_address = std::ptr::from_ref(restored.render_state.unwrap().as_ref()) as usize;
    assert!(
        (payload.as_ptr() as usize..payload.as_ptr() as usize + payload.len())
            .contains(&leaf_address)
    );
    let stage_address = std::ptr::from_ref(
        restored
            .stage_bindings
            .as_ref()
            .unwrap()
            .iter()
            .next()
            .unwrap()
            .1,
    ) as usize;
    let attr_address = restored.attrs.as_ref().unwrap().as_slice().as_ptr() as usize;
    let range = payload.as_ptr() as usize..payload.as_ptr() as usize + payload.len();
    assert!(range.contains(&stage_address));
    assert!(range.contains(&attr_address));
    assert!(range.contains(&(vs_address(restored.vs.unwrap().as_ref()) as usize)));
    assert_eq!(&payload[4..8], &[0; 4]);
    assert_eq!(payload[8 + 35], 0);
    let mut second = ScratchArena::new();
    let second_payload = encode_snapshot(&mut second, &full_delta(restored));
    assert_eq!(payload, second_payload);
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
fn packed_stage_and_attribute_arrays_decode_empty_and_publish_only_when_complete() {
    use crate::{draw_data::StageBinding, ids::TextureId};

    let stages = [
        StageBinding {
            texture_id: TextureId::new_unique(),
            sampler_state: [7; mtld3d_types::SAMPLER_STATE_COUNT],
        },
        StageBinding {
            texture_id: TextureId::new_unique(),
            sampler_state: [9; mtld3d_types::SAMPLER_STATE_COUNT],
        },
    ];
    let delta = SnapshotDelta {
        stages: Some((3, &stages)),
        ..SnapshotDelta::default()
    };
    let mut slab = ScratchArena::new();
    let payload = encode_snapshot(&mut slab, &delta);
    let mut scratch = ScratchArena::new();
    // SAFETY: this fixture contains only scalar stage fields, no borrowed byte ranges.
    let mut decoder = unsafe { DrawReader::new() };
    assert!(
        decode_snapshot(
            &mut decoder,
            // SAFETY: this truncated command still borrows the retained fixture arena.
            &mut unsafe { WireReader::new_trusted(&payload[..payload.len() - 1]) },
            &mut scratch,
        )
        .is_err()
    );
    assert!(decoder.current.stage_bindings.is_none());
    decoder.clear();
    // SAFETY: slab retains this initialized immutable canonical payload through every token use.
    let first = unsafe { decoder.decode_snapshot(payload, &mut scratch) }.unwrap();

    let empty = SnapshotDelta {
        stages: Some((0, &[])),
        attrs: Some(SnapshotAttributes {
            attrs: &[],
            extents: &[0; 16],
            used_streams: 0,
            vdecl_hash: 0,
        }),
        ..SnapshotDelta::default()
    };
    let mut empty_slab = ScratchArena::new();
    let empty_payload = encode_snapshot(&mut empty_slab, &empty);
    // SAFETY: empty_slab retains the initialized command through all decoded token uses.
    let mut empty_reader = unsafe { WireReader::new_trusted(empty_payload) };
    let second = decode_snapshot(&mut decoder, &mut empty_reader, &mut scratch).unwrap();
    // SAFETY: scratch remains live and unchanged; subsequent allocations cannot move its chunks.
    let first = unsafe { &*first.as_ptr() };
    // SAFETY: same arena lifetime covers the second snapshot and its empty arrays.
    let second = unsafe { &*second.as_ptr() };
    let first_stages = first.stage_bindings.as_ref().unwrap();
    assert_eq!(first_stages.iter().count(), 2);
    for ((_, actual), expected) in first_stages.iter().zip(&stages) {
        assert_eq!(actual.texture_id, expected.texture_id);
        assert_eq!(actual.sampler_state, expected.sampler_state);
    }
    assert_eq!(second.stage_bindings.as_ref().unwrap().iter().count(), 0);
    assert!(second.attrs.as_ref().unwrap().as_slice().is_empty());
}

#[test]
#[cfg(debug_assertions)]
fn canonical_pixel_source_rejects_invalid_bool_before_borrowing() {
    #[repr(align(16))]
    struct AlignedPayload([u8; 80]);
    let source = crate::draw_data::PsSource::FixedFunction(crate::draw_data::FixedPsSource {
        key: crate::dxso::FfPsKey {
            stages: [crate::dxso::FfStage::default(); 8],
            specular_add: false,
            tt_projected_mask: 0,
        },
        sampled_stage_mask: 0,
        constant_rows: 0,
        reserved: [0; 3],
    });
    let mut arena = ScratchArena::new();
    let payload = encode_snapshot(
        &mut arena,
        &SnapshotDelta {
            ps: Some(source.as_view()),
            ..SnapshotDelta::default()
        },
    );
    assert_eq!(payload.len(), 80);
    for (offset, invalid) in [(8, 2), (16 + 56, 2), (16 + 6, 128), (16 + 61, 1)] {
        let mut copied = AlignedPayload([0; 80]);
        copied.0.copy_from_slice(payload);
        copied.0[offset] = invalid;
        let mut native = ScratchArena::new();
        // SAFETY: the aligned fixture remains initialized and immutable through decoding.
        let mut reader = unsafe { WireReader::new_trusted(&copied.0) };
        // SAFETY: both fixture and native arena remain live through all decoded token uses.
        let mut decoder = unsafe { DrawReader::new() };
        assert!(decode_snapshot(&mut decoder, &mut reader, &mut native).is_err());
    }
}

#[test]
fn canonical_variant_hash_preserves_all_original_fields_in_order() {
    use std::hash::{Hash, Hasher};

    use crate::dxso::{VariantFlags, VariantKey};

    // The pre-canonical struct declaration, retained to pin its derived hash sequence.
    #[derive(Hash)]
    struct OriginalVariant {
        alpha_func: u8,
        fog_mode: u8,
        fog_table_mode: u8,
        depth_sampler_mask: u16,
        depth_fetch_mask: u16,
        fetch4_mask: u16,
        fetch4_alpha_mask: u16,
        raw_depth_red_mask: u16,
        volume_sampler_mask: u16,
        cube_sampler_mask: u16,
        tt_projected_mask: u8,
        color_out_mask: u8,
        sample_mask: u8,
        flags: VariantFlags,
    }
    struct HashBytes(Vec<u8>);
    impl Hasher for HashBytes {
        fn finish(&self) -> u64 {
            0
        }
        fn write(&mut self, bytes: &[u8]) {
            self.0.extend_from_slice(bytes);
        }
    }
    let original = OriginalVariant {
        alpha_func: 1,
        fog_mode: 2,
        fog_table_mode: 3,
        depth_sampler_mask: 4,
        depth_fetch_mask: 5,
        fetch4_mask: 6,
        fetch4_alpha_mask: 7,
        raw_depth_red_mask: 8,
        volume_sampler_mask: 9,
        cube_sampler_mask: 10,
        tt_projected_mask: 11,
        color_out_mask: 12,
        sample_mask: 13,
        flags: VariantFlags::FLAT_SHADE,
    };
    let current = VariantKey {
        reserved: 0,
        alpha_func: original.alpha_func,
        fog_mode: original.fog_mode,
        fog_table_mode: original.fog_table_mode,
        depth_sampler_mask: original.depth_sampler_mask,
        depth_fetch_mask: original.depth_fetch_mask,
        fetch4_mask: original.fetch4_mask,
        fetch4_alpha_mask: original.fetch4_alpha_mask,
        raw_depth_red_mask: original.raw_depth_red_mask,
        volume_sampler_mask: original.volume_sampler_mask,
        cube_sampler_mask: original.cube_sampler_mask,
        tt_projected_mask: original.tt_projected_mask,
        color_out_mask: original.color_out_mask,
        sample_mask: original.sample_mask,
        flags: original.flags,
    };
    let mut expected = HashBytes(Vec::new());
    original.hash(&mut expected);
    let mut actual = HashBytes(Vec::new());
    current.hash(&mut actual);
    assert_eq!(actual.0, expected.0);
}

#[test]
fn programmable_sources_borrow_canonical_fields_without_reconstruction() {
    use crate::draw_data::{
        ProgrammablePsSource, ProgrammableVsSource, PsSource, PsSourceView, ShaderSourceFlags,
        VsSourceView,
    };
    let vertex = VsSource::Programmable(ProgrammableVsSource {
        vs_id: ProgramId::from_tokens(&[1, 2]),
        max_const_used: 219,
        provided_input_mask: 0x1357,
        flags: ShaderSourceFlags::RELATIVE
            | ShaderSourceFlags::INTEGER
            | ShaderSourceFlags::BOOLEAN,
        clip_plane_count: 6,
        sampler_kinds: crate::dxso::VsSamplerKinds {
            volume_mask: 2,
            cube_mask: 4,
        },
    });
    let pixel = PsSource::Programmable(ProgrammablePsSource {
        ps_id: ProgramId::from_tokens(&[3, 4]),
        max_const_used: 127,
        flags: ShaderSourceFlags::INTEGER
            | ShaderSourceFlags::BOOLEAN
            | ShaderSourceFlags::BUMP_ENV,
        color_out_mask: 11,
        reserved: [0; 4],
    });
    let mut arena = ScratchArena::new();
    let payload = encode_snapshot(
        &mut arena,
        &SnapshotDelta {
            vs: Some(vertex.as_view()),
            ps: Some(pixel.as_view()),
            ..SnapshotDelta::default()
        },
    );
    assert_eq!(payload.len(), 56);
    let mut native = ScratchArena::new();
    // SAFETY: source arena and canonical payload remain immutable through all token uses.
    let mut decoder = unsafe { DrawReader::new() };
    // SAFETY: encode_snapshot constructed these typed records in the retained arena.
    let snapshot = unsafe { decoder.decode_snapshot(payload, &mut native) }.unwrap();
    // SAFETY: native retains the initialized immutable root through these assertions.
    let snapshot = unsafe { &*snapshot.as_ptr() };
    let vs_token = snapshot.vs.unwrap();
    let ps_token = snapshot.ps.unwrap();
    let VsSourceView::Programmable(vs) = vs_token.as_ref() else {
        panic!("vertex source kind changed")
    };
    let PsSourceView::Programmable(ps) = ps_token.as_ref() else {
        panic!("pixel source kind changed")
    };
    assert_eq!(vs.vs_id, ProgramId::from_tokens(&[1, 2]));
    assert_eq!(
        (
            vs.max_const_used,
            vs.provided_input_mask,
            vs.clip_plane_count
        ),
        (219, 0x1357, 6)
    );
    assert!(vs.uses_rel_const() && vs.uses_int_const() && vs.uses_bool_const());
    assert_eq!(
        (vs.sampler_kinds.volume_mask, vs.sampler_kinds.cube_mask),
        (2, 4)
    );
    assert_eq!(ps.ps_id, ProgramId::from_tokens(&[3, 4]));
    assert_eq!((ps.max_const_used, ps.color_out_mask), (127, 11));
    assert!(ps.uses_bump_env() && ps.uses_int_const() && ps.uses_bool_const());
    assert_eq!(
        std::ptr::from_ref(vs).cast::<u8>(),
        payload.as_ptr().wrapping_add(16)
    );
    assert_eq!(
        std::ptr::from_ref(ps).cast::<u8>(),
        payload.as_ptr().wrapping_add(40)
    );
}
