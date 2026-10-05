//! Unit tests for the vertex-stream frequency word.
//!
//! Covers the `SetStreamSourceFreq` validation rules (stream range, `INSTANCEDATA` on
//! stream 0, both flags at once, a literal zero) and the derivations a draw makes from a
//! frequency word: the instance count taken from stream 0, the Metal step function and
//! rate, and the saturating byte range an instanced draw reads from a per-instance stream.
//! A flag with a zero count is the corner case: accepted, one instance, a `Constant` layout.
//! The attributes that end past a short stride: their remap onto bindings of their own,
//! the advanced offsets Metal accepts, and the bytes such a draw reads.

use super::*;

#[test]
fn validation_follows_the_runtime_rules() {
    assert_eq!(validate_stream_freq(0, 1), Ok(()));
    assert_eq!(validate_stream_freq(1, 2), Ok(()));
    assert_eq!(
        validate_stream_freq(MAX_STREAMS, 1),
        Err(StreamFreqError::StreamOutOfRange)
    );
    assert_eq!(
        validate_stream_freq(0, D3DSTREAMSOURCE_INSTANCEDATA | 1),
        Err(StreamFreqError::InstanceDataOnStreamZero)
    );
    assert_eq!(
        validate_stream_freq(
            1,
            D3DSTREAMSOURCE_INSTANCEDATA | D3DSTREAMSOURCE_INDEXEDDATA
        ),
        Err(StreamFreqError::BothFlags)
    );
    assert_eq!(validate_stream_freq(1, 0), Err(StreamFreqError::Zero));
    // A flag with a zero count is a non-zero word: accepted.
    assert_eq!(validate_stream_freq(1, D3DSTREAMSOURCE_INDEXEDDATA), Ok(()));
    assert_eq!(
        validate_stream_freq(1, D3DSTREAMSOURCE_INSTANCEDATA),
        Ok(())
    );
    assert_eq!(validate_stream_freq(0, D3DSTREAMSOURCE_INDEXEDDATA), Ok(()));
}

#[test]
fn instance_count_reads_stream_zero_only_when_something_is_instanced() {
    assert_eq!(instance_count(D3DSTREAMSOURCE_INDEXEDDATA | 4, true), 4);
    // Stream 0 with a plain count and no flag still supplies the count.
    assert_eq!(instance_count(3, true), 3);
    // No per-instance stream in the draw: one instance regardless.
    assert_eq!(instance_count(D3DSTREAMSOURCE_INDEXEDDATA | 4, false), 1);
    // `INDEXEDDATA | 0` is driver-defined; one instance here.
    assert_eq!(instance_count(D3DSTREAMSOURCE_INDEXEDDATA, true), 1);
    // Bits above the count mask are ignored.
    assert_eq!(instance_count(0x3F80_0002, true), 2);
}

#[test]
fn step_function_follows_the_flags() {
    assert_eq!(stream_step(1), (VertexStepFunction::PerVertex, 1));
    assert_eq!(
        stream_step(D3DSTREAMSOURCE_INDEXEDDATA | 7),
        (VertexStepFunction::PerVertex, 1)
    );
    assert_eq!(
        stream_step(D3DSTREAMSOURCE_INSTANCEDATA | 1),
        (VertexStepFunction::PerInstance, 1)
    );
    assert_eq!(
        stream_step(D3DSTREAMSOURCE_INSTANCEDATA | 3),
        (VertexStepFunction::PerInstance, 3)
    );
    assert_eq!(
        stream_step(D3DSTREAMSOURCE_INSTANCEDATA),
        (VertexStepFunction::Constant, 0)
    );
}

#[test]
fn instanced_read_bytes_round_up_and_saturate() {
    assert_eq!(
        instanced_stream_read_bytes(4, VertexStepFunction::PerInstance, 1, 12),
        48
    );
    // 5 instances at rate 2 read 3 elements.
    assert_eq!(
        instanced_stream_read_bytes(5, VertexStepFunction::PerInstance, 2, 12),
        36
    );
    assert_eq!(
        instanced_stream_read_bytes(100, VertexStepFunction::Constant, 0, 12),
        12
    );
    assert_eq!(
        instanced_stream_read_bytes(u32::MAX, VertexStepFunction::PerInstance, 1, 16),
        u32::MAX
    );
}

#[test]
fn zero_stride_binds_one_constant_element() {
    // D3D9: a zero `SetStreamSource` stride feeds every vertex the element at
    // the stream offset; Metal spells that as a `Constant` layout.
    assert_eq!(
        bound_stream_layout(0, 28, STREAM_FREQ_DEFAULT),
        StreamLayout {
            stride: 28,
            step: VertexStepFunction::Constant,
            step_rate: 0,
        }
    );
    // The stride wins over an instancing frequency: one element either way.
    assert_eq!(
        bound_stream_layout(0, 12, D3DSTREAMSOURCE_INSTANCEDATA | 2),
        StreamLayout {
            stride: 12,
            step: VertexStepFunction::Constant,
            step_rate: 0,
        }
    );
}

#[test]
fn non_zero_stride_steps_per_frequency_word() {
    assert_eq!(
        bound_stream_layout(48, 36, STREAM_FREQ_DEFAULT),
        StreamLayout {
            stride: 48,
            step: VertexStepFunction::PerVertex,
            step_rate: 1,
        }
    );
    assert_eq!(
        bound_stream_layout(12, 12, D3DSTREAMSOURCE_INSTANCEDATA | 2),
        StreamLayout {
            stride: 12,
            step: VertexStepFunction::PerInstance,
            step_rate: 2,
        }
    );
}

#[test]
fn layout_stride_preserves_the_application_step() {
    assert_eq!(layout_stride(48, 36), 48);
    assert_eq!(layout_stride(36, 36), 36);
    // Crossing attributes will use a separate binding.
    assert_eq!(layout_stride(16, 28), 16);
    // Zero is the declaration extent for the inline (UP) path.
    assert_eq!(layout_stride(0, 28), 28);
}

fn attr(stream: u32, offset: u32, format: VertexFormat) -> VertexAttrDesc {
    VertexAttrDesc {
        attr_index: stream,
        buffer_index: stream,
        offset,
        format,
    }
}

#[test]
fn crossing_attributes_keep_the_original_stride_and_do_not_clobber_another_stream() {
    let mut attrs = [
        attr(0, 0, VertexFormat::Float3),
        attr(0, 28, VertexFormat::UChar4NormalizedBgra),
        attr(1, 0, VertexFormat::Float),
    ];
    let mut layouts = [StreamLayout::UNUSED; 16];
    layouts[0] = bound_stream_layout(16, 32, 1);
    layouts[1] = bound_stream_layout(4, 4, 1);
    let map = remap_crossing_attributes(&mut attrs, &mut layouts).unwrap();
    assert_eq!(attrs[1].buffer_index, 2);
    assert_eq!(attrs[1].offset, 0);
    assert_eq!(map[2].stream, 0);
    assert_eq!(map[2].offset, 28);
    assert_eq!(layouts[2].stride, 16);
    assert_eq!(attrs[2].buffer_index, 1);
}

#[test]
fn remapped_stream_keeps_its_instance_step_and_constant_rate() {
    for freq in [
        D3DSTREAMSOURCE_INSTANCEDATA | 2,
        D3DSTREAMSOURCE_INSTANCEDATA,
    ] {
        let mut attrs = [
            attr(0, 0, VertexFormat::Float3),
            attr(1, 16, VertexFormat::Float),
        ];
        let mut layouts = [StreamLayout::UNUSED; 16];
        layouts[0] = bound_stream_layout(12, 12, 1);
        let expected = bound_stream_layout(4, 20, freq);
        layouts[1] = expected;
        let map = remap_crossing_attributes(&mut attrs, &mut layouts).unwrap();
        assert_eq!(attrs[1].buffer_index, 1);
        assert_eq!(map[1].stream, 1);
        assert_eq!(map[1].offset, 16);
        assert_eq!(layouts[1], expected);
    }
}

#[test]
fn all_crossing_streams_reuse_their_own_slots() {
    let mut attrs: [VertexAttrDesc; 16] =
        std::array::from_fn(|i| attr(u32::try_from(i).unwrap(), 16, VertexFormat::Float));
    let mut layouts = [bound_stream_layout(4, 20, 1); 16];
    let map = remap_crossing_attributes(&mut attrs, &mut layouts).unwrap();
    for (i, a) in attrs.iter().enumerate() {
        assert_eq!(a.buffer_index, u32::try_from(i).unwrap());
        assert_eq!(a.offset, 0);
        assert_eq!(u32::from(map[i].stream), a.buffer_index);
        assert_eq!(map[i].offset, 16);
        assert_eq!(layouts[i].stride, 4);
    }
}

#[test]
fn unsupported_width_alignment_and_slot_pressure_fail_explicitly() {
    let mut layouts = [StreamLayout::UNUSED; 16];
    layouts[0] = bound_stream_layout(4, 16, 1);
    let mut wide = [attr(0, 4, VertexFormat::Float2)];
    assert_eq!(
        remap_crossing_attributes(&mut wide, &mut layouts).err(),
        Some(VertexFetchError::AttributeWiderThanStride)
    );
    // The advanced offset is the stream offset plus the attribute's.
    assert_eq!(
        advanced_binding_offset(0, 6, 64),
        Err(VertexFetchError::UnalignedOffset)
    );
    assert_eq!(advanced_binding_offset(2, 6, 64), Ok(8));
    assert_eq!(
        advanced_binding_offset(60, 4, 64),
        Err(VertexFetchError::OutsideBuffer)
    );
    assert_eq!(
        advanced_binding_offset(u32::MAX - 3, 4, u64::MAX),
        Err(VertexFetchError::OutsideBuffer)
    );
    let mut attrs: Vec<VertexAttrDesc> = (0..16)
        .map(|stream| attr(stream, 0, VertexFormat::Float))
        .collect();
    attrs.push(attr(0, 4, VertexFormat::Float));
    let mut layouts = [bound_stream_layout(4, 8, 1); 16];
    assert_eq!(
        remap_crossing_attributes(&mut attrs, &mut layouts).err(),
        Some(VertexFetchError::NoFreeSlot)
    );
}

#[test]
fn inline_capture_includes_the_last_crossing_attribute_and_checks_overflow() {
    assert_eq!(inline_vertex_span(3, 16, 32), Some(64));
    assert_eq!(inline_vertex_span(3, 16, 12), Some(48));
    assert_eq!(inline_vertex_span(0, 16, 32), Some(0));
    assert_eq!(inline_vertex_span(1, 0, 20), Some(20));
    assert_eq!(inline_vertex_span(u32::MAX, 16, 32), None);
    assert_eq!(inline_vertex_span(2, 4, u32::MAX), None);
}

#[test]
fn a_finite_read_range_grows_by_the_crossing_tail_and_the_end_marker_stays() {
    // Three 16-byte vertices whose last attribute ends at byte 32 of a vertex.
    assert_eq!(crossing_read_size(48, 32, 16), 64);
    // Size 0 is "to the end of the buffer", not a 16-byte range.
    assert_eq!(crossing_read_size(0, 32, 16), 0);
    // A stream that does not cross reads what it read.
    assert_eq!(crossing_read_size(48, 12, 16), 48);
    assert_eq!(crossing_read_size(u32::MAX - 4, 32, 16), u32::MAX);
}

#[test]
fn crossing_fetch_lists_each_slot_of_a_stream_with_its_advance() {
    let attrs = [
        attr(0, 0, VertexFormat::Float3),
        attr(0, 28, VertexFormat::UChar4NormalizedBgra),
        attr(1, 0, VertexFormat::Float),
    ];
    let mut layouts = [StreamLayout::UNUSED; 16];
    layouts[0] = bound_stream_layout(16, 32, 1);
    layouts[1] = bound_stream_layout(4, 4, 1);
    let fetch = CrossingFetch::new(&attrs, &layouts).expect("one crossing attribute");
    assert_eq!(fetch.slots_of(0).collect::<Vec<_>>(), [(0, 0), (2, 28)]);
    assert_eq!(fetch.slots_of(1).collect::<Vec<_>>(), [(1, 0)]);
    assert_eq!(fetch.attrs().len(), 3);
    assert_eq!(fetch.attrs()[1].buffer_index, 2);
    assert_eq!(fetch.attrs()[1].offset, 0);
    assert_eq!(fetch.layouts()[2], layouts[0]);
    // The input layouts are the draw's own: the remap works on a copy.
    assert!(!layouts[2].is_used());
}

#[test]
fn a_stream_whose_attributes_all_cross_binds_only_advanced_slots() {
    let attrs = [
        attr(1, 0, VertexFormat::Float3),
        attr(0, 4, VertexFormat::Float),
    ];
    let mut layouts = [StreamLayout::UNUSED; 16];
    layouts[0] = bound_stream_layout(4, 8, 1);
    layouts[1] = bound_stream_layout(12, 12, 1);
    let fetch = CrossingFetch::new(&attrs, &layouts).expect("one crossing attribute");
    // Slot 0 is released by stream 0 and taken back by its crossing attribute.
    assert_eq!(fetch.slots_of(0).collect::<Vec<_>>(), [(0, 4)]);
    assert_eq!(fetch.slots_of(1).collect::<Vec<_>>(), [(1, 0)]);
}

#[test]
fn equal_remapped_layouts_place_the_attributes_alike() {
    let attrs = [
        attr(0, 0, VertexFormat::Float),
        attr(0, 8, VertexFormat::Float),
        attr(1, 4, VertexFormat::Float),
        attr(2, 12, VertexFormat::Float2),
        attr(2, 0, VertexFormat::Float),
    ];
    let mut seen: Vec<([StreamLayout; 16], [VertexAttrDesc; 5])> = Vec::new();
    for strides in (0..216u32).map(|n| [n % 6, n / 6 % 6, n / 36].map(|s| 4 + 4 * s)) {
        let mut layouts = [StreamLayout::UNUSED; 16];
        for (stream, stride) in strides.into_iter().enumerate() {
            layouts[stream] = bound_stream_layout(stride, 20, 1);
        }
        let mut remapped = attrs;
        if remap_crossing_attributes(&mut remapped, &mut layouts).is_err() {
            continue;
        }
        if let Some((_, placed)) = seen
            .iter()
            .find(|(seen_layouts, _)| *seen_layouts == layouts)
        {
            assert!(
                *placed == remapped,
                "strides {strides:?} collide in the pipeline memo"
            );
        } else {
            seen.push((layouts, remapped));
        }
    }
}

#[test]
fn only_an_advanced_binding_is_held_to_metals_offset_rules() {
    let attrs = [
        attr(0, 0, VertexFormat::Float3),
        attr(0, 18, VertexFormat::UChar4NormalizedBgra),
    ];
    let mut layouts = [StreamLayout::UNUSED; 16];
    layouts[0] = bound_stream_layout(16, 22, 1);
    let fetch = CrossingFetch::new(&attrs, &layouts).expect("one crossing attribute");
    // Stream offset 2: the stream's own binding stays at 2, unchecked, and the
    // advanced one lands on 20.
    assert_eq!(fetch.check_advanced_offsets(1, |_| Some((2, 4096))), Ok(()));
    // Stream offset 0 leaves the advanced binding at 18.
    assert_eq!(
        fetch.check_advanced_offsets(1, |_| Some((0, 4096))),
        Err(VertexFetchError::UnalignedOffset)
    );
    // A stream fed nothing is skipped.
    assert_eq!(fetch.check_advanced_offsets(1, |_| None), Ok(()));
}
