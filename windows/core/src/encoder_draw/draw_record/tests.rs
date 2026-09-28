use mtld3d_shared::command_header::COMMAND_HEADER_BYTES;

use super::*;
use crate::{draw_data::ExtraStreams, ids::BufferId, scratch::ScratchArena};

fn captured(bytes: &'static [u8]) -> ScratchSlice {
    if bytes.is_empty() {
        return ScratchSlice::EMPTY;
    }
    // SAFETY: immutable static storage outlives every command and view in these fixtures.
    unsafe {
        ScratchSlice::from_raw_parts(
            NonNull::new(bytes.as_ptr().cast_mut()).unwrap(),
            u32::try_from(bytes.len()).unwrap(),
        )
    }
}

fn encode<'a>(draw: &DrawOp, arena: &'a mut ScratchArena) -> &'a [u8] {
    let size = payload_size(draw).unwrap();
    let command = arena
        .write_command(4, 0, size, |destination| {
            write_into(draw, destination, size)?;
            Ok(size)
        })
        .unwrap();
    // SAFETY: arena owns the complete initialized payload until this borrow ends.
    unsafe {
        core::slice::from_raw_parts(
            (command.address + COMMAND_HEADER_BYTES as u64) as *const u8,
            size,
        )
    }
}

fn stream(index: u8) -> StreamBinding {
    StreamBinding {
        stream: index,
        buffer_id: BufferId::new_unique(),
        backing_ptr: 0,
        backing_len: 4096,
        backing_generation: u64::from(index),
        offset: 0,
        stride: 16,
        freq: 1,
    }
}

fn bound(indices: IndexSource) -> DrawOp {
    DrawOp {
        metal_prim: PrimitiveType::Triangle,
        vertex_source: VertexSource::Bound {
            first: stream(0),
            extra: ExtraStreams::EMPTY,
            stream0_freq: 1,
        },
        index_source: indices,
    }
}

#[test]
fn incorrect_supplied_size_cannot_publish_unwritten_command_bytes() {
    let draw = bound(IndexSource::None {
        start_vertex: 0,
        vertex_count: 3,
    });
    let size = payload_size(&draw).unwrap();
    let mut arena = ScratchArena::new();
    let first = encode(&draw, &mut arena).to_vec();
    let committed = arena.bytes_used();
    let error = arena.write_command(4, 0, size + 8, |destination| {
        write_into(&draw, destination, size + 8)?;
        Ok(size + 8)
    });
    assert!(matches!(error, Err(WireError::InvalidValue)));
    assert_eq!(arena.bytes_used(), committed);
    assert_eq!(encode(&draw, &mut arena), first);
}

#[test]
fn undersized_supplied_reservation_cannot_commit_a_partial_draw() {
    let draw = bound(IndexSource::None {
        start_vertex: 0,
        vertex_count: 3,
    });
    let size = payload_size(&draw).unwrap();
    let mut arena = ScratchArena::new();
    let prefix = encode(&draw, &mut arena);
    let pointer = prefix.as_ptr();
    let first = prefix.to_vec();
    let committed = arena.bytes_used();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        arena.write_command(4, 0, size - 1, |destination| {
            write_into(&draw, destination, size - 1)?;
            Ok(size - 1)
        })
    }));
    assert!(result.is_err());
    assert_eq!(arena.bytes_used(), committed);
    // SAFETY: the arena retains the earlier immutable command across failed writes.
    let prefix = unsafe { core::slice::from_raw_parts(pointer, first.len()) };
    assert_eq!(prefix, first);
    assert_eq!(encode(&draw, &mut arena), first);
}

#[test]
fn destination_mismatch_is_rejected_before_writing() {
    let draw = bound(IndexSource::None {
        start_vertex: 0,
        vertex_count: 3,
    });
    let size = payload_size(&draw).unwrap();
    for length in [size - 1, size + 1] {
        let mut bytes = vec![0xa5; length];
        assert_eq!(
            write_into(&draw, &mut bytes, size),
            Err(WireError::InvalidValue)
        );
        assert!(bytes.iter().all(|byte| *byte == 0xa5));
    }
}

fn checked_draw_fields(bytes: &[u8]) -> Result<(), WireError> {
    DrawView::new(bytes).map(drop)
}

#[test]
fn all_index_sources_use_actual_fixed_views() {
    let fixtures = [
        IndexSource::None {
            start_vertex: 2,
            vertex_count: 3,
        },
        IndexSource::Bound {
            buffer_id: BufferId::new_unique(),
            backing_ptr: 0,
            backing_len: 4096,
            backing_generation: 9,
            offset: 4,
            index_count: 3,
            index_type: IndexType::UInt16,
            base_vertex: -2,
        },
        IndexSource::Fan {
            start_vertex: 5,
            primitive_count: 2,
        },
        IndexSource::Generated {
            data: captured(&[0, 0, 1, 0, 2, 0]),
            index_count: 3,
            index_type: IndexType::UInt16,
            min_vertex: 0,
            max_vertex: 2,
        },
        IndexSource::Up {
            bytes: captured(&[0, 0, 1, 0, 2, 0]),
            index_count: 3,
            index_type: IndexType::UInt16,
        },
    ];
    for (kind, indices) in fixtures.into_iter().enumerate() {
        let draw = bound(indices);
        for prefix_commands in 0..2 {
            let mut arena = ScratchArena::new();
            if prefix_commands != 0 {
                arena.write_command(1, 0, 0, |_| Ok(0)).unwrap();
            }
            let payload = encode(&draw, &mut arena);
            assert_eq!(
                payload.as_ptr() as usize % 16,
                (prefix_commands + 1) % 2 * 8
            );
            let view = DrawView::new(payload).unwrap();
            assert_eq!(view.metal_primitive(), PrimitiveType::Triangle);
            match (kind, view.indices()) {
                (
                    0,
                    &IndexView::None {
                        start_vertex,
                        vertex_count,
                    },
                ) => assert_eq!((start_vertex, vertex_count), (2, 3)),
                (
                    1,
                    &IndexView::Bound {
                        record,
                        index_count,
                        base_vertex,
                    },
                ) => {
                    assert_eq!(
                        (record.generation, record.offset, index_count, base_vertex),
                        (9, 4, 3, -2)
                    );
                    assert_eq!(record.index_type().unwrap(), IndexType::UInt16);
                }
                (
                    2,
                    &IndexView::Fan {
                        start_vertex,
                        primitive_count,
                    },
                ) => assert_eq!((start_vertex, primitive_count), (5, 2)),
                (
                    3,
                    &IndexView::Generated {
                        record,
                        index_count,
                        min_vertex,
                    },
                ) => assert_eq!((record.maximum, index_count, min_vertex), (2, 3, 0)),
                (
                    4,
                    &IndexView::Up {
                        record,
                        index_count,
                    },
                ) => {
                    assert_eq!(record.length, 6);
                    assert_eq!(index_count, 3);
                }
                _ => panic!("wrong fixed index variant"),
            }
            for length in 0..payload.len() {
                assert!(checked_draw_fields(&payload[..length]).is_err());
            }
        }
    }
}

#[test]
fn large_up_bytes_keep_original_capture_identity_and_fixed_record_size() {
    static LARGE: [u8; 4800] = [9; 4800];
    let draw = DrawOp {
        metal_prim: PrimitiveType::Triangle,
        vertex_source: VertexSource::Up {
            bytes: captured(&LARGE),
            size: 4800,
            stride: 16,
        },
        index_source: IndexSource::Up {
            bytes: captured(&[0, 0, 1, 0, 2, 0]),
            index_count: 3,
            index_type: IndexType::UInt16,
        },
    };
    let mut arena = ScratchArena::new();
    let bytes = encode(&draw, &mut arena);
    assert_eq!(bytes.len(), 56);
    let view = DrawView::new(bytes).unwrap();
    let &VertexView::Up { record, stride } = view.vertices() else {
        unreachable!()
    };
    assert_eq!(
        (record.address, record.length, record.size, stride),
        (LARGE.as_ptr() as u64, 4800, 4800, 16)
    );
    // SAFETY: the record references the immutable static fixture retained for this test.
    assert_eq!(unsafe { record.bytes() }.as_slice(), LARGE);
    let IndexView::Up { record, .. } = view.indices() else {
        unreachable!()
    };
    assert_eq!(record.index_type().unwrap(), IndexType::UInt16);
}

#[test]
fn sixteen_streams_borrow_fixed_records_without_rebuilding_bindings() {
    let draw = DrawOp {
        metal_prim: PrimitiveType::Triangle,
        vertex_source: VertexSource::Bound {
            first: stream(0),
            extra: ExtraStreams::Owned((1..16).map(stream).collect()),
            stream0_freq: 1,
        },
        index_source: IndexSource::None {
            start_vertex: 0,
            vertex_count: 3,
        },
    };
    let mut arena = ScratchArena::new();
    let bytes = encode(&draw, &mut arena);
    let view = DrawView::new(bytes).unwrap();
    let vertices = view.vertices();
    assert_eq!(vertices.bindings().len(), 16);
    for (index, record) in vertices.bindings().enumerate() {
        assert_eq!(usize::from(record.stream), index);
        assert_eq!(record.address, 0);
        assert_eq!(record.length, 4096);
        assert_eq!(record.reserved, [0; 3]);
        assert_eq!(
            core::ptr::from_ref(record) as usize,
            bytes.as_ptr() as usize + 16 + index * 48
        );
    }
}

#[test]
fn padding_is_initialized_and_unknown_fixed_tags_are_rejected() {
    let draw = bound(IndexSource::Bound {
        buffer_id: BufferId::new_unique(),
        backing_ptr: 0,
        backing_len: 4096,
        backing_generation: 0,
        offset: 0,
        index_count: 3,
        index_type: IndexType::UInt32,
        base_vertex: 0,
    });
    let mut arena = ScratchArena::new();
    let bytes = encode(&draw, &mut arena);
    assert_eq!(&bytes[61..64], &[0; 3]);
    assert_eq!(&bytes[101..104], &[0; 3]);
    let original = bytes.to_vec();
    for offset in [0, 1, 2, 3] {
        let mut changed = original.clone();
        changed[offset] = 255;
        let address = arena.alloc(&changed);
        // SAFETY: arena owns the initialized malformed scalar fixture through validation.
        let changed = unsafe { core::slice::from_raw_parts(address as *const u8, changed.len()) };
        // The primitive (0), vertex kind (1), index kind (2) and stream count (3)
        // are each rejected by the decode, before any consumer reads a field.
        assert_eq!(checked_draw_fields(changed), Err(WireError::InvalidValue));
    }
}

#[test]
fn ordinary_bound_capture_matches_generic_bytes_for_sparse_and_full_streams() {
    for stream_ids in [vec![7], vec![2, 7, 15], (0..16).collect()] {
        for index_kind in [None, Some(IndexType::UInt16), Some(IndexType::UInt32)] {
            let mut bindings = stream_ids.iter().copied().map(stream);
            let draw = DrawOp {
                metal_prim: PrimitiveType::Triangle,
                vertex_source: VertexSource::Bound {
                    first: bindings.next().unwrap(),
                    extra: ExtraStreams::Owned(bindings.collect()),
                    stream0_freq: 0x4000_0011,
                },
                index_source: index_kind.map_or(
                    IndexSource::None {
                        start_vertex: u32::MAX - 9,
                        vertex_count: 9,
                    },
                    |index_type| IndexSource::Bound {
                        buffer_id: BufferId::new_unique(),
                        backing_ptr: 0x1234,
                        backing_len: 8192,
                        backing_generation: 123,
                        offset: 12,
                        index_count: 9,
                        index_type,
                        base_vertex: i32::MIN,
                    },
                ),
            };
            let expected = encode(&draw, &mut ScratchArena::new()).to_vec();
            let VertexSource::Bound {
                first,
                extra,
                stream0_freq,
            } = draw.vertex_source
            else {
                unreachable!()
            };
            let vertices = BoundVertices {
                first: StreamRecord::from_binding(&first),
                extra,
                stream0_freq,
            };
            let (prefix, index) = match draw.index_source {
                IndexSource::None {
                    start_vertex,
                    vertex_count,
                } => (
                    DrawPrefix::nonindexed(draw.metal_prim, start_vertex, vertex_count),
                    None,
                ),
                IndexSource::Bound {
                    buffer_id,
                    backing_ptr,
                    backing_len,
                    backing_generation,
                    offset,
                    index_count,
                    index_type,
                    base_vertex,
                } => (
                    DrawPrefix::indexed(draw.metal_prim, base_vertex, index_count),
                    Some(IndexBuffer {
                        buffer: buffer_id.raw(),
                        address: backing_ptr as u64,
                        length: backing_len as u64,
                        generation: backing_generation,
                        offset,
                        kind: index_type as u8,
                        reserved: [0; 3],
                    }),
                ),
                _ => unreachable!(),
            };
            let length = bound_payload_size(&vertices, index.as_ref()).unwrap();
            let mut actual = vec![0xcc; length];
            write_bound_into(prefix, &vertices, index.as_ref(), &mut actual, length).unwrap();
            assert_eq!(actual, expected);
        }
    }
}

#[test]
fn ordinary_bound_capture_rejects_bad_extents_and_stream_counts_before_publication() {
    let mut vertices = BoundVertices {
        first: StreamRecord::from_binding(&stream(0)),
        extra: ExtraStreams::EMPTY,
        stream0_freq: 1,
    };
    let length = bound_payload_size(&vertices, None).unwrap();
    let mut arena = ScratchArena::new();
    let prefix = || DrawPrefix::nonindexed(PrimitiveType::Triangle, 0, 3);
    for extra_bytes in [0, 8] {
        let used = arena.bytes_used();
        let result = arena.write_command(4, 0, length + extra_bytes, |destination| {
            // The indexed prefix has no index tail; an oversized reservation has
            // unwritten bytes. Neither failure may publish a command.
            let header = if extra_bytes == 0 {
                DrawPrefix::indexed(PrimitiveType::Triangle, -1, 3)
            } else {
                prefix()
            };
            write_bound_into(header, &vertices, None, destination, length + extra_bytes)?;
            Ok(length + extra_bytes)
        });
        assert!(matches!(result, Err(WireError::InvalidValue)));
        assert_eq!(arena.bytes_used(), used);
    }
    let used = arena.bytes_used();
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = arena.write_command(4, 0, length - 8, |destination| {
            write_bound_into(prefix(), &vertices, None, destination, length - 8)?;
            Ok(length - 8)
        });
    }));
    assert!(panic.is_err());
    assert_eq!(arena.bytes_used(), used);
    arena
        .write_command(4, 0, length, |destination| {
            write_bound_into(prefix(), &vertices, None, destination, length)?;
            Ok(length)
        })
        .unwrap();
    assert_eq!(
        arena.bytes_used(),
        used + u64::try_from(COMMAND_HEADER_BYTES + length).unwrap()
    );
    vertices.extra = ExtraStreams::Owned((0..16).map(stream).collect());
    assert_eq!(
        bound_payload_size(&vertices, None),
        Err(WireError::InvalidValue)
    );
}

/// A copy of `bytes` in 8-aligned arena storage, `shift` bytes past an aligned start.
fn stored<'a>(arena: &'a mut ScratchArena, bytes: &[u8], shift: usize) -> &'a [u8] {
    let mut padded = vec![0; shift];
    padded.extend_from_slice(bytes);
    let address = usize::try_from(arena.alloc(&padded)).expect("host address") + shift;
    // SAFETY: the arena owns the initialized copy and cannot be reset or dropped while the
    // returned borrow of it lives.
    unsafe { core::slice::from_raw_parts(address as *const u8, bytes.len()) }
}

#[test]
fn decode_checks_index_type_extent_and_alignment_once() {
    let draw = bound(IndexSource::Bound {
        buffer_id: BufferId::new_unique(),
        backing_ptr: 0,
        backing_len: 4096,
        backing_generation: 0,
        offset: 0,
        index_count: 3,
        index_type: IndexType::UInt32,
        base_vertex: 0,
    });
    let mut arena = ScratchArena::new();
    let original = encode(&draw, &mut arena).to_vec();
    let mut checks = ScratchArena::new();
    let view = DrawView::new(stored(&mut checks, &original, 0)).unwrap();
    let &IndexView::Bound { record, .. } = view.indices() else {
        unreachable!()
    };
    assert_eq!(record.index_type(), Ok(IndexType::UInt32));
    // The index element type at offset 64 + 36.
    let mut unknown_type = original.clone();
    unknown_type[100] = 9;
    assert_eq!(
        DrawView::new(stored(&mut checks, &unknown_type, 0)).map(drop),
        Err(WireError::InvalidValue)
    );
    // Bytes past the index record, and a stream array cut short.
    let mut trailing = original.clone();
    trailing.extend_from_slice(&[0; 8]);
    assert_eq!(
        DrawView::new(stored(&mut checks, &trailing, 0)).map(drop),
        Err(WireError::InvalidValue)
    );
    assert_eq!(
        DrawView::new(stored(&mut checks, &original[..40], 0)).map(drop),
        Err(WireError::Truncated)
    );
    // The same bytes off their 8-byte alignment.
    assert_eq!(
        DrawView::new(stored(&mut checks, &original, 4)).map(drop),
        Err(WireError::InvalidValue)
    );
}
