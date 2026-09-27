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
            write_into(draw, destination)?;
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
            assert_eq!(view.metal_primitive().unwrap(), PrimitiveType::Triangle);
            match (kind, view.indices().unwrap()) {
                (
                    0,
                    IndexView::None {
                        start_vertex,
                        vertex_count,
                    },
                ) => assert_eq!((start_vertex, vertex_count), (2, 3)),
                (
                    1,
                    IndexView::Bound {
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
                    IndexView::Fan {
                        start_vertex,
                        primitive_count,
                    },
                ) => assert_eq!((start_vertex, primitive_count), (5, 2)),
                (
                    3,
                    IndexView::Generated {
                        record,
                        index_count,
                        min_vertex,
                    },
                ) => assert_eq!((record.maximum, index_count, min_vertex), (2, 3, 0)),
                (
                    4,
                    IndexView::Up {
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
                assert!(DrawView::new(&payload[..length]).is_err());
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
    let VertexView::Up { record, stride } = view.vertices().unwrap() else {
        unreachable!()
    };
    assert_eq!(
        (record.address, record.length, record.size, stride),
        (LARGE.as_ptr() as u64, 4800, 4800, 16)
    );
    // SAFETY: the record references the immutable static fixture retained for this test.
    assert_eq!(unsafe { record.bytes() }.as_slice(), LARGE);
    let IndexView::Up { record, .. } = view.indices().unwrap() else {
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
    let vertices = view.vertices().unwrap();
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
        assert!(DrawView::new(changed).is_err());
    }
}
