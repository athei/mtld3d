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
