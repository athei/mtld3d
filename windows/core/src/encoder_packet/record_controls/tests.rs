use std::sync::Arc;

use super::*;
use crate::{
    encoder_data::{BeginVisibilityOp, SetViewportOp},
    visibility::VisibilityQueryCore,
};

fn first_payload<T: CommandRecord>(recorder: &FrameRecorder) -> &T {
    let (address, length) = recorder.slab.ranges().next().unwrap();
    assert!(length >= 16 + size_of::<T>() as u64);
    // SAFETY: the test retains the recorder's scratch arena and this exact initialized
    // command region. Its fixed header is 16 bytes and its payload has the asserted size.
    let payload =
        unsafe { core::slice::from_raw_parts((address + 16) as *const u8, size_of::<T>()) };
    crate::encoder_records::borrow(payload).unwrap()
}

#[test]
fn typed_viewport_is_constructed_in_final_command_storage() {
    let mut scratch = ScratchArena::new();
    let mut recorder = FrameRecorder::new();
    recorder
        .record_typed(
            &mut scratch,
            SetViewportOp {
                x: 7,
                y: 9,
                width: 640,
                height: 480,
                min_z: 0.25,
                max_z: 0.75,
            },
        )
        .unwrap();
    let record = first_payload::<SetViewportRecord>(&recorder);
    assert_eq!(
        (record.x, record.y, record.width, record.height),
        (7, 9, 640, 480)
    );
    assert_eq!((record.min_z, record.max_z), (0.25, 0.75));
    assert_eq!(recorder.len(), 1);
    assert!(recorder.rejected_ops.is_empty());
}

#[test]
fn typed_query_retains_original_owner_and_captured_generation() {
    let mut scratch = ScratchArena::new();
    let mut recorder = FrameRecorder::new();
    let query = VisibilityQueryCore::new();
    let weak = Arc::downgrade(&query);
    recorder
        .record_typed(
            &mut scratch,
            BeginVisibilityOp {
                generation: 37,
                c: query,
            },
        )
        .unwrap();
    assert_eq!(first_payload::<QueryRecord>(&recorder).generation, 37);
    assert_eq!(recorder.queries.len(), 1);
    assert!(weak.upgrade().is_some());
    assert!(recorder.rejected_ops.is_empty());
    // SAFETY: this fixture was never published and no native reader can adopt the lease.
    unsafe { recorder.queries[0].cancel_unadopted() };
    drop(recorder);
    assert!(weak.upgrade().is_none());
}

#[test]
fn latched_failure_retains_typed_owner_without_publishing_command() {
    let mut scratch = ScratchArena::new();
    let mut recorder = FrameRecorder::new();
    recorder.error = Some(WireError::AllocationFailed);
    let query = VisibilityQueryCore::new();
    let weak = Arc::downgrade(&query);
    assert!(
        recorder
            .record_typed(
                &mut scratch,
                BeginVisibilityOp {
                    generation: 9,
                    c: query
                }
            )
            .is_err()
    );
    assert!(recorder.slab.ranges().next().is_none());
    assert!(recorder.queries.is_empty());
    assert_eq!(recorder.rejected_ops.len(), 1);
    assert!(weak.upgrade().is_some());
    drop(recorder);
    assert!(weak.upgrade().is_none());
}
