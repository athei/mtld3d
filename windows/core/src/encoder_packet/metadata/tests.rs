use mtld3d_shared::{MetalHandle, mtl::PixelFormat, record_handle::DeviceRecordHandle};

use super::*;
use crate::{encoder_data::FrameInit, passes::BackbufferContents, render_scale::RenderScale};

fn frame() -> FrameData {
    FrameData::new(&FrameInit {
        device_handle: MetalHandle::NULL,
        record_handle: DeviceRecordHandle::NULL,
        backbuffer_handle: MetalHandle::NULL,
        backbuffer_srgb_handle: MetalHandle::NULL,
        backbuffer_msaa_handle: MetalHandle::NULL,
        backbuffer_msaa_srgb_handle: MetalHandle::NULL,
        backbuffer_sample_count: 1,
        layer_handle: MetalHandle::NULL,
        view_handle: MetalHandle::NULL,
        backbuffer_width: 320,
        backbuffer_height: 200,
        backbuffer_format: PixelFormat::Bgra8Unorm,
        render_scale: RenderScale::IDENTITY,
        backbuffer_contents: BackbufferContents::Undefined,
        depth_texture: MetalHandle::NULL,
        depth_has_stencil: false,
    })
}

#[test]
fn header_borrows_the_original_arena_payload() {
    let mut frame = frame();
    frame.perf_mut().set_present_block_cycles(77);
    let payload = core::ptr::from_ref(frame.perf());
    let recorder = FrameRecorder::new();
    let mut metadata = MetadataStorage::new();
    metadata.seal(&mut frame, &recorder).unwrap();
    // SAFETY: frame and recorder retain the initialized header and its only optional payload.
    let bytes = unsafe { metadata.as_bytes() };
    // SAFETY: the retained frame contains the initialized canonical header.
    let view = unsafe { FrameView::from_bytes(bytes) }.unwrap();
    assert_eq!(view.header().backbuffer_width, 320);
    assert_eq!(view.header().backbuffer_height, 200);
    assert_eq!(size_of_val(view.header()), size_of::<FrameMetadata>());
    #[cfg(perf_tracking)]
    assert!(core::ptr::eq(view.perf().unwrap().unwrap(), payload));
    #[cfg(not(perf_tracking))]
    {
        let _ = payload;
        assert!(view.perf().unwrap().is_none());
    }
}

#[test]
fn warmup_capture_is_already_a_command_before_frame_sealing() {
    let mut frame = frame();
    frame.push_buffer_warmup(VbibWarmupEntry {
        buffer_id: crate::ids::BufferId::new_unique(),
        backing_ptr: 0x1000,
        backing_len: 4096,
        backing_generation: 7,
        map_mode: crate::buffer_rename::BufferMapMode::Direct,
    });
    let recorder = frame.recorder.as_ref().unwrap();
    assert_eq!(recorder.len(), 1);
    assert_eq!(recorder.metadata.header, 0);
}

#[test]
fn multiple_gamma_changes_retain_each_original_lut_until_replay() {
    let mut frame = frame();
    let first = Box::new([17; crate::gamma::LUT_LANES]);
    let second = Box::new([29; crate::gamma::LUT_LANES]);
    let first_pointer = first.as_ptr();
    let second_pointer = second.as_ptr();
    frame.set_apply_gamma(Some(crate::gamma::Change::Apply(first)));
    frame.set_apply_gamma(Some(crate::gamma::Change::Apply(second)));
    frame.set_apply_gamma(Some(crate::gamma::Change::Remove));
    let recorder = frame.recorder.as_ref().unwrap();
    assert_eq!(recorder.len(), 3);
    assert_eq!(recorder.gamma_tables.len(), 2);
    assert_eq!(recorder.gamma_tables[0].table.as_ptr(), first_pointer);
    assert_eq!(recorder.gamma_tables[1].table.as_ptr(), second_pointer);
    assert_eq!(recorder.gamma_tables[0].table[0], 17);
    assert_eq!(recorder.gamma_tables[1].table[0], 29);
}

#[test]
fn retired_scratch_forgets_its_perf_pointer_before_reuse() {
    let mut frame = frame();
    frame.perf_mut().set_present_block_cycles(19);
    let previous = core::ptr::from_ref(frame.perf());
    let scratch = frame.take_recording_scratch();
    #[cfg(perf_tracking)]
    assert!(!core::ptr::eq(previous, frame.perf()));
    #[cfg(not(perf_tracking))]
    let _ = previous;
    frame.scratch = scratch;
    frame.perf_mut().set_present_block_cycles(31);
    assert!(frame.scratch.bytes_used() > 0 || !cfg!(perf_tracking));
}

#[cfg(perf_tracking)]
#[test]
fn telemetry_follows_safe_arena_swaps_and_direct_clear() {
    let mut frame = frame();
    frame.perf_mut().set_present_block_cycles(3);
    let original = core::ptr::from_ref(frame.perf());
    let mut other = crate::scratch::ScratchArena::new();
    other.perf_mut().set_present_block_cycles(5);
    let replacement = core::ptr::from_ref(other.perf());
    core::mem::swap(&mut frame.scratch, &mut other);
    assert!(core::ptr::eq(frame.perf(), replacement));
    assert!(core::ptr::eq(other.perf(), original));
    frame.scratch.clear();
    assert!(!core::ptr::eq(frame.perf(), replacement));
    drop(other);
    frame.perf_mut().set_present_block_cycles(7);
}
