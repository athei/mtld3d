use super::{CommandBufferRole, GpuBusy, PipelineTimings, ShaderTimings, SubmitTimings};

/// The scalar timing records keep one layout for the i686 PE caller and the 64-bit handler.
#[test]
fn scalar_timing_layouts_are_stable() {
    assert_eq!(size_of::<ShaderTimings>(), 24);
    assert_eq!(size_of::<PipelineTimings>(), 16);
}

/// The submit timings keep one layout for the i686 PE caller and the 64-bit handler.
#[test]
fn submit_timings_layout_is_stable() {
    assert_eq!(size_of::<GpuBusy>(), 16);
    assert_eq!(align_of::<SubmitTimings>(), 8);
    assert_eq!(size_of::<SubmitTimings>(), 72);
    assert_eq!(core::mem::offset_of!(SubmitTimings, gpu), 24);
    let timings = SubmitTimings::new();
    let present = &timings.gpu[CommandBufferRole::Present as usize];
    assert_eq!((present.ns, present.buffers), (0, 0));
}
