use super::{PipelineTimings, ShaderTimings, TimingOutput};

#[test]
fn timing_output_layout_and_fallback_are_stable() {
    assert_eq!(size_of::<TimingOutput<ShaderTimings>>(), 24);
    assert_eq!(
        align_of::<TimingOutput<ShaderTimings>>(),
        align_of::<ShaderTimings>()
    );
    assert_eq!(size_of::<TimingOutput<PipelineTimings>>(), 16);
    let value = TimingOutput::<ShaderTimings>::new().into_inner();
    assert_eq!(
        (value.preparation_ns, value.library_ns, value.function_ns),
        (0, 0, 0)
    );
}

#[test]
fn timing_output_publishes_only_in_perf_builds() {
    let mut output = TimingOutput::new();
    output.write(PipelineTimings {
        preparation_ns: 12,
        build_ns: 34,
    });
    let value = output.into_inner();
    let expected = if cfg!(perf_tracking) {
        (12, 34)
    } else {
        (0, 0)
    };
    assert_eq!((value.preparation_ns, value.build_ns), expected);
}
