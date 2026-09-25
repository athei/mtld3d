use mtld3d_shared::perf::{CommandBufferRole, SubmitTimings};

use super::{GpuTime, busy_ns};

/// A buffer's host times become whole nanoseconds of GPU execution.
#[test]
fn busy_time_is_end_minus_start_in_nanoseconds() {
    assert_eq!(busy_ns(10.0, 10.25), Some(250_000_000));
    assert_eq!(busy_ns(1.5, 1.5), Some(0));
}

/// A buffer that never started, or whose end precedes its start, reports nothing.
#[test]
fn unstarted_or_inverted_buffers_report_nothing() {
    assert_eq!(busy_ns(0.0, 4.0), None);
    assert_eq!(busy_ns(5.0, 4.0), None);
    assert_eq!(busy_ns(f64::NAN, 4.0), None);
}

/// Sums accumulate per role, and a take hands them over once and leaves zero behind.
#[test]
fn take_hands_each_sum_over_once() {
    let time = GpuTime::new();
    time.add(CommandBufferRole::Frame, 3_000);
    time.add(CommandBufferRole::Frame, 4_000);
    time.add(CommandBufferRole::Present, 500);

    let mut first = SubmitTimings::new();
    time.take(&mut first.gpu);
    let mut second = SubmitTimings::new();
    time.take(&mut second.gpu);

    let frame = &first.gpu[CommandBufferRole::Frame as usize];
    let upload = &first.gpu[CommandBufferRole::Upload as usize];
    let present = &first.gpu[CommandBufferRole::Present as usize];
    assert_eq!((frame.ns, frame.buffers), (7_000, 2));
    assert_eq!((upload.ns, upload.buffers), (0, 0));
    assert_eq!((present.ns, present.buffers), (500, 1));
    assert!(
        second
            .gpu
            .iter()
            .all(|busy| busy.ns == 0 && busy.buffers == 0)
    );
}
