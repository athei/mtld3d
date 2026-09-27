use super::*;

fn sample() -> PendingSample {
    let mut sample = super::super::tests::sample(77, 0);
    sample.counters.api_cycles_by_category[0] = 200;
    sample.timing.frame_total_cycles = 600;
    sample.timing.present_block_cycles = 20;
    PendingSample {
        sample,
        compilation: super::super::compilation::CompilationPerf::new().defer_frame(),
        nanos: SubmitNanos {
            drawable: 1_000_000_000,
            ..SubmitNanos::default()
        },
        captured_at: 123,
    }
}

#[test]
fn delayed_frequency_preserves_all_samples_and_units() {
    let source = ClockCalibration::new();
    let native = Arc::new(ClockCalibration::new());
    // SAFETY: source outlives the queue and both mailboxes have one publisher.
    let mut queue =
        unsafe { ClockedSamples::new(core::ptr::from_ref(&source) as u64, Arc::clone(&native)) };
    queue.push(sample());
    queue.push(sample());
    assert!(queue.take_ready().is_none());
    // SAFETY: this test is the sole source publisher.
    unsafe {
        source.publish_ready(100);
    }
    assert!(queue.take_ready().is_none());
    // SAFETY: this test is the sole native publisher.
    unsafe {
        native.publish_ready(1_000);
    }
    for _ in 0..2 {
        let (sample, _, hz, captured_at) = queue.take_ready().unwrap();
        assert_eq!((hz, captured_at), (1_000, 123));
        assert_eq!(
            (sample.api_cyc, sample.api_work, sample.outside_d3d9),
            (2_000, 1_800, 4_000)
        );
        assert_eq!(sample.enc_cyc, 77);
        assert_eq!(sample.enc.drawable_wait_cycles, 1_000);
    }
    queue.finish();
    assert!(queue.pending.is_empty());
    assert!(queue.failure.is_none());
}

#[test]
fn failed_calibration_retains_evidence_and_stops_capture() {
    let source = ClockCalibration::new();
    let native = Arc::new(ClockCalibration::new());
    // SAFETY: source outlives the queue and has one publisher.
    let mut queue = unsafe { ClockedSamples::new(core::ptr::from_ref(&source) as u64, native) };
    queue.push(sample());
    // SAFETY: this test is the sole source publisher.
    unsafe {
        source.publish_failed();
    }
    assert!(queue.take_ready().is_none());
    queue.push(sample());
    queue.finish();
    assert_eq!(queue.pending.len(), 1);
    assert_eq!(queue.rejected, 1);
    assert_eq!(queue.failure, Some("source calibration failed"));
}

#[test]
fn pending_capture_is_bounded_and_incomplete_shutdown_invalidates() {
    let source = ClockCalibration::new();
    let native = Arc::new(ClockCalibration::new());
    // SAFETY: source outlives both queues.
    let mut queue =
        unsafe { ClockedSamples::new(core::ptr::from_ref(&source) as u64, Arc::clone(&native)) };
    for _ in 0..=MAX_PENDING_SAMPLES {
        queue.push(sample());
    }
    assert_eq!(queue.pending.len(), MAX_PENDING_SAMPLES);
    assert_eq!(queue.rejected, 1);
    assert!(queue.failure.is_some());
    // SAFETY: source outlives this queue.
    let mut unfinished =
        unsafe { ClockedSamples::new(core::ptr::from_ref(&source) as u64, native) };
    unfinished.push(sample());
    unfinished.finish();
    assert_eq!(
        unfinished.failure,
        Some("calibration incomplete at shutdown")
    );
    assert_eq!(unfinished.pending.len(), 1);
}
