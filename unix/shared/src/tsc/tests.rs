use super::{ns_to_cycles, tsc_hz};

/// A duration handed across the boundary lands on this side's own rate.
///
/// The conversion is the whole reason `drawable_wait_ns` travels in
/// nanoseconds; if it stopped scaling by the local `tsc_hz`, the perf
/// summary would report the unix side's wait in the wrong unit and nothing
/// would fail loudly.
#[test]
fn ns_convert_to_locally_calibrated_cycles() {
    // Zero short-circuits, which is what keeps a non-measuring build from
    // paying the calibration sleep.
    assert_eq!(ns_to_cycles(0), 0);

    // One second of nanoseconds is, by definition, `tsc_hz` cycles. The
    // integer division makes the result exact only to the tick, so allow
    // the one-cycle truncation.
    let hz = tsc_hz();
    let one_second = ns_to_cycles(1_000_000_000);
    assert!(
        one_second.abs_diff(hz) <= 1,
        "1 s converted to {one_second} cycles, calibrated rate is {hz}"
    );

    // A frame-sized wait stays well inside the range and keeps its order.
    assert!(ns_to_cycles(16_000_000) < one_second);
    assert!(ns_to_cycles(16_000_000) > ns_to_cycles(8_000_000));
}
