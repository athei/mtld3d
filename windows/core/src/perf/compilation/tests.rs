#[cfg(perf_tracking)]
use super::{CompilationPerf, Identity, Kind};

#[cfg(perf_tracking)]
fn identity() -> Identity {
    Identity::Depth { device: 7, key: 9 }
}

#[test]
#[cfg(perf_tracking)]
fn peaks_are_per_frame_and_residuals_use_the_same_frame() {
    let mut perf = CompilationPerf::new();
    perf.record_enabled(Kind::ShaderVs, 4_000_000, true, 1, identity);
    perf.record_enabled(Kind::ShaderVs, 6_000_000, false, 1, identity);
    perf.record_enabled(Kind::Library, 3_000_000, false, 1, identity);
    perf.record_enabled(Kind::Sibling, 2_000_000, true, 1, identity);
    perf.finish_frame(12_000_000, 8_000_000, 30_000_000);
    perf.record_enabled(Kind::ShaderVs, 1_000_000, true, 2, identity);
    perf.record_enabled(Kind::Library, 2_000_000, true, 2, identity);
    perf.finish_frame(9_000_000, 1_000_000, 10_000_000);
    let vs = &perf.window[Kind::ShaderVs as usize];
    assert_eq!(
        (vs.ns, vs.peak_ns, vs.calls, vs.failures),
        (11_000_000, 10_000_000, 3, 1)
    );
    assert_eq!(perf.window[Kind::ResolveOther as usize].peak_ns, 8_000_000);
    assert_eq!(perf.window[Kind::PipelineOther as usize].peak_ns, 6_000_000);
    assert_eq!(perf.slow[0].encoder_ns, Some(30_000_000));
    assert_eq!(perf.slow[1].encoder_ns, Some(10_000_000));
    assert!(perf.frame.iter().all(|metric| metric.calls == 0));
    let mut output = String::new();
    perf.append_window(&mut output, 2);
    assert!(output.contains("calls=3 failed=1"));
    assert!(output.contains("encoder_ops_same_submission=30.000ms"));
    assert!(output.contains("device=0x7"));
    assert!(perf.slow.is_empty());
    assert!(perf.window.iter().all(|metric| metric.ns == 0));
}

#[test]
#[cfg(perf_tracking)]
fn slow_records_are_bounded_lazy_and_exclude_parent_duplicates() {
    let mut perf = CompilationPerf::new();
    perf.record_enabled(Kind::Pipeline, 90_000_000, true, 1, || {
        panic!("parent retained")
    });
    perf.record_enabled(Kind::PipelineBuild, 1_999_999, true, 1, || {
        panic!("below threshold")
    });
    for ms in 2..=6 {
        perf.record_enabled(Kind::Library, ms * 1_000_000, true, ms, identity);
    }
    perf.record_enabled(Kind::Library, 2_000_000, true, 7, || {
        panic!("non-winning identity")
    });
    perf.record_enabled(Kind::PipelineBuild, 9_000_000, false, 8, identity);
    assert_eq!(perf.slow.len(), 5);
    assert_eq!(
        perf.slow.iter().map(|event| event.ns).collect::<Vec<_>>(),
        vec![9_000_000, 6_000_000, 5_000_000, 4_000_000, 3_000_000]
    );
    assert!(!perf.slow[0].success);
}

#[test]
#[cfg(perf_tracking)]
fn empty_windows_reset_remainders() {
    let mut perf = CompilationPerf::new();
    perf.finish_frame(90_000_000, 80_000_000, 100_000_000);
    let mut output = String::new();
    perf.append_window(&mut output, 1);
    assert!(output.is_empty());
    perf.record_enabled(Kind::Depth, 1_000_000, true, 2, identity);
    perf.finish_frame(2_000_000, 3_000_000, 4_000_000);
    assert_eq!(perf.window[Kind::ResolveOther as usize].ns, 2_000_000);
    assert_eq!(perf.window[Kind::PipelineOther as usize].ns, 2_000_000);
}

#[test]
#[cfg(not(perf_tracking))]
fn disabled_perf_has_no_storage_or_identity_work() {
    let mut perf = super::CompilationPerf::new();
    assert_eq!(core::mem::size_of_val(&perf), 0);
    perf.record(super::Kind::Library, u64::MAX, false, 1, || {
        panic!("disabled identity")
    });
    perf.shader_parts(&mtld3d_shared::perf::ShaderTimings::new(), false, 1, || {
        panic!("disabled identity")
    });
}

#[test]
#[cfg(perf_tracking)]
fn shader_failure_counts_stop_at_the_failed_phase() {
    let mut perf = CompilationPerf::new();
    for (library_ns, function_ns, success) in [
        (0, 0, false),
        (3_000_000, 0, false),
        (4_000_000, 2_000_000, false),
        (5_000_000, 1_000_000, true),
    ] {
        perf.shader_parts_enabled(
            &mtld3d_shared::perf::ShaderTimings {
                preparation_ns: 10,
                library_ns,
                function_ns,
            },
            success,
            1,
            identity,
        );
    }
    perf.shader_parts_enabled(
        &mtld3d_shared::perf::ShaderTimings::new(),
        false,
        1,
        identity,
    );
    assert_eq!(perf.frame[Kind::Library as usize].calls, 3);
    assert_eq!(perf.frame[Kind::Library as usize].failures, 1);
    assert_eq!(perf.frame[Kind::Function as usize].calls, 2);
    assert_eq!(perf.frame[Kind::Function as usize].failures, 1);
    assert_eq!(perf.frame[Kind::ShaderPreparation as usize].calls, 4);
    assert_eq!(perf.frame[Kind::ShaderPreparation as usize].failures, 1);
    perf.log_startup_enabled(7);
    assert!(perf.frame.iter().all(|metric| metric.calls == 0));
    assert!(perf.window.iter().all(|metric| metric.calls == 0));
    assert!(perf.slow.is_empty());
}
