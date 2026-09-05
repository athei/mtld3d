//! Unit tests for the Metal-validation gate and the GPU-hang stop.
//!
//! The recogniser picks the layer's error messages out of a subtest's stderr,
//! keeps the detail lines under each one, collapses volatile numbers so a
//! repeated message reports once, and leaves ordinary Wine and Metal chatter
//! alone. The gate then fails a leg for any message that survives: after the
//! warning filter there is no benign one left.
//!
//! The hang tests let `/bin/sh` play Wine and a script play the test binary:
//! the driver's line has to stop the subtest within seconds, not at its
//! budget, and has to mark the run so its counts never reach a baseline.

use std::{
    fs,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use super::{
    is_gpu_hang_line, normalize_numbers, run_subtest, validation_errors, validation_gate_failed,
};
use crate::model::{Arch, Gpu, Leg, Subtest, Variant};

/// The two draw errors the mismatched-sampler-type case logged, verbatim.
const MISMATCHED_SAMPLER: &str = "\
0000:err:d3d9: something unrelated\n\
2026-08-28 11:59:31.778 wine[27403:71732] Draw Errors Validation\n\
Fragment Function(mtld3d_ps_sm3_29b0ab41fc8da9b1): incorrect type of texture (MTLTextureType2D) bound at Texture binding at index 0 (expect MTLTextureType3D) for s0[0].\n\
Fragment Function(mtld3d_ps_sm3_fab23e4e59af0b3b): incorrect type of texture (MTLTextureType3D) bound at Texture binding at index 0 (expect MTLTextureType2D) for s0[0].\n";

/// The rejected sampler descriptor a paravirtual GPU logged, verbatim.
const REJECTED_SAMPLER_DESCRIPTOR: &str = "\
2026-08-28 11:59:28.959 wine[27403:71131] Metal API Validation Enabled\n\
2026-08-28 11:59:31.778 wine[27403:71732] Sampler Descriptor Validation\n\
MTLSamplerAddressModeMirrorClampToEdge is not supported on this device\n";

#[test]
fn draw_errors_are_detail_of_one_message() {
    let seen = validation_errors(MISMATCHED_SAMPLER);
    assert_eq!(seen.len(), 1, "one report, not one per line: {seen:?}");
    let msg = seen.iter().next().unwrap();
    assert_eq!(msg.lines().count(), 3, "banner and both draw errors: {msg}");
    assert!(validation_gate_failed(seen.len()));
}

#[test]
fn the_rejected_property_is_kept() {
    let seen = validation_errors(REJECTED_SAMPLER_DESCRIPTOR);
    assert_eq!(seen.len(), 1, "{seen:?}");
    let msg = seen.iter().next().unwrap();
    assert!(
        msg.contains("Sampler Descriptor Validation")
            && msg.contains("MTLSamplerAddressModeMirrorClampToEdge is not supported"),
        "the header alone does not say which property was rejected: {msg}"
    );
}

#[test]
fn a_message_ends_at_the_next_writer() {
    let stderr = "\
2026-08-28 11:59:31.778 wine[27403:71732] Sampler Descriptor Validation\n\
MTLSamplerAddressModeMirrorClampToEdge is not supported on this device\n\
0000:err:d3d9: unrelated, and not this message's detail\n\
2026-08-28 11:59:32.001 wine[27403:71732] Buffer Validation\n\
Insufficient buffer size at buffer binding at index 2\n";
    let seen = validation_errors(stderr);
    assert_eq!(seen.len(), 2, "{seen:?}");
    assert!(
        seen.iter().all(|msg| !msg.contains("unrelated")),
        "a Wine channel line was swallowed as detail: {seen:?}"
    );
}

#[test]
fn one_error_repeated_per_draw_reports_once() {
    let line = "2026-08-28 11:59:31.778 wine[27403:71732] Buffer Validation: \
                Insufficient buffer size at buffer binding at index 2";
    let stderr = format!("{line} 0x600002a1c000\n{line} 0x600002b40480\n");
    assert_eq!(validation_errors(&stderr).len(), 1);
}

#[test]
fn ordinary_output_logs_nothing() {
    let stderr = "\
        d3d9_test.exe: 1234 tests executed (0 marked as todo, 0 failures), 3 skipped.\n\
        visual.c:9100: Test failed: Got unexpected colour 0x00000000.\n\
        Metal API Validation Enabled\n";
    assert!(validation_errors(stderr).is_empty());
    assert!(!validation_gate_failed(0));
}

#[test]
fn numbers_and_addresses_collapse() {
    assert_eq!(
        normalize_numbers("texture at index 3 (0xdeadbeef) must be <= 16"),
        "texture at index N (0xN) must be <= N"
    );
}

/// The two lines the paravirtual GPU's driver printed on the Intel CI image, verbatim.
const DRIVER_HANG: &str = "2026-09-05 05:23:06.572 wine[86111:201735] Execution of the command \
    buffer was aborted due to an error during execution. Caused GPU Hang Error \
    (00000003:kIOAccelCommandBufferCallbackErrorHang)";
const DRIVER_IGNORED: &str = "2026-09-05 05:23:06.667 wine[86111:201735] Execution of the \
    command buffer was aborted due to an error during execution. Ignored (for causing \
    prior/excessive GPU errors) (00000004:kIOAccelCommandBufferCallbackErrorSubmissionsIgnored)";

#[test]
fn the_drivers_hang_lines_are_recognised_and_nothing_else_is() {
    assert!(is_gpu_hang_line(DRIVER_HANG));
    assert!(is_gpu_hang_line(DRIVER_IGNORED));
    for line in [
        "2026-08-28 11:59:28.959 wine[27403:71131] Metal API Validation Enabled",
        "visual.c:9100: Test failed: Got unexpected colour 0x00000000.",
        "0000:err:d3d9: GPU hang is not what this line reports",
    ] {
        assert!(!is_gpu_hang_line(line), "{line}");
    }
}

/// A script in a fresh directory under the temp dir, run as `sh <script> <subtest>`.
///
/// `/bin/sh` plays Wine and the script plays `d3d9_test.exe`; `run_subtest`
/// only asks that the exe be a file. A script that has to outlive its lines
/// sleeps in a child of its own, which holds the pipes past the shell: the
/// kill has to take the group, or the run would wait for that child.
fn script(name: &str, body: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "mtld3d-conformance-run-{}-{name}",
        std::process::id()
    ));
    fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("d3d9_test.sh");
    fs::write(&path, body).expect("script");
    path
}

const LEG: Leg = Leg {
    arch: Arch::I686,
    variant: Variant::Native,
    gpu: Gpu::Apple,
};

/// The bound the spawn tests hold the runner to: well under any subtest budget.
const PROMPT: Duration = Duration::from_secs(5);

#[test]
fn a_subtest_that_hangs_the_gpu_is_stopped_at_once() {
    let exe = script(
        "hang",
        &format!("echo 'visual.c:100: Test failed: x'\necho '{DRIVER_HANG}' >&2\nsleep 30\n"),
    );
    let started = Instant::now();
    let run = run_subtest(Path::new("/bin/sh"), &exe, LEG, Subtest::Visual).expect("spawn sh");
    let elapsed = started.elapsed();
    assert!(run.gpu_hang);
    assert!(run.result.crash, "a leg cut short is not a clean count");
    assert!(
        elapsed < PROMPT,
        "took {elapsed:?}: the runner waited instead of killing"
    );
}

#[test]
fn a_hang_line_before_a_clean_exit_still_counts() {
    let exe = script(
        "hang-exit",
        &format!(
            "echo '{DRIVER_IGNORED}' >&2\necho 'visual: 10 tests executed (0 marked as todo, 0 failures), 0 skipped.'\nexit 0\n"
        ),
    );
    let run = run_subtest(Path::new("/bin/sh"), &exe, LEG, Subtest::Visual).expect("spawn sh");
    assert!(run.gpu_hang);
    assert!(
        run.result.crash,
        "a process that survived its hang still read zeros; its counts must not reach the baseline"
    );
}
