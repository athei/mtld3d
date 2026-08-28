//! Unit tests for the Metal-validation reporting and its gate.
//!
//! The recogniser picks the layer's error messages out of a subtest's stderr,
//! keeps the detail lines under each one, collapses volatile numbers so a
//! repeated message reports once, and leaves ordinary Wine and Metal chatter
//! alone. The gate then fails a leg for any message that survives: after the
//! warning filter there is no benign one left.

use super::{normalize_numbers, validation_errors, validation_gate_failed};

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
