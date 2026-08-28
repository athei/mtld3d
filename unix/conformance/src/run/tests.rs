//! Unit tests for the Metal-validation reporting and its gate.
//!
//! The recogniser picks the layer's error lines out of a subtest's stderr,
//! collapses volatile numbers so a repeated error reports once, and leaves
//! ordinary Wine and Metal chatter alone. The gate then fails a leg for any
//! line that survives: after the warning filter there is no benign one left.

use super::{normalize_numbers, validation_errors, validation_gate_failed};

/// Two draw-error lines the mismatched-sampler-type case used to log, verbatim.
const MISMATCHED_SAMPLER: &str = "\
0000:err:d3d9: something unrelated\n\
Draw Errors Validation\n\
Fragment Function(mtld3d_ps_sm3_29b0ab41fc8da9b1): incorrect type of texture (MTLTextureType2D) bound at Texture binding at index 0 (expect MTLTextureType3D) for s0[0].\n\
Fragment Function(mtld3d_ps_sm3_fab23e4e59af0b3b): incorrect type of texture (MTLTextureType3D) bound at Texture binding at index 0 (expect MTLTextureType2D) for s0[0].\n";

#[test]
fn draw_error_lines_are_collected() {
    let seen = validation_errors(MISMATCHED_SAMPLER);
    assert_eq!(seen.len(), 3, "the banner and both draw errors: {seen:?}");
    assert!(validation_gate_failed(seen.len()));
}

#[test]
fn one_error_repeated_per_draw_reports_once() {
    let line = "Buffer Validation: Insufficient buffer size at buffer binding at index 2";
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
