//! Unit tests for the present pass shader library and its pipelines.
//!
//! The first three tests need a real Metal device and skip when there is none.
//! The first compiles the embedded MSL and resolves the shared vertex stage
//! and the three frame fragment entry points (copy, HDR pass-through,
//! BT.2446), so a typo or a rename landed on one side only fails here
//! instead of at the first present. The second builds every pipeline for
//! the drawable format its route writes, where a format that disagrees
//! with the fragment output would otherwise surface as a black frame. The
//! third does the same for the gamma twins, which the end-to-end suite
//! never builds: it creates windowed devices, and only a fullscreen device
//! applies a ramp. The fourth checks the BT.2446 constants folded into the MSL
//! against their formula and against `hdr_uniforms`, and needs no device.

use super::*;

/// Smoke test: the present MSL compiles and the frame pass's entry points resolve.
///
/// Catches shader typos at unit-test time so a regression never ships to
/// the game first. The entry-point lookups are what would fail if a
/// rename landed on one side only. They cover the vertex stage and the
/// three frame fragment stages; the cursor stages and the gamma twins
/// resolve in the pipeline builds of the next two tests, and the nearest
/// readback copy in the exact-copy test of `command/tests.rs`.
#[test]
fn present_msl_compiles_under_metal() {
    use objc2_metal::MTLCreateSystemDefaultDevice;
    let Some(device) = MTLCreateSystemDefaultDevice() else {
        eprintln!("MTLCreateSystemDefaultDevice returned nil, skipping");
        return;
    };
    let source = NSString::from_str(PRESENT_MSL);
    let options = MTLCompileOptions::new();
    options.setLanguageVersion(MTLLanguageVersion::Version2_4);
    options.setMathMode(MTLMathMode::Fast);
    let library = device
        .newLibraryWithSource_options_error(&source, Some(&options))
        .expect("present MSL must compile");
    for name in [
        "mtld3d_present_vs",
        "mtld3d_present_ps_copy",
        "mtld3d_present_ps_hdr_passthrough",
        "mtld3d_present_ps_hdr_bt2446",
    ] {
        assert!(
            library
                .newFunctionWithName(&NSString::from_str(name))
                .is_some(),
            "entry point {name} must exist"
        );
    }
}

/// Every pipeline builds against the drawable format its route writes.
///
/// A pixel format that disagrees with the fragment stage's output is a
/// runtime pipeline-creation failure, which without this test would
/// first show up as a black frame on a user's machine.
#[test]
fn present_pipelines_build_for_both_drawable_formats() {
    use objc2_metal::MTLCreateSystemDefaultDevice;
    let Some(device) = MTLCreateSystemDefaultDevice() else {
        eprintln!("MTLCreateSystemDefaultDevice returned nil, skipping");
        return;
    };
    assert!(
        create(&device).is_some(),
        "copy, pass-through and BT.2446 pipelines must all build"
    );
}

/// Every gamma twin builds, for the drawable format its stage writes.
///
/// The twins are compiled lazily, on the first present that applies a ramp,
/// so a pipeline that fails to build would otherwise surface only in a
/// fullscreen game with its brightness slider moved.
#[test]
fn gamma_pipelines_build_for_every_stage() {
    use objc2_metal::MTLCreateSystemDefaultDevice;
    let Some(device) = MTLCreateSystemDefaultDevice() else {
        eprintln!("MTLCreateSystemDefaultDevice returned nil, skipping");
        return;
    };
    for stage in [
        GammaStage::Copy,
        GammaStage::Passthrough,
        GammaStage::Bt2446,
        GammaStage::CursorCopy,
        GammaStage::CursorPassthrough,
        GammaStage::CursorBt2446,
    ] {
        let (name, _) = stage.function();
        assert!(
            ensure_gamma_pipeline(&device, stage).is_some(),
            "the {name} pipeline must build"
        );
    }
}

/// The literal of a `constant float <name> = <literal>;` line in the present MSL.
fn msl_float_constant(name: &str) -> f64 {
    let prefix = format!("constant float {name} = ");
    let rest = PRESENT_MSL
        .lines()
        .find_map(|line| line.strip_prefix(prefix.as_str()))
        .unwrap_or_else(|| panic!("present MSL must declare {name}"));
    let literal = rest
        .split_once(';')
        .map_or(rest, |(literal, _)| literal)
        .trim();
    literal
        .trim_end_matches('f')
        .parse()
        .unwrap_or_else(|_| panic!("{name} must be a float literal, found {literal}"))
}

/// The BT.2446 SDR constants match their formula and the HDR side at peak 1.0.
///
/// `P_SDR` and `LOG_P_SDR` are folded into the MSL while `hdr_uniforms` computes
/// the HDR side on the CPU, so nothing else ties the two together. A peak of 1.0
/// is the SDR reference white, where `p_hdr` has to equal `P_SDR`.
#[test]
fn bt2446_sdr_constants_match_the_formula() {
    let l_sdr = msl_float_constant("L_SDR");
    let p_sdr = msl_float_constant("P_SDR");
    let log_p_sdr = msl_float_constant("LOG_P_SDR");
    let expected = 32.0_f64.mul_add((l_sdr / 10000.0).powf(1.0 / 2.4), 1.0);
    assert!(
        (p_sdr - expected).abs() < 1e-6,
        "P_SDR is {p_sdr}, the formula gives {expected} at L_SDR = {l_sdr}"
    );
    assert!(
        (log_p_sdr - expected.ln()).abs() < 1e-6,
        "LOG_P_SDR is {log_p_sdr}, ln(P_SDR) is {}",
        expected.ln()
    );
    let hdr_at_unit_peak = f64::from(hdr_uniforms(1.0)[1]);
    assert!(
        (hdr_at_unit_peak - p_sdr).abs() < 1e-5,
        "p_hdr at peak 1.0 is {hdr_at_unit_peak}, P_SDR is {p_sdr}"
    );
}
