//! Unit tests for the present pass shader library and its pipelines.
//!
//! Every test needs a real Metal device and skips when there is none. The
//! first compiles the embedded MSL and resolves the shared vertex stage
//! plus all three fragment entry points (copy, HDR pass-through,
//! BT.2446), so a typo or a rename landed on one side only fails here
//! instead of at the first present. The second builds every pipeline for
//! the drawable format its route writes, where a format that disagrees
//! with the fragment output would otherwise surface as a black frame. The
//! third does the same for the gamma twins, which the end-to-end suite
//! never builds: it creates windowed devices, and only a fullscreen device
//! applies a ramp.

use super::*;

/// Smoke test: the present MSL compiles and every entry point resolves.
///
/// Catches shader typos at unit-test time so a regression never ships to
/// the game first. The entry-point lookups are what would fail if a
/// rename landed on one side only.
#[test]
fn present_msl_compiles_under_metal() {
    use objc2_metal::MTLCreateSystemDefaultDevice;
    let Some(device) = MTLCreateSystemDefaultDevice() else {
        eprintln!("MTLCreateSystemDefaultDevice returned nil — skipping");
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
        eprintln!("MTLCreateSystemDefaultDevice returned nil — skipping");
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
