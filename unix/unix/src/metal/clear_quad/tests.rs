//! Unit tests for the clear-quad shader sources, fixed and generated.
//!
//! Compiling `CLEAR_QUAD_MSL` on the host `MTLDevice` and resolving its entry points
//! catches a typo or a rename before a mid-pass `Clear` does. The generated multi-target
//! variants carry the real risk: Metal rejects a fragment function that writes a colour
//! slot the pass leaves unbound, so every present mask from 1 to 7 is generated, checked
//! slot by slot, and compiled.

use super::*;

/// Smoke test: the inline MSL compiles on the host `MTLDevice`.
///
/// Catches shader typos at unit-test time so a regression never ships
/// to the game first.
#[test]
fn clear_quad_msl_compiles_under_metal() {
    use objc2_metal::MTLCreateSystemDefaultDevice;
    let Some(device) = MTLCreateSystemDefaultDevice() else {
        eprintln!("MTLCreateSystemDefaultDevice returned nil — skipping");
        return;
    };
    let source = NSString::from_str(CLEAR_QUAD_MSL);
    let options = MTLCompileOptions::new();
    options.setLanguageVersion(MTLLanguageVersion::Version2_4);
    options.setMathMode(MTLMathMode::Fast);
    let library = device
        .newLibraryWithSource_options_error(&source, Some(&options))
        .expect("clear-quad MSL must compile");
    let _vs = library
        .newFunctionWithName(&NSString::from_str(VS_NAME))
        .expect("VS entry must exist");
    let _ps = library
        .newFunctionWithName(&NSString::from_str(PS_COLOR_NAME))
        .expect("PS color entry must exist");
}

/// The generated multi-target colour function compiles for every present mask.
#[test]
fn multi_target_clear_quad_msl_compiles_under_metal() {
    use objc2_metal::MTLCreateSystemDefaultDevice;
    let Some(device) = MTLCreateSystemDefaultDevice() else {
        eprintln!("MTLCreateSystemDefaultDevice returned nil — skipping");
        return;
    };
    for mask in 1..=7u8 {
        let msl = color_ps_source(mask);
        assert!(msl.contains("[[color(0)]]"), "{msl}");
        assert_eq!(msl.contains("[[color(1)]]"), mask & 0b001 != 0, "{msl}");
        assert_eq!(msl.contains("[[color(3)]]"), mask & 0b100 != 0, "{msl}");
        let source = NSString::from_str(&msl);
        let options = MTLCompileOptions::new();
        options.setLanguageVersion(MTLLanguageVersion::Version2_4);
        let library = device
            .newLibraryWithSource_options_error(&source, Some(&options))
            .unwrap_or_else(|e| panic!("mask {mask:#x} must compile: {e}\n{msl}"));
        assert!(
            library
                .newFunctionWithName(&NSString::from_str(&color_ps_name(mask)))
                .is_some(),
            "entry for mask {mask:#x}"
        );
    }
}
