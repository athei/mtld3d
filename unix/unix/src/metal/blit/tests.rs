//! Unit tests for the blit-quad shader source.
//!
//! The MSL is a string constant compiled at runtime, so a typo in it or a rename of an
//! entry point would first surface as a scaling `StretchRect` that silently stopped
//! working. Compiling `BLIT_MSL` on the host `MTLDevice` with the options the pipeline
//! builder uses, then resolving both entry-point names, moves that failure to the test.

use super::*;

/// Smoke test: the inline MSL compiles on the host `MTLDevice`.
///
/// Catches shader typos at unit-test time so a regression never ships to
/// the game first.
#[test]
fn blit_quad_msl_compiles_under_metal() {
    use objc2_metal::MTLCreateSystemDefaultDevice;
    let Some(device) = MTLCreateSystemDefaultDevice() else {
        eprintln!("MTLCreateSystemDefaultDevice returned nil — skipping");
        return;
    };
    let source = NSString::from_str(BLIT_MSL);
    let options = MTLCompileOptions::new();
    options.setLanguageVersion(MTLLanguageVersion::Version2_4);
    options.setMathMode(MTLMathMode::Fast);
    let library = device
        .newLibraryWithSource_options_error(&source, Some(&options))
        .expect("blit-quad MSL must compile");
    let _vs = library
        .newFunctionWithName(&NSString::from_str(VS_NAME))
        .expect("VS entry must exist");
    let _ps = library
        .newFunctionWithName(&NSString::from_str(PS_NAME))
        .expect("PS entry must exist");
}
