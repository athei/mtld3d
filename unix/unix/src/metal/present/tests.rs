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
//! applies a ramp. The last three need no device: they check the BT.2446
//! constants folded into the MSL against their formula, `hdr_uniforms` against
//! the headroom × 203 rule, and the colour matrices against their inverses.
//! The tone map's output is checked on the GPU in `command/tests.rs`.

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
        "the six frame and cursor pipelines (copy, pass-through, BT.2446) must all build"
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

/// The BT.2446 SDR constants match their formula.
///
/// `P_SDR` and `LOG_P_SDR` are folded into the MSL from `L_SDR`, so nothing
/// else ties the three together.
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
}

/// The uniforms put white at the headroom: `L_hdr` is the headroom × 203, from 203 to 10000.
///
/// The MSL divides by its own `REFERENCE_WHITE_NITS`, which has to be the 203 the
/// CPU multiplies by. A headroom under 1 or NaN is taken as 1, and PQ's
/// 10000 nits stop a headroom above 49, an infinite one included. The three
/// derived terms follow BT.2446's formula at that `L_hdr`.
#[test]
fn hdr_uniforms_put_white_at_the_headroom() {
    let white = msl_float_constant("REFERENCE_WHITE_NITS");
    assert!(
        (white - f64::from(REFERENCE_WHITE_NITS)).abs() < 1e-9 && (white - 203.0).abs() < 1e-9,
        "the MSL's REFERENCE_WHITE_NITS is {white}, the CPU's {REFERENCE_WHITE_NITS}, BT.2408's 203"
    );
    for (headroom, l_hdr) in [
        (1.0, 203.0_f64),
        (1000.0 / 203.0, 1000.0),
        (15.0, 3045.0),
        (0.5, 203.0),
        (f32::NAN, 203.0),
        (60.0, 10000.0),
        (f32::INFINITY, 10000.0),
    ] {
        let [nits, p_hdr, log2_p_hdr, inv_p_minus_one] = hdr_uniforms(headroom).map(f64::from);
        let p = 32.0_f64.mul_add((l_hdr / 10000.0).powf(1.0 / 2.4), 1.0);
        for (name, read, wanted) in [
            ("l_hdr_nits", nits, l_hdr),
            ("p_hdr", p_hdr, p),
            ("log2_p_hdr", log2_p_hdr, p.log2()),
            ("inv_p_minus_one", inv_p_minus_one, 1.0 / (p - 1.0)),
        ] {
            assert!(
                (read - wanted).abs() <= wanted.abs() * 1e-5,
                "{name} at a headroom of {headroom} is {read}, the formula gives {wanted}"
            );
        }
    }
}

/// The columns of a `constant float3x3 <name> = float3x3(...)` block in the present MSL.
fn msl_matrix(name: &str) -> [[f64; 3]; 3] {
    let start = format!("constant float3x3 {name} = float3x3(");
    let body = PRESENT_MSL
        .split_once(start.as_str())
        .and_then(|(_, rest)| rest.split_once(");"))
        .map_or_else(
            || panic!("present MSL must declare {name}"),
            |(body, _)| body,
        );
    let numbers: Vec<f64> = body
        .lines()
        .map(|line| line.split_once("//").map_or(line, |(code, _)| code))
        .flat_map(|code| code.split(['(', ')', ',']))
        .map(str::trim)
        .filter(|token| !token.is_empty() && *token != "float3")
        .map(|token| {
            token
                .trim_end_matches('f')
                .parse()
                .unwrap_or_else(|_| panic!("{name} holds {token}, not a float literal"))
        })
        .collect();
    assert_eq!(numbers.len(), 9, "{name} must hold nine literals");
    core::array::from_fn(|column| core::array::from_fn(|row| numbers[column * 3 + row]))
}

/// The present's colour matrices and their inverses round-trip to the identity.
///
/// `M_LMS_TO_BT709` and `M_ICTCP_TO_LMS_PQ` are the inverses of the forward
/// matrices as the MSL writes them, so a pixel the tone map leaves at its SDR
/// luminance comes back with its SDR value, SDR white as 1.0. Both products,
/// in both orders, are the identity within 1e-6 (the literals give 1e-9).
#[test]
fn present_matrix_inverses_round_trip() {
    for (forward, inverse) in [
        ("M_BT709_TO_LMS", "M_LMS_TO_BT709"),
        ("M_LMS_PQ_TO_ICTCP", "M_ICTCP_TO_LMS_PQ"),
    ] {
        let (a, b) = (msl_matrix(forward), msl_matrix(inverse));
        for (left, right, order) in [
            (a, b, "forward after inverse"),
            (b, a, "inverse after forward"),
        ] {
            // Column-major: column `column` of `left * right` is `left` applied
            // to column `column` of `right`.
            for (column, right_column) in right.iter().enumerate() {
                let product: [f64; 3] = core::array::from_fn(|row| {
                    left.iter()
                        .zip(right_column)
                        .map(|(left_column, weight)| left_column[row] * weight)
                        .sum()
                });
                for (row, value) in product.into_iter().enumerate() {
                    let identity = if row == column { 1.0 } else { 0.0 };
                    assert!(
                        (value - identity).abs() < 1e-6,
                        "{forward} and {inverse}, {order}: ({row}, {column}) is {value}"
                    );
                }
            }
        }
    }
}
