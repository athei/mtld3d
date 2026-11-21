//! Unit tests for the baseline text (de)serializer.
//!
//! A round-trip over a populated model pins canonical output: parsing then
//! re-serializing is byte-identical, so a re-baseline shows a minimal diff, and
//! the checked-in `baseline.txt` is held to the same round-trip. One
//! test lifts the `# Wine:` comment line into `wine_version`. Three error paths
//! follow: a site line before any header, a stale per-site `class=` token
//! (classes moved to CONFORMANCE.md), and a header crash flag that is not `0` or `1`.

use std::collections::BTreeMap;

use super::{Arch, Baseline, Leg, Site, Subtest, SubtestBaseline, Variant};

fn sample() -> Baseline {
    let mut device = SubtestBaseline {
        crash: true,
        sites: BTreeMap::new(),
    };
    device.sites.insert(
        Site {
            file: "device.c".to_owned(),
            line: 125,
        },
        37,
    );
    device.sites.insert(
        Site {
            file: "device.c".to_owned(),
            line: 792,
        },
        18,
    );
    let mut d3d9ex = SubtestBaseline {
        crash: false,
        sites: BTreeMap::new(),
    };
    d3d9ex.sites.insert(
        Site {
            file: "d3d9ex.c".to_owned(),
            line: 55,
        },
        1,
    );
    let mut entries = BTreeMap::new();
    entries.insert(
        (
            Leg {
                arch: Arch::I686Arm64,
                variant: Variant::Native,
            },
            Subtest::Device,
        ),
        device,
    );
    entries.insert(
        (
            Leg {
                arch: Arch::Arm64x,
                variant: Variant::Scale,
            },
            Subtest::D3d9Ex,
        ),
        d3d9ex,
    );
    Baseline {
        wine_version: "wine-11.0".to_owned(),
        entries,
    }
}

#[test]
fn model_text_roundtrip_is_canonical() {
    let baseline = sample();
    let text = baseline.to_text();
    assert_eq!(Baseline::from_text(&text).unwrap(), baseline);
    // Re-serializing parsed canonical text is byte-identical.
    assert_eq!(Baseline::from_text(&text).unwrap().to_text(), text);
}

#[test]
fn parse_reads_wine_version() {
    let baseline = Baseline::from_text("# Wine: wine-11.0-6\n[arm64x/device] crash=0\n").unwrap();
    assert_eq!(baseline.wine_version, "wine-11.0-6");
}

#[test]
fn parse_rejects_site_before_header() {
    let err = Baseline::from_text("  device.c:1 count=1\n").unwrap_err();
    assert!(err.contains("before any header"), "{err}");
}

#[test]
fn parse_rejects_legacy_class_token() {
    let err = Baseline::from_text("[arm64x/device] crash=0\n  device.c:1 count=1 class=real\n")
        .unwrap_err();
    assert!(err.contains("classes moved to CONFORMANCE.md"), "{err}");
}

#[test]
fn parse_rejects_bad_crash_token() {
    let err = Baseline::from_text("[arm64x/device] crash=maybe\n").unwrap_err();
    assert!(err.contains("crash=0|1"), "{err}");
}

/// The checked-in file is what the writer emits, so a re-record that changes no count is no diff.
#[test]
fn checked_in_baseline_is_canonical() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("baseline.txt");
    let text = std::fs::read_to_string(path).expect("baseline.txt must be readable");
    let baseline = Baseline::from_text(&text).expect("baseline.txt must parse");
    assert_eq!(baseline.to_text(), text);
}

#[test]
fn leg_names_round_trip_with_implicit_native() {
    for (text, arch, variant) in [
        ("arm64x", Arch::Arm64x, Variant::Native),
        ("arm64x+scale", Arch::Arm64x, Variant::Scale),
        ("i686-arm64", Arch::I686Arm64, Variant::Native),
        ("i686-arm64+scale", Arch::I686Arm64, Variant::Scale),
    ] {
        let leg = Leg { arch, variant };
        assert_eq!(leg.to_string(), text);
        assert_eq!(text.parse::<Leg>().unwrap(), leg);
    }
    assert!("i686".parse::<Leg>().is_err());
    assert!("x86_64".parse::<Leg>().is_err());
    assert!("arm64x+intel".parse::<Leg>().is_err());
    assert!("arm64x@mac2".parse::<Leg>().is_err());
}

#[test]
fn every_leg_is_listed_once_in_output_order() {
    let mut seen = Leg::ALL.to_vec();
    seen.dedup();
    assert_eq!(seen.len(), Leg::ALL.len());
    assert!(Leg::ALL.windows(2).all(|pair| pair[0] < pair[1]));
    let names: Vec<String> = Leg::ALL.iter().map(ToString::to_string).collect();
    assert_eq!(
        names,
        ["i686-arm64", "i686-arm64+scale", "arm64x", "arm64x+scale"]
    );
}
