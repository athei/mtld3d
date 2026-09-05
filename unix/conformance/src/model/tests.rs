//! Unit tests for the baseline text (de)serializer.
//!
//! A round-trip over a populated model pins canonical output: parsing then
//! re-serializing is byte-identical, so a re-baseline shows a minimal diff. One
//! test lifts the `# Wine:` comment line into `wine_version`. Three error paths
//! follow: a site line before any header, a stale per-site `class=` token
//! (classes moved to CONFORMANCE.md), and a header crash flag that is not `0` or `1`.

use std::collections::BTreeMap;

use super::{Arch, Baseline, Gpu, Leg, Site, Subtest, SubtestBaseline, Variant};

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
                arch: Arch::I686,
                variant: Variant::Native,
                gpu: Gpu::Apple,
            },
            Subtest::Device,
        ),
        device,
    );
    entries.insert(
        (
            Leg {
                arch: Arch::X64,
                variant: Variant::Intel,
                gpu: Gpu::Mac2,
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
    let baseline = Baseline::from_text("# Wine: wine-11.0-6\n[i686/device] crash=0\n").unwrap();
    assert_eq!(baseline.wine_version, "wine-11.0-6");
}

#[test]
fn parse_rejects_site_before_header() {
    let err = Baseline::from_text("  device.c:1 count=1\n").unwrap_err();
    assert!(err.contains("before any header"), "{err}");
}

#[test]
fn parse_rejects_legacy_class_token() {
    let err = Baseline::from_text("[i686/device] crash=0\n  device.c:1 count=1 class=real\n")
        .unwrap_err();
    assert!(err.contains("classes moved to CONFORMANCE.md"), "{err}");
}

#[test]
fn parse_rejects_bad_crash_token() {
    let err = Baseline::from_text("[i686/device] crash=maybe\n").unwrap_err();
    assert!(err.contains("crash=0|1"), "{err}");
}

#[test]
fn leg_names_round_trip_with_implicit_native_and_apple() {
    for (text, arch, variant, gpu) in [
        ("i686", Arch::I686, Variant::Native, Gpu::Apple),
        ("x86_64+intel", Arch::X64, Variant::Intel, Gpu::Apple),
        ("i686@mac2", Arch::I686, Variant::Native, Gpu::Mac2),
        ("x86_64+intel@mac2", Arch::X64, Variant::Intel, Gpu::Mac2),
    ] {
        let leg = Leg { arch, variant, gpu };
        assert_eq!(leg.to_string(), text);
        assert_eq!(text.parse::<Leg>().unwrap(), leg);
    }
    assert!("i686@apple2".parse::<Leg>().is_err());
    assert!("i686@mac2+intel".parse::<Leg>().is_err());
}

#[test]
fn every_leg_is_listed_once_in_output_order() {
    let mut seen = Leg::ALL.to_vec();
    seen.dedup();
    assert_eq!(seen.len(), Leg::ALL.len());
    assert!(Leg::ALL.windows(2).all(|pair| pair[0] < pair[1]));
    assert_eq!(Leg::ALL[0].to_string(), "i686");
    assert_eq!(Leg::ALL[3].to_string(), "i686+intel@mac2");
    assert_eq!(Leg::ALL[7].to_string(), "x86_64+intel@mac2");
}
