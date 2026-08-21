use super::{Aggregate, SiteFlap, render};
use crate::model::{Arch, Site, Subtest};

fn site(line: u32) -> Site {
    Site {
        file: "device.c".to_owned(),
        line,
    }
}

/// A flap from a per-run count series, where `0` means "did not fire".
fn flap_from(per_run: &[u32]) -> SiteFlap {
    let mut f = SiteFlap::default();
    for &c in per_run {
        if c > 0 {
            f.fired_runs += 1;
            *f.counts.entry(c).or_insert(0) += 1;
        }
    }
    f
}

#[test]
fn stable_site_fires_every_run_at_constant_count() {
    let flap = flap_from(&[3, 3, 3, 3, 3]);
    assert!(flap.is_stable(5));
    assert_eq!(flap.distribution(5), "3×5");
}

#[test]
fn flapping_site_is_unstable_and_shows_zeros() {
    let flap = flap_from(&[0, 1, 0, 1, 1]); // fired in 3 of 5 runs
    assert!(!flap.is_stable(5));
    assert_eq!(flap.distribution(5), "0×2, 1×3");
}

#[test]
fn report_separates_flapping_from_stable_and_lists_upstream_flaky() {
    let mut agg = Aggregate {
        runs: 5,
        ..Default::default()
    };
    agg.sites.insert(site(5368), flap_from(&[0, 1, 0, 1, 1]));
    agg.sites.insert(site(6516), flap_from(&[3, 3, 3, 3, 3]));
    agg.flaky_marked.insert(site(5406), 37);

    let text = render(Arch::I686, Subtest::Device, &agg);
    assert!(
        text.contains("device.c:5368") && text.contains("<- FLAPS"),
        "{text}"
    );
    assert!(text.contains("1 stable site"), "{text}");
    assert!(text.contains("device.c:6516"), "{text}");
    assert!(
        text.contains("upstream flaky-marked") && text.contains("device.c:5406×37"),
        "{text}"
    );
}
