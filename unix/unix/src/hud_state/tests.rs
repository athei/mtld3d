use std::sync::atomic::Ordering;

use mtld3d_core::hud_stats::{HudRowLatch, SampleGate};

use super::{DOMAIN, HudReporter, OWNER, ReporterState, domains_with_ours, report_attached};

#[test]
fn an_unset_or_empty_list_shows_only_ours() {
    assert_eq!(domains_with_ours(None).as_deref(), Some(DOMAIN));
    assert_eq!(domains_with_ours(Some("")).as_deref(), Some(DOMAIN));
    assert_eq!(domains_with_ours(Some(" \"\" ")).as_deref(), Some(DOMAIN));
}

#[test]
fn a_user_list_keeps_its_domains_and_gains_ours() {
    assert_eq!(
        domains_with_ours(Some("com.example.game")).as_deref(),
        Some("com.example.game,mtld3d")
    );
    assert_eq!(
        domains_with_ours(Some("\"com.example.a, com.example.b\"")).as_deref(),
        Some("com.example.a, com.example.b,mtld3d")
    );
}

#[test]
fn a_list_naming_the_old_upper_case_domain_still_gains_ours() {
    assert_eq!(
        domains_with_ours(Some("MTLD3D")).as_deref(),
        Some("MTLD3D,mtld3d")
    );
}

#[test]
fn a_list_that_already_shows_ours_is_left_alone() {
    assert_eq!(domains_with_ours(Some("ALL")), None);
    assert_eq!(domains_with_ours(Some("*")), None);
    assert_eq!(domains_with_ours(Some("\"*\"")), None);
    assert_eq!(domains_with_ours(Some("com.example.game, mtld3d")), None);
}

#[test]
fn reporting_twice_in_one_process_is_harmless() {
    report_attached();
    report_attached();
}

/// A reporter that waits for the HUD whatever this macOS carries, so the test runs everywhere.
fn waiting_reporter(owner_id: u64) -> HudReporter {
    HudReporter {
        owner_id,
        gate: SampleGate::default(),
        latch: HudRowLatch::default(),
        state: ReporterState::Waiting,
        claimed: false,
    }
}

#[test]
fn the_newest_rendering_reporter_holds_the_rows_until_its_teardown() {
    // Without the HUD loaded no sample reports, but a due sample still takes the rows.
    let mut first = waiting_reporter(0x1000);
    assert_eq!(OWNER.load(Ordering::Relaxed), 0, "creation takes nothing");
    assert!(!first.sample_due(0, 2));
    assert_eq!(OWNER.load(Ordering::Relaxed), 0x1000);
    let mut second = waiting_reporter(0x2000);
    assert_eq!(
        OWNER.load(Ordering::Relaxed),
        0x1000,
        "a device that has not rendered takes nothing"
    );
    assert!(!second.sample_due(0, 2));
    assert_eq!(OWNER.load(Ordering::Relaxed), 0x2000);
    assert!(!first.sample_due(1, 2));
    assert_eq!(
        OWNER.load(Ordering::Relaxed),
        0x2000,
        "the first sample alone takes the rows from another encoder"
    );

    // One test, since both phases share the process-wide rows. A period of
    // 1000 cycles: the holder goes stale 2000 cycles after its last sample.
    let period = 1000;
    assert!(second.hold_rows(10_000, period));
    assert!(
        !first.hold_rows(11_000, period),
        "an active holder keeps the rows"
    );
    assert!(second.hold_rows(11_500, period));
    assert!(
        !first.hold_rows(13_000, period),
        "1500 cycles without a sample is not yet stale"
    );
    assert!(
        !first.hold_rows(11_000, period),
        "a holder's reading taken after `now` counts as fresh"
    );
    assert!(
        first.hold_rows(13_500, period),
        "a holder that stops sampling loses the rows after two periods"
    );
    assert_eq!(OWNER.load(Ordering::Relaxed), 0x1000);
    assert!(
        !second.hold_rows(14_000, period),
        "the new holder's fresh sample keeps the old one from taking the rows back"
    );
    assert!(first.hold_rows(14_500, period));
    assert!(
        second.hold_rows(16_600, period),
        "and the old one can take them back once it is stale"
    );
    assert!(!first.hold_rows(16_700, period));
    drop(second);
    assert_eq!(
        OWNER.load(Ordering::Relaxed),
        0,
        "the holder's teardown frees the rows"
    );
    assert!(
        first.hold_rows(16_800, period),
        "a free row set goes to the next sample"
    );
    let mut third = waiting_reporter(0x3000);
    assert!(third.hold_rows(16_900, period));
    drop(first);
    assert_eq!(
        OWNER.load(Ordering::Relaxed),
        0x3000,
        "a reporter that does not hold the rows leaves them alone"
    );
    drop(third);
    assert_eq!(OWNER.load(Ordering::Relaxed), 0);
}
