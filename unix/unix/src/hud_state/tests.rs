use std::sync::atomic::Ordering;

use super::{DOMAIN, HudReporter, OWNER, domains_with_ours, report_attached};

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

#[test]
fn the_newest_reporter_holds_the_rows_until_its_teardown() {
    let first = HudReporter::new(0x1000);
    assert_eq!(OWNER.load(Ordering::Relaxed), 0x1000);
    let second = HudReporter::new(0x2000);
    assert_eq!(OWNER.load(Ordering::Relaxed), 0x2000);
    drop(first);
    assert_eq!(
        OWNER.load(Ordering::Relaxed),
        0x2000,
        "a reporter that does not hold the rows leaves them alone"
    );
    drop(second);
    assert_eq!(
        OWNER.load(Ordering::Relaxed),
        0,
        "the holder's teardown frees the rows"
    );
}
