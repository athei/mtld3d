use super::{DOMAIN, domains_with_ours, report_attached};

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
        Some("com.example.game,MTLD3D")
    );
    assert_eq!(
        domains_with_ours(Some("\"com.example.a, com.example.b\"")).as_deref(),
        Some("com.example.a, com.example.b,MTLD3D")
    );
}

#[test]
fn a_list_that_already_shows_ours_is_left_alone() {
    assert_eq!(domains_with_ours(Some("ALL")), None);
    assert_eq!(domains_with_ours(Some("*")), None);
    assert_eq!(domains_with_ours(Some("\"*\"")), None);
    assert_eq!(domains_with_ours(Some("com.example.game, MTLD3D")), None);
}

#[test]
fn reporting_twice_in_one_process_is_harmless() {
    report_attached();
    report_attached();
}
