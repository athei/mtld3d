//! Unit tests for the filter.

use super::{selected, skipped, test_id};

fn patterns(list: &[&str]) -> Vec<String> {
    list.iter().map(|p| (*p).to_owned()).collect()
}

#[test]
fn patterns_are_substrings_of_the_id_in_union() {
    let id = test_id("e2e", "msaa::resolve_counts_edge_pixels");
    assert_eq!(id, "e2e::msaa::resolve_counts_edge_pixels");
    assert!(selected(&id, &patterns(&["msaa::"])));
    assert!(selected(&id, &patterns(&["stencil", "edge_pixels"])));
    assert!(!selected(&id, &patterns(&["stencil"])));
    assert!(selected(&id, &[]), "no pattern selects everything");
}

#[test]
fn skip_patterns_leave_out_what_they_are_substrings_of() {
    let id = test_id("e2e", "window_lifecycle::devices_come_and_go");
    let other = test_id("e2e", "window_lifecycle::a_window_is_destroyed");
    let skip = patterns(&["e2e::window_lifecycle::devices_come_and_go"]);
    assert!(skipped(&id, &skip));
    assert!(
        !skipped(&other, &skip),
        "a skip leaves the rest of the module alone"
    );
    assert!(!skipped(&id, &[]), "no pattern skips nothing");
    let filter = patterns(&["window_lifecycle::"]);
    assert!(
        selected(&id, &filter) && skipped(&id, &skip),
        "a skip narrows a filter"
    );
}
