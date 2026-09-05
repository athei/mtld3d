//! Unit tests for the filter.

use super::{selected, test_id};

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
