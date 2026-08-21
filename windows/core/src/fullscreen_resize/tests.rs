use super::{ExternalResizeAction, ExternalResizeGuard, REASSERT_BUDGET};

const MONITOR: (u32, u32) = (3456, 2234);

#[test]
fn a_covered_window_is_left_alone() {
    let mut g = ExternalResizeGuard::new();
    assert_eq!(g.decide(MONITOR, MONITOR), ExternalResizeAction::Covered);
}

#[test]
fn a_shrunk_window_is_reasserted_once_per_size() {
    let mut g = ExternalResizeGuard::new();
    assert_eq!(
        g.decide((1288, 834), MONITOR),
        ExternalResizeAction::Reassert
    );
    // The same clamp coming back means the re-assert did not stick.
    assert_eq!(
        g.decide((1288, 834), MONITOR),
        ExternalResizeAction::Suppressed
    );
    // A different size is a new event and earns its own re-assert.
    assert_eq!(
        g.decide((1608, 934), MONITOR),
        ExternalResizeAction::Reassert
    );
}

#[test]
fn covering_the_monitor_clears_the_guard_and_refills_the_budget() {
    let mut g = ExternalResizeGuard::new();
    assert_eq!(
        g.decide((1288, 834), MONITOR),
        ExternalResizeAction::Reassert
    );
    assert_eq!(g.decide(MONITOR, MONITOR), ExternalResizeAction::Covered);
    // The previously re-asserted size counts as a fresh event again.
    assert_eq!(
        g.decide((1288, 834), MONITOR),
        ExternalResizeAction::Reassert
    );
}

#[test]
fn alternating_clamp_sizes_exhaust_the_budget() {
    let mut g = ExternalResizeGuard::new();
    for i in 0..REASSERT_BUDGET {
        let size = if i % 2 == 0 { (100, 100) } else { (200, 200) };
        assert_eq!(g.decide(size, MONITOR), ExternalResizeAction::Reassert);
    }
    assert_eq!(
        g.decide((100, 100), MONITOR),
        ExternalResizeAction::Suppressed
    );
    assert_eq!(
        g.decide((300, 300), MONITOR),
        ExternalResizeAction::Suppressed
    );
}

#[test]
fn reset_refills_the_budget() {
    let mut g = ExternalResizeGuard::new();
    for i in 0..REASSERT_BUDGET {
        let size = if i % 2 == 0 { (100, 100) } else { (200, 200) };
        assert_eq!(g.decide(size, MONITOR), ExternalResizeAction::Reassert);
    }
    g.reset();
    assert_eq!(
        g.decide((100, 100), MONITOR),
        ExternalResizeAction::Reassert
    );
}
