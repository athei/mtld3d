use strum::EnumCount;

use super::*;
#[test]
fn duration_scaling_preserves_counts_and_saturates_ticks() {
    let mut counters = FrameCounters::new();
    counters.vbib_pool_hits = 19;
    counters.api_cycles_by_category = [200; crate::perf::ApiCategory::COUNT];
    counters.query_wait_cycles = 50;
    counters.rescale_durations(100, 1000);
    assert_eq!(counters.vbib_pool_hits, 19);
    assert_eq!(counters.query_wait_cycles, 500);
    assert!(counters.api_cycles_by_category.iter().all(|&v| v == 2000));
    assert_eq!(scale_ticks(u64::MAX, 1, 2), u64::MAX);
    assert_eq!(scale_ticks(3, 2, 1), 1);
}
