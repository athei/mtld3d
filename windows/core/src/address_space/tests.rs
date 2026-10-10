//! The free-space tally, the threshold latch and the page-box holder split.

use super::*;

const MIB: u64 = 1 << 20;
const GRANULE: u64 = ALLOCATION_GRANULARITY;

/// Feed `regions` of `(free, size)` laid end to end from address zero.
fn walk(regions: &[(bool, u64)]) -> FreeSpace {
    let mut space = FreeSpace::new();
    let mut base = 0;
    for &(free, size) in regions {
        space.add_region(free, base, size);
        base += size;
    }
    space
}

#[test]
fn free_space_sums_free_regions_and_keeps_the_largest() {
    let space = walk(&[
        (false, 900 * MIB),
        (true, 300 * MIB),
        (false, 64 * MIB),
        (true, 1200 * MIB),
        (true, 5 * MIB),
    ]);
    assert_eq!(space.total_mib(), 1505);
    assert_eq!(space.largest_mib(), 1200);
    assert_eq!(space.regions(), 5);
}

#[test]
fn free_space_never_reports_a_largest_block_above_the_total() {
    let mut space = FreeSpace::new();
    let mut base = 0;
    for (free, size) in [(true, 7 * MIB), (false, MIB), (true, 2030 * MIB), (true, 0)] {
        space.add_region(free, base, size);
        base += size;
        assert!(space.largest_mib() <= space.total_mib());
    }
}

#[test]
fn an_empty_walk_reports_nothing_free() {
    let space = FreeSpace::new();
    assert_eq!(space.total_mib(), 0);
    assert_eq!(space.largest_mib(), 0);
    assert_eq!(space.regions(), 0);
}

#[test]
fn free_space_truncates_to_whole_mib() {
    let space = walk(&[
        (true, MIB - GRANULE),
        (false, GRANULE),
        (true, MIB - GRANULE),
    ]);
    assert_eq!(space.total_mib(), 1);
    assert_eq!(space.largest_mib(), 0);
}

#[test]
fn a_free_region_counts_only_whole_granules() {
    let mut space = FreeSpace::new();
    // From 4 KiB past the boundary of granule 1 to 4 KiB short of the end
    // of granule 9: the partial granule at each end is unusable, granules 2
    // to 8 are not.
    space.add_region(true, GRANULE + 4096, 9 * GRANULE - 8192);
    assert_eq!(space.total, 7 * GRANULE);
    assert_eq!(space.largest, 7 * GRANULE);
}

#[test]
fn a_sliver_inside_one_granule_counts_nothing() {
    let mut space = FreeSpace::new();
    space.add_region(true, 3 * GRANULE + 4096, GRANULE - 8192);
    space.add_region(true, 5 * GRANULE + 8192, GRANULE - 8192);
    assert_eq!(space.total, 0);
    assert_eq!(space.largest, 0);
    assert_eq!(space.regions(), 2);
}

#[test]
fn an_aligned_free_region_counts_whole() {
    let mut space = FreeSpace::new();
    space.add_region(true, 16 * GRANULE, 32 * GRANULE);
    assert_eq!(space.total, 32 * GRANULE);
}

#[test]
fn a_region_at_the_top_of_the_space_does_not_overflow() {
    let mut space = FreeSpace::new();
    space.add_region(true, u64::MAX - GRANULE, GRANULE * 2);
    assert_eq!(space.total, 0);
}

const fn step(next: u8, report: ThresholdReport) -> ThresholdStep {
    ThresholdStep { next, report }
}

#[test]
fn a_first_sample_above_every_threshold_arms_the_latch_at_the_first() {
    assert_eq!(
        threshold_step(&FREE_THRESHOLDS_MIB, UNSAMPLED, 3000),
        Some(step(0, ThresholdReport::StartsAbove))
    );
    assert_eq!(
        threshold_step(&FREE_THRESHOLDS_MIB, UNSAMPLED, 1536),
        Some(step(0, ThresholdReport::StartsAbove))
    );
}

#[test]
fn a_first_sample_below_thresholds_skips_them_without_a_crossing() {
    assert_eq!(
        threshold_step(&FREE_THRESHOLDS_MIB, UNSAMPLED, 1900),
        Some(step(0, ThresholdReport::StartsAbove))
    );
    assert_eq!(
        threshold_step(&FREE_THRESHOLDS_MIB, UNSAMPLED, 1400),
        Some(step(1, ThresholdReport::StartsBelow(1536)))
    );
    assert_eq!(
        threshold_step(&FREE_THRESHOLDS_MIB, UNSAMPLED, 0),
        Some(step(6, ThresholdReport::StartsBelow(128)))
    );
}

#[test]
fn no_threshold_is_crossed_above_the_next() {
    assert_eq!(threshold_step(&FREE_THRESHOLDS_MIB, 0, 1536), None);
    assert_eq!(threshold_step(&FREE_THRESHOLDS_MIB, 1, 1100), None);
}

#[test]
fn one_crossing_reports_that_threshold() {
    assert_eq!(
        threshold_step(&FREE_THRESHOLDS_MIB, 0, 1535),
        Some(step(1, ThresholdReport::Crossed(1536)))
    );
    assert_eq!(
        threshold_step(&FREE_THRESHOLDS_MIB, 1, 1000),
        Some(step(2, ThresholdReport::Crossed(1024)))
    );
}

#[test]
fn a_fall_past_several_thresholds_reports_the_lowest_once() {
    assert_eq!(
        threshold_step(&FREE_THRESHOLDS_MIB, 0, 600),
        Some(step(3, ThresholdReport::Crossed(768)))
    );
    assert_eq!(threshold_step(&FREE_THRESHOLDS_MIB, 3, 600), None);
    assert_eq!(
        threshold_step(&FREE_THRESHOLDS_MIB, 0, 0),
        Some(step(6, ThresholdReport::Crossed(128)))
    );
}

#[test]
fn reported_thresholds_are_not_reported_again_after_a_recovery() {
    assert_eq!(threshold_step(&FREE_THRESHOLDS_MIB, 2, 2000), None);
    assert_eq!(threshold_step(&FREE_THRESHOLDS_MIB, 2, 1000), None);
    assert_eq!(
        threshold_step(&FREE_THRESHOLDS_MIB, 2, 700),
        Some(step(3, ThresholdReport::Crossed(768)))
    );
}

#[test]
fn every_threshold_reported_leaves_nothing_to_cross() {
    let last = u8::try_from(FREE_THRESHOLDS_MIB.len()).unwrap();
    assert_eq!(threshold_step(&FREE_THRESHOLDS_MIB, last, 0), None);
}

#[test]
fn the_largest_block_latch_reports_fragmentation_the_total_does_not() {
    // 1800 MiB free in total is above every free threshold, but no block
    // of it is larger than 200 MiB.
    assert_eq!(threshold_step(&FREE_THRESHOLDS_MIB, 0, 1800), None);
    assert_eq!(
        threshold_step(&LARGEST_THRESHOLDS_MIB, 0, 200),
        Some(step(2, ThresholdReport::Crossed(256)))
    );
}

#[test]
fn cycles_convert_to_whole_microseconds() {
    assert_eq!(cycles_to_micros(3_000, 3_000_000_000), 1);
    assert_eq!(cycles_to_micros(2_999, 3_000_000_000), 0);
    assert_eq!(cycles_to_micros(24_000_000, 24_000_000), 1_000_000);
    assert_eq!(cycles_to_micros(5, 0), 0);
    assert_eq!(cycles_to_micros(u64::MAX, 1), u64::MAX);
}

fn holders() -> PageBoxHolders {
    PageBoxHolders {
        total: 1042 * MIB,
        texture_staging: 600 * MIB,
        surfaces: 40 * MIB,
        vertex_index_backing: 3 * MIB,
        encoder_leases: 2 * MIB,
        upload_leases: 21 * MIB,
        texture_snapshots: 4 * MIB,
        pool_parked: 126 * MIB,
    }
}

#[test]
fn other_is_the_total_less_every_named_holder() {
    assert_eq!(holders().other(), 246 * MIB);
}

#[test]
fn other_saturates_when_the_holders_overcount() {
    let over = PageBoxHolders {
        total: 10 * MIB,
        ..holders()
    };
    assert_eq!(over.other(), 0);
}

#[test]
fn the_holder_clause_names_every_holder_in_mib() {
    assert_eq!(
        holders().to_string(),
        "page boxes 1042 MiB: texture staging 600, surfaces 40, vertex/index backing 3, \
         encoder leases 2, upload leases 21, texture snapshots 4, pool parked 126, other 246"
    );
}
