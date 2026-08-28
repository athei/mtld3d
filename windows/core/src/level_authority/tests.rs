use super::*;

#[test]
fn a_fresh_mask_leaves_every_level_to_the_staging() {
    let mask = LevelAuthorityMask::new();
    for level in 0..15 {
        assert!(!mask.gpu_holds(level), "level {level}");
    }
}

#[test]
fn a_write_of_an_unclaimed_level_is_served_from_staging() {
    let mut mask = LevelAuthorityMask::new();
    assert_eq!(mask.plan_write(0, false), WritePlan::WriteStaging);
    assert_eq!(mask.plan_write(0, true), WritePlan::WriteStaging);
}

#[test]
fn a_partial_write_of_a_claimed_level_reads_it_back() {
    let mut mask = LevelAuthorityMask::new();
    mask.gpu_wrote(2);
    assert_eq!(mask.plan_write(2, false), WritePlan::ReadBackFirst);
}

#[test]
fn a_whole_level_write_of_a_claimed_level_skips_the_read_back() {
    let mut mask = LevelAuthorityMask::new();
    mask.gpu_wrote(0);
    assert_eq!(mask.plan_write(0, true), WritePlan::Overwrite);
}

/// The claim goes at the write that defines the level, so the next one is free.
///
/// A `D3DLOCK_DISCARD` map takes the whole-level branch: leaving the claim
/// standing would make the map after it read back pixels the application had
/// already overwritten.
#[test]
fn a_whole_level_write_releases_the_claim() {
    let mut mask = LevelAuthorityMask::new();
    mask.gpu_wrote(0);
    assert_eq!(mask.plan_write(0, true), WritePlan::Overwrite);
    assert!(
        !mask.gpu_holds(0),
        "the whole-level write released the claim"
    );
    assert_eq!(
        mask.plan_write(0, false),
        WritePlan::WriteStaging,
        "the write after it pays no read back"
    );
}

#[test]
fn a_read_back_releases_the_claim() {
    let mut mask = LevelAuthorityMask::new();
    mask.gpu_wrote(3);
    assert_eq!(mask.plan_write(3, false), WritePlan::ReadBackFirst);
    assert!(!mask.gpu_holds(3));
    assert_eq!(mask.plan_write(3, false), WritePlan::WriteStaging);
}

#[test]
fn levels_are_claimed_independently() {
    let mut mask = LevelAuthorityMask::new();
    mask.gpu_wrote(1);
    mask.gpu_wrote(4);
    assert_eq!(mask.plan_write(1, true), WritePlan::Overwrite);
    assert!(
        mask.gpu_holds(4),
        "one level's write leaves the others alone"
    );
    assert_eq!(mask.plan_write(0, false), WritePlan::WriteStaging);
    assert_eq!(mask.plan_write(4, false), WritePlan::ReadBackFirst);
}

/// A level past the mask's width is never claimed, so it never reads back.
#[test]
fn a_level_past_the_mask_stays_with_the_staging() {
    let mut mask = LevelAuthorityMask::new();
    mask.gpu_wrote(u32::BITS as usize);
    assert!(!mask.gpu_holds(u32::BITS as usize));
    assert_eq!(
        mask.plan_write(u32::BITS as usize, false),
        WritePlan::WriteStaging
    );
}
