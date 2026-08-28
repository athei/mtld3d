use super::*;

#[test]
fn a_fresh_mask_leaves_every_subresource_to_the_staging() {
    let mask = LevelAuthorityMask::new();
    for face in 0..FACE_COUNT {
        for level in 0..15 {
            assert!(!mask.gpu_holds(face, level), "face {face} level {level}");
        }
    }
}

#[test]
fn a_write_of_an_unclaimed_level_is_served_from_staging() {
    let mut mask = LevelAuthorityMask::new();
    assert_eq!(mask.plan_write(0, 0, false), WritePlan::WriteStaging);
    assert_eq!(mask.plan_write(0, 0, true), WritePlan::WriteStaging);
}

#[test]
fn a_partial_write_of_a_claimed_level_reads_it_back() {
    let mut mask = LevelAuthorityMask::new();
    mask.gpu_wrote(0, 2);
    assert_eq!(mask.plan_write(0, 2, false), WritePlan::ReadBackFirst);
}

#[test]
fn a_whole_level_write_of_a_claimed_level_skips_the_read_back() {
    let mut mask = LevelAuthorityMask::new();
    mask.gpu_wrote(0, 0);
    assert_eq!(mask.plan_write(0, 0, true), WritePlan::Overwrite);
}

/// The claim goes at the write that defines the level, so the next one is free.
///
/// A `D3DLOCK_DISCARD` map takes the whole-level branch: leaving the claim
/// standing would make the map after it read back pixels the application had
/// already overwritten.
#[test]
fn a_whole_level_write_releases_the_claim() {
    let mut mask = LevelAuthorityMask::new();
    mask.gpu_wrote(0, 0);
    assert_eq!(mask.plan_write(0, 0, true), WritePlan::Overwrite);
    assert!(
        !mask.gpu_holds(0, 0),
        "the whole-level write released the claim"
    );
    assert_eq!(
        mask.plan_write(0, 0, false),
        WritePlan::WriteStaging,
        "the write after it pays no read back"
    );
}

#[test]
fn a_read_back_releases_the_claim() {
    let mut mask = LevelAuthorityMask::new();
    mask.gpu_wrote(0, 3);
    assert_eq!(mask.plan_write(0, 3, false), WritePlan::ReadBackFirst);
    assert!(!mask.gpu_holds(0, 3));
    assert_eq!(mask.plan_write(0, 3, false), WritePlan::WriteStaging);
}

#[test]
fn levels_are_claimed_independently() {
    let mut mask = LevelAuthorityMask::new();
    mask.gpu_wrote(0, 1);
    mask.gpu_wrote(0, 4);
    assert_eq!(mask.plan_write(0, 1, true), WritePlan::Overwrite);
    assert!(
        mask.gpu_holds(0, 4),
        "one level's write leaves the others alone"
    );
    assert_eq!(mask.plan_write(0, 0, false), WritePlan::WriteStaging);
    assert_eq!(mask.plan_write(0, 4, false), WritePlan::ReadBackFirst);
}

/// A blit into one cube face says nothing about the same level of the others.
#[test]
fn faces_are_claimed_independently() {
    let mut mask = LevelAuthorityMask::new();
    mask.gpu_wrote(3, 0);
    assert!(mask.gpu_holds(3, 0));
    for face in [0, 1, 2, 4, 5] {
        assert!(!mask.gpu_holds(face, 0), "face {face}");
        assert_eq!(mask.plan_write(face, 0, false), WritePlan::WriteStaging);
    }
    assert_eq!(mask.plan_write(3, 0, false), WritePlan::ReadBackFirst);
}

/// Releasing one face's claim leaves the other faces' claims standing.
#[test]
fn a_write_of_one_face_leaves_the_others_claimed() {
    let mut mask = LevelAuthorityMask::new();
    for face in 0..FACE_COUNT {
        mask.gpu_wrote(face, 2);
    }
    assert_eq!(mask.plan_write(1, 2, true), WritePlan::Overwrite);
    assert!(!mask.gpu_holds(1, 2));
    for face in [0, 2, 3, 4, 5] {
        assert!(mask.gpu_holds(face, 2), "face {face}");
    }
}

/// A level past the mask's width is never claimed, so it never reads back.
#[test]
fn a_level_past_the_mask_stays_with_the_staging() {
    let mut mask = LevelAuthorityMask::new();
    mask.gpu_wrote(0, u32::BITS as usize);
    assert!(!mask.gpu_holds(0, u32::BITS as usize));
    assert_eq!(
        mask.plan_write(0, u32::BITS as usize, false),
        WritePlan::WriteStaging
    );
}

/// A face past the six a cube carries is never claimed either.
#[test]
fn a_face_past_the_mask_stays_with_the_staging() {
    let mut mask = LevelAuthorityMask::new();
    mask.gpu_wrote(FACE_COUNT, 0);
    assert!(!mask.gpu_holds(FACE_COUNT, 0));
    assert_eq!(
        mask.plan_write(FACE_COUNT, 0, false),
        WritePlan::WriteStaging
    );
}
