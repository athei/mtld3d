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
    let mask = LevelAuthorityMask::new();
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
    mask.staging_wrote(0, 0);
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
    mask.staging_wrote(0, 3);
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
    mask.staging_wrote(1, 2);
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

#[test]
fn failed_materialization_keeps_every_face_and_mip_for_retry() {
    for face in 0..FACE_COUNT {
        for level in 0..15 {
            let mut mask = LevelAuthorityMask::new();
            mask.gpu_wrote(face, level);
            assert_eq!(
                mask.plan_write(face, level, false),
                WritePlan::ReadBackFirst
            );
            // No staging write completed, so the next attempt still needs the GPU.
            assert!(mask.gpu_holds(face, level), "face {face} level {level}");
            assert_eq!(
                mask.plan_write(face, level, false),
                WritePlan::ReadBackFirst
            );
            assert!(
                mask.gpu_holds(face, level),
                "retry face {face} level {level}"
            );
        }
    }
}

#[test]
fn an_unfinished_whole_overwrite_keeps_the_gpu_copy() {
    for face in 0..FACE_COUNT {
        for level in 0..15 {
            let mut mask = LevelAuthorityMask::new();
            mask.gpu_wrote(face, level);
            assert_eq!(mask.plan_write(face, level, true), WritePlan::Overwrite);
            // A later copy bounds check can reject before the write completes.
            assert!(mask.gpu_holds(face, level), "face {face} level {level}");
            assert_eq!(
                mask.plan_write(face, level, false),
                WritePlan::ReadBackFirst
            );
        }
    }
}

#[test]
fn a_recreated_level_keeps_its_failed_readback_obligation() {
    let mut mask = LevelAuthorityMask::new();
    // Freshly allocated staging has no copy of the released level's GPU pixels.
    mask.gpu_wrote(0, 4);
    for _ in 0..3 {
        assert_eq!(mask.plan_write(0, 4, false), WritePlan::ReadBackFirst);
        assert!(mask.gpu_holds(0, 4));
    }
    assert_eq!(mask.plan_write(0, 4, true), WritePlan::Overwrite);
    assert!(mask.gpu_holds(0, 4));
}

#[test]
fn successful_retry_commits_only_its_face_and_mip() {
    for completed_face in 0..FACE_COUNT {
        for completed_level in 0..15 {
            let mut mask = LevelAuthorityMask::new();
            for face in 0..FACE_COUNT {
                for level in 0..15 {
                    mask.gpu_wrote(face, level);
                }
            }
            for readback_completed in [false, false, true] {
                assert_eq!(
                    mask.plan_write(completed_face, completed_level, false),
                    WritePlan::ReadBackFirst
                );
                if readback_completed {
                    mask.staging_wrote(completed_face, completed_level);
                }
                for face in 0..FACE_COUNT {
                    for level in 0..15 {
                        let committed = readback_completed
                            && face == completed_face
                            && level == completed_level;
                        assert_eq!(mask.gpu_holds(face, level), !committed);
                    }
                }
            }
            assert_eq!(
                mask.plan_write(completed_face, completed_level, false),
                WritePlan::WriteStaging
            );
        }
    }
}

#[test]
fn successful_whole_overwrite_commits_only_its_face_and_mip() {
    let mut mask = LevelAuthorityMask::new();
    for face in 0..FACE_COUNT {
        for level in 0..15 {
            mask.gpu_wrote(face, level);
        }
    }
    assert_eq!(mask.plan_write(3, 7, true), WritePlan::Overwrite);
    assert!(mask.gpu_holds(3, 7));
    mask.staging_wrote(3, 7);
    assert_eq!(mask.plan_write(3, 7, false), WritePlan::WriteStaging);
    for face in 0..FACE_COUNT {
        for level in 0..15 {
            assert_eq!(mask.gpu_holds(face, level), face != 3 || level != 7);
        }
    }
}

#[test]
fn recreated_staging_is_valid_only_after_successful_retry() {
    let mut mask = LevelAuthorityMask::new();
    let mut coverage = crate::staging_coverage::StagingCoverage::new();
    // Allocation reset coverage before its required readback failed.
    mask.gpu_wrote(0, 2);
    for readback_completed in [false, true] {
        assert!(!coverage.is_full());
        assert_eq!(mask.plan_write(0, 2, false), WritePlan::ReadBackFirst);
        if readback_completed {
            coverage.mark_full();
            mask.staging_wrote(0, 2);
        }
        assert_eq!(coverage.is_full(), readback_completed);
        assert_eq!(mask.gpu_holds(0, 2), !readback_completed);
    }
    assert_eq!(mask.plan_write(0, 2, false), WritePlan::WriteStaging);
}

#[test]
fn a_staging_commit_outside_the_mask_changes_no_claim() {
    let mut mask = LevelAuthorityMask::new();
    for face in 0..FACE_COUNT {
        for level in 0..15 {
            mask.gpu_wrote(face, level);
        }
    }
    mask.staging_wrote(FACE_COUNT, 0);
    mask.staging_wrote(0, u32::BITS as usize);
    mask.staging_wrote(u32::MAX, usize::MAX);
    for face in 0..FACE_COUNT {
        for level in 0..15 {
            assert!(mask.gpu_holds(face, level));
        }
    }
}
