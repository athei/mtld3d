use super::*;

/// `x`, `y`, `w`, `h` as a rect, shortening the table-driven cases below.
const fn r(x: u32, y: u32, w: u32, h: u32) -> DirtyRect {
    DirtyRect { x, y, w, h }
}

#[test]
fn a_fresh_tracker_covers_nothing() {
    let cov = StagingCoverage::new();
    assert!(!cov.is_full());
}

#[test]
fn a_whole_level_write_covers() {
    let mut cov = StagingCoverage::new();
    cov.add(r(0, 0, 64, 64), 64, 64);
    assert!(cov.is_full());
    assert!(cov.rects.is_empty(), "a full level keeps no rects");
}

#[test]
fn a_rect_larger_than_the_level_covers() {
    let mut cov = StagingCoverage::new();
    cov.add(r(0, 0, 128, 128), 64, 64);
    assert!(cov.is_full());
}

#[test]
fn one_partial_write_does_not_cover() {
    let mut cov = StagingCoverage::new();
    cov.add(r(0, 0, 64, 32), 64, 64);
    assert!(!cov.is_full());
}

#[test]
fn two_halves_cover() {
    let mut cov = StagingCoverage::new();
    cov.add(r(0, 0, 64, 32), 64, 64);
    assert!(!cov.is_full());
    cov.add(r(0, 32, 64, 32), 64, 64);
    assert!(cov.is_full());
}

/// Two rects on opposite corners: the bounding box spans the level, the union does not.
///
/// The case a bounding-box union answers wrong. Their areas sum past the
/// level's, so the sweep runs and has to reject them on the two corners
/// neither rect touches.
#[test]
fn diagonal_rects_do_not_cover_despite_a_covering_bounding_box() {
    let mut cov = StagingCoverage::new();
    cov.add(r(0, 0, 48, 48), 64, 64);
    cov.add(r(16, 16, 48, 48), 64, 64);
    assert!(
        cov.written_area >= 64 * 64,
        "the areas must sum past the level's, or the sweep never runs"
    );
    assert!(!cov.is_full());
    // The two corners the pair misses, written last, complete the level.
    cov.add(r(48, 0, 16, 16), 64, 64);
    cov.add(r(0, 48, 16, 16), 64, 64);
    assert!(cov.is_full());
}

#[test]
fn tiled_writes_cover_the_level() {
    let mut cov = StagingCoverage::new();
    for ty in 0..4 {
        for tx in 0..4 {
            cov.add(r(tx * 16, ty * 16, 16, 16), 64, 64);
        }
    }
    assert!(cov.is_full());
}

/// A tiling with one tile left out never covers, whichever tile it is.
#[test]
fn a_tiling_missing_one_tile_never_covers() {
    for skip in 0..16 {
        let mut cov = StagingCoverage::new();
        for tile in 0..16 {
            if tile == skip {
                continue;
            }
            cov.add(r((tile % 4) * 16, (tile / 4) * 16, 16, 16), 64, 64);
        }
        assert!(!cov.is_full(), "tile {skip} was never written");
    }
}

/// Writes reaching both the top and the bottom edge but leaving a column gap do not cover.
///
/// The overlapping pairs push the area sum past the level's, so this rejection
/// is the sweep's, not the area gate's.
#[test]
fn a_column_gap_does_not_cover() {
    let mut cov = StagingCoverage::new();
    for x in [0, 38] {
        cov.add(r(x, 0, 26, 40), 64, 64);
        cov.add(r(x, 20, 26, 44), 64, 64);
    }
    assert!(cov.written_area >= 64 * 64);
    assert!(!cov.is_full());
    cov.add(r(26, 0, 12, 64), 64, 64);
    assert!(cov.is_full());
}

/// Bands of unequal height still tile the level.
#[test]
fn unequal_bands_cover() {
    let mut cov = StagingCoverage::new();
    cov.add(r(0, 0, 64, 7), 64, 64);
    cov.add(r(0, 7, 30, 57), 64, 64);
    cov.add(r(30, 7, 34, 57), 64, 64);
    assert!(cov.is_full());
}

#[test]
fn a_rect_outside_the_level_is_ignored() {
    let mut cov = StagingCoverage::new();
    cov.add(r(64, 64, 16, 16), 64, 64);
    assert!(cov.rects.is_empty());
    assert!(!cov.is_full());
}

#[test]
fn a_repeated_rect_is_recorded_once() {
    let mut cov = StagingCoverage::new();
    for _ in 0..1000 {
        cov.add(r(0, 0, 32, 32), 64, 64);
    }
    assert_eq!(cov.rects.len(), 1);
    // A rect inside one already recorded is redundant too.
    cov.add(r(4, 4, 8, 8), 64, 64);
    assert_eq!(cov.rects.len(), 1);
}

#[test]
fn coverage_stops_being_tracked_past_the_rect_bound() {
    let mut cov = StagingCoverage::new();
    for i in 0..u32::try_from(MAX_TRACKED_RECTS).expect("bound fits u32") {
        cov.add(r(i, 0, 1, 1), 4096, 4096);
    }
    assert_eq!(cov.rects.len(), MAX_TRACKED_RECTS);
    cov.add(r(0, 1, 1, 1), 4096, 4096);
    assert_eq!(cov.state, CoverageState::Untracked);
    assert!(cov.rects.is_empty(), "an untracked level keeps no rects");
    // A whole-level write is still an answer the tracker can give.
    cov.add(r(0, 0, 4096, 4096), 4096, 4096);
    assert!(cov.is_full());
}

#[test]
fn reset_forgets_full_coverage() {
    let mut cov = StagingCoverage::new();
    cov.mark_full();
    assert!(cov.is_full());
    cov.reset();
    assert!(!cov.is_full());
    cov.add(r(0, 0, 64, 32), 64, 64);
    assert!(!cov.is_full());
}

#[test]
fn full_coverage_survives_a_further_partial_write() {
    let mut cov = StagingCoverage::new();
    cov.mark_full();
    cov.add(r(8, 8, 4, 4), 64, 64);
    assert!(cov.is_full());
}

#[test]
fn a_one_texel_level_is_covered_by_its_single_texel() {
    let mut cov = StagingCoverage::new();
    cov.add(r(0, 0, 1, 1), 1, 1);
    assert!(cov.is_full());
}
