//! Unit tests for the blit-source read guards and the staging gauge they feed.

use std::sync::Arc;

use mtld3d_core::{
    page_box::{PAGE_SIZE, PageBox, PageBoxRead},
    perf::EncoderPerfState,
};

use super::BlitRetention;

/// A Reset or shutdown that releases queued reads drops every guard and empties the gauge.
///
/// Both cleanups empty the queue without waiting for the reads to retire
/// one by one, so the bytes the submit added must leave with them, and the
/// reads of the frame being encoded, which were never added, leave nothing.
/// The queues are private, so `release_all` is the only way the cleanups
/// can empty them.
#[test]
fn releasing_every_read_drops_the_guards_and_returns_the_gauge_to_zero() {
    let first = Arc::new(PageBox::new_uninit(PAGE_SIZE));
    let second = Arc::new(PageBox::new_uninit(3 * PAGE_SIZE));
    let mut perf = EncoderPerfState::new();
    let mut retention = BlitRetention::default();
    retention.hold(PageBoxRead::new(Arc::clone(&first)), first.len());
    retention.hold(PageBoxRead::new(Arc::clone(&second)), second.len());
    retention.queue(&mut perf, 7);
    #[cfg(perf_tracking)]
    assert_eq!(perf.tex_staging_retained_bytes(), 4 * PAGE_SIZE);
    retention.hold(PageBoxRead::new(Arc::clone(&first)), first.len());
    assert_eq!(
        retention.queued(),
        2,
        "the current frame's read is not queued"
    );

    retention.release_all(&mut perf);
    #[cfg(perf_tracking)]
    assert_eq!(perf.tex_staging_retained_bytes(), 0);
    assert_eq!(retention.queued(), 0);
    assert_eq!(Arc::strong_count(&first), 1, "every read guard dropped");
    assert_eq!(Arc::strong_count(&second), 1);
    assert!(!first.has_readers() && !second.has_readers());
}

/// Retirement releases only the reads whose sequence the GPU has reached.
#[test]
fn reclaim_releases_only_retired_reads() {
    let first = Arc::new(PageBox::new_uninit(PAGE_SIZE));
    let second = Arc::new(PageBox::new_uninit(2 * PAGE_SIZE));
    let mut perf = EncoderPerfState::new();
    let mut retention = BlitRetention::default();
    retention.hold(PageBoxRead::new(Arc::clone(&first)), first.len());
    retention.queue(&mut perf, 3);
    retention.hold(PageBoxRead::new(Arc::clone(&second)), second.len());
    retention.queue(&mut perf, 4);

    retention.reclaim(&mut perf, 3);
    assert_eq!(retention.queued(), 1);
    assert_eq!(Arc::strong_count(&first), 1);
    assert_eq!(Arc::strong_count(&second), 2, "seq 4 has not retired");
    #[cfg(perf_tracking)]
    assert_eq!(perf.tex_staging_retained_bytes(), 2 * PAGE_SIZE);
}

/// A read of a shared chunk adds only the bytes it covers to the gauge.
///
/// A snapshot is one box in an arena chunk other uploads share, so the gauge
/// counts the box, not the chunk, and takes the same figure off at reclaim.
#[test]
fn a_held_snapshot_counts_only_its_own_bytes() {
    let chunk = Arc::new(PageBox::new_uninit(16 * PAGE_SIZE));
    let mut perf = EncoderPerfState::new();
    let mut retention = BlitRetention::default();
    retention.hold(PageBoxRead::new(Arc::clone(&chunk)), 4096);
    retention.hold(PageBoxRead::new(Arc::clone(&chunk)), 512);
    retention.queue(&mut perf, 5);
    #[cfg(perf_tracking)]
    assert_eq!(perf.tex_staging_retained_bytes(), 4608);
    retention.reclaim(&mut perf, 5);
    #[cfg(perf_tracking)]
    assert_eq!(perf.tex_staging_retained_bytes(), 0);
    assert!(!chunk.has_readers());
}
