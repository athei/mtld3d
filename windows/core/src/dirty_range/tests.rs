use super::{DirtyRange, indexed_vb_range_lower_bound, nonindexed_vb_range};

const LEN: u32 = 4096;

#[test]
fn empty_has_no_span() {
    let d = DirtyRange::empty();
    assert!(d.is_empty());
    assert_eq!(d.span(), None);
}

#[test]
fn single_conjoin_sets_span() {
    let mut d = DirtyRange::empty();
    d.conjoin(256, 1024, LEN);
    assert_eq!(d.span(), Some((256, 1280)));
}

#[test]
fn overlaps_half_open_semantics() {
    let mut d = DirtyRange::empty();
    // Drawn region [256, 1280).
    d.conjoin(256, 1024, LEN);
    // Empty range overlaps nothing.
    assert!(!DirtyRange::empty().overlaps(0, LEN));
    // Disjoint below and above (touching at the boundary does not
    // overlap, since both spans are half-open).
    assert!(!d.overlaps(0, 256));
    assert!(!d.overlaps(1280, 2048));
    // Genuine overlaps: straddling the start, fully inside, straddling
    // the end, and fully covering.
    assert!(d.overlaps(0, 257));
    assert!(d.overlaps(512, 600));
    assert!(d.overlaps(1279, 4096));
    assert!(d.overlaps(0, LEN));
}

#[test]
fn zero_size_means_to_end() {
    let mut d = DirtyRange::empty();
    d.conjoin(256, 0, LEN);
    assert_eq!(d.span(), Some((256, LEN)));
}

#[test]
fn disjoint_conjoins_widen_to_cover_gap() {
    let mut d = DirtyRange::empty();
    d.conjoin(0, 256, LEN);
    d.conjoin(2048, 256, LEN);
    // Single span covers the gap between the two writes.
    assert_eq!(d.span(), Some((0, 2304)));
}

#[test]
fn overlapping_conjoins_merge() {
    let mut d = DirtyRange::empty();
    d.conjoin(100, 200, LEN);
    d.conjoin(250, 200, LEN);
    assert_eq!(d.span(), Some((100, 450)));
}

#[test]
fn inner_conjoin_does_not_shrink() {
    let mut d = DirtyRange::empty();
    d.conjoin(0, 1000, LEN);
    d.conjoin(400, 100, LEN);
    assert_eq!(d.span(), Some((0, 1000)));
}

#[test]
fn offset_past_end_is_noop() {
    let mut d = DirtyRange::empty();
    d.conjoin(LEN + 100, 256, LEN);
    assert!(d.is_empty());
}

#[test]
fn size_clamped_to_buffer_end() {
    let mut d = DirtyRange::empty();
    d.conjoin(LEN - 256, 1024, LEN);
    assert_eq!(d.span(), Some((LEN - 256, LEN)));
}

#[test]
fn clear_resets() {
    let mut d = DirtyRange::empty();
    d.conjoin(0, 1024, LEN);
    d.clear();
    assert!(d.is_empty());
    assert_eq!(d.span(), None);
}

#[test]
fn nonindexed_exact_range() {
    // 100 verts from index 10, 32-byte stride, stream offset 256:
    // bytes [256 + 10*32, 256 + 110*32) = [576, 3776), size 3200.
    assert_eq!(nonindexed_vb_range(256, 32, 10, 100), Some((576, 3200)));
}

#[test]
fn nonindexed_zero_count_reads_nothing() {
    assert_eq!(nonindexed_vb_range(0, 32, 5, 0), None);
}

#[test]
fn nonindexed_start_overflow_falls_back_to_whole_tail() {
    // start_vertex * stride overflows u32 → conservative [offset, end).
    assert_eq!(nonindexed_vb_range(100, u32::MAX, 2, 4), Some((100, 0)));
}

#[test]
fn nonindexed_size_overflow_preserves_to_end_from_start() {
    // start fits but vertex_count * stride overflows → to-end from the
    // exact start (over-cover, never under-cover).
    assert_eq!(
        nonindexed_vb_range(0, 1 << 16, 1, 1 << 16),
        Some((1 << 16, 0))
    );
}

#[test]
fn nonindexed_zero_stride_is_to_end() {
    // Degenerate stride: size collapses to 0 (to-end), which is a safe
    // over-cover.
    assert_eq!(nonindexed_vb_range(512, 0, 4, 10), Some((512, 0)));
}

#[test]
fn indexed_positive_base_tightens_lower_bound() {
    // base_vertex 100, 32-byte stride, offset 256 → start 256 + 3200.
    assert_eq!(
        indexed_vb_range_lower_bound(256, 32, 100, 50),
        Some((3456, 0))
    );
}

#[test]
fn indexed_negative_base_is_conservative() {
    // Can't raise the floor below stream_offset → whole tail.
    assert_eq!(
        indexed_vb_range_lower_bound(256, 32, -5, 50),
        Some((256, 0))
    );
}

#[test]
fn indexed_zero_count_reads_nothing() {
    assert_eq!(indexed_vb_range_lower_bound(256, 32, 100, 0), None);
}

#[test]
fn indexed_start_overflow_falls_back_to_whole_tail() {
    assert_eq!(
        indexed_vb_range_lower_bound(100, u32::MAX, 2, 4),
        Some((100, 0))
    );
}

#[test]
fn nonindexed_range_covers_exactly_what_the_draw_reads() {
    // Draw reads bytes [576, 3776). Conjoined, the recorded range must
    // overlap every byte read (never under-cover) and nothing past it
    // (exact upper bound — no false rename on a disjoint later upload).
    const BIG: u32 = 1 << 20;
    let (off, size) = nonindexed_vb_range(256, 32, 10, 100).unwrap();
    let mut d = DirtyRange::empty();
    d.conjoin(off, size, BIG);
    // First and last bytes read overlap.
    assert!(d.overlaps(576, 577));
    assert!(d.overlaps(3775, 3776));
    // One byte before and after the read span do not.
    assert!(!d.overlaps(575, 576));
    assert!(!d.overlaps(3776, 3777));
}
