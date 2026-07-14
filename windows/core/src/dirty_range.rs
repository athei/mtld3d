//! Single conjoined byte span tracking the region a `Staged` VB/IB was dirtied.
//!
//! The dirtying spans one or more `Lock`s before its `Unlock` upload.
//! Pure arithmetic — no platform APIs — so it is host-testable.
//!
//! Coalesces sub-lock spans into one conjoined range: rather than
//! tracking disjoint sub-locks, it widens one half-open `[min, max)`
//! span to cover every write since the last upload. Gaps between
//! sub-locks get re-uploaded too, but that over-copy is negligible
//! against the simplicity and is an acceptable trade-off.

/// A half-open `[min, max)` byte range. Empty when `min >= max`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct DirtyRange {
    min: u32,
    max: u32,
}

impl DirtyRange {
    /// An empty range — nothing dirtied yet.
    #[must_use]
    pub const fn empty() -> Self {
        Self { min: 0, max: 0 }
    }

    /// Whether the range covers no bytes.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.min >= self.max
    }

    /// The dirtied `[min, max)` span, or `None` when empty.
    #[must_use]
    pub const fn span(&self) -> Option<(u32, u32)> {
        if self.is_empty() {
            None
        } else {
            Some((self.min, self.max))
        }
    }

    /// Reset to empty — call after the dirtied bytes have been uploaded.
    pub const fn clear(&mut self) {
        self.min = 0;
        self.max = 0;
    }

    /// True if this (non-empty) range overlaps the half-open `[off, end)`.
    ///
    /// Two half-open spans overlap iff each starts before the other ends.
    /// An empty range overlaps nothing.
    #[must_use]
    pub const fn overlaps(&self, off: u32, end: u32) -> bool {
        !self.is_empty() && self.min < end && off < self.max
    }

    /// Widen the range to include the `Lock` region `[offset, offset + size)`.
    ///
    /// Clamped to `logical_len`. `size == 0` means "to end of buffer"
    /// per D3D9. A region that starts at or past the end (or has a zero
    /// effective length) contributes nothing.
    pub const fn conjoin(&mut self, offset: u32, size: u32, logical_len: u32) {
        let start = if offset > logical_len {
            logical_len
        } else {
            offset
        };
        let remaining = logical_len - start;
        let span = if size == 0 || size > remaining {
            remaining
        } else {
            size
        };
        let end = start + span;
        if end <= start {
            return;
        }
        if self.is_empty() {
            self.min = start;
            self.max = end;
            return;
        }
        if start < self.min {
            self.min = start;
        }
        if end > self.max {
            self.max = end;
        }
    }
}

/// The vertex-buffer byte sub-range a non-indexed draw reads, as `(offset, size)`.
///
/// For [`DirtyRange::conjoin`] (`size == 0` = to end of buffer). `None`
/// means the draw reads nothing — skip recording.
///
/// Exact: vertices `[start_vertex, start_vertex + vertex_count)` at
/// `stride` bytes each, from `stream_offset`. The range may over-cover
/// but must never under-cover — a missed overlap reuses a buffer a later
/// upload corrupts — so any arithmetic overflow falls back to the
/// conservative whole-tail `[stream_offset, end)`.
#[must_use]
pub const fn nonindexed_vb_range(
    stream_offset: u32,
    stride: u32,
    start_vertex: u32,
    vertex_count: u32,
) -> Option<(u32, u32)> {
    if vertex_count == 0 {
        return None;
    }
    // start = stream_offset + start_vertex * stride
    let start = match start_vertex.checked_mul(stride) {
        Some(skip) => match stream_offset.checked_add(skip) {
            Some(start) => start,
            None => return Some((stream_offset, 0)),
        },
        None => return Some((stream_offset, 0)),
    };
    // size = vertex_count * stride; overflow → to-end from the exact
    // start (over-cover, never under-cover).
    match vertex_count.checked_mul(stride) {
        Some(size) => Some((start, size)),
        None => Some((start, 0)),
    }
}

/// The vertex-buffer byte sub-range an indexed draw reads, lower-bounded by `base_vertex`.
///
/// As `(offset, size)` for [`DirtyRange::conjoin`]. `None` means the
/// draw reads nothing — skip recording.
///
/// The exact upper bound needs the maximum index value (an index-buffer
/// scan we deliberately avoid), so the span stays "to end of buffer"
/// (`size 0`); only the lower bound is tightened. Safe because the
/// lowest vertex an index selects is `base_vertex + min_index` with
/// `min_index >= 0`, so reads never start below `base_vertex`. A
/// negative `base_vertex` (D3D9 allows it) or any overflow falls back to
/// the conservative whole-tail `[stream_offset, end)`.
#[must_use]
pub const fn indexed_vb_range_lower_bound(
    stream_offset: u32,
    stride: u32,
    base_vertex: i32,
    index_count: u32,
) -> Option<(u32, u32)> {
    if index_count == 0 {
        return None;
    }
    if base_vertex < 0 {
        return Some((stream_offset, 0));
    }
    // `base_vertex >= 0` here, so `unsigned_abs` is the value itself
    // (and sidesteps a sign-loss cast). start = stream_offset + base * stride.
    let base = base_vertex.unsigned_abs();
    match base.checked_mul(stride) {
        Some(skip) => match stream_offset.checked_add(skip) {
            Some(start) => Some((start, 0)),
            None => Some((stream_offset, 0)),
        },
        None => Some((stream_offset, 0)),
    }
}

#[cfg(test)]
mod tests {
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
}
