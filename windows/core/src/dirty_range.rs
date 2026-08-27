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

    /// Every byte of a `logical_len`-byte buffer.
    ///
    /// `Staged` VB/IB creation seeds this so the first `Unlock` uploads
    /// the whole staging buffer. Its device buffer starts undefined and
    /// nothing fills it, and a fill done entirely through `D3DLOCK_READONLY`
    /// locks records no range at all, so the opening upload has to carry
    /// everything or the GPU reads bytes no one wrote.
    #[must_use]
    pub const fn full(logical_len: u32) -> Self {
        Self {
            min: 0,
            max: logical_len,
        }
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
mod tests;
