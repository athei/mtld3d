//! Exact union coverage of the writes landing in one texture level's staging.
//!
//! A default-pool level the game cannot lock releases its staging once the GPU
//! holds every byte of it (`d3d9::texture::schedule_upload`). The decision needs
//! to know whether the writes since the staging was allocated cover the whole
//! level, and a bounding box cannot answer that: two rects on opposite corners
//! have a bounding box spanning the level while the other two corners were never
//! written, and dropping there would hand the GPU whatever the uninitialised
//! pages held.
//!
//! So the tracker keeps the rects themselves and answers with an exact union
//! test: a band sweep over the distinct row boundaries, each band covered only
//! when the rects spanning it tile `[0, level_w)` without a gap. The sweep runs
//! only once the recorded areas sum to at least the level's own, which no
//! partially written level ever reaches.
//!
//! Two bounds keep the bookkeeping cheap. A rect already contained in a
//! recorded one is dropped on arrival, so a game re-writing the same region
//! every frame stays at one entry. Past [`MAX_TRACKED_RECTS`] distinct rects the
//! tracker gives up and the level keeps its staging: the memory it would take to
//! stay exact is the memory the drop exists to save.

use crate::dirty_rect::DirtyRect;

/// Distinct rects tracked per level before coverage stops being followed.
pub const MAX_TRACKED_RECTS: usize = 32;

/// What the writes since a level's staging was allocated cover of that level.
pub struct StagingCoverage {
    /// Recorded writes, clamped to the level, none contained in another.
    ///
    /// Empty in every state but `Partial`: both terminal states are answers,
    /// and holding the rects past them would keep memory this exists to free.
    rects: Vec<DirtyRect>,
    /// Sum of the recorded rect areas, an upper bound on the union area.
    ///
    /// Under the level's area it proves the union cannot cover the level, which
    /// keeps the sweep off the common path.
    written_area: u64,
    state: CoverageState,
}

impl Default for StagingCoverage {
    fn default() -> Self {
        Self::new()
    }
}

impl StagingCoverage {
    /// A level with nothing written into its staging yet.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            rects: Vec::new(),
            written_area: 0,
            state: CoverageState::Partial,
        }
    }

    /// Whether every texel of the level has been written since its staging was allocated.
    #[must_use]
    pub const fn is_full(&self) -> bool {
        matches!(self.state, CoverageState::Full)
    }

    /// Forget every recorded write.
    ///
    /// Called when the staging goes away or its contents stop describing the
    /// level: a release, a re-create, or a rename that preserved nothing.
    pub fn reset(&mut self) {
        self.rects = Vec::new();
        self.written_area = 0;
        self.state = CoverageState::Partial;
    }

    /// Record a write covering the whole level.
    pub fn mark_full(&mut self) {
        self.rects = Vec::new();
        self.written_area = 0;
        self.state = CoverageState::Full;
    }

    /// Record a write of `rect`, clamped to the level.
    ///
    /// Reaching full coverage is sticky: a further write into a fully written
    /// level leaves it fully written.
    pub fn add(&mut self, rect: DirtyRect, level_w: u32, level_h: u32) {
        let Some(rect) = rect.clamp(level_w, level_h) else {
            return;
        };
        if rect.x == 0 && rect.y == 0 && rect.w >= level_w && rect.h >= level_h {
            self.mark_full();
            return;
        }
        if self.state != CoverageState::Partial {
            return;
        }
        if self.rects.iter().any(|r| contains(*r, rect)) {
            return;
        }
        if self.rects.len() >= MAX_TRACKED_RECTS {
            self.rects = Vec::new();
            self.state = CoverageState::Untracked;
            mtld3d_shared::log_once_info!(
                target: crate::LOG_TARGET,
                "staging coverage: more than {MAX_TRACKED_RECTS} distinct written rects on one \
                 level, coverage no longer tracked and the staging stays resident"
            );
            return;
        }
        self.rects.push(rect);
        self.written_area = self
            .written_area
            .saturating_add(u64::from(rect.w) * u64::from(rect.h));
        if self.written_area >= u64::from(level_w) * u64::from(level_h)
            && covers_level(&self.rects, level_w, level_h)
        {
            self.mark_full();
        }
    }
}

/// How much of a level the recorded rects are known to cover.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CoverageState {
    /// Some texels of the level have not been written yet.
    Partial,
    /// Every texel has been written.
    Full,
    /// Too many distinct rects arrived; coverage is not followed until a reset.
    Untracked,
}

/// Whether `outer` covers every texel of `inner`.
const fn contains(outer: DirtyRect, inner: DirtyRect) -> bool {
    outer.x <= inner.x
        && outer.y <= inner.y
        && outer.x + outer.w >= inner.x + inner.w
        && outer.y + outer.h >= inner.y + inner.h
}

/// Whether the union of `rects` covers `level_w` × `level_h` exactly.
///
/// Sweeps the horizontal bands the rect edges cut the level into: within one
/// band every rect either spans it fully or not at all, so the band is covered
/// when the rects spanning it, walked in ascending `x`, reach `level_w` with no
/// gap. Every rect is expected clamped to the level.
fn covers_level(rects: &[DirtyRect], level_w: u32, level_h: u32) -> bool {
    let mut bands: Vec<u32> = Vec::with_capacity(rects.len() * 2);
    for r in rects {
        bands.push(r.y);
        bands.push(r.y + r.h);
    }
    bands.sort_unstable();
    bands.dedup();
    // A band boundary missing at the top or the bottom edge is a row of the
    // level no rect touched, so the sweep has nothing to prove.
    if bands.first() != Some(&0) || bands.last() != Some(&level_h) {
        return false;
    }
    let mut ordered: Vec<DirtyRect> = rects.to_vec();
    ordered.sort_unstable_by_key(|r| r.x);
    bands.windows(2).all(|band| {
        let (top, bottom) = (band[0], band[1]);
        let mut reach = 0;
        for r in &ordered {
            if r.y > top || r.y + r.h < bottom {
                continue;
            }
            if r.x > reach {
                break;
            }
            reach = reach.max(r.x + r.w);
        }
        reach >= level_w
    })
}

#[cfg(test)]
mod tests;
