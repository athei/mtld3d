//! Pooled GPU versions of textures renamed between the draws that sample them.
//!
//! An upload into a texture that a draw earlier in the same submission
//! sampled cannot write the live `MTLTexture`: the upload runs at the head
//! of the submission, before that draw. The encoder renames the texture
//! instead, and the version the draw read retires once the GPU has
//! finished every submission that named it. A texture rewritten between
//! draws every frame (a sprite scratch the game relocks once per
//! character) would otherwise allocate one texture per rewrite and destroy
//! it a few frames later. This module keeps the retired versions for the
//! next rename of the same texture.
//!
//! A version goes through three states. It is live while the texture cache
//! names it; a rename moves it to the retiring queue under the submission
//! that last named it; once both retirement counters have passed that
//! submission the caller pops it, forgets every handle-keyed record of it
//! exactly as a destroy does, and parks it as a spare of its texture, or
//! destroys it when the texture is gone or the pool is at its cap. A spare
//! nobody takes for [`SPARE_MAX_AGE_SUBMITS`] submissions is trimmed, and a
//! texture with no spares that no rename touched for as long stops being
//! tracked: its next rename starts it over, write history included.
//!
//! A reused version still has to receive the texels the texture changed
//! since it retired, and only those, when the texture's GPU writes are all
//! known. For a 2D or cube colour texture the application did not create
//! as a render target or depth stencil they are: D3D9 refuses it as a
//! render target and as the destination of `StretchRect` and `ColorFill`,
//! so its only writers are uploads and mip regeneration. Such a texture
//! keeps a write counter and, per slice and level, a grid of the counter
//! value of the last write that touched each tile. A version records the
//! counter when it retires, and the preserve copy into it covers the
//! bounding box of the tiles stamped later. Every version stays a complete
//! snapshot of the texture at the point it was live.
//!
//! Pure bookkeeping, generic over the caller's view handles: no Metal
//! calls and no blit shapes, so the retirement order, the pool bounds and
//! the copy plan are host-testable.

use std::collections::{VecDeque, hash_map::Entry};

use mtld3d_shared::{
    TextureCreateDesc,
    mtl::{BlockLayout, PixelFormat, StorageMode, Swizzle, TextureCreateFlags, TextureUsage},
};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::{format::FormatMapping, ids::TextureId};

/// Bytes of texture storage the pool may park across every texture.
///
/// The busiest measured scene rewrites one 512x512 A4R4G4B4 sprite scratch
/// about 268 times a frame, so it retires about 134 MiB of versions per
/// frame, each reclaimed two to three frames later. 256 MiB holds about two
/// frames of that. Every spare stands in for an allocation the rename
/// would otherwise make in the same frame, so in steady state the pool
/// adds no storage beyond what the renames already held; the cap bounds
/// the transient after a burst of renames stops. The storage is `Private`
/// GPU memory, outside the 32-bit guest address space.
pub const VERSION_POOL_CAP_BYTES: u64 = 256 << 20;

/// Submissions a spare may sit unused before it is destroyed.
///
/// About two seconds at 60 frames a second: a texture that stopped being
/// renamed gives its storage back that much later, while one renamed in
/// bursts a second apart (a scene change, a loading screen) keeps it.
pub const SPARE_MAX_AGE_SUBMITS: u64 = 120;

/// Smallest tile edge of a write-history grid, in texels.
///
/// A sprite-sized write (tens of texels on a side) covers a handful of
/// tiles, so the copy it causes overshoots by at most one tile edge on each
/// side, while a small level keeps a grid of a few cells.
const MIN_TILE_TEXELS: u32 = 16;

/// Most tiles a write-history grid has along an axis.
///
/// Bounds the grid at 1024 stamps (8 KiB) per subresource whatever the
/// texture's size, and bounds the scan a rename makes over it; a 4096-texel
/// level gets 128-texel tiles.
const MAX_TILES_PER_AXIS: u32 = 32;

/// The creation shape a pooled version must share with the texture it serves.
///
/// Every field of the creation descriptor except the texture id, which
/// only labels the object. A texture's descriptor does not change over its
/// life, so a mismatch means the caller's descriptor changed and the spares
/// made for the old one cannot serve it.
#[derive(PartialEq, Eq)]
pub struct VersionShape {
    width: u32,
    height: u32,
    depth: u32,
    levels: u32,
    pixel_format: PixelFormat,
    storage_mode: StorageMode,
    flags: TextureCreateFlags,
    usage: TextureUsage,
    swizzle: [Swizzle; 4],
}

impl VersionShape {
    /// The shape of the texture `desc` creates.
    #[must_use]
    pub const fn of(desc: &TextureCreateDesc) -> Self {
        Self {
            width: desc.width,
            height: desc.height,
            depth: desc.depth,
            levels: desc.levels,
            pixel_format: desc.pixel_format,
            storage_mode: desc.storage_mode,
            flags: desc.flags,
            usage: desc.usage_flags,
            swizzle: [
                desc.swizzle_r,
                desc.swizzle_g,
                desc.swizzle_b,
                desc.swizzle_a,
            ],
        }
    }

    /// Array slices of the texture: six faces for a cube, one otherwise.
    #[must_use]
    pub const fn slices(&self) -> u32 {
        if self.flags.contains(TextureCreateFlags::TYPE_CUBE) {
            6
        } else {
            1
        }
    }

    /// Storage one version of this shape occupies, every slice and level counted.
    ///
    /// Each level is measured in whole blocks of the pixel format, so a
    /// compressed level narrower than a block still costs one block.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        let block = self.pixel_format.block_layout();
        let mut total = 0u64;
        for level in 0..self.levels {
            let w = level_extent(self.width, level).div_ceil(block.width());
            let h = level_extent(self.height, level).div_ceil(block.height());
            let d = if self.flags.contains(TextureCreateFlags::TYPE_3D) {
                level_extent(self.depth, level)
            } else {
                1
            };
            total += u64::from(w) * u64::from(h) * u64::from(d) * u64::from(block.bytes());
        }
        total * u64::from(self.slices())
    }
}

/// Whether a texture's write history can bound the preserve copies into its reused versions.
///
/// True for a 2D or cube colour texture whose application usage
/// (`application_usage`, before the encoder adds the render-target bit an
/// upload pass needs) has neither the render-target nor the depth-stencil
/// bit: D3D9 lets nothing but uploads and mip regeneration write one. A
/// volume, a depth texture and a render target take whole-level copies.
#[must_use]
pub fn records_every_write(desc: &TextureCreateDesc, application_usage: TextureUsage) -> bool {
    !desc.flags.contains(TextureCreateFlags::TYPE_3D)
        && !application_usage.intersects(TextureUsage::RENDER_TARGET | TextureUsage::DEPTH_STENCIL)
        && !matches!(
            desc.pixel_format,
            PixelFormat::Depth32Float | PixelFormat::Depth32FloatStencil8
        )
}

/// A rectangle of texels in one level of a texture.
#[derive(Debug, PartialEq, Eq)]
pub struct TexelRect {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

impl TexelRect {
    #[must_use]
    pub const fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// The whole of a level `width` by `height` texels.
    #[must_use]
    pub const fn level(width: u32, height: u32) -> Self {
        Self::new(0, 0, width, height)
    }

    #[must_use]
    pub const fn x(&self) -> u32 {
        self.x
    }

    #[must_use]
    pub const fn y(&self) -> u32 {
        self.y
    }

    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }
}

/// The texels an upload of `rect` writes into a level `extent` texels in size.
///
/// `compressed` is the source format's mapping for a block-compressed
/// upload, `None` for one by pixel. A compressed rectangle off the block
/// grid ([`FormatMapping::rect_on_block_grid`], the check the blit path
/// makes) has no blit form, so the upload copies the whole level from the
/// staging instead, and that is what it writes. Every other upload writes
/// `rect`.
#[must_use]
pub const fn upload_written_rect(
    rect: TexelRect,
    extent: (u32, u32),
    compressed: Option<&FormatMapping>,
) -> TexelRect {
    let on_grid = match compressed {
        Some(format) => {
            format.rect_on_block_grid((rect.x, rect.y), (rect.width, rect.height), extent)
        }
        None => true,
    };
    if on_grid {
        rect
    } else {
        TexelRect::level(extent.0, extent.1)
    }
}

/// The subresource an upload writes, as the copy plan needs it.
pub struct UploadFootprint {
    slice: u32,
    level: u32,
    full_cover: bool,
}

impl UploadFootprint {
    /// An upload into `slice`, `level`; `full_cover` when it rewrites that whole subresource.
    #[must_use]
    pub const fn new(slice: u32, level: u32, full_cover: bool) -> Self {
        Self {
            slice,
            level,
            full_cover,
        }
    }
}

/// One preserve copy from the live version into the version a rename makes live.
pub struct PreserveCopy {
    slice: u32,
    level: u32,
    region: Option<TexelRect>,
}

impl PreserveCopy {
    #[must_use]
    pub const fn slice(&self) -> u32 {
        self.slice
    }

    #[must_use]
    pub const fn level(&self) -> u32 {
        self.level
    }

    /// The rectangle to copy, or `None` for the whole level (every depth slice of a volume).
    #[must_use]
    pub const fn region(&self) -> Option<&TexelRect> {
        self.region.as_ref()
    }
}

/// A retired version of a texture, kept for the texture's next rename.
pub struct Spare<V> {
    views: V,
    held_writes: Option<u64>,
    last_seq: u64,
}

impl<V> Spare<V> {
    /// The version's view handles, consuming the spare.
    #[must_use]
    pub fn into_views(self) -> V {
        self.views
    }

    /// The view handles, and the texture's write count the version holds every write of.
    ///
    /// `None` when its contents are unknown: a submission it was live
    /// through failed, so a copy into it or an upload it carried may never
    /// have run. [`TextureVersions::preserve_plan`] then copies every level.
    #[must_use]
    pub fn into_parts(self) -> (V, Option<u64>) {
        (self.views, self.held_writes)
    }
}

/// A version a rename replaced, waiting for the GPU to finish with it.
pub struct Retiring<V> {
    id: TextureId,
    views: V,
    seq: u64,
    epoch: u64,
    birth_seq: u64,
    held_writes: u64,
}

impl<V> Retiring<V> {
    /// The view handles every handle-keyed record must forget before reuse.
    #[must_use]
    pub const fn views(&self) -> &V {
        &self.views
    }
}

/// The pool and retirement queue of every renamed texture.
///
/// Created empty; a texture enters at its first rename. `V` is whatever
/// the caller destroys or binds a version through (the encoder's set of
/// view handles); this type never looks inside it.
pub struct TextureVersions<V> {
    textures: FxHashMap<TextureId, VersionState<V>>,
    retiring: VecDeque<Retiring<V>>,
    parked_bytes: u64,
    /// The epoch the next tracking state gets; 0 is never handed out.
    ///
    /// Every state, a texture's first and every one after a shape change
    /// or after [`Self::forget_texture`], takes a new value, so a version
    /// retired under one state never matches another, even when the same
    /// `TextureId` comes back (an `EvictManagedResources`, a texture moving
    /// between devices) while that version is still retiring.
    next_epoch: u64,
    /// The textures with at least one spare, the only ones [`Self::trim`] visits for spares.
    with_spares: FxHashSet<TextureId>,
    /// One `(submission, texture, epoch)` per tracked state, oldest first.
    ///
    /// [`Self::trim`] looks at a state when its entry is
    /// [`SPARE_MAX_AGE_SUBMITS`] old, drops it when it has no spares and no
    /// rename since, and queues it again otherwise, so a drain visits only
    /// the states whose age came due. An entry whose epoch no longer
    /// matches its texture's state belongs to a state that is gone.
    idle: VecDeque<(u64, TextureId, u64)>,
}

impl<V> TextureVersions<V> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            textures: FxHashMap::default(),
            retiring: VecDeque::new(),
            parked_bytes: 0,
            next_epoch: 1,
            with_spares: FxHashSet::default(),
            idle: VecDeque::new(),
        }
    }

    /// Bytes of storage parked as spares across every texture.
    #[must_use]
    pub const fn parked_bytes(&self) -> u64 {
        self.parked_bytes
    }

    /// Whether `id` keeps a write history, so its writes must be reported.
    #[must_use]
    pub fn records_writes(&self, id: TextureId) -> bool {
        self.textures
            .get(&id)
            .is_some_and(|state| state.history.is_some())
    }

    /// Start a rename of `id`, handing back its newest spare if it has one.
    ///
    /// A texture seen for the first time starts tracking here with no
    /// spares, and with a write history when `records_writes` (see
    /// [`records_every_write`]). History starts empty: every version from
    /// here on holds whatever was written before, so only later writes are
    /// stamped. When `shape` differs from the shape the texture's spares
    /// were made for, those spares come back in the second value for the
    /// caller to destroy behind the current submission, and the texture
    /// starts over under the new shape; versions of the old shape still
    /// retiring are destroyed when they are popped. `seq` is the submission
    /// the rename belongs to, which keeps the texture tracked for another
    /// [`SPARE_MAX_AGE_SUBMITS`] submissions.
    pub fn take_spare(
        &mut self,
        id: TextureId,
        shape: VersionShape,
        records_writes: bool,
        seq: u64,
    ) -> (Option<Spare<V>>, Vec<V>) {
        let mut discarded = Vec::new();
        let epoch = self.next_epoch;
        let state = match self.textures.entry(id) {
            Entry::Vacant(slot) => {
                self.next_epoch += 1;
                self.idle.push_back((seq, id, epoch));
                slot.insert(VersionState::new(shape, epoch, records_writes))
            }
            Entry::Occupied(slot) => {
                let state = slot.into_mut();
                if state.shape != shape {
                    self.parked_bytes -= state.bytes * state.spares.len() as u64;
                    discarded.extend(state.spares.drain(..).map(Spare::into_views));
                    self.next_epoch += 1;
                    self.idle.push_back((seq, id, epoch));
                    *state = VersionState::new(shape, epoch, records_writes);
                }
                state
            }
        };
        state.last_rename_seq = seq;
        let spare = state.spares.pop();
        if spare.is_some() {
            self.parked_bytes -= state.bytes;
        }
        if state.spares.is_empty() {
            self.with_spares.remove(&id);
        }
        (spare, discarded)
    }

    /// The copies that bring a rename's new version up to the live one.
    ///
    /// `held_writes` is what [`Spare::into_parts`] reported for a reused
    /// spare, and `None` for a fresh allocation or a spare of unknown
    /// contents: every subresource is then copied whole. A known spare of a
    /// texture with a write history gets, per subresource, the bounding box
    /// of the tiles written after it retired, and nothing where no tile was.
    /// Either way the subresource the upload rewrites entirely is skipped.
    /// A box the upload's own rectangle covers is still copied: the upload
    /// can decline after the rename, and the version must be whole without
    /// it. `shape` is the texture's shape; a texture [`Self::take_spare`]
    /// never saw gets every subresource whole.
    #[must_use]
    pub fn preserve_plan(
        &self,
        id: TextureId,
        shape: &VersionShape,
        held_writes: Option<u64>,
        upload: &UploadFootprint,
    ) -> Vec<PreserveCopy> {
        let mut plan = Vec::new();
        let history = self.textures.get(&id).map_or_else(
            || {
                mtld3d_shared::log_once_warn!(
                    target: crate::LOG_TARGET,
                    "texture versions: a copy plan for untracked texture {:#x} → whole-level \
                     copies",
                    id.raw(),
                );
                None
            },
            |state| state.history.as_ref(),
        );
        let since = history.zip(held_writes);
        for slice in 0..shape.slices() {
            for level in 0..shape.levels {
                let rewritten = slice == upload.slice && level == upload.level;
                if rewritten && upload.full_cover {
                    continue;
                }
                let Some((history, held)) = since else {
                    plan.push(PreserveCopy {
                        slice,
                        level,
                        region: None,
                    });
                    continue;
                };
                let grid = history.grid(slice, level);
                let Some(dirty) = grid.written_since(held) else {
                    continue;
                };
                let whole = TexelRect::level(grid.width, grid.height);
                plan.push(PreserveCopy {
                    slice,
                    level,
                    region: (dirty != whole).then_some(dirty),
                });
            }
        }
        plan
    }

    /// Queue the version a rename of `id` just replaced, retiring behind submission `seq`.
    ///
    /// `seq` is the submission being built, which holds every command that
    /// named the version: the draws that sampled it and the copy out of it.
    /// It must not go backwards across calls, since [`Self::pop_retired`]
    /// gates on the front entry alone; the encoder's submission counter only
    /// advances. The version holds every write the texture's history has
    /// counted so far. The version that replaces it is live from `seq` on.
    /// A texture [`Self::take_spare`] never saw is not tracked, and its
    /// version is still queued: it is destroyed when popped.
    pub fn retire(&mut self, id: TextureId, views: V, seq: u64) {
        debug_assert!(
            self.retiring.back().is_none_or(|back| back.seq <= seq),
            "texture versions must retire in submission order",
        );
        let (epoch, birth_seq, held_writes) =
            self.textures.get_mut(&id).map_or((0, 0, 0), |state| {
                let birth = core::mem::replace(&mut state.live_birth_seq, seq);
                (
                    state.epoch,
                    birth,
                    state.history.as_ref().map_or(0, |h| h.writes),
                )
            });
        self.retiring.push_back(Retiring {
            id,
            views,
            seq,
            epoch,
            birth_seq,
            held_writes,
        });
    }

    /// Stamp a write of `rect` into `slice`, `level` of `id`.
    ///
    /// Clamped to the level; a no-op for a texture without a write history.
    /// The caller reports every upload it emitted into the live version,
    /// replays of aborted ones included, with the rectangle the upload
    /// actually writes ([`upload_written_rect`]). A subresource the
    /// texture's shape does not have means the history no longer describes
    /// the texture: it is dropped, so every later copy is whole.
    pub fn note_write(&mut self, id: TextureId, slice: u32, level: u32, rect: &TexelRect) {
        let Some(state) = self.textures.get_mut(&id) else {
            return;
        };
        let Some(history) = state.history.as_mut() else {
            return;
        };
        if slice >= history.slices || level >= history.levels {
            mtld3d_shared::log_once_warn!(
                target: crate::LOG_TARGET,
                "texture versions: a write to slice {slice} level {level} of texture {:#x}, \
                 which has {} slices of {} levels → write history dropped, whole-level copies",
                id.raw(),
                history.slices,
                history.levels,
            );
            state.history = None;
            return;
        }
        history.writes += 1;
        let stamp = history.writes;
        history.grid_mut(slice, level).stamp(rect, stamp);
    }

    /// Stamp a mip regeneration of `id`: every level but the first, on every slice.
    pub fn note_mip_regeneration(&mut self, id: TextureId) {
        let Some(history) = self
            .textures
            .get_mut(&id)
            .and_then(|state| state.history.as_mut())
        else {
            return;
        };
        history.writes += 1;
        let stamp = history.writes;
        let levels = history.levels.max(1) as usize;
        for grids in history.grids.chunks_exact_mut(levels) {
            for grid in &mut grids[1..] {
                let whole = TexelRect::level(grid.width, grid.height);
                grid.stamp(&whole, stamp);
            }
        }
    }

    /// Pop the oldest retiring version whose submission `settled` has passed.
    ///
    /// `settled` is the lower of the two retirement counters, the gate the
    /// encoder's retention drain uses, so no command buffer that named the
    /// version is still running. The caller forgets the version's
    /// handle-keyed records before handing it to [`Self::park`].
    pub fn pop_retired(&mut self, settled: u64) -> Option<Retiring<V>> {
        if self.retiring.front()?.seq > settled {
            return None;
        }
        self.retiring.pop_front()
    }

    /// Park a popped version as its texture's newest spare, or hand it back to destroy.
    ///
    /// It comes back when its texture is gone or changed shape since it
    /// retired, or when parking it would take the pool past
    /// [`VERSION_POOL_CAP_BYTES`]. `failed` is the highest submission the
    /// GPU aborted (0 for none), read with the counter the pop was gated
    /// on: a version live through an aborted submission may lack a copy or
    /// an upload that submission carried, so it parks with unknown contents.
    pub fn park(&mut self, entry: Retiring<V>, failed: u64) -> Option<V> {
        let Some(state) = self.textures.get_mut(&entry.id) else {
            return Some(entry.views);
        };
        if state.epoch != entry.epoch {
            return Some(entry.views);
        }
        if self.parked_bytes + state.bytes > VERSION_POOL_CAP_BYTES {
            mtld3d_shared::log_once_info!(
                target: crate::LOG_TARGET,
                "texture versions: the rename pool is at its {} MiB cap; further retired \
                 versions are destroyed until spares are taken or trimmed",
                VERSION_POOL_CAP_BYTES >> 20,
            );
            return Some(entry.views);
        }
        self.parked_bytes += state.bytes;
        let aborted = failed != 0 && entry.birth_seq <= failed;
        if state.spares.is_empty() {
            self.with_spares.insert(entry.id);
        }
        state.spares.push(Spare {
            views: entry.views,
            held_writes: (!aborted).then_some(entry.held_writes),
            last_seq: entry.seq,
        });
        None
    }

    /// Take every spare no rename used for [`SPARE_MAX_AGE_SUBMITS`] submissions before `now`.
    ///
    /// A spare's age counts from the submission it last served in. Spares
    /// park in submission order and renames take the newest, so each
    /// texture's oldest spares sit at the bottom of its stack. Only the
    /// textures holding spares are visited for that. Then every texture
    /// left without spares and without a rename for as long stops being
    /// tracked, its write history with it; only the states whose age came
    /// due are looked at.
    pub fn trim(&mut self, now: u64) -> Vec<V> {
        let mut aged = Vec::new();
        let aged_at = |seq: u64| seq + SPARE_MAX_AGE_SUBMITS < now;
        let (textures, parked_bytes) = (&mut self.textures, &mut self.parked_bytes);
        self.with_spares.retain(|id| {
            let Some(state) = textures.get_mut(id) else {
                return false;
            };
            let old = state
                .spares
                .iter()
                .take_while(|spare| aged_at(spare.last_seq))
                .count();
            *parked_bytes -= state.bytes * old as u64;
            aged.extend(state.spares.drain(..old).map(Spare::into_views));
            !state.spares.is_empty()
        });
        while let Some(&(seq, id, epoch)) = self.idle.front() {
            if !aged_at(seq) {
                break;
            }
            self.idle.pop_front();
            let Some(state) = self.textures.get(&id) else {
                continue;
            };
            if state.epoch != epoch {
                continue;
            }
            if aged_at(state.last_rename_seq) && state.spares.is_empty() {
                self.textures.remove(&id);
                continue;
            }
            // Renamed since, or holding a spare that parked late: look again
            // once the newer of the two has aged.
            let again = if aged_at(state.last_rename_seq) {
                now
            } else {
                state.last_rename_seq
            };
            self.idle.push_back((again, id, epoch));
        }
        aged
    }

    /// Stop tracking a released texture, handing back its spares.
    ///
    /// Its versions still retiring stay queued and are destroyed when
    /// popped: a texture that comes back under the same id starts a state
    /// of a new epoch, which they do not match.
    pub fn forget_texture(&mut self, id: TextureId) -> Vec<V> {
        let Some(state) = self.textures.remove(&id) else {
            return Vec::new();
        };
        self.with_spares.remove(&id);
        self.parked_bytes -= state.bytes * state.spares.len() as u64;
        state.spares.into_iter().map(Spare::into_views).collect()
    }

    /// Take every spare and every retiring version, leaving the textures tracked.
    ///
    /// For `Reset` and shutdown, whose caller waits for the GPU to go idle
    /// before it destroys what this returns.
    pub fn drain_versions(&mut self) -> Vec<V> {
        let mut all: Vec<V> = self.retiring.drain(..).map(|entry| entry.views).collect();
        for id in self.with_spares.drain() {
            if let Some(state) = self.textures.get_mut(&id) {
                all.extend(state.spares.drain(..).map(Spare::into_views));
            }
        }
        self.parked_bytes = 0;
        all
    }

    /// Textures tracked, with or without spares.
    #[cfg(test)]
    fn tracked(&self) -> usize {
        self.textures.len()
    }

    /// Textures holding at least one spare.
    #[cfg(test)]
    fn holding_spares(&self) -> usize {
        self.with_spares.len()
    }
}

impl<V> Default for TextureVersions<V> {
    fn default() -> Self {
        Self::new()
    }
}

/// One tracked texture: its shape, its write history and its spares, newest last.
///
/// `live_birth_seq` is the submission the live version became live in, 0
/// for the version that was live when tracking started (it may have lived
/// through any earlier failure). `last_rename_seq` is the submission of
/// the texture's latest rename.
struct VersionState<V> {
    shape: VersionShape,
    epoch: u64,
    bytes: u64,
    history: Option<WriteHistory>,
    live_birth_seq: u64,
    last_rename_seq: u64,
    spares: Vec<Spare<V>>,
}

impl<V> VersionState<V> {
    fn new(shape: VersionShape, epoch: u64, records_writes: bool) -> Self {
        Self {
            bytes: shape.bytes(),
            history: records_writes.then(|| WriteHistory::new(&shape)),
            shape,
            epoch,
            live_birth_seq: 0,
            last_rename_seq: 0,
            spares: Vec::new(),
        }
    }
}

/// A texture's write counter and one stamp grid per slice and level, slice-major.
struct WriteHistory {
    slices: u32,
    levels: u32,
    writes: u64,
    grids: Box<[StampGrid]>,
}

impl WriteHistory {
    fn new(shape: &VersionShape) -> Self {
        let block = shape.pixel_format.block_layout();
        let slices = shape.slices();
        let grids = (0..slices)
            .flat_map(|_| 0..shape.levels)
            .map(|level| {
                StampGrid::new(
                    level_extent(shape.width, level),
                    level_extent(shape.height, level),
                    block,
                )
            })
            .collect();
        Self {
            slices,
            levels: shape.levels,
            writes: 0,
            grids,
        }
    }

    fn grid(&self, slice: u32, level: u32) -> &StampGrid {
        &self.grids[(slice * self.levels + level) as usize]
    }

    fn grid_mut(&mut self, slice: u32, level: u32) -> &mut StampGrid {
        &mut self.grids[(slice * self.levels + level) as usize]
    }
}

/// The write stamps of one subresource, one per tile.
///
/// A tile edge is at least [`MIN_TILE_TEXELS`], a whole number of the
/// format's blocks, and long enough that an axis has at most
/// [`MAX_TILES_PER_AXIS`] tiles, so a tile-aligned rectangle clamped to the
/// level is a legal copy region for a compressed format too. `newest` lets
/// a subresource nothing wrote since a version retired skip the scan.
struct StampGrid {
    width: u32,
    height: u32,
    tile_width: u32,
    tile_height: u32,
    columns: u32,
    rows: u32,
    newest: u64,
    stamps: Box<[u64]>,
}

impl StampGrid {
    fn new(width: u32, height: u32, block: BlockLayout) -> Self {
        let tile_width = tile_edge(width, block.width());
        let tile_height = tile_edge(height, block.height());
        let columns = width.div_ceil(tile_width);
        let rows = height.div_ceil(tile_height);
        Self {
            width,
            height,
            tile_width,
            tile_height,
            columns,
            rows,
            newest: 0,
            stamps: vec![0; (columns * rows) as usize].into_boxed_slice(),
        }
    }

    /// Set every tile `rect` touches, clamped to the level, to `stamp`.
    fn stamp(&mut self, rect: &TexelRect, stamp: u64) {
        let x_end = rect.x.saturating_add(rect.width).min(self.width);
        let y_end = rect.y.saturating_add(rect.height).min(self.height);
        if rect.x >= x_end || rect.y >= y_end {
            return;
        }
        let (first_column, last_column) = (rect.x / self.tile_width, (x_end - 1) / self.tile_width);
        let (first_row, last_row) = (rect.y / self.tile_height, (y_end - 1) / self.tile_height);
        for row in first_row..=last_row {
            let start = (row * self.columns + first_column) as usize;
            let end = (row * self.columns + last_column) as usize;
            self.stamps[start..=end].fill(stamp);
        }
        self.newest = stamp;
    }

    /// The tile-aligned bounding box, clamped to the level, of the tiles stamped after `held`.
    fn written_since(&self, held: u64) -> Option<TexelRect> {
        if self.newest <= held {
            return None;
        }
        let mut columns = (u32::MAX, 0);
        let mut rows = (u32::MAX, 0);
        for (row, line) in (0..self.rows).zip(self.stamps.chunks_exact(self.columns as usize)) {
            for (column, &stamp) in (0..self.columns).zip(line) {
                if stamp > held {
                    columns = (columns.0.min(column), columns.1.max(column));
                    rows = (rows.0.min(row), rows.1.max(row));
                }
            }
        }
        let x = columns.0 * self.tile_width;
        let y = rows.0 * self.tile_height;
        let x_end = ((columns.1 + 1) * self.tile_width).min(self.width);
        let y_end = ((rows.1 + 1) * self.tile_height).min(self.height);
        Some(TexelRect::new(x, y, x_end - x, y_end - y))
    }
}

/// Tile edge along an axis of `extent` texels in a format with `block`-texel blocks.
fn tile_edge(extent: u32, block: u32) -> u32 {
    extent
        .div_ceil(MAX_TILES_PER_AXIS)
        .max(MIN_TILE_TEXELS)
        .next_multiple_of(block)
}

/// Extent of mip `level` along an axis whose level-0 extent is `base`.
fn level_extent(base: u32, level: u32) -> u32 {
    (base.max(1) >> level).max(1)
}

#[cfg(test)]
mod tests;
