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
//! exactly as a destroy does, and parks it as a spare of its texture. A
//! spare no rename takes within [`SPARE_MAX_IDLE_SUBMITS`] submissions is
//! destroyed, and so is a version whose texture is gone or changed shape, a
//! released texture's spares, and a version whose parking would take the
//! spares past the [`VersionBudget`] cap or that retires while the process
//! is short of GPU memory, when spares are given back oldest first as well.
//! Those destroys wait in a
//! queue the caller takes a batch from per drain
//! ([`TextureVersions::take_doomed`]): the recent rate of arrivals plus an
//! eighth of the queue, so a steady stream never keeps a queue and a burst
//! spreads over several drains. A texture with no spares that no rename touched for
//! [`STATE_MAX_IDLE_SUBMITS`] submissions stops being tracked: its next
//! rename starts it over, write history included.
//!
//! A reused version still has to receive the texels the texture changed
//! since it retired, and only those, when the texture's GPU writes are all
//! known. For a 2D or cube colour texture the application did not create
//! as a render target or depth stencil they are: D3D9 refuses it as a
//! render target and as a `ColorFill` destination, and a plain device as a
//! `StretchRect` destination too, so its writers are uploads, mip
//! regeneration and an extended device's whole-surface `StretchRect`, which
//! the encoder records like an upload. Such a texture
//! keeps a write counter and, per slice and level, a grid of the counter
//! value of the last write that touched each tile. A version records the
//! counter when it retires, and the preserve copy into it covers the
//! bounding box of the tiles stamped later. Every version stays a complete
//! snapshot of the texture at the point it was live.
//!
//! Pure bookkeeping, generic over the caller's view handles: no Metal
//! calls and no blit shapes, so the retirement order, the pool bounds and
//! the copy plan are host-testable.

use std::collections::{BTreeSet, VecDeque, hash_map::Entry};

use mtld3d_shared::{
    TextureCreateDesc,
    mtl::{BlockLayout, PixelFormat, StorageMode, Swizzle, TextureCreateFlags, TextureUsage},
};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::{format::FormatMapping, ids::TextureId};

/// Submissions a spare may wait for a rename before it is destroyed.
///
/// The bound that sizes the pool to the demand. A version is parked once
/// the GPU has finished its submission: in the busiest measured scene (a
/// 512x512 A4R4G4B4 sprite scratch rewritten 267 times a frame in one
/// submission, about 134 MiB of versions retired per frame) one to eight
/// submissions after it retired, three quarters of them four to eight. The
/// age counts from parking, so that lag does not use it up, and a texture
/// still being renamed takes its newest spares first, leaving the oldest
/// to age out when it renames less. Eight gives a texture that stopped being renamed its
/// storage back within about 130 ms at 60 frames a second. Counted from
/// retiring instead, the same bound destroyed about 87 versions a frame
/// there. A total-bytes cap does not size this: the pool a
/// scene needs is the per-frame retire times the frames the GPU lags, and
/// the lag grows as the CPU outruns the GPU; a 256 MiB cap destroyed and
/// reallocated about 110 versions a frame there, and 512 MiB still 45.
pub const SPARE_MAX_IDLE_SUBMITS: u64 = 8;

/// Most bytes of texture storage the pool holds as spares on any device.
///
/// Not the bound that normally acts: [`SPARE_MAX_IDLE_SUBMITS`] does. It is
/// the age bound's own worst case in the busiest measured scene (eight
/// submissions of 134 MiB of retired versions, about 1 GiB). Measured there,
/// the spares held 267 to 534 MiB while the scene ran and reached 1 GiB only
/// once the GPU caught up with every submission at its end. Versions already
/// queued to be destroyed do not count: they are committed, leave within a
/// few drains, and counting them would keep the pool turning versions away
/// for as long as that queue fed itself.
pub const VERSION_POOL_CEILING_BYTES: u64 = 1 << 30;

/// Divisor of the device's recommended working set that gives the spares' cap below the ceiling.
///
/// An eighth keeps the pool's share of a small machine's GPU memory small:
/// about 683 MiB of an 8 GB Apple silicon Mac's 5.3 GiB, 192 MiB of an
/// integrated Intel GPU's 1.5 GiB and 512 MiB of a 4 GB discrete GPU, while
/// every machine with a working set of 8 GiB or more (a 16 GB Mac and up)
/// reaches [`VERSION_POOL_CEILING_BYTES`].
pub const VERSION_POOL_WORKING_SET_DIVISOR: u64 = 8;

/// Quarters of the device's recommended working set the process may allocate before spares go.
///
/// Above three quarters of it the process is close to where Metal starts to
/// page or fail allocations, and spares are the cheapest memory to give
/// back: nothing names them, and losing one costs the next rename an
/// allocation and a whole copy, what it cost before the pool. Parking
/// resumes only below the line less the cap, so a process hovering at the
/// line does not park and destroy in turn. `currentAllocatedSize` may also
/// count the no-copy buffers that wrap upload staging, so on a discrete GPU
/// the line can trip early; that costs speed only, never texels.
pub const VERSION_POOL_PRESSURE_QUARTERS: u64 = 3;

/// How much GPU memory the pool may use, from the device's recommended working set.
///
/// Read once when the device is created. The cap is the smaller of
/// [`VERSION_POOL_CEILING_BYTES`] and the working set divided by
/// [`VERSION_POOL_WORKING_SET_DIVISOR`]; past the pressure line,
/// [`VERSION_POOL_PRESSURE_QUARTERS`] of the working set, the pool parks
/// nothing and gives its spares back until the process is below the line
/// less the cap. A working set of 0, which Metal does
/// not report for a real device, leaves the ceiling as the cap and no
/// pressure line.
pub struct VersionBudget {
    /// Most bytes of spares the pool parks.
    spare_cap: u64,
    /// Process-wide allocated bytes above which the pool gives its spares back.
    pressure_above: u64,
}

impl VersionBudget {
    /// The budget for a device whose recommended working set is `working_set` bytes.
    #[must_use]
    pub const fn from_working_set(working_set: u64) -> Self {
        if working_set == 0 {
            return Self {
                spare_cap: VERSION_POOL_CEILING_BYTES,
                pressure_above: u64::MAX,
            };
        }
        let share = working_set / VERSION_POOL_WORKING_SET_DIVISOR;
        Self {
            spare_cap: if share < VERSION_POOL_CEILING_BYTES {
                share
            } else {
                VERSION_POOL_CEILING_BYTES
            },
            pressure_above: working_set / 4 * VERSION_POOL_PRESSURE_QUARTERS,
        }
    }

    /// Most bytes of spares the pool parks.
    #[must_use]
    pub const fn spare_cap(&self) -> u64 {
        self.spare_cap
    }

    /// Process-wide allocated bytes above which the pool gives its spares back.
    #[must_use]
    pub const fn pressure_above(&self) -> u64 {
        self.pressure_above
    }

    /// Process-wide allocated bytes the process has to fall below before parking resumes.
    #[must_use]
    pub const fn resume_below(&self) -> u64 {
        self.pressure_above.saturating_sub(self.spare_cap)
    }
}

/// Submissions a texture with no spares may go without a rename before it stops being tracked.
///
/// Dropping a texture's state frees its write history, a few KiB of tile
/// stamps per level, and costs the next rename a fresh history and a
/// whole-level copy into a fresh allocation, which it would take anyway
/// with no spare left. The state is far smaller than a spare, so it is
/// kept longer: about two seconds at 60 frames a second, which spares a
/// texture renamed in bursts seconds apart the rebuild, while a texture
/// that stopped being renamed gives the memory back.
pub const STATE_MAX_IDLE_SUBMITS: u64 = 120;

/// Fewest destroyed versions handed to the caller per drain, when that many wait.
///
/// See [`TextureVersions::take_doomed`] for the batch. A drain that
/// destroyed 1357 handles at once (678 versions of two views each) took
/// 6.1 ms, about 4.5 us a handle, so 64 versions cost about 0.6 ms.
pub const MIN_DESTROYS_PER_DRAIN: usize = 64;

/// Drains over which the destroy queue's arrival rate is averaged, and the queue's divisor.
///
/// A burst raises the averaged rate for this many drains and is handed out
/// over about five of them: the 678 versions of the 6.1 ms drain go as
/// 170, 149, 130, 114, 100 and 15.
pub const DESTROY_SPREAD_DRAINS: usize = 8;

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
/// bit: D3D9 lets nothing but uploads, mip regeneration and an extended
/// device's whole-surface `StretchRect` write one, and the caller reports
/// each. A volume, a depth texture and a render target take whole-level
/// copies.
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
    parked_seq: u64,
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
    bytes: u64,
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
    /// Versions to destroy, with their bytes, oldest first; see [`Self::take_doomed`].
    doomed: VecDeque<(V, u64)>,
    /// Bytes of the versions in `doomed`, still allocated until destroyed.
    doomed_bytes: u64,
    /// Versions queued in `doomed` since the last [`Self::take_doomed`].
    doomed_since_take: usize,
    /// Versions queued between each of the last [`DESTROY_SPREAD_DRAINS`] takes, a ring.
    doomed_arrivals: [usize; DESTROY_SPREAD_DRAINS],
    /// Slot of `doomed_arrivals` the next take writes.
    arrival_slot: usize,
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
    /// One `(submission, raw texture id, epoch)` per tracked state, in submission order.
    ///
    /// [`Self::trim`] looks at a state when its entry is
    /// [`STATE_MAX_IDLE_SUBMITS`] old, drops it when it has no spares and no
    /// rename since, and enters it again under its latest rename otherwise,
    /// so a drain visits only the states whose age came due. The set keeps
    /// the order whatever submission an entry comes back under, so no state
    /// waits behind a younger one. An entry whose epoch no longer matches its
    /// texture's state belongs to a state that is gone.
    idle: BTreeSet<(u64, u64, u64)>,
    /// What the pool may hold; see [`VersionBudget`].
    budget: VersionBudget,
    /// Parking is off: past the pressure line and not yet back below the resume mark.
    pressed: bool,
    /// Bytes the next [`Self::take_doomed`] hands out at least, the excess over the pressure line.
    relieve_bytes: u64,
}

impl<V> TextureVersions<V> {
    #[must_use]
    pub fn new(budget: VersionBudget) -> Self {
        Self {
            textures: FxHashMap::default(),
            retiring: VecDeque::new(),
            parked_bytes: 0,
            doomed: VecDeque::new(),
            doomed_bytes: 0,
            doomed_since_take: 0,
            doomed_arrivals: [0; DESTROY_SPREAD_DRAINS],
            arrival_slot: 0,
            next_epoch: 1,
            with_spares: FxHashSet::default(),
            idle: BTreeSet::new(),
            budget,
            pressed: false,
            relieve_bytes: 0,
        }
    }

    /// Bytes of storage parked as spares across every texture.
    #[must_use]
    pub const fn parked_bytes(&self) -> u64 {
        self.parked_bytes
    }

    /// Bytes of storage queued to be destroyed and still allocated.
    #[must_use]
    pub const fn pending_destroy_bytes(&self) -> u64 {
        self.doomed_bytes
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
    /// were made for, those spares are queued to be destroyed, and the
    /// texture starts over under the new shape; versions of the old shape
    /// still retiring are destroyed when they are popped. `seq` is the submission
    /// the rename belongs to, which keeps the texture tracked for another
    /// [`STATE_MAX_IDLE_SUBMITS`] submissions.
    pub fn take_spare(
        &mut self,
        id: TextureId,
        shape: VersionShape,
        records_writes: bool,
        seq: u64,
    ) -> Option<Spare<V>> {
        let epoch = self.next_epoch;
        let state = match self.textures.entry(id) {
            Entry::Vacant(slot) => {
                self.next_epoch += 1;
                self.idle.insert((seq, id.raw(), epoch));
                slot.insert(VersionState::new(shape, epoch, records_writes))
            }
            Entry::Occupied(slot) => {
                let state = slot.into_mut();
                if state.shape != shape {
                    self.parked_bytes -= state.bytes * state.spares.len() as u64;
                    self.doomed_bytes += state.bytes * state.spares.len() as u64;
                    self.doomed_since_take += state.spares.len();
                    let bytes = state.bytes;
                    self.doomed
                        .extend(state.spares.drain(..).map(|spare| (spare.views, bytes)));
                    self.next_epoch += 1;
                    self.idle.insert((seq, id.raw(), epoch));
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
        spare
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
        let (epoch, birth_seq, held_writes, bytes) =
            self.textures.get_mut(&id).map_or((0, 0, 0, 0), |state| {
                let birth = core::mem::replace(&mut state.live_birth_seq, seq);
                (
                    state.epoch,
                    birth,
                    state.history.as_ref().map_or(0, |h| h.writes),
                    state.bytes,
                )
            });
        self.retiring.push_back(Retiring {
            id,
            views,
            seq,
            epoch,
            birth_seq,
            held_writes,
            bytes,
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

    /// Park a popped version as its texture's newest spare, or queue it to be destroyed.
    ///
    /// It is queued when its texture is gone or changed shape since it
    /// retired, or when parking it would take the spares past
    /// the budget's cap, or while the last [`Self::note_allocated`] found the
    /// process short of GPU memory. `failed` is
    /// the highest submission the GPU aborted (0 for none), read with the
    /// counter the pop was gated on: a version live through an aborted
    /// submission may lack a copy or an upload that submission carried, so
    /// it parks with unknown contents. `now` is the submission being built,
    /// from which the spare's idle time counts: a version the GPU retired
    /// late is no older for it.
    pub fn park(&mut self, entry: Retiring<V>, failed: u64, now: u64) {
        let Some(state) = self
            .textures
            .get_mut(&entry.id)
            .filter(|state| state.epoch == entry.epoch)
        else {
            self.doom(entry.views, entry.bytes);
            return;
        };
        if self.pressed {
            let bytes = state.bytes;
            self.doom(entry.views, bytes);
            return;
        }
        if self.parked_bytes + state.bytes > self.budget.spare_cap {
            mtld3d_shared::log_once_info!(
                target: crate::LOG_TARGET,
                "texture versions: the rename pool reached its {} MiB cap; retired \
                 versions past it are destroyed until spares are taken or age out",
                self.budget.spare_cap >> 20,
            );
            let bytes = state.bytes;
            self.doom(entry.views, bytes);
            return;
        }
        self.parked_bytes += state.bytes;
        let aborted = failed != 0 && entry.birth_seq <= failed;
        if state.spares.is_empty() {
            self.with_spares.insert(entry.id);
        }
        state.spares.push(Spare {
            views: entry.views,
            held_writes: (!aborted).then_some(entry.held_writes),
            parked_seq: now,
        });
    }

    /// Note the process's allocated GPU bytes, once per drain before parking.
    ///
    /// Past the budget's pressure line the pool parks nothing until a later
    /// drain finds the process below [`VersionBudget::resume_below`], and
    /// queues spares to be destroyed, oldest first, until what waits in the
    /// destroy queue covers the excess: the queue's versions are still
    /// allocated, so they count toward it. The next [`Self::take_doomed`]
    /// hands out at least that excess at once, so the pressure is relieved
    /// in one drain rather than spread like a burst.
    pub fn note_allocated(&mut self, allocated: u64) {
        self.relieve_bytes = allocated.saturating_sub(self.budget.pressure_above);
        self.pressed = if self.pressed {
            allocated >= self.budget.resume_below()
        } else {
            allocated > self.budget.pressure_above
        };
        if self.relieve_bytes == 0 {
            return;
        }
        mtld3d_shared::log_once_info!(
            target: crate::LOG_TARGET,
            "texture versions: {} MiB of GPU memory allocated, past the {} MiB pressure \
             line; the rename pool parks nothing and gives its spares back",
            allocated >> 20,
            self.budget.pressure_above >> 20,
        );
        let excess = self.relieve_bytes;
        if self.doomed_bytes >= excess {
            return;
        }
        let mut oldest: Vec<(u64, TextureId, u64)> = Vec::new();
        for id in &self.with_spares {
            if let Some(state) = self.textures.get(id) {
                oldest.extend(
                    state
                        .spares
                        .iter()
                        .map(|spare| (spare.parked_seq, *id, state.bytes)),
                );
            }
        }
        oldest.sort_unstable_by_key(|&(seq, id, _)| (seq, id.raw()));
        let mut freed = self.doomed_bytes;
        let mut counts: FxHashMap<TextureId, usize> = FxHashMap::default();
        for (_, id, bytes) in oldest {
            if freed >= excess {
                break;
            }
            freed += bytes;
            *counts.entry(id).or_default() += 1;
        }
        for (id, count) in counts {
            let Some(state) = self.textures.get_mut(&id) else {
                continue;
            };
            self.parked_bytes -= state.bytes * count as u64;
            self.doomed_bytes += state.bytes * count as u64;
            self.doomed_since_take += count;
            let bytes = state.bytes;
            self.doomed.extend(
                state
                    .spares
                    .drain(..count)
                    .map(|spare| (spare.into_views(), bytes)),
            );
            if state.spares.is_empty() {
                self.with_spares.remove(&id);
            }
        }
    }

    /// Hand back the next batch of versions to destroy.
    ///
    /// The average number queued per take over the last
    /// [`DESTROY_SPREAD_DRAINS`] takes plus that fraction of the queue, at
    /// least [`MIN_DESTROYS_PER_DRAIN`], oldest first; the rest wait for the
    /// next drain. A steady stream is handed out at the rate it arrives and
    /// whatever queued while the average caught up shrinks every drain, so
    /// no queue persists; a burst raises the average for a few drains and
    /// is spread over them. After a [`Self::note_allocated`] past the
    /// pressure line the batch also covers at least the excess bytes. Every
    /// queued version
    /// was popped after its submission settled and has had its handle-keyed
    /// records forgotten, and no rename takes it again, so no command of a
    /// submission in flight or being built can name it: destroying it a
    /// drain or several later is as safe as destroying it at once.
    pub fn take_doomed(&mut self) -> Vec<V> {
        self.doomed_arrivals[self.arrival_slot] = self.doomed_since_take;
        self.arrival_slot = (self.arrival_slot + 1) % DESTROY_SPREAD_DRAINS;
        self.doomed_since_take = 0;
        let rate = self
            .doomed_arrivals
            .iter()
            .sum::<usize>()
            .div_ceil(DESTROY_SPREAD_DRAINS);
        let mut batch = MIN_DESTROYS_PER_DRAIN
            .max(rate + self.doomed.len().div_ceil(DESTROY_SPREAD_DRAINS))
            .min(self.doomed.len());
        // Under pressure, at least the excess goes in this one batch.
        let mut covered: u64 = self
            .doomed
            .iter()
            .take(batch)
            .map(|&(_, bytes)| bytes)
            .sum();
        while covered < self.relieve_bytes && batch < self.doomed.len() {
            covered += self.doomed[batch].1;
            batch += 1;
        }
        self.relieve_bytes = 0;
        let mut taken = Vec::with_capacity(batch);
        for (views, bytes) in self.doomed.drain(..batch) {
            self.doomed_bytes -= bytes;
            taken.push(views);
        }
        taken
    }

    /// Queue `views`, of `bytes` bytes, for [`Self::take_doomed`].
    fn doom(&mut self, views: V, bytes: u64) {
        self.doomed_bytes += bytes;
        self.doomed_since_take += 1;
        self.doomed.push_back((views, bytes));
    }

    /// Queue for destruction every spare idle past the bound, and forget idle textures.
    ///
    /// A spare no rename took within [`SPARE_MAX_IDLE_SUBMITS`] submissions
    /// before `now` is queued for [`Self::take_doomed`]. A spare's age counts
    /// from the submission it was parked in. Spares park in submission
    /// order and renames take the newest, so each texture's oldest spares
    /// sit at the bottom of its stack. Only the
    /// textures holding spares are visited for that. Then every texture
    /// left without spares and without a rename for
    /// [`STATE_MAX_IDLE_SUBMITS`] submissions stops being tracked, its write
    /// history with it; only the states whose age came due are looked at.
    pub fn trim(&mut self, now: u64) {
        let spare_aged = |seq: u64| seq + SPARE_MAX_IDLE_SUBMITS < now;
        let (textures, parked_bytes) = (&mut self.textures, &mut self.parked_bytes);
        let (doomed, doomed_bytes, doomed_since_take) = (
            &mut self.doomed,
            &mut self.doomed_bytes,
            &mut self.doomed_since_take,
        );
        self.with_spares.retain(|id| {
            let Some(state) = textures.get_mut(id) else {
                return false;
            };
            let old = state
                .spares
                .iter()
                .take_while(|spare| spare_aged(spare.parked_seq))
                .count();
            *parked_bytes -= state.bytes * old as u64;
            *doomed_bytes += state.bytes * old as u64;
            *doomed_since_take += old;
            let bytes = state.bytes;
            doomed.extend(
                state
                    .spares
                    .drain(..old)
                    .map(|spare| (spare.into_views(), bytes)),
            );
            !state.spares.is_empty()
        });
        let aged_at = |seq: u64| seq + STATE_MAX_IDLE_SUBMITS < now;
        while let Some(&(seq, raw, epoch)) = self.idle.first() {
            if !aged_at(seq) {
                break;
            }
            self.idle.pop_first();
            let id = TextureId::from_raw(raw);
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
            self.idle.insert((again, raw, epoch));
        }
    }

    /// Stop tracking a released texture, queueing its spares to be destroyed.
    ///
    /// A spare was popped after its submission settled and has left every
    /// handle-keyed record, so it can wait in the destroy queue like any
    /// other. Its versions still retiring stay queued and are destroyed when
    /// popped: a texture that comes back under the same id starts a state
    /// of a new epoch, which they do not match.
    pub fn forget_texture(&mut self, id: TextureId) {
        let Some(state) = self.textures.remove(&id) else {
            return;
        };
        self.with_spares.remove(&id);
        self.parked_bytes -= state.bytes * state.spares.len() as u64;
        for spare in state.spares {
            self.doom(spare.views, state.bytes);
        }
    }

    /// Take every spare, retiring version and queued destroy, leaving the textures tracked.
    ///
    /// For `Reset` and shutdown, whose caller waits for the GPU to go idle
    /// before it destroys what this returns.
    pub fn drain_versions(&mut self) -> Vec<V> {
        let mut all: Vec<V> = self.retiring.drain(..).map(|entry| entry.views).collect();
        all.extend(self.doomed.drain(..).map(|(views, _)| views));
        self.doomed_bytes = 0;
        self.doomed_since_take = 0;
        self.doomed_arrivals = [0; DESTROY_SPREAD_DRAINS];
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

    /// Versions waiting in the destroy queue.
    #[cfg(test)]
    fn doomed(&self) -> usize {
        self.doomed.len()
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
