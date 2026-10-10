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
//! nobody takes for [`SPARE_MAX_AGE_SUBMITS`] submissions is trimmed.
//!
//! Pure bookkeeping, generic over the caller's view handles: no Metal
//! calls and no blit shapes, so the retirement order and the pool bounds
//! are host-testable.

use std::collections::{VecDeque, hash_map::Entry};

use mtld3d_shared::{
    TextureCreateDesc,
    mtl::{PixelFormat, StorageMode, Swizzle, TextureCreateFlags, TextureUsage},
};
use rustc_hash::FxHashMap;

use crate::ids::TextureId;

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

/// A retired version of a texture, kept for the texture's next rename.
pub struct Spare<V> {
    views: V,
    last_seq: u64,
}

impl<V> Spare<V> {
    /// The version's view handles, consuming the spare.
    #[must_use]
    pub fn into_views(self) -> V {
        self.views
    }
}

/// A version a rename replaced, waiting for the GPU to finish with it.
pub struct Retiring<V> {
    id: TextureId,
    views: V,
    seq: u64,
    epoch: u64,
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
}

impl<V> TextureVersions<V> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            textures: FxHashMap::default(),
            retiring: VecDeque::new(),
            parked_bytes: 0,
            next_epoch: 1,
        }
    }

    /// Bytes of storage parked as spares across every texture.
    #[must_use]
    pub const fn parked_bytes(&self) -> u64 {
        self.parked_bytes
    }

    /// Start a rename of `id`, handing back its newest spare if it has one.
    ///
    /// A texture seen for the first time starts tracking here with no
    /// spares. When `shape` differs from the shape the texture's spares were
    /// made for, those spares come back in the second value for the caller
    /// to destroy behind the current submission, and the texture starts
    /// over under the new shape; versions of the old shape still retiring
    /// are destroyed when they are popped.
    pub fn take_spare(&mut self, id: TextureId, shape: VersionShape) -> (Option<Spare<V>>, Vec<V>) {
        let mut discarded = Vec::new();
        let epoch = self.next_epoch;
        let state = match self.textures.entry(id) {
            Entry::Vacant(slot) => {
                self.next_epoch += 1;
                slot.insert(VersionState::new(shape, epoch))
            }
            Entry::Occupied(slot) => {
                let state = slot.into_mut();
                if state.shape != shape {
                    self.parked_bytes -= state.bytes * state.spares.len() as u64;
                    discarded.extend(state.spares.drain(..).map(Spare::into_views));
                    self.next_epoch += 1;
                    *state = VersionState::new(shape, epoch);
                }
                state
            }
        };
        let spare = state.spares.pop();
        if spare.is_some() {
            self.parked_bytes -= state.bytes;
        }
        (spare, discarded)
    }

    /// Queue the version a rename of `id` just replaced, retiring behind submission `seq`.
    ///
    /// `seq` is the submission being built, which holds every command that
    /// named the version: the draws that sampled it and the copy out of it.
    /// It must not go backwards across calls, since [`Self::pop_retired`]
    /// gates on the front entry alone; the encoder's submission counter only
    /// advances. A texture [`Self::take_spare`] never saw is not tracked,
    /// and its version is still queued: it is destroyed when popped.
    pub fn retire(&mut self, id: TextureId, views: V, seq: u64) {
        debug_assert!(
            self.retiring.back().is_none_or(|back| back.seq <= seq),
            "texture versions must retire in submission order",
        );
        let epoch = self.textures.get(&id).map_or(0, |state| state.epoch);
        self.retiring.push_back(Retiring {
            id,
            views,
            seq,
            epoch,
        });
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
    /// [`VERSION_POOL_CAP_BYTES`].
    pub fn park(&mut self, entry: Retiring<V>) -> Option<V> {
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
        state.spares.push(Spare {
            views: entry.views,
            last_seq: entry.seq,
        });
        None
    }

    /// Take every spare no rename used for [`SPARE_MAX_AGE_SUBMITS`] submissions before `now`.
    ///
    /// A spare's age counts from the submission it last served in. Spares
    /// park in submission order and renames take the newest, so each
    /// texture's oldest spares sit at the bottom of its stack.
    pub fn trim(&mut self, now: u64) -> Vec<V> {
        let mut aged = Vec::new();
        if self.parked_bytes == 0 {
            return aged;
        }
        for state in self.textures.values_mut() {
            let old = state
                .spares
                .iter()
                .take_while(|spare| spare.last_seq + SPARE_MAX_AGE_SUBMITS < now)
                .count();
            if old == 0 {
                continue;
            }
            self.parked_bytes -= state.bytes * old as u64;
            aged.extend(state.spares.drain(..old).map(Spare::into_views));
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
        self.parked_bytes -= state.bytes * state.spares.len() as u64;
        state.spares.into_iter().map(Spare::into_views).collect()
    }

    /// Take every spare and every retiring version, leaving the textures tracked.
    ///
    /// For `Reset` and shutdown, whose caller waits for the GPU to go idle
    /// before it destroys what this returns.
    pub fn drain_versions(&mut self) -> Vec<V> {
        let mut all: Vec<V> = self.retiring.drain(..).map(|entry| entry.views).collect();
        for state in self.textures.values_mut() {
            all.extend(state.spares.drain(..).map(Spare::into_views));
        }
        self.parked_bytes = 0;
        all
    }
}

impl<V> Default for TextureVersions<V> {
    fn default() -> Self {
        Self::new()
    }
}

/// One tracked texture: its shape and its spares, newest last.
struct VersionState<V> {
    shape: VersionShape,
    epoch: u64,
    bytes: u64,
    spares: Vec<Spare<V>>,
}

impl<V> VersionState<V> {
    fn new(shape: VersionShape, epoch: u64) -> Self {
        Self {
            bytes: shape.bytes(),
            shape,
            epoch,
            spares: Vec::new(),
        }
    }
}

/// Extent of mip `level` along an axis whose level-0 extent is `base`.
fn level_extent(base: u32, level: u32) -> u32 {
    (base.max(1) >> level).max(1)
}

#[cfg(test)]
mod tests;
