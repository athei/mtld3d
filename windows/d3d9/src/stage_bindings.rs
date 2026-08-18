//! Per-stage texture and sampler state (16 stages) owned by `DeviceInner`.
//!
//! Texture slots hold game-supplied COM pointers with the Release step
//! paired to `SetTexture` replacement and device teardown.
//!
//! PS3.0 allows pixel-shader samplers s0–s15; the `WoW` HD shadow path
//! binds the 4th cascade depth texture at slot 8 (cascades 0–3
//! land at slots 5/6/7/8), so the cap is 16 rather than the D3D9
//! `MaxSimultaneousTextures` FF-only floor of 8. The FF-only
//! `MaxSimultaneousTextures = 8` cap advertisement is kept independent
//! of the programmable-PS slot count.

use mtld3d_types::{
    D3DSAMP_ADDRESSU, D3DSAMP_ADDRESSV, D3DSAMP_ADDRESSW, D3DSAMP_BORDERCOLOR, D3DSAMP_MAGFILTER,
    D3DSAMP_MAXANISOTROPY, D3DSAMP_MAXMIPLEVEL, D3DSAMP_MINFILTER, D3DSAMP_MIPFILTER,
    D3DSAMP_MIPMAPLODBIAS, D3DSAMP_SRGBTEXTURE, SAMPLER_STATE_COUNT, sampler_state_defaults,
};

use super::{
    com_ref::{Bound, CachedComPtr},
    texture::Direct3DTexture9,
};
use crate::LOG_TARGET;

/// Number of D3D9 PS sampler stages we accept.
///
/// PS3.0 spec maximum is 16 (s0–s15). The FF combiner limit
/// (`MaxTextureBlendStages = 8`) and the FF cap
/// `MaxSimultaneousTextures = 8` are separate from this.
pub const STAGE_COUNT: usize = 16;

bitflags::bitflags! {
    /// Outcome of [`StageBindings::replace_texture`].
    ///
    /// Used by the `SetTexture` thunk to gate snapshot dirty-marking. The
    /// FF VS/PS keys depend only on the slot occupancy mask and the variant
    /// only on per-slot depth-format-ness, so a swap that flips neither
    /// needs only a fresh `STAGES` mark (the new handle the encoder binds).
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct TextureSwapDelta: u8 {
        /// Slot occupancy (null vs non-null) flipped.
        const OCCUPANCY_CHANGED = 1 << 0;
        /// The slot's depth-format-ness flipped (drives `depth_sampler_mask`).
        const DEPTH_CHANGED = 1 << 1;
        /// The slot's volume (3D) texture-ness flipped (drives `volume_sampler_mask`).
        const VOLUME_CHANGED = 1 << 2;
        /// The slot's cube texture-ness flipped (drives `cube_sampler_mask`).
        const CUBE_CHANGED = 1 << 3;
    }
}

pub struct StageBindings {
    /// Per-stage bound texture slot.
    ///
    /// Uses the `Bound` ownership marker: each swap increments the
    /// wrapper's `private_refcount` inline (no vtable indirection, no
    /// `ApiTimer` instrumentation) rather than going through the public
    /// `IUnknown` `AddRef`/`Release` thunks. The wrapper stays alive while
    /// bound via a dual public/private refcount invariant (a
    /// device-internal binding count kept alongside the public `IUnknown`
    /// count).
    textures: [CachedComPtr<Direct3DTexture9, Bound>; STAGE_COUNT],
    sampler_states: [[u32; SAMPLER_STATE_COUNT]; STAGE_COUNT],
    /// Per-(sampler, type_) once-warn latch for unsupported D3DSAMP_* writes.
    ///
    /// Bit `type_` of `samp_warn_fired[sampler]` is set after the first
    /// warn for that pair. `SAMPLER_STATE_COUNT == 14`, so `u16` per
    /// sampler covers every legal `type_` index with bits to spare.
    samp_warn_fired: [u16; STAGE_COUNT],
    /// Bit `stage` set ⇒ `textures[stage]` is non-null.
    ///
    /// Maintained by `replace_texture`/`teardown` so the per-draw stage
    /// walk visits only bound slots instead of all `STAGE_COUNT`.
    bound_mask: u16,
    /// Cached [`Self::depth_sampler_mask`] value, maintained incrementally.
    ///
    /// Depth-format-ness is fixed at texture creation, so the mask only
    /// changes when a slot is rebound; caching it turns the per-draw
    /// variant-key scans into field reads.
    depth_mask: u16,
    /// Cached [`Self::volume_sampler_mask`] value; same scheme as `depth_mask`.
    volume_mask: u16,
    /// Cached cube binding mask; updated only by `replace_texture`.
    cube_mask: u16,
    /// Cached [`Self::depth_fetch_mask`] value; same scheme as `depth_mask`.
    fetch_mask: u16,
}

impl StageBindings {
    pub const fn new(sampler_defaults: &[[u32; SAMPLER_STATE_COUNT]; STAGE_COUNT]) -> Self {
        Self {
            textures: [const { CachedComPtr::null() }; STAGE_COUNT],
            sampler_states: *sampler_defaults,
            samp_warn_fired: [0; STAGE_COUNT],
            bound_mask: 0,
            depth_mask: 0,
            volume_mask: 0,
            cube_mask: 0,
            fetch_mask: 0,
        }
    }

    /// Bit `stage` set ⇒ that slot has a texture bound.
    ///
    /// The per-draw snapshot walk iterates this instead of probing all
    /// `STAGE_COUNT` slots.
    pub const fn bound_mask(&self) -> u16 {
        self.bound_mask
    }

    pub const fn texture(&self, stage: usize) -> *mut Direct3DTexture9 {
        self.textures[stage].raw()
    }

    /// Bit `i` set ⇒ slot `i` is bound to a sampleable depth-format texture (shadow map).
    ///
    /// Folded into `VariantKey::depth_sampler_mask` so the PS shader cache
    /// compiles a `depth2d<float>` variant for matching slots.
    pub fn depth_sampler_mask(&self) -> u16 {
        debug_assert_eq!(
            self.depth_mask,
            self.scan_mask(Direct3DTexture9::is_depth_format),
            "cached depth_mask out of sync with texture slots"
        );
        self.depth_mask
    }

    /// Bit `i` set ⇒ slot `i` is bound to a volume (3D) texture.
    ///
    /// One whose backing `MTLTexture` is `MTLTextureType3D` (`depth > 1`).
    /// Folded into `VariantKey::volume_sampler_mask` so the FF PS compiles
    /// a `texture3d<float>` variant for matching slots.
    pub fn volume_sampler_mask(&self) -> u16 {
        debug_assert_eq!(
            self.volume_mask,
            self.scan_mask(Direct3DTexture9::is_volume),
            "cached volume_mask out of sync with texture slots"
        );
        self.volume_mask
    }

    pub fn cube_sampler_mask(&self) -> u16 {
        debug_assert_eq!(
            self.cube_mask,
            self.scan_mask(Direct3DTexture9::is_cube),
            "cached cube_mask out of sync with texture slots"
        );
        self.cube_mask
    }

    /// Bit `i` set ⇒ slot `i` is bound to a "readable raw depth" FOURCC texture (INTZ/DF24/DF16).
    ///
    /// A subset of [`Self::depth_sampler_mask`]. These slots fetch the RAW
    /// stored depth (`.sample` + a non-comparison sampler) instead of a
    /// hardware depth comparison (`sample_compare`), per the D3D9 rule that
    /// raw-depth FOURCC formats are read raw rather than as a shadow
    /// comparison. Folded into `VariantKey::depth_fetch_mask`.
    pub fn depth_fetch_mask(&self) -> u16 {
        debug_assert_eq!(
            self.fetch_mask,
            self.scan_mask(|t| mtld3d_core::format::is_raw_depth_fetch_format(t.d3d_format())),
            "cached fetch_mask out of sync with texture slots"
        );
        self.fetch_mask
    }

    /// Recompute a per-slot predicate mask by walking every texture slot.
    ///
    /// Only referenced by the `debug_assert` in-sync guards on the cached
    /// mask accessors; the per-draw path reads the cached fields.
    fn scan_mask(&self, pred: impl Fn(&Direct3DTexture9) -> bool) -> u16 {
        let mut mask = 0u16;
        for (stage, slot) in self.textures.iter().enumerate() {
            if slot.as_ref().is_some_and(&pred) {
                mask |= 1u16 << stage;
            }
        }
        mask
    }

    /// Bind `tex` at `stage`, transferring one refcount to the slot via [`CachedComPtr::adopt`].
    ///
    /// The prior slot value is released via the slot's auto-`Drop` on
    /// assignment. Null `tex` is a no-op for the new slot (`adopt` skips
    /// the refcount bump).
    ///
    /// Returns a [`TextureSwapDelta`] describing whether the slot's
    /// occupancy or depth-format-ness flipped, so the caller can gate
    /// snapshot dirty-marking: a swap that changes neither rebuilds
    /// byte-identical FF VS/PS keys and variant.
    pub fn replace_texture(
        &mut self,
        stage: usize,
        tex: *mut Direct3DTexture9,
    ) -> TextureSwapDelta {
        // Snapshot the old slot before `adopt` drops its ref below.
        let old = &self.textures[stage];
        let old_nonnull = !old.raw().is_null();
        let old_depth = old.as_ref().is_some_and(Direct3DTexture9::is_depth_format);
        let old_volume = old.as_ref().is_some_and(Direct3DTexture9::is_volume);
        let old_cube = old.as_ref().is_some_and(Direct3DTexture9::is_cube);

        let new_nonnull = !tex.is_null();
        // SAFETY: `tex` is null or a live IDirect3DTexture9 supplied by the
        // calling D3D9 vtable thunk; the game holds a ref across SetTexture,
        // so the stored `is_depth_format` / `is_volume` flags are readable
        // before `adopt`.
        let (new_depth, new_volume, new_cube, new_fetch) = if new_nonnull {
            // SAFETY: as above — take a shared reference to the live texture and
            // read the flags through it (one raw-pointer deref).
            let t = unsafe { &*tex };
            (
                t.is_depth_format(),
                t.is_volume(),
                t.is_cube(),
                mtld3d_core::format::is_raw_depth_fetch_format(t.d3d_format()),
            )
        } else {
            (false, false, false, false)
        };

        // SAFETY: `tex` is null or a live IDirect3DTexture9 supplied by the
        // calling D3D9 vtable thunk; AddRef/Release thunks valid for our
        // lifetime.
        self.textures[stage] = unsafe { CachedComPtr::adopt(tex) };

        // Keep the cached per-slot masks in step with the slot write; the
        // accessors' debug guards recompute and compare.
        let bit = 1u16 << stage;
        self.bound_mask = with_bit(self.bound_mask, bit, new_nonnull);
        self.depth_mask = with_bit(self.depth_mask, bit, new_depth);
        self.volume_mask = with_bit(self.volume_mask, bit, new_volume);
        self.cube_mask = with_bit(self.cube_mask, bit, new_cube);
        self.fetch_mask = with_bit(self.fetch_mask, bit, new_fetch);

        let mut delta = TextureSwapDelta::empty();
        delta.set(
            TextureSwapDelta::OCCUPANCY_CHANGED,
            old_nonnull != new_nonnull,
        );
        delta.set(TextureSwapDelta::DEPTH_CHANGED, old_depth != new_depth);
        delta.set(TextureSwapDelta::VOLUME_CHANGED, old_volume != new_volume);
        delta.set(TextureSwapDelta::CUBE_CHANGED, old_cube != new_cube);
        delta
    }

    pub const fn sampler_state(&self, sampler: usize, type_: usize) -> u32 {
        self.sampler_states[sampler][type_]
    }

    pub fn set_sampler_state(&mut self, sampler: usize, type_: usize, value: u32) {
        self.warn_samp_non_default_once(sampler, type_, value);
        self.sampler_states[sampler][type_] = value;
    }

    fn warn_samp_non_default_once(&mut self, sampler: usize, type_: usize, value: u32) {
        static SAMP_DEFAULTS: [u32; SAMPLER_STATE_COUNT] = sampler_state_defaults();

        if sampler >= STAGE_COUNT || type_ >= SAMPLER_STATE_COUNT {
            return;
        }
        if value == SAMP_DEFAULTS[type_] {
            if mtld3d_core::state_trace::enabled() {
                log::trace!(
                    target: mtld3d_core::state_trace::TARGET,
                    "D3DSAMP_{type_} (sampler {sampler}) = {value:#x} (default — write suppressed in warn machinery)"
                );
            }
            return;
        }
        if (self.samp_warn_fired[sampler] & (1u16 << type_)) != 0 {
            return;
        }
        let class = samp_classify(
            u32::try_from(type_).expect("D3DSAMP type fits u32 by SAMPLER_STATE_COUNT bound"),
        );
        if matches!(class, SampClass::Consumed) {
            if mtld3d_core::state_trace::enabled() {
                let default = SAMP_DEFAULTS[type_];
                log::trace!(
                    target: mtld3d_core::state_trace::TARGET,
                    "D3DSAMP_{type_} (sampler {sampler}) Consumed = {value:#x} (default {default:#x})"
                );
            }
            return;
        }
        self.samp_warn_fired[sampler] |= 1u16 << type_;
        let default = SAMP_DEFAULTS[type_];
        match class {
            SampClass::Consumed => {} // unreachable
            SampClass::PortCandidate(feat) => {
                log::warn!(
                    target: LOG_TARGET,
                    "D3DSAMP_{type_} (sampler {sampler}) = {value:#x} (default {default:#x}) set but {feat} not implemented"
                );
            }
            SampClass::NotImplemented => {
                log::warn!(
                    target: LOG_TARGET,
                    "D3DSAMP_{type_} (sampler {sampler}) = {value:#x} (default {default:#x}) written but not consumed"
                );
            }
        }
    }

    pub const fn sampler_states(&self, stage: usize) -> [u32; SAMPLER_STATE_COUNT] {
        self.sampler_states[stage]
    }

    /// Release and null every texture slot.
    ///
    /// Used from the device release path.
    pub fn teardown(&mut self) {
        for slot in &mut self.textures {
            *slot = CachedComPtr::null();
        }
        self.bound_mask = 0;
        self.depth_mask = 0;
        self.volume_mask = 0;
        self.fetch_mask = 0;
    }

    /// `IDirect3DDevice9::Reset` analog of `teardown`.
    ///
    /// Releases every texture slot and reseeds sampler states to the D3D9
    /// spec defaults captured at `CreateDevice`. The per-type silent-write
    /// warn latch (`samp_warn_fired`) is intentionally preserved across
    /// Reset — those latches are process-lifetime telemetry, not device
    /// state.
    pub fn reset_to_defaults(
        &mut self,
        sampler_defaults: &[[u32; SAMPLER_STATE_COUNT]; STAGE_COUNT],
    ) {
        self.teardown();
        self.sampler_states = *sampler_defaults;
    }
}

/// Return `mask` with `bit` set when `on`, cleared otherwise.
///
/// Shared by the cached-mask updates in [`StageBindings::replace_texture`].
const fn with_bit(mask: u16, bit: u16, on: bool) -> u16 {
    if on { mask | bit } else { mask & !bit }
}

// ── Silent-write audit: D3DSAMP_* classifier ──
// Per-(sampler, type_) latch lives on `StageBindings.samp_warn_fired`. This
// table classifies each type so the warn message is targeted.

enum SampClass {
    Consumed,
    PortCandidate(&'static str),
    NotImplemented,
}

const fn samp_classify(type_: u32) -> SampClass {
    match type_ {
        D3DSAMP_ADDRESSU
        | D3DSAMP_ADDRESSV
        | D3DSAMP_ADDRESSW
        | D3DSAMP_MAGFILTER
        | D3DSAMP_MINFILTER
        | D3DSAMP_MIPFILTER
        | D3DSAMP_MAXANISOTROPY
        // MAXMIPLEVEL consumed by sampler_state::key_from_snapshot (mtld3d-core),
        // plumbed to setLodMinClamp on the unix side.
        | D3DSAMP_MAXMIPLEVEL
        // SRGBTEXTURE consumed by sampler_state::key_from_snapshot
        // (bit 38) — feeds the sampler-key hash so distinct samplers
        // stay distinct, but no draw-time view swap. There is no
        // eager sRGB texture-view path: it would force PixelFormatView
        // usage on every BGRA8 / BC1-3 texture and block Metal
        // lossless compression; no mtld3d target game actually sets
        // this bit, and the CAMetalLayer DisplayP3 colorspace tag
        // handles colour-space correctness at the compositor level.
        | D3DSAMP_SRGBTEXTURE => SampClass::Consumed,

        // Known gap: Metal has no sampler-level LOD bias; bias is
        // expressed inside the shader's sample() call, which would
        // require DXSO→MSL emitter changes. Stays PortCandidate.
        D3DSAMP_MIPMAPLODBIAS => SampClass::PortCandidate("LOD bias (Metal shader-only)"),
        D3DSAMP_BORDERCOLOR => SampClass::PortCandidate("border color"),

        _ => SampClass::NotImplemented,
    }
}
