//! Fixed-Function pipeline state stored on `DeviceInner`.
//!
//! All state here is written by the `SetTransform` / `SetMaterial` /
//! `SetLight` / `LightEnable` / `SetTextureStageState` family and read on the
//! API thread when building FF shader keys and constant-buffer contents for
//! the draw path.
//!
//! Nothing here reaches the encoder thread directly; the draw path snapshots
//! the pieces it needs into closures.
//!
//! All fields are private. Writes go through the typed setters
//! (`set_transform`, `set_material`, ...) which pair the write with the
//! matching `FfDirty` mark so the "forgot to mark dirty" bug is not
//! representable.

use std::collections::BTreeMap;

use bitflags::bitflags;
use mtld3d_types::{
    D3DCOLORVALUE, D3DLIGHT_DIRECTIONAL, D3DLIGHT_SPOT, D3DLIGHT9, D3DMATERIAL9, D3DMATRIX,
    D3DRS_ALPHAFUNC, D3DRS_ALPHAREF, D3DRS_ALPHATESTENABLE, D3DRS_AMBIENT,
    D3DRS_AMBIENTMATERIALSOURCE, D3DRS_COLORVERTEX, D3DRS_DEPTHBIAS, D3DRS_DIFFUSEMATERIALSOURCE,
    D3DRS_EMISSIVEMATERIALSOURCE, D3DRS_FOGCOLOR, D3DRS_FOGDENSITY, D3DRS_FOGENABLE, D3DRS_FOGEND,
    D3DRS_FOGSTART, D3DRS_FOGTABLEMODE, D3DRS_FOGVERTEXMODE, D3DRS_INDEXEDVERTEXBLENDENABLE,
    D3DRS_LIGHTING, D3DRS_LOCALVIEWER, D3DRS_NORMALIZENORMALS, D3DRS_SPECULARENABLE,
    D3DRS_SPECULARMATERIALSOURCE, D3DRS_TEXTUREFACTOR, D3DRS_VERTEXBLEND, D3DTOP_DISABLE,
    D3DTSS_ALPHAARG1, D3DTSS_ALPHAARG2, D3DTSS_ALPHAOP, D3DTSS_BUMPENVLOFFSET,
    D3DTSS_BUMPENVLSCALE, D3DTSS_BUMPENVMAT00, D3DTSS_BUMPENVMAT01, D3DTSS_BUMPENVMAT10,
    D3DTSS_BUMPENVMAT11, D3DTSS_COLORARG1, D3DTSS_COLORARG2, D3DTSS_COLOROP, D3DTSS_TEXCOORDINDEX,
    D3DTSS_TEXTURETRANSFORMFLAGS, D3DTTFF_PROJECTED, RENDER_STATE_COUNT, StateBlockType,
    TEXTURE_STAGE_STATE_COUNT, texture_stage_state_defaults,
};

use crate::{
    LOG_TARGET,
    convert::FfVsLayout,
    dxso::{FfPsKey, FfStage, FfVsFlags, FfVsKey, VariantFlags, VariantKey},
    scratch::ScratchArena,
};

bitflags! {
    /// Per-section dirty bits for the FF VS const buffer.
    ///
    /// Set by `FfState` setters when their owning section changes; consumed
    /// (and cleared) by `take_dirty()` at `emit_snapshot_deltas` time. The
    /// API thread then walks each set bit and emits one `Op::SetFfVsConstRange`
    /// per dirty section, copying only the changed rows into the per-frame
    /// scratch arena and onto the encoder's `ff_vs_constants_mirror`.
    ///
    /// Initial state is [`FfVsDirty::all()`] so the first FF draw of the
    /// process uploads every section — the encoder mirror starts zero-init
    /// and must be brought up to date before any draw reads from it.
    ///
    /// **Why these granules**: each section is read by the FF VS shader as
    /// an independent block of rows in the layout (see the per-section
    /// `build_*_section` helpers and `ff_vs_row_count`). Sections never
    /// overlap, and the row ranges are fixed at emit time. RS-driven
    /// sections (`FOG`, `AMBIENT`) live here too — their setter is
    /// `SetRenderState` rather than an `FfState` method, but the dirty mark
    /// is recorded on `FfState` because the encoder mirror is the single
    /// source of truth.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct FfVsDirty: u32 {
        /// Rows 0-3: `transpose(world_palette[0] * view)`.
        const WV       = 1 << 0;
        /// Rows 4-7: `transpose(projection)`.
        const PROJ     = 1 << 1;
        /// Row 8: fog params from `D3DRS_FOGSTART/END/DENSITY`.
        const FOG      = 1 << 2;
        /// Row 9: ambient color from `D3DRS_AMBIENT`.
        const AMBIENT  = 1 << 3;
        /// Rows 10-14: material colors + power.
        const MATERIAL = 1 << 4;
        /// Rows 15-62: per-light × 8 (6 rows each).
        ///
        /// Activity per slot decided by the shader at read time via
        /// `FfVsKey.light_active_mask`.
        const LIGHTS   = 1 << 5;
        /// Rows 63-94: per-stage TTFF × 8 (4 rows each).
        ///
        /// Per-stage activity decided by `FfState.tt_active_mask`.
        const TT       = 1 << 6;
        /// Rows 95+: world-matrix palette (vertex-blend only).
        const PALETTE  = 1 << 7;
    }
}

/// `MaxActiveLights` per the D3D9 caps we advertise.
///
/// At most this many lights contribute to a single draw, regardless of the
/// addressable index range. Matches the 8 fast-path slots and the per-light
/// row budget in the FF VS const layout (rows 15..62 = 6 rows × 8 lights).
const MAX_ACTIVE_LIGHTS: usize = 8;

/// A light addressed beyond the 8 fast-path slots.
///
/// `defined` is implicit by the slot's presence in [`FfState::overflow_lights`];
/// `enabled` mirrors the `light_enabled` mask bit a fast-path slot would
/// carry.
struct OverflowLight {
    light: D3DLIGHT9,
    enabled: bool,
}

/// One entry of the compacted active-light list.
///
/// D3D9 lets `SetLight` / `LightEnable` address arbitrary indices but caps
/// the *simultaneously contributing* set at [`MAX_ACTIVE_LIGHTS`]; the FF VS
/// reads a dense `light[0..n]` slot range, so enabled lights at sparse
/// indices must be packed down into contiguous shader slots in ascending
/// D3D9-index order. `ty` is the `D3DLIGHT9::type_` (1=POINT, 2=SPOT,
/// 3=DIRECTIONAL); cached so neither `build_vs_key` nor
/// `build_lights_section` re-reads it.
struct ActiveLight {
    light: D3DLIGHT9,
    ty: u32,
}

/// Compacted ordered list of currently-active lights, capped at [`MAX_ACTIVE_LIGHTS`].
///
/// `len` is the number of valid leading entries; the tail is unspecified.
/// Returned by [`FfState::resolve_active_lights`] and consumed by both the
/// key derivation and the constant packing so the two agree on slot order by
/// construction.
struct ActiveLights {
    lights: [ActiveLight; MAX_ACTIVE_LIGHTS],
    len: usize,
}

impl ActiveLights {
    fn as_slice(&self) -> &[ActiveLight] {
        &self.lights[..self.len]
    }
}

/// Fixed-Function pipeline state.
///
/// `SetTransform` / `SetMaterial` etc. write into here; `build_ff_vs_key` /
/// `build_ff_ps_key` / `build_ff_vs_constants` read from it at draw time.
pub struct FfState {
    view: D3DMATRIX,
    projection: D3DMATRIX,
    /// World-matrix palette.
    ///
    /// `D3DTS_WORLD == D3DTS_WORLDMATRIX(0)` per spec (both equal raw state
    /// 256), so `palette[0]` is the single-world-matrix slot used when vertex
    /// blending is disabled. `palette[1..255]` are the additional matrices for
    /// `D3DRS_VERTEXBLEND` mode. 16 KB per device, one-time. Most games (e.g.
    /// `WoW`) only touch `palette[0]`; per-draw constant upload reads
    /// `world_palette[0..=world_palette_high_water]`, so non-blending
    /// workloads ship 64 bytes of world matrix as before.
    world_palette: [D3DMATRIX; 256],
    /// Maximum palette index ever written via `SetTransform`.
    ///
    /// Drives the per-draw upload extent so we don't ship unused identity
    /// matrices.
    world_palette_high_water: u16,
    texture_transforms: [D3DMATRIX; 8],
    material: D3DMATERIAL9,
    lights: [D3DLIGHT9; 8],
    /// One bit per light slot: bit `i` set iff `D3DLIGHT(i)` is enabled via `LightEnable(i, TRUE)`.
    ///
    /// D3D9 caps `MaxActiveLights` at 8, so `u8` covers every slot.
    light_enabled: u8,
    /// Bit `i` set iff `SetLight(i, &light)` was called with `light.Type != 0`.
    ///
    /// Updated incrementally by `set_light`. `light_active_mask()` = this
    /// AND `light_enabled` — the FF VS reads constants only for slots that
    /// are both set and enabled. Source of truth for `FfVsKey::light_active_mask`.
    light_set_mask: u8,
    /// Bit `i` set iff light slot `i` has been *defined*.
    ///
    /// I.e. `SetLight(i, ..)` or `LightEnable(i, ..)` has been called at least
    /// once. Distinct from `light_set_mask` (which gates the FF lighting
    /// contribution on a non-zero type): a defined light may have type 0.
    /// `GetLight`/`GetLightEnable` return `INVALIDCALL` for an undefined slot,
    /// per the D3D9 get-before-set contract.
    light_defined_mask: u8,
    /// Bit `i` set iff `lights[i].Type == D3DLIGHT_DIRECTIONAL`.
    ///
    /// Only meaningful when the corresponding `light_set_mask` bit is also
    /// set. Lets the emitter pick the per-light branch without re-reading
    /// `lights[i].Type`.
    light_directional_mask: u8,
    /// Bit `i` set iff `lights[i].Type == D3DLIGHT_SPOT`.
    ///
    /// Same lifecycle as `light_directional_mask`; a slot with neither bit set
    /// is POINT.
    light_spot_mask: u8,
    /// Sparse storage for light slots at or beyond the 8 fast-path slots.
    ///
    /// D3D9 lets `SetLight` / `LightEnable` address an unbounded set of indices;
    /// `MaxActiveLights` (8) caps only how many lights simultaneously contribute
    /// to a draw, not the addressable index range. Slots `0..8` live in the
    /// `lights` array — the only slots the FF vertex shader reads — so they stay
    /// on the fast path; higher indices are kept here purely so `GetLight` /
    /// `GetLightEnable` round-trip. Overflow lights never feed FF lighting and,
    /// like the array slots, are not captured by [`FfStateSnapshot`] beyond the
    /// 8 fast-path slots. Empty for every workload that stays within 8 lights.
    overflow_lights: BTreeMap<u32, OverflowLight>,
    texture_stage_states: [[u32; TEXTURE_STAGE_STATE_COUNT]; 8],
    /// Bit `s` set iff stage `s`'s `D3DTSS_TEXTURETRANSFORMFLAGS` is non-zero.
    ///
    /// I.e. the per-stage texture transform contributes to the FF VS const
    /// upload. Maintained by `set_texture_stage_state`.
    tt_active_mask: u8,
    /// Per-(stage, ty) once-warn latch for unsupported D3DTSS_* writes.
    ///
    /// Bit `ty` of `tss_warn_fired[stage]` is set after the first warn
    /// for that pair. `TEXTURE_STAGE_STATE_COUNT == 33`, so a single
    /// `u64` per stage covers every legal `ty` index with bits to spare.
    tss_warn_fired: [u64; 8],
    /// Per-section dirty bits for the FF VS const buffer.
    ///
    /// Set by setters (and by `SetRenderState` for the RS-driven sections)
    /// whenever a section's source values change; consumed and cleared by
    /// [`take_ff_vs_dirty`] when `emit_snapshot_deltas` walks dirty sections
    /// and pushes `Op::SetFfVsConstRange` deltas onto the encoder's mirror.
    /// Initial state is [`FfVsDirty::all()`] so the first FF draw uploads
    /// every section.
    ff_vs_dirty: FfVsDirty,
}

impl Default for FfState {
    fn default() -> Self {
        Self::new()
    }
}

impl FfState {
    #[must_use]
    pub fn new() -> Self {
        use mtld3d_types::texture_stage_state_defaults;
        Self {
            view: D3DMATRIX::IDENTITY,
            projection: D3DMATRIX::IDENTITY,
            world_palette: [D3DMATRIX::IDENTITY; 256],
            world_palette_high_water: 0,
            texture_transforms: [D3DMATRIX::IDENTITY; 8],
            material: D3DMATERIAL9::default(),
            lights: [D3DLIGHT9::default(); 8],
            light_enabled: 0,
            light_set_mask: 0,
            light_defined_mask: 0,
            light_directional_mask: 0,
            light_spot_mask: 0,
            overflow_lights: BTreeMap::new(),
            texture_stage_states: [
                texture_stage_state_defaults(0),
                texture_stage_state_defaults(1),
                texture_stage_state_defaults(2),
                texture_stage_state_defaults(3),
                texture_stage_state_defaults(4),
                texture_stage_state_defaults(5),
                texture_stage_state_defaults(6),
                texture_stage_state_defaults(7),
            ],
            tt_active_mask: 0,
            tss_warn_fired: [0; 8],
            // Cold-start: encoder mirror is zero-init, so every section
            // must be uploaded before the first FF draw reads.
            ff_vs_dirty: FfVsDirty::all(),
        }
    }

    /// Mark one or more sections of the FF VS const buffer as dirty.
    ///
    /// The next `emit_snapshot_deltas` re-emits them. Idempotent — repeated
    /// calls between draws coalesce into a single per-section delta.
    #[inline]
    pub fn mark_ff_vs_dirty(&mut self, bits: FfVsDirty) {
        self.ff_vs_dirty |= bits;
    }

    /// Returns the current dirty mask and clears it.
    ///
    /// Called from `emit_snapshot_deltas` at the start of the FF VS arm; the
    /// returned mask drives one `Op::SetFfVsConstRange` per dirty section.
    #[inline]
    pub const fn take_ff_vs_dirty(&mut self) -> FfVsDirty {
        let bits = self.ff_vs_dirty;
        self.ff_vs_dirty = FfVsDirty::empty();
        bits
    }

    /// Translate a D3DTS_* index to a mutable matrix slot.
    ///
    /// Returns `None` for indices outside the recognised set (texture
    /// transforms 8..255, etc.). `D3DTS_WORLD == D3DTS_WORLDMATRIX(0) == 256`,
    /// so the world-matrix palette covers both names through one range arm.
    fn transform_slot_mut(&mut self, state: u32) -> Option<&mut D3DMATRIX> {
        use mtld3d_types::{D3DTS_PROJECTION, D3DTS_TEXTURE0, D3DTS_TEXTURE7, D3DTS_VIEW};
        match state {
            D3DTS_VIEW => Some(&mut self.view),
            D3DTS_PROJECTION => Some(&mut self.projection),
            s if (D3DTS_TEXTURE0..=D3DTS_TEXTURE7).contains(&s) => {
                Some(&mut self.texture_transforms[(s - D3DTS_TEXTURE0) as usize])
            }
            // D3DTS_WORLDMATRIX(i) for i in 0..=255 → palette[i].
            s if (256..=511).contains(&s) => Some(&mut self.world_palette[(s - 256) as usize]),
            _ => None,
        }
    }

    /// Read-only counterpart of `transform_slot_mut`.
    ///
    /// Returns `None` for D3DTS_* indices we don't honour.
    #[must_use]
    pub fn transform(&self, state: u32) -> Option<&D3DMATRIX> {
        use mtld3d_types::{D3DTS_PROJECTION, D3DTS_TEXTURE0, D3DTS_TEXTURE7, D3DTS_VIEW};
        match state {
            D3DTS_VIEW => Some(&self.view),
            D3DTS_PROJECTION => Some(&self.projection),
            s if (D3DTS_TEXTURE0..=D3DTS_TEXTURE7).contains(&s) => {
                Some(&self.texture_transforms[(s - D3DTS_TEXTURE0) as usize])
            }
            s if (256..=511).contains(&s) => Some(&self.world_palette[(s - 256) as usize]),
            _ => None,
        }
    }

    /// Map a `D3DTS_*` index to the FF VS const-buffer section(s) it invalidates.
    ///
    /// Returns `FfVsDirty::empty()` for indices that don't feed the FF VS
    /// const buffer.
    ///
    /// `D3DTS_VIEW` invalidates `WV` (rows 0-3 = world×view), `PALETTE`
    /// (rows 95+ = palette[i]×view), AND `LIGHTS` (rows 15+): FF light
    /// positions/directions are packed in EYE space (transformed by VIEW), so
    /// a mid-frame view change restamps them. The conservative double-mark
    /// costs at most one redundant PALETTE delta-op when vertex blending
    /// is off — the encoder writes to a mirror slot the shader never
    /// reads — and avoids a stale-palette bug if vertex blending turns
    /// on later in the frame.
    fn transform_dirty_section(state: u32) -> FfVsDirty {
        use mtld3d_types::{
            D3DTS_PROJECTION, D3DTS_TEXTURE0, D3DTS_TEXTURE7, D3DTS_VIEW, D3DTS_WORLD,
        };
        match state {
            D3DTS_VIEW => FfVsDirty::WV | FfVsDirty::PALETTE | FfVsDirty::LIGHTS,
            D3DTS_PROJECTION => FfVsDirty::PROJ,
            s if (D3DTS_TEXTURE0..=D3DTS_TEXTURE7).contains(&s) => FfVsDirty::TT,
            // D3DTS_WORLD == D3DTS_WORLDMATRIX(0) feeds WV via
            // `palette[0] × view` (rows 0-3) AND PALETTE bone 0 (rows
            // 95-98) when vertex blending is active. Per-section emit
            // must mark both. PALETTE rebuild short-circuits to None
            // when blending is off, so the extra mark is free in the
            // common case.
            D3DTS_WORLD => FfVsDirty::WV | FfVsDirty::PALETTE,
            // D3DTS_WORLDMATRIX(i>0): only consulted when vertex blending
            // is enabled. Lives in the PALETTE section at rows 95+.
            s if (257..=511).contains(&s) => FfVsDirty::PALETTE,
            _ => FfVsDirty::empty(),
        }
    }

    /// Write matrix `m` into the slot identified by `state`.
    ///
    /// Returns `false` for unrecognised D3DTS_* indices (silently accepted by
    /// D3D9).
    pub fn set_transform(&mut self, state: u32, m: &D3DMATRIX) -> bool {
        self.bump_palette_high_water(state);
        let section = Self::transform_dirty_section(state);
        let Some(slot) = self.transform_slot_mut(state) else {
            mtld3d_shared::log_once_warn_by!(
                target: crate::LOG_TARGET,
                key: u64::from(state),
                "SetTransform: D3DTS_{state} not honoured — value dropped"
            );
            return false;
        };
        *slot = *m;
        self.ff_vs_dirty |= section;
        true
    }

    /// Left-multiply the slot identified by `state` by `rhs`.
    ///
    /// Returns `false` for unrecognised D3DTS_* indices.
    pub fn multiply_transform(&mut self, state: u32, rhs: &D3DMATRIX) -> bool {
        self.bump_palette_high_water(state);
        let section = Self::transform_dirty_section(state);
        let Some(slot) = self.transform_slot_mut(state) else {
            mtld3d_shared::log_once_warn_by!(
                target: crate::LOG_TARGET,
                key: u64::from(state),
                "MultiplyTransform: D3DTS_{state} not honoured — value dropped"
            );
            return false;
        };
        *slot = Self::mat_mul(slot, rhs);
        self.ff_vs_dirty |= section;
        true
    }

    /// Track the highest world-palette index ever set.
    ///
    /// Per-draw constant uploads then only pack
    /// `world_palette[0..=high_water]` instead of the full 16 KB array.
    /// `D3DTS_WORLD == D3DTS_WORLDMATRIX(0) == 256` always keeps
    /// `high_water >= 0` (the default value), so non-blending workloads pay
    /// one matrix as before.
    fn bump_palette_high_water(&mut self, state: u32) {
        if (256..=511).contains(&state) {
            // (state - 256) is in 0..=255, well inside u16.
            let idx = u16::try_from(state - 256)
                .expect("D3D9 D3DTS_WORLDMATRIX range bounds (state - 256) ≤ 255");
            if idx > self.world_palette_high_water {
                self.world_palette_high_water = idx;
            }
        }
    }

    /// Number of world matrices the game has set via `SetTransform`.
    ///
    /// Or 1 if only `D3DTS_WORLD` / `palette[0]` was touched. Drives the
    /// per-draw constant-upload extent in [`Self::build_palette_section`].
    #[must_use]
    pub const fn world_palette_used(&self) -> usize {
        self.world_palette_high_water as usize + 1
    }

    /// Read-only access to the world-matrix palette.
    ///
    /// Slice length is bounded by `world_palette_used()` for callers that want
    /// only the in-use range.
    #[must_use]
    pub const fn world_palette(&self) -> &[D3DMATRIX; 256] {
        &self.world_palette
    }

    #[must_use]
    pub const fn material(&self) -> &D3DMATERIAL9 {
        &self.material
    }

    pub fn set_material(&mut self, m: &D3DMATERIAL9) {
        self.material = *m;
        self.ff_vs_dirty |= FfVsDirty::MATERIAL;
    }

    #[must_use]
    pub const fn light(&self, index: usize) -> &D3DLIGHT9 {
        &self.lights[index]
    }

    pub fn set_light(&mut self, index: usize, light: &D3DLIGHT9) {
        self.lights[index] = *light;
        let bit = 1u8 << index;
        self.light_defined_mask |= bit;
        if light.type_ != 0 {
            self.light_set_mask |= bit;
        } else {
            self.light_set_mask &= !bit;
        }
        if light.type_ == D3DLIGHT_DIRECTIONAL {
            self.light_directional_mask |= bit;
        } else {
            self.light_directional_mask &= !bit;
        }
        if light.type_ == D3DLIGHT_SPOT {
            self.light_spot_mask |= bit;
        } else {
            self.light_spot_mask &= !bit;
        }
        self.ff_vs_dirty |= FfVsDirty::LIGHTS;
    }

    /// Bit `i` set iff slot `i` contributes constants to the FF VS.
    ///
    /// I.e. has a non-zero D3DLIGHT9 type AND is enabled via `LightEnable`.
    /// Source of truth for `FfVsKey::light_active_mask` and the inline
    /// `max_const_row` derivation in [`Self::ff_vs_row_count`].
    #[inline]
    #[must_use]
    pub const fn light_active_mask(&self) -> u8 {
        self.light_set_mask & self.light_enabled
    }

    /// Compact every currently-active light into contiguous shader slots.
    ///
    /// D3D9 allows up to `MaxActiveLights` (8) lights at *arbitrary* indices;
    /// the FF VS const layout reserves a dense `light[0..n]` range, so a light
    /// enabled at e.g. index 123 must occupy shader slot 0. Walk both light
    /// stores in overall ascending D3D9-index order — the fast-path slots
    /// `0..8` (bit set in `light_set_mask & light_enabled`) first, then the
    /// sparse `overflow_lights` (indices ≥ 8, already ascending in the
    /// `BTreeMap`) — and truncate to [`MAX_ACTIVE_LIGHTS`]. An overflow entry
    /// contributes only when enabled and carrying a non-zero light type,
    /// mirroring the `light_set_mask & light_enabled` gate the fast path uses.
    ///
    /// For the common contiguous layout (lights at 0,1,2,…) the result is
    /// identical to the physical slots, so the derived key masks and packed
    /// constants are byte-for-byte unchanged.
    fn resolve_active_lights(&self) -> ActiveLights {
        let mut out = ActiveLights {
            lights: core::array::from_fn(|_| ActiveLight {
                light: D3DLIGHT9::default(),
                ty: 0,
            }),
            len: 0,
        };
        let active = self.light_active_mask();
        for i in 0..8usize {
            if out.len == MAX_ACTIVE_LIGHTS {
                return out;
            }
            if (active & (1u8 << i)) != 0 {
                let light = self.lights[i];
                out.lights[out.len] = ActiveLight {
                    light,
                    ty: light.type_,
                };
                out.len += 1;
            }
        }
        for slot in self.overflow_lights.values() {
            if out.len == MAX_ACTIVE_LIGHTS {
                return out;
            }
            if slot.enabled && slot.light.type_ != 0 {
                out.lights[out.len] = ActiveLight {
                    light: slot.light,
                    ty: slot.light.type_,
                };
                out.len += 1;
            }
        }
        out
    }

    /// Bit `i` set iff the active light at slot `i` is DIRECTIONAL.
    ///
    /// Only meaningful when the corresponding `light_active_mask` bit is set.
    #[inline]
    #[must_use]
    pub const fn light_directional_mask(&self) -> u8 {
        self.light_directional_mask
    }

    /// Bit `i` set iff the active light at slot `i` is SPOT.
    ///
    /// Only meaningful when the corresponding `light_active_mask` bit is set.
    #[inline]
    #[must_use]
    pub const fn light_spot_mask(&self) -> u8 {
        self.light_spot_mask
    }

    /// Bit `s` set iff stage `s` has non-zero `D3DTSS_TEXTURETRANSFORMFLAGS`.
    #[inline]
    #[must_use]
    pub const fn tt_active_mask(&self) -> u8 {
        self.tt_active_mask
    }

    /// Re-derive `tt_active_mask` from the texture-stage-state array.
    ///
    /// The setter maintains it incrementally; a bulk array restore
    /// (state-block `Apply`) bypasses the setter and must call this afterwards
    /// or the mask goes stale against the restored array. The light masks
    /// cannot be re-derived the same way (an untouched slot's
    /// `D3DLIGHT9::default()` is indistinguishable from an explicit
    /// directional `SetLight`), so they travel with [`FfStateSnapshot`]
    /// instead.
    fn recompute_tt_active_mask(&mut self) {
        self.tt_active_mask = 0;
        for (s, stage) in self.texture_stage_states.iter().enumerate() {
            if stage[D3DTSS_TEXTURETRANSFORMFLAGS as usize] != 0 {
                self.tt_active_mask |= 1u8 << s;
            }
        }
    }

    #[must_use]
    pub const fn light_enabled(&self, index: usize) -> bool {
        (self.light_enabled & (1u8 << index)) != 0
    }

    /// Bit `i` set iff slot `i` has been defined via `SetLight`/`LightEnable`.
    ///
    /// `GetLight`/`GetLightEnable` fail (`INVALIDCALL`) for undefined slots.
    #[must_use]
    pub const fn light_defined(&self, index: usize) -> bool {
        (self.light_defined_mask & (1u8 << index)) != 0
    }

    pub fn set_light_enabled(&mut self, index: usize, enabled: bool) {
        let bit = 1u8 << index;
        // D3D9: `LightEnable` on an undefined slot first creates a light with
        // the default directional parameters (white diffuse, direction +Z), so
        // a subsequent `GetLight` reports those defaults. Only the FF lighting
        // contribution (gated by `light_set_mask`) still requires an explicit
        // `SetLight`, so this does not change rendering for enable-only lights.
        if self.light_defined_mask & bit == 0 {
            self.light_defined_mask |= bit;
            self.lights[index] = Self::enable_default_light();
        }
        if enabled {
            self.light_enabled |= bit;
        } else {
            self.light_enabled &= !bit;
        }
        // Toggling enable flips a single light's contribution; the type-w
        // value at row `15 + index*6` depends on `light_active_mask` which
        // includes this bit. Mark the whole LIGHTS section dirty rather
        // than tracking per-slot — the LIGHTS section ships as one delta.
        self.ff_vs_dirty |= FfVsDirty::LIGHTS;
    }

    /// The light D3D9 materializes when `LightEnable` targets a slot with no `SetLight`.
    ///
    /// White diffuse over the otherwise-zeroed `D3DLIGHT9` default.
    fn enable_default_light() -> D3DLIGHT9 {
        D3DLIGHT9 {
            diffuse: D3DCOLORVALUE {
                r: 1.0,
                g: 1.0,
                b: 1.0,
                a: 0.0,
            },
            ..D3DLIGHT9::default()
        }
    }

    /// `SetLight` for any D3D9 light index.
    ///
    /// Slots `0..8` take the fast path and feed FF lighting; higher indices
    /// land in `overflow_lights` for `GetLight` round-trip only.
    pub fn set_light_at(&mut self, index: u32, light: &D3DLIGHT9) {
        if index < 8 {
            self.set_light(index as usize, light);
        } else {
            self.overflow_lights
                .entry(index)
                .and_modify(|slot| slot.light = *light)
                .or_insert(OverflowLight {
                    light: *light,
                    enabled: false,
                });
        }
    }

    /// `GetLight` for any D3D9 light index.
    ///
    /// `None` for an undefined slot (the caller maps that to `INVALIDCALL`).
    #[must_use]
    pub fn get_light_at(&self, index: u32) -> Option<D3DLIGHT9> {
        if index < 8 {
            let i = index as usize;
            self.light_defined(i).then(|| *self.light(i))
        } else {
            self.overflow_lights.get(&index).map(|slot| slot.light)
        }
    }

    /// `LightEnable` for any D3D9 light index.
    ///
    /// An undefined slot first materializes the default light, matching the
    /// fast-path behavior.
    pub fn set_light_enabled_at(&mut self, index: u32, enabled: bool) {
        if index < 8 {
            self.set_light_enabled(index as usize, enabled);
        } else {
            self.overflow_lights
                .entry(index)
                .or_insert_with(|| OverflowLight {
                    light: Self::enable_default_light(),
                    enabled: false,
                })
                .enabled = enabled;
        }
    }

    /// `GetLightEnable` for any D3D9 light index.
    #[must_use]
    pub fn is_light_enabled_at(&self, index: u32) -> bool {
        if index < 8 {
            self.light_enabled(index as usize)
        } else {
            self.overflow_lights
                .get(&index)
                .is_some_and(|slot| slot.enabled)
        }
    }

    /// Whether any light index has been defined via `SetLight` / `LightEnable`.
    #[must_use]
    pub fn is_light_defined_at(&self, index: u32) -> bool {
        if index < 8 {
            self.light_defined(index as usize)
        } else {
            self.overflow_lights.contains_key(&index)
        }
    }

    #[must_use]
    pub const fn texture_stage_state(&self, stage: usize, ty: usize) -> u32 {
        self.texture_stage_states[stage][ty]
    }

    /// Pack the per-stage bump-environment matrix + luminance scale/offset for upload.
    ///
    /// The destination is the PS slot-12 uniform consumed by
    /// `texbem`/`texbeml`/`bem`. Layout per stage `s`:
    /// `bump_env[s*2] = (m00, m01, m10, m11)` and
    /// `bump_env[s*2+1] = (lscale, loffset, 0, 0)` — matching the indexing the
    /// DXSO emitter uses (`bump_matrix_exprs` / `bump_lum_exprs`). Eight stages,
    /// 256 bytes total. The TSS values are stored as `u32` bit patterns of the
    /// `f32` the game wrote via `SetTextureStageState`.
    #[must_use]
    pub fn build_bump_env_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(8 * 2 * 16);
        let f =
            |stage: usize, ty: u32| f32::from_bits(self.texture_stage_states[stage][ty as usize]);
        for stage in 0..8 {
            for v in [
                f(stage, D3DTSS_BUMPENVMAT00),
                f(stage, D3DTSS_BUMPENVMAT01),
                f(stage, D3DTSS_BUMPENVMAT10),
                f(stage, D3DTSS_BUMPENVMAT11),
                f(stage, D3DTSS_BUMPENVLSCALE),
                f(stage, D3DTSS_BUMPENVLOFFSET),
                0.0,
                0.0,
            ] {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        bytes
    }

    #[must_use]
    pub const fn tss_warn_fired(&self, stage: usize, ty: usize) -> bool {
        (self.tss_warn_fired[stage] & (1u64 << ty)) != 0
    }

    const fn mark_tss_warn(&mut self, stage: usize, ty: usize) {
        self.tss_warn_fired[stage] |= 1u64 << ty;
    }

    /// Returns whether the stored value actually changed.
    ///
    /// Callers gate snapshot dirty-marking on this: a same-value write (very
    /// common in state-block restores) leaves every FF VS/PS key
    /// byte-identical.
    pub fn set_texture_stage_state(&mut self, stage: usize, ty: usize, value: u32) -> bool {
        self.warn_tss_non_default_once(stage, ty, value);
        let changed = self.texture_stage_states[stage][ty] != value;
        self.texture_stage_states[stage][ty] = value;
        if ty == D3DTSS_TEXTURETRANSFORMFLAGS as usize && stage < 8 {
            let bit = 1u8 << stage;
            let was_active = (self.tt_active_mask & bit) != 0;
            if value != 0 {
                self.tt_active_mask |= bit;
            } else {
                self.tt_active_mask &= !bit;
            }
            let now_active = (self.tt_active_mask & bit) != 0;
            // Only mark TT dirty when the per-stage active bit actually
            // flipped — TTFF writes with the same flag value (very common
            // in WoW state-block restores) leave the row contents
            // unchanged.
            if was_active != now_active {
                self.ff_vs_dirty |= FfVsDirty::TT;
            }
        }
        changed
    }

    fn warn_tss_non_default_once(&mut self, stage: usize, ty: usize, value: u32) {
        static TSS_DEFAULTS: [[u32; TEXTURE_STAGE_STATE_COUNT]; 8] = [
            texture_stage_state_defaults(0),
            texture_stage_state_defaults(1),
            texture_stage_state_defaults(2),
            texture_stage_state_defaults(3),
            texture_stage_state_defaults(4),
            texture_stage_state_defaults(5),
            texture_stage_state_defaults(6),
            texture_stage_state_defaults(7),
        ];

        if stage >= 8 || ty >= TEXTURE_STAGE_STATE_COUNT {
            return;
        }
        if value == TSS_DEFAULTS[stage][ty] {
            if crate::state_trace::enabled() {
                log::trace!(
                    target: crate::state_trace::TARGET,
                    "D3DTSS_{ty} (stage {stage}) = {value:#x} (default — write suppressed in warn machinery)"
                );
            }
            return;
        }
        if self.tss_warn_fired(stage, ty) {
            return;
        }
        let class = tss_classify(
            u32::try_from(ty).expect("D3DTSS type fits u32 by TEXTURE_STAGE_STATE_COUNT bound"),
        );
        if matches!(class, TssClass::Consumed) {
            if crate::state_trace::enabled() {
                let default = TSS_DEFAULTS[stage][ty];
                log::trace!(
                    target: crate::state_trace::TARGET,
                    "D3DTSS_{ty} (stage {stage}) Consumed = {value:#x} (default {default:#x})"
                );
            }
            return;
        }
        self.mark_tss_warn(stage, ty);
        let default = TSS_DEFAULTS[stage][ty];
        match class {
            TssClass::Consumed => {} // unreachable
            TssClass::NotImplemented => {
                log::warn!(
                    target: LOG_TARGET,
                    "D3DTSS_{ty} (stage {stage}) = {value:#x} (default {default:#x}) written but not consumed"
                );
            }
        }
    }

    /// Row-major 4x4 matrix multiply: `out = a * b`.
    ///
    /// D3D9 matrices are stored in row-major order (same as the on-disk
    /// `D3DMATRIX`), and we feed them to MSL the same way — with
    /// `dot(row, vert)` matmul instructions.
    #[must_use]
    pub fn mat_mul(a: &D3DMATRIX, b: &D3DMATRIX) -> D3DMATRIX {
        let mut r = [0.0f32; 16];
        for i in 0..4 {
            for j in 0..4 {
                let mut s = 0.0;
                for k in 0..4 {
                    s += a.m[i * 4 + k] * b.m[k * 4 + j];
                }
                r[i * 4 + j] = s;
            }
        }
        D3DMATRIX { m: r }
    }

    /// Transform a position into eye space by the VIEW matrix, D3D9 row-vector convention.
    ///
    /// `p_eye = (pos.xyz, 1) * view`, i.e.
    /// `p_eye[j] = pos.x·view[0][j] + pos.y·view[1][j] + pos.z·view[2][j] + view[3][j]`
    /// for `j = 0..3` (the `w = 1` lane pulls in the translation row). Used to
    /// pack FF light positions, which the shader reads already in eye space.
    fn view_transform_point(&self, pos: &mtld3d_types::D3DVECTOR) -> [f32; 3] {
        let v = &self.view.m;
        let mut out = [0.0f32; 3];
        for (j, o) in out.iter_mut().enumerate() {
            // Array-sum rather than an `a*b + c*d` polynomial: the polynomial
            // tempts clippy's `suboptimal_flops` toward `mul_add`, which lowers
            // to a ucrtbase `fmaf` LIBCALL on the i686 SSE2 baseline (no FMA).
            // Summing separate products keeps the multiplies and adds unfused.
            *o = [pos.x * v[j], pos.y * v[4 + j], pos.z * v[8 + j], v[12 + j]]
                .iter()
                .sum();
        }
        out
    }

    /// Transform a direction into eye space by the VIEW matrix, D3D9 row-vector convention.
    ///
    /// With `w = 0` (no translation row): `d_eye[j] = dir.x·view[0][j] +
    /// dir.y·view[1][j] + dir.z·view[2][j]`. Used to pack FF light directions;
    /// the caller normalizes afterwards.
    fn view_transform_dir(&self, dir: &mtld3d_types::D3DVECTOR) -> [f32; 3] {
        let v = &self.view.m;
        let mut out = [0.0f32; 3];
        for (j, o) in out.iter_mut().enumerate() {
            // Array-sum, not a polynomial — see `view_transform_point` (avoids
            // an `fmaf` libcall on the no-FMA i686 baseline).
            *o = [dir.x * v[j], dir.y * v[4 + j], dir.z * v[8 + j]]
                .iter()
                .sum();
        }
        out
    }

    /// The composite `WorldViewProjection` in D3D9 row-vector order.
    ///
    /// The resulting matrix `M` is the one the app would use for
    /// `clip = pos * M`.
    #[must_use]
    pub fn world_view_projection(&self) -> D3DMATRIX {
        let wv = Self::mat_mul(&self.world_palette[0], &self.view);
        Self::mat_mul(&wv, &self.projection)
    }

    /// Transpose for upload.
    ///
    /// We expose D3D9 matrices to MSL via `dot(pos, c[i])`, which requires the
    /// stored `c[i]` to hold the i-th *column* of the original matrix — i.e.
    /// the row of the transpose.
    #[must_use]
    pub fn transpose(m: &D3DMATRIX) -> D3DMATRIX {
        let mut r = [0.0f32; 16];
        for i in 0..4 {
            for j in 0..4 {
                r[i * 4 + j] = m.m[j * 4 + i];
            }
        }
        D3DMATRIX { m: r }
    }

    /// `bound_texture_mask` has bit `i` set if stage `i` has a texture bound.
    ///
    /// Mirrors `build_ps_key`; used to decide which stages need a VS output
    /// varying.
    ///
    /// # Panics
    ///
    /// Panics if a stage's TCI mode falls outside the documented 0..=4 range
    /// — clamped by the decoder upstream, so unreachable on real D3D9 state.
    #[must_use]
    pub fn build_vs_key(
        &self,
        render_states: &[u32; RENDER_STATE_COUNT],
        layout: FfVsLayout,
        bound_texture_mask: u8,
    ) -> FfVsKey {
        // D3D9 spec: XYZRHW bypasses per-vertex lighting regardless of
        // D3DRS_LIGHTING. Encode that here so `emit_vs` doesn't gate on the
        // combination itself.
        let lighting_enabled = !layout.has_rhw() && render_states[D3DRS_LIGHTING as usize] != 0;
        // Derive the per-shader-slot masks from the COMPACTED active-light
        // list, not the physical slots: D3D9 lights live at arbitrary indices
        // but the FF VS reads a dense `light[0..n]` range, so bit `i` here
        // refers to compacted shader slot `i`, matching the packing order in
        // `build_lights_section`. Active = the low `n` bits set; the type
        // masks carry the per-slot DIRECTIONAL / SPOT branch (a slot with
        // neither bit is POINT). Lighting off (or XYZRHW) zeroes the
        // contribution. For the common contiguous layout the compacted list
        // equals the physical slots, so these masks are byte-identical to the
        // pre-compaction `light_active_mask()` / `light_directional_mask()` /
        // `light_spot_mask()`.
        let (light_active_mask, light_directional_mask, light_spot_mask) = if lighting_enabled {
            let active = self.resolve_active_lights();
            let mut dir = 0u8;
            let mut spot = 0u8;
            for (i, l) in active.as_slice().iter().enumerate() {
                let bit = 1u8 << i;
                if l.ty == D3DLIGHT_DIRECTIONAL {
                    dir |= bit;
                } else if l.ty == D3DLIGHT_SPOT {
                    spot |= bit;
                }
            }
            // `len` ≤ MAX_ACTIVE_LIGHTS (8); the active mask is the low `len`
            // bits set. `0xFFu8 >> (8 - len)` yields exactly that (and 0 for
            // len == 0, all-ones for len == 8) without an overflowing shift.
            let active_mask = if active.len == 0 {
                0u8
            } else {
                0xFFu8 >> (8 - active.len)
            };
            (active_mask, dir, spot)
        } else {
            (0, 0, 0)
        };
        // Decode per-stage TCI mode + input coord-set index from
        // `D3DTSS_TEXCOORDINDEX`, and texture-transform flags from
        // `D3DTSS_TEXTURETRANSFORMFLAGS`. High byte (bits 16..19) of
        // TEXCOORDINDEX = TCI_*; low 16 bits = coord-set index for passthru.
        // TEXTURETRANSFORMFLAGS: low 3 bits = count, bit 8 = PROJECTED;
        // pack into the VS key's one-byte format (low 3 = count,
        // bit 4 = projected).
        //
        // TCI / TTFF must be read for every stage regardless of `COLOROP` —
        // `D3DTSS_COLOROP == D3DTOP_DISABLE` ends the *FF PS* color-blend
        // chain, but programmable PS draws using FF VS still depend on the
        // VS routing the right coord-set to each stage the PS samples.
        // Stopping the loop at the first `COLOROP_DISABLE` would leave
        // `tci_coord_indices[1..]` at their `[0; 8]` init for the very
        // common pattern of "game leaves stages 1..N at default
        // `COLOROP_DISABLE` while a programmable PS does its own sampling",
        // collapsing every VS texcoord output onto coord-set 0 (v4) — fine
        // when the VB carries one UV set, catastrophically wrong when
        // stages expect distinct coord sets (e.g. v4 = tiled distortion UV,
        // v5 = normalized scene UV, v6 = second scene UV).
        //
        // `max_active_stage` and `tex_coord_count` keep the original FF-PS-
        // aware semantics: default-state stages (COLOROP=MODULATE + no
        // texture, or any COLOROP=DISABLE chain terminator) must NOT
        // inflate `tex_coord_count`, or every draw on defaults would emit
        // a dead varying and trip `passthru_rhs`'s out-of-range fallback
        // warn. `input_tex_coord_count` stays pinned to the vertex-stream
        // count so the `VertexIn` struct only declares attributes that
        // `resolve_attrs_for_ff` populates in the MTLVertexDescriptor.
        let mut tci_modes = [0u8; 8];
        let mut tci_coord_indices = [0u8; 8];
        let mut tt_flags = [0u8; 8];
        for (i, stage_state) in self.texture_stage_states.iter().enumerate() {
            let raw = stage_state[D3DTSS_TEXCOORDINDEX as usize];
            tci_modes[i] = raw.to_le_bytes()[2]; // bits 16..23
            tci_coord_indices[i] = raw.to_le_bytes()[0]; // bits 0..7
            let ttff = stage_state[D3DTSS_TEXTURETRANSFORMFLAGS as usize];
            // Only D3DTTFF_COUNT2..4 trigger the texture-matrix multiply.
            // D3DTTFF_DISABLE (0), D3DTTFF_COUNT1 (1), and any value above
            // COUNT4 (the test passes 0xffffff03 etc.) pass the coordinate
            // through untransformed — store count 0 for those. The PROJECTED
            // bit is orthogonal and preserved regardless.
            let eff = ttff & !D3DTTFF_PROJECTED;
            let count = if (2..=4).contains(&eff) {
                u8::try_from(eff).expect("eff in 2..=4 fits u8")
            } else {
                0
            };
            let projected = if (ttff & D3DTTFF_PROJECTED) != 0 {
                0x10u8
            } else {
                0
            };
            tt_flags[i] = count | projected;
        }
        let mut max_active_stage: Option<u8> = None;
        for (i, stage_state) in self.texture_stage_states.iter().enumerate() {
            if stage_state[D3DTSS_COLOROP as usize] == D3DTOP_DISABLE {
                break;
            }
            if (bound_texture_mask >> i) & 1 != 0 {
                max_active_stage = Some(u8::try_from(i).expect("stage index ≤ 7 fits u8"));
            }
        }
        // `.min(8)` is defensive: `ff_vs_layout_from_elements` already
        // clamps, but keep the invariant enforced here so a future layout
        // source can't reintroduce OOB into FfVsKey's [u8; 8] per-stage
        // arrays (tci_modes, tci_coord_indices, tt_flags).
        let tex_coord_count = layout
            .tex_coord_count
            .max(max_active_stage.map_or(0, |m| m + 1))
            .min(8);
        assert!(
            tex_coord_count <= 8,
            "FfState::build_vs_key clamp violated: tex_coord_count={tex_coord_count}"
        );

        let vertex_blend_indexed = render_states[D3DRS_INDEXEDVERTEXBLENDENABLE as usize] != 0;
        let vertex_blend_count = resolve_vertex_blend_count(
            render_states[D3DRS_VERTEXBLEND as usize],
            layout,
            vertex_blend_indexed,
        );

        let flags = build_vs_flags(
            render_states,
            layout,
            lighting_enabled,
            vertex_blend_indexed,
        );

        FfVsKey {
            flags,
            input_tex_coord_count: layout.tex_coord_count,
            tex_coord_count,
            light_active_mask,
            light_directional_mask,
            light_spot_mask,
            diffuse_source: clamp_material_source(
                render_states[D3DRS_DIFFUSEMATERIALSOURCE as usize],
                "DIFFUSEMATERIALSOURCE",
            ),
            ambient_source: clamp_material_source(
                render_states[D3DRS_AMBIENTMATERIALSOURCE as usize],
                "AMBIENTMATERIALSOURCE",
            ),
            specular_source: clamp_material_source(
                render_states[D3DRS_SPECULARMATERIALSOURCE as usize],
                "SPECULARMATERIALSOURCE",
            ),
            emissive_source: clamp_material_source(
                render_states[D3DRS_EMISSIVEMATERIALSOURCE as usize],
                "EMISSIVEMATERIALSOURCE",
            ),
            fog_mode: resolve_fog_config(render_states, layout.has_rhw()).vertex_mode,
            tci_modes,
            tci_coord_indices,
            tex_coord_dims: layout.tex_coord_dims,
            tt_flags,
            vertex_blend_count,
            declared_weights_count: layout.declared_weights_count,
        }
    }

    /// `bound_texture_mask` has bit `i` set if stage `i` has a texture bound.
    ///
    /// # Panics
    ///
    /// Panics if a D3DTSS COLOROP/ARG/ALPHAOP/ALPHAARG value exceeds `u8::MAX`
    /// — unreachable, the spec caps each at ≤ 24.
    #[must_use]
    pub fn build_ps_key(
        &self,
        render_states: &[u32; RENDER_STATE_COUNT],
        bound_texture_mask: u8,
    ) -> FfPsKey {
        let mut stages = [FfStage::default(); 8];
        for (i, stage) in stages.iter_mut().enumerate() {
            let s = &self.texture_stage_states[i];
            let to_u8 = |v: u32| u8::try_from(v).expect("D3DTSS op/arg ≤ 24");
            stage.color_op = to_u8(s[D3DTSS_COLOROP as usize]);
            stage.color_arg1 = to_u8(s[D3DTSS_COLORARG1 as usize]);
            stage.color_arg2 = to_u8(s[D3DTSS_COLORARG2 as usize]);
            stage.alpha_op = to_u8(s[D3DTSS_ALPHAOP as usize]);
            stage.alpha_arg1 = to_u8(s[D3DTSS_ALPHAARG1 as usize]);
            stage.alpha_arg2 = to_u8(s[D3DTSS_ALPHAARG2 as usize]);
            // `D3DTSS_TEXCOORDINDEX` is now consumed VS-side via
            // `FfVsKey::tci_modes` + `tci_coord_indices` (one entry per
            // stage). The PS samples `Varyings.texcoord[stage]` 1:1.
            stage.has_texture = (bound_texture_mask & (1 << i)) != 0;
        }
        FfPsKey {
            stages,
            specular_add: render_states[D3DRS_SPECULARENABLE as usize] != 0,
            tt_projected_mask: self.tt_projected_mask(),
        }
    }

    /// Per-stage `D3DTTFF_PROJECTED` mask.
    ///
    /// Bit `i` set ⇒ stage `i` has the PROJECTED transform flag. Drives the
    /// implicit per-pixel projective divide in the FF and `ps_1_0`..`ps_1_3`
    /// pixel pipelines (the programmable PS emitter folds this into
    /// `VariantKey::tt_projected_mask`).
    #[must_use]
    pub fn tt_projected_mask(&self) -> u8 {
        let mut mask = 0u8;
        for (i, s) in self.texture_stage_states.iter().enumerate() {
            if s[D3DTSS_TEXTURETRANSFORMFLAGS as usize] & D3DTTFF_PROJECTED != 0 {
                mask |= 1u8 << i;
            }
        }
        mask
    }

    /// `VariantKey` derived from current render states (alpha test + fog).
    ///
    /// Callers pass `has_rhw` so XYZRHW pipelines get `fog_mode = 0` (D3D9
    /// bypasses vertex fog for pre-transformed geometry) matching the VS key.
    ///
    /// # Panics
    ///
    /// Panics if `D3DRS_ALPHAFUNC` exceeds `u8::MAX` — unreachable, the spec
    /// caps it at ≤ 8 (`D3DCMP_ALWAYS`).
    #[must_use]
    pub fn variant_key(
        &self,
        render_states: &[u32; RENDER_STATE_COUNT],
        has_rhw: bool,
    ) -> VariantKey {
        use mtld3d_types::{D3DRS_SHADEMODE, D3DRS_SRGBWRITEENABLE, D3DSHADE_FLAT};
        let alpha_test_on = render_states[D3DRS_ALPHATESTENABLE as usize] != 0;
        let fog = resolve_fog_config(render_states, has_rhw);
        let mut flags = VariantFlags::empty();
        // Source select is only meaningful for table fog; held clear
        // otherwise so vertex-fog variants don't churn on projection changes.
        flags.set(
            VariantFlags::FOG_SOURCE_W,
            fog.table_mode != 0 && !self.projection_is_ortho(),
        );
        flags.set(
            VariantFlags::FLAT_SHADE,
            render_states[D3DRS_SHADEMODE as usize] == D3DSHADE_FLAT,
        );
        flags.set(
            VariantFlags::SRGB_WRITE,
            render_states[D3DRS_SRGBWRITEENABLE as usize] != 0,
        );
        VariantKey {
            alpha_func: if alpha_test_on {
                u8::try_from(render_states[D3DRS_ALPHAFUNC as usize]).expect("D3DCMP_* ≤ 8 fits u8")
            } else {
                0
            },
            fog_mode: fog.vertex_mode,
            fog_table_mode: fog.table_mode,
            // depth_sampler_mask / depth_fetch_mask / volume_sampler_mask /
            // tt_projected_mask are per-bind-state / texture-stage properties,
            // not render-state; the encoder folds them in at draw time.
            depth_sampler_mask: 0,
            depth_fetch_mask: 0,
            volume_sampler_mask: 0,
            tt_projected_mask: 0,
            flags,
        }
    }

    /// D3D9's pixel-fog source rule.
    ///
    /// A projection matrix whose 4th COLUMN is (0, 0, 0, 1) leaves clip W == 1
    /// for every vertex (orthographic), so table fog reads the pixel depth;
    /// anything else reads the eye W. `D3DMATRIX.m` is row-major, so the 4th
    /// column is elements 3, 7, 11, 15 (`_14`/`_24`/`_34`/`_44`).
    #[must_use]
    pub fn projection_is_ortho(&self) -> bool {
        let m = &self.projection.m;
        // Ordering compares (`abs() <= 0.0`) instead of `== 0.0` dodge
        // clippy::float_cmp while still treating -0.0 as zero; the `1.0` check
        // compares exact bits. The 4th column is elements 3/7/11/15.
        m[3].abs() <= 0.0
            && m[7].abs() <= 0.0
            && m[11].abs() <= 0.0
            && m[15].to_bits() == 1.0f32.to_bits()
    }

    #[must_use]
    pub fn alpha_ref(&self, render_states: &[u32; RENDER_STATE_COUNT]) -> f32 {
        // D3DRS_ALPHAREF is a D3DCOLOR-style unsigned byte (0..255) stored as u32.
        // Take the low byte directly so the divide is exact (u8 fits f32 mantissa).
        let [byte, _, _, _] = render_states[D3DRS_ALPHAREF as usize].to_le_bytes();
        f32::from(byte) / 255.0
    }

    /// Compute the FF VS const-buffer extent (number of rows) for a draw.
    ///
    /// Evaluated from `vs_key` against the current `FfState`. The extent
    /// matches the sum of section ranges contributed by the per-section
    /// `build_*_section` helpers; callers (the snapshot path, `emit_draw`)
    /// use it to size the encoder mirror snapshot and to bound
    /// `setVertexBytes`.
    ///
    /// # Panics
    ///
    /// Panics if `world_palette_used()` exceeds `u32` (unreachable —
    /// bounded by 256 per spec) or if the computed row count exceeds
    /// `u16` (also unreachable — `95 + 256*4 = 1119 < u16::MAX`).
    #[must_use]
    pub fn ff_vs_row_count(&self, vs_key: &FfVsKey) -> u16 {
        if vs_key.has_rhw() {
            return 1;
        }
        let lit = vs_key.lighting_enabled();
        let mut max_row: u16 = 7;
        if vs_key.fog_mode != 0 && 8 > max_row {
            max_row = 8;
        }
        if lit {
            let mat_extent = if vs_key.specular_enable() { 14 } else { 13 };
            if mat_extent > max_row {
                max_row = mat_extent;
            }
            let active = vs_key.light_active_mask;
            if active != 0 {
                let lz =
                    u8::try_from(active.leading_zeros()).expect("u8::leading_zeros ≤ 8 fits u8");
                let hi = 7u16 - u16::from(lz);
                let light_extent = 15 + hi * 6 + 5;
                if light_extent > max_row {
                    max_row = light_extent;
                }
            }
        } else if 10 > max_row {
            max_row = 10;
        }
        let tt_mask = self.tt_active_mask();
        if tt_mask != 0 {
            let lz = u8::try_from(tt_mask.leading_zeros()).expect("u8::leading_zeros ≤ 8 fits u8");
            let hi = 7u16 - u16::from(lz);
            let tt_extent = 63 + hi * 4 + 3;
            if tt_extent > max_row {
                max_row = tt_extent;
            }
        }
        let mut row_count: u32 = u32::from(max_row) + 1;
        if vs_key.vertex_blend_count > 0 {
            let used = self.world_palette_used();
            let palette_rows = 95 + u32::try_from(used).expect("palette ≤ 256") * 4;
            if palette_rows > row_count {
                row_count = palette_rows;
            }
        }
        u16::try_from(row_count).expect("FF VS row_count ≤ 95 + 256*4 fits u16")
    }

    /// XYZRHW path: pack `[vp_w, vp_h, vp_x, vp_y]` into row 0.
    ///
    /// Caller pushes a single
    /// `Op::SetFfVsConstRange { start_row: 0, rows: 1, .. }`. The non-XYZRHW
    /// path uses the per-section helpers below (one per `FfVsDirty` bit) and
    /// never enters this function.
    ///
    /// `viewport` is `(x, y, width, height)` in pixels.
    pub fn build_xyzrhw_row(viewport: (f32, f32, f32, f32), scratch: &mut ScratchArena) -> *mut u8 {
        let dst_ptr = scratch.alloc_uninit_slice::<core::mem::MaybeUninit<[f32; 4]>>(1);
        // SAFETY: `alloc_uninit_slice` reserved one 16-byte-aligned
        // `MaybeUninit<[f32; 4]>` slot; treating it as a `&mut` slice of
        // `MaybeUninit` lets the per-slot writes below be safe.
        let dst: &mut [core::mem::MaybeUninit<[f32; 4]>] =
            unsafe { core::slice::from_raw_parts_mut(dst_ptr, 1) };
        dst[0].write([viewport.2, viewport.3, viewport.0, viewport.1]);
        dst_ptr.cast::<u8>()
    }

    /// Bump-copy the row 0..4 WV section into the scratch arena.
    ///
    /// The section holds `transpose(world_palette[0] × view)`. Returns
    /// `(start_row=0, rows=4, ptr)`.
    pub fn build_wv_section(&self, scratch: &mut ScratchArena) -> (u16, u16, *mut u8) {
        let wv_t = Self::transpose(&Self::mat_mul(&self.world_palette[0], &self.view));
        let dst_ptr = scratch.alloc_uninit_slice::<core::mem::MaybeUninit<[f32; 4]>>(4);
        // SAFETY: see `build_xyzrhw_row`. Single boundary unsafe op;
        // subsequent slice writes are safe.
        let dst: &mut [core::mem::MaybeUninit<[f32; 4]>] =
            unsafe { core::slice::from_raw_parts_mut(dst_ptr, 4) };
        write_matrix_rows(dst, &wv_t);
        (0, 4, dst_ptr.cast::<u8>())
    }

    /// Bump-copy the row 4..8 PROJ section (`transpose(projection)`) into the scratch arena.
    ///
    /// Returns `(start_row=4, rows=4, ptr)`.
    pub fn build_proj_section(&self, scratch: &mut ScratchArena) -> (u16, u16, *mut u8) {
        let proj_t = Self::transpose(&self.projection);
        let dst_ptr = scratch.alloc_uninit_slice::<core::mem::MaybeUninit<[f32; 4]>>(4);
        // SAFETY: see `build_xyzrhw_row`.
        let dst: &mut [core::mem::MaybeUninit<[f32; 4]>] =
            unsafe { core::slice::from_raw_parts_mut(dst_ptr, 4) };
        write_matrix_rows(dst, &proj_t);
        (4, 4, dst_ptr.cast::<u8>())
    }

    /// Bump-copy the row 8 FOG section into the scratch arena.
    ///
    /// Reads `D3DRS_FOGSTART/END/DENSITY`; zero-fills when `fog_mode == 0` so
    /// the row contents are defined even if a different dim pulled the
    /// extent past 8. Returns `(start_row=8, rows=1, ptr)`.
    pub fn build_fog_section(
        render_states: &[u32; RENDER_STATE_COUNT],
        fog_mode: u8,
        scratch: &mut ScratchArena,
    ) -> (u16, u16, *mut u8) {
        let row = if fog_mode != 0 {
            [
                f32::from_bits(render_states[D3DRS_FOGSTART as usize]),
                f32::from_bits(render_states[D3DRS_FOGEND as usize]),
                f32::from_bits(render_states[D3DRS_FOGDENSITY as usize]),
                0.0,
            ]
        } else {
            [0.0; 4]
        };
        let dst_ptr = scratch.alloc_uninit_slice::<core::mem::MaybeUninit<[f32; 4]>>(1);
        // SAFETY: see `build_xyzrhw_row`.
        let dst: &mut [core::mem::MaybeUninit<[f32; 4]>] =
            unsafe { core::slice::from_raw_parts_mut(dst_ptr, 1) };
        dst[0].write(row);
        (8, 1, dst_ptr.cast::<u8>())
    }

    /// Bump-copy the row 9 AMBIENT section (global ambient color from `D3DRS_AMBIENT`).
    ///
    /// Returns `(start_row=9, rows=1, ptr)`.
    pub fn build_ambient_section(
        render_states: &[u32; RENDER_STATE_COUNT],
        scratch: &mut ScratchArena,
    ) -> (u16, u16, *mut u8) {
        let row = d3dcolor_to_rgba(render_states[D3DRS_AMBIENT as usize]);
        let dst_ptr = scratch.alloc_uninit_slice::<core::mem::MaybeUninit<[f32; 4]>>(1);
        // SAFETY: see `build_xyzrhw_row`.
        let dst: &mut [core::mem::MaybeUninit<[f32; 4]>] =
            unsafe { core::slice::from_raw_parts_mut(dst_ptr, 1) };
        dst[0].write(row);
        (9, 1, dst_ptr.cast::<u8>())
    }

    /// Bump-copy the row 10..14 MATERIAL section.
    ///
    /// Extent depends on the `FfVsKey`'s lit + specular bits:
    /// - unlit: 1 row (row 10; the unlit emitter defaults a missing COLOR0
    ///   to white and no longer reads it — kept so the MATERIAL section
    ///   keeps one shape across the lit flip)
    /// - lit, no specular: 4 rows (diffuse..emissive)
    /// - lit + specular: 5 rows (diffuse..emissive + power)
    ///
    /// Returns `(start_row=10, rows, ptr)`.
    pub fn build_material_section(
        &self,
        vs_key: &FfVsKey,
        scratch: &mut ScratchArena,
    ) -> (u16, u16, *mut u8) {
        let rows: u16 = if !vs_key.lighting_enabled() {
            1
        } else if vs_key.specular_enable() {
            5
        } else {
            4
        };
        let rows_usize = rows as usize;
        let dst_ptr = scratch.alloc_uninit_slice::<core::mem::MaybeUninit<[f32; 4]>>(rows_usize);
        // SAFETY: see `build_xyzrhw_row`. Slice length matches the
        // `rows_usize` reservation.
        let dst: &mut [core::mem::MaybeUninit<[f32; 4]>] =
            unsafe { core::slice::from_raw_parts_mut(dst_ptr, rows_usize) };
        dst[0].write(colorvalue_to_rgba(&self.material.diffuse));
        if rows_usize > 1 {
            dst[1].write(colorvalue_to_rgba(&self.material.ambient));
            dst[2].write(colorvalue_to_rgba(&self.material.specular));
            dst[3].write(colorvalue_to_rgba(&self.material.emissive));
        }
        if rows_usize > 4 {
            dst[4].write([self.material.power, 0.0, 0.0, 0.0]);
        }
        (10, rows, dst_ptr.cast::<u8>())
    }

    /// Bump-copy the row 15..62 LIGHTS section, one 6-row block per compacted active light.
    ///
    /// See `resolve_active_lights`. The active mask is always the low
    /// `n` bits set, so every covered slot is occupied — there are no gaps to
    /// zero-fill. Light positions/directions are packed in EYE space
    /// (transformed by the VIEW matrix). Returns `None` when
    /// `vs_key.light_active_mask == 0` (nothing to upload — the shader
    /// won't read these rows either).
    ///
    /// # Panics
    ///
    /// Panics if `light_active_mask.leading_zeros()` exceeds `u8::MAX`
    /// — unreachable (the mask is a `u8` so the count is ≤ 8).
    pub fn build_lights_section(
        &self,
        vs_key: &FfVsKey,
        scratch: &mut ScratchArena,
    ) -> Option<(u16, u16, *mut u8)> {
        let active = vs_key.light_active_mask;
        if active == 0 {
            return None;
        }
        // The compacted active-light list IS the shader-slot order; its length
        // must agree with `vs_key.light_active_mask` (both derive from
        // `resolve_active_lights`). The mask is always the low `n` bits set, so
        // its high-bit index = the number of slots minus one.
        let compacted = self.resolve_active_lights();
        // High bit index (0..=7) of the active mask.
        let lz = u8::try_from(active.leading_zeros()).expect("u8::leading_zeros ≤ 8 fits u8");
        let hi = 7usize - lz as usize;
        let slots = hi + 1;
        debug_assert_eq!(
            slots, compacted.len,
            "LIGHTS slot count must match the compacted active-light list"
        );
        let rows_usize = slots * 6;
        let rows = u16::try_from(rows_usize).expect("LIGHTS rows ≤ 48 fits u16");
        let dst_ptr = scratch.alloc_uninit_slice::<core::mem::MaybeUninit<[f32; 4]>>(rows_usize);
        // SAFETY: see `build_xyzrhw_row`.
        let dst: &mut [core::mem::MaybeUninit<[f32; 4]>] =
            unsafe { core::slice::from_raw_parts_mut(dst_ptr, rows_usize) };
        for (i, active_light) in compacted.as_slice().iter().enumerate() {
            let base = i * 6;
            let light = &active_light.light;
            let is_spot = active_light.ty == D3DLIGHT_SPOT;
            let ty = if active_light.ty == D3DLIGHT_DIRECTIONAL {
                3.0
            } else if is_spot {
                2.0
            } else {
                1.0
            };
            // The FF VS computes `posEye = world * view` and reads the light
            // position/direction in EYE space (`toL = light_pos - posEye`), so
            // pack them transformed by the VIEW matrix. Matrices are D3D9
            // row-vector (`v * M`, `out[j] = Σ_k v[k]·M[k][j]`).
            let pos_eye = self.view_transform_point(&light.position);
            dst[base].write([pos_eye[0], pos_eye[1], pos_eye[2], ty]);
            // Direction is a w=0 vector — transform without the translation
            // row, then normalize so the shader's dot products see a unit
            // vector even when the app supplies an unnormalized one. A
            // zero-length direction (typical for point lights, where it is
            // unused) packs as zero.
            let d_eye = self.view_transform_dir(&light.direction);
            // Array-sum, not a polynomial, keeps the products unfused (a
            // `mul_add` would be an `fmaf` libcall on the no-FMA i686 baseline).
            let len = [
                d_eye[0] * d_eye[0],
                d_eye[1] * d_eye[1],
                d_eye[2] * d_eye[2],
            ]
            .iter()
            .sum::<f32>()
            .sqrt();
            let dir = if len > 0.0 {
                [d_eye[0] / len, d_eye[1] / len, d_eye[2] / len]
            } else {
                [0.0; 3]
            };
            // Spot cone, folded for a single shader mad: the D3D9 factor is
            // 1 inside theta, 0 outside phi, and
            // ((rho − cos(phi/2)) / (cos(theta/2) − cos(phi/2)))^falloff in
            // the penumbra — i.e. saturate(rho·scale + offset)^falloff with
            // scale = 1/(ct − cp), offset = −cp·scale. The falloff exponent
            // gets a tiny floor: fast-math pow(0, 0) is NaN, and a 1e-6
            // exponent keeps the falloff = 0 "constant cone" shape while
            // pinning the outside-cone result to 0.
            // Shared fast cos, not `f32::cos`: the C-libm `cosf` libcall
            // returns in x87 ST0 on i686 (same class of deopt as the fmaf
            // libcall above; few-ULP polynomial is plenty for a cone angle).
            let (spot_scale, spot_offset) = if is_spot {
                let ct = mtld3d_shared::trig::cos(light.theta * 0.5);
                let cp = mtld3d_shared::trig::cos(light.phi * 0.5);
                let scale = 1.0 / (ct - cp).max(1e-6);
                (scale, -cp * scale)
            } else {
                (0.0, 0.0)
            };
            dst[base + 1].write([dir[0], dir[1], dir[2], light.falloff.max(1e-6)]);
            dst[base + 2].write(colorvalue_to_rgba(&light.diffuse));
            // Light-color alpha lanes are dead in the lighting math (the
            // accumulated alpha is overwritten by the material diffuse
            // alpha), so the ambient and specular rows donate .w to the
            // spot params.
            let amb = colorvalue_to_rgba(&light.ambient);
            dst[base + 3].write([amb[0], amb[1], amb[2], spot_offset]);
            dst[base + 4].write([
                light.attenuation0,
                light.attenuation1,
                light.attenuation2,
                light.range,
            ]);
            let spec = colorvalue_to_rgba(&light.specular);
            dst[base + 5].write([spec[0], spec[1], spec[2], spot_scale]);
        }
        Some((15, rows, dst_ptr.cast::<u8>()))
    }

    /// Bump-copy the row 63..94 TT section (per-stage texture transform).
    ///
    /// Extent: 4 rows per stage up through the high bit of
    /// `tt_active_mask`; inactive stages in the covered range get a
    /// transposed identity transform (their TTFF flag is 0, so the
    /// shader won't read these rows either). Returns `None` when
    /// `tt_active_mask == 0`.
    ///
    /// # Panics
    ///
    /// Panics if `tt_active_mask.leading_zeros()` exceeds `u8::MAX` —
    /// unreachable (the mask is a `u8` so the count is ≤ 8).
    pub fn build_tt_section(&self, scratch: &mut ScratchArena) -> Option<(u16, u16, *mut u8)> {
        let mask = self.tt_active_mask;
        if mask == 0 {
            return None;
        }
        let lz = u8::try_from(mask.leading_zeros()).expect("u8::leading_zeros ≤ 8 fits u8");
        let hi = 7usize - lz as usize;
        let stages = hi + 1;
        let rows_usize = stages * 4;
        let rows = u16::try_from(rows_usize).expect("TT rows ≤ 32 fits u16");
        let dst_ptr = scratch.alloc_uninit_slice::<core::mem::MaybeUninit<[f32; 4]>>(rows_usize);
        // SAFETY: see `build_xyzrhw_row`.
        let dst: &mut [core::mem::MaybeUninit<[f32; 4]>] =
            unsafe { core::slice::from_raw_parts_mut(dst_ptr, rows_usize) };
        for (s, chunk) in dst.chunks_exact_mut(4).enumerate() {
            let m_t = Self::transpose(&self.texture_transforms[s]);
            write_matrix_rows(chunk, &m_t);
        }
        Some((63, rows, dst_ptr.cast::<u8>()))
    }

    /// Bump-copy the row 95+ PALETTE section (world-matrix palette × view).
    ///
    /// Extent: `world_palette_used() * 4` rows. Returns `None` when
    /// `vs_key.vertex_blend_count == 0` (palette is never read by the
    /// shader in that case).
    ///
    /// # Panics
    ///
    /// Panics if `world_palette_used()` exceeds `u16` capacity — unreachable
    /// (bounded by 256 per spec → `256 * 4 = 1024 < u16::MAX`).
    pub fn build_palette_section(
        &self,
        vs_key: &FfVsKey,
        scratch: &mut ScratchArena,
    ) -> Option<(u16, u16, *mut u8)> {
        if vs_key.vertex_blend_count == 0 {
            return None;
        }
        let used = self.world_palette_used();
        let rows_usize = used * 4;
        let rows = u16::try_from(rows_usize).expect("PALETTE rows ≤ 256*4 fits u16");
        let dst_ptr = scratch.alloc_uninit_slice::<core::mem::MaybeUninit<[f32; 4]>>(rows_usize);
        // SAFETY: see `build_xyzrhw_row`.
        let dst: &mut [core::mem::MaybeUninit<[f32; 4]>] =
            unsafe { core::slice::from_raw_parts_mut(dst_ptr, rows_usize) };
        for (bone, chunk) in dst.chunks_exact_mut(4).enumerate() {
            let bone_view_t =
                Self::transpose(&Self::mat_mul(&self.world_palette[bone], &self.view));
            write_matrix_rows(chunk, &bone_view_t);
        }
        Some((95, rows, dst_ptr.cast::<u8>()))
    }

    /// Pack FF PS constants: `ps_c[0]` = texture factor.
    ///
    /// Fog color lives in its own dedicated buffer (slot 13) so the FF and
    /// programmable PS paths share the same binding — see
    /// `build_fog_color_bytes`.
    #[must_use]
    pub fn build_ps_constants(&self, render_states: &[u32; RENDER_STATE_COUNT]) -> Vec<u8> {
        let tfactor = d3dcolor_to_rgba(render_states[D3DRS_TEXTUREFACTOR as usize]);
        let mut bytes = Vec::with_capacity(16);
        for v in tfactor {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes
    }
}

/// Serialize the fog render states for upload to the PS slot-13 `fog_data` buffer.
///
/// Row 0 = fog colour RGBA, row 1 = (start, end, density, depth-bias). Row 1
/// feeds the per-pixel table-fog factor; the depth-bias lane carries
/// `D3DRS_DEPTHBIAS` because real hardware fogs the post-bias fragment depth.
/// Returns empty when fog is off (neither vertex nor table mode set), so
/// `emit_draw` can skip the bind for variants without fog.
#[must_use]
pub fn build_fog_color_bytes(
    render_states: &[u32; RENDER_STATE_COUNT],
    variant: VariantKey,
) -> Vec<u8> {
    if variant.fog_mode == 0 && variant.fog_table_mode == 0 {
        return Vec::new();
    }
    let fog_color = d3dcolor_to_rgba(render_states[D3DRS_FOGCOLOR as usize]);
    let mut bytes = Vec::with_capacity(32);
    for v in fog_color {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    for rs in [
        D3DRS_FOGSTART,
        D3DRS_FOGEND,
        D3DRS_FOGDENSITY,
        D3DRS_DEPTHBIAS,
    ] {
        // Float render states store raw f32 bits in the u32 slot.
        bytes.extend_from_slice(&render_states[rs as usize].to_le_bytes());
    }
    bytes
}

/// Assemble the `FfVsFlags` for a `FfVsKey`.
///
/// Pure helper used by `build_vs_key`; lifts the flag-set sequence out so the
/// parent stays under the `too_many_lines` threshold while keeping the
/// predicate resolution co-located.
fn build_vs_flags(
    render_states: &[u32; RENDER_STATE_COUNT],
    layout: FfVsLayout,
    lighting_enabled: bool,
    vertex_blend_indexed: bool,
) -> FfVsFlags {
    let mut flags = FfVsFlags::empty();
    flags.set(FfVsFlags::HAS_NORMAL, layout.has_normal());
    flags.set(FfVsFlags::HAS_COLOR0, layout.has_color0());
    flags.set(
        FfVsFlags::DIFFUSE_DECLARED_UNBOUND,
        layout.has_diffuse_declared_unbound(),
    );
    flags.set(FfVsFlags::USES_VERTEX_DECL, layout.uses_vertex_decl());
    flags.set(FfVsFlags::HAS_COLOR1, layout.has_color1());
    flags.set(FfVsFlags::LIGHTING_ENABLED, lighting_enabled);
    // D3DRS_NORMALIZENORMALS only affects a lit draw with a normal — gate the
    // variant fork on those so unlit / no-normal draws don't multiply pipelines.
    flags.set(
        FfVsFlags::NORMALIZE_NORMALS,
        lighting_enabled
            && layout.has_normal()
            && render_states[D3DRS_NORMALIZENORMALS as usize] != 0,
    );
    flags.set(FfVsFlags::HAS_RHW, layout.has_rhw());
    flags.set(
        FfVsFlags::COLOR_VERTEX,
        render_states[D3DRS_COLORVERTEX as usize] != 0,
    );
    flags.set(
        FfVsFlags::SPECULAR_ENABLE,
        render_states[D3DRS_SPECULARENABLE as usize] != 0,
    );
    // Canonicalized: the emitter only computes V when lighting + specular
    // are both on, so the bit stays clear otherwise and toggling
    // D3DRS_LOCALVIEWER on a non-specular draw doesn't fork a variant.
    flags.set(
        FfVsFlags::LOCAL_VIEWER,
        lighting_enabled
            && render_states[D3DRS_SPECULARENABLE as usize] != 0
            && render_states[D3DRS_LOCALVIEWER as usize] != 0,
    );
    flags.set(FfVsFlags::VERTEX_BLEND_INDEXED, vertex_blend_indexed);
    flags.set(FfVsFlags::DECLARED_INDICES, layout.declared_indices());
    flags
}

/// Clamp a raw D3DRS_*MATERIALSOURCE value into the [0..2] range.
///
/// That is the range `resolve_mat` in the DXSO FF emitter understands. D3D9
/// only defines MATERIAL=0 / COLOR1=1 / COLOR2=2; anything outside that is a
/// garbage write and surfaces once.
fn clamp_material_source(value: u32, which: &str) -> u8 {
    if value > 2 {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "FF: D3DRS_{which} = {value} out of range (expected 0..2) → MCS_MATERIAL"
        );
        return 0;
    }
    u8::try_from(value).expect("checked above: value ≤ 2")
}

/// Resolved fog configuration for a draw.
///
/// Exactly one of the two fields is non-zero when fog is enabled.
#[derive(Clone, Copy)]
struct FogConfig {
    /// Vertex-fog mode.
    ///
    /// 1..3 = the D3DFOG_* factor the FF VS computes from eye-space Z, 4 =
    /// per-vertex factor from the specular (COLOR1) alpha. The PS blends the
    /// interpolated `in.fog.x`.
    vertex_mode: u8,
    /// Per-pixel table-fog mode (D3DFOG_* 1..3).
    ///
    /// The PS computes the factor from the rasterizer position; the vertex
    /// factor is unused.
    table_mode: u8,
}

fn resolve_fog_config(render_states: &[u32; RENDER_STATE_COUNT], has_rhw: bool) -> FogConfig {
    const OFF: FogConfig = FogConfig {
        vertex_mode: 0,
        table_mode: 0,
    };
    if render_states[D3DRS_FOGENABLE as usize] == 0 {
        return OFF;
    }
    // A non-NONE table mode wins over any vertex mode: D3D9 fogs per-pixel
    // ("pixel fog") and ignores FOGVERTEXMODE — on every vertex path,
    // including pre-transformed XYZRHW.
    match render_states[D3DRS_FOGTABLEMODE as usize] {
        0 => {}
        t @ 1..=3 => {
            return FogConfig {
                vertex_mode: 0,
                table_mode: u8::try_from(t).expect("matched 1..=3 fits u8"),
            };
        }
        other => {
            mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "FF fog: D3DRS_FOGTABLEMODE={other} unrecognised → skipped");
            return OFF;
        }
    }
    if has_rhw {
        // Pre-transformed (XYZRHW) vertices bypass the vertex-fog
        // computation: D3D9 takes the per-vertex fog factor from the
        // specular (COLOR1) alpha when table mode is NONE.
        return FogConfig {
            vertex_mode: 4,
            table_mode: 0,
        };
    }
    let vertex_mode = render_states[D3DRS_FOGVERTEXMODE as usize];
    let vertex_mode = match vertex_mode {
        // Vertex fog = D3DFOG_NONE with table fog also NONE: D3D9 takes the
        // per-vertex fog factor straight from the specular (COLOR1) alpha.
        0 => 4,
        v @ 1..=3 => u8::try_from(v).expect("matched 1..=3 fits u8"),
        other => {
            mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "FF fog: D3DRS_FOGVERTEXMODE={other} unrecognised → skipped");
            0
        }
    };
    FogConfig {
        vertex_mode,
        table_mode: 0,
    }
}

/// Resolve `D3DRS_VERTEXBLEND` and the decl-presence flags to a matrix count.
///
/// The count is the one consumed by `emit_vs`. Returns 0 (single-world-matrix
/// fallback) whenever the game asks for blending but the decl can't satisfy
/// it.
///
/// D3DVBF values: `DISABLE=0`, `1WEIGHTS=1`, `2WEIGHTS=2`, `3WEIGHTS=3`,
/// `TWEENING=255`, `0WEIGHTS=256`.
fn resolve_vertex_blend_count(mode: u32, layout: FfVsLayout, indexed: bool) -> u8 {
    use mtld3d_types::{
        D3DVBF_0WEIGHTS, D3DVBF_1WEIGHTS, D3DVBF_2WEIGHTS, D3DVBF_3WEIGHTS, D3DVBF_DISABLE,
        D3DVBF_TWEENING,
    };
    let count = match mode {
        D3DVBF_DISABLE => return 0,
        D3DVBF_1WEIGHTS => 2,
        D3DVBF_2WEIGHTS => 3,
        D3DVBF_3WEIGHTS => 4,
        // D3DVBF_0WEIGHTS: indexed-only single-matrix mode. Requires BLENDINDICES.
        D3DVBF_0WEIGHTS => 1,
        D3DVBF_TWEENING => {
            mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
                "FF vertex blend: D3DVBF_TWEENING (255) not implemented → falling back to single-world-matrix"
            );
            return 0;
        }
        other => {
            mtld3d_shared::log_once_warn_by!(
                target: crate::LOG_TARGET,
                key: u64::from(other),
                "FF vertex blend: D3DRS_VERTEXBLEND={other} unrecognised → falling back to single-world-matrix"
            );
            return 0;
        }
    };
    // Sequential mode (mode = 1..=3) needs at least one explicit weight in
    // the decl; the implicit last weight comes from `1 - sum(explicit)`.
    // Indexed mode for D3DVBF_0WEIGHTS doesn't need BLENDWEIGHT but does
    // need BLENDINDICES.
    if mode != 256 && layout.declared_weights_count == 0 {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "FF vertex blend: D3DRS_VERTEXBLEND non-zero but vertex decl has no BLENDWEIGHT element → falling back to single-world-matrix"
        );
        return 0;
    }
    if indexed && !layout.declared_indices() {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "FF vertex blend: D3DRS_INDEXEDVERTEXBLENDENABLE=TRUE but vertex decl has no BLENDINDICES element → falling back to single-world-matrix"
        );
        return 0;
    }
    if mode == 256 && !indexed {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "FF vertex blend: D3DVBF_0WEIGHTS requires D3DRS_INDEXEDVERTEXBLENDENABLE=TRUE → falling back to single-world-matrix"
        );
        return 0;
    }
    count
}

/// Plain-value snapshot of every FF field the state block needs to round-trip.
///
/// Used by `IDirect3DStateBlock9::Capture` / `Apply`; kept here so the
/// invariant "writing each field marks the right dirty bit" stays next to
/// `FfState` instead of drifting in a sibling module.
pub struct FfStateSnapshot {
    view: D3DMATRIX,
    projection: D3DMATRIX,
    world_palette: [D3DMATRIX; 256],
    world_palette_high_water: u16,
    texture_transforms: [D3DMATRIX; 8],
    material: D3DMATERIAL9,
    lights: [D3DLIGHT9; 8],
    light_enabled: u8,
    light_defined_mask: u8,
    /// The three setter-maintained light masks travel with the lights array.
    ///
    /// They cannot be re-derived from it after a restore, because an untouched
    /// slot's `D3DLIGHT9::default()` carries a directional type yet must not
    /// contribute (its `light_set_mask` bit is clear).
    light_set_mask: u8,
    light_directional_mask: u8,
    light_spot_mask: u8,
    texture_stage_states: [[u32; TEXTURE_STAGE_STATE_COUNT]; 8],
}

impl FfStateSnapshot {
    #[must_use]
    pub const fn from(state: &FfState) -> Self {
        Self {
            view: state.view,
            projection: state.projection,
            world_palette: state.world_palette,
            world_palette_high_water: state.world_palette_high_water,
            texture_transforms: state.texture_transforms,
            material: state.material,
            lights: state.lights,
            light_enabled: state.light_enabled,
            light_defined_mask: state.light_defined_mask,
            light_set_mask: state.light_set_mask,
            light_directional_mask: state.light_directional_mask,
            light_spot_mask: state.light_spot_mask,
            texture_stage_states: state.texture_stage_states,
        }
    }

    pub fn restore_into(&self, ff: &mut FfState) {
        ff.view = self.view;
        ff.projection = self.projection;
        ff.world_palette = self.world_palette;
        ff.world_palette_high_water = self.world_palette_high_water;
        ff.texture_transforms = self.texture_transforms;
        ff.material = self.material;
        ff.lights = self.lights;
        ff.light_enabled = self.light_enabled;
        ff.light_defined_mask = self.light_defined_mask;
        ff.light_set_mask = self.light_set_mask;
        ff.light_directional_mask = self.light_directional_mask;
        ff.light_spot_mask = self.light_spot_mask;
        ff.texture_stage_states = self.texture_stage_states;
        ff.recompute_tt_active_mask();
    }

    /// Restore only the fixed-function state a `block_type` state block owns.
    ///
    /// Leaves the rest of `ff` untouched. `All` matches [`Self::restore_into`]
    /// exactly. The split follows the D3D9 `D3DSBT_VERTEXSTATE` /
    /// `D3DSBT_PIXELSTATE` classification of which states each filtered
    /// block owns:
    ///
    /// - transforms (view / projection / world palette / texture transforms)
    ///   and material are `D3DSBT_ALL`-only — neither filtered block captures
    ///   them;
    /// - lights (and the enable / defined masks that travel with them) belong
    ///   to the vertex pipeline (`Vertex` and `All`);
    /// - texture-stage states are filtered per index via
    ///   [`StateBlockType::includes_tss`].
    ///
    /// As with [`Self::restore_into`], the captured light masks travel with the
    /// lights array (they are not derivable from it — see the field doc),
    /// while `tt_active_mask` re-derives from the restored stage states via
    /// `FfState::recompute_tt_active_mask`; both are normally maintained
    /// incrementally by setters, which a bulk array restore bypasses.
    pub fn restore_filtered(&self, ff: &mut FfState, block_type: StateBlockType) {
        // Transforms + material: D3DSBT_ALL only.
        if matches!(block_type, StateBlockType::All) {
            ff.view = self.view;
            ff.projection = self.projection;
            ff.world_palette = self.world_palette;
            ff.world_palette_high_water = self.world_palette_high_water;
            ff.texture_transforms = self.texture_transforms;
            ff.material = self.material;
        }
        // Lights (+ enable / defined / type masks): vertex pipeline →
        // Vertex | All.
        if !matches!(block_type, StateBlockType::Pixel) {
            ff.lights = self.lights;
            ff.light_enabled = self.light_enabled;
            ff.light_defined_mask = self.light_defined_mask;
            ff.light_set_mask = self.light_set_mask;
            ff.light_directional_mask = self.light_directional_mask;
            ff.light_spot_mask = self.light_spot_mask;
        }
        // Texture-stage states: whole-array for All, per-index otherwise. The
        // `0u32..` counter zipped with the per-stage array yields the `D3DTSS_*`
        // index as a `u32` without a fallible width conversion.
        if matches!(block_type, StateBlockType::All) {
            ff.texture_stage_states = self.texture_stage_states;
        } else {
            for (dst_stage, src_stage) in ff
                .texture_stage_states
                .iter_mut()
                .zip(&self.texture_stage_states)
            {
                for (ty, (dst, src)) in (0u32..).zip(dst_stage.iter_mut().zip(src_stage)) {
                    if block_type.includes_tss(ty) {
                        *dst = *src;
                    }
                }
            }
        }
        ff.recompute_tt_active_mask();
    }
}

fn d3dcolor_to_rgba(c: u32) -> [f32; 4] {
    crate::convert::d3dcolor_to_rgba_f32(c)
}

const fn colorvalue_to_rgba(c: &mtld3d_types::D3DCOLORVALUE) -> [f32; 4] {
    [c.r, c.g, c.b, c.a]
}

/// Write the four rows of a transposed `D3DMATRIX` into a destination slice.
///
/// Slice length must be exactly 4 (panics otherwise via slice indexing). Each
/// row is one `[f32; 4]` — the i-th column of the original matrix.
fn write_matrix_rows(dst: &mut [core::mem::MaybeUninit<[f32; 4]>], m_t: &D3DMATRIX) {
    for (i, slot) in dst.iter_mut().enumerate() {
        slot.write([
            m_t.m[i * 4],
            m_t.m[i * 4 + 1],
            m_t.m[i * 4 + 2],
            m_t.m[i * 4 + 3],
        ]);
    }
}

// ── Silent-write audit: D3DTSS_* classifier ──
// Per-(stage, ty) latch lives on `FfState.tss_warn_fired`. This table
// classifies each slot so the warn message is targeted.

enum TssClass {
    Consumed,
    NotImplemented,
}

const fn tss_classify(ty: u32) -> TssClass {
    match ty {
        // Consumed by mtld3d (FF VS + PS keys).
        D3DTSS_COLOROP
        | D3DTSS_COLORARG1
        | D3DTSS_COLORARG2
        | D3DTSS_ALPHAOP
        | D3DTSS_ALPHAARG1
        | D3DTSS_ALPHAARG2
        | D3DTSS_TEXCOORDINDEX
        | D3DTSS_TEXTURETRANSFORMFLAGS => TssClass::Consumed,

        // Neither consumes.
        _ => TssClass::NotImplemented,
    }
}

#[cfg(test)]
mod tests {
    use mtld3d_types::{
        D3DMATRIX, D3DRS_DEPTHBIAS, D3DRS_FOGCOLOR, D3DRS_FOGDENSITY, D3DRS_FOGENABLE,
        D3DRS_FOGEND, D3DRS_FOGSTART, D3DRS_FOGTABLEMODE, D3DRS_FOGVERTEXMODE, D3DRS_TEXTUREFACTOR,
        D3DTOP_MODULATE, D3DTSS_BUMPENVMAT00, D3DTSS_COLOROP, D3DTSS_TEXTURETRANSFORMFLAGS,
        RENDER_STATE_COUNT, render_state_defaults,
    };

    use super::{FfState, FfVsLayout, VariantFlags, VariantKey, build_fog_color_bytes};
    use crate::convert::FfVsLayoutFlags;

    fn rs() -> [u32; RENDER_STATE_COUNT] {
        render_state_defaults()
    }

    #[test]
    fn build_ps_constants_is_tfactor_only() {
        let mut states = rs();
        states[D3DRS_TEXTUREFACTOR as usize] = 0xFF80_4020;
        let bytes = FfState::new().build_ps_constants(&states);
        assert_eq!(bytes.len(), 16, "ps_c now only carries texture factor");
        // First float4 decodes back to texture factor (RGBA float).
        let r = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let g = f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let b = f32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        let a = f32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        // 0xFF80_4020 = ARGB(255,128,64,32).
        assert!((r - 128.0 / 255.0).abs() < 1e-4);
        assert!((g - 64.0 / 255.0).abs() < 1e-4);
        assert!((b - 32.0 / 255.0).abs() < 1e-4);
        assert!((a - 1.0).abs() < 1e-4);
    }

    #[test]
    fn fog_color_bytes_empty_when_fog_off() {
        let mut states = rs();
        states[D3DRS_FOGCOLOR as usize] = 0xFFFF_00FF;
        let variant = VariantKey::default();
        assert_eq!(variant.fog_mode, 0);
        assert!(build_fog_color_bytes(&states, variant).is_empty());
    }

    #[test]
    fn projection_is_ortho_treats_negative_zero_as_zero() {
        use mtld3d_types::{D3DMATRIX, D3DTS_PROJECTION};
        let mut ff = FfState::new();
        // Identity's 4th column is (0,0,0,1) → orthographic.
        assert!(ff.projection_is_ortho());
        // A negative-zero in the column must still count as zero.
        let mut proj = D3DMATRIX::IDENTITY;
        proj.m[3] = -0.0;
        proj.m[7] = -0.0;
        proj.m[11] = -0.0;
        ff.set_transform(D3DTS_PROJECTION, &proj);
        assert!(
            ff.projection_is_ortho(),
            "-0.0 in the projection's 4th column must count as 0"
        );
        // A genuine perspective column is not orthographic.
        proj.m[11] = 0.5;
        ff.set_transform(D3DTS_PROJECTION, &proj);
        assert!(!ff.projection_is_ortho());
    }

    #[test]
    fn table_fog_wins_over_vertex_mode_and_keys_source_on_projection() {
        let mut states = rs();
        states[D3DRS_FOGENABLE as usize] = 1;
        states[D3DRS_FOGVERTEXMODE as usize] = 1; // D3DFOG_EXP
        states[D3DRS_FOGTABLEMODE as usize] = 3; // D3DFOG_LINEAR

        // Identity projection (4th column (0,0,0,1)) = orthographic → Z source.
        let mut ff = FfState::new();
        let variant = ff.variant_key(&states, false);
        assert_eq!(variant.fog_mode, 0, "table fog must zero the vertex mode");
        assert_eq!(variant.fog_table_mode, 3);
        assert!(
            !variant.flags.contains(VariantFlags::FOG_SOURCE_W),
            "ortho projection → Z source"
        );

        // Perspective-marked projection (_44 != 1) → W source.
        let mut proj = D3DMATRIX::IDENTITY;
        proj.m[15] = 1.01;
        ff.set_transform(mtld3d_types::D3DTS_PROJECTION, &proj);
        let variant = ff.variant_key(&states, false);
        assert!(
            variant.flags.contains(VariantFlags::FOG_SOURCE_W),
            "non-ortho projection → W source"
        );

        // Table fog applies on the RHW path too.
        let variant = ff.variant_key(&states, true);
        assert_eq!(variant.fog_table_mode, 3);
        assert_eq!(variant.fog_mode, 0);

        // Vertex fog only: no table mode, no source bit churn from the
        // (still perspective) projection.
        states[D3DRS_FOGTABLEMODE as usize] = 0;
        let variant = ff.variant_key(&states, false);
        assert_eq!(variant.fog_mode, 1);
        assert_eq!(variant.fog_table_mode, 0);
        assert!(!variant.flags.contains(VariantFlags::FOG_SOURCE_W));
    }

    /// `restore_filtered` writes back only the FF state owned by the block type.
    ///
    /// Transforms + material are `All`-only, lights are vertex-pipeline, and
    /// texture-stage states split per index. `All` must match `restore_into`.
    #[test]
    fn restore_filtered_respects_block_type() {
        use mtld3d_types::{
            D3DLIGHT_DIRECTIONAL, D3DLIGHT9, D3DMATERIAL9, D3DMATRIX, D3DTOP_DISABLE,
            D3DTOP_MODULATE, D3DTS_VIEW, D3DTSS_TEXCOORDINDEX, StateBlockType,
        };

        use super::FfStateSnapshot;

        const COLOROP: usize = D3DTSS_COLOROP as usize;
        const TCI: usize = D3DTSS_TEXCOORDINDEX as usize;

        // One distinctive value per category.
        let mut src = FfState::new();
        src.set_light(
            0,
            &D3DLIGHT9 {
                type_: D3DLIGHT_DIRECTIONAL,
                range: 42.0,
                ..Default::default()
            },
        );
        src.set_light_enabled(0, true);
        let mut view = D3DMATRIX::IDENTITY;
        view.m[0] = 7.0;
        src.set_transform(D3DTS_VIEW, &view);
        src.set_material(&D3DMATERIAL9 {
            power: 13.0,
            ..Default::default()
        });
        src.set_texture_stage_state(0, COLOROP, D3DTOP_DISABLE); // pixel-only TSS
        src.set_texture_stage_state(0, TCI, 5); // vertex + pixel TSS
        let snap = FfStateSnapshot::from(&src);

        // VERTEXSTATE: lights + vertex TSS restored; transforms/material/pixel TSS untouched.
        let mut v = FfState::new();
        snap.restore_filtered(&mut v, StateBlockType::Vertex);
        assert_eq!(
            v.light(0).range.to_bits(),
            42.0_f32.to_bits(),
            "vertex restores lights"
        );
        assert!(v.light_enabled(0), "vertex restores light-enable");
        assert_eq!(
            v.transform(D3DTS_VIEW).unwrap().m[0].to_bits(),
            1.0_f32.to_bits(),
            "vertex leaves transforms at default"
        );
        assert_eq!(
            v.material().power.to_bits(),
            0.0_f32.to_bits(),
            "vertex leaves material"
        );
        assert_eq!(
            v.texture_stage_state(0, COLOROP),
            D3DTOP_MODULATE,
            "vertex leaves pixel-only TSS at stage-0 default"
        );
        assert_eq!(
            v.texture_stage_state(0, TCI),
            5,
            "vertex restores texcoord index"
        );

        // PIXELSTATE: pixel TSS restored; lights/transforms untouched.
        let mut p = FfState::new();
        snap.restore_filtered(&mut p, StateBlockType::Pixel);
        assert_eq!(
            p.light(0).range.to_bits(),
            0.0_f32.to_bits(),
            "pixel leaves lights"
        );
        assert!(!p.light_enabled(0), "pixel leaves light-enable");
        assert_eq!(
            p.texture_stage_state(0, COLOROP),
            D3DTOP_DISABLE,
            "pixel restores color op"
        );
        assert_eq!(
            p.texture_stage_state(0, TCI),
            5,
            "pixel restores texcoord index"
        );

        // ALL: everything restored, identical to restore_into.
        let mut a = FfState::new();
        snap.restore_filtered(&mut a, StateBlockType::All);
        let mut a_ref = FfState::new();
        snap.restore_into(&mut a_ref);
        assert_eq!(a.light(0).range.to_bits(), 42.0_f32.to_bits());
        assert_eq!(
            a.transform(D3DTS_VIEW).unwrap().m[0].to_bits(),
            7.0_f32.to_bits()
        );
        assert_eq!(a.material().power.to_bits(), 13.0_f32.to_bits());
        assert_eq!(a.texture_stage_state(0, COLOROP), D3DTOP_DISABLE);
        assert_eq!(
            a.texture_stage_state(0, TCI),
            a_ref.texture_stage_state(0, TCI),
            "restore_filtered(All) matches restore_into"
        );
    }

    /// `build_vs_key` must populate `tci_coord_indices` for every stage the VB layout declares.
    ///
    /// This holds for every stage the layout declares an attribute for, even
    /// when the FF PS color-blend chain terminates earlier via
    /// `D3DTSS_COLOROP == D3DTOP_DISABLE`.
    ///
    /// A programmable PS bound over FF VS commonly samples several textures
    /// while the captured FF state leaves stage 1+'s `COLOROP` at its default
    /// `DISABLE` (the game doesn't enable FF blending when a programmable PS
    /// is bound). Stopping TCI decode at the first `COLOROP_DISABLE` would
    /// leave `tci_coord_indices[1..]` at their `[0; 8]` init, routing every
    /// VS texcoord output onto `v4`; the PS would then sample every texture
    /// at `v4`'s coord set instead of the distinct sets each stage expects,
    /// collapsing the intended multi-texture result.
    #[test]
    fn tci_indices_preserved_past_colorop_disable_terminator() {
        let mut ff = FfState::new();
        // Make stage 0 default-enabled (COLOROP=MODULATE) but leave stages
        // 1+ at their default `COLOROP_DISABLE` — exactly the shape a
        // programmable-PS draw with FF VS produces. Without the fix the
        // loop breaks at stage 1 before reading its TEXCOORDINDEX.
        assert_eq!(
            ff.texture_stage_state(0, D3DTSS_COLOROP as usize),
            D3DTOP_MODULATE,
            "stage 0 default COLOROP must be MODULATE"
        );
        ff.set_texture_stage_state(1, D3DTSS_COLOROP as usize, 1 /* D3DTOP_DISABLE */);

        let layout = FfVsLayout {
            flags: FfVsLayoutFlags::HAS_COLOR0,
            tex_coord_count: 3,
            tex_coord_dims: [0; 8],
            declared_weights_count: 0,
        };
        // bound_texture_mask = stages 0/1/2 all have textures bound.
        let key = ff.build_vs_key(&rs(), layout, 0b0000_0111);

        // D3D9 spec default for `D3DTSS_TEXCOORDINDEX` is the stage index.
        // The fix preserves that for stages past the FF PS chain
        // terminator; the broken behaviour collapsed them all to 0.
        assert_eq!(
            &key.tci_coord_indices[..3],
            &[0u8, 1, 2],
            "tci_coord_indices[1..3] must stay populated; collapsing them to 0 \
             would route every FF VS texcoord output onto v4",
        );
        assert_eq!(
            key.tex_coord_count, 3,
            "VS still emits 3 texcoord outputs driven by VB layout"
        );
    }

    #[test]
    fn local_viewer_flag_canonicalizes_on_lighting_and_specular() {
        use mtld3d_types::{D3DRS_LIGHTING, D3DRS_LOCALVIEWER, D3DRS_SPECULARENABLE};
        let ff = FfState::new();
        let layout = FfVsLayout {
            flags: FfVsLayoutFlags::HAS_NORMAL,
            tex_coord_count: 0,
            tex_coord_dims: [0; 8],
            declared_weights_count: 0,
        };

        // RS defaults: LIGHTING=1, LOCALVIEWER=1, SPECULARENABLE=0 — the
        // bit stays clear while no specular term reads V.
        let mut states = rs();
        let key = ff.build_vs_key(&states, layout, 0);
        assert!(!key.local_viewer(), "no specular → no LOCAL_VIEWER bit");

        // Specular on + default LOCALVIEWER=1 → set.
        states[D3DRS_SPECULARENABLE as usize] = 1;
        let key = ff.build_vs_key(&states, layout, 0);
        assert!(key.local_viewer(), "specular + RS default → set");

        // Explicit LOCALVIEWER=0 → infinite viewer.
        states[D3DRS_LOCALVIEWER as usize] = 0;
        let key = ff.build_vs_key(&states, layout, 0);
        assert!(!key.local_viewer(), "RS off → infinite viewer");

        // Lighting off clears it even with specular + localviewer on.
        states[D3DRS_LOCALVIEWER as usize] = 1;
        states[D3DRS_LIGHTING as usize] = 0;
        let key = ff.build_vs_key(&states, layout, 0);
        assert!(!key.local_viewer(), "unlit → no LOCAL_VIEWER bit");
    }

    #[test]
    fn fog_color_bytes_two_rows_when_fog_on() {
        let mut states = rs();
        // D3DCOLOR 0xFF80_40C0 = ARGB(255, 128, 64, 192) → R=128/255, G=64/255, B=192/255, A=1.0
        states[D3DRS_FOGCOLOR as usize] = 0xFF80_40C0;
        states[D3DRS_FOGSTART as usize] = 0.5f32.to_bits();
        states[D3DRS_FOGEND as usize] = 10.0f32.to_bits();
        states[D3DRS_FOGDENSITY as usize] = 2.0f32.to_bits();
        states[D3DRS_DEPTHBIAS as usize] = 0.1f32.to_bits();
        let variant = VariantKey {
            fog_mode: 3,
            ..Default::default()
        };
        let bytes = build_fog_color_bytes(&states, variant);
        assert_eq!(bytes.len(), 32);
        let comp = |i: usize| f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
        let (r, g, b, a) = (comp(0), comp(1), comp(2), comp(3));
        assert!((r - 128.0 / 255.0).abs() < 1e-4, "r = {r}");
        assert!((g - 64.0 / 255.0).abs() < 1e-4, "g = {g}");
        assert!((b - 192.0 / 255.0).abs() < 1e-4, "b = {b}");
        assert!((a - 1.0).abs() < 1e-4, "a = {a}");
        // Row 1: (start, end, density, depth-bias), raw f32 bit copies.
        assert_eq!((comp(4), comp(5), comp(6), comp(7)), (0.5, 10.0, 2.0, 0.1));
    }

    #[test]
    fn tss_warn_latch_is_per_stage() {
        // BUMPENVMAT00 is NotImplemented (not in the Consumed list), so a
        // non-default write fires warn_tss_non_default_once. Default for
        // BUMPENVMAT00 is 0; write 1 to stages 0 and 1. A latch keyed only
        // on `ty` would set tss_warn_fired[ty] on stage 0 and silently
        // swallow stage 1 — the per-stage latch must fire for both.
        let mut state = FfState::new();
        state.set_texture_stage_state(0, D3DTSS_BUMPENVMAT00 as usize, 1);
        state.set_texture_stage_state(1, D3DTSS_BUMPENVMAT00 as usize, 1);
        assert!(state.tss_warn_fired(0, D3DTSS_BUMPENVMAT00 as usize));
        assert!(state.tss_warn_fired(1, D3DTSS_BUMPENVMAT00 as usize));
    }

    #[test]
    fn set_transform_world_matrix_index_routes_to_palette() {
        use mtld3d_types::{D3DMATRIX, D3DTS_WORLD};
        let mut state = FfState::new();
        // D3DTS_WORLD is palette[0] — must not bump high-water above 0.
        let m = D3DMATRIX::IDENTITY;
        assert!(state.set_transform(D3DTS_WORLD, &m));
        assert_eq!(state.world_palette_used(), 1);
        // D3DTS_WORLDMATRIX(5) = state 261 → palette[5].
        let mut m5 = D3DMATRIX::IDENTITY;
        m5.m[3] = 7.0; // distinguishable value in row 0 col 3
        assert!(state.set_transform(256 + 5, &m5));
        assert_eq!(state.world_palette_used(), 6, "high water 5 → used = 6");
        assert!((state.world_palette()[5].m[3] - 7.0).abs() < f32::EPSILON);
        // Slot 0 unchanged by the slot-5 write.
        assert!(state.world_palette()[0].m[3].abs() < f32::EPSILON);
    }

    #[test]
    fn resolve_vertex_blend_count_normal_mode() {
        let layout_with_weights = FfVsLayout {
            flags: FfVsLayoutFlags::empty(),
            tex_coord_count: 0,
            tex_coord_dims: [0; 8],
            declared_weights_count: 3,
        };
        // D3DVBF_1WEIGHTS → 2 matrices; sequential mode.
        assert_eq!(
            super::resolve_vertex_blend_count(1, layout_with_weights, false),
            2
        );
        // D3DVBF_3WEIGHTS → 4 matrices.
        assert_eq!(
            super::resolve_vertex_blend_count(3, layout_with_weights, false),
            4
        );
        // D3DVBF_DISABLE → 0.
        assert_eq!(
            super::resolve_vertex_blend_count(0, layout_with_weights, false),
            0
        );
        // Tweening unsupported → 0.
        assert_eq!(
            super::resolve_vertex_blend_count(255, layout_with_weights, false),
            0
        );
    }

    #[test]
    fn resolve_vertex_blend_count_indexed_only() {
        let layout_with_indices = FfVsLayout {
            flags: FfVsLayoutFlags::DECLARED_INDICES,
            tex_coord_count: 0,
            tex_coord_dims: [0; 8],
            declared_weights_count: 0,
        };
        // D3DVBF_0WEIGHTS + INDEXED → 1 matrix (single-bone indexed).
        assert_eq!(
            super::resolve_vertex_blend_count(256, layout_with_indices, true),
            1
        );
        // D3DVBF_0WEIGHTS without INDEXED → 0 (mode requires indices).
        assert_eq!(
            super::resolve_vertex_blend_count(256, layout_with_indices, false),
            0
        );
    }

    #[test]
    fn set_light_sets_active_and_directional_masks() {
        use mtld3d_types::{D3DLIGHT_DIRECTIONAL, D3DLIGHT_POINT, D3DLIGHT9};
        let mut state = FfState::new();
        assert_eq!(state.light_active_mask(), 0);
        assert_eq!(state.light_directional_mask(), 0);

        let dir = D3DLIGHT9 {
            type_: D3DLIGHT_DIRECTIONAL,
            ..D3DLIGHT9::default()
        };
        state.set_light(3, &dir);
        state.set_light_enabled(3, true);
        assert_eq!(state.light_active_mask(), 1 << 3);
        assert_eq!(state.light_directional_mask(), 1 << 3);

        let pt = D3DLIGHT9 {
            type_: D3DLIGHT_POINT,
            ..D3DLIGHT9::default()
        };
        state.set_light(5, &pt);
        state.set_light_enabled(5, true);
        assert_eq!(state.light_active_mask(), (1 << 3) | (1 << 5));
        assert_eq!(
            state.light_directional_mask(),
            1 << 3,
            "POINT must not set dir bit"
        );
    }

    #[test]
    fn set_light_with_type_zero_clears_set_bit() {
        use mtld3d_types::{D3DLIGHT_POINT, D3DLIGHT9};
        let mut state = FfState::new();
        let pt = D3DLIGHT9 {
            type_: D3DLIGHT_POINT,
            ..D3DLIGHT9::default()
        };
        state.set_light(2, &pt);
        state.set_light_enabled(2, true);
        assert_eq!(state.light_active_mask(), 1 << 2);

        // SetLight with Type=0 should drop the slot from the set mask, so
        // light_active_mask clears even though LightEnable(2, TRUE) remains.
        // (D3DLIGHT9::default() is DIRECTIONAL — construct Type=0 directly.)
        let zero = D3DLIGHT9 {
            type_: 0,
            ..D3DLIGHT9::default()
        };
        state.set_light(2, &zero);
        assert_eq!(state.light_active_mask(), 0);
    }

    #[test]
    fn light_enable_toggles_active_mask_when_set_bit_present() {
        use mtld3d_types::{D3DLIGHT_DIRECTIONAL, D3DLIGHT9};
        let mut state = FfState::new();
        let dir = D3DLIGHT9 {
            type_: D3DLIGHT_DIRECTIONAL,
            ..D3DLIGHT9::default()
        };
        state.set_light(0, &dir);
        // Set but not enabled → not active.
        assert_eq!(state.light_active_mask(), 0);
        state.set_light_enabled(0, true);
        assert_eq!(state.light_active_mask(), 1);
        state.set_light_enabled(0, false);
        assert_eq!(state.light_active_mask(), 0);
        assert_eq!(
            state.light_directional_mask(),
            1,
            "dir bit persists across enable toggles"
        );
    }

    #[test]
    fn set_light_maintains_spot_mask() {
        use mtld3d_types::{D3DLIGHT_POINT, D3DLIGHT_SPOT, D3DLIGHT9};
        let mut state = FfState::new();
        let spot = D3DLIGHT9 {
            type_: D3DLIGHT_SPOT,
            ..D3DLIGHT9::default()
        };
        state.set_light(1, &spot);
        state.set_light_enabled(1, true);
        assert_eq!(state.light_spot_mask(), 1 << 1);
        assert_eq!(
            state.light_directional_mask(),
            0,
            "SPOT must not set the dir bit"
        );
        assert_eq!(state.light_active_mask(), 1 << 1);

        // Re-typing the slot clears the spot bit.
        let pt = D3DLIGHT9 {
            type_: D3DLIGHT_POINT,
            ..D3DLIGHT9::default()
        };
        state.set_light(1, &pt);
        assert_eq!(state.light_spot_mask(), 0);
    }

    #[test]
    fn restore_recomputes_derived_masks() {
        use mtld3d_types::{D3DLIGHT_DIRECTIONAL, D3DLIGHT_SPOT, D3DLIGHT9};

        use super::FfStateSnapshot;
        // State-block Apply restores the light/TSS arrays wholesale,
        // bypassing the setters that maintain the masks — the captured
        // light masks must land with the lights array, and tt_active_mask
        // must re-derive from the restored stage states.
        let mut src = FfState::new();
        src.set_light(
            0,
            &D3DLIGHT9 {
                type_: D3DLIGHT_SPOT,
                ..D3DLIGHT9::default()
            },
        );
        src.set_light_enabled(0, true);
        src.set_texture_stage_state(2, D3DTSS_TEXTURETRANSFORMFLAGS as usize, 2);
        let snap = FfStateSnapshot::from(&src);

        let mut dst = FfState::new();
        dst.set_light(
            0,
            &D3DLIGHT9 {
                type_: D3DLIGHT_DIRECTIONAL,
                ..D3DLIGHT9::default()
            },
        );
        snap.restore_into(&mut dst);
        assert_eq!(dst.light_spot_mask(), 1, "spot bit from restored light");
        assert_eq!(
            dst.light_directional_mask(),
            0,
            "stale dir bit must clear on restore"
        );
        assert_eq!(
            dst.light_active_mask(),
            1,
            "set mask re-derived from restored lights"
        );
        assert_eq!(
            dst.tt_active_mask(),
            1 << 2,
            "tt mask re-derived from restored stage states"
        );
    }

    #[test]
    fn light_defined_tracks_set_and_enable() {
        use mtld3d_types::{D3DLIGHT_DIRECTIONAL, D3DLIGHT9};
        let mut state = FfState::new();
        assert!(!state.light_defined(0));
        assert!(!state.light_defined(4));

        // LightEnable defines a previously-undefined slot with the D3D9 default
        // directional light (white diffuse), so GetLight can report it.
        state.set_light_enabled(4, true);
        assert!(state.light_defined(4));
        assert_eq!(state.light(4).type_, D3DLIGHT_DIRECTIONAL);
        assert_eq!(state.light(4).diffuse.r.to_bits(), 1.0f32.to_bits());
        // Enable-only does not feed FF lighting (no explicit SetLight).
        assert_eq!(state.light_active_mask(), 0);

        // SetLight defines a slot regardless of light type.
        let zero = D3DLIGHT9 {
            type_: 0,
            ..D3DLIGHT9::default()
        };
        state.set_light(0, &zero);
        assert!(state.light_defined(0));
    }

    #[test]
    fn set_tt_flags_toggles_tt_active_mask() {
        let mut state = FfState::new();
        assert_eq!(state.tt_active_mask(), 0);

        state.set_texture_stage_state(2, D3DTSS_TEXTURETRANSFORMFLAGS as usize, 2);
        assert_eq!(state.tt_active_mask(), 1 << 2);

        state.set_texture_stage_state(5, D3DTSS_TEXTURETRANSFORMFLAGS as usize, 0x101);
        assert_eq!(state.tt_active_mask(), (1 << 2) | (1 << 5));

        state.set_texture_stage_state(2, D3DTSS_TEXTURETRANSFORMFLAGS as usize, 0);
        assert_eq!(state.tt_active_mask(), 1 << 5);
    }

    #[test]
    fn ff_state_new_clears_all_masks() {
        let state = FfState::new();
        assert_eq!(state.light_active_mask(), 0);
        assert_eq!(state.light_directional_mask(), 0);
        assert_eq!(state.tt_active_mask(), 0);
    }

    #[test]
    fn set_texture_stage_state_reports_value_change() {
        // The `changed` return gates snapshot dirty-marking: a same-value
        // write must report `false` so the redundant FF-key rebuild is
        // skipped; a real change must report `true`.
        let mut state = FfState::new();
        let ty = D3DTSS_COLOROP as usize;
        let initial = state.texture_stage_state(0, ty);

        assert!(
            !state.set_texture_stage_state(0, ty, initial),
            "re-writing the existing value reports unchanged"
        );
        assert!(
            state.set_texture_stage_state(0, ty, initial + 1),
            "writing a new value reports changed"
        );
        assert!(
            !state.set_texture_stage_state(0, ty, initial + 1),
            "re-writing the now-current value reports unchanged"
        );
    }

    // ─────────────────────────────────────────────────────────────────────
    // FF VS const-row extent tests
    //
    // `ff_vs_row_count` derives the per-draw upload extent from `FfVsKey`
    // gating + `FfState.tt_active_mask` + `world_palette_used`. Row
    // indices (fog 8, material 10..14, lights 15..62, TTFF 63..94,
    // palette 95+) are load-bearing — these tests pin the cascade.
    // ─────────────────────────────────────────────────────────────────────

    fn make_vs_key(flags: super::FfVsFlags, fog_mode: u8) -> super::FfVsKey {
        super::FfVsKey {
            flags,
            input_tex_coord_count: 0,
            tex_coord_count: 0,
            light_active_mask: 0,
            light_directional_mask: 0,
            light_spot_mask: 0,
            diffuse_source: 0,
            ambient_source: 0,
            specular_source: 0,
            emissive_source: 0,
            fog_mode,
            tci_modes: [0; 8],
            tci_coord_indices: [0; 8],
            tex_coord_dims: [0; 8],
            tt_flags: [0; 8],
            vertex_blend_count: 0,
            declared_weights_count: 0,
        }
    }

    #[test]
    fn ff_vs_row_count_xyzrhw_is_one_row() {
        let mut key = make_vs_key(super::FfVsFlags::HAS_RHW, 0);
        // Add some noise to make sure has_rhw short-circuits past it.
        key.light_active_mask = 0xFF;
        key.tt_flags = [0xFF; 8];
        assert_eq!(FfState::new().ff_vs_row_count(&key), 1);
    }

    #[test]
    fn ff_vs_row_count_unlit_no_fog_no_tt() {
        // Unlit reads only WV/Proj (rows 0..7) + diffuse fallback (row 10).
        let key = make_vs_key(super::FfVsFlags::empty(), 0);
        assert_eq!(FfState::new().ff_vs_row_count(&key), 11);
    }

    #[test]
    fn ff_vs_row_count_unlit_only_fog() {
        // Fog at row 8 < diffuse fallback row 10, so row 10 still wins.
        let key = make_vs_key(super::FfVsFlags::empty(), 3);
        assert_eq!(FfState::new().ff_vs_row_count(&key), 11);
    }

    #[test]
    fn ff_vs_row_count_lit_no_lights() {
        let key = make_vs_key(super::FfVsFlags::LIGHTING_ENABLED, 0);
        // Lit reads through material.emissive (row 13).
        assert_eq!(FfState::new().ff_vs_row_count(&key), 14);
    }

    #[test]
    fn ff_vs_row_count_lit_specular_no_lights() {
        let flags = super::FfVsFlags::LIGHTING_ENABLED | super::FfVsFlags::SPECULAR_ENABLE;
        let key = make_vs_key(flags, 0);
        // Material power lives at row 14.
        assert_eq!(FfState::new().ff_vs_row_count(&key), 15);
    }

    #[test]
    fn ff_vs_row_count_one_light() {
        let mut key = make_vs_key(super::FfVsFlags::LIGHTING_ENABLED, 0);
        key.light_active_mask = 1;
        key.light_directional_mask = 1;
        // Light 0 tail = row 15 + 0*6 + 5 = 20.
        assert_eq!(FfState::new().ff_vs_row_count(&key), 21);
    }

    #[test]
    fn ff_vs_row_count_one_light_with_fog() {
        // Fog folds into the lit block, adding no extra rows beyond what
        // light 0 already forced.
        let mut key = make_vs_key(super::FfVsFlags::LIGHTING_ENABLED, 3);
        key.light_active_mask = 1;
        key.light_directional_mask = 1;
        assert_eq!(FfState::new().ff_vs_row_count(&key), 21);
    }

    #[test]
    fn ff_vs_row_count_light_7() {
        let mut key = make_vs_key(super::FfVsFlags::LIGHTING_ENABLED, 0);
        key.light_active_mask = 1 << 7;
        // Light 7 tail = 15 + 7*6 + 5 = 62.
        assert_eq!(FfState::new().ff_vs_row_count(&key), 63);
    }

    #[test]
    fn ff_vs_row_count_tt_stage_4() {
        let mut state = FfState::new();
        state.set_texture_stage_state(4, D3DTSS_TEXTURETRANSFORMFLAGS as usize, 2);
        let key = make_vs_key(super::FfVsFlags::empty(), 0);
        // Stage 4 tail = 63 + 4*4 + 3 = 82.
        assert_eq!(state.ff_vs_row_count(&key), 83);
    }

    #[test]
    fn ff_vs_row_count_full_no_blend() {
        let mut state = FfState::new();
        for s in 0..8usize {
            state.set_texture_stage_state(s, D3DTSS_TEXTURETRANSFORMFLAGS as usize, 2);
        }
        let mut key = make_vs_key(
            super::FfVsFlags::LIGHTING_ENABLED | super::FfVsFlags::SPECULAR_ENABLE,
            3,
        );
        key.light_active_mask = 0xFF;
        key.light_directional_mask = 0xFF;
        // Worst case without blend: TTFF stage 7 tail = 63+7*4+3 = 94.
        assert_eq!(state.ff_vs_row_count(&key), 95);
    }

    #[test]
    fn ff_vs_row_count_blend_extends_palette() {
        use mtld3d_types::{D3DMATRIX, D3DTS_WORLD};
        let mut state = FfState::new();
        // Touch palette[4] via D3DTS_WORLDMATRIX(4) = 260 to push the
        // high-water mark to 4 → world_palette_used() = 5.
        state.set_transform(D3DTS_WORLD, &D3DMATRIX::IDENTITY);
        state.set_transform(256 + 4, &D3DMATRIX::IDENTITY);
        let mut key = make_vs_key(super::FfVsFlags::empty(), 0);
        key.vertex_blend_count = 2;
        // 95 + 5*4 = 115 rows.
        assert_eq!(state.ff_vs_row_count(&key), 115);
    }

    // ─────────────────────────────────────────────────────────────────────
    // Drift guard: regex over the emitted MSL must agree with our
    // inline `max_const_row` derivation. Catches any future edit that
    // reorders rows in the emitter without updating the derivation.
    // ─────────────────────────────────────────────────────────────────────

    fn emitter_high_water(key: &super::FfVsKey) -> u16 {
        use crate::dxso::emit_vs_ff;
        let msl = emit_vs_ff(key);
        let mut max: u16 = 0;
        let mut scanner = msl.as_str();
        while let Some(pos) = scanner.find("vs_c[") {
            let rest = &scanner[pos + "vs_c[".len()..];
            let end = rest.find(']').expect("vs_c[ without closing ]");
            let n: u16 = rest[..end].parse().expect("vs_c index must be u16");
            if n > max {
                max = n;
            }
            scanner = &rest[end + 1..];
        }
        max
    }

    fn derive_max_const_row(state: &FfState, key: &super::FfVsKey) -> u16 {
        state.ff_vs_row_count(key) - 1
    }

    #[test]
    fn max_const_row_matches_emitter_high_water_unlit() {
        let key = make_vs_key(super::FfVsFlags::empty(), 0);
        let state = FfState::new();
        // Emitter reads diffuse fallback row 10. Our derive matches.
        let emit = emitter_high_water(&key);
        let derive = derive_max_const_row(&state, &key);
        assert!(
            emit <= derive,
            "emitter reads vs_c[{emit}] but we only wrote rows 0..={derive}"
        );
    }

    #[test]
    fn max_const_row_matches_emitter_high_water_lit_no_lights() {
        let mut key = make_vs_key(super::FfVsFlags::LIGHTING_ENABLED, 0);
        key.flags.set(super::FfVsFlags::HAS_NORMAL, true);
        let state = FfState::new();
        let emit = emitter_high_water(&key);
        let derive = derive_max_const_row(&state, &key);
        assert!(
            emit <= derive,
            "emitter reads vs_c[{emit}] but we only wrote rows 0..={derive}"
        );
    }

    #[test]
    fn max_const_row_matches_emitter_high_water_lit_light0() {
        let mut key = make_vs_key(super::FfVsFlags::LIGHTING_ENABLED, 0);
        key.flags.set(super::FfVsFlags::HAS_NORMAL, true);
        key.light_active_mask = 1;
        key.light_directional_mask = 1;
        let state = FfState::new();
        let emit = emitter_high_water(&key);
        let derive = derive_max_const_row(&state, &key);
        assert!(
            emit <= derive,
            "emitter reads vs_c[{emit}] but we only wrote rows 0..={derive}"
        );
    }

    #[test]
    fn max_const_row_matches_emitter_high_water_lit_light7() {
        let mut key = make_vs_key(super::FfVsFlags::LIGHTING_ENABLED, 0);
        key.flags.set(super::FfVsFlags::HAS_NORMAL, true);
        key.light_active_mask = 1 << 7;
        let state = FfState::new();
        let emit = emitter_high_water(&key);
        let derive = derive_max_const_row(&state, &key);
        assert!(
            emit <= derive,
            "emitter reads vs_c[{emit}] but we only wrote rows 0..={derive}"
        );
    }

    #[test]
    fn max_const_row_matches_emitter_high_water_lit_fog_tt_stage_4() {
        let mut key = make_vs_key(super::FfVsFlags::LIGHTING_ENABLED, 3);
        key.flags.set(super::FfVsFlags::HAS_NORMAL, true);
        key.light_active_mask = 1;
        key.light_directional_mask = 1;
        // Emitter reads tt_flags[s] to gate the TTFF rows; mirror in key.
        key.tt_flags[4] = 2;
        key.tex_coord_count = 5;
        let mut state = FfState::new();
        state.set_texture_stage_state(4, D3DTSS_TEXTURETRANSFORMFLAGS as usize, 2);
        let emit = emitter_high_water(&key);
        let derive = derive_max_const_row(&state, &key);
        assert!(
            emit <= derive,
            "emitter reads vs_c[{emit}] but we only wrote rows 0..={derive}"
        );
    }

    #[test]
    fn resolve_vertex_blend_count_decl_mismatch_falls_back() {
        let layout_no_blend = FfVsLayout {
            flags: FfVsLayoutFlags::empty(),
            tex_coord_count: 0,
            tex_coord_dims: [0; 8],
            declared_weights_count: 0,
        };
        // Game asks for blending but decl has no BLENDWEIGHT → 0.
        assert_eq!(
            super::resolve_vertex_blend_count(1, layout_no_blend, false),
            0
        );
        // Game enables INDEXED but decl has no BLENDINDICES → 0.
        let layout_weights_only = FfVsLayout {
            declared_weights_count: 2,
            ..layout_no_blend
        };
        assert_eq!(
            super::resolve_vertex_blend_count(2, layout_weights_only, true),
            0
        );
    }

    // ─────────────────────────────────────────────────────────────────────
    // Sparse-light compaction + eye-space packing.
    // ─────────────────────────────────────────────────────────────────────

    /// Read `rows` packed `[f32; 4]` rows back out of a section pointer.
    ///
    /// Decodes the raw bytes, sidestepping any pointer-alignment cast.
    ///
    /// # Safety
    ///
    /// `ptr` must point at `rows` consecutive `[f32; 4]` values (16 bytes
    /// each), as returned by a `build_*_section` helper.
    unsafe fn read_section_rows(ptr: *mut u8, rows: usize) -> Vec<[f32; 4]> {
        let byte_len = rows * 16;
        // SAFETY: caller guarantees `rows * 16` initialized bytes at `ptr`.
        let bytes = unsafe { core::slice::from_raw_parts(ptr.cast_const(), byte_len) };
        bytes
            .chunks_exact(16)
            .map(|row| {
                let lane = |k: usize| {
                    f32::from_le_bytes(
                        row[k * 4..k * 4 + 4]
                            .try_into()
                            .expect("4-byte f32 lane slice"),
                    )
                };
                [lane(0), lane(1), lane(2), lane(3)]
            })
            .collect()
    }

    /// Assert two `[f32; 4]` rows match within a tight tolerance.
    ///
    /// The section data is exact, but float-array `assert_eq!` trips clippy's
    /// `float_cmp`.
    fn assert_row_eq(got: [f32; 4], want: [f32; 4], what: &str) {
        for (k, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!((g - w).abs() < 1e-6, "{what} lane {k}: got {g}, want {w}");
        }
    }

    /// Build the [`super::FfVsKey`] a lit, normal-carrying draw would produce.
    ///
    /// Reads the real `D3DRS_LIGHTING` default-on state so `build_vs_key`
    /// derives the compacted light masks.
    fn lit_vs_key(state: &FfState) -> super::FfVsKey {
        use mtld3d_types::{D3DRS_LIGHTING, RENDER_STATE_COUNT};
        let mut rs = [0u32; RENDER_STATE_COUNT];
        rs[D3DRS_LIGHTING as usize] = 1;
        let layout = FfVsLayout {
            flags: FfVsLayoutFlags::HAS_NORMAL,
            tex_coord_count: 0,
            tex_coord_dims: [0; 8],
            declared_weights_count: 0,
        };
        state.build_vs_key(&rs, layout, 0)
    }

    #[test]
    fn sparse_light_index_compacts_to_slot_zero() {
        use mtld3d_types::{D3DLIGHT_POINT, D3DLIGHT9};
        // Sparse light addressing: a single light at index 123.
        let mut state = FfState::new();
        let light = D3DLIGHT9 {
            type_: D3DLIGHT_POINT,
            ..D3DLIGHT9::default()
        };
        state.set_light_at(123, &light);
        state.set_light_enabled_at(123, true);

        // The physical fast-path mask is empty — the light lives in overflow.
        assert_eq!(
            state.light_active_mask(),
            0,
            "overflow light not in fast mask"
        );

        let active = state.resolve_active_lights();
        assert_eq!(active.len, 1, "one enabled overflow light compacts");
        assert_eq!(active.as_slice()[0].ty, D3DLIGHT_POINT);

        // And the derived key mask is the single low bit.
        let key = lit_vs_key(&state);
        assert_eq!(
            key.light_active_mask, 0b1,
            "compacted active mask must be slot 0 set"
        );
    }

    #[test]
    fn build_vs_key_compacts_sparse_lights_in_index_order() {
        use mtld3d_types::{D3DLIGHT_DIRECTIONAL, D3DLIGHT_POINT, D3DLIGHT_SPOT, D3DLIGHT9};
        // Lights at fast-path 5 (POINT) and overflow 100 (SPOT), 200 (DIR).
        let mut state = FfState::new();
        state.set_light(
            5,
            &D3DLIGHT9 {
                type_: D3DLIGHT_POINT,
                ..D3DLIGHT9::default()
            },
        );
        state.set_light_enabled(5, true);
        state.set_light_at(
            100,
            &D3DLIGHT9 {
                type_: D3DLIGHT_SPOT,
                ..D3DLIGHT9::default()
            },
        );
        state.set_light_enabled_at(100, true);
        state.set_light_at(
            200,
            &D3DLIGHT9 {
                type_: D3DLIGHT_DIRECTIONAL,
                ..D3DLIGHT9::default()
            },
        );
        state.set_light_enabled_at(200, true);

        let key = lit_vs_key(&state);
        // Three compacted slots → low three bits.
        assert_eq!(key.light_active_mask, 0b111);
        // Slot 0 = index 5 = POINT (neither type bit), slot 1 = index 100 =
        // SPOT, slot 2 = index 200 = DIRECTIONAL.
        assert_eq!(key.light_spot_mask, 0b010, "spot at compacted slot 1");
        assert_eq!(
            key.light_directional_mask, 0b100,
            "directional at compacted slot 2"
        );
    }

    #[test]
    fn eye_space_point_light_position_matches_hand_calc() {
        use mtld3d_types::{D3DLIGHT_POINT, D3DLIGHT9, D3DMATRIX, D3DTS_VIEW, D3DVECTOR};

        use crate::scratch::ScratchArena;
        // VIEW = pure translation {0.5, 0.5, 0}.
        let view = D3DMATRIX {
            m: [
                1.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0, //
                0.5, 0.5, 0.0, 1.0, //
            ],
        };
        let mut state = FfState::new();
        state.set_transform(D3DTS_VIEW, &view);
        state.set_light(
            0,
            &D3DLIGHT9 {
                type_: D3DLIGHT_POINT,
                position: D3DVECTOR {
                    x: 1.0,
                    y: 2.0,
                    z: 3.0,
                },
                ..D3DLIGHT9::default()
            },
        );
        state.set_light_enabled(0, true);

        let key = lit_vs_key(&state);
        let mut scratch = ScratchArena::new();
        let (start, rows, ptr) = state
            .build_lights_section(&key, &mut scratch)
            .expect("one active light → section present");
        assert_eq!(start, 15);
        assert_eq!(rows, 6, "one light = 6 rows");
        // SAFETY: build_lights_section wrote `rows` [f32;4] rows at `ptr`.
        let data = unsafe { read_section_rows(ptr, rows as usize) };
        // Row 0 = eye-space position + type-w. Hand calc for `v * view`:
        //   x' = 1*1 + 2*0 + 3*0 + 0.5 = 1.5
        //   y' = 1*0 + 2*1 + 3*0 + 0.5 = 2.5
        //   z' = 1*0 + 2*0 + 3*1 + 0   = 3.0
        // type-w = 1.0 (POINT).
        assert_row_eq(data[0], [1.5, 2.5, 3.0, 1.0], "eye-space POINT position");
    }

    #[test]
    fn contiguous_index0_light_packing_is_byte_identical() {
        // Guards the common WoW / e2e path: a single light at index 0 with
        // identity VIEW (eye == world) must pack to the exact rows hard-coded
        // below, so a future refactor of the compaction / eye-space path
        // can't silently drift them.
        use mtld3d_types::{D3DCOLORVALUE, D3DLIGHT_POINT, D3DLIGHT9, D3DVECTOR};

        use crate::scratch::ScratchArena;
        let mut state = FfState::new();
        state.set_light(
            0,
            &D3DLIGHT9 {
                type_: D3DLIGHT_POINT,
                diffuse: D3DCOLORVALUE {
                    r: 0.25,
                    g: 0.5,
                    b: 0.75,
                    a: 1.0,
                },
                position: D3DVECTOR {
                    x: 4.0,
                    y: 5.0,
                    z: 6.0,
                },
                attenuation0: 1.0,
                attenuation1: 0.1,
                attenuation2: 0.01,
                range: 100.0,
                ..D3DLIGHT9::default()
            },
        );
        state.set_light_enabled(0, true);

        let key = lit_vs_key(&state);
        // Identity VIEW ⇒ compacted slot 0 == physical slot 0, eye == world.
        assert_eq!(key.light_active_mask, 0b1);
        let mut scratch = ScratchArena::new();
        let (start, rows, ptr) = state.build_lights_section(&key, &mut scratch).unwrap();
        assert_eq!((start, rows), (15, 6));
        // SAFETY: 6 [f32;4] rows written at ptr.
        let data = unsafe { read_section_rows(ptr, 6) };
        // Position row: world == eye under identity view; POINT type-w = 1.
        assert_row_eq(data[0], [4.0, 5.0, 6.0, 1.0], "position row");
        // Diffuse color row (row base+2).
        assert_row_eq(data[2], [0.25, 0.5, 0.75, 1.0], "diffuse row");
        // Attenuation row (row base+4): a0, a1, a2, range.
        assert_row_eq(data[4], [1.0, 0.1, 0.01, 100.0], "attenuation row");
    }

    #[test]
    fn view_change_marks_lights_dirty() {
        use mtld3d_types::{D3DMATRIX, D3DTS_VIEW};
        let mut state = FfState::new();
        // Clear the cold-start all-dirty so we observe the SetTransform mark.
        let _ = state.take_ff_vs_dirty();
        state.set_transform(D3DTS_VIEW, &D3DMATRIX::IDENTITY);
        let dirty = state.take_ff_vs_dirty();
        assert!(
            dirty.contains(super::FfVsDirty::LIGHTS),
            "a VIEW change must invalidate the eye-space LIGHTS section"
        );
    }
}
