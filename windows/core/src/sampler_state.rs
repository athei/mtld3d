//! Single source of truth for D3D9 → Metal sampler-state translation.
//!
//! Mirrors `pipeline_state` but for samplers: one `SamplerSnapshot` input
//! drives both the cache `SamplerKey` and the native
//! `SamplerDescription`. Per-field unit tests assert the
//! static invariant that mutating any snapshot field produces a
//! different key, so the pipeline-style silent-drop bug (state classified
//! Consumed but value never reaches the sampler) is unrepresentable.
//!
//! Translation adds no implicit promotes. Promoting
//! `MIPFILTER NONE → LINEAR` or `MINFILTER LINEAR → ANISOTROPIC` would layer
//! aniso onto box-filter-generated mip chains on textures the game intended
//! to be sampled bilinearly, producing distance shimmer that D3D9 does not.
//! The filters D3D9 names beyond what a sampler offers (the quad filters,
//! `D3DTEXF_CONVOLUTIONMONO`) read as LINEAR, and a min or mag filter of
//! NONE as POINT.

use std::fmt;

use mtld3d_shared::mtl::{AddressMode, BorderColor, MinMagFilter, MipFilter};
use mtld3d_types::{
    D3DSAMP_ADDRESSU, D3DSAMP_ADDRESSV, D3DSAMP_ADDRESSW, D3DSAMP_BORDERCOLOR, D3DSAMP_DMAPOFFSET,
    D3DSAMP_ELEMENTINDEX, D3DSAMP_MAGFILTER, D3DSAMP_MAXANISOTROPY, D3DSAMP_MAXMIPLEVEL,
    D3DSAMP_MINFILTER, D3DSAMP_MIPFILTER, D3DSAMP_MIPMAPLODBIAS, D3DSAMP_SRGBTEXTURE,
    D3DTADDRESS_MIRRORONCE, D3DTADDRESS_WRAP, D3DTEXF_ANISOTROPIC, D3DTEXF_LINEAR, D3DTEXF_NONE,
    D3DTEXF_POINT, SAMPLER_STATE_COUNT, sampler_state_defaults,
};

use crate::{
    caps::MAX_ANISOTROPY,
    convert::{
        border_color_preset, d3d_border_color_to_metal, d3d_to_metal_address_mode,
        d3d_to_metal_min_mag_filter, d3d_to_metal_mip_filter,
    },
};

/// Upper LOD clamp passed to every `MTLSamplerDescriptor`.
///
/// The fixed `1000.0f` is the D3D9 convention for "no upper clamp", with
/// Metal naturally capping selection at the texture's actual mip count.
const LOD_MAX_CLAMP: f32 = 1000.0;

/// Fragment sampler slots the LOD uniform carries, one `float4` row each.
///
/// Matches the pixel-shader sampler slot count (`s0`..`s15`). The d3d9 side
/// static-asserts its own stage count against this so the two cannot drift.
pub const LOD_BIAS_SLOTS: usize = 16;

/// Byte length of the fragment LOD uniform.
pub const LOD_BIAS_BYTES: usize = LOD_BIAS_SLOTS * 16;

/// Index of the sampler-state array that carries the bound texture's `SetLOD`.
///
/// D3D9 numbers sampler states from `D3DSAMP_ADDRESSU` = 1, so index 0 names
/// no state the sampler reads. The draw snapshot's per-stage copy stores the
/// bound texture's LOD there, which is how the texture's most detailed level
/// reaches the sampler translation and the explicit-LOD rows without widening
/// the per-draw stage record. A copy that has no texture LOD to give carries 0.
pub const TEXTURE_LOD_SLOT: usize = 0;

/// The explicit-LOD row of a stage that needs no adjustment.
///
/// `max(lod + 0, -f32::MAX)` is `lod`, so a sample through this row lands
/// where its shader asked. See [`explicit_lod_row`].
pub const EXPLICIT_LOD_OPEN: [f32; 2] = [0.0, -f32::MAX];

/// A whole table of [`EXPLICIT_LOD_OPEN`] rows, for a draw no stage of which needs one.
pub const EXPLICIT_LOD_OPEN_ROWS: [[f32; 2]; LOD_BIAS_SLOTS] = [EXPLICIT_LOD_OPEN; LOD_BIAS_SLOTS];

/// Vertex sampler slots the vertex LOD uniform carries, one `float2` row each.
///
/// The four `D3DVERTEXTEXTURESAMPLER0..3` slots a `vs_3_0` samples as
/// `s0`..`s3`.
pub const VS_LOD_SLOTS: usize = crate::passes::VERTEX_SAMPLER_SLOTS;

/// Byte length of the vertex LOD uniform.
pub const VS_LOD_BYTES: usize = VS_LOD_SLOTS * 8;

/// Fine-mip clamp `D3DSAMP_MAXMIPLEVEL` is limited to on decode.
///
/// D3D9 leaves the state a full DWORD, but the largest surface the API allows
/// is 16384 wide, so no texture has a level above this and a wider value
/// selects the same smallest mip that this one does. Limiting it here keeps
/// the key's 5-bit field and the sampler's `lodMinClamp` reading one value.
const MAX_MIP_LEVEL: u8 = 15;

/// Magnitude `D3DSAMP_MIPMAPLODBIAS` is clamped to on decode.
///
/// D3D9 leaves the accepted range to the driver. The largest surface the API
/// allows is 16384 wide (15 mip levels), so a bias past ±32 cannot select a
/// level that exists, and clamping keeps an infinity out of the sample site's
/// `bias()` argument.
const LOD_BIAS_LIMIT: f32 = 32.0;

/// The D3D9 sampler enum bounds at the byte width the snapshot carries.
///
/// Narrow copies of the ABI constants, each pinned to its `mtld3d-types`
/// definition by the asserts below, so [`filter_value`] and [`address_value`]
/// can name a bound in `u8` without a truncating cast.
const TEXF_NONE: u8 = 0;
const TEXF_POINT: u8 = 1;
const TEXF_LINEAR: u8 = 2;
const TADDRESS_FIRST: u8 = 1;
const TADDRESS_LAST: u8 = 5;

const _: () = assert!(TEXF_NONE as u32 == D3DTEXF_NONE);
const _: () = assert!(TEXF_POINT as u32 == D3DTEXF_POINT);
const _: () = assert!(TEXF_LINEAR as u32 == D3DTEXF_LINEAR);
const _: () = assert!(TADDRESS_FIRST as u32 == D3DTADDRESS_WRAP);
const _: () = assert!(TADDRESS_LAST as u32 == D3DTADDRESS_MIRRORONCE);
const _: () = assert!(MAX_ANISOTROPY <= u8::MAX as u32);
const _: () = assert!(TEXTURE_LOD_SLOT < D3DSAMP_ADDRESSU as usize);

bitflags::bitflags! {
    /// Sampler cache-key booleans that aren't sourced from a D3DSAMP slot.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct SamplerFlags: u8 {
        /// Set when the sampler is bound to a depth-format texture (sampleable shadow map).
        ///
        /// Adds `compareFunction = LessEqual` to the Metal sampler so MSL
        /// `sample_compare` returns the D3D9 hardware-shadow PCF result
        /// (1 = lit, 0 = shadowed). Folded into the cache key so a single
        /// D3D9 sampler bound to both colour and depth slots in different
        /// draws cleanly fans out into two `MTLSamplerState`s.
        const IS_COMPARE = 1 << 0;
        /// `D3DSAMP_SRGBTEXTURE`.
        ///
        /// The game requests `sRGB → linear` decode on texture read so
        /// subsequent shading math runs in linear space. Doesn't affect the
        /// Metal sampler descriptor — sRGB is expressed as the *texture
        /// view*'s pixel format, not sampler state — but folds into the cache
        /// key so the bind-side resolution of which `MTLTexture` handle to
        /// bind (linear vs. sRGB view) stays in sync with the sampler's
        /// intent.
        const SRGB_TEXTURE = 1 << 1;
    }
}

/// How the silent-write audit treats a non-default `SetSamplerState` write.
///
/// The sampler bindings keep one warn latch per (sampler, type) and ask
/// [`samp_classify`] the first time a slot receives a value other than its
/// D3D9 default. Same three classes as `render_state::RsClass`.
pub enum SampClass {
    /// A consumer reads the slot, so the write is honoured and nothing is logged.
    Consumed,
    /// A no-op by design, logged once at info with the reason.
    ///
    /// The slot belongs to a feature Metal has no analog for or that is
    /// obsolete on every modern driver, so the no-op is the complete correct
    /// behaviour and not a port candidate.
    Obsolete(&'static str),
    /// Nothing reads the slot, so the write is lost and warned once.
    NotImplemented,
}

/// Classify a non-default write to sampler state `type_`.
///
/// A slot is `Consumed` only while the sampler translation or the draw-time
/// bind reads it; the comment on each group names that reader.
#[must_use]
pub const fn samp_classify(type_: u32) -> SampClass {
    match type_ {
        // Address, filter and anisotropy: `snapshot_from_state`, packed by
        // `key_from_snapshot` and built by `description_from_snapshot`.
        D3DSAMP_ADDRESSU
        | D3DSAMP_ADDRESSV
        | D3DSAMP_ADDRESSW
        | D3DSAMP_MAGFILTER
        | D3DSAMP_MINFILTER
        | D3DSAMP_MIPFILTER
        | D3DSAMP_MAXANISOTROPY
        // MAXMIPLEVEL: `snapshot_from_state`, plumbed to `setLodMinClamp`
        // on the unix side, and `explicit_lod_row` for the samples that
        // name their level.
        | D3DSAMP_MAXMIPLEVEL
        // SRGBTEXTURE is consumed at the draw-time bind: the stage's texture
        // handle resolves to the eager sRGB twin view so the hardware
        // decodes sRGB to linear at sample time. It also feeds the sampler
        // key (`SamplerFlags::SRGB_TEXTURE`) so distinct samplers stay
        // distinct.
        | D3DSAMP_SRGBTEXTURE
        // BORDERCOLOR: `description_from_snapshot` picks the nearest Metal border
        // preset (transparent black, opaque black, opaque white; other
        // colours fall back to opaque black with a once-per-colour warn from
        // `convert::d3d_border_color_to_metal`).
        | D3DSAMP_BORDERCOLOR
        // MIPMAPLODBIAS is consumed at draw time: Metal has no sampler-level
        // LOD bias, so `lod_bias` decodes the slot into the per-draw fragment
        // uniform and the pixel-shader emitters put `bias(...)` on every
        // implicit-LOD sample (`VariantFlags::LOD_BIAS`), or `fetch4` latches
        // the GET4/GET1 commands.
        | D3DSAMP_MIPMAPLODBIAS => SampClass::Consumed,

        // DMAPOFFSET addresses the displacement map of N-patch tessellation
        // (`D3DDMAPSAMPLER`, which `SetSamplerState` rejects), and
        // ELEMENTINDEX an element of a multi-element texture, which no
        // creation entry point can make.
        D3DSAMP_DMAPOFFSET => {
            SampClass::Obsolete("displacement-mapped N-patch tessellation is obsolete")
        }
        D3DSAMP_ELEMENTINDEX => SampClass::Obsolete("multi-element textures do not exist in D3D9"),

        // Nothing reads the slot.
        _ => SampClass::NotImplemented,
    }
}

/// Input view of the D3DSAMP state that participates in pipeline/cache decisions.
///
/// [`snapshot_from_state`] narrows each state once on the way in: the filters
/// to the point and linear filtering a sampler applies, the address modes to
/// their D3D9 value space, the numeric ones to the range the sampler accepts.
/// `key_from_snapshot` packs exactly the bytes `description_from_snapshot`
/// translates, so a state can never be keyed as one thing and built as
/// another, and the key's four-bit fields are exact without a mask.
pub struct SamplerSnapshot {
    /// `D3DSAMP_MINFILTER`, `D3DTEXF_POINT` or `D3DTEXF_LINEAR`.
    pub min_filter: u8,
    /// `D3DSAMP_MAGFILTER`, `D3DTEXF_POINT` or `D3DTEXF_LINEAR`.
    pub mag_filter: u8,
    /// `D3DSAMP_MIPFILTER`, `D3DTEXF_NONE`, `D3DTEXF_POINT` or `D3DTEXF_LINEAR`.
    pub mip_filter: u8,
    /// `D3DSAMP_ADDRESSU`, inside the `D3DTADDRESS_*` space.
    pub address_u: u8,
    /// `D3DSAMP_ADDRESSV`, inside the `D3DTADDRESS_*` space.
    pub address_v: u8,
    /// `D3DSAMP_ADDRESSW`, inside the `D3DTADDRESS_*` space.
    pub address_w: u8,
    /// `D3DSAMP_MAXANISOTROPY` as the sampler applies it.
    ///
    /// Limited to the ceiling the caps advertise, and 1 unless one of the
    /// stage's filters is `D3DTEXF_ANISOTROPIC`.
    pub max_anisotropy: u8,
    /// The most detailed level the stage may sample, at most the deepest level of a D3D9 texture.
    ///
    /// With mipmapping on it is `D3DSAMP_MAXMIPLEVEL` (D3D9's name for the
    /// *finest* level the sampler may select) or the texture's `SetLOD`,
    /// whichever is coarser, and maps to Metal's `setLodMinClamp`. With
    /// `D3DSAMP_MIPFILTER` NONE D3D9 ignores `MAXMIPLEVEL` and samples the
    /// texture's LOD level alone, so it is that LOD, and a non-zero value
    /// pins the sampler to the level from both sides. Zero = default.
    pub max_mip_level: u8,
    /// `D3DSAMP_BORDERCOLOR` as the game set it (a D3DCOLOR).
    ///
    /// Kept full-width: the D3DCOLOR is a bit pattern, not an enum, and both
    /// consumers reduce it through `border_color_preset`.
    pub border_color: u32,
    /// Cache-key booleans not sourced from a D3DSAMP slot value (`IS_COMPARE` / `SRGB_TEXTURE`).
    ///
    /// See [`SamplerFlags`].
    pub flags: SamplerFlags,
}

impl SamplerSnapshot {
    /// Replace the filter state with unfiltered point sampling.
    ///
    /// A raw depth fetch takes this: Apple GPUs cannot filter `Depth32Float`,
    /// and a linear sample of one returns garbage rather than depth. D3D9
    /// games set LINEAR on everything, so the slot's sampler is forced to
    /// point and the shader reads exact stored depths, which is what position
    /// reconstruction wants anyway. Comparison samplers are left as
    /// configured: linear there is hardware PCF, which Apple GPUs do support.
    ///
    /// Mipmapping goes off with the filters, so a stage with a non-zero
    /// [`Self::max_mip_level`] becomes a pinned sampler
    /// (`lodMinClamp = lodMaxClamp` = that level, see
    /// [`description_from_snapshot`]), and a raw-depth `texldl` at a coarser
    /// LOD is held at that level, while one with no finest level reads level 0
    /// whatever its LOD.
    pub const fn force_point_filter(&mut self) {
        self.min_filter = TEXF_POINT;
        self.mag_filter = TEXF_POINT;
        self.mip_filter = TEXF_NONE;
        self.max_anisotropy = 1;
    }
}

/// Turn a stage's filtering into point sampling, for a texture the device cannot filter.
///
/// The single-precision float formats filter only on a device with 32-bit
/// float filtering; elsewhere `CheckDeviceFormat(D3DUSAGE_QUERY_FILTER)`
/// answers no for them and Metal does not filter them. The min and mag
/// filters read POINT, and a mip filter other than NONE reads POINT too, which
/// keeps the level selection and blends no texels. The PE side applies it to
/// the stored state on its way to the draw, as it folds a texture's LOD.
pub const fn sample_unfiltered(ss: &mut [u32; SAMPLER_STATE_COUNT]) {
    ss[D3DSAMP_MINFILTER as usize] = D3DTEXF_POINT;
    ss[D3DSAMP_MAGFILTER as usize] = D3DTEXF_POINT;
    if ss[D3DSAMP_MIPFILTER as usize] != D3DTEXF_NONE {
        ss[D3DSAMP_MIPFILTER as usize] = D3DTEXF_POINT;
    }
}

/// Cached fragment LOD uniform derived from effective per-slot inputs.
///
/// Input identity uses raw `f32` bits so signed zero and NaN payload changes
/// are not hidden by float equality. This cache describes only the derived
/// bytes; the render encoder's last-bound cache independently decides whether
/// those bytes need binding in the current pass.
pub struct LodBiasTableCache {
    input_bits: Option<([u32; LOD_BIAS_SLOTS], [[u32; 2]; LOD_BIAS_SLOTS])>,
    bytes: [u8; LOD_BIAS_BYTES],
}

impl LodBiasTableCache {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            input_bits: None,
            bytes: [0; LOD_BIAS_BYTES],
        }
    }

    /// Rebuild the table when any effective input bit pattern changed.
    ///
    /// Returns whether the cached bytes were rebuilt.
    #[must_use]
    pub fn update(
        &mut self,
        biases: &[f32; LOD_BIAS_SLOTS],
        explicit: &[[f32; 2]; LOD_BIAS_SLOTS],
    ) -> bool {
        let input_bits = (
            biases.map(f32::to_bits),
            explicit.map(|row| row.map(f32::to_bits)),
        );
        if self.input_bits == Some(input_bits) {
            return false;
        }
        self.bytes = build_lod_bias_bytes(biases, explicit);
        self.input_bits = Some(input_bits);
        true
    }

    /// The table derived by the latest [`Self::update`] call.
    #[must_use]
    pub const fn bytes(&self) -> &[u8; LOD_BIAS_BYTES] {
        &self.bytes
    }
}

impl Default for LodBiasTableCache {
    fn default() -> Self {
        Self::new()
    }
}

/// The vertex samplers' explicit-LOD rows, kept as the vertex LOD uniform's bytes.
///
/// Every vertex sample names its level (`texldl`), and Metal applies no
/// sampler LOD clamp to an explicit level, so each slot's [`explicit_lod_row`]
/// reaches the vertex function through this uniform. The encoder updates a
/// slot when its sampler state arrives, which is far rarer than a draw, so a
/// draw reads [`Self::mask`] to decide whether its shader needs the table and
/// binds [`Self::bytes`] as they stand.
pub struct VertexLodTable {
    /// Bit `i` set when slot `i`'s row is not [`EXPLICIT_LOD_OPEN`].
    mask: u8,
    bytes: [u8; VS_LOD_BYTES],
}

impl VertexLodTable {
    #[must_use]
    pub fn new() -> Self {
        let mut table = Self {
            mask: 0,
            bytes: [0; VS_LOD_BYTES],
        };
        for slot in 0..VS_LOD_SLOTS {
            table.write_row(slot, EXPLICIT_LOD_OPEN);
        }
        table
    }

    /// Record vertex slot `slot`'s sampler state, the texture LOD included.
    ///
    /// Slots at or past [`VS_LOD_SLOTS`] are ignored (D3D9 defines four).
    pub fn set_slot(&mut self, slot: usize, ss: &[u32; SAMPLER_STATE_COUNT]) {
        if slot >= VS_LOD_SLOTS {
            return;
        }
        let bit = 1u8 << slot;
        let row = explicit_lod_row(ss);
        if row.is_some() {
            self.mask |= bit;
        } else {
            self.mask &= !bit;
        }
        self.write_row(slot, row.unwrap_or(EXPLICIT_LOD_OPEN));
    }

    /// Bit `i` set when vertex slot `i` needs its row.
    ///
    /// A sample at such a slot lands elsewhere than the level its shader names.
    #[must_use]
    pub const fn mask(&self) -> u8 {
        self.mask
    }

    /// The uniform's bytes: row `i` is `(offset, floor)` for vertex slot `i`.
    #[must_use]
    pub const fn bytes(&self) -> &[u8; VS_LOD_BYTES] {
        &self.bytes
    }

    fn write_row(&mut self, slot: usize, [offset, floor]: [f32; 2]) {
        let base = slot * 8;
        self.bytes[base..base + 4].copy_from_slice(&offset.to_le_bytes());
        self.bytes[base + 4..base + 8].copy_from_slice(&floor.to_le_bytes());
    }
}

impl Default for VertexLodTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Packed-bits sampler cache key.
///
/// Layout (u64 low-to-high):
/// - 0..3   `min_filter`
/// - 4..7   `mag_filter`
/// - 8..11  `mip_filter`
/// - 12..15 `address_u`
/// - 16..19 `address_v`
/// - 20..23 `address_w`
/// - 24..31 `max_anisotropy`
/// - 32..36 `max_mip_level` (5 bits — D3D9 mip count fits in 5)
/// - 37     `is_compare` (depth-bound shadow sampler)
/// - 38     `srgb_texture` (`D3DSAMP_SRGBTEXTURE` — picks linear vs sRGB texture-view at bind)
/// - 39..40 border preset (`D3DSAMP_BORDERCOLOR` reduced to the Metal preset, not the raw colour)
///
/// Every field is packed from the narrowed snapshot value, so each one fits
/// its width without a mask. A mask would make two states that differ only
/// above a field's width share this key while translating to different Metal
/// enums, and the second state to arrive would be served the first one's
/// sampler.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SamplerKey(u64);

impl fmt::LowerHex for SamplerKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::LowerHex::fmt(&self.0, f)
    }
}

impl SamplerKey {
    #[must_use]
    pub const fn raw(&self) -> u64 {
        self.0
    }
}

/// Whether a stage's `D3DSAMP_SRGBTEXTURE` value turns the sRGB decode on.
///
/// Only the LSB counts. Reference D3D9 drivers disagree on non-boolean
/// values (some read the low bit, some keep the previous state, some treat
/// any non-zero as on); the conformance suite pins the low-bit reading, so
/// values like 0x7e41882a, 100 and 2 sample RAW. The single predicate for
/// both the sampler-key flag and the draw-time texture-view pick, so the
/// two can't drift.
#[must_use]
pub const fn srgb_texture_enabled(ss: &[u32; SAMPLER_STATE_COUNT]) -> bool {
    ss[D3DSAMP_SRGBTEXTURE as usize] & 1 != 0
}

/// `D3DSAMP_MIPMAPLODBIAS` decoded as the float a sample site applies.
///
/// The state slot holds an IEEE `f32` bit pattern. NaN and Fetch4 commands
/// decode to no bias; the magnitude is clamped to ±32. Metal samplers carry no LOD
/// bias, so the value reaches the GPU as a shader uniform instead; this is the
/// single decoder for both that uniform and the "is any stage biased"
/// predicate, so the compiled variant and the value it reads cannot disagree.
#[must_use]
pub fn lod_bias(ss: &[u32; SAMPLER_STATE_COUNT]) -> f32 {
    if crate::fetch4::command(ss[D3DSAMP_MIPMAPLODBIAS as usize]).is_some() {
        return 0.0;
    }
    let raw = f32::from_bits(ss[D3DSAMP_MIPMAPLODBIAS as usize]);
    if raw.is_nan() {
        return 0.0;
    }
    raw.clamp(-LOD_BIAS_LIMIT, LOD_BIAS_LIMIT)
}

/// Whether a decoded bias actually shifts mip selection.
///
/// Any non-zero magnitude counts; both signed zeroes and the NaN
/// [`lod_bias`] already folded to zero read as no bias, so a draw that leaves
/// the state at its default keeps the unbiased shader variant.
#[must_use]
pub fn lod_bias_active(bias: f32) -> bool {
    bias.abs() > 0.0
}

/// Where a sample that names its own level lands, as `(offset, floor)`.
///
/// Metal ignores a sampler's LOD clamps for a sample with an explicit
/// `level()`, so `texldl` and the depth samples pinned to a level apply the
/// stage's state in the shader instead: `texldl` samples
/// `level(max(lod + offset, floor))`, and a depth sample with no level of its
/// own samples `level(max(floor, 0))`.
///
/// The texture's `SetLOD` is a base level, so the shader's LOD counts from it
/// and `offset` carries it, together with `D3DSAMP_MIPMAPLODBIAS`, which D3D9
/// adds to an explicit LOD as it does to a computed one. `floor` is the finest
/// level the stage may sample ([`SamplerSnapshot::max_mip_level`]), so
/// `D3DSAMP_MAXMIPLEVEL` clamps the result rather than shifting it. With
/// `D3DSAMP_MIPFILTER` NONE D3D9 samples the LOD level whatever the shader
/// asks: the offset is `-f32::MAX`, so the sum never wins over the floor.
///
/// `None` when the stage needs nothing, the [`EXPLICIT_LOD_OPEN`] row.
#[must_use]
pub fn explicit_lod_row(ss: &[u32; SAMPLER_STATE_COUNT]) -> Option<[f32; 2]> {
    let lod = clamped_max_mip_level(ss[TEXTURE_LOD_SLOT]);
    let floor = finest_level(ss);
    if ss[D3DSAMP_MIPFILTER as usize] == D3DTEXF_NONE {
        // Level 0 of a sampler with no mipmapping is the only level Metal
        // reads, which is what D3D9 reads at LOD 0.
        return (floor > 0).then(|| [-f32::MAX, f32::from(floor)]);
    }
    let offset = f32::from(lod) + lod_bias(ss);
    if floor == 0 && !lod_bias_active(offset) {
        return None;
    }
    let floor = if floor == 0 {
        EXPLICIT_LOD_OPEN[1]
    } else {
        f32::from(floor)
    };
    Some([offset, floor])
}

/// Serialise the per-slot LOD inputs into the fragment uniform's bytes.
///
/// Row `i` is `(bias, exp2(bias), offset, floor)`. An implicit-LOD sample
/// adds `.x` through MSL's `bias()`; an explicit-gradient sample multiplies
/// its derivatives by `.y` instead, which shifts the computed LOD by the same
/// amount, because MSL accepts only one LOD option per sample call. A sample
/// with an explicit level reads `.z` and `.w`, the [`explicit_lod_row`] of the
/// stage.
#[must_use]
pub fn build_lod_bias_bytes(
    biases: &[f32; LOD_BIAS_SLOTS],
    explicit: &[[f32; 2]; LOD_BIAS_SLOTS],
) -> [u8; LOD_BIAS_BYTES] {
    let mut out = [0u8; LOD_BIAS_BYTES];
    for (row, (&bias, &[offset, floor])) in biases.iter().zip(explicit).enumerate() {
        let base = row * 16;
        out[base..base + 4].copy_from_slice(&bias.to_le_bytes());
        out[base + 4..base + 8].copy_from_slice(&bias.exp2().to_le_bytes());
        out[base + 8..base + 12].copy_from_slice(&offset.to_le_bytes());
        out[base + 12..base + 16].copy_from_slice(&floor.to_le_bytes());
    }
    out
}

/// Build a `SamplerSnapshot` from the device's per-stage D3DSAMP array.
///
/// Every state is narrowed here and nowhere else, so the key and the wire
/// params read one value. `is_compare` is supplied separately by the caller
/// (encoder) from the per-stage depth-sampler mask — it isn't a D3DSAMP_*
/// slot.
#[must_use]
pub fn snapshot_from_state(ss: &[u32; SAMPLER_STATE_COUNT], is_compare: bool) -> SamplerSnapshot {
    let mut flags = SamplerFlags::empty();
    flags.set(SamplerFlags::IS_COMPARE, is_compare);
    flags.set(SamplerFlags::SRGB_TEXTURE, srgb_texture_enabled(ss));
    SamplerSnapshot {
        min_filter: filter_value(ss, D3DSAMP_MINFILTER, TEXF_POINT),
        mag_filter: filter_value(ss, D3DSAMP_MAGFILTER, TEXF_POINT),
        mip_filter: filter_value(ss, D3DSAMP_MIPFILTER, TEXF_NONE),
        address_u: address_value(ss, D3DSAMP_ADDRESSU),
        address_v: address_value(ss, D3DSAMP_ADDRESSV),
        address_w: address_value(ss, D3DSAMP_ADDRESSW),
        max_anisotropy: effective_max_anisotropy(ss),
        max_mip_level: finest_level(ss),
        border_color: ss[D3DSAMP_BORDERCOLOR as usize],
        flags,
    }
}

/// A filter state, narrowed to the filtering a sampler applies.
///
/// A sampler filters by point or linearly, and a mip filter may also turn
/// mipmapping off. `SetSamplerState` stores whatever DWORD the game passed,
/// so the value is clamped into `floor..=LINEAR`: a min or mag filter of
/// `D3DTEXF_NONE` (`floor` POINT) samples as POINT, and every value above
/// LINEAR filters linearly. That covers `D3DTEXF_ANISOTROPIC`, whose
/// anisotropy [`effective_max_anisotropy`] reads from the raw state, the two
/// quad filters and `D3DTEXF_CONVOLUTIONMONO`, which no sampler offers, and
/// any DWORD no `D3DTEXF_*` names. Each value read as another filter than it
/// names surfaces once.
fn filter_value(ss: &[u32; SAMPLER_STATE_COUNT], state: u32, floor: u8) -> u8 {
    let value = ss[state as usize];
    if value == D3DTEXF_ANISOTROPIC {
        return TEXF_LINEAR;
    }
    let read = if value < u32::from(floor) {
        floor
    } else if value > u32::from(TEXF_LINEAR) {
        TEXF_LINEAR
    } else {
        // Exact: the branches above leave POINT..=LINEAR or NONE.
        return value.to_le_bytes()[0];
    };
    mtld3d_shared::log_once_warn_by!(
        target: crate::LOG_TARGET,
        key: (u64::from(state) << 32) | u64::from(value),
        "D3DSAMP_{state} = {value:#x} has no sampler filter of its own → reading D3DTEXF {read}"
    );
    read
}

/// An address-mode state, narrowed to the byte a snapshot carries.
///
/// `SetSamplerState` stores whatever DWORD the game passed, so these are game
/// input. A value outside the `D3DTADDRESS_*` space reads as the D3D9 default
/// WRAP from `sampler_state_defaults`, which is what a driver settles on for
/// a mode it does not recognise, and surfaces once.
fn address_value(ss: &[u32; SAMPLER_STATE_COUNT], state: u32) -> u8 {
    let value = ss[state as usize];
    // Exact for every value the space accepts: it fits in a byte.
    let byte = value.to_le_bytes()[0];
    if u32::from(byte) == value && (TADDRESS_FIRST..=TADDRESS_LAST).contains(&byte) {
        return byte;
    }
    // Exact: no sampler-state default is wider than a byte.
    let default = sampler_state_defaults()[state as usize].to_le_bytes()[0];
    mtld3d_shared::log_once_warn_by!(
        target: crate::LOG_TARGET,
        key: u64::from(state),
        "D3DSAMP_{state} = {value:#x} outside its {TADDRESS_FIRST}..={TADDRESS_LAST} value space → reading the D3D9 default {default:#x}"
    );
    default
}

#[must_use]
pub const fn key_from_snapshot(s: &SamplerSnapshot) -> SamplerKey {
    SamplerKey(
        (s.min_filter as u64)
            | ((s.mag_filter as u64) << 4)
            | ((s.mip_filter as u64) << 8)
            | ((s.address_u as u64) << 12)
            | ((s.address_v as u64) << 16)
            | ((s.address_w as u64) << 20)
            | ((s.max_anisotropy as u64) << 24)
            | ((s.max_mip_level as u64) << 32)
            | ((s.flags.contains(SamplerFlags::IS_COMPARE) as u64) << 37)
            | ((s.flags.contains(SamplerFlags::SRGB_TEXTURE) as u64) << 38)
            | ((border_preset_for_key(s.border_color) as u64 & 0x3) << 39),
    )
}

/// The most detailed level a stage may sample, see [`SamplerSnapshot::max_mip_level`].
fn finest_level(ss: &[u32; SAMPLER_STATE_COUNT]) -> u8 {
    let lod = clamped_max_mip_level(ss[TEXTURE_LOD_SLOT]);
    if ss[D3DSAMP_MIPFILTER as usize] == D3DTEXF_NONE {
        lod
    } else {
        lod.max(clamped_max_mip_level(ss[D3DSAMP_MAXMIPLEVEL as usize]))
    }
}

/// A level-valued state at the width the key packs and the sampler takes.
///
/// `D3DSAMP_MAXMIPLEVEL` and the texture LOD. `SetSamplerState` stores
/// whatever DWORD the game passed, so the state is game input, and a value
/// past the deepest level any D3D9 texture has reads as [`MAX_MIP_LEVEL`].
const fn clamped_max_mip_level(level: u32) -> u8 {
    if level > MAX_MIP_LEVEL as u32 {
        MAX_MIP_LEVEL
    } else {
        // Exact: the branch above leaves nothing wider than a byte.
        level.to_le_bytes()[0]
    }
}

/// `D3DSAMP_MAXANISOTROPY` as the stage's filters apply it.
///
/// D3D9 filters anisotropically only through `D3DTEXF_ANISOTROPIC`, so a stage
/// whose min, mag and mip filters all name another filter samples
/// isotropically whatever `D3DSAMP_MAXANISOTROPY` holds, and reads 1 here.
/// Any of the three naming it turns anisotropy on, the min filter being the
/// one that decides a minified sample and the mag filter the one the caps
/// also advertise.
fn effective_max_anisotropy(ss: &[u32; SAMPLER_STATE_COUNT]) -> u8 {
    let anisotropic = [D3DSAMP_MINFILTER, D3DSAMP_MAGFILTER, D3DSAMP_MIPFILTER]
        .iter()
        .any(|&state| ss[state as usize] == D3DTEXF_ANISOTROPIC);
    if anisotropic {
        clamped_max_anisotropy(ss[D3DSAMP_MAXANISOTROPY as usize])
    } else {
        1
    }
}

/// `D3DSAMP_MAXANISOTROPY` at the width the key packs and the sampler takes.
///
/// The state is a DWORD the game chose, D3D9's default is 1 and the caps
/// advertise a ceiling of [`MAX_ANISOTROPY`], so the value is limited to that
/// range here. Two DWORDs that limit alike then share one sampler rather than
/// minting two keys for the one object the unix side builds.
const fn clamped_max_anisotropy(value: u32) -> u8 {
    if value == 0 {
        return 1;
    }
    if value > MAX_ANISOTROPY {
        // Exact: the const assert above pins MAX_ANISOTROPY inside a byte.
        return MAX_ANISOTROPY.to_le_bytes()[0];
    }
    value.to_le_bytes()[0]
}

/// The border preset the key carries.
///
/// Exact presets map to themselves, anything else to opaque black, matching
/// what `description_from_snapshot` (which also logs the substitution) hands to the
/// unix side.
const fn border_preset_for_key(color: u32) -> BorderColor {
    match border_color_preset(color) {
        Some(preset) => preset,
        None => BorderColor::OpaqueBlack,
    }
}

/// Resolved native sampler inputs, independent of device and output ownership.
pub struct SamplerDescription {
    pub id: u64,
    pub min_filter: MinMagFilter,
    pub mag_filter: MinMagFilter,
    pub mip_filter: MipFilter,
    pub address_u: AddressMode,
    pub address_v: AddressMode,
    pub address_w: AddressMode,
    pub max_anisotropy: u32,
    pub lod_min_clamp: f32,
    pub lod_max_clamp: f32,
    pub flags: SamplerFlags,
    pub border_color: BorderColor,
}

/// Translate a snapshot into native sampler inputs.
///
/// A stage with mipmapping off and a non-zero [`SamplerSnapshot::max_mip_level`]
/// samples that one level: the sampler selects the nearest level with both
/// LOD clamps on it, since a sampler without mipmapping reads level 0 alone.
#[must_use]
pub fn description_from_snapshot(s: &SamplerSnapshot, key: SamplerKey) -> SamplerDescription {
    let pinned = s.mip_filter == TEXF_NONE && s.max_mip_level > 0;
    let mip_filter = if pinned { TEXF_POINT } else { s.mip_filter };
    SamplerDescription {
        id: key.raw(),
        min_filter: d3d_to_metal_min_mag_filter(u32::from(s.min_filter)),
        mag_filter: d3d_to_metal_min_mag_filter(u32::from(s.mag_filter)),
        mip_filter: d3d_to_metal_mip_filter(u32::from(mip_filter)),
        address_u: d3d_to_metal_address_mode(u32::from(s.address_u)),
        address_v: d3d_to_metal_address_mode(u32::from(s.address_v)),
        address_w: d3d_to_metal_address_mode(u32::from(s.address_w)),
        max_anisotropy: u32::from(s.max_anisotropy),
        lod_min_clamp: f32::from(s.max_mip_level),
        lod_max_clamp: if pinned {
            f32::from(s.max_mip_level)
        } else {
            LOD_MAX_CLAMP
        },
        flags: s.flags & SamplerFlags::IS_COMPARE,
        border_color: d3d_border_color_to_metal(s.border_color),
    }
}

#[cfg(test)]
mod tests;
