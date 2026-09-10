//! On-disk shader and render-pipeline cache in `mtld3d_shaders.bin` next to the host EXE.
//!
//! Each successful MSL or render-pipeline compile appends one *Single* chunk.
//! The startup prewarm thread recreates the recorded objects and atomically
//! compacts the file into one *Bundle* chunk when needed.
//!
//! ## File layout (v18)
//!
//! ```text
//! [file header  16B]  MTLD3DSH | format u32 LE | shader/translation schema u32 LE
//! [chunk]*
//! ```
//!
//! Each chunk has a 24-byte plaintext header followed by a zstd frame:
//!
//! ```text
//! [1B kind | 3B _pad | 8B key u64 LE | 4B frame_len u32 LE | 8B xxh3 u64 LE]
//! [frame: frame_len bytes]
//! ```
//!
//! * `kind` ∈ `0..=7` (`CachedKind`) → **Single** chunk. Frame decompresses to
//!   one UTF-8 MSL string. `key` is the `disk_key`.
//! * `kind == RECORD_KIND_PIPELINE` (`0xFE`) → **Single** pipeline recipe.
//!   The recipe contains stable shader references and explicit logical fields,
//!   never runtime handles or Rust struct memory.
//! * `kind == RECORD_KIND_BUNDLE` (`0xFF`) → **Bundle** chunk. Frame
//!   decompresses to a concatenation of plain records
//!   (`[1B kind|3B _pad|8B key|4B body_len][body]`, repeated). No inner
//!   checksum: the whole bundle frame is covered by the outer chunk's xxh3.
//! * `xxh3` = `xxh3_64` over `chunk_header[0..16] ++ frame_bytes`. Computed
//!   once at write, verified on read. Mismatch stops parsing because the
//!   length may be corrupt. This mechanism already covers the
//!   frame body, so zstd's own per-frame checksum is left off as redundant.
//!
//! Robustness comes from the plaintext `frame_len` prefix plus the xxh3:
//! torn writes (frame runs past EOF) ⇒ `break`; xxh3 mismatch ⇒ `break`;
//! decompress failure or bad UTF-8 ⇒ skip; unknown `kind` ⇒ skip via
//! `frame_len` (forward-compat hook); a file header found where a chunk
//! header belongs ⇒ skip its 16 bytes.
//!
//! This module owns the binary format and the creation of the file that
//! holds it, both pure-Rust and host-testable. Encoder-side write hooks
//! and pre-warm-thread plumbing live in `windows/d3d9`.

use std::{
    fs::{self, File, OpenOptions},
    hash::{Hash, Hasher},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    thread,
    time::Duration,
};

use mtld3d_shared::{
    MetalHandle, VertexAttrDesc,
    mtl::{PixelFormat, VertexFormat, VertexStepFunction},
};
use mtld3d_types::MAX_STREAMS;
use rustc_hash::FxHashSet;
use xxhash_rust::xxh3::Xxh3;

use crate::{
    pipeline_state::{
        ExtraColorAttachments, PipelineAttachFlags, PipelineRsBits, PipelineRsFlags,
        PipelineSnapshot, StreamLayout,
    },
    shader_compile_stats::CompileBucket,
};

/// Bumped for shader emission, shader keys, or pipeline-translation semantics.
///
/// A cache file with a different schema is wiped and rebuilt from scratch.
/// Pipeline recipes share this schema because unchanged recipe bytes can
/// describe different Metal state after a translation-rule change.
///
/// `14` adds `VariantKey::depth_sampler_mask` to the PS variant tuple
/// (sampleable shadow-map support). Pre-existing PS records hash with
/// the old key shape and would mis-resolve once the new emitter starts
/// producing `depth2d<float>` bindings, so the cache must be wiped.
///
/// `15` switches depth-bound sampler call sites from `sample()` to
/// `sample_compare()` (D3D9 hardware-shadow PCF). The MSL text differs
/// for every PS with `depth_sampler_mask != 0`, so the cache must be
/// wiped again.
///
/// `16` `saturate`s the `sample_compare` reference to `[0, 1]`. Apple Silicon
/// promotes every depth format to `Depth32Float`, which — unlike the D24
/// UNORM the game authored for — does not clamp the comparison reference;
/// the emitted MSL gains a `saturate(...)` for every depth-bound tap.
///
/// `17` switches the on-disk format to zstd-compressed chunks with a
/// per-chunk xxh3 checksum (Single chunks for appends, one Bundle chunk
/// for the post-pre-warm compacted form). Pre-existing v16 files are
/// plaintext and unparseable under v17, so the schema-mismatch path wipes
/// them.
///
/// `21` moves the FF VS fog params from `vs_c[54]` to `vs_c[8]` (shifting
/// material rows 8..13 → 9..14 and per-light rows 14..53 → 15..54) and
/// replaces the per-slot `light_types: [u8; 8]` on `FfVsKey` with the
/// `light_active_mask` + `light_directional_mask` pair. Emitted MSL +
/// key bytes both change; old-version entries are unreusable.
///
/// `22` FF PS emits `depth2d<float>` + `sample_compare` for depth-format
/// sampler slots (`depth_sampler_mask`), matching the programmable PS path.
///
/// `23` FF PS honors the `D3DTA_ALPHAREPLICATE` / `D3DTA_COMPLEMENT` texture-arg
/// modifiers (previously dropped) and emits `D3DTOP_DOTPRODUCT3`. The MSL text
/// differs for any stage using those, so pre-`23` records must be wiped.
///
/// `24` adds `FfPsKey::specular_add` (the D3D9 end-of-cascade specular add)
/// and `D3DTA_SPECULAR` resolution, and the FF VS passes a declared vertex
/// COLOR1 through `color1` on the unlit/XYZRHW paths (previously hardwired
/// to zero). Key bytes and MSL text both change.
///
/// `25` grows the FF VS per-light constant stride from 5 to 6 rows (the new
/// `+5` row carries the light's specular color, replacing the previous
/// lightDiffuse weighting of the specular term), shifting the texture-
/// transform block 55→63 and the blend palette 87→95. MSL text changes for
/// every lit/TT/blend key; key bytes are unchanged.
///
/// `26` adds `FfVsKey::light_spot_mask` and the spot cone factor (SPOT
/// previously collapsed to DIRECTIONAL): the emitter gains the rho/penumbra
/// block, and the pack side carries spot scale/offset in the ambient and
/// specular rows' .w lanes. Key bytes and MSL text both change.
///
/// `27` adds `FfVsFlags::LOCAL_VIEWER` (`D3DRS_LOCALVIEWER`): the specular
/// view vector becomes the constant infinite-viewer direction when the RS
/// is FALSE instead of always `normalize(-posEye)`. Key bytes and MSL text
/// both change for lit + specular keys.
///
/// `28` the unlit FF VS defaults a missing COLOR0 stream to opaque white
/// (the D3D9 missing-DIFFUSE default) instead of the material diffuse
/// constant. MSL text changes for unlit keys without COLOR0.
///
/// `29` the programmable VS emitter default-initialises out.color0 to opaque
/// white and out.color1 to black, so a shader that omits oD0/oD1 yields the
/// D3D9 spec defaults instead of undefined varyings.
///
/// `30` the FF VS vertex-blend implicit last-weight contribution reads the
/// world-matrix palette at row 95 (was 87, off by 8 rows / 2 bones), matching
/// the explicit-weight loop and the encoder upload base.
///
/// `31` the FF VS texture-coordinate transform is rewritten to the D3D9
/// fixed-function rule: dimension-aware input masking (unbacked components
/// fill 0, not 1), `D3DTTFF_COUNT2..4` expand-and-matrix-multiply with
/// component-count masking, `COUNT1`/`DISABLE`/garbage pass through
/// untransformed, `PROJECTED` stashes the projective divisor in `.w`, and
/// `CAMERASPACENORMAL` texgen uses the un-normalized eye-space normal.
///
/// `32` the FF PS applies the `D3DTTFF_PROJECTED` projective divide — for a
/// projected stage it samples at `texcoord.xy / texcoord.w` (origin when
/// `.w == 0`) instead of `texcoord.xy`.
///
/// `33` SM1 pixel shaders route `r0` to the colour output — `ps_1_x` has no
/// `D3DSPR_COLOROUT` register, so the final pixel colour is whatever the shader
/// left in `r0`; previously such shaders returned the `oC0` float4(0.0) default.
///
/// `34` FF lighting runs without a vertex normal — emissive and ambient (global
/// and per-light) are normal-independent, so a lit draw with no normal now
/// emits them instead of skipping lighting entirely. Only the per-light N·L
/// diffuse/specular terms are gated on the normal.
///
/// `35` `texdepth` (`ps_1_4`) emits the D3D9 reference formula
/// `saturate(r.x / min(r.y, 1.0))` (divisor clamped to 1.0, no `r.y == 0`
/// special-case) instead of the previous guarded `r.x / r.y`.
///
/// `36` relative constant addressing (`c[a0 + N]`) overlays `def`-declared
/// constants via a lookup helper instead of reading the uniform buffer alone —
/// a `def`'d register the app never uploaded previously read zero.
///
/// `37` `texldp` (`D3DSI_TEXLD_PROJECT`) divides the SM2+ `texld` coordinate by
/// its `.w` before sampling; the modifier was previously dropped so a projected
/// sample collapsed to the unprojected one.
///
/// `38` `cnd` honours the shader version and `D3DSI_COISSUE`: `ps_1_4` compares
/// per component, and a co-issued non-alpha `ps_1_1`..`1_3` `cnd` selects src1
/// unconditionally (previously every `cnd` ran the scalar `.x > 0.5` compare).
///
/// `39` the Varyings struct gains a `position1` user varying so a secondary
/// POSITION semantic (`dcl_positionN`, N>=1) survives VS→PS instead of
/// clobbering the clip-space `[[position]]`. Shifts every later varying's
/// positional index, so VS and PS MSL both change.
///
/// `40` fog enabled with both vertex and table fog modes `D3DFOG_NONE` takes
/// the per-vertex fog factor from the specular (COLOR1) alpha (`fog_mode` 4)
/// rather than disabling fog.
///
/// `41` `nrm` scales all written components by 1/length(src.xyz) (was a
/// hardcoded `w = 1.0`); SM1/SM2 VS epilogue saturates the colour outputs
/// (oD0/oD1) to `[0, 1]` like fixed-function, SM3 unchanged.
///
/// `42` FF PS "invalid op" rewrite: a texture stage with no bound texture whose
/// colour/alpha op consumes a `D3DTA_TEXTURE` arg now resolves to
/// SELECTARG1(CURRENT) (was opaque white), per the D3D9 spec.
///
/// `43` raw-depth-fetch samplers: INTZ/DF24/DF16 (`depth_fetch_mask`) read the
/// raw stored depth via `.sample()` instead of `sample_compare` (the implicit
/// depth formats stay comparison shadow samplers).
///
/// `46` `ps_1_x` float-constant clamp: `def` constants and `cN`/`ps_c[N]` reads
/// in a `ps_1_x` shader clamp to `[-1, 1]` (fixed-point hardware range).
///
/// `50` FF eye normal uses the inverse-transpose of the `WorldView` 3×3 (the
/// D3D9 normal matrix, computed inline via the cofactor form) instead of the
/// plain WV, and renormalizes only when `D3DRS_NORMALIZENORMALS` is set (new
/// `FfVsFlags::NORMALIZE_NORMALS` key bit). The per-light diffuse N·L clamps to
/// `[0, 1]`. MSL text + key bytes change for lit FF keys.
///
/// `51` `ps_1_x` `texbem`/`texbeml` applies the `D3DTTFF_PROJECTED` divide to
/// the base texcoord (`tN.xy / tN.w`) before perturbing, matching the plain
/// `tex`/`texld` path — a projected bump stage previously sampled the
/// un-divided coordinate.
///
/// `52` FF eye-normal cofactor fix: the normal matrix is built from the WV
/// columns (`vs_c[i].xyz`) instead of the transposed components
/// (`vs_c[0].x, vs_c[1].x, …`). Schema `50` fed the columns, which transposed
/// the matrix and applied the *inverse* rotation to the normal — lighting swam
/// as the camera turned. A diagonal-scale world matrix is unaffected by the
/// transpose; MSL text changes for lit FF keys with a normal.
///
/// `53` adds `VariantKey::volume_sampler_mask` (FF PS declares
/// `texture3d<float>` + samples `.xyz` for slots bound to a volume texture).
/// The `VariantKey` hash shape changes, so every PS disk key moves.
///
/// `54` D3D9 fog overhaul: programmable VS gains the
/// no-oFog specular-alpha fallback (`out.fog = float4(out.color1.w)` — VS MSL
/// changes for every non-fog-writing shader); per-pixel TABLE fog lands
/// (`VariantKey::{fog_table_mode, fog_source_w}` — hash shape changes); the
/// PS slot-13 fog binding grows from a single `&fog_color` row to the
/// two-row `*fog_data` (colour + start/end/density/depth-bias), changing the
/// MSL of every fogged PS; and the shared `Varyings` struct gains the
/// `fog_z [[center_no_perspective]]` NDC-depth field (the table-fog Z
/// source), changing the MSL of EVERY shader.
///
/// `55` adds `VariantKey::cube_sampler_mask`, which changes fixed-function
/// pixel-shader keys and emits `texturecube<float>` bindings with `.xyz`
/// coordinates.
///
/// `56` adds the `pos_fixup.z`-selected depth clamp to the FF XYZRHW vertex
/// epilogue (the D3D9 depth-clamp rule, previously `MTLDepthClipMode::Clamp`
/// on the encoder), changing the MSL of every FF RHW vertex shader.
///
/// `57` adds `VariantKey::color_out_mask` (multiple render targets): the
/// programmable PS declares `oC0..oC3` locals and returns a `PsOut` struct
/// with one `[[color(i)]]` member per exported target, changing the key hash
/// of every pixel shader and the MSL of any that writes `oC1` or above.
///
/// `59` derives the lit FF vertex shader's normal matrix from the full 4x4
/// inverse of the world-view matrix (a projective fourth column changes the
/// normal), changing the MSL of every lit FF vertex shader with a normal.
///
/// `60` point size and point sprites: every vertex shader declares the
/// per-draw `VsDraw` uniform and clamps `[[point_size]]` from it (the FF VS
/// also reads a PSIZE attribute and applies `D3DRS_POINTSCALE*`), and
/// `VariantFlags::POINT_SPRITE` makes a pixel shader take `[[point_coord]]`,
/// changing the MSL of every vertex shader and the PS key hash shape.
///
/// `61` user clip planes: `VsDraw` grows the inverse view and six planes, the
/// vertex shaders emit `[[clip_distance]]` lanes keyed on the enabled-plane
/// count (a new `FfVsKey` field and a new programmable-VS disk-key input),
/// changing every vertex shader's MSL and both VS key hash shapes.
///
/// `62` vertex texture fetch: a `vs_3_0` shader declaring samplers gains
/// `[[texture(n)]]` / `[[sampler(n)]]` vertex-function arguments read via
/// `texldl`, changing the MSL of every such vertex shader.
///
/// `63` point size converts to render pixels: the vertex epilogue multiplies
/// the `POINTSIZE_MIN/MAX`-clamped size by `pos_fixup.w` (render pixels per
/// logical pixel) and the fixed-function attenuation divides the same lane
/// back out of the viewport height, changing the MSL of every vertex shader.
///
/// `64` sampler LOD bias: `VariantFlags::LOD_BIAS` gives a pixel shader a
/// per-slot bias uniform and puts `bias(...)` on every implicit-LOD sample,
/// changing the MSL of both pixel-shader emitters and the PS key hash shape.
///
/// `65` `D3DRS_MULTISAMPLEMASK`: a pixel shader drawing into a maskable
/// multisampled render target under a narrowed mask returns a struct with a
/// `[[sample_mask]]` member, which is a new PS variant bit and new MSL for
/// both the fixed-function and the programmable emitter.
///
/// `66` types a programmable pixel shader's sampler slots from the texture
/// bound to each slot (`VariantKey::{volume_sampler_mask, cube_sampler_mask}`)
/// instead of from its `dcl_<dim>`, so the `[[texture(n)]]` argument type and
/// the coordinate swizzle of any shader whose declaration disagrees with the
/// binding change.
///
/// `67` does the same for the four vertex texture fetch slots: a `vs_3_0` slot
/// bound to a volume or cube texture emits `texture3d<float>` /
/// `texturecube<float>` and a `.xyz` coordinate whatever its `dcl_<dim>` said,
/// which is new MSL and a new programmable-VS disk-key input.
///
/// `68` preserves the `texldb` instruction control and adds the coordinate's
/// `.w` to any sampler-state bias in the emitted programmable pixel shader.
///
/// `69` lets predicate-based flow control read `p0` through the ordinary
/// source path and emits `breakp` from that source, changing programmable MSL.
///
/// `70` preserves each arithmetic destination component whose corresponding
/// SM3 predicate component is false, changing programmable shader MSL.
///
/// `71` gives `vs_1_1` `expp` its four distinct result components instead of
/// broadcasting `exp2(src)` into every lane, changing MSL for those shaders.
///
/// `72` applies sampler operand swizzles to programmable texture sample results,
/// changing the MSL of every shader that uses a non-identity sampler swizzle.
///
/// `73` applies componentwise predicate selection to `mova` writes of the
/// integer address register, changing programmable vertex shader MSL.
pub const SHADER_CACHE_SCHEMA_VERSION: u32 = 73;

/// On-disk container format.
///
/// Separate from [`SHADER_CACHE_SCHEMA_VERSION`] so a translation change can
/// invalidate shader and pipeline identities without pretending the binary
/// framing changed.
pub const CACHE_FORMAT_VERSION: u32 = 18;

/// File magic.
///
/// ASCII `MTLD3DSH`. Recognises *our* file vs. unrelated content under
/// the same name.
pub const SHADER_CACHE_MAGIC: [u8; 8] = *b"MTLD3DSH";

/// Bytes of `magic | format version | shader schema version`.
pub const HEADER_LEN: usize = 16;

/// Bytes of `kind | _pad | key | frame_len | xxh3` before the variable zstd frame body.
///
/// The first 16 bytes are the same `kind|_pad|key|frame_len` layout as
/// v16's record header (with `frame_len` now counting compressed bytes);
/// the trailing 8 bytes hold the per-chunk xxh3 checksum added in v17.
pub const CHUNK_HEADER_LEN: usize = 24;

/// Bytes of `kind | _pad | key | body_len` before an inner plain-record body.
///
/// Such records live inside a Bundle chunk's decompressed payload. Same
/// layout as v16's whole-file record header — no checksum, since the
/// outer chunk's xxh3 already covers every byte of the bundle.
pub const RECORD_HEADER_LEN: usize = 16;

/// Out-of-band chunk-kind discriminator for a **Bundle** chunk.
///
/// One zstd frame holding many plain records. Outside the `CachedKind`
/// enum range so it round-trips cleanly through `from_byte`.
pub const RECORD_KIND_BUNDLE: u8 = 0xFF;

/// Chunk-kind discriminator for a render-pipeline recipe.
pub const RECORD_KIND_PIPELINE: u8 = 0xFE;

/// zstd level for append records (Single chunks).
///
/// Runs on the encoder thread, which is on the hot path: keep it cheap.
/// The weak per-frame ratio is fine because Single chunks fold into a
/// high-level Bundle on the next launch's compaction.
const ZSTD_APPEND_LEVEL: i32 = 3;

/// zstd level for the startup compaction Bundle.
///
/// Runs once on the pre-warm thread, off any hot path; spend the cycles
/// for a dense long-lived form.
const ZSTD_BUNDLE_LEVEL: i32 = 19;

/// On-disk record kind.
///
/// Discriminants are wire bytes: never reorder without bumping
/// `SHADER_CACHE_SCHEMA_VERSION`.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CachedKind {
    FfVs = 0,
    FfPs = 1,
    Sm1Vs = 2,
    Sm1Ps = 3,
    Sm2Vs = 4,
    Sm2Ps = 5,
    Sm3Vs = 6,
    Sm3Ps = 7,
}

impl CachedKind {
    /// Round-trip helper for the parser.
    ///
    /// `None` if the byte is outside the discriminant range — the parser
    /// drops the record.
    #[must_use]
    pub const fn from_byte(b: u8) -> Option<Self> {
        match b {
            0 => Some(Self::FfVs),
            1 => Some(Self::FfPs),
            2 => Some(Self::Sm1Vs),
            3 => Some(Self::Sm1Ps),
            4 => Some(Self::Sm2Vs),
            5 => Some(Self::Sm2Ps),
            6 => Some(Self::Sm3Vs),
            7 => Some(Self::Sm3Ps),
            _ => None,
        }
    }

    /// Map to the live-compile bucket.
    ///
    /// Pre-warm uses the same `(FF, SM1, SM2, SM3)` breakdown as the
    /// existing burst log.
    #[must_use]
    pub const fn compile_bucket(self) -> CompileBucket {
        match self {
            Self::FfVs | Self::FfPs => CompileBucket::Ff,
            Self::Sm1Vs | Self::Sm1Ps => CompileBucket::Sm1,
            Self::Sm2Vs | Self::Sm2Ps => CompileBucket::Sm2,
            Self::Sm3Vs | Self::Sm3Ps => CompileBucket::Sm3,
        }
    }

    #[must_use]
    pub const fn is_vertex(self) -> bool {
        matches!(self, Self::FfVs | Self::Sm1Vs | Self::Sm2Vs | Self::Sm3Vs)
    }

    #[must_use]
    pub const fn is_pixel(self) -> bool {
        matches!(self, Self::FfPs | Self::Sm1Ps | Self::Sm2Ps | Self::Sm3Ps)
    }

    #[must_use]
    pub const fn is_programmable(self) -> bool {
        !matches!(self, Self::FfVs | Self::FfPs)
    }

    /// Programmable: derive the kind from `(sm_major, is_pixel_shader)`.
    ///
    /// `None` for SM majors d3d9 should never see (DX10+).
    #[must_use]
    pub const fn from_programmable(sm_major: u8, is_pixel: bool) -> Option<Self> {
        match (sm_major, is_pixel) {
            (1, false) => Some(Self::Sm1Vs),
            (1, true) => Some(Self::Sm1Ps),
            (2, false) => Some(Self::Sm2Vs),
            (2, true) => Some(Self::Sm2Ps),
            (3, false) => Some(Self::Sm3Vs),
            (3, true) => Some(Self::Sm3Ps),
            _ => None,
        }
    }

    /// Per-shader Metal entry-point name, e.g. `mtld3d_vs_ff_5f3a0001`, `mtld3d_ps_sm3_a2b1c4d8`.
    ///
    /// The same string is written into the MSL function definition by the
    /// emitter and looked up via `newFunctionWithName:` on the unix side,
    /// so each compiled `MTLFunction` reports a distinct name in Xcode's
    /// pipeline-state inspector. Live-path (`encoder.rs`) and cache-load
    /// (`shader_prewarm.rs`) must share this helper to stay consistent.
    #[must_use]
    pub fn entry_name(self, disk_key: u64) -> String {
        let stage = match self {
            Self::FfVs | Self::Sm1Vs | Self::Sm2Vs | Self::Sm3Vs => "vs",
            Self::FfPs | Self::Sm1Ps | Self::Sm2Ps | Self::Sm3Ps => "ps",
        };
        let kind_label = match self {
            Self::FfVs | Self::FfPs => "ff",
            Self::Sm1Vs | Self::Sm1Ps => "sm1",
            Self::Sm2Vs | Self::Sm2Ps => "sm2",
            Self::Sm3Vs | Self::Sm3Ps => "sm3",
        };
        format!("mtld3d_{stage}_{kind_label}_{disk_key:08x}")
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CacheEntry {
    pub kind: CachedKind,
    pub key: u64,
    pub msl: String,
}

/// The two independent versions carried by the cache header.
#[derive(Debug, PartialEq, Eq)]
pub struct CacheHeader {
    pub format_version: u32,
    pub shader_schema_version: u32,
}

impl CacheHeader {
    pub const CURRENT: Self = Self {
        format_version: CACHE_FORMAT_VERSION,
        shader_schema_version: SHADER_CACHE_SCHEMA_VERSION,
    };
}

/// Stable reference from a pipeline recipe to one shader source record.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ShaderRecordRef {
    kind: CachedKind,
    key: u64,
}

impl ShaderRecordRef {
    #[must_use]
    pub const fn new(kind: CachedKind, key: u64) -> Self {
        Self { kind, key }
    }

    #[must_use]
    pub const fn kind(self) -> CachedKind {
        self.kind
    }

    #[must_use]
    pub const fn key(self) -> u64 {
        self.key
    }
}

/// Persistent description of one successfully-created render pipeline.
///
/// The stored snapshot always has null function handles. [`Self::resolve`]
/// installs functions compiled for the current Metal device before the
/// ordinary pipeline-state builder consumes it.
#[derive(PartialEq, Eq)]
pub struct PipelineRecipe {
    vs: ShaderRecordRef,
    ps: ShaderRecordRef,
    snapshot: PipelineSnapshot,
    vertex_attrs: Vec<VertexAttrDesc>,
}

impl PipelineRecipe {
    /// Capture one live pipeline without retaining either runtime function handle.
    #[must_use]
    pub fn from_snapshot(
        vs: ShaderRecordRef,
        ps: ShaderRecordRef,
        snapshot: &PipelineSnapshot,
        vertex_attrs: &[VertexAttrDesc],
    ) -> Self {
        let mut stored = snapshot.clone();
        stored.vs_fn = MetalHandle::NULL;
        stored.ps_fn = MetalHandle::NULL;
        stored.sample_count = stored.sample_count.max(1);
        stored.extra.has_alpha_mask &= stored.extra.present_mask;
        for index in 0..stored.extra.formats.len() {
            if !stored.extra.is_present(index) {
                stored.extra.formats[index] = ExtraColorAttachments::NONE.formats[index];
            }
        }
        Self {
            vs,
            ps,
            snapshot: stored,
            vertex_attrs: vertex_attrs.to_vec(),
        }
    }

    #[must_use]
    pub const fn vs(&self) -> ShaderRecordRef {
        self.vs
    }

    #[must_use]
    pub const fn ps(&self) -> ShaderRecordRef {
        self.ps
    }

    #[must_use]
    pub fn vertex_attrs(&self) -> &[VertexAttrDesc] {
        &self.vertex_attrs
    }

    /// Recreate the runtime snapshot using functions owned by the current device.
    #[must_use]
    pub fn resolve(
        &self,
        vs_fn: MetalHandle<mtld3d_shared::mtl_handle::MTLFunctionKind>,
        ps_fn: MetalHandle<mtld3d_shared::mtl_handle::MTLFunctionKind>,
    ) -> PipelineSnapshot {
        let mut snapshot = self.snapshot.clone();
        snapshot.vs_fn = vs_fn;
        snapshot.ps_fn = ps_fn;
        snapshot
    }

    /// Stable content identity of the explicit recipe encoding.
    #[must_use]
    pub fn disk_key(&self) -> u64 {
        let mut bytes = Vec::new();
        self.encode(&mut bytes);
        let mut hash = Xxh3::new();
        hash.write(&bytes);
        hash.finish()
    }

    fn encode(&self, out: &mut Vec<u8>) {
        out.push(self.vs.kind as u8);
        out.push(self.ps.kind as u8);
        out.push(self.snapshot.attach.bits());
        out.push(self.snapshot.rs.flags.bits());
        out.extend_from_slice(&self.vs.key.to_le_bytes());
        out.extend_from_slice(&self.ps.key.to_le_bytes());
        out.extend_from_slice(&self.snapshot.vdecl_hash.to_le_bytes());
        out.extend_from_slice(&(self.snapshot.color_format as u16).to_le_bytes());
        out.push(self.snapshot.sample_count);
        out.push(self.snapshot.ps_color_out_mask);
        out.push(self.snapshot.extra.present_mask);
        out.push(self.snapshot.extra.has_alpha_mask);
        out.extend_from_slice(&[
            self.snapshot.rs.src_blend,
            self.snapshot.rs.dst_blend,
            self.snapshot.rs.blend_op,
            self.snapshot.rs.src_blend_alpha,
            self.snapshot.rs.dst_blend_alpha,
            self.snapshot.rs.blend_op_alpha,
            self.snapshot.rs.color_write_mask,
        ]);
        out.extend_from_slice(&self.snapshot.rs.color_write_mask_ext);
        for format in self.snapshot.extra.formats {
            out.extend_from_slice(&(format as u16).to_le_bytes());
        }
        for layout in self.snapshot.stream_layouts {
            out.extend_from_slice(&layout.stride.to_le_bytes());
            out.push(layout.step as u8);
            out.extend_from_slice(&layout.step_rate.to_le_bytes());
        }
        let attr_count =
            u8::try_from(self.vertex_attrs.len()).expect("D3D9 vertex attribute count fits u8");
        out.push(attr_count);
        for attr in &self.vertex_attrs {
            out.push(u8::try_from(attr.attr_index).expect("D3D9 attribute index fits u8"));
            out.push(u8::try_from(attr.buffer_index).expect("D3D9 buffer index fits u8"));
            out.extend_from_slice(
                &u16::try_from(attr.offset)
                    .expect("D3D9 vertex attribute offset fits u16")
                    .to_le_bytes(),
            );
            out.push(attr.format as u8);
        }
    }

    fn decode(bytes: &[u8]) -> Option<Self> {
        let mut reader = RecipeReader::new(bytes);
        let vs_kind = CachedKind::from_byte(reader.u8()?)?;
        let ps_kind = CachedKind::from_byte(reader.u8()?)?;
        let attach = PipelineAttachFlags::from_bits(reader.u8()?)?;
        let rs_flags = PipelineRsFlags::from_bits(reader.u8()?)?;
        let vs = ShaderRecordRef::new(vs_kind, reader.u64()?);
        let ps = ShaderRecordRef::new(ps_kind, reader.u64()?);
        let vdecl_hash = reader.u64()?;
        let color_format = PixelFormat::from_repr(u32::from(reader.u16()?))?;
        let sample_count = reader.u8()?;
        let ps_color_out_mask = reader.u8()?;
        let extra_present_mask = reader.u8()?;
        let extra_has_alpha_mask = reader.u8()?;
        let rs = PipelineRsBits {
            flags: rs_flags,
            src_blend: reader.u8()?,
            dst_blend: reader.u8()?,
            blend_op: reader.u8()?,
            src_blend_alpha: reader.u8()?,
            dst_blend_alpha: reader.u8()?,
            blend_op_alpha: reader.u8()?,
            color_write_mask: reader.u8()?,
            color_write_mask_ext: [reader.u8()?, reader.u8()?, reader.u8()?],
        };
        let extra = ExtraColorAttachments {
            formats: [
                PixelFormat::from_repr(u32::from(reader.u16()?))?,
                PixelFormat::from_repr(u32::from(reader.u16()?))?,
                PixelFormat::from_repr(u32::from(reader.u16()?))?,
            ],
            present_mask: extra_present_mask,
            has_alpha_mask: extra_has_alpha_mask,
        };
        let mut stream_layouts = [StreamLayout::UNUSED; MAX_STREAMS as usize];
        for layout in &mut stream_layouts {
            *layout = StreamLayout {
                stride: reader.u32()?,
                step: VertexStepFunction::from_repr(u32::from(reader.u8()?))?,
                step_rate: reader.u32()?,
            };
        }
        let attr_count = usize::from(reader.u8()?);
        if attr_count > MAX_STREAMS as usize {
            return None;
        }
        let mut vertex_attrs = Vec::with_capacity(attr_count);
        for _ in 0..attr_count {
            vertex_attrs.push(VertexAttrDesc {
                attr_index: u32::from(reader.u8()?),
                buffer_index: u32::from(reader.u8()?),
                offset: u32::from(reader.u16()?),
                format: VertexFormat::from_repr(u32::from(reader.u8()?))?,
            });
        }
        if !reader.is_empty() {
            return None;
        }
        let recipe = Self {
            vs,
            ps,
            snapshot: PipelineSnapshot {
                vs_fn: MetalHandle::NULL,
                ps_fn: MetalHandle::NULL,
                vdecl_hash,
                stream_layouts,
                color_format,
                attach,
                rs,
                extra,
                ps_color_out_mask,
                sample_count,
            },
            vertex_attrs,
        };
        recipe.is_valid().then_some(recipe)
    }

    fn is_valid(&self) -> bool {
        self.vs.kind.is_vertex()
            && self.ps.kind.is_pixel()
            // An attachmentless sibling cannot serve a depth-only pass and
            // fails pipeline validation on Mac2. Keep useful cache records
            // while dropping these unused recipes from older writers.
            && (self.snapshot.has_depth()
                || self.snapshot.has_color_output()
                || self.snapshot.extra.present_mask != 0)
            && self.snapshot.sample_count != 0
            && self.snapshot.ps_color_out_mask & !0x0F == 0
            && self.snapshot.rs.color_write_mask & !0x0F == 0
            && self
                .snapshot
                .rs
                .color_write_mask_ext
                .iter()
                .all(|mask| mask & !0x0F == 0)
            && self.snapshot.extra.present_mask & !0x07 == 0
            && self.snapshot.extra.has_alpha_mask & !self.snapshot.extra.present_mask == 0
            && self
                .snapshot
                .stream_layouts
                .iter()
                .all(|layout| layout.stride != 0 || *layout == StreamLayout::UNUSED)
            && self.vertex_attrs.iter().all(|attr| {
                attr.attr_index < MAX_STREAMS
                    && attr.buffer_index < MAX_STREAMS
                    && attr.format != VertexFormat::Invalid
                    && self.snapshot.stream_layouts[attr.buffer_index as usize].is_used()
            })
    }
}

struct RecipeReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> RecipeReader<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, len: usize) -> Option<&'a [u8]> {
        let end = self.offset.checked_add(len)?;
        let value = self.bytes.get(self.offset..end)?;
        self.offset = end;
        Some(value)
    }

    fn u8(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }

    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }

    fn u16(&mut self) -> Option<u16> {
        Some(u16::from_le_bytes(self.take(2)?.try_into().ok()?))
    }

    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }

    const fn is_empty(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

/// Parsed and deduplicated cache contents.
pub struct CacheRecords {
    pub shaders: Vec<CacheEntry>,
    pub pipelines: Vec<PipelineRecipe>,
    pub needs_compaction: bool,
    /// End of the last complete, checksum-verified chunk.
    valid_len: usize,
}

#[derive(Debug, PartialEq, Eq)]
pub enum CacheReadError {
    /// Buffer too short or magic mismatch — almost certainly not our file.
    ///
    /// Caller should leave the file alone.
    WrongMagic,
}

/// Validate the 16-byte file header and return both version fields.
///
/// # Errors
///
/// [`CacheReadError::WrongMagic`] if the file is shorter than the header
/// or the leading 8 bytes don't match `SHADER_CACHE_MAGIC`.
///
/// # Panics
///
/// Panics if the slice indexing internally yields an unexpected length —
/// guarded by the header-length check above, so unreachable in practice.
pub fn read_header(bytes: &[u8]) -> Result<CacheHeader, CacheReadError> {
    if bytes.len() < HEADER_LEN || bytes[..8] != SHADER_CACHE_MAGIC {
        return Err(CacheReadError::WrongMagic);
    }
    let format_version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    let shader_schema_version = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
    Ok(CacheHeader {
        format_version,
        shader_schema_version,
    })
}

/// Walk chunks starting after the 16-byte file header.
///
/// Decompresses each zstd frame and verifies its xxh3.
///
/// Returns deduplicated shader records and pipeline recipes. Recipes whose
/// shader references are absent are discarded.
///
/// `needs_compaction` is `false` only when the file is exactly one
///   well-formed Bundle chunk with no inner duplicates and reached EOF
///   cleanly. `true` whenever anything else was observed: any Single
///   chunks, more than one Bundle, a torn / corrupt / unknown-kind chunk,
///   a stray file header mid-file, trailing partial-header garbage, or
///   duplicate or dangling records. The prewarm thread uses this to decide whether to
///   rewrite the file as a single dense Bundle.
///
/// # Panics
///
/// Panics if the slice indexing internally yields an unexpected length —
/// guarded by the per-chunk-length checks above, so unreachable on a
/// well-formed file.
#[must_use]
pub fn read_records(bytes: &[u8]) -> CacheRecords {
    let mut shaders = Vec::new();
    let mut pipelines = Vec::new();
    if bytes.len() < HEADER_LEN {
        return CacheRecords {
            shaders,
            pipelines,
            needs_compaction: false,
            valid_len: 0,
        };
    }
    let mut off = HEADER_LEN;
    let mut single_count: usize = 0;
    let mut bundle_count: usize = 0;
    let mut other_chunk = false;
    let mut seen_shaders: FxHashSet<ShaderRecordRef> = FxHashSet::default();
    let mut seen_pipelines: FxHashSet<u64> = FxHashSet::default();
    let mut duplicates = false;

    while off + CHUNK_HEADER_LEN <= bytes.len() {
        if bytes[off..off + 8] == SHADER_CACHE_MAGIC {
            // A file header inside the file is a stray from a duplicate
            // creation: skip its 16 bytes and keep parsing. No chunk header
            // can be mistaken for it, since the magic's leading byte would
            // have to be a `kind` outside `0..=7` and `RECORD_KIND_BUNDLE`.
            other_chunk = true;
            off += HEADER_LEN;
            continue;
        }
        let header_start = off;
        let kind_byte = bytes[off];
        let key = u64::from_le_bytes(bytes[off + 4..off + 12].try_into().unwrap());
        let frame_len = u32::from_le_bytes(bytes[off + 12..off + 16].try_into().unwrap()) as usize;
        let stored_checksum = u64::from_le_bytes(bytes[off + 16..off + 24].try_into().unwrap());
        let frame_start = off + CHUNK_HEADER_LEN;
        let Some(frame_end) = frame_start.checked_add(frame_len) else {
            // Length overflow ⇒ torn / corrupt. Stop cleanly.
            break;
        };
        if frame_end > bytes.len() {
            // Frame runs past EOF — torn write at the tail. Stop cleanly
            // and leave `off` pointing at the partial chunk so the
            // trailing-bytes check below flags it.
            break;
        }

        let header16: &[u8; 16] = bytes[header_start..header_start + 16].try_into().unwrap();
        let frame = &bytes[frame_start..frame_end];
        if chunk_xxh3(header16, frame) != stored_checksum {
            // Plaintext chunk header or frame body corrupted. We can't
            // trust `frame_len` to skip past this chunk safely (a flipped
            // bit in that field would mis-align every subsequent parse),
            // so stop here. Everything earlier in the file is intact;
            // the compaction rewrite below produces a clean file from
            // it, and any lost trailing chunks recompile next session.
            other_chunk = true;
            break;
        }

        if kind_byte == RECORD_KIND_BUNDLE {
            bundle_count += 1;
            match zstd::decode_all(frame) {
                Ok(plain) => {
                    let parsed = parse_plain_records(&plain);
                    other_chunk |= parsed.malformed;
                    for record in parsed.records {
                        push_record(
                            record,
                            &mut shaders,
                            &mut pipelines,
                            &mut seen_shaders,
                            &mut seen_pipelines,
                            &mut duplicates,
                        );
                    }
                }
                Err(_) => other_chunk = true,
            }
        } else if kind_byte == RECORD_KIND_PIPELINE {
            single_count += 1;
            match zstd::decode_all(frame) {
                Ok(payload) => match PipelineRecipe::decode(&payload) {
                    Some(recipe) if recipe.disk_key() == key => push_record(
                        PlainRecord::Pipeline(Box::new(recipe)),
                        &mut shaders,
                        &mut pipelines,
                        &mut seen_shaders,
                        &mut seen_pipelines,
                        &mut duplicates,
                    ),
                    _ => other_chunk = true,
                },
                Err(_) => other_chunk = true,
            }
        } else if let Some(kind) = CachedKind::from_byte(kind_byte) {
            single_count += 1;
            match zstd::decode_all(frame) {
                Ok(payload) => match String::from_utf8(payload) {
                    Ok(msl) => push_record(
                        PlainRecord::Shader(CacheEntry { kind, key, msl }),
                        &mut shaders,
                        &mut pipelines,
                        &mut seen_shaders,
                        &mut seen_pipelines,
                        &mut duplicates,
                    ),
                    Err(_) => other_chunk = true,
                },
                Err(_) => other_chunk = true,
            }
        } else {
            // Unknown kind byte — wire-byte forward-compat hook.
            other_chunk = true;
        }
        off = frame_end;
    }

    // Any bytes between `off` and EOF after the loop are an incomplete
    // chunk header (or the torn-frame `break` above) — treat as garbage
    // that warrants a rewrite.
    let trailing_garbage = off < bytes.len();

    let shader_refs: FxHashSet<ShaderRecordRef> = shaders
        .iter()
        .map(|entry| ShaderRecordRef::new(entry.kind, entry.key))
        .collect();
    let before = pipelines.len();
    pipelines
        .retain(|recipe| shader_refs.contains(&recipe.vs()) && shader_refs.contains(&recipe.ps()));
    let dangling = pipelines.len() != before;

    let already_optimal = bundle_count == 1
        && single_count == 0
        && !other_chunk
        && !duplicates
        && !dangling
        && !trailing_garbage;
    CacheRecords {
        shaders,
        pipelines,
        needs_compaction: !already_optimal,
        valid_len: off,
    }
}

/// Emit the 16-byte file header into `buf`.
///
/// Caller writes the buffer to the freshly-created file.
pub fn write_header(buf: &mut Vec<u8>) {
    buf.extend_from_slice(&SHADER_CACHE_MAGIC);
    buf.extend_from_slice(&CACHE_FORMAT_VERSION.to_le_bytes());
    buf.extend_from_slice(&SHADER_CACHE_SCHEMA_VERSION.to_le_bytes());
}

/// Serialise one Single chunk into `buf`.
///
/// Compresses the MSL bytes at `ZSTD_APPEND_LEVEL` and stamps the
/// chunk header with an xxh3 over `header_first_16_bytes ++ frame_bytes`.
/// Caller issues a single `write_all` against the open file so the chunk
/// either lands whole or the trailing torn portion gets dropped on next
/// read.
///
/// # Panics
///
/// Panics if the compressed frame exceeds 4 GiB (the wire format encodes
/// `frame_len` as a `u32`), or if zstd's in-memory `encode_all` returns
/// an error (effectively impossible for an `&[u8]` source). Real shader
/// frames are kilobytes, so unreachable in practice.
pub fn write_record(buf: &mut Vec<u8>, entry: &CacheEntry) {
    let frame = zstd::encode_all(entry.msl.as_bytes(), ZSTD_APPEND_LEVEL)
        .expect("zstd encode_all of in-memory MSL bytes");
    push_chunk(buf, entry.kind as u8, entry.key, &frame);
}

/// Serialise one pipeline-recipe Single chunk into `buf`.
///
/// # Panics
///
/// Panics if the encoded frame exceeds 4 GiB or if zstd fails to encode
/// the in-memory recipe.
pub fn write_pipeline_record(buf: &mut Vec<u8>, recipe: &PipelineRecipe) {
    let mut payload = Vec::new();
    recipe.encode(&mut payload);
    let frame = zstd::encode_all(payload.as_slice(), ZSTD_APPEND_LEVEL)
        .expect("zstd encode_all of in-memory pipeline recipe");
    push_chunk(buf, RECORD_KIND_PIPELINE, recipe.disk_key(), &frame);
}

/// Serialise one Bundle chunk containing every entry into `buf`.
///
/// The entries are written into a scratch plain-record blob (no
/// per-record checksum — the outer chunk's xxh3 covers everything) then
/// compressed at `ZSTD_BUNDLE_LEVEL`. The pre-warm thread uses this
/// for the one-shot startup rewrite.
///
/// # Panics
///
/// Panics if the compressed frame exceeds 4 GiB or if any entry's MSL
/// exceeds 4 GiB, or if zstd's in-memory `encode_all` returns an error
/// (effectively impossible). Real bundles are hundreds of KB at most.
pub fn write_bundle(buf: &mut Vec<u8>, shaders: &[CacheEntry], pipelines: &[PipelineRecipe]) {
    let mut plain = Vec::new();
    for entry in shaders {
        write_plain_record(&mut plain, entry);
    }
    for recipe in pipelines {
        write_plain_pipeline_record(&mut plain, recipe);
    }
    let frame = zstd::encode_all(plain.as_slice(), ZSTD_BUNDLE_LEVEL)
        .expect("zstd encode_all of in-memory plain-record blob");
    push_chunk(buf, RECORD_KIND_BUNDLE, 0, &frame);
}

/// Result of opening and validating the cache under its sidecar lock.
pub enum CacheLoad {
    Missing,
    Current(CacheRecords),
    InvalidatedVersion(CacheHeader),
    InvalidatedWrongMagic,
}

/// A process-safe append endpoint for shader and pipeline records.
///
/// Each append locks a stable sidecar and reopens the data path. Reopening is
/// required because a concurrent compactor can atomically replace the data
/// file while this value remains alive.
pub struct CacheWriter {
    path: PathBuf,
}

impl CacheWriter {
    /// Open the stable sidecar lock for `path`.
    ///
    /// # Errors
    ///
    /// Any I/O error opening or creating the sidecar.
    pub fn open(path: &Path) -> std::io::Result<Self> {
        drop(open_lock(path)?);
        Ok(Self {
            path: path.to_owned(),
        })
    }

    /// Append one shader record as one locked write.
    ///
    /// # Errors
    ///
    /// Any locking, validation, open, or write error.
    pub fn append_shader(&self, entry: &CacheEntry) -> std::io::Result<()> {
        let mut buf = Vec::with_capacity(CHUNK_HEADER_LEN + entry.msl.len());
        write_record(&mut buf, entry);
        self.append_bytes(&buf)
    }

    /// Append one pipeline recipe as one locked write.
    ///
    /// # Errors
    ///
    /// Any locking, validation, open, or write error.
    pub fn append_pipeline(&self, recipe: &PipelineRecipe) -> std::io::Result<()> {
        let mut buf = Vec::new();
        write_pipeline_record(&mut buf, recipe);
        self.append_bytes(&buf)
    }

    fn append_bytes(&self, bytes: &[u8]) -> std::io::Result<()> {
        let lock = open_lock(&self.path)?;
        lock_exclusive(&lock)?;
        let result = (|| {
            let mut file = open_data_for_append(&self.path)?;
            std::io::Write::write_all(&mut file, bytes)
        })();
        drop(lock);
        result
    }
}

/// Read and validate a cache while excluding appenders and compactors.
///
/// A wrong header is removed and an unreadable tail is truncated before the
/// lock is released. Concurrent writers can then append while this device
/// prewarms without placing records after a tail the parser cannot traverse.
///
/// # Errors
///
/// Any sidecar, lock, read, truncate, or remove error.
pub fn load(path: &Path) -> std::io::Result<CacheLoad> {
    let lock = open_lock(path)?;
    lock_exclusive(&lock)?;
    let result = (|| {
        let bytes = match fs::read(path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(CacheLoad::Missing);
            }
            Err(error) => return Err(error),
        };
        match read_header(&bytes) {
            Ok(header) if header == CacheHeader::CURRENT => {
                let records = read_records(&bytes);
                if records.valid_len < bytes.len() {
                    let len = u64::try_from(records.valid_len)
                        .map_err(|error| std::io::Error::other(error.to_string()))?;
                    OpenOptions::new().write(true).open(path)?.set_len(len)?;
                    mtld3d_shared::log_once_warn!(
                        target: crate::LOG_TARGET,
                        "shader_cache: discarded unreadable cache tail before reopening for writes"
                    );
                }
                Ok(CacheLoad::Current(records))
            }
            Ok(header) => {
                fs::remove_file(path)?;
                Ok(CacheLoad::InvalidatedVersion(header))
            }
            Err(CacheReadError::WrongMagic) => {
                fs::remove_file(path)?;
                Ok(CacheLoad::InvalidatedWrongMagic)
            }
        }
    })();
    drop(lock);
    result
}

/// Rewrite the latest cache contents into one Bundle while holding the sidecar lock.
///
/// The function rereads after taking the lock, so records appended since a
/// prewarm read are included rather than lost across the atomic rename.
///
/// Returns `(record_count, byte_count)` when a rewrite occurred.
///
/// # Errors
///
/// Any sidecar, lock, read, write, or rename error.
pub fn compact(path: &Path) -> std::io::Result<Option<(usize, usize)>> {
    let lock = open_lock(path)?;
    lock_exclusive(&lock)?;
    let result = (|| {
        let bytes = match fs::read(path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error),
        };
        let header = read_header(&bytes)
            .map_err(|_| std::io::Error::other("cache magic changed before compaction"))?;
        if header != CacheHeader::CURRENT {
            return Err(std::io::Error::other(
                "cache version changed before compaction",
            ));
        }
        let records = read_records(&bytes);
        if !records.needs_compaction {
            return Ok(None);
        }
        let mut buf = Vec::new();
        write_header(&mut buf);
        write_bundle(&mut buf, &records.shaders, &records.pipelines);

        let tmp = temp_sibling(path);
        fs::write(&tmp, &buf)?;
        if let Err(error) = fs::rename(&tmp, path) {
            let _ = fs::remove_file(&tmp);
            return Err(error);
        }
        Ok(Some((
            records.shaders.len() + records.pipelines.len(),
            buf.len(),
        )))
    })();
    drop(lock);
    result
}

fn open_lock(path: &Path) -> std::io::Result<File> {
    let path = lock_path(path);
    loop {
        match OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
        {
            Ok(lock) => return Ok(lock),
            Err(error) if is_lock_contention(&error) => {
                thread::sleep(Duration::from_micros(200));
            }
            Err(error) => return Err(error),
        }
    }
}

fn lock_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".lock");
    PathBuf::from(name)
}

fn lock_exclusive(lock: &File) -> std::io::Result<()> {
    loop {
        match lock.lock() {
            Ok(()) => return Ok(()),
            Err(error) if is_lock_contention(&error) => {
                // Wine can return ERROR_LOCK_VIOLATION from a blocking
                // LockFileEx call. Treat it as contention and retry.
                thread::sleep(Duration::from_micros(200));
            }
            Err(error) => return Err(error),
        }
    }
}

fn is_lock_contention(error: &std::io::Error) -> bool {
    error.kind() == std::io::ErrorKind::WouldBlock
        || (cfg!(windows) && error.raw_os_error() == Some(33))
}

fn open_data_for_append(path: &Path) -> std::io::Result<File> {
    use std::io::{Read as _, Seek as _, Write as _};

    let mut file = OpenOptions::new()
        .read(true)
        .append(true)
        .create(true)
        .open(path)?;
    if file.metadata()?.len() == 0 {
        let mut header = Vec::with_capacity(HEADER_LEN);
        write_header(&mut header);
        file.write_all(&header)?;
        return Ok(file);
    }
    file.rewind()?;
    let mut header_bytes = [0u8; HEADER_LEN];
    file.read_exact(&mut header_bytes)?;
    if read_header(&header_bytes).is_ok_and(|header| header == CacheHeader::CURRENT) {
        Ok(file)
    } else {
        Err(std::io::Error::other("cache header changed before append"))
    }
}

// Name a scratch file beside `path`, unique to this process and this call.
fn temp_sibling(path: &Path) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let seq = NEXT.fetch_add(1, Ordering::Relaxed);
    let mut name = path.as_os_str().to_owned();
    name.push(format!(".{}-{seq}.tmp", std::process::id()));
    PathBuf::from(name)
}

/// Hash any `Hash`-implementing FF state key to a u64 disk identifier.
///
/// `FfVsKey` / `FfPsKey` already implement `Hash` via `derive`, so this
/// is a one-liner at every call site.
pub fn ff_key_hash<T: Hash>(key: &T) -> u64 {
    let mut h = Xxh3::new();
    key.hash(&mut h);
    h.finish()
}

// Emit one 24-byte chunk header + zstd frame into `buf`. The checksum is
// computed over the first 16 header bytes (kind/pad/key/frame_len)
// followed by the frame body — the 8-byte checksum field itself is
// excluded.
fn push_chunk(buf: &mut Vec<u8>, kind: u8, key: u64, frame: &[u8]) {
    let frame_len = u32::try_from(frame.len()).expect("compressed frame > 4 GiB");
    let header16 = build_chunk_header16(kind, key, frame_len);
    let checksum = chunk_xxh3(&header16, frame);
    buf.extend_from_slice(&header16);
    buf.extend_from_slice(&checksum.to_le_bytes());
    buf.extend_from_slice(frame);
}

// Build the first 16 bytes of a chunk header: kind | _pad | key |
// frame_len. The trailing 8-byte xxh3 lives outside this helper so the
// checksum can be computed over the produced bytes.
const fn build_chunk_header16(kind: u8, key: u64, frame_len: u32) -> [u8; 16] {
    let mut h = [0u8; 16];
    h[0] = kind;
    // h[1..4] = padding (zero).
    let key_bytes = key.to_le_bytes();
    h[4] = key_bytes[0];
    h[5] = key_bytes[1];
    h[6] = key_bytes[2];
    h[7] = key_bytes[3];
    h[8] = key_bytes[4];
    h[9] = key_bytes[5];
    h[10] = key_bytes[6];
    h[11] = key_bytes[7];
    let len_bytes = frame_len.to_le_bytes();
    h[12] = len_bytes[0];
    h[13] = len_bytes[1];
    h[14] = len_bytes[2];
    h[15] = len_bytes[3];
    h
}

// xxh3_64 over `header_first_16_bytes ++ frame_bytes`. The checksum
// field itself is excluded so the value is self-consistent: writer
// computes it before the field exists in the buffer, reader computes it
// from the same span.
fn chunk_xxh3(header16: &[u8; 16], frame: &[u8]) -> u64 {
    let mut h = Xxh3::new();
    h.write(header16);
    h.write(frame);
    h.finish()
}

// Serialise one v15-style plain record (16-byte header + raw MSL bytes)
// into `buf`. Used only as a Bundle chunk's decompressed payload — the
// per-record checksum is intentionally absent there, since the outer
// chunk's xxh3 + the zstd frame integrity already cover every byte.
fn write_plain_record(buf: &mut Vec<u8>, entry: &CacheEntry) {
    let msl_bytes = entry.msl.as_bytes();
    let msl_len = u32::try_from(msl_bytes.len()).expect("MSL > 4 GiB");
    buf.push(entry.kind as u8);
    buf.extend_from_slice(&[0u8; 3]);
    buf.extend_from_slice(&entry.key.to_le_bytes());
    buf.extend_from_slice(&msl_len.to_le_bytes());
    buf.extend_from_slice(msl_bytes);
}

fn write_plain_pipeline_record(buf: &mut Vec<u8>, recipe: &PipelineRecipe) {
    let mut body = Vec::new();
    recipe.encode(&mut body);
    let body_len = u32::try_from(body.len()).expect("pipeline recipe > 4 GiB");
    buf.push(RECORD_KIND_PIPELINE);
    buf.extend_from_slice(&[0u8; 3]);
    buf.extend_from_slice(&recipe.disk_key().to_le_bytes());
    buf.extend_from_slice(&body_len.to_le_bytes());
    buf.extend_from_slice(&body);
}

enum PlainRecord {
    Shader(CacheEntry),
    Pipeline(Box<PipelineRecipe>),
}

struct PlainRecords {
    records: Vec<PlainRecord>,
    malformed: bool,
}

fn push_record(
    record: PlainRecord,
    shaders: &mut Vec<CacheEntry>,
    pipelines: &mut Vec<PipelineRecipe>,
    seen_shaders: &mut FxHashSet<ShaderRecordRef>,
    seen_pipelines: &mut FxHashSet<u64>,
    duplicates: &mut bool,
) {
    match record {
        PlainRecord::Shader(entry) => {
            let reference = ShaderRecordRef::new(entry.kind, entry.key);
            if seen_shaders.insert(reference) {
                shaders.push(entry);
            } else {
                *duplicates = true;
            }
        }
        PlainRecord::Pipeline(recipe) => {
            if seen_pipelines.insert(recipe.disk_key()) {
                pipelines.push(*recipe);
            } else {
                *duplicates = true;
            }
        }
    }
}

// Parse a Bundle chunk's decompressed plain-record payload. Same
// torn-record / unknown-kind / bad-UTF-8 skip discipline as the outer
// chunk parser, applied to the v15 inner layout.
fn parse_plain_records(bytes: &[u8]) -> PlainRecords {
    let mut records = Vec::new();
    let mut off = 0usize;
    let mut malformed = false;
    while off + RECORD_HEADER_LEN <= bytes.len() {
        let kind_byte = bytes[off];
        let key = u64::from_le_bytes(bytes[off + 4..off + 12].try_into().unwrap());
        let body_len = u32::from_le_bytes(bytes[off + 12..off + 16].try_into().unwrap()) as usize;
        let body_start = off + RECORD_HEADER_LEN;
        let Some(body_end) = body_start.checked_add(body_len) else {
            malformed = true;
            break;
        };
        if body_end > bytes.len() {
            malformed = true;
            break;
        }
        let body = &bytes[body_start..body_end];
        if kind_byte == RECORD_KIND_PIPELINE {
            match PipelineRecipe::decode(body) {
                Some(recipe) if recipe.disk_key() == key => {
                    records.push(PlainRecord::Pipeline(Box::new(recipe)));
                }
                _ => malformed = true,
            }
        } else if let Some(kind) = CachedKind::from_byte(kind_byte) {
            match std::str::from_utf8(body) {
                Ok(msl) => records.push(PlainRecord::Shader(CacheEntry {
                    kind,
                    key,
                    msl: msl.to_owned(),
                })),
                Err(_) => malformed = true,
            }
        } else {
            malformed = true;
        }
        off = body_end;
    }
    malformed |= off != bytes.len();
    PlainRecords { records, malformed }
}

#[cfg(test)]
mod tests;
