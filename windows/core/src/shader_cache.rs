//! On-disk shader cache: append-only binary file `mtld3d_shaders.bin` next to the host EXE.
//!
//! Each successful MSL compile appends one *Single* chunk (per-frame zstd at
//! `ZSTD_APPEND_LEVEL`) onto the file. The startup pre-warm thread reads
//! every chunk, pre-compiles the MSL via the existing `CompileShaderLibrary`
//! thunk, and — if the file is not already a single dense *Bundle* — rewrites
//! it atomically as one Bundle chunk so cross-shader redundancy is captured
//! (zstd at `ZSTD_BUNDLE_LEVEL`). Session appends after that bundle land as
//! Single chunks at the tail and fold back into the bundle on the next launch.
//!
//! ## File layout (v17)
//!
//! ```text
//! [file header  16B]  MTLD3DSH | schema u32 LE | _pad 4B
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
//! * `kind == RECORD_KIND_BUNDLE` (`0xFF`) → **Bundle** chunk. Frame
//!   decompresses to a concatenation of *plain* records using the v15 layout
//!   (`[1B kind|3B _pad|8B key|4B msl_len][raw MSL]`, repeated). No inner
//!   checksum: the whole bundle frame is covered by the outer chunk's xxh3.
//! * `xxh3` = `xxh3_64` over `chunk_header[0..16] ++ frame_bytes`. Computed
//!   once at write, verified on read. Mismatch ⇒ skip the chunk via
//!   `frame_len`. This is the sole integrity mechanism: it already covers the
//!   frame body, so zstd's own per-frame checksum is left off as redundant.
//!
//! Robustness comes from the plaintext `frame_len` prefix plus the xxh3:
//! torn writes (frame runs past EOF) ⇒ `break`; xxh3 mismatch ⇒ skip;
//! decompress failure or bad UTF-8 ⇒ skip; unknown `kind` ⇒ skip via
//! `frame_len` (forward-compat hook).
//!
//! This module owns the binary format only — it is pure-Rust and host-
//! testable. Encoder-side write hooks and pre-warm-thread plumbing live in
//! `windows/d3d9`.

use std::{
    collections::HashSet,
    hash::{Hash, Hasher},
};

use xxhash_rust::xxh3::Xxh3;

use crate::shader_compile_stats::CompileBucket;

/// Bumped on any of: DXSO emitter changes, FF emitter changes, hash function, on-disk format.
///
/// A cache file with a different schema is wiped and rebuilt from
/// scratch.
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
pub const SHADER_CACHE_SCHEMA_VERSION: u32 = 54;

/// File magic.
///
/// ASCII `MTLD3DSH`. Recognises *our* file vs. unrelated content under
/// the same name.
pub const SHADER_CACHE_MAGIC: [u8; 8] = *b"MTLD3DSH";

/// Bytes of `magic | schema | _pad`.
pub const HEADER_LEN: usize = 16;

/// Bytes of `kind | _pad | key | frame_len | xxh3` before the variable zstd frame body.
///
/// The first 16 bytes are the same `kind|_pad|key|frame_len` layout as
/// v16's record header (with `frame_len` now counting compressed bytes);
/// the trailing 8 bytes hold the per-chunk xxh3 checksum added in v17.
pub const CHUNK_HEADER_LEN: usize = 24;

/// Bytes of `kind | _pad | key | msl_len` before the raw MSL bytes of an **inner plain record**.
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

/// zstd level for per-shader appends (Single chunks).
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

#[derive(Debug, PartialEq, Eq)]
pub enum CacheReadError {
    /// Buffer too short or magic mismatch — almost certainly not our file.
    ///
    /// Caller should leave the file alone.
    WrongMagic,
}

/// Validate the 16-byte file header and return its schema field.
///
/// Caller compares against `SHADER_CACHE_SCHEMA_VERSION` and decides between
/// proceeding to `read_records` (match) or wiping the file (mismatch).
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
pub fn read_header(bytes: &[u8]) -> Result<u32, CacheReadError> {
    if bytes.len() < HEADER_LEN || bytes[..8] != SHADER_CACHE_MAGIC {
        return Err(CacheReadError::WrongMagic);
    }
    let schema = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    Ok(schema)
}

/// Walk chunks starting after the 16-byte file header.
///
/// Decompresses each zstd frame and verifies its xxh3.
///
/// Returns `(entries, needs_compaction)`:
/// * `entries` — every successfully-parsed `CacheEntry`, in file order.
///   Duplicate `key`s appear in order; the caller dedupes.
/// * `needs_compaction` — `false` only when the file is exactly one
///   well-formed Bundle chunk with no inner duplicates and reached EOF
///   cleanly. `true` whenever anything else was observed: any Single
///   chunks, more than one Bundle, a torn / corrupt / unknown-kind chunk,
///   trailing partial-header garbage, or duplicate keys. The pre-warm
///   thread uses this to decide whether to rewrite the file as a single
///   dense Bundle.
///
/// Empty input (parsed zero entries) always returns
/// `needs_compaction = false` — there is nothing to compact.
///
/// # Panics
///
/// Panics if the slice indexing internally yields an unexpected length —
/// guarded by the per-chunk-length checks above, so unreachable on a
/// well-formed file.
#[must_use]
pub fn read_records(bytes: &[u8]) -> (Vec<CacheEntry>, bool) {
    let mut out = Vec::new();
    if bytes.len() < HEADER_LEN {
        return (out, false);
    }
    let mut off = HEADER_LEN;
    let mut single_count: usize = 0;
    let mut bundle_count: usize = 0;
    let mut other_chunk = false;
    let mut seen_keys: HashSet<u64> = HashSet::new();
    let mut duplicates = false;

    while off + CHUNK_HEADER_LEN <= bytes.len() {
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
                    for entry in parse_plain_records(&plain) {
                        if !seen_keys.insert(entry.key) {
                            duplicates = true;
                        }
                        out.push(entry);
                    }
                }
                Err(_) => other_chunk = true,
            }
        } else if let Some(kind) = CachedKind::from_byte(kind_byte) {
            single_count += 1;
            match zstd::decode_all(frame) {
                Ok(payload) => match String::from_utf8(payload) {
                    Ok(msl) => {
                        if !seen_keys.insert(key) {
                            duplicates = true;
                        }
                        out.push(CacheEntry { kind, key, msl });
                    }
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

    let needs_compaction = if out.is_empty() {
        // Nothing to compact regardless of what was in the file; the
        // caller will just leave it alone or treat it as cold-start.
        false
    } else {
        let already_optimal = bundle_count == 1
            && single_count == 0
            && !other_chunk
            && !duplicates
            && !trailing_garbage;
        !already_optimal
    };

    (out, needs_compaction)
}

/// Emit the 16-byte file header into `buf`.
///
/// Caller writes the buffer to the freshly-created file.
pub fn write_header(buf: &mut Vec<u8>) {
    buf.extend_from_slice(&SHADER_CACHE_MAGIC);
    buf.extend_from_slice(&SHADER_CACHE_SCHEMA_VERSION.to_le_bytes());
    buf.extend_from_slice(&[0u8; 4]);
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
pub fn write_bundle(buf: &mut Vec<u8>, entries: &[CacheEntry]) {
    let mut plain = Vec::new();
    for entry in entries {
        write_plain_record(&mut plain, entry);
    }
    let frame = zstd::encode_all(plain.as_slice(), ZSTD_BUNDLE_LEVEL)
        .expect("zstd encode_all of in-memory plain-record blob");
    push_chunk(buf, RECORD_KIND_BUNDLE, 0, &frame);
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

// Parse a Bundle chunk's decompressed plain-record payload. Same
// torn-record / unknown-kind / bad-UTF-8 skip discipline as the outer
// chunk parser, applied to the v15 inner layout.
fn parse_plain_records(bytes: &[u8]) -> Vec<CacheEntry> {
    let mut out = Vec::new();
    let mut off = 0usize;
    while off + RECORD_HEADER_LEN <= bytes.len() {
        let kind_byte = bytes[off];
        let key = u64::from_le_bytes(bytes[off + 4..off + 12].try_into().unwrap());
        let msl_len = u32::from_le_bytes(bytes[off + 12..off + 16].try_into().unwrap()) as usize;
        let body_start = off + RECORD_HEADER_LEN;
        let Some(body_end) = body_start.checked_add(msl_len) else {
            break;
        };
        if body_end > bytes.len() {
            break;
        }
        let Some(kind) = CachedKind::from_byte(kind_byte) else {
            off = body_end;
            continue;
        };
        let Ok(msl) = std::str::from_utf8(&bytes[body_start..body_end]) else {
            off = body_end;
            continue;
        };
        out.push(CacheEntry {
            kind,
            key,
            msl: msl.to_owned(),
        });
        off = body_end;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_file(entries_per_chunk: &[Vec<CacheEntry>], bundle_last: bool) -> Vec<u8> {
        let mut buf = Vec::new();
        write_header(&mut buf);
        let last_idx = entries_per_chunk.len().saturating_sub(1);
        for (i, group) in entries_per_chunk.iter().enumerate() {
            let as_bundle = bundle_last && i == last_idx;
            if as_bundle {
                write_bundle(&mut buf, group);
            } else {
                for entry in group {
                    write_record(&mut buf, entry);
                }
            }
        }
        buf
    }

    fn sample_entries() -> Vec<CacheEntry> {
        vec![
            CacheEntry {
                kind: CachedKind::Sm3Vs,
                key: 0xDEAD_BEEF_CAFE_BABE,
                msl: "vertex VsOut vs(Inputs in [[stage_in]]) { /* … */ }".into(),
            },
            CacheEntry {
                kind: CachedKind::FfPs,
                key: 0,
                msl: String::new(),
            },
            CacheEntry {
                kind: CachedKind::Sm2Ps,
                key: 0x0102_0304_0506_0708,
                msl: "fragment float4 ps() { return float4(1); }".into(),
            },
        ]
    }

    #[test]
    fn single_chunk_round_trip() {
        let entries = sample_entries();
        let buf = write_file(std::slice::from_ref(&entries), false);
        assert_eq!(read_header(&buf), Ok(SHADER_CACHE_SCHEMA_VERSION));
        let (read, needs_compaction) = read_records(&buf);
        assert_eq!(read, entries);
        // Singles only, no Bundle ⇒ compact next launch.
        assert!(needs_compaction);
    }

    #[test]
    fn bundle_chunk_round_trip_is_optimal() {
        let entries = sample_entries();
        let buf = write_file(std::slice::from_ref(&entries), true);
        assert_eq!(read_header(&buf), Ok(SHADER_CACHE_SCHEMA_VERSION));
        let (read, needs_compaction) = read_records(&buf);
        assert_eq!(read, entries);
        // Exactly one Bundle, no dupes, EOF clean ⇒ optimal.
        assert!(!needs_compaction);
        // First chunk byte after the file header is the Bundle discriminator.
        assert_eq!(buf[HEADER_LEN], RECORD_KIND_BUNDLE);
    }

    #[test]
    fn mixed_bundle_plus_singles_round_trip() {
        let bundle_entries = sample_entries();
        let later_appends = vec![CacheEntry {
            kind: CachedKind::Sm3Ps,
            key: 0xAAAA_BBBB_CCCC_DDDD,
            msl: "fragment float4 ps_later() { return float4(0,1,0,1); }".into(),
        }];
        let mut buf = Vec::new();
        write_header(&mut buf);
        write_bundle(&mut buf, &bundle_entries);
        for e in &later_appends {
            write_record(&mut buf, e);
        }
        let (read, needs_compaction) = read_records(&buf);
        let mut expected = bundle_entries.clone();
        expected.extend(later_appends);
        assert_eq!(read, expected);
        // Bundle + singles ⇒ not optimal.
        assert!(needs_compaction);
    }

    #[test]
    fn torn_trailing_chunk_dropped_and_flags_compaction() {
        let entries = sample_entries();
        let mut buf = write_file(std::slice::from_ref(&entries), false);
        // Truncate mid-frame of the final chunk.
        buf.truncate(buf.len() - 5);
        let (read, needs_compaction) = read_records(&buf);
        // Dropped the torn last chunk.
        assert_eq!(read.len(), entries.len() - 1);
        assert!(needs_compaction);
    }

    #[test]
    fn corrupt_chunk_header_caught_by_xxh3() {
        let mut buf = Vec::new();
        write_header(&mut buf);
        write_record(
            &mut buf,
            &CacheEntry {
                kind: CachedKind::Sm2Vs,
                key: 0xAABB,
                msl: "ok before".into(),
            },
        );
        let after_first = buf.len();
        write_record(
            &mut buf,
            &CacheEntry {
                kind: CachedKind::Sm2Ps,
                key: 0xCCDD,
                msl: "corrupted in header below".into(),
            },
        );
        // Flip a bit in the second chunk's `frame_len` field. Without
        // the xxh3 this would silently desync every subsequent parse;
        // with it, the chunk is detected as corrupt and we stop here.
        buf[after_first + 12] ^= 0x01;
        // Append one more well-formed chunk; since we can't trust the
        // corrupt frame_len to skip safely, this trailing chunk is
        // intentionally forfeit (recompiled next session).
        write_record(
            &mut buf,
            &CacheEntry {
                kind: CachedKind::Sm3Ps,
                key: 0xEEFF,
                msl: "ok after — forfeit on corruption-stop".into(),
            },
        );
        let (read, needs_compaction) = read_records(&buf);
        // Only the chunk before the corruption survives. The corrupt
        // chunk and everything after it are dropped; compaction rewrites
        // a clean file so the trailing chunk recompiles next launch.
        assert_eq!(read.len(), 1);
        assert_eq!(read[0].key, 0xAABB);
        assert!(needs_compaction);
    }

    #[test]
    fn corrupt_frame_body_caught_and_skipped() {
        let mut buf = Vec::new();
        write_header(&mut buf);
        write_record(
            &mut buf,
            &CacheEntry {
                kind: CachedKind::Sm2Vs,
                key: 0x1111,
                msl: "good".into(),
            },
        );
        let bad_start = buf.len();
        write_record(
            &mut buf,
            &CacheEntry {
                kind: CachedKind::Sm2Ps,
                key: 0x2222,
                msl: "frame body will be scrambled".into(),
            },
        );
        // Scramble a byte inside the second chunk's compressed frame
        // (past the 24-byte chunk header).
        buf[bad_start + CHUNK_HEADER_LEN + 2] ^= 0xFF;
        let (read, _) = read_records(&buf);
        assert_eq!(read.len(), 1);
        assert_eq!(read[0].key, 0x1111);
    }

    #[test]
    fn unknown_chunk_kind_skipped_via_frame_len() {
        let mut buf = Vec::new();
        write_header(&mut buf);
        // Hand-craft a chunk with kind = 0x42 (neither CachedKind nor Bundle),
        // valid xxh3, a tiny zstd frame as payload. Use write_record to
        // build a Single, then patch its kind byte after the fact and
        // recompute the checksum so the parser reaches the unknown-kind
        // arm rather than failing on xxh3.
        let weird_kind: u8 = 0x42;
        write_record(
            &mut buf,
            &CacheEntry {
                kind: CachedKind::FfVs,
                key: 0x9999,
                msl: "irrelevant".into(),
            },
        );
        let chunk_off = HEADER_LEN;
        buf[chunk_off] = weird_kind;
        let frame_len =
            u32::from_le_bytes(buf[chunk_off + 12..chunk_off + 16].try_into().unwrap()) as usize;
        let header16: [u8; 16] = buf[chunk_off..chunk_off + 16].try_into().unwrap();
        let frame_start = chunk_off + CHUNK_HEADER_LEN;
        let frame = &buf[frame_start..frame_start + frame_len];
        let new_checksum = chunk_xxh3(&header16, frame);
        buf[chunk_off + 16..chunk_off + 24].copy_from_slice(&new_checksum.to_le_bytes());
        // Followed by a valid chunk.
        write_record(
            &mut buf,
            &CacheEntry {
                kind: CachedKind::Sm3Ps,
                key: 0x4321,
                msl: "after weird".into(),
            },
        );
        let (read, needs_compaction) = read_records(&buf);
        assert_eq!(read.len(), 1);
        assert_eq!(read[0].key, 0x4321);
        assert!(needs_compaction);
    }

    #[test]
    fn duplicate_keys_flag_compaction() {
        let dup = CacheEntry {
            kind: CachedKind::Sm3Vs,
            key: 0x5555,
            msl: "first copy".into(),
        };
        let mut buf = Vec::new();
        write_header(&mut buf);
        write_bundle(&mut buf, &[dup.clone(), dup]);
        let (read, needs_compaction) = read_records(&buf);
        // Both entries are read; the dedupe is the caller's job.
        assert_eq!(read.len(), 2);
        assert!(needs_compaction);
    }

    #[test]
    fn empty_file_with_just_header_is_not_compacted() {
        let mut buf = Vec::new();
        write_header(&mut buf);
        let (read, needs_compaction) = read_records(&buf);
        assert!(read.is_empty());
        // Nothing to compact ⇒ pre-warm leaves the file alone.
        assert!(!needs_compaction);
    }

    #[test]
    fn read_header_rejects_wrong_magic() {
        let bytes = b"GARBAGE!\x01\x00\x00\x00\x00\x00\x00\x00";
        assert_eq!(read_header(bytes), Err(CacheReadError::WrongMagic));
    }

    #[test]
    fn read_header_rejects_short_input() {
        let bytes = b"MTLD3DSH";
        assert_eq!(read_header(bytes), Err(CacheReadError::WrongMagic));
    }

    #[test]
    fn read_header_returns_schema_for_caller_comparison() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&SHADER_CACHE_MAGIC);
        buf.extend_from_slice(&99u32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 4]);
        assert_eq!(read_header(&buf), Ok(99));
    }

    #[test]
    fn cached_kind_round_trips_via_byte() {
        for k in [
            CachedKind::FfVs,
            CachedKind::FfPs,
            CachedKind::Sm1Vs,
            CachedKind::Sm1Ps,
            CachedKind::Sm2Vs,
            CachedKind::Sm2Ps,
            CachedKind::Sm3Vs,
            CachedKind::Sm3Ps,
        ] {
            assert_eq!(CachedKind::from_byte(k as u8), Some(k));
        }
    }

    #[test]
    fn from_programmable_maps_supported_majors() {
        assert_eq!(
            CachedKind::from_programmable(1, false),
            Some(CachedKind::Sm1Vs)
        );
        assert_eq!(
            CachedKind::from_programmable(2, true),
            Some(CachedKind::Sm2Ps)
        );
        assert_eq!(
            CachedKind::from_programmable(3, false),
            Some(CachedKind::Sm3Vs)
        );
        assert_eq!(CachedKind::from_programmable(0, false), None);
        assert_eq!(CachedKind::from_programmable(4, true), None);
    }

    #[test]
    fn ff_key_hash_is_stable() {
        let a = (1u32, 2u32, 3u32);
        let b = (1u32, 2u32, 3u32);
        assert_eq!(ff_key_hash(&a), ff_key_hash(&b));
    }
}
