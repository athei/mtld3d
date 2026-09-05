//! `mtld3d.conf` parser.
//!
//! Pure-string in, typed config out — no I/O, no env reads. The
//! PE-side wrapper in `windows/d3d9/src/config.rs` does the
//! EXE-relative file lookup and feeds the file body to [`parse`];
//! this module is host-testable through `cargo test -p mtld3d-core
//! --target x86_64-apple-darwin`.

use log::info;
use mtld3d_shared::{
    log_once_warn,
    mtl::{ColorSpacePolicy, SoftwareCursorPolicy},
};

use crate::app_profile::AppProfile;

/// Byte cap of the `PageBox` recycle pool when `memory.pageboxPoolCapMB` is not set.
///
/// The pool is constructed before any configuration is resolved and sized
/// from this, so the number lives here and nowhere else.
pub const DEFAULT_PAGEBOX_POOL_CAP_BYTES: u64 = 128 * 1024 * 1024;

/// Resolved runtime configuration.
///
/// One instance per `IDirect3D9`, built at `Direct3DCreate9` from the user's
/// `mtld3d.conf` (or all-defaults if the file is absent) and shared with every
/// device that interface creates.
///
/// Field shape stays flat — the dotted file keys (`debug.capsAll`,
/// `color.hdr.enable`, …) are a file-namespace choice for the user,
/// not a nesting choice for the struct. A flat layout keeps call sites
/// a single field access (`cfg.caps_all` vs. `cfg.debug.caps_all`)
/// and avoids a pointless sub-struct.
#[derive(Debug, PartialEq, Eq)]
// File-shape: each key maps to one independent toggle; nesting them
// into a state machine or two-variant enums obscures the conf-file
// mapping for no real benefit.
#[allow(clippy::struct_excessive_bools)]
pub struct Mtld3dConfig {
    /// Diagnostic mode: OR-in spec-max capability bits so the game requests every feature.
    ///
    /// Surfaces unimplemented paths via `log_once_warn!`. Default:
    /// `false`. File key: `debug.capsAll`.
    pub caps_all: bool,
    /// Force the packed 16-bit expansion path used on non-Apple-family GPUs.
    ///
    /// Treats the device as lacking the native packed 16-bit pixel
    /// formats: A4R4G4B4 / R5G6B5 / A1R5G5B5 / X1R5G5B5 textures are
    /// backed by BGRA8, every upload into one is widened from its 2 bpp staging
    /// by the GPU upload pass (a render pass whose fragment function
    /// reads the staging slab as a buffer argument) instead of a blit
    /// copy, and the 16-bit render target formats stop being
    /// advertised. Exists so the Intel/AMD path can be exercised on
    /// Apple Silicon. Default: `false`. File key:
    /// `intel.expandPacked16`.
    pub expand_packed16: bool,
    /// Force the negative 32-bit float filtering answer of GPUs without it.
    ///
    /// Combined with the device's own `MTLDevice.supports32BitFloatFiltering`
    /// answer, so `true` makes `CheckDeviceFormat(D3DUSAGE_QUERY_FILTER)`
    /// report NOTAVAILABLE for R32F / G32R32F / A32B32G32R32F on any
    /// device, and `false` leaves the device's answer alone; it never
    /// claims filtering the device lacks. Exists so the Intel/AMD path
    /// can be exercised on a device that does filter them. Default:
    /// `false`. File key: `intel.denyFloat32Filtering`.
    pub deny_float32_filtering: bool,
    /// Force the storage policy of a GPU without unified memory.
    ///
    /// Clears `GpuCaps::unified_memory` where the snapshot is built, so
    /// every CPU-visible buffer is created `Managed` and the encoder
    /// enqueues `didModifyRange:` after each CPU write, as on an
    /// Intel/AMD Mac. Exists so that path can be exercised on Apple
    /// Silicon, where a missed notify stays invisible. Default: `false`.
    /// File key: `intel.managedMemory`.
    pub managed_memory: bool,
    /// Force the 256-byte linear texture alignment of a Mac2 GPU.
    ///
    /// Raises `GpuCaps::min_linear_texture_align` to 256 where the
    /// snapshot is built (a larger device value is kept), so blit
    /// staging pads its rows to that floor and the mips whose pitch falls
    /// under it take the padded-staging or GPU upload pass paths, as on
    /// an Intel/AMD Mac. Default: `false`. File key:
    /// `intel.linearAlign256`.
    pub linear_align256: bool,
    /// Enable HDR present pipeline on EDR-capable displays.
    ///
    /// The display gates this, not the value: the present pipeline only
    /// goes HDR when the attached screen reports EDR headroom, so a
    /// non-EDR display renders identically either way. On a panel that
    /// does have headroom the HDR route is the better-looking one (the
    /// SDR blit throws away every nit above paper white), which is why
    /// it is the default rather than an opt-in. Set `false` to force
    /// the SDR blit regardless of display capability. Default: `true`.
    /// File key: `color.hdr.enable`.
    pub hdr_enable: bool,
    /// Colorspace tagging policy for the `CAMetalLayer` (both SDR and HDR paths).
    ///
    /// Default: [`ColorSpacePolicy::Passthrough`] — tag with the
    /// display's native `CGColorSpace`, max-vibrance rendering. File
    /// key: `color.space` (`passthrough` | `accurate`).
    pub color_space: ColorSpacePolicy,
    /// Cursor bitmap enlargement factor, hardware HCURSOR and software sprite alike.
    ///
    /// Default: [`CursorScale::Auto`], which follows Wine's retina mode (2
    /// when the prefix runs in retina mode, else 1). `Fixed(n)` overrides with
    /// the user's chosen multiplier (still clamped to `[1, 8]` at use
    /// site). File key: `cursor.scale` (`auto` | positive integer).
    pub cursor_scale: CursorScale,
    /// Draw the cursor in the unix-side overlay window instead of the hardware HCURSOR.
    ///
    /// Default: [`SoftwareCursorPolicy::Auto`], on when the HDR present path
    /// is active and off otherwise; `On` / `Off` force it. Resolved on the unix
    /// side at device creation. File key: `cursor.software`
    /// (`auto` | `true` | `false`).
    pub cursor_software: SoftwareCursorPolicy,
    /// Use the persistent on-disk shader cache.
    ///
    /// Default: `true`. File key: `shaderCache.enable`.
    pub shader_cache_enable: bool,
    /// Directory the process's log file and GPU traces go into.
    ///
    /// Resolved against the executable's directory; an absolute path stands
    /// as is. Empty string = `mtld3d-logs` beside the executable. The log
    /// file is one per process, so the first `Direct3DCreate9` names its
    /// location and later ones leave it. Default: `""`. File key: `log.dir`.
    pub log_dir: String,
    /// Directory to dump raw DXSO bytecode into on first sight of each shader id.
    ///
    /// Empty string = disabled. Default: `""`. File key:
    /// `debug.bytecodeDumpDir`.
    pub bytecode_dump_dir: String,
    /// Shader-identity bisection probe.
    ///
    /// Drop any draw whose VS or PS `pair_id` hash appears in this
    /// list. Default: empty. File key: `debug.skipShaders` —
    /// comma-separated hex u64s, optional `0x` prefix.
    pub skip_shaders: Vec<u64>,
    /// Return `S_OK` immediately on `GetData(D3DGETDATA_FLUSH)` for a `Pending` occlusion query.
    ///
    /// Skips the kernel block on `MTLCommandBuffer::waitUntilCompleted` and
    /// answers with the permissive count instead. Default: `false`, the
    /// spec-correct wait, because an engine that polls with FLUSH reads the
    /// count it gets back (a pixel-visibility fraction, an occlusion cull).
    /// `true` is for a title that uses the poll loop only as a GPU fence and
    /// never reads the number: the `wow` profile sets it, since both clients
    /// fence every loading-screen upload batch that way and the wait costs
    /// seconds per load. File key: `query.flushImmediate`.
    pub query_flush_immediate: bool,
    /// A newly bound same-size depth-stencil texture inherits the previous one's contents.
    ///
    /// D3D9-era drivers commonly backed all equal-size depth-stencil
    /// surfaces with one physical allocation, so engines of that era
    /// bind a *different* depth texture of the same dimensions and rely
    /// on the just-rendered scene depth being visible through it (the
    /// point of the trick: z-test one handle while sampling the other,
    /// which D3D9 forbids on a single surface). When enabled, binding a
    /// texture-backed depth-stencil whose dimensions match the
    /// previously bound one queues a GPU copy of the previous contents
    /// into it. Default: `false` — engines that clear every depth
    /// target before use (the common case) would pay one full-surface
    /// copy per switch for nothing. File key: `depth.aliasSameSize`.
    pub depth_alias_same_size: bool,
    /// Disbelieve the `[OffsetToLock, SizeToLock)` window a `Lock` on a static VB/IB announces.
    ///
    /// Every vertex or index buffer except a `DEFAULT` + `DYNAMIC` one
    /// keeps its CPU staging separate from the device buffer the GPU
    /// reads, so only the bytes recorded at `Lock` are uploaded. A
    /// handful of D3D9-era titles write past the window they named, and a
    /// real driver never noticed because the pointer it handed back was
    /// into the one allocation the GPU read. Default: `false`, the
    /// announcement is taken at its word: that is the dirty range the
    /// D3D9 contract describes, and the only affordable one here. Set
    /// `true` for a title whose geometry is stretched, folded or missing
    /// where it re-locks a static buffer: the announcement then binds
    /// only for a `MANAGED` or `DYNAMIC` buffer and every other one
    /// uploads whole. A whole-buffer range always overlaps a range a draw
    /// already read this frame, so it is not a wider copy but a rename: a
    /// fresh device buffer, a full device-to-device preserve copy, and
    /// retention against `memory.vbibRetentionCapMB`.
    /// File key: `buffer.ignoreLockBounds`.
    pub buffer_ignore_lock_bounds: bool,
    /// Proactive cap on live VB/IB retained-`PageBox` bytes.
    ///
    /// When live retention reaches this, the capped alloc path (a Lock
    /// rename, or a `Staged` buffer's `Unlock` snapshot) drains retired
    /// backings and, if still over, forces a mid-frame GPU-sync before
    /// allocating, bounding peak PE-heap retention so a camera-turn
    /// rename burst can't thrash the 32-bit game process.
    /// This is the only bound: allocation itself is infallible, so `0`
    /// means unbounded retention and an abort if the address space does
    /// run out (the A/B baseline arm). Default: 512 MiB. File key:
    /// `memory.vbibRetentionCapMB` (value in MiB).
    pub vbib_retention_cap_bytes: u64,
    /// Ceiling for the figure `GetAvailableTextureMem` advertises.
    ///
    /// `0` lifts the ceiling.
    ///
    /// An engine of this era sizes its texture and streaming pools off that
    /// figure and commits against it, so on a 32-bit guest an unrestricted
    /// report invites a title past the process address space. What fails then
    /// is a Metal command buffer, out of memory, whose rendering is discarded.
    /// Default: 1 GiB on a 32-bit guest, no ceiling on 64-bit. File key:
    /// `memory.vramBudgetMB` (value in MiB).
    pub vram_budget_cap_bytes: u64,
    /// Byte cap for the `PageBox` recycle pool. `0` = pool disabled.
    ///
    /// VB/IB `PageBox`es retired by the encoder's retention drain are
    /// parked in a per-size-class pool and handed back to the next
    /// same-size Lock-rename alloc, instead of cycling through the
    /// global allocator (whose page-return policy decommits them,
    /// making the game's first touch of every fresh box fault). The
    /// A/B measured ~900 MB/s of rename traffic collapsing to a few
    /// MB/s and a ~50x drop in process fault rate with the pool on;
    /// parked bytes peaked at 53 MiB in a quiet scene, so the default
    /// leaves ~2x headroom for busy scenes. The cap bounds the
    /// committed bytes the pool may park; over-cap boxes drop to the
    /// allocator as before. The pool is one per process, so the value the
    /// most recent `Direct3DCreate9` resolved is the one in force. Default:
    /// 128 MiB. File key: `memory.pageboxPoolCapMB` (value in MiB).
    pub pagebox_pool_cap_bytes: u64,
    /// Frame-rate ceiling applied via the present-throttle duration.
    ///
    /// Independent of the guest's vsync request. When both this and
    /// vsync are active the lower rate wins (the throttle takes the
    /// longer of the two frame durations); with vsync off it caps the
    /// otherwise-unthrottled free-run. `0` = uncapped. Default: `0`.
    /// File key: `present.maxFps`.
    pub present_max_fps: u32,
    /// Render resolution as a percentage of the reported back buffer, `MetalFX`-upscaled.
    ///
    /// `100` (the default) renders at the size the game sees, which is an
    /// exact identity: every derived rect is unscaled and present stays a 1:1
    /// copy. Below 100 the scene rasterizes on a smaller grid and `MetalFX`
    /// upscales it on present, trading pixels for frame rate. Stored as a
    /// percentage rather than a float so the arithmetic is integer-only and
    /// the struct keeps its `Eq`; the file key is written as a float. File
    /// key: `render.scale` (e.g. `0.75`), accepted range `(0, 1.0]`.
    pub render_scale_percent: u32,
    /// Present the adapter as a well-known GPU vendor.
    ///
    /// D3D9-era engines pick vendor-specific render paths (a depth copy
    /// via `StretchRect` on one vendor, a resolve hack on another) and
    /// disable those paths for a vendor they do not recognize, leaving
    /// depth-dependent effects reading a buffer nothing updates. The
    /// spoof fills `GetAdapterIdentifier` with a consistent vendor id,
    /// device id, description, driver name and driver version. Default:
    /// [`AdapterSpoof::None`] — report the real identity. File key:
    /// `adapter.spoof` (`none` | `nvidia` | `amd`).
    pub adapter_spoof: AdapterSpoof,
    /// Advertise the `DF24` / `DF16` sampleable-depth formats.
    ///
    /// These fourccs existed on one vendor's hardware. A game that
    /// probes them next to `INTZ` and finds both can take a mixed path
    /// no real GPU ever ran; hiding them (`false`) keeps such engines on
    /// the plain `INTZ` route (`INTZ` stays advertised). Default:
    /// `true`. File key: `caps.dfFormats`.
    pub df_formats: bool,
}

/// `adapter.spoof` policy: the vendor identity `GetAdapterIdentifier` reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdapterSpoof {
    None,
    Nvidia,
    Amd,
}

/// `cursor.scale` policy.
///
/// `Auto` takes the multiplier from Wine's retina mode, 2 when it is on
/// and 1 otherwise; `Fixed(n)` forces it to `n` (still clamped to `[1, 8]`
/// at the use site to match the HCURSOR bitmap downstream's expected
/// range).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CursorScale {
    Auto,
    Fixed(u32),
}

impl CursorScale {
    /// Resolve the cursor upscale factor for Wine's retina factor.
    ///
    /// Called again whenever the unix side republishes the factor, so
    /// `Fixed` has to keep answering with the user's number: a setting that
    /// pins the cursor size means pinned. The clamp is the range the HCURSOR
    /// builder accepts.
    #[must_use]
    pub const fn resolve(self, backing_scale: u32) -> u32 {
        let requested = match self {
            Self::Auto => backing_scale,
            Self::Fixed(n) => n,
        };
        // `u32::clamp` is not const.
        if requested < 1 {
            1
        } else if requested > 8 {
            8
        } else {
            requested
        }
    }
}

impl Default for Mtld3dConfig {
    fn default() -> Self {
        Self {
            caps_all: false,
            expand_packed16: false,
            deny_float32_filtering: false,
            managed_memory: false,
            linear_align256: false,
            hdr_enable: true,
            color_space: ColorSpacePolicy::Passthrough,
            cursor_scale: CursorScale::Auto,
            cursor_software: SoftwareCursorPolicy::Auto,
            shader_cache_enable: true,
            log_dir: String::new(),
            bytecode_dump_dir: String::new(),
            skip_shaders: Vec::new(),
            query_flush_immediate: false,
            depth_alias_same_size: false,
            buffer_ignore_lock_bounds: false,
            vbib_retention_cap_bytes: 512 * 1024 * 1024,
            // The 32-bit address space is the ceiling that matters, not the
            // GPU's memory: a unified-memory Mac has far more than a D3D9
            // title can use, while the guest process runs out first.
            vram_budget_cap_bytes: if cfg!(target_pointer_width = "32") {
                1024 * 1024 * 1024
            } else {
                0
            },
            pagebox_pool_cap_bytes: DEFAULT_PAGEBOX_POOL_CAP_BYTES,
            present_max_fps: 0,
            render_scale_percent: 100,
            adapter_spoof: AdapterSpoof::None,
            df_formats: true,
        }
    }
}

/// Resolve the three configuration layers into a [`Mtld3dConfig`].
///
/// Weakest first: the [`Default`] values, then the built-in
/// [`AppProfile`] for this application if it has one, then the
/// `mtld3d.conf` body, then the `MTLD3D_CONFIG` env override. A key set
/// by a later layer wins, so a user can always take one option back
/// from a profile without losing the rest of it.
///
/// `file_src` is the file body (newline-separated `key = value`),
/// `env_override` is the env-var body (semicolon-separated
/// `key=value`), and a profile's settings use the env form. All three
/// flow through the same per-entry decode. Missing keys keep the value
/// the layer below left.
///
/// Unrecognised keys, malformed entries, and unparseable values fire
/// `log_once_warn!` (tagged with `mtld3d.conf line N` for file input,
/// `MTLD3D_CONFIG` for env input, and the profile's name for a profile)
/// so a typo doesn't silently no-op, then parsing continues. The
/// pure-string interface keeps the parser host-testable.
#[must_use]
pub fn parse(
    profile: Option<&AppProfile>,
    file_src: &str,
    env_override: Option<&str>,
) -> Mtld3dConfig {
    let mut cfg = Mtld3dConfig::default();
    if let Some(profile) = profile {
        let source = format!("app profile '{}'", profile.name());
        for segment in profile.settings().split(';') {
            apply_line(&mut cfg, segment, &source, None);
        }
    }
    for (lineno, raw_line) in file_src.lines().enumerate() {
        apply_line(&mut cfg, raw_line, "mtld3d.conf", Some(lineno));
    }
    if let Some(env) = env_override {
        for segment in env.split(';') {
            apply_line(&mut cfg, segment, "MTLD3D_CONFIG", None);
        }
    }
    cfg
}

fn apply_line(cfg: &mut Mtld3dConfig, raw: &str, source: &str, lineno: Option<usize>) {
    let line = raw.trim();
    if line.is_empty() || line.starts_with('#') {
        return;
    }
    let Some(eq) = line.find('=') else {
        if let Some(n) = lineno {
            log_once_warn!(
                target: crate::LOG_TARGET,
                "{source} line {}: missing '=' → ignored",
                n + 1
            );
        } else {
            log_once_warn!(
                target: crate::LOG_TARGET,
                "{source}: segment missing '=' → ignored"
            );
        }
        return;
    };
    let key = line[..eq].trim();
    let value = unquote(line[eq + 1..].trim());
    apply(cfg, source, key, value);
}

/// Emit one `info!` line per resolved option.
///
/// Called from the PE-side `load()` after parse; users see exactly what
/// the runtime is acting on, even when the file is absent (defaults
/// logged too).
pub fn log_options(cfg: &Mtld3dConfig) {
    info!(target: crate::LOG_TARGET, "config: debug.capsAll = {}", cfg.caps_all);
    info!(
        target: crate::LOG_TARGET,
        "config: intel.expandPacked16 = {}", cfg.expand_packed16
    );
    info!(
        target: crate::LOG_TARGET,
        "config: intel.denyFloat32Filtering = {}", cfg.deny_float32_filtering
    );
    info!(
        target: crate::LOG_TARGET,
        "config: intel.managedMemory = {}", cfg.managed_memory
    );
    info!(
        target: crate::LOG_TARGET,
        "config: intel.linearAlign256 = {}", cfg.linear_align256
    );
    info!(target: crate::LOG_TARGET, "config: color.hdr.enable = {}", cfg.hdr_enable);
    info!(
        target: crate::LOG_TARGET,
        "config: color.space = {}",
        color_space_label(cfg.color_space)
    );
    info!(
        target: crate::LOG_TARGET,
        "config: cursor.scale = {}",
        cursor_scale_label(cfg.cursor_scale)
    );
    info!(
        target: crate::LOG_TARGET,
        "config: cursor.software = {}",
        cursor_software_label(cfg.cursor_software)
    );
    info!(
        target: crate::LOG_TARGET,
        "config: shaderCache.enable = {}", cfg.shader_cache_enable
    );
    info!(target: crate::LOG_TARGET, "config: log.dir = {:?}", cfg.log_dir);
    info!(
        target: crate::LOG_TARGET,
        "config: debug.bytecodeDumpDir = {:?}", cfg.bytecode_dump_dir
    );
    info!(
        target: crate::LOG_TARGET,
        "config: debug.skipShaders = {} hash(es)", cfg.skip_shaders.len()
    );
    info!(
        target: crate::LOG_TARGET,
        "config: query.flushImmediate = {}", cfg.query_flush_immediate
    );
    info!(
        target: crate::LOG_TARGET,
        "config: depth.aliasSameSize = {}", cfg.depth_alias_same_size
    );
    info!(
        target: crate::LOG_TARGET,
        "config: buffer.ignoreLockBounds = {}", cfg.buffer_ignore_lock_bounds
    );
    info!(
        target: crate::LOG_TARGET,
        "config: memory.vbibRetentionCapMB = {}",
        cfg.vbib_retention_cap_bytes / (1024 * 1024)
    );
    info!(
        target: crate::LOG_TARGET,
        "config: memory.vramBudgetMB = {}",
        cfg.vram_budget_cap_bytes / (1024 * 1024)
    );
    info!(
        target: crate::LOG_TARGET,
        "config: memory.pageboxPoolCapMB = {}",
        cfg.pagebox_pool_cap_bytes / (1024 * 1024)
    );
    info!(
        target: crate::LOG_TARGET,
        "config: present.maxFps = {}", cfg.present_max_fps
    );
    info!(
        target: crate::LOG_TARGET,
        "config: render.scale = {}",
        f64::from(cfg.render_scale_percent) / 100.0
    );
    info!(
        target: crate::LOG_TARGET,
        "config: adapter.spoof = {}",
        adapter_spoof_label(cfg.adapter_spoof)
    );
    info!(
        target: crate::LOG_TARGET,
        "config: caps.dfFormats = {}", cfg.df_formats
    );
}

const fn color_space_label(p: ColorSpacePolicy) -> &'static str {
    match p {
        ColorSpacePolicy::Passthrough => "passthrough",
        ColorSpacePolicy::Accurate => "accurate",
    }
}

fn cursor_scale_label(s: CursorScale) -> String {
    match s {
        CursorScale::Auto => "auto".to_owned(),
        CursorScale::Fixed(n) => n.to_string(),
    }
}

const fn cursor_software_label(p: SoftwareCursorPolicy) -> &'static str {
    match p {
        SoftwareCursorPolicy::Auto => "auto",
        SoftwareCursorPolicy::On => "true",
        SoftwareCursorPolicy::Off => "false",
    }
}

fn apply(cfg: &mut Mtld3dConfig, source: &str, key: &str, value: &str) {
    match key {
        "debug.capsAll" => assign_bool(source, key, value, &mut cfg.caps_all),
        "intel.expandPacked16" => assign_bool(source, key, value, &mut cfg.expand_packed16),
        "intel.denyFloat32Filtering" => {
            assign_bool(source, key, value, &mut cfg.deny_float32_filtering);
        }
        "intel.managedMemory" => assign_bool(source, key, value, &mut cfg.managed_memory),
        "intel.linearAlign256" => assign_bool(source, key, value, &mut cfg.linear_align256),
        "color.hdr.enable" => assign_bool(source, key, value, &mut cfg.hdr_enable),
        "color.space" => assign_color_space(source, value, &mut cfg.color_space),
        "cursor.scale" => assign_cursor_scale(source, value, &mut cfg.cursor_scale),
        "cursor.software" => assign_cursor_software(source, value, &mut cfg.cursor_software),
        "shaderCache.enable" => assign_bool(source, key, value, &mut cfg.shader_cache_enable),
        "log.dir" => value.clone_into(&mut cfg.log_dir),
        "debug.bytecodeDumpDir" => value.clone_into(&mut cfg.bytecode_dump_dir),
        "debug.skipShaders" => cfg.skip_shaders = parse_hex_list(value),
        "query.flushImmediate" => assign_bool(source, key, value, &mut cfg.query_flush_immediate),
        "depth.aliasSameSize" => assign_bool(source, key, value, &mut cfg.depth_alias_same_size),
        "buffer.ignoreLockBounds" => {
            assign_bool(source, key, value, &mut cfg.buffer_ignore_lock_bounds);
        }
        "memory.vbibRetentionCapMB" => {
            assign_cap_mb(source, key, value, &mut cfg.vbib_retention_cap_bytes);
        }
        "memory.vramBudgetMB" => {
            assign_cap_mb(source, key, value, &mut cfg.vram_budget_cap_bytes);
        }
        "memory.pageboxPoolCapMB" => {
            assign_cap_mb(source, key, value, &mut cfg.pagebox_pool_cap_bytes);
        }
        "present.maxFps" => assign_max_fps(source, value, &mut cfg.present_max_fps),
        "render.scale" => assign_render_scale(source, value, &mut cfg.render_scale_percent),
        "adapter.spoof" => assign_adapter_spoof(source, value, &mut cfg.adapter_spoof),
        "caps.dfFormats" => assign_bool(source, key, value, &mut cfg.df_formats),
        _ => log_once_warn!(
            target: crate::LOG_TARGET,
            "{source}: unknown key '{key}' → ignored"
        ),
    }
}

fn assign_cursor_scale(source: &str, value: &str, slot: &mut CursorScale) {
    if value.eq_ignore_ascii_case("auto") {
        *slot = CursorScale::Auto;
        return;
    }
    match value.parse::<u32>() {
        Ok(n) if n > 0 => *slot = CursorScale::Fixed(n),
        _ => log_once_warn!(
            target: crate::LOG_TARGET,
            "{source}: 'cursor.scale = {value}' is not 'auto' or a positive integer → kept {kept}",
            kept = cursor_scale_label(*slot)
        ),
    }
}

fn assign_cursor_software(source: &str, value: &str, slot: &mut SoftwareCursorPolicy) {
    match value.to_ascii_lowercase().as_str() {
        "auto" => *slot = SoftwareCursorPolicy::Auto,
        "true" => *slot = SoftwareCursorPolicy::On,
        "false" => *slot = SoftwareCursorPolicy::Off,
        other => log_once_warn!(
            target: crate::LOG_TARGET,
            "{source}: 'cursor.software = {other}' is not auto, true or false → kept {kept}",
            kept = cursor_software_label(*slot)
        ),
    }
}

fn assign_adapter_spoof(source: &str, value: &str, slot: &mut AdapterSpoof) {
    match value.to_ascii_lowercase().as_str() {
        "none" => *slot = AdapterSpoof::None,
        "nvidia" => *slot = AdapterSpoof::Nvidia,
        "amd" | "ati" => *slot = AdapterSpoof::Amd,
        other => log_once_warn!(
            target: crate::LOG_TARGET,
            "{source}: 'adapter.spoof = {other}' is not a known vendor (expected none/nvidia/amd) → kept {kept}",
            kept = adapter_spoof_label(*slot)
        ),
    }
}

const fn adapter_spoof_label(s: AdapterSpoof) -> &'static str {
    match s {
        AdapterSpoof::None => "none",
        AdapterSpoof::Nvidia => "nvidia",
        AdapterSpoof::Amd => "amd",
    }
}

fn assign_color_space(source: &str, value: &str, slot: &mut ColorSpacePolicy) {
    match value.to_ascii_lowercase().as_str() {
        "passthrough" => *slot = ColorSpacePolicy::Passthrough,
        "accurate" => *slot = ColorSpacePolicy::Accurate,
        other => log_once_warn!(
            target: crate::LOG_TARGET,
            "{source}: 'color.space = {other}' is not a known policy (expected passthrough/accurate) → kept {kept}",
            kept = color_space_label(*slot)
        ),
    }
}

fn assign_cap_mb(source: &str, key: &str, value: &str, slot: &mut u64) {
    // `0` disables the capped feature; any other value is MiB → bytes.
    if let Ok(mb) = value.parse::<u32>() {
        *slot = u64::from(mb) * 1024 * 1024;
    } else {
        log_once_warn!(
            target: crate::LOG_TARGET,
            "{source}: '{key} = {value}' is not a non-negative integer (MiB) → kept {kept}",
            kept = *slot / (1024 * 1024)
        );
    }
}

fn assign_max_fps(source: &str, value: &str, slot: &mut u32) {
    // `0` means uncapped; any other value is a frame-rate ceiling in Hz.
    if let Ok(fps) = value.parse::<u32>() {
        *slot = fps;
    } else {
        log_once_warn!(
            target: crate::LOG_TARGET,
            "{source}: 'present.maxFps = {value}' is not a non-negative integer (Hz) → kept {kept}",
            kept = *slot
        );
    }
}

/// Lowest render scale the knob accepts, as a percentage.
///
/// Just above zero: the only real constraint is that a scaled dimension must
/// not collapse, and `RenderScale::dimension` already floors it at one pixel.
/// Anything tighter would be an invented limit, and a user who asks for a
/// silly resolution can see the result for themselves.
const RENDER_SCALE_MIN_PERCENT: u32 = 1;

/// Highest render scale the knob accepts, as a percentage.
///
/// Rendering *above* the presented size would need a downscale on present,
/// and `MTLFXSpatialScaler` only ever enlarges. There is nothing else on the
/// unix side that can resize a frame, so supersampling is simply not offered.
const RENDER_SCALE_MAX_PERCENT: u32 = 100;

fn assign_render_scale(source: &str, value: &str, slot: &mut u32) {
    // Written as a decimal (`0.75`) because that is how a resolution scale
    // reads, stored as a percentage because everything downstream is integer
    // arithmetic.
    let percent = parse_hundredths(value)
        .filter(|p| (RENDER_SCALE_MIN_PERCENT..=RENDER_SCALE_MAX_PERCENT).contains(p));
    if let Some(p) = percent {
        *slot = p;
    } else {
        log_once_warn!(
            target: crate::LOG_TARGET,
            "{source}: 'render.scale = {value}' is not a number in [{min}, {max}] → kept {kept}",
            min = f64::from(RENDER_SCALE_MIN_PERCENT) / 100.0,
            max = f64::from(RENDER_SCALE_MAX_PERCENT) / 100.0,
            kept = f64::from(*slot) / 100.0
        );
    }
}

/// Parse a non-negative decimal such as `0.75` into hundredths (`75`).
///
/// Deliberately integer-only. Going through `f32` would need a float-to-int
/// cast that no bound check can make total, and it would put the exactness of
/// a user-visible setting at the mercy of binary rounding. Digits past the
/// hundredths place are truncated; anything that is not a plain decimal
/// (`-1`, `1e20`, `inf`, `NaN`, `half`) fails the digit parse and returns
/// `None`.
fn parse_hundredths(value: &str) -> Option<u32> {
    let trimmed = value.trim();
    let (int_part, frac_part) = trimmed.split_once('.').unwrap_or((trimmed, ""));
    let units: u32 = if int_part.is_empty() {
        0
    } else {
        int_part.parse().ok()?
    };
    let mut hundredths = 0;
    for (i, ch) in frac_part.chars().enumerate() {
        let digit = ch.to_digit(10)?;
        if i < 2 {
            hundredths = hundredths * 10 + digit;
        }
    }
    // Pad a single fractional digit out to hundredths (`0.5` → 50).
    if frac_part.len() == 1 {
        hundredths *= 10;
    }
    units.checked_mul(100)?.checked_add(hundredths)
}

fn assign_bool(source: &str, key: &str, value: &str, slot: &mut bool) {
    match value.to_ascii_lowercase().as_str() {
        "true" => *slot = true,
        "false" => *slot = false,
        other => log_once_warn!(
            target: crate::LOG_TARGET,
            "{source}: '{key} = {other}' is not a boolean (expected true/false) → kept {slot}",
            slot = *slot
        ),
    }
}

fn parse_hex_list(value: &str) -> Vec<u64> {
    value
        .split(',')
        .filter_map(|s| {
            let s = s.trim().trim_start_matches("0x");
            if s.is_empty() {
                return None;
            }
            u64::from_str_radix(s, 16).ok()
        })
        .collect()
}

fn unquote(value: &str) -> &str {
    if value.len() >= 2 && value.starts_with('"') && value.ends_with('"') {
        &value[1..value.len() - 1]
    } else {
        value
    }
}

#[cfg(test)]
mod tests;
