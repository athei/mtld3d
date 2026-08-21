//! `mtld3d.conf` parser.
//!
//! Pure-string in, typed config out — no I/O, no env reads. The
//! PE-side wrapper in `windows/d3d9/src/config.rs` does the
//! EXE-relative file lookup and feeds the file body to [`parse`];
//! this module is host-testable through `cargo test -p mtld3d-core
//! --target x86_64-apple-darwin`.

use log::info;
use mtld3d_shared::{log_once_warn, mtl::ColorSpacePolicy};

/// Resolved runtime configuration.
///
/// One instance built at startup from the user's `mtld3d.conf` (or
/// all-defaults if the file is absent).
///
/// Field shape stays flat — the dotted file keys (`debug.capsAll`,
/// `color.hdr.enable`, …) are a file-namespace choice for the user,
/// not a nesting choice for the struct. A flat layout keeps call sites
/// a single field access (`CONFIG.caps_all` vs. `CONFIG.debug.caps_all`)
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
    /// Hardware-cursor (`HCURSOR`) bitmap enlargement factor.
    ///
    /// Default: [`CursorScale::Auto`] — derive from the display's
    /// `NSWindow.backingScaleFactor`. `Fixed(n)` overrides with the
    /// user's chosen multiplier (still clamped to `[1, 8]` at use
    /// site). File key: `cursor.scale` (`auto` | positive integer).
    pub cursor_scale: CursorScale,
    /// Use the persistent on-disk shader cache.
    ///
    /// Default: `true`. File key: `shaderCache.enable`.
    pub shader_cache_enable: bool,
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
    /// Skips the kernel block on `MTLCommandBuffer::waitUntilCompleted`.
    /// Default: `true` — D3D9-era games use the FLUSH-poll loop as a
    /// poor-man's GPU fence to work around 2004-era drivers that lacked
    /// resource hazard tracking. Metal tracks hazards explicitly, so the
    /// fence buys nothing and just throttles our API thread (the
    /// project's bottleneck). Flip to `false` to restore the
    /// spec-correct kernel wait if a game ever needs the actual
    /// pixel count immediately after FLUSH. File key:
    /// `query.flushImmediate`.
    pub query_flush_immediate: bool,
    /// Proactive cap on live VB/IB retained-`PageBox` bytes.
    ///
    /// When live retention reaches this, the Lock-rename alloc path
    /// drains retired backings and, if still over, forces a mid-frame
    /// GPU-sync before allocating — bounding peak PE-heap retention so
    /// a camera-turn rename burst can't thrash the 32-bit game process.
    /// This is the only bound: allocation itself is infallible, so `0`
    /// means unbounded retention and an abort if the address space does
    /// run out (the A/B baseline arm). Default: 512 MiB. File key:
    /// `memory.vbibRetentionCapMB` (value in MiB).
    pub vbib_retention_cap_bytes: u64,
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
    /// allocator as before. Default: 128 MiB. File key:
    /// `memory.pageboxPoolCapMB` (value in MiB).
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
}

/// `cursor.scale` policy.
///
/// `Auto` derives the multiplier from the display's
/// `backingScaleFactor`; `Fixed(n)` forces it to `n` (still clamped to
/// `[1, 8]` at the use site to match the HCURSOR bitmap downstream's
/// expected range).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CursorScale {
    Auto,
    Fixed(u32),
}

impl Default for Mtld3dConfig {
    fn default() -> Self {
        Self {
            caps_all: false,
            hdr_enable: true,
            color_space: ColorSpacePolicy::Passthrough,
            cursor_scale: CursorScale::Auto,
            shader_cache_enable: true,
            bytecode_dump_dir: String::new(),
            skip_shaders: Vec::new(),
            query_flush_immediate: true,
            vbib_retention_cap_bytes: 512 * 1024 * 1024,
            pagebox_pool_cap_bytes: 128 * 1024 * 1024,
            present_max_fps: 0,
            render_scale_percent: 100,
        }
    }
}

/// Parse `mtld3d.conf` source text into a [`Mtld3dConfig`].
///
/// An optional `MTLD3D_CONFIG` env-var override is applied on top.
///
/// `file_src` is the file body (newline-separated `key = value`),
/// `env_override` is the env-var body (semicolon-separated
/// `key=value`). Both flow through the same per-entry decode; env
/// segments are applied after file lines so env wins on conflict.
/// Missing keys keep their [`Default`] value.
///
/// Unrecognised keys, malformed entries, and unparseable values fire
/// `log_once_warn!` (tagged with `mtld3d.conf line N` for file input,
/// `MTLD3D_CONFIG` for env input) so a typo doesn't silently no-op, then
/// parsing continues. The pure-string interface keeps the parser
/// host-testable.
#[must_use]
pub fn parse(file_src: &str, env_override: Option<&str>) -> Mtld3dConfig {
    let mut cfg = Mtld3dConfig::default();
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
        "config: shaderCache.enable = {}", cfg.shader_cache_enable
    );
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
        "config: memory.vbibRetentionCapMB = {}",
        cfg.vbib_retention_cap_bytes / (1024 * 1024)
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

fn apply(cfg: &mut Mtld3dConfig, source: &str, key: &str, value: &str) {
    match key {
        "debug.capsAll" => assign_bool(source, key, value, &mut cfg.caps_all),
        "color.hdr.enable" => assign_bool(source, key, value, &mut cfg.hdr_enable),
        "color.space" => assign_color_space(source, value, &mut cfg.color_space),
        "cursor.scale" => assign_cursor_scale(source, value, &mut cfg.cursor_scale),
        "shaderCache.enable" => assign_bool(source, key, value, &mut cfg.shader_cache_enable),
        "debug.bytecodeDumpDir" => value.clone_into(&mut cfg.bytecode_dump_dir),
        "debug.skipShaders" => cfg.skip_shaders = parse_hex_list(value),
        "query.flushImmediate" => assign_bool(source, key, value, &mut cfg.query_flush_immediate),
        "memory.vbibRetentionCapMB" => {
            assign_cap_mb(source, key, value, &mut cfg.vbib_retention_cap_bytes);
        }
        "memory.pageboxPoolCapMB" => {
            assign_cap_mb(source, key, value, &mut cfg.pagebox_pool_cap_bytes);
        }
        "present.maxFps" => assign_max_fps(source, value, &mut cfg.present_max_fps),
        "render.scale" => assign_render_scale(source, value, &mut cfg.render_scale_percent),
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
