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
    /// Default: `false` — users on HDR displays opt in explicitly. File
    /// key: `color.hdr.enable`.
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
    /// `0` disables the cap. Default: 512 MiB. File key:
    /// `memory.vbibRetentionCapMB` (value in MiB).
    pub vbib_retention_cap_bytes: u64,
    /// Frame-rate ceiling applied via the present-throttle duration.
    ///
    /// Independent of the guest's vsync request. When both this and
    /// vsync are active the lower rate wins (the throttle takes the
    /// longer of the two frame durations); with vsync off it caps the
    /// otherwise-unthrottled free-run. `0` = uncapped. Default: `0`.
    /// File key: `present.maxFps`.
    pub present_max_fps: u32,
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
            hdr_enable: false,
            color_space: ColorSpacePolicy::Passthrough,
            cursor_scale: CursorScale::Auto,
            shader_cache_enable: true,
            bytecode_dump_dir: String::new(),
            skip_shaders: Vec::new(),
            query_flush_immediate: true,
            vbib_retention_cap_bytes: 512 * 1024 * 1024,
            present_max_fps: 0,
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
        "config: present.maxFps = {}", cfg.present_max_fps
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
            assign_retention_cap_mb(source, value, &mut cfg.vbib_retention_cap_bytes);
        }
        "present.maxFps" => assign_max_fps(source, value, &mut cfg.present_max_fps),
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

fn assign_retention_cap_mb(source: &str, value: &str, slot: &mut u64) {
    // `0` disables the cap; any other value is MiB → bytes.
    if let Ok(mb) = value.parse::<u32>() {
        *slot = u64::from(mb) * 1024 * 1024;
    } else {
        log_once_warn!(
            target: crate::LOG_TARGET,
            "{source}: 'memory.vbibRetentionCapMB = {value}' is not a non-negative integer (MiB) → kept {kept}",
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
mod tests {
    use mtld3d_shared::mtl::ColorSpacePolicy;

    use super::{CursorScale, Mtld3dConfig, parse};

    #[test]
    fn empty_input_returns_defaults() {
        assert_eq!(parse("", None), Mtld3dConfig::default());
    }

    #[test]
    fn defaults_match_documented_values() {
        let d = Mtld3dConfig::default();
        assert!(!d.caps_all);
        assert!(!d.hdr_enable);
        assert_eq!(d.color_space, ColorSpacePolicy::Passthrough);
        assert_eq!(d.cursor_scale, CursorScale::Auto);
        assert!(d.shader_cache_enable);
        assert!(d.bytecode_dump_dir.is_empty());
        assert!(d.skip_shaders.is_empty());
        assert!(d.query_flush_immediate);
        assert_eq!(d.present_max_fps, 0);
    }

    #[test]
    fn present_max_fps_positive_integer_parses() {
        let cfg = parse("present.maxFps = 60\n", None);
        assert_eq!(cfg.present_max_fps, 60);
    }

    #[test]
    fn present_max_fps_zero_means_uncapped() {
        let cfg = parse("present.maxFps = 60\npresent.maxFps = 0\n", None);
        assert_eq!(cfg.present_max_fps, 0);
    }

    #[test]
    fn present_max_fps_garbage_keeps_default() {
        let cfg = parse("present.maxFps = fast\n", None);
        assert_eq!(cfg.present_max_fps, 0);
    }

    #[test]
    fn query_flush_immediate_round_trips_false() {
        let cfg = parse("query.flushImmediate = false\n", None);
        assert!(!cfg.query_flush_immediate);
    }

    #[test]
    fn query_flush_immediate_round_trips_true() {
        let cfg = parse("query.flushImmediate = true\n", None);
        assert!(cfg.query_flush_immediate);
    }

    #[test]
    fn cursor_scale_auto_keyword_case_insensitive() {
        let cfg = parse("cursor.scale = Auto\n", None);
        assert_eq!(cfg.cursor_scale, CursorScale::Auto);
    }

    #[test]
    fn cursor_scale_positive_integer_parses_to_fixed() {
        let cfg = parse("cursor.scale = 3\n", None);
        assert_eq!(cfg.cursor_scale, CursorScale::Fixed(3));
    }

    #[test]
    fn cursor_scale_zero_keeps_default() {
        let cfg = parse("cursor.scale = 0\n", None);
        assert_eq!(cfg.cursor_scale, CursorScale::Auto);
    }

    #[test]
    fn cursor_scale_garbage_keeps_default() {
        let cfg = parse("cursor.scale = jumbo\n", None);
        assert_eq!(cfg.cursor_scale, CursorScale::Auto);
    }

    #[test]
    fn color_space_accepts_both_policies_case_insensitive() {
        let cfg = parse("color.space = accurate\n", None);
        assert_eq!(cfg.color_space, ColorSpacePolicy::Accurate);

        let cfg = parse("color.space = PASSTHROUGH\n", None);
        assert_eq!(cfg.color_space, ColorSpacePolicy::Passthrough);
    }

    #[test]
    fn color_space_unknown_value_keeps_default() {
        let cfg = parse("color.space = vivid\n", None);
        assert_eq!(cfg.color_space, ColorSpacePolicy::Passthrough);
    }

    #[test]
    fn comments_and_blank_lines_are_skipped() {
        let src = "\
            # comment\n\
            \n\
            \t  \n\
            color.hdr.enable = true\n\
            # debug.capsAll = true\n\
        ";
        let cfg = parse(src, None);
        assert!(cfg.hdr_enable);
        assert!(!cfg.caps_all);
    }

    #[test]
    fn boolean_keys_round_trip_both_values() {
        let cfg = parse(
            "debug.capsAll = true\ncolor.hdr.enable = false\nshaderCache.enable = false\n",
            None,
        );
        assert!(cfg.caps_all);
        assert!(!cfg.hdr_enable);
        assert!(!cfg.shader_cache_enable);
    }

    #[test]
    fn booleans_are_case_insensitive() {
        let cfg = parse("debug.capsAll = TRUE\ncolor.hdr.enable = False\n", None);
        assert!(cfg.caps_all);
        assert!(!cfg.hdr_enable);
    }

    #[test]
    fn whitespace_around_equals_is_tolerated() {
        let cfg = parse("  debug.capsAll=true  \ncolor.hdr.enable\t=\ttrue\n", None);
        assert!(cfg.caps_all);
        assert!(cfg.hdr_enable);
    }

    #[test]
    fn quoted_string_value_preserves_inner_whitespace() {
        let cfg = parse("debug.bytecodeDumpDir = \" /tmp/x \"\n", None);
        assert_eq!(cfg.bytecode_dump_dir, " /tmp/x ");
    }

    #[test]
    fn unquoted_string_value_is_trimmed() {
        let cfg = parse("debug.bytecodeDumpDir = /tmp/x\n", None);
        assert_eq!(cfg.bytecode_dump_dir, "/tmp/x");
    }

    #[test]
    fn empty_string_disables_bytecode_dump() {
        let cfg = parse("debug.bytecodeDumpDir =\n", None);
        assert!(cfg.bytecode_dump_dir.is_empty());
    }

    #[test]
    fn hex_list_parses_with_and_without_0x_prefix() {
        let cfg = parse("debug.skipShaders = 0xabc, def, 0x12345\n", None);
        assert_eq!(cfg.skip_shaders, vec![0xabc, 0xdef, 0x1_2345]);
    }

    #[test]
    fn hex_list_drops_unparseable_entries_silently() {
        let cfg = parse("debug.skipShaders = abc, gggg, def,, , 0\n", None);
        assert_eq!(cfg.skip_shaders, vec![0xabc, 0xdef, 0]);
    }

    #[test]
    fn unknown_key_does_not_corrupt_other_assignments() {
        let cfg = parse("bogusKey = whatever\ncolor.hdr.enable = true\n", None);
        assert!(cfg.hdr_enable);
    }

    #[test]
    fn missing_equals_line_is_skipped() {
        let cfg = parse("not a key value pair\ncolor.hdr.enable = true\n", None);
        assert!(cfg.hdr_enable);
    }

    #[test]
    fn non_boolean_value_keeps_default() {
        let cfg = parse("color.hdr.enable = maybe\n", None);
        assert!(!cfg.hdr_enable, "default must be preserved");
    }

    #[test]
    fn later_assignment_wins() {
        let cfg = parse("debug.capsAll = false\ndebug.capsAll = true\n", None);
        assert!(cfg.caps_all);
    }

    #[test]
    fn env_override_after_file_wins() {
        let cfg = parse("color.hdr.enable = false\n", Some("color.hdr.enable=true"));
        assert!(cfg.hdr_enable);
    }

    #[test]
    fn env_override_merges_with_file() {
        let cfg = parse("cursor.scale = 2\n", Some("color.hdr.enable=true"));
        assert_eq!(cfg.cursor_scale, CursorScale::Fixed(2));
        assert!(cfg.hdr_enable);
    }

    #[test]
    fn env_override_supports_all_keys() {
        let env = "debug.capsAll=true\
            ;color.hdr.enable=true\
            ;color.space=accurate\
            ;cursor.scale=4\
            ;shaderCache.enable=false\
            ;debug.bytecodeDumpDir=/tmp/x\
            ;debug.skipShaders=0xabc,def\
            ;query.flushImmediate=false\
            ;present.maxFps=72";
        let cfg = parse("", Some(env));
        assert!(cfg.caps_all);
        assert!(cfg.hdr_enable);
        assert_eq!(cfg.color_space, ColorSpacePolicy::Accurate);
        assert_eq!(cfg.cursor_scale, CursorScale::Fixed(4));
        assert!(!cfg.shader_cache_enable);
        assert_eq!(cfg.bytecode_dump_dir, "/tmp/x");
        assert_eq!(cfg.skip_shaders, vec![0xabc, 0xdef]);
        assert!(!cfg.query_flush_immediate);
        assert_eq!(cfg.present_max_fps, 72);
    }

    #[test]
    fn env_override_empty_segments_skipped() {
        let cfg = parse("", Some(";;color.hdr.enable=true;;"));
        assert!(cfg.hdr_enable);
    }

    #[test]
    fn env_override_lists_keep_comma_separator() {
        let from_file = parse("debug.skipShaders = 0xabc, 0xdef\n", None);
        let from_env = parse("", Some("debug.skipShaders=0xabc,0xdef"));
        assert_eq!(from_file.skip_shaders, from_env.skip_shaders);
        assert_eq!(from_env.skip_shaders, vec![0xabc, 0xdef]);
    }

    #[test]
    fn env_override_unknown_key_keeps_other_assignments() {
        let cfg = parse("", Some("bogus.key=foo;color.hdr.enable=true"));
        assert!(cfg.hdr_enable);
    }

    #[test]
    fn env_override_none_matches_file_only() {
        let src = "debug.capsAll = true\ncursor.scale = 3\n";
        assert_eq!(parse(src, None), parse(src, Some("")));
    }
}
