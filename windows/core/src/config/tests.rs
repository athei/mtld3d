//! Unit tests for the `mtld3d.conf` parser.
//!
//! `parse` is driven with file text plus optional env-override text and the result checked
//! field by field: documented defaults, both values of every boolean, later assignments and
//! env segments winning in that order, and malformed or unknown entries keeping the previous
//! value instead of derailing the rest of the input. Per-value shapes (MiB caps, decimal
//! render scale and its bounds, hex id lists, quoted strings) get their own cases.
//!
//! Every case here passes no app profile; the profile layer and its precedence against these
//! two live in `app_profile/tests.rs`, where a profile can be built.

use mtld3d_shared::mtl::{ColorSpacePolicy, SoftwareCursorPolicy};

use super::{AdapterSpoof, CursorScale, Mtld3dConfig, parse};

#[test]
fn empty_input_returns_defaults() {
    assert_eq!(parse(None, "", None), Mtld3dConfig::default());
}

#[test]
fn defaults_match_documented_values() {
    let d = Mtld3dConfig::default();
    assert!(!d.caps_all);
    assert!(!d.expand_packed16);
    assert!(!d.deny_float32_filtering);
    assert!(!d.managed_memory);
    assert!(!d.linear_align256);
    assert!(d.hdr_enable);
    assert_eq!(d.color_space, ColorSpacePolicy::Passthrough);
    assert_eq!(d.cursor_scale, CursorScale::Auto);
    assert_eq!(d.cursor_software, SoftwareCursorPolicy::Auto);
    assert!(d.shader_cache_enable);
    assert!(d.log_dir.is_empty());
    assert!(d.bytecode_dump_dir.is_empty());
    assert!(d.skip_shaders.is_empty());
    assert!(!d.query_flush_immediate);
    assert!(!d.buffer_ignore_lock_bounds);
    assert_eq!(d.vbib_retention_cap_bytes, 512 * 1024 * 1024);
    assert_eq!(d.pagebox_pool_cap_bytes, 128 * 1024 * 1024);
    assert_eq!(d.present_max_fps, 0);
    assert_eq!(d.render_scale_percent, 100);
}

/// The advertised video-memory ceiling defaults by guest width.
///
/// A 32-bit guest runs out of process address space long before a
/// unified-memory Mac runs out of GPU memory, and an engine that sizes its
/// streaming pool from the advertised figure will commit until it does.
#[test]
fn vram_budget_defaults_to_a_ceiling_only_on_a_32_bit_guest() {
    let d = parse(None, "", None);
    if cfg!(target_pointer_width = "32") {
        assert_eq!(d.vram_budget_cap_bytes, 1024 * 1024 * 1024);
    } else {
        assert_eq!(
            d.vram_budget_cap_bytes, 0,
            "a 64-bit guest keeps no ceiling"
        );
    }
}

#[test]
fn vram_budget_parses_as_mib_and_zero_lifts_the_ceiling() {
    assert_eq!(
        parse(None, "memory.vramBudgetMB = 512\n", None).vram_budget_cap_bytes,
        512 * 1024 * 1024
    );
    assert_eq!(
        parse(None, "memory.vramBudgetMB = 0\n", None).vram_budget_cap_bytes,
        0
    );
}

#[test]
fn pagebox_pool_cap_parses_as_mib() {
    let cfg = parse(None, "memory.pageboxPoolCapMB = 96\n", None);
    assert_eq!(cfg.pagebox_pool_cap_bytes, 96 * 1024 * 1024);
}

#[test]
fn pagebox_pool_cap_zero_disables() {
    let cfg = parse(
        None,
        "memory.pageboxPoolCapMB = 96\nmemory.pageboxPoolCapMB = 0\n",
        None,
    );
    assert_eq!(cfg.pagebox_pool_cap_bytes, 0);
}

#[test]
fn pagebox_pool_cap_garbage_keeps_default() {
    let cfg = parse(None, "memory.pageboxPoolCapMB = lots\n", None);
    assert_eq!(cfg.pagebox_pool_cap_bytes, 128 * 1024 * 1024);
}

#[test]
fn present_max_fps_positive_integer_parses() {
    let cfg = parse(None, "present.maxFps = 60\n", None);
    assert_eq!(cfg.present_max_fps, 60);
}

#[test]
fn present_max_fps_zero_means_uncapped() {
    let cfg = parse(None, "present.maxFps = 60\npresent.maxFps = 0\n", None);
    assert_eq!(cfg.present_max_fps, 0);
}

#[test]
fn present_max_fps_garbage_keeps_default() {
    let cfg = parse(None, "present.maxFps = fast\n", None);
    assert_eq!(cfg.present_max_fps, 0);
}

#[test]
fn render_scale_float_becomes_a_percentage() {
    assert_eq!(
        parse(None, "render.scale = 0.5\n", None).render_scale_percent,
        50
    );
    assert_eq!(
        parse(None, "render.scale = 0.75\n", None).render_scale_percent,
        75
    );
    assert_eq!(
        parse(None, "render.scale = 1\n", None).render_scale_percent,
        100
    );
}

#[test]
fn render_scale_accepts_its_exact_bounds() {
    assert_eq!(
        parse(None, "render.scale = 0.01\n", None).render_scale_percent,
        1
    );
    assert_eq!(
        parse(None, "render.scale = 1.0\n", None).render_scale_percent,
        100
    );
}

#[test]
fn render_scale_out_of_range_keeps_default() {
    for src in [
        // Above 1.0 has no path home: the present-side scaler only
        // enlarges, so a render bigger than the drawable cannot resolve.
        "render.scale = 1.5\n",
        "render.scale = 2.5\n",
        "render.scale = 0\n",
        "render.scale = -1\n",
        // Far outside u32: must be rejected, not saturated into range.
        "render.scale = 1e20\n",
        "render.scale = inf\n",
        "render.scale = NaN\n",
        "render.scale = half\n",
    ] {
        assert_eq!(
            parse(None, src, None).render_scale_percent,
            100,
            "{src:?} must keep the default"
        );
    }
}

#[test]
fn query_flush_immediate_round_trips_false() {
    let cfg = parse(None, "query.flushImmediate = false\n", None);
    assert!(!cfg.query_flush_immediate);
}

#[test]
fn query_flush_immediate_round_trips_true() {
    let cfg = parse(None, "query.flushImmediate = true\n", None);
    assert!(cfg.query_flush_immediate);
}

#[test]
fn depth_alias_same_size_defaults_off_and_round_trips() {
    let cfg = parse(None, "", None);
    assert!(!cfg.depth_alias_same_size);
    let cfg = parse(None, "depth.aliasSameSize = true\n", None);
    assert!(cfg.depth_alias_same_size);
}

#[test]
fn buffer_ignore_lock_bounds_defaults_off_and_round_trips() {
    let cfg = parse(None, "", None);
    assert!(!cfg.buffer_ignore_lock_bounds);
    let cfg = parse(None, "buffer.ignoreLockBounds = true\n", None);
    assert!(cfg.buffer_ignore_lock_bounds);
    let cfg = parse(None, "buffer.ignoreLockBounds = false\n", None);
    assert!(!cfg.buffer_ignore_lock_bounds);
}

#[test]
fn cursor_scale_auto_keyword_case_insensitive() {
    let cfg = parse(None, "cursor.scale = Auto\n", None);
    assert_eq!(cfg.cursor_scale, CursorScale::Auto);
}

#[test]
fn cursor_scale_positive_integer_parses_to_fixed() {
    let cfg = parse(None, "cursor.scale = 3\n", None);
    assert_eq!(cfg.cursor_scale, CursorScale::Fixed(3));
}

#[test]
fn cursor_scale_zero_keeps_default() {
    let cfg = parse(None, "cursor.scale = 0\n", None);
    assert_eq!(cfg.cursor_scale, CursorScale::Auto);
}

#[test]
fn cursor_scale_garbage_keeps_default() {
    let cfg = parse(None, "cursor.scale = jumbo\n", None);
    assert_eq!(cfg.cursor_scale, CursorScale::Auto);
}

#[test]
fn cursor_scale_auto_follows_the_display() {
    // Resolved again on every display move, so both readings of the same
    // policy have to answer for the display they were handed.
    assert_eq!(CursorScale::Auto.resolve(1), 1);
    assert_eq!(CursorScale::Auto.resolve(2), 2);
}

#[test]
fn cursor_scale_override_ignores_the_display() {
    assert_eq!(CursorScale::Fixed(3).resolve(1), 3);
    assert_eq!(CursorScale::Fixed(3).resolve(2), 3);
}

#[test]
fn cursor_scale_resolves_into_the_hcursor_range() {
    assert_eq!(CursorScale::Auto.resolve(0), 1);
    assert_eq!(CursorScale::Auto.resolve(99), 8);
    assert_eq!(CursorScale::Fixed(0).resolve(2), 1);
    assert_eq!(CursorScale::Fixed(99).resolve(2), 8);
}

#[test]
fn cursor_software_accepts_all_three_values_case_insensitive() {
    let cfg = parse(None, "cursor.software = TRUE\n", None);
    assert_eq!(cfg.cursor_software, SoftwareCursorPolicy::On);

    let cfg = parse(None, "cursor.software = False\n", None);
    assert_eq!(cfg.cursor_software, SoftwareCursorPolicy::Off);

    let cfg = parse(
        None,
        "cursor.software = true\ncursor.software = auto\n",
        None,
    );
    assert_eq!(cfg.cursor_software, SoftwareCursorPolicy::Auto);
}

#[test]
fn cursor_software_garbage_keeps_default() {
    let cfg = parse(None, "cursor.software = maybe\n", None);
    assert_eq!(cfg.cursor_software, SoftwareCursorPolicy::Auto);
}

#[test]
fn cursor_software_auto_follows_the_hdr_path() {
    assert!(SoftwareCursorPolicy::Auto.resolve(true));
    assert!(!SoftwareCursorPolicy::Auto.resolve(false));
}

#[test]
fn cursor_software_overrides_ignore_the_hdr_path() {
    assert!(SoftwareCursorPolicy::On.resolve(false));
    assert!(SoftwareCursorPolicy::On.resolve(true));
    assert!(!SoftwareCursorPolicy::Off.resolve(false));
    assert!(!SoftwareCursorPolicy::Off.resolve(true));
}

#[test]
fn color_space_accepts_both_policies_case_insensitive() {
    let cfg = parse(None, "color.space = accurate\n", None);
    assert_eq!(cfg.color_space, ColorSpacePolicy::Accurate);

    let cfg = parse(None, "color.space = PASSTHROUGH\n", None);
    assert_eq!(cfg.color_space, ColorSpacePolicy::Passthrough);
}

#[test]
fn color_space_unknown_value_keeps_default() {
    let cfg = parse(None, "color.space = vivid\n", None);
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
    let cfg = parse(None, src, None);
    assert!(cfg.hdr_enable);
    assert!(!cfg.caps_all);
}

#[test]
fn boolean_keys_round_trip_both_values() {
    let cfg = parse(
        None,
        "debug.capsAll = true\ncolor.hdr.enable = false\nshaderCache.enable = false\n\
         intel.expandPacked16 = true\nintel.denyFloat32Filtering = true\n\
         intel.managedMemory = true\nintel.linearAlign256 = true\n",
        None,
    );
    assert!(cfg.caps_all);
    assert!(!cfg.hdr_enable);
    assert!(!cfg.shader_cache_enable);
    assert!(cfg.expand_packed16);
    assert!(cfg.deny_float32_filtering);
    assert!(cfg.managed_memory);
    assert!(cfg.linear_align256);
}

#[test]
fn retired_debug_keys_are_ignored() {
    // The two keys moved into the `intel.*` family; their old names are
    // unknown keys now and must not reach the fields they used to set.
    let cfg = parse(
        None,
        "debug.expandPacked16 = true\ndebug.float32Filtering = false\n",
        None,
    );
    assert_eq!(cfg, Mtld3dConfig::default());
}

#[test]
fn booleans_are_case_insensitive() {
    let cfg = parse(
        None,
        "debug.capsAll = TRUE\ncolor.hdr.enable = False\n",
        None,
    );
    assert!(cfg.caps_all);
    assert!(!cfg.hdr_enable);
}

#[test]
fn whitespace_around_equals_is_tolerated() {
    let cfg = parse(
        None,
        "  debug.capsAll=true  \ncolor.hdr.enable\t=\ttrue\n",
        None,
    );
    assert!(cfg.caps_all);
    assert!(cfg.hdr_enable);
}

#[test]
fn quoted_string_value_preserves_inner_whitespace() {
    let cfg = parse(None, "debug.bytecodeDumpDir = \" /tmp/x \"\n", None);
    assert_eq!(cfg.bytecode_dump_dir, " /tmp/x ");
}

#[test]
fn unquoted_string_value_is_trimmed() {
    let cfg = parse(None, "debug.bytecodeDumpDir = /tmp/x\n", None);
    assert_eq!(cfg.bytecode_dump_dir, "/tmp/x");
}

#[test]
fn log_dir_is_a_plain_string() {
    let cfg = parse(None, "log.dir = logs\\mtld3d\n", None);
    assert_eq!(cfg.log_dir, "logs\\mtld3d");
    let cfg = parse(None, "", Some("log.dir=C:\\mtld3d-logs"));
    assert_eq!(cfg.log_dir, "C:\\mtld3d-logs");
}

#[test]
fn empty_string_disables_bytecode_dump() {
    let cfg = parse(None, "debug.bytecodeDumpDir =\n", None);
    assert!(cfg.bytecode_dump_dir.is_empty());
}

#[test]
fn hex_list_parses_with_and_without_0x_prefix() {
    let cfg = parse(None, "debug.skipShaders = 0xabc, def, 0x12345\n", None);
    assert_eq!(cfg.skip_shaders, vec![0xabc, 0xdef, 0x1_2345]);
}

#[test]
fn hex_list_drops_unparseable_entries_silently() {
    let cfg = parse(None, "debug.skipShaders = abc, gggg, def,, , 0\n", None);
    assert_eq!(cfg.skip_shaders, vec![0xabc, 0xdef, 0]);
}

#[test]
fn unknown_key_does_not_corrupt_other_assignments() {
    let cfg = parse(None, "bogusKey = whatever\ncolor.hdr.enable = true\n", None);
    assert!(cfg.hdr_enable);
}

#[test]
fn missing_equals_line_is_skipped() {
    let cfg = parse(
        None,
        "not a key value pair\ncolor.hdr.enable = true\n",
        None,
    );
    assert!(cfg.hdr_enable);
}

#[test]
fn non_boolean_value_keeps_default() {
    // Canary is a key that defaults to `false`, so a parser that
    // wrongly assigned `true` on garbage would fail here; with a
    // `true`-defaulting key the two outcomes are indistinguishable.
    let cfg = parse(None, "debug.capsAll = maybe\n", None);
    assert!(!cfg.caps_all, "default must be preserved");
}

#[test]
fn later_assignment_wins() {
    let cfg = parse(None, "debug.capsAll = false\ndebug.capsAll = true\n", None);
    assert!(cfg.caps_all);
}

#[test]
fn env_override_after_file_wins() {
    let cfg = parse(
        None,
        "color.hdr.enable = false\n",
        Some("color.hdr.enable=true"),
    );
    assert!(cfg.hdr_enable);
}

#[test]
fn env_override_merges_with_file() {
    let cfg = parse(None, "cursor.scale = 2\n", Some("color.hdr.enable=true"));
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
        ;memory.vbibRetentionCapMB=256\
        ;memory.pageboxPoolCapMB=96\
        ;present.maxFps=72\
        ;render.scale=0.5";
    let cfg = parse(None, "", Some(env));
    assert!(cfg.caps_all);
    assert!(cfg.hdr_enable);
    assert_eq!(cfg.color_space, ColorSpacePolicy::Accurate);
    assert_eq!(cfg.cursor_scale, CursorScale::Fixed(4));
    assert!(!cfg.shader_cache_enable);
    assert_eq!(cfg.bytecode_dump_dir, "/tmp/x");
    assert_eq!(cfg.skip_shaders, vec![0xabc, 0xdef]);
    assert!(!cfg.query_flush_immediate);
    assert_eq!(cfg.vbib_retention_cap_bytes, 256 * 1024 * 1024);
    assert_eq!(cfg.pagebox_pool_cap_bytes, 96 * 1024 * 1024);
    assert_eq!(cfg.present_max_fps, 72);
    assert_eq!(cfg.render_scale_percent, 50);
}

#[test]
fn env_override_empty_segments_skipped() {
    let cfg = parse(None, "", Some(";;color.hdr.enable=true;;"));
    assert!(cfg.hdr_enable);
}

#[test]
fn env_override_lists_keep_comma_separator() {
    let from_file = parse(None, "debug.skipShaders = 0xabc, 0xdef\n", None);
    let from_env = parse(None, "", Some("debug.skipShaders=0xabc,0xdef"));
    assert_eq!(from_file.skip_shaders, from_env.skip_shaders);
    assert_eq!(from_env.skip_shaders, vec![0xabc, 0xdef]);
}

#[test]
fn env_override_unknown_key_keeps_other_assignments() {
    let cfg = parse(None, "", Some("bogus.key=foo;color.hdr.enable=true"));
    assert!(cfg.hdr_enable);
}

#[test]
fn env_override_none_matches_file_only() {
    let src = "debug.capsAll = true\ncursor.scale = 3\n";
    assert_eq!(parse(None, src, None), parse(None, src, Some("")));
}

#[test]
fn adapter_spoof_parses_all_values() {
    assert_eq!(
        parse(None, "adapter.spoof = nvidia\n", None).adapter_spoof,
        AdapterSpoof::Nvidia
    );
    assert_eq!(
        parse(None, "adapter.spoof = AMD\n", None).adapter_spoof,
        AdapterSpoof::Amd
    );
    assert_eq!(
        parse(None, "adapter.spoof = ati\n", None).adapter_spoof,
        AdapterSpoof::Amd
    );
    assert_eq!(
        parse(None, "adapter.spoof = none\n", None).adapter_spoof,
        AdapterSpoof::None
    );
}

#[test]
fn adapter_spoof_rejects_unknown_vendor() {
    let cfg = parse(None, "adapter.spoof = matrox\n", None);
    assert_eq!(cfg.adapter_spoof, AdapterSpoof::None, "default preserved");
}

#[test]
fn df_formats_defaults_on_and_parses_off() {
    assert!(parse(None, "", None).df_formats);
    assert!(!parse(None, "caps.dfFormats = false\n", None).df_formats);
}
