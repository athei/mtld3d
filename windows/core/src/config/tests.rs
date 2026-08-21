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
    assert!(d.hdr_enable);
    assert_eq!(d.color_space, ColorSpacePolicy::Passthrough);
    assert_eq!(d.cursor_scale, CursorScale::Auto);
    assert!(d.shader_cache_enable);
    assert!(d.bytecode_dump_dir.is_empty());
    assert!(d.skip_shaders.is_empty());
    assert!(d.query_flush_immediate);
    assert_eq!(d.vbib_retention_cap_bytes, 512 * 1024 * 1024);
    assert_eq!(d.pagebox_pool_cap_bytes, 128 * 1024 * 1024);
    assert_eq!(d.present_max_fps, 0);
    assert_eq!(d.render_scale_percent, 100);
}

#[test]
fn pagebox_pool_cap_parses_as_mib() {
    let cfg = parse("memory.pageboxPoolCapMB = 96\n", None);
    assert_eq!(cfg.pagebox_pool_cap_bytes, 96 * 1024 * 1024);
}

#[test]
fn pagebox_pool_cap_zero_disables() {
    let cfg = parse(
        "memory.pageboxPoolCapMB = 96\nmemory.pageboxPoolCapMB = 0\n",
        None,
    );
    assert_eq!(cfg.pagebox_pool_cap_bytes, 0);
}

#[test]
fn pagebox_pool_cap_garbage_keeps_default() {
    let cfg = parse("memory.pageboxPoolCapMB = lots\n", None);
    assert_eq!(cfg.pagebox_pool_cap_bytes, 128 * 1024 * 1024);
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
fn render_scale_float_becomes_a_percentage() {
    assert_eq!(parse("render.scale = 0.5\n", None).render_scale_percent, 50);
    assert_eq!(
        parse("render.scale = 0.75\n", None).render_scale_percent,
        75
    );
    assert_eq!(parse("render.scale = 1\n", None).render_scale_percent, 100);
}

#[test]
fn render_scale_accepts_its_exact_bounds() {
    assert_eq!(parse("render.scale = 0.01\n", None).render_scale_percent, 1);
    assert_eq!(
        parse("render.scale = 1.0\n", None).render_scale_percent,
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
            parse(src, None).render_scale_percent,
            100,
            "{src:?} must keep the default"
        );
    }
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
    // Canary is a key that defaults to `false`, so a parser that
    // wrongly assigned `true` on garbage would fail here; with a
    // `true`-defaulting key the two outcomes are indistinguishable.
    let cfg = parse("debug.capsAll = maybe\n", None);
    assert!(!cfg.caps_all, "default must be preserved");
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
        ;memory.vbibRetentionCapMB=256\
        ;memory.pageboxPoolCapMB=96\
        ;present.maxFps=72\
        ;render.scale=0.5";
    let cfg = parse("", Some(env));
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
