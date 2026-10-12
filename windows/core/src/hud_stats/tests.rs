//! The HUD rows' counters, values, width budget, key order and header label.

use super::*;

/// A sample with every figure at the most it can hold.
fn widest_sample() -> HudSample {
    HudSample {
        draws_up: u32::MAX,
        draws_vb: u32::MAX,
        passes: u32::MAX,
        pipelines: usize::MAX,
        samplers: usize::MAX,
        textures: usize::MAX,
        free_va: u64::MAX,
        cpu_heap_bytes: u64::MAX,
        in_flight_bytes: u64::MAX,
        reuse_pool_bytes: u64::MAX,
        fixed_function: [usize::MAX; 2],
        models: [[u32::MAX; 3]; 2],
    }
}

/// A sample with every figure at zero.
fn empty_sample() -> HudSample {
    HudSample {
        draws_up: 0,
        draws_vb: 0,
        passes: 0,
        pipelines: 0,
        samplers: 0,
        textures: 0,
        free_va: 0,
        cpu_heap_bytes: 0,
        in_flight_bytes: 0,
        reuse_pool_bytes: 0,
        fixed_function: [0; 2],
        models: [[0; 3]; 2],
    }
}

/// The index of `key` in [`HUD_ROW_KEYS`].
fn row(key: &str) -> usize {
    HUD_ROW_KEYS
        .iter()
        .position(|candidate| *candidate == key)
        .unwrap_or_else(|| panic!("no row {key}"))
}

#[test]
fn every_key_with_its_widest_value_leaves_a_column_free() {
    let rows = hud_rows(&widest_sample());
    for (key, value) in HUD_ROW_KEYS.iter().zip(rows) {
        let value = value.unwrap_or_else(|| panic!("{key} has no row at its widest"));
        let width = HUD_KEY_INDENT + key.len() + 1 + value.to_string().len();
        assert!(
            width <= HUD_COLUMNS,
            "{key} with {value} takes {width} of {HUD_COLUMNS} columns"
        );
    }
}

#[test]
fn values_stay_bounded() {
    let rows = hud_rows(&widest_sample());
    assert_eq!(rows[row("Mem 32-bit free MiB")], Some(4096));
    assert_eq!(rows[row("Mem 32-bit largest MiB")], Some(4096));
    assert_eq!(rows[row("Mem CPU heap MiB")], Some(u32::MAX));
    assert_eq!(rows[row("Live textures")], Some(u32::MAX));
    assert_eq!(rows[row("Shaders VS FF")], Some(u32::MAX));
}

#[test]
fn keys_are_ascii_and_sorted_by_bytes_and_ignoring_case_alike() {
    for key in HUD_ROW_KEYS {
        assert!(key.is_ascii(), "{key} is not ASCII");
    }
    for pair in HUD_ROW_KEYS.windows(2) {
        let (first, second) = (pair[0], pair[1]);
        assert!(first < second, "{first} sorts after {second} by bytes");
        assert!(
            first.to_ascii_lowercase() < second.to_ascii_lowercase(),
            "{first} sorts after {second} ignoring case"
        );
    }
}

#[test]
fn keys_group_in_reading_order() {
    let mut groups: Vec<&str> = HUD_ROW_KEYS
        .iter()
        .filter_map(|key| key.split(' ').next())
        .collect();
    groups.dedup();
    assert_eq!(groups, ["Frame", "Live", "Mem", "Shaders"]);
}

#[test]
fn the_header_label_fits_beside_the_domain() {
    for build in [
        "v0.12.0",
        "v0.12.0-20-gbc8bc1e6",
        "v0.12.0-20-gbc8bc1e6-dirty",
        "v0.12.0-12345-gbc8bc1e6d3f4a5b6c7",
        "v0.12.0-12345-gbc8bc1e6d3f4a5b6c7-dirty",
        "v10.123.456-rc.12-12345-gbc8bc1e6",
        "bc8bc1e6",
    ] {
        let label = hud_label(build);
        assert!(
            HUD_DOMAIN.len() + 1 + label.len() <= HUD_COLUMNS,
            "{build} shows as {label}, too wide"
        );
    }
}

#[test]
fn a_release_and_a_short_development_identity_show_unchanged() {
    assert_eq!(hud_label("v0.12.0"), "v0.12.0");
    assert_eq!(hud_label("v0.12.0-20-gbc8bc1e6"), "v0.12.0-20-gbc8bc1e6");
    assert_eq!(
        hud_label("v0.12.0-20-gbc8bc1e6-dirty"),
        "v0.12.0-20-gbc8bc1e6-dirty"
    );
}

#[test]
fn a_long_development_identity_gives_up_hash_digits_first() {
    assert_eq!(
        hud_label("v0.12.0-12345-gbc8bc1e6d3f4a5b6c7"),
        "v0.12.0-12345-gbc8bc1e6d3f"
    );
    assert_eq!(
        hud_label("v0.12.0-12345-gbc8bc1e6d3f4a5b6c7-dirty"),
        "v0.12.0-12345-gbc8bc-dirty"
    );
    // No room for four hash digits: the first bytes stay.
    assert_eq!(
        hud_label("v10.123.456-rc.12-12345-gbc8bc1e6"),
        "v10.123.456-rc.12-12345-gb"
    );
}

#[test]
fn draws_and_passes_add_up_across_a_flush_and_freeze_at_present() {
    let mut stats = HudFrameStats::default();
    stats.count_draw(false);
    stats.count_draw(true);
    stats.end_submit(2, false);
    stats.count_draw(false);
    stats.count_draw(false);
    stats.count_draw(true);
    assert_eq!(
        stats.shown().draws_vb(),
        0,
        "nothing is shown before a present"
    );
    stats.end_submit(3, true);
    assert_eq!(stats.shown().draws_up(), 2);
    assert_eq!(stats.shown().draws_vb(), 3);
    assert_eq!(stats.shown().passes(), 5);
    stats.count_draw(true);
    stats.end_submit(1, false);
    assert_eq!(
        stats.shown().draws_up(),
        2,
        "a flush leaves the shown frame alone"
    );
    stats.end_submit(0, true);
    assert_eq!(stats.shown().draws_up(), 1);
    assert_eq!(stats.shown().draws_vb(), 0);
    assert_eq!(stats.shown().passes(), 1);
}

#[test]
fn counters_wrap_instead_of_panicking() {
    let mut stats = HudFrameStats {
        draws: u32::MAX,
        draws_up: u32::MAX,
        passes: u32::MAX,
        shown: HudFrame::default(),
    };
    stats.count_draw(true);
    stats.end_submit(usize::MAX, true);
    assert_eq!(stats.shown().draws_up(), 0);
    assert_eq!(stats.shown().draws_vb(), 0);
}

#[test]
fn shader_programs_count_by_stage_and_model() {
    let mut models = ShaderModels::default();
    models.adopt(ShaderType::Pixel, 2);
    models.adopt(ShaderType::Pixel, 2);
    models.adopt(ShaderType::Pixel, 1);
    models.adopt(ShaderType::Vertex, 3);
    models.adopt(ShaderType::Vertex, 0);
    models.adopt(ShaderType::Vertex, 4);
    assert_eq!(models.counts(ShaderType::Pixel), [1, 2, 0]);
    assert_eq!(models.counts(ShaderType::Vertex), [0, 0, 1]);
}

#[test]
fn shader_rows_show_only_the_models_in_use() {
    let mut sample = empty_sample();
    sample.fixed_function = [12, 8];
    sample.models = [[4, 210, 0], [0, 118, 0]];
    let rows = hud_rows(&sample);
    assert_eq!(rows[row("Shaders PS FF")], Some(12));
    assert_eq!(rows[row("Shaders PS SM1")], Some(4));
    assert_eq!(rows[row("Shaders PS SM2")], Some(210));
    assert_eq!(rows[row("Shaders PS SM3")], None);
    assert_eq!(rows[row("Shaders VS FF")], Some(8));
    assert_eq!(rows[row("Shaders VS SM1")], None);
    assert_eq!(rows[row("Shaders VS SM2")], Some(118));
    assert_eq!(rows[row("Shaders VS SM3")], None);
}

#[test]
fn the_other_rows_show_even_at_zero() {
    let rows = hud_rows(&empty_sample());
    for key in [
        "Frame draws UP",
        "Frame draws VB",
        "Frame passes",
        "Live pipelines",
        "Live samplers",
        "Live textures",
        "Mem CPU heap MiB",
        "Mem in flight MiB",
        "Mem reuse pool MiB",
    ] {
        assert_eq!(rows[row(key)], Some(0), "{key}");
    }
}

#[test]
fn byte_figures_show_in_whole_mib() {
    let mut sample = empty_sample();
    sample.cpu_heap_bytes = (96 << 20) + (300 << 10);
    sample.in_flight_bytes = (1 << 20) - 1;
    sample.reuse_pool_bytes = 64 << 20;
    let rows = hud_rows(&sample);
    assert_eq!(rows[row("Mem CPU heap MiB")], Some(96));
    assert_eq!(rows[row("Mem in flight MiB")], Some(0));
    assert_eq!(rows[row("Mem reuse pool MiB")], Some(64));
}

#[test]
fn the_address_space_word_round_trips_and_zero_leaves_its_rows_out() {
    let packed = pack_free_va(1453, 512);
    assert_eq!(unpack_free_va(packed), (1453, 512));
    assert_eq!(
        unpack_free_va(pack_free_va(u64::MAX, u64::MAX)),
        (u32::MAX, u32::MAX)
    );
    let mut sample = empty_sample();
    sample.free_va = packed;
    let rows = hud_rows(&sample);
    assert_eq!(rows[row("Mem 32-bit free MiB")], Some(1453));
    assert_eq!(rows[row("Mem 32-bit largest MiB")], Some(512));
    sample.free_va = 0;
    let rows = hud_rows(&sample);
    assert_eq!(rows[row("Mem 32-bit free MiB")], None);
    assert_eq!(rows[row("Mem 32-bit largest MiB")], None);
}

#[test]
fn the_gate_lets_the_first_sample_through_then_one_per_period() {
    let mut gate = SampleGate::default();
    assert!(gate.due(1000, 500));
    assert!(!gate.due(1000, 500));
    assert!(!gate.due(1499, 500));
    assert!(gate.due(1500, 500));
    assert!(!gate.due(1999, 500));
    assert!(
        gate.due(5000, 500),
        "a long pause lets the next sample through"
    );
    assert!(gate.due(u64::MAX, 500));
    assert!(
        gate.due(u64::MAX.wrapping_add(500), 500),
        "the clock wrapping is a period"
    );
}

#[test]
fn the_index_constants_name_their_rows() {
    assert_eq!(HUD_ROW_KEYS[ROW_MEM_32_FREE], "Mem 32-bit free MiB");
    assert_eq!(HUD_ROW_KEYS[ROW_SHADERS], "Shaders PS FF");
    assert_eq!(HUD_ROW_KEYS[ROW_SHADERS + 4], "Shaders VS FF");
}
