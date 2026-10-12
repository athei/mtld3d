//! The HUD rows' counters, values, width budget, key order and header label.

use super::*;

/// A sample with every figure at the most it can hold.
fn widest_sample() -> HudSample {
    HudSample {
        draws_up: u32::MAX,
        draws_vb: u32::MAX,
        passes: u32::MAX,
        textures: usize::MAX,
        largest_free_word: u64::MAX,
        guest_page_bytes: u64::MAX,
        in_flight_bytes: u64::MAX,
        reuse_pool_bytes: u64::MAX,
        fixed_function: usize::MAX,
        programs: usize::MAX,
    }
}

/// A sample with every figure at zero.
fn empty_sample() -> HudSample {
    HudSample {
        draws_up: 0,
        draws_vb: 0,
        passes: 0,
        textures: 0,
        largest_free_word: 0,
        guest_page_bytes: 0,
        in_flight_bytes: 0,
        reuse_pool_bytes: 0,
        fixed_function: 0,
        programs: 0,
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
    assert_eq!(rows[row("Mem 32-bit largest MiB")], Some(4096));
    assert_eq!(rows[row("Mem guest pages MiB")], Some(u32::MAX));
    assert_eq!(rows[row("Live textures")], Some(u32::MAX));
    assert_eq!(rows[row("Shaders programmable")], Some(u32::MAX));
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
    // No room for four hash digits: the hash goes with its `-g`.
    assert_eq!(
        hud_label("v10.123.456-rc.12-12345-gbc8bc1e6"),
        "v10.123.456-rc.12-12345"
    );
    assert_eq!(
        hud_label("v10.123.456-rc-12345-gbc8bc1e6-dirty"),
        "v10.123.456-rc-12345-dirty"
    );
    // Not a development identity: the first bytes stay.
    assert_eq!(
        hud_label("v10.123.456-release-candidate-12"),
        "v10.123.456-release-candid"
    );
}

#[test]
fn every_row_carries_its_own_field() {
    let sample = HudSample {
        draws_up: 1,
        draws_vb: 2,
        passes: 3,
        textures: 4,
        largest_free_word: largest_free_word(5),
        guest_page_bytes: 6 << 20,
        in_flight_bytes: 7 << 20,
        reuse_pool_bytes: 8 << 20,
        fixed_function: 9,
        programs: 10,
    };
    let rows = hud_rows(&sample);
    for (index, (key, value)) in HUD_ROW_KEYS.iter().zip(rows).enumerate() {
        let expected = u32::try_from(index + 1).expect("ten rows");
        assert_eq!(value, Some(expected), "{key}");
    }
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
fn every_row_but_the_address_space_shows_even_at_zero() {
    let rows = hud_rows(&empty_sample());
    for (key, value) in HUD_ROW_KEYS.iter().zip(rows) {
        if *key == "Mem 32-bit largest MiB" {
            assert_eq!(value, None, "no walk yet leaves the row out");
        } else {
            assert_eq!(value, Some(0), "{key}");
        }
    }
}

#[test]
fn byte_figures_show_in_whole_mib() {
    let mut sample = empty_sample();
    sample.guest_page_bytes = (96 << 20) + (300 << 10);
    sample.in_flight_bytes = (1 << 20) - 1;
    sample.reuse_pool_bytes = 64 << 20;
    let rows = hud_rows(&sample);
    assert_eq!(rows[row("Mem guest pages MiB")], Some(96));
    assert_eq!(rows[row("Mem in flight MiB")], Some(0));
    assert_eq!(rows[row("Mem reuse pool MiB")], Some(64));
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
fn a_walk_that_found_under_a_mib_shows_zero_and_no_walk_shows_nothing() {
    let mut sample = empty_sample();
    let largest = row("Mem 32-bit largest MiB");
    assert_eq!(hud_rows(&sample)[largest], None, "no walk yet");
    sample.largest_free_word = largest_free_word(0);
    assert_eq!(hud_rows(&sample)[largest], Some(0), "a walked 0 MiB");
    sample.largest_free_word = largest_free_word(512);
    assert_eq!(hud_rows(&sample)[largest], Some(512));
    sample.largest_free_word = largest_free_word(u64::MAX);
    assert_eq!(
        hud_rows(&sample)[largest],
        Some(4096),
        "saturates, then clamps"
    );
}
