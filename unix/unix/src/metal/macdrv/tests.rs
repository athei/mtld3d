//! Unit tests for the present-throttle duration and the layer-mode decision.
//!
//! `min_present_duration` folds the guest's vsync request and the user's
//! `present.maxFps` ceiling into the minimum seconds handed to
//! `presentDrawable:afterMinimumDuration:`. Bit-exact assertions pin every
//! combination of the two inputs: the lower of the two rates wins, an
//! unknown panel rate still honours the user cap, and only IMMEDIATE with
//! no cap resolves to `0.0` for an unthrottled free run.
//!
//! `layer_mode_for` and `layer_mode_change` decide which `CAMetalLayer`
//! configuration a screen asks for and whether the applied one has to be
//! replaced. `min_present_duration_change` and `backing_scale_change` are
//! their two neighbours in the same reconciliation. All four are the whole of
//! the display-follow decision, so the tests below pin both directions of a
//! display change, the user's off switch, the degenerate ceilings, and the
//! case that must *not* reconfigure. A Reset that flips the guest's
//! `PresentationInterval` reaches the throttle through the same
//! reconciliation with the panel unchanged, so both directions of that flip
//! and the user cap that holds through it are pinned there too.
//!
//! `screen_params_filter_step` decides what one attempt to take
//! `NSApplicationDidChangeScreenParametersNotification` over from Wine does.
//! The tests pin all three outcomes, and in particular that an attempt made
//! before `NSApp` has a delegate resolves to a retry rather than to a filter
//! marked installed, which is what keeps a process whose first `CreateDevice`
//! beats Wine's application delegate from running unfiltered for its lifetime.
//!
//! `MetalViewPark` is the bookkeeping behind keeping a retired device's metal
//! view for the next device on its window. The tests pin that a window's view
//! is kept and taken back by that window alone, that a second view for the
//! same window displaces the first, that views of several windows are kept
//! beside each other up to the park's size, and that a full park displaces
//! its oldest view.
//!
//! The other half of that path, what a device's teardown retires and what
//! it leaves alone for the devices still attached, is the attachment
//! registry's, and lives in `attachment/tests.rs`.
//!
//! `MacdrvFuncs::load` is pinned for the one case a test process can
//! produce: no `macdrv_functions` table in the symbol space, which is every
//! process that is not Wine's. The table is the only door, so the load
//! resolves nothing and the attach above it fails rather than reading a
//! window record through a layout the process does not have.
//!
//! What that load does with a table it did get is pinned on tables built
//! here: `first_null_required_entry` names the first entry the layer calls
//! that a table leaves null, which is what turns a null entry into a warn
//! and a clean failure instead of a transmuted null called as a function,
//! and it passes a table whose only null is `macdrv_get_cocoa_window`,
//! which the layer treats as optional. The layout the table is read
//! through is pinned beside them, field offset by field offset, since
//! nothing at run time compares it against the fork's.

use core::ffi::c_void;

use super::{
    KEPT_METAL_VIEWS, LayerMode, MacdrvFuncs, MacdrvFunctionsTable, MetalViewPark, PresentPacing,
    ScreenParamsFilterStep, backing_scale_change, backing_scale_from, first_null_required_entry,
    layer_mode_change, layer_mode_for, min_present_duration, min_present_duration_change,
    pack_pacing, screen_params_filter_step, unpack_pacing,
};

#[test]
fn vsync_only_paces_at_panel_rate() {
    let pacing = PresentPacing {
        vsync_requested: true,
        max_fps: 0,
    };
    let d = min_present_duration(120.0, &pacing);
    assert_eq!(d.to_bits(), (1.0_f64 / 120.0).to_bits());
}

#[test]
fn cap_only_bounds_the_free_run() {
    let pacing = PresentPacing {
        vsync_requested: false,
        max_fps: 30,
    };
    let d = min_present_duration(120.0, &pacing);
    assert_eq!(d.to_bits(), (1.0_f64 / 30.0).to_bits());
}

#[test]
fn lower_rate_wins_when_both_active() {
    let below_panel = PresentPacing {
        vsync_requested: true,
        max_fps: 60,
    };
    let d = min_present_duration(120.0, &below_panel);
    assert_eq!(d.to_bits(), (1.0_f64 / 60.0).to_bits());

    let above_panel = PresentPacing {
        vsync_requested: true,
        max_fps: 240,
    };
    let d = min_present_duration(120.0, &above_panel);
    assert_eq!(d.to_bits(), (1.0_f64 / 120.0).to_bits());
}

#[test]
fn immediate_and_uncapped_free_runs() {
    let pacing = PresentPacing {
        vsync_requested: false,
        max_fps: 0,
    };
    let d = min_present_duration(120.0, &pacing);
    assert_eq!(d.to_bits(), 0.0_f64.to_bits());
}

#[test]
fn unknown_panel_rate_still_honours_the_cap() {
    let pacing = PresentPacing {
        vsync_requested: true,
        max_fps: 60,
    };
    let d = min_present_duration(0.0, &pacing);
    assert_eq!(d.to_bits(), (1.0_f64 / 60.0).to_bits());
}

#[test]
fn an_edr_panel_asks_for_the_hdr_layer() {
    assert_eq!(layer_mode_for(16.0, true), LayerMode::Hdr);
    assert_eq!(layer_mode_for(1.5, true), LayerMode::Hdr);
}

#[test]
fn an_sdr_panel_asks_for_the_sdr_layer() {
    assert_eq!(layer_mode_for(1.0, true), LayerMode::Sdr);
    assert_eq!(layer_mode_for(0.0, true), LayerMode::Sdr);
}

#[test]
fn the_user_switch_forces_sdr_on_any_panel() {
    assert_eq!(layer_mode_for(16.0, false), LayerMode::Sdr);
}

#[test]
fn degenerate_ceilings_are_sdr() {
    assert_eq!(layer_mode_for(f64::NAN, true), LayerMode::Sdr);
    assert_eq!(layer_mode_for(f64::INFINITY, true), LayerMode::Sdr);
}

#[test]
fn staying_on_one_display_never_reconfigures() {
    assert_eq!(layer_mode_change(LayerMode::Hdr, 16.0, true), None);
    assert_eq!(layer_mode_change(LayerMode::Sdr, 1.0, true), None);
    // The user switch is off, so an EDR panel is already correctly on the
    // SDR layer and a poll must leave it there.
    assert_eq!(layer_mode_change(LayerMode::Sdr, 16.0, false), None);
}

#[test]
fn moving_onto_an_sdr_display_asks_for_the_sdr_layer() {
    assert_eq!(
        layer_mode_change(LayerMode::Hdr, 1.0, true),
        Some(LayerMode::Sdr)
    );
}

#[test]
fn moving_onto_an_edr_display_asks_for_the_hdr_layer() {
    assert_eq!(
        layer_mode_change(LayerMode::Sdr, 16.0, true),
        Some(LayerMode::Hdr)
    );
}

#[test]
fn a_brightness_change_alone_does_not_reconfigure() {
    // The decision reads the static panel ceiling, which brightness and
    // thermal state do not move; the live headroom they do move drives the
    // present shader's peak instead. Both readings of the same XDR panel
    // therefore keep the HDR layer.
    assert_eq!(layer_mode_change(LayerMode::Hdr, 16.0, true), None);
    assert_eq!(layer_mode_change(LayerMode::Hdr, 16.0, true), None);
}

#[test]
fn pacing_survives_the_round_trip_through_one_word() {
    for (vsync_requested, max_fps) in [(true, 0), (false, 0), (true, 60), (false, 240)] {
        let packed = pack_pacing(&PresentPacing {
            vsync_requested,
            max_fps,
        });
        let back = unpack_pacing(packed);
        assert_eq!(back.vsync_requested, vsync_requested);
        assert_eq!(back.max_fps, max_fps);
    }
}

#[test]
fn the_widest_cap_survives_the_round_trip() {
    let packed = pack_pacing(&PresentPacing {
        vsync_requested: true,
        max_fps: u32::MAX,
    });
    let back = unpack_pacing(packed);
    assert!(back.vsync_requested);
    assert_eq!(back.max_fps, u32::MAX);
}

#[test]
fn staying_on_one_panel_never_rederives_the_throttle() {
    let pacing = PresentPacing {
        vsync_requested: true,
        max_fps: 0,
    };
    let applied = min_present_duration(120.0, &pacing);
    assert_eq!(min_present_duration_change(applied, 120.0, &pacing), None);
}

#[test]
fn moving_onto_a_slower_panel_lengthens_the_throttle() {
    let pacing = PresentPacing {
        vsync_requested: true,
        max_fps: 0,
    };
    let applied = min_present_duration(120.0, &pacing);
    let changed = min_present_duration_change(applied, 60.0, &pacing).expect("panel rate moved");
    assert_eq!(changed.to_bits(), (1.0_f64 / 60.0).to_bits());
}

#[test]
fn moving_onto_a_faster_panel_shortens_the_throttle() {
    let pacing = PresentPacing {
        vsync_requested: true,
        max_fps: 0,
    };
    let applied = min_present_duration(60.0, &pacing);
    let changed = min_present_duration_change(applied, 120.0, &pacing).expect("panel rate moved");
    assert_eq!(changed.to_bits(), (1.0_f64 / 120.0).to_bits());
}

#[test]
fn a_user_cap_below_both_panels_holds_the_throttle_still() {
    // The cap is the lower rate on either display, so the duration the
    // present site uses does not move and nothing is rewritten or logged.
    let pacing = PresentPacing {
        vsync_requested: true,
        max_fps: 30,
    };
    let applied = min_present_duration(120.0, &pacing);
    assert_eq!(min_present_duration_change(applied, 60.0, &pacing), None);
}

#[test]
fn a_free_running_session_stays_unthrottled_on_any_panel() {
    let pacing = PresentPacing {
        vsync_requested: false,
        max_fps: 0,
    };
    let applied = min_present_duration(120.0, &pacing);
    assert_eq!(applied.to_bits(), 0.0_f64.to_bits());
    assert_eq!(min_present_duration_change(applied, 60.0, &pacing), None);
}

#[test]
fn a_reset_to_immediate_lifts_the_throttle() {
    let vsync = PresentPacing {
        vsync_requested: true,
        max_fps: 0,
    };
    let applied = min_present_duration(60.0, &vsync);
    assert_eq!(applied.to_bits(), (1.0_f64 / 60.0).to_bits());
    let immediate = PresentPacing {
        vsync_requested: false,
        max_fps: 0,
    };
    let changed = min_present_duration_change(applied, 60.0, &immediate).expect("the pacing moved");
    assert_eq!(changed.to_bits(), 0.0_f64.to_bits());
}

#[test]
fn a_reset_back_to_vsync_restores_the_panel_throttle() {
    let immediate = PresentPacing {
        vsync_requested: false,
        max_fps: 0,
    };
    let applied = min_present_duration(60.0, &immediate);
    assert_eq!(applied.to_bits(), 0.0_f64.to_bits());
    let vsync = PresentPacing {
        vsync_requested: true,
        max_fps: 0,
    };
    let changed = min_present_duration_change(applied, 60.0, &vsync).expect("the pacing moved");
    assert_eq!(changed.to_bits(), (1.0_f64 / 60.0).to_bits());
}

#[test]
fn a_user_cap_holds_the_throttle_through_a_vsync_flip() {
    // The cap is the lower rate whichever way the guest's interval goes, so
    // the duration the present site uses does not move in either direction.
    let capped_vsync = PresentPacing {
        vsync_requested: true,
        max_fps: 30,
    };
    let capped_immediate = PresentPacing {
        vsync_requested: false,
        max_fps: 30,
    };
    let applied = min_present_duration(60.0, &capped_vsync);
    assert_eq!(applied.to_bits(), (1.0_f64 / 30.0).to_bits());
    assert_eq!(
        min_present_duration_change(applied, 60.0, &capped_immediate),
        None
    );
    let applied = min_present_duration(60.0, &capped_immediate);
    assert_eq!(
        min_present_duration_change(applied, 60.0, &capped_vsync),
        None
    );
}

#[test]
fn backing_scale_rounds_and_clamps_into_the_hcursor_range() {
    assert_eq!(backing_scale_from(1.0), 1);
    assert_eq!(backing_scale_from(2.0), 2);
    assert_eq!(backing_scale_from(1.4), 1);
    assert_eq!(backing_scale_from(1.5), 2);
    // No screen at all, and a pathological reading, both land on identity
    // rather than a factor the HCURSOR builder would reject.
    assert_eq!(backing_scale_from(0.0), 1);
    assert_eq!(backing_scale_from(-4.0), 1);
    assert_eq!(backing_scale_from(f64::NAN), 1);
    assert_eq!(backing_scale_from(99.0), 8);
}

#[test]
fn staying_on_one_display_never_republishes_the_scale() {
    assert_eq!(backing_scale_change(2, 2.0), None);
    assert_eq!(backing_scale_change(1, 1.0), None);
}

#[test]
fn moving_between_displays_of_different_scale_republishes() {
    assert_eq!(backing_scale_change(2, 1.0), Some(1));
    assert_eq!(backing_scale_change(1, 2.0), Some(2));
}

#[test]
fn an_attempt_without_a_delegate_installs_nothing_and_retries() {
    assert_eq!(
        screen_params_filter_step(false, false),
        ScreenParamsFilterStep::AwaitDelegate
    );
}

#[test]
fn the_first_attempt_that_finds_a_delegate_takes_the_notification_over() {
    assert_eq!(
        screen_params_filter_step(false, true),
        ScreenParamsFilterStep::TakeOver
    );
}

#[test]
fn an_installed_filter_is_never_installed_twice() {
    assert_eq!(
        screen_params_filter_step(true, true),
        ScreenParamsFilterStep::AlreadyOurs
    );
    // A delegate that went away after the take-over does not reopen the
    // decision: the observer is registered with the center, not with it.
    assert_eq!(
        screen_params_filter_step(true, false),
        ScreenParamsFilterStep::AlreadyOurs
    );
}

#[test]
fn an_empty_park_keeps_the_first_view_and_displaces_nothing() {
    let mut park = MetalViewPark::new();
    assert_eq!(park.park(0x40, 0x1000, 0x2000), None);
    assert_eq!(park.take_for(0x40), Some((0x1000, 0x2000)));
    assert_eq!(
        park.take_for(0x40),
        None,
        "taken once; the slot is empty again"
    );
}

#[test]
fn a_second_view_for_the_same_window_displaces_the_first() {
    let mut park = MetalViewPark::new();
    assert_eq!(park.park(0x40, 0x1000, 0x2000), None);
    assert_eq!(park.park(0x40, 0x3000, 0x4000), Some(0x1000));
    assert_eq!(park.take_for(0x40), Some((0x3000, 0x4000)));
}

#[test]
fn parking_the_kept_view_again_displaces_nothing() {
    let mut park = MetalViewPark::new();
    assert_eq!(park.park(0x40, 0x1000, 0x2000), None);
    assert_eq!(park.park(0x40, 0x1000, 0x2000), None);
    assert_eq!(park.take_for(0x40), Some((0x1000, 0x2000)));
}

/// One kept view per window, as many as the park holds.
const KEPT: [(u64, usize); KEPT_METAL_VIEWS] = [(1, 0x1000), (2, 0x2000), (3, 0x3000), (4, 0x4000)];

#[test]
fn views_of_other_windows_are_kept_beside_each_other() {
    let mut park = MetalViewPark::new();
    for (window, view) in KEPT {
        assert_eq!(park.park(window, view, 0x20), None);
    }
    assert_eq!(
        park.take_for(0x99),
        None,
        "no view for a window that had none"
    );
    for (window, view) in KEPT.iter().rev() {
        assert_eq!(
            park.take_for(*window),
            Some((*view, 0x20)),
            "still kept for a device that comes back"
        );
    }
}

#[test]
fn a_full_park_displaces_the_oldest_view() {
    let mut park = MetalViewPark::new();
    for (window, view) in KEPT {
        assert_eq!(park.park(window, view, 0x20), None);
    }
    assert_eq!(park.park(0x99, 0x9000, 0x20), Some(0x1000));
    assert_eq!(park.take_for(1), None, "the oldest went");
    assert_eq!(
        park.take_for(2),
        Some((0x2000, 0x20)),
        "the next oldest stays"
    );
    assert_eq!(park.take_for(0x99), Some((0x9000, 0x20)));
}

#[test]
fn a_process_without_the_macdrv_table_loads_nothing() {
    assert!(
        MacdrvFuncs::load().is_none(),
        "a process with no macdrv_functions table resolves no macdrv entry point",
    );
}

/// A table with every entry the layer declares filled, as the fork publishes one.
///
/// The addresses are not functions and are never called: what reads them
/// here reads them as pointers, and only asks whether one is null.
fn filled_table() -> MacdrvFunctionsTable {
    let entry = core::ptr::NonNull::<c_void>::dangling().as_ptr();
    MacdrvFunctionsTable {
        macdrv_init_display_devices: entry,
        get_win_data: entry,
        release_win_data: entry,
        macdrv_get_cocoa_window: entry,
        macdrv_create_metal_device: entry,
        macdrv_release_metal_device: entry,
        macdrv_view_create_metal_view: entry,
        macdrv_view_get_metal_layer: entry,
        macdrv_view_release_metal_view: entry,
        on_main_thread: entry,
    }
}

#[test]
fn a_table_the_fork_filled_is_taken_as_it_is() {
    assert_eq!(first_null_required_entry(&filled_table()), None);
}

#[test]
fn the_first_null_entry_the_layer_calls_is_named() {
    let mut table = filled_table();
    table.get_win_data = core::ptr::null_mut();
    assert_eq!(first_null_required_entry(&table), Some("get_win_data"));

    let mut table = filled_table();
    table.release_win_data = core::ptr::null_mut();
    assert_eq!(first_null_required_entry(&table), Some("release_win_data"));

    let mut table = filled_table();
    table.macdrv_view_create_metal_view = core::ptr::null_mut();
    assert_eq!(
        first_null_required_entry(&table),
        Some("macdrv_view_create_metal_view")
    );

    let mut table = filled_table();
    table.macdrv_view_get_metal_layer = core::ptr::null_mut();
    assert_eq!(
        first_null_required_entry(&table),
        Some("macdrv_view_get_metal_layer")
    );

    // The release path holds this entry as a typed pointer, so a table
    // missing it is refused here rather than at a release that has a view
    // in hand and nothing to release it with.
    let mut table = filled_table();
    table.macdrv_view_release_metal_view = core::ptr::null_mut();
    assert_eq!(
        first_null_required_entry(&table),
        Some("macdrv_view_release_metal_view")
    );
}

#[test]
fn a_null_cocoa_window_entry_leaves_the_table_usable() {
    let mut table = filled_table();
    table.macdrv_get_cocoa_window = core::ptr::null_mut();
    assert_eq!(
        first_null_required_entry(&table),
        None,
        "the entry is optional: without it a kept view is not reused",
    );
    // The entries that are not optional are unaffected by it.
    table.get_win_data = core::ptr::null_mut();
    assert_eq!(first_null_required_entry(&table), Some("get_win_data"));
}

#[test]
fn the_table_is_read_through_the_layout_the_fork_publishes() {
    // Wine's `struct macdrv_functions_t` is 24 function pointers, pinned
    // there by a `C_ASSERT` that its size is 192 bytes. The layer declares
    // the first ten of them and reads that prefix alone, so every entry it
    // reads has to sit at the offset the fork's field order gives it, and
    // the prefix has to be the ten pointers and nothing else. The table
    // carries no version or size word, so nothing at run time can check
    // this; the assertions below are the check, and they fail here rather
    // than in a game if the declaration drifts.
    assert_eq!(
        core::mem::offset_of!(MacdrvFunctionsTable, macdrv_init_display_devices),
        0
    );
    assert_eq!(core::mem::offset_of!(MacdrvFunctionsTable, get_win_data), 8);
    assert_eq!(
        core::mem::offset_of!(MacdrvFunctionsTable, release_win_data),
        16
    );
    assert_eq!(
        core::mem::offset_of!(MacdrvFunctionsTable, macdrv_get_cocoa_window),
        24
    );
    assert_eq!(
        core::mem::offset_of!(MacdrvFunctionsTable, macdrv_create_metal_device),
        32
    );
    assert_eq!(
        core::mem::offset_of!(MacdrvFunctionsTable, macdrv_release_metal_device),
        40
    );
    assert_eq!(
        core::mem::offset_of!(MacdrvFunctionsTable, macdrv_view_create_metal_view),
        48
    );
    assert_eq!(
        core::mem::offset_of!(MacdrvFunctionsTable, macdrv_view_get_metal_layer),
        56
    );
    assert_eq!(
        core::mem::offset_of!(MacdrvFunctionsTable, macdrv_view_release_metal_view),
        64
    );
    assert_eq!(
        core::mem::offset_of!(MacdrvFunctionsTable, on_main_thread),
        72
    );
    assert_eq!(size_of::<MacdrvFunctionsTable>(), 80);
}
