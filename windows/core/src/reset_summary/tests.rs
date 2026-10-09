//! Unit tests for the `Reset` log line.
//!
//! The line is what a user's log carries about a resolution change or a
//! windowed/fullscreen switch, so these pin every field it names: the old and
//! new sizes, the format by name (and by code when it has none), the buffer
//! count, the window mode with its refresh rate, the swap effect, the
//! interval, the auto depth-stencil and the outcome the device supplies. The
//! comparison that picks a success's level is pinned too: a windowed drag
//! step that repeats what the device holds changes nothing, while a vsync
//! toggle, a window-mode switch and a new fullscreen mode do.

use mtld3d_types::{
    D3DFMT_D24S8, D3DFMT_X8R8G8B8, D3DPRESENT_INTERVAL_IMMEDIATE, D3DPRESENT_INTERVAL_ONE,
    D3DPRESENT_PARAMETERS, D3DSWAPEFFECT_DISCARD, D3DSWAPEFFECT_FLIP, D3DSWAPEFFECT_FLIPEX,
    D3DSWAPEFFECT_OVERLAY,
};

use super::{
    ResetDone, ResetFailed, ResetFailureEffect, ResetMethod, ResetSummary, changes_presentation,
    failure_warns, success_logs_at_info,
};

const fn params(windowed: bool) -> D3DPRESENT_PARAMETERS {
    D3DPRESENT_PARAMETERS {
        back_buffer_width: 1280,
        back_buffer_height: 720,
        back_buffer_format: D3DFMT_X8R8G8B8,
        back_buffer_count: 1,
        multi_sample_type: 0,
        multi_sample_quality: 0,
        swap_effect: D3DSWAPEFFECT_DISCARD,
        device_window: 0,
        windowed: if windowed { 1 } else { 0 },
        enable_auto_depth_stencil: 1,
        auto_depth_stencil_format: D3DFMT_D24S8,
        flags: 0,
        full_screen_refresh_rate_in_hz: 0,
        presentation_interval: D3DPRESENT_INTERVAL_ONE,
    }
}

#[test]
fn a_fullscreen_reset_names_the_mode_and_its_refresh_rate() {
    let mut pp = params(false);
    pp.full_screen_refresh_rate_in_hz = 120;
    let line = ResetSummary::new(
        "IDirect3DDevice9",
        (1728, 1117),
        &pp,
        "ok, back buffer recreated",
    )
    .to_string();
    assert_eq!(
        line,
        "IDirect3DDevice9::Reset 1728x1117 -> 1280x720 X8R8G8B8, 1 back buffer, fullscreen at \
         120 Hz, swap effect DISCARD, interval ONE, auto depth-stencil D24S8: ok, back buffer \
         recreated"
    );
}

#[test]
fn a_fullscreen_reset_without_a_rate_says_the_default_rate() {
    let pp = params(false);
    let line = ResetSummary::new("IDirect3DDevice9", (1280, 720), &pp, "ok").to_string();
    assert!(
        line.contains(", fullscreen at the default refresh rate,"),
        "{line}"
    );
}

#[test]
fn a_windowed_reset_names_no_refresh_rate() {
    let mut pp = params(true);
    pp.full_screen_refresh_rate_in_hz = 60;
    let line = ResetSummary::new("IDirect3DDevice9", (800, 600), &pp, "ok").to_string();
    assert!(line.contains("800x600 -> 1280x720"), "{line}");
    assert!(line.contains(", windowed, "), "{line}");
    assert!(!line.contains("Hz"), "{line}");
}

#[test]
fn the_interface_names_the_device_the_call_came_through() {
    let pp = params(true);
    let line = ResetSummary::new("IDirect3DDevice9Ex", (1280, 720), &pp, "ok").to_string();
    assert!(line.starts_with("IDirect3DDevice9Ex::Reset "), "{line}");
}

#[test]
fn several_buffers_multisample_and_other_effects_are_named() {
    let mut pp = params(true);
    pp.back_buffer_count = 2;
    pp.multi_sample_type = 4;
    pp.multi_sample_quality = 1;
    pp.swap_effect = D3DSWAPEFFECT_FLIP;
    pp.presentation_interval = D3DPRESENT_INTERVAL_IMMEDIATE;
    pp.enable_auto_depth_stencil = 0;
    let line = ResetSummary::new("IDirect3DDevice9", (1280, 720), &pp, "ok").to_string();
    assert_eq!(
        line,
        "IDirect3DDevice9::Reset 1280x720 -> 1280x720 X8R8G8B8, 2 back buffers, multisample \
         type 4 quality 1, windowed, swap effect FLIP, interval IMMEDIATE, no auto \
         depth-stencil: ok"
    );
}

#[test]
fn values_the_line_has_no_name_for_print_their_codes() {
    let mut pp = params(false);
    pp.back_buffer_format = 0x1234;
    pp.swap_effect = 6;
    pp.presentation_interval = 0x10;
    pp.auto_depth_stencil_format = 0;
    let outcome = format_args!(
        "rejected ({}): {}",
        "D3DERR_INVALIDCALL", "invalid present params"
    );
    let line = ResetSummary::new("IDirect3DDevice9", (1280, 720), &pp, outcome).to_string();
    assert!(line.contains(" format 4660, "), "{line}");
    assert!(line.contains("swap effect 6,"), "{line}");
    assert!(line.contains("interval 0x10,"), "{line}");
    assert!(line.contains("auto depth-stencil UNKNOWN:"), "{line}");
    assert!(
        line.ends_with(": rejected (D3DERR_INVALIDCALL): invalid present params"),
        "{line}"
    );
}

#[test]
fn a_reset_that_repeats_the_stored_params_changes_nothing() {
    let pp = params(true);
    assert!(!changes_presentation(&pp, &pp));
}

#[test]
fn a_windowed_size_alone_is_no_change() {
    // The device follows the client area on `WM_SIZE`; a game that resets
    // on each drag step may still name the size it last passed.
    let previous = params(true);
    let mut next = params(true);
    next.back_buffer_width = 1440;
    next.back_buffer_height = 900;
    assert!(!changes_presentation(&previous, &next));
}

#[test]
fn a_vsync_toggle_alone_is_a_change() {
    let previous = params(true);
    let mut next = params(true);
    next.presentation_interval = D3DPRESENT_INTERVAL_IMMEDIATE;
    assert!(changes_presentation(&previous, &next));
}

#[test]
fn a_window_mode_switch_is_a_change() {
    assert!(changes_presentation(&params(true), &params(false)));
    assert!(changes_presentation(&params(false), &params(true)));
}

#[test]
fn a_new_fullscreen_mode_or_refresh_rate_is_a_change() {
    let previous = params(false);
    let mut resized = params(false);
    resized.back_buffer_width = 1920;
    resized.back_buffer_height = 1080;
    assert!(changes_presentation(&previous, &resized));
    let mut faster = params(false);
    faster.full_screen_refresh_rate_in_hz = 120;
    assert!(changes_presentation(&previous, &faster));
}

#[test]
fn every_other_field_counts() {
    let previous = params(true);
    let edits: [fn(&mut D3DPRESENT_PARAMETERS); 9] = [
        |pp| pp.back_buffer_format = 21,
        |pp| pp.back_buffer_count = 2,
        |pp| pp.multi_sample_type = 4,
        |pp| pp.multi_sample_quality = 1,
        |pp| pp.swap_effect = D3DSWAPEFFECT_FLIP,
        |pp| pp.device_window = 0x1234,
        |pp| pp.enable_auto_depth_stencil = 0,
        |pp| pp.auto_depth_stencil_format = 80,
        |pp| pp.flags = 1,
    ];
    for edit in edits {
        let mut next = params(true);
        edit(&mut next);
        assert!(changes_presentation(&previous, &next));
    }
}

#[test]
fn a_plain_success_names_only_the_back_buffer_and_a_recovery() {
    assert_eq!(ResetDone::RESIZED.to_string(), "ok, back buffer recreated");
    assert_eq!(
        ResetDone::empty().to_string(),
        "ok, same size, back buffer kept"
    );
    assert_eq!(
        (ResetDone::RESIZED | ResetDone::RECOVERED).to_string(),
        "ok, back buffer recreated, device recovered"
    );
}

#[test]
fn an_extended_success_says_what_it_kept_detached_and_rebound() {
    assert_eq!(
        ResetDone::EXTENDED.to_string(),
        "ok, state and default-pool resources kept, same size, back buffer kept, targets \
         rebound to the swap chain"
    );
    assert_eq!(
        (ResetDone::EXTENDED
            | ResetDone::RESIZED
            | ResetDone::BACK_BUFFER_DETACHED
            | ResetDone::DEPTH_DETACHED)
            .to_string(),
        "ok, state and default-pool resources kept, back buffer recreated, the held back buffer \
         and depth surface detached, targets rebound to the swap chain"
    );
    assert_eq!(
        (ResetDone::EXTENDED | ResetDone::RESIZED | ResetDone::DEPTH_DETACHED).to_string(),
        "ok, state and default-pool resources kept, back buffer recreated, the held depth \
         surface detached, targets rebound to the swap chain"
    );
    assert_eq!(
        (ResetDone::EXTENDED | ResetDone::FAILURES_ENDED).to_string(),
        "ok, state and default-pool resources kept, same size, back buffer kept, targets \
         rebound to the swap chain, ending a run of failed Resets"
    );
}

#[test]
fn a_failure_says_what_an_extended_device_kept_and_nothing_more_on_a_plain_one() {
    let reason = format_args!("rejected, no usable windowed client area");
    assert_eq!(
        ResetFailed::new(reason, ResetFailureEffect::Unstated).to_string(),
        "rejected, no usable windowed client area"
    );
    assert_eq!(
        ResetFailed::new(reason, ResetFailureEffect::Unchanged).to_string(),
        "rejected, no usable windowed client area; the extended device is unchanged"
    );
    assert_eq!(
        ResetFailed::new(
            format_args!("failed recreating the back buffer (0x88760870)"),
            ResetFailureEffect::StateKeptResetOwed
        )
        .to_string(),
        "failed recreating the back buffer (0x88760870); the extended device keeps its state and \
         needs another Reset"
    );
}

#[test]
fn a_reset_ex_line_names_the_method_it_came_through() {
    let pp = params(false);
    let line = ResetSummary::new("IDirect3DDevice9Ex", (1280, 720), &pp, "ok")
        .via(ResetMethod::ResetEx)
        .to_string();
    assert!(
        line.starts_with("IDirect3DDevice9Ex::ResetEx 1280x720 -> "),
        "{line}"
    );
}

#[test]
fn the_extended_swap_effects_are_named() {
    let mut pp = params(true);
    for (effect, name) in [
        (D3DSWAPEFFECT_FLIPEX, "FLIPEX"),
        (D3DSWAPEFFECT_OVERLAY, "OVERLAY"),
    ] {
        pp.swap_effect = effect;
        let line = ResetSummary::new("IDirect3DDevice9Ex", (1280, 720), &pp, "ok").to_string();
        assert!(line.contains(&format!(", swap effect {name},")), "{line}");
    }
}

#[test]
fn a_recreate_forced_only_by_a_held_surface_stays_at_debug() {
    let held_only = ResetDone::EXTENDED | ResetDone::RESIZED | ResetDone::BACK_BUFFER_DETACHED;
    assert!(
        !success_logs_at_info(false, held_only),
        "a drag step with a held surface"
    );
    assert!(success_logs_at_info(true, held_only), "a real change");
    assert!(
        success_logs_at_info(false, ResetDone::RECOVERED),
        "a recovery"
    );
    assert!(
        success_logs_at_info(false, ResetDone::EXTENDED | ResetDone::FAILURES_ENDED),
        "the end of a run of failures"
    );
    assert!(
        !success_logs_at_info(false, ResetDone::empty()),
        "nothing changed"
    );
}

#[test]
fn a_failure_that_leaves_a_reset_owed_warns_after_quieted_rejections() {
    use ResetFailureEffect::{StateKeptResetOwed, Unchanged, Unstated};
    assert!(
        failure_warns(false, false, &Unchanged),
        "the first rejection"
    );
    assert!(
        !failure_warns(false, true, &Unchanged),
        "a repeated rejection"
    );
    assert!(
        failure_warns(false, true, &StateKeptResetOwed),
        "owing a Reset after rejections"
    );
    assert!(
        !failure_warns(true, true, &StateKeptResetOwed),
        "already owing one"
    );
    assert!(
        failure_warns(false, false, &Unstated),
        "a plain device's first failure"
    );
    assert!(
        !failure_warns(true, false, &Unstated),
        "a plain device's retry"
    );
}
