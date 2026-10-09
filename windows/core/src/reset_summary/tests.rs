//! Unit tests for the `Reset` log line.
//!
//! The line is what a user's log carries about a resolution change or a
//! windowed/fullscreen switch, so these pin every field it names: the old and
//! new sizes, the format by name (and by code when it has none), the buffer
//! count, the window mode with its refresh rate, the swap effect, the
//! interval, the auto depth-stencil and the outcome the device supplies.

use mtld3d_types::{
    D3DFMT_D24S8, D3DFMT_X8R8G8B8, D3DPRESENT_INTERVAL_IMMEDIATE, D3DPRESENT_INTERVAL_ONE,
    D3DPRESENT_PARAMETERS, D3DSWAPEFFECT_DISCARD, D3DSWAPEFFECT_FLIP,
};

use super::ResetSummary;

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
    pp.swap_effect = 5;
    pp.presentation_interval = 0x10;
    pp.auto_depth_stencil_format = 0;
    let outcome = format_args!(
        "rejected ({}): {}",
        "D3DERR_INVALIDCALL", "invalid present params"
    );
    let line = ResetSummary::new("IDirect3DDevice9", (1280, 720), &pp, outcome).to_string();
    assert!(line.contains(" format 4660, "), "{line}");
    assert!(line.contains("swap effect 5,"), "{line}");
    assert!(line.contains("interval 0x10,"), "{line}");
    assert!(line.contains("auto depth-stencil UNKNOWN:"), "{line}");
    assert!(
        line.ends_with(": rejected (D3DERR_INVALIDCALL): invalid present params"),
        "{line}"
    );
}
