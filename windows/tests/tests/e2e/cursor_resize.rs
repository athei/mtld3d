//! The D3D cursor across window resizes, minimise and restore.
//!
//! A shown cursor stays shown while the game handles the message dispatch a
//! resize arrived in: a `ShowCursor(FALSE)` from a handler that runs after
//! the `WM_SIZE` inside it is dropped. A hide the game makes once that
//! dispatch has returned, such as the one a mouselook starts with, hides the
//! cursor, and a resize never shows a cursor the game hid. Most tests run the
//! hardware cursor, whose visibility is the thread cursor the device pushes;
//! the software cursor keeps its blank HCURSOR selected either way, so its
//! test reads visibility from `ShowCursor`'s previous state.

use mtld3d_core::cursor::{SIZE_MAXIMIZED, SIZE_MINIMIZED, SIZE_RESTORED};
use mtld3d_tests::{
    Harness, HarnessConfig, WA_ACTIVE, WA_INACTIVE, WM_ACTIVATE, WM_ACTIVATEAPP,
    WM_HARNESS_RESTORE, WM_SIZE, set_window_pos, window_rect,
};
use mtld3d_types::{D3D_OK, D3DFMT_A8R8G8B8, D3DPOOL_SCRATCH};

/// The harness window's client size, which a restore brings back.
const WIDTH: isize = 640;
const HEIGHT: isize = 480;

/// A `WM_SIZE` `lParam`: the client height in the high word, the width in the low.
const fn client_size(width: isize, height: isize) -> isize {
    (height << 16) | width
}

/// A device drawing the hardware cursor.
fn hardware_cursor_harness() -> Harness {
    Harness::with_config("cursor.software=false")
}

/// Set a cursor on `h` and show it; the HCURSOR the device realized for it.
fn show_a_cursor(h: &Harness) -> usize {
    let bitmap = h.create_offscreen_plain_surface(32, 32, D3DFMT_A8R8G8B8, D3DPOOL_SCRATCH);
    assert_eq!(h.set_cursor_properties_hr(0, 0, &bitmap), D3D_OK);
    assert_eq!(h.show_cursor(true), 0, "the cursor starts hidden");
    let ours = h.thread_cursor();
    assert_ne!(ours, 0, "ShowCursor(TRUE) realizes the device's HCURSOR");
    ours
}

/// Shrink, minimise and deactivate the window, then restore it the way Wine does.
///
/// The restore runs inside one outer message (`WM_HARNESS_RESTORE`), which
/// resizes the window back for real, so the `WM_SIZE` comes from the default
/// procedure's `WM_WINDOWPOSCHANGED`, and then sends `WM_ACTIVATEAPP` and
/// `WM_ACTIVATE`. The window's procedure hides `h`'s cursor from its handler
/// of `hide_on` during the restore alone, and the script then stays set.
fn minimise_and_restore(h: &Harness, hide_on: u32) {
    h.hide_cursor_from_window_procedure_on(0);
    let rect = window_rect(h.hwnd());
    set_window_pos(h.hwnd(), rect.left, rect.top, 320, 240);
    h.send_window_message(WM_SIZE, SIZE_MINIMIZED, 0);
    h.send_window_message(WM_ACTIVATE, WA_INACTIVE, 0);
    h.hide_cursor_from_window_procedure_on(hide_on);
    h.send_window_message(WM_HARNESS_RESTORE, 0, client_size(WIDTH, HEIGHT));
}

/// A hide the game makes outside any window message, the way a mouselook starts.
fn assert_mouselook_hides(h: &Harness, ours: usize, after: &str) {
    assert_eq!(
        h.show_cursor(false),
        1,
        "{after}: the hide reports the cursor previously shown"
    );
    assert_eq!(
        h.thread_cursor(),
        0,
        "{after}: a hide outside the resize's dispatch hides the cursor"
    );
    assert_eq!(
        h.show_cursor(true),
        0,
        "{after}: the hide took, so the show reports the cursor previously hidden"
    );
    assert_eq!(
        h.thread_cursor(),
        ours,
        "{after}: the show realizes the cursor again"
    );
}

#[test]
fn a_mouselook_after_a_restore_from_minimised_hides_the_cursor() {
    let h = hardware_cursor_harness();
    let ours = show_a_cursor(&h);
    // Answer the synthesized activation in the test window's own procedure.
    h.hide_cursor_from_window_procedure_on(0);

    // Minimise, deactivate, restore and activate, each its own dispatch.
    h.send_window_message(WM_SIZE, SIZE_MINIMIZED, 0);
    h.send_window_message(WM_ACTIVATE, WA_INACTIVE, 0);
    h.send_window_message(WM_SIZE, SIZE_RESTORED, client_size(WIDTH, HEIGHT));
    h.send_window_message(WM_ACTIVATE, WA_ACTIVE, 0);
    assert_eq!(
        h.thread_cursor(),
        ours,
        "the restore keeps the cursor shown"
    );

    assert_mouselook_hides(&h, ours, "the first mouselook after a restore");
    assert_mouselook_hides(&h, ours, "the second mouselook after a restore");
}

#[test]
fn a_hide_inside_the_restore_dispatch_is_dropped_and_the_next_one_is_not() {
    let h = hardware_cursor_harness();
    let ours = show_a_cursor(&h);
    // Wine's restore sends `WM_SIZE`, `WM_ACTIVATEAPP` and `WM_ACTIVATE` from
    // inside one `WM_SYSCOMMAND`; a hide from any of their handlers is part of
    // the restore.
    for (hide_on, name) in [
        (WM_SIZE, "WM_SIZE"),
        (WM_ACTIVATEAPP, "WM_ACTIVATEAPP"),
        (WM_ACTIVATE, "WM_ACTIVATE"),
    ] {
        minimise_and_restore(&h, hide_on);
        assert_eq!(
            h.window_procedure_hides(),
            (1, 1),
            "the {name} handler hid the cursor once, reported previously shown"
        );
        assert_eq!(
            h.thread_cursor(),
            ours,
            "a hide from the {name} handler inside the restore leaves the cursor shown"
        );
        h.hide_cursor_from_window_procedure_on(0);
        assert_mouselook_hides(&h, ours, &format!("a restore whose {name} handler hid"));
    }
}

#[test]
fn a_hide_from_the_games_resize_handler_is_dropped_and_a_later_one_is_not() {
    let h = hardware_cursor_harness();
    let ours = show_a_cursor(&h);
    h.hide_cursor_from_window_procedure_on(WM_SIZE);
    h.send_window_message(WM_SIZE, SIZE_RESTORED, client_size(WIDTH, HEIGHT));
    assert_eq!(
        h.window_procedure_hides(),
        (1, 1),
        "the WM_SIZE handler hid the cursor once, reported previously shown"
    );
    assert_eq!(
        h.thread_cursor(),
        ours,
        "a hide from the game's own WM_SIZE handler leaves the cursor shown"
    );
    assert_eq!(
        h.show_cursor(true),
        1,
        "the dropped hide left the cursor shown"
    );
    h.hide_cursor_from_window_procedure_on(0);
    assert_mouselook_hides(&h, ours, "a WM_SIZE whose handler hid");
}

#[test]
fn a_mouselook_after_a_resize_or_a_maximise_hides_the_cursor() {
    let h = hardware_cursor_harness();
    let ours = show_a_cursor(&h);
    h.send_window_message(WM_SIZE, SIZE_RESTORED, client_size(600, 400));
    assert_mouselook_hides(&h, ours, "a resize");
    h.send_window_message(WM_SIZE, SIZE_MAXIMIZED, client_size(WIDTH, HEIGHT));
    assert_mouselook_hides(&h, ours, "a maximise");
}

#[test]
fn a_resize_never_shows_a_cursor_the_game_hid() {
    let h = hardware_cursor_harness();
    show_a_cursor(&h);
    assert_eq!(h.show_cursor(false), 1);
    assert_eq!(h.thread_cursor(), 0, "the hide pushes the null cursor");
    for size_type in [SIZE_MINIMIZED, SIZE_RESTORED, SIZE_MAXIMIZED] {
        h.send_window_message(WM_SIZE, size_type, client_size(WIDTH, HEIGHT));
        assert_eq!(
            h.thread_cursor(),
            0,
            "a WM_SIZE of type {size_type} leaves the hidden cursor hidden"
        );
    }
    assert_eq!(
        h.show_cursor(true),
        0,
        "the cursor stayed hidden across the resizes"
    );
}

/// A hide outside any message, read from what the next `ShowCursor(TRUE)` reports.
fn assert_hide_takes(h: &Harness, after: &str) {
    assert_eq!(
        h.show_cursor(false),
        1,
        "{after}: the hide reports the cursor previously shown"
    );
    assert_eq!(
        h.show_cursor(true),
        0,
        "{after}: the hide took, so the show reports the cursor previously hidden"
    );
}

#[test]
fn the_software_cursor_keeps_its_blank_and_honours_the_hide_after_a_restore() {
    let h = Harness::with_config("cursor.software=true");
    let blank = show_a_cursor(&h);

    minimise_and_restore(&h, WM_ACTIVATEAPP);
    assert_eq!(
        h.window_procedure_hides(),
        (1, 1),
        "the WM_ACTIVATEAPP handler hid the cursor once, reported previously shown"
    );
    h.hide_cursor_from_window_procedure_on(0);
    // The mouselook's hide reporting the cursor shown is what says the hide
    // inside the restore was dropped.
    assert_hide_takes(&h, "the first mouselook after a restore");

    assert_eq!(h.show_cursor(false), 1);
    for size_type in [SIZE_MINIMIZED, SIZE_RESTORED] {
        h.send_window_message(WM_SIZE, size_type, client_size(WIDTH, HEIGHT));
        assert_eq!(
            h.thread_cursor(),
            blank,
            "a WM_SIZE of type {size_type} keeps the blank selected while the cursor is hidden"
        );
    }
    assert_eq!(
        h.show_cursor(true),
        0,
        "the cursor stayed hidden across the resizes"
    );
}

#[test]
fn a_second_device_on_the_window_leaves_the_restore_to_the_first() {
    let first = hardware_cursor_harness();
    let second = Harness::create(&HarnessConfig {
        device_window: first.hwnd(),
        config_entries: "cursor.software=false",
        ..HarnessConfig::default()
    });
    show_a_cursor(&first);
    show_a_cursor(&second);

    // The window's messages go to the first device, so its restore is the
    // one that holds a hide its handlers make.
    minimise_and_restore(&first, WM_SIZE);
    assert_eq!(
        first.window_procedure_hides(),
        (1, 1),
        "the WM_SIZE handler hid the first device's cursor once"
    );
    first.hide_cursor_from_window_procedure_on(0);
    // The first device's hide reporting the cursor shown is what says its
    // hide inside the restore was dropped.
    assert_hide_takes(&first, "the first device after the restore");
    assert_hide_takes(&second, "the second device after the restore");
}
