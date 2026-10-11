//! Wine's client surface for a device's window stays shown and sized after the attach.
//!
//! The layer presents through `CAMetalLayer`'s own `nextDrawable`, not
//! winemac's override, which had Wine show the client surface the metal view
//! sits in, and size it to the window's client area, on every present. Some
//! Wines hide that surface's cocoa view whenever they flush the window
//! surface and leave its frame alone when the window resizes, counting on the
//! override for both. So a device whose client view Wine hid, or whose
//! window's client area moved, has Wine show its surface again, sized to the
//! client area, at its next `Present`.
//!
//! The Wine the suite runs on neither hides the view on a flush nor skips the
//! surface on a resize, so each test makes Wine do what it can here and reads
//! the outcome from the layer's log, in a process of its own. A hidden client
//! view is made by giving the window an OpenGL pixel format: Wine shows the
//! GL client surface it creates for that, hiding the device's. The resize is
//! a `SetWindowPos` to another size, after which this Wine has already sized
//! the surface itself; what the test pins there is that the device asks Wine
//! to present it, which is what sizes it on a Wine that does not. On a
//! machine that offers no OpenGL pixel format, such as a CI runner whose
//! virtual GPU has no OpenGL, Wine cannot be made to hide the view this way,
//! so the two workloads that need it end with a note there, which the run
//! repeats.

use mtld3d_tests::{Harness, set_opengl_pixel_format, set_window_pos};

use super::device::{
    await_logged_lines, logged_lines, run_in_private_log_child, running_as, spawn_private_log_child,
};

/// The line a workload child prints when this machine offers no OpenGL pixel format.
///
/// Wine cannot be made to hide the client view without one, which is the
/// case on a CI runner whose virtual GPU has no OpenGL. The parent repeats
/// the line, so a run that could not test the hidden view says so.
const NO_OPENGL_NOTE: &str = "[e2e] note: no OpenGL pixel format on this machine";

/// The name the hidden-view workload child runs under.
const HIDDEN_CHILD_NAME: &str = "client-view-hidden.exe";

/// The name the resize workload child runs under.
const RESIZE_CHILD_NAME: &str = "client-area-resized.exe";

/// The unix side's debug records: the client view watch and every surface shown on request.
const CLIENT_SURFACE_LOG_FILTER: &str = "warn,mtld3d::unix=debug";

const RED: u32 = 0xFFFF_0000;

#[test]
fn a_client_view_wine_hides_is_shown_again_at_the_next_present() {
    if running_as(HIDDEN_CHILD_NAME) {
        hidden_view_workload();
        return;
    }
    run_opengl_child(
        HIDDEN_CHILD_NAME,
        "client_surface::a_client_view_wine_hides_is_shown_again_at_the_next_present",
    );
}

#[test]
fn a_resized_window_has_its_client_surface_presented_again() {
    if running_as(RESIZE_CHILD_NAME) {
        resized_window_workload();
        return;
    }
    run_in_private_log_child(
        RESIZE_CHILD_NAME,
        "client_surface::a_resized_window_has_its_client_surface_presented_again",
        CLIENT_SURFACE_LOG_FILTER,
    );
}

/// The name the teardown workload child runs under.
const TEARDOWN_CHILD_NAME: &str = "client-view-unwatched.exe";

#[test]
fn a_released_device_no_longer_watches_its_client_view() {
    if running_as(TEARDOWN_CHILD_NAME) {
        released_device_workload();
        return;
    }
    run_opengl_child(
        TEARDOWN_CHILD_NAME,
        "client_surface::a_released_device_no_longer_watches_its_client_view",
    );
}

/// Run a workload that hides the client view through OpenGL, repeating its note when it could not.
fn run_opengl_child(child_name: &str, test: &str) {
    let (output, dir) = spawn_private_log_child(child_name, test, CLIENT_SURFACE_LOG_FILTER, "");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "workload child {child_name} failed: {stderr}"
    );
    if let Some(note) = stderr.lines().find(|line| line.contains(NO_OPENGL_NOTE)) {
        eprintln!("{note}");
    }
    std::fs::remove_dir_all(&dir).expect("remove the child's private directory");
}

/// Have Wine hide the device's client view with an OpenGL pixel format on its window.
///
/// `false`, after printing [`NO_OPENGL_NOTE`], on a machine that offers no
/// OpenGL pixel format: there is no other way to make this Wine hide the
/// view, so the workload ends there.
fn hide_client_view_with_opengl(hwnd: usize) -> bool {
    if set_opengl_pixel_format(hwnd) {
        return true;
    }
    eprintln!("{NO_OPENGL_NOTE}, so Wine cannot be made to hide the client view; not tested here");
    false
}

/// A released device's client view hidden by Wine afterwards: nothing watches it any more.
///
/// The release keeps the metal view in its client view for the window's next
/// device and removes the watch. The OpenGL pixel format then makes Wine hide
/// that client view, and no line reports it: a watch left behind would log
/// it, and one whose observer had gone would take the process down.
fn released_device_workload() {
    let h = Harness::new();
    h.render_once(RED, |_| {});
    let (view, _) = attached_view_and_surface();
    await_logged_lines(&format!("of view {view} (hidden=false)"), 1);
    assert_eq!(h.release_device(), 0, "the device is fully released");
    // The removal runs on the main queue, and Wine's hide on a request source
    // of its own, so the two are not ordered: wait until the watch is gone.
    await_logged_lines(&format!("client view watch of view {view} removed"), 1);

    if !hide_client_view_with_opengl(h.hwnd()) {
        return;
    }
    // The same pixel format hides a watched view within this time in the
    // hidden-view test above, so a watch left behind would have logged by now.
    // Nothing here confirms the hide itself; that test is what shows the
    // trigger works.
    std::thread::sleep(std::time::Duration::from_secs(1));
    let changes = logged_lines(&format!("of view {view} hidden="));
    assert!(
        changes.is_empty(),
        "a released device's client view is not watched: {changes:?}"
    );
}

/// The metal view and the client surface the device's attach line names.
fn attached_view_and_surface() -> (String, String) {
    let attached = await_logged_lines("presents through CAMetalLayer", 1);
    let line = &attached[0];
    let view = line
        .split_once("of view ")
        .and_then(|(_, rest)| rest.split_once(' '))
        .map(|(view, _)| view.to_owned())
        .expect("the attach line names the metal view");
    let surface = line
        .split_once("client surface ")
        .map(|(_, rest)| rest.trim().to_owned())
        .expect("the attach line names the client surface");
    // Every Wine the suite runs on keeps the client surface on the cocoa
    // view, so a null one means the field moved and nothing here is tested.
    assert_ne!(surface, "0x0", "the attach read the client surface: {line}");
    (view, surface)
}

/// A device's client view hidden by another client surface on its window, then shown again.
///
/// The OpenGL pixel format makes Wine show a client surface of its own on
/// the window, which hides the device's client view; the watch on that view
/// logs it. The device's next presents have Wine show its surface again, and
/// the watch logs the view shown.
fn hidden_view_workload() {
    let h = Harness::new();
    h.render_once(RED, |_| {});
    let (view, surface) = attached_view_and_surface();
    let states = format!("of view {view} hidden=");

    if !hide_client_view_with_opengl(h.hwnd()) {
        return;
    }
    await_logged_lines(&format!("{states}true"), 1);
    for _ in 0..3 {
        h.render_once(RED, |_| {});
    }
    await_logged_lines(&format!("{states}false"), 1);
    let shown = logged_lines(&format!("Wine shows client surface {surface} again"));
    assert!(
        shown.iter().any(|line| line.contains("VIEW_HIDDEN")),
        "the device asked Wine to show its surface because the view was hidden: {shown:?}"
    );
    let changes = logged_lines(&states);
    let hidden_at = changes
        .iter()
        .position(|line| line.contains(&format!("{states}true")))
        .expect("the view was hidden");
    assert!(
        changes[hidden_at..]
            .iter()
            .any(|line| line.contains(&format!("{states}false"))),
        "the view is shown again after Wine hid it: {changes:?}"
    );
    assert!(
        changes
            .last()
            .is_some_and(|line| line.contains(&format!("{states}false"))),
        "the view ends shown: {changes:?}"
    );
    assert_pixel_after_presents(&h);
}

/// A device whose window is resized asks Wine to present its client surface again.
fn resized_window_workload() {
    let h = Harness::new();
    h.render_once(RED, |_| {});
    let (_, surface) = attached_view_and_surface();

    set_window_pos(h.hwnd(), 0, 0, 320, 200);
    for _ in 0..3 {
        h.render_once(RED, |_| {});
    }
    let shown = await_logged_lines(&format!("Wine shows client surface {surface} again"), 1);
    assert!(
        shown
            .iter()
            .any(|line| line.contains("CLIENT_AREA_CHANGED")),
        "the device asked Wine to present its surface because the window resized: {shown:?}"
    );
    assert_pixel_after_presents(&h);
}

/// The device still presents its own colour after the client surface was presented again.
fn assert_pixel_after_presents(h: &Harness) {
    h.render_once(RED, |_| {});
    mtld3d_tests::assert_pixel_eq(h.read_pixel(1, 1), RED, "the device after the request");
}
