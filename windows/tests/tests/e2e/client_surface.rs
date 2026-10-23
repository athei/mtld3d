//! Wine's client surface for a device's window stays shown and sized after the attach.
//!
//! The layer presents through `CAMetalLayer`'s own `nextDrawable`, not
//! winemac's override, which had Wine show the client surface the metal view
//! sits in, and size it to the window's client area, on every present. Some
//! Wines hide that surface's cocoa view whenever they flush the window
//! surface and leave its frame alone when the window resizes, counting on the
//! override for both. So a device whose client view Wine hid, or whose
//! window's client area moved, has Wine show its surface again, sized to the
//! client area, at its next `Present` that submits a frame. A device's first
//! submitted frame asks too, since a winemac that keeps a client view shown
//! across window surface flushes does so from the surface's second present
//! on, and until a device has drawn no other device's call presents its
//! surface.
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

use std::time::{Duration, Instant};

use mtld3d_tests::{Harness, HarnessConfig, set_opengl_pixel_format, set_window_pos};

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

/// The name the first-frame workload child runs under.
const FIRST_FRAME_CHILD_NAME: &str = "client-surface-first-frame.exe";

#[test]
fn a_new_device_asks_wine_once_at_its_first_submitted_frame() {
    if running_as(FIRST_FRAME_CHILD_NAME) {
        first_frame_workload();
        return;
    }
    run_in_private_log_child(
        FIRST_FRAME_CHILD_NAME,
        "client_surface::a_new_device_asks_wine_once_at_its_first_submitted_frame",
        CLIENT_SURFACE_LOG_FILTER,
    );
}

/// The name the undrawn-sibling workload child runs under.
const UNDRAWN_SIBLING_CHILD_NAME: &str = "client-surface-undrawn-sibling.exe";

#[test]
fn a_departing_device_leaves_an_undrawn_siblings_surface_to_its_first_frame() {
    if running_as(UNDRAWN_SIBLING_CHILD_NAME) {
        undrawn_sibling_workload();
        return;
    }
    run_in_private_log_child(
        UNDRAWN_SIBLING_CHILD_NAME,
        "client_surface::a_departing_device_leaves_an_undrawn_siblings_surface_to_its_first_frame",
        CLIENT_SURFACE_LOG_FILTER,
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
    attached_view_and_surface_of(0)
}

/// The metal view and the client surface the attach line of the `index`th device names.
fn attached_view_and_surface_of(index: usize) -> (String, String) {
    let attached = await_logged_lines("presents through CAMetalLayer", index + 1);
    let line = &attached[index];
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
    // The first frame's own request is logged before anything the resize
    // asks, and may carry a client area change of the device's creation.
    await_request_lines(&surface, "ATTACHED", 1);
    let before = request_lines(&surface, "CLIENT_AREA_CHANGED").len();

    set_window_pos(h.hwnd(), 0, 0, 320, 200);
    for _ in 0..3 {
        h.render_once(RED, |_| {});
    }
    await_request_lines(&surface, "CLIENT_AREA_CHANGED", before + 1);
    assert_pixel_after_presents(&h);
}

/// The presents of `surface` the device asked for, whose reasons name `reason`.
fn request_lines(surface: &str, reason: &str) -> Vec<String> {
    logged_lines(&format!("Wine shows client surface {surface} again ("))
        .into_iter()
        .filter(|line| line.contains(reason))
        .collect()
}

/// Wait for `expected` presents of `surface` the device asked for with `reason`.
fn await_request_lines(surface: &str, reason: &str, expected: usize) -> Vec<String> {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let lines = request_lines(surface, reason);
        if lines.len() >= expected {
            return lines;
        }
        assert!(
            Instant::now() < deadline,
            "{} request(s) of client surface {surface} for {reason} logged, expected \
             {expected}: {lines:?}",
            lines.len()
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// A new device has Wine present its surface at the attach, and once more at its first frame.
///
/// The attach present runs before the device has drawn; the device's first
/// submitted frame asks again with `ATTACHED`, which is the present after
/// which Wine keeps the client view shown across window surface flushes.
/// Later frames ask nothing for it.
///
/// Every line counted here is the unix side's, logged on this thread inside
/// the call that caused it (the attach in `CreateDevice`, a request in
/// `Present`), and the unix log writes each line to the file with one
/// unbuffered `write(2)` once the file is open, which the attach line read
/// back shows it is. So a count read after the call returned is complete:
/// none of these lines can still be on its way to the file.
fn first_frame_workload() {
    let h = Harness::new();
    let (_, surface) = attached_view_and_surface();
    let attach = await_logged_lines(&format!("Wine shows client surface {surface} (attach)"), 1);
    assert_eq!(attach.len(), 1, "one present at the attach: {attach:?}");
    let before_present = request_lines(&surface, "ATTACHED");
    assert!(
        before_present.is_empty(),
        "nothing asks before the device submits a frame: {before_present:?}"
    );

    h.render_once(RED, |_| {});
    let first_frame = request_lines(&surface, "ATTACHED");
    assert_eq!(
        first_frame.len(),
        1,
        "the first submitted frame asks once: {first_frame:?}"
    );
    for _ in 0..3 {
        h.render_once(RED, |_| {});
    }
    let later = request_lines(&surface, "ATTACHED");
    assert_eq!(
        later.len(),
        1,
        "the later frames do not ask for the attach again: {later:?}"
    );
    assert_pixel_after_presents(&h);
}

/// A device that leaves a window does not present the surface of a device there that has not drawn.
///
/// A second device on the first one's window attaches and has Wine show its
/// surface, then the first device is released before the second presents.
/// The release leaves the second device's surface to that device's first
/// frame, logged, and presents nothing: one present of that surface so far,
/// the attach's. The second device's first frame then asks with `ATTACHED`
/// and reads back its own colour.
fn undrawn_sibling_workload() {
    const BLUE: u32 = 0xFF00_00FF;

    let first = Harness::new();
    first.render_once(RED, |_| {});
    let second = Harness::create(&HarnessConfig {
        device_window: first.hwnd(),
        ..HarnessConfig::default()
    });
    let (_, surface) = attached_view_and_surface_of(1);
    assert_eq!(first.release_device(), 0, "release the earlier device");
    await_logged_lines(
        &format!("client surface {surface}, the newest left on window"),
        1,
    );
    let presents = logged_lines(&format!("Wine shows client surface {surface} ("));
    assert_eq!(
        presents.len(),
        1,
        "only the attach presented the surface of a device that has not drawn: {presents:?}"
    );
    assert!(
        presents[0].contains("(attach)"),
        "that present is the attach's: {presents:?}"
    );

    second.render_once(BLUE, |_| {});
    await_request_lines(&surface, "ATTACHED", 1);
    mtld3d_tests::assert_pixel_eq(
        second.read_pixel(1, 1),
        BLUE,
        "the device left on the window",
    );
    // The window is the first harness's to destroy, after the device on it.
    drop(second);
    drop(first);
}

/// The device still presents its own colour after the client surface was presented again.
fn assert_pixel_after_presents(h: &Harness) {
    h.render_once(RED, |_| {});
    mtld3d_tests::assert_pixel_eq(h.read_pixel(1, 1), RED, "the device after the request");
}
