//! Ctrl+Shift+P hotkey poll: one press arms the frame dump and the Metal GPU capture.
//!
//! Both diagnostics cover the same [`FrameDump::FRAMES`] consecutive frames
//! (see `device::frame_dump`): the dump logs the D3D9-level events a GPU
//! trace cannot know, the trace holds everything Metal saw, and the dump's
//! draw numbering and frame labels name the trace's nodes.
//!
//! Apple gates the capture itself on `MTL_CAPTURE_ENABLED=1` at process
//! launch, so there is no mtld3d-side env guard; without the Apple env the
//! unix-side `start_capture` handler logs a warn and returns, and the dump
//! still runs.
//!
//! The chord, its key codes and what counts as a press are
//! `mtld3d_core::capture_chord`'s: P going down while Control and Shift are
//! held and Alt is not, a chord that neither Steam, the Metal HUD nor macOS
//! takes. Polling cost is one `GetAsyncKeyState` syscall per `Present()`
//! (~100 ns), plus up to three for the modifiers on the present where P goes
//! down.
//!
//! Flow: `device_present` → `poll()` → on a chord press sets
//! `CAPTURE_REQUESTED`; the same `Present` takes it through
//! `take_request()` and arms `frame_dump_present`, which marks the first
//! and last frame of the run with `FrameDataFlags::GPU_CAPTURE_START` /
//! `GPU_CAPTURE_STOP`. The encoder thread brackets those frames with the
//! `StartGpuCapture` / `StopGpuCapture` thunks. The trace lands next to
//! the process's log file, numbered per press.
//!
//! [`FrameDump::FRAMES`]: crate::device::frame_dump::FrameDump::FRAMES

use std::sync::atomic::{AtomicBool, Ordering};

/// A chord press not yet taken by a `Present`.
///
/// The resource is process-wide: `GetAsyncKeyState` reads the one keyboard
/// every device in the process shares, so one press arms one capture. The
/// next `Present` that asks takes it: usually the one that saw the press,
/// but another device's, or a later one when the polling `Present` returned
/// early with `D3DERR_DEVICENOTRESET` or a send failure, can take it too.
static CAPTURE_REQUESTED: AtomicBool = AtomicBool::new(false);

/// Poll the chord once per present, firing when P goes down with Control and Shift held.
///
/// Idempotent across frames where P is held down.
pub fn poll() {
    if crate::platform::capture_hotkey_pressed() {
        CAPTURE_REQUESTED.store(true, Ordering::Release);
    }
}

/// Take the pending chord request, if any; the next frames are then dumped and captured.
pub fn take_request() -> bool {
    CAPTURE_REQUESTED.swap(false, Ordering::AcqRel)
}
