//! Ctrl+Shift+F12 hotkey poll: one press arms the frame dump and the Metal GPU capture.
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
//! The chord and what counts as a press are `mtld3d_core::capture_chord`'s:
//! F12 going down while Control and Shift are held and Alt is not. A bare
//! F12 is Steam's screenshot key and a common game bind, and Shift+F12
//! opens the Metal HUD's configuration panel on macOS 27. Polling cost is
//! one `GetAsyncKeyState` syscall per `Present()` (~100 ns), plus up to
//! three for the modifiers on the present where F12 goes down.
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

use mtld3d_core::capture_chord::{self, Modifier};

const VK_SHIFT: i32 = 0x10;
const VK_CONTROL: i32 = 0x11;
const VK_MENU: i32 = 0x12;
const VK_F12: i32 = 0x7B;

/// A chord press not yet taken by a `Present`.
///
/// The resource is process-wide: `GetAsyncKeyState` reads the one keyboard
/// every device in the process shares, so one press arms one capture. The
/// next `Present` that asks takes it: usually the one that saw the press,
/// but another device's, or a later one when the polling `Present` returned
/// early with `D3DERR_DEVICENOTRESET` or a send failure, can take it too.
static CAPTURE_REQUESTED: AtomicBool = AtomicBool::new(false);

#[link(name = "user32")]
unsafe extern "system" {
    fn GetAsyncKeyState(vkey: i32) -> i16;
}

fn key_down(vkey: i32) -> bool {
    // SAFETY: `GetAsyncKeyState` is a thread-safe Win32 syscall taking an
    // `int vkey`; every caller passes a valid virtual-key constant.
    unsafe { GetAsyncKeyState(vkey) }.cast_unsigned() & 0x8000 != 0
}

/// Poll the chord once per present, firing when F12 goes down with Control and Shift held.
///
/// Idempotent across frames where F12 is held down.
pub fn poll() {
    /// F12's state at the previous poll, by any device.
    ///
    /// The resource is process-wide: the keyboard `GetAsyncKeyState` reads is
    /// one for every device, so one latch turns one press into one request.
    static F12_DOWN_LAST: AtomicBool = AtomicBool::new(false);
    let down = key_down(VK_F12);
    let was_down = F12_DOWN_LAST.swap(down, Ordering::Relaxed);
    let pressed = capture_chord::chord_pressed(was_down, down, |modifier| {
        key_down(match modifier {
            Modifier::Control => VK_CONTROL,
            Modifier::Shift => VK_SHIFT,
            Modifier::Alt => VK_MENU,
        })
    });
    if pressed {
        CAPTURE_REQUESTED.store(true, Ordering::Release);
    }
}

/// Take the pending chord request, if any; the next frames are then dumped and captured.
pub fn take_request() -> bool {
    CAPTURE_REQUESTED.swap(false, Ordering::AcqRel)
}
