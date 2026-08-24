//! F12 hotkey poll for one-shot Metal GPU frame capture.
//!
//! Apple gates capture itself on `MTL_CAPTURE_ENABLED=1` at process
//! launch — no mtld3d-side env guard needed; without the Apple env, the
//! unix-side `start_capture` handler logs a warn and returns. Polling
//! cost is one `GetAsyncKeyState` syscall per `Present()` (~100 ns),
//! free in practice.
//!
//! Flow: `device_present` → `poll()` → on F12 rising-edge sets
//! `CAPTURE_REQUESTED`. The encoder thread reads + clears the flag at
//! the next frame and brackets `run_frame` with `StartGpuCapture` /
//! `StopGpuCapture` thunks. Output is `/tmp/mtld3d_capture.gputrace`.

use std::sync::atomic::{AtomicBool, Ordering};

const VK_F12: i32 = 0x7B;
const VK_SHIFT: i32 = 0x10;
const VK_CONTROL: i32 = 0x11;
/// The `D` key: Ctrl+Shift+D arms the one-frame draw-state dump.
///
/// A plain function key is a bad trigger on a Mac: F11 is the system's
/// "show desktop" key, and others double as media keys.
const VK_D: i32 = 0x44;

static CAPTURE_REQUESTED: AtomicBool = AtomicBool::new(false);
static F12_DOWN_LAST: AtomicBool = AtomicBool::new(false);
/// Ctrl+Shift+D asked for a one-frame draw-state dump (`device::frame_dump`).
static FRAME_DUMP_REQUESTED: AtomicBool = AtomicBool::new(false);
static DUMP_CHORD_DOWN_LAST: AtomicBool = AtomicBool::new(false);

#[link(name = "user32")]
unsafe extern "system" {
    fn GetAsyncKeyState(vkey: i32) -> i16;
}

fn key_down(vkey: i32) -> bool {
    // SAFETY: `GetAsyncKeyState` is a thread-safe Win32 syscall taking an
    // `int vkey`; every caller passes a valid virtual-key constant.
    unsafe { GetAsyncKeyState(vkey) }.cast_unsigned() & 0x8000 != 0
}

/// Poll the trigger keys once per present, firing on the press transition.
///
/// Idempotent across frames where a key is held down.
pub fn poll() {
    let down = key_down(VK_F12);
    let was_down = F12_DOWN_LAST.swap(down, Ordering::Relaxed);
    if down && !was_down {
        CAPTURE_REQUESTED.store(true, Ordering::Release);
    }
    let chord = key_down(VK_CONTROL) && key_down(VK_SHIFT) && key_down(VK_D);
    let was_chord = DUMP_CHORD_DOWN_LAST.swap(chord, Ordering::Relaxed);
    if chord && !was_chord {
        FRAME_DUMP_REQUESTED.store(true, Ordering::Release);
    }
}

/// Take the pending Ctrl+Shift+D request, if any; the next frame is then dumped.
pub fn take_frame_dump_request() -> bool {
    FRAME_DUMP_REQUESTED.swap(false, Ordering::AcqRel)
}

/// Encoder-thread side: read-and-clear. Returns true once per F12 press.
pub fn take_request() -> bool {
    CAPTURE_REQUESTED.swap(false, Ordering::AcqRel)
}
