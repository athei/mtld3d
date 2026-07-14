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

static CAPTURE_REQUESTED: AtomicBool = AtomicBool::new(false);
static F12_DOWN_LAST: AtomicBool = AtomicBool::new(false);

#[link(name = "user32")]
unsafe extern "system" {
    fn GetAsyncKeyState(vkey: i32) -> i16;
}

/// Poll F12, set `CAPTURE_REQUESTED` on rising edge.
///
/// Idempotent across frames where the key is held down — only the press
/// transition fires.
pub fn poll() {
    // SAFETY: `GetAsyncKeyState` is a thread-safe Win32 syscall taking an
    // `int vkey`; `VK_F12` is a valid virtual-key constant.
    let down = unsafe { GetAsyncKeyState(VK_F12) }.cast_unsigned() & 0x8000 != 0;
    let was_down = F12_DOWN_LAST.swap(down, Ordering::Relaxed);
    if down && !was_down {
        CAPTURE_REQUESTED.store(true, Ordering::Release);
    }
}

/// Encoder-thread side: read-and-clear. Returns true once per F12 press.
pub fn take_request() -> bool {
    CAPTURE_REQUESTED.swap(false, Ordering::AcqRel)
}
