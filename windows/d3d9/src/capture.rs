//! F12 hotkey poll: one press arms the frame dump and the Metal GPU capture.
//!
//! Both diagnostics cover the same [`FrameDump::FRAMES`] consecutive frames
//! (see `device::frame_dump`): the dump logs the D3D9-level events a GPU
//! trace cannot know, the trace holds everything Metal saw, and the dump's
//! draw numbering and frame labels name the trace's nodes.
//!
//! Apple gates the capture itself on `MTL_CAPTURE_ENABLED=1` at process
//! launch, so there is no mtld3d-side env guard; without the Apple env the
//! unix-side `start_capture` handler logs a warn and returns, and the dump
//! still runs. Polling cost is one `GetAsyncKeyState` syscall per
//! `Present()` (~100 ns), free in practice.
//!
//! A plain function key is a weak trigger on a Mac (F11 is the system's
//! "show desktop" key, others double as media keys); F12 is the one Apple
//! leaves alone and the one Xcode uses for the same purpose.
//!
//! Flow: `device_present` → `poll()` → on the F12 rising edge sets
//! `CAPTURE_REQUESTED`; the same `Present` takes it through
//! `take_request()` and arms `frame_dump_present`, which marks the first
//! and last frame of the run with `FrameDataFlags::GPU_CAPTURE_START` /
//! `GPU_CAPTURE_STOP`. The encoder thread brackets those frames with the
//! `StartGpuCapture` / `StopGpuCapture` thunks. The trace lands next to
//! the process's log file, numbered per press.
//!
//! [`FrameDump::FRAMES`]: crate::device::frame_dump::FrameDump::FRAMES

use std::sync::atomic::{AtomicBool, Ordering};

const VK_F12: i32 = 0x7B;

static CAPTURE_REQUESTED: AtomicBool = AtomicBool::new(false);
static F12_DOWN_LAST: AtomicBool = AtomicBool::new(false);

#[link(name = "user32")]
unsafe extern "system" {
    fn GetAsyncKeyState(vkey: i32) -> i16;
}

fn key_down(vkey: i32) -> bool {
    // SAFETY: `GetAsyncKeyState` is a thread-safe Win32 syscall taking an
    // `int vkey`; every caller passes a valid virtual-key constant.
    unsafe { GetAsyncKeyState(vkey) }.cast_unsigned() & 0x8000 != 0
}

/// Poll the trigger key once per present, firing on the press transition.
///
/// Idempotent across frames where the key is held down.
pub fn poll() {
    let down = key_down(VK_F12);
    let was_down = F12_DOWN_LAST.swap(down, Ordering::Relaxed);
    if down && !was_down {
        CAPTURE_REQUESTED.store(true, Ordering::Release);
    }
}

/// Take the pending F12 request, if any; the next frames are then dumped and captured.
pub fn take_request() -> bool {
    CAPTURE_REQUESTED.swap(false, Ordering::AcqRel)
}
