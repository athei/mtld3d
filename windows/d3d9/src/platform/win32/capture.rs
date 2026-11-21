//! The Ctrl+Shift+P chord, read from the keyboard through `GetAsyncKeyState`.

use std::sync::atomic::{AtomicBool, Ordering};

use mtld3d_core::capture_chord::{self, CAPTURE_KEY};

#[link(name = "user32")]
unsafe extern "system" {
    fn GetAsyncKeyState(vkey: i32) -> i16;
}

/// Whether the capture chord was pressed since the previous poll, by any device.
///
/// True on the poll that sees P go down while Control and Shift are held and
/// Alt is not; false while P stays held.
pub fn capture_hotkey_pressed() -> bool {
    /// The capture key's state at the previous poll, by any device.
    ///
    /// The resource is process-wide: the keyboard `GetAsyncKeyState` reads is
    /// one for every device, so one latch turns one press into one request.
    static CAPTURE_KEY_DOWN_LAST: AtomicBool = AtomicBool::new(false);
    let down = key_down(CAPTURE_KEY);
    let was_down = CAPTURE_KEY_DOWN_LAST.swap(down, Ordering::Relaxed);
    capture_chord::chord_pressed(was_down, down, |modifier| key_down(modifier.virtual_key()))
}

fn key_down(vkey: i32) -> bool {
    // SAFETY: `GetAsyncKeyState` is a thread-safe Win32 syscall taking an
    // `int vkey`; every caller passes a valid virtual-key constant.
    unsafe { GetAsyncKeyState(vkey) }.cast_unsigned() & 0x8000 != 0
}
