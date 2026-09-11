//! Visible probe for games that hide the native cursor without supplying a D3D cursor image.

use std::time::{Duration, Instant};

use mtld3d_tests::{Harness, HarnessConfig, window_rect};
use mtld3d_types::{D3D_OK, D3DCLEAR_TARGET};

#[link(name = "user32")]
unsafe extern "system" {
    fn ShowCursor(show: i32) -> i32;
}

fn main() {
    let swapchain = std::env::args().any(|arg| arg == "--swapchain");
    println!("native-cursor swapchain={swapchain}");
    let present = if swapchain {
        Harness::present_swapchain
    } else {
        Harness::present
    };
    let h = Harness::create(&HarnessConfig {
        visible: true,
        ..HarnessConfig::default()
    });
    // WM_SETCURSOR for the client area selects the harness window's class arrow.
    h.send_window_message(0x0020, h.hwnd(), (0x0200 << 16) | 1);
    let arrow = h.thread_cursor();
    assert_ne!(arrow, 0);
    // SAFETY: user32 ShowCursor takes a BOOL; this probe owns its thread's display count.
    assert_eq!(unsafe { ShowCursor(0) }, -1);
    assert!(h.foreground(), "probe window must be foreground");
    println!("native-cursor ready window={:?}", window_rect(h.hwnd()));
    phase(&h, present, "hidden by ShowCursor from startup", 12);
    // SAFETY: balances the one hide above, restoring this thread's initial display count.
    assert_eq!(unsafe { ShowCursor(1) }, 0);
    phase(&h, present, "visible native arrow", 4);
    h.set_thread_cursor(0);
    phase(&h, present, "hidden by SetCursor(NULL)", 12);
    h.set_thread_cursor(arrow);
    phase(&h, present, "visible native arrow restored", 4);
}

fn phase(h: &Harness, present: fn(&Harness) -> i32, label: &str, seconds: u64) {
    println!("native-cursor phase={label}");
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(seconds) {
        assert_eq!(h.clear(D3DCLEAR_TARGET, 0xFF20_3050, 1.0, 0), D3D_OK);
        assert_eq!(present(h), D3D_OK);
        h.pump();
        std::thread::sleep(Duration::from_millis(5));
    }
}
