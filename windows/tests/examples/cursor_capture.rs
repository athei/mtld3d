//! Visible Wine cursor probe. See windows/tests/COVERAGE.md for running and checking its log.

use std::time::{Duration, Instant};

use mtld3d_tests::{Harness, HarnessConfig, window_rect};
use mtld3d_types::{D3D_OK, D3DCLEAR_TARGET, D3DFMT_A8R8G8B8, D3DPOOL_SCRATCH};

fn main() {
    let h = Harness::create(&HarnessConfig {
        visible: true,
        config_entries: "cursor.software=true;cursor.scale=1",
        ..HarnessConfig::default()
    });
    assert!(h.foreground(), "probe window must be foreground");
    let bitmap = h.create_offscreen_plain_surface(32, 32, D3DFMT_A8R8G8B8, D3DPOOL_SCRATCH);
    bitmap
        .lock_rect(0)
        .write_u32_rect(32, 32, &[0xFFFF_8040; 32 * 32]);
    assert_eq!(h.set_cursor_properties_hr(0, 0, &bitmap), D3D_OK);
    h.show_cursor(true);
    println!("cursor-probe ready window={:?}", window_rect(h.hwnd()));
    let start = Instant::now();
    let mut second = 0;
    while start.elapsed() < Duration::from_secs(24) {
        let elapsed = start.elapsed().as_secs();
        if elapsed != second {
            second = elapsed;
            if elapsed == 2 {
                assert!(h.capture_mouse(true), "SetCapture without ClipCursor");
                println!("cursor-probe capture begin");
            }
            if elapsed == 20 {
                assert!(!h.capture_mouse(false), "ReleaseCapture");
                println!("cursor-probe capture end");
            }
            // Force a new draw while captured, beyond merely inspecting the backbuffer.
            let color = if elapsed.is_multiple_of(2) {
                0xFFFF_8040
            } else {
                0xFF40_FF80
            };
            bitmap
                .lock_rect(0)
                .write_u32_rect(32, 32, &[color; 32 * 32]);
            assert_eq!(h.set_cursor_properties_hr(0, 0, &bitmap), D3D_OK);
            println!(
                "cursor-probe second={elapsed} captured={}",
                (2..20).contains(&elapsed)
            );
        }
        if (10..12).contains(&elapsed) {
            // Loading pause: native input and the run-loop observer must keep working.
            h.pump();
            std::thread::sleep(Duration::from_millis(5));
            continue;
        }
        if (14..16).contains(&elapsed) {
            for _ in 0..20 {
                h.show_cursor(false);
                h.show_cursor(true);
            }
        }
        assert_eq!(h.clear(D3DCLEAR_TARGET, 0xFF20_3050, 1.0, 0), D3D_OK);
        assert_eq!(h.present(), D3D_OK);
        h.pump();
        std::thread::sleep(Duration::from_millis(5));
    }
    drop(bitmap);
    handoffs(h);
    println!("cursor-probe complete");
}

fn handoffs(h: Harness) {
    let next = Harness::create(&HarnessConfig {
        visible: true,
        config_entries: "cursor.software=true;cursor.scale=1;color.space=accurate",
        ..HarnessConfig::default()
    });
    assert!(next.foreground(), "new software owner must be foreground");
    let next_bitmap = next.create_offscreen_plain_surface(32, 32, D3DFMT_A8R8G8B8, D3DPOOL_SCRATCH);
    next_bitmap
        .lock_rect(0)
        .write_u32_rect(32, 32, &[0xFF40_FF80; 32 * 32]);
    assert_eq!(next.set_cursor_properties_hr(0, 0, &next_bitmap), D3D_OK);
    next.show_cursor(true);
    assert_eq!(h.release_device(), 0, "old owner must detach");
    drop(h);
    present_phase(&next, "software owner/color handoff and old detach");

    let hardware = Harness::create(&HarnessConfig {
        visible: true,
        config_entries: "cursor.software=false;cursor.scale=1",
        ..HarnessConfig::default()
    });
    assert!(hardware.foreground(), "hardware owner must be foreground");
    assert_eq!(
        hardware.set_cursor_properties_hr(0, 0, &next_bitmap),
        D3D_OK
    );
    hardware.show_cursor(true);
    present_phase(&hardware, "hardware takeover");
    assert!(
        next.foreground(),
        "software owner must return to foreground"
    );
    next.show_cursor(true);
    present_phase(&next, "unchanged software sprite takeover");
    next.show_cursor(false);
}

fn present_phase(h: &Harness, name: &str) {
    println!("cursor-probe phase={name}");
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(2) {
        assert_eq!(h.clear(D3DCLEAR_TARGET, 0xFF20_3050, 1.0, 0), D3D_OK);
        assert_eq!(h.present(), D3D_OK);
        h.pump();
        std::thread::sleep(Duration::from_millis(5));
    }
}
