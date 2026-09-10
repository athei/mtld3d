//! Bitmap bounds and acknowledged-upload recovery.

use mtld3d_types::{D3DFMT_A8, D3DFMT_A8R8G8B8, D3DFMT_R5G6B5, D3DFMT_X8R8G8B8};

use super::{BitmapLayout, reconcile_upload};

fn bitmap() -> BitmapLayout {
    BitmapLayout {
        width: 32,
        height: 32,
        x_hotspot: 4,
        y_hotspot: 6,
    }
}

#[test]
fn only_argb_cursors_are_read_as_four_byte_pixels() {
    assert!(bitmap().valid(D3DFMT_A8R8G8B8, (1920, 1080), 2));
    for format in [D3DFMT_A8, D3DFMT_R5G6B5, D3DFMT_X8R8G8B8] {
        assert!(!bitmap().valid(format, (1920, 1080), 2));
    }
}

#[test]
fn extent_and_hotspot_scaling_are_checked_before_reading() {
    for width in [0, 3, 2048, u32::MAX] {
        let layout = BitmapLayout { width, ..bitmap() };
        assert!(!layout.valid(D3DFMT_A8R8G8B8, (1920, 1080), 2));
    }
    let layout = BitmapLayout {
        x_hotspot: u32::MAX,
        ..bitmap()
    };
    assert!(layout.scaled(1).is_some());
    assert!(layout.scaled(2).is_none());
    let layout = BitmapLayout {
        width: 1 << 29,
        height: 1 << 29,
        ..bitmap()
    };
    assert!(layout.scaled(8).is_none());
    assert!(bitmap().scaled(0).is_none());
    assert!(bitmap().scaled(9).is_none());
}

#[test]
fn locked_rows_must_cover_every_pixel_without_pointer_arithmetic_overflow() {
    let layout = bitmap();
    assert_eq!(layout.row_pitch(0x1000, 128), Some(128));
    assert_eq!(layout.row_pitch(0x1000, 256), Some(256));
    for pitch in [-128, 0, 1, 64, 127] {
        assert!(layout.row_pitch(0x1000, pitch).is_none());
    }
    assert!(layout.row_pitch(0, 128).is_none());
    assert!(layout.row_pitch(usize::MAX - 128, 128).is_none());
    let layout = BitmapLayout {
        height: u32::MAX,
        ..bitmap()
    };
    // This layout overflows a 32-bit slice even though each individual row fits.
    if usize::BITS == 32 {
        assert!(layout.row_pitch(0x1000, i32::MAX).is_none());
    }
}

#[test]
fn failed_first_upload_is_retried_on_retarget_or_show() {
    let mut calls = Vec::new();
    let known = reconcile_upload(false, |pixels| {
        calls.push(pixels);
        false
    });
    assert!(!known);
    assert_eq!(calls, [true]);
    calls.clear();
    assert!(reconcile_upload(known, |pixels| {
        calls.push(pixels);
        true
    }));
    assert_eq!(calls, [true]);
}

#[test]
fn an_unchanged_known_hash_still_reconciles_ownership_and_visibility() {
    let mut calls = Vec::new();
    assert!(reconcile_upload(true, |pixels| {
        calls.push(pixels);
        true
    }));
    assert_eq!(calls, [false]);
}

#[test]
fn rejected_hash_gets_exactly_one_full_retry_and_forgets_failed_acknowledgment() {
    let mut calls = Vec::new();
    let known = reconcile_upload(true, |pixels| {
        calls.push(pixels);
        false
    });
    assert!(!known);
    assert_eq!(calls, [false, true]);
    calls.clear();
    assert!(reconcile_upload(true, |pixels| {
        calls.push(pixels);
        pixels
    }));
    assert_eq!(calls, [false, true]);
}
