//! Unit tests for the Ctrl+Shift+P capture chord.
//!
//! Each test drives `chord_pressed` through a run of per-present samples the
//! way the d3d9 poll does: a sample is the set of virtual keys held, the
//! capture key and the modifiers are read from it by the codes the module
//! names, and the capture key's state is carried from one sample to the next.
//! The codes are restated here from the Win32 headers so a test fails if the
//! module's codes move. The chords that must not fire are the near misses (P,
//! Shift+P, Ctrl+P, Ctrl+Alt+Shift+P) and the function-key chords other
//! programs take before the game sees them (F12, the Metal HUD's Shift+F8 to
//! Shift+F12 and Ctrl+Shift+F9 to Ctrl+Shift+F12, and macOS's Ctrl+F1 to
//! Ctrl+F8 with or without Shift, which include the earlier Ctrl+Shift+F7).

use super::{CAPTURE_KEY, Modifier, chord_pressed};

const VK_SHIFT: i32 = 0x10;
const VK_CONTROL: i32 = 0x11;
const VK_MENU: i32 = 0x12;
const VK_P: i32 = 0x50;
const VK_F1: i32 = 0x70;
const VK_F7: i32 = 0x76;
const VK_F8: i32 = 0x77;
const VK_F9: i32 = 0x78;
const VK_F12: i32 = 0x7B;

/// Run the samples in order from released keys; return which of them fired.
fn fired(samples: &[&[i32]]) -> Vec<bool> {
    let mut key_was_down = false;
    samples
        .iter()
        .map(|held| {
            let key_down = held.contains(&CAPTURE_KEY);
            let fires = chord_pressed(key_was_down, key_down, |modifier| {
                held.contains(&modifier.virtual_key())
            });
            key_was_down = key_down;
            fires
        })
        .collect()
}

/// Press and release `key` with `modifiers` held throughout; return which samples fired.
fn press(modifiers: &[i32], key: i32) -> Vec<bool> {
    let mut down = modifiers.to_vec();
    down.push(key);
    fired(&[modifiers, &down, modifiers])
}

#[test]
fn ctrl_shift_p_fires_once_per_press() {
    let chord = [VK_CONTROL, VK_SHIFT];
    let down = [VK_CONTROL, VK_SHIFT, VK_P];
    assert_eq!(
        fired(&[&chord, &down, &chord, &down, &[]]),
        [false, true, false, true, false]
    );
}

#[test]
fn plain_p_does_not_fire() {
    assert_eq!(press(&[], VK_P), [false, false, false]);
}

#[test]
fn shift_p_does_not_fire() {
    assert_eq!(press(&[VK_SHIFT], VK_P), [false, false, false]);
}

#[test]
fn ctrl_p_does_not_fire() {
    assert_eq!(press(&[VK_CONTROL], VK_P), [false, false, false]);
}

#[test]
fn ctrl_alt_shift_p_does_not_fire() {
    assert_eq!(
        press(&[VK_CONTROL, VK_MENU, VK_SHIFT], VK_P),
        [false, false, false]
    );
}

#[test]
fn ctrl_shift_f7_does_not_fire() {
    assert_eq!(press(&[VK_CONTROL, VK_SHIFT], VK_F7), [false, false, false]);
}

#[test]
fn ctrl_shift_f12_does_not_fire() {
    assert_eq!(
        press(&[VK_CONTROL, VK_SHIFT], VK_F12),
        [false, false, false]
    );
}

#[test]
fn plain_f12_does_not_fire() {
    assert_eq!(press(&[], VK_F12), [false, false, false]);
}

#[test]
fn metal_hud_keys_do_not_fire() {
    for key in VK_F8..=VK_F12 {
        assert_eq!(press(&[VK_SHIFT], key), [false, false, false]);
    }
    for key in VK_F9..=VK_F12 {
        assert_eq!(press(&[VK_CONTROL, VK_SHIFT], key), [false, false, false]);
    }
}

#[test]
fn macos_keyboard_navigation_keys_do_not_fire() {
    for key in VK_F1..=VK_F8 {
        assert_eq!(press(&[VK_CONTROL], key), [false, false, false]);
        assert_eq!(press(&[VK_CONTROL, VK_SHIFT], key), [false, false, false]);
    }
}

#[test]
fn keys_going_down_between_the_same_two_presents_fire() {
    assert_eq!(fired(&[&[], &[VK_CONTROL, VK_SHIFT, VK_P]]), [false, true]);
}

#[test]
fn held_chord_does_not_repeat() {
    let chord = [VK_CONTROL, VK_SHIFT];
    let down = [VK_CONTROL, VK_SHIFT, VK_P];
    assert_eq!(
        fired(&[&chord, &down, &down, &down, &down]),
        [false, true, false, false, false]
    );
}

#[test]
fn modifiers_pressed_after_the_key_do_not_fire() {
    assert_eq!(
        fired(&[
            &[],
            &[VK_P],
            &[VK_P, VK_CONTROL],
            &[VK_P, VK_CONTROL, VK_SHIFT],
            &[VK_P, VK_CONTROL, VK_SHIFT]
        ]),
        [false, false, false, false, false]
    );
}

#[test]
fn modifiers_are_read_only_when_the_key_goes_down() {
    let unread = |_: Modifier| -> bool { panic!("a modifier was read without a key press") };
    assert!(!chord_pressed(false, false, unread));
    assert!(!chord_pressed(true, true, unread));
    assert!(!chord_pressed(true, false, unread));
}
