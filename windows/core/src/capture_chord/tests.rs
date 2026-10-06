//! Unit tests for the Ctrl+Shift+F12 capture chord.
//!
//! Each test drives `chord_pressed` through a run of per-present samples the
//! way the d3d9 poll does, carrying F12's state from one sample to the next,
//! and checks which samples fire. The chords that must not fire are the ones
//! another program or the system owns (F12, Shift+F12) and the near misses
//! (Ctrl+F12, Ctrl+Alt+Shift+F12).

use super::{Modifier, chord_pressed};

const CONTROL: u8 = 1 << 0;
const SHIFT: u8 = 1 << 1;
const ALT: u8 = 1 << 2;

/// One per-present sample: whether F12 is down, and the modifiers held.
struct Sample {
    f12: bool,
    modifiers: u8,
}

const fn up(modifiers: u8) -> Sample {
    Sample {
        f12: false,
        modifiers,
    }
}

const fn down(modifiers: u8) -> Sample {
    Sample {
        f12: true,
        modifiers,
    }
}

const fn bit(modifier: &Modifier) -> u8 {
    match modifier {
        Modifier::Control => CONTROL,
        Modifier::Shift => SHIFT,
        Modifier::Alt => ALT,
    }
}

/// Run the samples in order from a released F12; return which of them fired.
fn fired(samples: &[Sample]) -> Vec<bool> {
    let mut f12_was_down = false;
    samples
        .iter()
        .map(|sample| {
            let fires = chord_pressed(f12_was_down, sample.f12, |modifier| {
                sample.modifiers & bit(&modifier) != 0
            });
            f12_was_down = sample.f12;
            fires
        })
        .collect()
}

#[test]
fn plain_f12_does_not_fire() {
    assert_eq!(fired(&[up(0), down(0), up(0)]), [false, false, false]);
}

#[test]
fn shift_f12_does_not_fire() {
    assert_eq!(
        fired(&[up(SHIFT), down(SHIFT), up(SHIFT)]),
        [false, false, false]
    );
}

#[test]
fn ctrl_f12_does_not_fire() {
    assert_eq!(
        fired(&[up(CONTROL), down(CONTROL), up(CONTROL)]),
        [false, false, false]
    );
}

#[test]
fn ctrl_alt_shift_f12_does_not_fire() {
    let all = CONTROL | SHIFT | ALT;
    assert_eq!(fired(&[up(all), down(all), up(all)]), [false, false, false]);
}

#[test]
fn ctrl_shift_f12_fires_once_per_press() {
    let chord = CONTROL | SHIFT;
    assert_eq!(
        fired(&[up(chord), down(chord), up(chord), down(chord), up(0)]),
        [false, true, false, true, false]
    );
}

#[test]
fn keys_going_down_between_the_same_two_presents_fire() {
    assert_eq!(fired(&[up(0), down(CONTROL | SHIFT)]), [false, true]);
}

#[test]
fn held_chord_does_not_repeat() {
    let chord = CONTROL | SHIFT;
    assert_eq!(
        fired(&[
            up(chord),
            down(chord),
            down(chord),
            down(chord),
            down(chord)
        ]),
        [false, true, false, false, false]
    );
}

#[test]
fn modifiers_pressed_after_f12_do_not_fire() {
    assert_eq!(
        fired(&[
            up(0),
            down(0),
            down(CONTROL),
            down(CONTROL | SHIFT),
            down(CONTROL | SHIFT)
        ]),
        [false, false, false, false, false]
    );
}

#[test]
fn modifiers_are_read_only_when_f12_goes_down() {
    let unread = |_: Modifier| -> bool { panic!("a modifier was read without an F12 press") };
    assert!(!chord_pressed(false, false, unread));
    assert!(!chord_pressed(true, true, unread));
    assert!(!chord_pressed(true, false, unread));
}
