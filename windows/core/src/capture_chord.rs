//! The Ctrl+Shift+F12 chord that arms the three-frame dump and the Metal GPU capture.
//!
//! A bare F12 is Steam's default screenshot key and a common in-game bind,
//! and on macOS 27 Shift+F12 opens the Metal performance HUD's configuration
//! panel, so the trigger is the chord with Control and Shift held and Alt up.
//! It is fixed: no configuration key and no environment variable moves it.
//!
//! The keys are sampled once per `Present`, so a press is read from two
//! consecutive samples. The chord fires when F12 is up at one sample and
//! down at the next while, at that next sample, Control and Shift are down
//! and Alt is up. Consequences:
//!
//! - Holding the chord fires once; releasing F12 and pressing it again with
//!   the modifiers still held fires again.
//! - Pressing Control and Shift after F12 is already down never fires, since
//!   F12 made no transition at the sample that first sees the modifiers.
//! - Keys that all go down between the same two presents count as pressed
//!   together, whatever their order, and a chord pressed and released
//!   entirely between two presents is not seen.
//!
//! The modifiers are read only on the sample where F12 goes down, so the
//! steady-state cost is the one F12 read.

/// A modifier key the chord reads.
pub enum Modifier {
    /// Either Control key.
    Control,
    /// Either Shift key.
    Shift,
    /// Either Alt key, which winemac maps from either Command key.
    Alt,
}

/// Whether this sample completes a press of the capture chord.
///
/// `f12_was_down` and `f12_down` are F12's state at the previous sample and
/// at this one. `modifier_down` reads a modifier's current state; it is
/// called only when F12 has just gone down, and stops at the first modifier
/// that rules the press out.
#[must_use]
pub fn chord_pressed(
    f12_was_down: bool,
    f12_down: bool,
    modifier_down: impl Fn(Modifier) -> bool,
) -> bool {
    f12_down
        && !f12_was_down
        && modifier_down(Modifier::Control)
        && modifier_down(Modifier::Shift)
        && !modifier_down(Modifier::Alt)
}

#[cfg(test)]
mod tests;
