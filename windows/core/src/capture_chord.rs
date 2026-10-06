//! The Ctrl+Shift+F7 chord that arms the three-frame dump and the Metal GPU capture.
//!
//! The chord stays clear of the keys other programs take before Wine sees
//! them: a bare F12 is Steam's default screenshot key, and the Metal
//! performance HUD's menu takes Shift+F8 to Shift+F12 and Ctrl+Shift+F9 to
//! Ctrl+Shift+F12. The trigger is F7 with Control and Shift held and Alt up.
//! It is fixed: no configuration key and no environment variable moves it.
//!
//! The keys are sampled once per `Present`, so a press is read from two
//! consecutive samples. The chord fires when the capture key is up at one
//! sample and down at the next while, at that next sample, Control and Shift
//! are down and Alt is up. Consequences:
//!
//! - Holding the chord fires once; releasing the key and pressing it again
//!   with the modifiers still held fires again.
//! - Pressing Control and Shift after the key is already down never fires,
//!   since the key made no transition at the sample that first sees the
//!   modifiers.
//! - Keys that all go down between the same two presents count as pressed
//!   together, whatever their order, and a chord pressed and released
//!   entirely between two presents is not seen.
//!
//! The modifiers are read only on the sample where the key goes down, so the
//! steady-state cost is the one key read.

/// The capture key, F7, as a Win32 virtual-key code (`VK_F7`).
pub const CAPTURE_KEY: i32 = 0x76;

/// A modifier key the chord reads.
pub enum Modifier {
    /// Either Control key.
    Control,
    /// Either Shift key.
    Shift,
    /// Either Alt key, which winemac maps from either Command key.
    Alt,
}

impl Modifier {
    /// The Win32 virtual-key code that reads either key of this modifier.
    #[must_use]
    pub const fn virtual_key(self) -> i32 {
        match self {
            Self::Control => 0x11, // VK_CONTROL
            Self::Shift => 0x10,   // VK_SHIFT
            Self::Alt => 0x12,     // VK_MENU
        }
    }
}

/// Whether this sample completes a press of the capture chord.
///
/// `key_was_down` and `key_down` are the capture key's state at the previous
/// sample and at this one. `modifier_down` reads a modifier's current state;
/// it is called only when the key has just gone down, and stops at the first
/// modifier that rules the press out.
#[must_use]
pub fn chord_pressed(
    key_was_down: bool,
    key_down: bool,
    modifier_down: impl Fn(Modifier) -> bool,
) -> bool {
    key_down
        && !key_was_down
        && modifier_down(Modifier::Control)
        && modifier_down(Modifier::Shift)
        && !modifier_down(Modifier::Alt)
}

#[cfg(test)]
mod tests;
