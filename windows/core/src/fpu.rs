//! The x87 control word a device leaves on the thread that creates it.
//!
//! Per the D3D9 contract a 32-bit device switches the creating thread's x87
//! unit to single precision, round to nearest, with every exception masked,
//! unless the application passes `D3DCREATE_FPU_PRESERVE`, and never changes
//! it afterwards. A comparison between a stored `f32` and a value the x87
//! stack computed then matches only because the value was rounded to 24 bits.
//! A 64-bit device leaves the control word alone.

use mtld3d_types::D3DCREATE_FPU_PRESERVE;

/// The precision-control, rounding-control and exception-mask fields of the control word.
const DEVICE_FIELDS_MASK: u16 = 0x0f3f;

/// Single precision and round to nearest (both fields zero), every exception masked.
const DEVICE_FIELDS: u16 = 0x003f;

/// Whether a device created with `behavior_flags` sets the control word.
#[must_use]
pub const fn sets_control_word(behavior_flags: u32) -> bool {
    behavior_flags & D3DCREATE_FPU_PRESERVE == 0
}

/// The control word a device leaves on a thread whose control word was `current`.
///
/// Bits outside the three fields it sets (the infinity-control bit and the
/// reserved ones) keep their value.
#[must_use]
pub const fn device_control_word(current: u16) -> u16 {
    (current & !DEVICE_FIELDS_MASK) | DEVICE_FIELDS
}

#[cfg(test)]
mod tests;
