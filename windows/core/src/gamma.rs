//! Gamma-ramp validation, identity detection and the present-pass lookup table.
//!
//! `IDirect3DDevice9::SetGammaRamp` hands over three 256-entry channel ramps
//! that map each 8-bit code to the 16-bit level the display should show.
//! Nothing in D3D9 applies them to the rendered image: the ramp is the
//! display's transfer function, applied after the frame is composed. Here it
//! rides the present pass instead, as a lookup table the fragment stage
//! samples with the gamma-encoded value it just read, which is the last point
//! before the frame leaves for the compositor.
//!
//! This module is the whole D3D9 half of that: whether a ramp is usable,
//! whether it would change anything, and the texture payload the unix side
//! uploads. The unix side receives finished `RGBA16Unorm` texels and needs no
//! knowledge of `D3DGAMMARAMP`.

use mtld3d_types::{D3DGAMMARAMP, GAMMA_RAMP_ENTRIES};

/// Texel count of the lookup table, one per 8-bit code.
pub const LUT_TEXELS: usize = GAMMA_RAMP_ENTRIES;

/// `u16` lanes of the lookup table: four per texel, `RGBA16Unorm`.
pub const LUT_LANES: usize = LUT_TEXELS * 4;

/// The level an identity ramp holds for code `index`.
///
/// `65535 * index / 255` exactly, which is what a driver reports for the ramp
/// it starts with and what every entry of a ramp that changes nothing holds.
#[must_use]
pub const fn identity_level(index: u8) -> u16 {
    // 65535 == 255 * 257, so the quotient is exact and needs no rounding.
    (index as u16) * 257
}

/// An identity ramp, the value a device reports before the application sets one.
#[must_use]
pub fn identity_ramp() -> D3DGAMMARAMP {
    let mut channel = [0u16; GAMMA_RAMP_ENTRIES];
    // Both sides are exactly 256 long, so the index needs no conversion and
    // the range never has to step past its end.
    for (level, index) in channel.iter_mut().zip(0..=u8::MAX) {
        *level = identity_level(index);
    }
    D3DGAMMARAMP {
        red: channel,
        green: channel,
        blue: channel,
    }
}

/// Whether one channel's levels describe a usable transfer function.
///
/// A channel is rejected when it is inverted or flat end to end, when it
/// decreases anywhere, or when it jumps by half the range or more between
/// neighbours. D3D9 itself takes any ramp, but those three shapes are what a
/// game produces by handing over uninitialized or byte-swapped memory, and
/// applying one leaves a display unreadable with no way back inside the
/// game. A ramp meant to darken or brighten passes all three.
#[must_use]
pub fn is_usable_channel(channel: &[u16; GAMMA_RAMP_ENTRIES]) -> bool {
    if channel[0] >= channel[GAMMA_RAMP_ENTRIES - 1] {
        return false;
    }
    channel
        .windows(2)
        .all(|pair| pair[1] >= pair[0] && pair[1] - pair[0] < u16::MAX / 2)
}

/// Whether a ramp is usable as a whole.
///
/// A ramp passes as long as **one** channel is usable. Rejecting a whole ramp
/// because one channel is unusual would throw away the brightness a player
/// asked for whenever the other two carry it, and the rejection exists only
/// to catch a ramp that is garbage in every channel.
#[must_use]
pub fn is_usable(ramp: &D3DGAMMARAMP) -> bool {
    is_usable_channel(&ramp.red) || is_usable_channel(&ramp.green) || is_usable_channel(&ramp.blue)
}

/// Whether a ramp leaves every code where it found it.
///
/// An identity ramp needs no lookup table and no present-pass transform, so
/// detecting it keeps the ordinary present on the route it already takes.
#[must_use]
pub fn is_identity(ramp: &D3DGAMMARAMP) -> bool {
    (0..=u8::MAX).all(|index| {
        let level = identity_level(index);
        let index = usize::from(index);
        ramp.red[index] == level && ramp.green[index] == level && ramp.blue[index] == level
    })
}

/// The ramp as one row of `RGBA16Unorm` texels.
///
/// Texel `i` carries the three channels' level for code `i` and an opaque
/// alpha, so the present fragment stage reads all three with one sample per
/// channel and the alpha lane never participates. The alpha is written rather
/// than left zero because the sample's `.a` is otherwise a Metal-visible
/// uninitialized lane.
#[must_use]
pub fn to_lut(ramp: &D3DGAMMARAMP) -> [u16; LUT_LANES] {
    let mut lut = [0u16; LUT_LANES];
    let (texels, rest) = lut.as_chunks_mut::<4>();
    debug_assert!(rest.is_empty(), "the table is four lanes per entry");
    for (index, texel) in texels.iter_mut().enumerate() {
        *texel = [
            ramp.red[index],
            ramp.green[index],
            ramp.blue[index],
            u16::MAX,
        ];
    }
    lut
}

/// What a queued ramp change asks the layer's present pass to do.
///
/// The two cases are not one `Option`: a ramp that stops applying has to be
/// sent, so "no change queued" and "queued a removal" are different states.
pub enum Change {
    /// Apply this table, the ramp laid out by [`to_lut`].
    Apply(Box<[u16; LUT_LANES]>),
    /// Remove the table the layer carries.
    Remove,
}

#[cfg(test)]
mod tests;
