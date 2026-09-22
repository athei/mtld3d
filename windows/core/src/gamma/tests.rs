//! Unit tests for gamma-ramp validation, identity detection and the lookup table.
//!
//! The identity mapping is exact arithmetic, so it is pinned at both ends and in
//! the middle: an off-by-one there would make every ramp a game sets look
//! non-identity and put the present pass on the shader route for nothing. The
//! validation tests cover each rejection shape on its own, plus the rule that a
//! ramp survives as long as one channel is usable, which is the lenient
//! behaviour a game may depend on. The lookup-table tests pin the channel order
//! and the opaque alpha lane the fragment stage never reads but Metal still
//! samples.

use mtld3d_types::{D3DGAMMARAMP, GAMMA_RAMP_ENTRIES};

use super::{
    LUT_LANES, identity_level, identity_ramp, is_identity, is_usable, is_usable_channel, to_lut,
};

/// A channel that rises by `step` per entry from `first`, saturating.
fn ramped_channel(first: u16, step: u16) -> [u16; GAMMA_RAMP_ENTRIES] {
    let mut channel = [0u16; GAMMA_RAMP_ENTRIES];
    for (level, index) in channel.iter_mut().zip(0..=u8::MAX) {
        *level = first.saturating_add(u16::from(index).saturating_mul(step));
    }
    channel
}

#[test]
fn identity_level_spans_the_full_range_exactly() {
    assert_eq!(identity_level(0), 0);
    assert_eq!(identity_level(1), 257);
    assert_eq!(identity_level(128), 32896);
    assert_eq!(identity_level(255), u16::MAX);
}

#[test]
fn the_identity_ramp_reads_as_identity_on_every_channel() {
    let ramp = identity_ramp();
    assert!(is_identity(&ramp));
    assert!(is_usable(&ramp));
    assert_eq!(ramp.red[7], identity_level(7));
    assert_eq!(ramp.green[7], identity_level(7));
    assert_eq!(ramp.blue[7], identity_level(7));
}

#[test]
fn one_changed_entry_on_one_channel_is_no_longer_identity() {
    let mut ramp = identity_ramp();
    ramp.green[200] = identity_level(200) - 1;
    assert!(!is_identity(&ramp));
}

#[test]
fn a_usable_channel_rises_end_to_end() {
    assert!(is_usable_channel(&ramped_channel(0, 257)));
    assert!(is_usable_channel(&ramped_channel(0, 1)));
}

#[test]
fn a_flat_or_inverted_channel_is_rejected() {
    assert!(!is_usable_channel(&[0u16; GAMMA_RAMP_ENTRIES]));
    assert!(!is_usable_channel(&[4242u16; GAMMA_RAMP_ENTRIES]));
    let mut inverted = ramped_channel(0, 257);
    inverted.reverse();
    assert!(!is_usable_channel(&inverted));
}

#[test]
fn a_channel_that_dips_anywhere_is_rejected() {
    let mut dipping = ramped_channel(0, 257);
    dipping[100] = dipping[99] - 1;
    assert!(!is_usable_channel(&dipping));
}

#[test]
fn a_channel_that_jumps_half_the_range_is_rejected() {
    let mut jumping = [0u16; GAMMA_RAMP_ENTRIES];
    // Everything below the jump stays at zero, so the only defect is the step.
    for (index, level) in jumping.iter_mut().enumerate() {
        *level = if index < 128 { 0 } else { u16::MAX / 2 + 1 };
    }
    assert!(!is_usable_channel(&jumping));
}

#[test]
fn a_ramp_survives_when_one_channel_is_usable() {
    // The ramp is rejected only when no channel validates: a game whose red
    // carries the brightness keeps it even with a flat green.
    let ramp = D3DGAMMARAMP {
        red: ramped_channel(0, 257),
        green: [0u16; GAMMA_RAMP_ENTRIES],
        blue: [0u16; GAMMA_RAMP_ENTRIES],
    };
    assert!(is_usable(&ramp));
}

#[test]
fn a_ramp_with_no_usable_channel_is_rejected() {
    let mut inverted = ramped_channel(0, 257);
    inverted.reverse();
    let ramp = D3DGAMMARAMP {
        red: [0u16; GAMMA_RAMP_ENTRIES],
        green: inverted,
        blue: [9u16; GAMMA_RAMP_ENTRIES],
    };
    assert!(!is_usable(&ramp));
}

#[test]
fn the_lookup_table_carries_rgb_in_order_and_an_opaque_alpha() {
    let ramp = D3DGAMMARAMP {
        red: ramped_channel(1, 2),
        green: ramped_channel(2, 3),
        blue: ramped_channel(3, 4),
    };
    let lut = to_lut(&ramp);
    assert_eq!(lut.len(), LUT_LANES);
    for index in [0usize, 1, 128, GAMMA_RAMP_ENTRIES - 1] {
        assert_eq!(lut[index * 4], ramp.red[index]);
        assert_eq!(lut[index * 4 + 1], ramp.green[index]);
        assert_eq!(lut[index * 4 + 2], ramp.blue[index]);
        assert_eq!(lut[index * 4 + 3], u16::MAX);
    }
}

#[test]
fn the_identity_lookup_table_maps_every_code_to_itself() {
    let lut = to_lut(&identity_ramp());
    for code in 0..=u8::MAX {
        let level = identity_level(code);
        let lane = usize::from(code) * 4;
        assert_eq!(lut[lane], level);
        assert_eq!(lut[lane + 1], level);
        assert_eq!(lut[lane + 2], level);
    }
}
