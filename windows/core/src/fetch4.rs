//! Latched Fetch4 commands and texture eligibility.

use mtld3d_types::{
    D3DFMT_A8, D3DFMT_DF16, D3DFMT_DF24, D3DFMT_INTZ, D3DFMT_L8, D3DFMT_L16, D3DFMT_R16F,
    D3DFMT_R32F, D3DSAMP_MAGFILTER, D3DSAMP_MIPMAPLODBIAS, D3DTEXF_POINT, FETCH4_DISABLE,
    FETCH4_ENABLE,
};

/// Incremental sampler specialization state, updated only by API setters.
pub struct Fetch4State {
    enabled: u16,
    point: u16,
    eligible: u16,
    alpha: u16,
    raw_red: u16,
}

impl Fetch4State {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            enabled: 0,
            point: u16::MAX,
            eligible: 0,
            alpha: 0,
            raw_red: 0,
        }
    }

    pub const fn set_sampler(&mut self, slot: usize, state: usize, value: u32) {
        let bit = 1u16 << slot;
        if state == D3DSAMP_MIPMAPLODBIAS as usize {
            // Numeric bias writes preserve the command latch.
            if let Some(on) = command(value) {
                self.enabled = with_bit(self.enabled, bit, on);
            }
        } else if state == D3DSAMP_MAGFILTER as usize {
            self.point = with_bit(self.point, bit, value == D3DTEXF_POINT);
        }
    }

    pub const fn set_texture(&mut self, slot: usize, format: Option<u32>, two_dimensional: bool) {
        let bit = 1u16 << slot;
        let eligible = two_dimensional
            && matches!(
                format,
                Some(
                    D3DFMT_A8
                        | D3DFMT_DF16
                        | D3DFMT_DF24
                        | D3DFMT_INTZ
                        | D3DFMT_L8
                        | D3DFMT_L16
                        | D3DFMT_R16F
                        | D3DFMT_R32F
                )
            );
        self.raw_red = with_bit(
            self.raw_red,
            bit,
            matches!(format, Some(D3DFMT_DF16 | D3DFMT_DF24)),
        );
        self.eligible = with_bit(self.eligible, bit, eligible);
        self.alpha = with_bit(
            self.alpha,
            bit,
            eligible && matches!(format, Some(D3DFMT_A8)),
        );
    }

    #[must_use]
    pub const fn raw_red_mask(&self) -> u16 {
        self.raw_red
    }

    #[must_use]
    pub const fn enabled(&self) -> u16 {
        self.enabled
    }

    pub const fn restore_enabled(&mut self, enabled: u16) {
        self.enabled = enabled;
    }

    #[must_use]
    pub const fn masks(&self) -> (u16, u16) {
        let active = self.enabled & self.point & self.eligible;
        (active, active & self.alpha)
    }
}

impl Default for Fetch4State {
    fn default() -> Self {
        Self::new()
    }
}

/// Decode vendor commands without treating their bit patterns as float biases.
#[must_use]
pub const fn command(value: u32) -> Option<bool> {
    match value {
        FETCH4_ENABLE => Some(true),
        FETCH4_DISABLE => Some(false),
        _ => None, // Every other bit pattern is an ordinary LOD bias.
    }
}

const fn with_bit(mask: u16, bit: u16, on: bool) -> u16 {
    if on { mask | bit } else { mask & !bit }
}

#[cfg(test)]
mod tests;
