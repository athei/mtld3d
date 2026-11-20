//! The sampler-filter rule `IDirect3DDevice9::ValidateDevice` answers with.
//!
//! D3D9 reports the first sampler stage whose filter setup cannot run: a
//! stage that disables magnification or minification. Every format the
//! device samples takes linear filtering, so the texture bound to the stage
//! does not enter the decision. It is a decision over the stage's
//! `D3DSAMP_*` slots, so the COM entry point only gathers that state and
//! routes the verdict.

use mtld3d_types::{D3DSAMP_MAGFILTER, D3DSAMP_MINFILTER, D3DTEXF_NONE, SAMPLER_STATE_COUNT};

/// What one sampler stage lets `ValidateDevice` answer.
#[derive(Debug, PartialEq, Eq)]
pub enum StageFilterVerdict {
    /// The stage's filters run as set.
    Valid,
    /// Magnification or minification is `D3DTEXF_NONE`.
    ///
    /// No stage may disable either, whether or not it has a texture bound;
    /// D3D9 answers `D3DERR_UNSUPPORTEDTEXTUREFILTER`.
    FilterDisabled,
}

/// Judge one sampler stage's filters.
#[must_use]
pub const fn stage_filter_verdict(
    sampler_states: &[u32; SAMPLER_STATE_COUNT],
) -> StageFilterVerdict {
    let mag = sampler_states[D3DSAMP_MAGFILTER as usize];
    let min = sampler_states[D3DSAMP_MINFILTER as usize];
    if mag == D3DTEXF_NONE || min == D3DTEXF_NONE {
        return StageFilterVerdict::FilterDisabled;
    }
    StageFilterVerdict::Valid
}

#[cfg(test)]
mod tests;
