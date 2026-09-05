//! The sampler-filter rules `IDirect3DDevice9::ValidateDevice` answers with.
//!
//! D3D9 reports the first sampler stage whose filter setup cannot run. Two
//! conditions produce a failure: a stage that disables magnification or
//! minification, and a stage that asks for a filtered fetch from a texture
//! whose format the device only point-samples. Both are decisions over the
//! stage's `D3DSAMP_*` slots and the format's `D3DUSAGE_QUERY_FILTER` answer,
//! so the COM entry point only gathers that state and routes the verdict.

use mtld3d_types::{
    D3DSAMP_MAGFILTER, D3DSAMP_MINFILTER, D3DSAMP_MIPFILTER, D3DTEXF_NONE, D3DTEXF_POINT,
    SAMPLER_STATE_COUNT,
};

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
    /// A filtered fetch from a texture the device point-samples.
    ///
    /// Only the three filters that stay at point sampling survive:
    /// magnification and minification at `D3DTEXF_POINT`, mip selection at
    /// `D3DTEXF_POINT` or `D3DTEXF_NONE`. Anything else answers `E_FAIL`.
    TextureNotFilterable,
}

/// Judge one sampler stage's filters against the texture bound to it.
///
/// `texture_filterable` is `None` when the stage holds no texture, and
/// otherwise carries the bound format's `D3DUSAGE_QUERY_FILTER` answer. The
/// disabled-filter rule holds either way and is decided first, so a stage
/// that both disables a filter and carries an unfilterable texture reports
/// the disabled filter, which is the order the two answers are specified in.
#[must_use]
pub const fn stage_filter_verdict(
    sampler_states: &[u32; SAMPLER_STATE_COUNT],
    texture_filterable: Option<bool>,
) -> StageFilterVerdict {
    let mag = sampler_states[D3DSAMP_MAGFILTER as usize];
    let min = sampler_states[D3DSAMP_MINFILTER as usize];
    let mip = sampler_states[D3DSAMP_MIPFILTER as usize];
    if mag == D3DTEXF_NONE || min == D3DTEXF_NONE {
        return StageFilterVerdict::FilterDisabled;
    }
    if matches!(texture_filterable, Some(false))
        && (mag != D3DTEXF_POINT
            || min != D3DTEXF_POINT
            || (mip != D3DTEXF_NONE && mip != D3DTEXF_POINT))
    {
        return StageFilterVerdict::TextureNotFilterable;
    }
    StageFilterVerdict::Valid
}

#[cfg(test)]
mod tests;
