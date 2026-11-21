use mtld3d_types::{
    D3DSAMP_MAGFILTER, D3DSAMP_MINFILTER, D3DSAMP_MIPFILTER, D3DTEXF_ANISOTROPIC, D3DTEXF_LINEAR,
    D3DTEXF_NONE, D3DTEXF_POINT, SAMPLER_STATE_COUNT, sampler_state_defaults,
};

use super::{StageFilterVerdict, stage_filter_verdict};

/// A sampler slot array at the D3D9 defaults with the three filters overridden.
fn filters(mag: u32, min: u32, mip: u32) -> [u32; SAMPLER_STATE_COUNT] {
    let mut ss = sampler_state_defaults();
    ss[D3DSAMP_MAGFILTER as usize] = mag;
    ss[D3DSAMP_MINFILTER as usize] = min;
    ss[D3DSAMP_MIPFILTER as usize] = mip;
    ss
}

#[test]
fn the_defaults_validate() {
    assert_eq!(
        stage_filter_verdict(&sampler_state_defaults()),
        StageFilterVerdict::Valid,
        "point mag/min and no mip filter run"
    );
}

#[test]
fn a_disabled_mag_or_min_filter_fails() {
    for (mag, min) in [
        (D3DTEXF_NONE, D3DTEXF_NONE),
        (D3DTEXF_POINT, D3DTEXF_NONE),
        (D3DTEXF_NONE, D3DTEXF_POINT),
        (D3DTEXF_LINEAR, D3DTEXF_NONE),
    ] {
        assert_eq!(
            stage_filter_verdict(&filters(mag, min, D3DTEXF_NONE)),
            StageFilterVerdict::FilterDisabled,
            "mag {mag} min {min}"
        );
    }
}

#[test]
fn a_disabled_mip_filter_is_not_a_disabled_stage() {
    // D3DTEXF_NONE is the default mip filter and selects a single level; only
    // mag and min carry the disabled-filter rule.
    assert_eq!(
        stage_filter_verdict(&filters(D3DTEXF_POINT, D3DTEXF_POINT, D3DTEXF_NONE)),
        StageFilterVerdict::Valid
    );
}

#[test]
fn every_enabled_filter_validates() {
    for (mag, min, mip) in [
        (D3DTEXF_LINEAR, D3DTEXF_POINT, D3DTEXF_NONE),
        (D3DTEXF_POINT, D3DTEXF_LINEAR, D3DTEXF_NONE),
        (D3DTEXF_POINT, D3DTEXF_POINT, D3DTEXF_LINEAR),
        (D3DTEXF_ANISOTROPIC, D3DTEXF_POINT, D3DTEXF_NONE),
    ] {
        assert_eq!(
            stage_filter_verdict(&filters(mag, min, mip)),
            StageFilterVerdict::Valid,
            "mag {mag} min {min} mip {mip}"
        );
    }
}
