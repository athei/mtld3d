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
fn the_defaults_validate_with_and_without_a_texture() {
    let ss = sampler_state_defaults();
    for bound in [None, Some(true), Some(false)] {
        assert_eq!(
            stage_filter_verdict(&ss, bound),
            StageFilterVerdict::Valid,
            "point mag/min and no mip filter run on any texture ({bound:?})"
        );
    }
}

#[test]
fn a_disabled_mag_or_min_filter_fails_with_or_without_a_texture() {
    for (mag, min) in [
        (D3DTEXF_NONE, D3DTEXF_NONE),
        (D3DTEXF_POINT, D3DTEXF_NONE),
        (D3DTEXF_NONE, D3DTEXF_POINT),
        (D3DTEXF_LINEAR, D3DTEXF_NONE),
    ] {
        for bound in [None, Some(true), Some(false)] {
            assert_eq!(
                stage_filter_verdict(&filters(mag, min, D3DTEXF_NONE), bound),
                StageFilterVerdict::FilterDisabled,
                "mag {mag} min {min} with texture {bound:?}"
            );
        }
    }
}

#[test]
fn a_disabled_mip_filter_is_not_a_disabled_stage() {
    // D3DTEXF_NONE is the default mip filter and selects a single level; only
    // mag and min carry the disabled-filter rule.
    assert_eq!(
        stage_filter_verdict(&filters(D3DTEXF_POINT, D3DTEXF_POINT, D3DTEXF_NONE), None),
        StageFilterVerdict::Valid
    );
}

#[test]
fn a_filtered_fetch_from_an_unfilterable_texture_fails() {
    for (mag, min, mip) in [
        (D3DTEXF_LINEAR, D3DTEXF_POINT, D3DTEXF_NONE),
        (D3DTEXF_POINT, D3DTEXF_LINEAR, D3DTEXF_NONE),
        (D3DTEXF_POINT, D3DTEXF_POINT, D3DTEXF_LINEAR),
        (D3DTEXF_ANISOTROPIC, D3DTEXF_POINT, D3DTEXF_NONE),
    ] {
        assert_eq!(
            stage_filter_verdict(&filters(mag, min, mip), Some(false)),
            StageFilterVerdict::TextureNotFilterable,
            "mag {mag} min {min} mip {mip} on an unfilterable format"
        );
        assert_eq!(
            stage_filter_verdict(&filters(mag, min, mip), Some(true)),
            StageFilterVerdict::Valid,
            "mag {mag} min {min} mip {mip} on a filterable format"
        );
        assert_eq!(
            stage_filter_verdict(&filters(mag, min, mip), None),
            StageFilterVerdict::Valid,
            "mag {mag} min {min} mip {mip} with no texture bound"
        );
    }
}

#[test]
fn point_sampling_an_unfilterable_texture_validates() {
    for mip in [D3DTEXF_NONE, D3DTEXF_POINT] {
        assert_eq!(
            stage_filter_verdict(&filters(D3DTEXF_POINT, D3DTEXF_POINT, mip), Some(false)),
            StageFilterVerdict::Valid,
            "point mag/min with mip {mip} samples any format"
        );
    }
}
