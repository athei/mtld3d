use mtld3d_shared::mtl::DeviceCapsFlags;
use mtld3d_types::{
    D3DFMT_A8R8G8B8, D3DFMT_D24S8, D3DFMT_DXT1, D3DFMT_INTZ, D3DMULTISAMPLE_2_SAMPLES,
    D3DMULTISAMPLE_4_SAMPLES, D3DMULTISAMPLE_8_SAMPLES, D3DMULTISAMPLE_NONE,
    D3DMULTISAMPLE_NONMASKABLE,
};

use super::*;

/// Every device advertises 2x and 4x; only some advertise 8x.
fn caps_4x() -> DeviceCapsFlags {
    DeviceCapsFlags::SAMPLE_COUNT_2 | DeviceCapsFlags::SAMPLE_COUNT_4
}

fn caps_8x() -> DeviceCapsFlags {
    caps_4x() | DeviceCapsFlags::SAMPLE_COUNT_8
}

#[test]
fn maskable_types_are_their_own_sample_count() {
    assert_eq!(sample_count_of(D3DMULTISAMPLE_NONE, 0), Ok(1));
    assert_eq!(sample_count_of(D3DMULTISAMPLE_2_SAMPLES, 0), Ok(2));
    assert_eq!(sample_count_of(D3DMULTISAMPLE_4_SAMPLES, 0), Ok(4));
    assert_eq!(sample_count_of(D3DMULTISAMPLE_8_SAMPLES, 0), Ok(8));
}

#[test]
fn nonmaskable_reads_the_quality_as_an_exponent() {
    assert_eq!(sample_count_of(D3DMULTISAMPLE_NONMASKABLE, 0), Ok(1));
    assert_eq!(sample_count_of(D3DMULTISAMPLE_NONMASKABLE, 1), Ok(2));
    assert_eq!(sample_count_of(D3DMULTISAMPLE_NONMASKABLE, 2), Ok(4));
    assert_eq!(sample_count_of(D3DMULTISAMPLE_NONMASKABLE, 3), Ok(8));
    assert_eq!(sample_count_of(D3DMULTISAMPLE_NONMASKABLE, 4), Ok(16));
    assert_eq!(
        sample_count_of(D3DMULTISAMPLE_NONMASKABLE, 5),
        Err(MultiSampleReject::Invalid)
    );
}

#[test]
fn a_type_outside_the_enumeration_is_invalid_and_one_inside_it_is_unavailable() {
    // 3 and 15 are legal `D3DMULTISAMPLE_TYPE` values that name a sample count
    // no hardware offers; 17 is outside the enumeration entirely.
    assert_eq!(sample_count_of(3, 0), Err(MultiSampleReject::Unavailable));
    assert_eq!(sample_count_of(15, 0), Err(MultiSampleReject::Unavailable));
    assert_eq!(sample_count_of(17, 0), Err(MultiSampleReject::Invalid));
    assert_eq!(
        resolve_sample_count(15, 0, D3DFMT_A8R8G8B8, caps_4x()),
        Err(MultiSampleReject::Unavailable)
    );
}

#[test]
fn a_quality_past_the_types_level_count_is_invalid() {
    // `D3DMULTISAMPLE_NONE` has no levels to index, so the argument is ignored
    // rather than rejected: a title that leaves a stale quality beside a
    // single-sampled create still gets its surface.
    assert_eq!(
        resolve_sample_count(D3DMULTISAMPLE_NONE, 1, D3DFMT_A8R8G8B8, caps_4x()),
        Ok(1)
    );
    // Every maskable type has exactly one quality level, so only 0 is in
    // range; NONMASKABLE has as many as the device advertises.
    assert_eq!(
        resolve_sample_count(D3DMULTISAMPLE_4_SAMPLES, 1, D3DFMT_A8R8G8B8, caps_4x()),
        Err(MultiSampleReject::Invalid)
    );
    assert_eq!(
        resolve_sample_count(D3DMULTISAMPLE_NONMASKABLE, 2, D3DFMT_A8R8G8B8, caps_4x()),
        Ok(4)
    );
    assert_eq!(
        resolve_sample_count(D3DMULTISAMPLE_NONMASKABLE, 3, D3DFMT_A8R8G8B8, caps_4x()),
        Err(MultiSampleReject::Invalid),
        "quality 3 is one past the three levels a 4x device offers"
    );
}

#[test]
fn an_unknown_format_has_nothing_to_answer_for() {
    assert_eq!(
        resolve_sample_count(D3DMULTISAMPLE_NONE, 0, 0, caps_4x()),
        Err(MultiSampleReject::Invalid)
    );
}

#[test]
fn resolve_answers_the_device_for_a_colour_format() {
    assert_eq!(
        resolve_sample_count(D3DMULTISAMPLE_4_SAMPLES, 0, D3DFMT_A8R8G8B8, caps_4x()),
        Ok(4)
    );
    assert_eq!(
        resolve_sample_count(D3DMULTISAMPLE_8_SAMPLES, 0, D3DFMT_A8R8G8B8, caps_4x()),
        Err(MultiSampleReject::Unavailable)
    );
    assert_eq!(
        resolve_sample_count(D3DMULTISAMPLE_8_SAMPLES, 0, D3DFMT_A8R8G8B8, caps_8x()),
        Ok(8)
    );
}

#[test]
fn depth_formats_multisample_but_readable_depth_does_not() {
    assert_eq!(
        resolve_sample_count(D3DMULTISAMPLE_4_SAMPLES, 0, D3DFMT_D24S8, caps_4x()),
        Ok(4)
    );
    assert_eq!(
        resolve_sample_count(D3DMULTISAMPLE_4_SAMPLES, 0, D3DFMT_INTZ, caps_4x()),
        Err(MultiSampleReject::Unavailable)
    );
    assert_eq!(
        resolve_sample_count(D3DMULTISAMPLE_2_SAMPLES, 0, D3DFMT_DXT1, caps_4x()),
        Err(MultiSampleReject::Unavailable)
    );
}

#[test]
fn single_sampled_requests_pass_for_every_format() {
    // A `D3DMULTISAMPLE_NONE` probe on a block-compressed format is what a
    // game uses to ask "is this format renderable at all", so it must not be
    // rejected by the multisample rules.
    assert_eq!(
        resolve_sample_count(D3DMULTISAMPLE_NONE, 0, D3DFMT_DXT1, caps_4x()),
        Ok(1)
    );
}

#[test]
fn quality_levels_track_the_highest_supported_count() {
    assert_eq!(nonmaskable_quality_levels(DeviceCapsFlags::empty()), 1);
    assert_eq!(
        nonmaskable_quality_levels(DeviceCapsFlags::SAMPLE_COUNT_2),
        2
    );
    assert_eq!(nonmaskable_quality_levels(caps_4x()), 3);
    assert_eq!(nonmaskable_quality_levels(caps_8x()), 4);
}

#[test]
fn the_sample_mask_applies_only_to_maskable_types() {
    assert!(!mask_applies(D3DMULTISAMPLE_NONE));
    assert!(!mask_applies(D3DMULTISAMPLE_NONMASKABLE));
    assert!(mask_applies(D3DMULTISAMPLE_2_SAMPLES));
    assert!(mask_applies(D3DMULTISAMPLE_4_SAMPLES));
}

#[test]
fn the_effective_mask_is_all_ones_when_the_state_has_no_effect() {
    // Single-sampled: the state is ignored whatever it holds.
    assert_eq!(
        effective_sample_mask(0x1, 1, D3DMULTISAMPLE_NONE),
        SAMPLE_MASK_ALL
    );
    // Non-maskable: the sample pattern is the driver's.
    assert_eq!(
        effective_sample_mask(0x1, 4, D3DMULTISAMPLE_NONMASKABLE),
        SAMPLE_MASK_ALL
    );
    // Every sample selected, whether the state says so in four bits or in
    // all thirty-two.
    assert_eq!(
        effective_sample_mask(0xF, 4, D3DMULTISAMPLE_4_SAMPLES),
        SAMPLE_MASK_ALL
    );
    assert_eq!(
        effective_sample_mask(0xFFFF_FFFF, 4, D3DMULTISAMPLE_4_SAMPLES),
        SAMPLE_MASK_ALL
    );
    assert_eq!(
        effective_sample_mask(0xFFFF_FFFF, 8, D3DMULTISAMPLE_8_SAMPLES),
        SAMPLE_MASK_ALL
    );
}

#[test]
fn the_effective_mask_narrows_to_the_sample_count() {
    assert_eq!(effective_sample_mask(0x1, 4, D3DMULTISAMPLE_4_SAMPLES), 0x1);
    // Bits above the sample count are dropped, so 0x35 on a 4x target is 0x5.
    assert_eq!(
        effective_sample_mask(0x35, 4, D3DMULTISAMPLE_4_SAMPLES),
        0x5
    );
    // Selecting nothing is a legitimate request and stays distinct from
    // "no masking".
    assert_eq!(effective_sample_mask(0, 2, D3DMULTISAMPLE_2_SAMPLES), 0);
    assert_eq!(
        effective_sample_mask(0x7F, 8, D3DMULTISAMPLE_8_SAMPLES),
        0x7F
    );
}
