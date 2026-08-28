//! `D3DMULTISAMPLE_TYPE` decoding and the device's answer for it.
//!
//! One home for the two questions every multisample path asks: what sample
//! count does a `(type, quality)` pair mean, and can this Metal device create
//! a texture with it. `CheckDeviceMultiSampleType`, the surface creators and
//! `CreateDevice` all route through here so a request that passes the check is
//! exactly one the create paths accept.

use mtld3d_shared::mtl::DeviceCapsFlags;
use mtld3d_types::{
    D3DFMT_D16_LOCKABLE, D3DFMT_D32F_LOCKABLE, D3DFMT_DF16, D3DFMT_DF24, D3DFMT_DXT1, D3DFMT_DXT2,
    D3DFMT_DXT3, D3DFMT_DXT4, D3DFMT_DXT5, D3DFMT_INTZ, D3DMULTISAMPLE_16_SAMPLES,
    D3DMULTISAMPLE_NONE, D3DMULTISAMPLE_NONMASKABLE,
};

/// Why a `(multi_sample_type, quality)` pair cannot be served.
///
/// `Invalid` is a malformed request (a type outside the enum, or a quality
/// level that scales past 16 samples); `Unavailable` is a well-formed request
/// this device cannot satisfy. They map onto `D3DERR_INVALIDCALL` and
/// `D3DERR_NOTAVAILABLE` respectively, which is the distinction D3D9 draws.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MultiSampleReject {
    Invalid,
    Unavailable,
}

/// Sample count named by a `(D3DMULTISAMPLE_TYPE, MultiSampleQuality)` pair.
///
/// The maskable levels are their own count; `D3DMULTISAMPLE_NONMASKABLE`
/// selects `1 << quality` instead, which is how a driver exposes a ladder of
/// counts under one type.
///
/// A type outside the enumeration is malformed; a type inside it that is not a
/// power of two (`D3DMULTISAMPLE_15_SAMPLES` and its neighbours) is well
/// formed and merely unavailable, which is the distinction D3D9 draws between
/// `D3DERR_INVALIDCALL` and `D3DERR_NOTAVAILABLE`.
///
/// # Errors
///
/// Returns [`MultiSampleReject::Invalid`] for a pair no device could serve and
/// [`MultiSampleReject::Unavailable`] for a count no device does.
pub const fn sample_count_of(
    multi_sample_type: u32,
    quality: u32,
) -> Result<u32, MultiSampleReject> {
    if multi_sample_type > D3DMULTISAMPLE_16_SAMPLES {
        return Err(MultiSampleReject::Invalid);
    }
    let count = if multi_sample_type == D3DMULTISAMPLE_NONMASKABLE {
        if quality >= 5 {
            return Err(MultiSampleReject::Invalid);
        }
        1u32 << quality
    } else if multi_sample_type == D3DMULTISAMPLE_NONE {
        1
    } else {
        multi_sample_type
    };
    if count.count_ones() != 1 {
        return Err(MultiSampleReject::Unavailable);
    }
    Ok(count)
}

/// Whether a D3D9 surface format may be multisampled at all.
///
/// Block-compressed formats have no render-target path, and the lockable and
/// FOURCC readable-depth formats exist precisely so their samples can be read
/// back one by one, which a multisampled surface cannot offer. D3D9 answers
/// `D3DERR_NOTAVAILABLE` for each of them at any count above one.
#[must_use]
pub const fn format_allows_multisample(format: u32) -> bool {
    !matches!(
        format,
        D3DFMT_D16_LOCKABLE
            | D3DFMT_D32F_LOCKABLE
            | D3DFMT_INTZ
            | D3DFMT_DF24
            | D3DFMT_DF16
            | D3DFMT_DXT1
            | D3DFMT_DXT2
            | D3DFMT_DXT3
            | D3DFMT_DXT4
            | D3DFMT_DXT5
    )
}

/// Resolve a `(type, quality, format)` request against the device.
///
/// The single predicate `CheckDeviceMultiSampleType` and every create path
/// share, so a game that asks first and creates second never gets a different
/// answer the second time.
///
/// `quality` must name a level the type actually offers: one
/// (`MultiSampleQuality` 0) for every maskable type, and
/// [`nonmaskable_quality_levels`] of them under `D3DMULTISAMPLE_NONMASKABLE`.
/// D3D9 calls a quality past that end malformed rather than unavailable, since
/// `CheckDeviceMultiSampleType` told the caller how many there were.
/// `D3DMULTISAMPLE_NONE` is the exception: there is no multisampling to pick a
/// level of, so whatever the caller passed is ignored rather than rejected.
///
/// # Errors
///
/// [`MultiSampleReject::Invalid`] for a malformed request,
/// [`MultiSampleReject::Unavailable`] when the format or the device rules it
/// out.
pub fn resolve_sample_count(
    multi_sample_type: u32,
    quality: u32,
    format: u32,
    caps: DeviceCapsFlags,
) -> Result<u32, MultiSampleReject> {
    // `D3DFMT_UNKNOWN` names no surface, so there is nothing to answer for.
    if format == 0 {
        return Err(MultiSampleReject::Invalid);
    }
    if multi_sample_type != D3DMULTISAMPLE_NONE {
        let levels = if multi_sample_type == D3DMULTISAMPLE_NONMASKABLE {
            nonmaskable_quality_levels(caps)
        } else {
            1
        };
        if quality >= levels {
            return Err(MultiSampleReject::Invalid);
        }
    }
    let count = sample_count_of(multi_sample_type, quality)?;
    if count == 1 {
        return Ok(1);
    }
    if !format_allows_multisample(format) {
        return Err(MultiSampleReject::Unavailable);
    }
    if caps.supports_sample_count(count) {
        Ok(count)
    } else {
        Err(MultiSampleReject::Unavailable)
    }
}

/// Number of `D3DMULTISAMPLE_NONMASKABLE` quality levels this device offers.
///
/// Quality `q` means `1 << q` samples, so the level count is one more than the
/// exponent of the highest supported count. Always at least 1: quality 0 is
/// single-sampled, which every device can do.
#[must_use]
pub const fn nonmaskable_quality_levels(caps: DeviceCapsFlags) -> u32 {
    if caps.contains(DeviceCapsFlags::SAMPLE_COUNT_8) {
        4
    } else if caps.contains(DeviceCapsFlags::SAMPLE_COUNT_4) {
        3
    } else if caps.contains(DeviceCapsFlags::SAMPLE_COUNT_2) {
        2
    } else {
        1
    }
}

/// The value [`effective_sample_mask`] returns when no sample is masked out.
///
/// All eight bits set: Metal caps a texture at eight samples, so a mask that
/// selects every sample of any supported count is indistinguishable from this
/// and needs no shader variant.
pub const SAMPLE_MASK_ALL: u8 = 0xFF;

/// Narrow `D3DRS_MULTISAMPLEMASK` to the samples a draw actually covers.
///
/// Returns [`SAMPLE_MASK_ALL`] when the state has no effect: a
/// single-sampled target, a `D3DMULTISAMPLE_NONMASKABLE` one (the sample
/// pattern is the driver's, so there is nothing to select), or a mask that
/// already selects every sample. Anything else is the coverage the pixel
/// shader has to write to its `[[sample_mask]]` output, which is the only
/// place Metal accepts one.
#[must_use]
pub const fn effective_sample_mask(mask: u32, sample_count: u8, multi_sample_type: u32) -> u8 {
    if sample_count <= 1 || !mask_applies(multi_sample_type) {
        return SAMPLE_MASK_ALL;
    }
    let all = if sample_count >= 8 {
        SAMPLE_MASK_ALL
    } else {
        (1u8 << sample_count) - 1
    };
    // The mask is `& all` first, so every remaining bit is inside `all`'s
    // eight and the narrowing is exact.
    let narrowed = (mask & (all as u32)).to_le_bytes()[0];
    if narrowed == all {
        SAMPLE_MASK_ALL
    } else {
        narrowed
    }
}

/// Whether `D3DRS_MULTISAMPLEMASK` applies to a surface created with this type.
///
/// D3D9 defines the mask only for the maskable levels; under
/// `D3DMULTISAMPLE_NONMASKABLE` the sample pattern is the driver's, so there
/// is nothing for the mask to select and the state is ignored.
#[must_use]
pub const fn mask_applies(multi_sample_type: u32) -> bool {
    multi_sample_type > D3DMULTISAMPLE_NONMASKABLE
}

#[cfg(test)]
mod tests;
