use mtld3d_shared::{
    CreateSamplerStateParams, MetalHandle,
    mtl::{AddressMode, MinMagFilter, MipFilter},
    mtl_handle::MTLSamplerStateKind,
};
use objc2::rc::Retained;
use objc2_metal::{
    MTLCompareFunction, MTLDevice, MTLSamplerAddressMode, MTLSamplerDescriptor,
    MTLSamplerMinMagFilter, MTLSamplerMipFilter,
};

use crate::metal::handle::{IntoRetained, ReleaseRetain};

/// Sub-target for the depth-path diagnostic probe in `create_sampler_state`.
///
/// Confirms the `is_compare` flag (and the filters) reached the unix side
/// and produced a comparison sampler. Permanent probe (zero-cost when off);
/// `RUST_LOG=mtld3d::unix::depth=trace` opts in. The PE-side
/// `mtld3d::d3d9::depth` sub-target is the windows-side mirror.
const DEPTH_TRACE_TARGET: &str = "mtld3d::unix::depth";

pub fn create_sampler_state(
    params: &CreateSamplerStateParams,
) -> Option<MetalHandle<MTLSamplerStateKind>> {
    let device = params.device_handle.into_retained()?;

    let desc = MTLSamplerDescriptor::new();
    desc.setMinFilter(mtl_min_mag_filter(params.min_filter));
    desc.setMagFilter(mtl_min_mag_filter(params.mag_filter));
    desc.setMipFilter(mtl_mip_filter(params.mip_filter));
    desc.setSAddressMode(mtl_address_mode(params.address_u));
    desc.setTAddressMode(mtl_address_mode(params.address_v));
    desc.setRAddressMode(mtl_address_mode(params.address_w));
    if params.max_anisotropy > 1 {
        desc.setMaxAnisotropy(params.max_anisotropy as usize);
    }
    // D3DSAMP_MAXMIPLEVEL → setLodMinClamp. 0.0 is the default (no
    // clamp); any higher value pins the sampler to "at least mip N".
    let lod_min = f32::from_bits(params.lod_min_clamp);
    if lod_min > 0.0 {
        desc.setLodMinClamp(lod_min);
    }
    // Upper LOD clamp — fixed `1000.0f` per the D3D9 convention. Metal's
    // own default is `FLT_MAX`, so the value is semantically a no-op
    // for any plausible mip chain; the explicit assignment just makes
    // the field's intent visible in the descriptor.
    desc.setLodMaxClamp(f32::from_bits(params.lod_max_clamp));

    // Shadow-comparison sampler variant. MSL `sample_compare(...)` against
    // a `depth2d<float>` evaluates "ref <= sampled_depth" with the
    // sampler's `compareFunction`. D3D9's hardware shadow filter uses
    // LessEqual semantics (1 = lit = pixel is closer than what the light
    // occluder wrote). Without this, `sample_compare` returns 0 and the
    // PCF accumulator math in the terrain shader produces a
    // character-locked oval instead of an actual world-space shadow.
    if params.is_compare != 0 {
        desc.setCompareFunction(MTLCompareFunction::LessEqual);
    }

    // Diag probe: confirms the `is_compare` flag and the filters reached
    // the unix side and produced the comparison sampler. Once per unique
    // (is_compare, id) combo — filters are deterministic from the id so
    // they don't widen the key. Zero-cost when `mtld3d::unix::depth=trace`
    // isn't enabled.
    mtld3d_shared::log_once_trace_by!(
        target: DEPTH_TRACE_TARGET,
        key: (u64::from(params.is_compare) << 56) | params.id,
        "depth: create_sampler_state(id={:#x}) is_compare={} min={:?} mag={:?}",
        params.id,
        params.is_compare,
        params.min_filter,
        params.mag_filter
    );

    let label_kind = if params.is_compare != 0 {
        "sampcmp"
    } else {
        "samp"
    };
    let label =
        objc2_foundation::NSString::from_str(&format!("mtld3d-{label_kind}-{:#x}", params.id));
    desc.setLabel(Some(&label));

    let sampler = device.newSamplerStateWithDescriptor(&desc)?;
    // SAFETY: Retained::into_raw transfers the retain into the typed handle.
    Some(unsafe { MetalHandle::<MTLSamplerStateKind>::new(Retained::into_raw(sampler) as u64) })
}

/// Release a Metal sampler-state handle.
pub fn destroy_sampler_state(sampler_handle: u64) {
    // SAFETY: bulk-destroy thunk; PE side has dropped its only copy of `sampler_handle`.
    let handle = unsafe { MetalHandle::<MTLSamplerStateKind>::new(sampler_handle) };
    // SAFETY: just wrapped the unique canonical retain.
    unsafe { handle.release_retain() };
}

const fn mtl_min_mag_filter(wire: MinMagFilter) -> MTLSamplerMinMagFilter {
    match wire {
        MinMagFilter::Nearest => MTLSamplerMinMagFilter::Nearest,
        MinMagFilter::Linear => MTLSamplerMinMagFilter::Linear,
    }
}

const fn mtl_mip_filter(wire: MipFilter) -> MTLSamplerMipFilter {
    match wire {
        MipFilter::NotMipmapped => MTLSamplerMipFilter::NotMipmapped,
        MipFilter::Nearest => MTLSamplerMipFilter::Nearest,
        MipFilter::Linear => MTLSamplerMipFilter::Linear,
    }
}

const fn mtl_address_mode(wire: AddressMode) -> MTLSamplerAddressMode {
    match wire {
        AddressMode::ClampToEdge => MTLSamplerAddressMode::ClampToEdge,
        AddressMode::MirrorClampToEdge => MTLSamplerAddressMode::MirrorClampToEdge,
        AddressMode::Repeat => MTLSamplerAddressMode::Repeat,
        AddressMode::MirrorRepeat => MTLSamplerAddressMode::MirrorRepeat,
        AddressMode::ClampToZero => MTLSamplerAddressMode::ClampToZero,
    }
}
