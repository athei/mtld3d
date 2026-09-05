//! The 32-bit float filtering caps split, forced on via `intel.denyFloat32Filtering`.
//!
//! `MTLDevice.supports32BitFloatFiltering` is true on Apple-family GPUs and
//! commonly false on Intel/AMD Macs, where R32F / G32R32F / A32B32G32R32F are
//! point-sampled only. These tests force the negative answer so the caps split
//! an engine probes for runs on Apple Silicon too, instead of only on the rare
//! Intel-hardware run.

use mtld3d_tests::Harness;
use mtld3d_types::{
    D3D_OK, D3DERR_NOTAVAILABLE, D3DFMT_A8R8G8B8, D3DFMT_A16B16G16R16, D3DFMT_A16B16G16R16F,
    D3DFMT_A32B32G32R32F, D3DFMT_G16R16F, D3DFMT_G32R32F, D3DFMT_R16F, D3DFMT_R32F,
    D3DFMT_X8R8G8B8, D3DPOOL_MANAGED, D3DRTYPE_SURFACE, D3DRTYPE_TEXTURE, D3DSAMP_MAGFILTER,
    D3DSAMP_MINFILTER, D3DSAMP_MIPFILTER, D3DTEXF_LINEAR, D3DTEXF_NONE, D3DTEXF_POINT,
    D3DUSAGE_QUERY_FILTER, D3DUSAGE_RENDERTARGET, E_FAIL,
};

/// The formats `supports32BitFloatFiltering` covers.
const SINGLE_FLOATS: [(u32, &str); 3] = [
    (D3DFMT_R32F, "R32F"),
    (D3DFMT_G32R32F, "G32R32F"),
    (D3DFMT_A32B32G32R32F, "A32B32G32R32F"),
];

/// The float members that filter on every GPU family.
const HALF_FLOATS: [(u32, &str); 3] = [
    (D3DFMT_R16F, "R16F"),
    (D3DFMT_G16R16F, "G16R16F"),
    (D3DFMT_A16B16G16R16F, "A16B16G16R16F"),
];

/// The key that forces the no-32-bit-float-filtering answer.
///
/// Resolved by each harness's own `Direct3DCreate9`, so the rest of the
/// suite, sharing the process, keeps the device's own answer.
const DENY_FLOAT32_FILTERING: &str = "intel.denyFloat32Filtering=true";

#[test]
fn filter_query_is_denied_for_the_single_precision_floats() {
    let h = Harness::factory_only_with_config(DENY_FLOAT32_FILTERING);
    for (fmt, name) in SINGLE_FLOATS {
        assert_eq!(
            h.check_device_format(
                D3DFMT_X8R8G8B8,
                D3DUSAGE_QUERY_FILTER,
                D3DRTYPE_TEXTURE,
                fmt
            ),
            D3DERR_NOTAVAILABLE,
            "{name} must not be advertised as filterable",
        );
        assert_eq!(
            h.check_device_format(
                D3DFMT_X8R8G8B8,
                D3DUSAGE_RENDERTARGET | D3DUSAGE_QUERY_FILTER,
                D3DRTYPE_TEXTURE,
                fmt
            ),
            D3DERR_NOTAVAILABLE,
            "{name} render-then-filter probe follows the filter answer",
        );
    }
}

#[test]
fn only_filtering_drops_out_for_the_single_precision_floats() {
    let h = Harness::factory_only_with_config(DENY_FLOAT32_FILTERING);
    for (fmt, name) in SINGLE_FLOATS {
        assert_eq!(
            h.check_device_format(D3DFMT_X8R8G8B8, 0, D3DRTYPE_TEXTURE, fmt),
            D3D_OK,
            "{name} stays a sampleable texture format",
        );
        assert_eq!(
            h.check_device_format(
                D3DFMT_X8R8G8B8,
                D3DUSAGE_RENDERTARGET,
                D3DRTYPE_SURFACE,
                fmt
            ),
            D3D_OK,
            "{name} stays renderable",
        );
    }
}

#[test]
fn the_rest_of_the_advertised_set_still_filters() {
    let h = Harness::factory_only_with_config(DENY_FLOAT32_FILTERING);
    for (fmt, name) in HALF_FLOATS {
        assert_eq!(
            h.check_device_format(
                D3DFMT_X8R8G8B8,
                D3DUSAGE_QUERY_FILTER,
                D3DRTYPE_TEXTURE,
                fmt
            ),
            D3D_OK,
            "{name} filters on every GPU family",
        );
    }
    for (fmt, name) in [
        (D3DFMT_A8R8G8B8, "A8R8G8B8"),
        (D3DFMT_X8R8G8B8, "X8R8G8B8"),
        (D3DFMT_A16B16G16R16, "A16B16G16R16"),
    ] {
        assert_eq!(
            h.check_device_format(
                D3DFMT_X8R8G8B8,
                D3DUSAGE_QUERY_FILTER,
                D3DRTYPE_TEXTURE,
                fmt
            ),
            D3D_OK,
            "{name} filters on every GPU family",
        );
    }
}

#[test]
fn single_precision_floats_stay_creatable_and_renderable() {
    let h = Harness::with_config(DENY_FLOAT32_FILTERING);
    for (fmt, name) in SINGLE_FLOATS {
        // Unfilterable is not unusable: the create paths are unchanged, so a
        // title that samples one point-filtered still gets its texture.
        drop(h.create_texture(32, 32, 1, 0, fmt, D3DPOOL_MANAGED));
        assert_eq!(
            h.create_render_target_hr(32, 32, fmt),
            D3D_OK,
            "{name} CreateRenderTarget",
        );
    }
}

/// The sentinel a failing `ValidateDevice` must leave in the pass count.
const PASSES_SENTINEL: u32 = 0xdead_beef;

/// Set the three filters of sampler 0, asserting each write succeeds.
fn set_filters(h: &Harness, mag: u32, min: u32, mip: u32) {
    for (state, value) in [
        (D3DSAMP_MAGFILTER, mag),
        (D3DSAMP_MINFILTER, min),
        (D3DSAMP_MIPFILTER, mip),
    ] {
        assert_eq!(
            h.set_sampler_state(0, state, value),
            D3D_OK,
            "sampler state"
        );
    }
}

#[test]
fn validate_device_rejects_filtering_an_unfilterable_texture() {
    let h = Harness::with_config(DENY_FLOAT32_FILTERING);
    let float32 = h.create_texture(32, 32, 1, 0, D3DFMT_A32B32G32R32F, D3DPOOL_MANAGED);
    assert_eq!(
        h.set_texture(0, &float32),
        D3D_OK,
        "SetTexture(A32B32G32R32F)"
    );
    // Point sampling is what the device offers for this format, so it validates.
    for mip in [D3DTEXF_NONE, D3DTEXF_POINT] {
        set_filters(&h, D3DTEXF_POINT, D3DTEXF_POINT, mip);
        assert_eq!(
            h.validate_device(PASSES_SENTINEL),
            (D3D_OK, 1),
            "point sampling with mip {mip}"
        );
    }
    // Asking any of the three for a filtered fetch is the E_FAIL an engine
    // reads as "take the fallback"; the pass count stays untouched.
    for (mag, min, mip) in [
        (D3DTEXF_LINEAR, D3DTEXF_POINT, D3DTEXF_NONE),
        (D3DTEXF_POINT, D3DTEXF_LINEAR, D3DTEXF_NONE),
        (D3DTEXF_POINT, D3DTEXF_POINT, D3DTEXF_LINEAR),
    ] {
        set_filters(&h, mag, min, mip);
        assert_eq!(
            h.validate_device(PASSES_SENTINEL),
            (E_FAIL, PASSES_SENTINEL),
            "mag {mag} min {min} mip {mip} on a point-sampled format"
        );
    }
    // The rule is the bound format's, not the filters': the same trilinear
    // setup on a format the device filters validates.
    let filterable = h.create_texture(32, 32, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
    assert_eq!(
        h.set_texture(0, &filterable),
        D3D_OK,
        "SetTexture(A8R8G8B8)"
    );
    set_filters(&h, D3DTEXF_LINEAR, D3DTEXF_LINEAR, D3DTEXF_LINEAR);
    assert_eq!(
        h.validate_device(PASSES_SENTINEL),
        (D3D_OK, 1),
        "trilinear on a filterable format"
    );
    // Unbinding drops the rule with the texture: the same trilinear stage
    // with nothing bound has no format to answer for.
    assert_eq!(h.clear_texture(0), D3D_OK, "SetTexture(0, null)");
    assert_eq!(
        h.validate_device(PASSES_SENTINEL),
        (D3D_OK, 1),
        "trilinear with no texture bound"
    );
    assert_eq!(
        h.set_texture(0, &float32),
        D3D_OK,
        "SetTexture(A32B32G32R32F)"
    );
    assert_eq!(
        h.validate_device(PASSES_SENTINEL),
        (E_FAIL, PASSES_SENTINEL),
        "rebinding the point-sampled format brings the rule back"
    );
    assert_eq!(
        h.clear_texture(0),
        D3D_OK,
        "unbind before the textures drop"
    );
}
