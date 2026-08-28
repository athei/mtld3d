//! The 32-bit float filtering caps split, forced on via `debug.float32Filtering`.
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
    D3DFMT_X8R8G8B8, D3DPOOL_MANAGED, D3DRTYPE_SURFACE, D3DRTYPE_TEXTURE, D3DUSAGE_QUERY_FILTER,
    D3DUSAGE_RENDERTARGET,
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

/// Force the no-32-bit-float-filtering answer for this test process.
///
/// Must run before the first `Harness` (the config is read once at factory
/// bring-up). nextest runs each test in its own process, so the append is
/// test-local.
fn force_no_float32_filtering() {
    let merged = format!(
        "{};debug.float32Filtering=false",
        std::env::var("MTLD3D_CONFIG").unwrap_or_default()
    );
    // SAFETY: single-threaded at this point in the test process (the harness
    // and with it the config read are only constructed afterwards).
    unsafe { std::env::set_var("MTLD3D_CONFIG", merged) };
}

#[test]
fn filter_query_is_denied_for_the_single_precision_floats() {
    force_no_float32_filtering();
    let h = Harness::factory_only();
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
    force_no_float32_filtering();
    let h = Harness::factory_only();
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
    force_no_float32_filtering();
    let h = Harness::factory_only();
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
    force_no_float32_filtering();
    let h = Harness::new();
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
