//! Device + factory lifecycle.
//!
//! `IDirect3D9` queries, caps, `TestCooperativeLevel`, and `Reset`
//! (state-default restore, resize, malformed input).

use mtld3d_core::display_mode::MAX_SERVED_SIZES;
use mtld3d_tests::{
    Harness, HarnessConfig, TexturedVertex, WM_ACTIVATEAPP, WS_CAPTION, WS_EX_TOPMOST, WS_POPUP,
    WS_VISIBLE, assert_pixel_eq, enumerate_display_sizes,
};
use mtld3d_types::{
    D3D_OK, D3DCLEAR_TARGET, D3DCREATE_HARDWARE_VERTEXPROCESSING, D3DCREATE_NOWINDOWCHANGES,
    D3DDISPLAYMODE, D3DERR_DEVICENOTRESET, D3DERR_INVALIDCALL, D3DERR_NOTAVAILABLE, D3DFILL_SOLID,
    D3DFMT_A2R10G10B10, D3DFMT_A8B8G8R8, D3DFMT_A8R8G8B8, D3DFMT_A16B16G16R16,
    D3DFMT_A16B16G16R16F, D3DFMT_A32B32G32R32F, D3DFMT_ATI1, D3DFMT_D24S8, D3DFMT_DF24,
    D3DFMT_DXT1, D3DFMT_G16R16, D3DFMT_G16R16F, D3DFMT_G32R32F, D3DFMT_L8, D3DFMT_R5G6B5,
    D3DFMT_R8G8B8, D3DFMT_R16F, D3DFMT_R32F, D3DFMT_UYVY, D3DFMT_X8B8G8R8, D3DFMT_X8R8G8B8,
    D3DFMT_YUY2, D3DFVF_DIFFUSE, D3DFVF_TEX1, D3DFVF_XYZ, D3DOK_NOAUTOGEN, D3DPOOL_DEFAULT,
    D3DPOOL_MANAGED, D3DPOOL_SCRATCH, D3DPOOL_SYSTEMMEM, D3DPRESENT_INTERVAL_IMMEDIATE,
    D3DPRESENT_INTERVAL_ONE, D3DPRESENT_PARAMETERS, D3DPT_TRIANGLELIST, D3DRS_FILLMODE,
    D3DRS_LIGHTING, D3DRTYPE_CUBETEXTURE, D3DRTYPE_SURFACE, D3DRTYPE_TEXTURE,
    D3DSWAPEFFECT_DISCARD, D3DUSAGE_AUTOGENMIPMAP, D3DUSAGE_DEPTHSTENCIL, D3DUSAGE_DYNAMIC,
    D3DUSAGE_QUERY_FILTER, D3DUSAGE_QUERY_POSTPIXELSHADER_BLENDING, D3DUSAGE_QUERY_SRGBREAD,
    D3DUSAGE_QUERY_SRGBWRITE, D3DUSAGE_QUERY_VERTEXTEXTURE, D3DUSAGE_QUERY_WRAPANDMIP,
    D3DUSAGE_RENDERTARGET, D3DVIEWPORT9, DevCaps, TextureCaps,
};

#[test]
fn adapter_basics() {
    let h = Harness::factory_only();
    assert_eq!(h.adapter_count(), 1, "single adapter expected");

    let id = h.adapter_identifier();
    assert_ne!(id.driver[0], 0, "driver string should be populated");
    assert_ne!(
        id.description[0], 0,
        "description string should be populated"
    );

    let mut mode = D3DDISPLAYMODE {
        width: 0,
        height: 0,
        refresh_rate: 0,
        format: 0,
    };
    assert_eq!(
        h.adapter_display_mode(&mut mode),
        0,
        "GetAdapterDisplayMode"
    );
    assert!(mode.width > 0 && mode.height > 0, "display mode is empty");
    assert_eq!(mode.format, D3DFMT_X8R8G8B8, "display mode format");
}

#[test]
fn adapter_mode_enumeration() {
    let h = Harness::factory_only();
    let n = h.adapter_mode_count(D3DFMT_X8R8G8B8);
    assert!(n > 0, "GetAdapterModeCount should be > 0");

    let mut mode = D3DDISPLAYMODE {
        width: 0,
        height: 0,
        refresh_rate: 0,
        format: 0,
    };
    assert_eq!(
        h.enum_adapter_modes(D3DFMT_X8R8G8B8, 0, &mut mode),
        0,
        "EnumAdapterModes(0)"
    );
    assert!(
        mode.width > 0 && mode.height > 0,
        "enumerated mode is empty"
    );

    assert_ne!(
        h.enum_adapter_modes(D3DFMT_X8R8G8B8, n + 10, &mut mode),
        0,
        "EnumAdapterModes out-of-range must reject",
    );
}

#[test]
fn the_main_module_enumerates_the_sizes_the_adapter_serves() {
    // The test binary is the process's main module, so its own
    // EnumDisplaySettingsW import is the one d3d9 redirects: the list it
    // walks is user32's, thinned to the sizes EnumAdapterModes serves, each
    // still at every depth and rate user32 lists it. The current mode stays
    // readable through the same import.
    let h = Harness::factory_only();
    let mut served = Vec::new();
    for index in 0..h.adapter_mode_count(D3DFMT_X8R8G8B8) {
        let mut mode = D3DDISPLAYMODE {
            width: 0,
            height: 0,
            refresh_rate: 0,
            format: 0,
        };
        assert_eq!(
            h.enum_adapter_modes(D3DFMT_X8R8G8B8, index, &mut mode),
            D3D_OK,
            "EnumAdapterModes({index})"
        );
        served.push((mode.width, mode.height));
    }
    assert!(
        served.len() <= MAX_SERVED_SIZES,
        "EnumAdapterModes serves at most the bound: {served:?}"
    );

    let enumerated = enumerate_display_sizes();
    assert!(
        !enumerated.is_empty(),
        "the main module enumerates no display mode"
    );
    for size in &enumerated {
        assert!(
            served.contains(size),
            "the main module enumerated {size:?}, which EnumAdapterModes does not serve"
        );
    }
    let mut distinct = enumerated;
    distinct.sort_unstable();
    distinct.dedup();
    assert!(
        distinct.len() <= MAX_SERVED_SIZES,
        "the main module enumerates more sizes than the bound: {distinct:?}"
    );
    let current = Harness::current_display_mode();
    assert!(
        current.0 > 0 && current.1 > 0,
        "ENUM_CURRENT_SETTINGS still answers"
    );
}

#[test]
fn check_device_type_accept_and_reject() {
    let h = Harness::factory_only();
    assert_eq!(
        h.check_device_type(D3DFMT_X8R8G8B8, D3DFMT_X8R8G8B8, true),
        0,
        "X8R8G8B8 windowed device should be supported",
    );
    assert_eq!(
        h.check_device_type(D3DFMT_A2R10G10B10, D3DFMT_X8R8G8B8, true),
        D3DERR_NOTAVAILABLE,
        "A2R10G10B10 adapter format must be NOTAVAILABLE",
    );
}

#[test]
fn check_device_format_accept_and_reject() {
    let h = Harness::factory_only();
    assert_eq!(
        h.check_device_format(D3DFMT_X8R8G8B8, 0, D3DRTYPE_TEXTURE, D3DFMT_DXT1),
        0,
        "DXT1 texture should be supported",
    );
    assert_eq!(
        h.check_device_format(
            D3DFMT_X8R8G8B8,
            D3DUSAGE_DEPTHSTENCIL,
            D3DRTYPE_TEXTURE,
            D3DFMT_D24S8
        ),
        0,
        "D24S8 depth-stencil should be supported",
    );
    assert_eq!(
        h.check_device_format(D3DFMT_X8R8G8B8, 0, D3DRTYPE_TEXTURE, D3DFMT_A2R10G10B10),
        D3DERR_NOTAVAILABLE,
        "A2R10G10B10 texture must be NOTAVAILABLE",
    );
    // D3DUSAGE_AUTOGENMIPMAP needs render-target capability. A renderable
    // format succeeds; a supported-but-non-renderable format (DXT1) returns the
    // success code D3DOK_NOAUTOGEN, not D3D_OK.
    assert_eq!(
        h.check_device_format(
            D3DFMT_X8R8G8B8,
            D3DUSAGE_AUTOGENMIPMAP,
            D3DRTYPE_TEXTURE,
            D3DFMT_X8R8G8B8
        ),
        0,
        "AUTOGENMIPMAP on a renderable format is D3D_OK",
    );
    assert_eq!(
        h.check_device_format(
            D3DFMT_X8R8G8B8,
            D3DUSAGE_AUTOGENMIPMAP,
            D3DRTYPE_TEXTURE,
            D3DFMT_DXT1
        ),
        D3DOK_NOAUTOGEN,
        "AUTOGENMIPMAP on a non-renderable format is D3DOK_NOAUTOGEN",
    );
    assert_eq!(
        h.check_device_format(D3DFMT_X8R8G8B8, 0, D3DRTYPE_CUBETEXTURE, D3DFMT_DXT1),
        D3D_OK,
        "DXT1 cube sampling must agree with CreateCubeTexture",
    );
    assert_eq!(
        h.check_device_format(D3DFMT_X8R8G8B8, 0, D3DRTYPE_CUBETEXTURE, D3DFMT_ATI1),
        D3DERR_NOTAVAILABLE,
        "ATI1 cube sampling remains unavailable",
    );
    assert_eq!(
        h.check_device_format(
            D3DFMT_X8R8G8B8,
            D3DUSAGE_AUTOGENMIPMAP,
            D3DRTYPE_CUBETEXTURE,
            D3DFMT_A8R8G8B8,
        ),
        D3D_OK,
        "cube autogen is advertised for renderable color formats",
    );
    assert_eq!(
        h.check_device_format(
            D3DFMT_X8R8G8B8,
            D3DUSAGE_RENDERTARGET | D3DUSAGE_AUTOGENMIPMAP,
            D3DRTYPE_CUBETEXTURE,
            D3DFMT_A8R8G8B8,
        ),
        D3D_OK,
        "cube render-target autogen query agrees with creation",
    );
}

#[test]
fn surface_queries_reject_the_sampling_only_usage_bits() {
    // A query may only carry the usage its resource type expresses. A plain
    // D3DRTYPE_SURFACE is never bound as a shader resource, so every
    // sampling-only bit answers NOTAVAILABLE on one whatever the format,
    // while the same probe on a D3DRTYPE_TEXTURE answers on the format.
    let h = Harness::factory_only();
    for (usage, name) in [
        (D3DUSAGE_QUERY_FILTER, "QUERY_FILTER"),
        (D3DUSAGE_QUERY_SRGBREAD, "QUERY_SRGBREAD"),
        (D3DUSAGE_QUERY_VERTEXTEXTURE, "QUERY_VERTEXTEXTURE"),
        (D3DUSAGE_QUERY_WRAPANDMIP, "QUERY_WRAPANDMIP"),
        (D3DUSAGE_DYNAMIC, "DYNAMIC"),
    ] {
        assert_eq!(
            h.check_device_format(D3DFMT_X8R8G8B8, usage, D3DRTYPE_SURFACE, D3DFMT_A8R8G8B8),
            D3DERR_NOTAVAILABLE,
            "{name} is not a question a plain surface can answer",
        );
        assert_eq!(
            h.check_device_format(D3DFMT_X8R8G8B8, usage, D3DRTYPE_TEXTURE, D3DFMT_A8R8G8B8),
            D3D_OK,
            "{name} on a sampled texture answers on the format",
        );
    }
    // The bits a surface does express keep their answers: the two bindings,
    // the blend question, and the sRGB encode beside a render target.
    for (usage, name) in [
        (D3DUSAGE_RENDERTARGET, "RENDERTARGET"),
        (
            D3DUSAGE_QUERY_POSTPIXELSHADER_BLENDING,
            "QUERY_POSTPIXELSHADER_BLENDING",
        ),
        (
            D3DUSAGE_RENDERTARGET | D3DUSAGE_QUERY_SRGBWRITE,
            "RENDERTARGET | QUERY_SRGBWRITE",
        ),
    ] {
        assert_eq!(
            h.check_device_format(D3DFMT_X8R8G8B8, usage, D3DRTYPE_SURFACE, D3DFMT_A8R8G8B8),
            D3D_OK,
            "{name} stays advertised for a surface",
        );
    }
    // SRGBWRITE describes the render pass, so on its own it is not a
    // surface question.
    assert_eq!(
        h.check_device_format(
            D3DFMT_X8R8G8B8,
            D3DUSAGE_QUERY_SRGBWRITE,
            D3DRTYPE_SURFACE,
            D3DFMT_A8R8G8B8
        ),
        D3DERR_NOTAVAILABLE,
        "SRGBWRITE without RENDERTARGET is not a surface question",
    );
}

/// The D3D9 wide-channel texture formats.
///
/// 16-bit unorm plus the half- and single-precision floats.
/// Engines that render HDR internally pick a scene target from this set after
/// probing `CheckDeviceFormat`, so the probe and the create path have to give
/// the same answer for every member.
const WIDE_FORMATS: [(u32, &str); 8] = [
    (D3DFMT_G16R16, "G16R16"),
    (D3DFMT_A16B16G16R16, "A16B16G16R16"),
    (D3DFMT_R16F, "R16F"),
    (D3DFMT_G16R16F, "G16R16F"),
    (D3DFMT_A16B16G16R16F, "A16B16G16R16F"),
    (D3DFMT_R32F, "R32F"),
    (D3DFMT_G32R32F, "G32R32F"),
    (D3DFMT_A32B32G32R32F, "A32B32G32R32F"),
];

#[test]
fn check_device_format_advertises_the_wide_channel_family() {
    let h = Harness::factory_only();
    for (fmt, name) in WIDE_FORMATS {
        assert_eq!(
            h.check_device_format(D3DFMT_X8R8G8B8, 0, D3DRTYPE_TEXTURE, fmt),
            D3D_OK,
            "{name} texture must be advertised",
        );
        assert_eq!(
            h.check_device_format(
                D3DFMT_X8R8G8B8,
                D3DUSAGE_RENDERTARGET,
                D3DRTYPE_SURFACE,
                fmt
            ),
            D3D_OK,
            "{name} render target must be advertised",
        );
        assert_eq!(
            h.check_device_format(
                D3DFMT_X8R8G8B8,
                D3DUSAGE_QUERY_POSTPIXELSHADER_BLENDING,
                D3DRTYPE_TEXTURE,
                fmt
            ),
            D3D_OK,
            "{name} blends as a render target",
        );
        // No float format has an sRGB twin, so the sRGB queries stay negative.
        assert_eq!(
            h.check_device_format(
                D3DFMT_X8R8G8B8,
                D3DUSAGE_QUERY_SRGBREAD,
                D3DRTYPE_TEXTURE,
                fmt
            ),
            D3DERR_NOTAVAILABLE,
            "{name} has no sRGB decode",
        );
    }
}

#[test]
fn wide_channel_family_creates_what_check_device_format_advertises() {
    // The bug this pins: the probe answered NOTAVAILABLE for formats the
    // create paths accepted, so an engine that asks first concluded the whole
    // family was missing and shut down instead of falling back.
    let h = Harness::new();
    for (fmt, name) in WIDE_FORMATS {
        let advertised = h.check_device_format(D3DFMT_X8R8G8B8, 0, D3DRTYPE_TEXTURE, fmt);
        assert_eq!(advertised, D3D_OK, "{name} texture probe");
        // Panics with the HRESULT if the create disagrees with the probe.
        drop(h.create_texture(32, 32, 1, 0, fmt, D3DPOOL_MANAGED));

        // Cube maps too: an environment-map lookup table in G16R16 is the
        // second thing an HDR engine creates after its scene target.
        let advertised = h.check_device_format(D3DFMT_X8R8G8B8, 0, D3DRTYPE_CUBETEXTURE, fmt);
        assert_eq!(advertised, D3D_OK, "{name} cube probe");
        assert_eq!(
            h.create_cube_texture(16, 1, 0, fmt, D3DPOOL_MANAGED),
            D3D_OK,
            "{name} CreateCubeTexture",
        );

        let advertised = h.check_device_format(
            D3DFMT_X8R8G8B8,
            D3DUSAGE_RENDERTARGET,
            D3DRTYPE_SURFACE,
            fmt,
        );
        assert_eq!(advertised, D3D_OK, "{name} render-target probe");
        assert_eq!(
            h.create_render_target_hr(32, 32, fmt),
            D3D_OK,
            "{name} CreateRenderTarget",
        );
    }
}

/// The three colour formats a Source-engine title probes for its image cache.
///
/// `A8B8G8R8` / `X8B8G8R8` back Metal's native `RGBA8Unorm`, so they answer
/// yes to everything the 32-bit family does, sRGB decode included. `R8G8B8`
/// has no Metal counterpart and is widened into a BGRA8 backing by the upload
/// pass, which serves sampling but not rendering, so the render-target answers
/// stay negative and `CreateRenderTarget` is rejected to match.
#[test]
fn check_device_format_answers_for_the_reversed_channel_and_24_bit_formats() {
    let h = Harness::new();
    for (fmt, name) in [
        (D3DFMT_A8B8G8R8, "A8B8G8R8"),
        (D3DFMT_X8B8G8R8, "X8B8G8R8"),
        (D3DFMT_R8G8B8, "R8G8B8"),
    ] {
        assert_eq!(
            h.check_device_format(D3DFMT_X8R8G8B8, 0, D3DRTYPE_TEXTURE, fmt),
            D3D_OK,
            "{name} texture must be advertised",
        );
        assert_eq!(
            h.check_device_format(D3DFMT_X8R8G8B8, 0, D3DRTYPE_CUBETEXTURE, fmt),
            D3D_OK,
            "{name} cube texture must be advertised",
        );
        assert_eq!(
            h.check_device_format(
                D3DFMT_X8R8G8B8,
                D3DUSAGE_QUERY_FILTER,
                D3DRTYPE_TEXTURE,
                fmt
            ),
            D3D_OK,
            "{name} filters",
        );
        // Every one of the three is backed by an sRGB-twinned Metal format,
        // so `D3DSAMP_SRGBTEXTURE` is a real hardware decode on all of them.
        assert_eq!(
            h.check_device_format(
                D3DFMT_X8R8G8B8,
                D3DUSAGE_QUERY_SRGBREAD,
                D3DRTYPE_TEXTURE,
                fmt
            ),
            D3D_OK,
            "{name} has an sRGB decode",
        );
        // The probe and the create agree, which is the whole point of the
        // advertisement: an engine that asks first must not be told no.
        drop(h.create_texture(32, 32, 1, 0, fmt, D3DPOOL_MANAGED));
    }

    for (fmt, name) in [(D3DFMT_A8B8G8R8, "A8B8G8R8"), (D3DFMT_X8B8G8R8, "X8B8G8R8")] {
        assert_eq!(
            h.check_device_format(
                D3DFMT_X8R8G8B8,
                D3DUSAGE_RENDERTARGET,
                D3DRTYPE_SURFACE,
                fmt
            ),
            D3D_OK,
            "{name} renders",
        );
        assert_eq!(
            h.check_device_format(
                D3DFMT_X8R8G8B8,
                D3DUSAGE_RENDERTARGET | D3DUSAGE_QUERY_SRGBWRITE,
                D3DRTYPE_SURFACE,
                fmt
            ),
            D3D_OK,
            "{name} encodes sRGB on write",
        );
        assert_eq!(
            h.create_render_target_hr(32, 32, fmt),
            D3D_OK,
            "{name} CreateRenderTarget",
        );
    }

    assert_eq!(
        h.check_device_format(
            D3DFMT_X8R8G8B8,
            D3DUSAGE_RENDERTARGET,
            D3DRTYPE_SURFACE,
            D3DFMT_R8G8B8
        ),
        D3DERR_NOTAVAILABLE,
        "a format widened on upload is not a render target",
    );
    assert_ne!(
        h.create_render_target_hr(32, 32, D3DFMT_R8G8B8),
        D3D_OK,
        "CreateRenderTarget(R8G8B8) rejected to match the probe",
    );
    // A 24-bit system-memory surface is what a title locks to feed its
    // texture cache, and it is the store `GetDC` wraps a 24-bit DIB around.
    drop(h.create_offscreen_plain_surface(32, 32, D3DFMT_R8G8B8, D3DPOOL_SYSTEMMEM));
}

#[test]
fn check_format_conversion() {
    let h = Harness::factory_only();
    assert_eq!(
        h.check_device_format_conversion(D3DFMT_A8R8G8B8, D3DFMT_A8R8G8B8),
        0,
        "identity conversion should succeed",
    );
    // X8R8G8B8 and A8R8G8B8 are the same 32-bit RGB family, so this is a
    // present-compatible conversion that must succeed — consistent with the
    // CheckDeviceType format matrix that treats the X8/A8 pair as equivalent.
    assert_eq!(
        h.check_device_format_conversion(D3DFMT_A8R8G8B8, D3DFMT_X8R8G8B8),
        0,
        "X8R8G8B8 <-> A8R8G8B8 is a valid 32-bit-family conversion",
    );
    // A cross-family target (compressed DXT1) is not present-compatible.
    assert_eq!(
        h.check_device_format_conversion(D3DFMT_A8R8G8B8, D3DFMT_DXT1),
        D3DERR_NOTAVAILABLE,
        "mismatched conversion must reject",
    );
    // StretchRect converts any renderable colour format and the packed YUV
    // formats into a renderable colour format (render-quad sample/decode),
    // so the query says so, as every desktop driver does for these rows.
    assert_eq!(
        h.check_device_format_conversion(D3DFMT_R5G6B5, D3DFMT_X8R8G8B8),
        D3D_OK,
        "R5G6B5 -> X8R8G8B8 is a supported StretchRect conversion",
    );
    assert_eq!(
        h.check_device_format_conversion(D3DFMT_YUY2, D3DFMT_X8R8G8B8),
        D3D_OK,
        "YUY2 -> X8R8G8B8 is decoded by the StretchRect render quad",
    );
    // A conversion destination has to be renderable, since the quad draws into
    // it: R5G6B5 is one on a device with the packed 16-bit formats and is not
    // on one without them, and the conversion answer follows that answer.
    let r5g6b5_renders = h.check_device_format(
        D3DFMT_X8R8G8B8,
        D3DUSAGE_RENDERTARGET,
        D3DRTYPE_SURFACE,
        D3DFMT_R5G6B5,
    ) == D3D_OK;
    assert_eq!(
        h.check_device_format_conversion(D3DFMT_UYVY, D3DFMT_R5G6B5),
        if r5g6b5_renders {
            D3D_OK
        } else {
            D3DERR_NOTAVAILABLE
        },
        "UYVY -> R5G6B5 is decoded by the StretchRect render quad where R5G6B5 renders",
    );
    // Only renderable colour formats are conversion targets: YUV and
    // luminance destinations reject.
    assert_eq!(
        h.check_device_format_conversion(D3DFMT_X8R8G8B8, D3DFMT_YUY2),
        D3DERR_NOTAVAILABLE,
        "RGB -> YUY2 is not a conversion target",
    );
    assert_eq!(
        h.check_device_format_conversion(D3DFMT_X8R8G8B8, D3DFMT_L8),
        D3DERR_NOTAVAILABLE,
        "RGB -> L8 is not a conversion target",
    );
    // A format converts to itself, L8 included.
    assert_eq!(
        h.check_device_format_conversion(D3DFMT_L8, D3DFMT_L8),
        D3D_OK,
        "identity conversion holds for non-RGB formats too",
    );
}

#[test]
fn windowed_device_type_follows_format_conversion() {
    // The runtime requires windowed CheckDeviceType to equal
    // CheckDeviceFormat(RT, bb) && CheckDeviceFormatConversion(bb, display),
    // so a 16-bit windowed backbuffer on a 32-bit display is advertised
    // (CreateDevice substitutes the BGRA8 layer format for it). Fullscreen has
    // no present conversion and keeps rejecting the pair.
    let h = Harness::factory_only();
    assert_eq!(
        h.check_device_format_conversion(D3DFMT_R5G6B5, D3DFMT_X8R8G8B8),
        D3D_OK,
        "precondition: the conversion is supported",
    );
    assert_eq!(
        h.check_device_type(D3DFMT_X8R8G8B8, D3DFMT_R5G6B5, true),
        D3D_OK,
        "windowed R5G6B5 backbuffer on an X8R8G8B8 display follows the conversion predicate",
    );
    assert_eq!(
        h.check_device_type(D3DFMT_X8R8G8B8, D3DFMT_R5G6B5, false),
        D3DERR_NOTAVAILABLE,
        "fullscreen has no present conversion: the pair stays rejected",
    );
    // A conversion source that is not a render target (YUY2) is never a
    // backbuffer, windowed or not.
    assert_eq!(
        h.check_device_type(D3DFMT_X8R8G8B8, D3DFMT_YUY2, true),
        D3DERR_NOTAVAILABLE,
        "YUY2 converts but is not renderable, so it is no backbuffer",
    );
}

#[test]
fn device_caps_are_sane() {
    let h = Harness::factory_only();
    let caps = h.device_caps();
    assert!(
        caps.max_texture_width >= 4096,
        "max texture width too small"
    );
    assert!(
        caps.max_texture_height >= 4096,
        "max texture height too small"
    );
    // VS/PS at least 2.0 (high byte = major version).
    assert!(
        (caps.vertex_shader_version >> 8) & 0xFF >= 2,
        "VS version < 2.0"
    );
    assert!(
        (caps.pixel_shader_version >> 8) & 0xFF >= 2,
        "PS version < 2.0"
    );
    assert_ne!(
        caps.dev_caps & DevCaps::HWRASTERIZATION.bits(),
        0,
        "hardware rasterization not advertised"
    );
    assert_ne!(
        caps.texture_caps & TextureCaps::CUBEMAP.bits(),
        0,
        "cube maps not advertised"
    );
    assert_ne!(
        caps.texture_caps & TextureCaps::MIPCUBEMAP.bits(),
        0,
        "mipmapped cube maps not advertised"
    );
    assert_eq!(
        caps.texture_caps & TextureCaps::RESTRICTIONS.bits(),
        0,
        "no texture-creation restriction is advertised"
    );
    assert!(caps.max_streams >= 1, "no vertex streams");
    // A 2.0+ device reports its SM2 sub-structs; all-zero reads as "no
    // ps_2_x profile" to engines of that era (3DMark05 refused to start).
    assert!(
        caps.ps20_caps.num_temps >= 12,
        "PS20Caps.NumTemps below the ps_2_0 floor"
    );
    assert!(
        caps.ps20_caps.num_instruction_slots >= 96,
        "PS20Caps.NumInstructionSlots below the ps_2_0 floor"
    );
    assert_ne!(caps.ps20_caps.caps, 0, "PS20Caps.Caps is empty");
    assert!(
        caps.vs20_caps.num_temps >= 12,
        "VS20Caps.NumTemps below the vs_2_0 floor"
    );
    assert_eq!(
        caps.cube_texture_filter_caps, caps.texture_filter_caps,
        "cube filter caps differ from the 2D ones"
    );
    assert_eq!(
        caps.volume_texture_filter_caps, caps.texture_filter_caps,
        "volume filter caps differ from the 2D ones"
    );
    assert_eq!(
        caps.volume_texture_address_caps, caps.texture_address_caps,
        "volume address caps differ from the 2D ones"
    );
    assert_ne!(
        caps.texture_caps & TextureCaps::VOLUMEMAP.bits(),
        0,
        "volume maps not advertised"
    );
    // Both honoured presentation intervals are advertised; IMMEDIATE is a hard
    // requirement of 3DMark05's startup check.
    assert_ne!(
        caps.presentation_intervals & D3DPRESENT_INTERVAL_IMMEDIATE,
        0,
        "IMMEDIATE presentation interval not advertised"
    );
    assert_ne!(
        caps.presentation_intervals & D3DPRESENT_INTERVAL_ONE,
        0,
        "display-rate presentation interval not advertised"
    );
    // Vertex texture fetch: the caps bit and the per-format probe must
    // agree, and both now advertise it (titles gate whole effect paths on
    // the pair).
    assert_ne!(
        caps.vertex_texture_filter_caps, 0,
        "VTF filter caps not advertised"
    );
    assert_eq!(
        h.check_device_format(
            D3DFMT_X8R8G8B8,
            D3DUSAGE_QUERY_VERTEXTEXTURE,
            D3DRTYPE_TEXTURE,
            D3DFMT_A8R8G8B8
        ),
        0,
        "QUERY_VERTEXTEXTURE denied despite VTF caps"
    );
}

#[test]
fn cooperative_level_ok() {
    let h = Harness::new();
    assert_eq!(
        h.test_cooperative_level(),
        0,
        "device should be cooperative"
    );
}

#[test]
fn reset_rejects_outstanding_default_pool_resources() {
    // D3D9 rejects Reset while the app still references a D3DPOOL_DEFAULT
    // resource or an implicit surface, and TestCooperativeLevel reports
    // DEVICENOTRESET until a later Reset succeeds.
    let h = Harness::new();
    let vb = h.create_vertex_buffer(64, 0, D3DFVF_XYZ, D3DPOOL_DEFAULT);
    assert_eq!(
        h.reset(640, 480),
        D3DERR_INVALIDCALL,
        "a referenced DEFAULT-pool vertex buffer blocks Reset"
    );
    assert_eq!(
        h.test_cooperative_level(),
        D3DERR_DEVICENOTRESET,
        "a failed Reset latches DEVICENOTRESET"
    );
    drop(vb);
    assert_eq!(h.reset(640, 480), D3D_OK, "Reset succeeds once released");
    assert_eq!(
        h.test_cooperative_level(),
        D3D_OK,
        "a successful Reset clears the latch"
    );

    let backbuffer = h.back_buffer(0);
    assert_eq!(
        h.reset(640, 480),
        D3DERR_INVALIDCALL,
        "a held implicit back buffer blocks Reset"
    );
    drop(backbuffer);
    assert_eq!(h.reset(640, 480), D3D_OK, "Reset succeeds once released");

    // Other pools never block, and neither does the device's own binding of
    // a DEFAULT resource the app has released.
    let managed = h.create_vertex_buffer(64, 0, D3DFVF_XYZ, D3DPOOL_MANAGED);
    let sysmem = h.create_offscreen_plain_surface(16, 16, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    let bound = h.create_texture(16, 16, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    assert_eq!(h.set_texture(0, &bound), 0);
    drop(bound);
    assert_eq!(
        h.reset(640, 480),
        D3D_OK,
        "MANAGED / SYSTEMMEM resources and device-held bindings do not block Reset"
    );
    drop(managed);
    drop(sysmem);
}

#[test]
fn reset_bad_dims_rejected() {
    let h = Harness::new();
    // A *fullscreen* Reset must carry explicit dimensions — zero dims are
    // rejected. (A windowed zero-dimension Reset instead resolves against the
    // device window's client rect and succeeds, matching D3D9, so it is NOT a
    // rejection path.)
    let mut pp = D3DPRESENT_PARAMETERS {
        back_buffer_width: 0,
        back_buffer_height: 0,
        back_buffer_format: 0,
        back_buffer_count: 1,
        multi_sample_type: 0,
        multi_sample_quality: 0,
        swap_effect: D3DSWAPEFFECT_DISCARD,
        device_window: 0,
        windowed: 0,
        enable_auto_depth_stencil: 0,
        auto_depth_stencil_format: 0,
        flags: 0,
        full_screen_refresh_rate_in_hz: 0,
        presentation_interval: 0,
    };
    assert_eq!(
        h.reset_params(&mut pp),
        D3DERR_INVALIDCALL,
        "fullscreen 0x0 Reset must be INVALIDCALL"
    );
}

#[test]
fn reset_same_size_restores_state_defaults() {
    let h = Harness::new();

    // Pollute device state, then confirm the writes stuck.
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0);
    let custom = D3DVIEWPORT9 {
        x: 100,
        y: 50,
        width: 200,
        height: 150,
        min_z: 0.25,
        max_z: 0.75,
    };
    assert_eq!(h.set_viewport(&custom), 0);
    assert_eq!(
        h.render_state(D3DRS_LIGHTING),
        0,
        "LIGHTING write should stick"
    );
    assert_eq!(h.viewport().x, 100, "viewport write should stick");

    assert_eq!(h.reset(640, 480), 0, "same-size Reset must succeed");

    // State back to D3D9 defaults.
    assert_eq!(
        h.render_state(D3DRS_LIGHTING),
        1,
        "LIGHTING default after Reset"
    );
    assert_eq!(
        h.render_state(D3DRS_FILLMODE),
        D3DFILL_SOLID,
        "FILLMODE default after Reset"
    );
    let vp = h.viewport();
    assert_eq!(
        (vp.x, vp.y, vp.width, vp.height),
        (0, 0, 640, 480),
        "viewport reset to full target"
    );
    assert_eq!(
        vp.min_z.to_bits(),
        0.0_f32.to_bits(),
        "viewport min_z default"
    );
    assert_eq!(
        vp.max_z.to_bits(),
        1.0_f32.to_bits(),
        "viewport max_z default"
    );
    assert!(
        h.texture_raw(0).is_null(),
        "stage-0 texture unbound after Reset"
    );

    // Device still renders after Reset (backbuffer recreated).
    let red = 0xFFFF_0000;
    h.render_once(red, |_| {});
    assert_pixel_eq(h.read_pixel(320, 240), red, "renders after Reset");
}

#[test]
fn reset_clears_scene_state() {
    let h = Harness::new();

    // A normal pair still works — the scene flag tracks Begin/End correctly.
    assert_eq!(h.begin_scene(), 0, "BeginScene must succeed");
    assert_eq!(h.end_scene(), 0, "EndScene must succeed");

    // Reset abandons an open scene: the following EndScene has no matching
    // BeginScene and must fail.
    assert_eq!(h.begin_scene(), 0, "BeginScene before Reset");
    assert_eq!(h.reset(640, 480), 0, "same-size Reset must succeed");
    assert_eq!(
        h.end_scene(),
        D3DERR_INVALIDCALL,
        "EndScene after Reset must be INVALIDCALL"
    );
}

/// A full-target quad whose texture coordinates address a cube's +X face.
///
/// The whole face carries one colour, so the sampled texel does not depend on
/// where the coordinate lands within it.
const fn cube_face_quad() -> [CubeQuadVertex; 6] {
    [
        cube_quad_vertex(-1.0, 1.0),
        cube_quad_vertex(1.0, 1.0),
        cube_quad_vertex(-1.0, -1.0),
        cube_quad_vertex(1.0, 1.0),
        cube_quad_vertex(1.0, -1.0),
        cube_quad_vertex(-1.0, -1.0),
    ]
}

/// A full-target quad with UVs spanning the unit square and one vertex colour.
fn flat_quad(color: u32) -> [TexturedVertex; 6] {
    let v = |x: f32, y: f32, u: f32, v: f32| TexturedVertex {
        x,
        y,
        z: 0.5,
        color,
        u,
        v,
    };
    [
        v(-1.0, 1.0, 0.0, 0.0),
        v(1.0, 1.0, 1.0, 0.0),
        v(-1.0, -1.0, 0.0, 1.0),
        v(1.0, 1.0, 1.0, 0.0),
        v(1.0, -1.0, 1.0, 1.0),
        v(-1.0, -1.0, 0.0, 1.0),
    ]
}

#[repr(C)]
struct CubeQuadVertex {
    x: f32,
    y: f32,
    z: f32,
    color: u32,
    u: f32,
    v: f32,
    w: f32,
}

const fn cube_quad_vertex(x: f32, y: f32) -> CubeQuadVertex {
    CubeQuadVertex {
        x,
        y,
        z: 0.5,
        color: 0xFFFF_FFFF,
        u: 1.0,
        v: 0.0,
        w: 0.0,
    }
}

/// An upload queued before a same-size `Reset` still reaches the GPU.
///
/// A draw's bind-time flush queues the texture upload onto the pending frame
/// and clears the mip's dirty bit; with no `Present` in between, that op is
/// still queued when `Reset` replaces the frame. Dropping it with the
/// bookkeeping already advanced loses the level's content on the GPU for
/// good: the game believes it uploaded and never rewrites it. HL2's cached
/// VGUI text meshes ride exactly this queue through the same-size Reset its
/// windowed toggle issues, which is issue #76's garbled menu text.
#[test]
fn reset_same_size_keeps_uploads_queued_before_it() {
    const RED: u32 = 0xFFFF_0000;
    const BLACK: u32 = 0xFF00_0000;

    let h = Harness::new();
    let tex = h.create_texture(4, 4, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
    {
        let mut level = tex.lock_rect(0, 0);
        level.write_u32(&[RED; 16]);
    }
    // Bind and draw without presenting: the draw schedules the upload onto
    // the pending frame and spends the dirty bit.
    assert_eq!(h.set_texture(0, &tex), 0, "SetTexture before Reset");
    h.select_texture_stage(0);
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0, "LIGHTING off");
    assert_eq!(
        h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1),
        0,
        "SetFVF for the pre-Reset draw"
    );
    assert_eq!(h.begin_scene(), 0, "BeginScene before Reset");
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &flat_quad(0xFFFF_FFFF)),
        0,
        "pre-Reset draw"
    );
    assert_eq!(h.end_scene(), 0, "EndScene before Reset");

    assert_eq!(h.reset(640, 480), 0, "same-size Reset must succeed");

    // The dirty bit is spent, so only the pre-Reset upload can have put the
    // texels on the GPU; a Reset that dropped the queued op samples nothing.
    assert_eq!(h.set_texture(0, &tex), 0, "SetTexture after Reset");
    h.select_texture_stage(0);
    assert_eq!(
        h.set_render_state(D3DRS_LIGHTING, 0),
        0,
        "LIGHTING off again"
    );
    assert_eq!(
        h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1),
        0,
        "SetFVF for the post-Reset draw"
    );
    h.render_once(BLACK, |d| {
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &flat_quad(0xFFFF_FFFF)),
            0,
            "post-Reset draw"
        );
    });
    assert_pixel_eq(
        h.read_pixel(320, 240),
        RED,
        "the upload queued before the Reset must survive it",
    );
}

#[test]
fn reset_clears_the_stage_cube_binding_mask() {
    const RED: u32 = 0xFFFF_0000;
    const GREEN: u32 = 0xFF00_FF00;
    const BLUE: u32 = 0xFF00_00FF;
    const BLACK: u32 = 0xFF00_0000;
    // D3DFVF_TEXCOORDSIZE3(0), the three-component texcoord a cube sample needs.
    const TEXCOORDSIZE3_0: u32 = 0x0001_0000;

    let h = Harness::new();

    // A cube on stage 0, sampled once so the stage's cached cube bit is live
    // going into the Reset. Managed pool, so the texture may outlive the Reset.
    let cube = h.create_cube_texture_owned(4, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
    {
        let mut face = cube.lock_rect(0, 0, 0);
        face.write_u32(&[RED; 16]);
    }
    assert_eq!(h.set_cube_texture(0, &cube), 0, "SetTexture(cube)");
    h.select_texture_stage(0);
    assert_eq!(
        h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1 | TEXCOORDSIZE3_0),
        0,
        "SetFVF for the cube draw"
    );
    h.render_once(BLACK, |d| {
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &cube_face_quad()),
            0,
            "cube draw"
        );
    });
    assert_pixel_eq(h.read_pixel(320, 240), RED, "cube sample before Reset");

    assert_eq!(h.reset(640, 480), 0, "same-size Reset must succeed");

    // Reset unbinds every stage, and the first draw after it rebuilds the
    // shader variant key from the per-stage kind masks. A cube bit carried
    // over from before the Reset describes a stage that now holds nothing.
    h.select_diffuse_stage(0);
    // Reset restores the D3D9 default `D3DRS_LIGHTING = TRUE`, which the
    // normal-less vertices below would light to black.
    assert_eq!(
        h.set_render_state(D3DRS_LIGHTING, 0),
        0,
        "SetRenderState(LIGHTING)"
    );
    assert_eq!(
        h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1),
        0,
        "SetFVF for the diffuse draw"
    );
    h.render_once(BLACK, |d| {
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &flat_quad(BLUE)),
            0,
            "diffuse draw with stage 0 unbound"
        );
    });
    assert_pixel_eq(h.read_pixel(320, 240), BLUE, "diffuse draw after Reset");

    // The stage the cube held samples a 2D texture as a 2D texture.
    let flat = h.create_texture(4, 4, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
    {
        let mut level = flat.lock_rect(0, 0);
        level.write_u32(&[GREEN; 16]);
    }
    assert_eq!(h.set_texture(0, &flat), 0, "SetTexture(2D)");
    h.select_texture_stage(0);
    h.render_once(BLACK, |d| {
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &flat_quad(0xFFFF_FFFF)),
            0,
            "2D draw"
        );
    });
    assert_pixel_eq(h.read_pixel(320, 240), GREEN, "2D sample after Reset");
}

#[test]
fn present_after_resize_reset_without_drawing_reads_black() {
    let h = Harness::new();

    // Fill the current backbuffer with a loud colour and present it, so the
    // device heap holds recycled non-zero memory when the Reset below
    // recreates the backbuffer.
    h.render_once(0xFFFF_00FF, |_| {});

    // A resized Reset destroys the old backbuffer and creates a fresh
    // texture; presenting before any draw or clear publishes it as-is (a
    // scene transition routinely does exactly this). The creation-time
    // clear must make that frame opaque black, not whatever memory the
    // allocation recycled. Note the failure is only guaranteed to
    // reproduce when the heap actually recycles dirty pages, hence the
    // magenta frame above.
    assert_eq!(h.reset(512, 384), 0, "resize Reset must succeed");
    assert_eq!(h.present(), 0, "Present with no draws must succeed");

    for (x, y) in [(0, 0), (511, 0), (0, 383), (511, 383), (256, 192)] {
        assert_pixel_eq(
            h.read_pixel(x, y),
            0xFF00_0000,
            &format!("undrawn post-Reset backbuffer at ({x},{y})"),
        );
    }
}

#[test]
fn reset_resize_grows_backbuffer() {
    let h = Harness::new();
    assert_eq!(h.reset(800, 600), 0, "resize Reset must succeed");
    assert_eq!(h.dims(), (800, 600), "harness tracks new dims");

    let vp = h.viewport();
    assert_eq!((vp.width, vp.height), (800, 600), "viewport follows resize");

    let blue = 0xFF00_00FF;
    h.render_once(blue, |_| {});
    assert_pixel_eq(h.read_pixel(400, 300), blue, "new center renders");
    // (700,500) only exists in the grown 800x600 backbuffer.
    assert_pixel_eq(h.read_pixel(700, 500), blue, "grown backbuffer reachable");
}

/// Whether the display lists the 640x480 mode the fullscreen tests request.
///
/// A fullscreen create or Reset sets the mode through user32, which accepts
/// only a mode the display lists, so on a display with a single mode (a
/// runner's virtual display) the request is a non-mode one and the back
/// buffer follows the window instead; that path has its own test, and the
/// mode tests have nothing to measure there.
fn display_lists_640x480() -> bool {
    enumerate_display_sizes().contains(&(640, 480))
}

/// Present parameters for a fullscreen Reset at `width`x`height`.
const fn fullscreen_params(hwnd: usize, width: u32, height: u32) -> D3DPRESENT_PARAMETERS {
    D3DPRESENT_PARAMETERS {
        back_buffer_width: width,
        back_buffer_height: height,
        back_buffer_format: D3DFMT_X8R8G8B8,
        back_buffer_count: 1,
        multi_sample_type: 0,
        multi_sample_quality: 0,
        swap_effect: D3DSWAPEFFECT_DISCARD,
        device_window: hwnd,
        windowed: 0,
        enable_auto_depth_stencil: 0,
        auto_depth_stencil_format: 0,
        flags: 0,
        full_screen_refresh_rate_in_hz: 0,
        presentation_interval: 0,
    }
}

#[test]
fn reset_fullscreen_adopts_monitor_rect_and_restores() {
    if !display_lists_640x480() {
        return;
    }
    let h = Harness::new();
    let hwnd = h.hwnd();
    let windowed_rect = h.window_rect();
    let windowed_style = h.window_style();

    // 640x480 is a settable mode (one user32 accepts), so the Reset sets it and the monitor
    // rect the window adopts is the mode's. Read after the transition: the
    // metric answers in the mode while one is set.
    let mut pp = fullscreen_params(hwnd, 640, 480);
    assert_eq!(h.reset_params(&mut pp), D3D_OK, "fullscreen Reset");
    let (screen_w, screen_h) = Harness::screen_size();

    let rect = h.window_rect();
    assert_eq!(
        (
            u32::try_from(rect.right - rect.left).expect("width is positive"),
            u32::try_from(rect.bottom - rect.top).expect("height is positive")
        ),
        (screen_w, screen_h),
        "fullscreen device window must fill the monitor rect",
    );
    let style = h.window_style();
    assert_ne!(style & WS_POPUP, 0, "fullscreen window must be a popup");
    assert_eq!(
        style & WS_CAPTION,
        0,
        "fullscreen window must lose its caption"
    );
    // Deliberately *not* topmost: raising the window's level deadlocks Wine's
    // mac driver (see `fullscreen::apply_fullscreen_window`), and a borderless
    // window covering the monitor needs no help from the z-order.
    assert_eq!(
        h.window_exstyle() & WS_EX_TOPMOST,
        0,
        "fullscreen window must leave the z-order alone",
    );

    let (bb_hr, bb) = h.back_buffer(0).desc();
    assert_eq!(bb_hr, D3D_OK, "GetDesc on the fullscreen backbuffer");
    assert_eq!(
        (bb.width, bb.height),
        (640, 480),
        "backbuffer is the requested mode; the window covers the monitor, which is the mode",
    );

    // Back to windowed: the window we took over is handed back as it was.
    assert_eq!(h.reset(640, 480), D3D_OK, "windowed Reset");
    assert_eq!(
        h.window_rect(),
        windowed_rect,
        "leaving fullscreen restores the window rect",
    );
    assert_eq!(
        h.window_style() & !WS_VISIBLE,
        windowed_style & !WS_VISIBLE,
        "leaving fullscreen restores the window style",
    );
}

/// `D3DCREATE_NOWINDOWCHANGES` hands window management to the app.
///
/// A fullscreen Reset must then leave the device window's style, rect and
/// visibility exactly as the app left them, and the windowed Reset back must
/// not show a window the app kept hidden.
#[test]
fn nowindowchanges_leaves_the_device_window_alone() {
    if !display_lists_640x480() {
        return;
    }
    let h = Harness::create(&HarnessConfig {
        behavior_flags: D3DCREATE_HARDWARE_VERTEXPROCESSING | D3DCREATE_NOWINDOWCHANGES,
        ..HarnessConfig::default()
    });
    let windowed_rect = h.window_rect();
    let windowed_style = h.window_style();
    let windowed_exstyle = h.window_exstyle();
    assert_eq!(
        windowed_style & WS_VISIBLE,
        0,
        "the harness window starts hidden, which is what this test turns on",
    );

    let mut pp = fullscreen_params(h.hwnd(), 640, 480);
    assert_eq!(h.reset_params(&mut pp), D3D_OK, "fullscreen Reset");

    assert_eq!(
        h.window_rect(),
        windowed_rect,
        "NOWINDOWCHANGES: a fullscreen Reset must not move the device window",
    );
    assert_eq!(
        h.window_style(),
        windowed_style,
        "NOWINDOWCHANGES: a fullscreen Reset must not restyle or show the device window",
    );
    assert_eq!(
        h.window_exstyle(),
        windowed_exstyle,
        "NOWINDOWCHANGES: a fullscreen Reset must not touch the extended style",
    );

    // The back buffer still follows the D3D9 contract: the window is the
    // app's, the mode is ours.
    let (bb_hr, bb) = h.back_buffer(0).desc();
    assert_eq!(bb_hr, D3D_OK, "GetDesc on the fullscreen backbuffer");
    assert_eq!(
        (bb.width, bb.height),
        (640, 480),
        "back buffer keeps the requested mode even when the window is untouched",
    );

    assert_eq!(h.reset(640, 480), D3D_OK, "windowed Reset");
    assert_eq!(
        h.window_style(),
        windowed_style,
        "NOWINDOWCHANGES: leaving fullscreen must not show a window the app kept hidden",
    );
    assert_eq!(
        h.window_rect(),
        windowed_rect,
        "NOWINDOWCHANGES: leaving fullscreen must not move the device window",
    );
}

#[test]
fn reset_fullscreen_honors_a_settable_mode() {
    if !display_lists_640x480() {
        return;
    }
    let h = Harness::new();
    // 640x480 is a mode user32 accepts (whether or not the bounded list
    // EnumAdapterModes serves carries it), so the Reset sets that mode and
    // the back buffer keeps the requested size; a game that sizes its viewport
    // from its own request covers the frame. Present scales the back buffer
    // to the drawable, which stays at the display's size.
    let mut pp = fullscreen_params(h.hwnd(), 640, 480);
    assert_eq!(
        h.reset_params(&mut pp),
        D3D_OK,
        "fullscreen Reset at a settable mode must succeed",
    );
    assert_eq!(
        (pp.back_buffer_width, pp.back_buffer_height),
        (640, 480),
        "Reset must report the requested mode back unchanged",
    );
    let (bb_hr, bb) = h.back_buffer(0).desc();
    assert_eq!(bb_hr, D3D_OK, "GetDesc on the fullscreen backbuffer");
    assert_eq!(
        (bb.width, bb.height),
        (640, 480),
        "back buffer keeps the requested mode, not the window's size",
    );
    let vp = h.viewport();
    assert_eq!(
        (vp.width, vp.height),
        (640, 480),
        "default viewport covers the requested back buffer",
    );
    let (screen_w, screen_h) = Harness::screen_size();
    let rect = h.window_rect();
    assert_eq!(
        (
            u32::try_from(rect.right - rect.left).expect("width is positive"),
            u32::try_from(rect.bottom - rect.top).expect("height is positive")
        ),
        (screen_w, screen_h),
        "the window covers the monitor, which is the mode while one is set",
    );

    // A second fullscreen Reset at another mode takes the recreate path: the
    // resized gate compares against the honored (not window) size.
    let mut pp = fullscreen_params(h.hwnd(), 800, 600);
    assert_eq!(
        h.reset_params(&mut pp),
        D3D_OK,
        "fullscreen Reset to a second mode must succeed",
    );
    let (bb_hr, bb) = h.back_buffer(0).desc();
    assert_eq!(bb_hr, D3D_OK, "GetDesc after the second fullscreen Reset");
    assert_eq!(
        (bb.width, bb.height),
        (800, 600),
        "an in-game mode change recreates the back buffer at the new request",
    );
}

#[test]
fn reset_fullscreen_non_mode_request_follows_the_window() {
    let h = Harness::new();
    let (screen_w, screen_h) = Harness::screen_size();
    // 137x101 is in no display-mode list, so no game can depend on it being
    // honored: native would reject the request outright. Games that ask for
    // sizes like this carried their window size into the request (WoW's
    // windowed-to-fullscreen toggle) and size their rendering and mouse
    // handling from the window, so the back buffer follows the client rect
    // and the resolved size is reported back through the present params.
    let mut pp = fullscreen_params(h.hwnd(), 137, 101);
    assert_eq!(
        h.reset_params(&mut pp),
        D3D_OK,
        "fullscreen Reset at a non-mode size must still succeed",
    );
    assert_eq!(
        (pp.back_buffer_width, pp.back_buffer_height),
        (screen_w, screen_h),
        "Reset must report the size it actually used",
    );
    let (bb_hr, bb) = h.back_buffer(0).desc();
    assert_eq!(bb_hr, D3D_OK, "GetDesc on the fullscreen backbuffer");
    assert_eq!(
        (bb.width, bb.height),
        (screen_w, screen_h),
        "a non-mode request follows the monitor-covering window",
    );
}

#[test]
fn create_fullscreen_honors_the_requested_resolution() {
    if !display_lists_640x480() {
        return;
    }
    let h = Harness::fullscreen(640, 480);

    // Read after the create: the metric answers in the mode while one is set.
    let (screen_w, screen_h) = Harness::screen_size();
    let rect = h.window_rect();
    assert_eq!(
        (
            u32::try_from(rect.right - rect.left).expect("width is positive"),
            u32::try_from(rect.bottom - rect.top).expect("height is positive")
        ),
        (screen_w, screen_h),
        "a fullscreen create covers the monitor, which is the mode",
    );
    let (bb_hr, bb) = h.back_buffer(0).desc();
    assert_eq!(bb_hr, D3D_OK, "GetDesc on the fullscreen backbuffer");
    assert_eq!(
        (bb.width, bb.height),
        (640, 480),
        "back buffer keeps the requested mode",
    );
    let mut mode = D3DDISPLAYMODE {
        width: 0,
        height: 0,
        refresh_rate: 0,
        format: 0,
    };
    assert_eq!(h.display_mode(&mut mode), D3D_OK, "GetDisplayMode");
    assert_eq!(
        (mode.width, mode.height),
        (640, 480),
        "GetDisplayMode reports the requested mode",
    );
}

#[test]
fn fullscreen_window_reasserts_monitor_rect_after_external_resize() {
    if !display_lists_640x480() {
        return;
    }
    let h = Harness::fullscreen(640, 480);
    let (screen_w, screen_h) = Harness::screen_size();

    // The move a self-managing game makes after a mode change: apply the
    // mode's outer rect to its own window (GMmark2 does exactly this after
    // every fullscreen Reset). Native D3D9 leaves the app-set rect in
    // place until window events are processed and only then restores the
    // monitor rect, so the re-cover must not fire synchronously inside
    // the SetWindowPos call.
    mtld3d_tests::set_window_pos(h.hwnd(), 0, 0, 520, 418);
    let rect = h.window_rect();
    assert_eq!(
        (rect.right - rect.left, rect.bottom - rect.top),
        (520, 418),
        "the app-set rect survives its own SetWindowPos call",
    );

    // Coverage, not equality: this is the first rect in the test the window
    // manager has had a chance to answer, and its answer can be a pixel
    // larger than the monitor when a monitor dimension does not land on its
    // coordinate grid. Such a window covers the display, which is what a
    // fullscreen window owes, so the assertion reads the same rule the
    // re-cover does.
    assert!(h.pump(), "no WM_QUIT expected");
    let rect = h.window_rect();
    let covered = (
        u32::try_from(rect.right - rect.left).expect("width is positive"),
        u32::try_from(rect.bottom - rect.top).expect("height is positive"),
    );
    assert!(
        mtld3d_core::fullscreen_resize::covers_monitor(covered, (screen_w, screen_h)),
        "processing window events re-covers the monitor: {covered:?} against ({screen_w}, {screen_h})",
    );
    let (bb_hr, bb) = h.back_buffer(0).desc();
    assert_eq!(bb_hr, D3D_OK, "GetDesc after the external resize");
    assert_eq!(
        (bb.width, bb.height),
        (640, 480),
        "the back buffer never follows an external resize",
    );
}

/// A fullscreen device sets the requested mode through user32, like native.
///
/// The test prefix runs with Wine's `EmulateModeset`, so the mode-set is
/// virtual: win32u answers every metric in the mode and maps mouse input
/// into it. The invariant that keeps a game's clicks on its UI is the client
/// rect equalling the back buffer, which only a mode-set can produce while
/// the window covers the monitor.
#[test]
fn reset_fullscreen_sets_the_display_mode() {
    if !display_lists_640x480() {
        return;
    }
    let h = Harness::new();
    let native = Harness::current_display_mode();
    let windowed_client = h.client_size();
    assert_ne!(
        native,
        (640, 480),
        "the desktop is not already at the test mode"
    );

    let mut pp = fullscreen_params(h.hwnd(), 640, 480);
    assert_eq!(h.reset_params(&mut pp), D3D_OK, "fullscreen Reset");
    assert_eq!(
        Harness::current_display_mode(),
        (640, 480),
        "a fullscreen Reset at a settable mode sets that display mode",
    );
    assert_eq!(
        Harness::screen_size(),
        (640, 480),
        "GetSystemMetrics answers in the mode",
    );
    assert_eq!(
        h.client_size(),
        (640, 480),
        "the client rect is the mode, so mouse coordinates arrive in back-buffer space",
    );
    let rect = h.window_rect();
    assert_eq!(
        (rect.right - rect.left, rect.bottom - rect.top),
        (640, 480),
        "the window covers the monitor, which is the mode",
    );
    let (bb_hr, bb) = h.back_buffer(0).desc();
    assert_eq!(bb_hr, D3D_OK, "GetDesc on the fullscreen backbuffer");
    assert_eq!((bb.width, bb.height), (640, 480), "back buffer is the mode");

    assert_eq!(h.reset(640, 480), D3D_OK, "windowed Reset");
    assert_eq!(
        Harness::current_display_mode(),
        native,
        "leaving fullscreen restores the registry display mode",
    );
    assert_eq!(
        h.client_size(),
        windowed_client,
        "the windowed client rect comes back with the desktop mode",
    );

    let mut pp = fullscreen_params(h.hwnd(), 640, 480);
    assert_eq!(h.reset_params(&mut pp), D3D_OK, "second fullscreen Reset");
    assert_eq!(
        Harness::current_display_mode(),
        (640, 480),
        "mode set again"
    );
    drop(h);
    assert_eq!(
        Harness::current_display_mode(),
        native,
        "releasing a fullscreen device restores the registry display mode",
    );
}

/// The focus half of the mode contract: restore on deactivation, re-set on activation.
#[test]
fn fullscreen_device_restores_the_mode_on_deactivation_and_re_sets_it_on_activation() {
    if !display_lists_640x480() {
        return;
    }
    let native = Harness::current_display_mode();
    let h = Harness::fullscreen(640, 480);
    assert_eq!(
        Harness::current_display_mode(),
        (640, 480),
        "fullscreen create sets the mode"
    );

    h.send_window_message(WM_ACTIVATEAPP, 0, 0);
    assert_eq!(
        Harness::current_display_mode(),
        native,
        "WM_ACTIVATEAPP FALSE puts the registry display mode back",
    );

    // The re-set is posted from the activation and runs when the game next
    // pumps messages, so it can never run inside a Reset in flight.
    h.send_window_message(WM_ACTIVATEAPP, 1, 0);
    assert!(h.pump(), "no WM_QUIT expected");
    assert_eq!(
        Harness::current_display_mode(),
        (640, 480),
        "WM_ACTIVATEAPP TRUE re-sets the device's mode",
    );
    let rect = h.window_rect();
    assert_eq!(
        (rect.right - rect.left, rect.bottom - rect.top),
        (640, 480),
        "the window covers the monitor again",
    );
}

#[test]
fn reset_balances_device_refcount() {
    // Reset must not leak a device reference. A leak would mean the device's
    // refcount never returns to zero after a resolution change, so it could
    // never be destroyed (WoW resets on resolution change).
    let h = Harness::new();
    let base = h.device_refcount();
    assert_eq!(h.reset(640, 480), 0, "same-size Reset");
    assert_eq!(
        h.device_refcount(),
        base,
        "same-size Reset must not leak a device reference",
    );
    assert_eq!(h.reset(800, 600), 0, "resize Reset");
    assert_eq!(
        h.device_refcount(),
        base,
        "resize Reset must not leak a device reference",
    );
}

#[test]
fn set_cursor_properties_rejects_oversize() {
    // A cursor bitmap larger than the adapter display mode is rejected with
    // D3DERR_INVALIDCALL, while an in-bounds bitmap is accepted. The bound
    // check sizes the cursor relative to GetAdapterDisplayMode (the desktop
    // resolution, not the backbuffer).
    let h = Harness::new();

    let mut mode = D3DDISPLAYMODE {
        width: 0,
        height: 0,
        refresh_rate: 0,
        format: 0,
    };
    assert_eq!(
        h.adapter_display_mode(&mut mode),
        D3D_OK,
        "GetAdapterDisplayMode",
    );

    // Largest power-of-two width within the display mode; doubling it exceeds
    // the mode regardless of the host resolution.
    let mut fit_w = 1u32;
    while fit_w * 2 <= mode.width {
        fit_w *= 2;
    }

    let small = h.create_offscreen_plain_surface(32, 32, D3DFMT_A8R8G8B8, D3DPOOL_SCRATCH);
    assert_eq!(
        h.set_cursor_properties_hr(0, 0, &small),
        D3D_OK,
        "in-bounds 32x32 cursor must be accepted",
    );

    let oversize =
        h.create_offscreen_plain_surface(fit_w * 2, 32, D3DFMT_A8R8G8B8, D3DPOOL_SCRATCH);
    assert_eq!(
        h.set_cursor_properties_hr(0, 0, &oversize),
        D3DERR_INVALIDCALL,
        "cursor wider than the display mode must be rejected",
    );
}

#[test]
fn show_cursor_previous_state_survives_wm_size() {
    // A macdrv-posted WM_SIZE arms the cursor module's post-resize visibility
    // pin (keeps the physical cursor up across WoW's bogus post-resize hide).
    // The pin must not leak into ShowCursor's previous-state bookkeeping: the
    // first ShowCursor(TRUE) after SetCursorProperties reports the cursor
    // hidden.
    const WM_SIZE: u32 = 0x0005;
    let h = Harness::new();

    // Same-size WM_SIZE (lparam = client height << 16 | width): arms the pin
    // without churning the backbuffer (apply_auto_resize no-ops on equal dims).
    let (width, height): (isize, isize) = (640, 480);
    h.send_window_message(WM_SIZE, 0, (height << 16) | width);

    let cursor = h.create_offscreen_plain_surface(32, 32, D3DFMT_A8R8G8B8, D3DPOOL_SCRATCH);
    assert_eq!(h.set_cursor_properties_hr(0, 0, &cursor), D3D_OK);
    assert_eq!(
        h.show_cursor(true),
        0,
        "first ShowCursor(TRUE) must report the cursor previously hidden \
         even after a WM_SIZE armed the post-resize pin",
    );
    assert_eq!(
        h.show_cursor(true),
        1,
        "second ShowCursor(TRUE) reports the cursor previously shown",
    );
    assert_eq!(
        h.show_cursor(false),
        1,
        "ShowCursor(FALSE) after the pin cleared reports previously shown",
    );
}

#[test]
fn cursor_realization_recovers_from_external_clobber() {
    // Entering the window does not re-apply a previously-set cursor: the
    // display shows whatever the last SetCursor pushed, and while the pointer
    // is outside, the native cursor takes over. Both re-entry paths must
    // therefore PUSH the current cursor rather than assume it still sticks:
    // a consumed WM_SETCURSOR (even when not dirty) and ShowCursor (even
    // without a visibility transition). Gating realization on a visibility
    // transition would leave the in-game cursor invisible after a
    // pointer-outside startup until the game's next full hide/show cycle.
    const WM_SETCURSOR: u32 = 0x0020;
    /// `WM_MOUSEMOVE` as the trigger message in `WM_SETCURSOR`'s lparam.
    const WM_MOUSEMOVE_LP: isize = 0x0200;
    const HTCLIENT: isize = 1;
    let h = Harness::new();
    let lp_client_move = (WM_MOUSEMOVE_LP << 16) | HTCLIENT;

    let bitmap = h.create_offscreen_plain_surface(32, 32, D3DFMT_A8R8G8B8, D3DPOOL_SCRATCH);
    assert_eq!(h.set_cursor_properties_hr(0, 0, &bitmap), D3D_OK);
    assert_eq!(h.show_cursor(true), 0, "cursor starts hidden");
    let ours = h.thread_cursor();
    assert_ne!(ours, 0, "ShowCursor(TRUE) must realize an HCURSOR");

    // First consumed WM_SETCURSOR clears the initial DIRTY flag.
    h.send_window_message(WM_SETCURSOR, h.hwnd(), lp_client_move);
    assert_eq!(h.thread_cursor(), ours);

    // Pointer leaves; something else owns the cursor. A later non-dirty
    // WM_SETCURSOR (pointer re-entered the client area) must push ours back.
    h.set_thread_cursor(0);
    h.send_window_message(WM_SETCURSOR, h.hwnd(), lp_client_move);
    assert_eq!(
        h.thread_cursor(),
        ours,
        "non-dirty consumed WM_SETCURSOR must re-assert the cursor",
    );

    // Same for a ShowCursor(TRUE) with no visibility transition.
    h.set_thread_cursor(0);
    assert_eq!(h.show_cursor(true), 1, "already visible (no transition)");
    assert_eq!(
        h.thread_cursor(),
        ours,
        "transition-less ShowCursor(TRUE) must re-assert the cursor",
    );

    // And hide must push the null cursor, not merely flip the flag.
    assert_eq!(h.show_cursor(false), 1);
    assert_eq!(
        h.thread_cursor(),
        0,
        "ShowCursor(FALSE) must clear the cursor"
    );
}

#[test]
fn wm_setcursor_forwarded_to_game_while_cursor_hidden() {
    // Native d3d9 never intercepts WM_SETCURSOR: while the D3D cursor is not
    // shown, the game owns the win32 cursor (WoW's login screen never calls
    // ShowCursor(TRUE) — its glove is set by the game's own wndproc). Our
    // subclass must forward in that state; consuming and pushing null would
    // leave the login cursor invisible whenever the pointer entered the window.
    const WM_SETCURSOR: u32 = 0x0020;
    /// `WM_MOUSEMOVE` as the trigger message in `WM_SETCURSOR`'s lparam.
    const WM_MOUSEMOVE_LP: isize = 0x0200;
    const HTCLIENT: isize = 1;
    let h = Harness::new();
    let lp_client_move = (WM_MOUSEMOVE_LP << 16) | HTCLIENT;

    let bitmap = h.create_offscreen_plain_surface(32, 32, D3DFMT_A8R8G8B8, D3DPOOL_SCRATCH);
    assert_eq!(h.set_cursor_properties_hr(0, 0, &bitmap), D3D_OK);

    // Hidden (ShowCursor(TRUE) never called): SetCursorProperties must not
    // have touched the win32 cursor, and WM_SETCURSOR must reach the window's
    // own wndproc — DefWindowProc applies the window-class arrow.
    h.set_thread_cursor(0);
    h.send_window_message(WM_SETCURSOR, h.hwnd(), lp_client_move);
    let class_arrow = h.thread_cursor();
    assert_ne!(
        class_arrow, 0,
        "hidden: WM_SETCURSOR must be forwarded so the class cursor applies",
    );

    // Shown: the subclass owns the cursor and pushes the device HCURSOR.
    assert_eq!(h.show_cursor(true), 0);
    let ours = h.thread_cursor();
    assert_ne!(ours, 0);
    assert_ne!(
        ours, class_arrow,
        "device cursor is distinct from the arrow"
    );
    h.send_window_message(WM_SETCURSOR, h.hwnd(), lp_client_move);
    assert_eq!(
        h.thread_cursor(),
        ours,
        "visible: consumed WM_SETCURSOR pushes the device cursor",
    );

    // Hidden again after an explicit hide: back to forwarding.
    assert_eq!(h.show_cursor(false), 1);
    assert_eq!(h.thread_cursor(), 0, "hide pushes the null cursor");
    h.send_window_message(WM_SETCURSOR, h.hwnd(), lp_client_move);
    assert_eq!(
        h.thread_cursor(),
        class_arrow,
        "hidden again: WM_SETCURSOR forwarded to the class cursor",
    );
}

/// A device with the software cursor on.
///
/// The suite pins `color.hdr.enable=false`, under which the default `auto`
/// resolves to the hardware cursor; this harness's interface forces the
/// overlay, and no other harness in the process sees the key.
fn software_cursor_harness() -> Harness {
    Harness::with_config("cursor.software=true")
}

#[test]
fn each_direct3d9_resolves_its_own_configuration() {
    // Configuration belongs to the interface: a second `Direct3DCreate9` in
    // the same process resolves `MTLD3D_CONFIG` afresh and neither interface
    // sees the other's answers. `caps.dfFormats` is observable on the factory
    // alone through `CheckDeviceFormat`, so no device is needed.
    let hidden = Harness::factory_only_with_config("caps.dfFormats=false");
    let probe = |h: &Harness| {
        h.check_device_format(
            D3DFMT_X8R8G8B8,
            D3DUSAGE_DEPTHSTENCIL,
            D3DRTYPE_SURFACE,
            D3DFMT_DF24,
        )
    };
    assert_eq!(
        probe(&hidden),
        D3DERR_NOTAVAILABLE,
        "the first interface hides DF24"
    );

    let advertised = Harness::factory_only_with_config("caps.dfFormats=true");
    assert_eq!(
        probe(&advertised),
        D3D_OK,
        "the second interface resolved its own configuration"
    );
    assert_eq!(
        probe(&hidden),
        D3DERR_NOTAVAILABLE,
        "the first interface kept its configuration"
    );
}

#[test]
fn a_device_keeps_the_configuration_of_the_interface_that_created_it() {
    // The device takes its configuration from the interface that created it.
    // `memory.vramBudgetMB` caps what `GetAvailableTextureMem` reports, so a
    // device from each of two interfaces reports each interface's own cap.
    // The devices are sequential: the second is created after the first is
    // released.
    const MIB: u32 = 1024 * 1024;
    let first = Harness::with_config("memory.vramBudgetMB=64");
    assert!(
        first.available_texture_mem() <= 64 * MIB,
        "the first device reports at most its interface's 64 MiB budget"
    );
    assert_eq!(first.release_device(), 0, "the first device is released");

    let second = Harness::with_config("memory.vramBudgetMB=256");
    let reported = second.available_texture_mem();
    assert!(
        reported > 64 * MIB && reported <= 256 * MIB,
        "the second device reports its own interface's 256 MiB budget, got {reported}"
    );
}

#[test]
fn software_cursor_never_pushes_a_null_thread_cursor() {
    // With the software cursor on, the overlay window draws the cursor and the
    // Win32 cursor is a blank HCURSOR that is never taken away: a show pushes
    // the blank (a real handle, distinct from the class arrow the forwarded
    // WM_SETCURSOR applies while hidden) and a hide pushes nothing, so the
    // WindowServer cursor plane never toggles. The hardware path pins the
    // opposite in `cursor_realization_recovers_from_external_clobber`.
    const WM_SETCURSOR: u32 = 0x0020;
    /// `WM_MOUSEMOVE` as the trigger message in `WM_SETCURSOR`'s lparam.
    const WM_MOUSEMOVE_LP: isize = 0x0200;
    const HTCLIENT: isize = 1;
    let h = software_cursor_harness();
    let lp_client_move = (WM_MOUSEMOVE_LP << 16) | HTCLIENT;

    let bitmap = h.create_offscreen_plain_surface(32, 32, D3DFMT_A8R8G8B8, D3DPOOL_SCRATCH);
    assert_eq!(h.set_cursor_properties_hr(0, 0, &bitmap), D3D_OK);

    // Hidden: forwarded like the hardware path, the class arrow applies.
    h.set_thread_cursor(0);
    h.send_window_message(WM_SETCURSOR, h.hwnd(), lp_client_move);
    let class_arrow = h.thread_cursor();
    assert_ne!(
        class_arrow, 0,
        "hidden: WM_SETCURSOR must still be forwarded"
    );

    assert_eq!(h.show_cursor(true), 0, "cursor starts hidden");
    let blank = h.thread_cursor();
    assert_ne!(blank, 0, "ShowCursor(TRUE) must realize the blank HCURSOR");
    assert_ne!(blank, class_arrow, "the blank is not the class arrow");

    assert_eq!(h.show_cursor(false), 1);
    assert_eq!(
        h.thread_cursor(),
        blank,
        "ShowCursor(FALSE) must leave the blank in place, never push null",
    );

    // A show after something else took the thread cursor re-asserts the blank.
    h.set_thread_cursor(0);
    assert_eq!(h.show_cursor(true), 0);
    assert_eq!(
        h.thread_cursor(),
        blank,
        "ShowCursor(TRUE) re-asserts the blank"
    );
}

#[test]
fn software_cursor_presents_with_the_sprite_shown() {
    // The overlay path end to end inside the harness: a sprite upload, a
    // main-thread window creation, a sprite render, and show/hide/show across
    // presents. Nothing of it may disturb the frame, and a cursor change
    // (second bitmap) ships a second sprite.
    let h = software_cursor_harness();

    let first = h.create_offscreen_plain_surface(32, 32, D3DFMT_A8R8G8B8, D3DPOOL_SCRATCH);
    assert_eq!(h.set_cursor_properties_hr(2, 3, &first), D3D_OK);
    assert_eq!(h.show_cursor(true), 0);
    for _ in 0..20 {
        assert_eq!(h.clear(D3DCLEAR_TARGET, 0xFF00_80FF, 1.0, 0), D3D_OK);
        assert_eq!(h.present(), D3D_OK);
    }
    assert_eq!(
        h.read_pixel(5, 5) & 0x00FF_FFFF,
        0x0000_80FF,
        "frame unaffected"
    );

    assert_eq!(h.show_cursor(false), 1);
    assert_eq!(h.present(), D3D_OK);
    let second = h.create_offscreen_plain_surface(64, 64, D3DFMT_A8R8G8B8, D3DPOOL_SCRATCH);
    assert_eq!(h.set_cursor_properties_hr(0, 0, &second), D3D_OK);
    assert_eq!(h.show_cursor(true), 0);
    for _ in 0..20 {
        assert_eq!(h.clear(D3DCLEAR_TARGET, 0xFF40_C020, 1.0, 0), D3D_OK);
        assert_eq!(h.present(), D3D_OK);
    }
    assert_eq!(
        h.read_pixel(5, 5) & 0x00FF_FFFF,
        0x0040_C020,
        "frame unaffected"
    );
}

#[test]
fn a_second_device_renders_after_the_first_is_destroyed() {
    // The unix side keeps an attachment record per device, holding the metal
    // view, its layer and its window from CreateDevice on, and reconciles it
    // against the display from the main thread. Releasing a device retires
    // its record before the view is released, and the next device's own
    // attach registers a record of its own. Both devices present past the
    // interval at which the presenting thread asks the main thread for a
    // reconciliation, so that walk runs on the first device's record while it
    // is live and on the second device's once it has replaced it.
    const PRESENTS: u32 = 40;
    const RED: u32 = 0xFFFF_0000;
    const GREEN: u32 = 0xFF00_FF00;

    let first = Harness::new();
    for _ in 0..PRESENTS {
        first.render_once(RED, |_| {});
    }
    assert_pixel_eq(first.read_pixel(1, 1), RED, "first device");
    // The window outlives the device it served: destroying it here would post
    // WM_QUIT into the thread queue the second device then pumps.
    assert_eq!(
        first.release_device(),
        0,
        "the first device is fully released"
    );

    let second = Harness::new();
    for _ in 0..PRESENTS {
        second.render_once(GREEN, |_| {});
    }
    assert_pixel_eq(
        second.read_pixel(1, 1),
        GREEN,
        "second device after the first was released",
    );
}

#[test]
fn a_harness_with_its_own_configuration_leaves_the_environment_alone() {
    // The entries a harness carries are resolved by its own `Direct3DCreate9`
    // and never stay in `MTLD3D_CONFIG`, where every later interface in the
    // process would read them.
    let before = std::env::var("MTLD3D_CONFIG").ok();
    let own = Harness::factory_only_with_config("caps.dfFormats=false");
    assert_eq!(
        std::env::var("MTLD3D_CONFIG").ok(),
        before,
        "the variable is back before the constructor returns"
    );
    let plain = Harness::factory_only();
    let probe = |h: &Harness| {
        h.check_device_format(
            D3DFMT_X8R8G8B8,
            D3DUSAGE_DEPTHSTENCIL,
            D3DRTYPE_SURFACE,
            D3DFMT_DF24,
        )
    };
    assert_eq!(
        probe(&own),
        D3DERR_NOTAVAILABLE,
        "the entry reached its own interface"
    );
    assert_eq!(probe(&plain), D3D_OK, "and no interface created after it");
}
