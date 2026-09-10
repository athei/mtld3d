//! Device + factory lifecycle.
//!
//! `IDirect3D9` queries, caps, `TestCooperativeLevel`, and `Reset`
//! (state-default restore, resize, malformed input).

use std::{
    sync::{
        Barrier, Mutex, PoisonError,
        atomic::{AtomicUsize, Ordering},
        mpsc::{Receiver, RecvTimeoutError, Sender, TryRecvError, channel},
    },
    time::Duration,
};

use mtld3d_core::display_mode::MAX_SERVED_SIZES;
use mtld3d_tests::{
    Harness, HarnessConfig, TexturedVertex, WM_ACTIVATEAPP, WS_CAPTION, WS_EX_TOPMOST, WS_POPUP,
    WS_VISIBLE, WindowStyle, assert_pixel_eq, config_var, create_window, cursor_is_live,
    cursor_mask_bits, destroy_window, enumerate_display_sizes, spawn_scoped, window_rect,
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
    D3DRS_LIGHTING, D3DRTYPE_CUBETEXTURE, D3DRTYPE_SURFACE, D3DRTYPE_TEXTURE, D3DRTYPE_VOLUME,
    D3DRTYPE_VOLUMETEXTURE, D3DSWAPEFFECT_DISCARD, D3DUSAGE_AUTOGENMIPMAP, D3DUSAGE_DEPTHSTENCIL,
    D3DUSAGE_DYNAMIC, D3DUSAGE_QUERY_FILTER, D3DUSAGE_QUERY_POSTPIXELSHADER_BLENDING,
    D3DUSAGE_QUERY_SRGBREAD, D3DUSAGE_QUERY_SRGBWRITE, D3DUSAGE_QUERY_VERTEXTEXTURE,
    D3DUSAGE_QUERY_WRAPANDMIP, D3DUSAGE_RENDERTARGET, D3DVIEWPORT9, DevCaps, TextureCaps,
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
fn volume_format_queries_match_gpu_creation() {
    let h = Harness::new();
    for resource_type in [D3DRTYPE_VOLUMETEXTURE, D3DRTYPE_VOLUME] {
        assert_eq!(
            h.check_device_format(D3DFMT_X8R8G8B8, 0, resource_type, D3DFMT_A8R8G8B8),
            D3D_OK,
            "an A8R8G8B8 volume is advertised for resource type {resource_type}",
        );
    }
    assert_eq!(
        h.create_volume_texture([4, 4, 4], 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED),
        D3D_OK,
        "the advertised managed volume creates",
    );

    assert_eq!(
        h.check_device_format(
            D3DFMT_X8R8G8B8,
            D3DUSAGE_DYNAMIC,
            D3DRTYPE_VOLUMETEXTURE,
            D3DFMT_A8R8G8B8,
        ),
        D3D_OK,
        "a dynamic A8R8G8B8 volume is advertised",
    );
    assert_eq!(
        h.create_volume_texture(
            [4, 4, 4],
            1,
            D3DUSAGE_DYNAMIC,
            D3DFMT_A8R8G8B8,
            D3DPOOL_DEFAULT,
        ),
        D3D_OK,
        "the advertised dynamic volume creates",
    );

    for (usage, name) in [
        (D3DUSAGE_RENDERTARGET, "RENDERTARGET"),
        (D3DUSAGE_DEPTHSTENCIL, "DEPTHSTENCIL"),
        (D3DUSAGE_AUTOGENMIPMAP, "AUTOGENMIPMAP"),
    ] {
        assert_eq!(
            h.check_device_format(
                D3DFMT_X8R8G8B8,
                usage,
                D3DRTYPE_VOLUMETEXTURE,
                D3DFMT_A8R8G8B8,
            ),
            D3DERR_NOTAVAILABLE,
            "{name} is not available on a volume texture",
        );
        assert_eq!(
            h.create_volume_texture([4, 4, 4], 1, usage, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT,),
            D3DERR_INVALIDCALL,
            "CreateVolumeTexture rejects {name}",
        );
    }

    assert_eq!(
        h.check_device_format(D3DFMT_X8R8G8B8, 0, D3DRTYPE_VOLUMETEXTURE, D3DFMT_DXT1),
        D3DERR_NOTAVAILABLE,
        "a scratch-only DXT1 volume is not a GPU volume capability",
    );
    assert_eq!(
        h.create_volume_texture([4, 4, 4], 1, 0, D3DFMT_DXT1, D3DPOOL_DEFAULT),
        D3DERR_INVALIDCALL,
        "a GPU-backed DXT1 volume is rejected",
    );
    assert_eq!(
        h.create_volume_texture([4, 4, 4], 1, 0, D3DFMT_DXT1, D3DPOOL_SCRATCH),
        D3D_OK,
        "the CPU-only scratch exception remains creatable",
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
    // so the answer for a 16-bit backbuffer on a 32-bit display is whatever
    // this device says about rendering into R5G6B5: yes where the packed
    // 16-bit Metal formats are native, no where they are expansion-backed
    // (`expand16` pins that side). Deriving it here rather than pinning one
    // of the two keeps the identity the assertion, which is the contract.
    // Fullscreen has no present conversion and rejects the pair either way.
    let h = Harness::factory_only();
    assert_eq!(
        h.check_device_format_conversion(D3DFMT_R5G6B5, D3DFMT_X8R8G8B8),
        D3D_OK,
        "precondition: the conversion is supported",
    );
    let renderable = h.check_device_format(
        D3DFMT_X8R8G8B8,
        D3DUSAGE_RENDERTARGET,
        D3DRTYPE_SURFACE,
        D3DFMT_R5G6B5,
    ) == D3D_OK;
    let expected = if renderable {
        D3D_OK
    } else {
        D3DERR_NOTAVAILABLE
    };
    assert_eq!(
        h.check_device_type(D3DFMT_X8R8G8B8, D3DFMT_R5G6B5, true),
        expected,
        "windowed R5G6B5 backbuffer on an X8R8G8B8 display follows the render-target answer",
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
fn failed_auto_resize_requires_a_reset_before_presenting() {
    const WM_SIZE: u32 = 0x0005;
    const OVERSIZE_DIMENSION: isize = 0xffff;

    let h = Harness::new();
    h.send_window_message(WM_SIZE, 0, (OVERSIZE_DIMENSION << 16) | OVERSIZE_DIMENSION);

    assert_eq!(
        h.test_cooperative_level(),
        D3DERR_DEVICENOTRESET,
        "an oversized auto-resize back-buffer failure requires Reset"
    );
    assert_eq!(
        h.present(),
        D3DERR_DEVICENOTRESET,
        "Present must not submit the frame with null implicit handles"
    );

    assert_eq!(
        h.reset(640, 480),
        D3D_OK,
        "a valid Reset rebuilds the back buffer"
    );
    assert_eq!(
        h.test_cooperative_level(),
        D3D_OK,
        "a successful Reset clears the latch"
    );
    assert_eq!(h.present(), D3D_OK, "presentation resumes after Reset");
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

#[test]
fn reset_flips_the_presentation_interval() {
    let h = Harness::new();
    assert_eq!(
        h.present(),
        D3D_OK,
        "a present at the interval the device was created with"
    );
    // A Reset queues the new pacing for the frames that follow it, and the
    // presents after each one carry it to the layer, which re-derives the
    // present throttle off the panel's cadence and back onto it. Several
    // presents rather than one, so the frame that carries the pacing is sent
    // and the device outlives the re-derivation it queues.
    let mut pp = windowed_params(h.hwnd(), 640, 480);
    pp.presentation_interval = D3DPRESENT_INTERVAL_IMMEDIATE;
    assert_eq!(h.reset_params(&mut pp), D3D_OK, "Reset to IMMEDIATE");
    for _ in 0..8 {
        assert_eq!(h.present(), D3D_OK, "a present of the free run");
    }
    let mut pp = windowed_params(h.hwnd(), 640, 480);
    pp.presentation_interval = D3DPRESENT_INTERVAL_ONE;
    assert_eq!(h.reset_params(&mut pp), D3D_OK, "Reset back to ONE");
    for _ in 0..8 {
        assert_eq!(h.present(), D3D_OK, "a present at the display rate");
    }
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

/// The state defaults a same-size `Reset` applies reach the encoder.
///
/// `Reset` restores the full-target viewport, and the encoder's viewport is
/// sticky across frames: nothing else re-asserts it, so the ops the
/// state-default restore queues have to land in a frame that is submitted. A
/// pre-`Reset` viewport that survives clips the following frame's `Clear` and
/// draw to the corner it covered.
#[test]
fn reset_same_size_restores_the_rasterized_viewport() {
    const RED: u32 = 0xFFFF_0000;
    const BLUE: u32 = 0xFF00_00FF;
    const BLACK: u32 = 0xFF00_0000;
    const CORNER: u32 = 64;

    let h = Harness::new();
    // Paint the whole target first, so the centre carries a known colour that
    // a viewport-clipped clear and draw would leave untouched.
    h.render_once(BLUE, |_| {});
    assert_pixel_eq(h.read_pixel(320, 240), BLUE, "full-target clear");

    let corner = D3DVIEWPORT9 {
        x: 0,
        y: 0,
        width: CORNER,
        height: CORNER,
        min_z: 0.0,
        max_z: 1.0,
    };
    assert_eq!(h.set_viewport(&corner), 0, "SetViewport before Reset");
    assert_eq!(h.reset(640, 480), 0, "same-size Reset must succeed");

    // No `SetViewport` after the Reset: the default the Reset applies is the
    // only thing that can widen the frame back to the full target.
    h.select_diffuse_stage(0);
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0, "LIGHTING off");
    assert_eq!(
        h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1),
        0,
        "SetFVF for the post-Reset draw"
    );
    h.render_once(BLACK, |d| {
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &flat_quad(RED)),
            0,
            "post-Reset draw"
        );
    });
    assert_pixel_eq(
        h.read_pixel(320, 240),
        RED,
        "the Reset's default viewport must reach the encoder",
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

/// The same shape as [`fullscreen_params`], windowed.
const fn windowed_params(hwnd: usize, width: u32, height: u32) -> D3DPRESENT_PARAMETERS {
    D3DPRESENT_PARAMETERS {
        back_buffer_width: width,
        back_buffer_height: height,
        back_buffer_format: D3DFMT_X8R8G8B8,
        back_buffer_count: 1,
        multi_sample_type: 0,
        multi_sample_quality: 0,
        swap_effect: D3DSWAPEFFECT_DISCARD,
        device_window: hwnd,
        windowed: 1,
        enable_auto_depth_stencil: 0,
        auto_depth_stencil_format: 0,
        flags: 0,
        full_screen_refresh_rate_in_hz: 0,
        presentation_interval: 0,
    }
}

#[test]
fn rejected_reset_keeps_the_window_mode() {
    let h = Harness::create(&HarnessConfig {
        window_style: WindowStyle::Framed,
        ..HarnessConfig::default()
    });
    let original_rect = h.window_rect();
    let original_style = h.window_style();
    let original_exstyle = h.window_exstyle();

    let mut pp = fullscreen_params(h.hwnd(), 137, 101);
    pp.multi_sample_type = u32::MAX;
    assert_eq!(h.reset_params(&mut pp), D3DERR_INVALIDCALL);
    assert_eq!(h.window_rect(), original_rect, "rejected fullscreen entry");
    assert_eq!(h.window_style(), original_style);
    assert_eq!(h.window_exstyle(), original_exstyle);
    assert_eq!(h.test_cooperative_level(), D3DERR_DEVICENOTRESET);

    pp.multi_sample_type = 0;
    assert_eq!(h.reset_params(&mut pp), D3D_OK);
    let fullscreen_rect = h.window_rect();
    let fullscreen_style = h.window_style();
    let fullscreen_exstyle = h.window_exstyle();
    let fullscreen_screen = Harness::screen_size();
    let other = create_window(173, 119, false);
    let other_rect = window_rect(other);

    for mut rejected in [
        windowed_params(h.hwnd(), 320, 240),
        fullscreen_params(other, 800, 600),
    ] {
        rejected.multi_sample_type = u32::MAX;
        assert_eq!(h.reset_params(&mut rejected), D3DERR_INVALIDCALL);
        assert_eq!(h.window_rect(), fullscreen_rect, "rejected mode change");
        assert_eq!(h.window_style(), fullscreen_style);
        assert_eq!(h.window_exstyle(), fullscreen_exstyle);
        assert_eq!(Harness::screen_size(), fullscreen_screen);
        assert_eq!(window_rect(other), other_rect, "rejected retarget");
        assert_eq!(h.test_cooperative_level(), D3DERR_DEVICENOTRESET);
    }
    destroy_window(other);
    assert_eq!(h.reset(320, 240), D3D_OK, "retry can leave fullscreen");
    assert_eq!(h.window_rect(), original_rect);
    assert_eq!(h.window_style() & !WS_VISIBLE, original_style & !WS_VISIBLE);
    assert_eq!(h.test_cooperative_level(), D3D_OK);
}

#[test]
fn rejected_zero_dimension_reset_restores_fullscreen() {
    let h = Harness::create(&HarnessConfig {
        window_style: WindowStyle::Framed,
        ..HarnessConfig::default()
    });
    let original_rect = h.window_rect();
    let original_client = h.client_size();
    let original_style = h.window_style();
    let mut pp = fullscreen_params(h.hwnd(), 137, 101);
    assert_eq!(h.reset_params(&mut pp), D3D_OK);
    let fullscreen_rect = h.window_rect();
    let fullscreen_style = h.window_style();
    let fullscreen_exstyle = h.window_exstyle();
    let fullscreen_screen = Harness::screen_size();

    let gone = create_window(173, 119, false);
    destroy_window(gone);
    let mut rejected = windowed_params(gone, 0, 0);
    assert_eq!(h.reset_params(&mut rejected), D3DERR_INVALIDCALL);
    assert_eq!(
        h.window_rect(),
        fullscreen_rect,
        "failed client-area resolution"
    );
    assert_eq!(h.window_style(), fullscreen_style);
    assert_eq!(h.window_exstyle(), fullscreen_exstyle);
    assert_eq!(Harness::screen_size(), fullscreen_screen);
    assert_eq!(h.test_cooperative_level(), D3DERR_DEVICENOTRESET);

    let mut retry = windowed_params(0, 0, 0);
    assert_eq!(h.reset_params(&mut retry), D3D_OK);
    assert_eq!(
        h.window_rect(),
        original_rect,
        "retry retains the original saved window"
    );
    assert_eq!(h.window_style() & !WS_VISIBLE, original_style & !WS_VISIBLE);
    assert_eq!(
        (retry.back_buffer_width, retry.back_buffer_height),
        original_client,
        "successful zero dimensions use the restored client rect"
    );
    assert_eq!(h.test_cooperative_level(), D3D_OK);
}

#[test]
fn zero_dimension_reset_can_leave_a_destroyed_fullscreen_window() {
    let h = Harness::new();
    let old_target = create_window(320, 240, false);
    let mut pp = fullscreen_params(old_target, 137, 101);
    assert_eq!(h.reset_params(&mut pp), D3D_OK);
    let target = create_window(320, 240, false);
    destroy_window(old_target);

    let mut pp = windowed_params(target, 0, 0);
    assert_eq!(
        h.reset_params(&mut pp),
        D3D_OK,
        "retarget from a destroyed window"
    );
    assert!(pp.back_buffer_width > 0 && pp.back_buffer_height > 0);
    let (hr, backbuffer) = h.back_buffer(0).desc();
    assert_eq!(hr, D3D_OK);
    assert_eq!(
        (backbuffer.width, backbuffer.height),
        (pp.back_buffer_width, pp.back_buffer_height)
    );
    assert_eq!(h.test_cooperative_level(), D3D_OK);
    drop(h);
    destroy_window(target);
}

#[test]
fn reset_fullscreen_adopts_monitor_rect_and_restores() {
    if !display_lists_640x480() {
        return;
    }
    let h = Harness::create(&HarnessConfig {
        window_style: WindowStyle::Framed,
        ..HarnessConfig::default()
    });
    let hwnd = h.hwnd();
    let windowed_rect = h.window_rect();
    let windowed_style = h.window_style();
    assert_ne!(
        windowed_style & WS_CAPTION,
        0,
        "starts with a caption to restore"
    );

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
        window_style: WindowStyle::Framed,
        behavior_flags: D3DCREATE_HARDWARE_VERTEXPROCESSING | D3DCREATE_NOWINDOWCHANGES,
        ..HarnessConfig::default()
    });
    let windowed_rect = h.window_rect();
    let windowed_style = h.window_style();
    assert_ne!(
        windowed_style & WS_CAPTION,
        0,
        "starts with a caption to preserve"
    );
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
    h.hold_display_mode();
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

/// A fullscreen `Reset` naming another device window hands the session over.
///
/// The window the device came from keeps the borderless style and the monitor
/// rect it was put in: the session moves across rather than being given back,
/// so an app that retargets its device sees no screen uncovered behind it. The
/// window that is handed back on the way out is the one the device presents
/// into, the retarget target.
#[test]
fn reset_fullscreen_retarget_keeps_the_previous_window_covered() {
    let h = Harness::new();
    let second = create_window(320, 240, false);
    // The current resolution is a settable mode on any display, so this runs
    // wherever the suite does, unlike the tests that request 640x480.
    h.hold_display_mode();
    let second_rect = window_rect(second);
    let (screen_w, screen_h) = Harness::screen_size();

    let mut pp = fullscreen_params(h.hwnd(), screen_w, screen_h);
    assert_eq!(
        h.reset_params(&mut pp),
        D3D_OK,
        "fullscreen Reset on the device's own window",
    );
    let covered = h.window_rect();

    let mut pp = fullscreen_params(second, screen_w, screen_h);
    assert_eq!(
        h.reset_params(&mut pp),
        D3D_OK,
        "fullscreen Reset onto a second device window",
    );
    assert_eq!(
        window_rect(second),
        covered,
        "the new device window must cover the monitor",
    );
    assert_eq!(
        h.window_rect(),
        covered,
        "the window the device came from keeps the fullscreen rect",
    );

    assert_eq!(h.reset(screen_w, screen_h), D3D_OK, "windowed Reset");
    assert_eq!(
        window_rect(second),
        second_rect,
        "leaving fullscreen gives back the window the device presented into",
    );
    destroy_window(second);
}

/// A windowed `Reset` naming another device window moves the device onto it.
///
/// The presentation surface and the window subclass are both bound to the
/// window `CreateDevice` attached, and `Reset` re-specifies the swap chain on
/// the window its parameters name. The subclass is the observable half: after
/// the retarget a `WM_SIZE` on the new window resizes the back buffer, and one
/// on the window the device came from no longer reaches it.
#[test]
fn reset_windowed_retarget_moves_the_device_onto_the_new_window() {
    const WM_SIZE: u32 = 0x0005;
    // The client size each WM_SIZE announces, as lparam's low and high words.
    let (new_width, new_height): (isize, isize) = (400, 300);
    let (old_width, old_height): (isize, isize) = (320, 200);
    let h = Harness::new();
    let second = create_window(640, 480, false);

    let mut pp = windowed_params(second, 640, 480);
    assert_eq!(
        h.reset_params(&mut pp),
        D3D_OK,
        "windowed Reset onto a second device window",
    );

    // lparam = client height << 16 | width, the shape macdrv posts.
    let _ = mtld3d_tests::send_message(second, WM_SIZE, 0, (new_height << 16) | new_width);
    {
        let (bb_hr, bb) = h.back_buffer(0).desc();
        assert_eq!(bb_hr, D3D_OK, "GetDesc after the resize of the new window");
        assert_eq!(
            (bb.width, bb.height),
            (400, 300),
            "a WM_SIZE on the new device window resizes the back buffer",
        );
    }

    let _ = h.send_window_message(WM_SIZE, 0, (old_height << 16) | old_width);
    {
        let (bb_hr, bb) = h.back_buffer(0).desc();
        assert_eq!(bb_hr, D3D_OK, "GetDesc after the resize of the old window");
        assert_eq!(
            (bb.width, bb.height),
            (400, 300),
            "the window the device left no longer resizes it",
        );
    }

    // The frame goes through the layer the retarget attached, so a released or
    // stale one is a failing Present or a readback of the wrong buffer rather
    // than a silent no-op.
    h.render_once(0xFF20_4060, |_| {});
    assert_pixel_eq(
        h.read_pixel(8, 8),
        0xFF20_4060,
        "the frame presented after the retarget",
    );

    // The device holds the subclass and the metal view of the second window,
    // so it goes first.
    drop(h);
    destroy_window(second);
}

/// The cursor subclass follows a windowed `Reset` onto another device window.
///
/// `WM_SETCURSOR` on the window the device presents into has to reach the
/// device's cursor state: the pointer over that window otherwise keeps the
/// class cursor while the bitmap the application set through
/// `SetCursorProperties` is realized on a window it stopped drawing in.
#[test]
fn reset_windowed_retarget_moves_the_cursor_subclass() {
    const WM_SETCURSOR: u32 = 0x0020;
    /// `WM_MOUSEMOVE` as the trigger message in `WM_SETCURSOR`'s lparam.
    const WM_MOUSEMOVE_LP: isize = 0x0200;
    const HTCLIENT: isize = 1;
    let h = Harness::new();
    let lp_client_move = (WM_MOUSEMOVE_LP << 16) | HTCLIENT;

    {
        let bitmap = h.create_offscreen_plain_surface(32, 32, D3DFMT_A8R8G8B8, D3DPOOL_SCRATCH);
        assert_eq!(h.set_cursor_properties_hr(0, 0, &bitmap), D3D_OK);
    }
    assert_eq!(h.show_cursor(true), 0, "cursor starts hidden");
    let ours = h.thread_cursor();
    assert_ne!(ours, 0, "ShowCursor(TRUE) must realize an HCURSOR");

    let second = create_window(640, 480, false);
    let mut pp = windowed_params(second, 640, 480);
    assert_eq!(
        h.reset_params(&mut pp),
        D3D_OK,
        "windowed Reset onto a second device window",
    );

    // `Reset` clobbers device state, not the cursor the application set, so
    // the bitmap and its visibility are still the device's.
    h.set_thread_cursor(0);
    let _ = mtld3d_tests::send_message(second, WM_SETCURSOR, second, lp_client_move);
    assert_eq!(
        h.thread_cursor(),
        ours,
        "WM_SETCURSOR on the new device window must reach the device's cursor",
    );

    drop(h);
    destroy_window(second);
}

/// Rounds every windowed worker of the concurrent-retarget test runs at least.
const RETARGET_ROUNDS: u32 = 16;

/// Fullscreen round trips the windowed workers keep retargeting through.
const FULLSCREEN_TRIPS: u32 = 8;

/// Rounds a windowed worker stops at even if the fullscreen worker is slow.
const MAX_RETARGET_ROUNDS: u32 = 2000;

/// How long workers wait for a concurrent-retarget handshake or transition.
const FULLSCREEN_PROGRESS_TIMEOUT: Duration = Duration::from_secs(30);

/// Time the check deliberately leaves the fullscreen worker without display ownership.
const DELAYED_FULLSCREEN_START: Duration = Duration::from_millis(100);

/// Wait for every windowed worker to reach one handshake boundary.
fn wait_for_windowed_workers(
    receiver: &Receiver<()>,
    workers: usize,
    boundary: &str,
) -> Result<(), String> {
    for reached in 0..workers {
        match receiver.recv_timeout(FULLSCREEN_PROGRESS_TIMEOUT) {
            Ok(()) => {}
            Err(RecvTimeoutError::Timeout) => {
                return Err(format!(
                    "only {reached} of {workers} windowed workers reached {boundary} within {} \
                     seconds",
                    FULLSCREEN_PROGRESS_TIMEOUT.as_secs(),
                ));
            }
            Err(RecvTimeoutError::Disconnected) => {
                return Err(format!(
                    "a windowed worker stopped before all {workers} reached {boundary}"
                ));
            }
        }
    }
    Ok(())
}

/// Publish a fullscreen transition to every windowed worker.
fn publish_transition(senders: &[Sender<u32>], trips: u32) -> Result<(), String> {
    for (index, sender) in senders.iter().enumerate() {
        if sender.send(trips).is_err() {
            return Err(format!(
                "windowed worker {index} stopped before fullscreen trip {trips}"
            ));
        }
    }
    Ok(())
}

/// Wait until the fullscreen worker owns the display mode.
fn wait_for_fullscreen_start(receiver: &Receiver<u32>) -> Result<(), String> {
    match receiver.recv_timeout(FULLSCREEN_PROGRESS_TIMEOUT) {
        Ok(0) => Ok(()),
        Ok(trips) => Err(format!(
            "the first fullscreen progress signal was trip {trips}, not the ownership signal"
        )),
        Err(RecvTimeoutError::Timeout) => Err(format!(
            "the fullscreen worker did not acquire the display mode within {} seconds",
            FULLSCREEN_PROGRESS_TIMEOUT.as_secs(),
        )),
        Err(RecvTimeoutError::Disconnected) => {
            Err("the fullscreen worker stopped before acquiring the display mode".to_owned())
        }
    }
}

/// Wait for the fullscreen worker to publish its next transition.
fn wait_for_transition(receiver: &Receiver<u32>, previous: u32) -> Result<u32, String> {
    match receiver.recv_timeout(FULLSCREEN_PROGRESS_TIMEOUT) {
        Ok(trips) => Ok(trips),
        Err(RecvTimeoutError::Timeout) => Err(format!(
            "the fullscreen worker made no progress after {previous} round trips for {} seconds",
            FULLSCREEN_PROGRESS_TIMEOUT.as_secs(),
        )),
        Err(RecvTimeoutError::Disconnected) => Err(format!(
            "the fullscreen worker stopped after {previous} of {FULLSCREEN_TRIPS} round trips"
        )),
    }
}

/// A `WM_SIZE` lparam: the client height in the high word, the width in the low.
fn client_size_lparam(width: u32, height: u32) -> isize {
    isize::try_from((height << 16) | width).expect("a client size fits 16 bits per axis")
}

/// Retarget one windowed device between its two windows, checking its messages each round.
///
/// Every round moves the device onto the other window with a windowed
/// `Reset`, then sends that window a `WM_SIZE` and a `WM_SETCURSOR` and reads
/// the back buffer and this thread's cursor back; a `WM_SIZE` on the window
/// the device left has to change nothing. Runs `RETARGET_ROUNDS` rounds at
/// least, and on until the fullscreen worker has completed `FULLSCREEN_TRIPS`
/// round trips, so the rounds overlap its window moves. A worker that runs a
/// full burst without a new transition waits for bounded progress instead of
/// consuming the total round cap. `Ok` carries the rounds run; `Err` names the
/// first message that missed the device or the bound that expired.
fn retarget_and_check_messages(
    h: &Harness,
    second: usize,
    ours: usize,
    progress: &Receiver<u32>,
    started: &Sender<()>,
) -> Result<u32, String> {
    const WM_SIZE: u32 = 0x0005;
    const WM_SETCURSOR: u32 = 0x0020;
    /// `WM_MOUSEMOVE` as the trigger message in `WM_SETCURSOR`'s lparam.
    const WM_MOUSEMOVE_LP: isize = 0x0200;
    const HTCLIENT: isize = 1;
    let lp_client_move = (WM_MOUSEMOVE_LP << 16) | HTCLIENT;
    let mut round = 0;
    wait_for_fullscreen_start(progress)?;
    let mut observed_transitions = 0;
    let mut rounds_without_transition = 0;
    while round < MAX_RETARGET_ROUNDS
        && (round < RETARGET_ROUNDS || observed_transitions < FULLSCREEN_TRIPS)
    {
        let even = round.is_multiple_of(2);
        let (target, left) = if even {
            (second, h.hwnd())
        } else {
            (h.hwnd(), second)
        };
        let (width, height): (u32, u32) = if even { (400, 300) } else { (320, 200) };

        let mut pp = windowed_params(target, 640, 480);
        if round == 0 {
            started.send(()).map_err(|_| {
                "the fullscreen worker stopped before windowed work began".to_owned()
            })?;
        }
        let hr = h.reset_params(&mut pp);
        if hr != D3D_OK {
            return Err(format!(
                "round {round}: windowed Reset onto window {target:#x} failed: 0x{hr:08X}"
            ));
        }

        let _ = mtld3d_tests::send_message(target, WM_SIZE, 0, client_size_lparam(width, height));
        let hr = h.test_cooperative_level();
        if hr != D3D_OK {
            return Err(format!(
                "round {round}: WM_SIZE {width}x{height} on device window {target:#x} left the \
                 device unavailable: TestCooperativeLevel returned 0x{hr:08X}"
            ));
        }
        let (hr, bb) = h.back_buffer(0).desc();
        if hr != D3D_OK {
            return Err(format!(
                "round {round}: GetDesc after the resize failed: 0x{hr:08X}"
            ));
        }
        if (bb.width, bb.height) != (width, height) {
            return Err(format!(
                "round {round}: WM_SIZE {width}x{height} on the device window {target:#x} left \
                 the back buffer at {}x{}",
                bb.width, bb.height,
            ));
        }

        let _ = mtld3d_tests::send_message(left, WM_SIZE, 0, client_size_lparam(200, 150));
        let (_, bb) = h.back_buffer(0).desc();
        if (bb.width, bb.height) != (width, height) {
            return Err(format!(
                "round {round}: WM_SIZE on the window the device left {left:#x} resized its back \
                 buffer to {}x{}",
                bb.width, bb.height,
            ));
        }

        h.set_thread_cursor(0);
        let _ = mtld3d_tests::send_message(target, WM_SETCURSOR, target, lp_client_move);
        let cursor = h.thread_cursor();
        if cursor != ours {
            return Err(format!(
                "round {round}: WM_SETCURSOR on the device window {target:#x} realized cursor \
                 {cursor:#x}, the device's is {ours:#x}"
            ));
        }
        // A mode-set broadcasts `WM_DISPLAYCHANGE` to every window and waits
        // for each to answer; the pump is that answer.
        let _ = h.pump();
        round += 1;
        rounds_without_transition += 1;
        while let Ok(transitions) = progress.try_recv() {
            if transitions > observed_transitions {
                observed_transitions = transitions;
                rounds_without_transition = 0;
            }
        }
        if round >= RETARGET_ROUNDS
            && observed_transitions < FULLSCREEN_TRIPS
            && rounds_without_transition >= RETARGET_ROUNDS
        {
            observed_transitions = wait_for_transition(progress, observed_transitions)?;
            rounds_without_transition = 0;
        }
    }
    if round == MAX_RETARGET_ROUNDS && observed_transitions < FULLSCREEN_TRIPS {
        return Err(format!(
            "reached the {MAX_RETARGET_ROUNDS}-round bound before the fullscreen worker completed \
             {FULLSCREEN_TRIPS} round trips"
        ));
    }
    Ok(round)
}

/// Take one device through the required fullscreen round trips.
///
/// Each round trip is the `Reset` pair that moves the device's window: onto
/// the monitor at the display's own resolution, which is a settable mode
/// wherever the suite runs, and back to a windowed 640x480. `Ok` carries the
/// round trips completed; `Err` names the first `Reset` that failed.
fn cycle_fullscreen(
    h: &Harness,
    progress: &[Sender<u32>],
    finished: &AtomicUsize,
) -> Result<u32, String> {
    let mut trips = 0;
    while trips < FULLSCREEN_TRIPS {
        if finished.load(Ordering::Acquire) != 0 {
            return Err(format!(
                "a windowed worker stopped after {trips} of {FULLSCREEN_TRIPS} round trips"
            ));
        }
        h.hold_display_mode();
        let (screen_w, screen_h) = Harness::screen_size();
        let mut pp = fullscreen_params(h.hwnd(), screen_w, screen_h);
        let hr = h.reset_params(&mut pp);
        if hr != D3D_OK {
            return Err(format!(
                "round trip {trips}: fullscreen Reset failed: 0x{hr:08X}"
            ));
        }
        let mut pp = windowed_params(h.hwnd(), 640, 480);
        let hr = h.reset_params(&mut pp);
        if hr != D3D_OK {
            return Err(format!(
                "round trip {trips}: windowed Reset failed: 0x{hr:08X}"
            ));
        }
        trips += 1;
        publish_transition(progress, trips)?;
    }
    Ok(trips)
}

fn concurrent_retarget_failures(
    windowed: &[Result<u32, String>],
    fullscreen: &Result<u32, String>,
) -> Vec<String> {
    let mut failures = Vec::new();
    for (index, outcome) in windowed.iter().enumerate() {
        match outcome {
            Ok(rounds) if *rounds < RETARGET_ROUNDS => failures.push(format!(
                "windowed device {index} ran {rounds} rounds, fewer than {RETARGET_ROUNDS}"
            )),
            Err(failure) => failures.push(format!("windowed device {index}: {failure}")),
            Ok(_) => {}
        }
    }
    match fullscreen {
        Ok(trips) if *trips < FULLSCREEN_TRIPS => failures.push(format!(
            "the fullscreen device made {trips} round trips, fewer than the {FULLSCREEN_TRIPS} \
             the windowed rounds overlap with"
        )),
        Err(failure) => failures.push(format!("fullscreen device: {failure}")),
        Ok(_) => {}
    }
    failures
}

#[test]
fn concurrent_retarget_outcomes_report_every_returned_failure() {
    let windowed = [
        Err("the fullscreen worker stopped after 2 of 8 round trips".to_owned()),
        Ok(RETARGET_ROUNDS - 1),
        Ok(RETARGET_ROUNDS),
    ];
    let fullscreen = Err("round trip 2: fullscreen Reset failed: 0xC0000001".to_owned());

    assert_eq!(
        concurrent_retarget_failures(&windowed, &fullscreen),
        [
            "windowed device 0: the fullscreen worker stopped after 2 of 8 round trips",
            "windowed device 1 ran 15 rounds, fewer than 16",
            "fullscreen device: round trip 2: fullscreen Reset failed: 0xC0000001",
        ],
    );
    assert_eq!(
        concurrent_retarget_failures(&[Ok(RETARGET_ROUNDS)], &Ok(FULLSCREEN_TRIPS - 1)),
        [
            "the fullscreen device made 7 round trips, fewer than the 8 the windowed rounds overlap with"
        ],
    );
    assert!(
        concurrent_retarget_failures(
            &[Ok(RETARGET_ROUNDS), Ok(RETARGET_ROUNDS + 1)],
            &Ok(FULLSCREEN_TRIPS + 1),
        )
        .is_empty(),
        "workers at or above their bounds have no failures",
    );
}

/// Retargets on several threads keep their messages while another device moves its window.
///
/// Three threads each own a windowed device and a second window and move the
/// device back and forth between the two, checking after every move that a
/// `WM_SIZE` on the device window resizes that device's back buffer, that a
/// `WM_SIZE` on the window it left does not, and that a `WM_SETCURSOR` on the
/// device window realizes that device's cursor. A fourth thread takes its own
/// device in and out of fullscreen the whole time, so the checks land inside
/// its window moves. A device's window management filters the messages its
/// own moves bounce back, and that filter has to be the window's: one that
/// is shared by every device in the process sends the other devices'
/// messages to the default procedure for the duration of the move, so the
/// back buffer stays at the size the `Reset` gave it and the class cursor
/// replaces the device's. The fullscreen worker waits until all three
/// windowed workers are held at the start boundary, then owns the display
/// mode before releasing them, so a delayed fullscreen start cannot consume
/// the bounded retarget rounds. After release, every windowed worker enters a
/// measured round before the fullscreen cycles begin, and each pauses after a
/// bounded burst without fullscreen progress. Every device and window is
/// created and torn down one thread at a time: the driver's window teardown
/// and another thread's window update take two locks in opposite orders.
#[test]
fn concurrent_retargets_deliver_every_window_message_to_its_own_device() {
    const WINDOWED_WORKERS: usize = 3;
    let finished = AtomicUsize::new(0);
    let one_at_a_time = Mutex::new(());
    let done = Barrier::new(WINDOWED_WORKERS + 1);
    let (ready_sender, ready_receiver) = channel();
    let (started_sender, started_receiver) = channel();
    let (progress_senders, progress_receivers): (Vec<_>, Vec<_>) =
        (0..WINDOWED_WORKERS).map(|_| channel()).unzip();

    std::thread::scope(|scope| {
        let fullscreen = spawn_scoped(scope, {
            let done = &done;
            let finished = &finished;
            let one_at_a_time = &one_at_a_time;
            move || {
                let h = {
                    let _serial = one_at_a_time.lock().unwrap_or_else(PoisonError::into_inner);
                    Harness::new()
                };
                let outcome = wait_for_windowed_workers(
                    &ready_receiver,
                    WINDOWED_WORKERS,
                    "the fullscreen ownership boundary",
                )
                .and_then(|()| {
                    std::thread::sleep(DELAYED_FULLSCREEN_START);
                    match started_receiver.try_recv() {
                        Err(TryRecvError::Empty) => Ok(()),
                        Ok(()) => Err(
                            "a windowed worker began retargeting before fullscreen display \
                             ownership"
                                .to_owned(),
                        ),
                        Err(TryRecvError::Disconnected) => {
                            Err("a windowed worker stopped during the delayed start".to_owned())
                        }
                    }
                })
                .and_then(|()| {
                    h.hold_display_mode();
                    publish_transition(&progress_senders, 0)
                })
                .and_then(|()| {
                    wait_for_windowed_workers(
                        &started_receiver,
                        WINDOWED_WORKERS,
                        "the first measured round",
                    )
                })
                .and_then(|()| cycle_fullscreen(&h, &progress_senders, finished));
                drop(progress_senders);
                done.wait();
                let _serial = one_at_a_time.lock().unwrap_or_else(PoisonError::into_inner);
                drop(h);
                outcome
            }
        });
        let windowed: Vec<_> = progress_receivers
            .into_iter()
            .map(|progress| {
                let done = &done;
                let finished = &finished;
                let one_at_a_time = &one_at_a_time;
                let ready_sender = ready_sender.clone();
                let started_sender = started_sender.clone();
                spawn_scoped(scope, move || {
                    let (h, second, ours) = {
                        let _serial = one_at_a_time.lock().unwrap_or_else(PoisonError::into_inner);
                        let h = Harness::new();
                        let second = create_window(640, 480, false);
                        {
                            let bitmap = h.create_offscreen_plain_surface(
                                32,
                                32,
                                D3DFMT_A8R8G8B8,
                                D3DPOOL_SCRATCH,
                            );
                            assert_eq!(h.set_cursor_properties_hr(0, 0, &bitmap), D3D_OK);
                        }
                        assert_eq!(h.show_cursor(true), 0, "cursor starts hidden");
                        let ours = h.thread_cursor();
                        assert_ne!(ours, 0, "ShowCursor(TRUE) must realize an HCURSOR");
                        (h, second, ours)
                    };
                    let outcome = ready_sender
                        .send(())
                        .map_err(|_| {
                            "the fullscreen worker stopped before the start boundary".to_owned()
                        })
                        .and_then(|()| {
                            retarget_and_check_messages(
                                &h,
                                second,
                                ours,
                                &progress,
                                &started_sender,
                            )
                        });
                    finished.fetch_add(1, Ordering::AcqRel);
                    done.wait();
                    let _serial = one_at_a_time.lock().unwrap_or_else(PoisonError::into_inner);
                    drop(h);
                    destroy_window(second);
                    outcome
                })
            })
            .collect();
        drop(ready_sender);
        drop(started_sender);

        let windowed_outcomes: Vec<_> = windowed
            .into_iter()
            .map(|worker| worker.join().expect("a windowed worker panicked"))
            .collect();
        let fullscreen_outcome = fullscreen.join().expect("the fullscreen worker panicked");
        let failures = concurrent_retarget_failures(&windowed_outcomes, &fullscreen_outcome);
        assert!(
            failures.is_empty(),
            "concurrent retarget worker failures:\n{}",
            failures.join("\n"),
        );
    });
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

/// A `WM_SIZE` that arrives during a fullscreen device's release resizes nothing.
///
/// Releasing a fullscreen device gives the window back. The mode restore is the
/// first thing the release does to the window, and where the mode-set is real
/// the window manager answers it with a `WM_SIZE` for the restored window
/// trimmed to the visible frame, delivered inside the restore call itself,
/// before the device's own window moves are guarded. Under the test prefix's
/// emulated mode-set nothing answers, so a second thread stands in for the
/// window manager: a cross-thread `SendMessage` waits until the window's thread
/// next pumps, which is that restore. A release that answered the message would
/// destroy the back buffer, its sRGB twin and the depth texture a second time
/// and leak their replacements, which is what ended the process on the Intel CI
/// image. The unix side refuses a destroy of a handle that is no longer live,
/// and a build with debug assertions ends the process at that refusal, which is
/// what makes this test fail without the guard.
#[test]
fn releasing_a_fullscreen_device_ignores_a_resize_during_the_release() {
    const WM_SIZE: u32 = 0x0005;
    if !display_lists_640x480() {
        return;
    }
    let h = Harness::with_depth();
    let mut pp = fullscreen_params(h.hwnd(), 640, 480);
    pp.enable_auto_depth_stencil = 1;
    pp.auto_depth_stencil_format = D3DFMT_D24S8;
    assert_eq!(
        h.reset_params(&mut pp),
        D3D_OK,
        "Reset to fullscreen 640x480"
    );
    {
        let (bb_hr, bb) = h.back_buffer(0).desc();
        assert_eq!(bb_hr, D3D_OK, "GetDesc on the fullscreen back buffer");
        assert_eq!(
            (bb.width, bb.height),
            (640, 480),
            "the mode is the back buffer"
        );
    }

    // The stand-in for the window manager: a size the back buffer does not
    // have, sent from another thread so it queues until the release pumps.
    let hwnd = h.hwnd();
    let armed = Barrier::new(2);
    std::thread::scope(|scope| {
        let sender = spawn_scoped(scope, || {
            armed.wait();
            let _ = mtld3d_tests::send_message(hwnd, WM_SIZE, 0, (0x1C8 << 16) | 0x258);
        });
        armed.wait();
        // Long enough for the sender to reach its `SendMessage` and block there;
        // a message that arrives after the release is pumped harmlessly below.
        std::thread::sleep(std::time::Duration::from_millis(100));

        assert_eq!(
            h.release_device(),
            0,
            "the harness held the only device reference"
        );
        assert!(h.pump(), "no WM_QUIT expected");
        sender
            .join()
            .expect("the sender thread ends once its message is answered");
    });
    let rect = h.window_rect();
    assert!(
        rect.right - rect.left > 0 && rect.bottom - rect.top > 0,
        "the window outlived the device: {rect:?}",
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
fn cursor_and_mask_uses_word_aligned_ddb_rows() {
    let h = Harness::with_config("cursor.scale=1;cursor.software=false");
    let mut shown = false;

    for side in [8usize, 16, 32] {
        let side_u32 = u32::try_from(side).expect("cursor side fits u32");
        let bitmap =
            h.create_offscreen_plain_surface(side_u32, side_u32, D3DFMT_A8R8G8B8, D3DPOOL_SCRATCH);
        let mut pixels = vec![0xFF00_FFFF; side * side];
        for row in 0..side {
            pixels[row * side + row] = 0x0000_FFFF;
        }
        {
            let mut locked = bitmap.lock_rect(0);
            locked.write_u32_rect(side, side, &pixels);
        }
        assert_eq!(h.set_cursor_properties_hr(0, 0, &bitmap), D3D_OK);
        if !shown {
            assert_eq!(h.show_cursor(true), 0, "cursor starts hidden");
            shown = true;
        }

        let word_stride = side.div_ceil(16) * 2;
        let actual = cursor_mask_bits(h.thread_cursor(), word_stride * side);
        let mut expected = vec![0u8; word_stride * side];
        for row in 0..side {
            expected[row * word_stride + row / 8] = 1u8 << (7 - (row & 7));
        }
        let reversed: Vec<u8> = expected
            .chunks_exact(word_stride)
            .rev()
            .flatten()
            .copied()
            .collect();
        assert!(
            actual == expected || actual == reversed,
            "{side}-pixel cursor AND rows were not preserved: {actual:02x?}",
        );
    }
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

/// Releasing the device destroys every HCURSOR it built.
///
/// `SetCursorProperties` builds one Win32 cursor per distinct bitmap and
/// keeps every one of them for the device's lifetime, so a game that cycles
/// through pointers hands the device a growing set of handles. They are the
/// device's alone and go with it. The handle that is the thread's cursor at
/// release is replaced by the window's class cursor first: user32 frees a
/// cursor even while it is current, and the thread would otherwise keep a
/// destroyed handle as its cursor.
#[test]
fn device_release_destroys_the_cursors_it_built() {
    const WM_SETCURSOR: u32 = 0x0020;
    /// `WM_MOUSEMOVE` as the trigger message in `WM_SETCURSOR`'s lparam.
    const WM_MOUSEMOVE_LP: isize = 0x0200;
    const HTCLIENT: isize = 1;
    const SIDE: usize = 32;
    let h = Harness::new();
    let lp_client_move = (WM_MOUSEMOVE_LP << 16) | HTCLIENT;

    // Hidden, the message is forwarded and the class cursor is what applies.
    h.set_thread_cursor(0);
    h.send_window_message(WM_SETCURSOR, h.hwnd(), lp_client_move);
    let class_arrow = h.thread_cursor();
    assert_ne!(class_arrow, 0, "the window class carries a cursor");

    let mut built = Vec::new();
    for fill in [0xFF00_0000_u32, 0xFFFF_0000, 0xFF00_FF00] {
        let bitmap = h.create_offscreen_plain_surface(
            u32::try_from(SIDE).expect("cursor side fits u32"),
            u32::try_from(SIDE).expect("cursor side fits u32"),
            D3DFMT_A8R8G8B8,
            D3DPOOL_SCRATCH,
        );
        {
            let mut locked = bitmap.lock_rect(0);
            locked.write_u32_rect(SIDE, SIDE, &[fill; SIDE * SIDE]);
        }
        assert_eq!(h.set_cursor_properties_hr(0, 0, &bitmap), D3D_OK);
        assert_eq!(
            h.show_cursor(true),
            i32::from(!built.is_empty()),
            "ShowCursor(TRUE) reports the previous visibility",
        );
        let handle = h.thread_cursor();
        assert_ne!(handle, 0, "ShowCursor(TRUE) must realize an HCURSOR");
        assert_ne!(
            handle, class_arrow,
            "the device cursor is not the class cursor"
        );
        assert!(
            !built.contains(&handle),
            "a distinct bitmap builds a distinct cursor: {handle:#x} again",
        );
        assert!(
            cursor_is_live(handle),
            "the realized handle is a live cursor"
        );
        built.push(handle);
    }
    assert_eq!(
        h.release_device(),
        0,
        "the harness held the only device reference"
    );

    for handle in &built {
        assert!(
            !cursor_is_live(*handle),
            "cursor {handle:#x} outlived the device that built it",
        );
    }
    assert!(
        !built.contains(&h.thread_cursor()),
        "the thread cursor must not be a destroyed handle: {:#x}",
        h.thread_cursor(),
    );
    assert_eq!(
        h.thread_cursor(),
        class_arrow,
        "the window's class cursor replaces the device's on the thread",
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
fn cursor_rejects_invalid_bitmaps_without_replacing_the_previous_cursor() {
    for config in [
        "cursor.software=false;cursor.scale=2",
        "cursor.software=true;cursor.scale=2",
    ] {
        let h = Harness::with_config(config);
        let bitmap = h.create_offscreen_plain_surface(32, 32, D3DFMT_A8R8G8B8, D3DPOOL_SCRATCH);
        bitmap
            .lock_rect(0)
            .write_u32_rect(32, 32, &[0xFFFF_8040; 32 * 32]);
        assert_eq!(h.set_cursor_properties_hr(2, 3, &bitmap), D3D_OK);
        assert_eq!(h.show_cursor(true), 0);
        let previous = h.thread_cursor();
        assert_ne!(previous, 0);
        for format in [D3DFMT_R5G6B5, D3DFMT_L8, D3DFMT_X8R8G8B8] {
            let invalid = h.create_offscreen_plain_surface(128, 128, format, D3DPOOL_SCRATCH);
            assert_eq!(
                h.set_cursor_properties_hr(0, 0, &invalid),
                D3DERR_INVALIDCALL
            );
            assert_eq!(h.thread_cursor(), previous, "{config}: format {format}");
            assert_eq!(h.show_cursor(true), 1, "rejection preserves visibility");
        }
        for (pitch, null_bits) in [(-128, false), (0, false), (64, false), (128, true)] {
            assert_eq!(
                h.cursor_with_invalid_layout(&bitmap, pitch, null_bits),
                (D3DERR_INVALIDCALL, 1, 1)
            );
            assert_eq!(h.thread_cursor(), previous, "{config}: invalid lock layout");
        }
        assert_eq!(
            h.set_cursor_properties_hr(u32::MAX, 0, &bitmap),
            D3DERR_INVALIDCALL
        );
        assert_eq!(
            h.thread_cursor(),
            previous,
            "hotspot scaling rejection preserves cursor"
        );
        assert_eq!(
            h.set_cursor_properties_hr(2, 3, &bitmap),
            D3D_OK,
            "successful locks were balanced"
        );
        assert_eq!(h.show_cursor(false), 1);
        assert_eq!(h.show_cursor(true), 0);
        assert_eq!(h.thread_cursor(), previous);
    }
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
    // process would read them. Both reads go through the shared lock, so a
    // window another test holds open on its own thread cannot be mistaken
    // for a leak.
    let before = config_var();
    let own = Harness::factory_only_with_config("caps.dfFormats=false");
    assert_eq!(
        config_var(),
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
