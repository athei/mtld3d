//! Device + factory lifecycle.
//!
//! `IDirect3D9` queries, caps, `TestCooperativeLevel`, and `Reset`
//! (state-default restore, resize, malformed input).

use mtld3d_tests::{Harness, WS_CAPTION, WS_EX_TOPMOST, WS_POPUP, WS_VISIBLE, assert_pixel_eq};
use mtld3d_types::{
    D3D_OK, D3DDISPLAYMODE, D3DERR_DEVICENOTRESET, D3DERR_INVALIDCALL, D3DERR_NOTAVAILABLE,
    D3DFILL_SOLID, D3DFMT_A2R10G10B10, D3DFMT_A8R8G8B8, D3DFMT_A16B16G16R16, D3DFMT_A16B16G16R16F,
    D3DFMT_A32B32G32R32F, D3DFMT_ATI1, D3DFMT_D24S8, D3DFMT_DXT1, D3DFMT_G16R16, D3DFMT_G16R16F,
    D3DFMT_G32R32F, D3DFMT_L8, D3DFMT_R5G6B5, D3DFMT_R16F, D3DFMT_R32F, D3DFMT_UYVY,
    D3DFMT_X8R8G8B8, D3DFMT_YUY2, D3DFVF_XYZ, D3DOK_NOAUTOGEN, D3DPOOL_DEFAULT, D3DPOOL_MANAGED,
    D3DPOOL_SCRATCH, D3DPOOL_SYSTEMMEM, D3DPRESENT_INTERVAL_IMMEDIATE, D3DPRESENT_INTERVAL_ONE,
    D3DPRESENT_PARAMETERS, D3DRS_FILLMODE, D3DRS_LIGHTING, D3DRTYPE_CUBETEXTURE, D3DRTYPE_SURFACE,
    D3DRTYPE_TEXTURE, D3DUSAGE_AUTOGENMIPMAP, D3DUSAGE_DEPTHSTENCIL,
    D3DUSAGE_QUERY_POSTPIXELSHADER_BLENDING, D3DUSAGE_QUERY_SRGBREAD, D3DUSAGE_QUERY_VERTEXTEXTURE,
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
    assert_eq!(
        h.check_device_format_conversion(D3DFMT_UYVY, D3DFMT_R5G6B5),
        D3D_OK,
        "UYVY -> R5G6B5 is decoded by the StretchRect render quad",
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
        swap_effect: 1, // D3DSWAPEFFECT_DISCARD (irrelevant — rejected on dims first)
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

/// Present parameters for a fullscreen Reset at `width`x`height`.
const fn fullscreen_params(hwnd: usize, width: u32, height: u32) -> D3DPRESENT_PARAMETERS {
    D3DPRESENT_PARAMETERS {
        back_buffer_width: width,
        back_buffer_height: height,
        back_buffer_format: D3DFMT_X8R8G8B8,
        back_buffer_count: 1,
        multi_sample_type: 0,
        multi_sample_quality: 0,
        swap_effect: 1, // D3DSWAPEFFECT_DISCARD
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
    let h = Harness::new();
    let hwnd = h.hwnd();
    let windowed_rect = h.window_rect();
    let windowed_style = h.window_style();

    let (screen_w, screen_h) = Harness::screen_size();
    // An enumerable non-monitor mode keeps the backbuffer assertion below
    // sharp: the window must adopt the monitor rect while the back buffer
    // must not.
    let mut pp = fullscreen_params(hwnd, 640, 480);
    assert_eq!(h.reset_params(&mut pp), D3D_OK, "fullscreen Reset");

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
        "backbuffer keeps the requested mode; the window covers the monitor",
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

#[test]
fn reset_fullscreen_honors_an_enumerable_mode() {
    let h = Harness::new();
    let (screen_w, screen_h) = Harness::screen_size();
    // 640x480 is served by EnumAdapterModes, so the back buffer keeps the
    // requested mode and a game that sizes its viewport from its own request
    // covers the frame. Present scales the back buffer to the drawable.
    let mut pp = fullscreen_params(h.hwnd(), 640, 480);
    assert_eq!(
        h.reset_params(&mut pp),
        D3D_OK,
        "fullscreen Reset at an enumerable mode must succeed",
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
    let rect = h.window_rect();
    assert_eq!(
        (
            u32::try_from(rect.right - rect.left).expect("width is positive"),
            u32::try_from(rect.bottom - rect.top).expect("height is positive")
        ),
        (screen_w, screen_h),
        "the window still covers the monitor",
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
    let (screen_w, screen_h) = Harness::screen_size();
    let h = Harness::fullscreen(640, 480);

    let rect = h.window_rect();
    assert_eq!(
        (
            u32::try_from(rect.right - rect.left).expect("width is positive"),
            u32::try_from(rect.bottom - rect.top).expect("height is positive")
        ),
        (screen_w, screen_h),
        "a fullscreen create covers the monitor",
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

    assert!(h.pump(), "no WM_QUIT expected");
    let rect = h.window_rect();
    assert_eq!(
        (
            u32::try_from(rect.right - rect.left).expect("width is positive"),
            u32::try_from(rect.bottom - rect.top).expect("height is positive")
        ),
        (screen_w, screen_h),
        "processing window events re-covers the monitor",
    );
    let (bb_hr, bb) = h.back_buffer(0).desc();
    assert_eq!(bb_hr, D3D_OK, "GetDesc after the external resize");
    assert_eq!(
        (bb.width, bb.height),
        (640, 480),
        "the back buffer never follows an external resize",
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
