//! Device + factory lifecycle.
//!
//! `IDirect3D9` queries, caps, `TestCooperativeLevel`, and `Reset`
//! (state-default restore, resize, malformed input).

use mtld3d_tests::{Harness, assert_pixel_eq};
use mtld3d_types::{
    D3D_OK, D3DDISPLAYMODE, D3DERR_INVALIDCALL, D3DERR_NOTAVAILABLE, D3DFILL_SOLID,
    D3DFMT_A2R10G10B10, D3DFMT_A8R8G8B8, D3DFMT_D24S8, D3DFMT_DXT1, D3DFMT_X8R8G8B8,
    D3DOK_NOAUTOGEN, D3DPOOL_SCRATCH, D3DPRESENT_PARAMETERS, D3DRS_FILLMODE, D3DRS_LIGHTING,
    D3DRTYPE_TEXTURE, D3DUSAGE_AUTOGENMIPMAP, D3DUSAGE_DEPTHSTENCIL, D3DVIEWPORT9,
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
    assert!(caps.max_streams >= 1, "no vertex streams");
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
