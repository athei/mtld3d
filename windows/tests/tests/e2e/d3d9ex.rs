//! `Direct3DCreate9Ex` and the extended interfaces it leads to.
//!
//! The export resolves by name and hands back an `IDirect3D9Ex`; a null out
//! slot is `D3DERR_INVALIDCALL`. The extended factory, device and swap chain
//! are the base objects with an extended flag: each answers its extended IID
//! and shares one object and one count with its base interface, while a
//! plain interface, device or swap chain refuses those IIDs. A device made by
//! the base `CreateDevice` on an extended factory is extended too. The
//! factory's extended mode list matches the base one, the adapter identifier
//! reports WHQL level 1, and the adapter has one stable non-zero LUID. The
//! extended stubs answer what a device with nothing to do there answers, and
//! `GetPresentStats` writes the struct's own size on each architecture and
//! nothing past it. `TestCooperativeLevel` answers `D3D_OK` on an extended
//! device, after a failed submission too, while `PresentEx`, `Present`, the
//! swap chain's `Present`, `CheckDeviceState`, `ResetEx` and `Reset` report
//! the failure with one code. A whole-surface `StretchRect` into a texture
//! level that is no render target reaches the versions later renames of that
//! texture reuse.

use core::ffi::{c_char, c_void};

use mtld3d_tests::{
    Factory, Harness, HarnessConfig, PRESENT_STATS_GUARD, PRESENT_STATS_PROBE_BYTES, Surface,
    Texture, TexturedVertex, assert_pixel_eq,
};
use mtld3d_types::{
    D3D_OK, D3DDISPLAYMODE, D3DDISPLAYMODEEX_SIZE, D3DDISPLAYMODEFILTER,
    D3DDISPLAYROTATION_IDENTITY, D3DERR_DEVICELOST, D3DERR_INVALIDCALL, D3DFMT_A8R8G8B8,
    D3DFMT_R5G6B5, D3DFMT_X8R8G8B8, D3DFVF_DIFFUSE, D3DFVF_TEX1, D3DFVF_XYZ, D3DPOOL_DEFAULT,
    D3DPOOL_SYSTEMMEM, D3DPRESENT_DONOTWAIT, D3DPRESENT_FORCEIMMEDIATE, D3DPT_TRIANGLELIST,
    D3DRECT, D3DSAMP_MAGFILTER, D3DSAMP_MINFILTER, D3DSCANLINEORDERING_INTERLACED,
    D3DSCANLINEORDERING_PROGRESSIVE, D3DSCANLINEORDERING_UNKNOWN, D3DSDK_VERSION, D3DTEXF_NONE,
    D3DTEXF_POINT, E_NOINTERFACE, IDirect3D9Vtbl, IID_IDIRECT3D9, IID_IDIRECT3D9EX,
    IID_IDIRECT3DDEVICE9, IID_IDIRECT3DDEVICE9EX, IID_IDIRECT3DSWAPCHAIN9,
    IID_IDIRECT3DSWAPCHAIN9EX,
};

#[link(name = "kernel32")]
unsafe extern "system" {
    fn LoadLibraryA(name: *const c_char) -> *mut c_void;
    fn FreeLibrary(module: *mut c_void) -> i32;
    fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
}

type CreateExFn = unsafe extern "system" fn(u32, *mut *mut c_void) -> i32;

/// The bytes `D3DPRESENTSTATS` takes in this process, by the SDK header's layout.
///
/// `d3d9types.h` packs its structs to 4 bytes on x86, so the two
/// `LARGE_INTEGER` times follow the three counts at offset 12 there and the
/// struct is 28 bytes; elsewhere they sit at 16 and it is 32.
const PRESENT_STATS_BYTES: usize = if cfg!(target_arch = "x86") { 28 } else { 32 };

fn extended(factory: Factory) -> Harness {
    Harness::create(&HarnessConfig {
        factory,
        ..HarnessConfig::default()
    })
}

const fn filter(format: u32, scan_line_ordering: u32) -> D3DDISPLAYMODEFILTER {
    D3DDISPLAYMODEFILTER {
        size: 12,
        format,
        scan_line_ordering,
    }
}

#[test]
fn direct3d_create9_ex_resolves_and_creates_an_extended_interface() {
    // SAFETY: plain kernel32 call with a NUL-terminated name.
    let lib = unsafe { LoadLibraryA(c"d3d9.dll".as_ptr()) };
    assert!(!lib.is_null(), "LoadLibrary(d3d9.dll)");
    // SAFETY: `lib` is a live module handle and the name is NUL-terminated.
    let addr = unsafe { GetProcAddress(lib, c"Direct3DCreate9Ex".as_ptr()) };
    assert!(!addr.is_null(), "GetProcAddress(Direct3DCreate9Ex)");
    // SAFETY: the export has the documented `Direct3DCreate9Ex` signature.
    let create_ex: CreateExFn = unsafe { core::mem::transmute(addr) };

    let mut out: *mut c_void = core::ptr::null_mut();
    // SAFETY: the resolved export with the SDK version and a live out slot.
    let hr = unsafe { create_ex(D3DSDK_VERSION, &raw mut out) };
    assert_eq!(hr, D3D_OK, "Direct3DCreate9Ex creates an interface");
    assert!(!out.is_null(), "the out slot holds the interface");
    // SAFETY: `out` is a live interface whose first field is its vtable pointer.
    let vtbl_ptr = unsafe { *out.cast::<*const IDirect3D9Vtbl>() };
    // SAFETY: the vtable is a static the interface points to for its lifetime.
    let vtbl = unsafe { &*vtbl_ptr };
    let mut ex: *mut c_void = core::ptr::null_mut();
    // SAFETY: base vtable thunk on the live interface with a writable slot.
    let hr = unsafe { (vtbl.query_interface)(out, &IID_IDIRECT3D9EX, &raw mut ex) };
    assert_eq!(hr, D3D_OK, "the interface answers IID_IDirect3D9Ex");
    assert_eq!(ex, out, "as itself");
    // SAFETY: releases the query's reference, then the create's.
    assert_eq!(unsafe { (vtbl.release)(ex) }, 1, "one reference left");
    // SAFETY: the create's reference, the last one.
    assert_eq!(unsafe { (vtbl.release)(out) }, 0, "the interface is freed");

    // SAFETY: the resolved export; a null out slot is a documented invalid call.
    let hr = unsafe { create_ex(D3DSDK_VERSION, core::ptr::null_mut()) };
    assert_eq!(hr, D3DERR_INVALIDCALL, "a null out slot is an invalid call");

    // SAFETY: balancing the LoadLibrary above.
    assert_ne!(unsafe { FreeLibrary(lib) }, 0, "FreeLibrary(d3d9.dll)");
}

#[test]
fn the_extended_and_base_interfaces_are_one_object_with_one_count() {
    let h = extended(Factory::Extended);
    h.add_ref_factory();
    let factory_base = h.release_factory();
    for iid in [IID_IDIRECT3D9, IID_IDIRECT3D9EX] {
        let (hr, same, held) = h.factory_query_interface(&iid);
        assert_eq!(hr, D3D_OK, "factory QueryInterface");
        assert!(same, "the factory hands back itself");
        assert_eq!(
            held,
            factory_base + 1,
            "the query took one reference on the shared count"
        );
    }
    let device_base = h.device_refcount();
    for iid in [IID_IDIRECT3DDEVICE9, IID_IDIRECT3DDEVICE9EX] {
        let (hr, same, held) = h.device_query_interface(&iid);
        assert_eq!(hr, D3D_OK, "device QueryInterface");
        assert!(same, "the device hands back itself");
        assert_eq!(
            held,
            device_base + 1,
            "the query took one reference on the shared count"
        );
    }
    let chain = h.implicit_swapchain();
    for iid in [IID_IDIRECT3DSWAPCHAIN9, IID_IDIRECT3DSWAPCHAIN9EX] {
        assert_eq!(
            chain.query_interface(&iid),
            (D3D_OK, true),
            "swap chain QueryInterface"
        );
    }
}

#[test]
fn a_plain_factory_device_and_swap_chain_refuse_the_extended_iids() {
    let h = Harness::new();
    let (hr, same, _) = h.factory_query_interface(&IID_IDIRECT3D9EX);
    assert_eq!((hr, same), (E_NOINTERFACE, false), "plain factory");
    let (hr, same, _) = h.device_query_interface(&IID_IDIRECT3DDEVICE9EX);
    assert_eq!((hr, same), (E_NOINTERFACE, false), "plain device");
    let chain = h.implicit_swapchain();
    assert_eq!(
        chain.query_interface(&IID_IDIRECT3DSWAPCHAIN9EX),
        (E_NOINTERFACE, false),
        "plain swap chain"
    );
}

#[test]
fn every_device_an_extended_factory_creates_is_extended() {
    for factory in [Factory::Extended, Factory::ExtendedDeviceEx] {
        let h = extended(factory);
        let (hr, same, _) = h.device_query_interface(&IID_IDIRECT3DDEVICE9EX);
        assert_eq!(
            (hr, same),
            (D3D_OK, true),
            "the device answers IID_IDirect3DDevice9Ex"
        );
        assert_eq!(
            h.implicit_swapchain()
                .query_interface(&IID_IDIRECT3DSWAPCHAIN9EX),
            (D3D_OK, true),
            "its swap chain answers IID_IDirect3DSwapChain9Ex"
        );
        h.render_once(0xFF00_FF00, |_| {});
        assert_pixel_eq(
            h.read_pixel(320, 240),
            0xFF00_FF00,
            "the extended device draws",
        );
    }
}

#[test]
fn whql_level_is_one_on_an_extended_interface_and_zero_on_a_plain_one() {
    assert_eq!(
        Harness::factory_only_extended()
            .adapter_identifier()
            .whql_level,
        1
    );
    assert_eq!(Harness::factory_only().adapter_identifier().whql_level, 0);
}

#[test]
fn the_adapter_luid_is_nonzero_and_the_same_for_every_interface() {
    let first = Harness::factory_only_extended();
    let (hr, luid) = first.adapter_luid();
    assert_eq!(hr, D3D_OK, "GetAdapterLUID");
    assert_ne!(luid, (0, 0), "a LUID names the adapter");
    assert_eq!(first.adapter_luid(), (D3D_OK, luid), "a second call agrees");
    let second = Harness::factory_only_extended();
    assert_eq!(
        second.adapter_luid(),
        (D3D_OK, luid),
        "another interface agrees"
    );
    assert_eq!(
        first.adapter_luid_null_hr(),
        D3DERR_INVALIDCALL,
        "a null out slot"
    );
}

#[test]
fn the_extended_mode_list_is_the_base_list_progressive() {
    let h = Harness::factory_only_extended();
    h.hold_display_mode();
    for format in [D3DFMT_X8R8G8B8, D3DFMT_R5G6B5] {
        let count = h.adapter_mode_count(format);
        for ordering in [D3DSCANLINEORDERING_UNKNOWN, D3DSCANLINEORDERING_PROGRESSIVE] {
            let any = filter(format, ordering);
            assert_eq!(
                h.adapter_mode_count_ex(Some(&any)),
                count,
                "format {format}"
            );
            for index in 0..count {
                let mut base = D3DDISPLAYMODE {
                    width: 0,
                    height: 0,
                    refresh_rate: 0,
                    format: 0,
                };
                assert_eq!(h.enum_adapter_modes(format, index, &mut base), D3D_OK);
                let (hr, mode) = h.enum_adapter_modes_ex(&any, index);
                assert_eq!(hr, D3D_OK, "EnumAdapterModesEx {index}");
                assert_eq!(
                    (mode.width, mode.height, mode.refresh_rate, mode.format),
                    (base.width, base.height, base.refresh_rate, base.format),
                    "mode {index} matches the base list"
                );
                assert_eq!(
                    mode.size, D3DDISPLAYMODEEX_SIZE,
                    "the size field is filled in"
                );
                assert_eq!(mode.scan_line_ordering, D3DSCANLINEORDERING_PROGRESSIVE);
            }
            assert_eq!(
                h.enum_adapter_modes_ex(&any, count).0,
                D3DERR_INVALIDCALL,
                "past the list"
            );
        }
        let interlaced = filter(format, D3DSCANLINEORDERING_INTERLACED);
        assert_eq!(
            h.adapter_mode_count_ex(Some(&interlaced)),
            0,
            "no interlaced modes"
        );
        assert_eq!(
            h.enum_adapter_modes_ex(&interlaced, 0).0,
            D3DERR_INVALIDCALL
        );
    }
    assert_eq!(h.adapter_mode_count_ex(None), 0, "a null filter");
    let no_display_format = filter(D3DFMT_A8R8G8B8, D3DSCANLINEORDERING_UNKNOWN);
    assert_eq!(
        h.adapter_mode_count_ex(Some(&no_display_format)),
        0,
        "not a display format"
    );
}

#[test]
fn display_mode_ex_checks_the_size_and_reports_the_identity_rotation() {
    let h = extended(Factory::Extended);
    h.hold_display_mode();
    let mut adapter = D3DDISPLAYMODE {
        width: 0,
        height: 0,
        refresh_rate: 0,
        format: 0,
    };
    assert_eq!(h.adapter_display_mode(&mut adapter), D3D_OK);
    let (hr, mode, rotation) = h.adapter_display_mode_ex(D3DDISPLAYMODEEX_SIZE);
    assert_eq!(hr, D3D_OK, "GetAdapterDisplayModeEx");
    assert_eq!(
        (mode.width, mode.height, mode.format),
        (adapter.width, adapter.height, adapter.format),
        "the adapter's current mode"
    );
    assert_eq!(mode.scan_line_ordering, D3DSCANLINEORDERING_PROGRESSIVE);
    assert_eq!(rotation, D3DDISPLAYROTATION_IDENTITY);
    assert_eq!(
        h.adapter_display_mode_ex(0).0,
        D3DERR_INVALIDCALL,
        "adapter, wrong size"
    );

    let mut device = D3DDISPLAYMODE {
        width: 0,
        height: 0,
        refresh_rate: 0,
        format: 0,
    };
    assert_eq!(h.display_mode(&mut device), D3D_OK);
    let (hr, mode, rotation) = h.device_display_mode_ex(0, D3DDISPLAYMODEEX_SIZE);
    assert_eq!(hr, D3D_OK, "device GetDisplayModeEx");
    assert_eq!(
        (mode.width, mode.height, mode.refresh_rate, mode.format),
        (
            device.width,
            device.height,
            device.refresh_rate,
            device.format
        ),
        "the mode GetDisplayMode reports"
    );
    assert_eq!(rotation, D3DDISPLAYROTATION_IDENTITY);
    assert_eq!(
        h.device_display_mode_ex(0, 0).0,
        D3DERR_INVALIDCALL,
        "device, wrong size"
    );
    assert_eq!(
        h.device_display_mode_ex(1, D3DDISPLAYMODEEX_SIZE).0,
        D3DERR_INVALIDCALL,
        "no second swap chain"
    );

    let chain = h.implicit_swapchain();
    let (hr, mode, rotation) = chain.display_mode_ex(D3DDISPLAYMODEEX_SIZE);
    assert_eq!(hr, D3D_OK, "swap chain GetDisplayModeEx");
    assert_eq!((mode.width, mode.height), (device.width, device.height));
    assert_eq!(rotation, D3DDISPLAYROTATION_IDENTITY);
    assert_eq!(
        chain.display_mode_ex(0).0,
        D3DERR_INVALIDCALL,
        "swap chain, wrong size"
    );
}

#[test]
fn the_extended_stubs_answer_what_a_device_with_nothing_to_do_answers() {
    let h = extended(Factory::Extended);
    assert_eq!(h.compose_rects(), D3D_OK, "ComposeRects");
    assert_eq!(h.gpu_thread_priority(), (D3D_OK, 0), "GetGPUThreadPriority");
    assert_eq!(h.set_gpu_thread_priority(3), D3D_OK, "SetGPUThreadPriority");
    assert_eq!(
        h.gpu_thread_priority(),
        (D3D_OK, 0),
        "the priority is not stored"
    );
    assert_eq!(h.wait_for_vblank(0), D3D_OK, "WaitForVBlank");
    assert_eq!(
        h.wait_for_vblank(1),
        D3DERR_INVALIDCALL,
        "no second swap chain"
    );
    assert_eq!(
        h.check_resource_residency(),
        D3D_OK,
        "CheckResourceResidency"
    );
    assert_eq!(
        h.set_convolution_mono_kernel(),
        D3DERR_INVALIDCALL,
        "SetConvolutionMonoKernel"
    );
    assert_eq!(h.check_device_state(), D3D_OK, "CheckDeviceState");
    let chain = h.implicit_swapchain();
    assert_eq!(
        chain.last_present_count(),
        (D3D_OK, 0),
        "GetLastPresentCount"
    );
    let (hr, stats) = chain.present_stats();
    assert_eq!(hr, D3D_OK, "GetPresentStats");
    assert_eq!(
        (
            stats.present_count,
            stats.present_refresh_count,
            stats.sync_refresh_count,
            stats.sync_qpc_time,
            stats.sync_gpu_time,
        ),
        (0, 0, 0, [0; 2], [0; 2]),
        "zeroed statistics"
    );
}

#[test]
fn present_stats_fill_the_struct_and_nothing_past_it() {
    let h = extended(Factory::Extended);
    let (hr, bytes) = h.implicit_swapchain().present_stats_guarded();
    assert_eq!(hr, D3D_OK, "GetPresentStats");
    assert!(
        bytes[..PRESENT_STATS_BYTES].iter().all(|&b| b == 0),
        "the {PRESENT_STATS_BYTES} bytes of the struct are zeroed: {bytes:02x?}"
    );
    assert!(
        bytes[PRESENT_STATS_BYTES..PRESENT_STATS_PROBE_BYTES]
            .iter()
            .all(|&b| b == PRESENT_STATS_GUARD),
        "nothing is written past the struct: {bytes:02x?}"
    );
}

#[test]
fn an_extended_device_answers_test_cooperative_level_with_ok_after_a_failure() {
    let h = Harness::create(&HarnessConfig {
        factory: Factory::Extended,
        config_entries: "debug.failNextSubmit=true",
        ..HarnessConfig::default()
    });
    assert_eq!(h.test_cooperative_level(), D3D_OK, "before any failure");
    assert_eq!(
        h.present_ex(0),
        D3DERR_DEVICELOST,
        "the first submission is refused"
    );
    assert_eq!(
        h.test_cooperative_level(),
        D3D_OK,
        "an extended device's TestCooperativeLevel always answers D3D_OK"
    );
    assert_eq!(
        h.check_device_state(),
        D3DERR_DEVICELOST,
        "CheckDeviceState reports the failure"
    );
    assert_eq!(
        h.present_ex(0),
        D3DERR_DEVICELOST,
        "PresentEx reports it again"
    );
    assert_eq!(h.present(), D3DERR_DEVICELOST, "so does Present");
    assert_eq!(
        h.present_swapchain(),
        D3DERR_DEVICELOST,
        "and the swap chain's Present"
    );
    let (width, height) = h.dims();
    let mut pp = h.windowed_present_params(width, height);
    assert_eq!(
        h.reset_ex(&mut pp, None),
        D3DERR_DEVICELOST,
        "ResetEx does not clear it"
    );
    let mut pp = h.windowed_present_params(width, height);
    assert_eq!(h.reset_params(&mut pp), D3DERR_DEVICELOST, "nor does Reset");
    assert_eq!(
        h.test_cooperative_level(),
        D3D_OK,
        "TestCooperativeLevel still answers D3D_OK"
    );
}

#[test]
fn present_ex_presents_whatever_flags_it_carries() {
    let h = extended(Factory::ExtendedDeviceEx);
    for flags in [0, D3DPRESENT_DONOTWAIT, D3DPRESENT_FORCEIMMEDIATE] {
        assert!(h.pump(), "WM_QUIT before render");
        assert_eq!(h.begin_scene(), D3D_OK);
        assert_eq!(h.clear_target(0xFF00_00FF), D3D_OK);
        assert_eq!(h.end_scene(), D3D_OK);
        assert_eq!(h.present_ex(flags), D3D_OK, "PresentEx({flags:#x})");
    }
    assert_pixel_eq(h.read_pixel(320, 240), 0xFF00_00FF, "the frame drew");
}

#[test]
fn an_overlay_probe_of_the_extended_device_leaves_a_plain_device_working() {
    // An overlay creates an extended device, presents through PresentEx and
    // resets through ResetEx to learn the slots it hooks, then lets go of it;
    // the title's own plain device comes after.
    {
        let probe = extended(Factory::ExtendedDeviceEx);
        assert_eq!(probe.present_ex(0), D3D_OK, "PresentEx");
        let (width, height) = probe.dims();
        let mut pp = probe.windowed_present_params(width, height);
        assert_eq!(probe.reset_ex(&mut pp, None), D3D_OK, "ResetEx");
        assert_eq!(probe.present_ex(0), D3D_OK, "PresentEx after ResetEx");
    }
    let h = Harness::new();
    h.render_once(0xFFFF_0000, |_| {});
    assert_pixel_eq(
        h.read_pixel(320, 240),
        0xFFFF_0000,
        "the plain device draws",
    );
}

/// A whole-surface `StretchRect` into a renamed texture reaches the versions it reuses.
///
/// An extended device copies a whole surface into a default-pool texture
/// level that is no render target. Such a texture's renames reuse retired
/// versions and copy into them only what the texture's write history says
/// changed since they retired, so the copy has to be in that history: frame
/// 0 renames the texture (its first version retires holding only the fill),
/// frame 1 copies a whole surface of another colour over it and patches it
/// once before drawing, and frame 2's rename reuses that first version, which
/// must come back holding the copy everywhere the patches do not reach. The
/// copy leaves the level on the GPU alone, so the first CPU write after it
/// reads the level back with a blocking flush; frame 1's patch takes that
/// flush before any draw, so frame 2's draws and patch share one submission
/// and its upload renames.
#[test]
fn a_whole_surface_stretch_rect_reaches_the_versions_a_rename_reuses() {
    const FILL: u32 = 0xFF80_8080;
    const PATCH: u32 = 0xFFFF_0000;
    const COPY: u32 = 0xFF00_00FF;
    const LATER: u32 = 0xFF00_FF00;
    const READ_BACK: u32 = 0xFFFF_FF00;
    let h = extended(Factory::Extended);
    let texture = h.create_texture(256, 256, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    let level = texture.surface_level(0);
    let source = h.create_texture(256, 256, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    let staging = h.create_offscreen_plain_surface(256, 256, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    let fill = |dst: &Surface<'_>, rect: [u32; 4], color: u32| {
        let [x0, y0, x1, y1] = rect.map(|v| i32::try_from(v).expect("small"));
        let (w, hgt) = ((rect[2] - rect[0]) as usize, (rect[3] - rect[1]) as usize);
        staging
            .lock_rect_partial(&[x0, y0, x1, y1], 0)
            .write_u32_rect(w, hgt, &vec![color; w * hgt]);
        let src_rect = D3DRECT {
            x1: x0,
            y1: y0,
            x2: x1,
            y2: y1,
        };
        assert_eq!(
            h.update_surface_region_hr(&staging, &src_rect, dst, (x0, y0)),
            D3D_OK
        );
    };
    fill(&level, [0, 0, 256, 256], FILL);
    fill(&source.surface_level(0), [0, 0, 256, 256], COPY);
    bind_point_sampled(&h, &texture);
    let quad = fullscreen_textured_quad();
    let draw = |d: &Harness| {
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad), D3D_OK);
    };
    // Frame 0: the second draw's upload renames the texture.
    h.render_once(0xFF00_0000, |d| {
        draw(d);
        fill(&level, [0, 0, 64, 64], PATCH);
        draw(d);
    });
    assert_pixel_eq(h.read_pixel(40, 30), PATCH, "frame 0 patch");
    // Frame 1: the extended whole-surface copy, then a patch that reads the
    // level back ahead of the frame's draw.
    assert_eq!(
        h.stretch_rect(&source.surface_level(0), &level, D3DTEXF_NONE),
        D3D_OK
    );
    h.render_once(0xFF00_0000, |d| {
        fill(&level, [64, 0, 128, 64], READ_BACK);
        draw(d);
    });
    assert_pixel_eq(h.read_pixel(320, 240), COPY, "the copy landed");
    assert_pixel_eq(h.read_pixel(240, 30), READ_BACK, "the patch after it");
    // Frame 2: this rename reuses the version frame 0 retired.
    h.render_once(0xFF00_0000, |d| {
        draw(d);
        fill(&level, [192, 192, 256, 256], LATER);
        draw(d);
    });
    for (x, y, expected, what) in [
        (40, 30, COPY, "where frame 0's patch was"),
        (240, 30, READ_BACK, "frame 1's patch"),
        (320, 240, COPY, "the middle"),
        (600, 30, COPY, "the top right"),
        (600, 450, LATER, "frame 2's patch"),
    ] {
        assert_pixel_eq(h.read_pixel(x, y), expected, what);
    }
    assert_eq!(h.clear_texture(0), D3D_OK);
}

/// Bind `texture` at stage 0, point-sampled, its colour alone.
fn bind_point_sampled(h: &Harness, texture: &Texture<'_>) {
    assert_eq!(h.set_texture(0, texture), D3D_OK);
    h.select_texture_stage(0);
    for (state, value) in [
        (D3DSAMP_MINFILTER, D3DTEXF_POINT),
        (D3DSAMP_MAGFILTER, D3DTEXF_POINT),
    ] {
        assert_eq!(h.set_sampler_state(0, state, value), D3D_OK);
    }
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1), D3D_OK);
}

/// Two clockwise triangles over the whole back buffer, sampling the unit square.
const fn fullscreen_textured_quad() -> [TexturedVertex; 6] {
    const fn vertex(x: f32, y: f32, u: f32, v: f32) -> TexturedVertex {
        TexturedVertex {
            x,
            y,
            z: 0.5,
            color: 0xFFFF_FFFF,
            u,
            v,
        }
    }
    [
        vertex(-1.0, 1.0, 0.0, 0.0),
        vertex(1.0, 1.0, 1.0, 0.0),
        vertex(-1.0, -1.0, 0.0, 1.0),
        vertex(1.0, 1.0, 1.0, 0.0),
        vertex(1.0, -1.0, 1.0, 1.0),
        vertex(-1.0, -1.0, 0.0, 1.0),
    ]
}
