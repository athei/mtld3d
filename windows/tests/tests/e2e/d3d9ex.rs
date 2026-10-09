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
//! device, after a failed submission too, while `PresentEx`,
//! `CheckDeviceState` and `ResetEx` report the failure.

use core::ffi::{c_char, c_void};

use mtld3d_tests::{
    Factory, Harness, HarnessConfig, PRESENT_STATS_GUARD, PRESENT_STATS_PROBE_BYTES,
    assert_pixel_eq,
};
use mtld3d_types::{
    D3D_OK, D3DDISPLAYMODE, D3DDISPLAYMODEEX_SIZE, D3DDISPLAYMODEFILTER,
    D3DDISPLAYROTATION_IDENTITY, D3DERR_DEVICELOST, D3DERR_INVALIDCALL, D3DFMT_A8R8G8B8,
    D3DFMT_R5G6B5, D3DFMT_X8R8G8B8, D3DPRESENT_DONOTWAIT, D3DPRESENT_FORCEIMMEDIATE,
    D3DSCANLINEORDERING_INTERLACED, D3DSCANLINEORDERING_PROGRESSIVE, D3DSCANLINEORDERING_UNKNOWN,
    D3DSDK_VERSION, E_NOINTERFACE, IDirect3D9Vtbl, IID_IDIRECT3D9, IID_IDIRECT3D9EX,
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
    let (width, height) = h.dims();
    let mut pp = h.windowed_present_params(width, height);
    assert_eq!(
        h.reset_ex(&mut pp, None),
        D3DERR_DEVICELOST,
        "ResetEx does not clear it"
    );
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
