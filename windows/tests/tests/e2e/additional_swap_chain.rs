//! An additional swap chain has a back buffer of its own, at the size it asked for.
//!
//! WPF creates its device with a 1x1 back buffer on the application's window
//! and renders each window through an additional swap chain at the window's
//! size, from a thread that is not the window's. A window it presents through
//! GDI sizes a DIB from `GetDesc` of the chain's back buffer once, when the
//! chain is created, and copies the back buffer into it at the size it asked
//! for, so these tests run that sequence on a device of the same shape.

use core::{
    ffi::c_void,
    sync::atomic::{AtomicU32, Ordering},
};
use std::time::Duration;

use mtld3d_tests::{
    Factory, Harness, HarnessConfig, Surface, SwapChain, WM_SIZE, deref_vtbl, set_window_pos,
};
use mtld3d_types::{
    D3D_OK, D3DCREATE_HARDWARE_VERTEXPROCESSING, D3DCREATE_MULTITHREADED, D3DFMT_A8R8G8B8,
    D3DFMT_R5G6B5, D3DFMT_X8R8G8B8, D3DLOCK_READONLY, D3DMULTISAMPLE_4_SAMPLES, D3DPOOL_DEFAULT,
    D3DPOOL_SYSTEMMEM, D3DPRESENT_INTERVAL_IMMEDIATE, D3DPRESENT_PARAMETERS,
    D3DPRESENTFLAG_LOCKABLE_BACKBUFFER, D3DSWAPEFFECT_COPY, D3DTEXF_NONE, D3DUSAGE_RENDERTARGET,
    E_NOINTERFACE, Guid, IDirect3D9Vtbl,
};

use super::{
    device::{
        DRAWABLE_LOG_FILTER, PRESENT_OCCLUDED, await_acquired_drawables, await_logged_lines_within,
        run_in_private_log_child, run_in_private_log_child_with, running_as,
    },
    implicit_surface::client_size,
    render_target::rect,
};

/// The private-data key the release probe is stored under.
const RELEASE_PROBE_KEY: Guid = Guid {
    data1: 0x6d74_6c64,
    data2: 0x1196,
    data3: 0x0001,
    data4: *b"lastrele",
};
const RED: u32 = 0xFFFF_0000;
const GREEN: u32 = 0xFF00_FF00;
const BLUE: u32 = 0xFF00_00FF;
const RED_COLORREF: u32 = 0x0000_00FF;

/// The name the workload child of the present test runs under.
const PRESENT_CHILD_NAME: &str = "additional-chain-present.exe";

/// The name the workload child of the format warning test runs under.
const FORMAT_CHILD_NAME: &str = "additional-chain-format.exe";

/// An extended multithreaded device with a 1x1 back buffer on a window of real size.
fn one_pixel_device() -> Harness {
    let h = Harness::create(&HarnessConfig {
        factory: Factory::Extended,
        behavior_flags: D3DCREATE_HARDWARE_VERTEXPROCESSING | D3DCREATE_MULTITHREADED,
        back_buffer_format: D3DFMT_A8R8G8B8,
        ..HarnessConfig::default()
    });
    assert_eq!(h.reset(1, 1), D3D_OK, "the 1x1 back buffer");
    h
}

/// A lockable 32-bit copy-effect chain of `width`x`height` on the device window.
fn gdi_chain_params(h: &Harness, width: u32, height: u32) -> D3DPRESENT_PARAMETERS {
    D3DPRESENT_PARAMETERS {
        back_buffer_format: D3DFMT_A8R8G8B8,
        swap_effect: D3DSWAPEFFECT_COPY,
        flags: D3DPRESENTFLAG_LOCKABLE_BACKBUFFER,
        ..h.windowed_present_params(width, height)
    }
}

/// A chain the size of the window, and that size.
fn window_sized_chain(h: &Harness) -> (SwapChain<'_>, u32, u32) {
    let (width, height) = h.client_size();
    let mut pp = gdi_chain_params(h, width, height);
    (h.additional_swapchain_params(&mut pp), width, height)
}

/// The `IUnknown` head of [`ReleaseProbe`].
#[repr(C)]
struct ProbeVtbl {
    query_interface: extern "system" fn(*mut c_void, *const Guid, *mut *mut c_void) -> i32,
    add_ref: extern "system" fn(*mut c_void) -> u32,
    release: extern "system" fn(*mut c_void) -> u32,
}

static PROBE_VTBL: ProbeVtbl = ProbeVtbl {
    query_interface: probe_query_interface,
    add_ref: probe_add_ref,
    release: probe_release,
};

/// A COM object stored as private data, whose last `Release` counts the factory's references.
///
/// A surface's private data goes as the surface is finalized, right after its
/// textures retire. A device holds a reference on the factory that made it
/// until the end of its teardown, so the count says whether the device was
/// still there at that moment.
#[repr(C)]
struct ReleaseProbe {
    vtbl: &'static ProbeVtbl,
    refcount: AtomicU32,
    factory: *mut c_void,
    /// The factory's references when the last `Release` ran, 0 until it does.
    factory_refs: AtomicU32,
}

impl ReleaseProbe {
    const fn new(factory: *mut c_void) -> Self {
        Self {
            vtbl: &PROBE_VTBL,
            refcount: AtomicU32::new(0),
            factory,
            factory_refs: AtomicU32::new(0),
        }
    }

    const fn as_unknown(&self) -> *mut c_void {
        core::ptr::from_ref(self).cast_mut().cast::<c_void>()
    }
}

fn probe_from(this: *mut c_void) -> &'static ReleaseProbe {
    // SAFETY: the runtime only calls the probe through the pointer the test
    // stored, which names a `ReleaseProbe` that outlives the surface holding it.
    unsafe { &*this.cast::<ReleaseProbe>() }
}

extern "system" fn probe_query_interface(
    _this: *mut c_void,
    _riid: *const Guid,
    ppv: *mut *mut c_void,
) -> i32 {
    if !ppv.is_null() {
        // SAFETY: the caller's out-param, checked non-null.
        unsafe { *ppv = core::ptr::null_mut() };
    }
    E_NOINTERFACE
}

extern "system" fn probe_add_ref(this: *mut c_void) -> u32 {
    probe_from(this).refcount.fetch_add(1, Ordering::AcqRel) + 1
}

extern "system" fn probe_release(this: *mut c_void) -> u32 {
    let probe = probe_from(this);
    let remaining = probe.refcount.fetch_sub(1, Ordering::AcqRel) - 1;
    if remaining == 0 {
        probe
            .factory_refs
            .store(factory_refs(probe.factory), Ordering::Release);
    }
    remaining
}

/// The references on `factory`, a live `IDirect3D9`, read through `AddRef` and `Release`.
fn factory_refs(factory: *mut c_void) -> u32 {
    // SAFETY: the caller's contract; the vtable is `'static`.
    let vtbl = unsafe { deref_vtbl::<IDirect3D9Vtbl>(factory) };
    // SAFETY: a live object, and the release below balances the reference.
    let held = unsafe { (vtbl.add_ref)(factory) };
    // SAFETY: gives back the reference just taken.
    unsafe { (vtbl.release)(factory) };
    held - 1
}

/// The back buffer describes the chain it belongs to, whatever the device's size.
#[test]
fn a_back_buffer_reports_the_size_its_swap_chain_asked_for() {
    let h = one_pixel_device();
    let mut pp = gdi_chain_params(&h, 320, 200);
    let chain = h.additional_swapchain_params(&mut pp);
    let (hr, desc) = chain.back_buffer().desc();
    assert_eq!(hr, D3D_OK, "GetDesc on the chain's back buffer");
    assert_eq!((desc.width, desc.height), (320, 200), "the chain's size");
    assert_eq!(desc.format, D3DFMT_A8R8G8B8, "the chain's format");
    assert_eq!(desc.usage, D3DUSAGE_RENDERTARGET);
    assert_eq!(desc.pool, D3DPOOL_DEFAULT);
    let (hr, desc) = h.back_buffer(0).desc();
    assert_eq!(hr, D3D_OK, "GetDesc on the device's back buffer");
    assert_eq!((desc.width, desc.height), (1, 1), "the device's own size");
}

/// A format outside the A8/X8 pair reports X8R8G8B8, as the implicit back buffer does.
///
/// The storage is BGRA8 whatever the chain asked for, and the pixel size a
/// read-back works from has to be the storage's.
#[test]
fn a_back_buffer_reports_x8r8g8b8_for_a_format_it_does_not_store() {
    let h = one_pixel_device();
    let mut pp = D3DPRESENT_PARAMETERS {
        back_buffer_format: D3DFMT_R5G6B5,
        flags: 0,
        ..gdi_chain_params(&h, 320, 200)
    };
    let chain = h.additional_swapchain_params(&mut pp);
    let (hr, desc) = chain.back_buffer().desc();
    assert_eq!(hr, D3D_OK, "GetDesc on the chain's back buffer");
    assert_eq!(desc.format, D3DFMT_X8R8G8B8, "the format of the storage");
    let (hr, reported) = chain.present_parameters();
    assert_eq!(hr, D3D_OK);
    assert_eq!(
        reported.back_buffer_format, D3DFMT_R5G6B5,
        "the present parameters keep the request"
    );
}

/// A chain that asks for a format outside the A8/X8 pair says so in the log.
///
/// The warning latches once per process, so the workload runs in a child of
/// its own, whose device asks for A8R8G8B8 and leaves the chain the only asker.
#[test]
fn a_format_the_back_buffer_does_not_store_is_warned() {
    if running_as(FORMAT_CHILD_NAME) {
        unsupported_format_workload();
        return;
    }
    run_in_private_log_child(
        FORMAT_CHILD_NAME,
        "additional_swap_chain::a_format_the_back_buffer_does_not_store_is_warned",
        DRAWABLE_LOG_FILTER,
    );
}

/// Create an R5G6B5 chain on an A8R8G8B8 device and read the warning naming the format.
fn unsupported_format_workload() {
    let h = one_pixel_device();
    let mut pp = D3DPRESENT_PARAMETERS {
        back_buffer_format: D3DFMT_R5G6B5,
        flags: 0,
        ..gdi_chain_params(&h, 320, 200)
    };
    let _chain = h.additional_swapchain_params(&mut pp);
    await_logged_lines_within(
        &format!("back_buffer_format {D3DFMT_R5G6B5:#x} requested"),
        1,
        Duration::from_secs(10),
    );
}

/// A zero request on a window with no client area makes a back buffer of eight texels a side.
#[test]
fn a_zero_request_on_a_window_with_no_client_area_makes_eight_texels() {
    let h = one_pixel_device();
    set_window_pos(h.hwnd(), 0, 0, 0, 0);
    assert_eq!(h.client_size(), (0, 0), "the window has no client area");
    let mut pp = gdi_chain_params(&h, 0, 0);
    let chain = h.additional_swapchain_params(&mut pp);
    assert_eq!(
        (pp.back_buffer_width, pp.back_buffer_height),
        (8, 8),
        "the resolved size reaches the caller"
    );
    let (hr, desc) = chain.back_buffer().desc();
    assert_eq!(hr, D3D_OK, "GetDesc on the chain's back buffer");
    assert_eq!((desc.width, desc.height), (8, 8), "the chain's size");
}

/// A frame drawn into the back buffer reads back whole, at the size the chain asked for.
///
/// A sub-rect `StretchRect` out of the back buffer and a `GetRenderTargetData`
/// of all of it are the two reads a GDI present and a render into a
/// destination texture make of it; a row the read-back hands over is as wide
/// as the back buffer's `GetDesc` said.
#[test]
fn a_back_buffer_reads_back_at_its_own_size() {
    let h = one_pixel_device();
    let (chain, width, height) = window_sized_chain(&h);
    let back_buffer = chain.back_buffer();
    let (hr, desc) = back_buffer.desc();
    assert_eq!(hr, D3D_OK, "GetDesc on the chain's back buffer");
    assert_eq!((desc.width, desc.height), (width, height));

    assert_eq!(h.set_render_target(0, &back_buffer), D3D_OK);
    let vp = h.viewport();
    assert_eq!((vp.width, vp.height), (width, height), "the viewport");
    assert_eq!(h.clear_target(BLUE), D3D_OK, "clear the chain");
    assert_eq!(
        h.clear_target_rects(RED, &[rect(0, 0, 64, 64)]),
        D3D_OK,
        "clear a block of it"
    );

    let (right, bottom) = (
        i32::try_from(width).expect("window width"),
        i32::try_from(height).expect("window height"),
    );
    let inset = h.create_render_target(width - 22, height - 22, D3DFMT_A8R8G8B8);
    assert_eq!(
        h.stretch_rect_regions(
            &back_buffer,
            &rect(11, 11, right - 11, bottom - 11),
            &inset,
            &rect(0, 0, right - 22, bottom - 22),
            D3DTEXF_NONE,
        ),
        D3D_OK,
        "a rect past the device's 1x1 back buffer lies inside the chain's"
    );

    let sysmem =
        h.create_offscreen_plain_surface(width, height, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    assert_eq!(
        h.get_render_target_data_hr(&back_buffer, &sysmem),
        D3D_OK,
        "the whole back buffer reads back"
    );
    let pitch = width as usize;
    let lock = sysmem.lock_rect(D3DLOCK_READONLY);
    assert_eq!(lock.pitch().cast_unsigned(), width * 4, "the row pitch");
    let pixels = lock.as_u32(pitch * 100 + 101);
    assert_eq!(pixels[10], RED, "the block");
    assert_eq!(pixels[pitch * 100 + 100], BLUE, "the rest of the frame");
}

/// A `GetDC` of a lockable back buffer reads and writes the back buffer at its own size.
#[test]
fn a_lockable_back_buffer_hands_out_a_dc() {
    let h = one_pixel_device();
    let mut pp = gdi_chain_params(&h, 128, 128);
    let chain = h.additional_swapchain_params(&mut pp);
    let back_buffer = chain.back_buffer();
    assert_eq!(h.set_render_target(0, &back_buffer), D3D_OK);
    assert_eq!(h.clear_target(GREEN), D3D_OK, "clear the chain");
    {
        let dc = back_buffer.dc();
        assert_eq!(dc.get_pixel(100, 100), GREEN & 0x00FF_FFFF, "the DC reads");
        dc.fill_block(64, RED_COLORREF);
        assert_eq!(dc.release(), D3D_OK, "ReleaseDC");
    }
    let sysmem = h.create_offscreen_plain_surface(128, 128, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    assert_eq!(h.get_render_target_data_hr(&back_buffer, &sysmem), D3D_OK);
    let lock = sysmem.lock_rect(D3DLOCK_READONLY);
    assert_eq!(
        lock.as_u32(17)[16] | 0xFF00_0000,
        RED,
        "what GDI drew reached the back buffer"
    );
}

/// A chain and the device draw into two textures, so neither sees the other's frame.
#[test]
fn a_back_buffer_does_not_alias_the_device_back_buffer() {
    let h = one_pixel_device();
    let (chain, width, height) = window_sized_chain(&h);
    let chain_buffer = chain.back_buffer();
    let device_buffer = h.back_buffer(0);
    assert_eq!(h.set_render_target(0, &chain_buffer), D3D_OK);
    assert_eq!(h.clear_target(RED), D3D_OK, "clear the chain");
    assert_eq!(h.set_render_target(0, &device_buffer), D3D_OK);
    assert_eq!(h.clear_target(GREEN), D3D_OK, "clear the device");
    assert_eq!(h.read_pixel(0, 0), GREEN, "the device's back buffer");
    let sysmem =
        h.create_offscreen_plain_surface(width, height, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    assert_eq!(h.get_render_target_data_hr(&chain_buffer, &sysmem), D3D_OK);
    let lock = sysmem.lock_rect(D3DLOCK_READONLY);
    assert_eq!(lock.as_u32(1)[0], RED, "the chain's back buffer");
}

/// A device that presents through additional chains keeps its back buffer across a `WM_SIZE`.
///
/// The message runs on the window's thread, and such an application drives the
/// device from another one, so the resize has no safe moment to run in. The
/// application resizes its chains itself, and D3D9 changes a back buffer only
/// at `Reset`.
#[test]
fn a_device_with_an_additional_chain_keeps_its_back_buffer_across_a_resize() {
    let h = one_pixel_device();
    let (chain, width, height) = window_sized_chain(&h);
    let _ = h.send_window_message(WM_SIZE, 0, client_size(800, 600));
    let (hr, desc) = h.back_buffer(0).desc();
    assert_eq!(hr, D3D_OK);
    assert_eq!(
        (desc.width, desc.height),
        (1, 1),
        "the device's back buffer"
    );
    let (hr, desc) = chain.back_buffer().desc();
    assert_eq!(hr, D3D_OK);
    assert_eq!((desc.width, desc.height), (width, height), "the chain's");
}

/// A multisampled chain draws into its own samples and resolves at its own size.
#[test]
fn a_multisampled_back_buffer_resolves_at_its_own_size() {
    let h = one_pixel_device();
    let (width, height) = h.client_size();
    let mut pp = D3DPRESENT_PARAMETERS {
        flags: 0,
        multi_sample_type: D3DMULTISAMPLE_4_SAMPLES,
        swap_effect: h.windowed_present_params(width, height).swap_effect,
        ..gdi_chain_params(&h, width, height)
    };
    let chain = h.additional_swapchain_params(&mut pp);
    let back_buffer = chain.back_buffer();
    let (hr, desc) = back_buffer.desc();
    assert_eq!(hr, D3D_OK);
    assert_eq!(desc.multi_sample_type, D3DMULTISAMPLE_4_SAMPLES);
    assert_eq!(h.set_render_target(0, &back_buffer), D3D_OK);
    assert_eq!(h.clear_target(BLUE), D3D_OK, "clear the samples");
    let resolved = h.create_render_target(width, height, D3DFMT_A8R8G8B8);
    let (right, bottom) = (
        i32::try_from(width).expect("window width"),
        i32::try_from(height).expect("window height"),
    );
    let whole = rect(0, 0, right, bottom);
    assert_eq!(
        h.stretch_rect_regions(&back_buffer, &whole, &resolved, &whole, D3DTEXF_NONE),
        D3D_OK,
        "the resolve"
    );
    let sysmem =
        h.create_offscreen_plain_surface(width, height, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    assert_eq!(h.get_render_target_data_hr(&resolved, &sysmem), D3D_OK);
    let lock = sysmem.lock_rect(D3DLOCK_READONLY);
    assert_eq!(lock.as_u32(1)[0], BLUE, "the resolved frame");
}

/// A back buffer that outlives its chain and its device retires its texture on the way out.
///
/// Its release is the device's last reference, so the texture has to leave
/// through the device before the device goes. The [`ReleaseProbe`] on the back
/// buffer goes right after the texture retires, and has to find the device's
/// reference on the factory still there. Once the release has returned, it is
/// gone. A frame drawn into the back buffer keeps its texture named by work
/// the teardown waits for.
#[test]
fn a_back_buffer_released_last_retires_its_texture_before_the_device_goes() {
    let h = one_pixel_device();
    let factory = h.factory();
    // SAFETY: the harness's live factory; the vtable is `'static`.
    let vtbl = unsafe { deref_vtbl::<IDirect3D9Vtbl>(factory) };
    // SAFETY: this test's own reference, so the factory outlives the harness;
    // given back at the end.
    unsafe { (vtbl.add_ref)(factory) };
    let probe = ReleaseProbe::new(factory);
    let (chain, _, _) = window_sized_chain(&h);
    let raw = {
        let back_buffer = chain.back_buffer();
        // SAFETY: the probe lives on this frame past the back buffer's
        // finalize, which releases the reference.
        let hr =
            unsafe { back_buffer.set_private_data_unknown(&RELEASE_PROBE_KEY, probe.as_unknown()) };
        assert_eq!(hr, D3D_OK, "SetPrivateData(D3DSPD_IUNKNOWN)");
        assert_eq!(h.set_render_target(0, &back_buffer), D3D_OK);
        assert_eq!(h.clear_target(RED), D3D_OK, "draw into the chain");
        let implicit = h.back_buffer(0);
        assert_eq!(h.set_render_target(0, &implicit), D3D_OK);
        let raw = back_buffer.as_ptr();
        core::mem::forget(back_buffer);
        raw
    };
    drop(chain);
    drop(h);
    assert_eq!(
        factory_refs(factory),
        2,
        "this test's reference and that of the device the back buffer keeps"
    );
    drop(Surface::from_raw(raw));
    assert_eq!(
        probe.factory_refs.load(Ordering::Acquire),
        2,
        "the device was still there when the back buffer's texture retired"
    );
    assert_eq!(
        factory_refs(factory),
        1,
        "the device went with the back buffer's release"
    );
    // SAFETY: gives back this test's own reference.
    unsafe { (vtbl.release)(factory) };
}

/// A chain's `Present` shows that chain's back buffer, not the device's.
///
/// The child presents into its occluded window (`debug.presentOccluded`), and
/// the present target's line for the drawable names the size of what went
/// into it. The chain asks for an interval the device did not, which the layer
/// does not take, and the log says so.
#[test]
fn a_swap_chain_present_shows_its_own_back_buffer() {
    if running_as(PRESENT_CHILD_NAME) {
        chain_present_workload();
        return;
    }
    run_in_private_log_child_with(
        PRESENT_CHILD_NAME,
        "additional_swap_chain::a_swap_chain_present_shows_its_own_back_buffer",
        DRAWABLE_LOG_FILTER,
        PRESENT_OCCLUDED,
    );
}

/// Present a 320x200 chain on a 1x1 device and read the size the drawable was drawn from.
fn chain_present_workload() {
    let h = one_pixel_device();
    let mut pp = D3DPRESENT_PARAMETERS {
        presentation_interval: D3DPRESENT_INTERVAL_IMMEDIATE,
        ..gdi_chain_params(&h, 320, 200)
    };
    let chain = h.additional_swapchain_params(&mut pp);
    let back_buffer = chain.back_buffer();
    assert_eq!(h.set_render_target(0, &back_buffer), D3D_OK);
    assert_eq!(h.clear_target(RED), D3D_OK, "draw into the chain");
    assert_eq!(chain.present(), D3D_OK, "Present through the chain");
    let lines = await_acquired_drawables(h.hwnd(), 1);
    assert!(
        lines[0].ends_with("from 320x200"),
        "the drawable came from the chain: {}",
        lines[0]
    );
    await_logged_lines_within(
        "of the swap chain is not applied",
        1,
        Duration::from_secs(10),
    );
}
