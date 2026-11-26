//! Drive the native `libd3d9.dylib` through its two factories in a hidden `NSWindow`.
//!
//! The library is loaded from the path given as the first argument, and the
//! layer logs into the directory given as the second, which the checks read.
//! The window is never ordered front, and `debug.presentOccluded` has the
//! presenter acquire a drawable for it anyway; each `Present` waits for the
//! layer's line saying it did.
//!
//! Each check prints one `PASS` or `FAIL` line, and the exit status is the
//! number of failures. The checks run on the main thread, which `AppKit`
//! needs. One device is created on another thread while the main thread runs
//! its run loop, which is how a host serves the layer's synchronous
//! main-thread hop, and one presents from another thread after a resize while
//! the main thread runs nothing.

use core::ffi::c_void;
use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use libloading::Library;
use mtld3d_types::{
    D3D_OK, D3DCLEAR_TARGET, D3DCREATE_HARDWARE_VERTEXPROCESSING, D3DCREATE_MULTITHREADED,
    D3DCULL_NONE, D3DDEVTYPE_HAL, D3DERR_INVALIDCALL, D3DFMT_A8R8G8B8, D3DFMT_X8R8G8B8,
    D3DFVF_DIFFUSE, D3DFVF_XYZRHW, D3DLOCK_READONLY, D3DLOCKED_RECT, D3DPOOL_SCRATCH,
    D3DPOOL_SYSTEMMEM, D3DPRESENT_INTERVAL_IMMEDIATE, D3DPRESENT_PARAMETERS, D3DPT_TRIANGLELIST,
    D3DRS_CULLMODE, D3DRS_LIGHTING, D3DSDK_VERSION, D3DSURFACE_DESC, D3DSWAPEFFECT_DISCARD,
    IDirect3D9ExVtbl, IDirect3D9Vtbl, IDirect3DDevice9Vtbl, IDirect3DSurface9Vtbl,
};
use objc2::{MainThreadMarker, MainThreadOnly, rc::Retained};
use objc2_app_kit::{NSApplication, NSBackingStoreType, NSView, NSWindow, NSWindowStyleMask};
use objc2_core_foundation::{CFRunLoop, CGPoint, CGRect, CGSize, kCFRunLoopDefaultMode};

/// The configuration the checks run under, over any the caller set.
const CONFIG: &str = "shaderCache.enable=false;shader.asyncCompile=false;color.hdr.enable=false;\
                      debug.mainThreadChecker=true;debug.presentOccluded=true";

/// The window's first content size, in points.
const FIRST_SIZE: (f64, f64) = (320.0, 240.0);
/// The content size the resize check moves the window to, in points.
const RESIZED: (f64, f64) = (400.0, 300.0);
/// The back-buffer size the `Reset` after a resize asks for, in pixels.
const RESET_SIZE: (u32, u32) = (256, 128);
/// The back-buffer size the later explicit `Reset` asks for, which the device does not have yet.
const SECOND_RESET_SIZE: (u32, u32) = (192, 96);
/// The colour the clear writes, as an X8R8G8B8 texel.
const CLEAR_COLOR: u32 = 0x00FF_0000;
/// The colour the triangle is drawn in, as an X8R8G8B8 texel.
const TRIANGLE_COLOR: u32 = 0x0000_FF00;
/// How long a device made on another thread may take before the run is called hung.
const WORKER_TIMEOUT: Duration = Duration::from_secs(60);
/// The log filter under which the presenter logs one line per drawable it acquires.
const DRAWABLE_LOG_FILTER: &str = "warn,mtld3d::unix=info,mtld3d::unix::present=debug";
/// How long the drawable line of one `Present` may take to reach the log.
///
/// The layer allows `nextDrawable` to time out after one second; the rest is
/// slack for a slow runner.
const DRAWABLE_WAIT: Duration = Duration::from_secs(11);

type Create9 = unsafe extern "system" fn(u32) -> *mut c_void;
type Create9Ex = unsafe extern "system" fn(u32, *mut *mut c_void) -> i32;

/// A pre-transformed, coloured vertex (`D3DFVF_XYZRHW | D3DFVF_DIFFUSE`).
#[repr(C)]
struct RhwVertex {
    x: f32,
    y: f32,
    z: f32,
    rhw: f32,
    color: u32,
}

/// The checks' tally, and the log they read the drawable lines from.
struct Run {
    failures: i32,
    log: PathBuf,
    /// The view the devices present into, unless a check names another.
    view: usize,
    /// The drawable lines counted so far, per view.
    acquired: Vec<(usize, usize)>,
}

impl Run {
    /// Wait for the drawable line of the `Present` just made into the main view.
    fn presents_acquired(&mut self, name: &str) {
        self.presents_acquired_on(self.view, name);
    }

    /// Wait for one more drawable line on `view` than the last check counted.
    ///
    /// Every `Present` the checks make is followed by one of these, so each
    /// counts its own drawable.
    fn presents_acquired_on(&mut self, view: usize, name: &str) {
        let slot = self
            .acquired
            .iter()
            .position(|&(v, _)| v == view)
            .unwrap_or_else(|| {
                self.acquired.push((view, 0));
                self.acquired.len() - 1
            });
        let expected = self.acquired[slot].1 + 1;
        let needle = format!("drawable acquired on window {view:#x} for");
        let deadline = Instant::now() + DRAWABLE_WAIT;
        let mut acquired = logged_lines(&self.log, &needle);
        while acquired < expected && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(20));
            acquired = logged_lines(&self.log, &needle);
        }
        self.acquired[slot].1 = acquired;
        self.check(
            &format!("{name} acquires a drawable"),
            acquired == expected,
            &format!("{acquired} drawable line(s) on view {view:#x}, {expected} expected"),
        );
    }

    fn check(&mut self, name: &str, ok: bool, detail: &str) {
        if ok {
            println!("PASS {name}");
        } else {
            println!("FAIL {name}: {detail}");
            self.failures += 1;
        }
    }
}

/// A COM object's vtable.
///
/// # Safety
///
/// `p` is a live COM object whose vtable has the layout of `T`.
unsafe fn vtbl<T>(p: *mut c_void) -> &'static T {
    // SAFETY: per the contract, the first field of `p` points at a static `T`.
    let table = unsafe { *p.cast::<*const T>() };
    // SAFETY: per the contract, the vtable is static.
    unsafe { &*table }
}

/// Release one COM reference.
fn release(p: *mut c_void) {
    if p.is_null() {
        return;
    }
    // SAFETY: every pointer handed here is a live object the layer returned.
    let table = unsafe { vtbl::<IDirect3D9Vtbl>(p) };
    // SAFETY: `release` is the third slot of every COM vtable.
    unsafe { (table.release)(p) };
}

/// Windowed (`windowed` 1) or fullscreen present parameters on `view`.
const fn present_params(view: usize, size: (u32, u32), windowed: u32) -> D3DPRESENT_PARAMETERS {
    D3DPRESENT_PARAMETERS {
        back_buffer_width: size.0,
        back_buffer_height: size.1,
        back_buffer_format: D3DFMT_X8R8G8B8,
        back_buffer_count: 1,
        multi_sample_type: 0,
        multi_sample_quality: 0,
        swap_effect: D3DSWAPEFFECT_DISCARD,
        device_window: view,
        windowed,
        enable_auto_depth_stencil: 0,
        auto_depth_stencil_format: 0,
        flags: 0,
        full_screen_refresh_rate_in_hz: 0,
        presentation_interval: D3DPRESENT_INTERVAL_IMMEDIATE,
    }
}

/// `IDirect3D9::CreateDevice` on `view`; the device or the failing `HRESULT`.
fn create_device(d3d: usize, view: usize, pp: &mut D3DPRESENT_PARAMETERS) -> Result<usize, i32> {
    create_device_with(d3d, view, pp, 0)
}

/// `IDirect3D9::CreateDevice` on `view` with `flags` added to hardware vertex processing.
fn create_device_with(
    d3d: usize,
    view: usize,
    pp: &mut D3DPRESENT_PARAMETERS,
    flags: u32,
) -> Result<usize, i32> {
    let d3d = d3d as *mut c_void;
    // SAFETY: `d3d` is a live `IDirect3D9`.
    let table = unsafe { vtbl::<IDirect3D9Vtbl>(d3d) };
    let mut device = core::ptr::null_mut();
    // SAFETY: the arguments follow `CreateDevice`'s contract; `pp` and `device` are live.
    let hr = unsafe {
        (table.create_device)(
            d3d,
            0,
            D3DDEVTYPE_HAL,
            view as *mut c_void,
            D3DCREATE_HARDWARE_VERTEXPROCESSING | flags,
            (&raw mut *pp).cast::<c_void>(),
            &raw mut device,
        )
    };
    if hr == D3D_OK && !device.is_null() {
        Ok(device as usize)
    } else {
        Err(hr)
    }
}

/// `IDirect3D9Ex::CreateDeviceEx` on `view`, windowed; the device or the failing `HRESULT`.
fn create_device_ex(
    d3d_ex: usize,
    view: usize,
    pp: &mut D3DPRESENT_PARAMETERS,
) -> Result<usize, i32> {
    let d3d_ex = d3d_ex as *mut c_void;
    // SAFETY: `d3d_ex` is a live `IDirect3D9Ex`.
    let table = unsafe { vtbl::<IDirect3D9ExVtbl>(d3d_ex) };
    let mut device = core::ptr::null_mut();
    // SAFETY: the arguments follow `CreateDeviceEx`'s contract; a windowed
    // device takes no display mode, and `pp` and `device` are live.
    let hr = unsafe {
        (table.create_device_ex)(
            d3d_ex,
            0,
            D3DDEVTYPE_HAL,
            view as *mut c_void,
            D3DCREATE_HARDWARE_VERTEXPROCESSING,
            (&raw mut *pp).cast::<c_void>(),
            core::ptr::null_mut(),
            &raw mut device,
        )
    };
    if hr == D3D_OK && !device.is_null() {
        Ok(device as usize)
    } else {
        Err(hr)
    }
}

/// A live `IDirect3DDevice9` and the calls the checks make on it.
struct Device {
    ptr: *mut c_void,
}

impl Device {
    fn table(&self) -> &'static IDirect3DDevice9Vtbl {
        // SAFETY: `ptr` is a live device until `drop`.
        unsafe { vtbl::<IDirect3DDevice9Vtbl>(self.ptr) }
    }

    fn clear(&self, color: u32) -> i32 {
        // SAFETY: a full-target clear; no rects.
        unsafe {
            (self.table().clear)(
                self.ptr,
                0,
                core::ptr::null(),
                D3DCLEAR_TARGET,
                color,
                1.0,
                0,
            )
        }
    }

    /// The top-left half of a `width`x`height` target as one triangle, in `TRIANGLE_COLOR`.
    fn draw_triangle(&self, width: u32, height: u32) -> i32 {
        let (w, h) = (f32_of(width), f32_of(height));
        let vertex = |x, y| RhwVertex {
            x,
            y,
            z: 0.5,
            rhw: 1.0,
            color: 0xFF00_0000 | TRIANGLE_COLOR,
        };
        let vertices = [vertex(0.0, 0.0), vertex(w, 0.0), vertex(0.0, h)];
        let table = self.table();
        // SAFETY: plain state setters on a live device.
        unsafe {
            (table.set_render_state)(self.ptr, D3DRS_CULLMODE, D3DCULL_NONE);
        }
        // SAFETY: as above.
        unsafe { (table.set_render_state)(self.ptr, D3DRS_LIGHTING, 0) };
        // SAFETY: as above.
        unsafe { (table.set_fvf)(self.ptr, D3DFVF_XYZRHW | D3DFVF_DIFFUSE) };
        // SAFETY: a scene bracket on a live device.
        unsafe { (table.begin_scene)(self.ptr) };
        // SAFETY: `vertices` holds one triangle of `RhwVertex`, the stride passed.
        let hr = unsafe {
            (table.draw_primitive_up)(
                self.ptr,
                D3DPT_TRIANGLELIST,
                1,
                vertices.as_ptr().cast::<c_void>(),
                u32::try_from(size_of::<RhwVertex>()).expect("a vertex is 20 bytes"),
            )
        };
        // SAFETY: closes the bracket opened above.
        unsafe { (table.end_scene)(self.ptr) };
        hr
    }

    fn present(&self) -> i32 {
        // SAFETY: a whole-surface present into the device window.
        unsafe {
            (self.table().present)(
                self.ptr,
                core::ptr::null(),
                core::ptr::null(),
                core::ptr::null_mut(),
                core::ptr::null(),
            )
        }
    }

    fn reset(&self, pp: &mut D3DPRESENT_PARAMETERS) -> i32 {
        // SAFETY: `pp` is a live, writable `D3DPRESENT_PARAMETERS`.
        unsafe { (self.table().reset)(self.ptr, (&raw mut *pp).cast::<c_void>()) }
    }

    fn create_additional_swap_chain(&self, pp: &mut D3DPRESENT_PARAMETERS) -> i32 {
        let mut chain = core::ptr::null_mut();
        // SAFETY: `pp` is a live, writable `D3DPRESENT_PARAMETERS`; `chain` a live out slot.
        let hr = unsafe {
            (self.table().create_additional_swap_chain)(
                self.ptr,
                (&raw mut *pp).cast::<c_void>(),
                &raw mut chain,
            )
        };
        release(chain);
        hr
    }

    fn show_cursor(&self, show: bool) -> i32 {
        // SAFETY: a plain call on a live device.
        unsafe { (self.table().show_cursor)(self.ptr, i32::from(show)) }
    }

    fn back_buffer(&self) -> Option<Surface> {
        let mut surface = core::ptr::null_mut();
        // SAFETY: swap chain 0, back buffer 0, type 0 (`D3DBACKBUFFER_TYPE_MONO`);
        // `surface` is a live out slot.
        let hr = unsafe { (self.table().get_back_buffer)(self.ptr, 0, 0, 0, &raw mut surface) };
        (hr == D3D_OK && !surface.is_null()).then_some(Surface { ptr: surface })
    }

    fn offscreen(&self, size: (u32, u32), format: u32, pool: u32) -> Option<Surface> {
        let mut surface = core::ptr::null_mut();
        // SAFETY: a plain create; `surface` is a live out slot, no shared handle.
        let hr = unsafe {
            (self.table().create_offscreen_plain_surface)(
                self.ptr,
                size.0,
                size.1,
                format,
                pool,
                &raw mut surface,
                core::ptr::null_mut(),
            )
        };
        (hr == D3D_OK && !surface.is_null()).then_some(Surface { ptr: surface })
    }

    /// The back buffer's texel at (`x`, `y`), read back through `GetRenderTargetData`.
    fn read_back(&self, x: u32, y: u32) -> Option<u32> {
        let back = self.back_buffer()?;
        let (width, height) = back.size()?;
        let copy = self.offscreen((width, height), D3DFMT_X8R8G8B8, D3DPOOL_SYSTEMMEM)?;
        // SAFETY: both surfaces are live and of one size and format.
        let hr = unsafe { (self.table().get_render_target_data)(self.ptr, back.ptr, copy.ptr) };
        if hr != D3D_OK {
            return None;
        }
        copy.texel(x, y)
    }

    fn set_cursor_properties(&self, bitmap: &Surface) -> i32 {
        // SAFETY: `bitmap` is a live A8R8G8B8 surface.
        unsafe { (self.table().set_cursor_properties)(self.ptr, 0, 0, bitmap.ptr) }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        release(self.ptr);
    }
}

/// A live `IDirect3DSurface9`.
struct Surface {
    ptr: *mut c_void,
}

impl Surface {
    fn table(&self) -> &'static IDirect3DSurface9Vtbl {
        // SAFETY: `ptr` is a live surface until `drop`.
        unsafe { vtbl::<IDirect3DSurface9Vtbl>(self.ptr) }
    }

    fn size(&self) -> Option<(u32, u32)> {
        let mut desc = D3DSURFACE_DESC {
            format: 0,
            resource_type: 0,
            usage: 0,
            pool: 0,
            multi_sample_type: 0,
            multi_sample_quality: 0,
            width: 0,
            height: 0,
        };
        // SAFETY: `desc` is a live out struct.
        let hr = unsafe { (self.table().get_desc)(self.ptr, &raw mut desc) };
        (hr == D3D_OK).then_some((desc.width, desc.height))
    }

    fn texel(&self, x: u32, y: u32) -> Option<u32> {
        let (width, height) = self.size()?;
        if x >= width || y >= height {
            return None;
        }
        let mut locked = D3DLOCKED_RECT {
            pitch: 0,
            bits: core::ptr::null_mut(),
        };
        // SAFETY: a read-only lock of the whole surface into a live out struct.
        let hr = unsafe {
            (self.table().lock_rect)(
                self.ptr,
                &raw mut locked,
                core::ptr::null(),
                D3DLOCK_READONLY,
            )
        };
        if hr != D3D_OK || locked.bits.is_null() {
            return None;
        }
        let pitch = usize::try_from(locked.pitch).ok()?;
        let offset = usize::try_from(y).ok()? * pitch + usize::try_from(x).ok()? * 4;
        // SAFETY: the lock maps the whole surface, and (`x`, `y`) lies inside it.
        let at = unsafe { locked.bits.cast::<u8>().add(offset) };
        // SAFETY: `at` names four readable bytes of the locked texel; no alignment is assumed.
        let texel = unsafe { at.cast::<u32>().read_unaligned() };
        // SAFETY: pairs with the lock above.
        unsafe { (self.table().unlock_rect)(self.ptr) };
        Some(texel & 0x00FF_FFFF)
    }

    fn get_dc(&self) -> i32 {
        let mut dc = core::ptr::null_mut();
        // SAFETY: `dc` is a live out slot.
        unsafe { (self.table().get_dc)(self.ptr, &raw mut dc) }
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        release(self.ptr);
    }
}

fn main() {
    let mtm = MainThreadMarker::new().expect("the checks run on the main thread");
    let (Some(path), Some(log_dir)) = (std::env::args_os().nth(1), std::env::args_os().nth(2))
    else {
        eprintln!("usage: native-d3d9-test <path to libd3d9.dylib> <log directory>");
        std::process::exit(2);
    };
    let log_dir = PathBuf::from(log_dir);
    if !log_dir.is_absolute() {
        eprintln!("the log directory {} is not absolute", log_dir.display());
        std::process::exit(2);
    }
    set_config(&log_dir);
    // SAFETY: the library is the layer's own build; loading it runs no
    // initialiser that needs more than the process it is loaded into.
    let library = match unsafe { Library::new(&path) } {
        Ok(library) => library,
        Err(e) => {
            eprintln!("cannot load {}: {e}", path.display());
            std::process::exit(2);
        }
    };
    // SAFETY: both exports have these signatures in the native build.
    let create9 = unsafe { library.get::<Create9>(b"Direct3DCreate9\0") }.map(|s| *s);
    // SAFETY: as above.
    let create9_ex = unsafe { library.get::<Create9Ex>(b"Direct3DCreate9Ex\0") }.map(|s| *s);
    let (Ok(create9), Ok(create9_ex)) = (create9, create9_ex) else {
        eprintln!("{} does not export both factories", path.display());
        std::process::exit(2);
    };
    // `AppKit` needs its shared application before a window is made.
    let _app = NSApplication::sharedApplication(mtm);
    let (window, view) = hidden_window(mtm);
    let view_addr = Retained::as_ptr(&view) as usize;
    let log = log_dir.join(format!("native-d3d9-test-{}.log", std::process::id()));
    // A file of an earlier process with this id would hold lines that are not this run's.
    if let Err(e) = std::fs::remove_file(&log)
        && e.kind() != std::io::ErrorKind::NotFound
    {
        eprintln!("cannot remove {}: {e}", log.display());
        std::process::exit(2);
    }
    let mut run = Run {
        failures: 0,
        log,
        view: view_addr,
        acquired: Vec::new(),
    };
    run_checks(&mut run, (create9, create9_ex), &window, &view, view_addr);
    drop(view);
    drop(window);
    drop(library);
    println!("{} failed", run.failures);
    std::process::exit(run.failures);
}

/// Put the checks' entries after whatever `MTLD3D_CONFIG` the caller set, and log the drawables.
///
/// The checks' entries and `log_dir`, where the layer logs, go after the
/// caller's entries, as the drawable filter goes after the caller's
/// `RUST_LOG`: a later entry wins, and the checks need theirs.
fn set_config(log_dir: &Path) {
    let mut value = std::env::var_os("MTLD3D_CONFIG").unwrap_or_default();
    if !value.is_empty() {
        value.push(";");
    }
    value.push(CONFIG);
    value.push(";log.dir=");
    value.push(log_dir);
    let mut filter = std::env::var_os("RUST_LOG").unwrap_or_default();
    if !filter.is_empty() {
        filter.push(",");
    }
    filter.push(DRAWABLE_LOG_FILTER);
    // SAFETY: no other thread exists yet, so nothing reads the environment meanwhile.
    unsafe { std::env::set_var("MTLD3D_CONFIG", value) };
    // SAFETY: as above.
    unsafe { std::env::set_var("RUST_LOG", filter) };
}

/// The number of lines of the log at `path` that contain `needle`; none while it does not exist.
fn logged_lines(path: &Path, needle: &str) -> usize {
    std::fs::read_to_string(path).map_or(0, |text| {
        text.lines().filter(|line| line.contains(needle)).count()
    })
}

/// A borderless `NSWindow` that is never ordered front, and its content view.
fn hidden_window(mtm: MainThreadMarker) -> (Retained<NSWindow>, Retained<NSView>) {
    let rect = CGRect {
        origin: CGPoint { x: 0.0, y: 0.0 },
        size: CGSize {
            width: FIRST_SIZE.0,
            height: FIRST_SIZE.1,
        },
    };
    // SAFETY: a plain designated initializer on a fresh allocation.
    let window = unsafe {
        NSWindow::initWithContentRect_styleMask_backing_defer(
            NSWindow::alloc(mtm),
            rect,
            NSWindowStyleMask::Borderless,
            NSBackingStoreType::Buffered,
            false,
        )
    };
    // SAFETY: the window is released by the `Retained` above, not on close.
    unsafe { window.setReleasedWhenClosed(false) };
    let view = NSView::initWithFrame(NSView::alloc(mtm), rect);
    window.setContentView(Some(&view));
    (window, view)
}

/// The view's size in backing pixels.
fn backing_size(view: &NSView) -> (u32, u32) {
    let backing = view.convertRectToBacking(view.bounds());
    (pixels(backing.size.width), pixels(backing.size.height))
}

fn run_checks(
    run: &mut Run,
    (create9, create9_ex): (Create9, Create9Ex),
    window: &NSWindow,
    view: &NSView,
    view_addr: usize,
) {
    // SAFETY: the export takes the SDK version and returns an interface or null.
    let d3d = unsafe { create9(D3DSDK_VERSION) };
    run.check("Direct3DCreate9", !d3d.is_null(), "null interface");
    let mut d3d_ex = core::ptr::null_mut();
    // SAFETY: `d3d_ex` is a live out slot.
    let hr = unsafe { create9_ex(D3DSDK_VERSION, &raw mut d3d_ex) };
    run.check(
        "Direct3DCreate9Ex",
        hr == D3D_OK && !d3d_ex.is_null(),
        &format!("hr={hr:#x}"),
    );
    if d3d.is_null() {
        return;
    }
    let d3d_addr = d3d as usize;

    let mut fullscreen = present_params(view_addr, (640, 480), 0);
    let hr = create_device(d3d_addr, view_addr, &mut fullscreen)
        .err()
        .unwrap_or(D3D_OK);
    run.check(
        "CreateDevice(Windowed = FALSE) is INVALIDCALL",
        hr == D3DERR_INVALIDCALL,
        &format!("hr={hr:#x}"),
    );
    let window_addr = core::ptr::from_ref(window) as usize;
    let mut on_window = present_params(window_addr, (0, 0), 1);
    let hr = create_device(d3d_addr, window_addr, &mut on_window)
        .err()
        .unwrap_or(D3D_OK);
    run.check(
        "CreateDevice on an NSWindow is INVALIDCALL",
        hr == D3DERR_INVALIDCALL,
        &format!("hr={hr:#x}"),
    );

    let mut pp = present_params(view_addr, (0, 0), 1);
    let device = match create_device(d3d_addr, view_addr, &mut pp) {
        Ok(device) => Device {
            ptr: device as *mut c_void,
        },
        Err(hr) => {
            run.check(
                "CreateDevice on the main thread",
                false,
                &format!("hr={hr:#x}"),
            );
            release(d3d);
            release(d3d_ex);
            return;
        }
    };
    run.check("CreateDevice on the main thread", true, "");
    let expected = backing_size(view);
    let size = device.back_buffer().and_then(|b| b.size());
    run.check(
        "a zero-size request takes the view's size in pixels",
        size == Some(expected),
        &format!("back buffer {size:?}, view {expected:?}"),
    );
    render_checks(run, &device, expected);

    resize_checks(run, &device, window, view);
    reset_after_resize_check(run, &device, window, view_addr);
    zero_reset_after_resize_check(run, &device, window, view);

    // A second view in the window, which a Reset moves the device to while the
    // first has a resize pending; the later Resets move it back.
    let other = NSView::initWithFrame(NSView::alloc(view.mtm()), view.bounds());
    view.addSubview(&other);
    window.setContentSize(CGSize {
        width: FIRST_SIZE.0,
        height: FIRST_SIZE.1,
    });
    let mut onto_other = present_params(core::ptr::from_ref(&*other) as usize, RESET_SIZE, 1);
    let reset = device.reset(&mut onto_other);
    let presented = device.present();
    let size = device.back_buffer().and_then(|b| b.size());
    run.check(
        "a Reset onto another view keeps its size over the old view's pending resize",
        reset == D3D_OK && presented == D3D_OK && size == Some(RESET_SIZE),
        &format!("reset={reset:#x}, present={presented:#x}, back buffer {size:?}"),
    );
    run.presents_acquired_on(
        core::ptr::from_ref(&*other) as usize,
        "the present on the other view",
    );

    let mut chain_on_window = present_params(window_addr, (0, 0), 1);
    let hr = device.create_additional_swap_chain(&mut chain_on_window);
    run.check(
        "CreateAdditionalSwapChain on an NSWindow is INVALIDCALL",
        hr == D3DERR_INVALIDCALL,
        &format!("hr={hr:#x}"),
    );
    let mut reset_on_window = present_params(window_addr, (0, 0), 1);
    let hr = device.reset(&mut reset_on_window);
    run.check(
        "Reset onto an NSWindow is INVALIDCALL",
        hr == D3DERR_INVALIDCALL,
        &format!("hr={hr:#x}"),
    );

    let mut windowed_off = present_params(view_addr, (640, 480), 0);
    let hr = device.reset(&mut windowed_off);
    run.check(
        "Reset(Windowed = FALSE) is INVALIDCALL",
        hr == D3DERR_INVALIDCALL,
        &format!("hr={hr:#x}"),
    );
    let mut sized = present_params(view_addr, SECOND_RESET_SIZE, 1);
    let hr = device.reset(&mut sized);
    let size = device.back_buffer().and_then(|b| b.size());
    run.check(
        "Reset to an explicit size",
        hr == D3D_OK && size == Some(SECOND_RESET_SIZE),
        &format!("hr={hr:#x}, back buffer {size:?}"),
    );
    render_checks(run, &device, SECOND_RESET_SIZE);

    cursor_and_dc_checks(run, &device);

    drop(device);
    let mut again = present_params(view_addr, (0, 0), 1);
    match create_device(d3d_addr, view_addr, &mut again) {
        Ok(device) => {
            let device = Device {
                ptr: device as *mut c_void,
            };
            run.check("a device recreated on the same view", true, "");
            render_checks(run, &device, backing_size(view));
        }
        Err(hr) => run.check(
            "a device recreated on the same view",
            false,
            &format!("hr={hr:#x}"),
        ),
    }

    if !d3d_ex.is_null() {
        let mut pp = present_params(view_addr, (0, 0), 1);
        match create_device_ex(d3d_ex as usize, view_addr, &mut pp) {
            Ok(device) => {
                let device = Device {
                    ptr: device as *mut c_void,
                };
                run.check("CreateDeviceEx", true, "");
                render_checks(run, &device, backing_size(view));
            }
            Err(hr) => run.check("CreateDeviceEx", false, &format!("hr={hr:#x}")),
        }
    }

    worker_check(run, d3d_addr, view_addr);
    present_without_main_loop_check(run, d3d_addr, window, view);

    release(d3d);
    release(d3d_ex);
}

/// Clear, draw, read back and present on a back buffer of `size`.
fn render_checks(run: &mut Run, device: &Device, size: (u32, u32)) {
    let (width, height) = size;
    let hr = device.clear(0xFF00_0000 | CLEAR_COLOR);
    let corner = device.read_back(width - 1, height - 1);
    run.check(
        &format!("clear reads back at {width}x{height}"),
        hr == D3D_OK && corner == Some(CLEAR_COLOR),
        &format!("hr={hr:#x}, texel {corner:x?}"),
    );
    let hr = device.draw_triangle(width, height);
    let inside = device.read_back(width / 4, height / 4);
    let outside = device.read_back(width - 1, height - 1);
    run.check(
        &format!("a triangle reads back at {width}x{height}"),
        hr == D3D_OK && inside == Some(TRIANGLE_COLOR) && outside == Some(CLEAR_COLOR),
        &format!("hr={hr:#x}, inside {inside:x?}, outside {outside:x?}"),
    );
    let hr = device.present();
    run.check(
        &format!("Present at {width}x{height}"),
        hr == D3D_OK,
        &format!("hr={hr:#x}"),
    );
    run.presents_acquired(&format!("Present at {width}x{height}"));
}

/// Grow the view and check the back buffer follows it at the next present.
fn resize_checks(run: &mut Run, device: &Device, window: &NSWindow, view: &NSView) {
    window.setContentSize(CGSize {
        width: RESIZED.0,
        height: RESIZED.1,
    });
    let expected = backing_size(view);
    // The present after the frame change is the one that follows it.
    let hr = device.present();
    let size = device.back_buffer().and_then(|b| b.size());
    run.check(
        "the back buffer follows a resized view at the next present",
        hr == D3D_OK && size == Some(expected),
        &format!("hr={hr:#x}, back buffer {size:?}, view {expected:?}"),
    );
    run.presents_acquired("the present after a resize");
    render_checks(run, device, expected);
}

/// Resize the view, then `Reset` to an explicit size: the next present keeps that size.
fn reset_after_resize_check(run: &mut Run, device: &Device, window: &NSWindow, view: usize) {
    window.setContentSize(CGSize {
        width: FIRST_SIZE.0,
        height: FIRST_SIZE.1,
    });
    let mut sized = present_params(view, RESET_SIZE, 1);
    let reset = device.reset(&mut sized);
    let presented = device.present();
    let size = device.back_buffer().and_then(|b| b.size());
    run.check(
        "a Reset after a view resize keeps its size at the next present",
        reset == D3D_OK && presented == D3D_OK && size == Some(RESET_SIZE),
        &format!("reset={reset:#x}, present={presented:#x}, back buffer {size:?}"),
    );
    run.presents_acquired("the present after that Reset");
}

/// Resize the view, then `Reset` with zero dimensions: the back buffer takes the new size.
fn zero_reset_after_resize_check(run: &mut Run, device: &Device, window: &NSWindow, view: &NSView) {
    window.setContentSize(CGSize {
        width: RESIZED.0,
        height: RESIZED.1,
    });
    let expected = backing_size(view);
    let mut zero = present_params(core::ptr::from_ref(view) as usize, (0, 0), 1);
    let reset = device.reset(&mut zero);
    let size = device.back_buffer().and_then(|b| b.size());
    run.check(
        "a zero-size Reset after a view resize takes the new size",
        reset == D3D_OK && size == Some(expected),
        &format!("reset={reset:#x}, back buffer {size:?}, view {expected:?}"),
    );
    render_checks(run, device, expected);
}

/// `ShowCursor` reports the previous state; `GetDC` is refused.
fn cursor_and_dc_checks(run: &mut Run, device: &Device) {
    let before = device.show_cursor(true);
    run.check(
        "ShowCursor before a cursor surface reports hidden and changes nothing",
        before == 0 && device.show_cursor(true) == 0,
        &format!("previous {before}"),
    );
    let Some(bitmap) = device.offscreen((32, 32), D3DFMT_A8R8G8B8, D3DPOOL_SCRATCH) else {
        run.check(
            "a cursor bitmap",
            false,
            "CreateOffscreenPlainSurface failed",
        );
        return;
    };
    let hr = device.set_cursor_properties(&bitmap);
    run.check("SetCursorProperties", hr == D3D_OK, &format!("hr={hr:#x}"));
    let first = device.show_cursor(true);
    let second = device.show_cursor(false);
    let third = device.show_cursor(false);
    run.check(
        "ShowCursor returns the previous state",
        (first, second, third) == (0, 1, 0),
        &format!("{first}, {second}, {third}"),
    );
    let surface = device.offscreen((16, 16), D3DFMT_X8R8G8B8, D3DPOOL_SYSTEMMEM);
    let hr = surface.as_ref().map_or(D3D_OK, Surface::get_dc);
    run.check(
        "GetDC is INVALIDCALL",
        hr == D3DERR_INVALIDCALL,
        &format!("hr={hr:#x}"),
    );
}

/// Create, render and release a device on another thread while the main thread runs its loop.
fn worker_check(run: &mut Run, d3d: usize, view: usize) {
    let worker = std::thread::spawn(move || {
        let mut pp = present_params(view, (64, 64), 1);
        create_device(d3d, view, &mut pp).map(|device| {
            let device = Device {
                ptr: device as *mut c_void,
            };
            let hr = device.clear(0xFF00_0000 | CLEAR_COLOR);
            let texel = device.read_back(0, 0);
            (hr, texel, device.present())
        })
    });
    let started = Instant::now();
    while !worker.is_finished() && started.elapsed() < WORKER_TIMEOUT {
        // SAFETY: the default mode is a framework constant.
        let mode = unsafe { kCFRunLoopDefaultMode };
        CFRunLoop::run_in_mode(mode, 0.01, false);
    }
    if !worker.is_finished() {
        // The worker still uses the device and the dylib: nothing may be released under it.
        eprintln!("FAIL the worker thread did not finish");
        std::process::exit(1);
    }
    let outcome = worker.join();
    run.check(
        "CreateDevice, clear, read back and Present from another thread",
        matches!(outcome, Ok(Ok((D3D_OK, Some(CLEAR_COLOR), D3D_OK)))),
        &format!("{outcome:x?}"),
    );
    run.presents_acquired("the Present from another thread");
}

/// Resize the view, then `Present` from another thread while the main thread runs nothing.
///
/// The `Present` follows the new size without asking the main thread, so it
/// returns although nothing serves a main-thread hop, as when the host's main
/// thread waits for the render thread.
fn present_without_main_loop_check(run: &mut Run, d3d: usize, window: &NSWindow, view: &NSView) {
    let view_addr = core::ptr::from_ref(view) as usize;
    let mut pp = present_params(view_addr, (0, 0), 1);
    let device = match create_device_with(d3d, view_addr, &mut pp, D3DCREATE_MULTITHREADED) {
        Ok(device) => Device {
            ptr: device as *mut c_void,
        },
        Err(hr) => {
            run.check("a multithreaded device", false, &format!("hr={hr:#x}"));
            return;
        }
    };
    let before = backing_size(view);
    render_checks(run, &device, before);
    let current = view.bounds().size;
    let grown = (current.width + 40.0, current.height + 30.0);
    window.setContentSize(CGSize {
        width: grown.0,
        height: grown.1,
    });
    let expected = backing_size(view);
    run.check(
        "the view changed size before the Present from another thread",
        expected != before,
        &format!("view {expected:?}"),
    );
    let device_addr = device.ptr as usize;
    let worker = std::thread::spawn(move || {
        let device = core::mem::ManuallyDrop::new(Device {
            ptr: device_addr as *mut c_void,
        });
        device.present()
    });
    let started = Instant::now();
    while !worker.is_finished() && started.elapsed() < DRAWABLE_WAIT {
        std::thread::sleep(Duration::from_millis(10));
    }
    if !worker.is_finished() {
        // The worker still uses the device and the dylib: nothing may be released under it.
        eprintln!("FAIL a Present after a resize waited for the main thread");
        std::process::exit(1);
    }
    let presented = worker.join();
    let size = device.back_buffer().and_then(|b| b.size());
    run.check(
        "a Present after a resize returns while the main thread runs nothing",
        matches!(presented, Ok(D3D_OK)) && size == Some(expected),
        &format!("{presented:x?}, back buffer {size:?}, view {expected:?}"),
    );
    run.presents_acquired("a Present from another thread after a resize");
}

/// A point extent in whole pixels.
fn pixels(extent: f64) -> u32 {
    format!("{:.0}", extent.max(0.0)).parse().unwrap_or(0)
}

/// A pixel count as a vertex coordinate.
fn f32_of(value: u32) -> f32 {
    u16::try_from(value).map_or(f32::MAX, f32::from)
}
