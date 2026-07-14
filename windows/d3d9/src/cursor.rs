//! Hardware cursor implementation.
//!
//! Implements `D3D9Device::SetCursor*` / `ShowCursor` / `CursorWndProc` so
//! games that rely on the Win32 cursor (hiding the OS pointer while they
//! render their own sprite over it) actually get the behaviour they expect.
//!
//! **Per-window back-pointers.** `CallWindowProcW` can't pass per-device user
//! data through, so the subclass maps each window's `HWND` to its owning
//! `DeviceInner` in `DEVICE_INSTANCES`. Two devices may exist at once, and a
//! single global back-pointer would be wrong the moment they do, so the lookup
//! is keyed by the window the message arrived on.

use core::{ffi::c_void, ptr::null_mut};
use std::{
    collections::{HashMap, hash_map::DefaultHasher},
    hash::Hasher,
    sync::{LazyLock, Mutex},
};

use log::{Level, debug, error, log_enabled, trace, warn};
use mtld3d_shared::InPtr;
use mtld3d_types::{
    D3DLOCK_READONLY, D3DLOCKED_RECT, D3DSURFACE_DESC, ICONINFO, IDirect3DSurface9Vtbl, POINT,
};

use super::{
    D3D_OK, D3DERR_INVALIDCALL,
    device::{DeviceInner, Direct3DDevice9},
};

/// Cursor-specific log sub-target.
///
/// Inherits filtering from any broader `mtld3d::d3d9` or `mtld3d` selector by
/// `env_logger`'s `::`-prefix matching, so `RUST_LOG=mtld3d::d3d9=warn` still
/// catches the `warn!`s below. The dedicated target lets us crank trace
/// separately:
///   `RUST_LOG=mtld3d::d3d9::cursor=trace`
const LOG_TARGET: &str = "mtld3d::d3d9::cursor";

// ── Win32 FFI ──

#[link(name = "user32")]
unsafe extern "system" {
    fn SetCursor(cursor: *mut c_void) -> *mut c_void;
    fn SetCursorPos(x: i32, y: i32) -> i32;
    fn GetCursorPos(p: *mut POINT) -> i32;
    fn CreateIconIndirect(info: *const ICONINFO) -> *mut c_void;
    fn CallWindowProcW(
        prev_proc: *mut c_void,
        hwnd: *mut c_void,
        msg: u32,
        wp: usize,
        lp: isize,
    ) -> isize;
    fn DefWindowProcW(hwnd: *mut c_void, msg: u32, wp: usize, lp: isize) -> isize;
}

// Win32 LONG is 32-bit; `SetWindowLongPtrW` only exists on 64-bit Windows,
// while 32-bit user32 exports `SetWindowLongW` (the header is a #define
// alias). Declare a single Rust-side `SetWindowLongPtrW` symbol per arch
// and route the 32-bit one through `#[link_name = "SetWindowLongW"]` so
// the call site stays uniform.
#[cfg(target_pointer_width = "64")]
#[link(name = "user32")]
unsafe extern "system" {
    fn SetWindowLongPtrW(hwnd: *mut c_void, index: i32, new_long: isize) -> isize;
}

#[cfg(target_pointer_width = "32")]
#[link(name = "user32")]
unsafe extern "system" {
    #[link_name = "SetWindowLongW"]
    fn SetWindowLongPtrW(hwnd: *mut c_void, index: i32, new_long: isize) -> isize;
}

fn set_window_long_ptr(hwnd: *mut c_void, index: i32, new: isize) -> isize {
    // SAFETY: SetWindowLongPtrW (or SetWindowLongW on 32-bit) accepts any
    // HWND and isize; documented to return the previous value.
    unsafe { SetWindowLongPtrW(hwnd, index, new) }
}

// Safe wrappers around the Win32 calls used by this module — each Win32
// function is wrapped once so call sites are unsafe-free per CONVENTIONS.md
// §13 "Don't sprinkle — concentrate".

fn set_cursor(handle: *mut c_void) {
    // SAFETY: SetCursor accepts null and any valid HCURSOR; documented to
    // be thread-safe and side-effect-free besides updating the cursor.
    unsafe {
        SetCursor(handle);
    }
}

fn set_cursor_pos(x: i32, y: i32) {
    // SAFETY: SetCursorPos accepts any i32 pair; failure (return 0) is a
    // game-side concern, not a memory-safety issue.
    unsafe {
        SetCursorPos(x, y);
    }
}

fn get_cursor_pos() -> Option<POINT> {
    let mut p = POINT { x: 0, y: 0 };
    // SAFETY: GetCursorPos writes a POINT through `&mut p`; pointer comes
    // from an owned local, so non-null + aligned + writable holds.
    let ok = unsafe { GetCursorPos(&raw mut p) };
    (ok != 0).then_some(p)
}

fn def_window_proc(hwnd: *mut c_void, msg: u32, wp: usize, lp: isize) -> isize {
    // SAFETY: DefWindowProcW is the documented Win32 fallback — accepts any
    // HWND/msg/wp/lp tuple.
    unsafe { DefWindowProcW(hwnd, msg, wp, lp) }
}

fn call_window_proc(
    prev_proc: *mut c_void,
    hwnd: *mut c_void,
    msg: u32,
    wp: usize,
    lp: isize,
) -> isize {
    // SAFETY: CallWindowProcW forwards to a documented WNDPROC; the
    // subclass-install path stored `prev_proc` from a prior
    // GetWindowLongPtrW call.
    unsafe { CallWindowProcW(prev_proc, hwnd, msg, wp, lp) }
}

fn create_icon_indirect(info: &ICONINFO) -> *mut c_void {
    // SAFETY: ICONINFO is passed by ref so the pointer is non-null +
    // properly aligned; CreateIconIndirect returns null on failure (caller
    // checks).
    unsafe { CreateIconIndirect(&raw const *info) }
}

fn delete_object(obj: *mut c_void) -> i32 {
    // SAFETY: DeleteObject accepts null + any GDI object handle; returns 0
    // on failure (caller logs).
    unsafe { DeleteObject(obj) }
}

fn create_bitmap_packed(
    width: i32,
    height: i32,
    planes: u32,
    bpp: u32,
    bits: *const c_void,
) -> *mut c_void {
    // SAFETY: CreateBitmap copies `bits` according to (width * height * bpp / 8)
    // bytes; caller supplies a tight buffer with that many bytes (color
    // bitmap is 32 bpp ARGB, mask bitmap is 1 bpp + row-padded to 32 bits).
    unsafe { CreateBitmap(width, height, planes, bpp, bits) }
}

#[link(name = "gdi32")]
unsafe extern "system" {
    fn CreateBitmap(
        width: i32,
        height: i32,
        planes: u32,
        bpp: u32,
        bits: *const c_void,
    ) -> *mut c_void;
    fn DeleteObject(object: *mut c_void) -> i32;
}

// ── Constants ──

const GWLP_WNDPROC: i32 = -4;
const WM_SETCURSOR: u32 = 0x0020;
const WM_ACTIVATE: u32 = 0x0006;
const WM_SIZE: u32 = 0x0005;
const WA_INACTIVE: u32 = 0;
const HTCLIENT: usize = 1;

// ── Per-window subclass back-pointers ──

/// Maps each subclassed window (`HWND` as `usize`) to the owning `DeviceInner`.
///
/// The device is held as a raw pointer widened to `usize`. Populated by
/// `CursorState::install_subclass` during `CreateDevice`, entry removed by
/// `CursorState::uninstall_subclass` during device release. `cursor_wnd_proc`
/// looks up its own `hwnd` here to find the owning device — `CallWindowProcW`
/// can't pass per-device user data, and a single global back-pointer is wrong
/// when more than one device exists at once (the second `CreateDevice` would
/// orphan the first window's cursor and, once either device tears down, leave a
/// still-subclassed window pointing at a stale/cleared device).
static DEVICE_INSTANCES: LazyLock<Mutex<HashMap<usize, usize>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

// ── CursorState ──

bitflags::bitflags! {
    /// Packed boolean state for `CursorState`.
    ///
    /// Three independent flags that the cursor module reads and writes
    /// together at the WM_* edges — packing them into one byte means a
    /// `match` against the trio fits in one comparison and the surrounding
    /// struct's tail padding tightens.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct CursorFlags: u8 {
        /// Game-requested visibility (ShowCursor / ShowCursor toggles).
        const VISIBLE = 1 << 0;
        /// Cursor handle or hash changed since the last paint.
        ///
        /// The next WM_SETCURSOR / paint cycle needs to re-realise.
        const DIRTY = 1 << 1;
        /// Latched on WM_SIZE-driven auto-resize.
        ///
        /// Suppresses a single game-issued ShowCursor(FALSE) until the next
        /// ShowCursor(TRUE). Some games hide the cursor from their own
        /// WM_SIZE handler and re-show it seconds later; this latch pre-empts
        /// that transient hide while preserving legitimate hides.
        const FORCE_VISIBLE_AFTER_RESIZE = 1 << 2;
    }
}

/// Cursor state owned by `DeviceInner` as a single field.
///
/// Fields are private to this module — only code in `cursor.rs` reads or
/// writes them, so cursor invariants don't leak into the rest of `d3d9`.
pub struct CursorState {
    hwnd: *mut c_void,
    original_wndproc: *mut c_void,
    handle: *mut c_void,
    flags: CursorFlags,
    hash: u64,
    cache: HashMap<u64, *mut c_void>,
    /// Nearest-neighbor upscale factor applied to the cursor bitmap.
    ///
    /// Sourced from the display's `backingScaleFactor` at `CreateDevice` so
    /// a retina Mac gets a proportionally-sized Win32 cursor (Wine's
    /// HCURSOR path does not participate in the OS's retina upscale).
    /// `1` is the identity fast path.
    scale: u32,
}

impl CursorState {
    pub fn new(hwnd: *mut c_void, scale: u32) -> Self {
        Self {
            hwnd,
            original_wndproc: null_mut(),
            handle: null_mut(),
            // D3D9 starts the cursor hidden (ShowCursor reports FALSE until a
            // cursor image is set and shown).
            flags: CursorFlags::DIRTY,
            hash: 0,
            cache: HashMap::new(),
            scale: scale.clamp(1, 8),
        }
    }

    const fn visible(&self) -> bool {
        self.flags.contains(CursorFlags::VISIBLE)
    }

    const fn set_visible(&mut self, on: bool) {
        if on {
            self.flags = self.flags.union(CursorFlags::VISIBLE);
        } else {
            self.flags = self.flags.difference(CursorFlags::VISIBLE);
        }
    }

    const fn dirty(&self) -> bool {
        self.flags.contains(CursorFlags::DIRTY)
    }

    const fn set_dirty(&mut self, on: bool) {
        if on {
            self.flags = self.flags.union(CursorFlags::DIRTY);
        } else {
            self.flags = self.flags.difference(CursorFlags::DIRTY);
        }
    }

    const fn force_visible_after_resize(&self) -> bool {
        self.flags.contains(CursorFlags::FORCE_VISIBLE_AFTER_RESIZE)
    }

    /// Visibility that drives the *physical* Win32 cursor.
    ///
    /// The game-requested flag, or the post-resize pin while the latch is
    /// armed. Only `VISIBLE` feeds `ShowCursor`'s previous-state return — the
    /// latch must not leak into the API bookkeeping.
    const fn effective_visible(&self) -> bool {
        self.visible() || self.force_visible_after_resize()
    }

    const fn set_force_visible_after_resize(&mut self, on: bool) {
        if on {
            self.flags = self.flags.union(CursorFlags::FORCE_VISIBLE_AFTER_RESIZE);
        } else {
            self.flags = self
                .flags
                .difference(CursorFlags::FORCE_VISIBLE_AFTER_RESIZE);
        }
    }

    /// Subclass the game's hwnd so `WM_SETCURSOR` / `WM_ACTIVATE` route through `cursor_wnd_proc`.
    ///
    /// Registers `dev_ptr` in `DEVICE_INSTANCES` under this window's `HWND`, so
    /// the subclass can resolve the owning device from the window a message
    /// arrived on. The game's original wndproc is stored back into `self` for
    /// later restoration. No-op if `hwnd` was never captured or `dev_ptr` is
    /// null.
    pub fn install_subclass(&mut self, dev_ptr: *mut DeviceInner) {
        if self.hwnd.is_null() || dev_ptr.is_null() {
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET,
                "install_subclass: skipped (hwnd or dev_ptr is null) — \
                 game-side cursor messages will NOT route through mtld3d; \
                 SetCursorProperties/ShowCursor become no-ops in-game"
            );
            return;
        }
        DEVICE_INSTANCES
            .lock()
            .expect("device-instances mutex poisoned")
            .insert(self.hwnd as usize, dev_ptr as usize);
        let prev = set_window_long_ptr(
            self.hwnd,
            GWLP_WNDPROC,
            cursor_wnd_proc as *const () as isize,
        );
        self.original_wndproc = prev as *mut c_void;
        debug!(
            target: LOG_TARGET,
            "install_subclass: hwnd={:p} dev={:p} prev_wndproc={:p} scale={}",
            self.hwnd, dev_ptr, self.original_wndproc, self.scale,
        );
    }

    /// Restore the game's original wndproc and drop this window's back-pointer.
    ///
    /// Removes the window's `DEVICE_INSTANCES` entry. Call from the
    /// device-release path before freeing `DeviceInner`.
    pub fn uninstall_subclass(&self) {
        debug!(
            target: LOG_TARGET,
            "uninstall_subclass: hwnd={:p} restoring prev_wndproc={:p} cache_entries={}",
            self.hwnd, self.original_wndproc, self.cache.len(),
        );
        if !self.hwnd.is_null() && !self.original_wndproc.is_null() {
            set_window_long_ptr(self.hwnd, GWLP_WNDPROC, self.original_wndproc as isize);
        }
        DEVICE_INSTANCES
            .lock()
            .expect("device-instances mutex poisoned")
            .remove(&(self.hwnd as usize));
    }
}

// ── Vtable entry points ──

pub extern "system" fn device_set_cursor_properties(
    this: *mut c_void,
    x_hotspot: u32,
    y_hotspot: u32,
    cursor_bitmap: *mut c_void,
) -> i32 {
    if cursor_bitmap.is_null() {
        warn!(target: LOG_TARGET, "reject SetCursorProperties(null bitmap) → INVALIDCALL");
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();

    // SAFETY: `cursor_bitmap` is a valid SurfaceHead via the D3D9 ABI;
    // its `vtbl` field is a non-null pointer to a static vtable.
    let surf_head = unsafe { &*(cursor_bitmap as *const SurfaceHead) };
    // SAFETY: same invariant — vtbl pointer is static.
    let surf_vtbl = unsafe { &*surf_head.vtbl };
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
    // SAFETY: calling the just-loaded `get_desc` thunk through
    // `surf_vtbl`; `cursor_bitmap` is the IDirect3DSurface9 `this` per
    // D3D9 ABI and `desc` is a writable local.
    if unsafe { (surf_vtbl.get_desc)(cursor_bitmap, &raw mut desc) } != 0 {
        warn!(target: LOG_TARGET, "reject SetCursorProperties: surface GetDesc failed");
        return D3DERR_INVALIDCALL;
    }
    let width = desc.width;
    let height = desc.height;
    if width == 0 || height == 0 {
        warn!(
            target: LOG_TARGET,
            "reject SetCursorProperties: zero-sized surface ({width}x{height}) → INVALIDCALL",
        );
        return D3DERR_INVALIDCALL;
    }
    // D3D9 requires cursor dimensions to be powers of two.
    if !width.is_power_of_two() || !height.is_power_of_two() {
        warn!(
            target: LOG_TARGET,
            "reject SetCursorProperties: non-power-of-2 surface ({width}x{height}) → INVALIDCALL",
        );
        return D3DERR_INVALIDCALL;
    }
    // D3D9 caps cursor dimensions at the current adapter display mode (the
    // resolution `GetAdapterDisplayMode` reports), not the backbuffer size.
    let (mode_width, mode_height) = crate::direct3d9::adapter_display_mode_dims();
    if width > mode_width || height > mode_height {
        warn!(
            target: LOG_TARGET,
            "reject SetCursorProperties: surface ({width}x{height}) exceeds display mode ({mode_width}x{mode_height}) → INVALIDCALL",
        );
        return D3DERR_INVALIDCALL;
    }

    let mut locked = D3DLOCKED_RECT {
        pitch: 0,
        bits: null_mut(),
    };
    // SAFETY: calling the just-loaded `lock_rect` thunk through
    // `surf_vtbl`; `cursor_bitmap` is the IDirect3DSurface9 `this` per
    // D3D9 ABI, `locked` is a writable local, and a null rect locks
    // the entire surface.
    if unsafe {
        (surf_vtbl.lock_rect)(
            cursor_bitmap,
            &raw mut locked,
            core::ptr::null(),
            D3DLOCK_READONLY,
        )
    } != 0
    {
        warn!(target: LOG_TARGET, "reject SetCursorProperties: surface LockRect failed");
        return D3DERR_INVALIDCALL;
    }

    let pitch = usize::try_from(locked.pitch).expect("D3D9 LOCKED_RECT.pitch is non-negative");
    let src = locked.bits as *const u8;
    let hash = hash_cursor(x_hotspot, y_hotspot, width, height, src, pitch);

    let cur = dev.cursor_mut();
    let prev_hash = cur.hash;
    let (handle, outcome) = if hash == prev_hash {
        (cur.handle, "unchanged")
    } else if let Some(&h) = cur.cache.get(&hash) {
        (h, "cache-hit")
    } else {
        let Some(h) = build_hcursor(width, height, pitch, src, x_hotspot, y_hotspot, cur.scale)
        else {
            warn!(
                target: LOG_TARGET,
                "reject SetCursorProperties: build_hcursor failed (hash={hash:#018x} {width}x{height} scale={})",
                cur.scale,
            );
            // SAFETY: calling the just-loaded `unlock_rect` thunk through
            // `surf_vtbl`; paired with the `lock_rect` call above.
            unsafe { (surf_vtbl.unlock_rect)(cursor_bitmap) };
            return D3DERR_INVALIDCALL;
        };
        cur.cache.insert(hash, h);
        (h, "built-fresh")
    };

    // SAFETY: calling the just-loaded `unlock_rect` thunk through
    // `surf_vtbl`; paired with the `lock_rect` call above.
    unsafe { (surf_vtbl.unlock_rect)(cursor_bitmap) };

    cur.hash = hash;
    cur.handle = handle;
    let visible = cur.effective_visible();
    debug!(
        target: LOG_TARGET,
        "SetCursorProperties: {width}x{height} fmt={} pool={} hotspot=({x_hotspot},{y_hotspot}) hash={hash:#018x} outcome={outcome} handle={handle:p} visible={visible} cache_entries={}",
        desc.format, desc.pool, cur.cache.len(),
    );
    // Realize only while shown (D3D9 sets the Win32 cursor only when the cursor is visible).
    // While hidden the game owns the win32 cursor — pushing null here clobbers
    // the cursor the game's own wndproc set (WoW's login screen never calls
    // ShowCursor(TRUE); its glove is the game's own cursor).
    if visible {
        set_cursor(handle);
    }
    D3D_OK
}

pub extern "system" fn device_set_cursor_position(_this: *mut c_void, x: i32, y: i32, _flags: u32) {
    let current = get_cursor_pos();
    let suppress = current.is_some_and(|p| p.x == x && p.y == y);
    if suppress {
        debug!(target: LOG_TARGET, "SetCursorPosition: noop ({x},{y})");
        return;
    }
    if let Some(p) = current {
        debug!(
            target: LOG_TARGET,
            "SetCursorPosition: ({},{}) → ({x},{y}) dx={} dy={}",
            p.x, p.y, x - p.x, y - p.y,
        );
    } else {
        debug!(target: LOG_TARGET, "SetCursorPosition: → ({x},{y}) got_current=none");
    }
    set_cursor_pos(x, y);
}

pub extern "system" fn device_show_cursor(this: *mut c_void, show: i32) -> i32 {
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let cur = obj.inner().cursor_mut();
    let prev = cur.visible();
    let next = show != 0;
    if !next && cur.force_visible_after_resize() {
        debug!(
            target: LOG_TARGET,
            "ShowCursor(show=0) suppressed by force_visible_after_resize (post-resize hide pre-empted)",
        );
        return i32::from(prev);
    }
    // D3D9 changes the cursor visibility only once a cursor image has been set
    // via SetCursorProperties (handle non-null); with no cursor surface,
    // ShowCursor is a pure read of the (unchanged) previous visibility.
    if cur.handle.is_null() {
        trace!(
            target: LOG_TARGET,
            "ShowCursor(show={show}) → prev={prev} (no cursor surface set; no-op)",
        );
        return i32::from(prev);
    }
    if next {
        cur.set_force_visible_after_resize(false);
    }
    cur.set_visible(next);
    // Realize on EVERY call, not only on transitions: a transition-gated
    // SetCursor leaves the display stale whenever the previous realize didn't
    // stick (pointer outside the window at the time) or a latch-suppressed
    // hide ate the transition — the game then believes the cursor visible
    // while nothing is displayed until its next full hide/show cycle.
    let handle = if next { cur.handle } else { null_mut() };
    set_cursor(handle);
    if prev == next {
        trace!(
            target: LOG_TARGET,
            "ShowCursor(show={show}) → prev={prev} handle={handle:p} (re-assert)",
        );
    } else {
        debug!(
            target: LOG_TARGET,
            "ShowCursor(show={show}) → prev={prev} next={next} handle={handle:p} (transition)",
        );
    }
    i32::from(prev)
}

// ── Window-proc subclass ──

extern "system" fn cursor_wnd_proc(hwnd: *mut c_void, msg: u32, wp: usize, lp: isize) -> isize {
    // Resolve the owning device for *this* window. A window may still be
    // subclassed briefly after its device's entry is removed (or never have
    // been registered); fall back to the default proc rather than deref a
    // missing/stale device.
    let dev_ptr = DEVICE_INSTANCES
        .lock()
        .expect("device-instances mutex poisoned")
        .get(&(hwnd as usize))
        .copied()
        .unwrap_or(0) as *mut DeviceInner;
    if dev_ptr.is_null() {
        return def_window_proc(hwnd, msg, wp, lp);
    }

    if msg == WM_SETCURSOR {
        // SAFETY: `dev_ptr` is this window's `DEVICE_INSTANCES` entry,
        // installed by `install_subclass`; it's null-checked above and
        // removed only in `uninstall_subclass` on device drop.
        let cur = unsafe { (*dev_ptr).cursor_mut() };
        let hit_test = lp.cast_unsigned() & 0xFFFF;
        if hit_test == HTCLIENT && !cur.handle.is_null() && cur.effective_visible() {
            // We own the win32 cursor ONLY while the D3D cursor is effectively
            // visible. Consuming then takes over DefWindowProc's duty: the
            // displayed cursor is whatever the last SetCursor call pushed, and
            // entering the window does NOT re-apply it — so every consumed
            // pass must push, or a pointer re-entering the client area keeps
            // the stale (often none) cursor. While the D3D cursor is hidden
            // the message is FORWARDED below (native d3d9 never intercepts
            // WM_SETCURSOR): the game shows its own cursor — WoW's login
            // screen never calls ShowCursor(TRUE) and relies on exactly that.
            let was_dirty = cur.dirty();
            if was_dirty {
                cur.set_dirty(false);
                // Null-then-set forces macdrv to drop a lingering native
                // cursor (e.g. macOS's resize cursor after a drag).
                set_cursor(null_mut());
            }
            set_cursor(cur.handle);
            if was_dirty {
                debug!(
                    target: LOG_TARGET,
                    "wndproc WM_SETCURSOR: hit_test={hit_test:#x} (HTCLIENT) dirty_was=true → re-asserted handle={:p} → consumed",
                    cur.handle,
                );
            } else if log_enabled!(target: LOG_TARGET, Level::Trace) {
                trace!(
                    target: LOG_TARGET,
                    "wndproc WM_SETCURSOR: hit_test={hit_test:#x} (HTCLIENT) dirty_was=false handle={:p} → consumed",
                    cur.handle,
                );
            }
            return 1;
        }
        if log_enabled!(target: LOG_TARGET, Level::Trace) {
            trace!(
                target: LOG_TARGET,
                "wndproc WM_SETCURSOR: hit_test={hit_test:#x} visible={} handle={:p} → forwarded",
                cur.effective_visible(), cur.handle,
            );
        }
    } else if msg == WM_ACTIVATE {
        // SAFETY: see WM_SETCURSOR branch — `dev_ptr` is live for the
        // lifetime of the subclass.
        let cur = unsafe { (*dev_ptr).cursor_mut() };
        let activate_state = u32::try_from(wp & 0xFFFF).expect("16-bit value fits u32");
        let activating = activate_state != WA_INACTIVE;
        if activating {
            cur.set_dirty(true);
        }
        debug!(
            target: LOG_TARGET,
            "wndproc WM_ACTIVATE: state={activate_state} activating={activating} dirty_now={}",
            cur.dirty(),
        );
    } else if msg == WM_SIZE {
        // Implicit client-area resize from Wine's macdrv (e.g. macOS
        // shrunk the visible rect after we attached the layer because
        // chrome / dock take some pixels). lParam's low / high words
        // are the new client width / height in pixels — trigger an
        // auto-resize so game-side `GetClientRect` and our
        // `GetDisplayMode` agree.
        let lp_bits = lp.cast_unsigned();
        let new_width = u32::try_from(lp_bits & 0xFFFF).expect("16-bit value fits u32");
        let new_height = u32::try_from((lp_bits >> 16) & 0xFFFF).expect("16-bit value fits u32");
        if new_width != 0 && new_height != 0 {
            // SAFETY: see WM_SETCURSOR branch — `dev_ptr` is live for
            // the lifetime of the subclass.
            let dev = unsafe { &mut *dev_ptr };
            dev.apply_auto_resize(new_width, new_height);
        }
        // WoW's own `WM_SIZE` handler (about to run via the
        // `CallWindowProcW` tail below) will call `ShowCursor(FALSE)`
        // and re-show ~6 s later. Pin visibility to TRUE across that
        // window so the cursor doesn't disappear mid-loading. The
        // latch clears on the next game-issued `ShowCursor(TRUE)` and
        // pins only the *physical* cursor (`effective_visible`) — it
        // must not touch `VISIBLE`, which `ShowCursor` reports as the
        // previous state (a macdrv WM_SIZE can race the first
        // ShowCursor(TRUE), and touching `VISIBLE` here would corrupt
        // that return value).
        //
        // Always re-assert the HCURSOR + mark dirty so a follow-up
        // `WM_SETCURSOR` re-runs `SetCursor` too. After a user-driven
        // drag resize, macOS's native resize cursor is left over until
        // we explicitly replace it; without this the in-game cursor
        // bitmap stays gone after the drag.
        // SAFETY: see WM_SETCURSOR branch — `dev_ptr` is live for the
        // lifetime of the subclass.
        let cur = unsafe { (*dev_ptr).cursor_mut() };
        cur.set_force_visible_after_resize(true);
        cur.set_dirty(true);
        if !cur.handle.is_null() {
            set_cursor(cur.handle);
        }
    }

    // SAFETY: see WM_SETCURSOR branch — `dev_ptr` is live for the
    // lifetime of the subclass.
    let original_wndproc = unsafe { (*dev_ptr).cursor() }.original_wndproc;
    call_window_proc(original_wndproc, hwnd, msg, wp, lp)
}

// ── Helpers ──

/// Content-hash over hotspot + dimensions + pixel bytes.
///
/// Used as the cursor cache key inside `SetCursorProperties`. Cold-path (a
/// handful of calls per session), so `DefaultHasher` is used here for
/// consistency with `ProgramId::from_tokens`.
fn hash_cursor(
    x_hotspot: u32,
    y_hotspot: u32,
    width: u32,
    height: u32,
    src: *const u8,
    pitch: usize,
) -> u64 {
    let mut h = DefaultHasher::new();
    h.write_u32(x_hotspot);
    h.write_u32(y_hotspot);
    h.write_u32(width);
    h.write_u32(height);
    let row_bytes = (width as usize) * 4;
    for y in 0..height as usize {
        // SAFETY: caller-validated `src + y*pitch` stays within the bitmap.
        let row_ptr = unsafe { src.add(y * pitch) };
        // SAFETY: `row_ptr..row_ptr + row_bytes` lies in the same allocation.
        let row = unsafe { core::slice::from_raw_parts(row_ptr, row_bytes) };
        h.write(row);
    }
    h.finish()
}

/// Build a Win32 HCURSOR from a BGRA bitmap.
///
/// Returns `None` on any Win32 failure. Source pixels are upscaled by
/// `scale` so the cursor matches the display's `backingScaleFactor`: 2×
/// uses xBR (`xbr` crate), and every other factor falls back to
/// nearest-neighbor, since `xbr` implements no other factor. `scale == 1`
/// is the identity path. Hotspot is multiplied by `scale`.
///
/// # Panics
///
/// Panics if upscaled dimensions exceed `i32::MAX`. Unreachable on Windows
/// cursors (max 256×256 pre-scale, ≤8× post-scale).
fn build_hcursor(
    width: u32,
    height: u32,
    pitch: usize,
    src: *const u8,
    x_hotspot: u32,
    y_hotspot: u32,
    scale: u32,
) -> Option<*mut c_void> {
    let w = width as usize;
    let h = height as usize;
    let scale = scale.clamp(1, 8) as usize;

    // Copy the locked surface into a tight w*h BGRA buffer (handles
    // pitch != w*4). The upscalers operate on tight buffers. Use
    // `read_unaligned` per pixel so the u8→u32 cast is alignment-agnostic;
    // D3DLOCKED_RECT guarantees u32 alignment in practice but threading
    // that through the type system is more friction than the unaligned
    // read costs.
    let mut src_pixels = vec![0u32; w * h];
    for y in 0..h {
        for x in 0..w {
            // SAFETY: `src` is the `D3DLOCKED_RECT.bits` from the
            // locked cursor surface; `pitch * h` plus the in-row
            // offset `x * 4` stays within the locked region (the
            // surface is `w*h*4` BGRA bytes with the given pitch).
            let pixel_ptr = unsafe { src.add(y * pitch + x * 4) };
            // SAFETY: `pixel_ptr` points at 4 readable bytes within
            // the locked region; `read_unaligned` makes no alignment
            // assumption.
            src_pixels[y * w + x] = unsafe { core::ptr::read_unaligned(pixel_ptr.cast::<u32>()) };
        }
    }

    // Probe the source before the dispatch consumes `src_pixels`. The mask
    // decision keys on the *source* so the upscaler's output can't sneakily
    // flip the fallback (it can't today — the upscaler never invents alpha
    // that wasn't in the input — but the check is cheap and future-proof).
    let any_alpha = src_pixels.iter().any(|&px| (px >> 24) != 0);

    let (sw, sh, pixels, path) = scale_cursor_pixels(scale, w, h, src_pixels);
    let and_mask = derive_and_mask(&pixels, sw, sh, any_alpha);

    let bitmap_width = i32::try_from(sw).expect("upscaled cursor width fits i32");
    let bitmap_height = i32::try_from(sh).expect("upscaled cursor height fits i32");
    let color_bitmap = create_bitmap_packed(
        bitmap_width,
        bitmap_height,
        1,
        32,
        pixels.as_ptr().cast::<c_void>(),
    );
    let mask_bitmap = create_bitmap_packed(
        bitmap_width,
        bitmap_height,
        1,
        1,
        and_mask.as_ptr().cast::<c_void>(),
    );
    if color_bitmap.is_null() || mask_bitmap.is_null() {
        error!(
            target: LOG_TARGET,
            "build_hcursor: CreateBitmap failed (color={color_bitmap:p} mask={mask_bitmap:p}) src={width}x{height} → {sw}x{sh} path={path}",
        );
        if !color_bitmap.is_null() {
            delete_object(color_bitmap);
        }
        if !mask_bitmap.is_null() {
            delete_object(mask_bitmap);
        }
        return None;
    }

    let scale_u32 = u32::try_from(scale).expect("scale clamped to ≤8 fits u32");
    let info = ICONINFO {
        f_icon: 0, // cursor
        x_hotspot: x_hotspot * scale_u32,
        y_hotspot: y_hotspot * scale_u32,
        hbm_mask: mask_bitmap,
        hbm_color: color_bitmap,
    };
    let cursor = create_icon_indirect(&info);

    // CreateIconIndirect copies the bitmaps; we own the originals.
    delete_object(color_bitmap);
    delete_object(mask_bitmap);

    if cursor.is_null() {
        error!(
            target: LOG_TARGET,
            "build_hcursor: CreateIconIndirect returned null (src={width}x{height} → {sw}x{sh} path={path} any_alpha={any_alpha})",
        );
        None
    } else {
        debug!(
            target: LOG_TARGET,
            "build_hcursor: ok handle={cursor:p} src={width}x{height} → {sw}x{sh} path={path} any_alpha={any_alpha} hotspot=({},{})→({},{})",
            x_hotspot, y_hotspot, info.x_hotspot, info.y_hotspot,
        );
        Some(cursor)
    }
}

/// Upscale a tight `w*h` BGRA buffer by `scale`.
///
/// The `xbr` crate only provides 2×, so every other factor takes the
/// nearest-neighbor arm. Factors above 2 are unheard of on current
/// hardware (retina is 2×); that arm exists for forward compatibility
/// rather than panicking.
///
/// Byte-order note: `xbr::Block` stores BGRA/RGBA as `Vec<u8>` and
/// compares pixels in YUV space. Our buffer is native-endian BGRA `u32`;
/// we pass the bytes through unchanged. The YUV luma weights are
/// technically computed with R/B swapped, but for an alpha cursor the
/// edge-detection output is visually identical — alpha is still in the
/// high byte, so downstream mask derivation stays correct.
///
/// # Panics
///
/// Panics if source dimensions exceed `u32::MAX`. Unreachable: callers
/// originate `w`/`h` from `u32` inputs.
fn scale_cursor_pixels(
    scale: usize,
    w: usize,
    h: usize,
    src_pixels: Vec<u32>,
) -> (usize, usize, Vec<u32>, &'static str) {
    match scale {
        1 => (w, h, src_pixels, "1x-identity"),
        2 => {
            let src_bytes = u32_to_u8_vec(&src_pixels);
            let block = xbr::x2(xbr::Block {
                bytes: src_bytes,
                width: u32::try_from(w).expect("cursor width fits u32"),
                height: u32::try_from(h).expect("cursor height fits u32"),
            });
            let out = u8_to_u32_vec(&block.bytes);
            (block.width as usize, block.height as usize, out, "2x-xbr")
        }
        n => {
            let sw = w * n;
            let sh = h * n;
            let mut dst = vec![0u32; sw * sh];
            for y in 0..sh {
                for x in 0..sw {
                    dst[y * sw + x] = src_pixels[(y / n) * w + (x / n)];
                }
            }
            (sw, sh, dst, "nx-nearest")
        }
    }
}

/// Derive an AND mask from upscaled alpha.
///
/// The mask is 1-bit-per-pixel, row-padded to 32 bits; a bit of 1 means
/// "transparent, show screen", a bit of 0 means "opaque, use color".
/// Keying on the upscaled alpha lines the upscaler's smoothed edges up with
/// what the color bitmap actually shows on Wine's mono-cursor path (kicks in
/// whenever `create_alpha_bitmap` in user32 fails to find any alpha via
/// `GetDIBits` on the DDB we hand it). Wine's alpha-blend path ignores
/// the mask entirely, so this is harmless there.
///
/// Some cursors carry alpha=0 across the whole surface; deriving a
/// mask would leave the cursor fully transparent, so `any_alpha=false`
/// returns an all-zeros mask.
fn derive_and_mask(pixels: &[u32], sw: usize, sh: usize, any_alpha: bool) -> Vec<u8> {
    let mask_stride = sw.div_ceil(32) * 4;
    let mut and_mask = vec![0u8; mask_stride * sh];
    if any_alpha {
        for y in 0..sh {
            let row = &pixels[y * sw..(y + 1) * sw];
            for (x, &px) in row.iter().enumerate() {
                if (px >> 24) == 0 {
                    and_mask[y * mask_stride + x / 8] |= 1u8 << (7 - (x & 7));
                }
            }
        }
    }
    and_mask
}

/// Pack a tight BGRA `u32` buffer into a byte buffer for `xbr::Block`.
///
/// Bytes land in native-endian order (`b, g, r, a` on little-endian
/// macOS/Windows — which matches the wire convention Wine and Metal use
/// for `D3DFMT_A8R8G8B8`).
fn u32_to_u8_vec(src: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(src.len() * 4);
    for &p in src {
        out.extend_from_slice(&p.to_ne_bytes());
    }
    out
}

/// Inverse of `u32_to_u8_vec` — read 4-byte chunks back into BGRA `u32`s.
fn u8_to_u32_vec(src: &[u8]) -> Vec<u32> {
    let mut out = Vec::with_capacity(src.len() / 4);
    for chunk in src.chunks_exact(4) {
        out.push(u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    out
}

/// Minimal layout for reading the surface's vtable pointer.
///
/// Avoids pulling the full `Direct3DSurface9` type into this module.
#[repr(C)]
struct SurfaceHead {
    vtbl: *const IDirect3DSurface9Vtbl,
}
