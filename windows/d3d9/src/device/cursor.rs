//! The cursor methods of `IDirect3DDevice9`, their state, and the words the unix side publishes.
//!
//! The entry points unpack the D3D9 arguments and hand the operation to the
//! device's [`CursorState`], which keeps the D3D9 cursor rules and the
//! software overlay and hands what the system realizes to its
//! [`SystemCursor`].
//!
//! **Software mode.** With `cursor.software` resolved on, the game's cursor is
//! drawn by the unix side in an overlay window: `SetCursorProperties` ships the
//! upscaled bitmap through the `SetCursorOverlay` thunk, `ShowCursor` ships the
//! visibility, and the system cursor is a blank one.

use core::{
    ffi::c_void,
    hash::Hasher,
    ptr::null_mut,
    sync::atomic::{AtomicU32, Ordering},
};
use std::time::Instant;

use log::{debug, info, trace, warn};
use mtld3d_core::{
    cursor::{BitmapLayout, reconcile_upload},
    perf::DeviceSubCategory,
};
use mtld3d_shared::{
    InPtr, MetalHandle, PresentClientSurfaceParams, SetCursorOverlayParams,
    bounded_cache::BoundedCache,
    mtl::{CURSOR_SPRITE_CACHE_ENTRIES, ClientSurfaceRequest, CursorOverlayFlags},
    mtl_handle::NSViewKind,
};
use mtld3d_types::{D3DLOCK_READONLY, D3DLOCKED_RECT, D3DSURFACE_DESC, IDirect3DSurface9Vtbl};
use xxhash_rust::xxh3::Xxh3;

use super::{DeviceInner, Direct3DDevice9, device_api_lock, device_timer};
use crate::{
    D3D_OK, D3DERR_INVALIDCALL,
    platform::cursor::{CursorImage, SystemCursor, current_thread_id},
    unix_call::unix_call,
};

/// Cursor-specific log sub-target.
///
/// Inherits filtering from any broader `mtld3d::d3d9` or `mtld3d` selector by
/// `env_logger`'s `::`-prefix matching, so `RUST_LOG=mtld3d::d3d9=warn` still
/// catches the `warn!`s below. The dedicated target lets us crank trace
/// separately:
///   `RUST_LOG=mtld3d::d3d9::cursor=trace`
pub const LOG_TARGET: &str = "mtld3d::d3d9::cursor";

/// The three words the unix side writes into from its main thread, at a stable address.
///
/// `AttachMetalLayer` hands the unix side the addresses, and the record it
/// keeps for the device's view writes through them outside any thunk: the
/// backing scale at attach and whenever the display-follow reconciliation
/// derives another, the cursor kick when the pointer comes back from another
/// process, the client surface request at attach and when Wine hides the
/// device's client view. The window procedure sets the request too, when the
/// window's client area may have moved. The box is created before the attach
/// and owned by the device's `CursorState`, so the addresses hold for the
/// device's lifetime. Ordering against teardown is what makes that enough:
/// device release issues `DestroyCommandQueue` while the inner box is still
/// alive, that thunk unregisters the record under the unix registry lock,
/// every unix write into a sink runs under the same lock with a liveness
/// check, and the inner box drops only after the thunk has returned. So a
/// write lands before the record is gone or not at all.
pub struct DisplaySinks {
    /// Wine's retina factor for the layer (2 in retina mode, else 1); `0` until published.
    backing_scale: AtomicU32,
    /// Set by the unix side to ask for a cursor re-apply; taken back to zero here.
    ///
    /// A system tool that borrows the pointer (the screenshot crosshair)
    /// leaves its own cursor on screen, and Wine re-applies its cursor only
    /// on a handle change. The cursor module takes the flag at the next
    /// `WM_SETCURSOR` or `ShowCursor(TRUE)` and answers it with the
    /// null-then-set kick.
    cursor_kick: AtomicU32,
    /// The `ClientSurfaceRequest` bits asked since the last `Present` took them.
    ///
    /// Some Wines hide the client view a device presents through whenever
    /// they flush the window surface, and leave its frame where it was when
    /// the window resizes, counting on winemac's `nextDrawable` override to
    /// put both right; the layer presents without it. The attach sets
    /// `ATTACHED` here too, so the device's first submitted frame tells Wine
    /// it has drawn. The next `Present` that submits a frame takes the bits
    /// and sends `PresentClientSurface`, which has Wine show the surface
    /// again, sized to the client area.
    client_surface_request: AtomicU32,
}

impl DisplaySinks {
    pub const fn new() -> Self {
        Self {
            backing_scale: AtomicU32::new(0),
            cursor_kick: AtomicU32::new(0),
            client_surface_request: AtomicU32::new(0),
        }
    }

    /// The address `AttachMetalLayerParams::client_surface_request_ptr` carries.
    pub fn client_surface_request_ptr(&self) -> u64 {
        (&raw const self.client_surface_request) as u64
    }

    /// Ask the next `Present` to have Wine show the client surface again, for `reason`.
    pub fn request_client_surface(&self, reason: ClientSurfaceRequest) {
        self.client_surface_request
            .fetch_or(reason.bits(), Ordering::Release);
    }

    /// Take the client surface request, empty when nothing asked since the last take.
    ///
    /// One relaxed load on a frame nothing asked on, which is every frame but
    /// the first after an attach and the ones after a resize or a hidden
    /// client view; the swap that takes the bits pairs its `Acquire` with the
    /// writers' `Release`.
    pub fn take_client_surface_request(&self) -> ClientSurfaceRequest {
        if self.client_surface_request.load(Ordering::Relaxed) == 0 {
            return ClientSurfaceRequest::empty();
        }
        ClientSurfaceRequest::from_bits_truncate(
            self.client_surface_request.swap(0, Ordering::AcqRel),
        )
    }

    /// The address `AttachMetalLayerParams::backing_scale_ptr` carries.
    pub fn backing_scale_ptr(&self) -> u64 {
        (&raw const self.backing_scale) as u64
    }

    /// The address `AttachMetalLayerParams::cursor_kick_ptr` carries.
    pub fn cursor_kick_ptr(&self) -> u64 {
        (&raw const self.cursor_kick) as u64
    }

    /// Wine's retina factor as last published, or `None` before attach published one.
    pub fn display_backing_scale(&self) -> Option<u32> {
        match self.backing_scale.load(Ordering::Relaxed) {
            0 => None,
            scale => Some(scale),
        }
    }

    /// Take the pending cursor re-apply request, if the unix side left one.
    ///
    /// `Acquire` pairs with the unix side's `Release` store; the flag is the
    /// whole message, so nothing else is read behind it.
    pub fn take_cursor_kick(&self) -> bool {
        self.cursor_kick.swap(0, Ordering::AcqRel) != 0
    }
}

impl Default for DisplaySinks {
    fn default() -> Self {
        Self::new()
    }
}

/// Cursor state owned by `DeviceInner` as a single field.
///
/// It holds the D3D9 cursor rules and the overlay sync. What the platform
/// realizes is the one [`SystemCursor`] field, reached through
/// [`Self::system`] / [`Self::system_mut`].
pub struct CursorState {
    /// The metal view the device attached, naming its unix-side attachment record.
    ///
    /// Every `SetCursorOverlay` carries it so the overlay follows this
    /// device's window. Null for a headless device, which then sends no
    /// overlay state at all.
    view_handle: MetalHandle<NSViewKind>,
    /// The words the unix side publishes the backing scale and the cursor kick into.
    sinks: Box<DisplaySinks>,
    flags: CursorFlags,
    hash: u64,
    /// Sprite hashes the unix overlay is expected to hold (software mode).
    ///
    /// Mirrors the platform's image cache for the other mode: a hash in here
    /// goes out with no pixels attached. Bounded alike; a stale entry costs
    /// one rejected hash-only call, which is answered by sending the pixels.
    uploaded: BoundedCache<u64, ()>,
    /// Nearest-neighbor upscale factor applied to the cursor bitmap.
    ///
    /// Sourced from the display's `backingScaleFactor` at `CreateDevice`, and
    /// re-sourced whenever the window lands on a display of another one, so a
    /// retina Mac gets a proportionally-sized Win32 cursor (Wine's HCURSOR
    /// path does not participate in the OS's retina upscale). `1` is the
    /// identity fast path.
    scale: u32,
    /// The bitmap the game last handed `SetCursorProperties`.
    ///
    /// Kept so a backing-scale change can rebuild the pointer the game is
    /// already showing. Without it the new factor would only reach the
    /// display on the game's next cursor change, which for a title that sets
    /// its pointer once is never. `None` until the first accepted call.
    source: Option<CursorSource>,
    /// The cursor the platform realizes for this device's window.
    system: SystemCursor,
}

impl CursorState {
    pub fn new(
        hwnd: *mut c_void,
        scale: u32,
        software: bool,
        view_handle: MetalHandle<NSViewKind>,
        sinks: Box<DisplaySinks>,
    ) -> Self {
        // D3D9 starts the cursor hidden (ShowCursor reports FALSE until a
        // cursor image is set and shown).
        let mut flags = CursorFlags::empty();
        if software {
            flags |= CursorFlags::SOFTWARE;
        }
        Self {
            view_handle,
            sinks,
            flags,
            hash: 0,
            uploaded: BoundedCache::new(CURSOR_SPRITE_CACHE_ENTRIES),
            scale: scale.clamp(1, 8),
            source: None,
            system: SystemCursor::new(hwnd),
        }
    }

    /// The cursor the platform realizes, for its window procedure.
    pub const fn system(&self) -> &SystemCursor {
        &self.system
    }

    /// The cursor the platform realizes, for its window procedure.
    pub const fn system_mut(&mut self) -> &mut SystemCursor {
        &mut self.system
    }

    /// Follow the backing scale the unix side last published for this device's window.
    ///
    /// Called once per `Present`: one relaxed load and a compare on an
    /// unchanged value, which is every frame that stays put. The unix side
    /// republishes whenever the window's display changes the factor, so the
    /// cursor upscale follows the window between displays without a thunk.
    pub fn follow_published_scale(&mut self, cursor_scale: mtld3d_core::config::CursorScale) {
        if let Some(backing_scale) = self.sinks.display_backing_scale() {
            let (scale, _origin) =
                crate::direct3d9::resolve_cursor_scale(backing_scale, cursor_scale);
            self.follow_scale(scale);
        }
    }

    /// Take the pending cursor re-apply request, if the unix side left one for this device.
    pub fn take_cursor_kick(&self) -> bool {
        self.sinks.take_cursor_kick()
    }

    /// Have Wine show this device's client surface again, sized to the client area, if asked.
    ///
    /// Called once per `Present`, after the frame was handed to the encoder,
    /// on the API thread, a Wine thread, which win32u's surface calls need. So
    /// a request is taken only by a `Present` that submits a frame, and the
    /// unix side reads the call as the device having drawn. A frame nothing
    /// asked on pays one relaxed load; the device's first frame, and one
    /// after the client view was hidden or the window's client area moved,
    /// sends `PresentClientSurface`.
    pub fn present_client_surface_if_asked(&self) {
        if self.view_handle.is_null() {
            return;
        }
        let reasons = self.sinks.take_client_surface_request();
        if reasons.is_empty() {
            return;
        }
        let mut params = PresentClientSurfaceParams {
            view_handle: self.view_handle,
            reasons,
            pad0: 0,
        };
        let status = unix_call(&mut params);
        if status != 0 {
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET,
                "PresentClientSurface ({reasons:?}) failed (status={status:#x}); the client surface \
                 stays as Wine left it",
            );
        }
    }

    /// Ask the next `Present` to show the client surface again: the client area may have moved.
    pub fn request_client_surface(&self) {
        self.sinks
            .request_client_surface(ClientSurfaceRequest::CLIENT_AREA_CHANGED);
    }

    /// Re-scale the pointer after the window moved to a display of another backing scale.
    ///
    /// A no-op while the factor is unchanged, which is every frame of a
    /// session that stays on one display. On a real change the bitmap the
    /// game is currently showing is rebuilt at the new factor and re-realised
    /// straight away, so the pointer resizes with the window rather than at
    /// the game's next `SetCursorProperties`.
    pub fn follow_scale(&mut self, scale: u32) {
        let scale = scale.clamp(1, 8);
        let previous = self.scale;
        if scale == previous {
            return;
        }
        if let Some(source) = &self.source
            && source.layout().scaled(scale).is_none()
        {
            warn!(target: LOG_TARGET, "cursor scale: bitmap or hotspot overflows at {scale}x");
            return;
        }
        self.scale = scale;
        info!(
            target: LOG_TARGET,
            "cursor scale: {previous}x -> {scale}x (display backing scale changed)",
        );
        if self.software() {
            self.resync_sprite();
        } else {
            self.rebuild_current();
        }
    }

    /// Re-hash and re-ship the current sprite at the current scale (software mode).
    ///
    /// The overlay keys sprites by the same hash as the HCURSOR cache, scale
    /// folded in, so a scale change is a new sprite to it: uploaded once, then
    /// named by hash.
    fn resync_sprite(&mut self) {
        let Some(source) = self.source.as_ref() else {
            return;
        };
        self.hash = hash_cursor(
            source.x_hotspot,
            source.y_hotspot,
            source.width,
            source.height,
            source.pixels.as_ptr(),
            source.width as usize * 4,
            self.scale,
        );
        self.sync_sprite();
    }

    /// Ship the current sprite and visibility to the overlay (software mode).
    ///
    /// Pixels ride along only for a hash the overlay has not seen; every other
    /// call is the hash and the visible flag. Nothing here touches Win32.
    fn sync_sprite(&mut self) {
        let hash = self.hash;
        if hash == 0 {
            return;
        }
        let flags = self.overlay_flags();
        let Some(source) = self.source.as_ref() else {
            mtld3d_shared::log_once_warn!(target: LOG_TARGET, "cursor: no source for sprite retry");
            return;
        };
        let accepted = reconcile_upload(self.uploaded.contains(&hash), |with_pixels| {
            let sprite = with_pixels.then(|| upscale_sprite(source, self.scale));
            send_overlay_state(self.view_handle, hash, flags, sprite.as_ref())
        });
        if accepted {
            self.uploaded.insert(hash, ());
        } else {
            self.uploaded.remove(&hash);
        }
    }

    /// Ship the current visibility to the unix side.
    ///
    /// Software mode names the sprite too; the hardware path sends visibility
    /// alone, which the unix side's pointer watch needs to know whatever
    /// draws the cursor.
    fn push_overlay_state(&mut self) {
        if self.software() && self.source.is_some() {
            self.sync_sprite();
        } else {
            // Native visibility also belongs to games that draw their own cursor
            // and never supply a D3D sprite.
            send_overlay_state(
                self.view_handle,
                0,
                self.overlay_flags() | CursorOverlayFlags::HARDWARE,
                None,
            );
        }
    }

    /// The flags word `SetCursorOverlay` carries: the visibility once a cursor surface is set.
    const fn overlay_flags(&self) -> CursorOverlayFlags {
        let mut flags = if self.visible() && self.flags.contains(CursorFlags::SET) {
            CursorOverlayFlags::VISIBLE
        } else {
            CursorOverlayFlags::empty()
        };
        if self.flags.contains(CursorFlags::NATIVE_HIDDEN) {
            flags = flags.union(CursorOverlayFlags::NATIVE_HIDDEN);
        }
        flags
    }

    /// Publish the foreground window's native hide, even without a D3D cursor surface.
    pub fn sync_native_visibility(&mut self) {
        let Some(hidden) = self.system.native_hidden() else {
            return;
        };
        if hidden != self.flags.contains(CursorFlags::NATIVE_HIDDEN) {
            self.flags.set(CursorFlags::NATIVE_HIDDEN, hidden);
            debug!(
                target: LOG_TARGET,
                "native cursor hidden={hidden} hwnd={:p}",
                self.system.window(),
            );
            self.push_overlay_state();
        }
    }

    const fn software(&self) -> bool {
        self.flags.contains(CursorFlags::SOFTWARE)
    }

    /// Rebuild and re-realise the current pointer at the current scale.
    ///
    /// Keyed through the same cache as `SetCursorProperties`, whose key folds
    /// the scale in, so moving back to the first display reuses the bitmap
    /// built for it instead of building a third.
    fn rebuild_current(&mut self) {
        let Some(source) = self.source.as_ref() else {
            return;
        };
        let pitch = source.width as usize * 4;
        let hash = hash_cursor(
            source.x_hotspot,
            source.y_hotspot,
            source.width,
            source.height,
            source.pixels.as_ptr(),
            pitch,
            self.scale,
        );
        let visible = self.visible();
        let rebuilt =
            self.system
                .rebuild(hash, &source.layout(), &source.pixels, self.scale, visible);
        if !rebuilt {
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET,
                "cursor: build_hcursor failed while following a backing-scale change; \
                 the pointer stays at its previous size",
            );
            return;
        }
        self.hash = hash;
    }

    /// Fold one `Present` into the platform's frame-hitch attribution.
    pub fn note_present(&mut self) {
        self.system.note_present();
    }

    /// Whether the game's `ShowCursor` last asked for the cursor shown.
    pub const fn visible(&self) -> bool {
        self.flags.contains(CursorFlags::VISIBLE)
    }

    const fn set_visible(&mut self, on: bool) {
        if on {
            self.flags = self.flags.union(CursorFlags::VISIBLE);
        } else {
            self.flags = self.flags.difference(CursorFlags::VISIBLE);
        }
    }

    const fn hide_suppressed_in_resize(&self) -> bool {
        self.flags.contains(CursorFlags::HIDE_SUPPRESSED_IN_RESIZE)
    }

    const fn set_hide_suppressed_in_resize(&mut self, on: bool) {
        if on {
            self.flags = self.flags.union(CursorFlags::HIDE_SUPPRESSED_IN_RESIZE);
        } else {
            self.flags = self
                .flags
                .difference(CursorFlags::HIDE_SUPPRESSED_IN_RESIZE);
        }
    }

    /// Answer a `WM_SIZE` of `size_type` before the game's window procedure sees it.
    ///
    /// Arms the hide suppression when `mtld3d_core::cursor::resize_suppresses_hide`
    /// says so and returns whether it did; the caller holds the window's
    /// message dispatch, whose end disarms it ([`Self::end_resize_dispatch`]).
    /// The platform re-asserts its cursor ([`SystemCursor::note_resize`]).
    pub fn enter_resize(&mut self, size_type: usize) -> bool {
        let suppress = mtld3d_core::cursor::resize_suppresses_hide(size_type, self.visible());
        if suppress {
            self.set_hide_suppressed_in_resize(true);
        }
        self.system.note_resize(self.visible() || self.software());
        self.push_overlay_state();
        suppress
    }

    /// Disarm the hide suppression: the message dispatch a resize arrived in has returned.
    ///
    /// Returns whether it was still armed; a `ShowCursor(TRUE)` inside the
    /// dispatch may have disarmed it already.
    pub const fn end_resize_dispatch(&mut self) -> bool {
        let armed = self.hide_suppressed_in_resize();
        self.set_hide_suppressed_in_resize(false);
        armed
    }

    /// Route this window's messages to `dev_ptr`; see [`SystemCursor::attach`].
    pub fn install_subclass(&self, dev_ptr: *mut DeviceInner) {
        self.system.attach(dev_ptr, self.scale);
    }

    /// Stop routing this window's messages to `dev_ptr`; see [`SystemCursor::detach`].
    ///
    /// Call from the device-release path before freeing `DeviceInner`, and
    /// before a retarget moves the device to another window.
    pub fn uninstall_subclass(&self, dev_ptr: *mut DeviceInner) {
        self.system.detach(dev_ptr);
    }

    /// Destroy every cursor image the platform built for this device.
    ///
    /// Call from the device-release path after `uninstall_subclass`, so no
    /// window message realizes one of them again. No cursor surface is set
    /// afterwards.
    pub fn destroy_handles(&mut self) {
        let keep_hidden = self.software() && !self.visible();
        self.flags.remove(CursorFlags::SET);
        self.system.destroy_images(keep_hidden);
    }

    /// Move the subclass and the overlay's view onto another device window.
    ///
    /// A `Reset` naming a different `hDeviceWindow` moves the presentation
    /// surface onto that window, and the messages the cursor rides move with
    /// it. The device leaves the old window before it joins the new one, so a
    /// message arriving between the two finds no device rather than the wrong
    /// one, and a new window another device already subclasses is joined
    /// rather than hooked a second time. The realized `HCURSOR`, both sprite
    /// caches and the visibility latches carry over: `Reset` re-specifies the
    /// swap chain, not the cursor the application set. The resize hide
    /// suppression does not: it belongs to a message dispatch on the window
    /// the device leaves, whose end would no longer find the device there.
    ///
    /// `view_handle` names the view the new window's attach produced, null
    /// when it produced none. The overlay state is re-sent against it because
    /// the record the previous state named is gone; the sprites themselves are
    /// content-addressed on the unix side and outlive the detach, so the
    /// uploaded set stays valid.
    pub fn retarget(
        &mut self,
        hwnd: *mut c_void,
        view_handle: MetalHandle<NSViewKind>,
        dev_ptr: *mut DeviceInner,
    ) {
        self.uninstall_subclass(dev_ptr);
        self.set_hide_suppressed_in_resize(false);
        self.system.set_window(hwnd);
        self.view_handle = view_handle;
        self.install_subclass(dev_ptr);
        self.push_overlay_state();
    }

    /// The words the unix side publishes into for this device.
    ///
    /// The attach that a device-window retarget issues hands their addresses
    /// over again, and they are the same box: the device owns it for its
    /// lifetime, so the address the new attachment record holds is the one
    /// the cursor still reads.
    pub const fn sinks(&self) -> &DisplaySinks {
        &self.sinks
    }

    /// The upscale factor a cursor bitmap is checked and built at.
    pub const fn scale(&self) -> u32 {
        self.scale
    }

    /// Pick the image for a locked `SetCursorProperties` bitmap, while the surface is locked.
    ///
    /// `src` and `pitch` are the locked rows `layout` describes. `None` when
    /// the cursor cannot be built, which the caller answers with
    /// `D3DERR_INVALIDCALL` after unlocking.
    ///
    /// # Safety
    ///
    /// `src` points at `layout.height` rows `pitch` bytes apart, each holding
    /// `layout.width` 4-byte pixels, readable for the duration of the call.
    pub unsafe fn select_bitmap(
        &mut self,
        layout: &BitmapLayout,
        src: *const u8,
        pitch: usize,
    ) -> Option<SelectedBitmap> {
        let &BitmapLayout {
            width,
            height,
            x_hotspot,
            y_hotspot,
        } = layout;
        let hash = hash_cursor(x_hotspot, y_hotspot, width, height, src, pitch, self.scale);
        let prev_hash = self.hash;
        let (image, outcome) = if self.software() {
            // The overlay draws the bitmap; the system cursor only has to be blank.
            // Building the blank can fail the same way `build_hcursor` can, and
            // is reported the same way.
            let Some(blank) = self.system.blank() else {
                warn!(
                    target: LOG_TARGET,
                    "reject SetCursorProperties: blank HCURSOR failed (hash={hash:#018x} {width}x{height})",
                );
                return None;
            };
            let outcome = if hash == prev_hash {
                "unchanged"
            } else if self.uploaded.contains(&hash) {
                "sprite-known"
            } else {
                "sprite-upload"
            };
            (blank, outcome)
        } else {
            // SAFETY: the caller's contract on `src`, `pitch` and `layout` is
            // the one `select` states.
            unsafe {
                self.system
                    .select(hash, hash == prev_hash, layout, src, pitch, self.scale)
            }?
        };

        // Keep the pixels: the backing scale can change under a running session
        // when the window moves to another display, and rebuilding the pointer at
        // the new factor needs the bitmap the caller's unlock hands back to the
        // guest. Only on a new bitmap, so a game re-setting the same cursor every
        // frame pays nothing.
        if hash != prev_hash {
            self.source = Some(CursorSource {
                width,
                height,
                x_hotspot,
                y_hotspot,
                pixels: tight_copy(width, height, pitch, src),
            });
        }
        Some(SelectedBitmap {
            hash,
            prev_hash,
            image,
            outcome,
        })
    }

    /// Make the bitmap [`Self::select_bitmap`] picked the current cursor, after the unlock.
    pub fn realize_bitmap(
        &mut self,
        selected: &SelectedBitmap,
        layout: &BitmapLayout,
        desc: &D3DSURFACE_DESC,
    ) {
        let &SelectedBitmap {
            hash,
            prev_hash,
            ref image,
            outcome,
        } = selected;
        let &BitmapLayout {
            width,
            height,
            x_hotspot,
            y_hotspot,
        } = layout;
        self.hash = hash;
        self.system.set_image(image);
        self.flags.insert(CursorFlags::SET);
        let visible = self.visible();
        // Software mode: the overlay gets the new sprite (or just its hash) and the
        // current visibility. The blank Win32 cursor is unchanged by a cursor
        // change and already in place from the last show, so it is realized here
        // only when no cursor was set before.
        self.push_overlay_state();
        // Realize only while shown (D3D9 sets the Win32 cursor only when the cursor is visible).
        // While hidden the game owns the win32 cursor — pushing null here clobbers
        // the cursor the game's own wndproc set (WoW's login screen never calls
        // ShowCursor(TRUE); its glove is the game's own cursor).
        // `set_cursor_us` is 0 when nothing was realized.
        let set_cursor_us = if visible && (!self.software() || prev_hash == 0) {
            self.system.apply(image)
        } else {
            0
        };
        debug!(
            target: LOG_TARGET,
            "SetCursorProperties: {width}x{height} fmt={} pool={} hotspot=({x_hotspot},{y_hotspot}) hash={hash:#018x} outcome={outcome} handle={handle:p} visible={visible} set_cursor_us={set_cursor_us} cache_entries={} tid={}",
            desc.format, desc.pool, self.system.cache_len(), current_thread_id(),
            handle = *image,
        );
    }

    /// `SetCursorPosition`: move the pointer to `(x, y)` unless it is already there.
    pub fn set_position(&mut self, x: i32, y: i32) {
        self.system.set_position(x, y);
    }

    /// `ShowCursor`: set the visibility to `show`, returning the previous one.
    pub fn show(&mut self, show: i32) -> i32 {
        let prev = self.visible();
        let next = show != 0;
        if !next && self.hide_suppressed_in_resize() {
            debug!(
                target: LOG_TARGET,
                "ShowCursor(show=0) suppressed: issued inside the dispatch a resize arrived in tid={}",
                current_thread_id(),
            );
            self.push_overlay_state();
            return i32::from(prev);
        }
        // D3D9 changes the cursor visibility only once a cursor image has been set
        // via SetCursorProperties; with no cursor surface,
        // ShowCursor is a pure read of the (unchanged) previous visibility.
        if !self.flags.contains(CursorFlags::SET) {
            trace!(
                target: LOG_TARGET,
                "ShowCursor(show={show}) → prev={prev} (no cursor surface set; no-op)",
            );
            return i32::from(prev);
        }
        if next {
            self.set_hide_suppressed_in_resize(false);
            // A pointer that came back from another process while the cursor was
            // hidden: the game's cursor was not on screen to kick then, so the
            // show pushes null first and Wine re-applies on the handle change.
            if self.take_cursor_kick() {
                SystemCursor::kick();
            }
        }
        self.set_visible(next);
        // Realize on EVERY call, not only on transitions: a transition-gated
        // SetCursor leaves wine's cursor state stale whenever a latch-suppressed
        // hide ate the transition — the game then believes the cursor visible
        // while wine still holds the hidden state until its next full hide/show
        // cycle.
        //
        // What this does NOT recover from is a pointer image replaced *below*
        // wine: wine's `set_cursor` server request only notifies its display
        // driver when the handle changes (`prev_cursor != new_cursor`), so
        // re-pushing the handle wine already holds is dropped before it reaches
        // the driver. Only a different handle, the null-then-handle kick in the
        // window procedure's WM_SETCURSOR branch, or the pointer re-entering the
        // window gets through.
        //
        // Software mode keeps the Win32 cursor blank in both directions: a show
        // re-asserts the blank (the same recovery role, the game may have set its
        // own cursor meanwhile), a hide pushes nothing, so the WindowServer cursor
        // plane never toggles on our account. The overlay gets the visibility.
        let handle = self.system.image(next || self.software());
        let set_cursor_us = if next || !self.software() {
            self.system.apply(&handle)
        } else {
            0
        };
        // The hardware path sends visibility on transitions only, beside the
        // SetCursor it already makes; software mode sends every call, coalesced
        // on the unix side.
        if self.software() || prev != next {
            self.push_overlay_state();
        }
        if prev != next {
            self.system.note_transition(next);
        }
        if prev == next {
            trace!(
                target: LOG_TARGET,
                "ShowCursor(show={show}) → prev={prev} handle={handle:p} set_cursor_us={set_cursor_us} tid={} (re-assert)",
                current_thread_id(),
            );
        } else {
            debug!(
                target: LOG_TARGET,
                "ShowCursor(show={show}) → prev={prev} next={next} handle={handle:p} set_cursor_us={set_cursor_us} tid={} (transition)",
                current_thread_id(),
            );
        }
        i32::from(prev)
    }
}

/// The image [`CursorState::select_bitmap`] picked, handed to [`CursorState::realize_bitmap`].
pub struct SelectedBitmap {
    hash: u64,
    prev_hash: u64,
    image: CursorImage,
    outcome: &'static str,
}

/// Microseconds since `started`, saturating.
pub fn elapsed_us(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX)
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
pub fn scale_cursor_pixels(
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

// ── Vtable entry points ──

pub extern "system" fn device_set_cursor_properties(
    this: *mut c_void,
    x_hotspot: u32,
    y_hotspot: u32,
    cursor_bitmap: *mut c_void,
) -> i32 {
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::Misc);
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
    let layout = BitmapLayout {
        width,
        height,
        x_hotspot,
        y_hotspot,
    };
    let mode = crate::direct3d9::current_adapter_display_mode();
    if !layout.valid(
        desc.format,
        (mode.width, mode.height),
        dev.cursor_mut().scale(),
    ) {
        warn!(
            target: LOG_TARGET,
            "reject SetCursorProperties: invalid format, extent or scaled bitmap/hotspot \
             (format={} {width}x{height} hotspot={x_hotspot},{y_hotspot})",
            desc.format,
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

    let Some(pitch) = layout.row_pitch(locked.bits as usize, locked.pitch) else {
        warn!(target: LOG_TARGET, "reject SetCursorProperties: invalid locked pointer or pitch");
        // SAFETY: this surface was successfully locked above; no pixels were read.
        unsafe { (surf_vtbl.unlock_rect)(cursor_bitmap) };
        return D3DERR_INVALIDCALL;
    };
    let src = locked.bits as *const u8;
    let cur = dev.cursor_mut();
    // SAFETY: `src` is the surface `LockRect` locked above, whose extent
    // `layout` holds, and `row_pitch` checked `pitch` against it; the lock
    // holds until the unlock below.
    let selected = unsafe { cur.select_bitmap(&layout, src, pitch) };
    // SAFETY: calling the just-loaded `unlock_rect` thunk through
    // `surf_vtbl`; paired with the `lock_rect` call above.
    unsafe { (surf_vtbl.unlock_rect)(cursor_bitmap) };
    let Some(selected) = selected else {
        return D3DERR_INVALIDCALL;
    };
    cur.realize_bitmap(&selected, &layout, &desc);
    D3D_OK
}

pub extern "system" fn device_set_cursor_position(this: *mut c_void, x: i32, y: i32, _flags: u32) {
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return;
    };
    obj.inner().cursor_mut().set_position(x, y);
}

pub extern "system" fn device_show_cursor(this: *mut c_void, show: i32) -> i32 {
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    obj.inner().cursor_mut().show(show)
}

/// Minimal layout for reading the surface's vtable pointer.
///
/// Avoids pulling the full `Direct3DSurface9` type into this module.
#[repr(C)]
struct SurfaceHead {
    vtbl: *const IDirect3DSurface9Vtbl,
}

bitflags::bitflags! {
    /// Packed boolean state for `CursorState`.
    ///
    /// Packing the flags into one byte means a `match` against them fits in
    /// one comparison and the surrounding struct's tail padding tightens.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    struct CursorFlags: u8 {
        /// Game-requested visibility (ShowCursor / ShowCursor toggles).
        const VISIBLE = 1 << 0;
        /// A `SetCursorProperties` was accepted, and no release destroyed its image since.
        ///
        /// `ShowCursor` changes the visibility only while it is set, and the
        /// overlay shows a cursor only then.
        const SET = 1 << 1;
        /// Set by a resize `WM_SIZE` until the message dispatch it arrived in returns.
        ///
        /// A game-issued `ShowCursor(FALSE)` is dropped while it is set, and
        /// a `ShowCursor(TRUE)` clears it. Armed only while the cursor is
        /// shown, so it never shows a cursor the game hid; the rule is
        /// `mtld3d_core::cursor::resize_suppresses_hide`, and the dispatch is
        /// counted by `mtld3d_core::window_subclass::WindowSubclasses`.
        const HIDE_SUPPRESSED_IN_RESIZE = 1 << 2;
        /// The unix-side overlay window draws the cursor.
        ///
        /// Fixed for the device's lifetime. The platform then realizes a
        /// blank cursor, and every cursor change and show/hide also goes out
        /// through `SetCursorOverlay`.
        const SOFTWARE = 1 << 3;
        /// Latest foreground Win32 cursor state, independent of the D3D cursor image.
        const NATIVE_HIDDEN = 1 << 4;
    }
}

/// A cursor bitmap in the shape `build_hcursor` consumes.
///
/// `pixels` is a tight BGRA copy of the locked surface, so the row pitch is
/// `width * 4` and the buffer outlives the lock it came from.
struct CursorSource {
    width: u32,
    height: u32,
    x_hotspot: u32,
    y_hotspot: u32,
    pixels: Vec<u8>,
}

impl CursorSource {
    const fn layout(&self) -> BitmapLayout {
        BitmapLayout {
            width: self.width,
            height: self.height,
            x_hotspot: self.x_hotspot,
            y_hotspot: self.y_hotspot,
        }
    }
}

/// Content-hash over hotspot + dimensions + pixel bytes.
///
/// Used as the cursor cache key inside `SetCursorProperties`. xxh3 is the
/// content hash for byte buffers throughout the tree (see `ProgramId`); the
/// value is the whole identity of the cursor, so it needs real avalanche, not
/// a map hasher.
///
/// The upscale factor is part of that identity: the same bitmap on displays
/// of different backing scale produces different HCURSORs, and one key for
/// both would serve the wrong-sized one out of the cache.
fn hash_cursor(
    x_hotspot: u32,
    y_hotspot: u32,
    width: u32,
    height: u32,
    src: *const u8,
    pitch: usize,
    scale: u32,
) -> u64 {
    let mut h = Xxh3::new();
    h.write_u32(x_hotspot);
    h.write_u32(y_hotspot);
    h.write_u32(width);
    h.write_u32(height);
    h.write_u32(scale);
    let row_bytes = (width as usize) * 4;
    for y in 0..height as usize {
        // SAFETY: caller-validated `src + y*pitch` stays within the bitmap.
        let row_ptr = unsafe { src.add(y * pitch) };
        // SAFETY: `row_ptr..row_ptr + row_bytes` lies in the same allocation.
        let row = unsafe { core::slice::from_raw_parts(row_ptr, row_bytes) };
        h.write(row);
    }
    // `0` is the wire's "no sprite"; fold the one-in-2^64 collision away.
    match h.finish() {
        0 => 1,
        hash => hash,
    }
}

/// Copy a locked BGRA cursor surface into a tight `width * 4`-pitch buffer.
///
/// The caller validates that `src` covers `height` rows of at least
/// `width * 4` readable bytes, `pitch` apart, which is what
/// `D3DLOCKED_RECT` describes for the locked region.
fn tight_copy(width: u32, height: u32, pitch: usize, src: *const u8) -> Vec<u8> {
    let row_bytes = width as usize * 4;
    let mut out = vec![0u8; row_bytes * height as usize];
    for y in 0..height as usize {
        // SAFETY: caller-validated `src + y*pitch` stays within the bitmap.
        let row_ptr = unsafe { src.add(y * pitch) };
        // SAFETY: `row_ptr..row_ptr + row_bytes` lies in the same allocation.
        let row = unsafe { core::slice::from_raw_parts(row_ptr, row_bytes) };
        out[y * row_bytes..(y + 1) * row_bytes].copy_from_slice(row);
    }
    out
}

/// A cursor bitmap in the shape the overlay window consumes.
///
/// Tight BGRA rows at `scale` pixels per point, hotspot in those pixels.
struct SpriteUpload {
    width: u32,
    height: u32,
    x_hotspot: u32,
    y_hotspot: u32,
    scale: u32,
    pixels: Vec<u8>,
}

/// Upscale the game's bitmap for the overlay the way `build_hcursor` does for the HCURSOR.
///
/// Same dispatch (`scale_cursor_pixels`), same hotspot rule, so the software
/// cursor is the hardware cursor's sprite drawn by a different compositor.
fn upscale_sprite(source: &CursorSource, scale: u32) -> SpriteUpload {
    let scale = scale.clamp(1, 8);
    let scaled = source
        .layout()
        .scaled(scale)
        .expect("validated cursor scale");
    let src_pixels = u8_to_u32_vec(&source.pixels);
    // Same rule as the hardware path: a bitmap with no alpha anywhere is an
    // opaque cursor (there its AND mask keeps every pixel); the overlay
    // blends premultiplied, so those pixels get an opaque alpha instead.
    let any_alpha = src_pixels.iter().any(|&px| (px >> 24) != 0);
    let (sw, sh, mut pixels, path) = scale_cursor_pixels(
        scale as usize,
        source.width as usize,
        source.height as usize,
        src_pixels,
    );
    if !any_alpha {
        for px in &mut pixels {
            *px |= 0xFF00_0000;
        }
    }
    trace!(
        target: LOG_TARGET,
        "upscale_sprite: {}x{} → {sw}x{sh} path={path} any_alpha={any_alpha}",
        source.width, source.height,
    );
    SpriteUpload {
        width: u32::try_from(sw).expect("upscaled cursor width fits u32"),
        height: u32::try_from(sh).expect("upscaled cursor height fits u32"),
        x_hotspot: scaled.x_hotspot,
        y_hotspot: scaled.y_hotspot,
        scale,
        pixels: u32_to_u8_vec(&pixels),
    }
}

/// One `SetCursorOverlay` call: the wanted sprite and visibility, pixels attached or not.
///
/// Returns whether the unix side accepted it. The pixel buffer only has to
/// outlive the call: the unix side copies what it keeps. A device with no
/// metal view (headless) has no attachment record for the overlay to follow,
/// so nothing is sent and the call reports as not accepted.
fn send_overlay_state(
    view_handle: MetalHandle<NSViewKind>,
    hash: u64,
    flags: CursorOverlayFlags,
    sprite: Option<&SpriteUpload>,
) -> bool {
    if view_handle.is_null() {
        mtld3d_shared::log_once_info!(
            target: LOG_TARGET,
            "SetCursorOverlay: no metal view attached (headless device), overlay state not sent",
        );
        return false;
    }
    let started = Instant::now();
    let mut params = SetCursorOverlayParams {
        hash,
        pixels_ptr: 0,
        pixels_len: 0,
        width: 0,
        height: 0,
        x_hotspot: 0,
        y_hotspot: 0,
        scale: 0,
        flags,
        pad0: 0,
        view_handle,
    };
    if let Some(s) = sprite {
        params.pixels_ptr = s.pixels.as_ptr() as u64;
        params.pixels_len =
            u32::try_from(s.pixels.len()).expect("a cursor sprite is far below 4 GiB");
        params.width = s.width;
        params.height = s.height;
        params.x_hotspot = s.x_hotspot;
        params.y_hotspot = s.y_hotspot;
        params.scale = s.scale;
    }
    let status = unix_call(&mut params);
    let us = elapsed_us(started);
    if status != 0 {
        warn!(
            target: LOG_TARGET,
            "SetCursorOverlay: hash={hash:#018x} flags={flags:?} pixels={} rejected (status={status:#x}) us={us}",
            sprite.is_some(),
        );
        return false;
    }
    debug!(
        target: LOG_TARGET,
        "SetCursorOverlay: hash={hash:#018x} flags={flags:?} pixels={} us={us}",
        sprite.is_some(),
    );
    true
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
    for chunk in src.as_chunks::<4>().0 {
        out.push(u32::from_ne_bytes(*chunk));
    }
    out
}
