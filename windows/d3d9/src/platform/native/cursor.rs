//! The system half of a device's cursor in the native build, which realizes nothing.
//!
//! The host owns its cursor: no image is built and the pointer does not move.
//! The D3D9 cursor state stays in the shared `CursorState`.

use core::{ffi::c_void, fmt};

use log::debug;
use mtld3d_core::cursor::BitmapLayout;
use mtld3d_shared::{MetalHandle, mtl::ClientSurfaceRequest, mtl_handle::NSViewKind};

use crate::{
    device::{
        DeviceInner,
        cursor::{DisplaySinks, LOG_TARGET},
    },
    platform::cursor::SystemCursorOps,
};

/// A cursor image; the native build builds none.
pub struct CursorImage;

impl fmt::Pointer for CursorImage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Pointer::fmt(&core::ptr::null::<c_void>(), f)
    }
}

/// The calling thread's id for the cursor log lines; the native build logs none.
pub const fn current_thread_id() -> u32 {
    0
}

/// The view a device's cursor state follows.
pub struct SystemCursor {
    view: *mut c_void,
}

impl SystemCursor {
    /// No overlay window: the host draws its own cursor.
    pub const HAS_OVERLAY: bool = false;

    /// The system cursor of a device on `view`.
    pub const fn new(view: *mut c_void) -> Self {
        Self { view }
    }

    /// The view the device presents into.
    pub const fn window(&self) -> *mut c_void {
        self.view
    }

    /// Follow another view.
    pub const fn set_window(&mut self, view: *mut c_void) {
        self.view = view;
    }

    /// The view's size in pixels the request carried, when it says the size may have changed.
    ///
    /// The unix side measured it on the main thread and put it in the
    /// request, so `Present` asks the main thread nothing.
    pub fn present_client_surface(
        view_handle: MetalHandle<NSViewKind>,
        reasons: ClientSurfaceRequest,
        pixels: Option<(u32, u32)>,
    ) -> Option<(u32, u32)> {
        if !reasons.intersects(ClientSurfaceRequest::CLIENT_AREA_CHANGED) {
            return None;
        }
        debug!(target: LOG_TARGET, "view {view_handle:?} re-framed ({reasons:?}): {pixels:?} pixels");
        pixels
    }

    /// Drop a pending resize: the view that asked for it is no longer the device's.
    pub fn drop_pending_resize(sinks: &DisplaySinks) {
        sinks.client_surface_request().fetch_and(
            !ClientSurfaceRequest::RESIZE,
            core::sync::atomic::Ordering::AcqRel,
        );
    }
}

impl SystemCursorOps for SystemCursor {
    /// The host's cursor visibility is its own: nothing to publish.
    fn native_hidden(&self) -> Option<bool> {
        None
    }

    /// No blank cursor is needed; software mode is off without an overlay.
    fn blank(&mut self) -> Option<CursorImage> {
        Some(CursorImage)
    }

    /// Accept a bitmap whose scaled size fits; its pixels are not read.
    ///
    /// # Safety
    ///
    /// None: the pointer is not read. The signature is the platform interface's.
    unsafe fn select(
        &mut self,
        _hash: u64,
        _unchanged: bool,
        layout: &BitmapLayout,
        _src: *const u8,
        _pitch: usize,
        scale: u32,
    ) -> Option<(CursorImage, &'static str)> {
        layout.scaled(scale).map(|_| (CursorImage, "state-only"))
    }

    /// Nothing to rebuild.
    fn rebuild(
        &mut self,
        _hash: u64,
        _layout: &BitmapLayout,
        _pixels: &[u8],
        _scale: u32,
        _visible: bool,
    ) -> bool {
        true
    }

    /// No image to keep.
    fn set_image(&mut self, _image: &CursorImage) {}

    /// No image to hand out.
    fn image(&self, _shown: bool) -> CursorImage {
        CursorImage
    }

    /// Nothing is realized, in no time.
    fn apply(&mut self, _image: &CursorImage) -> u64 {
        0
    }

    /// Nothing to kick.
    fn kick() {}

    /// No cursor to re-assert.
    fn note_resize(&mut self, _reassert: bool) {}

    /// No image cache.
    fn cache_len(&self) -> usize {
        0
    }

    /// No hitch probe.
    fn note_transition(&mut self, _shown: bool) {}

    /// No hitch probe.
    fn note_present(&mut self) {}

    /// No window procedure: the back buffer follows the view at present.
    fn attach(&self, _dev_ptr: *mut DeviceInner, _scale: u32) {
        mtld3d_shared::log_once_info!(
            target: LOG_TARGET,
            "view {:p}: no window procedure in the native build; the back buffer follows the \
             view's size at present",
            self.view,
        );
    }

    /// No window procedure to restore.
    fn detach(&self, _dev_ptr: *mut DeviceInner) {
        debug!(target: LOG_TARGET, "view {:p}: the device leaves it", self.view);
    }

    /// No image was built.
    fn destroy_images(&mut self, _keep_hidden: bool) {}

    /// The pointer is the host's; the position is not applied.
    fn set_position(&mut self, x: i32, y: i32) {
        mtld3d_shared::log_once_info!(
            target: LOG_TARGET,
            "SetCursorPosition ({x}, {y}) on view {:p}: the native build leaves the host's \
             pointer alone",
            self.view,
        );
    }
}
