//! What `d3d9` needs from the system it runs on, behind one interface.
//!
//! Wine loads the PE build as `d3d9.dll` and gets the Win32 implementation; a
//! native build links the unix side into the same library and gets the native
//! one. Both define the same items, and this declaration is the only place the
//! crate tells the two builds apart.

#[cfg_attr(windows, path = "platform/win32.rs")]
#[cfg_attr(not(windows), path = "platform/native.rs")]
mod imp;
pub use imp::{
    ADDRESS_SPACE_WALKS, SurfaceDc, acquire_logger, adapter_luid, address_space_map,
    apply_device_control_word, caller_holds_loader_lock, capture_hotkey_pressed, client_size,
    create_surface_dc, current_display_mode, destroy_surface_dc, dispatch_unix_call,
    enumerate_display_modes, exclusive_fullscreen, free_space, is_device_window, is_maximized,
    main_image_version_blob, note_create_device, primary_monitor, release_logger, surface_dcs,
    unix_path, warn_if_dpi_scaled,
};

/// What the system realizes of the device's cursor.
pub mod cursor {
    use mtld3d_core::cursor::BitmapLayout;

    pub use super::imp::cursor::{CursorImage, SystemCursor, current_thread_id};
    use crate::device::DeviceInner;

    /// The operations shared code asks of the system cursor, which each platform defines.
    pub trait SystemCursorOps {
        /// Whether the foreground window's own cursor is hidden; `None` when unread.
        fn native_hidden(&self) -> Option<bool>;
        /// The blank cursor software mode realizes; `None` when it cannot be built.
        fn blank(&mut self) -> Option<CursorImage>;
        /// Pick the image for the game's bitmap at `scale`, and its log outcome.
        ///
        /// # Safety
        ///
        /// `src` points at `layout.height` rows `pitch` bytes apart, each holding
        /// `layout.width` 4-byte pixels, readable for the duration of the call.
        unsafe fn select(
            &mut self,
            hash: u64,
            unchanged: bool,
            layout: &BitmapLayout,
            src: *const u8,
            pitch: usize,
            scale: u32,
        ) -> Option<(CursorImage, &'static str)>;
        /// Rebuild the current pointer from tight `pixels`; `false` keeps the current image.
        fn rebuild(
            &mut self,
            hash: u64,
            layout: &BitmapLayout,
            pixels: &[u8],
            scale: u32,
            visible: bool,
        ) -> bool;
        /// Make `image` the current one, without realizing it.
        fn set_image(&mut self, image: &CursorImage);
        /// The current image when `shown`, no cursor otherwise.
        fn image(&self, shown: bool) -> CursorImage;
        /// Realize `image`, returning the microseconds the call took.
        fn apply(&mut self, image: &CursorImage) -> u64;
        /// Realize no cursor, so the next realize reaches the display as a change.
        fn kick();
        /// Re-assert the cursor after a resize, when `reassert` says the device owns it.
        fn note_resize(&mut self, reassert: bool);
        /// The number of images the platform caches.
        fn cache_len(&self) -> usize;
        /// Record a `ShowCursor` transition to `shown`.
        fn note_transition(&mut self, shown: bool);
        /// Fold one `Present` into the platform's diagnostics.
        fn note_present(&mut self);
        /// Start following the window's events for `dev_ptr`.
        fn attach(&self, dev_ptr: *mut DeviceInner, scale: u32);
        /// Stop following the window's events for `dev_ptr`.
        fn detach(&self, dev_ptr: *mut DeviceInner);
        /// Destroy every image this device built.
        fn destroy_images(&mut self, keep_hidden: bool);
        /// `SetCursorPosition`: move the pointer to `(x, y)`.
        fn set_position(&mut self, x: i32, y: i32);
    }
}

/// A fullscreen device's window and display mode.
pub mod fullscreen {
    pub use super::imp::fullscreen::{
        SavedWindow, activate_focus_window, enter, leave, try_leave, update,
    };
}
