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
    ADDRESS_SPACE_WALKS, SurfaceDc, SurfaceDcSource, acquire_logger, adapter_luid,
    address_space_map, apply_device_control_word, caller_holds_loader_lock, capture_hotkey_pressed,
    client_size, create_surface_dc, current_display_mode, cursor, destroy_surface_dc,
    dispatch_unix_call, enumerate_display_modes, free_space, fullscreen, is_maximized,
    main_image_version_blob, note_create_device, primary_monitor, release_logger, system_dpi,
    unix_path, warn_if_dpi_scaled,
};
