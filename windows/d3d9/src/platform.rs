//! What `d3d9` needs from the system it runs on, behind one interface.
//!
//! Wine loads the PE build as `d3d9.dll` and gets the Win32 implementation; a
//! native build links the unix side into the same library and gets the native
//! one. Both define the same items, and this declaration is the only place the
//! crate tells the two builds apart.

#[cfg_attr(windows, path = "platform/win32.rs")]
#[cfg_attr(not(windows), path = "platform/native.rs")]
mod imp;
pub use imp::*;
