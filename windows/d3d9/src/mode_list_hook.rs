//! The game's view of user32's display-mode list.
//!
//! A game that builds its resolution menu from `EnumDisplaySettings` rather
//! than `EnumAdapterModes` (`WoW` 1.12 walks every mode the primary display
//! reports, keeps those of at least 640x480 at 16 bits or more and lists each
//! size once) sees the whole list a fullscreen device may set, which under
//! Wine's `EmulateModeset` is forty-odd sizes and overflows a menu built for
//! a driver's short list. The `EnumDisplaySettings*` imports of the process's
//! main module are redirected here at `DllMain`: an index into the list maps
//! onto the n-th mode whose size `EnumAdapterModes` serves, so both menu
//! paths show one bounded list ([`served_mode_sizes`]). `ENUM_CURRENT_SETTINGS`
//! and `ENUM_REGISTRY_SETTINGS` pass through untouched, user32's own list is
//! never changed (a mode-set validates against all of it), and this DLL's own
//! imports are not patched, so `fullscreen::enumerate_display_modes` keeps
//! seeing everything. A game that resolves the entry point through
//! `GetProcAddress` is not covered.
//!
//! [`served_mode_sizes`]: mtld3d_core::display_mode::served_mode_sizes

use core::ffi::c_void;
use std::sync::LazyLock;

use log::info;
use mtld3d_core::display_mode::served_mode_indices;

use super::{
    LOG_TARGET,
    import_patch::{Hook, PatchedImports},
};

/// `EnumDisplaySettings` mode index for the current mode; passed through.
const ENUM_CURRENT_SETTINGS: u32 = 0xFFFF_FFFF;
/// `EnumDisplaySettings` mode index for the registry mode; passed through.
const ENUM_REGISTRY_SETTINGS: u32 = 0xFFFF_FFFE;
/// How many `EnumDisplaySettings*` entry points are redirected.
const HOOKS: usize = 4;
/// Slot of each redirected entry point, in the order [`install`] lists them.
const SETTINGS_A: usize = 0;
const SETTINGS_W: usize = 1;
const SETTINGS_EX_A: usize = 2;
const SETTINGS_EX_W: usize = 3;

/// The redirected `EnumDisplaySettings*` imports of the main module.
static IMPORTS: PatchedImports<HOOKS> = PatchedImports::empty();

/// For each served index, the index in user32's list of the mode it stands for.
///
/// Built on the first enumeration through a patched import, from this DLL's
/// own (unpatched) enumeration of the primary display and the sizes the
/// adapter table serves.
static SERVED_INDICES: LazyLock<Vec<u32>> = LazyLock::new(|| {
    let modes = crate::fullscreen::enumerate_display_modes();
    let indices = served_mode_indices(
        modes.iter().map(|mode| (mode.width, mode.height)),
        crate::direct3d9::served_sizes(),
    );
    info!(
        target: LOG_TARGET,
        "display-mode list filter: the main module enumerates {} of user32's {} modes",
        indices.len(),
        modes.len()
    );
    indices
});

type EnumDisplaySettingsFn<N> = extern "system" fn(*const N, u32, *mut c_void) -> i32;
type EnumDisplaySettingsExFn<N> = extern "system" fn(*const N, u32, *mut c_void, u32) -> i32;

/// Redirect the main module's `EnumDisplaySettings*` imports here. Idempotent.
///
/// Called from `DllMain` `PROCESS_ATTACH`, which runs before the game's
/// entry point when d3d9 is a static import, so the list is bounded before
/// the game's first enumeration.
pub fn install() {
    let user32 = Some(&b"user32.dll"[..]);
    let patched = IMPORTS.install(&[
        Hook {
            dll: user32,
            func: b"EnumDisplaySettingsA",
            replacement: enum_display_settings_a as *const (),
        },
        Hook {
            dll: user32,
            func: b"EnumDisplaySettingsW",
            replacement: enum_display_settings_w as *const (),
        },
        Hook {
            dll: user32,
            func: b"EnumDisplaySettingsExA",
            replacement: enum_display_settings_ex_a as *const (),
        },
        Hook {
            dll: user32,
            func: b"EnumDisplaySettingsExW",
            replacement: enum_display_settings_ex_w as *const (),
        },
    ]);
    if patched != 0 {
        info!(
            target: LOG_TARGET,
            "display-mode list filter: redirected {patched} EnumDisplaySettings import(s) of the \
             main module"
        );
    }
}

/// Put the original entry points back. Idempotent.
///
/// Called from `DllMain` `PROCESS_DETACH` on the path the process survives,
/// so no slot keeps pointing into an image that is about to unmap.
pub fn uninstall() {
    IMPORTS.uninstall();
}

/// The index in user32's list that `mode_num` from the game stands for; `None` past the end.
fn map_mode_index(mode_num: u32) -> Option<u32> {
    if mode_num == ENUM_CURRENT_SETTINGS || mode_num == ENUM_REGISTRY_SETTINGS {
        return Some(mode_num);
    }
    SERVED_INDICES.get(usize::try_from(mode_num).ok()?).copied()
}

extern "system" fn enum_display_settings_a(name: *const u8, mode_num: u32, dm: *mut c_void) -> i32 {
    let Some(mapped) = map_mode_index(mode_num) else {
        return 0;
    };
    IMPORTS.original::<EnumDisplaySettingsFn<u8>>(SETTINGS_A)(name, mapped, dm)
}

extern "system" fn enum_display_settings_w(
    name: *const u16,
    mode_num: u32,
    dm: *mut c_void,
) -> i32 {
    let Some(mapped) = map_mode_index(mode_num) else {
        return 0;
    };
    IMPORTS.original::<EnumDisplaySettingsFn<u16>>(SETTINGS_W)(name, mapped, dm)
}

extern "system" fn enum_display_settings_ex_a(
    name: *const u8,
    mode_num: u32,
    dm: *mut c_void,
    flags: u32,
) -> i32 {
    let Some(mapped) = map_mode_index(mode_num) else {
        return 0;
    };
    IMPORTS.original::<EnumDisplaySettingsExFn<u8>>(SETTINGS_EX_A)(name, mapped, dm, flags)
}

extern "system" fn enum_display_settings_ex_w(
    name: *const u16,
    mode_num: u32,
    dm: *mut c_void,
    flags: u32,
) -> i32 {
    let Some(mapped) = map_mode_index(mode_num) else {
        return 0;
    };
    IMPORTS.original::<EnumDisplaySettingsExFn<u16>>(SETTINGS_EX_W)(name, mapped, dm, flags)
}
