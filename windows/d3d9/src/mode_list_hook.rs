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

use core::{
    ffi::c_void,
    ptr,
    sync::atomic::{AtomicPtr, Ordering},
};
use std::sync::LazyLock;

use log::{info, warn};
use mtld3d_core::display_mode::served_mode_indices;

use super::LOG_TARGET;

/// `EnumDisplaySettings` mode index for the current mode; passed through.
const ENUM_CURRENT_SETTINGS: u32 = 0xFFFF_FFFF;
/// `EnumDisplaySettings` mode index for the registry mode; passed through.
const ENUM_REGISTRY_SETTINGS: u32 = 0xFFFF_FFFE;
/// `VirtualProtect` protection that lets the import slot be written.
const PAGE_READWRITE: u32 = 0x04;
/// `IMAGE_DOS_HEADER.e_lfanew`: file offset of the NT headers.
const E_LFANEW_OFFSET: usize = 0x3c;
/// Size of `IMAGE_FILE_HEADER`, which sits between the PE signature and the optional header.
const FILE_HEADER_SIZE: usize = 20;
/// `IMAGE_OPTIONAL_HEADER.Magic` for PE32 and PE32+.
const PE32_MAGIC: u16 = 0x10b;
const PE32PLUS_MAGIC: u16 = 0x20b;
/// Offset of `DataDirectory` inside the optional header, per format.
const PE32_DATA_DIRECTORY_OFFSET: usize = 96;
const PE32PLUS_DATA_DIRECTORY_OFFSET: usize = 112;
/// `IMAGE_DIRECTORY_ENTRY_IMPORT`.
const IMPORT_DIRECTORY: usize = 1;
/// Size of `IMAGE_IMPORT_DESCRIPTOR`.
const IMPORT_DESCRIPTOR_SIZE: usize = 20;
/// Thunk entries with this bit import by ordinal and carry no name.
const ORDINAL_FLAG: usize = 1 << (usize::BITS - 1);

/// The imports redirected and their replacements, in the order of [`SLOTS`] and [`ORIGINALS`].
fn hooks() -> [(&'static [u8], *const ()); 4] {
    [
        (
            b"EnumDisplaySettingsA",
            enum_display_settings_a as *const (),
        ),
        (
            b"EnumDisplaySettingsW",
            enum_display_settings_w as *const (),
        ),
        (
            b"EnumDisplaySettingsExA",
            enum_display_settings_ex_a as *const (),
        ),
        (
            b"EnumDisplaySettingsExW",
            enum_display_settings_ex_w as *const (),
        ),
    ]
}

/// The main module's import slots that were patched; null where the import is absent.
static SLOTS: [AtomicPtr<*const ()>; 4] = [const { AtomicPtr::new(ptr::null_mut()) }; 4];
/// user32's entry points the slots held before the patch; null where none was patched.
static ORIGINALS: [AtomicPtr<()>; 4] = [const { AtomicPtr::new(ptr::null_mut()) }; 4];

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

/// A PE image mapped by the loader, addressed by its base.
///
/// The invariant the constructor asserts is that `base` is the base of an
/// image the loader mapped in this process, so every RVA in its headers
/// resolves inside the mapping; the methods read headers on that basis.
struct MappedImage {
    base: *const u8,
}

impl MappedImage {
    /// # Safety
    ///
    /// `base` must be the base address of a PE image the loader mapped in
    /// this process (an `HMODULE`).
    const unsafe fn new(base: *const u8) -> Self {
        Self { base }
    }

    const fn read<T: Copy>(&self, offset: usize) -> T {
        // SAFETY: `base` is a mapped image (the type's invariant) and every
        // offset read here comes from its own headers, which the loader
        // validated when it mapped the image, so the sum stays inside it.
        let at = unsafe { self.base.add(offset) };
        // SAFETY: `at` is inside the mapping (above); `read_unaligned`
        // tolerates the packed header layouts.
        unsafe { at.cast::<T>().read_unaligned() }
    }

    /// The bytes of a NUL-terminated string at `offset`, without the terminator.
    const fn c_string(&self, offset: usize) -> &[u8] {
        let mut len = 0;
        while self.read::<u8>(offset + len) != 0 {
            len += 1;
        }
        // SAFETY: `offset` was just read from inside the mapping (see `read`).
        let start = unsafe { self.base.add(offset) };
        // SAFETY: the bytes `offset..offset + len` were just read one by one
        // inside the mapping, so the slice lies within it and stays mapped
        // for the image's lifetime, which outlives `self`.
        unsafe { core::slice::from_raw_parts(start, len) }
    }

    /// The import-address-table slot for `func` imported from `dll`, if the image has one.
    fn import_slot(&self, dll: &[u8], func: &[u8]) -> Option<*mut *const ()> {
        if self.read::<[u8; 2]>(0) != *b"MZ" {
            return None;
        }
        let nt = usize::try_from(self.read::<u32>(E_LFANEW_OFFSET)).ok()?;
        if self.read::<[u8; 4]>(nt) != *b"PE\0\0" {
            return None;
        }
        let optional = nt + 4 + FILE_HEADER_SIZE;
        let directory = match self.read::<u16>(optional) {
            PE32_MAGIC => PE32_DATA_DIRECTORY_OFFSET,
            PE32PLUS_MAGIC => PE32PLUS_DATA_DIRECTORY_OFFSET,
            _ => return None,
        };
        let imports =
            usize::try_from(self.read::<u32>(optional + directory + IMPORT_DIRECTORY * 8)).ok()?;
        if imports == 0 {
            return None;
        }
        for i in 0.. {
            let descriptor = imports + i * IMPORT_DESCRIPTOR_SIZE;
            let name_rva = usize::try_from(self.read::<u32>(descriptor + 12)).ok()?;
            if name_rva == 0 {
                return None;
            }
            if !self.c_string(name_rva).eq_ignore_ascii_case(dll) {
                continue;
            }
            let names = usize::try_from(self.read::<u32>(descriptor)).ok()?;
            let addresses = usize::try_from(self.read::<u32>(descriptor + 16)).ok()?;
            // Without a separate name table the address table still holds
            // the names before the loader overwrote it; the loader has, so
            // an image bound that way is not searchable and yields nothing.
            if names == 0 || addresses == 0 {
                return None;
            }
            for j in 0.. {
                let entry = self.read::<usize>(names + j * size_of::<usize>());
                if entry == 0 {
                    return None;
                }
                if entry & ORDINAL_FLAG != 0 {
                    continue;
                }
                // `IMAGE_IMPORT_BY_NAME`: a u16 hint, then the name.
                if self.c_string(entry + 2) == func {
                    // An import-address-table entry is pointer-aligned by
                    // the PE layout; the address stays inside the mapping
                    // (see `read`) and keeps the image's provenance.
                    let slot = addresses + j * size_of::<usize>();
                    return Some(ptr::with_exposed_provenance_mut(
                        self.base.expose_provenance() + slot,
                    ));
                }
            }
        }
        None
    }
}

unsafe extern "system" {
    fn GetModuleHandleA(module_name: *const u8) -> *mut c_void;
    fn VirtualProtect(
        address: *mut c_void,
        size: usize,
        new_protect: u32,
        old_protect: *mut u32,
    ) -> i32;
}

/// Write `value` into an import slot, returning what it held; `None` if the page cannot be opened.
fn write_slot(slot: *mut *const (), value: *const ()) -> Option<*const ()> {
    let mut old_protect = 0;
    // SAFETY: `slot` lies inside the main module's mapped import table
    // (`MappedImage::import_slot`); `old_protect` is an owned local.
    let opened = unsafe {
        VirtualProtect(
            slot.cast::<c_void>(),
            size_of::<usize>(),
            PAGE_READWRITE,
            &raw mut old_protect,
        )
    };
    if opened == 0 {
        return None;
    }
    // SAFETY: the slot's page is writable as of the call above and the slot
    // is a valid, aligned pointer-sized entry of the import address table.
    let previous = unsafe { slot.replace(value) };
    // SAFETY: restoring the protection the page had; a failure here leaves
    // the page writable, which is harmless.
    unsafe {
        VirtualProtect(
            slot.cast::<c_void>(),
            size_of::<usize>(),
            old_protect,
            &raw mut old_protect,
        )
    };
    Some(previous)
}

/// Redirect the main module's `EnumDisplaySettings*` imports here. Idempotent.
///
/// Called from `DllMain` `PROCESS_ATTACH`, which runs before the game's
/// entry point when d3d9 is a static import, so the list is bounded before
/// the game's first enumeration.
pub fn install() {
    if !SLOTS[0].load(Ordering::Acquire).is_null() {
        return;
    }
    // SAFETY: Win32; a null name selects the process's main module.
    let base = unsafe { GetModuleHandleA(ptr::null()) };
    if base.is_null() {
        return;
    }
    // SAFETY: `GetModuleHandleA(NULL)` is the base of the main module the
    // loader mapped in this process.
    let image = unsafe { MappedImage::new(base.cast::<u8>().cast_const()) };
    let mut patched = 0;
    for (i, (name, hook)) in hooks().into_iter().enumerate() {
        let Some(slot) = image.import_slot(b"user32.dll", name) else {
            continue;
        };
        let Some(original) = write_slot(slot, hook) else {
            warn!(
                target: LOG_TARGET,
                "display-mode list filter: the import slot of {} is not writable",
                String::from_utf8_lossy(name)
            );
            continue;
        };
        ORIGINALS[i].store(original.cast_mut(), Ordering::Release);
        SLOTS[i].store(slot, Ordering::Release);
        patched += 1;
    }
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
    for (slot, original) in SLOTS.iter().zip(&ORIGINALS) {
        let slot = slot.swap(ptr::null_mut(), Ordering::AcqRel);
        if slot.is_null() {
            continue;
        }
        write_slot(slot, original.swap(ptr::null_mut(), Ordering::AcqRel));
    }
}

/// The index in user32's list that `mode_num` from the game stands for; `None` past the end.
fn map_mode_index(mode_num: u32) -> Option<u32> {
    if mode_num == ENUM_CURRENT_SETTINGS || mode_num == ENUM_REGISTRY_SETTINGS {
        return Some(mode_num);
    }
    SERVED_INDICES.get(usize::try_from(mode_num).ok()?).copied()
}

fn original<F>(index: usize) -> F {
    let original = ORIGINALS[index].load(Ordering::Acquire);
    // SAFETY: the slot held user32's entry point of exactly the signature
    // `F` names before `install` replaced it, and a hook only runs while its
    // slot is patched, so the pointer is that entry point and `F` is a
    // pointer-sized `fn` type.
    unsafe { core::mem::transmute_copy(&original) }
}

extern "system" fn enum_display_settings_a(name: *const u8, mode_num: u32, dm: *mut c_void) -> i32 {
    let Some(mapped) = map_mode_index(mode_num) else {
        return 0;
    };
    original::<EnumDisplaySettingsFn<u8>>(0)(name, mapped, dm)
}

extern "system" fn enum_display_settings_w(
    name: *const u16,
    mode_num: u32,
    dm: *mut c_void,
) -> i32 {
    let Some(mapped) = map_mode_index(mode_num) else {
        return 0;
    };
    original::<EnumDisplaySettingsFn<u16>>(1)(name, mapped, dm)
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
    original::<EnumDisplaySettingsExFn<u8>>(2)(name, mapped, dm, flags)
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
    original::<EnumDisplaySettingsExFn<u16>>(3)(name, mapped, dm, flags)
}
