//! Redirecting imports of the process's main module to entry points of ours.
//!
//! The loader resolves a static import into one slot of the importing
//! image's import address table, and every call the image makes goes through
//! that slot, so writing one redirects the main module's calls without
//! touching the exporting DLL or any other importer. This DLL's own imports
//! are never patched, so it keeps reaching the real entry points. A module
//! that resolves the same entry point through `GetProcAddress` is not
//! covered: nothing it calls goes through an import slot.

use core::{
    ffi::c_void,
    ptr,
    sync::atomic::{AtomicBool, AtomicPtr, Ordering},
};

use log::warn;

use super::LOG_TARGET;

/// `VirtualProtect` protection that lets an import slot be written.
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

unsafe extern "system" {
    fn GetModuleHandleA(module_name: *const u8) -> *mut c_void;
    fn VirtualProtect(
        address: *mut c_void,
        size: usize,
        new_protect: u32,
        old_protect: *mut u32,
    ) -> i32;
}

/// One import to redirect.
pub struct Hook<'a> {
    /// The exporting module's file name, or `None` for whichever imports the name.
    pub dll: Option<&'a [u8]>,
    /// The imported entry point's name, as `IMAGE_IMPORT_BY_NAME` spells it.
    pub func: &'a [u8],
    /// The entry point calls are sent to instead.
    pub replacement: *const (),
}

/// A PE image mapped by the loader, addressed by its base.
///
/// The invariant the constructor asserts is that `base` is the base of an
/// image the loader mapped in this process, so every RVA in its headers
/// resolves inside the mapping; the methods read headers on that basis.
pub struct MappedImage {
    base: *const u8,
}

/// A fixed set of the main module's import slots, redirected to entry points of ours.
///
/// One static per group of related imports. The slots keep the order of the
/// hooks they were installed from, so a replacement reaches the entry point
/// it displaced through [`PatchedImports::original`] with its own index.
pub struct PatchedImports<const N: usize> {
    /// The patched slots, in hook order; null where the import is absent.
    slots: [AtomicPtr<*const ()>; N],
    /// The entry points the slots held; null where no slot was patched.
    originals: [AtomicPtr<()>; N],
    /// Whether [`PatchedImports::install`] has already run.
    installed: AtomicBool,
}

impl MappedImage {
    /// # Safety
    ///
    /// `base` must be the base address of a PE image the loader mapped in
    /// this process (an `HMODULE`).
    const unsafe fn new(base: *const u8) -> Self {
        Self { base }
    }

    /// The import-address-table slot for `func`, if the image imports it.
    ///
    /// `dll` names the exporting module the import must come from; `None`
    /// takes the first import descriptor that carries the name, whichever
    /// module it is.
    pub fn import_slot(&self, dll: Option<&[u8]>, func: &[u8]) -> Option<*mut *const ()> {
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
            if dll.is_some_and(|dll| !self.c_string(name_rva).eq_ignore_ascii_case(dll)) {
                continue;
            }
            let names = usize::try_from(self.read::<u32>(descriptor)).ok()?;
            let addresses = usize::try_from(self.read::<u32>(descriptor + 16)).ok()?;
            // Without a separate name table the address table still holds
            // the names before the loader overwrote it; the loader has, so
            // an image bound that way is not searchable here.
            if names == 0 || addresses == 0 {
                continue;
            }
            if let Some(slot) = self.named_slot(names, addresses, func) {
                return Some(slot);
            }
        }
        None
    }

    /// The address-table slot of `func` in one descriptor's name table.
    fn named_slot(&self, names: usize, addresses: usize, func: &[u8]) -> Option<*mut *const ()> {
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
                // An import-address-table entry is pointer-aligned by the PE
                // layout; the address stays inside the mapping (see `read`)
                // and keeps the image's provenance.
                let slot = addresses + j * size_of::<usize>();
                return Some(ptr::with_exposed_provenance_mut(
                    self.base.expose_provenance() + slot,
                ));
            }
        }
        None
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
}

impl<const N: usize> PatchedImports<N> {
    /// A set with nothing patched yet, for a `static`.
    pub const fn empty() -> Self {
        Self {
            slots: [const { AtomicPtr::new(ptr::null_mut()) }; N],
            originals: [const { AtomicPtr::new(ptr::null_mut()) }; N],
            installed: AtomicBool::new(false),
        }
    }

    /// Redirect each hook's import of the main module; how many were patched.
    ///
    /// Idempotent, and every hook is optional: an import the main module
    /// does not have and a slot whose page will not open are both skipped,
    /// leaving that index without an original and its replacement unreached.
    pub fn install(&self, hooks: &[Hook<'_>; N]) -> usize {
        if self.installed.swap(true, Ordering::AcqRel) {
            return 0;
        }
        let Some(image) = main_module() else {
            return 0;
        };
        let mut patched = 0;
        for (i, hook) in hooks.iter().enumerate() {
            let Some(slot) = image.import_slot(hook.dll, hook.func) else {
                continue;
            };
            let Some(original) = write_slot(slot, hook.replacement) else {
                warn!(
                    target: LOG_TARGET,
                    "import patch: the import slot of {} is not writable",
                    String::from_utf8_lossy(hook.func)
                );
                continue;
            };
            self.originals[i].store(original.cast_mut(), Ordering::Release);
            self.slots[i].store(slot, Ordering::Release);
            patched += 1;
        }
        patched
    }

    /// Put the original entry points back. Idempotent.
    ///
    /// Called on the `FreeLibrary` path the process survives, so no slot
    /// keeps pointing into an image that is about to unmap.
    pub fn uninstall(&self) {
        for (slot, original) in self.slots.iter().zip(&self.originals) {
            let slot = slot.swap(ptr::null_mut(), Ordering::AcqRel);
            if slot.is_null() {
                continue;
            }
            write_slot(slot, original.swap(ptr::null_mut(), Ordering::AcqRel));
        }
        self.installed.store(false, Ordering::Release);
    }

    /// The entry point the slot at `index` held, as the function type `F`.
    pub fn original<F>(&self, index: usize) -> F {
        let original = self.originals[index].load(Ordering::Acquire);
        // SAFETY: the slot held an entry point of exactly the signature `F`
        // names before `install` replaced it, and a replacement only runs
        // while its slot is patched, so the pointer is that entry point and
        // `F` is a pointer-sized `fn` type.
        unsafe { core::mem::transmute_copy(&original) }
    }
}

/// The process's main module, if the loader can name it.
fn main_module() -> Option<MappedImage> {
    // SAFETY: Win32; a null name selects the process's main module.
    let base = unsafe { GetModuleHandleA(ptr::null()) };
    if base.is_null() {
        return None;
    }
    // SAFETY: `GetModuleHandleA(NULL)` is the base of the main module the
    // loader mapped in this process.
    Some(unsafe { MappedImage::new(base.cast::<u8>().cast_const()) })
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
