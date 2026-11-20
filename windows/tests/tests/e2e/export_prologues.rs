//! Every export of the d3d9 API starts with the fixed prologue inline-hook engines relocate.
//!
//! Overlays hook `Direct3DCreate9` and its siblings by decoding the first
//! instructions of the export with a length decoder of their own, copying at
//! least five bytes of whole instructions into a trampoline and writing a
//! `jmp rel32` over them; a decoder that meets an instruction it does not know
//! skips the hook. The i386 build pins the first bytes of each export to the
//! hot-patch prologue such decoders know, so this reads them through
//! `GetProcAddress`, the address a hook engine patches, and compares them
//! byte for byte. The list of exports is read from `d3d9.def`, so an export
//! added there as a plain function fails here. Each is resolved by its
//! ordinal, and a named one by its name too, which has to give the same
//! address, so an export the link moved to another ordinal fails as well.
//!
//! No shared harness: the point is the export table, as in `d3dperf.rs`.

use core::ffi::{c_char, c_void};
#[cfg(target_arch = "x86")]
use std::ffi::CString;

#[link(name = "kernel32")]
unsafe extern "system" {
    fn LoadLibraryA(name: *const c_char) -> *mut c_void;
    fn FreeLibrary(module: *mut c_void) -> i32;
    #[cfg(target_arch = "x86")]
    fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
}

/// The `.def` file `d3d9.dll` links with: its `EXPORTS` section lists every export and its ordinal.
#[cfg(target_arch = "x86")]
const D3D9_DEF: &str = include_str!("../../../d3d9/d3d9.def");

/// `mov edi, edi; push ebp; mov ebp, esp; pop ebp`, in the encodings MSVC emits.
#[cfg(target_arch = "x86")]
const PROLOGUE: [u8; 6] = [0x8B, 0xFF, 0x55, 0x8B, 0xEC, 0x5D];

#[test]
fn every_api_export_starts_with_the_hook_prologue() {
    // SAFETY: plain kernel32 call with a NUL-terminated name.
    let lib = unsafe { LoadLibraryA(c"d3d9.dll".as_ptr()) };
    assert!(!lib.is_null(), "LoadLibrary(d3d9.dll)");

    // An x64 or a native ARM64 process loads the ARM64X build, whose exports
    // are ARM64EC or ARM64 code, which an x86 hook engine cannot decode
    // whatever their first bytes are.
    #[cfg(not(target_arch = "x86"))]
    eprintln!("[e2e] d3d9.dll is an ARM64X image: its exports are ARM64 code, not checked");
    #[cfg(target_arch = "x86")]
    check_prologues(lib);

    // SAFETY: balancing the LoadLibrary above.
    assert_ne!(unsafe { FreeLibrary(lib) }, 0, "FreeLibrary(d3d9.dll)");
}

/// Compare the first bytes of every export `d3d9.def` lists with [`PROLOGUE`].
#[cfg(target_arch = "x86")]
fn check_prologues(lib: *mut c_void) {
    let exports = def_exports();
    assert!(
        exports
            .iter()
            .any(|export| export.name.as_deref() == Some(c"Direct3DCreate9")),
        "d3d9.def lists Direct3DCreate9: {exports:?}"
    );
    let wrong: Vec<String> = exports
        .iter()
        .filter_map(|export| {
            let found = first_bytes(lib, export);
            (found != PROLOGUE).then(|| format!("{}: {found:02X?}", export.label()))
        })
        .collect();
    assert!(
        wrong.is_empty(),
        "exports that do not start with {PROLOGUE:02X?}:\n{}",
        wrong.join("\n")
    );
}

/// One entry of the `EXPORTS` section of `d3d9.def`.
#[cfg(target_arch = "x86")]
#[derive(Debug)]
struct DefExport {
    /// The export's name, `None` for a `NONAME` entry.
    name: Option<CString>,
    ordinal: u16,
}

#[cfg(target_arch = "x86")]
impl DefExport {
    /// The name the entry has in a failure message.
    fn label(&self) -> String {
        let name = self
            .name
            .as_ref()
            .map_or("NONAME", |name| name.to_str().unwrap_or("?"));
        format!("{name} @{}", self.ordinal)
    }
}

/// Every entry the `EXPORTS` section of `d3d9.def` lists: `Name @ordinal [NONAME]`.
#[cfg(target_arch = "x86")]
fn def_exports() -> Vec<DefExport> {
    let (_, exports) = D3D9_DEF
        .split_once("\nEXPORTS")
        .expect("d3d9.def has an EXPORTS section");
    exports
        .lines()
        .filter_map(|line| {
            let entry = line.split_once(';').map_or(line, |(entry, _)| entry);
            let mut fields = entry.split_whitespace();
            let name = fields.next()?;
            let ordinal = fields
                .next()
                .and_then(|field| field.strip_prefix('@'))
                .and_then(|ordinal| ordinal.parse().ok())
                .unwrap_or_else(|| panic!("d3d9.def: {name} has no `@ordinal`"));
            let named = fields.next() != Some("NONAME");
            Some(DefExport {
                name: named.then(|| CString::new(name).expect("an export name has no NUL")),
                ordinal,
            })
        })
        .collect()
}

/// The first bytes of `export`, as many as the prologue has.
///
/// Resolved by ordinal, and a named export by name too, the two addresses
/// required to match.
#[cfg(target_arch = "x86")]
fn first_bytes(lib: *mut c_void, export: &DefExport) -> [u8; PROLOGUE.len()] {
    // `MAKEINTRESOURCE`: an ordinal is passed as the low word of the name pointer.
    let by_ordinal = core::ptr::without_provenance::<c_char>(usize::from(export.ordinal));
    // SAFETY: `lib` is a live module handle and an ordinal below 0x10000 is a
    // valid name argument.
    let addr = unsafe { GetProcAddress(lib, by_ordinal) };
    assert!(!addr.is_null(), "GetProcAddress({})", export.label());
    if let Some(name) = &export.name {
        // SAFETY: `lib` is a live module handle and the name is NUL-terminated.
        let by_name = unsafe { GetProcAddress(lib, name.as_ptr()) };
        assert_eq!(by_name, addr, "{} by name and by ordinal", export.label());
    }
    // SAFETY: an export of a loaded image points into its mapped, readable
    // code section, which holds at least the prologue's length past every
    // entry: the function itself or the padding that aligns the next one.
    unsafe { core::ptr::read_unaligned(addr.cast::<[u8; PROLOGUE.len()]>()) }
}
