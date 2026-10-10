//! Every export of the d3d9 API starts with the fixed prologue inline-hook engines relocate.
//!
//! Overlays hook `Direct3DCreate9` and its siblings by decoding the first
//! instructions of the export with a length decoder of their own, copying at
//! least five bytes of whole instructions into a trampoline and writing a
//! `jmp rel32` over them; a decoder that meets an instruction it does not know
//! skips the hook. The layer pins the first bytes of each export to one
//! prologue per arch that such decoders know, so this reads them through
//! `GetProcAddress`, the address a hook engine patches, and compares them
//! byte for byte.
//!
//! No shared harness: the point is the export table, as in `d3dperf.rs`.

use core::ffi::{CStr, c_char, c_void};

#[link(name = "kernel32")]
unsafe extern "system" {
    fn LoadLibraryA(name: *const c_char) -> *mut c_void;
    fn FreeLibrary(module: *mut c_void) -> i32;
    fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
}

/// Every export `d3d9.def` lists.
const EXPORTS: [&CStr; 10] = [
    c"Direct3DCreate9",
    c"Direct3DCreate9Ex",
    c"Direct3DShaderValidatorCreate9",
    c"D3DPERF_BeginEvent",
    c"D3DPERF_EndEvent",
    c"D3DPERF_GetStatus",
    c"D3DPERF_QueryRepeatFrame",
    c"D3DPERF_SetMarker",
    c"D3DPERF_SetOptions",
    c"D3DPERF_SetRegion",
];

/// `mov edi, edi; push ebp; mov ebp, esp; pop ebp`, in the encodings MSVC emits.
#[cfg(target_arch = "x86")]
const PROLOGUE: [u8; 6] = [0x8B, 0xFF, 0x55, 0x8B, 0xEC, 0x5D];

/// The five-byte NOP, `nop dword ptr [rax + rax]`.
#[cfg(target_arch = "x86_64")]
const PROLOGUE: [u8; 5] = [0x0F, 0x1F, 0x44, 0x00, 0x00];

#[test]
fn every_api_export_starts_with_the_hook_prologue() {
    // SAFETY: plain kernel32 call with a NUL-terminated name.
    let lib = unsafe { LoadLibraryA(c"d3d9.dll".as_ptr()) };
    assert!(!lib.is_null(), "LoadLibrary(d3d9.dll)");

    if is_hybrid_image(lib) {
        // The ARM64X build's exports are ARM64EC code, which an x64 hook
        // engine cannot decode whatever its first bytes are.
        eprintln!("[e2e] d3d9.dll is an ARM64X image: its exports are ARM64EC code, not checked");
    } else {
        let wrong: Vec<String> = EXPORTS
            .iter()
            .filter_map(|name| {
                let found = first_bytes(lib, name);
                (found != PROLOGUE)
                    .then(|| format!("{}: {found:02X?}", name.to_str().unwrap_or("?")))
            })
            .collect();
        assert!(
            wrong.is_empty(),
            "exports that do not start with {PROLOGUE:02X?}:\n{}",
            wrong.join("\n")
        );
    }

    // SAFETY: balancing the LoadLibrary above.
    assert_ne!(unsafe { FreeLibrary(lib) }, 0, "FreeLibrary(d3d9.dll)");
}

/// The first bytes of the export `name`, as many as the prologue has.
fn first_bytes(lib: *mut c_void, name: &CStr) -> [u8; PROLOGUE.len()] {
    // SAFETY: `lib` is a live module handle and the name is NUL-terminated.
    let addr = unsafe { GetProcAddress(lib, name.as_ptr()) };
    assert!(
        !addr.is_null(),
        "GetProcAddress({})",
        name.to_str().unwrap_or("?")
    );
    // SAFETY: an export of a loaded image points into its mapped, readable
    // code section, which holds at least the prologue's length past every
    // entry: the function itself or the padding that aligns the next one.
    unsafe { core::ptr::read_unaligned(addr.cast::<[u8; PROLOGUE.len()]>()) }
}

/// Whether the loaded image is an ARM64X one, whose load config names CHPE metadata.
#[cfg(target_arch = "x86_64")]
fn is_hybrid_image(lib: *mut c_void) -> bool {
    /// The offset of `e_lfanew` in the DOS header, the start of the NT headers.
    const E_LFANEW: usize = 0x3C;
    /// The offset of the load-config data directory in the PE32+ NT headers.
    const LOAD_CONFIG_DIRECTORY: usize = 24 + 112 + 10 * 8;
    /// The offset of `CHPEMetadataPointer` in the 64-bit load-config directory.
    const CHPE_METADATA_POINTER: usize = 0xC8;

    let base = lib.cast::<u8>();
    let read_u32 = |offset: usize| {
        // SAFETY: every offset read here lies in the mapped headers or the
        // load-config directory of the image `base` is the module handle of.
        unsafe { core::ptr::read_unaligned(base.wrapping_add(offset).cast::<u32>()) }
    };
    let as_usize = |value: u32| usize::try_from(value).expect("u32 fits usize on x86_64");
    let nt = as_usize(read_u32(E_LFANEW));
    let load_config = as_usize(read_u32(nt + LOAD_CONFIG_DIRECTORY));
    // The directory's own `Size` field says which fields this image carries.
    if load_config == 0 || as_usize(read_u32(load_config)) < CHPE_METADATA_POINTER + 8 {
        return false;
    }
    let field = base.wrapping_add(load_config + CHPE_METADATA_POINTER);
    // SAFETY: the load-config directory is mapped with the image and its size
    // covers the field, checked above.
    let chpe = unsafe { core::ptr::read_unaligned(field.cast::<u64>()) };
    chpe != 0
}

/// An i386 process never loads the ARM64X image.
#[cfg(target_arch = "x86")]
const fn is_hybrid_image(_lib: *mut c_void) -> bool {
    false
}
