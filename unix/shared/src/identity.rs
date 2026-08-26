//! Who this binary is: which release it was built from, and which exact image it is.
//!
//! [`BUILD`] is the release identity, stamped in by `build.rs` from `git
//! describe`. It says which source a build came from, but two builds of the
//! same source share it.
//!
//! [`image_id`] is the binary identity: the ID the linker itself assigned to
//! this image, read back out of the loaded image at runtime. Both `lld-link`
//! and `ld64` derive it from the output's contents, so it changes whenever the
//! binary does and is stable across a reproducible rebuild. It is also the key
//! the symbol tools pair on (the PDB's GUID, the `.dSYM`'s UUID), so one value
//! both names the build and selects the file that symbolicates it.
//!
//! Every cdylib logs both on load. A crash report then names the debug archive
//! that resolves it, instead of leaving us to fingerprint the build from
//! indirect evidence.
//!
//! `version_blob` answers the other identity question, and about a different
//! image: who the host application is. It hands back the raw `VS_VERSIONINFO`
//! resource of a mapped PE, which is where a built-in app profile reads the
//! vendor's own `CompanyName` and `ProductName` from.

/// Release identity, stamped by `build.rs` from `git describe --tags --always`.
///
/// `v0.2.0` on a release tag, `v0.2.0-3-g119c0db` between tags, and the
/// manifest version (`v0.2.0`) when built outside a git checkout.
pub const BUILD: &str = env!("MTLD3D_BUILD");

pub use platform::image_id;
#[cfg(target_family = "windows")]
pub use platform::version_blob;

#[cfg(target_family = "windows")]
mod platform {
    use core::{ffi::c_void, fmt::Write as _};

    /// Offset of `e_lfanew` in the DOS header: the file offset of the NT headers.
    const E_LFANEW: usize = 0x3c;
    /// Offset of the optional header from the start of the NT headers.
    ///
    /// Four-byte `PE\0\0` signature plus the 20-byte `IMAGE_FILE_HEADER`.
    const OPTIONAL_HEADER: usize = 0x18;
    /// `IMAGE_NT_OPTIONAL_HDR32_MAGIC`: a PE32 image, as shipped for i686.
    const MAGIC_PE32: u16 = 0x010b;
    /// `IMAGE_NT_OPTIONAL_HDR64_MAGIC`: a PE32+ image, as shipped for `x86_64`.
    const MAGIC_PE32_PLUS: u16 = 0x020b;
    /// Offset of `NumberOfRvaAndSizes` within a PE32 optional header.
    const NUM_RVA_PE32: usize = 0x5c;
    /// Offset of `NumberOfRvaAndSizes` within a PE32+ optional header.
    const NUM_RVA_PE32_PLUS: usize = 0x6c;
    /// Offset of `SizeOfImage` within an optional header, PE32 and PE32+ alike.
    const SIZE_OF_IMAGE: usize = 0x38;
    /// Index of the debug directory among the data directories.
    const DIRECTORY_DEBUG: usize = 6;
    /// Index of the resource directory among the data directories.
    const DIRECTORY_RESOURCE: usize = 2;
    /// Size of one data directory entry: an RVA and a size, both dwords.
    const DIRECTORY_ENTRY_SIZE: usize = 8;
    /// `RT_VERSION`, the resource type of a `VS_VERSIONINFO` blob.
    const RT_VERSION: u32 = 16;
    /// Size of one `IMAGE_RESOURCE_DIRECTORY`, after which its entries start.
    const RESOURCE_DIRECTORY_SIZE: usize = 16;
    /// Size of one `IMAGE_RESOURCE_DIRECTORY_ENTRY`: a name and an offset.
    const RESOURCE_ENTRY_SIZE: usize = 8;
    /// Offset of `NumberOfNamedEntries` within an `IMAGE_RESOURCE_DIRECTORY`.
    const NAMED_ENTRIES: usize = 12;
    /// Offset of `NumberOfIdEntries` within an `IMAGE_RESOURCE_DIRECTORY`.
    const ID_ENTRIES: usize = 14;
    /// High bit of a resource entry's offset: the child is another directory.
    const SUBDIRECTORY: u32 = 0x8000_0000;
    /// Size of one `IMAGE_DEBUG_DIRECTORY` entry.
    const DEBUG_ENTRY_SIZE: usize = 28;
    /// `IMAGE_DEBUG_TYPE_CODEVIEW`, the entry pointing at the PDB record.
    const DEBUG_TYPE_CODEVIEW: u32 = 2;
    /// `RSDS`, little-endian: the PDB 7.0 `CodeView` record signature.
    const RSDS: u32 = 0x5344_5352;

    /// This image's PDB GUID, read from its own `CodeView` debug record.
    ///
    /// `lld-link` derives the GUID from the PDB's contents, so it identifies
    /// this exact build and is what pairs the DLL with its `.pdb`. Returns
    /// `None` if the image carries no `CodeView` record, which is the case for a
    /// build with debug info switched off.
    ///
    /// # Safety
    ///
    /// `module` must be the `HMODULE` the loader passed to `DllMain` (this
    /// module's own handle), so that the whole image is mapped.
    #[must_use]
    pub unsafe fn image_id(module: *mut c_void) -> Option<String> {
        let base = module as usize;
        if base == 0 {
            return None;
        }
        // SAFETY: `base` is a loaded image, per the contract.
        let (dir_rva, dir_size) = unsafe { data_directory(base, DIRECTORY_DEBUG) }?;
        if dir_size < DEBUG_ENTRY_SIZE {
            return None;
        }
        // SAFETY: the debug directory RVA points into a mapped section (the
        // linker places the record in `.rdata`), and `dir_size` is its extent.
        unsafe { codeview_guid(base, base + dir_rva, dir_size / DEBUG_ENTRY_SIZE) }
    }

    /// The raw `VS_VERSIONINFO` resource of a mapped image, as UTF-16 units.
    ///
    /// Returns `None` when the image carries no version resource, which is
    /// normal: only vendors that set the linker's resource fields have one.
    /// Every read is bounded by the image's own `SizeOfImage`, so a malformed
    /// resource tree yields `None` rather than a wild read.
    ///
    /// Reading the mapped image keeps this free of imports. The obvious
    /// alternative, `version.dll`, is one of the DLL names game mods install
    /// proxies under, and so is the `d3d9.dll` this code ships as, so calling it
    /// would route our startup through whatever a modded install dropped beside
    /// the executable.
    ///
    /// # Safety
    ///
    /// `module` must be the `HMODULE` of an image the loader has mapped, so
    /// that its headers and every section its `SizeOfImage` covers are readable.
    #[must_use]
    pub unsafe fn version_blob(module: *mut c_void) -> Option<Vec<u16>> {
        let base = module as usize;
        if base == 0 {
            return None;
        }
        // SAFETY: `base` is a loaded image, per the contract.
        let (res, _) = unsafe { data_directory(base, DIRECTORY_RESOURCE) }?;
        // SAFETY: same image, so its optional header is mapped.
        let size = unsafe { image_size(base) }?;
        let img = Image { base, size };

        // Three levels, by the resource tree's fixed shape: type, then name,
        // then language. Only the type is looked up by id; a version resource
        // carries one name and one language in practice, and taking the first
        // of each is what the loader's own `FindResource` fallback settles on.
        // SAFETY: `img` bounds every read to the mapped image.
        let by_name = unsafe { img.subdirectory(res, res, RT_VERSION) }?;
        // SAFETY: as above.
        let (by_language, is_dir) = unsafe { img.first_child(res, by_name) }?;
        if !is_dir {
            return None;
        }
        // SAFETY: as above.
        let (leaf, is_dir) = unsafe { img.first_child(res, by_language) }?;
        if is_dir {
            return None;
        }

        // `IMAGE_RESOURCE_DATA_ENTRY`: the blob's own RVA, then its size.
        // SAFETY: as above.
        let data = unsafe { img.u32(leaf) }? as usize;
        // SAFETY: the size dword follows the RVA in the same entry.
        let size = unsafe { img.u32(leaf + 4) }? as usize;
        // SAFETY: as above.
        unsafe { img.utf16(data, size) }
    }

    /// Scan the debug directory for the `CodeView` entry and format its GUID.
    ///
    /// # Safety
    ///
    /// `dir` must point at `count` mapped `IMAGE_DEBUG_DIRECTORY` entries, and
    /// `base` must be the load base their RVAs are relative to.
    unsafe fn codeview_guid(base: usize, dir: usize, count: usize) -> Option<String> {
        for i in 0..count {
            let entry = dir + i * DEBUG_ENTRY_SIZE;
            // SAFETY: per the contract, entry `i` of `count` is mapped; `Type`
            // is the dword at `+12`.
            let kind = unsafe { read_u32(entry + 12) };
            if kind != DEBUG_TYPE_CODEVIEW {
                continue;
            }
            // SAFETY: same entry; `AddressOfRawData` is the dword at `+20`.
            let rva = unsafe { read_u32(entry + 20) } as usize;
            if rva == 0 {
                continue;
            }
            let record = base + rva;
            // SAFETY: `AddressOfRawData` is an RVA into a mapped section of the
            // same image, so the record's signature dword is readable.
            if unsafe { read_u32(record) } != RSDS {
                continue;
            }
            // SAFETY: an `RSDS` record carries its 16-byte GUID right after the
            // signature.
            let guid = unsafe { read_bytes(record + 4) };
            return Some(format_guid(&guid));
        }
        None
    }

    /// Format a 16-byte Windows GUID in the canonical 8-4-4-4-12 form.
    ///
    /// The first three fields are little-endian integers and the last eight
    /// bytes are in wire order, which is what makes this different from simply
    /// hex-dumping the buffer.
    fn format_guid(g: &[u8; 16]) -> String {
        let d1 = u32::from_le_bytes([g[0], g[1], g[2], g[3]]);
        let d2 = u16::from_le_bytes([g[4], g[5]]);
        let d3 = u16::from_le_bytes([g[6], g[7]]);
        let mut out = format!("{d1:08X}-{d2:04X}-{d3:04X}-");
        for byte in &g[8..10] {
            let _ = write!(out, "{byte:02X}");
        }
        out.push('-');
        for byte in &g[10..] {
            let _ = write!(out, "{byte:02X}");
        }
        out
    }

    /// One data directory's RVA and size.
    ///
    /// `None` when the image declares no entry at `index`, or when the entry is
    /// empty.
    ///
    /// # Safety
    ///
    /// `base` must be the load base of a mapped image.
    const unsafe fn data_directory(base: usize, index: usize) -> Option<(usize, usize)> {
        // SAFETY: a loaded image has its DOS header mapped, and `e_lfanew` is
        // the dword at `+0x3c`.
        let nt = base + unsafe { read_u32(base + E_LFANEW) } as usize;
        // SAFETY: `nt` is the NT header the DOS stub points at, mapped in the
        // same image; the optional header magic is the word at `+0x18`.
        let magic = unsafe { read_u16(nt + OPTIONAL_HEADER) };
        // The data directories follow `NumberOfRvaAndSizes`, whose offset in
        // the optional header differs between PE32 and PE32+ (the wider image
        // base and the four larger reserve/commit fields push it out by 0x10).
        let num_rva_offset = match magic {
            MAGIC_PE32 => NUM_RVA_PE32,
            MAGIC_PE32_PLUS => NUM_RVA_PE32_PLUS,
            _ => return None,
        };
        let num_rva_at = nt + OPTIONAL_HEADER + num_rva_offset;
        // SAFETY: within the optional header of a mapped image.
        let num_rva = unsafe { read_u32(num_rva_at) } as usize;
        if num_rva <= index {
            return None;
        }
        let entry = num_rva_at + 4 + index * DIRECTORY_ENTRY_SIZE;
        // SAFETY: entry `index` exists, per the `num_rva` check.
        let rva = unsafe { read_u32(entry) } as usize;
        // SAFETY: the size dword follows the RVA in the same entry.
        let size = unsafe { read_u32(entry + 4) } as usize;
        if rva == 0 {
            return None;
        }
        Some((rva, size))
    }

    /// The extent a mapped image covers, from its own `SizeOfImage`.
    ///
    /// # Safety
    ///
    /// `base` must be the load base of a mapped image.
    unsafe fn image_size(base: usize) -> Option<usize> {
        // SAFETY: a loaded image has its DOS header mapped.
        let nt = base + unsafe { read_u32(base + E_LFANEW) } as usize;
        // SAFETY: `nt` is the mapped NT header.
        let size = unsafe { read_u32(nt + OPTIONAL_HEADER + SIZE_OF_IMAGE) } as usize;
        (size != 0).then_some(size)
    }

    /// A mapped image, with every read bounded by its own extent.
    ///
    /// Resource walking follows offsets the image itself supplies, and the
    /// image here is the host application's rather than our own, so a bound is
    /// the difference between "no version resource" and a fault in someone's
    /// game.
    struct Image {
        base: usize,
        size: usize,
    }

    impl Image {
        /// The address of `len` bytes at `off`, or `None` if they leave the image.
        const fn at(&self, off: usize, len: usize) -> Option<usize> {
            match off.checked_add(len) {
                Some(end) if end <= self.size => self.base.checked_add(off),
                _ => None,
            }
        }

        /// Read a bounded `u16` at image offset `off`.
        ///
        /// # Safety
        ///
        /// `self` must describe a mapped image.
        unsafe fn u16(&self, off: usize) -> Option<u16> {
            let addr = self.at(off, 2)?;
            // SAFETY: `at` placed two readable bytes at `addr`.
            Some(unsafe { read_u16(addr) })
        }

        /// Read a bounded `u32` at image offset `off`.
        ///
        /// # Safety
        ///
        /// `self` must describe a mapped image.
        unsafe fn u32(&self, off: usize) -> Option<u32> {
            let addr = self.at(off, 4)?;
            // SAFETY: `at` placed four readable bytes at `addr`.
            Some(unsafe { read_u32(addr) })
        }

        /// Read `bytes` bytes at image offset `off` as UTF-16 units.
        ///
        /// # Safety
        ///
        /// `self` must describe a mapped image.
        unsafe fn utf16(&self, off: usize, bytes: usize) -> Option<Vec<u16>> {
            let mut out = Vec::with_capacity(bytes / 2);
            for i in 0..bytes / 2 {
                // SAFETY: each offset is bounds-checked by `u16` in turn.
                out.push(unsafe { self.u16(off + i * 2) }?);
            }
            Some(out)
        }

        /// The sub-directory of the entry with numeric id `id`.
        ///
        /// `res` is the resource directory's RVA, which every entry offset
        /// inside the tree is relative to.
        ///
        /// # Safety
        ///
        /// `self` must describe a mapped image.
        unsafe fn subdirectory(&self, res: usize, dir: usize, id: u32) -> Option<usize> {
            // SAFETY: bounded reads on a mapped image.
            let count = unsafe { self.entry_count(dir) }?;
            for i in 0..count {
                let entry = dir + RESOURCE_DIRECTORY_SIZE + i * RESOURCE_ENTRY_SIZE;
                // SAFETY: as above.
                let name = unsafe { self.u32(entry) }?;
                // SAFETY: the offset dword follows the name in the same entry.
                let offset = unsafe { self.u32(entry + 4) }?;
                // A numeric id (high bit clear) that matches, pointing at a
                // sub-directory (high bit of the offset set).
                if name == id && offset & SUBDIRECTORY != 0 {
                    return res.checked_add((offset & !SUBDIRECTORY) as usize);
                }
            }
            None
        }

        /// The first entry of a resource directory.
        ///
        /// Reports where its child sits, and whether that child is another
        /// directory rather than a leaf.
        ///
        /// # Safety
        ///
        /// `self` must describe a mapped image.
        unsafe fn first_child(&self, res: usize, dir: usize) -> Option<(usize, bool)> {
            // SAFETY: bounded reads on a mapped image.
            if unsafe { self.entry_count(dir) }? == 0 {
                return None;
            }
            // SAFETY: as above; the offset dword is the second half of the
            // first entry.
            let offset = unsafe { self.u32(dir + RESOURCE_DIRECTORY_SIZE + 4) }?;
            let child = res.checked_add((offset & !SUBDIRECTORY) as usize)?;
            Some((child, offset & SUBDIRECTORY != 0))
        }

        /// How many entries a resource directory holds, named and id-keyed.
        ///
        /// # Safety
        ///
        /// `self` must describe a mapped image.
        unsafe fn entry_count(&self, dir: usize) -> Option<usize> {
            // SAFETY: bounded reads on a mapped image.
            let named = usize::from(unsafe { self.u16(dir + NAMED_ENTRIES) }?);
            // SAFETY: as above.
            let ids = usize::from(unsafe { self.u16(dir + ID_ENTRIES) }?);
            named.checked_add(ids)
        }
    }

    /// Read a `u16` at `addr` without assuming alignment.
    ///
    /// # Safety
    ///
    /// `addr` must point at two readable bytes.
    const unsafe fn read_u16(addr: usize) -> u16 {
        // SAFETY: per the contract, two readable bytes live at `addr`.
        unsafe { (addr as *const u16).read_unaligned() }
    }

    /// Read a `u32` at `addr` without assuming alignment.
    ///
    /// # Safety
    ///
    /// `addr` must point at four readable bytes.
    const unsafe fn read_u32(addr: usize) -> u32 {
        // SAFETY: per the contract, four readable bytes live at `addr`.
        unsafe { (addr as *const u32).read_unaligned() }
    }

    /// Read 16 bytes at `addr` without assuming alignment.
    ///
    /// # Safety
    ///
    /// `addr` must point at sixteen readable bytes.
    const unsafe fn read_bytes(addr: usize) -> [u8; 16] {
        // SAFETY: per the contract, sixteen readable bytes live at `addr`.
        unsafe { (addr as *const [u8; 16]).read_unaligned() }
    }
}

#[cfg(unix)]
mod platform {
    use core::{ffi::c_void, fmt::Write as _};

    /// `LC_UUID`, the load command carrying the linker's image UUID.
    const LC_UUID: u32 = 0x1b;
    /// Size of a `mach_header_64`, after which the load commands start.
    const MACH_HEADER_64_SIZE: usize = 32;
    /// Offset of `ncmds` within a `mach_header_64`.
    const NCMDS: usize = 16;

    /// This image's Mach-O `LC_UUID`, read from its own load commands.
    ///
    /// `ld64` derives the UUID from the output's contents, so it identifies
    /// this exact build and is what pairs the dylib with its `.dSYM`. Returns
    /// `None` if the image carries no `LC_UUID`.
    #[must_use]
    pub fn image_id() -> Option<String> {
        let base = image_base()?;
        // SAFETY: `base` is this image's mach_header, so `ncmds` is the dword
        // at `+16`.
        let ncmds = unsafe { read_u32(base + NCMDS) };
        let mut cmd = base + MACH_HEADER_64_SIZE;
        for _ in 0..ncmds {
            // SAFETY: `cmd` walks the load-command chain of a mapped image,
            // bounded by the header's own `ncmds`; `cmd` is the first dword.
            let kind = unsafe { read_u32(cmd) };
            // SAFETY: `cmdsize` is the second dword of the same command.
            let size = unsafe { read_u32(cmd + 4) } as usize;
            if size == 0 {
                return None;
            }
            if kind == LC_UUID {
                // SAFETY: an `LC_UUID` command carries its 16 raw bytes right
                // after the `cmd`/`cmdsize` pair.
                let uuid = unsafe { read_bytes(cmd + 8) };
                return Some(format_uuid(&uuid));
            }
            cmd += size;
        }
        None
    }

    /// Load base of the image this code was linked into.
    ///
    /// Asking `dladdr` about one of our own functions is what makes this the
    /// *containing* image rather than the main executable, which matters
    /// because we are loaded as a library beside the Wine host.
    fn image_base() -> Option<usize> {
        let mut info = libc::Dl_info {
            dli_fname: core::ptr::null(),
            dli_fbase: core::ptr::null_mut(),
            dli_sname: core::ptr::null(),
            dli_saddr: core::ptr::null_mut(),
        };
        let probe: *const c_void = (image_base as *const ()).cast();
        // SAFETY: `probe` is a function pointer into this image and `info` is a
        // live `Dl_info`; `dladdr` only writes through the latter.
        let found = unsafe { libc::dladdr(probe, &raw mut info) };
        if found == 0 || info.dli_fbase.is_null() {
            return None;
        }
        Some(info.dli_fbase as usize)
    }

    /// Format a 16-byte Mach-O UUID in the canonical 8-4-4-4-12 form.
    ///
    /// Unlike a Windows GUID the bytes are already in wire order, so this is a
    /// plain hex dump with separators.
    fn format_uuid(u: &[u8; 16]) -> String {
        let mut out = String::with_capacity(36);
        for (i, byte) in u.iter().enumerate() {
            if matches!(i, 4 | 6 | 8 | 10) {
                out.push('-');
            }
            let _ = write!(out, "{byte:02X}");
        }
        out
    }

    /// Read a `u32` at `addr` without assuming alignment.
    ///
    /// # Safety
    ///
    /// `addr` must point at four readable bytes.
    const unsafe fn read_u32(addr: usize) -> u32 {
        // SAFETY: per the contract, four readable bytes live at `addr`.
        unsafe { (addr as *const u32).read_unaligned() }
    }

    /// Read 16 bytes at `addr` without assuming alignment.
    ///
    /// # Safety
    ///
    /// `addr` must point at sixteen readable bytes.
    const unsafe fn read_bytes(addr: usize) -> [u8; 16] {
        // SAFETY: per the contract, sixteen readable bytes live at `addr`.
        unsafe { (addr as *const [u8; 16]).read_unaligned() }
    }
}
