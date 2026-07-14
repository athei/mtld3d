//! Shared crash-diagnostic breadcrumb.
//!
//! A 1024-entry ring buffer mapped from the same file on both sides of the
//! Wine PE/unix bridge (`Z:\tmp\mtld3d-crumb.bin` ≡ `/tmp/mtld3d-crumb.bin`).
//! Records into the buffer via `record()` are lock-free (one `fetch_add`
//! on a shared atomic + a volatile entry write). On crash, the per-side
//! crash handler calls `dump_recent(N)` to print the last N entries to
//! the side's stderr, interleaving PE and unix events in `seq` order.
//!
//! The module compiles to no-op stubs unless `cfg(mtld3d_crumb)` is set,
//! so probes have zero cost in production. Crash handlers (which live in
//! the consumer crates, not here) call `dump_recent` unconditionally — when
//! the cfg is off the call is an empty inline.
//!
//! ## Build
//!
//! ```sh
//! MTLD3D_CRUMB=1 make install
//! ```
//!
//! Routed through each consumer's `build.rs` (see `unix/shared/build.rs`,
//! `unix/unix/build.rs`, `windows/d3d9/build.rs`).

#[cfg(not(mtld3d_crumb))]
mod disabled {
    #[inline(always)]
    pub const fn init() {}

    #[inline(always)]
    pub const fn record(_tag: &str, _p0: u64, _p1: u64) {}

    #[inline(always)]
    pub const fn dump_recent(_n: usize) {}

    #[inline(always)]
    pub const fn dump_on_stall_edge(_stalled: bool) {}
}

#[cfg(not(mtld3d_crumb))]
pub use disabled::{dump_on_stall_edge, dump_recent, init, record};

#[cfg(mtld3d_crumb)]
mod enabled {
    use core::sync::atomic::{AtomicBool, AtomicPtr, AtomicU8, AtomicU64, Ordering};

    pub const NUM_ENTRIES: usize = 1024;
    const MAGIC: u64 = 0x4D54_4C44_3344_4342; // "MTLD3DCB" (BE bytes)
    const VERSION: u32 = 1;

    /// On-disk header. First 64 bytes of the mmap.
    #[repr(C)]
    pub struct Header {
        pub magic: u64,
        pub version: u32,
        pub entry_size: u32,
        pub entry_count: u64,
        pub write_index: AtomicU64,
        pub init_marker: AtomicU8,
        _pad: [u8; 31],
    }

    /// Single ring entry. 48 bytes.
    #[repr(C)]
    pub struct Entry {
        pub seq: u64,
        pub tid: u32,
        pub side: u8,
        pub tag_len: u8,
        pub _pad: [u8; 2],
        pub tag: [u8; 16],
        pub payload: [u64; 2],
    }

    const _: () = {
        assert!(size_of::<Header>() == 64);
        assert!(size_of::<Entry>() == 48);
    };

    pub const FILE_SIZE: usize = size_of::<Header>() + NUM_ENTRIES * size_of::<Entry>();

    pub(super) static HEADER_PTR: AtomicPtr<Header> = AtomicPtr::new(core::ptr::null_mut());
    pub(super) static ENTRIES_PTR: AtomicPtr<Entry> = AtomicPtr::new(core::ptr::null_mut());

    #[cfg(unix)]
    pub(super) const SIDE_TAG: u8 = b'U';
    #[cfg(target_family = "windows")]
    pub(super) const SIDE_TAG: u8 = b'P';

    /// Map the breadcrumb file and validate (or initialize) its header.
    ///
    /// Idempotent — first call per process opens and CAS-claims; later
    /// calls observe `HEADER_PTR` already populated and return.
    pub fn init() {
        if !HEADER_PTR.load(Ordering::Acquire).is_null() {
            return;
        }
        // SAFETY: side-specific FFI; failure leaves pointers null and
        // record()/dump_recent() short-circuit.
        let (header_ptr, entries_ptr) = unsafe {
            let Some(p) = platform::map_file() else {
                return;
            };
            p
        };
        // First mapper to claim init_marker (0 → 1) initializes the
        // header; others wait for the marker to reach 2.
        // SAFETY: header_ptr points at the mmap'd Header.
        let hdr = unsafe { &*header_ptr };
        let claimed = hdr.init_marker.load(Ordering::Acquire) == 0
            && hdr
                .init_marker
                .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire)
                .is_ok();
        if claimed {
            let fresh = Header {
                magic: MAGIC,
                version: VERSION,
                // size_of::<Entry>() == 48 — bounded above by the
                // compile-time const_assert near the type definition.
                entry_size: u32::try_from(size_of::<Entry>())
                    .expect("Entry size fits u32 (const_assert-bounded)"),
                entry_count: NUM_ENTRIES as u64,
                write_index: AtomicU64::new(0),
                init_marker: AtomicU8::new(2),
                _pad: [0; 31],
            };
            // SAFETY: header_ptr valid for one Header; we just claimed
            // exclusive init via the CAS.
            unsafe { core::ptr::write(header_ptr, fresh) };
            // SAFETY: entries_ptr valid for NUM_ENTRIES * size_of::<Entry> bytes.
            unsafe {
                core::ptr::write_bytes(
                    entries_ptr.cast::<u8>(),
                    0,
                    NUM_ENTRIES * size_of::<Entry>(),
                );
            }
        } else {
            while hdr.init_marker.load(Ordering::Acquire) != 2 {
                core::hint::spin_loop();
            }
            if hdr.magic != MAGIC || hdr.version != VERSION {
                return;
            }
        }
        ENTRIES_PTR.store(entries_ptr, Ordering::Release);
        HEADER_PTR.store(header_ptr, Ordering::Release);
    }

    /// Record one breadcrumb entry.
    ///
    /// Lock-free: one `fetch_add` + an entry write. Safe to call from any
    /// thread, including signal handlers (no allocation, no locks). Torn
    /// writes on concurrent record paths are detected at read time via the
    /// `seq != slot_position` check in `dump_recent`.
    #[inline]
    pub fn record(tag: &str, p0: u64, p1: u64) {
        let hdr = HEADER_PTR.load(Ordering::Acquire);
        let entries = ENTRIES_PTR.load(Ordering::Acquire);
        if hdr.is_null() || entries.is_null() {
            return;
        }
        // SAFETY: hdr non-null per check; init() mapped FILE_SIZE bytes
        // alive for process lifetime.
        let seq = unsafe { (*hdr).write_index.fetch_add(1, Ordering::Relaxed) };
        let slot = slot_for(seq);
        let n = tag.len().min(16);
        let mut tag_buf = [0u8; 16];
        tag_buf[..n].copy_from_slice(&tag.as_bytes()[..n]);
        // n <= 16 by construction (the `.min(16)` above), so this never panics.
        let tag_len = u8::try_from(n).expect("n is tag.len().min(16) — fits u8");
        let new_entry = Entry {
            seq,
            tid: platform::current_tid(),
            side: SIDE_TAG,
            tag_len,
            _pad: [0; 2],
            tag: tag_buf,
            payload: [p0, p1],
        };
        // SAFETY: slot in bounds (NUM_ENTRIES is a power of two and `slot`
        // is masked). The write races concurrent readers; `dump_recent`
        // skips entries with mismatched `seq`.
        let entry_ptr = unsafe { entries.add(slot) };
        // SAFETY: same justification as above.
        unsafe { core::ptr::write(entry_ptr, new_entry) };
    }

    /// Dump the last `n` entries to the side's stderr sink.
    ///
    /// Async-signal-safe: only `write` (unix) / `WriteFile` (PE), no
    /// allocator, no locks. Tolerates torn writes on the most recent slot
    /// (skipped if `seq` doesn't match the expected position).
    #[inline]
    pub fn dump_recent(n: usize) {
        let hdr = HEADER_PTR.load(Ordering::Acquire);
        let entries = ENTRIES_PTR.load(Ordering::Acquire);
        if hdr.is_null() || entries.is_null() {
            return;
        }
        // SAFETY: hdr non-null per check.
        let cur = unsafe { (*hdr).write_index.load(Ordering::Acquire) };
        if cur == 0 {
            return;
        }
        let count = u64::try_from(n).unwrap_or(u64::MAX);
        let start = cur.saturating_sub(count);
        platform::write_str(b"[trail] last entries:\n");
        for i in start..cur {
            let slot = slot_for(i);
            // SAFETY: slot in bounds (masked).
            let entry_ptr = unsafe { entries.add(slot) };
            // SAFETY: same justification; concurrent record() races
            // detected via the `seq != i` filter below.
            let entry = unsafe { core::ptr::read(entry_ptr) };
            if entry.seq != i {
                continue;
            }
            emit_entry(&entry);
        }
    }

    /// Dump the recent ring to stderr once on each edge of a stall condition.
    ///
    /// An intermittent present stall then self-documents in the log with
    /// no manual timing. `stalled` is whether the current present failed
    /// to acquire its drawable; the ring is written when that flips in
    /// either direction — the rising edge captures the lead-up, the
    /// falling edge captures the whole episode (few entries accrue while
    /// stalled, so the recovery dump still spans it).
    pub fn dump_on_stall_edge(stalled: bool) {
        static STALLED: AtomicBool = AtomicBool::new(false);
        if STALLED.swap(stalled, Ordering::Relaxed) != stalled {
            dump_recent(512);
        }
    }

    #[inline]
    fn slot_for(seq: u64) -> usize {
        let masked = seq & (NUM_ENTRIES as u64 - 1);
        usize::try_from(masked).expect("masked seq fits NUM_ENTRIES range")
    }

    fn emit_entry(e: &Entry) {
        let mut buf = [0u8; 128];
        let mut pos = 0;
        let push = |buf: &mut [u8; 128], pos: &mut usize, bytes: &[u8]| {
            let avail = buf.len() - *pos;
            let take = bytes.len().min(avail);
            buf[*pos..*pos + take].copy_from_slice(&bytes[..take]);
            *pos += take;
        };
        push(&mut buf, &mut pos, b"  [");
        push_hex_u64(&mut buf, &mut pos, e.seq);
        push(&mut buf, &mut pos, b" tid=");
        push_hex_u64(&mut buf, &mut pos, u64::from(e.tid));
        push(&mut buf, &mut pos, b" ");
        push(&mut buf, &mut pos, core::slice::from_ref(&e.side));
        push(&mut buf, &mut pos, b" ");
        let n = (e.tag_len as usize).min(16);
        push(&mut buf, &mut pos, &e.tag[..n]);
        push(&mut buf, &mut pos, b"] p0=");
        push_hex_u64(&mut buf, &mut pos, e.payload[0]);
        push(&mut buf, &mut pos, b" p1=");
        push_hex_u64(&mut buf, &mut pos, e.payload[1]);
        push(&mut buf, &mut pos, b"\n");
        platform::write_str(&buf[..pos]);
    }

    fn push_hex_u64(buf: &mut [u8; 128], pos: &mut usize, v: u64) {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        if *pos + 18 > buf.len() {
            return;
        }
        buf[*pos] = b'0';
        buf[*pos + 1] = b'x';
        *pos += 2;
        for i in (0..16).rev() {
            let nib = ((v >> (i * 4)) & 0xf) as usize;
            buf[*pos] = HEX[nib];
            *pos += 1;
        }
    }

    #[cfg(unix)]
    mod platform {
        use core::ffi::c_void;

        use super::{Entry, FILE_SIZE, Header};

        pub fn current_tid() -> u32 {
            // SAFETY: pthread_self is async-signal-safe.
            let tid = unsafe { libc::pthread_self() };
            // pthread_t is opaque pointer-sized on Apple; we only want a short
            // label, so keep the low 32 bits via a byte slice — a total
            // truncation with no `as` and no panic path.
            let bytes = (tid as usize).to_le_bytes();
            u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
        }

        pub fn write_str(bytes: &[u8]) {
            // SAFETY: write(2) on fd 2 is async-signal-safe.
            unsafe {
                let _ = libc::write(2, bytes.as_ptr().cast::<c_void>(), bytes.len());
            }
        }

        /// Open + size + mmap the breadcrumb file.
        ///
        /// Returns `(header, entries)` pointers or `None` on failure.
        ///
        /// # Safety
        ///
        /// Performs libc FFI; failure paths leave nothing live. Caller must
        /// ensure single-threaded init.
        pub unsafe fn map_file() -> Option<(*mut Header, *mut Entry)> {
            // /tmp/mtld3d-crumb.bin — same path the PE side maps via
            // CreateFileMapping for coherent shared backing.
            let path = c"/tmp/mtld3d-crumb.bin";
            // SAFETY: open(2) with literal flags + path is sound.
            let fd = unsafe {
                libc::open(
                    path.as_ptr(),
                    libc::O_RDWR | libc::O_CREAT,
                    libc::c_int::from(0o644u16),
                )
            };
            if fd < 0 {
                return None;
            }
            let size = libc::off_t::try_from(FILE_SIZE).expect("FILE_SIZE fits off_t");
            // SAFETY: ftruncate(2) on a writable fd.
            if unsafe { libc::ftruncate(fd, size) } != 0 {
                // SAFETY: closing a fd we just opened.
                unsafe { libc::close(fd) };
                return None;
            }
            // SAFETY: mmap(2) on a writable fd with PROT_READ|WRITE.
            let p = unsafe {
                libc::mmap(
                    core::ptr::null_mut(),
                    FILE_SIZE,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED,
                    fd,
                    0,
                )
            };
            // SAFETY: fd lifecycle ends here; mmap holds its own
            // reference to the underlying file.
            unsafe { libc::close(fd) };
            if p == libc::MAP_FAILED {
                return None;
            }
            let header_ptr = p.cast::<Header>();
            // SAFETY: entries start immediately after the header.
            let entries_ptr = unsafe { header_ptr.add(1).cast::<Entry>() };
            Some((header_ptr, entries_ptr))
        }
    }

    #[cfg(target_family = "windows")]
    mod platform {
        use core::ffi::c_void;

        use super::{Entry, FILE_SIZE, Header};

        const GENERIC_READ: u32 = 0x8000_0000;
        const GENERIC_WRITE: u32 = 0x4000_0000;
        const OPEN_ALWAYS: u32 = 4;
        const FILE_ATTRIBUTE_NORMAL: u32 = 0x80;
        const PAGE_READWRITE: u32 = 0x04;
        const FILE_MAP_WRITE: u32 = 0x02;
        const INVALID_HANDLE_VALUE: *mut c_void = !0usize as *mut c_void;
        const STD_ERROR_HANDLE: u32 = 0xFFFF_FFF4;

        unsafe extern "system" {
            fn CreateFileA(
                lp_file_name: *const u8,
                dw_desired_access: u32,
                dw_share_mode: u32,
                lp_security_attributes: *const c_void,
                dw_creation_disposition: u32,
                dw_flags_and_attributes: u32,
                h_template_file: *mut c_void,
            ) -> *mut c_void;
            fn CreateFileMappingA(
                h_file: *mut c_void,
                lp_attributes: *const c_void,
                fl_protect: u32,
                dw_max_size_high: u32,
                dw_max_size_low: u32,
                lp_name: *const u8,
            ) -> *mut c_void;
            fn MapViewOfFile(
                h_file_mapping_object: *mut c_void,
                dw_desired_access: u32,
                dw_file_offset_high: u32,
                dw_file_offset_low: u32,
                dw_number_of_bytes_to_map: u32,
            ) -> *mut c_void;
            fn GetCurrentThreadId() -> u32;
            fn GetStdHandle(n_std_handle: u32) -> *mut c_void;
            fn WriteFile(
                h_file: *mut c_void,
                lp_buffer: *const u8,
                n_number_of_bytes_to_write: u32,
                lp_number_of_bytes_written: *mut u32,
                lp_overlapped: *mut c_void,
            ) -> i32;
        }

        pub fn current_tid() -> u32 {
            // SAFETY: kernel32 stable export.
            unsafe { GetCurrentThreadId() }
        }

        pub fn write_str(bytes: &[u8]) {
            // SAFETY: kernel32 WriteFile on the std error handle. Wine
            // routes fd 2 and STD_ERROR_HANDLE to the same terminal.
            unsafe {
                let h = GetStdHandle(STD_ERROR_HANDLE);
                if h.is_null() || h == INVALID_HANDLE_VALUE {
                    return;
                }
                let mut written = 0u32;
                let _ = WriteFile(
                    h,
                    bytes.as_ptr(),
                    u32::try_from(bytes.len()).unwrap_or(u32::MAX),
                    &raw mut written,
                    core::ptr::null_mut(),
                );
            }
        }

        /// Open + size + mmap the breadcrumb file via Win32.
        ///
        /// # Safety
        ///
        /// Performs Win32 FFI; failure paths leave nothing live. Caller
        /// must ensure single-threaded init.
        pub unsafe fn map_file() -> Option<(*mut Header, *mut Entry)> {
            // Z:\tmp\mtld3d-crumb.bin under Wine = /tmp/mtld3d-crumb.bin
            // — the unix side opens the same backing file via libc::open
            // so both mmaps share the page cache and writes are coherent.
            let path = b"Z:\\tmp\\mtld3d-crumb.bin\0";
            // SAFETY: CreateFileA with literal path + flags.
            let h_file = unsafe {
                CreateFileA(
                    path.as_ptr(),
                    GENERIC_READ | GENERIC_WRITE,
                    0,
                    core::ptr::null(),
                    OPEN_ALWAYS,
                    FILE_ATTRIBUTE_NORMAL,
                    core::ptr::null_mut(),
                )
            };
            if h_file == INVALID_HANDLE_VALUE || h_file.is_null() {
                return None;
            }
            // SAFETY: CreateFileMappingA on a valid file handle.
            let h_mapping = unsafe {
                CreateFileMappingA(
                    h_file,
                    core::ptr::null(),
                    PAGE_READWRITE,
                    0,
                    u32::try_from(FILE_SIZE).expect("crumb file size fits u32"),
                    core::ptr::null(),
                )
            };
            if h_mapping.is_null() {
                return None;
            }
            // SAFETY: MapViewOfFile on a valid mapping handle.
            let view = unsafe {
                MapViewOfFile(
                    h_mapping,
                    FILE_MAP_WRITE,
                    0,
                    0,
                    u32::try_from(FILE_SIZE).expect("crumb file size fits u32"),
                )
            };
            if view.is_null() {
                return None;
            }
            let header_ptr = view.cast::<Header>();
            // SAFETY: header is the first sizeof(Header) bytes; entries
            // immediately after.
            let entries_ptr = unsafe { header_ptr.add(1).cast::<Entry>() };
            Some((header_ptr, entries_ptr))
        }
    }
}

#[cfg(mtld3d_crumb)]
pub use enabled::{dump_on_stall_edge, dump_recent, init, record};

/// Probe macro.
///
/// Records one breadcrumb entry with up to two `u64` payload slots.
/// No-op when `cfg(mtld3d_crumb)` is off.
#[macro_export]
macro_rules! crumb {
    ($tag:expr $(,)?) => {
        $crate::crumb::record($tag, 0, 0)
    };
    ($tag:expr, $p0:expr $(,)?) => {
        $crate::crumb::record($tag, $p0 as u64, 0)
    };
    ($tag:expr, $p0:expr, $p1:expr $(,)?) => {
        $crate::crumb::record($tag, $p0 as u64, $p1 as u64)
    };
}
