//! Always-on PE-side crash diagnostics.
//!
//! Installed once from `init_logger()` during `DllMain` `PROCESS_ATTACH`, and
//! the vectored handler is removed again on a `PROCESS_DETACH` the process
//! survives: a VEH registration is a process-wide pointer into this image, and
//! launchers and benchmarks routinely `LoadLibrary` d3d9, probe the caps and
//! `FreeLibrary` it before carrying on. Left behind, the registration turns
//! the process's next exception (C++ throws included) into an instruction
//! fetch from unmapped memory, which raises the next exception, which reaches
//! the same stale entry, until the main thread's stack is gone. The panic hook
//! needs no such care: it lives in this cdylib's own `std` and nothing else in
//! the process can reach it once the image is gone.
//!
//! Two pieces, both diagnostic-only — they print the crumb trail and
//! delegate termination to the normal SEH / `panic_abort` paths:
//!
//! 1. A Vectored Exception Handler that filters to truly-fatal `NTSTATUS`
//!    codes AND faults that originate inside our `d3d9.dll` image. On a
//!    match it writes a one-line FATAL banner + dumps the shared crumb
//!    trail, then returns `EXCEPTION_CONTINUE_SEARCH` — `WoW`'s
//!    unhandled-exception filter still gets a chance to write
//!    `Crash.txt` before the process dies.
//!
//! 2. A panic hook. `panic = "abort"` on Windows calls `__fastfail`,
//!    which bypasses VEH entirely, so the panic path needs its own
//!    diagnostic — we just dump the crumb trail and chain to the
//!    previous (default) hook. The default hook prints the usual
//!    `thread '…' panicked at …` line and (if `RUST_BACKTRACE=1`) a
//!    backtrace. Empty backtraces under Wine are accepted as-is.
//!
//! Faults in other modules (game code, `ClientExtensions.dll` `VMProtect`
//! probes, system DLLs) pass through to SEH normally, so `VMProtect`'s
//! first-chance recovery still works and `WoW` continues.
//!
//! `RUST_BACKTRACE=1` is set here (if unset) so the default panic hook
//! attempts a backtrace.

use core::{
    ffi::c_void,
    sync::atomic::{AtomicBool, AtomicPtr, AtomicU32, Ordering},
};

use mtld3d_core::address_space::FreeSpace;
use mtld3d_shared::crumb;

// NTSTATUS codes the handler filters on.
const STATUS_ACCESS_VIOLATION: u32 = 0xC000_0005;
const STATUS_DATATYPE_MISALIGNMENT: u32 = 0x8000_0002;
const STATUS_HEAP_CORRUPTION: u32 = 0xC000_0374;
const STATUS_PRIVILEGED_INSTRUCTION: u32 = 0xC000_0096;
const STATUS_ILLEGAL_INSTRUCTION: u32 = 0xC000_001D;
const STATUS_STACK_BUFFER_OVERRUN: u32 = 0xC000_0409;
const STATUS_ASSERTION_FAILURE: u32 = 0xC000_0420;

const EXCEPTION_CONTINUE_SEARCH: i32 = 0;
pub const GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS: u32 = 4;
/// `GetModuleHandleEx` flag that keeps the module loaded until the process ends.
pub const GET_MODULE_HANDLE_EX_FLAG_PIN: u32 = 1;

static INSTALLED: AtomicBool = AtomicBool::new(false);
static D3D9_HMODULE: AtomicPtr<c_void> = AtomicPtr::new(core::ptr::null_mut());
/// The registration `RtlAddVectoredExceptionHandler` handed back, for [`uninstall`].
static VEH_HANDLE: AtomicPtr<c_void> = AtomicPtr::new(core::ptr::null_mut());
/// Faults outside our image reported so far; the report stops after a few.
static FOREIGN_REPORTS: AtomicU32 = AtomicU32::new(0);

/// How many faults outside our image get a line in the log.
///
/// Some programs probe with deliberate access violations, and every one
/// of those would otherwise cost a module lookup and a crumb dump.
const FOREIGN_REPORT_LIMIT: u32 = 4;

/// `MEMORY_BASIC_INFORMATION`, one region of the address space.
#[repr(C)]
struct MemoryBasicInformation {
    base_address: *mut c_void,
    allocation_base: *mut c_void,
    allocation_protect: u32,
    region_size: usize,
    state: u32,
    protect: u32,
    kind: u32,
}

const MEM_FREE: u32 = 0x1_0000;

/// Free address space of this process: the sum of the usable free regions and the largest.
///
/// The free figure the address-space watch and the crash lines report. It is
/// not `GlobalMemoryStatusEx`'s `ullAvailVirtual`, which Wine computes as the
/// total minus the process working set (see `mtld3d_core::address_space`).
/// Free space that no single allocation can use is what fails a DLL load or a
/// game's streaming block long before the total runs out, so both come from
/// the same walk. Allocation-free, so the exception handler can call it.
pub fn free_space() -> FreeSpace {
    let mut space = FreeSpace::new();
    walk_regions(|addr, info| {
        space.add_region(info.state == MEM_FREE, addr as u64, info.region_size as u64);
    });
    space
}

/// Visit every region of the address space in address order, with its base address.
///
/// One `VirtualQuery` per region (a few thousand calls), stopping where the
/// query fails or the address would wrap.
fn walk_regions(mut visit: impl FnMut(usize, &MemoryBasicInformation)) {
    let mut addr = 0usize;
    loop {
        let mut info = MemoryBasicInformation {
            base_address: core::ptr::null_mut(),
            allocation_base: core::ptr::null_mut(),
            allocation_protect: 0,
            region_size: 0,
            state: 0,
            protect: 0,
            kind: 0,
        };
        // SAFETY: kernel32 export filling a struct of the size passed.
        let got = unsafe {
            VirtualQuery(
                addr as *const c_void,
                &raw mut info,
                size_of::<MemoryBasicInformation>(),
            )
        };
        if got == 0 || info.region_size == 0 {
            break;
        }
        visit(addr, &info);
        let Some(next) = addr.checked_add(info.region_size) else {
            break;
        };
        addr = next;
    }
}

const MEM_COMMIT: u32 = 0x1000;
const MEM_RESERVE: u32 = 0x2000;
const MEM_IMAGE: u32 = 0x100_0000;
const MEM_MAPPED: u32 = 0x4_0000;

/// A summary of the address space: region counts by state and the largest regions.
///
/// One line of text for the log, built when the free space crosses one of
/// the watch's thresholds, so the log names who owns the space (image,
/// mapped file, private commit, private reserve) and where the biggest holes
/// are.
pub fn address_space_map() -> String {
    let mut regions: Vec<(usize, usize, u32, u32)> = Vec::new();
    walk_regions(|addr, info| regions.push((addr, info.region_size, info.state, info.kind)));
    let mut free = 0usize;
    let mut committed = 0usize;
    let mut reserved = 0usize;
    let mut image = 0usize;
    let mut mapped = 0usize;
    let mut free_holes = 0usize;
    let mut used_regions = 0usize;
    for &(_, size, state, kind) in &regions {
        if state == MEM_FREE {
            free += size;
            free_holes += 1;
            continue;
        }
        used_regions += 1;
        if kind == MEM_IMAGE {
            image += size;
        } else if kind == MEM_MAPPED {
            mapped += size;
        } else if state == MEM_COMMIT {
            committed += size;
        } else if state == MEM_RESERVE {
            reserved += size;
        }
    }
    let mut out = format!(
        "regions used={used_regions} free_holes={free_holes}; MiB: free={} image={} mapped={} \
         private_commit={} private_reserve={}; largest used:",
        free >> 20,
        image >> 20,
        mapped >> 20,
        committed >> 20,
        reserved >> 20
    );
    let mut used: Vec<_> = regions
        .iter()
        .filter(|r| r.2 != MEM_FREE)
        .copied()
        .collect();
    used.sort_by_key(|r| core::cmp::Reverse(r.1));
    for (base, size, state, kind) in used.into_iter().take(12) {
        let what = if kind == MEM_IMAGE {
            "image"
        } else if kind == MEM_MAPPED {
            "mapped"
        } else if state == MEM_COMMIT {
            "commit"
        } else {
            "reserve"
        };
        let _ = core::fmt::Write::write_fmt(
            &mut out,
            format_args!(" {base:#010x}+{}M({what})", size >> 20),
        );
    }
    let mut holes: Vec<_> = regions
        .iter()
        .filter(|r| r.2 == MEM_FREE)
        .copied()
        .collect();
    holes.sort_by_key(|r| core::cmp::Reverse(r.1));
    out.push_str("; largest holes:");
    for (base, size, _, _) in holes.into_iter().take(6) {
        let _ =
            core::fmt::Write::write_fmt(&mut out, format_args!(" {base:#010x}+{}M", size >> 20));
    }
    out
}

#[repr(C)]
struct ExceptionRecord {
    code: u32,
    flags: u32,
    nested: *mut Self,
    address: *mut c_void,
    n_params: u32,
    information: [usize; 15],
}

#[repr(C)]
struct ExceptionPointers {
    record: *mut ExceptionRecord,
    context: *mut c_void,
}

type VectoredHandler = extern "system" fn(*mut ExceptionPointers) -> i32;

unsafe extern "system" {
    fn RtlAddVectoredExceptionHandler(first: u32, handler: VectoredHandler) -> *mut c_void;
    fn RtlRemoveVectoredExceptionHandler(handle: *mut c_void) -> u32;
    pub fn GetModuleHandleExA(flags: u32, module_name: *const u8, out: *mut *mut c_void) -> i32;
    fn GetModuleFileNameA(module: *mut c_void, filename: *mut u8, size: u32) -> u32;
    fn VirtualQuery(
        address: *const c_void,
        buffer: *mut MemoryBasicInformation,
        length: usize,
    ) -> usize;
}

/// Install the VEH and the panic hook. Idempotent.
///
/// `d3d9_module` is the `HMODULE` passed to `DllMain` — saved so we can
/// later check whether a faulting PC lives in our DLL image.
pub fn install(d3d9_module: *mut c_void) {
    if INSTALLED.swap(true, Ordering::AcqRel) {
        return;
    }

    if std::env::var_os("RUST_BACKTRACE").is_none() {
        // `full` over `1` because Wine's unwinder returns very few
        // frames; the `1`-mode elision of std-internal frames usually
        // strips the result to empty. `full` keeps everything captured.
        // SAFETY: DllMain runs single-threaded on the main thread
        // before any of our spawned threads exist.
        unsafe { std::env::set_var("RUST_BACKTRACE", "full") };
    }

    D3D9_HMODULE.store(d3d9_module, Ordering::Release);
    // SAFETY: ntdll export; safe to call from DllMain.
    let veh = unsafe { RtlAddVectoredExceptionHandler(1, handler) };
    VEH_HANDLE.store(veh, Ordering::Release);
    install_panic_hook();
}

/// Remove the VEH before the image goes away. Idempotent.
///
/// Called from `DllMain` `PROCESS_DETACH` when no device was created: a
/// `FreeLibrary` the process survives, or the exit of a process that never
/// created one. Once a device exists the image is pinned, so the only
/// detach left is process exit, which terminates the process instead and
/// never gets here.
pub fn uninstall() {
    if !INSTALLED.swap(false, Ordering::AcqRel) {
        return;
    }
    let veh = VEH_HANDLE.swap(core::ptr::null_mut(), Ordering::AcqRel);
    if veh.is_null() {
        return;
    }
    // SAFETY: `veh` is the live registration `install` stored; ntdll export.
    unsafe { RtlRemoveVectoredExceptionHandler(veh) };
}

fn install_panic_hook() {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        // A panic ends the process: its report may open the early log.
        crate::log_sink::write_crash(b"[mtld3d::d3d9] PANIC - dumping crumb trail:\n");
        crumb::dump_recent(32);
        emit_image_bases();
        // Chain to the default hook so the usual "thread '…' panicked
        // at …" line + std backtrace (possibly empty on Wine) still
        // appears. Our crumb dump is the load-bearing diagnostic;
        // backtrace quality is best-effort.
        prev(info);
    }));
}

/// Print the runtime load bases for our own DLL(s).
///
/// Wine's `dbghelp` rarely loads PDBs end-to-end, so symbolicated frames
/// inside d3d9.dll come out as `<unknown>` in std's backtrace. Knowing the
/// load base lets you compute RVAs (`pc - base`) and resolve them
/// externally:
///
/// ```sh
/// llvm-symbolizer --obj=windows/target/i686-pc-windows-msvc/release/d3d9.dll \
///                 --pdb=windows/target/i686-pc-windows-msvc/release/d3d9.pdb \
///                 <RVA>
/// ```
fn emit_image_bases() {
    let our = D3D9_HMODULE.load(Ordering::Acquire);
    if our.is_null() {
        return;
    }
    let mut buf = [0u8; 64];
    let mut pos = 0;
    push(&mut buf, &mut pos, b"[mtld3d::d3d9] d3d9.dll base=");
    push_hex(&mut buf, &mut pos, our as usize as u64);
    push(&mut buf, &mut pos, b"\n");
    crate::log_sink::write_crash(&buf[..pos]);
}

extern "system" fn handler(ep: *mut ExceptionPointers) -> i32 {
    if ep.is_null() {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    // SAFETY: ep non-null per check; kernel-supplied for handler lifetime.
    let rec = unsafe { (*ep).record };
    if rec.is_null() {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    // SAFETY: rec non-null per check.
    let code = unsafe { (*rec).code };
    // SAFETY: rec non-null per check.
    let addr = unsafe { (*rec).address };

    let always_fatal = matches!(
        code,
        STATUS_HEAP_CORRUPTION | STATUS_STACK_BUFFER_OVERRUN | STATUS_ASSERTION_FAILURE
    );
    let possibly_fatal = matches!(
        code,
        STATUS_ACCESS_VIOLATION
            | STATUS_DATATYPE_MISALIGNMENT
            | STATUS_PRIVILEGED_INSTRUCTION
            | STATUS_ILLEGAL_INSTRUCTION
    );

    // Diagnostic-only. Do NOT terminate — let SEH unwind so the game's own
    // unhandled-exception filter still gets to write its crash report.
    // Only an always-fatal code is known to end the process here; a fault in
    // our image is still a first chance a frame up the stack may handle.
    if always_fatal || (possibly_fatal && fault_in_our_dll(addr)) {
        let line = fatal_line(code, addr);
        if always_fatal {
            crate::log_sink::write_crash(line.bytes());
        } else {
            crate::log_sink::write_fault(line.bytes());
        }
        crumb::dump_recent(32);
    } else if possibly_fatal {
        report_foreign_fault(code, addr);
    }
    EXCEPTION_CONTINUE_SEARCH
}

/// Note a fault in someone else's code, with the module it landed in.
///
/// Wine names the address of an unhandled fault but not the module, and a
/// launcher-spawned game leaves no way to run the debugger afterwards. The
/// first few faults get the owning module's path and our most recent API
/// crumbs, which is usually enough to tell "the game dereferenced what we
/// returned" from "unrelated".
fn report_foreign_fault(code: u32, addr: *mut c_void) {
    if FOREIGN_REPORTS.fetch_add(1, Ordering::AcqRel) >= FOREIGN_REPORT_LIMIT {
        return;
    }
    let mut module: *mut c_void = core::ptr::null_mut();
    // SAFETY: kernel32 export; `addr` is only used as a lookup key.
    let found = unsafe {
        GetModuleHandleExA(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
            addr.cast::<u8>(),
            &raw mut module,
        )
    };
    let mut path = [0u8; 260];
    let path_len = if found == 0 || module.is_null() {
        0
    } else {
        // SAFETY: kernel32 export writing at most `path.len()` bytes.
        unsafe { GetModuleFileNameA(module, path.as_mut_ptr(), 260) }
    };
    let mut buf = [0u8; 400];
    let mut pos = 0;
    push(
        &mut buf,
        &mut pos,
        b"[mtld3d::d3d9] fault outside d3d9.dll: code=",
    );
    push_hex(&mut buf, &mut pos, u64::from(code));
    push(&mut buf, &mut pos, b" addr=");
    push_hex(&mut buf, &mut pos, addr as usize as u64);
    push(&mut buf, &mut pos, b" module=");
    if path_len == 0 {
        push(&mut buf, &mut pos, b"?");
    } else {
        let n = (path_len as usize).min(path.len());
        push(&mut buf, &mut pos, &path[..n]);
        push(&mut buf, &mut pos, b" base=");
        push_hex(&mut buf, &mut pos, module as usize as u64);
    }
    push_free_space(&mut buf, &mut pos);
    push(&mut buf, &mut pos, b"\n");
    // A first chance: the fault may be handled, so the line opens no log.
    crate::log_sink::write_fault(&buf[..pos]);
    crumb::dump_recent(16);
}

/// Append the free address space so a crash line carries it.
fn push_free_space(buf: &mut [u8], pos: &mut usize) {
    let space = free_space();
    push(buf, pos, b" free_mib=");
    push_hex(buf, pos, space.total_mib());
    push(buf, pos, b" largest_free_mib=");
    push_hex(buf, pos, space.largest_mib());
}

fn fault_in_our_dll(addr: *mut c_void) -> bool {
    let our = D3D9_HMODULE.load(Ordering::Acquire);
    if our.is_null() {
        return false;
    }
    let mut module: *mut c_void = core::ptr::null_mut();
    // SAFETY: GetModuleHandleExA with FROM_ADDRESS returns the HMODULE
    // containing `addr` without incrementing its refcount when paired
    // with UNCHANGED_REFCOUNT (flag 2). Passing just FROM_ADDRESS
    // (flag 4) adds a refcount — the leak is acceptable because this
    // only runs on a fault the process does not survive.
    let ok = unsafe {
        GetModuleHandleExA(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
            addr.cast::<u8>(),
            &raw mut module,
        )
    };
    ok != 0 && module == our
}

/// The `FATAL` line for an exception in our image or with a fatal code, in a fixed buffer.
struct FatalLine {
    buf: [u8; 160],
    len: usize,
}

impl FatalLine {
    /// The line's bytes.
    fn bytes(&self) -> &[u8] {
        &self.buf[..self.len]
    }
}

fn fatal_line(code: u32, addr: *mut c_void) -> FatalLine {
    let mut buf = [0u8; 160];
    let mut pos = 0;
    push(&mut buf, &mut pos, b"[mtld3d::d3d9] FATAL: code=");
    push_hex(&mut buf, &mut pos, u64::from(code));
    push(&mut buf, &mut pos, b" addr=");
    push_hex(&mut buf, &mut pos, addr as usize as u64);
    push_free_space(&mut buf, &mut pos);
    push(&mut buf, &mut pos, b"\n");
    FatalLine { buf, len: pos }
}

fn push(buf: &mut [u8], pos: &mut usize, bytes: &[u8]) {
    let avail = buf.len() - *pos;
    let take = bytes.len().min(avail);
    buf[*pos..*pos + take].copy_from_slice(&bytes[..take]);
    *pos += take;
}

fn push_hex(buf: &mut [u8], pos: &mut usize, v: u64) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    if *pos + 18 > buf.len() {
        return;
    }
    buf[*pos] = b'0';
    buf[*pos + 1] = b'x';
    *pos += 2;
    for i in (0..16).rev() {
        let nib = usize::try_from((v >> (i * 4)) & 0xf).expect("4-bit nibble fits usize");
        buf[*pos] = HEX[nib];
        *pos += 1;
    }
}
