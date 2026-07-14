//! Always-on PE-side crash diagnostics.
//!
//! Installed once from `init_logger()` during `DllMain` `PROCESS_ATTACH`.
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
    sync::atomic::{AtomicBool, AtomicPtr, Ordering},
};

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
const STD_ERROR_HANDLE: u32 = 0xFFFF_FFF4;
const GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS: u32 = 4;

static INSTALLED: AtomicBool = AtomicBool::new(false);
static D3D9_HMODULE: AtomicPtr<c_void> = AtomicPtr::new(core::ptr::null_mut());

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
    fn GetModuleHandleExA(flags: u32, module_name: *const u8, out: *mut *mut c_void) -> i32;
    fn GetStdHandle(handle: u32) -> *mut c_void;
    fn WriteFile(
        h_file: *mut c_void,
        buf: *const u8,
        n: u32,
        written: *mut u32,
        overlapped: *mut c_void,
    ) -> i32;
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
    // SAFETY: kernel32 export; safe to call from DllMain.
    unsafe {
        RtlAddVectoredExceptionHandler(1, handler);
    }
    install_panic_hook();
}

fn install_panic_hook() {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        write_stderr(b"[mtld3d::d3d9] PANIC - dumping crumb trail:\n");
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
    write_stderr(&buf[..pos]);
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

    if !(always_fatal || (possibly_fatal && fault_in_our_dll(addr))) {
        return EXCEPTION_CONTINUE_SEARCH;
    }

    // Diagnostic-only. Do NOT terminate — let SEH unwind so WoW's
    // unhandled-exception filter still gets to write `Crash.txt`.
    emit_fatal(code, addr);
    crumb::dump_recent(32);
    EXCEPTION_CONTINUE_SEARCH
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

fn emit_fatal(code: u32, addr: *mut c_void) {
    let mut buf = [0u8; 128];
    let mut pos = 0;
    push(&mut buf, &mut pos, b"[mtld3d::d3d9] FATAL: code=");
    push_hex(&mut buf, &mut pos, u64::from(code));
    push(&mut buf, &mut pos, b" addr=");
    push_hex(&mut buf, &mut pos, addr as usize as u64);
    push(&mut buf, &mut pos, b"\n");
    write_stderr(&buf[..pos]);
}

fn write_stderr(bytes: &[u8]) {
    // SAFETY: kernel32 stable export.
    let h = unsafe { GetStdHandle(STD_ERROR_HANDLE) };
    if h.is_null() {
        return;
    }
    let mut written = 0u32;
    // SAFETY: WriteFile on STD_ERROR_HANDLE with a valid buffer slice.
    unsafe {
        let _ = WriteFile(
            h,
            bytes.as_ptr(),
            u32::try_from(bytes.len()).unwrap_or(u32::MAX),
            &raw mut written,
            core::ptr::null_mut(),
        );
    }
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
