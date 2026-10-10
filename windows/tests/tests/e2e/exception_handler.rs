//! The layer's exception handler on a thread whose thread-locals are not set up.
//!
//! Wine starts a thread created with `THREAD_CREATE_FLAGS_SKIP_THREAD_ATTACH`
//! without allocating its TLS block, so `TEB.ThreadLocalStoragePointer` is
//! null on it, as it is on any thread past its thread detach. Copy
//! protectors start such threads and raise exceptions on them that their own
//! handlers resume. `d3d9.dll`'s vectored handler sees every exception in the
//! process first and reports a fault in another module, so it must do that
//! without reading a Rust thread-local: a read through the null pointer
//! faults inside the handler, and the thread never gets back to its own
//! handler.
//!
//! The thread raises an illegal instruction (`ud2`), as a protector's
//! instructions that Rosetta rejects do, and a vectored handler the test
//! appends after the layer's resumes it past the instruction. It runs in a
//! process of its own before any `Direct3DCreate9`, the state a protector
//! runs in at a game's start: the layer's log queue has no thread draining
//! it, and the layer reports the fault, one of the first few in the process.

use core::ffi::{c_char, c_void};
use std::sync::atomic::{AtomicBool, Ordering};

use super::device::{ntdll_export, run_in_private_log_child, running_as};

#[link(name = "kernel32")]
unsafe extern "system" {
    fn AddVectoredExceptionHandler(first: u32, handler: VectoredHandler) -> *mut c_void;
    fn RemoveVectoredExceptionHandler(handle: *mut c_void) -> u32;
    fn GetModuleHandleA(name: *const c_char) -> *mut c_void;
    fn GetCurrentProcess() -> *mut c_void;
    fn TerminateProcess(process: *mut c_void, exit_code: u32) -> i32;
    fn WaitForSingleObject(handle: *mut c_void, millis: u32) -> u32;
    fn CloseHandle(handle: *mut c_void) -> i32;
}

type VectoredHandler = unsafe extern "system" fn(*mut ExceptionPointers) -> i32;
type ThreadStart = unsafe extern "system" fn(*mut c_void) -> u32;
/// `NtCreateThreadEx`, which takes the thread-creation flags `CreateThread` does not.
type NtCreateThreadExFn = unsafe extern "system" fn(
    *mut *mut c_void,
    u32,
    *mut c_void,
    *mut c_void,
    ThreadStart,
    *mut c_void,
    u32,
    usize,
    usize,
    usize,
    *mut c_void,
) -> i32;

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

const EXCEPTION_CONTINUE_SEARCH: i32 = 0;
const EXCEPTION_CONTINUE_EXECUTION: i32 = -1;
const STATUS_ILLEGAL_INSTRUCTION: u32 = 0xC000_001D;
/// The bytes of `ud2`, the instruction the thread raises its exception with.
const UD2: [u8; 2] = [0x0F, 0x0B];
/// Offset of the instruction pointer (`Eip`) in an i386 `CONTEXT`.
#[cfg(target_arch = "x86")]
const CONTEXT_PC_OFFSET: usize = 0xB8;
/// Offset of the instruction pointer (`Rip`) in an `x86_64` `CONTEXT`.
#[cfg(target_arch = "x86_64")]
const CONTEXT_PC_OFFSET: usize = 0xF8;
const THREAD_ALL_ACCESS: u32 = 0x001F_FFFF;
/// Start the thread without calling any DLL's thread attach or allocating its TLS block.
const THREAD_CREATE_FLAGS_SKIP_THREAD_ATTACH: u32 = 0x2;
const WAIT_OBJECT_0: u32 = 0;
/// How long the thread may take to get past its exception before it counts as hung, in ms.
const THREAD_LIMIT_MS: u32 = 10_000;
/// The exit code the child ends with when the thread hung, libtest's failure code.
const HUNG_EXIT_CODE: u32 = 101;

/// The copy of the test executable that runs the thread, with no interface created before it.
const CHILD_NAME: &str = "exception-without-tls.exe";

/// Set by the thread once its exception was resumed and it ran on past it.
static RESUMED: AtomicBool = AtomicBool::new(false);

/// An exception on a thread without thread-locals reaches the thread's own handler and resumes.
///
/// Before `Direct3DCreate9`, on a thread started without thread attach, the
/// layer's handler reports the illegal instruction and passes it on, the
/// test's handler resumes it, and the thread finishes within ten seconds.
/// A handler that reads a thread-local there faults inside `d3d9.dll`
/// instead, which ends the process or, under a protector that resumes the
/// fault, loops the thread.
#[test]
fn an_exception_on_a_thread_without_thread_locals_reaches_its_own_handler() {
    if running_as(CHILD_NAME) {
        exception_without_thread_locals_workload();
        return;
    }
    run_in_private_log_child(
        CHILD_NAME,
        "exception_handler::an_exception_on_a_thread_without_thread_locals_reaches_its_own_handler",
        "warn",
    );
}

/// Start the thread without thread attach, wait for it, and check it ran past its exception.
///
/// A thread that does not finish ends the process with a failing exit code
/// rather than a panic: the panic's backtrace would take the loader lock,
/// which the layer's handler takes too on every pass of a looping thread.
fn exception_without_thread_locals_workload() {
    // SAFETY: plain kernel32 lookup by a NUL-terminated name.
    let d3d9 = unsafe { GetModuleHandleA(c"d3d9.dll".as_ptr()) };
    assert!(
        !d3d9.is_null(),
        "d3d9.dll is not loaded, so its exception handler is not installed"
    );
    // SAFETY: appending (first = 0) behind the layer's handler, which
    // registered itself first; the handler stays valid for the process.
    let handler = unsafe { AddVectoredExceptionHandler(0, resume_past_ud2) };
    assert!(!handler.is_null(), "AddVectoredExceptionHandler");
    // SAFETY: the export has the `NtCreateThreadEx` signature declared above.
    let create: NtCreateThreadExFn =
        unsafe { core::mem::transmute(ntdll_export(c"NtCreateThreadEx")) };
    let mut thread = core::ptr::null_mut();
    // SAFETY: plain kernel32 call answering this process's pseudo-handle.
    let process = unsafe { GetCurrentProcess() };
    // SAFETY: `thread` receives the new handle; no attributes, the default
    // stack, and a start routine that lives as long as the process.
    let status = unsafe {
        create(
            &raw mut thread,
            THREAD_ALL_ACCESS,
            core::ptr::null_mut(),
            process,
            raise_and_resume,
            core::ptr::null_mut(),
            THREAD_CREATE_FLAGS_SKIP_THREAD_ATTACH,
            0,
            0,
            0,
            core::ptr::null_mut(),
        )
    };
    assert_eq!(status, 0, "NtCreateThreadEx");
    // SAFETY: `thread` is the live handle NtCreateThreadEx returned.
    let waited = unsafe { WaitForSingleObject(thread, THREAD_LIMIT_MS) };
    if waited != WAIT_OBJECT_0 {
        eprintln!(
            "the thread without thread-locals did not finish within {THREAD_LIMIT_MS} ms \
             (wait result {waited:#x}); its exception never got back to its own handler"
        );
        // SAFETY: the current process's pseudo-handle; the documented
        // self-terminate form.
        unsafe { TerminateProcess(process, HUNG_EXIT_CODE) };
    }
    // SAFETY: closing the handle NtCreateThreadEx returned, exactly once.
    unsafe { CloseHandle(thread) };
    // SAFETY: `handler` came from AddVectoredExceptionHandler above.
    assert_ne!(unsafe { RemoveVectoredExceptionHandler(handler) }, 0);
    assert!(
        RESUMED.load(Ordering::Acquire),
        "the thread ended without running past its exception"
    );
}

/// The thread: clear the TLS array pointer, raise the exception, restore the pointer.
///
/// Wine already leaves the pointer null on a thread started without thread
/// attach; clearing it here keeps the test's precondition whatever the Wine.
/// Nothing in this function or in [`resume_past_ud2`] reads a thread-local.
extern "system" fn raise_and_resume(_parameter: *mut c_void) -> u32 {
    let saved = tls_array();
    // SAFETY: the thread's own TEB field; nothing on this thread reads a
    // thread-local until it is restored below.
    unsafe { set_tls_array(0) };
    // SAFETY: `ud2` raises an illegal-instruction exception, which
    // `resume_past_ud2` resumes at the next instruction.
    unsafe { core::arch::asm!("ud2") };
    // SAFETY: putting back the value this thread's TEB held.
    unsafe { set_tls_array(saved) };
    RESUMED.store(true, Ordering::Release);
    0
}

/// Resume the thread's `ud2` at the instruction after it; pass every other exception on.
extern "system" fn resume_past_ud2(ep: *mut ExceptionPointers) -> i32 {
    // SAFETY: the dispatcher hands a valid EXCEPTION_POINTERS for the call.
    let record = unsafe { (*ep).record };
    // SAFETY: as above.
    let context = unsafe { (*ep).context };
    // SAFETY: the record pointer is valid for the handler's duration.
    if unsafe { (*record).code } != STATUS_ILLEGAL_INSTRUCTION {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    // SAFETY: the context is this architecture's `CONTEXT`, which holds the
    // instruction pointer at `CONTEXT_PC_OFFSET`.
    let pc_slot =
        unsafe { context.cast::<u8>().add(CONTEXT_PC_OFFSET) }.cast::<[u8; size_of::<usize>()]>();
    // SAFETY: inside the context, as above; a byte array has no alignment.
    let pc = usize::from_ne_bytes(unsafe { pc_slot.read() });
    // SAFETY: the faulting instruction's bytes, mapped since it just ran.
    let bytes = unsafe { (pc as *const [u8; 2]).read_unaligned() };
    if bytes != UD2 {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    // SAFETY: the same slot; the thread resumes past the two-byte `ud2`.
    unsafe { pc_slot.write((pc + UD2.len()).to_ne_bytes()) };
    EXCEPTION_CONTINUE_EXECUTION
}

/// `TEB.ThreadLocalStoragePointer` of the calling thread, read as thread-local accesses read it.
#[cfg(target_arch = "x86")]
fn tls_array() -> usize {
    let value: usize;
    // SAFETY: `fs` addresses the calling thread's TEB on i386 Windows; 0x2c is the field.
    unsafe {
        core::arch::asm!("mov {}, dword ptr fs:[0x2c]", out(reg) value, options(nostack, readonly));
    }
    value
}

/// `TEB.ThreadLocalStoragePointer` of the calling thread, read as thread-local accesses read it.
#[cfg(target_arch = "x86_64")]
fn tls_array() -> usize {
    let value: usize;
    // SAFETY: `gs` addresses the calling thread's TEB on x86_64 Windows; 0x58 is the field.
    unsafe {
        core::arch::asm!("mov {}, qword ptr gs:[0x58]", out(reg) value, options(nostack, readonly));
    }
    value
}

/// Write the calling thread's `TEB.ThreadLocalStoragePointer`.
///
/// # Safety
/// Nothing on the calling thread may read a thread-local while the value is
/// not the one the loader set up.
#[cfg(target_arch = "x86")]
unsafe fn set_tls_array(value: usize) {
    // SAFETY: the calling thread's own TEB field, as `tls_array` reads it.
    unsafe {
        core::arch::asm!("mov dword ptr fs:[0x2c], {}", in(reg) value, options(nostack));
    }
}

/// Write the calling thread's `TEB.ThreadLocalStoragePointer`.
///
/// # Safety
/// Nothing on the calling thread may read a thread-local while the value is
/// not the one the loader set up.
#[cfg(target_arch = "x86_64")]
unsafe fn set_tls_array(value: usize) {
    // SAFETY: the calling thread's own TEB field, as `tls_array` reads it.
    unsafe {
        core::arch::asm!("mov qword ptr gs:[0x58], {}", in(reg) value, options(nostack));
    }
}
