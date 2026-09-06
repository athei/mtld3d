//! Self-test for the crash handler's fault report.
//!
//! The handler reads the faulting frame out of `__darwin_mcontext64` at byte
//! offsets hand-derived per architecture, which nothing else checks: a wrong one
//! prints zeros in the single report a crash leaves behind. A signal handler can
//! only be exercised by taking the signal, so the test re-executes the binary,
//! faults on a known address, and asserts the child died through the handler's
//! own exit path with a decodable banner, PC, stack pointer and argument labels.
//!
//! The second test faults inside libsystem instead: a fault the handler does
//! not own is handed back to the signal's default action, and the report it
//! writes first has to name the image with an offset, the thread, and our
//! frames on the faulting stack. The third traps in our own code, which is
//! fatal like a fault. The fourth takes the same foreign fault on a thread
//! that answers with no TEB, where handing it back would fault inside Wine,
//! and pins that the process ends here with the first fault named.
//!
//! Wine is not in a unit test's process, so the branch that asks it for the
//! calling thread's TEB is exercised through a stand-in `NtCurrentTeb` stored
//! where the install-time lookup would have put one: what the handler reads is
//! a function address either way, so the stand-in exercises the same branch.

use core::{ffi::c_void, ptr};
use std::{os::unix::process::ExitStatusExt as _, sync::atomic::Ordering};

/// Set in the re-executed child so it faults instead of asserting.
///
/// A signal handler can only be exercised by actually taking the signal,
/// which terminates the process, so the test spawns itself.
const SELFTEST_ENV: &str = "MTLD3D_CRASH_SELFTEST";

/// Set in the re-executed child so it faults in someone else's code.
const FOREIGN_SELFTEST_ENV: &str = "MTLD3D_CRASH_FOREIGN_SELFTEST";

/// Set in the re-executed child so it executes a trap instruction.
const ILL_SELFTEST_ENV: &str = "MTLD3D_CRASH_ILL_SELFTEST";

/// Set in the re-executed child so its foreign fault has no TEB behind it.
const NO_TEB_SELFTEST_ENV: &str = "MTLD3D_CRASH_NO_TEB_SELFTEST";

/// The byte a stand-in TEB pointer points at; only its address is ever used.
static FAKE_TEB: u8 = 0;

/// Stands in for Wine's `NtCurrentTeb` on a thread Wine never created.
extern "C" fn no_teb_stub() -> *mut c_void {
    ptr::null_mut()
}

/// Stands in for Wine's `NtCurrentTeb` on a thread Wine created.
extern "C" fn teb_stub() -> *mut c_void {
    (&raw const FAKE_TEB).cast_mut().cast::<c_void>()
}

/// Install a stand-in `NtCurrentTeb`, replacing whatever the lookup found.
///
/// The handler reads one function address out of a static, so a stand-in put
/// there reaches exactly the code a resolved `ntdll.so` would.
fn pin_wine_teb(entry: extern "C" fn() -> *mut c_void) {
    super::WINE_CURRENT_TEB.store(entry as usize, Ordering::Relaxed);
}

/// The bad pointer the child dereferences.
///
/// Below every mapping and page-aligned nowhere useful, so the fault is a
/// read of exactly this address and the report's `fault=` line pins it.
const BAD_ADDR: usize = 0xdead_beef;

/// Fault through a garbage object pointer, the shape the dump decodes.
///
/// `extern "C"` and never inlined so the argument really travels in the
/// first-argument register and the return address really is a caller frame.
#[inline(never)]
extern "C" fn deref_this(this: *const u64) -> u64 {
    // SAFETY: deliberately unsound; this is the fault under test, taken in
    // a child process that never returns from the handler.
    unsafe { this.read() }
}

/// The saved-register decode names the faulting frame on the running arch.
///
/// Guards the `mcontext` offsets, which are hand-derived per arch and have
/// no compiler check: a wrong one silently reports zeros in the crash
/// report, exactly when nobody can re-run the crash.
#[test]
fn fault_report_decodes_registers() {
    if std::env::var_os(SELFTEST_ENV).is_some() {
        super::install();
        let _ = deref_this(BAD_ADDR as *const u64);
        unreachable!("the read above must fault");
    }

    let exe = std::env::current_exe().expect("test binary path");
    let out = std::process::Command::new(exe)
        .args([
            "--exact",
            "crash::tests::fault_report_decodes_registers",
            "--nocapture",
        ])
        .env(SELFTEST_ENV, "1")
        .output()
        .expect("re-exec the test binary");
    let report = String::from_utf8_lossy(&out.stderr);

    // The handler ran to its own `_exit(1)` rather than dying on the
    // signal's default action (or looping in the re-entrancy guard).
    assert_eq!(out.status.code(), Some(1), "{report}");
    assert!(report.contains("FATAL: SIGSEGV"), "{report}");
    assert!(
        report.contains(&format!("fault=0x{BAD_ADDR:016x}")),
        "{report}"
    );

    // Reads the hex word printed right after `label`.
    let value_after = |label: &str| -> String {
        report
            .split_once(label)
            .unwrap_or_else(|| panic!("{label} missing from report:\n{report}"))
            .1
            .chars()
            .take(18)
            .collect()
    };
    let zero = format!("0x{:016x}", 0);

    // `fault_pc` is why this handler beats the one Wine would print: it must
    // name the faulting instruction, and the stack pointer must be real.
    assert_ne!(value_after("fault_pc="), zero, "{report}");
    assert_ne!(value_after(" sp="), zero, "{report}");

    // The two per-arch offsets. On arm64 both decode exactly: AAPCS64 passes
    // the argument in `x0` and `BLR` leaves the return address in `lr`, so
    // the sentinel pins `ARG0_OFFSET` to the byte and a non-zero `lr` pins
    // `LR_OFFSET`. The x86_64 pair can only be checked for presence, because
    // neither is reproducible from a native call: `rcx` is the *Win64* first
    // argument (what Wine's COM calls use, not System V's `rdi`), and
    // `[rsp]` holds a return address only for a fault at a callee's first
    // instruction, which is the jump-through-garbage shape it exists for.
    let arg0_label = std::str::from_utf8(super::ARG0_LABEL).expect("ascii label");
    let caller_label = std::str::from_utf8(super::CALLER_LABEL).expect("ascii label");
    assert!(report.contains(arg0_label), "{report}");
    assert!(report.contains(caller_label), "{report}");
    #[cfg(target_arch = "aarch64")]
    {
        assert_eq!(
            value_after(arg0_label),
            format!("0x{BAD_ADDR:016x}"),
            "{report}"
        );
        assert_ne!(value_after(caller_label), zero, "{report}");
    }
}

/// A fault outside our image is named, then handed back.
///
/// `strlen` on the bad address faults inside libsystem, which the handler
/// does not own: the child dies by the signal's default action rather than
/// the handler's `_exit(1)`, and the report before that names the image and
/// the calling thread.
#[test]
fn foreign_fault_is_named_then_forwarded() {
    if std::env::var_os(FOREIGN_SELFTEST_ENV).is_some() {
        super::install();
        // SAFETY: deliberately unsound; this is the fault under test, taken
        // in a child process that dies on it.
        let len = unsafe { libc::strlen(std::hint::black_box(BAD_ADDR as *const libc::c_char)) };
        unreachable!("the strlen above must fault, not return {len}");
    }

    let exe = std::env::current_exe().expect("test binary path");
    let out = std::process::Command::new(exe)
        .args([
            "--exact",
            "crash::tests::foreign_fault_is_named_then_forwarded",
            "--nocapture",
        ])
        .env(FOREIGN_SELFTEST_ENV, "1")
        .output()
        .expect("re-exec the test binary");
    let report = String::from_utf8_lossy(&out.stderr);

    assert_eq!(out.status.signal(), Some(libc::SIGSEGV), "{report}");
    assert!(!report.contains("FATAL"), "{report}");
    let line = report
        .lines()
        .find(|l| l.contains("fault outside mtld3d.so"))
        .unwrap_or_else(|| panic!("no foreign-fault line:\n{report}"));
    assert!(line.contains("signo=11"), "{line}");
    assert!(line.contains(" tid=0x"), "{line}");
    // The released or messaged object, for a fault inside the runtime.
    assert!(line.contains(" arg0=0x"), "{line}");
    // The image as path plus offset: the load address alone names nothing
    // once the process is gone.
    assert!(line.contains("image=/"), "{line}");
    assert!(line.contains(".dylib+0x"), "{line}");
    assert!(line.contains("thread="), "{line}");
    // The faulting stack was scanned for our frames: this test binary is
    // one of ours by path, so the caller of `strlen` is on the list.
    assert!(
        report.contains("mtld3d.so return addrs on stack:"),
        "{report}"
    );
}

/// A thread with no TEB is not one Wine's handler can serve.
///
/// Three cases, since the guard has to leave the forwarding path alone
/// wherever Wine is not the previous owner: no Wine in the process at all,
/// which is what a plain pthread here is; a thread that has a TEB; and a
/// thread that does not, on the calling thread and on a spawned one.
#[test]
fn a_thread_without_a_teb_is_not_one_wine_serves() {
    super::resolve_wine_current_teb();
    assert_eq!(super::wine_teb(), 0);
    assert!(super::wine_serves_this_thread());

    pin_wine_teb(teb_stub);
    assert_ne!(super::wine_teb(), 0);
    assert!(super::wine_serves_this_thread());

    pin_wine_teb(no_teb_stub);
    assert_eq!(super::wine_teb(), 0);
    assert!(!super::wine_serves_this_thread());
    let off_thread = std::thread::spawn(super::wine_serves_this_thread)
        .join()
        .expect("the spawned thread returns");
    super::WINE_CURRENT_TEB.store(0, Ordering::Relaxed);
    assert!(!off_thread);
}

/// A foreign fault on a thread with no TEB ends the process here.
///
/// Handing it back would fault inside Wine reading the TEB the thread does not
/// have, and that second fault is what the report would name. The child dies
/// through the handler's own `_exit(1)` instead, with the foreign fault named
/// first and the fatal report under it.
#[test]
fn foreign_fault_without_a_teb_is_reported_not_forwarded() {
    if std::env::var_os(NO_TEB_SELFTEST_ENV).is_some() {
        super::install();
        pin_wine_teb(no_teb_stub);
        // SAFETY: deliberately unsound; this is the fault under test, taken
        // in a child process that never returns from the handler.
        let len = unsafe { libc::strlen(std::hint::black_box(BAD_ADDR as *const libc::c_char)) };
        unreachable!("the strlen above must fault, not return {len}");
    }

    let exe = std::env::current_exe().expect("test binary path");
    let out = std::process::Command::new(exe)
        .args([
            "--exact",
            "crash::tests::foreign_fault_without_a_teb_is_reported_not_forwarded",
            "--nocapture",
        ])
        .env(NO_TEB_SELFTEST_ENV, "1")
        .output()
        .expect("re-exec the test binary");
    let report = String::from_utf8_lossy(&out.stderr);

    // The handler ran to its own `_exit(1)` rather than dying on the signal's
    // default action, which is where a forward would have ended up.
    assert_eq!(out.status.code(), Some(1), "{report}");
    assert_eq!(out.status.signal(), None, "{report}");
    // The first fault is the one named, image and symbol included, and the
    // fatal report that follows carries the stack the forwarded case lacks.
    let foreign = report
        .find("fault outside mtld3d.so")
        .unwrap_or_else(|| panic!("no foreign-fault line:\n{report}"));
    let fatal = report
        .find("FATAL: SIGSEGV")
        .unwrap_or_else(|| panic!("no fatal banner:\n{report}"));
    assert!(foreign < fatal, "{report}");
    assert!(report.contains("has no Wine TEB"), "{report}");
    assert!(report.contains("native backtrace:"), "{report}");
}

/// A trap instruction in our own code is fatal and named like a fault.
///
/// A framework's assertion ends a process with the same signal, so the
/// handler has to own it: the child dies through the handler's `_exit(1)`
/// with the banner, not by the signal's default action.
#[test]
fn illegal_instruction_in_our_code_is_fatal() {
    if std::env::var_os(ILL_SELFTEST_ENV).is_some() {
        super::install();
        // SAFETY: deliberately traps; this is the signal under test, taken
        // in a child process that never returns from the handler.
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::asm!("ud2");
        }
        // SAFETY: as above, the permanently undefined encoding.
        #[cfg(target_arch = "aarch64")]
        unsafe {
            std::arch::asm!("udf #0");
        }
        unreachable!("the trap above must not return");
    }

    let exe = std::env::current_exe().expect("test binary path");
    let out = std::process::Command::new(exe)
        .args([
            "--exact",
            "crash::tests::illegal_instruction_in_our_code_is_fatal",
            "--nocapture",
        ])
        .env(ILL_SELFTEST_ENV, "1")
        .output()
        .expect("re-exec the test binary");
    let report = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{report}");
    assert!(report.contains("FATAL: SIGILL"), "{report}");
}
