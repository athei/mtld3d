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
//! not own is handed back to the signal's default action, and the one line it
//! writes first has to name the image and the thread.

use std::os::unix::process::ExitStatusExt as _;

/// Set in the re-executed child so it faults instead of asserting.
///
/// A signal handler can only be exercised by actually taking the signal,
/// which terminates the process, so the test spawns itself.
const SELFTEST_ENV: &str = "MTLD3D_CRASH_SELFTEST";

/// Set in the re-executed child so it faults in someone else's code.
const FOREIGN_SELFTEST_ENV: &str = "MTLD3D_CRASH_FOREIGN_SELFTEST";

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
    assert!(line.contains("image=/"), "{line}");
    assert!(line.contains("thread="), "{line}");
}
