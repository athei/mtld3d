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
//! The trap-object tests take a `SIGILL` whose PC is a real `CoreFoundation`
//! symbol on a thread that answers with no TEB, the shape of the framework's
//! own trap, and point `rbx` at memory of a known kind: a readable object
//! whose first word resolves to a symbol, an unmapped page, an object that
//! ends one word before a protected page, and the null it may hold. The dump
//! has to name what it can read and stop at the first word it cannot, and
//! must stay absent from a trap in our own code and from a memory fault.
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

/// Set in a child that scans across a protected stack boundary.
const STACK_SELFTEST_ENV: &str = "MTLD3D_CRASH_STACK_SELFTEST";

/// Set in the re-executed child that takes a `CoreFoundation` trap; its value picks the object.
#[cfg(target_arch = "x86_64")]
const TRAP_OBJECT_SELFTEST_ENV: &str = "MTLD3D_CRASH_TRAP_OBJECT_SELFTEST";

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
    #[cfg(target_arch = "x86_64")]
    for label in [
        " rbx=", " rbp=", " r12=", " r13=", " r14=", " r15=", " rflags=",
    ] {
        assert!(value_after(label).starts_with("0x"), "{report}");
    }
    #[cfg(target_arch = "aarch64")]
    {
        assert!(!report.contains(" rflags="), "{report}");
        assert_eq!(
            value_after(arg0_label),
            format!("0x{BAD_ADDR:016x}"),
            "{report}"
        );
        assert_ne!(value_after(caller_label), zero, "{report}");
    }
}

/// Typed Darwin fields must survive the terminal report at their full width.
///
/// Distinct sentinels catch swapped offsets; all-one values pin the longest
/// line including its newline. The synthetic context is read only by the
/// handler, never restored as executable machine state.
#[cfg(target_arch = "x86_64")]
#[test]
fn nonvolatile_registers_preserve_context() {
    if let Ok(mode) = std::env::var(SELFTEST_ENV) {
        // SAFETY: Darwin's machine context contains only integer state.
        let mut registers: libc::__darwin_mcontext64 = unsafe { core::mem::zeroed() };
        let sentinel = |value| if mode == "max" { u64::MAX } else { value };
        registers.__ss.__rbx = sentinel(0x8123_4567_89ab_cdef);
        registers.__ss.__rbp = sentinel(0x9234_5678_9abc_def0);
        registers.__ss.__r12 = sentinel(0xa345_6789_abcd_ef01);
        registers.__ss.__r13 = sentinel(0xb456_789a_bcde_f012);
        registers.__ss.__r14 = sentinel(0xc567_89ab_cdef_0123);
        registers.__ss.__r15 = sentinel(0xd678_9abc_def0_1234);
        registers.__ss.__rflags = sentinel(0xe789_abcd_ef01_2345);
        // SAFETY: ucontext contains integers and nullable raw pointers.
        let mut context: libc::ucontext_t = unsafe { core::mem::zeroed() };
        context.uc_mcontext = &raw mut registers;
        context.uc_mcsize = core::mem::size_of_val(&registers);
        super::handler(libc::SIGABRT, ptr::null_mut(), (&raw mut context).cast());
        unreachable!("the terminal handler must end the child");
    }

    for (mode, expected) in [
        (
            "sentinels",
            concat!(
                "[mtld3d::unix] rbx=0x8123456789abcdef rbp=0x923456789abcdef0",
                " r12=0xa3456789abcdef01 r13=0xb456789abcdef012",
                " r14=0xc56789abcdef0123 r15=0xd6789abcdef01234",
                " rflags=0xe789abcdef012345\n",
            ),
        ),
        (
            "max",
            concat!(
                "[mtld3d::unix] rbx=0xffffffffffffffff rbp=0xffffffffffffffff",
                " r12=0xffffffffffffffff r13=0xffffffffffffffff",
                " r14=0xffffffffffffffff r15=0xffffffffffffffff",
                " rflags=0xffffffffffffffff\n",
            ),
        ),
    ] {
        let out = std::process::Command::new(std::env::current_exe().expect("test binary path"))
            .args([
                "--exact",
                "crash::tests::nonvolatile_registers_preserve_context",
                "--nocapture",
            ])
            .env(SELFTEST_ENV, mode)
            .output()
            .expect("re-exec the test binary");
        let report = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(1), "{report}");
        assert!(report.contains("FATAL: SIGABRT"), "{report}");
        assert!(report.contains(expected), "{mode}: {report}");
        assert_eq!(expected.len(), 179);
        assert_eq!(report.matches(" rflags=").count(), 1, "{report}");
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
    assert!(!report.contains(" rflags="), "{report}");
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
    #[cfg(target_arch = "x86_64")]
    assert!(report.contains(" rflags=0x"), "{report}");
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
    assert!(!report.contains("trap object"), "{report}");
}

/// Address of a `CoreFoundation` function, the PC a framework trap reports.
///
/// Looked up by name so the address is inside the framework's own image and
/// not a wrapper of ours; the crate links the framework, so it is loaded.
#[cfg(target_arch = "x86_64")]
fn core_foundation_pc() -> u64 {
    // SAFETY: `dlsym` with the default handle reads the loaded images only.
    let symbol = unsafe { libc::dlsym(libc::RTLD_DEFAULT, c"CFRunLoopGetCurrent".as_ptr()) };
    assert!(!symbol.is_null(), "CoreFoundation is loaded");
    symbol as usize as u64
}

/// Take a `CoreFoundation` trap in the child with `rbx` naming `object`.
///
/// The context is read only by the handler, never restored as machine state;
/// the stack pointer stays zero so the stack scans stay out of the report.
#[cfg(target_arch = "x86_64")]
fn trap_with_object(signo: libc::c_int, object: u64) {
    // The install resolves Wine's `NtCurrentTeb` and finds none here, so the
    // stand-in goes in after it.
    super::install();
    pin_wine_teb(no_teb_stub);
    // SAFETY: Darwin's machine context contains only integer state.
    let mut registers: libc::__darwin_mcontext64 = unsafe { core::mem::zeroed() };
    registers.__ss.__rip = core_foundation_pc();
    registers.__ss.__rbx = object;
    // SAFETY: ucontext contains integers and nullable raw pointers.
    let mut context: libc::ucontext_t = unsafe { core::mem::zeroed() };
    context.uc_mcontext = &raw mut registers;
    context.uc_mcsize = core::mem::size_of_val(&registers);
    super::handler(signo, ptr::null_mut(), (&raw mut context).cast());
    unreachable!("the terminal handler must end the child");
}

/// Re-execute this test binary into one trap-object child and return its report.
#[cfg(target_arch = "x86_64")]
fn trap_object_report(test: &str, mode: &str) -> String {
    let out = std::process::Command::new(std::env::current_exe().expect("test binary path"))
        .args(["--exact", test, "--nocapture"])
        .env(TRAP_OBJECT_SELFTEST_ENV, mode)
        .output()
        .expect("re-exec the test binary");
    let report = String::from_utf8_lossy(&out.stderr).into_owned();
    assert_eq!(out.status.code(), Some(1), "{mode}: {report}");
    report
}

/// The dump names a word that resolves to a symbol and prints the rest as hex.
///
/// The object's first word is the address of a function in this binary, so
/// its line has to carry this binary's name and an offset; the second word is
/// a bit pattern no image covers, so its line ends at the hex.
#[cfg(target_arch = "x86_64")]
#[test]
fn trap_object_words_are_dumped_and_resolved() {
    const TEST: &str = "crash::tests::trap_object_words_are_dumped_and_resolved";

    if std::env::var_os(TRAP_OBJECT_SELFTEST_ENV).is_some() {
        let mut object = [0u64; super::TRAP_OBJECT_WORDS];
        object[0] = super::install as usize as u64;
        object[1] = 0x8000_0000_0000_0001;
        object[super::TRAP_OBJECT_WORDS - 1] = 0x0123_4567_89ab_cdef;
        trap_with_object(libc::SIGILL, object.as_ptr() as usize as u64);
    }

    let report = trap_object_report(TEST, "resolved");
    assert!(report.contains("FATAL: SIGILL"), "{report}");
    assert!(report.contains("no Wine TEB"), "{report}");
    assert!(
        report.contains("CoreFoundation trap object words at rbx:\n"),
        "{report}"
    );
    let exe = std::env::current_exe().expect("test binary path");
    let basename = exe
        .file_name()
        .and_then(|n| n.to_str())
        .expect("utf-8 test binary name");
    let first = report
        .lines()
        .find(|line| line.starts_with("  +0x0000000000000000 "))
        .unwrap_or_else(|| panic!("first word missing:\n{report}"));
    // The child's load address differs from this process's, so the line is
    // checked by the names it resolves to rather than by the address.
    assert!(
        first.contains(&basename[..basename.len().min(32)]),
        "{first}\n{report}"
    );
    assert!(
        first.contains("install+0x0000000000000000"),
        "{first}\n{report}"
    );
    assert!(
        report.contains("  +0x0000000000000008 0x8000000000000001\n"),
        "{report}"
    );
    assert!(
        report.contains("  +0x00000000000000b8 0x0123456789abcdef\n"),
        "{report}"
    );
    assert!(!report.contains("trap object read unavailable"), "{report}");
    assert_eq!(
        report
            .lines()
            .filter(|line| line.starts_with("  +0x"))
            .count(),
        super::TRAP_OBJECT_WORDS,
        "{report}"
    );
}

/// An unreadable object costs one line and nothing else in the report.
#[cfg(target_arch = "x86_64")]
#[test]
fn trap_object_at_an_unmapped_page_stops_the_dump() {
    const TEST: &str = "crash::tests::trap_object_at_an_unmapped_page_stops_the_dump";

    if std::env::var_os(TRAP_OBJECT_SELFTEST_ENV).is_some() {
        trap_with_object(libc::SIGILL, 0xdead_b000);
    }

    let report = trap_object_report(TEST, "unmapped");
    assert!(report.contains("FATAL: SIGILL"), "{report}");
    assert!(
        report.contains("CoreFoundation trap object words at rbx:\n"),
        "{report}"
    );
    assert!(
        report.contains("trap object read unavailable, stopping dump\n"),
        "{report}"
    );
    assert!(!report.contains("  +0x"), "{report}");
    assert!(report.contains("native backtrace:"), "{report}");
}

/// An object that ends at a protected page yields its readable words, then stops.
#[cfg(target_arch = "x86_64")]
#[test]
fn trap_object_dump_stops_at_a_protected_page() {
    const TEST: &str = "crash::tests::trap_object_dump_stops_at_a_protected_page";

    if std::env::var_os(TRAP_OBJECT_SELFTEST_ENV).is_some() {
        // SAFETY: sysconf reads the process's constant page size.
        let page = usize::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) })
            .expect("positive page size");
        // SAFETY: anonymous mapping with no fixed address or backing file.
        let mapping = unsafe {
            libc::mmap(
                ptr::null_mut(),
                page * 2,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        assert_ne!(mapping, libc::MAP_FAILED);
        let boundary = mapping as usize + page;
        // SAFETY: the second page is inside the live mapping and page-aligned.
        let protected = unsafe { libc::mprotect(boundary as *mut c_void, page, libc::PROT_NONE) };
        assert_eq!(protected, 0);
        // SAFETY: the two words below the boundary are inside the readable page.
        unsafe {
            (boundary as *mut u64).sub(2).write(0x1111_1111_1111_1111);
            (boundary as *mut u64).sub(1).write(0x2222_2222_2222_2222);
        }
        trap_with_object(libc::SIGILL, (boundary - 16) as u64);
    }

    let report = trap_object_report(TEST, "boundary");
    assert!(
        report.contains("  +0x0000000000000000 0x1111111111111111\n"),
        "{report}"
    );
    assert!(
        report.contains("  +0x0000000000000008 0x2222222222222222\n"),
        "{report}"
    );
    assert!(!report.contains("  +0x0000000000000010"), "{report}");
    assert!(
        report.contains("trap object read unavailable, stopping dump\n"),
        "{report}"
    );
}

/// A null `rbx` and a memory fault in the framework both leave the dump out.
#[cfg(target_arch = "x86_64")]
#[test]
fn trap_object_dump_needs_a_framework_trap_with_an_object() {
    const TEST: &str = "crash::tests::trap_object_dump_needs_a_framework_trap_with_an_object";

    if let Ok(mode) = std::env::var(TRAP_OBJECT_SELFTEST_ENV) {
        let object = [super::install as usize as u64; super::TRAP_OBJECT_WORDS];
        match mode.as_str() {
            "null" => trap_with_object(libc::SIGILL, 0),
            "segv" => trap_with_object(libc::SIGSEGV, object.as_ptr() as usize as u64),
            other => panic!("unknown mode {other}"),
        }
    }

    let report = trap_object_report(TEST, "null");
    assert!(report.contains("FATAL: SIGILL"), "{report}");
    assert!(!report.contains("trap object"), "{report}");
    let report = trap_object_report(TEST, "segv");
    assert!(report.contains("FATAL: SIGSEGV"), "{report}");
    assert!(!report.contains("trap object"), "{report}");
}

#[test]
fn native_stack_scan_stops_at_unreadable_memory() {
    check_stack_boundaries(
        "crash::tests::native_stack_scan_stops_at_unreadable_memory",
        |sp| super::our_frames_on_stack(sp, 16),
    );
}

#[test]
fn stack_words_preserve_unaligned_values() {
    let bytes = [
        0x11u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd,
    ];
    let sp = bytes.as_ptr() as usize as u64 + 1;
    assert_eq!(
        super::stack_word::<4>(sp, 0),
        Some([0x22, 0x33, 0x44, 0x55])
    );
    assert_eq!(
        super::stack_word::<8>(sp, 1),
        Some([0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd])
    );
    assert_eq!(super::stack_word::<8>(sp, usize::MAX), None);
    assert_eq!(super::stack_word::<4>(u64::MAX - 1, 1), None);
}

#[test]
fn unreadable_stack_does_not_prevent_signal_forwarding() {
    check_stack_boundaries(
        "crash::tests::unreadable_stack_does_not_prevent_signal_forwarding",
        forward_with_stack,
    );
}

/// A previous signal owner reached with the original signal ends the child successfully.
extern "C" fn forwarded_signal(signo: libc::c_int, _info: *mut libc::siginfo_t, _ctx: *mut c_void) {
    // SAFETY: ends only the regression-test child, without running exit hooks.
    unsafe { libc::_exit(i32::from(signo != libc::SIGSEGV)) }
}

/// Run the real handler with a foreign PC and a stack that ends at a guard page.
fn forward_with_stack(sp: u64) {
    // SAFETY: all-zero sigaction is a valid starting value for initialization.
    let mut action: libc::sigaction = unsafe { core::mem::zeroed() };
    action.sa_sigaction = forwarded_signal as *const () as usize;
    action.sa_flags = libc::SA_SIGINFO;
    // SAFETY: action contains a valid mask out-parameter.
    assert_eq!(unsafe { libc::sigemptyset(&raw mut action.sa_mask) }, 0);
    assert_eq!(
        // SAFETY: installs a correctly typed handler in this child process only.
        unsafe { libc::sigaction(libc::SIGSEGV, &raw const action, ptr::null_mut()) },
        0
    );
    super::install();

    // Model the kernel-owned context with live, aligned local storage. Only
    // the register slots read by the handler need nonzero values.
    let mut registers = [0u64; 40];
    #[cfg(target_arch = "x86_64")]
    let pc_slot = 144 / 8;
    #[cfg(target_arch = "aarch64")]
    let pc_slot = 272 / 8;
    registers[pc_slot] = libc::strlen as *const () as usize as u64;
    registers[super::SP_OFFSET / 8] = sp;
    let mut context = [0u64; 7];
    context[0x30 / 8] = registers.as_ptr() as usize as u64;
    super::handler(libc::SIGSEGV, ptr::null_mut(), context.as_mut_ptr().cast());
    unreachable!("the previous signal owner must terminate the child");
}

#[test]
fn guest_stack_scan_stops_at_unreadable_memory() {
    check_stack_boundaries(
        "crash::tests::guest_stack_scan_stops_at_unreadable_memory",
        super::scan_stack_for_our_frames,
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn caller_stack_read_tolerates_unreadable_memory() {
    check_stack_boundaries(
        "crash::tests::caller_stack_read_tolerates_unreadable_memory",
        |sp| assert_eq!(super::caller_pc(ptr::null_mut(), sp), 0),
    );
}

/// Exercise stack readers in a child so an unsafe read fails one test only.
///
/// The first page is readable and zero-filled, the second is mapped with no
/// access. Reads start just before the boundary, on it, and after unmapping
/// both pages; overflowing addresses must also return without a signal.
fn check_stack_boundaries(test: &str, scan: fn(u64)) {
    if std::env::var_os(STACK_SELFTEST_ENV).is_some() {
        // SAFETY: sysconf reads the process's constant page size.
        let page = usize::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) })
            .expect("positive page size");
        // SAFETY: anonymous mapping with no fixed address or backing file.
        let mapping = unsafe {
            libc::mmap(
                ptr::null_mut(),
                page * 2,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        assert_ne!(mapping, libc::MAP_FAILED);
        let base = mapping as usize as u64;
        let boundary = base + page as u64;
        // SAFETY: the second page is inside the live mapping and page-aligned.
        let protected = unsafe { libc::mprotect(boundary as *mut c_void, page, libc::PROT_NONE) };
        assert_eq!(protected, 0);
        scan(boundary - 8);
        scan(boundary - 4);
        scan(boundary - 2);
        scan(boundary);
        // SAFETY: releases the complete mapping created above, exactly once.
        assert_eq!(unsafe { libc::munmap(mapping, page * 2) }, 0);
        scan(base);
        scan(0);
        scan(u64::MAX - 3);
        return;
    }

    let out = std::process::Command::new(std::env::current_exe().expect("test binary path"))
        .args(["--exact", test, "--nocapture"])
        .env(STACK_SELFTEST_ENV, "1")
        .output()
        .expect("re-exec the test binary");
    let report = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{}: {report}", out.status);
    assert!(!report.contains("FATAL"), "{report}");
    assert!(report.contains("stack read unavailable"), "{report}");
}

/// The file name decides ownership, not the path.
///
/// Wine lives under a directory named after this project in a normal
/// developer install, and claiming its images sends a fault Wine would have
/// recovered down the terminal path instead of back to it.
#[test]
fn only_the_image_file_name_decides_whether_a_fault_is_ours() {
    fn names_ours(path: &str) -> bool {
        let c = std::ffi::CString::new(path).expect("no interior NUL");
        super::path_names_our_image(c.as_ptr())
    }

    assert!(names_ours("/opt/mtld3d/lib/wine/x86_64-unix/mtld3d.so"));
    assert!(names_ours("mtld3d.so"));
    assert!(
        !names_ours("/opt/mtld3d-toolchain/components/wine/lib/wine/x86_64-unix/ntdll.so"),
        "Wine's own image under a directory named for this project is not ours",
    );
    assert!(
        !names_ours("ntdll.so"),
        "a bare foreign file name is not ours"
    );
    assert!(
        !names_ours("/opt/mtld3d/"),
        "a trailing separator names no file"
    );
    assert!(!names_ours("/usr/lib/system/libsystem_platform.dylib"));
    assert!(!names_ours("/opt/mtld3d/lib/wine/x86_64-unix/winemetal.so"));
    assert!(
        !names_ours("/x/libmtld3d_unix.dylib"),
        "the file name has to begin with the needle, not merely contain it",
    );
    assert!(!names_ours(""));
}
