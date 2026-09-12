//! Always-on unix-side crash handler.
//!
//! Installed once from `init_logger_handler`. Catches SIGSEGV, SIGBUS,
//! SIGILL, SIGABRT. The handler is async-signal-safe — it only calls
//! `libc::write` on the log file's descriptor and `mtld3d_shared::crumb::dump_recent` (which is
//! itself async-signal-safe). On a fatal signal in OUR code the handler:
//!
//! 1. Writes a single-line fatal banner identifying the signal and the
//!    fault address (for SIGSEGV/SIGBUS).
//! 2. If `cfg(mtld3d_crumb)` is on, dumps the last 32 ring-buffer
//!    entries (interleaved PE + unix events).
//! 3. Calls `libc::_exit(1)` — does **not** chain to Wine's prior
//!    handler.
//!
//! The point of terminating directly is that any unix-side fatal event
//! has corrupted state we can't recover from; continuing into Wine's
//! NTSTATUS-translation path lets the encoder thread keep churning
//! until `WoW` eventually crashes downstream. `_exit(1)` produces one
//! clean diagnostic and one termination event.
//!
//! A memory fault raised by anything OTHER than our own code is forwarded to
//! whoever owned the signal before us instead, because we are not the only
//! consumer of a fault in this process: Wine translates guest faults into
//! Windows exceptions, and on an arm64 host the x86 emulator (`xtajit`) takes
//! memory faults as part of ordinary work. Terminating on those killed the game
//! at startup on `CrossOver`'s arm64 Wine, and on any host it would have eaten
//! the guest's own exception handling.
//!
//! One line is still written before such a fault is forwarded when its PC
//! lies in a native image (a framework, libobjc, libsystem, Wine's own
//! `.so`): `fault outside mtld3d.so:` with the signal number, the thread id
//! and name, the PC as image plus offset and nearest symbol, then the return
//! addresses into our own dylib found on the faulting stack. Guest code and
//! translated code resolve to no image and stay silent: those faults are
//! Wine's ordinary work. The report is capped at a few per process and
//! carries no signal name, since a fault Wine recovers must not read as a
//! crash to anything that scans the log for one.
//!
//! Forwarding needs a thread Wine can serve. Wine's unix side keeps each
//! thread's TEB in a pthread key and reads it as soon as a fault reaches
//! `segv_handler`, so on a thread it did not create (the Cocoa main thread
//! `winemac` runs its event loop on, or one of ours) the forward faults
//! inside Wine, and the fault the report then names is that second one. The
//! handler therefore asks Wine for the calling thread's TEB first, through
//! the `NtCurrentTeb` its own code goes through, and a foreign fault on a
//! thread without one is reported in full here and ends the process, so the
//! report leads with the fault that caused it.
//!
//! A trap `CoreFoundation` raises itself (`SIGILL` on a deliberate `ud2`, the
//! way the framework halts on a run-loop timer it refuses to reschedule) names
//! its condition in a string and nothing else, while the object it was checking
//! is in `rbx`. The terminal report of such a trap therefore adds the first
//! words of that object, each copied by the kernel rather than dereferenced and
//! annotated with the image and symbol `dladdr` resolves it to, so the object's
//! class, callout and context read as their owners without the handler assuming
//! any layout.
//!
//! `RUST_BACKTRACE=1` is also set here (if unset) so the default Rust
//! panic hook prints message + backtrace before `abort()` flows through
//! to the SIGABRT branch.

use core::{
    ffi::{c_int, c_void},
    mem, ptr,
};
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, AtomicUsize, Ordering};

use mtld3d_shared::{crumb, fatal};

/// Re-entrancy guard: reading the faulting stack/registers can itself fault on a corrupted context.
///
/// With `SA_NODEFER` that would re-enter `handler`; the first re-entry exits
/// immediately so we never loop.
static IN_HANDLER: AtomicBool = AtomicBool::new(false);

/// The disposition a signal had before we took it over.
///
/// Kept in atomics rather than a `sigaction` copy because the handler reads it
/// from a signal context, where a lock is not an option.
struct PrevDisposition {
    /// `sa_sigaction`, or `SIG_DFL` / `SIG_IGN`.
    action: AtomicUsize,
    /// `sa_flags`, which says whether `action` is a three-argument handler.
    flags: AtomicI32,
}

impl PrevDisposition {
    const fn new() -> Self {
        Self {
            action: AtomicUsize::new(libc::SIG_DFL),
            flags: AtomicI32::new(0),
        }
    }
}

/// Previous dispositions of the signals we install, indexed by [`signal_slot`].
static PREV: [PrevDisposition; 4] = [
    PrevDisposition::new(),
    PrevDisposition::new(),
    PrevDisposition::new(),
    PrevDisposition::new(),
];

/// Wine's `NtCurrentTeb`, resolved once so the handler can ask from a signal.
///
/// Zero when Wine's unix library is not in the process, which is every context
/// but a Wine one, the unit tests among them. `dlsym` allocates and takes the
/// loader's lock, so the lookup happens at install time and the handler only
/// reads the atomic.
static WINE_CURRENT_TEB: AtomicUsize = AtomicUsize::new(0);

// macOS interfaces absent or deprecated in libc, provided by libSystem.
unsafe extern "C" {
    static mut mach_task_self_: libc::mach_port_t;
    fn mach_vm_read_overwrite(
        target_task: libc::vm_map_t,
        address: libc::mach_vm_address_t,
        size: libc::mach_vm_size_t,
        data: libc::mach_vm_address_t,
        outsize: *mut libc::mach_vm_size_t,
    ) -> libc::kern_return_t;
    fn backtrace(array: *mut *mut c_void, size: c_int) -> c_int;
    fn backtrace_symbols_fd(array: *const *mut c_void, size: c_int, fd: c_int);
    /// macOS `pthread_getname_np` (not exposed by the `libc` crate).
    ///
    /// Reads the calling thread's name into `buf`.
    fn pthread_getname_np(
        thread: libc::pthread_t,
        buf: *mut core::ffi::c_char,
        len: usize,
    ) -> c_int;
}

/// Index into [`PREV`] for a signal we handle, or `None` for anything else.
const fn signal_slot(signo: c_int) -> Option<usize> {
    match signo {
        libc::SIGSEGV => Some(0),
        libc::SIGBUS => Some(1),
        libc::SIGABRT => Some(2),
        libc::SIGILL => Some(3),
        _ => None,
    }
}

/// Install the crash handler.
///
/// Called once from the first-thunk init, whose `Once` guards the whole init
/// sequence; installing signal handlers is itself idempotent at the OS level
/// (same handler), so a stray re-call is harmless.
pub fn install() {
    if std::env::var_os("RUST_BACKTRACE").is_none() {
        // `full` over `1` so std-internal frames don't get elided —
        // matches the PE side's choice for the same reason.
        // SAFETY: init_logger_handler runs on the API thread before the
        // encoder thread is spawned (encoder spawns from CreateDevice,
        // which always follows InitLogger); `set_var` is unsound only on
        // concurrent reads/writes, which can't happen here.
        unsafe { std::env::set_var("RUST_BACKTRACE", "full") };
    }

    resolve_wine_current_teb();

    // Diagnostic escape hatch: with `MTLD3D_NO_CRASH_HANDLER=1` we do NOT
    // intercept SIGSEGV/SIGBUS, so Wine's own SEH machinery translates the
    // fault into a Windows exception and prints a PE-side backtrace
    // (`d3d9.dll`/`winemac.drv`+offset) — the frame our async-signal-safe
    // handler can't recover when the stack chain is broken.
    if std::env::var_os("MTLD3D_NO_CRASH_HANDLER").is_none() {
        install_signal_handler(libc::SIGSEGV);
        install_signal_handler(libc::SIGBUS);
        // A trap instruction: a framework's own assertion (`__builtin_trap`)
        // ends a process this way, and so does a jump into non-code.
        install_signal_handler(libc::SIGILL);
    }
    install_signal_handler(libc::SIGABRT);
}

fn install_signal_handler(signo: libc::c_int) {
    // SAFETY: writing a zero-initialized sigaction with our handler.
    let mut act: libc::sigaction = unsafe { mem::zeroed() };
    act.sa_sigaction = handler as *const () as usize;
    act.sa_flags = libc::SA_SIGINFO | libc::SA_ONSTACK | libc::SA_NODEFER;
    // SAFETY: sigemptyset on a zeroed sigaction.
    unsafe { libc::sigemptyset(&raw mut act.sa_mask) };
    // SAFETY: zeroed `sigaction` is a valid out-param for the old disposition.
    let mut old: libc::sigaction = unsafe { mem::zeroed() };
    // SAFETY: sigaction(2) with a valid `act` and a valid out-param.
    unsafe {
        libc::sigaction(signo, &raw const act, &raw mut old);
    }
    // Kept so a fault that is not ours can go back to its owner, which is
    // Wine's exception translation, or the x86 emulator on an arm64 host.
    if let Some(slot) = signal_slot(signo) {
        PREV[slot].action.store(old.sa_sigaction, Ordering::Relaxed);
        PREV[slot].flags.store(old.sa_flags, Ordering::Relaxed);
    }
}

/// Resolve Wine's `NtCurrentTeb` into [`WINE_CURRENT_TEB`].
///
/// `ntdll.so` publishes it into the process's global symbol space, and the
/// handle `dlopen` returns for a null path is how to reach it without naming a
/// path of Wine's. That handle is kept rather than closed: it stands for the
/// process itself, and the address has to stay resolvable for as long as the
/// handler can run.
fn resolve_wine_current_teb() {
    // SAFETY: `dlopen` with a null path loads nothing; it hands back a handle
    // for the symbol space the process already has.
    let handle = unsafe { libc::dlopen(ptr::null(), libc::RTLD_LAZY | libc::RTLD_LOCAL) };
    if handle.is_null() {
        return;
    }
    // SAFETY: `handle` is the process handle just opened and the name is a
    // NUL-terminated C string; `dlsym` answers null when nothing exports it.
    let entry = unsafe { libc::dlsym(handle, c"NtCurrentTeb".as_ptr()) };
    WINE_CURRENT_TEB.store(entry as usize, Ordering::Relaxed);
}

/// The calling thread's Wine TEB, or 0 when the thread has none.
///
/// Wine's unix side stores it in a pthread key and reaches it through
/// `NtCurrentTeb`, which reads that key and nothing else, so a signal handler
/// can afford to ask. Zero on every thread Wine did not create, and in every
/// process where the symbol did not resolve.
fn wine_teb() -> u64 {
    let entry = WINE_CURRENT_TEB.load(Ordering::Relaxed);
    if entry == 0 {
        return 0;
    }
    // SAFETY: `entry` is the address `dlsym` resolved for Wine's
    // `NtCurrentTeb`, which takes no argument and returns the calling thread's
    // TEB pointer.
    let current_teb: extern "C" fn() -> *mut c_void = unsafe { mem::transmute::<usize, _>(entry) };
    current_teb() as usize as u64
}

/// Whether Wine's own handler can serve a fault taken on the calling thread.
///
/// A forward lands in `segv_handler`, which saves the faulting context, and
/// the debug registers it saves live in the TEB: a thread without one faults
/// there instead of having its fault translated. True when Wine's unix library
/// is absent, where the previous disposition is not Wine's and there is
/// nothing to guard against.
fn wine_serves_this_thread() -> bool {
    WINE_CURRENT_TEB.load(Ordering::Relaxed) == 0 || wine_teb() != 0
}

/// True when the faulting instruction is in our own `.so`.
///
/// The ownership test for a memory fault, and one `dladdr` is cheap enough for
/// a signal handler. It fails closed: a PC we cannot decode reads as not-ours,
/// so the fault goes back to its owner rather than terminating the process. A
/// fault taken inside a system framework we called (Metal, `AppKit`) is likewise
/// not ours by this rule; Wine reports that one as a guest exception instead of
/// our crumb dump, which is the price of staying out of the emulator's way.
fn fault_is_ours(ctx: *mut c_void) -> bool {
    let pc = fault_pc(ctx);
    pc != 0 && dladdr_is_ours(pc)
}

/// Hand a signal back to whoever owned it before us.
///
/// Only the three-argument (`SA_SIGINFO`) form is called through, which is what
/// Wine and the emulator install. Anything else, including `SIG_DFL`, restores
/// the previous disposition and returns: the faulting instruction re-executes
/// and takes that disposition instead, which is how a genuine fault still
/// reaches the default action.
fn forward_to_previous(signo: c_int, info: *mut libc::siginfo_t, ctx: *mut c_void) {
    let Some(slot) = signal_slot(signo) else {
        return;
    };
    let action = PREV[slot].action.load(Ordering::Relaxed);
    let flags = PREV[slot].flags.load(Ordering::Relaxed);

    if action != libc::SIG_DFL && action != libc::SIG_IGN && flags & libc::SA_SIGINFO != 0 {
        // SAFETY: `action` came from `sigaction`'s out-param for this signal
        // and its `SA_SIGINFO` flag says it takes these three arguments. The
        // arguments are the kernel's own, passed through unchanged.
        let previous: extern "C" fn(c_int, *mut libc::siginfo_t, *mut c_void) =
            unsafe { mem::transmute::<usize, _>(action) };
        previous(signo, info, ctx);
        return;
    }

    // SAFETY: writing a zero-initialized sigaction that restores the saved
    // disposition; the handler returns immediately after, so the faulting
    // instruction re-runs under it.
    let mut act: libc::sigaction = unsafe { mem::zeroed() };
    act.sa_sigaction = action;
    act.sa_flags = flags;
    // SAFETY: sigemptyset on a zeroed sigaction.
    unsafe { libc::sigemptyset(&raw mut act.sa_mask) };
    // SAFETY: sigaction(2) with a valid `act`; no out-param wanted.
    unsafe {
        libc::sigaction(signo, &raw const act, ptr::null_mut());
    }
}

extern "C" fn handler(signo: libc::c_int, info: *mut libc::siginfo_t, ctx: *mut c_void) {
    // A memory fault outside our own image belongs to somebody else: Wine turns
    // guest faults into Windows exceptions, and on an arm64 host the x86
    // emulator faults as part of ordinary work. Hand those straight back,
    // before the re-entrancy latch below, which would otherwise arm itself on
    // the first one and turn every later fault into an immediate `_exit`.
    // SIGABRT is not shared this way: an abort is always terminal, and its PC
    // is inside libsystem rather than our code, so it stays ours to report.
    if signo != libc::SIGABRT && !fault_is_ours(ctx) {
        /// Says why a foreign fault ends here rather than going back.
        const NO_TEB: &[u8] =
            b"[mtld3d::unix] faulting thread has no Wine TEB: reporting here, not forwarding\n";

        // Wine's handler reads the TEB of the thread it is serving. On a
        // thread that has none it faults doing so, and its fault, not this
        // one, is what a report would name. Report this one in full instead
        // and end the process below.
        let terminal = !wine_serves_this_thread();
        report_foreign_fault(signo, ctx, terminal);
        if !terminal {
            forward_to_previous(signo, info, ctx);
            return;
        }
        // SAFETY: write(2) is async-signal-safe; the descriptor is the log file's or fd 2.
        unsafe {
            let _ = libc::write(
                crate::log_file::raw_fd(),
                NO_TEB.as_ptr().cast::<c_void>(),
                NO_TEB.len(),
            );
        }
    }

    // Bail on the first re-entry (a faulting register/stack read below would
    // otherwise loop under `SA_NODEFER`).
    if IN_HANDLER.swap(true, Ordering::AcqRel) {
        // SAFETY: _exit(2) is async-signal-safe.
        unsafe { libc::_exit(1) };
    }
    // Async-signal-safe path: no allocator, no `log!`, no formatting that
    // takes locks. Stack-buffered hex formatting via `write_hex`.
    let mut buf = [0u8; 192];
    let mut pos = 0;
    push(&mut buf, &mut pos, fatal::BANNER.as_bytes());
    push(&mut buf, &mut pos, signal_name(signo));

    if !info.is_null() && (signo == libc::SIGSEGV || signo == libc::SIGBUS) {
        // SAFETY: info non-null per check; kernel-supplied for handler lifetime.
        let info_ref = unsafe { &*info };
        // SAFETY: si_addr() is the libc accessor for the relevant union.
        let fault_addr = unsafe { info_ref.si_addr() };
        let fault = fault_addr as usize as u64;
        push(&mut buf, &mut pos, b" fault=");
        push_hex(&mut buf, &mut pos, fault);
        let code = info_ref.si_code;
        push(&mut buf, &mut pos, b" si_code=");
        // `cast_signed`'s inverse: a total bit-pattern reinterpret to u32, with
        // no panic path (this runs in a signal handler) and no sign-loss lint.
        let code_u32 = code.cast_unsigned();
        push_hex(&mut buf, &mut pos, u64::from(code_u32));
    }
    push(&mut buf, &mut pos, b"\n");

    // SAFETY: write(2) is async-signal-safe; the descriptor is the log file's or fd 2.
    unsafe {
        let _ = libc::write(
            crate::log_file::raw_fd(),
            buf.as_ptr().cast::<c_void>(),
            pos,
        );
    }

    // Faulting thread name. For a teardown race the *which thread* (API vs
    // `mtld3d-encoder` / `mtld3d-submit` / `mtld3d-prewarm`) is the first clue.
    {
        let mut b = [0u8; 192];
        let mut p = 0;
        push(&mut b, &mut p, b"[mtld3d::unix] thread=");
        push_thread_name(&mut b, &mut p);
        push(&mut b, &mut p, b"\n");
        // SAFETY: write(2) is async-signal-safe; the descriptor is the log file's or fd 2.
        unsafe {
            let _ = libc::write(crate::log_file::raw_fd(), b.as_ptr().cast::<c_void>(), p);
        }
    }

    // Faulting program counter, pulled from the signal `ucontext` (see
    // `fault_pc`). The frame-pointer `backtrace` below can't cross the
    // `_sigtramp` boundary, so without this the actual faulting frame is
    // invisible — and for a jump through a freed/garbage object the PC *is* the
    // bad address, which is the tell. `dladdr` (via `backtrace_symbols_fd`)
    // names the enclosing module/symbol.
    let rip = fault_pc(ctx);
    if rip != 0 {
        let mut b = [0u8; 192];
        let mut p = 0;
        push(&mut b, &mut p, b"[mtld3d::unix] fault_pc=");
        push_hex(&mut b, &mut p, rip);
        push(&mut b, &mut p, b"\n");
        // SAFETY: write(2) is async-signal-safe; the descriptor is the log file's or fd 2.
        unsafe {
            let _ = libc::write(crate::log_file::raw_fd(), b.as_ptr().cast::<c_void>(), p);
        }
        let mut frame = [rip as *mut c_void; 1];
        // SAFETY: single in-bounds frame pointer; `backtrace_symbols_fd` is
        // async-signal-safe (resolves via `dladdr`, no malloc) and writes to the log descriptor.
        unsafe { backtrace_symbols_fd(frame.as_mut_ptr(), 1, crate::log_file::raw_fd()) };
    }

    // For a jump-through-garbage fault (`fault_pc` is a tiny/invalid value), the
    // saved registers name the culprit: the first-argument register holds a COM
    // call's `this` (the freed object), and the caller is one register or one
    // stack slot away. The register names differ per arch, the roles do not.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        let sp = mcontext_u64(ctx, SP_OFFSET);
        let mut b = [0u8; 192];
        let mut p = 0;
        push(&mut b, &mut p, b"[mtld3d::unix] ");
        push(&mut b, &mut p, ARG0_LABEL);
        push_hex(&mut b, &mut p, mcontext_u64(ctx, ARG0_OFFSET));
        #[cfg(target_arch = "x86_64")]
        {
            // The vtable pointer the faulting `CALL` loaded. No arm64
            // counterpart: an indirect branch there goes through whichever
            // register the compiler picked, and `fault_pc` above already
            // carries the value it jumped to.
            push(&mut b, &mut p, b" rax(vtbl)=");
            push_hex(&mut b, &mut p, mcontext_u64(ctx, RAX_OFFSET));
        }
        push(&mut b, &mut p, b" sp=");
        push_hex(&mut b, &mut p, sp);
        push(&mut b, &mut p, b"\n");
        // SAFETY: write(2) is async-signal-safe; the descriptor is the log file's or fd 2.
        unsafe {
            let _ = libc::write(crate::log_file::raw_fd(), b.as_ptr().cast::<c_void>(), p);
        }
        #[cfg(target_arch = "x86_64")]
        report_nonvolatile_registers(ctx);
        #[cfg(target_arch = "x86_64")]
        report_trap_object(signo, ctx);
        let ret = caller_pc(ctx, sp);
        if ret != 0 {
            let mut rb = [0u8; 192];
            let mut rp = 0;
            push(&mut rb, &mut rp, b"[mtld3d::unix] ");
            push(&mut rb, &mut rp, CALLER_LABEL);
            push_hex(&mut rb, &mut rp, ret);
            push(&mut rb, &mut rp, b"\n");
            // SAFETY: write(2) is async-signal-safe; the descriptor is the log file's or fd 2.
            unsafe {
                let _ = libc::write(crate::log_file::raw_fd(), rb.as_ptr().cast::<c_void>(), rp);
            }
            let mut frame = [ret as *mut c_void; 1];
            // SAFETY: single in-bounds frame pointer; `backtrace_symbols_fd` is
            // async-signal-safe (resolves via `dladdr`) and writes to the log descriptor.
            unsafe { backtrace_symbols_fd(frame.as_mut_ptr(), 1, crate::log_file::raw_fd()) };
        }
    }

    crumb::dump_recent(256);

    // Native backtrace of the faulting thread. `backtrace` only walks frame
    // pointers (no allocation) and `backtrace_symbols_fd` resolves each via
    // `dladdr` straight to the log descriptor — both async-signal-safe (unlike
    // `backtrace_symbols`, which mallocs). Symbolises our `.so`, Wine, and
    // system frames (Metal/CoreAnimation), turning a bare fault address into a
    // call chain.
    let mut frames = [ptr::null_mut::<c_void>(); 64];
    // SAFETY: `frames` is a valid 64-element buffer; `backtrace` writes at most
    // `len` entries and returns the count actually written.
    let n = unsafe { backtrace(frames.as_mut_ptr(), FRAME_CAP) };
    if n > 0 {
        const HDR: &[u8] = b"[mtld3d::unix] native backtrace:\n";
        // SAFETY: write(2) is async-signal-safe; the descriptor is the log file's or fd 2.
        unsafe {
            let _ = libc::write(
                crate::log_file::raw_fd(),
                HDR.as_ptr().cast::<c_void>(),
                HDR.len(),
            );
        }
        // SAFETY: `frames[..n]` were filled by `backtrace`; `backtrace_symbols_fd`
        // is async-signal-safe and writes the resolved frames to fd 2.
        unsafe { backtrace_symbols_fd(frames.as_ptr(), n, crate::log_file::raw_fd()) };
    }

    // Last resort for a jump-to-NULL whose frame chain is broken: scan the raw
    // stack for words that `dladdr` resolves into *our* dylib and symbolise
    // them. This reconstructs the call chain the frame-pointer walk can't —
    // the return addresses spilled by the calls leading to the bad jump are
    // still on the stack even when the frame pointer is garbage. Done last
    // because an unmapped read re-faults into the re-entrancy guard (`_exit`),
    // which would otherwise drop the crumb dump + native backtrace above.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        let sp = mcontext_u64(ctx, SP_OFFSET);
        if sp != 0 {
            const HDR: &[u8] = b"[mtld3d::unix] mtld3d.so return addrs on stack:\n";
            // SAFETY: write(2) is async-signal-safe; the descriptor is the log file's or fd 2.
            unsafe {
                let _ = libc::write(
                    crate::log_file::raw_fd(),
                    HDR.as_ptr().cast::<c_void>(),
                    HDR.len(),
                );
            }
            scan_stack_for_our_frames(sp);
        }
    }

    // SAFETY: _exit(2) is async-signal-safe; skips atexit handlers and
    // libc cleanup.
    unsafe { libc::_exit(1) };
}

/// Name a fault in someone else's native code before it is handed back.
///
/// Only when the PC resolves to an image: guest and translated code do not,
/// and a fault there is Wine's ordinary work. The line carries the thread id,
/// the PC as image plus offset (a load address on its own names nothing once
/// the process is gone) and the nearest symbol dyld knows, and is followed by
/// the return addresses into our dylib on the faulting stack, which is what
/// ties a fault in a framework to the call of ours that provoked it. Written
/// in pieces rather than one buffer, because an image path can be longer than
/// the line buffer. Stops after [`FOREIGN_REPORT_LIMIT`] reports, since a game
/// may probe with faults of its own and every one would cost the lookups.
///
/// `terminal` says the fault is not going back to anyone, so the cap does not
/// apply to it and the stack pass is left to the fatal report that follows,
/// which prints a superset of it.
fn report_foreign_fault(signo: c_int, ctx: *mut c_void, terminal: bool) {
    /// How many faults outside our image get a report.
    const FOREIGN_REPORT_LIMIT: u32 = 4;
    static REPORTS: AtomicU32 = AtomicU32::new(0);

    let pc = fault_pc(ctx);
    let Some(info) = dladdr_info(pc) else {
        return;
    };
    if REPORTS.fetch_add(1, Ordering::AcqRel) >= FOREIGN_REPORT_LIMIT && !terminal {
        return;
    }
    let fd = crate::log_file::raw_fd();
    let mut b = [0u8; 192];
    let mut p = 0;
    push(
        &mut b,
        &mut p,
        b"[mtld3d::unix] fault outside mtld3d.so: signo=",
    );
    push_decimal(&mut b, &mut p, signo.cast_unsigned());
    push(&mut b, &mut p, b" tid=");
    push_hex(&mut b, &mut p, thread_id());
    push(&mut b, &mut p, b" pc=");
    push_hex(&mut b, &mut p, pc);
    // The native first argument: for a fault in `objc_release` or
    // `objc_msgSend` it is the object that was gone.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        push(&mut b, &mut p, b" arg0=");
        push_hex(&mut b, &mut p, mcontext_u64(ctx, NATIVE_ARG0_OFFSET));
    }
    push(&mut b, &mut p, b" image=");
    // SAFETY: write(2) is async-signal-safe; the descriptor is the log file's or fd 2.
    unsafe {
        let _ = libc::write(fd, b.as_ptr().cast::<c_void>(), p);
    }
    // SAFETY: `dli_fname` is the NUL-terminated path dyld owns for a loaded
    // image; `strlen` is async-signal-safe.
    let image_len = unsafe { libc::strlen(info.dli_fname) };
    // SAFETY: write(2) is async-signal-safe; `image_len` bytes at `dli_fname`
    // are the path just measured.
    unsafe {
        let _ = libc::write(fd, info.dli_fname.cast::<c_void>(), image_len);
    }
    let mut b = [0u8; 192];
    let mut p = 0;
    push(&mut b, &mut p, b"+");
    push_hex(
        &mut b,
        &mut p,
        pc.wrapping_sub(info.dli_fbase as usize as u64),
    );
    if !info.dli_sname.is_null() {
        push(&mut b, &mut p, b" sym=");
        // SAFETY: `dli_sname` is the NUL-terminated symbol name dyld owns;
        // `strlen` is async-signal-safe.
        let name_len = unsafe { libc::strlen(info.dli_sname) };
        // SAFETY: `name_len` bytes at `dli_sname` are the name just measured.
        let name = unsafe { core::slice::from_raw_parts(info.dli_sname.cast::<u8>(), name_len) };
        push(&mut b, &mut p, &name[..name_len.min(96)]);
        push(&mut b, &mut p, b"+");
        push_hex(
            &mut b,
            &mut p,
            pc.wrapping_sub(info.dli_saddr as usize as u64),
        );
    }
    push(&mut b, &mut p, b" thread=");
    push_thread_name(&mut b, &mut p);
    push(&mut b, &mut p, b"\n");
    // SAFETY: as above.
    unsafe {
        let _ = libc::write(fd, b.as_ptr().cast::<c_void>(), p);
    }

    // The faulting stack. The fault may be one Wine recovers, so the scan
    // must not fault itself: the kernel copies each word or reports failure.
    // A mapping probe cannot establish readability, and a Wine thread runs
    // unix calls on a stack Wine allocated, not its pthread stack.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        let sp = mcontext_u64(ctx, SP_OFFSET);
        if sp != 0 && !terminal {
            const HDR: &[u8] = b"[mtld3d::unix] mtld3d.so return addrs on stack:\n";
            // SAFETY: write(2) is async-signal-safe; the descriptor is the log file's or fd 2.
            unsafe {
                let _ = libc::write(fd, HDR.as_ptr().cast::<c_void>(), HDR.len());
            }
            our_frames_on_stack(sp, STACK_SCAN_WORDS);
        }
    }
}

/// The calling thread's system-wide id, the one `ps -M` and a sample show.
fn thread_id() -> u64 {
    let mut id = 0u64;
    // SAFETY: `pthread_threadid_np` with a null thread reads the calling
    // thread's own id into `id`; it touches thread-local state only.
    unsafe {
        libc::pthread_threadid_np(0, &raw mut id);
    }
    id
}

/// Append the calling thread's name.
///
/// `pthread_getname_np` only reads thread-local storage, which is
/// signal-safe enough for a terminating handler. A thread nobody named is
/// named by what it is instead, because the one that matters here carries no
/// name: the process's main thread is `AppKit`'s, and a fault under
/// `winemac`'s Cocoa event loop lands on it.
fn push_thread_name(buf: &mut [u8; 192], pos: &mut usize) {
    let mut name = [0u8; 64];
    // SAFETY: `pthread_self` is always safe to call; reads the current TLS.
    let tid = unsafe { libc::pthread_self() };
    // SAFETY: writes a NUL-terminated name (≤ len) into the buffer.
    unsafe {
        pthread_getname_np(
            tid,
            name.as_mut_ptr().cast::<core::ffi::c_char>(),
            name.len(),
        );
    }
    let nlen = name.iter().position(|&b| b == 0).unwrap_or(name.len());
    if nlen == 0 {
        // SAFETY: `pthread_main_np` compares the calling thread against the
        // process's first one and reads nothing else.
        let label: &[u8] = if unsafe { libc::pthread_main_np() } == 0 {
            b"unnamed"
        } else {
            b"cocoa main"
        };
        push(buf, pos, label);
        return;
    }
    push(buf, pos, &name[..nlen.min(96)]);
}

/// Scan the raw stack for return addresses into our own dylib and print them.
///
/// Walks up to 4096 words from `sp` and `backtrace_symbols_fd`-prints each
/// value that `dladdr` resolves into a module whose file name is ours.
/// Caps the printed count so a deep stack can't flood. Stack words are copied
/// by the kernel into local storage, without dereferencing the faulting stack.
/// Arch-neutral: a spilled return address is a stack word on both arches, and
/// the 4-byte step below already tolerates either alignment.
fn scan_stack_for_our_frames(sp: u64) {
    /// Header for the guest-stack pass below.
    const GUEST_HDR: &[u8] = b"[mtld3d::unix] guest (PE) stack words:\n";
    /// Print cap for the guest pass.
    const GUEST_CAP: u32 = 64;

    our_frames_on_stack(sp, STACK_SCAN_WORDS);
    // Second pass: the 32-bit guest stack. `dladdr` can't see Wine's PE
    // builtins (not dyld images), so collect raw 4-byte words that land in the
    // PE-builtin zone [0x7A00_0000, 0x7C00_0000) (ntdll/user32/win32u/d3d9/…)
    // or the guest EXE image [0x0040_0000, 0x0080_0000) — covering all of the
    // guest client's `.text` (up to ~0x7ff000), not just the first page, so the
    // real guest call chain (its 0x4xxxxx–0x7xxxxx return addresses) is shown,
    // mapped to modules by their logged load bases.
    //
    // SAFETY: write(2) is async-signal-safe; the descriptor is the log file's or fd 2.
    unsafe {
        let _ = libc::write(
            crate::log_file::raw_fd(),
            GUEST_HDR.as_ptr().cast::<c_void>(),
            GUEST_HDR.len(),
        );
    }
    let mut guest_printed = 0u32;
    let mut slot = 0usize;
    while slot < STACK_SCAN_WORDS && guest_printed < GUEST_CAP {
        let Some(bytes) = stack_word(sp, slot) else {
            break;
        };
        let guest_addr = u32::from_ne_bytes(bytes);
        let in_builtin = (0x7A00_0000..0x7C00_0000).contains(&guest_addr);
        let in_exe = (0x0040_0000..0x0080_0000).contains(&guest_addr);
        if in_builtin || in_exe {
            let mut b = [0u8; 192];
            let mut p = 0;
            push(&mut b, &mut p, b"  g=");
            push_hex(&mut b, &mut p, u64::from(guest_addr));
            push(&mut b, &mut p, b"\n");
            // SAFETY: write(2) is async-signal-safe; the descriptor is the log file's or fd 2.
            unsafe {
                let _ = libc::write(crate::log_file::raw_fd(), b.as_ptr().cast::<c_void>(), p);
            }
            guest_printed += 1;
        }
        slot += 1;
    }
}

/// Stack words a scan inspects from the faulting stack pointer upward.
const STACK_SCAN_WORDS: usize = 4096;

/// Print the return addresses into our own dylib among `words` stack words above `sp`.
///
/// `backtrace_symbols_fd`-prints each value that `dladdr` resolves into a
/// module whose file name is ours, capped so a deep stack can't flood.
/// The 4-byte step tolerates a 32-bit guest stack's alignment; a spilled
/// return address is a stack word on both arches.
fn our_frames_on_stack(sp: u64, words: usize) {
    /// Print cap for the pass.
    const OURS_CAP: u32 = 48;

    let mut printed = 0u32;
    let mut slot = 0usize;
    while slot < words && printed < OURS_CAP {
        let Some(bytes) = stack_word(sp, slot) else {
            break;
        };
        let addr = u64::from_ne_bytes(bytes);
        if addr >= 0x1000 && dladdr_is_ours(addr) {
            let mut frame = [addr as *mut c_void; 1];
            // SAFETY: single in-bounds frame pointer; `backtrace_symbols_fd`
            // resolves via `dladdr` and writes to fd 2.
            unsafe { backtrace_symbols_fd(frame.as_mut_ptr(), 1, crate::log_file::raw_fd()) };
            printed += 1;
        }
        slot += 1;
    }
}

/// Copy a stack word without dereferencing the faulting stack.
///
/// The address is `sp` plus four bytes per slot, the step that tolerates a
/// 32-bit guest stack's alignment. Only a complete copy is used; a failed
/// read writes directly to the crash descriptor and ends that scan.
fn stack_word<const N: usize>(sp: u64, slot: usize) -> Option<[u8; N]> {
    const UNREADABLE: &[u8] = b"[mtld3d::unix] stack read unavailable, stopping scan\n";

    let address = slot
        .checked_mul(4)
        .and_then(|offset| sp.checked_add(offset as u64));
    let word = address.and_then(copy_word::<N>);
    if word.is_none() {
        // SAFETY: write(2) is async-signal-safe; the bytes have static storage.
        unsafe {
            let _ = libc::write(
                crate::log_file::raw_fd(),
                UNREADABLE.as_ptr().cast::<c_void>(),
                UNREADABLE.len(),
            );
        }
    }
    word
}

/// Copy the `index`th `N`-byte word of the object at `base`, without dereferencing it.
///
/// The object's words are consecutive, so the address is `base` plus `N`
/// bytes per index. `None` for an address that overflows or that the kernel
/// declines to copy; the caller decides what a failed read ends.
#[cfg(target_arch = "x86_64")]
fn object_word<const N: usize>(base: u64, index: usize) -> Option<[u8; N]> {
    index
        .checked_mul(N)
        .and_then(|offset| base.checked_add(offset as u64))
        .and_then(copy_word::<N>)
}

/// Copy `N` bytes at `address` into local storage through the kernel.
///
/// The Mach call reports an unreadable source, including a mapping whose
/// protection changes during the copy, instead of faulting the caller; a
/// mapping probe could not promise that. Only a complete copy is returned.
/// No allocation and no lock, so a signal handler may ask.
fn copy_word<const N: usize>(address: u64) -> Option<[u8; N]> {
    address.checked_add(N as u64)?;
    let mut bytes = [0u8; N];
    let mut copied = 0;
    // SAFETY: reads libSystem's task port for this process.
    let task = unsafe { mach_task_self_ };
    // SAFETY: the kernel validates the source address. The destination
    // holds N writable bytes, and `copied` is a live size out-parameter.
    let status = unsafe {
        mach_vm_read_overwrite(
            task,
            address,
            N as u64,
            bytes.as_mut_ptr() as usize as u64,
            &raw mut copied,
        )
    };
    (status == libc::KERN_SUCCESS && copied == N as u64).then_some(bytes)
}

/// What dyld knows about `addr`: its image, and the nearest symbol below it.
///
/// `None` outside every image: guest (PE) pages and translated code are not
/// images dyld knows. `dladdr` allocates nothing, which is what lets a signal
/// handler ask.
fn dladdr_info(addr: u64) -> Option<libc::Dl_info> {
    // SAFETY: zeroed `Dl_info` is a valid out-param for `dladdr`.
    let mut info: libc::Dl_info = unsafe { mem::zeroed() };
    // SAFETY: `dladdr` reads `addr` only as an opaque value and fills `info`.
    let ok = unsafe { libc::dladdr(addr as *const c_void, &raw mut info) };
    if ok == 0 || info.dli_fname.is_null() {
        return None;
    }
    Some(info)
}

/// The path of the loaded image `addr` lies in, or `None` outside every image.
fn dladdr_image(addr: u64) -> Option<*const core::ffi::c_char> {
    dladdr_info(addr).map(|info| info.dli_fname)
}

/// True when `addr` resolves (via `dladdr`) into our own `.so`.
///
/// The match is on a loaded image whose file name begins with `mtld3d`.
/// Filters stack garbage and libsystem/Wine/Metal frames down to our own call
/// chain.
fn dladdr_is_ours(addr: u64) -> bool {
    let Some(path) = dladdr_image(addr) else {
        return false;
    };
    path_names_our_image(path)
}

/// Whether the NUL-terminated image path names one of our own images.
///
/// The file name decides, not the path. A directory anywhere above the image
/// can carry our name, and matching the whole path then claims every image
/// under it: Wine installed beside us is the layout where that matters, where claiming
/// its `ntdll.so` sends a fault Wine would have turned into a Windows
/// exception down the terminal path instead of back to it.
///
/// Split from [`dladdr_is_ours`] so the suite can pin the rule without a live
/// `dladdr`.
fn path_names_our_image(path: *const core::ffi::c_char) -> bool {
    /// Matched against the image's file name.
    const NEEDLE: &[u8] = b"mtld3d";

    let len = image_path_len(path);
    // The file name starts one past the last separator, or at the beginning
    // when the path carries none.
    let start = (0..len)
        .rfind(|&i| image_path_byte(path, i) == b'/')
        .map_or(0, |i| i + 1);
    // It has to begin with the needle, so `mtld3d.so` matches and an
    // unrelated `ntdll.so` under an `mtld3d`-named directory does not.
    len - start >= NEEDLE.len()
        && NEEDLE
            .iter()
            .enumerate()
            .all(|(i, &want)| image_path_byte(path, start + i) == want)
}

/// Bound on every scan of an image path, so a corrupt `dli_fname` can't spin.
const PATH_MAX_SCAN: usize = 4096;

/// Length of the NUL-terminated image path dyld owns, bounded by [`PATH_MAX_SCAN`].
const fn image_path_len(path: *const core::ffi::c_char) -> usize {
    let mut len = 0usize;
    while len < PATH_MAX_SCAN && image_path_byte(path, len) != 0 {
        len += 1;
    }
    len
}

/// One byte of the NUL-terminated image path dyld owns.
///
/// Callers stay within the length [`image_path_len`] measured, or stop at
/// the first zero this returns.
const fn image_path_byte(path: *const core::ffi::c_char, index: usize) -> u8 {
    // SAFETY: an offset within a NUL-terminated C string owned by dyld, which
    // the caller bounds by its measured length or by the NUL itself.
    let at = unsafe { path.cast::<u8>().add(index) };
    // SAFETY: as above; reads one byte of that string.
    unsafe { at.read() }
}

/// True when the image path dyld owns ends in `suffix`.
#[cfg(target_arch = "x86_64")]
fn image_path_ends_with(path: *const core::ffi::c_char, suffix: &[u8]) -> bool {
    let len = image_path_len(path);
    if len < suffix.len() {
        return false;
    }
    let start = len - suffix.len();
    suffix
        .iter()
        .enumerate()
        .all(|(i, &byte)| image_path_byte(path, start + i) == byte)
}

/// Append the last component of the image path dyld owns, at most 32 bytes of it.
#[cfg(target_arch = "x86_64")]
fn push_image_basename(buf: &mut [u8; 192], pos: &mut usize, path: *const core::ffi::c_char) {
    /// The longest basename a line carries; keeps the whole line in its buffer.
    const BASENAME_MAX: usize = 32;

    let len = image_path_len(path);
    let mut start = 0usize;
    for index in 0..len {
        if image_path_byte(path, index) == b'/' {
            start = index + 1;
        }
    }
    let mut name = [0u8; BASENAME_MAX];
    let take = (len - start).min(BASENAME_MAX);
    for (i, slot) in name.iter_mut().enumerate().take(take) {
        *slot = image_path_byte(path, start + i);
    }
    push(buf, pos, &name[..take]);
}

/// Words copied out of the object a `CoreFoundation` trap holds in `rbx`.
///
/// Enough to cover a run-loop timer in every layout the framework has had:
/// the runtime base, the lock, the run loop and mode pointers, the fire date,
/// the interval, the tolerance, the fire ticks, the order, the callout and
/// the context. Fixed, so the dump's length is too.
#[cfg(target_arch = "x86_64")]
const TRAP_OBJECT_WORDS: usize = 24;

/// Dump the object a `CoreFoundation` trap holds in `rbx`, without touching it.
///
/// A framework trap (`SIGILL` on a deliberate `ud2`) names its condition in a
/// string and nothing else, while the object it was checking is in `rbx`,
/// which the trapping function keeps live across the cold call. Each word of
/// that object is copied by the kernel into local storage, printed as hex at
/// its offset, and followed by the image and symbol `dladdr` resolves it to,
/// so a class pointer, a callout or a context reads as its owner while the
/// handler assumes nothing about the layout. Only a `SIGILL` whose PC lies in
/// `CoreFoundation` qualifies: a trap in our own code has its own report, and
/// a memory fault's `rbx` is arbitrary. The first unreadable word ends the
/// dump, and nothing here dereferences the object or calls into the framework.
#[cfg(target_arch = "x86_64")]
fn report_trap_object(signo: c_int, ctx: *mut c_void) {
    /// The image whose traps carry their object in `rbx`.
    const IMAGE_SUFFIX: &[u8] = b"/CoreFoundation";
    const HDR: &[u8] = b"[mtld3d::unix] CoreFoundation trap object words at rbx:\n";
    const UNREADABLE: &[u8] = b"[mtld3d::unix] trap object read unavailable, stopping dump\n";
    /// The longest symbol name a word's line carries.
    const SYMBOL_MAX: usize = 64;

    if signo != libc::SIGILL {
        return;
    }
    let Some(trap) = dladdr_info(fault_pc(ctx)) else {
        return;
    };
    if !image_path_ends_with(trap.dli_fname, IMAGE_SUFFIX) {
        return;
    }
    let base = mcontext_u64(ctx, mem::offset_of!(libc::__darwin_mcontext64, __ss.__rbx));
    if base == 0 {
        return;
    }
    let fd = crate::log_file::raw_fd();
    // SAFETY: write(2) is async-signal-safe; the bytes have static storage.
    unsafe {
        let _ = libc::write(fd, HDR.as_ptr().cast::<c_void>(), HDR.len());
    }
    for index in 0..TRAP_OBJECT_WORDS {
        let Some(bytes) = object_word::<8>(base, index) else {
            // SAFETY: as above.
            unsafe {
                let _ = libc::write(fd, UNREADABLE.as_ptr().cast::<c_void>(), UNREADABLE.len());
            }
            return;
        };
        let word = u64::from_ne_bytes(bytes);
        let mut b = [0u8; 192];
        let mut p = 0;
        push(&mut b, &mut p, b"  +");
        push_hex(&mut b, &mut p, (index * 8) as u64);
        push(&mut b, &mut p, b" ");
        push_hex(&mut b, &mut p, word);
        if word >= 0x1000
            && let Some(target) = dladdr_info(word)
        {
            push(&mut b, &mut p, b" ");
            push_image_basename(&mut b, &mut p, target.dli_fname);
            push(&mut b, &mut p, b"+");
            push_hex(
                &mut b,
                &mut p,
                word.wrapping_sub(target.dli_fbase as usize as u64),
            );
            if !target.dli_sname.is_null() {
                // SAFETY: `dli_sname` is the NUL-terminated symbol name dyld
                // owns; `strlen` is async-signal-safe.
                let name_len = unsafe { libc::strlen(target.dli_sname) };
                // SAFETY: `name_len` bytes at `dli_sname` are the name just measured.
                let name =
                    unsafe { core::slice::from_raw_parts(target.dli_sname.cast::<u8>(), name_len) };
                push(&mut b, &mut p, b" ");
                push(&mut b, &mut p, &name[..name_len.min(SYMBOL_MAX)]);
                push(&mut b, &mut p, b"+");
                push_hex(
                    &mut b,
                    &mut p,
                    word.wrapping_sub(target.dli_saddr as usize as u64),
                );
            }
        }
        push(&mut b, &mut p, b"\n");
        // SAFETY: write(2) is async-signal-safe; the buffer holds p initialized bytes.
        unsafe {
            let _ = libc::write(fd, b.as_ptr().cast::<c_void>(), p);
        }
    }
}

/// Capacity of the frame buffer handed to `backtrace`.
const FRAME_CAP: c_int = 64;

/// Byte offset of the saved stack pointer within `__darwin_mcontext64`.
///
/// `x86_64` `__rsp`; `arm64` `__sp`, which follows `__x[29]`, `__fp`, `__lr`.
#[cfg(target_arch = "x86_64")]
const SP_OFFSET: usize = 72;
#[cfg(target_arch = "aarch64")]
const SP_OFFSET: usize = 264;

/// Byte offset of the register carrying a called method's first argument.
///
/// `x86_64` `__rcx` is the Win64 first argument, i.e. a COM call's `this`;
/// `arm64` `__x0` is the AAPCS64 first argument, the same role.
#[cfg(target_arch = "x86_64")]
const ARG0_OFFSET: usize = 32;
#[cfg(target_arch = "aarch64")]
const ARG0_OFFSET: usize = 16;

/// Byte offset of the register carrying a native call's first argument.
///
/// `x86_64` `__rdi`, the System V first argument, which is what a framework
/// or the Objective-C runtime was handed; `arm64` `__x0`, the same role.
#[cfg(target_arch = "x86_64")]
const NATIVE_ARG0_OFFSET: usize = 48;
#[cfg(target_arch = "aarch64")]
const NATIVE_ARG0_OFFSET: usize = 16;

/// Byte offset of `__rax`, which holds the vtable pointer a `CALL` loaded.
#[cfg(target_arch = "x86_64")]
const RAX_OFFSET: usize = 16;

/// Byte offset of `__lr`, where `BLR` leaves the return address.
#[cfg(target_arch = "aarch64")]
const LR_OFFSET: usize = 256;

/// Label for the first-argument register in the fault report.
#[cfg(target_arch = "x86_64")]
const ARG0_LABEL: &[u8] = b"rcx(this)=";
#[cfg(target_arch = "aarch64")]
const ARG0_LABEL: &[u8] = b"x0(this)=";

/// Label for the caller address in the fault report, naming where it came from.
#[cfg(target_arch = "x86_64")]
const CALLER_LABEL: &[u8] = b"caller(ret@rsp)=";
#[cfg(target_arch = "aarch64")]
const CALLER_LABEL: &[u8] = b"caller(lr)=";

/// Retain architectural nonvolatile registers and flags from the faulting context.
///
/// Darwin places the 64-bit thread state after its 16-byte exception state.
/// The libc fields mirror that SDK layout, including the 64-bit RFLAGS slot.
/// A separate line fits all seven full-width values and its newline in the
/// existing 192-byte buffer without crowding out the argument or stack fields.
#[cfg(target_arch = "x86_64")]
fn report_nonvolatile_registers(ctx: *mut c_void) {
    const REGISTERS: [(&[u8], usize); 7] = [
        (
            b"rbx=",
            mem::offset_of!(libc::__darwin_mcontext64, __ss.__rbx),
        ),
        (
            b" rbp=",
            mem::offset_of!(libc::__darwin_mcontext64, __ss.__rbp),
        ),
        (
            b" r12=",
            mem::offset_of!(libc::__darwin_mcontext64, __ss.__r12),
        ),
        (
            b" r13=",
            mem::offset_of!(libc::__darwin_mcontext64, __ss.__r13),
        ),
        (
            b" r14=",
            mem::offset_of!(libc::__darwin_mcontext64, __ss.__r14),
        ),
        (
            b" r15=",
            mem::offset_of!(libc::__darwin_mcontext64, __ss.__r15),
        ),
        (
            b" rflags=",
            mem::offset_of!(libc::__darwin_mcontext64, __ss.__rflags),
        ),
    ];

    let mut buf = [0u8; 192];
    let mut pos = 0;
    push(&mut buf, &mut pos, b"[mtld3d::unix] ");
    for (label, offset) in REGISTERS {
        push(&mut buf, &mut pos, label);
        push_hex(&mut buf, &mut pos, mcontext_u64(ctx, offset));
    }
    push(&mut buf, &mut pos, b"\n");
    // SAFETY: write(2) is async-signal-safe; the buffer holds pos initialized bytes.
    unsafe {
        let _ = libc::write(
            crate::log_file::raw_fd(),
            buf.as_ptr().cast::<c_void>(),
            pos,
        );
    }
}

/// The return address of the frame that faulted, or 0 if it can't be read.
///
/// `x86_64` `CALL` pushes it, so for a jump-through-garbage fault (which faults
/// at the callee's first instruction, before any prologue) it is the word at
/// `[rsp]`; an unreadable `rsp` reports no return address.
#[cfg(target_arch = "x86_64")]
fn caller_pc(_ctx: *mut c_void, sp: u64) -> u64 {
    stack_word(sp, 0).map_or(0, u64::from_ne_bytes)
}

/// The return address of the frame that faulted, or 0 if it can't be read.
///
/// `arm64` `BLR` leaves it in `__lr` rather than on the stack, so this needs no
/// memory read at all and stays valid even when the stack pointer is garbage.
#[cfg(target_arch = "aarch64")]
const fn caller_pc(ctx: *mut c_void, _sp: u64) -> u64 {
    mcontext_u64(ctx, LR_OFFSET)
}

/// The faulting program counter from a signal `ucontext`, or 0 if it can't be read.
///
/// `uc_mcontext` is a pointer to `__darwin_mcontext64`, sitting at byte offset 0x30 in
/// `ucontext_t` (same on both macOS arches). The PC offset *within* the
/// `mcontext` is arch-specific: `x86_64` `__rip` follows the 16-byte exception
/// state + 16 thread-state `u64`s (144); `arm64` `__pc` follows the 16-byte
/// exception state + 32 thread-state `u64`s (272). Both are shipped: the `.so`
/// follows the arch of the Wine that loads it.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const fn fault_pc(ctx: *mut c_void) -> u64 {
    #[cfg(target_arch = "x86_64")]
    const PC_OFFSET: usize = 144;
    #[cfg(target_arch = "aarch64")]
    const PC_OFFSET: usize = 272;

    if ctx.is_null() {
        return 0;
    }
    // SAFETY: `ctx` is a non-null `ucontext_t*` from the kernel; `uc_mcontext`
    // lives at +0x30 and stays valid for the handler's lifetime.
    let mctx_field = unsafe { ctx.cast::<u8>().add(0x30) };
    // SAFETY: reads the `uc_mcontext` pointer (unaligned-safe, no write).
    let mctx = unsafe { mctx_field.cast::<*const u8>().read_unaligned() };
    if mctx.is_null() {
        return 0;
    }
    // SAFETY: `mctx` points at a live `__darwin_mcontext64`; the PC lives at
    // `PC_OFFSET` within it.
    let pc_field = unsafe { mctx.add(PC_OFFSET) };
    // SAFETY: reads the saved PC (unaligned-safe, no write).
    unsafe { pc_field.cast::<u64>().read_unaligned() }
}

/// Architectures where we can't decode the saved PC: report 0.
///
/// The handler falls back to the frame-pointer backtrace alone.
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
const fn fault_pc(_ctx: *mut c_void) -> u64 {
    0
}

/// Read a `u64` at byte `offset` within the signal `ucontext`'s `mcontext`.
///
/// Offsets follow `__darwin_mcontext64` for the arch this built for: a 16-byte
/// exception state, then the thread-state registers. `x86_64` has `rax` at 16,
/// `rcx` at 32, `rsp` at 72, `rip` at 144; `arm64` has `x0` at 16 (the rest of
/// `__x[29]` following), then `fp` at 248, `lr` at 256, `sp` at 264, `pc` at
/// 272. Returns 0 if the context can't be read.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const fn mcontext_u64(ctx: *mut c_void, offset: usize) -> u64 {
    if ctx.is_null() {
        return 0;
    }
    // SAFETY: `ctx` is a non-null `ucontext_t*`; `uc_mcontext` lives at +0x30.
    let mctx_field = unsafe { ctx.cast::<u8>().add(0x30) };
    // SAFETY: reads the `uc_mcontext` pointer (unaligned-safe, no write).
    let mctx = unsafe { mctx_field.cast::<*const u8>().read_unaligned() };
    if mctx.is_null() {
        return 0;
    }
    // SAFETY: `mctx` points at a live `__darwin_mcontext64`; `offset` is within it.
    let field = unsafe { mctx.add(offset) };
    // SAFETY: reads the saved register (unaligned-safe, no write).
    unsafe { field.cast::<u64>().read_unaligned() }
}

const fn signal_name(signo: libc::c_int) -> &'static [u8] {
    match signo {
        libc::SIGSEGV => b"SIGSEGV",
        libc::SIGBUS => b"SIGBUS",
        libc::SIGABRT => b"SIGABRT",
        libc::SIGILL => b"SIGILL",
        _ => b"SIG?",
    }
}

fn push(buf: &mut [u8; 192], pos: &mut usize, bytes: &[u8]) {
    let avail = buf.len() - *pos;
    let take = bytes.len().min(avail);
    buf[*pos..*pos + take].copy_from_slice(&bytes[..take]);
    *pos += take;
}

/// Append `v` in decimal, the form a signal number is read in.
fn push_decimal(buf: &mut [u8; 192], pos: &mut usize, v: u32) {
    let mut digits = [0u8; 10];
    let mut n = digits.len();
    let mut v = v;
    loop {
        n -= 1;
        digits[n] = b'0' + u8::try_from(v % 10).expect("a digit fits u8");
        v /= 10;
        if v == 0 {
            break;
        }
    }
    push(buf, pos, &digits[n..]);
}

fn push_hex(buf: &mut [u8; 192], pos: &mut usize, v: u64) {
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

#[cfg(test)]
mod tests;
