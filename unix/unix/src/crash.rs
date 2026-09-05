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
//! addresses into our own dylib found on the faulting stack. A fault there on
//! a thread Wine does not own (ours, or a Metal completion thread) ends the
//! process inside Wine's handler with nothing else said, and the game's log
//! then has to name the image, the thread and our frames under it at least.
//! Guest code and translated code resolve to no image and stay silent: those
//! faults are Wine's ordinary work. The report is capped at a few per process
//! and carries no signal name, since a fault Wine recovers
//! must not read as a crash to anything that scans the log for one.
//!
//! `RUST_BACKTRACE=1` is also set here (if unset) so the default Rust
//! panic hook prints message + backtrace before `abort()` flows through
//! to the SIGABRT branch.

use core::{
    ffi::{c_int, c_void},
    mem, ptr,
};
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, AtomicUsize, Ordering};

use mtld3d_shared::crumb;

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
        report_foreign_fault(signo, ctx);
        forward_to_previous(signo, info, ctx);
        return;
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
    push(&mut buf, &mut pos, b"[mtld3d::unix] FATAL: ");
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
fn report_foreign_fault(signo: c_int, ctx: *mut c_void) {
    /// How many faults outside our image get a report.
    const FOREIGN_REPORT_LIMIT: u32 = 4;
    static REPORTS: AtomicU32 = AtomicU32::new(0);

    let pc = fault_pc(ctx);
    let Some(info) = dladdr_info(pc) else {
        return;
    };
    if REPORTS.fetch_add(1, Ordering::AcqRel) >= FOREIGN_REPORT_LIMIT {
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

    // The faulting stack, bounded by the thread's own stack top: the fault
    // may be one Wine recovers, and a read past the top would re-fault into
    // the re-entrancy guard and end a process that would have lived.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        let sp = mcontext_u64(ctx, SP_OFFSET);
        let words = stack_words_above(sp);
        if words > 0 {
            let mut b = [0u8; 192];
            let mut p = 0;
            push(
                &mut b,
                &mut p,
                b"[mtld3d::unix] mtld3d.so return addrs on stack (",
            );
            push_decimal(&mut b, &mut p, u32::try_from(words).unwrap_or(u32::MAX));
            push(&mut b, &mut p, b" words scanned):\n");
            // SAFETY: write(2) is async-signal-safe; the descriptor is the log file's or fd 2.
            unsafe {
                let _ = libc::write(fd, b.as_ptr().cast::<c_void>(), p);
            }
            our_frames_on_stack(sp, words);
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

/// How many 4-byte words lie between `sp` and the calling thread's stack top.
///
/// Capped at [`STACK_SCAN_WORDS`]. Zero when the top is unknown or below
/// `sp`, which skips the scan rather than guessing.
fn stack_words_above(sp: u64) -> usize {
    // SAFETY: `pthread_self` is always safe to call; reads the current TLS.
    let me = unsafe { libc::pthread_self() };
    // SAFETY: `pthread_get_stackaddr_np` on the calling thread reads its
    // pthread record; it is the documented way to the stack's high end.
    let top = unsafe { libc::pthread_get_stackaddr_np(me) } as usize as u64;
    if top <= sp {
        return 0;
    }
    usize::try_from((top - sp) / 4).map_or(STACK_SCAN_WORDS, |w| w.min(STACK_SCAN_WORDS))
}

/// Append the calling thread's name.
///
/// `pthread_getname_np` only reads thread-local storage, which is
/// signal-safe enough for a terminating handler.
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
    push(buf, pos, &name[..nlen.min(96)]);
}

/// Scan the raw stack for return addresses into our own dylib and print them.
///
/// Walks up to 4096 words from `sp` and `backtrace_symbols_fd`-prints each
/// value that `dladdr` resolves into a module whose path contains `mtld3d`.
/// Caps the printed count so a deep stack can't flood. Async-signal-safe: only
/// `dladdr` (no malloc on the resolve path here) + `backtrace_symbols_fd` + raw
/// stack reads. Arch-neutral: a spilled return address is a stack word on both
/// arches, and the 4-byte step below already tolerates either alignment.
fn scan_stack_for_our_frames(sp: u64) {
    /// Header for the guest-stack pass below.
    const GUEST_HDR: &[u8] = b"[mtld3d::unix] guest (PE) stack words:\n";
    /// Print cap for the guest pass.
    const GUEST_CAP: u32 = 64;

    our_frames_on_stack(sp, STACK_SCAN_WORDS);
    let base = sp as *const u8;
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
        // SAFETY: as the loop above, an offset into stack memory near `sp`.
        let word = unsafe { base.add(slot * 4) };
        // SAFETY: as above; an unmapped read re-faults into the re-entrancy
        // guard, bounding the walk.
        let guest_addr = unsafe { word.cast::<u32>().read_unaligned() };
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
/// module whose path contains `mtld3d`, capped so a deep stack can't flood.
/// The 4-byte step tolerates a 32-bit guest stack's alignment; a spilled
/// return address is a stack word on both arches.
fn our_frames_on_stack(sp: u64, words: usize) {
    /// Print cap for the pass.
    const OURS_CAP: u32 = 48;

    let base = sp as *const u8;
    let mut printed = 0u32;
    let mut slot = 0usize;
    while slot < words && printed < OURS_CAP {
        // SAFETY: an offset into stack memory near `sp`; an unmapped read below
        // re-faults into the re-entrancy guard (terminating), which bounds the
        // walk.
        let word = unsafe { base.add(slot * 4) };
        // SAFETY: as above; `read_unaligned` tolerates a 4-byte-aligned 32-bit
        // stack.
        let addr = unsafe { word.cast::<u64>().read_unaligned() };
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
/// The match is on a loaded image whose filename contains the bytes `mtld3d`.
/// Filters stack garbage and libsystem/Wine/Metal frames down to our own call
/// chain.
fn dladdr_is_ours(addr: u64) -> bool {
    /// Scanned for in the image path, without allocating.
    const NEEDLE: &[u8] = b"mtld3d";
    /// Bound on the path scan, so a corrupt `dli_fname` can't spin.
    const PATH_MAX_SCAN: usize = 4096;

    let Some(path) = dladdr_image(addr) else {
        return false;
    };
    let path = path.cast::<u8>();
    let mut idx = 0usize;
    let mut matched = 0usize;
    while idx < PATH_MAX_SCAN {
        // SAFETY: an offset within a NUL-terminated C string owned by dyld,
        // bounded by the NUL check below.
        let at = unsafe { path.add(idx) };
        // SAFETY: as above; reads one byte of that string.
        let byte = unsafe { at.read() };
        if byte == 0 {
            break;
        }
        matched = if byte == NEEDLE[matched] {
            matched + 1
        } else {
            usize::from(byte == NEEDLE[0])
        };
        if matched == NEEDLE.len() {
            return true;
        }
        idx += 1;
    }
    false
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

/// The return address of the frame that faulted, or 0 if it can't be read.
///
/// `x86_64` `CALL` pushes it, so for a jump-through-garbage fault (which faults
/// at the callee's first instruction, before any prologue) it is the word at
/// `[rsp]`; a bad `rsp` re-faults into the re-entrancy guard rather than
/// looping.
#[cfg(target_arch = "x86_64")]
fn caller_pc(_ctx: *mut c_void, sp: u64) -> u64 {
    if sp == 0 {
        return 0;
    }
    // SAFETY: `sp` is the faulting stack pointer, whose top word is the return
    // address the faulting `CALL` pushed. An unmapped read terminates through
    // the re-entrancy guard.
    unsafe { (sp as *const u64).read() }
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
/// `uc_mcontext` is a pointer to an opaque `__darwin_mcontext64` (the `libc`
/// crate exposes it only as padding), sitting at byte offset 0x30 in
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

// Declared here because the `libc` crate does not expose the macOS
// `<execinfo.h>` family; both live in libSystem, which is always linked.
unsafe extern "C" {
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
