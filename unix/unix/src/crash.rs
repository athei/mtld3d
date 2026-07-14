//! Always-on unix-side crash handler.
//!
//! Installed once from `init_logger_handler`. Catches SIGSEGV, SIGBUS,
//! SIGABRT. The handler is async-signal-safe — it only calls
//! `libc::write` on fd 2 and `mtld3d_shared::crumb::dump_recent` (which is
//! itself async-signal-safe). On a fatal signal the handler:
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
//! `RUST_BACKTRACE=1` is also set here (if unset) so the default Rust
//! panic hook prints message + backtrace before `abort()` flows through
//! to the SIGABRT branch.

use core::{
    ffi::{c_int, c_void},
    mem, ptr,
};
use std::sync::atomic::{AtomicBool, Ordering};

use mtld3d_shared::crumb;

/// Re-entrancy guard: reading the faulting stack/registers can itself fault on a corrupted context.
///
/// With `SA_NODEFER` that would re-enter `handler`; the first re-entry exits
/// immediately so we never loop.
static IN_HANDLER: AtomicBool = AtomicBool::new(false);

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
    // SAFETY: sigaction(2) with valid `act`; ignoring the prior handler
    // because we never chain.
    unsafe {
        libc::sigaction(signo, &raw const act, ptr::null_mut());
    }
}

extern "C" fn handler(signo: libc::c_int, info: *mut libc::siginfo_t, ctx: *mut c_void) {
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

    // SAFETY: write(2) on fd 2 is async-signal-safe.
    unsafe {
        let _ = libc::write(2, buf.as_ptr().cast::<c_void>(), pos);
    }

    // Faulting thread name. For a teardown race the *which thread* (API vs
    // `mtld3d-encoder` / `mtld3d-submit` / `mtld3d-prewarm`) is the first clue.
    // `pthread_getname_np` only reads thread-local storage — signal-safe enough
    // for a terminating handler.
    {
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
        let mut b = [0u8; 192];
        let mut p = 0;
        push(&mut b, &mut p, b"[mtld3d::unix] thread=");
        push(&mut b, &mut p, &name[..nlen.min(96)]);
        push(&mut b, &mut p, b"\n");
        // SAFETY: write(2) on fd 2 is async-signal-safe.
        unsafe {
            let _ = libc::write(2, b.as_ptr().cast::<c_void>(), p);
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
        // SAFETY: write(2) on fd 2 is async-signal-safe.
        unsafe {
            let _ = libc::write(2, b.as_ptr().cast::<c_void>(), p);
        }
        let mut frame = [rip as *mut c_void; 1];
        // SAFETY: single in-bounds frame pointer; `backtrace_symbols_fd` is
        // async-signal-safe (resolves via `dladdr`, no malloc) and writes to fd 2.
        unsafe { backtrace_symbols_fd(frame.as_mut_ptr(), 1, 2) };
    }

    // For a jump-through-garbage fault (`fault_pc` is a tiny/invalid value), the
    // saved registers + top-of-stack name the culprit: `rcx` is the Win64 first
    // argument (a COM call's `this` — the freed object), `rax` the loaded vtable,
    // and the return address the faulting `CALL` pushed at `[rsp]` is the caller.
    #[cfg(target_arch = "x86_64")]
    {
        let rsp = mcontext_u64(ctx, 72);
        let rcx = mcontext_u64(ctx, 32);
        let rax = mcontext_u64(ctx, 16);
        let mut b = [0u8; 192];
        let mut p = 0;
        push(&mut b, &mut p, b"[mtld3d::unix] rcx(this)=");
        push_hex(&mut b, &mut p, rcx);
        push(&mut b, &mut p, b" rax(vtbl)=");
        push_hex(&mut b, &mut p, rax);
        push(&mut b, &mut p, b" rsp=");
        push_hex(&mut b, &mut p, rsp);
        push(&mut b, &mut p, b"\n");
        // SAFETY: write(2) on fd 2 is async-signal-safe.
        unsafe {
            let _ = libc::write(2, b.as_ptr().cast::<c_void>(), p);
        }
        if rsp != 0 {
            // SAFETY: `rsp` is the faulting stack pointer; the `CALL` that jumped
            // to garbage pushed its return address there. A bad `rsp` re-faults
            // into the re-entrancy guard rather than looping.
            let ret = unsafe { (rsp as *const u64).read() };
            let mut rb = [0u8; 192];
            let mut rp = 0;
            push(&mut rb, &mut rp, b"[mtld3d::unix] caller(ret@rsp)=");
            push_hex(&mut rb, &mut rp, ret);
            push(&mut rb, &mut rp, b"\n");
            // SAFETY: write(2) on fd 2 is async-signal-safe.
            unsafe {
                let _ = libc::write(2, rb.as_ptr().cast::<c_void>(), rp);
            }
            let mut frame = [ret as *mut c_void; 1];
            // SAFETY: single in-bounds frame pointer; `backtrace_symbols_fd` is
            // async-signal-safe (resolves via `dladdr`) and writes to fd 2.
            unsafe { backtrace_symbols_fd(frame.as_mut_ptr(), 1, 2) };
        }
    }

    crumb::dump_recent(256);

    // Native backtrace of the faulting thread. `backtrace` only walks frame
    // pointers (no allocation) and `backtrace_symbols_fd` resolves each via
    // `dladdr` straight to fd 2 — both async-signal-safe (unlike
    // `backtrace_symbols`, which mallocs). Symbolises our `.so`, Wine, and
    // system frames (Metal/CoreAnimation), turning a bare fault address into a
    // call chain.
    let mut frames = [ptr::null_mut::<c_void>(); 64];
    // SAFETY: `frames` is a valid 64-element buffer; `backtrace` writes at most
    // `len` entries and returns the count actually written.
    let n = unsafe { backtrace(frames.as_mut_ptr(), FRAME_CAP) };
    if n > 0 {
        const HDR: &[u8] = b"[mtld3d::unix] native backtrace:\n";
        // SAFETY: write(2) on fd 2 is async-signal-safe.
        unsafe {
            let _ = libc::write(2, HDR.as_ptr().cast::<c_void>(), HDR.len());
        }
        // SAFETY: `frames[..n]` were filled by `backtrace`; `backtrace_symbols_fd`
        // is async-signal-safe and writes the resolved frames to fd 2.
        unsafe { backtrace_symbols_fd(frames.as_ptr(), n, 2) };
    }

    // Last resort for a jump-to-NULL whose frame chain is broken: scan the raw
    // stack for words that `dladdr` resolves into *our* dylib and symbolise
    // them. This reconstructs the call chain the frame-pointer walk can't —
    // the return addresses pushed by the calls leading to the bad jump are
    // still on the stack even when `rbp` is garbage. Done last because an
    // unmapped read re-faults into the re-entrancy guard (`_exit`), which would
    // otherwise drop the crumb dump + native backtrace above.
    #[cfg(target_arch = "x86_64")]
    {
        let rsp = mcontext_u64(ctx, 72);
        if rsp != 0 {
            const HDR: &[u8] = b"[mtld3d::unix] mtld3d.so return addrs on stack:\n";
            // SAFETY: write(2) on fd 2 is async-signal-safe.
            unsafe {
                let _ = libc::write(2, HDR.as_ptr().cast::<c_void>(), HDR.len());
            }
            scan_stack_for_our_frames(rsp);
        }
    }

    // SAFETY: _exit(2) is async-signal-safe; skips atexit handlers and
    // libc cleanup.
    unsafe { libc::_exit(1) };
}

/// Scan the raw stack for return addresses into our own dylib and print them.
///
/// Walks up to 4096 words from `rsp` and `backtrace_symbols_fd`-prints each
/// value that `dladdr` resolves into a module whose path contains `mtld3d`.
/// Caps the printed count so a deep stack can't flood. Async-signal-safe: only
/// `dladdr` (no malloc on the resolve path here) + `backtrace_symbols_fd` + raw
/// stack reads.
#[cfg(target_arch = "x86_64")]
fn scan_stack_for_our_frames(rsp: u64) {
    let base = rsp as *const u64;
    let mut printed = 0u32;
    let mut i = 0usize;
    while i < 4096 && printed < 48 {
        // SAFETY: reads stack memory at increasing offsets from `rsp`; an
        // unmapped read re-faults into the re-entrancy guard (terminating),
        // which bounds the walk. `read_unaligned` tolerates a 4-byte-aligned
        // 32-bit stack.
        let v = unsafe { base.cast::<u8>().add(i * 4).cast::<u64>().read_unaligned() };
        if v >= 0x1000 && dladdr_is_ours(v) {
            let mut frame = [v as *mut c_void; 1];
            // SAFETY: single in-bounds frame pointer; `backtrace_symbols_fd`
            // resolves via `dladdr` and writes to fd 2.
            unsafe { backtrace_symbols_fd(frame.as_mut_ptr(), 1, 2) };
            printed += 1;
        }
        i += 1;
    }
    // Second pass: the 32-bit guest stack. `dladdr` can't see Wine's PE
    // builtins (not dyld images), so collect raw 4-byte words that land in the
    // PE-builtin zone [0x7A00_0000, 0x7C00_0000) (ntdll/user32/win32u/d3d9/…)
    // or the guest EXE image [0x0040_0000, 0x0080_0000) — covering all of the
    // guest client's `.text` (up to ~0x7ff000), not just the first page, so the
    // real guest call chain (its 0x4xxxxx–0x7xxxxx return addresses) is shown,
    // mapped to modules by their logged load bases.
    const GHDR: &[u8] = b"[mtld3d::unix] guest (PE) stack words:\n";
    // SAFETY: write(2) on fd 2 is async-signal-safe.
    unsafe {
        let _ = libc::write(2, GHDR.as_ptr().cast::<c_void>(), GHDR.len());
    }
    let mut gprinted = 0u32;
    let mut j = 0usize;
    while j < 4096 && gprinted < 64 {
        // SAFETY: as the loop above — reads near `rsp`; an unmapped read
        // re-faults into the re-entrancy guard, bounding the walk.
        let w = unsafe { base.cast::<u8>().add(j * 4).cast::<u32>().read_unaligned() };
        let in_builtin = (0x7A00_0000..0x7C00_0000).contains(&w);
        let in_exe = (0x0040_0000..0x0080_0000).contains(&w);
        if in_builtin || in_exe {
            let mut b = [0u8; 192];
            let mut p = 0;
            push(&mut b, &mut p, b"  g=");
            push_hex(&mut b, &mut p, u64::from(w));
            push(&mut b, &mut p, b"\n");
            // SAFETY: write(2) on fd 2 is async-signal-safe.
            unsafe {
                let _ = libc::write(2, b.as_ptr().cast::<c_void>(), p);
            }
            gprinted += 1;
        }
        j += 1;
    }
}

/// True when `addr` resolves (via `dladdr`) into our own `.so`.
///
/// The match is on a loaded image whose filename contains the bytes `mtld3d`.
/// Filters stack garbage and libsystem/Wine/Metal frames down to our own call
/// chain.
#[cfg(target_arch = "x86_64")]
fn dladdr_is_ours(addr: u64) -> bool {
    // SAFETY: zeroed `Dl_info` is a valid out-param for `dladdr`.
    let mut info: libc::Dl_info = unsafe { mem::zeroed() };
    // SAFETY: `dladdr` reads `addr` only as an opaque value and fills `info`.
    let ok = unsafe { libc::dladdr(addr as *const c_void, &raw mut info) };
    if ok == 0 || info.dli_fname.is_null() {
        return false;
    }
    // Scan the NUL-terminated path for the substring "mtld3d" without alloc.
    const NEEDLE: &[u8] = b"mtld3d";
    let p = info.dli_fname.cast::<u8>();
    let mut idx = 0usize;
    let mut matched = 0usize;
    while idx < 4096 {
        // SAFETY: `dli_fname` is a NUL-terminated C string owned by dyld.
        let c = unsafe { p.add(idx).read() };
        if c == 0 {
            break;
        }
        matched = if c == NEEDLE[matched] {
            matched + 1
        } else {
            usize::from(c == NEEDLE[0])
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

/// The faulting program counter from a signal `ucontext`, or 0 if it can't be read.
///
/// `uc_mcontext` is a pointer to an opaque `__darwin_mcontext64` (the `libc`
/// crate exposes it only as padding), sitting at byte offset 0x30 in
/// `ucontext_t` (same on both macOS arches). The PC offset *within* the
/// `mcontext` is arch-specific: `x86_64` `__rip` follows the 16-byte exception
/// state + 16 thread-state `u64`s (144); `arm64` `__pc` follows the 16-byte
/// exception state + 32 thread-state `u64`s (272). The shipped `.so` is
/// `x86_64` under Wine.
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
/// That `mcontext` is the `x86_64` register file. Offsets follow
/// `__darwin_mcontext64`: the 16-byte exception state then the thread-state
/// registers — `rax` at 16, `rcx` at 32, `rsp` at 72, `rip` at 144. Returns 0
/// if the context can't be read.
#[cfg(target_arch = "x86_64")]
fn mcontext_u64(ctx: *mut c_void, offset: usize) -> u64 {
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
        _ => b"SIG?",
    }
}

fn push(buf: &mut [u8; 192], pos: &mut usize, bytes: &[u8]) {
    let avail = buf.len() - *pos;
    let take = bytes.len().min(avail);
    buf[*pos..*pos + take].copy_from_slice(&bytes[..take]);
    *pos += take;
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
