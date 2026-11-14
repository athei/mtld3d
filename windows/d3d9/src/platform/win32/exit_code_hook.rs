//! The status the process asked to exit with, kept for the detach that ends it.
//!
//! Wine hands the status to the wineserver before it runs the
//! `DLL_PROCESS_DETACH` callbacks, and this DLL's detach ends the process
//! there with `TerminateProcess`, whose own status is the one the unix side
//! of Wine exits with. Nothing readable from inside the detach carries the
//! original: the server answers `STATUS_PENDING` for a running thread's exit
//! code and `STILL_ACTIVE` for the live process, and the PE side keeps
//! nothing. So the exit entry points the main module imports are redirected
//! here, each recording the status it was handed before it forwards to the
//! real one, and the detach passes the first status recorded to
//! `TerminateProcess`, which is what a unix parent's `wait` then reads.
//!
//! Three entry points cover the ways a main module ends its own process:
//! `ExitProcess`, `RtlExitUserProcess` under it, and the C runtime's `exit`,
//! which the statically linked startup stub calls with what `main` returned.
//! A process that resolves one of them through `GetProcAddress`, or that
//! exits from a call inside another module, is not covered and still ends
//! with 0.

use core::{
    ffi::c_int,
    sync::atomic::{AtomicBool, AtomicU32, Ordering},
};

use log::info;

use super::{
    LOG_TARGET,
    import_patch::{Hook, PatchedImports},
};

/// How many exit entry points are redirected.
const HOOKS: usize = 3;
/// Slot of `ExitProcess`, `RtlExitUserProcess` and the C runtime's `exit`.
const EXIT_PROCESS: usize = 0;
const RTL_EXIT_USER_PROCESS: usize = 1;
const CRT_EXIT: usize = 2;

/// The status of the first exit call the main module made; 0 until then.
static STATUS: AtomicU32 = AtomicU32::new(0);
/// Whether [`STATUS`] has been written, so the first call is the one that counts.
static RECORDED: AtomicBool = AtomicBool::new(false);
/// The redirected exit imports of the main module.
static IMPORTS: PatchedImports<HOOKS> = PatchedImports::empty();

/// `ExitProcess` and `RtlExitUserProcess`, which share a signature.
type ExitProcessFn = extern "system" fn(u32) -> !;
/// The C runtime's `exit`, which is `cdecl` on i686.
type CrtExitFn = extern "C" fn(c_int) -> !;

/// Redirect the main module's exit entry points here. Idempotent.
///
/// Called from `DllMain` `PROCESS_ATTACH`, which runs before the game's
/// entry point when d3d9 is a static import, so every exit the process
/// makes on its own is seen.
pub fn install() {
    let patched = IMPORTS.install(&[
        Hook {
            dll: None,
            func: b"ExitProcess",
            replacement: exit_process as *const (),
        },
        Hook {
            dll: None,
            func: b"RtlExitUserProcess",
            replacement: rtl_exit_user_process as *const (),
        },
        Hook {
            dll: None,
            func: b"exit",
            replacement: crt_exit as *const (),
        },
    ]);
    if patched != 0 {
        info!(
            target: LOG_TARGET,
            "exit status: redirected {patched} exit import(s) of the main module"
        );
    }
}

/// Put the original exit entry points back. Idempotent.
///
/// Called from `DllMain` `PROCESS_DETACH` on the path the process survives,
/// so no slot keeps pointing into an image that is about to unmap.
pub fn uninstall() {
    IMPORTS.uninstall();
}

/// The status the process asked to exit with; 0 until an exit call names one.
pub fn status() -> u32 {
    STATUS.load(Ordering::Acquire)
}

/// Record the status of an exit call. The first one wins.
///
/// `exit` forwards to `ExitProcess` and that to `RtlExitUserProcess`, each
/// with the same status, but a C runtime runs its `atexit` handlers in
/// between and one of those may exit with a status of its own. The status
/// the process named first is the one it meant.
fn record(status: u32) {
    if !RECORDED.swap(true, Ordering::AcqRel) {
        STATUS.store(status, Ordering::Release);
    }
}

extern "system" fn exit_process(exit_code: u32) -> ! {
    record(exit_code);
    IMPORTS.original::<ExitProcessFn>(EXIT_PROCESS)(exit_code)
}

extern "system" fn rtl_exit_user_process(exit_code: u32) -> ! {
    record(exit_code);
    IMPORTS.original::<ExitProcessFn>(RTL_EXIT_USER_PROCESS)(exit_code)
}

extern "C" fn crt_exit(exit_code: c_int) -> ! {
    record(exit_code.cast_unsigned());
    IMPORTS.original::<CrtExitFn>(CRT_EXIT)(exit_code)
}
