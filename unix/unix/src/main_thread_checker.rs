//! Loading Apple's Main Thread Checker into the process on request.
//!
//! The layer runs its API, encoder and submit threads off the `AppKit` main
//! thread, and `AppKit`'s view, window and screen objects may only be touched
//! there. A call that breaks that rule does not fail where it is made: it
//! corrupts state `AppKit` owns on the main thread and the process dies later,
//! typically inside an autorelease pool pop in Wine's own code, with no frame
//! of the layer on the stack. A checked `MainThreadMarker` catches the class
//! methods that take one, but not an instance method on a view or window the
//! code already holds, and nothing the Wine driver does.
//!
//! Apple's Main Thread Checker does catch those. It is a dynamic library that
//! swizzles every `AppKit` method that requires the main thread and reports a
//! call from any other thread on stderr, as `Main Thread Checker: UI API
//! called on a background thread: -[NSView window]` followed by the thread's
//! name; with `MTC_CRASH_ON_REPORT=1` in the environment it ends the process
//! at the report instead, on the offending thread, so a crash log names the
//! caller. The library lives in the dyld shared cache, not on disk, so it is
//! reached by `dlopen` of its canonical path rather than by a file check.
//!
//! It is loaded here, from the first thunk that carries the resolved
//! configuration, rather than inserted at launch: the checker installs its
//! swizzles for the frameworks present when it initializes, and under Wine a
//! copy inserted through `DYLD_INSERT_LIBRARIES` reports nothing, while one
//! opened after `AppKit` is in the process does. `mtld3d.so` links `AppKit`, so
//! by the time any of its thunks runs the framework is loaded, and touching
//! an `AppKit` class first makes that certain.

use libloading::os::unix::{Library, RTLD_NOW};
use log::info;
use objc2::ClassType;

use crate::LOG_TARGET;

/// The checker's canonical path, resolved by dyld out of the shared cache.
const CHECKER_PATH: &str = "/usr/lib/libMainThreadChecker.dylib";

/// Load the Main Thread Checker and leave it resident for the process.
///
/// Failure is logged once and otherwise ignored: a machine without the
/// library (or a macOS that moved it) runs unchecked rather than not at all.
/// The handle is forgotten on purpose, since the swizzles the checker
/// installed have to outlive any scope that could drop it.
pub fn load() {
    // Touching a class through the runtime forces AppKit to be initialized
    // ahead of the checker, which swizzles what is loaded when it starts.
    let view_class = objc2_app_kit::NSView::class();
    // SAFETY: loading a library runs its initializers; this one is Apple's
    // own diagnostic library, whose initializer only swizzles AppKit and
    // reads its `MTC_*` environment variables, and it is loaded once per
    // process by the once-per-process `OpenLog` thunk.
    match unsafe { Library::open(Some(CHECKER_PATH), RTLD_NOW) } {
        Ok(lib) => {
            core::mem::forget(lib);
            info!(
                target: LOG_TARGET,
                "main thread checker: loaded {CHECKER_PATH} ({} present)", view_class.name().to_string_lossy()
            );
        }
        Err(err) => {
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET,
                "main thread checker: {CHECKER_PATH} did not load ({err}), AppKit use is unchecked"
            );
        }
    }
}
