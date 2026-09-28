//! Forward a notification into Wine's application delegate with its Objective-C exceptions caught.
//!
//! This library is built with `panic = "abort"`, under which every Rust frame
//! aborts the process when an unwind reaches it, a foreign Objective-C
//! exception included. `objc2::exception::catch` cannot help there, because
//! the closure it runs is a Rust frame between its `@try` and the send. So the
//! send and its `@try` live together in `delegate_forward.m`, and what the
//! delegate throws is handed back here as an object instead of unwinding.
//! The same file answers whether a delegate implements the method at all,
//! which a driver's delegate need not.

use objc2::{
    rc::Retained,
    runtime::{AnyObject, Bool},
};
use objc2_foundation::{NSException, NSNotification};

unsafe extern "C" {
    /// Send `applicationDidChangeScreenParameters:` to `delegate` inside `@try`.
    ///
    /// Returns null when the send returns, and otherwise the object the
    /// delegate threw, retained once. Defined in `delegate_forward.m`.
    fn mtld3d_forward_screen_parameters(
        delegate: *const AnyObject,
        notification: *const NSNotification,
    ) -> *mut AnyObject;

    /// Whether `delegate` responds to `applicationDidChangeScreenParameters:`.
    ///
    /// Defined in `delegate_forward.m`, where the selector is checked
    /// against the `NSApplicationDelegate` declaration at compile time.
    fn mtld3d_delegate_handles_screen_parameters(delegate: *const AnyObject) -> Bool;
}

/// Whether `delegate` implements `applicationDidChangeScreenParameters:`.
///
/// `AppKit` subscribes a delegate to the notification only when it does, so
/// this is also whether the delegate is an observer to take the notification
/// over from.
pub fn delegate_handles_screen_parameters(delegate: &AnyObject) -> bool {
    // SAFETY: the pointer comes from a live reference, and
    // `respondsToSelector:` is a runtime query on the object's class that
    // any thread may send.
    unsafe { mtld3d_delegate_handles_screen_parameters(delegate) }.as_bool()
}

/// Send `applicationDidChangeScreenParameters:` to `delegate`, catching what it throws.
///
/// # Errors
///
/// Returns the caught exception's name and reason when the delegate, or
/// anything it called, threw an Objective-C exception. The send's own effects
/// up to the throw stay as they are, as they would after `AppKit` caught the
/// exception in its run loop.
///
/// # Safety
///
/// `delegate` must be messaged only on the thread it belongs to (the main
/// thread for Wine's application delegate), and `notification` must be the
/// notification being delivered.
pub unsafe fn forward_screen_parameters(
    delegate: &AnyObject,
    notification: &NSNotification,
) -> Result<(), String> {
    // SAFETY: both pointers come from live references; the caller upholds the
    // thread the delegate is messaged on; the function catches every
    // Objective-C exception, so nothing unwinds out of it.
    let thrown = unsafe { mtld3d_forward_screen_parameters(delegate, notification) };
    // SAFETY: a non-null return is an object retained once for us.
    let Some(thrown) = (unsafe { Retained::from_raw(thrown) }) else {
        return Ok(());
    };
    Err(describe_exception(thrown))
}

/// Name and reason of a caught exception, or the object's description when it is no `NSException`.
fn describe_exception(thrown: Retained<AnyObject>) -> String {
    match thrown.downcast::<NSException>() {
        Ok(exception) => {
            let reason = exception.reason();
            format!(
                "{}: {}",
                exception.name(),
                reason
                    .as_deref()
                    .map_or_else(|| "(no reason)".to_owned(), ToString::to_string),
            )
        }
        Err(other) => format!("a non-NSException object {other:?}"),
    }
}

#[cfg(test)]
mod tests;
