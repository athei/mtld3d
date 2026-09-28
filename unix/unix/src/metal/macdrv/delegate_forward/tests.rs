//! Unit tests for the screen-parameter forward into the application delegate.
//!
//! The forward has to come back with the exception a delegate throws, its
//! name and reason in hand, instead of letting it unwind. The tests pin a
//! delegate method that raises an `NSException`, a delegate that returns
//! normally, and a receiver with no such method, whose unrecognized-selector
//! exception is raised by the runtime with no Rust frame on the way. They also
//! pin that a receiver without the method is told apart from one with it,
//! which is what decides whether the notification is taken over at all.
//!
//! The test harness always builds with unwinding, so these tests cannot show
//! that a Rust frame would have aborted; they show that the exception stops in
//! the Objective-C `@try`, which is the same under either panic strategy.

use objc2::{define_class, extern_methods, rc::Retained, runtime::NSObject};
use objc2_foundation::{NSException, NSNotification, NSString};

use super::*;

const EXCEPTION_NAME: &str = "Mtld3dTestException";
const EXCEPTION_REASON: &str = "the test delegate raised";

define_class!(
    /// A delegate whose screen-parameter handler raises an `NSException`.
    #[unsafe(super(NSObject))]
    #[name = "Mtld3dTestThrowingDelegate"]
    struct ThrowingDelegate;

    impl ThrowingDelegate {
        #[unsafe(method(applicationDidChangeScreenParameters:))]
        fn application_did_change_screen_parameters(&self, _notification: &NSNotification) {
            let name = NSString::from_str(EXCEPTION_NAME);
            let reason = NSString::from_str(EXCEPTION_REASON);
            // SAFETY: the name and reason are strings and the user info is absent.
            let exception =
                unsafe { NSException::exceptionWithName_reason_userInfo(&name, Some(&reason), None) };
            exception.raise();
        }
    }
);

define_class!(
    /// A delegate whose screen-parameter handler returns normally.
    #[unsafe(super(NSObject))]
    #[name = "Mtld3dTestReturningDelegate"]
    struct ReturningDelegate;

    impl ReturningDelegate {
        #[unsafe(method(applicationDidChangeScreenParameters:))]
        fn application_did_change_screen_parameters(&self, _notification: &NSNotification) {}
    }
);

impl ThrowingDelegate {
    extern_methods!(
        #[unsafe(method(new))]
        #[unsafe(method_family = new)]
        fn new() -> Retained<Self>;
    );
}

impl ReturningDelegate {
    extern_methods!(
        #[unsafe(method(new))]
        #[unsafe(method_family = new)]
        fn new() -> Retained<Self>;
    );
}

fn notification() -> Retained<NSNotification> {
    let name = NSString::from_str("Mtld3dTestScreenParameters");
    // SAFETY: the name is a string and the notification carries no object.
    unsafe { NSNotification::notificationWithName_object(&name, None) }
}

#[test]
fn an_exception_the_delegate_raises_comes_back_with_its_name_and_reason() {
    let delegate = ThrowingDelegate::new();
    // SAFETY: the test delegate is a plain `NSObject` subclass, usable from any thread.
    let outcome = unsafe { forward_screen_parameters(&delegate, &notification()) };
    assert_eq!(
        outcome,
        Err(format!("{EXCEPTION_NAME}: {EXCEPTION_REASON}"))
    );
}

#[test]
fn a_delegate_that_returns_forwards_cleanly() {
    let delegate = ReturningDelegate::new();
    // SAFETY: the test delegate is a plain `NSObject` subclass, usable from any thread.
    let outcome = unsafe { forward_screen_parameters(&delegate, &notification()) };
    assert_eq!(outcome, Ok(()));
}

#[test]
fn a_receiver_without_the_method_comes_back_as_an_unrecognized_selector() {
    let receiver = NSObject::new();
    // SAFETY: a plain `NSObject` is usable from any thread.
    let outcome = unsafe { forward_screen_parameters(&receiver, &notification()) };
    let described = outcome.expect_err("an unrecognized selector raises");
    assert!(
        described.starts_with("NSInvalidArgumentException: ")
            && described.contains("applicationDidChangeScreenParameters:"),
        "unexpected description: {described}",
    );
}

#[test]
fn a_delegate_with_the_method_is_one_to_take_the_notification_over_from() {
    assert!(delegate_handles_screen_parameters(&ReturningDelegate::new()));
    assert!(delegate_handles_screen_parameters(&ThrowingDelegate::new()));
}

#[test]
fn a_delegate_without_the_method_is_left_alone() {
    assert!(!delegate_handles_screen_parameters(&NSObject::new()));
}
