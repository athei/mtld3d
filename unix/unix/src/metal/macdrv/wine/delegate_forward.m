// Screen-parameter forward into the application delegate, with its exceptions caught.
//
// The send and the @try that guards it have to be in one Objective-C frame:
// mtld3d.so is built with panic = "abort", so an Objective-C exception that
// unwinds into any Rust frame ends the process there, and a Rust-side catch
// would put Rust frames between the @try and the send.

#import <AppKit/AppKit.h>

// Send applicationDidChangeScreenParameters: to the delegate.
//
// Returns nil when the send returns, and otherwise the object the delegate
// threw, retained once for the caller to release.
id mtld3d_forward_screen_parameters(id<NSApplicationDelegate> delegate,
                                    NSNotification *notification)
{
    @try {
        [delegate applicationDidChangeScreenParameters:notification];
    } @catch (id exception) {
        return [exception retain];
    }
    return nil;
}

// Whether the delegate implements applicationDidChangeScreenParameters:.
//
// AppKit registers a delegate for the notification only when it does, so a
// delegate without the method has nothing to take over or forward to.
BOOL mtld3d_delegate_handles_screen_parameters(id<NSApplicationDelegate> delegate)
{
    return [delegate respondsToSelector:@selector(applicationDidChangeScreenParameters:)];
}
