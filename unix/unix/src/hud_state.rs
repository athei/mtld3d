//! Naming the layer and its build in the Metal Performance HUD.
//!
//! macOS 27 lets a process report a labelled state per domain through
//! `SRStateReporter` (the `StateReporting` framework), and the Metal HUD lists
//! the process's reporting domains under State Reporters in its configuration
//! panel; once one is ticked there, the overlay shows the domain and its state
//! label as a row beside the frame metrics. Reporting the release identity
//! under `MTLD3D` makes the HUD of a game say which build is drawing it.
//!
//! The framework is opened at runtime rather than linked: macOS 15 and 26 do
//! not carry it, and an SDK older than macOS 27 has nothing to link against.

use std::sync::Once;

use libloading::os::unix::{Library, RTLD_NOW};
use log::info;
use mtld3d_shared::identity;
use objc2::{extern_class, extern_methods, rc::Retained, runtime::AnyClass};
use objc2_foundation::{NSDictionary, NSObject, NSString};

use crate::LOG_TARGET;

/// The framework's canonical path, resolved by dyld out of the shared cache.
const FRAMEWORK_PATH: &str = "/System/Library/Frameworks/StateReporting.framework/StateReporting";

/// The domain the HUD names the row by.
///
/// The framework documents reverse-DNS domains, but the overlay prints the
/// domain in a narrow column and elides a long one, so the project's name is
/// used as is.
const DOMAIN: &str = "MTLD3D";

extern_class!(
    /// The reporter `StateReporting` hands out, one per domain.
    ///
    /// Declared here because no binding crate carries the framework yet. The
    /// class is looked up by name before use, so a macOS without it never
    /// reaches these methods.
    #[unsafe(super(NSObject))]
    #[name = "SRStateReporter"]
    struct StateReporter;
);

impl StateReporter {
    extern_methods!(
        #[unsafe(method(reporterForDomain:))]
        #[unsafe(method_family = none)]
        fn for_domain(domain: &NSString) -> Retained<Self>;

        /// Report a transition, a no-op when the label and stable metadata are unchanged.
        ///
        /// The metadata values are declared as `NSString` so this safe method cannot
        /// be handed one the framework raises on; it also takes `NSNumber` and
        /// `NSDate`.
        #[unsafe(method(reportTransitionToStateLabel:stableMetadata:volatileMetadata:))]
        #[unsafe(method_family = none)]
        fn report_transition(
            &self,
            label: Option<&NSString>,
            stable_metadata: Option<&NSDictionary<NSString, NSString>>,
            volatile_metadata: Option<&NSDictionary<NSString, NSString>>,
        );
    );
}

/// Report the layer and its build to the HUD once per process, where the system offers it.
///
/// Called for every attached layer, so a process that never attaches one
/// reports nothing. The label never changes, so the first call reports and
/// the rest return on the latch.
pub fn report_attached() {
    /// Whether this process has made its report, or found that it cannot.
    ///
    /// The resource is process-wide: the framework hands out one reporter per
    /// domain for the process, and it is opened on first use and left resident.
    static REPORTED: Once = Once::new();
    REPORTED.call_once(|| {
        // SAFETY: loading a library runs its initializers; this is an Apple
        // system framework whose initializer only registers its Objective-C
        // classes, and it is opened once per process by this latch.
        match unsafe { Library::open(Some(FRAMEWORK_PATH), RTLD_NOW) } {
            Ok(lib) => core::mem::forget(lib),
            Err(err) => {
                info!(
                    target: LOG_TARGET,
                    "hud state: {FRAMEWORK_PATH} did not load ({err}), no state is reported"
                );
                return;
            }
        }
        if AnyClass::get(c"SRStateReporter").is_none() {
            info!(
                target: LOG_TARGET,
                "hud state: {FRAMEWORK_PATH} has no SRStateReporter, no state is reported"
            );
            return;
        }
        let reporter = StateReporter::for_domain(&NSString::from_str(DOMAIN));
        reporter.report_transition(Some(&NSString::from_str(identity::BUILD)), None, None);
        info!(target: LOG_TARGET, "hud state: reported {} under {DOMAIN}", identity::BUILD);
    });
}
