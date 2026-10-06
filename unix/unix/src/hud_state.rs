//! Naming the layer and its build in the Metal Performance HUD.
//!
//! macOS 27 lets a process report a labelled state per domain through
//! `SRStateReporter` (the `StateReporting` framework), and the Metal HUD shows
//! the domains named by `MTL_HUD_STATE_REPORTER_DOMAINS` as rows of their own
//! beside the frame metrics. Reporting the release identity under `MTLD3D`,
//! and naming that domain in the game layer's `developerHUDProperties`, makes
//! the HUD of a game say which build is drawing it.
//!
//! The framework is opened at runtime rather than linked: macOS 15 and 26 do
//! not carry it, and an SDK older than macOS 27 has nothing to link against.

use std::sync::Once;

use libloading::os::unix::{Library, RTLD_NOW};
use log::info;
use mtld3d_shared::identity;
use objc2::{
    extern_class, extern_methods,
    rc::Retained,
    runtime::{AnyClass, AnyObject, ProtocolObject},
};
use objc2_foundation::{
    NSDictionary, NSMutableCopying, NSMutableDictionary, NSObject, NSOperatingSystemVersion,
    NSProcessInfo, NSString,
};
use objc2_quartz_core::CAMetalLayer;

use crate::LOG_TARGET;

/// The framework's canonical path, resolved by dyld out of the shared cache.
const FRAMEWORK_PATH: &str = "/System/Library/Frameworks/StateReporting.framework/StateReporting";

/// The first macOS major version that carries `StateReporting`.
const FIRST_MACOS: isize = 27;

/// The domain the HUD names the row by.
///
/// The framework documents reverse-DNS domains, but the overlay prints the
/// domain in a narrow column and elides a long one, so the project's name is
/// used as is.
const DOMAIN: &str = "MTLD3D";

/// The HUD key, and the environment variable of the same name, that lists the domains shown.
///
/// The HUD reads it from a layer's `developerHUDProperties` as well as from
/// the environment, and a set value also turns its State Reporters metric on.
/// The layer's value replaces the whole list, so it carries the user's own.
const DOMAINS_KEY: &str = "MTL_HUD_STATE_REPORTER_DOMAINS";

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
        /// The process's reporter for `domain`.
        ///
        /// Declared optional although the framework declares it non-null: a nil
        /// answer would otherwise panic, and the unix crate aborts on panic.
        #[unsafe(method(reporterForDomain:))]
        #[unsafe(method_family = none)]
        fn for_domain(domain: &NSString) -> Option<Retained<Self>>;

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
        if !is_supported() {
            info!(target: LOG_TARGET, "hud state: the Metal HUD row requires macOS {FIRST_MACOS}");
            return;
        }
        // SAFETY: loading a library runs its initializers; this is an Apple
        // system framework, written in Swift, so they register its Objective-C
        // classes and load the Swift runtime libraries it links; it is opened
        // once per process by this latch.
        match unsafe { Library::open(Some(FRAMEWORK_PATH), RTLD_NOW) } {
            Ok(lib) => core::mem::forget(lib),
            Err(err) => {
                mtld3d_shared::log_once_warn!(
                    target: LOG_TARGET,
                    "hud state: {FRAMEWORK_PATH} did not load ({err}), no state is reported"
                );
                return;
            }
        }
        if AnyClass::get(c"SRStateReporter").is_none() {
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET,
                "hud state: {FRAMEWORK_PATH} has no SRStateReporter, no state is reported"
            );
            return;
        }
        let Some(reporter) = StateReporter::for_domain(&NSString::from_str(DOMAIN)) else {
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET,
                "hud state: no reporter for {DOMAIN}, no state is reported"
            );
            return;
        };
        reporter.report_transition(Some(&NSString::from_str(identity::BUILD)), None, None);
        info!(target: LOG_TARGET, "hud state: reported {} under {DOMAIN}", identity::BUILD);
    });
}

/// Name the domain in a game layer's HUD properties, so the row shows without the panel.
///
/// Runs on the main thread with the rest of the layer's configuration. Every
/// key the layer already carries is kept, and `mode`, which turns the HUD on
/// or off, is never written, so the row shows only where the HUD already does.
pub fn show_row(layer: &CAMetalLayer) {
    if !is_supported() {
        return;
    }
    let key = NSString::from_str(DOMAINS_KEY);
    let existing = layer.developerHUDProperties();
    let current = existing
        .as_deref()
        .and_then(|props| props.objectForKey(key.as_ref()))
        .and_then(|value| value.downcast::<NSString>().ok())
        .map(|value| value.to_string())
        .or_else(|| std::env::var(DOMAINS_KEY).ok());
    let Some(domains) = domains_with_ours(current.as_deref()) else {
        return;
    };
    let merged = existing.map_or_else(NSMutableDictionary::<AnyObject, AnyObject>::new, |props| {
        props.mutableCopy()
    });
    let value = NSString::from_str(&domains);
    // SAFETY: the key is an `NSString`, which conforms to `NSCopying`, and
    // the layer's HUD properties are a string-keyed dictionary.
    unsafe { merged.setObject_forKey(value.as_ref(), ProtocolObject::from_ref(&*key)) };
    // SAFETY: objc2 typed binding; the dictionary is copied by the layer.
    unsafe { layer.setDeveloperHUDProperties(Some(&merged)) };
    mtld3d_shared::log_once_info!(
        target: LOG_TARGET,
        "hud state: the game layer's HUD shows the {DOMAIN} row ({DOMAINS_KEY}={domains})"
    );
}

/// Whether this macOS carries `StateReporting`.
fn is_supported() -> bool {
    NSProcessInfo::processInfo().isOperatingSystemAtLeastVersion(NSOperatingSystemVersion {
        majorVersion: FIRST_MACOS,
        minorVersion: 0,
        patchVersion: 0,
    })
}

/// The domain list that adds ours to `current`, or `None` when `current` already shows it.
///
/// The HUD accepts a comma-separated list, optionally wrapped in double
/// quotes, and `ALL` or `*` for every domain.
fn domains_with_ours(current: Option<&str>) -> Option<String> {
    let list = current.map_or("", |value| value.trim().trim_matches('"').trim());
    if list.is_empty() {
        return Some(DOMAIN.to_owned());
    }
    if list == "ALL" || list == "*" || list.split(',').any(|domain| domain.trim() == DOMAIN) {
        return None;
    }
    Some(format!("{list},{DOMAIN}"))
}

#[cfg(test)]
mod tests;
