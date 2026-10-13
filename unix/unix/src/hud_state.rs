//! Naming the layer, its build and its live figures in the Metal Performance HUD.
//!
//! macOS 27 lets a process report a labelled state per domain through
//! `SRStateReporter` (the `StateReporting` framework), and the Metal HUD shows
//! the domains named by `MTL_HUD_STATE_REPORTER_DOMAINS` under its "State
//! Reporters" caption: the domain and its label on one row, then one indented
//! row per metadata key. Reporting the release identity as the label of the
//! `mtld3d` domain, and naming that domain in the game layer's
//! `developerHUDProperties`, makes the HUD of a game say which build is
//! drawing it; the encoder adds its live figures as volatile metadata
//! ([`HudReporter`], with the rows themselves in `mtld3d_core::hud_stats`).
//!
//! The framework is opened at runtime rather than linked: macOS 15 and 26 do
//! not carry it, and an SDK older than macOS 27 has nothing to link against.

use std::sync::{
    Once,
    atomic::{AtomicU64, Ordering},
};

use libloading::os::unix::{Library, RTLD_NOW};
use log::info;
use mtld3d_core::hud_stats::{
    HUD_DOMAIN, HUD_ROW_COUNT, HUD_ROW_KEYS, HUD_SAMPLE_HZ, HudRowLatch, SampleGate, hud_label,
};
use mtld3d_shared::identity;
use objc2::{
    extern_class, extern_methods,
    rc::Retained,
    runtime::{AnyClass, AnyObject, NSObjectProtocol, ProtocolObject, Sel},
};
use objc2_foundation::{
    NSDictionary, NSMutableCopying, NSMutableDictionary, NSNumber, NSObject,
    NSOperatingSystemVersion, NSProcessInfo, NSString,
};
use objc2_quartz_core::CAMetalLayer;

use crate::LOG_TARGET;

/// The framework's canonical path, resolved by dyld out of the shared cache.
const FRAMEWORK_PATH: &str = "/System/Library/Frameworks/StateReporting.framework/StateReporting";

/// The first macOS major version that carries `StateReporting`.
const FIRST_MACOS: isize = 27;

/// The domain the HUD names the rows by, and prints as their header.
///
/// The framework documents reverse-DNS domains, but the overlay prints the
/// domain in a narrow column and elides a long one, so the project's name is
/// used as is, in lower case so the header row reads as the project's name.
const DOMAIN: &str = HUD_DOMAIN;

/// The HUD key, and the environment variable of the same name, that lists the domains shown.
///
/// The HUD reads it from a layer's `developerHUDProperties` as well as from
/// the environment, and a set value also turns its State Reporters metric on.
/// The layer's value replaces the whole list, so it carries the user's own.
const DOMAINS_KEY: &str = "MTL_HUD_STATE_REPORTER_DOMAINS";

/// The encoder whose figures the HUD's `mtld3d` rows show, by its owner id, or 0 for none.
///
/// The resource is process-wide: `StateReporting` hands out one reporter per
/// domain for the process, so the HUD shows one set of `mtld3d` rows however
/// many devices are live. An encoder takes the rows at its first sample,
/// which follows its first encoded frame, so the newest device that renders
/// holds them; an encoder reports only while it holds them, and the holder's
/// teardown clears the rows and leaves them to the next surviving encoder that
/// samples. A holder that stops sampling, a device that submitted once and
/// went idle, loses the rows to an encoder that samples once
/// [`OWNER_SAMPLED`] is [`STALE_PERIODS`] sample periods old. Touched at an
/// encoder's teardown and at most [`HUD_SAMPLE_HZ`] times a second per encoder.
static OWNER: AtomicU64 = AtomicU64::new(0);

/// The TSC reading of the rows' holder's last due sample.
///
/// Process-wide with [`OWNER`], for the same reason: one set of rows per
/// process, so one record of when its holder last sampled. Written by the
/// holder at each due sample and by an encoder that takes the rows, read by
/// the others at theirs. The two words are not updated together, and need not
/// be: a takeover that races the holder's own sample can leave both encoders
/// reporting once, after which the loser sees it no longer holds the rows and
/// the winner's fresh reading keeps it from being taken back, so the rows
/// settle within one sample period without a lock.
static OWNER_SAMPLED: AtomicU64 = AtomicU64::new(0);

/// Sample periods without a sample from the holder after which another encoder may take the rows.
const STALE_PERIODS: u64 = 2;

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

        /// Replace the domain's volatile metadata; `None` removes every row it added.
        ///
        /// Not every macOS 27 build need answer it, so a caller checks
        /// `respondsToSelector:` first. The values are numbers because the HUD
        /// draws no string value.
        #[unsafe(method(reportVolatileMetadataUpdate:))]
        #[unsafe(method_family = none)]
        fn report_volatile_update(&self, metadata: Option<&NSDictionary<NSString, NSNumber>>);
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
            info!(target: LOG_TARGET, "hud state: the Metal HUD rows require macOS {FIRST_MACOS}");
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
        let label = hud_label(identity::BUILD);
        reporter.report_transition(Some(&NSString::from_str(&label)), None, None);
        info!(target: LOG_TARGET, "hud state: reported {label} under {DOMAIN}");
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

/// One encoder's live rows in the HUD: when to sample, and the reporter to send them to.
///
/// Lives on the `FrameEncoder` and runs on its thread only. A sample is due at
/// most [`HUD_SAMPLE_HZ`] times a second, and only once the HUD is loaded:
/// `libMTLHud` carries the HUD's reporter manager, and neither Metal nor
/// Core Animation links it, so its class appearing is taken as the HUD
/// appearing.
pub struct HudReporter {
    /// The id this encoder holds the domain's rows under in [`OWNER`]; never 0.
    owner_id: u64,
    gate: SampleGate,
    /// The rows this device has shown, kept across a `Reset` and while another device holds them.
    latch: HudRowLatch,
    state: ReporterState,
    /// Whether this encoder has had a sample due, and with it taken the rows once.
    claimed: bool,
}

/// How far an encoder's reporter has come.
enum ReporterState {
    /// This macOS carries no `StateReporting`: nothing is ever reported.
    Unsupported,
    /// Waiting for the framework (opened by the first attached layer) and the HUD to load.
    Waiting,
    /// The framework gave no reporter, or one without the volatile update: logged once.
    Failed,
    /// Reporting, with the reporter and the row keys built once.
    Live(Box<LiveReporter>),
}

/// The retained reporter and the row keys, each built once per encoder.
struct LiveReporter {
    reporter: Retained<StateReporter>,
    keys: [Retained<NSString>; HUD_ROW_COUNT],
}

impl HudReporter {
    /// The reporter of the encoder `owner_id` names, which takes the rows at its first sample.
    ///
    /// `owner_id` must be nonzero and unique among live encoders; the address
    /// of an allocation the encoder owns for its life serves. A device that
    /// never encodes a frame never samples, so it never takes the rows from
    /// one that renders.
    pub fn new(owner_id: u64) -> Self {
        Self {
            owner_id,
            gate: SampleGate::default(),
            latch: HudRowLatch::default(),
            state: if is_supported() {
                ReporterState::Waiting
            } else {
                ReporterState::Unsupported
            },
            claimed: false,
        }
    }

    /// Whether the encoder can stop here: nothing it gathers would ever be reported.
    #[must_use]
    pub const fn is_inert(&self) -> bool {
        matches!(
            self.state,
            ReporterState::Unsupported | ReporterState::Failed
        )
    }

    /// Whether a sample is due at TSC reading `now` on a clock of `hz`.
    ///
    /// Due at most [`HUD_SAMPLE_HZ`] times a second, once the reporter is
    /// live, and only while this encoder holds the domain's rows
    /// ([`Self::hold_rows`]).
    pub fn sample_due(&mut self, now: u64, hz: u64) -> bool {
        let period = hz / HUD_SAMPLE_HZ;
        if self.is_inert() || !self.gate.due(now, period) {
            return false;
        }
        let holds = self.hold_rows(now, period);
        if matches!(self.state, ReporterState::Waiting) {
            self.state = go_live();
        }
        holds && matches!(self.state, ReporterState::Live(_))
    }

    /// Whether this encoder holds the rows at a due sample at `now`, taking them if it may.
    ///
    /// It takes them at its first due sample, whoever held them, and
    /// afterwards when no encoder holds them or when the holder has not had a
    /// due sample for [`STALE_PERIODS`] periods of `period` cycles. The holder
    /// records each of its samples in [`OWNER_SAMPLED`].
    fn hold_rows(&mut self, now: u64, period: u64) -> bool {
        if !self.claimed {
            self.claimed = true;
            OWNER.store(self.owner_id, Ordering::Relaxed);
            OWNER_SAMPLED.store(now, Ordering::Relaxed);
            return true;
        }
        let owner = OWNER.load(Ordering::Relaxed);
        // Saturating: a holder's reading taken on another core just after
        // `now` counts as fresh, never as a wrapped-around age.
        let free = owner == 0
            || (owner != self.owner_id
                && now.saturating_sub(OWNER_SAMPLED.load(Ordering::Relaxed))
                    >= period.saturating_mul(STALE_PERIODS));
        let holds = owner == self.owner_id
            || (free
                && OWNER
                    .compare_exchange(owner, self.owner_id, Ordering::Relaxed, Ordering::Relaxed)
                    .is_ok());
        if holds {
            OWNER_SAMPLED.store(now, Ordering::Relaxed);
        }
        holds
    }

    /// Replace the domain's rows with `rows`, indexed like `HUD_ROW_KEYS`; `None` leaves one out.
    ///
    /// The rows pass through this device's [`HudRowLatch`] first, so a row
    /// shows only once it has carried information on this device.
    pub fn report(&mut self, rows: &[Option<u32>; HUD_ROW_COUNT]) {
        let ReporterState::Live(live) = &self.state else {
            return;
        };
        let rows = self.latch.visible(*rows);
        let mut keys: Vec<&NSString> = Vec::with_capacity(HUD_ROW_COUNT);
        let mut values = Vec::with_capacity(HUD_ROW_COUNT);
        for (key, value) in live.keys.iter().zip(rows) {
            if let Some(value) = value {
                keys.push(key);
                values.push(NSNumber::new_u32(value));
            }
        }
        let metadata = NSDictionary::from_retained_objects(&keys, &values);
        live.reporter.report_volatile_update(Some(&metadata));
    }
}

impl Drop for HudReporter {
    /// Clear the rows when this encoder holds them, and leave them to the next encoder.
    ///
    /// The clear goes out while this encoder still holds the rows, so a
    /// surviving encoder that takes them afterwards reports over the clear
    /// rather than having its rows wiped by it.
    fn drop(&mut self) {
        if OWNER.load(Ordering::Relaxed) != self.owner_id {
            return;
        }
        if let ReporterState::Live(live) = &self.state {
            live.reporter.report_volatile_update(None);
        }
        let _ = OWNER.compare_exchange(self.owner_id, 0, Ordering::Relaxed, Ordering::Relaxed);
    }
}

/// The reporter, once the framework and the HUD are loaded; `Waiting` until then.
fn go_live() -> ReporterState {
    if AnyClass::get(c"SRStateReporter").is_none()
        || AnyClass::get(c"HUDStateReporterManager").is_none()
    {
        return ReporterState::Waiting;
    }
    let Some(reporter) = StateReporter::for_domain(&NSString::from_str(DOMAIN)) else {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "hud state: no reporter for {DOMAIN}, the HUD shows no live rows"
        );
        return ReporterState::Failed;
    };
    if !reporter.respondsToSelector(Sel::register(c"reportVolatileMetadataUpdate:")) {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "hud state: SRStateReporter has no reportVolatileMetadataUpdate:, the HUD shows no \
             live rows"
        );
        return ReporterState::Failed;
    }
    info!(target: LOG_TARGET, "hud state: the HUD is loaded, reporting live rows under {DOMAIN}");
    ReporterState::Live(Box::new(LiveReporter {
        reporter,
        keys: core::array::from_fn(|row| NSString::from_str(HUD_ROW_KEYS[row])),
    }))
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
/// quotes, and `ALL` or `*` for every domain. Domains compare case-sensitively,
/// as the HUD compares them.
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
