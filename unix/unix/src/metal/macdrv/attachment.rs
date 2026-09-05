//! One attachment record per device: the display state a metal view owns.
//!
//! `AttachMetalLayer` creates a metal view and its `CAMetalLayer` for one
//! device's window and registers an [`Attachment`] for them here, keyed by
//! the view's address, which is the handle every later thunk of that device
//! already carries (`SubmitFrame`, `DestroyCommandQueue`, `SetCursorOverlay`).
//! Everything the display decides for that window lives on the record:
//! whether the layer carries the HDR configuration, the live EDR headroom,
//! the present throttle, the window's occlusion, the backing scale published
//! to the PE side and the present-geometry streak. Several devices attached
//! at once each own their record, and one device's teardown touches only its
//! own.
//!
//! Liveness. A record is live exactly while it is in [`ATTACHMENTS`]. The
//! `view` and `layer` addresses and the two PE-side sink addresses a record
//! holds are valid only while it is live: `DestroyCommandQueue` unregisters
//! the record before it releases the view, and the PE side drops the box
//! behind the sinks after that thunk returns. So every dereference of one of
//! those addresses happens inside a helper in this file that holds the
//! registry lock and checks the record is still the one the map holds for
//! its view (`Arc::ptr_eq`, not key equality, so a view address the
//! allocator hands out again names a new record rather than resurrecting an
//! old one). Outside those helpers a record is plain data: atomics and
//! immutable words a main-thread observer, the submit thread and the API
//! thread may read at any time, whether or not the record is still live.
//!
//! The lock is held for a lookup plus one retain or one atomic store, never
//! across `AppKit` work, and nothing in this file takes any other lock, so
//! the registry lock sits at the bottom of every order. It is not taken
//! anywhere else.

use core::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

use mtld3d_shared::mtl::ColorSpacePolicy;
use objc2::rc::Retained;
use rustc_hash::FxHashMap;

use crate::metal::command::{GeometryStreak, PresentGeometry};

/// How many presents may pass between headroom refreshes.
///
/// Refreshing every present would put a main-queue block in every frame for
/// a value that tracks display brightness, so it is sampled every this many
/// presents instead. A present counter rather than a clock because the call
/// site is per-present and this needs no new time source; the cost is that
/// the interval is measured in frames, which at 30 to 300 fps puts the
/// refresh somewhere between one second and a tenth of one.
pub const HEADROOM_REFRESH_PRESENTS: u64 = 32;

/// The live records, keyed by the raw `NSView*` address attach created.
///
/// `LazyLock` because the map's constructor is not `const`.
static ATTACHMENTS: LazyLock<Mutex<FxHashMap<usize, Arc<Attachment>>>> =
    LazyLock::new(|| Mutex::new(FxHashMap::default()));

bitflags::bitflags! {
    /// The two switches attach resolves once for the record's lifetime.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct AttachFlags: u8 {
        /// `color.hdr.enable` from `mtld3d.conf`, as the attach carried it.
        ///
        /// Deciding the layer configuration for a screen first seen
        /// mid-session needs the user's gate, and that gate only ever
        /// arrives on the attach wire.
        const HDR_ENABLE_REQUESTED = 1 << 0;
        /// Attach configured the layer for HDR.
        ///
        /// Seeds the record's live `hdr_active` state; the display-follow
        /// path moves that state afterwards, never this bit.
        const HDR_ACTIVE = 1 << 1;
    }
}

/// What attach latched for a record, handed to [`register`].
pub struct AttachLatches {
    pub flags: AttachFlags,
    /// `color.space` from `mtld3d.conf`.
    ///
    /// Reconfiguring the layer for a new screen has to pick the colorspace
    /// family attach would have picked for it.
    pub color_space: ColorSpacePolicy,
    /// The guest's vsync ask and the user's frame cap, as `pack_pacing` folds them.
    pub pacing_bits: u64,
    /// The Wine layer's contents scale as attach read it, already published.
    pub backing_scale: u32,
    /// Address of the PE-side `AtomicU32` a changed backing scale is published into, `0` = none.
    pub backing_scale_sink: usize,
    /// Address of the PE-side `AtomicU32` a cursor re-apply is asked through, `0` = none.
    pub cursor_kick_sink: usize,
}

/// The display state of one attached metal view.
///
/// Immutable words identify the record and carry what attach latched; the
/// atomics are what the main thread derives from the display and the
/// presenting thread reads per present. See the module doc for which of
/// them may be dereferenced, and where.
pub struct Attachment {
    /// Raw `NSView*`, the registry key.
    view: usize,
    /// Raw `CAMetalLayer*` of that view.
    layer: usize,
    /// See [`AttachLatches::backing_scale_sink`].
    backing_scale_sink: usize,
    /// See [`AttachLatches::cursor_kick_sink`].
    cursor_kick_sink: usize,
    flags: AttachFlags,
    color_space: ColorSpacePolicy,
    /// Raw `NSWindow*` the occlusion observer filters notifications by.
    ///
    /// Written on the main thread, compared against the notification's own
    /// live object, never dereferenced. `0` while the view has no window.
    window: AtomicUsize,
    /// Whether the layer currently carries the EDR configuration.
    ///
    /// Written at attach and again on the main thread whenever the window's
    /// screen changes its EDR capability, so it always names the
    /// configuration the layer actually has. The present route is taken
    /// from the drawable's own pixel format rather than from here, so a
    /// reconfiguration landing between two presents cannot mismatch the pass.
    hdr_active: AtomicBool,
    /// Whether the window is fully occluded (covered or minimised).
    ///
    /// Seeded at attach and updated by the occlusion observer, both on the
    /// main thread; read by `submit_frame` per present to skip the
    /// `nextDrawable` acquire while nothing reaches the screen. Relaxed is
    /// enough: a one-frame lag at the transition is harmless and bounded by
    /// the retained `allowsNextDrawableTimeout` safety valve.
    window_occluded: AtomicBool,
    /// Live EDR headroom as `f32::to_bits`, published by the main thread.
    ///
    /// `submit_frame` needs this every present to drive the HDR tone-map
    /// shader, but deriving it means walking `NSView.window -> NSWindow.screen`,
    /// and those two are main-thread-only however read-only the `NSScreen`
    /// property at the end of the walk is. Seeded to `1.0`, which the HDR
    /// shader treats as the identity curve, so the presents before the first
    /// refresh lands are correct rather than merely safe.
    current_headroom_bits: AtomicU32,
    /// Last headroom an `info!` line was emitted for, as `f32::to_bits`.
    ///
    /// `0` = never logged; the first refresh always logs to establish a
    /// baseline distinct from the attach-time line. Written only on the main
    /// thread.
    last_logged_headroom_bits: AtomicU32,
    /// Whether a headroom refresh is already queued on the main thread.
    ///
    /// Bounds the main queue to one outstanding refresh per record no matter
    /// how far ahead the presenting thread runs.
    headroom_refresh_pending: AtomicBool,
    /// Presents since the last headroom refresh was queued.
    ///
    /// Starts already due so the first present after attach queues a refresh
    /// rather than waiting out a full interval on the seeded `1.0`.
    presents_since_headroom_refresh: AtomicU64,
    /// Minimum seconds between presents, as `f64::to_bits`; `0.0` = no throttle.
    ///
    /// The longer of the vsync-equivalent cap (`1 / panel_max_hz`) and the
    /// user's `present.maxFps` cap. Set at attach from the panel under the
    /// window; the D3D9 Reset path re-queries the panel and overwrites, and
    /// the display-follow path re-derives it for another panel.
    min_present_duration_bits: AtomicU64,
    /// Present pacing latched from the PE side, encoded by `pack_pacing`.
    ///
    /// Attach and the D3D9 Reset path both write it, and the display-follow
    /// reconciliation reads it back so a re-derivation on another panel
    /// still honours the guest's vsync request and the user's ceiling. One
    /// packed word rather than two atomics, so a Reset landing between two
    /// reads cannot hand the reconciliation half of each.
    present_pacing_bits: AtomicU64,
    /// Backing scale last published to the PE side.
    ///
    /// Compared against before publishing, so the PE side only hears about
    /// real changes.
    current_backing_scale: AtomicU32,
    /// Consecutive presents at one geometry; gates the `MetalFX` route.
    present_streak: GeometryStreak,
}

impl Attachment {
    const fn new(view: usize, layer: usize, latches: &AttachLatches) -> Self {
        Self {
            view,
            layer,
            backing_scale_sink: latches.backing_scale_sink,
            cursor_kick_sink: latches.cursor_kick_sink,
            flags: latches.flags,
            color_space: latches.color_space,
            window: AtomicUsize::new(0),
            hdr_active: AtomicBool::new(latches.flags.contains(AttachFlags::HDR_ACTIVE)),
            window_occluded: AtomicBool::new(false),
            current_headroom_bits: AtomicU32::new(1.0_f32.to_bits()),
            last_logged_headroom_bits: AtomicU32::new(0),
            headroom_refresh_pending: AtomicBool::new(false),
            presents_since_headroom_refresh: AtomicU64::new(HEADROOM_REFRESH_PRESENTS),
            min_present_duration_bits: AtomicU64::new(0),
            present_pacing_bits: AtomicU64::new(latches.pacing_bits),
            current_backing_scale: AtomicU32::new(latches.backing_scale),
            present_streak: GeometryStreak::new(),
        }
    }

    /// The raw `NSView*` address this record is keyed by; compared, never dereferenced here.
    #[must_use]
    pub const fn view(&self) -> usize {
        self.view
    }

    /// The raw `CAMetalLayer*` address; compared, never dereferenced here.
    #[must_use]
    pub const fn layer(&self) -> usize {
        self.layer
    }

    /// `color.hdr.enable` as the attach carried it.
    #[must_use]
    pub const fn hdr_enable_requested(&self) -> bool {
        self.flags.contains(AttachFlags::HDR_ENABLE_REQUESTED)
    }

    /// `color.space` as the attach carried it.
    #[must_use]
    pub const fn color_space(&self) -> ColorSpacePolicy {
        self.color_space
    }

    /// The raw `NSWindow*` the occlusion observer matches, `0` = none.
    #[must_use]
    pub fn window(&self) -> usize {
        self.window.load(Ordering::Relaxed)
    }

    pub fn set_window(&self, window: usize) {
        self.window.store(window, Ordering::Relaxed);
    }

    #[must_use]
    pub fn hdr_active(&self) -> bool {
        self.hdr_active.load(Ordering::Relaxed)
    }

    pub fn set_hdr_active(&self, active: bool) {
        self.hdr_active.store(active, Ordering::Relaxed);
    }

    #[must_use]
    pub fn window_occluded(&self) -> bool {
        self.window_occluded.load(Ordering::Relaxed)
    }

    pub fn set_window_occluded(&self, occluded: bool) {
        self.window_occluded.store(occluded, Ordering::Relaxed);
    }

    /// The live EDR headroom the main thread last published.
    #[must_use]
    pub fn headroom(&self) -> f32 {
        f32::from_bits(self.current_headroom_bits.load(Ordering::Relaxed))
    }

    pub fn set_headroom(&self, headroom: f32) {
        self.current_headroom_bits
            .store(headroom.to_bits(), Ordering::Relaxed);
    }

    /// The headroom last logged, `None` = never logged.
    #[must_use]
    pub fn last_logged_headroom(&self) -> Option<f32> {
        match self.last_logged_headroom_bits.load(Ordering::Relaxed) {
            0 => None,
            bits => Some(f32::from_bits(bits)),
        }
    }

    pub fn set_last_logged_headroom(&self, headroom: f32) {
        self.last_logged_headroom_bits
            .store(headroom.to_bits(), Ordering::Relaxed);
    }

    /// Count one present and say whether it is this record's turn to queue a headroom refresh.
    ///
    /// `true` at most once per [`HEADROOM_REFRESH_PRESENTS`] presents, and
    /// never while a refresh queued earlier has not run yet. The caller
    /// queues the refresh; [`Self::end_headroom_refresh`] is the refresh
    /// reporting back.
    pub fn begin_headroom_refresh(&self) -> bool {
        let due = self
            .presents_since_headroom_refresh
            .fetch_add(1, Ordering::Relaxed)
            >= HEADROOM_REFRESH_PRESENTS;
        if !due
            || self
                .headroom_refresh_pending
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
        {
            return false;
        }
        self.presents_since_headroom_refresh
            .store(0, Ordering::Relaxed);
        true
    }

    /// The queued headroom refresh has run; the next due present may queue another.
    pub fn end_headroom_refresh(&self) {
        self.headroom_refresh_pending
            .store(false, Ordering::Release);
    }

    /// The present throttle in seconds, `0.0` = none.
    #[must_use]
    pub fn min_present_duration_sec(&self) -> f64 {
        f64::from_bits(self.min_present_duration_bits.load(Ordering::Relaxed))
    }

    pub fn set_min_present_duration(&self, seconds: f64) {
        self.min_present_duration_bits
            .store(seconds.to_bits(), Ordering::Relaxed);
    }

    /// The pacing word `pack_pacing` produced.
    #[must_use]
    pub fn pacing_bits(&self) -> u64 {
        self.present_pacing_bits.load(Ordering::Relaxed)
    }

    pub fn set_pacing_bits(&self, bits: u64) {
        self.present_pacing_bits.store(bits, Ordering::Relaxed);
    }

    /// The backing scale last published to the PE side.
    #[must_use]
    pub fn backing_scale(&self) -> u32 {
        self.current_backing_scale.load(Ordering::Relaxed)
    }

    pub fn set_backing_scale(&self, scale: u32) {
        self.current_backing_scale.store(scale, Ordering::Relaxed);
    }

    /// Advance this record's present-geometry streak, and say whether it has settled.
    pub fn present_settled(&self, geometry: PresentGeometry) -> bool {
        self.present_streak.settled(geometry)
    }
}

fn lock() -> std::sync::MutexGuard<'static, FxHashMap<usize, Arc<Attachment>>> {
    ATTACHMENTS
        .lock()
        .expect("attachment registry mutex poisoned")
}

/// Whether `att` is the record the registry holds for its view.
///
/// The liveness check every dereference in this file rests on; `map` is the
/// locked registry.
fn is_live(map: &FxHashMap<usize, Arc<Attachment>>, att: &Arc<Attachment>) -> bool {
    map.get(&att.view)
        .is_some_and(|live| Arc::ptr_eq(live, att))
}

/// Create the record for a freshly attached view and make it live.
///
/// A view address that is already registered names a record whose teardown
/// never ran; the new record replaces it so the live device wins.
pub fn register(view: usize, layer: usize, latches: &AttachLatches) -> Arc<Attachment> {
    let att = Arc::new(Attachment::new(view, layer, latches));
    if lock().insert(view, Arc::clone(&att)).is_some() {
        mtld3d_shared::log_once_warn!(
            target: crate::LOG_TARGET,
            "present: view {view:#x} attached twice without a teardown between; \\
             the earlier record is dropped",
        );
    }
    att
}

/// Take the record for `view` out of the registry. **Device teardown only.**
///
/// Runs before the teardown releases the view, so from here on no helper in
/// this file dereferences the view, its layer or the sinks. `None` for a
/// view no record names, which is what a device that never attached looks
/// like.
pub fn unregister(view: usize) -> Option<Arc<Attachment>> {
    lock().remove(&view)
}

/// The live record for `view`, if any.
#[must_use]
pub fn find(view: usize) -> Option<Arc<Attachment>> {
    lock().get(&view).cloned()
}

/// The live record whose layer is `layer`, if any.
///
/// A linear scan over a handful of records at most; the callers are the
/// Reset path, never a present.
#[must_use]
pub fn find_by_layer(layer: usize) -> Option<Arc<Attachment>> {
    lock().values().find(|att| att.layer == layer).cloned()
}

/// A snapshot of every live record, for a main-thread observer to walk.
#[must_use]
pub fn live() -> Vec<Arc<Attachment>> {
    lock().values().cloned().collect()
}

/// Retain the record's `NSView`, or `None` once it is no longer live.
///
/// **Main thread only**, because the caller goes on to walk the view.
pub fn retain_view(att: &Arc<Attachment>) -> Option<Retained<objc2_app_kit::NSView>> {
    let map = lock();
    if !is_live(&map, att) {
        return None;
    }
    // SAFETY: the record is live under the registry lock, and teardown
    // unregisters it under the same lock before it releases the metal view,
    // so the address names a live `NSView` here and the retain this takes
    // keeps it alive for the caller's walk.
    unsafe { Retained::retain(att.view as *mut objc2_app_kit::NSView) }
}

/// Retain the record's `CAMetalLayer`, or `None` once it is no longer live.
///
/// **Main thread only**, for the same reason [`retain_view`] is.
pub fn retain_layer(att: &Arc<Attachment>) -> Option<Retained<objc2_quartz_core::CAMetalLayer>> {
    let map = lock();
    if !is_live(&map, att) {
        return None;
    }
    // SAFETY: wine retains the layer for the metal view's lifetime, the
    // record is live under the registry lock, and teardown unregisters it
    // under the same lock before releasing that view, so the address names a
    // live layer here.
    unsafe { Retained::retain(att.layer as *mut objc2_quartz_core::CAMetalLayer) }
}

/// Publish a backing scale into the record's PE-side sink, while the record is live.
///
/// The store is `Relaxed`: the value stands alone, the PE side reads it once
/// per present, and there is nothing for it to order against. A record that
/// was unregistered, or that has no sink, publishes nowhere.
pub fn publish_backing_scale(att: &Arc<Attachment>, scale: u32) {
    let map = lock();
    if att.backing_scale_sink == 0 || !is_live(&map, att) {
        return;
    }
    // SAFETY: the PE side backs this address with an `AtomicU32` in a box it
    // keeps at that address until after the teardown thunk that unregisters
    // this record has returned; the record is live under the registry lock,
    // so the box is still there. `AtomicU32` has the same size and alignment
    // in both images.
    let sink = unsafe { &*(att.backing_scale_sink as *const AtomicU32) };
    sink.store(scale, Ordering::Relaxed);
}

/// Ask every live device to re-apply its cursor through Wine.
///
/// Called when the pointer comes back after another process held it. The
/// kick is idempotent (a null-then-set at the device's next `WM_SETCURSOR`),
/// so every device gets it rather than guessing which one the pointer
/// concerns. The store is `Release` against the PE side's `AcqRel` swap; the
/// flag is the whole message.
pub fn request_cursor_kick_all() {
    let map = lock();
    for att in map.values().filter(|att| att.cursor_kick_sink != 0) {
        // SAFETY: as in `publish_backing_scale`: the record is live under the
        // registry lock, so the PE-side box behind the address is still
        // there.
        let sink = unsafe { &*(att.cursor_kick_sink as *const AtomicU32) };
        sink.store(1, Ordering::Release);
    }
}

/// The record's live EDR headroom, counting this present towards the next refresh.
///
/// Reads the value the main thread last published and queues a refresh when
/// one is due. Never touches `AppKit`, so it is safe to call from the
/// presenting thread. Called every present whatever the layer is configured
/// for, because the refresh it queues is also what notices the window
/// moving onto a display of the other class: an SDR session that never
/// polled would never see a panel with headroom appear under it.
///
/// `1.0` until the first refresh lands, which is still correct for the HDR
/// shader: BT.2446-A at `L_hdr = L_sdr = 100` is the identity curve.
pub fn current_headroom(att: &Arc<Attachment>) -> f32 {
    if att.begin_headroom_refresh() {
        let att = Arc::clone(att);
        super::run_on_main_thread_async(move || super::refresh_attachment_on_main(&att));
    }
    att.headroom()
}

#[cfg(test)]
mod tests;
