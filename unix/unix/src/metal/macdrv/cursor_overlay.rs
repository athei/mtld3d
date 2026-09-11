//! The software cursor: the game's cursor bitmap drawn in a transparent overlay window.
//!
//! The PE side keeps the Win32 cursor blank while `cursor.software` is on and
//! sends the cursor's sprite and visibility through the `SetCursorOverlay`
//! thunk; this module draws that sprite in a borderless, click-through
//! `NSWindow` one level above the game window. The hardware cursor plane is
//! never toggled, and under HDR the sprite goes through the same tone map as
//! the frame, so the cursor is as bright as the UI it hovers over.
//!
//! The window never moves with the pointer: it covers the whole screen the
//! game window is on, and the sprite is a `CAMetalLayer` moved inside it. A
//! window frame change makes `AppKit` re-resolve the cursor for the pointer's
//! location, and with no cursor of our own to offer it lands on the arrow over
//! the game's blank cursor on every mouse move; a layer moving inside a fixed
//! window is invisible to that machinery. Show and hide swap the layer's
//! pixels, a sprite or a transparent clear, so its surface stays in the
//! window's scene: taking a surface out from above the game layer is free,
//! putting one back costs the game's next present a refresh, and a game
//! hiding the cursor while a button is held would pay that on every click.
//!
//! Threads. The thunk runs on the API thread and only writes [`SHARED`] and
//! queues one main-thread wakeup, coalesced through [`APPLY_PENDING`]. The
//! pre-commit run-loop observer applies the latest state once per transaction.
//! Everything that touches `AppKit`, Core Animation or the overlay's Metal objects
//! runs on the main thread: input observations, activation notifications, and
//! the observer that reconciles position, visibility, layer mode and EDR headroom.
//! Those objects live in a main-thread `thread_local`, which makes the split
//! sound without a lock around `Retained` handles.
//!
//! Nothing about the game window is latched: the game `NSWindow`, its level,
//! its client rectangle and its screen are read when the sprite can be shown,
//! from the view of the attachment record the overlay follows, so in-game resolution
//! changes, windowed/fullscreen switches and display moves need no signal
//! from the PE side. There is one system cursor and one overlay window for
//! the process, and they follow the device whose `SetCursorOverlay` arrived
//! most recently (under [`SHARED`]): `SetCursorProperties` and `ShowCursor`
//! are per-device calls, so the device that last spoke is the device the
//! game means.
//!
//! The sprite's position and its pixels reach the compositor together. The
//! layer presents with the Core Animation transaction, so a hide and the move
//! made with it land in one frame and the old sprite is never seen at a new
//! place. That matters because the apply and the game's own pointer warps
//! reach the main thread through different queues (ours the dispatch main
//! queue, winemac's its request source) and a game warps right after showing
//! or hiding its cursor: the two have no order. A warp delivers no event
//! either, so a run-loop observer ahead of Core Animation's commit reads
//! winemac's last warp time and repositions the sprite in the same iteration,
//! whichever of the warp and the apply ran first.
//!
//! Two things the pointer can do without telling this process, both handled
//! here. A system tool that takes the pointer (the interactive screenshot
//! crosshair) delivers no mouse events to the application while the pointer
//! keeps moving; every present notices the pointer moving away from its last
//! legitimate position with no event since by requesting a main-thread check.
//! The sprite stays hidden until actual input resumes. Wine can consume captured
//! input before `AppKit`'s local monitor, so the run-loop observer also observes
//! unseen mouse events through `NSApplication.currentEvent`. Retaining the last
//! event prevents idle history and recycled addresses from counting as new input.
//! The check is asked only while the game shows its cursor, the
//! application is active and winemac is not clipping the cursor, since in
//! every other state the events stay away from the application by design
//! (an inactive application gets none, mouselook clips the cursor for the
//! drag), and a warp the game made through winemac counts as a legitimate
//! move. And when such a tool ends, the window server shows
//! the standard arrow rather than the cursor Wine set, which Wine never
//! re-applies because its handle did not change; the first event after a
//! capture asks the PE side for its null-then-set kick, which makes Wine
//! re-apply through a handle change. That pointer watch serves the hardware
//! cursor too, so it is installed at attach for every device, overlay or not.

use core::{
    cell::{Cell, RefCell},
    ptr::NonNull,
};
use std::{
    collections::hash_map::Entry,
    sync::{
        Arc, LazyLock, Mutex, MutexGuard,
        atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering},
    },
    time::Instant,
};

use block2::RcBlock;
use log::{debug, info};
use mtld3d_shared::{SetCursorOverlayParams, mtl::CursorOverlayFlags};
use objc2::{
    AnyThread, MainThreadMarker, MainThreadOnly, Message, extern_class, extern_methods,
    rc::{Retained, autoreleasepool},
    runtime::{AnyClass, NSObject, ProtocolObject},
};
use objc2_app_kit::{
    NSApplication, NSApplicationDidBecomeActiveNotification,
    NSApplicationDidResignActiveNotification, NSBackingStoreType, NSBitmapImageRep, NSColor,
    NSCursor, NSDeviceRGBColorSpace, NSEvent, NSEventMask, NSImage, NSScreen, NSView, NSWindow,
    NSWindowAnimationBehavior, NSWindowCollectionBehavior, NSWindowStyleMask,
};
use objc2_core_foundation::{
    CFRunLoop, CFRunLoopActivity, CFRunLoopObserver, CGPoint, CGRect, CGSize, kCFRunLoopCommonModes,
};
use objc2_core_graphics::CGColorSpace;
use objc2_foundation::{
    NSDictionary, NSInteger, NSNotification, NSNotificationCenter, NSNull, NSString,
};
use objc2_metal::{
    MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandQueue, MTLDevice, MTLDrawable, MTLOrigin,
    MTLPixelFormat, MTLRegion, MTLResource, MTLSize, MTLTexture, MTLTextureDescriptor,
    MTLTextureType, MTLTextureUsage,
};
use objc2_quartz_core::{CAAction, CALayer, CAMetalDrawable, CAMetalLayer};
use rustc_hash::FxHashMap;

use super::{
    LayerMode,
    attachment::{self, Attachment},
    run_on_main_thread_async, run_on_main_thread_sync,
};
use crate::metal::{command, device::cpu_written_texture_storage, present};

/// Log sub-target of the software cursor.
///
/// Inherits `mtld3d::unix` filters by prefix; `mtld3d::unix::cursor=debug`
/// shows every apply with the sprite, visibility and layer mode it landed.
const LOG_TARGET: &str = "mtld3d::unix::cursor";

/// Where the sprite layer sits before it has ever been positioned.
const PARKED: CGPoint = CGPoint {
    x: -100_000.0,
    y: -100_000.0,
};

/// What the sprite layer's drawable currently shows.
#[derive(Debug, PartialEq)]
enum Content {
    /// The submitted drawable is a transparent clear.
    Transparent,
    /// A sprite, tone-mapped for a layer mode and a headroom.
    Sprite {
        hash: u64,
        mode: LayerMode,
        peak: f32,
        geometry: SpriteGeometry,
    },
}

/// How long without a mouse event before a moved pointer counts as captured.
///
/// A moving pointer delivers an event every few milliseconds, so a gap this
/// long with the pointer elsewhere than the last event put it means someone
/// else is receiving the events. Checked on existing input, run-loop and
/// present opportunities, without a timer or an immediate retry loop.
const CAPTURE_SILENCE_MS: u128 = 60;

extern_class!(
    /// winemac's application controller, read for its cursor-clipping state.
    ///
    /// Declared here because no binding crate carries Wine's classes. Only
    /// the two members below are touched, both part of the driver's own
    /// header for as long as it has clipped the cursor; the class is looked
    /// up by name before use so a driver without it reads as never clipping.
    #[unsafe(super(NSObject))]
    #[name = "WineApplicationController"]
    struct WineApplicationController;
);

impl WineApplicationController {
    extern_methods!(
        #[unsafe(method(sharedController))]
        #[unsafe(method_family = none)]
        fn shared_controller() -> Option<Retained<Self>>;

        #[unsafe(method(clippingCursor))]
        #[unsafe(method_family = none)]
        fn clipping_cursor(&self) -> bool;

        #[unsafe(method(lastSetCursorPositionTime))]
        #[unsafe(method_family = none)]
        fn last_set_cursor_position_time(&self) -> f64;
    );
}

/// Whether the running driver has a `WineApplicationController` class at all.
static HAS_WINE_CONTROLLER: LazyLock<bool> =
    LazyLock::new(|| AnyClass::get(c"WineApplicationController").is_some());

/// Whether winemac is clipping the cursor. Main thread only.
///
/// Clipping disassociates the pointer from the mouse, so its position is not
/// evidence of external capture while the game constrains it.
fn wine_clips_cursor(_mtm: MainThreadMarker) -> bool {
    *HAS_WINE_CONTROLLER
        && WineApplicationController::shared_controller()
            .is_some_and(|controller| controller.clipping_cursor())
}

/// The uptime of winemac's last `SetCursorPos` warp, `0.0` when none is outstanding.
///
/// The game moving the pointer through Wine is the one pointer move that
/// neither comes from the mouse nor from another process; winemac records
/// its time so its own mouse handling can discard the events the warp
/// crosses, and the capture check reads the same record.
fn wine_last_warp_uptime(_mtm: MainThreadMarker) -> f64 {
    if !*HAS_WINE_CONTROLLER {
        return 0.0;
    }
    WineApplicationController::shared_controller()
        .map_or(0.0, |controller| controller.last_set_cursor_position_time())
}

/// Clock shared by main-thread input observations and queued-apply diagnostics.
static EPOCH: LazyLock<Instant> = LazyLock::new(Instant::now);

/// Input history owned entirely by the main thread.
///
/// Event identity is retained by `PointerWatch`, so an address cannot be reused
/// while it is the deduplication key. A repeated `currentEvent` is idle history,
/// never evidence that new input reached Wine.
#[derive(Default)]
struct InputState {
    event: usize,
    at_ns: Option<u64>,
    position: CGPoint,
    warp: f64,
    capture_epoch: u64,
    captured: bool,
}

impl InputState {
    const fn suspend(&mut self) -> bool {
        // No silence interval spans a hidden or inactive period. The first
        // active observation establishes a fresh position without native
        // pointer or Wine-controller queries while the watch is suspended.
        self.at_ns = None;
        core::mem::replace(&mut self.captured, false)
    }

    const fn note_position(&mut self, position: CGPoint, now: u64) {
        self.position = position;
        self.at_ns = Some(now);
    }

    const fn observe(&mut self, event: usize, position: CGPoint, now: u64) -> Option<bool> {
        if self.event == event {
            // The local monitor and the run-loop observer can see the same event.
            return None;
        }
        self.event = event;
        self.note_position(position, now);
        Some(core::mem::replace(&mut self.captured, false))
    }

    fn reconcile(&mut self, observation: &PointerObservation) {
        if self.capture_epoch != observation.capture_epoch {
            self.capture_epoch = observation.capture_epoch;
            self.captured = false;
            self.note_position(observation.position, observation.now);
        }
        if warp_since(self.warp, observation.warp) {
            self.warp = observation.warp;
            self.note_position(observation.position, observation.now);
        }
        if !observation
            .flags
            .contains(PointerFlags::SHOWN | PointerFlags::ACTIVE)
        {
            self.captured = false;
            return;
        }
        if observation.flags.contains(PointerFlags::CLIPPED) {
            // A clipped move is legitimate, including after a capture was suspected.
            self.captured = false;
            self.note_position(observation.position, observation.now);
            return;
        }
        let Some(at) = self.at_ns else {
            // Establish a baseline before attempting a silence measurement.
            self.note_position(observation.position, observation.now);
            return;
        };
        let silence = u128::from(observation.now.saturating_sub(at)) / 1_000_000;
        let moved = (observation.position.x - self.position.x).abs() > 0.5
            || (observation.position.y - self.position.y).abs() > 0.5;
        self.captured |= pointer_captured(silence, moved);
    }
}

bitflags::bitflags! {
    /// Main-thread inputs to the external-capture decision.
    struct PointerFlags: u8 {
        const SHOWN = 1 << 0;
        const ACTIVE = 1 << 1;
        const CLIPPED = 1 << 2;
    }
}

struct PointerObservation {
    position: CGPoint,
    now: u64,
    warp: f64,
    capture_epoch: u64,
    flags: PointerFlags,
}

#[derive(Default)]
struct PointerWatch {
    event: Option<Retained<NSEvent>>,
    input: InputState,
    /// Native cursor ownership changes on activation, independently of mouse input.
    activation: u64,
}

/// Request one coalesced main-thread check from the submit thread.
///
/// Wine's controller fields and `NSApplication.currentEvent` are main-thread
/// state. No decision taken on the submit thread can race a recovering event.
pub fn poll_capture_from_present() {
    let check = {
        let shared = lock_shared();
        shared.pending
            || (shared.owner.is_some() && shared.flags.contains(CursorOverlayFlags::VISIBLE))
    };
    if check {
        queue_apply();
    }
}

fn now_ns() -> u64 {
    u64::try_from(EPOCH.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

/// Whether the pointer moved away from its last legitimate position with no event since.
///
/// A system tool that takes the pointer (the screenshot crosshair) leaves the
/// application eventless while the pointer keeps moving. The last legitimate
/// position is where the last mouse event or the last winemac warp put the
/// pointer, so a `SetCursorPos` by the game counts as the game's own move
/// rather than another process's. Only asked while the game shows its cursor
/// and Wine is not clipping it; either way otherwise the game owns the
/// pointer and the events legitimately stay away from the application.
const fn pointer_captured(silence_ms: u128, moved: bool) -> bool {
    moved && silence_ms >= CAPTURE_SILENCE_MS
}

/// `developerHUDProperties` mode that keeps the Metal performance HUD off this layer.
///
/// With `MTL_HUD_ENABLED` in the environment the HUD attaches to every
/// `CAMetalLayer` in the process, and on a cursor-sized layer that presents
/// once per sprite it is a black box reading "inf" over the cursor.
const HUD_MODE_OFF: &str = "disabled";

/// A cursor bitmap as the PE side shipped it: tight BGRA rows, already upscaled.
struct Sprite {
    width: u32,
    height: u32,
    x_hotspot: u32,
    y_hotspot: u32,
    /// Sprite pixels per point; the overlay layer's `contentsScale`.
    scale: u32,
    pixels: Box<[u8]>,
}

/// State written by the thunk on the API thread and read by the main thread.
#[derive(Default)]
struct Shared {
    /// Content-addressed uploads remain available to every device's acknowledged set.
    sprites: FxHashMap<u64, Arc<Sprite>>,
    /// Identity, mode, sprite and visibility are published under this one mutex.
    owner: Option<Arc<Attachment>>,
    hash: u64,
    flags: CursorOverlayFlags,
    revision: u64,
    /// Remembers hides even when a later show coalesces into the same apply.
    capture_epoch: u64,
    /// Native work to complete; hardware-only state has no overlay to apply.
    pending: bool,
}

impl Shared {
    fn update(
        &mut self,
        view: usize,
        params: &SetCursorOverlayParams,
        pixels: Option<&[u8]>,
    ) -> bool {
        // Admission is inside SHARED. Unregister releases the registry lock
        // before detaching, so an admitted update cannot resurrect a retired owner.
        let Some(owner) = attachment::find(view) else {
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET, "SetCursorOverlay: view {view:#x} has no live attachment",
            );
            return false;
        };
        let hardware = params.flags.contains(CursorOverlayFlags::HARDWARE);
        if !hardware && pixels.is_none() && !self.sprites.contains_key(&params.hash) {
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET, "SetCursorOverlay: unknown sprite {:#018x}; pixels required",
                params.hash,
            );
            return false;
        }
        let hash = if hardware { 0 } else { params.hash };
        if same_owner(self.owner.as_ref(), Some(&owner))
            && self.hash == hash
            && self.flags == params.flags
        {
            // Admission and upload acknowledgment still run on every call. An
            // identical request keeps any failed work pending without waking
            // main again after the same state has successfully completed.
            return true;
        }
        if !same_owner(self.owner.as_ref(), Some(&owner))
            || !params.flags.contains(CursorOverlayFlags::VISIBLE)
        {
            self.capture_epoch += 1;
        }
        if !hardware && let Some(pixels) = pixels {
            self.sprites.entry(params.hash).or_insert_with(|| {
                Arc::new(Sprite {
                    width: params.width,
                    height: params.height,
                    x_hotspot: params.x_hotspot,
                    y_hotspot: params.y_hotspot,
                    scale: params.scale,
                    pixels: pixels.into(),
                })
            });
        }
        self.owner = Some(owner);
        self.hash = hash;
        self.flags = params.flags;
        self.revision += 1;
        // Until a software sprite has been accepted, no overlay can exist.
        // Hardware visibility is already published for the main-thread watch;
        // it needs no separate UI wakeup. Once software has been used, retain
        // the apply so hardware takeover clears any sprite or failed draw.
        self.pending = !hardware || !self.sprites.is_empty();
        true
    }

    fn detach(&mut self, retired: &Arc<Attachment>) -> bool {
        if !same_owner(self.owner.as_ref(), Some(retired)) {
            // Another attachment owns the cursor, even if its view address was reused.
            return false;
        }
        self.owner = None;
        self.hash = 0;
        self.flags = CursorOverlayFlags::empty();
        self.revision += 1;
        self.capture_epoch += 1;
        self.pending = true;
        true
    }

    fn snapshot(&self) -> WantedSnapshot {
        WantedSnapshot {
            owner: self.owner.as_ref().map(Arc::clone),
            sprite: self.sprites.get(&self.hash).map(Arc::clone),
            hash: self.hash,
            flags: self.flags,
            revision: self.revision,
            capture_epoch: self.capture_epoch,
        }
    }

    const fn applied(&mut self, revision: u64, completed: bool) {
        if self.revision == revision {
            self.pending = !completed;
        }
    }
}

fn same_owner(a: Option<&Arc<Attachment>>, b: Option<&Arc<Attachment>>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => Arc::ptr_eq(a, b),
        (None, None) => true,
        _ => false, // Only one snapshot has an attachment.
    }
}

static SHARED: LazyLock<Mutex<Shared>> = LazyLock::new(|| Mutex::new(Shared::default()));

/// Whether an apply is already queued on the main thread.
///
/// Bounds the main queue to one outstanding apply however fast the API thread
/// toggles the cursor; the apply reads the latest wanted state when it runs.
static APPLY_PENDING: AtomicBool = AtomicBool::new(false);

/// When the pending apply was queued, nanoseconds since [`EPOCH`].
///
/// The apply logs how long it waited for the main thread at debug level,
/// which is the number that says whether a cursor change landed late.
static APPLY_QUEUED_NS: AtomicU64 = AtomicU64::new(0);

/// The sprite's extent and hotspot in points, the window's coordinate unit.
#[derive(Debug, Default, PartialEq)]
struct SpriteGeometry {
    width: f64,
    height: f64,
    hotspot_x: f64,
    hotspot_y: f64,
    /// Sprite pixels per point: the overlay layer's `contentsScale`.
    scale: f64,
}

impl SpriteGeometry {
    /// Size sprite pixels the way winemac sizes a hardware cursor's image.
    ///
    /// winemac divides the cursor bitmap's pixel size by the prefix's retina
    /// factor (2 in retina mode, else 1) to get its point size, whatever the
    /// bitmap's own scale; `cursor.scale` therefore enlarges both cursors
    /// alike only when the sprite is divided by the same factor, which is
    /// the layer scale the attach published, never the sprite's own.
    fn of(sprite: &Sprite, retina_factor: u32) -> Self {
        let scale = f64::from(retina_factor.max(1));
        Self {
            width: f64::from(sprite.width) / scale,
            height: f64::from(sprite.height) / scale,
            hotspot_x: f64::from(sprite.x_hotspot) / scale,
            hotspot_y: f64::from(sprite.y_hotspot) / scale,
            scale,
        }
    }
}

bitflags::bitflags! {
    /// Everything the visibility decision looks at, in one word.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    struct VisibilityInputs: u8 {
        /// The PE side shows the cursor and a sprite is rendered.
        const WANTED = 1 << 0;
        /// The Wine process is the active application.
        ///
        /// macOS gives the pointer to the frontmost application; a sprite
        /// over an inactive game window would sit next to the real arrow.
        const APP_ACTIVE = 1 << 1;
        /// The pointer is over the game's client area, with no other window above it there.
        ///
        /// A dialog or another application's panel over the game shows its
        /// own hardware cursor; a sprite drawn over that would be a second
        /// pointer.
        const POINTER_INSIDE = 1 << 2;
        /// The game window is fully covered or minimised.
        const OCCLUDED = 1 << 3;
        /// The game window sits in the Dock.
        const MINIATURIZED = 1 << 4;
        /// A system tool has the pointer: it moves while no events reach us.
        const CAPTURED = 1 << 5;
    }
}

/// Whether the overlay shows its sprite for these inputs.
const fn overlay_visible(inputs: VisibilityInputs) -> bool {
    inputs.contains(VisibilityInputs::WANTED.union(VisibilityInputs::APP_ACTIVE))
        && inputs.contains(VisibilityInputs::POINTER_INSIDE)
        && !inputs.intersects(
            VisibilityInputs::OCCLUDED
                .union(VisibilityInputs::MINIATURIZED)
                .union(VisibilityInputs::CAPTURED),
        )
}

/// The sprite layer's origin that puts the sprite's hotspot under the pointer.
///
/// `mouse` and the result are in the overlay window's coordinates, which grow
/// upwards with the origin at the bottom left like the screen's, while the
/// hotspot is measured from the sprite's top left.
const fn sprite_origin(mouse: (f64, f64), geometry: &SpriteGeometry) -> (f64, f64) {
    (
        mouse.0 - geometry.hotspot_x,
        mouse.1 - (geometry.height - geometry.hotspot_y),
    )
}

/// Whether `point` lies inside `rect` (left and bottom inclusive, right and top exclusive).
fn rect_contains(rect: CGRect, point: CGPoint) -> bool {
    point.x >= rect.origin.x
        && point.y >= rect.origin.y
        && point.x < rect.origin.x + rect.size.width
        && point.y < rect.origin.y + rect.size.height
}

/// Whether a headroom move is worth re-rendering the sprite for.
///
/// The same 5% relative rule the headroom log uses, plus the `1.0` boundary,
/// where the frame switches between the pass-through and BT.2446 pipelines and
/// the sprite has to switch with it.
fn peak_changed(applied: f32, current: f32) -> bool {
    let relative = (current - applied).abs() / applied.max(f32::EPSILON);
    relative > 0.05 || (applied <= 1.0) != (current <= 1.0)
}

/// `SetCursorOverlay`: record the wanted sprite and visibility, queue one apply.
///
/// `pixels` is `Some` when this hash is new to the unix side; the bytes are
/// copied here, so the PE buffer only has to live for the call. Never blocks
/// on the main thread. `false` when `params.view_handle` names no attachment
/// record, in which case nothing is recorded: the overlay has no window to
/// draw over for a device that never attached, and a retired view must not
/// become the one it follows.
pub fn set_cursor_overlay(params: &SetCursorOverlayParams, pixels: Option<&[u8]>) -> bool {
    let view = usize::try_from(params.view_handle.raw())
        .expect("a 64-bit host addresses every view pointer");
    let (accepted, pending) = {
        let mut shared = lock_shared();
        let accepted = shared.update(view, params, pixels);
        (accepted, shared.pending)
    };
    if accepted && pending {
        queue_apply();
    }
    accepted
}

/// The device that attached `view` is going away: stop following it.
///
/// Only the device the overlay follows changes anything on screen: its sprite
/// is hidden and the cursor is no longer shown, so the capture checks stop.
/// Another device's teardown leaves the overlay where it is. The uploaded
/// sprites stay: they are content-addressed, so a second device's uploaded
/// set may name an entry the first one sent, and its next call would name a
/// sprite the unix side no longer held. Input observers stay for the process
/// lifetime; the cursor surface retires before the followed game surface so
/// Metal cannot promote the surviving cursor to its main timing layer.
pub fn detach(retired: &Arc<Attachment>) {
    let changed = lock_shared().detach(retired);
    if changed {
        let retired = Arc::clone(retired);
        run_on_main_thread_sync(move || {
            let mtm = MainThreadMarker::new().expect("cursor detach runs on the main thread");
            OVERLAY.with_borrow_mut(|slot| {
                if slot
                    .as_ref()
                    .is_some_and(|overlay| same_owner(overlay.owner.as_ref(), Some(&retired)))
                    && let Some(overlay) = slot.take()
                {
                    overlay.retire(mtm);
                }
            });
        });
        queue_apply();
    }
}

/// Let the game reach the display before introducing a second presenting surface.
///
/// A HUD-disabled cursor layer can still become Metal's main timing layer when
/// it presents first. A zero presented time denotes a drawable that did not
/// reach the display and must leave the observation armed for the next frame.
pub fn observe_game_present(att: &Arc<Attachment>, drawable: &ProtocolObject<dyn CAMetalDrawable>) {
    if !att.needs_first_display() {
        // Startup observation has already queued or published its result.
        return;
    }
    let att = Arc::clone(att);
    let handler = RcBlock::new(move |ptr: NonNull<ProtocolObject<dyn MTLDrawable>>| {
        // SAFETY: Metal supplies the drawable for the duration of this callback.
        let drawable = unsafe { ptr.as_ref() };
        if super::host_seconds_to_ns(drawable.presentedTime()) == 0 || !att.queue_first_display() {
            // Discarded frame or another displayed frame already queued publication.
            return;
        }
        let att = Arc::clone(&att);
        run_on_main_thread_async(move || {
            let mtm = MainThreadMarker::new().expect("first display runs on the main thread");
            if attachment::retain_layer(&att, mtm).is_none() {
                mtld3d_shared::log_once_info!(target: LOG_TARGET, "cursor: first display arrived after attachment retirement");
                return;
            }
            att.publish_first_display();
            debug!(target: LOG_TARGET, "cursor: game reached display view={:#x}; overlay may present", att.view());
            queue_apply();
        });
    });
    // SAFETY: Metal copies the callback. It owns only an Arc to the attachment
    // record and validates liveness on main before accessing native objects.
    unsafe { drawable.addPresentedHandler(RcBlock::as_ptr(&handler)) };
}

/// Observe input now; reconcile pixels and position at the run-loop commit. Main thread only.
pub fn reconcile_on_main() {
    let mtm = MainThreadMarker::new().expect("reconcile_on_main runs on the main thread");
    install_pointer_watch(mtm);
    observe_current_event(mtm);
}

fn lock_shared() -> MutexGuard<'static, Shared> {
    SHARED.lock().expect("cursor overlay mutex poisoned")
}

fn queue_apply() {
    if APPLY_PENDING
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_ok()
    {
        let ns = u64::try_from(EPOCH.elapsed().as_nanos()).unwrap_or(u64::MAX);
        APPLY_QUEUED_NS.store(ns, Ordering::Relaxed);
        run_on_main_thread_async(apply_on_main);
    }
}

/// Wake the input watch; the pre-commit observer applies the latest state. Main thread only.
fn apply_on_main() {
    APPLY_PENDING.store(false, Ordering::Release);
    let queued_ns = APPLY_QUEUED_NS.load(Ordering::Relaxed);
    let waited_us = EPOCH
        .elapsed()
        .as_nanos()
        .saturating_sub(u128::from(queued_ns))
        / 1_000;
    debug!(target: LOG_TARGET, "cursor: apply ran {waited_us} us after it was queued");
    reconcile_on_main();
}

/// One owned snapshot used throughout a native reconciliation.
struct WantedSnapshot {
    owner: Option<Arc<Attachment>>,
    sprite: Option<Arc<Sprite>>,
    hash: u64,
    flags: CursorOverlayFlags,
    revision: u64,
    capture_epoch: u64,
}

bitflags::bitflags! {
    /// Installed observer components; failed components are retried at existing opportunities.
    #[derive(Clone, Copy)]
    struct WatchInstalled: u8 {
        const MONITOR = 1 << 0;
        const RUN_LOOP = 1 << 1;
        const ACTIVATION = 1 << 2;
    }
}

thread_local! {
    static POINTER_WATCH_INSTALLED: Cell<WatchInstalled> = const {
        Cell::new(WatchInstalled::empty())
    };
    static POINTER_WATCH: RefCell<PointerWatch> = RefCell::new(PointerWatch::default());
    /// Every access runs on main; try_borrow_mut also handles AppKit reentrancy.
    static OVERLAY: RefCell<Option<Overlay>> = const { RefCell::new(None) };
}

/// Present at most one cursor drawable before each Core Animation commit.
///
/// Multiple presents to one layer in an implicit transaction can display its
/// first drawable while later GPU completions report success. Input, activation
/// and queued requests therefore converge here before any pixels are submitted.
fn apply_on_main_inner() {
    let mtm = MainThreadMarker::new().expect("apply_on_main_inner runs on the main thread");
    let wanted = {
        let shared = lock_shared();
        if shared.owner.is_none() && !shared.pending {
            // No device has a D3D cursor and no detach or failed draw remains to apply.
            return;
        }
        shared.snapshot()
    };
    install_pointer_watch(mtm);
    observe_current_event(mtm);
    let captured = reconcile_pointer(mtm, &wanted);
    OVERLAY.with(|cell| {
        let Ok(mut slot) = cell.try_borrow_mut() else {
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET,
                "cursor: reentrant apply deferred to the next event, run loop or present",
            );
            return;
        };
        if slot.is_none() {
            if wanted.sprite.is_none() {
                // Hardware-only or detached: no overlay needs creating.
                lock_shared().applied(wanted.revision, true);
                return;
            }
            if !wanted.owner.as_ref().is_some_and(|att| att.has_displayed()) {
                // The displayed-drawable callback will wake the observer. The
                // request remains pending while the game has no on-screen frame.
                mtld3d_shared::log_once_info!(target: LOG_TARGET, "cursor: overlay waits for the game's first displayed frame");
                return;
            }
            *slot = Overlay::create(mtm, &wanted);
        }
        if let Some(overlay) = slot.as_mut() {
            let completed = overlay.apply(mtm, &wanted, captured);
            lock_shared().applied(wanted.revision, completed);
        }
    });
}

fn observe_mouse_event(event: &NSEvent, route: &str, _mtm: MainThreadMarker) {
    let mask = NSEventMask(
        1u64.checked_shl(u32::try_from(event.r#type().0).unwrap_or(64))
            .unwrap_or(0),
    );
    if !mouse_mask().intersects(mask) {
        // currentEvent may still be a key, scroll, or application event.
        return;
    }
    let recovered = POINTER_WATCH.with_borrow_mut(|watch| {
        let recovered = watch.input.observe(
            core::ptr::from_ref(event) as usize,
            NSEvent::mouseLocation(),
            now_ns(),
        )?;
        watch.event = Some(event.retain());
        Some(recovered)
    });
    if recovered.is_some() {
        log::trace!(target: LOG_TARGET, "cursor: input route={route} type={:?} event={}", event.r#type(), event.eventNumber());
    }
    if recovered == Some(true) {
        debug!(target: LOG_TARGET, "cursor: input resumed after external capture; requesting cursor kick");
        attachment::request_cursor_kick_all();
    }
}

fn observe_current_event(mtm: MainThreadMarker) {
    if let Some(event) = NSApplication::sharedApplication(mtm).currentEvent() {
        observe_mouse_event(&event, "currentEvent", mtm);
    }
}

fn reconcile_pointer(mtm: MainThreadMarker, wanted: &WantedSnapshot) -> bool {
    if !wanted.flags.contains(CursorOverlayFlags::VISIBLE)
        || !NSApplication::sharedApplication(mtm).isActive()
    {
        if POINTER_WATCH.with_borrow_mut(|watch| watch.input.suspend()) {
            debug!(target: LOG_TARGET, "cursor: external capture=false (watch suspended)");
        }
        return false;
    }
    let mut flags = PointerFlags::SHOWN | PointerFlags::ACTIVE;
    flags.set(PointerFlags::CLIPPED, wine_clips_cursor(mtm));
    let observation = PointerObservation {
        position: NSEvent::mouseLocation(),
        now: now_ns(),
        warp: wine_last_warp_uptime(mtm),
        capture_epoch: wanted.capture_epoch,
        flags,
    };
    POINTER_WATCH.with_borrow_mut(|watch| {
        let previous = watch.input.captured;
        watch.input.reconcile(&observation);
        if previous != watch.input.captured {
            debug!(target: LOG_TARGET, "cursor: external capture={}", watch.input.captured);
        }
        watch.input.captured
    })
}

/// Actual layer inputs; HDR/SDR mode alone cannot identify a color configuration.
#[derive(Debug, PartialEq)]
struct LayerConfiguration<'a> {
    format: MTLPixelFormat,
    colorspace: Option<&'a CGColorSpace>,
    edr: bool,
}

/// A submission owns its result cell; stale callbacks can only update their own cell.
struct Submission {
    content: Content,
    result: Arc<AtomicU8>,
}

const SUBMITTED: u8 = 0;
const COMPLETED: u8 = 1;
const FAILED: u8 = 2;

/// Submitted content is reusable while pending, but only completion settles the request.
#[derive(Default)]
struct ContentState {
    generation: u64,
    submission: Option<Submission>,
}

impl ContentState {
    fn current(&self) -> Option<&Content> {
        self.submission
            .as_ref()
            .filter(|s| s.result.load(Ordering::Acquire) != FAILED)
            .map(|s| &s.content)
    }

    fn completed(&self) -> bool {
        self.submission
            .as_ref()
            .is_some_and(|s| s.result.load(Ordering::Acquire) == COMPLETED)
    }

    fn invalidate(&mut self) {
        self.submission = None;
    }

    fn ensure(
        &mut self,
        content: Content,
        draw: impl FnOnce(&Content, u64, Arc<AtomicU8>) -> bool,
    ) -> bool {
        if self.current() == Some(&content) {
            // Reuse a scheduled submission. Its completion may still invalidate it.
            return true;
        }
        self.invalidate();
        self.generation += 1;
        let result = Arc::new(AtomicU8::new(SUBMITTED));
        if !draw(&content, self.generation, Arc::clone(&result)) {
            // Allocation or encoding failed. Retry only at the next existing opportunity.
            return false;
        }
        debug!(target: LOG_TARGET, "cursor: submitted generation={} content={content:?}", self.generation);
        self.submission = Some(Submission { content, result });
        true
    }
}

/// The overlay window and everything rendered into it. **Main thread only.**
struct Overlay {
    window: Retained<NSWindow>,
    /// Native blank selected on startup and application activation.
    native_cursor: Retained<NSCursor>,
    /// Last activation whose native cursor was realized over the game.
    cursor_activation: Option<u64>,
    /// The sprite: a sublayer of the window's content layer, moved per event.
    layer: Retained<CAMetalLayer>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    /// One `MTLTexture` per sprite hash, uploaded on first render.
    textures: FxHashMap<u64, Retained<ProtocolObject<dyn MTLTexture>>>,
    /// What the layer's drawable shows right now.
    content: ContentState,
    /// Identity of the last reconciled attachment, never just its recyclable address.
    owner: Option<Arc<Attachment>>,
    mode: Option<LayerMode>,
    visibility: Option<VisibilityInputs>,
}

impl Overlay {
    /// Remove the cursor surface before its followed game surface is released.
    fn retire(self, _mtm: MainThreadMarker) {
        self.layer.removeFromSuperlayer();
        self.window.orderOut(None);
        self.window.close();
        debug!(target: LOG_TARGET, "cursor: retired overlay with its game attachment");
    }

    /// Create the window, its layer and the input hooks. **Main thread only.**
    ///
    /// `None` when there is no attachment to borrow a layer's device from or
    /// native cursor or Metal queue allocation fails; the next apply tries again.
    fn create(mtm: MainThreadMarker, wanted: &WantedSnapshot) -> Option<Self> {
        let native_cursor = native_blank_cursor(mtm)?;
        let Some(game_layer) = wanted
            .owner
            .as_ref()
            .and_then(|att| attachment::retain_layer(att, mtm))
        else {
            mtld3d_shared::log_once_warn!(target: LOG_TARGET, "cursor: overlay creation deferred without a live game layer");
            return None;
        };
        let Some(device) = game_layer.device() else {
            mtld3d_shared::log_once_warn!(target: LOG_TARGET, "cursor: game layer has no device; create deferred");
            return None;
        };
        let Some(queue) = device.newCommandQueue() else {
            mtld3d_shared::log_once_warn!(target: LOG_TARGET, "cursor: command queue allocation failed; create deferred");
            return None;
        };
        queue.setLabel(Some(&NSString::from_str("mtld3d-cursor-queue")));

        let layer = CAMetalLayer::new();
        layer.setDevice(Some(&device));
        // The sprite has an alpha channel and the window behind it is clear:
        // the compositor blends the whole window onto the game.
        layer.setOpaque(false);
        layer.setFramebufferOnly(true);
        // Allow two visibility flips in flight alongside the compositor's
        // current drawable. nextDrawable can still wait when all three are busy.
        layer.setMaximumDrawableCount(3);
        layer.setAllowsNextDrawableTimeout(true);
        // The pixels ride the Core Animation transaction that carries the
        // layer's position, so a hide and the move that goes with it reach
        // the compositor in one frame; see `Overlay::sync_position`.
        layer.setPresentsWithTransaction(true);
        // No implicit animation on anything written here: without this, the
        // layer having no delegate, every position or bounds write would ease
        // over Core Animation's default quarter second.
        layer.setActions(Some(&no_actions()));
        layer.setName(Some(&NSString::from_str("mtld3d-cursor-overlay")));
        // Positioned by its bottom-left corner, like the window it lives in.
        layer.setAnchorPoint(CGPoint { x: 0.0, y: 0.0 });
        layer.setPosition(PARKED);
        layer.setDrawableSize(CGSize {
            width: 1.0,
            height: 1.0,
        });
        layer.setBounds(CGRect {
            origin: CGPoint::default(),
            size: CGSize {
                width: 1.0,
                height: 1.0,
            },
        });
        let hud = NSDictionary::from_slices::<NSString>(
            &[&NSString::from_str("mode")],
            &[&*NSString::from_str(HUD_MODE_OFF)],
        );
        // SAFETY: an `NSDictionary<NSString, NSString>` is an `NSDictionary`
        // of objects; the erased view is what the setter is declared with.
        let hud = unsafe { Retained::cast_unchecked::<NSDictionary>(hud) };
        // SAFETY: objc2 typed binding; the dictionary is copied by the layer.
        unsafe { layer.setDeveloperHUDProperties(Some(&hud)) };

        let frame = overlay_frame(mtm, wanted.owner.as_ref());
        // SAFETY: standard NSWindow initialiser on a fresh allocation; the
        // borderless mask and buffered backing are the documented values for
        // an overlay, and `defer = false` gives the window its server-side
        // counterpart now so the ordering and level calls below take effect.
        let window = unsafe {
            NSWindow::initWithContentRect_styleMask_backing_defer(
                NSWindow::alloc(mtm),
                frame,
                NSWindowStyleMask::Borderless,
                NSBackingStoreType::Buffered,
                false,
            )
        };
        // SAFETY: the window is owned by this `Retained` and never closed
        // through `close`, so AppKit must not release it on our behalf.
        unsafe { window.setReleasedWhenClosed(false) };
        window.setOpaque(false);
        window.setBackgroundColor(Some(&NSColor::clearColor()));
        // Click-through: every mouse event lands on the game window below.
        window.setIgnoresMouseEvents(true);
        window.setHasShadow(false);
        window.setAnimationBehavior(NSWindowAnimationBehavior::None);
        // On every Space: the overlay is never ordered in or out, so it must
        // be wherever the game window is moved to, and a window on no visible
        // Space would be an uncomposited layer whose presents block.
        window.setCollectionBehavior(
            NSWindowCollectionBehavior::CanJoinAllSpaces
                | NSWindowCollectionBehavior::Transient
                | NSWindowCollectionBehavior::IgnoresCycle
                | NSWindowCollectionBehavior::FullScreenAuxiliary,
        );
        let view = NSView::initWithFrame(
            NSView::alloc(mtm),
            CGRect {
                origin: CGPoint { x: 0.0, y: 0.0 },
                size: frame.size,
            },
        );
        // Layer-hosting: a plain content layer sized with the view carries the
        // sprite as a sublayer, so moving the sprite touches no view or window
        // geometry.
        let host = CALayer::new();
        host.addSublayer(&layer);
        view.setLayer(Some(&host));
        view.setWantsLayer(true);
        window.setContentView(Some(&view));
        window.orderFrontRegardless();

        info!(
            target: LOG_TARGET,
            "cursor: overlay window created over ({:.0},{:.0}) {:.0}x{:.0} (borderless, click-through)",
            frame.origin.x, frame.origin.y, frame.size.width, frame.size.height,
        );
        Some(Self {
            window,
            native_cursor,
            cursor_activation: None,
            layer,
            queue,
            textures: FxHashMap::default(),
            content: ContentState::default(),
            owner: None,
            mode: None,
            visibility: None,
        })
    }

    /// Reconcile one owner and request throughout all native work.
    fn apply(&mut self, mtm: MainThreadMarker, wanted: &WantedSnapshot, captured: bool) -> bool {
        if !same_owner(self.owner.as_ref(), wanted.owner.as_ref()) {
            self.content.invalidate();
            self.owner = wanted.owner.as_ref().map(Arc::clone);
        }
        if let Some(att) = wanted.owner.as_ref() {
            let Some(game_layer) = attachment::retain_layer(att, mtm) else {
                mtld3d_shared::log_once_warn!(target: LOG_TARGET, "cursor: attachment retired during apply");
                self.ensure_content(Content::Transparent, None);
                return false;
            };
            if !self.reconfigure_layer(&game_layer, att) {
                self.ensure_content(Content::Transparent, None);
                return false;
            }
        }
        self.sync_position(mtm, wanted, captured);
        self.content.completed()
    }

    /// Mirror the actual followed layer, including same-mode profile and device handoffs.
    fn reconfigure_layer(&mut self, game: &CAMetalLayer, att: &Attachment) -> bool {
        let Some(device) = game.device() else {
            mtld3d_shared::log_once_warn!(target: LOG_TARGET, "cursor: followed layer lost its device");
            return false;
        };
        if self.queue.device().registryID() != device.registryID() {
            let Some(queue) = device.newCommandQueue() else {
                mtld3d_shared::log_once_warn!(target: LOG_TARGET, "cursor: handoff queue allocation failed");
                return false;
            };
            queue.setLabel(Some(&NSString::from_str("mtld3d-cursor-queue")));
            self.queue = queue;
            self.layer.setDevice(Some(&device));
            self.textures.clear();
            self.content.invalidate();
        }
        let mode = if att.hdr_active() {
            LayerMode::Hdr
        } else {
            LayerMode::Sdr
        };
        let colorspace = game.colorspace();
        let previous_colorspace = self.layer.colorspace();
        let current = LayerConfiguration {
            format: self.layer.pixelFormat(),
            colorspace: previous_colorspace.as_deref(),
            edr: self.layer.wantsExtendedDynamicRangeContent(),
        };
        let next = LayerConfiguration {
            format: game.pixelFormat(),
            colorspace: colorspace.as_deref(),
            edr: game.wantsExtendedDynamicRangeContent(),
        };
        if self.mode != Some(mode) || current != next {
            self.layer.setPixelFormat(game.pixelFormat());
            self.layer.setColorspace(colorspace.as_deref());
            self.layer
                .setWantsExtendedDynamicRangeContent(game.wantsExtendedDynamicRangeContent());
            self.mode = Some(mode);
            self.content.invalidate();
            info!(target: LOG_TARGET, "cursor: layer configured {mode:?} pixelFormat={:?} colorspace={colorspace:?} EDR={}",
                game.pixelFormat(), game.wantsExtendedDynamicRangeContent());
        }
        true
    }

    /// Schedule the requested pixels, leaving failed attempts pending for an existing opportunity.
    fn ensure_content(&mut self, wanted: Content, sprite: Option<&Sprite>) -> bool {
        let Self {
            layer,
            queue,
            textures,
            content,
            ..
        } = self;
        content.ensure(wanted, |requested, generation, result| {
            let draw = match requested {
                Content::Transparent => Self::present_transparent(layer, queue),
                Content::Sprite { .. } => sprite.and_then(|sprite| {
                    Self::render(layer, queue, textures, sprite, requested)
                }),
            };
            let Some(draw) = draw else {
                mtld3d_shared::log_once_warn!(target: LOG_TARGET, "cursor: draw preparation failed; latest request remains pending");
                return false;
            };
            present_with_transaction(&draw.command, &draw.drawable, generation, result)
        })
    }

    /// Encode a transparent drawable. A refused clear is not a hidden cursor.
    fn present_transparent(
        layer: &CAMetalLayer,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Option<CursorDraw> {
        let Some(drawable) = layer.nextDrawable() else {
            mtld3d_shared::log_once_warn!(target: LOG_TARGET, "cursor: clear nextDrawable returned nil");
            return None;
        };
        let Some(command) = queue.commandBuffer() else {
            mtld3d_shared::log_once_warn!(target: LOG_TARGET, "cursor: clear command buffer allocation failed");
            return None;
        };
        command.setLabel(Some(&NSString::from_str("mtld3d-cursor-clear")));
        if !command::clear_cursor_drawable(&command, &drawable.texture()) {
            return None;
        }
        Some(CursorDraw { command, drawable })
    }

    /// Render sprite `hash` into the overlay's drawable, sized to the sprite.
    ///
    /// A failed allocation or encode leaves no cached submission, so an existing
    /// input, run-loop or present opportunity retries the latest request.
    fn render(
        layer: &CAMetalLayer,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
        textures: &mut FxHashMap<u64, Retained<ProtocolObject<dyn MTLTexture>>>,
        sprite: &Sprite,
        content: &Content,
    ) -> Option<CursorDraw> {
        let Content::Sprite {
            hash,
            geometry,
            mode,
            peak,
        } = content
        else {
            mtld3d_shared::log_once_warn!(target: LOG_TARGET, "cursor: sprite renderer received transparent content");
            return None;
        };
        let device = queue.device();
        let texture = match textures.entry(*hash) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(upload_sprite_texture(&device, *hash, sprite)?),
        };
        layer.setBounds(CGRect {
            origin: CGPoint { x: 0.0, y: 0.0 },
            size: CGSize {
                width: geometry.width,
                height: geometry.height,
            },
        });
        layer.setContentsScale(geometry.scale);
        layer.setDrawableSize(CGSize {
            width: geometry.width * geometry.scale,
            height: geometry.height * geometry.scale,
        });
        let Some(drawable) = layer.nextDrawable() else {
            mtld3d_shared::log_once_warn!(target: LOG_TARGET, "cursor: sprite nextDrawable returned nil");
            return None;
        };
        let Some(command) = queue.commandBuffer() else {
            mtld3d_shared::log_once_warn!(target: LOG_TARGET, "cursor: sprite command buffer allocation failed");
            return None;
        };
        command.setLabel(Some(&NSString::from_str("mtld3d-cursor")));
        let Some(pipelines) = present::ensure_resources(&device) else {
            mtld3d_shared::log_once_warn!(target: LOG_TARGET, "cursor: pipeline allocation failed");
            return None;
        };
        let (pipeline, uniforms) = match mode {
            LayerMode::Sdr => (pipelines.cursor_copy, None),
            LayerMode::Hdr if *peak <= 1.0 => (pipelines.cursor_passthrough, None),
            LayerMode::Hdr => (pipelines.cursor_bt2446, Some(present::hdr_uniforms(*peak))),
        };
        if !command::encode_cursor_pass(&command, texture, &drawable.texture(), pipeline, uniforms)
        {
            mtld3d_shared::log_once_warn!(target: LOG_TARGET, "cursor: sprite encoder allocation failed");
            return None;
        }
        Some(CursorDraw { command, drawable })
    }

    /// Level, position and content against the pointer and the game window as they are now.
    fn sync_position(&mut self, mtm: MainThreadMarker, wanted: &WantedSnapshot, captured: bool) {
        let Some(att) = wanted.owner.as_ref() else {
            self.ensure_content(Content::Transparent, None);
            return;
        };
        let mut inputs = VisibilityInputs::empty();
        inputs.set(
            VisibilityInputs::WANTED,
            wanted.flags.contains(CursorOverlayFlags::VISIBLE) && wanted.sprite.is_some(),
        );
        inputs.set(
            VisibilityInputs::APP_ACTIVE,
            NSApplication::sharedApplication(mtm).isActive(),
        );
        inputs.set(VisibilityInputs::CAPTURED, captured);
        if !inputs.contains(VisibilityInputs::WANTED | VisibilityInputs::APP_ACTIVE) {
            // Transparent content has no position to follow. Avoid window,
            // screen and pointer queries until a sprite can be shown again;
            // that apply resolves its pixels and position in one transaction.
            self.update_visibility(inputs);
            self.ensure_content(Content::Transparent, None);
            return;
        }
        let game = attachment::retain_view(att, mtm)
            .and_then(|view| view.window().map(|window| (view, window)));
        let Some((view, game_window)) = game else {
            self.ensure_content(Content::Transparent, None);
            return;
        };
        // Wine re-levels its windows across fullscreen transitions; stay one
        // above whatever the game window is at right now.
        let level = game_window.level() + 1;
        if self.window.level() != level {
            self.window.setLevel(level);
        }
        // Follow the game window onto another screen. A window frame change
        // costs one cursor re-resolution by AppKit, which is why it is done
        // only here and never per event.
        let frame = overlay_frame(mtm, Some(att));
        if self.window.frame() != frame {
            self.window.setFrame_display(frame, false);
            info!(
                target: LOG_TARGET,
                "cursor: overlay window moved over ({:.0},{:.0}) {:.0}x{:.0}",
                frame.origin.x, frame.origin.y, frame.size.width, frame.size.height,
            );
        }
        let mouse = NSEvent::mouseLocation();
        let client = game_window.convertRectToScreen(view.convertRect_toView(view.bounds(), None));
        inputs.set(
            VisibilityInputs::POINTER_INSIDE,
            rect_contains(client, mouse)
                && window_under_pointer(mouse, mtm) == game_window.windowNumber(),
        );
        inputs.set(VisibilityInputs::OCCLUDED, att.window_occluded());
        inputs.set(VisibilityInputs::MINIATURIZED, game_window.isMiniaturized());
        self.update_visibility(inputs);
        let shown = overlay_visible(inputs);
        let Some(sprite) = wanted.sprite.as_deref() else {
            self.ensure_content(Content::Transparent, None);
            return;
        };
        let hash = wanted.hash;
        let geometry = SpriteGeometry::of(sprite, att.backing_scale());
        let local = self.window.convertPointFromScreen(mouse);
        let (origin_x, origin_y) = sprite_origin((local.x, local.y), &geometry);
        let content = if shown {
            let mode = self.mode.unwrap_or(LayerMode::Sdr);
            let peak = att.headroom();
            // Re-render on a sprite or layer-mode change, and on a headroom
            // move worth it; otherwise the drawable already shows this sprite.
            let peak = match self.content.current() {
                Some(Content::Sprite {
                    hash: h,
                    mode: m,
                    peak: p,
                    ..
                }) if *h == hash && *m == mode && !peak_changed(*p, peak) => *p,
                _ => peak,
            };
            Content::Sprite {
                hash,
                mode,
                peak,
                geometry,
            }
        } else {
            Content::Transparent
        };
        // Pixels first, the position second, and the position only once the
        // drawable shows the wanted pixels. Both are part of the run loop
        // iteration's one transaction (the layer presents with it), so the
        // compositor sees a hide and the move that comes with it in the same
        // frame, never the old sprite at the new place; and a hide whose
        // present found no drawable keeps the old sprite where it is until
        // the retry, rather than moving it. A show resolves the current pointer
        // before this write, so hidden or inactive sprites need no position updates.
        if self.ensure_content(content, Some(sprite))
            && inputs.contains(VisibilityInputs::POINTER_INSIDE)
        {
            self.layer.setPosition(CGPoint {
                x: origin_x,
                y: origin_y,
            });
        }
    }

    fn update_visibility(&mut self, inputs: VisibilityInputs) {
        if overlay_visible(inputs) {
            let activation = POINTER_WATCH.with_borrow(|watch| watch.activation);
            if self.cursor_activation != Some(activation) {
                self.native_cursor.set();
                self.cursor_activation = Some(activation);
                debug!(target: LOG_TARGET, "cursor: native blank applied for activation={activation}");
            }
        }
        if self.visibility != Some(inputs) {
            debug!(target: LOG_TARGET, "cursor: visibility inputs={inputs:?}");
            self.visibility = Some(inputs);
        }
    }
}

/// A native blank for startup and activation without pointer motion.
///
/// Win32 may already hold our blank HCURSOR while Wine has no cursor window to
/// notify or macOS still displays the previous application's image. Select this
/// transparent image once per activation when the overlay can be shown. Ordinary
/// mouse input and hide/show cycles then leave native cursor handling to Wine,
/// without changing the system cursor's hide count or synthesizing mouse input.
fn native_blank_cursor(_mtm: MainThreadMarker) -> Option<Retained<NSCursor>> {
    // SAFETY: immutable AppKit constant, initialized before the main thread starts.
    let color_space = unsafe { NSDeviceRGBColorSpace };
    // SAFETY: null planes asks AppKit to allocate storage for one RGBA pixel.
    let bitmap = unsafe {
        NSBitmapImageRep::initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel(
            NSBitmapImageRep::alloc(),
            core::ptr::null_mut(),
            1,
            1,
            8,
            4,
            true,
            false,
            color_space,
            4,
            32,
        )
    };
    let Some(bitmap) = bitmap else {
        mtld3d_shared::log_once_warn!(target: LOG_TARGET, "cursor: native blank allocation failed");
        return None;
    };
    bitmap.setColor_atX_y(&NSColor::clearColor(), 0, 0);
    let image = NSImage::initWithSize(
        NSImage::alloc(),
        CGSize {
            width: 1.0,
            height: 1.0,
        },
    );
    image.addRepresentation(&bitmap);
    Some(NSCursor::initWithImage_hotSpot(
        NSCursor::alloc(),
        &image,
        CGPoint { x: 0.0, y: 0.0 },
    ))
}

/// Encoded native work, with no references to caller-owned pixels or PE state.
struct CursorDraw {
    command: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    drawable: Retained<ProtocolObject<dyn CAMetalDrawable>>,
}

/// Commit `cmd_buf` and present `drawable` as part of the current Core Animation transaction.
///
/// The sequence `presentsWithTransaction` asks for: the command buffer must
/// be scheduled before the drawable is handed to the transaction, and the
/// transaction is the run loop iteration's implicit one, committed once the
/// iteration ends with every layer write made meanwhile. The wait is for
/// scheduling only, on the main thread, for a pass that draws one sprite.
fn present_with_transaction(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    drawable: &ProtocolObject<dyn CAMetalDrawable>,
    generation: u64,
    result: Arc<AtomicU8>,
) -> bool {
    let handler = RcBlock::new(move |ptr: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
        autoreleasepool(|_| {
            // SAFETY: Metal supplies a live completed buffer for this invocation.
            let buffer = unsafe { ptr.as_ref() };
            let completed = buffer.status() == MTLCommandBufferStatus::Completed;
            result.store(
                if completed { COMPLETED } else { FAILED },
                Ordering::Release,
            );
            if completed {
                debug!(target: LOG_TARGET, "cursor: completed generation={generation}");
            } else {
                log::warn!(target: LOG_TARGET, "cursor: completion failed generation={generation} status={:?} error={:?}", buffer.status(), buffer.error());
            }
        });
    });
    // SAFETY: Metal copies the block before commit. It retains only its own result
    // cell and generation, never an attachment, PE sink, sprite, or native UI object.
    // As for the present callbacks, device use pins this Unix image until process exit.
    unsafe { cmd_buf.addCompletedHandler(RcBlock::as_ptr(&handler)) };
    cmd_buf.commit();
    cmd_buf.waitUntilScheduled();
    if cmd_buf.status() == MTLCommandBufferStatus::Error {
        log::warn!(target: LOG_TARGET, "cursor: scheduling failed generation={generation}");
        return false;
    }
    drawable.present();
    true
}

/// An actions table that switches implicit animations off for everything the overlay writes.
fn no_actions() -> Retained<NSDictionary<NSString, ProtocolObject<dyn CAAction>>> {
    let null = NSNull::null();
    let none: &ProtocolObject<dyn CAAction> = ProtocolObject::from_ref(&*null);
    let keys = [
        NSString::from_str("position"),
        NSString::from_str("bounds"),
        NSString::from_str("contentsScale"),
        NSString::from_str("contents"),
    ];
    NSDictionary::from_slices(
        &[&*keys[0], &*keys[1], &*keys[2], &*keys[3]],
        &[none, none, none, none],
    )
}

/// Whether winemac warped the pointer since the last warp the observer acted on.
///
/// winemac reports `0` once an event newer than its warp arrived, which the
/// pointer watch handled as that event; only a fresh warp time is a move
/// nothing else told us about. Compared bit for bit: `followed` is a copy of
/// an earlier `now`, never a computed value.
const fn warp_since(followed: f64, now: f64) -> bool {
    now != 0.0 && now.to_bits() != followed.to_bits()
}

/// Observe Wine-consumed input and warps before Core Animation commits the transaction.
extern "C-unwind" fn follow_warp(
    _observer: *mut CFRunLoopObserver,
    _activity: CFRunLoopActivity,
    _info: *mut core::ffi::c_void,
) {
    autoreleasepool(|_| apply_on_main_inner());
}

/// The number of the window a click at `point` would land on, in any application.
///
/// The overlay ignores mouse events, so it is never the answer; the game
/// window is, unless something sits over it there.
fn window_under_pointer(point: CGPoint, mtm: MainThreadMarker) -> NSInteger {
    NSWindow::windowNumberAtPoint_belowWindowWithWindowNumber(point, 0, mtm)
}

/// The screen the game window is on; the main screen when it is on none or none is followed.
fn game_screen(
    mtm: MainThreadMarker,
    owner: Option<&Arc<Attachment>>,
) -> Option<Retained<NSScreen>> {
    owner
        .and_then(|att| attachment::retain_view(att, mtm))
        .and_then(|view| view.window())
        .and_then(|window| window.screen())
        .or_else(|| NSScreen::mainScreen(mtm))
}

/// The frame the overlay window covers: the screen the game window is on.
///
/// The main screen when the game window is not on any (mid-move between
/// displays) or no attachment is followed; the window follows on the next event.
fn overlay_frame(mtm: MainThreadMarker, owner: Option<&Arc<Attachment>>) -> CGRect {
    game_screen(mtm, owner).map_or(
        CGRect {
            origin: CGPoint { x: 0.0, y: 0.0 },
            size: CGSize {
                width: 1.0,
                height: 1.0,
            },
        },
        |screen| screen.frame(),
    )
}

/// Upload the owned snapshot's sprite outside the shared mutex.
///
/// The Arc snapshot keeps the tight pixels alive through replaceRegion without
/// blocking API updates while Metal allocates or copies the texture.
fn upload_sprite_texture(
    device: &ProtocolObject<dyn MTLDevice>,
    hash: u64,
    sprite: &Sprite,
) -> Option<Retained<ProtocolObject<dyn MTLTexture>>> {
    let desc = MTLTextureDescriptor::new();
    desc.setTextureType(MTLTextureType::Type2D);
    // The bytes are D3D9 A8R8G8B8, which is B, G, R, A in memory.
    desc.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
    // SAFETY: plain property setter on a fresh descriptor.
    unsafe { desc.setWidth(sprite.width as usize) };
    // SAFETY: plain property setter on a fresh descriptor.
    unsafe { desc.setHeight(sprite.height as usize) };
    desc.setUsage(MTLTextureUsage::ShaderRead);
    desc.setStorageMode(cpu_written_texture_storage(device));
    let Some(texture) = device.newTextureWithDescriptor(&desc) else {
        mtld3d_shared::log_once_warn!(target: LOG_TARGET, "cursor: sprite texture allocation failed");
        return None;
    };
    texture.setLabel(Some(&NSString::from_str(&format!(
        "mtld3d-cursor-sprite-{hash:#x}"
    ))));
    let region = MTLRegion {
        origin: MTLOrigin { x: 0, y: 0, z: 0 },
        size: MTLSize {
            width: sprite.width as usize,
            height: sprite.height as usize,
            depth: 1,
        },
    };
    let bytes_per_row = sprite.width as usize * 4;
    let pixels = NonNull::from(&*sprite.pixels).cast::<core::ffi::c_void>();
    // SAFETY: the handler checked `pixels.len() == width * height * 4`, so the
    // rows described by `bytes_per_row` over `region` lie inside the buffer,
    // and the texture was just created at exactly that extent.
    unsafe {
        texture.replaceRegion_mipmapLevel_slice_withBytes_bytesPerRow_bytesPerImage(
            region,
            0,
            0,
            pixels,
            bytes_per_row,
            sprite.pixels.len(),
        );
    }
    debug!(
        target: LOG_TARGET,
        "cursor: sprite {hash:#018x} uploaded ({}x{} px, upscaled {}x)",
        sprite.width, sprite.height, sprite.scale,
    );
    Some(texture)
}

/// Mouse moves and button transitions used by both observation routes.
fn mouse_mask() -> NSEventMask {
    NSEventMask::MouseMoved
        | NSEventMask::LeftMouseDragged
        | NSEventMask::RightMouseDragged
        | NSEventMask::OtherMouseDragged
        | NSEventMask::LeftMouseDown
        | NSEventMask::LeftMouseUp
        | NSEventMask::RightMouseDown
        | NSEventMask::RightMouseUp
        | NSEventMask::OtherMouseDown
        | NSEventMask::OtherMouseUp
}

/// Install process-lifetime monitors; retry only components whose creation failed.
///
/// Wine can consume captured events before calling `AppKit`'s `sendEvent`, bypassing
/// the local monitor. The existing run-loop observer also reads currentEvent,
/// which Wine's dequeue already updates. Both routes share one deduplicator.
pub fn install_pointer_watch(mtm: MainThreadMarker) {
    let mut installed = POINTER_WATCH_INSTALLED.get();
    if !installed.contains(WatchInstalled::MONITOR) {
        let monitor = RcBlock::new(|event: NonNull<NSEvent>| -> *mut NSEvent {
            autoreleasepool(|_| {
                let mtm =
                    MainThreadMarker::new().expect("cursor local monitor runs on the main thread");
                // SAFETY: AppKit passes the live event for this monitor invocation.
                observe_mouse_event(unsafe { event.as_ref() }, "local", mtm);
                event.as_ptr()
            })
        });
        // SAFETY: AppKit copies this block; the token is retained for process lifetime.
        let token = unsafe {
            NSEvent::addLocalMonitorForEventsMatchingMask_handler(mouse_mask(), &monitor)
        };
        if let Some(token) = token {
            core::mem::forget(token);
            installed.insert(WatchInstalled::MONITOR);
        } else {
            mtld3d_shared::log_once_warn!(target: LOG_TARGET, "cursor: local monitor installation failed");
        }
    }
    if !installed.contains(WatchInstalled::RUN_LOOP) {
        // Order 0 runs before Core Animation's commit observer (order 2 000 000).
        let activities = CFRunLoopActivity::BeforeWaiting | CFRunLoopActivity::Exit;
        // SAFETY: the main-loop callback uses no context and touches UI only on main.
        let observer = unsafe {
            CFRunLoopObserver::new(
                None,
                activities.0,
                true,
                0,
                Some(follow_warp),
                core::ptr::null_mut(),
            )
        };
        // SAFETY: immutable CoreFoundation constant, initialized before main.
        let common_modes = unsafe { kCFRunLoopCommonModes };
        if let (Some(observer), Some(main_loop)) = (observer, CFRunLoop::main()) {
            main_loop.add_observer(Some(&observer), common_modes);
            core::mem::forget(observer);
            installed.insert(WatchInstalled::RUN_LOOP);
        } else {
            mtld3d_shared::log_once_warn!(target: LOG_TARGET, "cursor: run-loop observer installation failed");
        }
    }
    if !installed.contains(WatchInstalled::ACTIVATION) {
        install_activation_watch(mtm);
        installed.insert(WatchInstalled::ACTIVATION);
    }
    POINTER_WATCH_INSTALLED.set(installed);
}

fn install_activation_watch(_mtm: MainThreadMarker) {
    let center = NSNotificationCenter::defaultCenter();
    // SAFETY: immutable AppKit notification names, initialized before main.
    let names = unsafe {
        [
            NSApplicationDidBecomeActiveNotification,
            NSApplicationDidResignActiveNotification,
        ]
    };
    for name in names {
        let block = RcBlock::new(|_: NonNull<NSNotification>| {
            autoreleasepool(|_| {
                let mtm = MainThreadMarker::new()
                    .expect("cursor activation observer runs on the main thread");
                let active = NSApplication::sharedApplication(mtm).isActive();
                let recovered = POINTER_WATCH.with_borrow_mut(|watch| {
                    if active {
                        watch.activation = watch.activation.wrapping_add(1);
                    }
                    watch
                        .input
                        .note_position(NSEvent::mouseLocation(), now_ns());
                    core::mem::replace(&mut watch.input.captured, false)
                });
                if recovered {
                    attachment::request_cursor_kick_all();
                }
            });
        });
        // SAFETY: the notification center copies the block; the token is kept for life.
        let token = unsafe {
            center.addObserverForName_object_queue_usingBlock(Some(name), None, None, &block)
        };
        core::mem::forget(token);
    }
}

#[cfg(test)]
mod tests;
