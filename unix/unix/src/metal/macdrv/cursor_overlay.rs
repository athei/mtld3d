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
//! queues one main-thread apply, coalesced through [`APPLY_PENDING`] so a burst
//! of clicks is one apply of the latest state. Everything that touches
//! `AppKit`, Core Animation or the overlay's Metal objects runs on the main
//! thread: the apply, the mouse-move monitor that repositions the window, the
//! activation observers, and the display reconciliation that re-renders the
//! sprite when the layer mode or the EDR headroom moved. Those objects live in
//! a main-thread `thread_local`, which is what makes the split sound without a
//! lock around `Retained` handles.
//!
//! Nothing about the game window is latched: the game `NSWindow`, its level,
//! its client rectangle and its screen are read from the bound view at every
//! event, so in-game resolution changes, windowed/fullscreen switches and
//! display moves need no signal from the PE side.
//!
//! Two things the pointer can do without telling this process, both handled
//! here. A system tool that takes the pointer (the interactive screenshot
//! crosshair) delivers no mouse events to the application while the pointer
//! keeps moving; every present, and a timer for when presents stop, notices
//! the pointer moving with no events arriving and hides the sprite until
//! events resume. And when such a tool
//! ends, the window server shows the standard arrow rather than the cursor
//! Wine set, which Wine never re-applies because its handle did not change;
//! the first event after a capture asks the PE side for its null-then-set
//! kick, which makes Wine re-apply through a handle change. That pointer
//! watch serves the hardware cursor too, so it is installed at attach for
//! every device, overlay or not.

use core::{
    cell::{Cell, RefCell},
    ptr::NonNull,
};
use std::{
    collections::hash_map::Entry,
    sync::{
        LazyLock, Mutex, MutexGuard,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::Instant,
};

use block2::RcBlock;
use log::{debug, info};
use mtld3d_shared::{
    SetCursorOverlayParams,
    mtl::{ColorSpacePolicy, CursorOverlayFlags},
};
use objc2::{MainThreadMarker, MainThreadOnly, rc::Retained, runtime::ProtocolObject};
use objc2_app_kit::{
    NSApplication, NSApplicationDidBecomeActiveNotification,
    NSApplicationDidResignActiveNotification, NSBackingStoreType, NSColor, NSEvent, NSEventMask,
    NSScreen, NSView, NSWindow, NSWindowAnimationBehavior, NSWindowCollectionBehavior,
    NSWindowStyleMask,
};
use objc2_core_foundation::{CGPoint, CGRect, CGSize};
use objc2_foundation::{
    NSDictionary, NSNotification, NSNotificationCenter, NSRunLoop, NSRunLoopCommonModes, NSString,
    NSTimer,
};
use objc2_metal::{
    MTLCommandBuffer, MTLCommandQueue, MTLDevice, MTLOrigin, MTLPixelFormat, MTLRegion,
    MTLResource, MTLSize, MTLStorageMode, MTLTexture, MTLTextureDescriptor, MTLTextureType,
    MTLTextureUsage,
};
use objc2_quartz_core::{CALayer, CAMetalDrawable, CAMetalLayer, CATransaction};
use rustc_hash::FxHashMap;

use super::{
    COLOR_SPACE_POLICY, CURRENT_HEADROOM_BITS, HDR_ACTIVE, LayerColorRefs, LayerMode,
    apply_layer_color, request_cursor_kick, retain_bound_layer, retain_bound_view,
    run_on_main_thread_async, screen_color_profile, window_occluded,
};
use crate::metal::{command, present};

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
#[derive(Clone, Copy, Debug, PartialEq)]
enum Content {
    /// Nothing yet, or the last present was a transparent clear.
    Transparent,
    /// A sprite, tone-mapped for a layer mode and a headroom.
    Sprite {
        hash: u64,
        mode: LayerMode,
        peak: f32,
    },
}

/// How often the timer checks the pointer for moving without events reaching us, seconds.
///
/// The timer is the fallback for a game that has stopped presenting; while
/// it presents, every frame runs the same check for free. Ten times a
/// second, with a tolerance so the system folds it into other wakeups.
const CAPTURE_TICK_SECONDS: f64 = 0.1;

/// How long without a mouse event before a moved pointer counts as captured.
///
/// A moving pointer delivers an event every few milliseconds, so a gap this
/// long with the pointer elsewhere than the last event put it means someone
/// else is receiving the events. Checked every present, so this is also the
/// delay before the sprite goes away.
const CAPTURE_SILENCE_MS: u128 = 60;

/// When the last mouse event reached this process, nanoseconds since [`EPOCH`].
///
/// Written by the pointer watch on the main thread, read by the capture
/// checks on the main and submit threads. `u64::MAX` until the first event.
static LAST_EVENT_NS: AtomicU64 = AtomicU64::new(u64::MAX);

/// Where the pointer was at the previous capture check, packed like [`LAST_EVENT_POS`].
///
/// A capture has the pointer moving between checks; a warp the game asked
/// for (`SetCursorPos` after a mouselook drag) is one jump followed by a
/// still pointer, and generates no event either. Comparing consecutive
/// checks is what tells them apart.
static LAST_CHECK_POS: AtomicU64 = AtomicU64::new(0);

/// Where the pointer was at the last mouse event: `x` bits high, `y` bits low.
///
/// The two coordinates are `f32` so one `u64` carries both and a reader
/// never sees an `x` from one event next to a `y` from another.
static LAST_EVENT_POS: AtomicU64 = AtomicU64::new(0);

/// The pointer is moving without events reaching us: another process has it.
///
/// Set by whichever capture check notices first, cleared by the next event
/// and whenever the game hides its cursor.
static CAPTURED: AtomicBool = AtomicBool::new(false);

/// The game shows its cursor, in either cursor mode.
///
/// The capture checks run only while it does: a game that hides its cursor
/// owns the pointer for as long as it likes (mouselook, raw input) and there
/// is no cursor of ours or Wine's on screen for another process to replace,
/// so a pointer moving with no events reaching us means nothing then.
static CURSOR_SHOWN: AtomicBool = AtomicBool::new(false);

/// The instant [`LAST_EVENT_NS`] counts from.
static EPOCH: LazyLock<Instant> = LazyLock::new(Instant::now);

fn pack_point(point: CGPoint) -> u64 {
    // Screen coordinates fit `f32` with room to spare; the lost fraction is
    // far below the half-point movement threshold.
    let x = super::bounded_cast::f64_to_f32(point.x).to_bits();
    let y = super::bounded_cast::f64_to_f32(point.y).to_bits();
    (u64::from(x) << 32) | u64::from(y)
}

fn unpack_point(bits: u64) -> (f64, f64) {
    let x = u32::try_from(bits >> 32).expect("the high word is 32 bits");
    let y = u32::try_from(bits & u64::from(u32::MAX)).expect("masked to 32 bits");
    (f64::from(f32::from_bits(x)), f64::from(f32::from_bits(y)))
}

/// Record a mouse event's time and pointer position for the capture checks.
fn note_event(position: CGPoint) {
    let ns = u64::try_from(EPOCH.elapsed().as_nanos()).unwrap_or(u64::MAX - 1);
    LAST_EVENT_POS.store(pack_point(position), Ordering::Relaxed);
    LAST_EVENT_NS.store(ns, Ordering::Release);
}

/// Whether the pointer has moved since the last event with no event for the silence period.
///
/// Callable from any thread: the answer is the whole message, and a stale
/// read only delays it by one check.
fn capture_suspected() -> bool {
    if !CURSOR_SHOWN.load(Ordering::Relaxed) {
        return false;
    }
    let at = LAST_EVENT_NS.load(Ordering::Acquire);
    if at == u64::MAX {
        return false;
    }
    let silence_ms = EPOCH.elapsed().as_nanos().saturating_sub(u128::from(at)) / 1_000_000;
    let now = NSEvent::mouseLocation();
    let now_bits = pack_point(now);
    let (seen_x, seen_y) = unpack_point(LAST_EVENT_POS.load(Ordering::Relaxed));
    let (prev_x, prev_y) = unpack_point(LAST_CHECK_POS.swap(now_bits, Ordering::Relaxed));
    let moved_since_event = (now.x - seen_x).abs() > 0.5 || (now.y - seen_y).abs() > 0.5;
    let moved_since_check = (now.x - prev_x).abs() > 0.5 || (now.y - prev_y).abs() > 0.5;
    pointer_captured(silence_ms, moved_since_event, moved_since_check)
}

/// Run the capture check from the present path; no wakeup of its own.
///
/// Called once per present on the submit thread. When it is the first to
/// notice, it flags the capture and asks the main thread to hide the sprite;
/// the timer does the same for a game that has stopped presenting.
pub fn poll_capture_from_present() {
    if CAPTURED.load(Ordering::Relaxed) || !capture_suspected() {
        return;
    }
    if CAPTURED
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_ok()
    {
        debug!(target: LOG_TARGET, "cursor: pointer moves without events, another process has it");
        run_on_main_thread_async(sync_overlay_on_main);
    }
}

/// Whether the pointer is moving while no mouse event reaches this process.
///
/// A system tool that takes the pointer (the screenshot crosshair) leaves the
/// application eventless while the pointer keeps moving. A pointer that sits
/// away from the last event but no longer moves was warped there by the game
/// (`SetCursorPos`), which generates no event either and must not hide the
/// cursor. Only asked while the game shows its cursor; with it hidden the
/// game owns the pointer and Wine may keep the events from the application.
const fn pointer_captured(
    silence_ms: u128,
    moved_since_event: bool,
    moved_since_check: bool,
) -> bool {
    moved_since_event && moved_since_check && silence_ms >= CAPTURE_SILENCE_MS
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

/// What the PE side last asked for.
struct Wanted {
    /// Sprite identity, `0` = none (also what detach leaves behind).
    hash: u64,
    visible: bool,
}

/// State written by the thunk on the API thread and read by the main thread.
struct Shared {
    /// Every sprite the PE side has handed over, by hash; never evicted.
    ///
    /// Mirrors the PE side's own uploaded set, so a hash arriving without
    /// pixels always names something here.
    sprites: FxHashMap<u64, Sprite>,
    wanted: Wanted,
}

static SHARED: LazyLock<Mutex<Shared>> = LazyLock::new(|| {
    Mutex::new(Shared {
        sprites: FxHashMap::default(),
        wanted: Wanted {
            hash: 0,
            visible: false,
        },
    })
});

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
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct SpriteGeometry {
    width: f64,
    height: f64,
    hotspot_x: f64,
    hotspot_y: f64,
}

impl SpriteGeometry {
    fn of(sprite: &Sprite) -> Self {
        let scale = f64::from(sprite.scale.max(1));
        Self {
            width: f64::from(sprite.width) / scale,
            height: f64::from(sprite.height) / scale,
            hotspot_x: f64::from(sprite.x_hotspot) / scale,
            hotspot_y: f64::from(sprite.y_hotspot) / scale,
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
        /// The pointer is over the game's client area.
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
const fn sprite_origin(mouse: (f64, f64), geometry: SpriteGeometry) -> (f64, f64) {
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
/// on the main thread.
pub fn set_cursor_overlay(params: &SetCursorOverlayParams, pixels: Option<&[u8]>) {
    let visible = params.flags.contains(CursorOverlayFlags::VISIBLE);
    CURSOR_SHOWN.store(visible, Ordering::Relaxed);
    if !visible {
        // The game took the pointer back; whatever another process did with
        // it meanwhile is over as far as the cursor on screen is concerned.
        CAPTURED.store(false, Ordering::Relaxed);
    }
    if params.flags.contains(CursorOverlayFlags::HARDWARE) {
        // Visibility only: the hardware cursor path has no sprite.
        return;
    }
    {
        let mut shared = lock_shared();
        if let Some(pixels) = pixels {
            shared.sprites.insert(
                params.hash,
                Sprite {
                    width: params.width,
                    height: params.height,
                    x_hotspot: params.x_hotspot,
                    y_hotspot: params.y_hotspot,
                    scale: params.scale,
                    pixels: pixels.into(),
                },
            );
        }
        shared.wanted = Wanted {
            hash: params.hash,
            visible,
        };
    }
    queue_apply();
}

/// The bound device is going away: hide the sprite and forget every sprite it uploaded.
///
/// The window and its observers stay for the process lifetime like the other
/// `AppKit` observers; the next device's PE side has an empty uploaded set of
/// its own and re-sends what it shows.
pub fn detach() {
    CURSOR_SHOWN.store(false, Ordering::Relaxed);
    CAPTURED.store(false, Ordering::Relaxed);
    {
        let mut shared = lock_shared();
        shared.sprites.clear();
        shared.wanted = Wanted {
            hash: 0,
            visible: false,
        };
    }
    queue_apply();
}

/// Re-apply against the layer mode and headroom the display reconciliation just refreshed.
///
/// **Main thread only.** Called at the end of the headroom refresh, so a game
/// window that moved onto a display of the other class, or whose EDR headroom
/// drifted, gets a sprite rendered for what the frame now looks like.
pub fn reconcile_on_main() {
    apply_on_main_inner(false);
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

/// The queued apply: creates the overlay on first use. **Main thread only.**
fn apply_on_main() {
    APPLY_PENDING.store(false, Ordering::Release);
    let queued_ns = APPLY_QUEUED_NS.load(Ordering::Relaxed);
    let waited_us = EPOCH
        .elapsed()
        .as_nanos()
        .saturating_sub(u128::from(queued_ns))
        / 1_000;
    debug!(target: LOG_TARGET, "cursor: apply ran {waited_us} us after it was queued");
    apply_on_main_inner(true);
}

/// A snapshot of the wanted state with the sprite's geometry resolved.
#[derive(Clone, Copy)]
struct WantedSnapshot {
    hash: u64,
    visible: bool,
    /// `None` when the hash names no sprite the unix side holds.
    geometry: Option<SpriteGeometry>,
}

fn snapshot_wanted() -> WantedSnapshot {
    let shared = lock_shared();
    WantedSnapshot {
        hash: shared.wanted.hash,
        visible: shared.wanted.visible,
        geometry: shared
            .sprites
            .get(&shared.wanted.hash)
            .map(SpriteGeometry::of),
    }
}

thread_local! {
    /// Whether the pointer watch has been installed.
    static POINTER_WATCH_INSTALLED: Cell<bool> = const { Cell::new(false) };
    /// The overlay's `AppKit` and Metal objects; main thread only, by construction.
    ///
    /// Every access is from a block dispatched to the main queue or from an
    /// `AppKit` callback, so the `RefCell` is never contended across threads;
    /// `try_borrow_mut` guards the one re-entrancy `AppKit` could produce.
    static OVERLAY: RefCell<Option<Overlay>> = const { RefCell::new(None) };
}

fn apply_on_main_inner(create_if_missing: bool) {
    // SAFETY: every caller runs on the main thread (a main-queue block or an
    // AppKit callback), where NSWindow and NSView access is valid.
    let mtm = unsafe { MainThreadMarker::new_unchecked() };
    let wanted = snapshot_wanted();
    OVERLAY.with(|cell| {
        let Ok(mut slot) = cell.try_borrow_mut() else {
            return;
        };
        if slot.is_none() {
            if !create_if_missing || wanted.hash == 0 {
                return;
            }
            *slot = Overlay::create(mtm);
        }
        if let Some(overlay) = slot.as_mut() {
            overlay.apply(mtm, wanted);
        }
    });
}

/// Mouse moved or dragged, or the application (de)activated. **Main thread only.**
///
/// Runs for every device whatever its cursor mode: the end of a capture is
/// what puts the hardware cursor back after a system tool borrowed the pointer.
fn on_pointer_event_main() {
    note_event(NSEvent::mouseLocation());
    if CAPTURED.swap(false, Ordering::AcqRel) {
        // The other process's cursor is still on screen; Wine only replaces
        // it on a handle change, which the PE side's kick provides.
        request_cursor_kick();
        debug!(
            target: LOG_TARGET,
            "cursor: mouse events resumed, pointer released, cursor re-apply requested",
        );
    }
    sync_overlay_on_main();
}

/// Bring the overlay, if there is one, in line with the pointer. **Main thread only.**
fn sync_overlay_on_main() {
    // SAFETY: every caller runs on the main thread (a main-queue block or an
    // AppKit callback).
    let mtm = unsafe { MainThreadMarker::new_unchecked() };
    OVERLAY.with(|cell| {
        let Ok(mut slot) = cell.try_borrow_mut() else {
            return;
        };
        if let Some(overlay) = slot.as_mut()
            && overlay.wanted.is_some()
        {
            overlay.sync_position(mtm);
        }
    });
}

/// The capture timer: the present-path check for a game that has stopped presenting.
///
/// **Main thread only.** Runs for every device. The software cursor hides
/// its sprite for the duration; the hardware cursor needs nothing until the
/// capture ends.
fn on_capture_tick_main() {
    if CAPTURED.load(Ordering::Relaxed) || !capture_suspected() {
        return;
    }
    if CAPTURED
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_ok()
    {
        debug!(target: LOG_TARGET, "cursor: pointer moves without events, another process has it");
        sync_overlay_on_main();
    }
}

/// The overlay window and everything rendered into it. **Main thread only.**
struct Overlay {
    window: Retained<NSWindow>,
    /// The sprite: a sublayer of the window's content layer, moved per event.
    layer: Retained<CAMetalLayer>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    /// One `MTLTexture` per sprite hash, uploaded on first render.
    textures: FxHashMap<u64, Retained<ProtocolObject<dyn MTLTexture>>>,
    /// What the layer's drawable shows right now.
    content: Content,
    /// The layer configuration in place; `None` until the first apply.
    mode: Option<LayerMode>,
    /// The sprite the PE side wants shown, `None` until one was uploaded.
    wanted: Option<(u64, SpriteGeometry)>,
    /// The PE side's last word on visibility.
    wanted_visible: bool,
}

impl Overlay {
    /// Create the window, its layer and the input hooks. **Main thread only.**
    ///
    /// `None` when there is no bound layer to borrow a device from or Metal
    /// refuses a command queue; the next apply tries again.
    fn create(mtm: MainThreadMarker) -> Option<Self> {
        let game_layer = retain_bound_layer()?;
        let device = game_layer.device()?;
        let queue = device.newCommandQueue()?;
        queue.setLabel(Some(&NSString::from_str("mtld3d-cursor-queue")));

        let layer = CAMetalLayer::new();
        layer.setDevice(Some(&device));
        // The sprite has an alpha channel and the window behind it is clear:
        // the compositor blends the whole window onto the game.
        layer.setOpaque(false);
        layer.setFramebufferOnly(true);
        layer.setMaximumDrawableCount(2);
        layer.setAllowsNextDrawableTimeout(true);
        layer.setPresentsWithTransaction(false);
        layer.setName(Some(&NSString::from_str("mtld3d-cursor-overlay")));
        // Positioned by its bottom-left corner, like the window it lives in.
        layer.setAnchorPoint(CGPoint { x: 0.0, y: 0.0 });
        layer.setPosition(PARKED);
        let hud = NSDictionary::from_slices::<NSString>(
            &[&NSString::from_str("mode")],
            &[&*NSString::from_str(HUD_MODE_OFF)],
        );
        // SAFETY: an `NSDictionary<NSString, NSString>` is an `NSDictionary`
        // of objects; the erased view is what the setter is declared with.
        let hud = unsafe { Retained::cast_unchecked::<NSDictionary>(hud) };
        // SAFETY: objc2 typed binding; the dictionary is copied by the layer.
        unsafe { layer.setDeveloperHUDProperties(Some(&hud)) };

        let frame = overlay_frame(mtm);
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
        window.setCollectionBehavior(
            NSWindowCollectionBehavior::Transient
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
            layer,
            queue,
            textures: FxHashMap::default(),
            content: Content::Transparent,
            mode: None,
            wanted: None,
            wanted_visible: false,
        })
    }

    /// Bring the window in line with the wanted state, the layer mode and the headroom.
    fn apply(&mut self, mtm: MainThreadMarker, wanted: WantedSnapshot) {
        let mode = if HDR_ACTIVE.load(Ordering::Relaxed) {
            LayerMode::Hdr
        } else {
            LayerMode::Sdr
        };
        if self.mode != Some(mode) {
            self.reconfigure_layer(mtm, mode);
        }
        self.wanted_visible = wanted.visible;
        if wanted.hash == 0 {
            // Nothing to show, and after a detach nothing to keep either.
            self.textures.clear();
            self.wanted = None;
        } else if let Some(geometry) = wanted.geometry {
            self.wanted = Some((wanted.hash, geometry));
        } else {
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET,
                "cursor: sprite {:#018x} was never uploaded; keeping the previous sprite",
                wanted.hash,
            );
        }
        self.sync_position(mtm);
    }

    /// Give the overlay layer the game layer's format, colorspace and EDR opt-in.
    fn reconfigure_layer(&mut self, mtm: MainThreadMarker, mode: LayerMode) {
        let raw_policy = COLOR_SPACE_POLICY.load(Ordering::Relaxed);
        let color_space = ColorSpacePolicy::from_repr(raw_policy).unwrap_or_else(|| {
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET,
                "cursor: attach latched an unknown color.space policy {raw_policy}; \
                 configuring the overlay as passthrough",
            );
            ColorSpacePolicy::Passthrough
        });
        let screen = self.window.screen().or_else(|| NSScreen::mainScreen(mtm));
        let (native_colorspace, screen_profile_name) =
            screen.as_deref().map_or((None, None), screen_color_profile);
        let screen_name = screen.as_deref().map(|s| s.localizedName().to_string());
        let cs_label = apply_layer_color(
            &self.layer,
            LayerColorRefs {
                mode,
                color_space,
                native_colorspace: native_colorspace.as_deref(),
                screen_name: screen_name.as_deref(),
                screen_profile_name: screen_profile_name.as_deref(),
            },
        );
        // `apply_layer_color` names the layer after the frame; ours is the cursor.
        self.layer
            .setName(Some(&NSString::from_str("mtld3d-cursor-overlay")));
        self.mode = Some(mode);
        info!(
            target: LOG_TARGET,
            "cursor: overlay layer configured {mode:?}: pixelFormat={:?} colorspace={cs_label}",
            self.layer.pixelFormat(),
        );
    }

    /// Present the pixels `content` names, if the drawable does not show them already.
    ///
    /// The layer's surface stays in the window's scene either way: hidden is a
    /// transparent clear, never a removed layer.
    fn ensure_content(&mut self, content: Content) {
        if self.content == content {
            return;
        }
        let presented = match content {
            Content::Transparent => self.present_transparent(),
            Content::Sprite { hash, mode, peak } => {
                let Some((_, geometry)) = self.wanted.filter(|(wanted, _)| *wanted == hash) else {
                    return;
                };
                self.render(hash, geometry, mode, peak)
            }
        };
        if presented {
            self.content = content;
            debug!(target: LOG_TARGET, "cursor: overlay shows {content:?}");
        }
    }

    /// Present a transparent drawable: the hidden state.
    fn present_transparent(&self) -> bool {
        let Some(drawable) = self.layer.nextDrawable() else {
            return false;
        };
        let Some(cmd_buf) = self.queue.commandBuffer() else {
            return false;
        };
        cmd_buf.setLabel(Some(&NSString::from_str("mtld3d-cursor-clear")));
        command::clear_cursor_drawable(&cmd_buf, &drawable.texture());
        cmd_buf.presentDrawable(ProtocolObject::from_ref(&*drawable));
        cmd_buf.commit();
        true
    }

    /// Render sprite `hash` into the overlay's drawable, sized to the sprite.
    ///
    /// `false` leaves the content state alone so the next event retries: the
    /// drawable pool can be momentarily empty, and a missing texture upload is
    /// reported once.
    fn render(&mut self, hash: u64, geometry: SpriteGeometry, mode: LayerMode, peak: f32) -> bool {
        let device = self.queue.device();
        let texture = match self.textures.entry(hash) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let Some(texture) = upload_sprite_texture(&device, hash) else {
                    return false;
                };
                entry.insert(texture)
            }
        };
        let scale = f64::from(sprite_scale(hash));
        self.layer.setBounds(CGRect {
            origin: CGPoint { x: 0.0, y: 0.0 },
            size: CGSize {
                width: geometry.width,
                height: geometry.height,
            },
        });
        self.layer.setContentsScale(scale);
        self.layer.setDrawableSize(CGSize {
            width: geometry.width * scale,
            height: geometry.height * scale,
        });
        let Some(drawable) = self.layer.nextDrawable() else {
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET,
                "cursor: nextDrawable returned nil; the sprite is retried on the next change",
            );
            return false;
        };
        let Some(cmd_buf) = self.queue.commandBuffer() else {
            return false;
        };
        cmd_buf.setLabel(Some(&NSString::from_str("mtld3d-cursor")));
        let Some(pipelines) = present::ensure_resources(&device) else {
            return false;
        };
        let (pipeline, uniforms) = match mode {
            LayerMode::Sdr => (pipelines.cursor_copy, None),
            LayerMode::Hdr if peak <= 1.0 => (pipelines.cursor_passthrough, None),
            LayerMode::Hdr => (pipelines.cursor_bt2446, Some(present::hdr_uniforms(peak))),
        };
        if !command::encode_cursor_pass(&cmd_buf, texture, &drawable.texture(), pipeline, uniforms)
        {
            return false;
        }
        cmd_buf.presentDrawable(ProtocolObject::from_ref(&*drawable));
        cmd_buf.commit();
        debug!(
            target: LOG_TARGET,
            "cursor: sprite {hash:#018x} rendered ({:.0}x{:.0} pt, hotspot ({:.0},{:.0}), {mode:?}, peak {peak:.2}x)",
            geometry.width, geometry.height, geometry.hotspot_x, geometry.hotspot_y,
        );
        true
    }

    /// Level, position and content against the pointer and the game window as they are now.
    fn sync_position(&mut self, mtm: MainThreadMarker) {
        let game = retain_bound_view().and_then(|view| view.window().map(|window| (view, window)));
        let Some((view, game_window)) = game else {
            self.ensure_content(Content::Transparent);
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
        let frame = overlay_frame(mtm);
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
        let mut inputs = VisibilityInputs::empty();
        inputs.set(
            VisibilityInputs::WANTED,
            self.wanted_visible && self.wanted.is_some(),
        );
        inputs.set(
            VisibilityInputs::APP_ACTIVE,
            NSApplication::sharedApplication(mtm).isActive(),
        );
        inputs.set(
            VisibilityInputs::POINTER_INSIDE,
            rect_contains(client, mouse),
        );
        inputs.set(VisibilityInputs::OCCLUDED, window_occluded());
        inputs.set(VisibilityInputs::MINIATURIZED, game_window.isMiniaturized());
        inputs.set(VisibilityInputs::CAPTURED, CAPTURED.load(Ordering::Relaxed));
        let shown = overlay_visible(inputs);
        let Some((hash, geometry)) = self.wanted else {
            self.ensure_content(Content::Transparent);
            return;
        };
        // Follow the pointer whenever it is over the game, shown or not: the
        // position write is free and keeps a hidden sprite where it will
        // reappear.
        if inputs.contains(VisibilityInputs::POINTER_INSIDE) {
            let local = self.window.convertPointFromScreen(mouse);
            let (x, y) = sprite_origin((local.x, local.y), geometry);
            self.set_position(CGPoint { x, y });
        }
        if shown {
            let mode = self.mode.unwrap_or(LayerMode::Sdr);
            let peak = f32::from_bits(CURRENT_HEADROOM_BITS.load(Ordering::Relaxed));
            // Re-render on a sprite or layer-mode change, and on a headroom
            // move worth it; otherwise the drawable already shows this sprite.
            let peak = match self.content {
                Content::Sprite {
                    hash: h,
                    mode: m,
                    peak: p,
                } if h == hash && m == mode && !peak_changed(p, peak) => p,
                _ => peak,
            };
            self.ensure_content(Content::Sprite { hash, mode, peak });
        } else {
            self.ensure_content(Content::Transparent);
        }
    }

    /// Move the sprite layer without an implicit animation.
    fn set_position(&self, position: CGPoint) {
        CATransaction::begin();
        CATransaction::setDisableActions(true);
        self.layer.setPosition(position);
        CATransaction::commit();
    }
}

/// The frame the overlay window covers: the screen the game window is on.
///
/// The main screen when the game window is not on any (mid-move between
/// displays) or no game window is bound; the window follows on the next event.
fn overlay_frame(mtm: MainThreadMarker) -> CGRect {
    let screen = retain_bound_view()
        .and_then(|view| view.window())
        .and_then(|window| window.screen())
        .or_else(|| NSScreen::mainScreen(mtm));
    screen.map_or(
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

/// The `scale` the PE side upscaled sprite `hash` by, `1` if it is gone.
fn sprite_scale(hash: u64) -> u32 {
    lock_shared()
        .sprites
        .get(&hash)
        .map_or(1, |sprite| sprite.scale.max(1))
}

/// Copy sprite `hash` out of [`SHARED`] into a shared-storage `MTLTexture`.
///
/// The lock is held across the `replaceRegion` copy, which is the only reader
/// of the pixel bytes; the API thread's insert of a different sprite waits
/// that long and no longer.
fn upload_sprite_texture(
    device: &ProtocolObject<dyn MTLDevice>,
    hash: u64,
) -> Option<Retained<ProtocolObject<dyn MTLTexture>>> {
    let shared = lock_shared();
    let sprite = shared.sprites.get(&hash)?;
    let desc = MTLTextureDescriptor::new();
    desc.setTextureType(MTLTextureType::Type2D);
    // The bytes are D3D9 A8R8G8B8, which is B, G, R, A in memory.
    desc.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
    // SAFETY: plain property setter on a fresh descriptor.
    unsafe { desc.setWidth(sprite.width as usize) };
    // SAFETY: plain property setter on a fresh descriptor.
    unsafe { desc.setHeight(sprite.height as usize) };
    desc.setUsage(MTLTextureUsage::ShaderRead);
    desc.setStorageMode(MTLStorageMode::Shared);
    let texture = device.newTextureWithDescriptor(&desc)?;
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
    drop(shared);
    Some(texture)
}

/// Install the capture tick, a repeating timer on the main run loop. **Main thread only.**
///
/// Part of the pointer watch, so it runs for every device. Added in the
/// common modes so it keeps firing while the main thread tracks a drag. The
/// timer is leaked for the process lifetime like the observers.
fn install_capture_tick() {
    let block = RcBlock::new(|_: NonNull<NSTimer>| on_capture_tick_main());
    // SAFETY: objc2 typed binding; the run loop copies the block.
    let timer =
        unsafe { NSTimer::timerWithTimeInterval_repeats_block(CAPTURE_TICK_SECONDS, true, &block) };
    // Let the system fire it together with other wakeups: a capture noticed
    // half a tick late costs nothing, a timer that cannot drift costs power.
    timer.setTolerance(CAPTURE_TICK_SECONDS / 2.0);
    // SAFETY: reading Foundation's run-loop-mode constant, an immutable static.
    let mode = unsafe { NSRunLoopCommonModes };
    // SAFETY: objc2 typed binding; the run loop retains the timer, and the
    // timer is leaked below so the block it holds outlives every tick.
    unsafe { NSRunLoop::mainRunLoop().addTimer_forMode(&timer, mode) };
    core::mem::forget(timer);
}

/// Install the pointer watch: the mouse-move monitor and the activation observers.
///
/// **Main thread only.** Called at every attach and installed once per
/// process, for every device whatever its cursor mode. The monitor sees every
/// mouse-moved and dragged event the Wine process receives, before Wine
/// dispatches it, and returns it unchanged. Every token is leaked for the
/// process lifetime like the other `AppKit` observers in this module's parent.
pub fn install_pointer_watch() {
    if POINTER_WATCH_INSTALLED.replace(true) {
        return;
    }
    install_capture_tick();
    let monitor = RcBlock::new(|event: NonNull<NSEvent>| -> *mut NSEvent {
        on_pointer_event_main();
        event.as_ptr()
    });
    // Presses and releases count as events too: the last event's position
    // has to be fresh when a game warps the pointer on release.
    let mask = NSEventMask::MouseMoved
        | NSEventMask::LeftMouseDragged
        | NSEventMask::RightMouseDragged
        | NSEventMask::OtherMouseDragged
        | NSEventMask::LeftMouseDown
        | NSEventMask::LeftMouseUp
        | NSEventMask::RightMouseDown
        | NSEventMask::RightMouseUp
        | NSEventMask::OtherMouseDown
        | NSEventMask::OtherMouseUp;
    // SAFETY: objc2 typed binding; AppKit copies the block and the returned
    // token is leaked below so the monitor is never removed.
    let token = unsafe { NSEvent::addLocalMonitorForEventsMatchingMask_handler(mask, &monitor) };
    core::mem::forget(token);

    let center = NSNotificationCenter::defaultCenter();
    // SAFETY: reading AppKit's notification-name constants, immutable statics
    // the framework initialised before `main`.
    let names = unsafe {
        [
            NSApplicationDidBecomeActiveNotification,
            NSApplicationDidResignActiveNotification,
        ]
    };
    for name in names {
        let block = RcBlock::new(|_: NonNull<NSNotification>| on_pointer_event_main());
        // SAFETY: objc2 typed binding; the center copies the block, and the
        // token is leaked so the observer lives for the process.
        let token = unsafe {
            center.addObserverForName_object_queue_usingBlock(Some(name), None, None, &block)
        };
        core::mem::forget(token);
    }
}

#[cfg(test)]
mod tests;
