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
//! game window is on, and the sprite is an image layer moved inside it. A
//! window frame change makes `AppKit` re-resolve the cursor for the pointer's
//! location, and with no cursor of our own to offer it lands on the arrow over
//! the game's blank cursor on every mouse move; a layer moving inside a fixed
//! window is invisible to that machinery. Show and hide swap the layer's
//! pixels, a sprite or a transparent image, so its surface stays in the
//! window's scene: taking a surface out from above the game layer is free,
//! putting one back costs the game's next present a refresh, and a game
//! hiding the cursor while a button is held would pay that on every click.
//!
//! Threads. The thunk runs on the API thread and only writes [`SHARED`] and
//! queues one main-thread wakeup, coalesced through [`APPLY_PENDING`]. The
//! pre-commit run-loop observer reconciles the latest state once per
//! transaction, and only when the pointer moved or an apply was requested
//! ([`RECONCILE_REQUESTED`]): the thunk, a detach, a GPU completion, a present,
//! an activation change and a headroom refresh each request one, so an
//! iteration of Wine's busy main loop that changed nothing costs one pointer
//! read. Everything that touches `AppKit`, Core Animation or the overlay's
//! Metal objects runs on the main thread: the observer, the activation
//! notifications and the reconciliation of position, visibility, layer mode
//! and EDR headroom. Those objects live in a main-thread `thread_local`, which
//! makes the split sound without a lock around `Retained` handles.
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
//! completed image is assigned in the Core Animation transaction, so a hide
//! and the move made with it land in one frame and the old sprite is never
//! seen at a new place. The observer reads the pointer's position at every
//! pass, so a warp the game makes through winemac, which runs on this thread
//! and delivers no event, is followed in the same transaction as the hide or
//! show beside it. While a changed sprite renders, the sprite already on
//! screen keeps following the pointer with its own geometry.
//!
//! Another process can take the pointer without telling this one: the
//! interactive screenshot crosshair. Its window is what a click at the pointer
//! would hit, so the hit test that decides whether the pointer is over the
//! game answers that window, the sprite hides, and the native cursor is left
//! to the tool. When the hit test answers the game window again, the PE side
//! is asked for its null-then-set kick, which makes Wine re-apply the cursor
//! the tool replaced. A window of this process over the game (a dialog) hides
//! the sprite the same way but needs no kick: Wine re-applies its cursor when
//! the pointer re-enters the game window. The kick serves the hardware cursor
//! too, so the observer is installed at attach for every device, overlay or
//! not.

use core::{
    cell::{Cell, RefCell},
    ptr::NonNull,
};
use std::{
    sync::{
        Arc, LazyLock, Mutex, MutexGuard,
        atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering},
    },
    time::Instant,
};

use block2::RcBlock;
use log::{debug, info};
use mtld3d_shared::{
    SetCursorOverlayParams,
    bounded_cache::BoundedCache,
    mtl::{CURSOR_SPRITE_CACHE_ENTRIES, CursorOverlayFlags},
};
use objc2::{
    AnyThread, MainThreadMarker, MainThreadOnly,
    rc::{Retained, autoreleasepool},
    runtime::ProtocolObject,
};
use objc2_app_kit::{
    NSApplication, NSApplicationDidBecomeActiveNotification,
    NSApplicationDidResignActiveNotification, NSBackingStoreType, NSBitmapImageRep, NSColor,
    NSCursor, NSDeviceRGBColorSpace, NSEvent, NSImage, NSScreen, NSView, NSWindow,
    NSWindowAnimationBehavior, NSWindowCollectionBehavior, NSWindowStyleMask,
};
use objc2_core_foundation::{
    CFRetained, CFRunLoop, CFRunLoopActivity, CFRunLoopObserver, CGPoint, CGRect, CGSize,
    kCFRunLoopCommonModes,
};
use objc2_core_graphics::{CGColorSpace, CGImage};
use objc2_foundation::{
    NSDictionary, NSInteger, NSNotification, NSNotificationCenter, NSNull, NSString,
};
use objc2_metal::{
    MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder,
    MTLCommandQueue, MTLDevice, MTLOrigin, MTLPixelFormat, MTLRegion, MTLResource, MTLSize,
    MTLStorageMode, MTLTexture, MTLTextureDescriptor, MTLTextureType, MTLTextureUsage,
};
use objc2_quartz_core::{CAAction, CALayer, CAMetalLayer};

use super::{
    LayerMode,
    attachment::{self, Attachment},
    run_on_main_thread_async,
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

/// What the sprite layer's image currently shows.
#[derive(Clone, Debug, PartialEq)]
enum Content {
    /// The published image is transparent.
    Transparent,
    /// A sprite, tone-mapped for a layer mode and a headroom.
    Sprite {
        hash: u64,
        mode: LayerMode,
        peak: f32,
        geometry: SpriteGeometry,
        /// The owner layer whose gamma ramp the sprite was rendered through.
        ///
        /// `0` when no ramp applies. A D3D9 gamma ramp is the display's
        /// transfer function, so the sprite takes it too: ramping the frame
        /// and leaving the cursor at full brightness is what a player reads
        /// as a bug.
        gamma_layer: usize,
        /// Which of that layer's ramps it was, so a new one re-renders.
        ///
        /// The image is cached and only rebuilt when this record changes,
        /// which is why the revision is part of the identity rather than the
        /// table itself.
        gamma_revision: u64,
    },
}

/// Clock for the queued-apply latency diagnostic.
static EPOCH: LazyLock<Instant> = LazyLock::new(Instant::now);

/// Main-thread state that outlives any one apply.
#[derive(Default)]
struct PointerWatch {
    /// Shared native blank, also used when the game never supplies a D3D cursor image.
    native_cursor: Option<Retained<NSCursor>>,
    /// A window of another process was frontmost at the pointer, inside the client rect.
    captured: bool,
    /// The pointer position the observer last reconciled against.
    last_location: CGPoint,
}

/// Request one coalesced main-thread reconciliation from the submit thread.
///
/// A present is the one regular wakeup while the pointer is still: it is what
/// notices a tool's window appearing over the pointer, and a native hide or
/// show the PE side found on the same present.
pub fn poll_from_present() {
    let check = {
        let shared = lock_shared();
        shared.pending
            || (shared.owner.is_some()
                && shared
                    .flags
                    .intersects(CursorOverlayFlags::VISIBLE | CursorOverlayFlags::NATIVE_HIDDEN))
    };
    if check {
        queue_apply();
    }
}

/// Whether another process's window has the pointer over the game's client area.
///
/// Inside the client rectangle, the hit test answers the game window unless
/// something sits over it there. A window of this process (a Wine dialog) is
/// what Wine re-applies its cursor for on re-entry; a foreign window, the
/// screenshot crosshair being the one that matters, is a capture, and the
/// return from it asks the PE side for a cursor kick.
const fn pointer_captured(inside_client: bool, hit_is_game: bool, hit_is_ours: bool) -> bool {
    inside_client && !hit_is_game && !hit_is_ours
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
struct Shared {
    /// Content-addressed uploads, shared by every device, bounded.
    ///
    /// A hash-only request for an evicted sprite is rejected, and the PE side
    /// answers that by sending the pixels again.
    sprites: BoundedCache<u64, Arc<Sprite>>,
    /// Identity, mode, sprite and visibility are published under this one mutex.
    owner: Option<Arc<Attachment>>,
    hash: u64,
    flags: CursorOverlayFlags,
    revision: u64,
    /// Native work to complete; hardware-only state has no overlay to apply.
    pending: bool,
}

impl Default for Shared {
    fn default() -> Self {
        Self {
            sprites: BoundedCache::new(CURSOR_SPRITE_CACHE_ENTRIES),
            owner: None,
            hash: 0,
            flags: CursorOverlayFlags::empty(),
            revision: 0,
            pending: false,
        }
    }
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
        if !hardware && pixels.is_none() && !self.sprites.contains(&params.hash) {
            // The designed miss: never uploaded, or evicted since. The PE side
            // sends the pixels on this answer.
            debug!(
                target: LOG_TARGET, "SetCursorOverlay: sprite {:#018x} not held; pixels required",
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
        if !hardware
            && let Some(pixels) = pixels
            && !self.sprites.contains(&params.hash)
        {
            self.sprites.insert(
                params.hash,
                Arc::new(Sprite {
                    width: params.width,
                    height: params.height,
                    x_hotspot: params.x_hotspot,
                    y_hotspot: params.y_hotspot,
                    scale: params.scale,
                    pixels: pixels.into(),
                }),
            );
        }
        let native_was_hidden = self.flags.contains(CursorOverlayFlags::NATIVE_HIDDEN);
        self.owner = Some(owner);
        self.hash = hash;
        self.flags = params.flags;
        self.revision += 1;
        // Until a software sprite has been accepted, no overlay can exist.
        // A native hide needs a UI wakeup even without a D3D sprite. Visible
        // hardware cursors need only the main-thread watch. Once software has
        // been used, retain the apply so hardware takeover clears its content.
        self.pending = !hardware
            || native_was_hidden
            || params.flags.contains(CursorOverlayFlags::NATIVE_HIDDEN)
            || !self.sprites.is_empty();
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
        self.pending = true;
        true
    }

    /// The wanted state, counting the current sprite as used.
    fn snapshot(&mut self) -> WantedSnapshot {
        WantedSnapshot {
            owner: self.owner.as_ref().map(Arc::clone),
            sprite: self.sprites.get(&self.hash).map(Arc::clone),
            hash: self.hash,
            flags: self.flags,
            revision: self.revision,
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

/// Whether something other than pointer motion wants the next observer pass to reconcile.
///
/// Set by every `queue_apply`, taken by the pre-commit observer. Without it
/// the observer, which runs on every iteration of the main run loop, would
/// walk the game window and ask the window server what is under the pointer
/// on iterations where nothing changed.
static RECONCILE_REQUESTED: AtomicBool = AtomicBool::new(false);

/// When the pending apply was queued, nanoseconds since [`EPOCH`].
///
/// The apply logs how long it waited for the main thread at debug level,
/// which is the number that says whether a cursor change landed late.
static APPLY_QUEUED_NS: AtomicU64 = AtomicU64::new(0);

/// The sprite's extent and hotspot in points, the window's coordinate unit.
#[derive(Clone, Debug, Default, PartialEq)]
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
    }
}

/// Whether the overlay shows its sprite for these inputs.
const fn overlay_visible(inputs: VisibilityInputs) -> bool {
    inputs.contains(VisibilityInputs::WANTED.union(VisibilityInputs::APP_ACTIVE))
        && inputs.contains(VisibilityInputs::POINTER_INSIDE)
        && !inputs.intersects(VisibilityInputs::OCCLUDED.union(VisibilityInputs::MINIATURIZED))
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
/// is hidden and the cursor is no longer shown. Another device's teardown
/// leaves the overlay where it is. The uploaded sprites stay: they are
/// content-addressed, so a second device's uploaded set may name an entry the
/// first one sent. The window and its observers stay for the process lifetime
/// like the other `AppKit` observers.
pub fn detach(retired: &Arc<Attachment>) {
    let changed = lock_shared().detach(retired);
    if changed {
        queue_apply();
    }
}

/// Make sure the observer is installed and ask it to reconcile at its next pass. Main thread only.
pub fn reconcile_on_main() {
    let mtm = MainThreadMarker::new().expect("reconcile_on_main runs on the main thread");
    install_pointer_watch(mtm);
    RECONCILE_REQUESTED.store(true, Ordering::Release);
}

fn lock_shared() -> MutexGuard<'static, Shared> {
    SHARED.lock().expect("cursor overlay mutex poisoned")
}

fn queue_apply() {
    RECONCILE_REQUESTED.store(true, Ordering::Release);
    if APPLY_PENDING
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_ok()
    {
        let ns = u64::try_from(EPOCH.elapsed().as_nanos()).unwrap_or(u64::MAX);
        APPLY_QUEUED_NS.store(ns, Ordering::Relaxed);
        run_on_main_thread_async(apply_on_main);
    }
}

/// Wake the main thread; the pre-commit observer applies the latest state. Main thread only.
fn apply_on_main() {
    APPLY_PENDING.store(false, Ordering::Release);
    let queued_ns = APPLY_QUEUED_NS.load(Ordering::Relaxed);
    let waited_us = EPOCH
        .elapsed()
        .as_nanos()
        .saturating_sub(u128::from(queued_ns))
        / 1_000;
    debug!(target: LOG_TARGET, "cursor: apply ran {waited_us} us after it was queued");
    let mtm = MainThreadMarker::new().expect("apply_on_main runs on the main thread");
    install_pointer_watch(mtm);
}

/// One owned snapshot used throughout a native reconciliation.
struct WantedSnapshot {
    owner: Option<Arc<Attachment>>,
    sprite: Option<Arc<Sprite>>,
    hash: u64,
    flags: CursorOverlayFlags,
    revision: u64,
}

bitflags::bitflags! {
    /// Installed observer components; failed components are retried at existing opportunities.
    #[derive(Clone, Copy)]
    struct WatchInstalled: u8 {
        const RUN_LOOP = 1 << 0;
        const ACTIVATION = 1 << 1;
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

/// Whether the pre-commit pass has anything to reconcile: pointer motion or a request.
fn reconcile_due() -> bool {
    let requested = RECONCILE_REQUESTED.swap(false, Ordering::AcqRel);
    let location = NSEvent::mouseLocation();
    let moved = POINTER_WATCH.with_borrow_mut(|watch| {
        let moved = watch.last_location != location;
        watch.last_location = location;
        moved
    });
    moved || requested
}

/// Reconcile cursor content and position before each Core Animation commit.
///
/// Input, activation and GPU completion requests converge here so image and
/// visibility updates share the latest pointer state.
fn apply_on_main_inner(mtm: MainThreadMarker) {
    let wanted = {
        let mut shared = lock_shared();
        if shared.owner.is_none() && !shared.pending {
            // No device has a D3D cursor and no detach or failed draw remains to apply.
            return;
        }
        shared.snapshot()
    };
    install_pointer_watch(mtm);
    let active = NSApplication::sharedApplication(mtm).isActive();
    let hit = pointer_hit(mtm, &wanted, active);
    note_capture(hit.as_ref());
    reconcile_native_cursor(mtm, &wanted, hit.as_ref());
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
            *slot = Overlay::create(mtm, &wanted);
        }
        if let Some(overlay) = slot.as_mut() {
            let completed = overlay.apply(mtm, &wanted, active, hit.as_ref());
            lock_shared().applied(wanted.revision, completed);
        }
    });
}

/// Where the pointer is relative to the followed game window, from one hit test.
struct PointerHit {
    /// The game window, retained for the level and screen reads that follow.
    window: Retained<NSWindow>,
    /// The pointer in screen coordinates, as read for this pass.
    mouse: CGPoint,
    /// Inside the client rectangle with the game window frontmost there.
    over_game: bool,
    /// Inside the client rectangle with a window of another process frontmost there.
    captured: bool,
    /// The game window sits in the Dock.
    miniaturized: bool,
}

/// Hit-test the pointer against the followed window, when a decision needs it.
///
/// `None` while the cursor is neither shown nor natively hidden, while the
/// application is inactive (macOS gives the pointer to the frontmost
/// application), and when the followed view is gone.
fn pointer_hit(mtm: MainThreadMarker, wanted: &WantedSnapshot, active: bool) -> Option<PointerHit> {
    if !active
        || !wanted
            .flags
            .intersects(CursorOverlayFlags::VISIBLE | CursorOverlayFlags::NATIVE_HIDDEN)
    {
        return None;
    }
    let view = attachment::retain_view(wanted.owner.as_ref()?, mtm)?;
    let window = view.window()?;
    let mouse = NSEvent::mouseLocation();
    let client = window.convertRectToScreen(view.convertRect_toView(view.bounds(), None));
    let inside = rect_contains(client, mouse);
    let hit = window_under_pointer(mouse, mtm);
    let hit_is_game = hit == window.windowNumber();
    let hit_is_ours = hit_is_game
        || NSApplication::sharedApplication(mtm)
            .windowWithWindowNumber(hit)
            .is_some();
    Some(PointerHit {
        miniaturized: window.isMiniaturized(),
        window,
        mouse,
        over_game: inside && hit_is_game,
        captured: pointer_captured(inside, hit_is_game, hit_is_ours),
    })
}

/// Track external capture; its end asks the PE side to re-apply Wine's cursor.
fn note_capture(hit: Option<&PointerHit>) {
    let captured = hit.is_some_and(|hit| hit.captured);
    POINTER_WATCH.with_borrow_mut(|watch| {
        if watch.captured == captured {
            return;
        }
        watch.captured = captured;
        debug!(target: LOG_TARGET, "cursor: external capture={captured}");
        if !captured {
            // The tool leaves the system cursor behind and Wine re-applies its
            // own only on a handle change; the kick is that change.
            attachment::request_cursor_kick_all();
        }
    });
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
    /// The sprite: a sublayer of the window's content layer, moved per event.
    layer: Retained<CAMetalLayer>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    /// One `MTLTexture` per sprite hash, uploaded on first render, bounded.
    textures: BoundedCache<u64, Retained<ProtocolObject<dyn MTLTexture>>>,
    /// What has been submitted or published for the current request.
    content: ContentState,
    /// Offscreen output owned until the corresponding GPU completion is observed.
    pending: Option<CursorDraw>,
    /// Completed images by sprite hash, so a sprite seen before shows again without GPU work.
    images: BoundedCache<u64, (Content, CFRetained<CGImage>)>,
    /// What the layer shows now; a pending render moves this with its own geometry.
    published: Option<Content>,
    /// A transparent image keeps a composited surface present while hidden.
    transparent: CFRetained<CGImage>,
    /// Identity of the last reconciled attachment, never just its recyclable address.
    owner: Option<Arc<Attachment>>,
    mode: Option<LayerMode>,
    visibility: Option<VisibilityInputs>,
}

impl Overlay {
    /// Create the window, its layer and the input hooks. **Main thread only.**
    ///
    /// `None` when there is no attachment to borrow a layer's device from or
    /// Metal queue allocation fails; the next apply tries again.
    fn create(mtm: MainThreadMarker, wanted: &WantedSnapshot) -> Option<Self> {
        let transparent = image::transparent()?;
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
        // This layer hosts immutable images and never acquires or presents a
        // Metal drawable. Keep CAMetalLayer's HDR controls for macOS 15 too.
        // No implicit animation on anything written here: without this, the
        // layer having no delegate, every position or bounds write would ease
        // over Core Animation's default quarter second.
        layer.setActions(Some(&no_actions()));
        layer.setName(Some(&NSString::from_str("mtld3d-cursor-overlay")));
        // Positioned by its bottom-left corner, like the window it lives in.
        layer.setAnchorPoint(CGPoint { x: 0.0, y: 0.0 });
        layer.setPosition(PARKED);
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

        let frame = screen_frame(game_screen(mtm, wanted.owner.as_ref()));
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
        // be wherever the game window is moved to.
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
            layer,
            queue,
            textures: BoundedCache::new(CURSOR_SPRITE_CACHE_ENTRIES),
            content: ContentState::default(),
            pending: None,
            images: BoundedCache::new(CURSOR_SPRITE_CACHE_ENTRIES),
            published: None,
            transparent,
            owner: None,
            mode: None,
            visibility: None,
        })
    }

    /// Reconcile one owner and request throughout all native work.
    fn apply(
        &mut self,
        mtm: MainThreadMarker,
        wanted: &WantedSnapshot,
        active: bool,
        hit: Option<&PointerHit>,
    ) -> bool {
        if !same_owner(self.owner.as_ref(), wanted.owner.as_ref()) {
            self.content.invalidate();
            self.pending = None;
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
        self.sync_position(mtm, wanted, active, hit);
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
            self.pending = None;
            self.images.clear();
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
            self.pending = None;
            self.images.clear();
            info!(target: LOG_TARGET, "cursor: layer configured {mode:?} pixelFormat={:?} colorspace={colorspace:?} EDR={}",
                game.pixelFormat(), game.wantsExtendedDynamicRangeContent());
        }
        true
    }

    /// Publish completed pixels and defer GPU work without blocking the main run loop.
    ///
    /// `true` once the layer shows `wanted`; `false` while its render is in
    /// flight or could not be started, in which case the layer keeps what it
    /// showed before.
    fn ensure_content(&mut self, wanted: Content, sprite: Option<&Sprite>) -> bool {
        let Self {
            layer,
            queue,
            textures,
            content,
            pending,
            images,
            published,
            transparent,
            ..
        } = self;
        let scheduled = content.ensure(wanted, |requested, generation, result| {
            *pending = None;
            let cached = match requested {
                Content::Transparent => Some(&**transparent),
                Content::Sprite { hash, .. } => images
                    .get(hash)
                    .filter(|(key, _)| key == requested)
                    .map(|(_, pixels)| &**pixels),
            };
            if let Some(pixels) = cached {
                apply_image(layer, pixels, requested);
                *published = Some(requested.clone());
                result.store(COMPLETED, Ordering::Release);
                return true;
            }
            let Some(draw) = sprite.and_then(|sprite| Self::render(layer, queue, textures, sprite, requested)) else {
                mtld3d_shared::log_once_warn!(target: LOG_TARGET, "cursor: draw preparation failed; latest request remains pending");
                return false;
            };
            submit_image(&draw.command, generation, result);
            *pending = Some(draw);
            true
        });
        if !scheduled || !content.completed() {
            // The completion wakes the existing observer; no GPU wait on main.
            return false;
        }
        if let Some(draw) = pending.take() {
            let colorspace = layer.colorspace();
            let Some(pixels) = image::readback(&draw.texture, colorspace.as_deref()) else {
                content.invalidate();
                mtld3d_shared::log_once_warn!(target: LOG_TARGET, "cursor: image publication failed; request remains pending");
                return false;
            };
            let requested = content.current().expect("completed content exists");
            apply_image(layer, &pixels, requested);
            *published = Some(requested.clone());
            if let Content::Sprite { hash, .. } = requested {
                images.insert(*hash, (requested.clone(), pixels));
            }
        }
        true
    }

    /// Render sprite `hash` offscreen, sized to its source pixels.
    ///
    /// A failed allocation or encode leaves no cached submission, so an existing
    /// input, run-loop or present opportunity retries the latest request.
    fn render(
        layer: &CAMetalLayer,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
        textures: &mut BoundedCache<u64, Retained<ProtocolObject<dyn MTLTexture>>>,
        sprite: &Sprite,
        content: &Content,
    ) -> Option<CursorDraw> {
        let Content::Sprite {
            hash,
            geometry: _,
            mode,
            peak,
            gamma_layer,
            gamma_revision: _,
        } = content
        else {
            mtld3d_shared::log_once_warn!(target: LOG_TARGET, "cursor: sprite renderer received transparent content");
            return None;
        };
        let device = queue.device();
        if !textures.contains(hash) {
            textures.insert(*hash, upload_sprite_texture(&device, *hash, sprite)?);
        }
        let texture = textures
            .get(hash)
            .expect("the sprite texture was just kept");
        let descriptor = MTLTextureDescriptor::new();
        descriptor.setTextureType(MTLTextureType::Type2D);
        descriptor.setPixelFormat(layer.pixelFormat());
        // SAFETY: validated sprite dimensions on a fresh offscreen descriptor.
        unsafe { descriptor.setWidth(sprite.width as usize) };
        // SAFETY: validated sprite dimensions on a fresh offscreen descriptor.
        unsafe { descriptor.setHeight(sprite.height as usize) };
        descriptor.setUsage(MTLTextureUsage::RenderTarget);
        descriptor.setStorageMode(cpu_written_texture_storage(&device));
        let Some(output) = device.newTextureWithDescriptor(&descriptor) else {
            mtld3d_shared::log_once_warn!(target: LOG_TARGET, "cursor: offscreen output allocation failed");
            return None;
        };
        output.setLabel(Some(&NSString::from_str("mtld3d-cursor-image")));
        let Some(command) = queue.commandBuffer() else {
            mtld3d_shared::log_once_warn!(target: LOG_TARGET, "cursor: sprite command buffer allocation failed");
            return None;
        };
        command.setLabel(Some(&NSString::from_str("mtld3d-cursor")));
        let Some(pipelines) = present::ensure_resources(&device) else {
            mtld3d_shared::log_once_warn!(target: LOG_TARGET, "cursor: pipeline allocation failed");
            return None;
        };
        let (stage, pipeline, uniforms) = match mode {
            LayerMode::Sdr => (present::GammaStage::CursorCopy, pipelines.cursor_copy, None),
            LayerMode::Hdr if *peak <= 1.0 => (
                present::GammaStage::CursorPassthrough,
                pipelines.cursor_passthrough,
                None,
            ),
            LayerMode::Hdr => (
                present::GammaStage::CursorBt2446,
                pipelines.cursor_bt2446,
                Some(present::hdr_uniforms(*peak)),
            ),
        };
        // The ramped twin, compiled on the first sprite that needs it. A
        // compile that fails renders the sprite unramped rather than leaving
        // the pointer invisible.
        let gamma = if *gamma_layer == 0 {
            None
        } else {
            present::ensure_gamma_pipeline(&device, stage).map(|handle| (handle, *gamma_layer))
        };
        let (pipeline, gamma_layer) = gamma.unwrap_or((pipeline, 0));
        if !command::encode_cursor_pass(&command, texture, &output, pipeline, uniforms, gamma_layer)
        {
            mtld3d_shared::log_once_warn!(target: LOG_TARGET, "cursor: sprite encoder allocation failed");
            return None;
        }
        if output.storageMode() == MTLStorageMode::Managed {
            let Some(blit) = command.blitCommandEncoder() else {
                mtld3d_shared::log_once_warn!(target: LOG_TARGET, "cursor: managed image sync allocation failed");
                return None;
            };
            blit.synchronizeResource(ProtocolObject::from_ref(&*output));
            blit.endEncoding();
        }
        Some(CursorDraw {
            command,
            texture: output,
        })
    }

    /// Level, position and content against the pointer and the game window as they are now.
    fn sync_position(
        &mut self,
        mtm: MainThreadMarker,
        wanted: &WantedSnapshot,
        active: bool,
        hit: Option<&PointerHit>,
    ) {
        let Some(att) = wanted.owner.as_ref() else {
            self.ensure_content(Content::Transparent, None);
            return;
        };
        let mut inputs = VisibilityInputs::empty();
        inputs.set(
            VisibilityInputs::WANTED,
            wanted.flags.contains(CursorOverlayFlags::VISIBLE) && wanted.sprite.is_some(),
        );
        inputs.set(VisibilityInputs::APP_ACTIVE, active);
        if !inputs.contains(VisibilityInputs::WANTED | VisibilityInputs::APP_ACTIVE) {
            // Transparent content has no position to follow. Avoid window,
            // screen and pointer queries until a sprite can be shown again;
            // that apply resolves its pixels and position in one transaction.
            self.update_visibility(inputs, mtm);
            self.ensure_content(Content::Transparent, None);
            return;
        }
        let Some(hit) = hit else {
            // The followed view retired between the snapshot and this pass.
            self.ensure_content(Content::Transparent, None);
            return;
        };
        // Wine re-levels its windows across fullscreen transitions; stay one
        // above whatever the game window is at right now.
        let level = hit.window.level() + 1;
        if self.window.level() != level {
            self.window.setLevel(level);
        }
        // Follow the game window onto another screen. A window frame change
        // costs one cursor re-resolution by AppKit, which is why it is done
        // only here and never per event.
        let frame = screen_frame(hit.window.screen().or_else(|| NSScreen::mainScreen(mtm)));
        if self.window.frame() != frame {
            self.window.setFrame_display(frame, false);
            info!(
                target: LOG_TARGET,
                "cursor: overlay window moved over ({:.0},{:.0}) {:.0}x{:.0}",
                frame.origin.x, frame.origin.y, frame.size.width, frame.size.height,
            );
        }
        inputs.set(VisibilityInputs::POINTER_INSIDE, hit.over_game);
        inputs.set(VisibilityInputs::OCCLUDED, att.window_occluded());
        inputs.set(VisibilityInputs::MINIATURIZED, hit.miniaturized);
        self.update_visibility(inputs, mtm);
        let shown = overlay_visible(inputs);
        let Some(sprite) = wanted.sprite.as_deref() else {
            self.ensure_content(Content::Transparent, None);
            return;
        };
        let hash = wanted.hash;
        let geometry = SpriteGeometry::of(sprite, att.backing_scale());
        let local = self.window.convertPointFromScreen(hit.mouse);
        let content = if shown {
            let mode = self.mode.unwrap_or(LayerMode::Sdr);
            let peak = att.headroom();
            // Re-render on a sprite or layer-mode change, and on a headroom
            // move worth it; otherwise the cached image already shows this sprite.
            let peak = match self.content.current() {
                Some(Content::Sprite {
                    hash: h,
                    mode: m,
                    peak: p,
                    ..
                }) if *h == hash && *m == mode && !peak_changed(*p, peak) => *p,
                _ => peak,
            };
            let gamma_layer = if att.gamma_active() { att.layer() } else { 0 };
            Content::Sprite {
                hash,
                mode,
                peak,
                geometry: geometry.clone(),
                gamma_layer,
                gamma_revision: crate::metal::gamma::revision(gamma_layer),
            }
        } else {
            Content::Transparent
        };
        // Publish a completed image before its position, in the same Core
        // Animation transaction. While a changed sprite renders, the sprite
        // on screen keeps following the pointer with its own geometry; a hide
        // uses the cached transparent image and lands at once.
        let positioned = self.ensure_content(content, Some(sprite));
        if !inputs.contains(VisibilityInputs::POINTER_INSIDE) {
            return;
        }
        let geometry = if positioned {
            Some(&geometry)
        } else if let Some(Content::Sprite { geometry, .. }) = self.published.as_ref() {
            Some(geometry)
        } else {
            None
        };
        if let Some(geometry) = geometry {
            let (origin_x, origin_y) = sprite_origin((local.x, local.y), geometry);
            self.layer.setPosition(CGPoint {
                x: origin_x,
                y: origin_y,
            });
        }
    }

    fn update_visibility(&mut self, inputs: VisibilityInputs, mtm: MainThreadMarker) {
        if overlay_visible(inputs) {
            select_native_blank(mtm);
        }
        if self.visibility != Some(inputs) {
            debug!(target: LOG_TARGET, "cursor: visibility inputs={inputs:?}");
            self.visibility = Some(inputs);
        }
    }
}

/// Whether this cursor mode owns native blank selection.
const fn native_blank_needed(flags: CursorOverlayFlags) -> bool {
    !flags.contains(CursorOverlayFlags::HARDWARE)
        && flags.contains(CursorOverlayFlags::NATIVE_HIDDEN)
}

/// Honor a software cursor's native hide over the active device's unobscured client area.
fn reconcile_native_cursor(
    mtm: MainThreadMarker,
    wanted: &WantedSnapshot,
    hit: Option<&PointerHit>,
) {
    if !native_blank_needed(wanted.flags) {
        // Wine owns hardware cursors, including their hide/show transitions.
        // A sampled native hide can outlive its Win32 show and must not
        // overwrite the cursor Wine has already restored.
        return;
    }
    let (Some(att), Some(hit)) = (wanted.owner.as_ref(), hit) else {
        // Detached, inactive, or the view retired before the queued apply reached main.
        return;
    };
    if hit.over_game && !hit.miniaturized && !att.window_occluded() {
        select_native_blank(mtm);
    }
}

/// Select the native blank unless it is already the current image.
///
/// The blank stays until Wine selects a cursor of its own, which it does on
/// every handle change. What it displaced is never put back: that image may be
/// the arrow `AppKit` resolved for the pointer rather than Wine's cursor, and
/// putting it back showed the system arrow during a mouselook.
fn select_native_blank(mtm: MainThreadMarker) {
    POINTER_WATCH.with_borrow_mut(|watch| {
        if watch.native_cursor.is_none() {
            watch.native_cursor = native_blank_cursor(mtm);
        }
        let current = NSCursor::currentCursor();
        if let Some(cursor) = watch.native_cursor.as_ref()
            && !core::ptr::eq(&raw const *current, &raw const **cursor)
        {
            cursor.set();
            debug!(target: LOG_TARGET, "cursor: native blank applied");
        }
    });
}

/// A native blank for startup and activation without pointer motion.
///
/// Win32 may already hold our blank HCURSOR while Wine has no cursor window to
/// notify or macOS still displays the previous application's image. Select this
/// transparent image when the overlay can be shown or Win32 requests a native
/// hide. Replacing an image changes no hide count and synthesizes no mouse input.
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
    texture: Retained<ProtocolObject<dyn MTLTexture>>,
}

/// Complete an offscreen sprite without waiting for scheduling or GPU execution.
fn submit_image(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    generation: u64,
    result: Arc<AtomicU8>,
) {
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
            queue_apply();
        });
    });
    // SAFETY: Metal copies the block before commit. It retains only its own result
    // cell and generation, never an attachment, PE sink, sprite, or native UI object.
    // As for the present callbacks, device use pins this Unix image until process exit.
    unsafe { cmd_buf.addCompletedHandler(RcBlock::as_ptr(&handler)) };
    cmd_buf.commit();
}

/// Change image and geometry together in the observer's Core Animation transaction.
fn apply_image(layer: &CAMetalLayer, pixels: &CGImage, content: &Content) {
    if let Content::Sprite { geometry, .. } = content {
        layer.setBounds(CGRect {
            origin: CGPoint::default(),
            size: CGSize {
                width: geometry.width,
                height: geometry.height,
            },
        });
        layer.setContentsScale(geometry.scale);
    }
    image::set_contents(layer, pixels);
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

/// Reconcile before Core Animation commits the transaction, when there is something to do.
extern "C-unwind" fn before_commit(
    _observer: *mut CFRunLoopObserver,
    _activity: CFRunLoopActivity,
    _info: *mut core::ffi::c_void,
) {
    autoreleasepool(|_| {
        let mtm = MainThreadMarker::new().expect("before_commit runs on the main thread");
        if reconcile_due() {
            apply_on_main_inner(mtm);
        }
    });
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
/// A unit rectangle when there is no screen at all; the window follows on the
/// next pass that finds one.
fn screen_frame(screen: Option<Retained<NSScreen>>) -> CGRect {
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

/// Install the process-lifetime observers; retry only components whose creation failed.
///
/// The run-loop observer is the one place the cursor is reconciled; the
/// activation observers only ask it for a pass, since which application is
/// frontmost is a visibility input that no pointer motion announces.
pub fn install_pointer_watch(mtm: MainThreadMarker) {
    let mut installed = POINTER_WATCH_INSTALLED.get();
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
                Some(before_commit),
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
            queue_apply();
        });
        // SAFETY: the notification center copies the block; the token is kept for life.
        let token = unsafe {
            center.addObserverForName_object_queue_usingBlock(Some(name), None, None, &block)
        };
        core::mem::forget(token);
    }
}

mod image;

#[cfg(test)]
mod tests;
