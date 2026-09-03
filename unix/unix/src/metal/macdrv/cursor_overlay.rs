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
//! keeps moving; a low-rate tick notices the pointer moving with no events
//! arriving and hides the sprite until events resume. And when such a tool
//! ends, the window server shows the standard arrow rather than the blank
//! cursor Wine set, which Wine never re-applies because its handle did not
//! change; the first event after such a silence re-sets a blank cursor of
//! our own while the sprite is shown.

use core::{cell::RefCell, ptr::NonNull};
use std::{
    collections::hash_map::Entry,
    sync::{
        LazyLock, Mutex, MutexGuard,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

use block2::RcBlock;
use log::{debug, info};
use mtld3d_shared::{
    SetCursorOverlayParams,
    mtl::{ColorSpacePolicy, CursorOverlayFlags},
};
use objc2::{
    AllocAnyThread, MainThreadMarker, MainThreadOnly, rc::Retained, runtime::ProtocolObject,
};
use objc2_app_kit::{
    NSApplication, NSApplicationDidBecomeActiveNotification,
    NSApplicationDidResignActiveNotification, NSBackingStoreType, NSColor, NSCursor, NSEvent,
    NSEventMask, NSImage, NSScreen, NSView, NSWindow, NSWindowAnimationBehavior,
    NSWindowCollectionBehavior, NSWindowStyleMask,
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
    apply_layer_color, retain_bound_layer, retain_bound_view, run_on_main_thread_async,
    screen_color_profile, window_occluded,
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

/// How often the pointer is checked for moving without events reaching us, seconds.
///
/// Four times a second is enough to take the sprite away within a quarter
/// second of a system tool grabbing the pointer, and costs one `mouseLocation`
/// read per tick.
const CAPTURE_TICK_SECONDS: f64 = 0.25;

/// How long without a mouse event before a moved pointer counts as captured.
const CAPTURE_SILENCE_MS: u128 = 200;

/// How long a silence has to be for the next event to re-set the blank cursor.
///
/// Anything that borrows the pointer, the screenshot crosshair included,
/// silences this process for longer than that and can leave a visible cursor
/// behind; an idle pointer that resumes moving costs one cursor set.
const HEAL_SILENCE_MS: u128 = 500;

/// Whether the first event after a silence should re-set the blank cursor.
const fn heal_after_silence(silence_ms: u128) -> bool {
    silence_ms >= HEAL_SILENCE_MS
}

/// Whether the pointer moved while no mouse event reached this process.
///
/// A system tool that takes the pointer (the screenshot crosshair) leaves the
/// application eventless while the pointer keeps moving; that is the only way
/// the two disagree.
const fn pointer_captured(silence_ms: u128, pointer_moved: bool) -> bool {
    pointer_moved && silence_ms >= CAPTURE_SILENCE_MS
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
            visible: params.flags.contains(CursorOverlayFlags::VISIBLE),
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
        run_on_main_thread_async(apply_on_main);
    }
}

/// The queued apply: creates the overlay on first use. **Main thread only.**
fn apply_on_main() {
    APPLY_PENDING.store(false, Ordering::Release);
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
fn on_pointer_event_main() {
    // SAFETY: local event monitors and notification blocks run on the main thread.
    let mtm = unsafe { MainThreadMarker::new_unchecked() };
    OVERLAY.with(|cell| {
        let Ok(mut slot) = cell.try_borrow_mut() else {
            return;
        };
        if let Some(overlay) = slot.as_mut()
            && overlay.wanted.is_some()
        {
            let silence_ms = overlay.last_event.0.elapsed().as_millis();
            overlay.last_event = (Instant::now(), NSEvent::mouseLocation());
            if overlay.captured {
                overlay.captured = false;
                debug!(target: LOG_TARGET, "cursor: mouse events resumed, pointer released");
            }
            overlay.sync_position(mtm);
            if heal_after_silence(silence_ms) {
                overlay.heal_system_cursor();
            }
        }
    });
}

/// The capture tick: hide the sprite while a system tool has the pointer. **Main thread only.**
fn on_capture_tick_main() {
    // SAFETY: the timer fires on the main run loop.
    let mtm = unsafe { MainThreadMarker::new_unchecked() };
    OVERLAY.with(|cell| {
        let Ok(mut slot) = cell.try_borrow_mut() else {
            return;
        };
        let Some(overlay) = slot.as_mut() else {
            return;
        };
        if overlay.captured || !matches!(overlay.content, Content::Sprite { .. }) {
            return;
        }
        let (at, seen) = overlay.last_event;
        let now = NSEvent::mouseLocation();
        let moved = (now.x - seen.x).abs() > 0.5 || (now.y - seen.y).abs() > 0.5;
        if pointer_captured(at.elapsed().as_millis(), moved) {
            overlay.captured = true;
            debug!(target: LOG_TARGET, "cursor: pointer moves without events, hiding the sprite");
            overlay.sync_position(mtm);
        }
    });
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
    /// Our one-pixel transparent cursor, set over whatever replaced Wine's blank.
    blank: Retained<NSCursor>,
    /// When the last mouse event reached us, and where the pointer was then.
    last_event: (Instant, CGPoint),
    /// The pointer is moving without events reaching us: a system tool has it.
    captured: bool,
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

        install_input_hooks();
        install_capture_tick();
        let blank = NSCursor::initWithImage_hotSpot(
            NSCursor::alloc(),
            &NSImage::initWithSize(
                NSImage::alloc(),
                CGSize {
                    width: 1.0,
                    height: 1.0,
                },
            ),
            CGPoint { x: 0.0, y: 0.0 },
        );
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
            blank,
            last_event: (Instant::now(), NSEvent::mouseLocation()),
            captured: false,
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
        inputs.set(VisibilityInputs::CAPTURED, self.captured);
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

    /// Put a blank cursor back after something else may have set a visible one.
    ///
    /// The window server keeps whatever the last process set: a system tool
    /// that borrowed the pointer leaves the standard arrow behind, and Wine
    /// does not re-apply a cursor whose handle did not change. Called on the
    /// first event after a silence, so a tool that still holds the pointer is
    /// never fought; only while the sprite is shown, so a cursor the game set
    /// through user32 while its own cursor is hidden is left alone.
    fn heal_system_cursor(&self) {
        if !matches!(self.content, Content::Sprite { .. }) {
            return;
        }
        self.blank.set();
        debug!(target: LOG_TARGET, "cursor: events resumed after a silence, blank cursor re-set");
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
/// Added in the common modes so it keeps firing while the main thread tracks
/// a drag. The timer is leaked for the process lifetime like the observers.
fn install_capture_tick() {
    let block = RcBlock::new(|_: NonNull<NSTimer>| on_capture_tick_main());
    // SAFETY: objc2 typed binding; the run loop copies the block.
    let timer =
        unsafe { NSTimer::timerWithTimeInterval_repeats_block(CAPTURE_TICK_SECONDS, true, &block) };
    // SAFETY: reading Foundation's run-loop-mode constant, an immutable static.
    let mode = unsafe { NSRunLoopCommonModes };
    // SAFETY: objc2 typed binding; the run loop retains the timer, and the
    // timer is leaked below so the block it holds outlives every tick.
    unsafe { NSRunLoop::mainRunLoop().addTimer_forMode(&timer, mode) };
    core::mem::forget(timer);
}

/// Install the mouse-move monitor and the activation observers. **Main thread only.**
///
/// Called once, from the overlay's creation. The monitor sees every
/// mouse-moved and dragged event the Wine process receives, before Wine
/// dispatches it, and returns it unchanged; the block only repositions the
/// overlay. Every token is leaked for the process lifetime like the other
/// `AppKit` observers in this module's parent.
fn install_input_hooks() {
    let monitor = RcBlock::new(|event: NonNull<NSEvent>| -> *mut NSEvent {
        on_pointer_event_main();
        event.as_ptr()
    });
    let mask = NSEventMask::MouseMoved
        | NSEventMask::LeftMouseDragged
        | NSEventMask::RightMouseDragged
        | NSEventMask::OtherMouseDragged;
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
