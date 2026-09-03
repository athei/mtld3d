//! The software cursor: the game's cursor bitmap drawn in a transparent overlay window.
//!
//! The PE side keeps the Win32 cursor blank while `cursor.software` is on and
//! sends the cursor's sprite and visibility through the `SetCursorOverlay`
//! thunk; this module draws that sprite in a borderless, click-through
//! `NSWindow` that rides one level above the game window and follows the
//! pointer. The hardware cursor plane is never toggled, and under HDR the
//! sprite goes through the same tone map as the frame, so the cursor is as
//! bright as the UI it hovers over.
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

use core::{cell::RefCell, ptr::NonNull};
use std::{
    collections::hash_map::Entry,
    sync::{
        LazyLock, Mutex, MutexGuard,
        atomic::{AtomicBool, Ordering},
    },
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
use objc2_foundation::{NSNotification, NSNotificationCenter, NSString};
use objc2_metal::{
    MTLCommandBuffer, MTLCommandQueue, MTLDevice, MTLOrigin, MTLPixelFormat, MTLRegion,
    MTLResource, MTLSize, MTLStorageMode, MTLTexture, MTLTextureDescriptor, MTLTextureType,
    MTLTextureUsage,
};
use objc2_quartz_core::{CAMetalDrawable, CAMetalLayer};
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
    }
}

/// Whether the overlay shows its sprite for these inputs.
const fn overlay_visible(inputs: VisibilityInputs) -> bool {
    inputs.contains(VisibilityInputs::WANTED.union(VisibilityInputs::APP_ACTIVE))
        && inputs.contains(VisibilityInputs::POINTER_INSIDE)
        && !inputs.intersects(VisibilityInputs::OCCLUDED.union(VisibilityInputs::MINIATURIZED))
}

/// The overlay window's frame origin that puts the sprite's hotspot under the pointer.
///
/// Cocoa screen coordinates grow upwards and a window's origin is its bottom
/// left, while the hotspot is measured from the sprite's top left.
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
            && overlay.applied_hash != 0
        {
            overlay.sync_position(mtm);
        }
    });
}

/// The overlay window and everything rendered into it. **Main thread only.**
struct Overlay {
    window: Retained<NSWindow>,
    layer: Retained<CAMetalLayer>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    /// One `MTLTexture` per sprite hash, uploaded on first render.
    textures: FxHashMap<u64, Retained<ProtocolObject<dyn MTLTexture>>>,
    /// The sprite currently rendered into the layer, `0` = none.
    applied_hash: u64,
    /// The headroom the rendered sprite was tone-mapped for.
    applied_peak: f32,
    /// The layer configuration in place; `None` until the first apply.
    mode: Option<LayerMode>,
    geometry: SpriteGeometry,
    flags: OverlayFlags,
}

bitflags::bitflags! {
    /// The overlay's two booleans.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    struct OverlayFlags: u8 {
        /// The window is at full alpha.
        const SHOWN = 1 << 0;
        /// The PE side's last word was "visible".
        const WANTED_VISIBLE = 1 << 1;
    }
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

        let frame = CGRect {
            origin: CGPoint { x: 0.0, y: 0.0 },
            size: CGSize {
                width: 1.0,
                height: 1.0,
            },
        };
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
        // Hidden by alpha, never by ordering out: show/hide happens at click
        // rate and an ordering change is the class of cost this replaces.
        window.setAlphaValue(0.0);

        let view = NSView::initWithFrame(NSView::alloc(mtm), frame);
        // Layer-hosting: our layer is the view's layer, sized with it.
        view.setLayer(Some(&layer));
        view.setWantsLayer(true);
        window.setContentView(Some(&view));
        window.orderFrontRegardless();

        install_input_hooks();
        info!(
            target: LOG_TARGET,
            "cursor: overlay window created (borderless, click-through, alpha-hidden)",
        );
        Some(Self {
            window,
            layer,
            queue,
            textures: FxHashMap::default(),
            applied_hash: 0,
            applied_peak: 1.0,
            mode: None,
            geometry: SpriteGeometry::default(),
            flags: OverlayFlags::empty(),
        })
    }

    /// Bring the window in line with the wanted state, the layer mode and the headroom.
    fn apply(&mut self, mtm: MainThreadMarker, wanted: WantedSnapshot) {
        let mode = if HDR_ACTIVE.load(Ordering::Relaxed) {
            LayerMode::Hdr
        } else {
            LayerMode::Sdr
        };
        let mode_changed = self.mode != Some(mode);
        if mode_changed {
            self.reconfigure_layer(mtm, mode);
        }
        let peak = f32::from_bits(CURRENT_HEADROOM_BITS.load(Ordering::Relaxed));
        self.flags.set(OverlayFlags::WANTED_VISIBLE, wanted.visible);

        if wanted.hash == 0 {
            // Nothing to show, and after a detach nothing to keep either.
            if self.applied_hash != 0 {
                self.textures.clear();
                self.applied_hash = 0;
            }
        } else if let Some(geometry) = wanted.geometry {
            let stale = wanted.hash != self.applied_hash
                || mode_changed
                || peak_changed(self.applied_peak, peak);
            if stale && self.render(wanted.hash, geometry, mode, peak) {
                self.applied_hash = wanted.hash;
                self.applied_peak = peak;
                self.geometry = geometry;
                debug!(
                    target: LOG_TARGET,
                    "cursor: sprite {:#018x} rendered ({:.0}x{:.0} pt, hotspot ({:.0},{:.0}), {mode:?}, peak {peak:.2}x)",
                    wanted.hash, geometry.width, geometry.height,
                    geometry.hotspot_x, geometry.hotspot_y,
                );
            }
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

    /// Render sprite `hash` into the overlay's drawable, sized to the sprite.
    ///
    /// `false` leaves the applied state alone so the next apply retries: the
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
        self.window.setContentSize(CGSize {
            width: geometry.width,
            height: geometry.height,
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
        true
    }

    /// Level, position and alpha against the pointer and the game window as they are now.
    fn sync_position(&mut self, mtm: MainThreadMarker) {
        let game = retain_bound_view().and_then(|view| view.window().map(|window| (view, window)));
        let Some((view, game_window)) = game else {
            self.set_shown(false);
            return;
        };
        // Wine re-levels its windows across fullscreen transitions; stay one
        // above whatever the game window is at right now.
        let level = game_window.level() + 1;
        if self.window.level() != level {
            self.window.setLevel(level);
        }
        let mouse = NSEvent::mouseLocation();
        let client = game_window.convertRectToScreen(view.convertRect_toView(view.bounds(), None));
        let mut inputs = VisibilityInputs::empty();
        inputs.set(
            VisibilityInputs::WANTED,
            self.flags.contains(OverlayFlags::WANTED_VISIBLE) && self.applied_hash != 0,
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
        let shown = overlay_visible(inputs);
        if shown {
            let (x, y) = sprite_origin((mouse.x, mouse.y), self.geometry);
            self.window.setFrameOrigin(CGPoint { x, y });
        }
        self.set_shown(shown);
    }

    fn set_shown(&mut self, shown: bool) {
        if self.flags.contains(OverlayFlags::SHOWN) == shown {
            return;
        }
        self.window.setAlphaValue(if shown { 1.0 } else { 0.0 });
        self.flags.set(OverlayFlags::SHOWN, shown);
        debug!(target: LOG_TARGET, "cursor: overlay {}", if shown { "shown" } else { "hidden" });
    }
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
