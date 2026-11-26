//! The `CAMetalLayer` code that does not depend on how the layer was obtained.
//!
//! It holds the attach request and answer, the layer settings both attach
//! paths apply, present pacing, host-time conversion and drawable sizing; the
//! panel cap itself is `mtld3d_shared::render_size`'s, which the API side
//! sizes a back buffer with too. None of it depends on Wine or `AppKit`.

use log::debug;
use mtld3d_shared::mtl::{ColorSpacePolicy, SoftwareCursorPolicy};

use super::bounded_cast;

/// Drawables the layer hands out before one has to come back from the display.
///
/// Three explicit drawables; two starves at 120 Hz under jitter. The
/// presenter's snapshot ring is sized from it: a present queued deeper than
/// this reaches the screen no sooner for a slot of its own.
pub const DRAWABLE_POOL_DEPTH: usize = 3;

/// Present-throttle request resolved PE-side.
///
/// The guest's vsync ask (`D3DPRESENT_PARAMETERS::PresentationInterval`
/// mapped through `display_sync_for`) plus the effective frame-rate ceiling
/// (`0` = uncapped): the lower of the user's `present.maxFps` from
/// `mtld3d.conf` and the rate a divided interval (TWO/THREE/FOUR) asks for.
/// Bundled so the attach/Reset entry points stay inside clippy's
/// `too_many_arguments` threshold.
pub struct PresentPacing {
    /// `true` for every presentation interval but IMMEDIATE.
    ///
    /// Caps presents at the panel ceiling when set.
    pub vsync_requested: bool,
    /// Effective frame-rate ceiling in Hz; `0` = uncapped.
    ///
    /// When both this and vsync are active the lower rate wins.
    pub max_fps: u32,
}

/// Derive the present-throttle duration from the panel ceiling and the PE-side pacing request.
///
/// A vsync request (any interval but IMMEDIATE) contributes `1 / panel_max_hz`, capping
/// presents at the panel ceiling; on `ProMotion` the system fills the gap
/// with adaptive cadence below that. A non-zero `max_fps` contributes
/// `1 / max_fps` regardless of the vsync state. The throttle takes the
/// longer of the two durations, so the lower frame rate always wins; when
/// neither contributes (IMMEDIATE + uncapped, or a zero / unknown
/// `panel_max_hz` with no user cap) the result is `0.0` for unthrottled
/// free-run.
pub fn min_present_duration(panel_max_hz: f64, pacing: &PresentPacing) -> f64 {
    let vsync_duration = if pacing.vsync_requested && panel_max_hz > 0.0 {
        1.0 / panel_max_hz
    } else {
        0.0
    };
    let cap_duration = if pacing.max_fps > 0 {
        1.0 / f64::from(pacing.max_fps)
    } else {
        0.0
    };
    vsync_duration.max(cap_duration)
}

/// Fold a [`PresentPacing`] into the one word an attachment record holds for it.
///
/// The vsync request is the low bit and the frame cap rides above it, so the
/// pair is written and read as a unit.
pub fn pack_pacing(pacing: &PresentPacing) -> u64 {
    (u64::from(pacing.max_fps) << 1) | u64::from(pacing.vsync_requested)
}

/// Read back what [`pack_pacing`] wrote.
pub fn unpack_pacing(bits: u64) -> PresentPacing {
    let max_fps = u32::try_from((bits >> 1) & u64::from(u32::MAX))
        .expect("masked to u32::MAX on the line above");
    PresentPacing {
        vsync_requested: bits & 1 != 0,
        max_fps,
    }
}

/// Everything the PE side's `AttachMetalLayer` request carries in.
///
/// Bundled so the entry point stays inside clippy's `too_many_arguments`
/// threshold, the same reason [`PresentPacing`] exists.
pub struct LayerAttachRequest {
    /// The guest window the layer is attached to.
    pub hwnd: u64,
    /// Back-buffer width, for the geometry log line.
    pub width: u32,
    /// Back-buffer height, for the geometry log line.
    pub height: u32,
    /// The guest's vsync ask plus the user's frame-rate ceiling.
    pub pacing: PresentPacing,
    /// `color.hdr.enable` from `mtld3d.conf`.
    pub hdr_enable: bool,
    /// `color.space` from `mtld3d.conf`.
    pub color_space: ColorSpacePolicy,
    /// Address of the PE-side `AtomicU32` that receives a changed backing scale.
    ///
    /// Recorded on the attachment record and written only while that record
    /// is live. `0` leaves the display-follow path with nothing to publish
    /// into, which is what a headless smoke test that never built one looks
    /// like.
    pub backing_scale_sink_ptr: u64,
    /// Where a cursor re-apply request is written on the PE side, `0` = nowhere.
    pub cursor_kick_sink_ptr: u64,
    /// Where a client surface present request is written on the PE side, `0` = nowhere.
    pub client_surface_request_sink_ptr: u64,
    /// Where the window's screen sizes are published on the PE side, `0` = nowhere.
    pub screen_sizes_sink_ptr: u64,
    /// `cursor.software`, resolved by attach against the layer mode it picks.
    pub software_cursor: SoftwareCursorPolicy,
}

/// What attach answers the PE side about the display and the cursor.
pub struct DisplayCaps {
    /// The layer's `contentsScale` rounded + clamped to `[1, 8]`: 2 in retina mode, else 1.
    pub backing_scale: u32,
    /// Whether the device draws its cursor through the overlay window.
    ///
    /// `cursor.software` resolved against the layer mode; `Auto` follows HDR.
    pub software_cursor_active: bool,
    /// The native pixel size of the panel under the window's screen, `None` = unknown.
    pub panel: Option<(u32, u32)>,
    /// The screen's frame at the layer's scale ([`backing_pixels`]), `None` = unknown.
    pub backing: Option<(u32, u32)>,
}

/// Host-time seconds (`CFTimeInterval`) to nanoseconds, saturating.
///
/// `presentedTime` is a `CACurrentMediaTime`-based host time; a session's
/// uptime in nanoseconds sits far below `u64::MAX`.
pub fn host_seconds_to_ns(secs: f64) -> u64 {
    bounded_cast::f64_to_u64_saturating(secs * 1e9)
}

/// The sizes of the screen under a window, which cap its drawable.
///
/// Published together by the main thread and read together by the
/// presenter, so the pair always names one screen.
pub struct ScreenPixels {
    /// The panel's native pixel size, `None` = unknown.
    pub panel: Option<(u32, u32)>,
    /// The screen's frame in points, `None` = unknown.
    ///
    /// Points rather than pixels: the reader multiplies by the layer's own
    /// `contentsScale` ([`backing_pixels`]), the same scale it reads for the
    /// layer's size, so a retina-mode change cannot leave the two at
    /// different scales. The screen's `backingScaleFactor` would be wrong for
    /// a layer winemac keeps at scale 1 on a 2x screen.
    pub frame: Option<(u32, u32)>,
}

impl ScreenPixels {
    /// Nothing known yet: the drawable keeps the layer's own size.
    pub const UNKNOWN: Self = Self {
        panel: None,
        frame: None,
    };

    /// Fold both sizes into one word, 16 bits per dimension; a size that does not fit is unknown.
    #[must_use]
    pub fn pack(&self) -> u64 {
        let half = |size: Option<(u32, u32)>| {
            size.and_then(|(width, height)| {
                Some(
                    (u64::from(u16::try_from(width).ok()?) << 16)
                        | u64::from(u16::try_from(height).ok()?),
                )
            })
            .unwrap_or(0)
        };
        (half(self.panel) << 32) | half(self.frame)
    }

    /// Read back what [`Self::pack`] wrote.
    #[must_use]
    pub fn unpack(bits: u64) -> Self {
        let half = |half: u64| {
            let width = u32::try_from((half >> 16) & 0xffff).expect("masked to 16 bits");
            let height = u32::try_from(half & 0xffff).expect("masked to 16 bits");
            (width != 0 && height != 0).then_some((width, height))
        };
        Self {
            panel: half(bits >> 32),
            frame: half(bits & 0xffff_ffff),
        }
    }
}

/// A screen frame in points, in the pixels of a layer at `scale`; `None` when unknown or empty.
#[must_use]
pub fn backing_pixels(frame: Option<(u32, u32)>, scale: f64) -> Option<(u32, u32)> {
    let (width, height) = frame?;
    let pixels = points_to_pixels(f64::from(width), f64::from(height), scale);
    (pixels.0 != 0 && pixels.1 != 0).then_some(pixels)
}

/// The drawable size the layer's own geometry asks for: `bounds × contentsScale`.
///
/// `(0, 0)` while the view has no frame yet, which the callers treat as "no
/// answer" rather than a size.
pub fn natural_drawable_size(layer: &objc2_quartz_core::CAMetalLayer) -> (u32, u32) {
    let bounds = layer.bounds();
    points_to_pixels(bounds.size.width, bounds.size.height, layer.contentsScale())
}

/// The size [`sync_drawable_size`] gave the drawable, beside the layer's own.
pub struct DrawableSizes {
    /// The layer's `bounds × contentsScale`.
    pub natural: (u32, u32),
    /// What the drawable is now: `natural`, or the panel cap of it.
    pub drawable: (u32, u32),
    /// The screen's frame at the layer's scale, the backing the cap was taken against.
    pub backing: Option<(u32, u32)>,
}

/// Point `drawableSize` at the layer's own backing store, capped at the panel.
///
/// `CAMetalLayer` documents `drawableSize` as defaulting to `bounds ×
/// contentsScale`, but it captures that once and does **not** follow the
/// layer afterwards: a freshly created wine metal view reports a real
/// `bounds` beside a `0x0` `drawableSize`. So the value has to be pushed,
/// and pushed again whenever the window resizes.
///
/// Present calls this before every `nextDrawable`, which is what keeps the
/// drawable equal to the backing store without waiting for a `WM_SIZE` to
/// make its way through the guest. Pushing is not free (the layer drops its
/// drawable pool), hence the compare first. Degenerate geometry is left
/// alone rather than written as a zero size Metal would reject, and answers
/// `None`.
///
/// `screen` and `source` feed
/// [`panel_drawable_size`](mtld3d_shared::render_size::panel_drawable_size):
/// in a scaled display mode the drawable is the panel's share of the layer
/// rather than the backing store's, when the frame fits it.
///
/// The presenter calls this on its own thread. It touches the layer alone,
/// never the view it backs: `drawableSize` is `CAMetalLayer`'s own and asks
/// the view for no action, unlike an animatable property, so the write needs
/// no main-thread hop. A layer write off the main thread would open an
/// implicit transaction there, and the presenter thread has no run loop to
/// commit one, so the write rides an explicit transaction of its own,
/// committed at once with actions off; only a resize pays for it, since the
/// compare before it returns first on every other present. Reading
/// `bounds`/`contentsScale` off the main thread races an in-flight `AppKit`
/// resize; the cost of losing that race is one frame at the previous size,
/// corrected on the next present.
pub fn sync_drawable_size(
    layer: &objc2_quartz_core::CAMetalLayer,
    screen: &ScreenPixels,
    source: Option<(u32, u32)>,
) -> Option<DrawableSizes> {
    use objc2_core_foundation::CGSize;
    use objc2_quartz_core::CATransaction;

    let bounds = layer.bounds();
    let scale = layer.contentsScale();
    let natural = points_to_pixels(bounds.size.width, bounds.size.height, scale);
    if natural.0 == 0 || natural.1 == 0 {
        return None;
    }
    let backing = backing_pixels(screen.frame, scale);
    let drawable =
        mtld3d_shared::render_size::panel_drawable_size(natural, backing, screen.panel, source);
    let sizes = DrawableSizes {
        natural,
        drawable,
        backing,
    };
    let (native_w, native_h) = drawable;
    let current = layer.drawableSize();
    let width = f64::from(native_w);
    let height = f64::from(native_h);
    if (current.width - width).abs() <= 0.0 && (current.height - height).abs() <= 0.0 {
        return Some(sizes);
    }
    debug!(
        target: super::command::PRESENT_LOG_TARGET,
        "present: drawable resized {:.0}x{:.0} -> {native_w}x{native_h}",
        current.width,
        current.height,
    );
    CATransaction::begin();
    CATransaction::setDisableActions(true);
    layer.setDrawableSize(CGSize { width, height });
    CATransaction::commit();
    Some(sizes)
}

/// A size in points at `scale`, rounded to whole pixels and saturated to `u32`.
fn points_to_pixels(width: f64, height: f64, scale: f64) -> (u32, u32) {
    (
        bounded_cast::f64_to_u32_saturating((width * scale).round()),
        bounded_cast::f64_to_u32_saturating((height * scale).round()),
    )
}

/// Apply the layer settings that do not depend on how the layer was obtained.
///
/// Both attach paths apply them on the main thread.
pub fn apply_shared_settings(layer: &objc2_quartz_core::CAMetalLayer) {
    use objc2_core_graphics::CGColor;
    use objc2_quartz_core::kCAGravityResize;

    // Games are fullscreen-style: no alpha blending with desktop.
    layer.setOpaque(true);
    // Gravity decides how Core Animation fits a drawable that is not the size
    // of the layer's backing store. In a scaled display mode the drawable is
    // capped at the panel's pixels (`sync_drawable_size`), which keeps the
    // panel's aspect, and that differs from the layer's by up to about 0.1 %:
    // aspect-fit would leave a 1 px bar, so the drawable is stretched over the
    // whole layer. An uncapped drawable is the backing store's size, and
    // gravity is inert. During a live resize the previous drawable is
    // stretched over the new bounds until the next present resizes it. The
    // magnification filter follows the screen in the main-thread refresh
    // (`follow_screen_pixels`).
    //
    // SAFETY: `kCAGravityResize` is a CoreAnimation string constant with
    // static storage duration, an immutable global the framework
    // initialised before `main`.
    let gravity = unsafe { kCAGravityResize };
    layer.setContentsGravity(gravity);
    let backdrop = CGColor::new_generic_gray(0.0, 1.0);
    layer.setBackgroundColor(Some(&backdrop));
    // Nothing reads the drawable. Every guest read of the frame (a lock,
    // `GetRenderTargetData`, `StretchRect`, `GetFrontBufferData`) goes to the
    // persistent back-buffer texture, and present only writes the drawable,
    // through render passes. A framebuffer-only drawable stays compressible
    // and eligible for direct-to-display; one that is not keeps an EDR layer
    // in the compositor.
    layer.setFramebufferOnly(true);
    // We always disable Metal-side vsync and instead throttle presents
    // via `presentDrawable:afterMinimumDuration:` set to `1/panel_max_hz`
    // when the guest asked for vsync. On a fixed-Hz panel that matches
    // the old "snap to vblank" cadence; on a ProMotion panel the system
    // adapts the panel rate down to whatever sub-max cadence the API
    // thread sustains under the cap (transparent VRR): fractional
    // production rates land at their actual rate instead of being
    // rounded down to the next vsync divisor. PE-side
    // `D3DPRESENT_INTERVAL_*` mapping (`display_sync_for`): every
    // interval but IMMEDIATE → vsync requested, IMMEDIATE → free-run.
    // A divided interval (TWO/THREE/FOUR) arrives as a frame-rate
    // ceiling, folded PE-side with the user's `present.maxFps` into
    // the one ceiling that rides the same throttle: the lower rate
    // wins, and it also bounds the IMMEDIATE free-run. The duration
    // itself is derived by the caller and kept on the attachment record.
    layer.setDisplaySyncEnabled(false);
    layer.setMaximumDrawableCount(DRAWABLE_POOL_DEPTH);
    // Default true; surface stalls surface as errors, not hangs.
    layer.setAllowsNextDrawableTimeout(true);
    // Default false; no AppKit surface sync needed.
    layer.setPresentsWithTransaction(false);
    // The drawable is the layer's own backing store, never the guest's
    // back-buffer size: a drawable the layer has to rescale into its
    // backing store is a second resample, after whatever present already
    // did, with a phase we do not control. Owning the resample ourselves is
    // what keeps the frame on the pixel grid the screen actually has.
    // Present re-syncs this before every `nextDrawable`, capped at the panel
    // once the screen's sizes are known; the push here is so the first frame
    // does not have to.
    sync_drawable_size(layer, &ScreenPixels::UNKNOWN, None);
}

#[cfg(test)]
mod tests;
