//! The `CAMetalLayer` a device presents into, apart from how it is obtained.
//!
//! The attach request and its answer, the present pacing a layer is held to,
//! the host-time conversion of a presented drawable, and the drawable size
//! that follows the layer's own geometry. None of it depends on Wine or
//! `AppKit`; `macdrv` obtains the layer from the metal view Wine creates in the
//! window's client area.

use log::debug;
use mtld3d_shared::mtl::{ColorSpacePolicy, SoftwareCursorPolicy};

use super::bounded_cast;

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
    /// `cursor.software`, resolved by attach against the layer mode it picks.
    pub software_cursor: SoftwareCursorPolicy,
}

/// What attach answers the PE side about the display and the cursor.
#[derive(Clone, Copy)]
pub struct DisplayCaps {
    /// The layer's `contentsScale` rounded + clamped to `[1, 8]`: 2 in retina mode, else 1.
    pub backing_scale: u32,
    /// Whether the device draws its cursor through the overlay window.
    ///
    /// `cursor.software` resolved against the layer mode; `Auto` follows HDR.
    pub software_cursor_active: bool,
}

/// Host-time seconds (`CFTimeInterval`) to nanoseconds, saturating.
///
/// `presentedTime` is a `CACurrentMediaTime`-based host time; a session's
/// uptime in nanoseconds sits far below `u64::MAX`.
pub fn host_seconds_to_ns(secs: f64) -> u64 {
    bounded_cast::f64_to_u64_saturating(secs * 1e9)
}

/// The drawable size the layer's own geometry asks for: `bounds × contentsScale`.
///
/// `(0, 0)` while the view has no frame yet, which the callers treat as "no
/// answer" rather than a size.
pub fn natural_drawable_size(layer: &objc2_quartz_core::CAMetalLayer) -> (u32, u32) {
    let bounds = layer.bounds();
    let scale = layer.contentsScale();
    (
        bounded_cast::f64_to_u32_saturating((bounds.size.width * scale).round()),
        bounded_cast::f64_to_u32_saturating((bounds.size.height * scale).round()),
    )
}

/// Point `drawableSize` at the layer's own backing store.
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
/// alone rather than written as a zero size Metal would reject.
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
pub fn sync_drawable_size(layer: &objc2_quartz_core::CAMetalLayer) {
    use objc2_core_foundation::CGSize;
    use objc2_quartz_core::CATransaction;

    let (native_w, native_h) = natural_drawable_size(layer);
    if native_w == 0 || native_h == 0 {
        return;
    }
    let current = layer.drawableSize();
    let width = f64::from(native_w);
    let height = f64::from(native_h);
    if (current.width - width).abs() <= 0.0 && (current.height - height).abs() <= 0.0 {
        return;
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
}

#[cfg(test)]
mod tests;
