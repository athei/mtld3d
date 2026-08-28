//! Display-mode policy for a fullscreen device.
//!
//! The pure rules behind `IDirect3D9::EnumAdapterModes` and the mode-set a
//! fullscreen device performs: which of the modes Win32 enumerates are served
//! to the game, and how a mode request is retried. The Win32 calls live in
//! the d3d9 crate; this module only decides.

#[cfg(test)]
mod tests;

/// A display mode a fullscreen device asks user32 for.
///
/// `refresh_hz` is the game's `FullScreen_RefreshRateInHz`, 0 for "any".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModeRequest {
    pub width: u32,
    pub height: u32,
    pub refresh_hz: u32,
}

/// The mode-set attempts for one request, in order.
///
/// A request with a refresh rate is tried as asked and then without it: the
/// game picked the rate from a list that need not name the rate the display
/// runs at, and where a native driver rounds, win32u rejects a rate its mode
/// list does not carry. A request without a rate is a single attempt.
pub fn mode_set_attempts(request: ModeRequest) -> impl Iterator<Item = ModeRequest> {
    let without_rate = (request.refresh_hz != 0).then_some(ModeRequest {
        refresh_hz: 0,
        ..request
    });
    core::iter::once(request).chain(without_rate)
}

/// Maximum tolerated difference between a mode's aspect ratio and the desktop's.
///
/// Expressed as a fraction of the desktop aspect.
///
/// 15 % keeps 4:3 (1.333), 16:10 (1.6), and 16:9 (1.778) alongside the
/// MBP-native 3:2-ish (1.547) desktop, and drops 5:4 (1.250, ~19 % off) and
/// 21:9 (2.333). The intent is "no obviously-wrong aspect in the resolution
/// dropdown", not a hard mathematical filter; if a future desktop aspect
/// surprises us, widen this number.
pub const ASPECT_TOLERANCE: f64 = 0.15;

/// The sizes `EnumAdapterModes` serves, from Win32's mode list.
///
/// `current` (the desktop mode) comes first so it doubles as the adapter
/// display mode. Candidates keep their enumeration order, minus duplicates,
/// anything larger than the desktop on either axis (the display cannot show
/// more pixels than it has, whatever a mode list says), degenerate sizes, and
/// aspects further than [`ASPECT_TOLERANCE`] from the desktop's. The result is
/// never empty: a list that filters down to nothing serves the desktop mode
/// alone.
pub fn select_mode_sizes(
    current: (u32, u32),
    candidates: impl IntoIterator<Item = (u32, u32)>,
) -> Vec<(u32, u32)> {
    let (host_w, host_h) = current;
    let host_aspect = f64::from(host_w) / f64::from(host_h);
    let mut sizes = vec![current];
    for (w, h) in candidates {
        if w == 0 || h == 0 || w > host_w || h > host_h || sizes.contains(&(w, h)) {
            continue;
        }
        let aspect = f64::from(w) / f64::from(h);
        let aspect_off = (aspect - host_aspect).abs() / host_aspect;
        if aspect_off <= ASPECT_TOLERANCE {
            sizes.push((w, h));
        }
    }
    sizes
}
