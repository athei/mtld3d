//! Display-mode policy for a fullscreen device.
//!
//! The pure rules behind `IDirect3D9::EnumAdapterModes` and the mode-set a
//! fullscreen device performs: which of the modes Win32 enumerates a
//! fullscreen device may set, which of those are served to the game, and how
//! a mode request is retried. The Win32 calls live in the d3d9 crate; this
//! module only decides.

use core::cmp::Reverse;

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

/// How many sizes `EnumAdapterModes` serves at most, per adapter format.
///
/// Era games size their resolution menus for a driver's list, and Wine's
/// Win32 view under `EmulateModeset` is long: the panel's own modes plus a
/// synthesised bank of standard sizes, 40 sizes on a 3456x2234 MBP once
/// filtered. `WoW` 1.12's video-options dropdown holds 32 buttons (40 on
/// Turtle `WoW`) and overflowed with a Lua error once that many sizes were
/// served at each of the two adapter formats; the fixed bank this list
/// replaced came to 16 sizes on that display, 32 entries, and never
/// overflowed. 15 per format keeps both formats under the 32 with a slot to
/// spare, a ceiling the panel-aspect filter of [`served_mode_sizes`] rarely
/// reaches. The bound is on what a menu shows, not on what a fullscreen
/// request may set.
pub const MAX_SERVED_SIZES: usize = 15;

// Two adapter formats inside the 32-button menu named above.
const _: () = assert!(MAX_SERVED_SIZES * 2 < 32);

/// Aspect difference within which a size counts as one of the panel's own modes.
///
/// Expressed as a fraction of the desktop aspect. The panel's modes are
/// scaled from one shape, but integer rounding leaves them a hair apart
/// (3456x2234, 2992x1934 and 2624x1696 span 1.5470 to 1.5472); 0.5 % covers
/// that and stays well inside the 3 % to 3:2 (1.5) and the 3.4 % to 16:10
/// (1.6) on that panel.
pub const PANEL_ASPECT_TOLERANCE: f64 = 0.005;

/// The sizes a fullscreen device may set, from Win32's mode list.
///
/// `current` (the desktop mode) comes first so it doubles as the adapter
/// display mode. Candidates keep their enumeration order, minus duplicates,
/// anything larger than the desktop on either axis (the display cannot show
/// more pixels than it has, whatever a mode list says), degenerate sizes, and
/// aspects further than [`ASPECT_TOLERANCE`] from the desktop's. The result is
/// never empty: a list that filters down to nothing holds the desktop mode
/// alone. [`served_mode_sizes`] bounds what games enumerate from it.
pub fn select_mode_sizes(
    current: (u32, u32),
    candidates: impl IntoIterator<Item = (u32, u32)>,
) -> Vec<(u32, u32)> {
    let (host_w, host_h) = current;
    let host_aspect = aspect(current);
    let mut sizes = vec![current];
    for (w, h) in candidates {
        if w == 0 || h == 0 || w > host_w || h > host_h || sizes.contains(&(w, h)) {
            continue;
        }
        if aspect_off((w, h), host_aspect) <= ASPECT_TOLERANCE {
            sizes.push((w, h));
        }
    }
    sizes
}

/// The sizes `EnumAdapterModes` serves: the settable sizes of the panel's aspect, bounded to `max`.
///
/// The desktop (the first entry, which doubles as the adapter display mode)
/// comes first, then the other sizes of the panel's own aspect (within
/// [`PANEL_ASPECT_TOLERANCE`] of the desktop's), largest first, at most
/// `max` in all; ties keep their enumeration order. Only that aspect fills
/// the display: under Wine's `EmulateModeset` any other is letterboxed by
/// win32u's uniform scale with the desktop showing in the bars, and a menu
/// is no place to offer that. Every other settable size stays settable for a
/// game's own config, since this never touches [`select_mode_sizes`]' list.
/// A `max` of 0 still serves the desktop.
#[must_use]
pub fn served_mode_sizes(settable: &[(u32, u32)], max: usize) -> Vec<(u32, u32)> {
    let Some((&desktop, rest)) = settable.split_first() else {
        return Vec::new();
    };
    let desktop_aspect = aspect(desktop);
    let mut panel: Vec<(u32, u32)> = rest
        .iter()
        .copied()
        .filter(|&size| aspect_off(size, desktop_aspect) <= PANEL_ASPECT_TOLERANCE)
        .collect();
    panel.sort_by_key(|&size| Reverse(pixels(size)));
    core::iter::once(desktop)
        .chain(panel)
        .take(max.max(1))
        .collect()
}

/// The positions in a mode list of the modes whose size is served.
///
/// The list a game enumerates through `EnumDisplaySettings` is user32's,
/// every depth and refresh rate of every size; the positions returned are
/// those of the modes at a size in `served`, in the list's own order, so a
/// game walking indices 0.. sees the served sizes and nothing else.
#[must_use]
pub fn served_mode_indices(
    sizes: impl IntoIterator<Item = (u32, u32)>,
    served: &[(u32, u32)],
) -> Vec<u32> {
    sizes
        .into_iter()
        .enumerate()
        .filter(|(_, size)| served.contains(size))
        .filter_map(|(index, _)| u32::try_from(index).ok())
        .collect()
}

fn aspect((w, h): (u32, u32)) -> f64 {
    f64::from(w) / f64::from(h)
}

/// A size's aspect distance from `reference`, as a fraction of `reference`.
fn aspect_off(size: (u32, u32), reference: f64) -> f64 {
    (aspect(size) - reference).abs() / reference
}

fn pixels((w, h): (u32, u32)) -> u64 {
    u64::from(w) * u64::from(h)
}
