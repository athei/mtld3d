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
/// spare. The panel's aspect and the notch area's that [`served_mode_sizes`]
/// keeps can reach it: they come to 15 on a 3456x2234 display. The bound is
/// on what a menu shows, not on what a fullscreen request may set.
pub const MAX_SERVED_SIZES: usize = 15;

// Two adapter formats inside the 32-button menu named above.
const _: () = assert!(MAX_SERVED_SIZES * 2 < 32);

/// Aspect difference within which a size counts as one of the panel's own modes.
///
/// Expressed as a fraction of the desktop aspect, and of the notch area's
/// aspect for the sizes [`served_mode_sizes`] adds after them. The panel's
/// modes are scaled from one shape, but integer rounding leaves them a hair
/// apart (3456x2234, 2992x1934 and 2624x1696 span 1.5470 to 1.5472); 0.5 %
/// covers that and stays well inside the 3 % to 3:2 (1.5) and the 3.4 % to
/// 16:10 (1.6) on that panel.
pub const PANEL_ASPECT_TOLERANCE: f64 = 0.005;

/// The smallest notch strip [`notch_area`] accepts, as a fraction of the physical height.
pub const NOTCH_STRIP_MIN: f64 = 0.03;

/// The fraction of the physical height a notch strip [`notch_area`] accepts stays under.
pub const NOTCH_STRIP_MAX: f64 = 0.04;

/// The bound on the reduced numerator of win32u's monitor scale ratio.
///
/// win32u packs the numerator and the denominator of that ratio into 16 bits
/// each, so a numerator of this value or more is one it cannot represent.
const MONITOR_RATIO_LIMIT: u64 = 1 << 16;

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

/// The physical display size win32u scales a mode onto, from the desktop and Win32's mode list.
///
/// Under `EmulateModeset` win32u lists the physical mode and virtual modes no
/// larger than it on either axis, so the largest extent on each axis across
/// the desktop and the list is the physical mode, even while a virtual mode
/// is current. A list the driver reports itself need have no such entry, and
/// the extent is then no mode at all; after a mode-set there the physical
/// mode is the new mode, so [`monitor_ratio_fits`] holds for every size and
/// the extent only leaves out sizes that did not need it.
pub fn physical_extent(
    desktop: (u32, u32),
    sizes: impl IntoIterator<Item = (u32, u32)>,
) -> (u32, u32) {
    sizes.into_iter().fold(desktop, |(w, h), (size_w, size_h)| {
        (w.max(size_w), h.max(size_h))
    })
}

/// Whether win32u can represent the scale from a mode of `size` onto `physical` at `dpi`.
///
/// After a mode-set win32u recomputes each monitor's scale as the ratio
/// `dpi * physical / size` on each axis, reduced by the greatest common
/// divisor of its two terms, and packs the reduced terms into 16 bits each.
/// Some Wine builds assert that the reduced numerator fits, and abort the
/// process when it does not: at 96 dpi on a 2234-pixel-high display that is
/// every height sharing a factor of 3 or less with 214464, such as 1934. The
/// denominator is a mode size and always fits. A zero term never fails.
#[must_use]
pub fn monitor_ratio_fits(size: (u32, u32), physical: (u32, u32), dpi: u32) -> bool {
    axis_ratio_fits(size.0, physical.0, dpi) && axis_ratio_fits(size.1, physical.1, dpi)
}

/// Leave out of `settable` the sizes whose monitor scale [`monitor_ratio_fits`] rejects.
///
/// The desktop, the first entry, always stays: it is the adapter display
/// mode, so the list is never empty. Every other entry keeps its place.
/// Returns the sizes left out, in list order.
pub fn drop_unscalable_sizes(
    settable: &mut Vec<(u32, u32)>,
    physical: (u32, u32),
    dpi: u32,
) -> Vec<(u32, u32)> {
    let Some(&desktop) = settable.first() else {
        return Vec::new();
    };
    let mut dropped = Vec::new();
    settable.retain(|&size| {
        let keep = size == desktop || monitor_ratio_fits(size, physical, dpi);
        if !keep {
            dropped.push(size);
        }
        keep
    });
    dropped
}

/// The area below a notch, from the physical display size and Win32's mode list.
///
/// A notched `MacBook` panel's physical mode includes the strip beside the
/// notch, and macOS lists a second mode of the same width that leaves it
/// out: 3456x2160 below a 3456x2234 display (a 74-pixel strip, 3.3 % of the
/// height), 1728x1080 below the same panel at 1728x1117 with Wine's Retina
/// mode off, and a 32 to 37 point strip on every notched model, 3.3 % to
/// 3.9 % of the height. Wine keeps macOS's own modes in its list, so the
/// area shows up there, and this relies on macOS listing it. The strip must
/// be at least [`NOTCH_STRIP_MIN`] and under [`NOTCH_STRIP_MAX`] of the
/// physical height, which no two same-width sizes of Wine's own table come
/// within (the closest pair, 1280x800 and 1280x768, are 4 % apart). The
/// tallest such mode wins. `None` for a display without a notch, which is
/// every external one.
pub fn notch_area(
    physical: (u32, u32),
    sizes: impl IntoIterator<Item = (u32, u32)>,
) -> Option<(u32, u32)> {
    let (width, height) = physical;
    sizes
        .into_iter()
        .filter(|&(w, h)| {
            if w != width || h == 0 || h >= height {
                return false;
            }
            let strip = f64::from(height - h) / f64::from(height);
            (NOTCH_STRIP_MIN..NOTCH_STRIP_MAX).contains(&strip)
        })
        .max_by_key(|&(_, h)| h)
}

/// The sizes `EnumAdapterModes` serves: the panel's aspect, then the notch area's, up to `max`.
///
/// The desktop (the first entry, which doubles as the adapter display mode)
/// comes first, then the other sizes of the panel's own aspect (within
/// [`PANEL_ASPECT_TOLERANCE`] of the desktop's), largest first, then, when
/// `notch` names the area below a notch ([`notch_area`]), the sizes of that
/// area's aspect (within the same tolerance) that are not of the panel's,
/// largest first, at most `max` in all; ties keep their enumeration order.
/// Under Wine's `EmulateModeset` win32u scales a mode uniformly onto the
/// display and centres it, so a mode of the panel's aspect fills it and any
/// other is letterboxed with the desktop showing in the bars. On a notched
/// panel the second tier is the area below the notch, which win32u centres,
/// so it straddles the notch strip. Without a notch there is no second tier
/// and only the panel's own aspect is served. The sizes are the primary
/// display's, the only one this layer describes. Every other settable size
/// stays settable for a game's own config, since this never touches
/// [`select_mode_sizes`]' list. A `max` of 0 still serves the desktop.
#[must_use]
pub fn served_mode_sizes(
    settable: &[(u32, u32)],
    notch: Option<(u32, u32)>,
    max: usize,
) -> Vec<(u32, u32)> {
    let Some((&desktop, rest)) = settable.split_first() else {
        return Vec::new();
    };
    let desktop_aspect = aspect(desktop);
    let is_panel = |size: (u32, u32)| aspect_off(size, desktop_aspect) <= PANEL_ASPECT_TOLERANCE;
    let panel = largest_first(rest, is_panel);
    let below_notch = notch.map_or_else(Vec::new, |area| {
        let notch_aspect = aspect(area);
        largest_first(rest, |size| {
            !is_panel(size) && aspect_off(size, notch_aspect) <= PANEL_ASPECT_TOLERANCE
        })
    });
    core::iter::once(desktop)
        .chain(panel)
        .chain(below_notch)
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

/// The sizes `keep` accepts, largest first, ties in list order.
fn largest_first(sizes: &[(u32, u32)], keep: impl Fn((u32, u32)) -> bool) -> Vec<(u32, u32)> {
    let mut kept: Vec<(u32, u32)> = sizes.iter().copied().filter(|&size| keep(size)).collect();
    kept.sort_by_key(|&size| Reverse(pixels(size)));
    kept
}

/// One axis of [`monitor_ratio_fits`].
fn axis_ratio_fits(size: u32, physical: u32, dpi: u32) -> bool {
    let num = u64::from(dpi) * u64::from(physical);
    let divisor = gcd(num, u64::from(size)).max(1);
    num / divisor < MONITOR_RATIO_LIMIT
}

/// The greatest common divisor, `gcd(n, 0) = n`.
const fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}
