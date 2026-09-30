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

/// How many sizes `EnumAdapterModes` serves at most, per adapter format.
///
/// Era games size their resolution menus for a driver's list, and Wine's
/// Win32 view under `EmulateModeset` is long: the panel's own modes plus a
/// synthesised bank of standard sizes, 43 sizes on a 3456x2234 MBP.
/// `WoW` 1.12's video-options dropdown holds 32 buttons (40 on
/// Turtle `WoW`) and overflowed with a Lua error once that many sizes were
/// served at each of the two adapter formats; the fixed bank this list
/// replaced came to 16 sizes on that display, 32 entries, and never
/// overflowed. 15 per format keeps both formats under the 32 with a slot to
/// spare. [`served_mode_sizes`] cuts its list at the bound from the end, so
/// the standard sizes go first, smallest first, and then the smallest sizes
/// that fill the display. The bound is on what a menu shows, not on what a
/// fullscreen request may set.
pub const MAX_SERVED_SIZES: usize = 15;

// Two adapter formats inside the 32-button menu named above.
const _: () = assert!(MAX_SERVED_SIZES * 2 < 32);

/// The standard sizes [`served_mode_sizes`] lists after those that fill the display.
///
/// A size is served only when user32 lists it for the display and it is
/// settable, so this adds no mode of its own: it picks from Win32's list the
/// sizes a game's menu is expected to offer whatever the display's shape.
/// The order here, largest first, is the order they are served in.
const STANDARD_SIZES: [(u32, u32); 7] = [
    (2560, 1440),
    (1920, 1080),
    (1600, 900),
    (1280, 720),
    (1024, 768),
    (800, 600),
    (640, 480),
];

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
/// more pixels than it has, whatever a mode list says) and degenerate sizes.
/// A size of any aspect stays: win32u letterboxes a mode whose aspect is not
/// the display's. The result is never empty: a list that filters down to
/// nothing holds the desktop mode alone. [`served_mode_sizes`] bounds what
/// games enumerate from it.
pub fn select_mode_sizes(
    current: (u32, u32),
    candidates: impl IntoIterator<Item = (u32, u32)>,
) -> Vec<(u32, u32)> {
    let (host_w, host_h) = current;
    let mut sizes = vec![current];
    for (w, h) in candidates {
        if w == 0 || h == 0 || w > host_w || h > host_h || sizes.contains(&(w, h)) {
            continue;
        }
        sizes.push((w, h));
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

/// The sizes `EnumAdapterModes` serves: those that fill the display, then the standard sizes.
///
/// The desktop (the first entry, which doubles as the adapter display mode)
/// comes first whatever its shape, then the other sizes that fill the
/// `physical` display (scaled onto it they leave a bar of less than one
/// physical pixel), largest first, then the standard sizes in the list that
/// do not fill it (2560x1440, 1920x1080, 1600x900, 1280x720, 1024x768,
/// 800x600 and 640x480), largest first, at most `max` in all. Under Wine's
/// `EmulateModeset` win32u scales a mode uniformly onto the display and
/// centres it, so a mode of another shape is letterboxed with the desktop
/// showing in the bars. The sizes that fill the display therefore come first,
/// and the standard sizes follow as further window sizes to pick from.
/// `physical` is [`physical_extent`]'s answer: under `EmulateModeset` the
/// extent win32u scales a mode onto, and without it the largest extent on
/// each axis of the driver's own list. The sizes are the primary display's,
/// the only one this layer describes. Every other settable size stays
/// settable for a game's own config, since this never touches
/// [`select_mode_sizes`]' list. A `max` of 0 still serves the desktop.
#[must_use]
pub fn served_mode_sizes(
    settable: &[(u32, u32)],
    physical: (u32, u32),
    max: usize,
) -> Vec<(u32, u32)> {
    let Some((&desktop, rest)) = settable.split_first() else {
        return Vec::new();
    };
    let filling = largest_first(rest, |size| fills_display(size, physical));
    let standard = STANDARD_SIZES
        .into_iter()
        .filter(|&size| !fills_display(size, physical) && rest.contains(&size));
    core::iter::once(desktop)
        .chain(filling)
        .chain(standard)
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

/// Whether a mode of `size` covers a display of `physical` pixels with no visible bar.
///
/// win32u scales a mode uniformly onto the physical display, as far as the
/// tighter axis allows, and centres it. The mode covers the display when
/// the bar left on the other axis, both sides together, is under one
/// physical pixel: win32u rounds each edge of the scaled rectangle to a
/// pixel, and half of less than one pixel rounds to none. A panel's own
/// scaled modes pass although integer rounding leaves their aspects a hair
/// apart (2624x1696 on 3456x2234 leaves a quarter of a pixel). A zero extent
/// on either side fills nothing.
fn fills_display(size: (u32, u32), physical: (u32, u32)) -> bool {
    let (w, h) = (u64::from(size.0), u64::from(size.1));
    let (phys_w, phys_h) = (u64::from(physical.0), u64::from(physical.1));
    if w == 0 || h == 0 || phys_w == 0 || phys_h == 0 {
        return false;
    }
    if w * phys_h >= h * phys_w {
        // Fitted to the width: the bar is `phys_h - h * phys_w / w` rows.
        phys_h * w - h * phys_w < w
    } else {
        // Fitted to the height: the bar is `phys_w - w * phys_h / h` columns.
        phys_w * h - w * phys_h < h
    }
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
