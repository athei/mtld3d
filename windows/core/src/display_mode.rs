//! Display-mode policy for a fullscreen device.
//!
//! The pure rules behind `IDirect3D9::EnumAdapterModes` and the mode-set a
//! fullscreen device performs: which of the modes Win32 enumerates a
//! fullscreen device may set, which of those are served to the game, and how
//! a mode request is retried. The Win32 calls live in the d3d9 crate; this
//! module only decides.

use core::{cmp::Reverse, fmt};

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

impl fmt::Display for ModeRequest {
    /// `1280x720@60Hz`, or `1280x720` for a request that takes any refresh rate.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{}", self.width, self.height)?;
        if self.refresh_hz != 0 {
            write!(f, "@{}Hz", self.refresh_hz)?;
        }
        Ok(())
    }
}

/// What one mode-set did, as a fullscreen transition's log line names it.
pub enum ModeSetOutcome {
    /// No settable mode was asked for, so none was set.
    ///
    /// A fullscreen request whose size is no display mode follows the window
    /// instead, and a session that set no mode has none to re-assert.
    NoMode,
    /// The requested size was the current mode already, so nothing was set.
    AlreadyCurrent(ModeRequest),
    /// User32 set `set` for `requested`, replacing a mode of size `was`.
    ///
    /// `set` is `requested` without its refresh rate when user32 refused the
    /// rate and took the retry without it ([`mode_set_attempts`]).
    Changed {
        requested: ModeRequest,
        set: ModeRequest,
        was: (u32, u32),
    },
    /// User32 refused every attempt at `requested`.
    Refused(ModeRequest),
}

impl ModeSetOutcome {
    /// The mode the session holds after this outcome, `None` when it set none.
    ///
    /// It is the request, not the attempt that succeeded, so a re-assert
    /// asks for the rate the game asked for again.
    #[must_use]
    pub const fn mode_in_place(&self) -> Option<ModeRequest> {
        match *self {
            Self::AlreadyCurrent(requested) | Self::Changed { requested, .. } => Some(requested),
            Self::NoMode | Self::Refused(_) => None,
        }
    }
}

impl fmt::Display for ModeSetOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoMode => f.write_str("no display mode set"),
            Self::AlreadyCurrent(mode) => write!(f, "display mode {mode} already current"),
            Self::Changed {
                requested,
                set,
                was,
            } => {
                write!(f, "display mode {set} set")?;
                if set != requested {
                    write!(f, " for a {requested} request")?;
                }
                write!(f, " (was {}x{})", was.0, was.1)
            }
            Self::Refused(mode) => write!(f, "display mode {mode} refused"),
        }
    }
}

/// What putting the registry display mode back did, as a log line names it.
pub enum RegistryRestore {
    /// The current or the registry mode could not be read, so nothing changed.
    Unreadable,
    /// The registry mode, of this size, was the current mode already.
    AlreadyCurrent((u32, u32)),
    /// The registry mode of size `registry` replaced the mode of size `was`.
    Restored {
        registry: (u32, u32),
        was: (u32, u32),
    },
    /// `ChangeDisplaySettingsW` answered `ret` putting `registry` back over `was`.
    Failed {
        registry: (u32, u32),
        was: (u32, u32),
        ret: i32,
    },
}

impl RegistryRestore {
    /// `true` when the restore was attempted and user32 refused it.
    #[must_use]
    pub const fn failed(&self) -> bool {
        matches!(self, Self::Failed { .. })
    }
}

impl fmt::Display for RegistryRestore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unreadable => f.write_str("display mode unreadable, left as it is"),
            Self::AlreadyCurrent((w, h)) => {
                write!(f, "registry display mode {w}x{h} already current")
            }
            Self::Restored { registry, was } => write!(
                f,
                "registry display mode {}x{} restored (was {}x{})",
                registry.0, registry.1, was.0, was.1
            ),
            Self::Failed { registry, was, ret } => write!(
                f,
                "restoring the registry display mode {}x{} over {}x{} failed (ret={ret})",
                registry.0, registry.1, was.0, was.1
            ),
        }
    }
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
/// nothing holds the desktop mode alone. [`served_sizes`] picks what games
/// enumerate from it.
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

/// The sizes `EnumAdapterModes` serves, from the settable ones: smallest first, at most `max`.
///
/// Every settable size is served, ordered by width and then height as a
/// display driver lists its modes, unless `max` is non-zero and smaller than
/// the number of sizes. The bound then keeps the `max` largest sizes by pixel
/// count, ties going to the wider size, and always keeps the desktop (the
/// first entry of `settable`), which is the adapter display mode. The
/// largest sizes stay because they are the display's own and the ones a
/// bounded menu is least able to do without; the smallest go first. The
/// bound is on what a menu shows, not on what a fullscreen request may set,
/// since this never touches [`select_mode_sizes`]' list. `max` comes from
/// `display.maxResolutions`, 0 for no bound.
#[must_use]
pub fn served_sizes(settable: &[(u32, u32)], max: usize) -> Vec<(u32, u32)> {
    let Some(&desktop) = settable.first() else {
        return Vec::new();
    };
    let mut sizes = settable.to_vec();
    sizes.sort_unstable();
    sizes.dedup();
    if max == 0 || sizes.len() <= max {
        return sizes;
    }
    let mut by_area = sizes;
    by_area.sort_unstable_by_key(|&size| Reverse((pixels(size), size)));
    by_area.truncate(max);
    if !by_area.contains(&desktop) {
        by_area.pop();
        by_area.push(desktop);
    }
    by_area.sort_unstable();
    by_area
}

/// The refresh rate a Win32 mode means: its own, or `fallback` for Win32's "default" of 0 or 1.
///
/// `EnumDisplaySettings` reports 0 or 1 for a display whose rate it does not
/// know; neither is a rate a game can ask for.
#[must_use]
pub const fn effective_refresh(hz: u32, fallback: u32) -> u32 {
    if hz > 1 { hz } else { fallback }
}

/// The modes the adapter reads from Win32's list at the desktop's colour depth.
///
/// Each listed `((width, height), refresh_hz)` with its rate passed through
/// [`effective_refresh`] against the desktop's rate, followed by the desktop
/// mode itself, which a display that lists its own size only at other rates
/// would otherwise leave out. Unsorted and possibly repeated; [`served_modes`]
/// orders and dedups.
#[must_use]
pub fn listed_modes(
    listed: impl IntoIterator<Item = ((u32, u32), u32)>,
    desktop: ((u32, u32), u32),
) -> Vec<((u32, u32), u32)> {
    let (_, desktop_hz) = desktop;
    listed
        .into_iter()
        .map(|(size, hz)| (size, effective_refresh(hz, desktop_hz)))
        .chain(core::iter::once(desktop))
        .collect()
}

/// The modes `EnumAdapterModes` serves: every listed refresh rate of each served size.
///
/// `modes` is the Win32 list as `((width, height), refresh_hz)` at the
/// desktop's colour depth. Each mode whose size is in `served` is kept once,
/// ordered by width, then height, then refresh rate, lowest first, so each
/// size's rates sit together the way a driver lists them.
#[must_use]
pub fn served_modes(
    modes: impl IntoIterator<Item = ((u32, u32), u32)>,
    served: &[(u32, u32)],
) -> Vec<((u32, u32), u32)> {
    let mut kept: Vec<((u32, u32), u32)> = modes
        .into_iter()
        .filter(|(size, _)| served.contains(size))
        .collect();
    kept.sort_unstable();
    kept.dedup();
    kept
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

fn pixels((w, h): (u32, u32)) -> u64 {
    u64::from(w) * u64::from(h)
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
