//! The clauses a fullscreen transition's log line is built from.
//!
//! Each fullscreen transition (entering, staying across a `Reset`, leaving,
//! losing and regaining focus, an external resize answered) logs one line on
//! the display target. The Win32 calls and the decision of what to log live
//! in the d3d9 crate; this module renders the parts from plain numbers: why a
//! session ended, where the window went, a rect, and the session's mode.

use core::fmt;

use crate::display_mode::ModeRequest;

/// Why a fullscreen session ends, as the line the leave writes names it.
pub enum LeaveReason {
    /// A windowed `Reset` with its size given.
    WindowedReset,
    /// A windowed `Reset` that asks for the client area, pending that area's check.
    ///
    /// Such a `Reset` leaves fullscreen before it can read the client area,
    /// and puts fullscreen back when the area is unusable, which a later line
    /// then says.
    ProvisionalWindowedReset,
    /// The device's final release.
    Release,
    /// A fullscreen `CreateDevice` that failed after taking the window over.
    FailedCreate,
}

impl fmt::Display for LeaveReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::WindowedReset => "windowed Reset",
            Self::ProvisionalWindowedReset => {
                "windowed Reset, provisional until its client area is checked"
            }
            Self::Release => "device release",
            Self::FailedCreate => "failed CreateDevice",
        })
    }
}

/// A window rect as a log line names it: `1728x1117 at (0, 0)`.
pub struct RectLabel {
    left: i32,
    top: i32,
    width: i32,
    height: i32,
}

impl RectLabel {
    /// Label the rect with these edges, its extent their saturating difference.
    #[must_use]
    pub const fn new(left: i32, top: i32, right: i32, bottom: i32) -> Self {
        Self {
            left,
            top,
            width: right.saturating_sub(left),
            height: bottom.saturating_sub(top),
        }
    }
}

impl fmt::Display for RectLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}x{} at ({}, {})",
            self.width, self.height, self.left, self.top
        )
    }
}

/// Where a fullscreen transition put the device window, as its log line names it.
pub enum WindowPlacement {
    /// The window was moved to this rect.
    Placed(RectLabel),
    /// The monitor rect could not be read, so the window was not moved.
    NoMonitor,
    /// `D3DCREATE_NOWINDOWCHANGES`: the window is the app's and was not touched.
    AppOwned,
    /// The window no longer exists.
    Gone,
    /// The pre-fullscreen rect was never read, so the window stayed where it is.
    Unmoved,
    /// An extended device's leave: the window stays at its fullscreen rect.
    ///
    /// `shown` is the visibility it was given back, the one it had before
    /// fullscreen showed it. `put_back` names where the mode restore had
    /// moved the window when the leave put it back.
    KeptFullscreen {
        shown: bool,
        put_back: Option<RectLabel>,
    },
    /// An extended device's leave after the application moved its window during the mode restore.
    ///
    /// The window stays where the application put it; `shown` is as for
    /// [`Self::KeptFullscreen`].
    KeptAppMove { shown: bool },
}

impl fmt::Display for WindowPlacement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Placed(rect) => write!(f, "window {rect}"),
            Self::NoMonitor => f.write_str("window not moved, the monitor rect is unreadable"),
            Self::AppOwned => f.write_str("window left to the app (D3DCREATE_NOWINDOWCHANGES)"),
            Self::Gone => f.write_str("window already destroyed"),
            Self::Unmoved => f.write_str("window not moved, its windowed rect was never read"),
            Self::KeptFullscreen { shown, put_back } => {
                f.write_str("window kept at its fullscreen rect")?;
                if let Some(moved) = put_back {
                    write!(f, ", put back from {moved} where the mode restore moved it")?;
                }
                f.write_str(visibility(*shown))
            }
            Self::KeptAppMove { shown } => {
                f.write_str("window left where the application moved it during the mode restore")?;
                f.write_str(visibility(*shown))
            }
        }
    }
}

/// How an extended leave's line ends: the visibility the window was given back.
const fn visibility(shown: bool) -> &'static str {
    if shown {
        ", shown as before fullscreen"
    } else {
        ", hidden again as before fullscreen"
    }
}

/// The display mode a fullscreen session holds, as a log line names it.
pub struct SessionMode(Option<ModeRequest>);

impl SessionMode {
    /// Label `mode`, `None` for a session that set no display mode.
    #[must_use]
    pub const fn new(mode: Option<ModeRequest>) -> Self {
        Self(mode)
    }
}

impl fmt::Display for SessionMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            Some(mode) => write!(f, "session display mode {mode}"),
            None => f.write_str("no session display mode"),
        }
    }
}

#[cfg(test)]
mod tests;
