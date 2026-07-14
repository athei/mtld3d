//! Rect parsing + validation for `IDirect3DDevice9::StretchRect`.
//!
//! The actual blit dispatch lives in `windows/d3d9` (it needs a Metal
//! handle); only the pure host-testable parts live here.

/// Parsed source / destination region for a `StretchRect`.
///
/// Coordinates are clamped against the surface dimensions; an empty
/// region (after clamping) is reported as `None` by `parse`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StretchRegion {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

/// Parse a D3D9 `RECT*` (4 × i32, `left/top/right/bottom`) clamped against `(full_w, full_h)`.
///
/// `NULL` means "full surface". Returns `None` for a degenerate / empty rect
/// — caller treats as `D3DERR_INVALIDCALL`.
///
/// `rect_ptr` is opaque: the wrapper crate owns the unsafe deref since
/// `D3DRECT` lives in `mtld3d-types` (not depended on here). Caller must
/// either pass `None` or a `Some((x1, y1, x2, y2))` already extracted.
#[must_use]
pub fn parse_rect(
    extracted: Option<(i32, i32, i32, i32)>,
    full_w: u32,
    full_h: u32,
) -> Option<StretchRegion> {
    let Some((x1, y1, x2, y2)) = extracted else {
        return Some(StretchRegion {
            x: 0,
            y: 0,
            w: full_w,
            h: full_h,
        });
    };
    let x1 = x1.max(0).cast_unsigned();
    let y1 = y1.max(0).cast_unsigned();
    let x2 = x2.max(0).cast_unsigned();
    let y2 = y2.max(0).cast_unsigned();
    let x = x1.min(full_w);
    let y = y1.min(full_h);
    let right = x2.min(full_w);
    let bottom = y2.min(full_h);
    if right <= x || bottom <= y {
        return None;
    }
    Some(StretchRegion {
        x,
        y,
        w: right - x,
        h: bottom - y,
    })
}

/// Why a `StretchRect` was rejected.
///
/// Carried in `log_once_warn_by!` keys so each distinct mismatch fires
/// exactly once instead of flooding the warn surface with the same line
/// per draw.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RejectReason {
    /// Source / destination differ in pixel format.
    FormatMismatch,
    /// Source / destination region differ in size — scaling is not supported (1:1 only).
    Scaling,
    /// Source surface has no Metal backing.
    ///
    /// E.g. a depth-stencil standalone surface, or a surface type we
    /// don't recognise.
    UnsupportedSource,
    /// Destination surface has no Metal backing.
    UnsupportedDestination,
    /// Source and destination resolve to the same Metal texture handle.
    ///
    /// Metal disallows self-overlap blits.
    SameSurface,
}

impl RejectReason {
    /// Stable u64 key used by `log_once_warn_by!` so each reason fires once.
    ///
    /// Keying on the discriminant keeps the reasons distinct: they are
    /// neither collapsed into a single `log_once_warn!` nor repeated per draw.
    #[must_use]
    pub const fn key(self) -> u64 {
        self as u64
    }

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FormatMismatch => "format mismatch (no conversion path)",
            Self::Scaling => "src and dst dimensions differ (no scaling)",
            Self::UnsupportedSource => "source surface has no Metal backing",
            Self::UnsupportedDestination => "destination surface has no Metal backing",
            Self::SameSurface => "src and dst are the same Metal texture",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_rect_is_full_surface() {
        assert_eq!(
            parse_rect(None, 100, 200),
            Some(StretchRegion {
                x: 0,
                y: 0,
                w: 100,
                h: 200
            })
        );
    }

    #[test]
    fn rect_clamped_against_surface() {
        assert_eq!(
            parse_rect(Some((-10, -20, 50, 60)), 100, 100),
            Some(StretchRegion {
                x: 0,
                y: 0,
                w: 50,
                h: 60
            })
        );
        assert_eq!(
            parse_rect(Some((10, 20, 200, 300)), 100, 100),
            Some(StretchRegion {
                x: 10,
                y: 20,
                w: 90,
                h: 80
            })
        );
    }

    #[test]
    fn empty_rect_returns_none() {
        assert_eq!(parse_rect(Some((50, 50, 50, 50)), 100, 100), None);
        assert_eq!(parse_rect(Some((100, 0, 200, 100)), 100, 100), None);
    }

    #[test]
    fn reject_keys_are_distinct() {
        let keys: Vec<u64> = [
            RejectReason::FormatMismatch,
            RejectReason::Scaling,
            RejectReason::UnsupportedSource,
            RejectReason::UnsupportedDestination,
            RejectReason::SameSurface,
        ]
        .iter()
        .map(|r| r.key())
        .collect();
        let mut sorted = keys.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(keys.len(), sorted.len());
    }
}
