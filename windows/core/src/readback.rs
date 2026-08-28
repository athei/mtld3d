//! Destination rules for the `GetRenderTargetData` / `GetFrontBufferData` read-backs.
//!
//! Both entry points copy a colour texture into a system-memory destination:
//! a `D3DPOOL_SYSTEMMEM` / `D3DPOOL_SCRATCH` offscreen surface, or a level of
//! a texture in either of those pools. D3D9 copies 1:1 and converts nothing,
//! so the destination has to carry the source's extent and byte layout, and
//! its backing has to hold every row the copy writes. The surface plumbing
//! lives in `windows/d3d9`; the geometry decision is here.

use mtld3d_shared::mtl::PixelFormat;

/// The colour image a read-back copies out of.
pub struct ReadbackSource {
    /// Logical width, the extent the destination is measured against.
    ///
    /// Under a non-default `render.scale` the Metal texture is smaller and
    /// the unix side resolves it up, so this stays the size D3D9 reports.
    pub width: u32,
    /// Logical height. See [`ReadbackSource::width`].
    pub height: u32,
    /// Byte layout of the source texture.
    pub format: PixelFormat,
}

/// Where a read-back lands and how the destination is laid out.
pub struct ReadbackDestination {
    /// Width the destination reports from `GetDesc` / `GetLevelDesc`.
    pub width: u32,
    /// Height the destination reports from `GetDesc` / `GetLevelDesc`.
    pub height: u32,
    /// Byte layout of the destination's D3D9 format.
    pub format: PixelFormat,
    /// Row stride the destination's own `LockRect` reports.
    pub bytes_per_row: u32,
    /// Bytes the destination's backing holds from its first row.
    pub len: u64,
}

/// Why a read-back destination cannot receive the source.
///
/// Carried in `log_once_warn_by!` keys so each distinct rejection fires once
/// instead of once per frame for an application that retries every frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReadbackReject {
    /// The destination is not a system-memory surface.
    ///
    /// Neither a `D3DPOOL_SYSTEMMEM` / `D3DPOOL_SCRATCH` offscreen surface nor
    /// a level of a texture in one of those pools.
    NotSystemMemory,
    /// The source has no pixels to copy.
    EmptySource,
    /// The destination's extent differs from the source's; D3D9 does not scale.
    ExtentMismatch,
    /// The destination's byte layout differs from the source's; D3D9 does not convert.
    FormatMismatch,
    /// The destination's backing is shorter than the rows the copy writes.
    DestinationTooSmall,
}

impl ReadbackReject {
    /// Stable `u64` key for `log_once_warn_by!` so each reason fires once.
    #[must_use]
    pub const fn key(self) -> u64 {
        self as u64
    }

    /// The rejection as the log line states it.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NotSystemMemory => "destination is not a system-memory surface",
            Self::EmptySource => "source has a zero extent",
            Self::ExtentMismatch => "src and dst extents differ (no scaling)",
            Self::FormatMismatch => "src and dst byte layouts differ (no conversion)",
            Self::DestinationTooSmall => "destination backing is shorter than the copy",
        }
    }
}

/// Check a read-back destination against its source; `None` accepts the copy.
///
/// The extent and format rules are D3D9's: the destination of
/// `GetRenderTargetData` and `GetFrontBufferData` has the source's size and
/// format, and anything else is `D3DERR_INVALIDCALL`. The length rule is ours:
/// the copy writes `height` rows of `bytes_per_row`, and the Metal blit takes
/// the whole backing as its destination buffer.
#[must_use]
pub fn reject_readback_dst(
    src: &ReadbackSource,
    dst: &ReadbackDestination,
) -> Option<ReadbackReject> {
    if src.width == 0 || src.height == 0 {
        return Some(ReadbackReject::EmptySource);
    }
    if dst.width != src.width || dst.height != src.height {
        return Some(ReadbackReject::ExtentMismatch);
    }
    if dst.format != src.format {
        return Some(ReadbackReject::FormatMismatch);
    }
    let needed = u64::from(dst.bytes_per_row).saturating_mul(u64::from(dst.height));
    if needed == 0 || dst.len < needed {
        return Some(ReadbackReject::DestinationTooSmall);
    }
    None
}

#[cfg(test)]
mod tests;
