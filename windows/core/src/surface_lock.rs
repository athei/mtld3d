//! Which store `LockRect` maps for a standalone colour surface.
//!
//! A colour surface with no parent texture reaches `LockRect` in one of three
//! shapes. `CreateRenderTarget` with `Lockable == TRUE` carries a CPU staging
//! buffer alongside its colour texture; the device's implicit back buffer
//! carries no persistent CPU store but is read back on demand; and a
//! `Lockable == FALSE` render target has no CPU-visible store at all, which
//! D3D9 answers with `D3DERR_INVALIDCALL`. The surface plumbing lives in
//! `windows/d3d9`; the routing decision is here.

/// The store a standalone colour surface's `LockRect` hands back.
#[derive(Debug, PartialEq, Eq)]
pub enum ColorSurfaceLock {
    /// The surface's own CPU staging buffer, uploaded back at `UnlockRect`.
    Staging,
    /// A one-shot read-back page, blitted out of the colour texture.
    ///
    /// Held until `UnlockRect` drops it. The implicit back buffer keeps no
    /// persistent CPU store, so this is the only way its pixels reach the
    /// application.
    BackBufferReadback,
    /// No CPU-visible store: `D3DERR_INVALIDCALL`.
    Reject,
}

/// Route a standalone colour surface's `LockRect` to the store that serves it.
///
/// `has_staging` marks a lockable render target, the one standalone colour
/// surface that owns CPU bytes. `is_back_buffer` marks the device's implicit
/// back buffer, whose read-back serves the screenshot and portrait paths an
/// application drives through a read-only lock. Anything else is a
/// `Lockable == FALSE` render target: D3D9 gives it no lock, so neither do we.
#[must_use]
pub const fn classify_color_surface_lock(
    has_staging: bool,
    is_back_buffer: bool,
) -> ColorSurfaceLock {
    if has_staging {
        ColorSurfaceLock::Staging
    } else if is_back_buffer {
        ColorSurfaceLock::BackBufferReadback
    } else {
        ColorSurfaceLock::Reject
    }
}

#[cfg(test)]
mod tests;
