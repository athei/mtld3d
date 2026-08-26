//! Pure decision helper for VB/IB `Lock` contention handling.
//!
//! `plan_lock` serves only `Direct` (DYNAMIC) buffers — see
//! [`classify_map_mode`]. Their backing is CPU-placed memory the GPU
//! reads directly at command-buffer execution time, so overwriting bytes
//! while a prior submit is still in flight would corrupt draws the GPU
//! hasn't reached yet. Non-DYNAMIC buffers are `Staged`: they write a
//! separate CPU staging buffer and upload only the dirtied range to a
//! device buffer on Unlock, so they never reach `plan_lock`.
//!
//! Rename only when there's no other option. The decision tree:
//! - `D3DLOCK_NOOVERWRITE` / `D3DLOCK_READONLY`, or uncontended
//!   (`last_submit_seq <= coherent_seq`): `WriteInPlace`.
//! - `D3DLOCK_DISCARD`: Rename, no preserve (game promised the old
//!   bytes are gone).
//! - Whole-buffer contended: Rename, because the game has access to
//!   every byte and might overwrite anything the GPU is currently
//!   reading. Preserve the old contents only if the buffer wasn't
//!   created `D3DUSAGE_WRITEONLY` (game might read what it didn't
//!   write).
//! - Partial non-DISCARD contended: `WriteInPlace`. The (DYNAMIC) game
//!   opted into the "I manage timing" contract (DISCARD/NOOVERWRITE
//!   discipline), the same one non-persistent mapped-buffer APIs (e.g.
//!   OpenGL `glBufferSubData`) make implicitly. Append-only UI batchers
//!   live here; renaming them on every call drives
//!   memory-allocation-failure symptoms. D3D9 promises no such thing on
//!   a plain Lock, so this arm is a deliberate divergence: the README
//!   lists it under "Faster than conformant" and
//!   `unix/conformance/CONFORMANCE.md` carries its conformance-site
//!   rationale. It is also the only arm with no other side effect, so
//!   `d3d9` counts it into a perf row.
//!
//! Side effects (allocate `PageBox`, sync memcpy preserve, queue
//! retention, bump perf counters) stay in `d3d9`; this module just
//! returns a verdict.

use mtld3d_types::{
    D3DLOCK_DISCARD, D3DLOCK_NOOVERWRITE, D3DLOCK_READONLY, D3DPOOL_DEFAULT, D3DUSAGE_DYNAMIC,
};

/// What the caller should do with the old backing's contents on a rename.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreserveKind {
    /// No preserve needed: `D3DLOCK_DISCARD` was set.
    ///
    /// The game explicitly abandons the prior bytes.
    None,
    /// Carry the old bytes across via synchronous memcpy inside the `Lock` call.
    ///
    /// Reached by every whole-buffer non-DISCARD contended Lock (the game
    /// may read the whole buffer via the Lock pointer — D3D9 does not
    /// discard on a plain Lock, even for a `D3DUSAGE_WRITEONLY` buffer).
    /// The memcpy is synchronous because the game is allowed to read back
    /// via CPU, and a deferred GPU blit would not be visible in time.
    Cpu,
}

/// Decision for a single `Lock` call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockPlan {
    /// Hand back a pointer into the existing backing.
    ///
    /// Either uncontended, the caller promised no in-flight overlap
    /// (NOOVERWRITE / READONLY), or the lock range is partial enough that
    /// a well-behaved game won't write bytes any in-flight draw reads.
    WriteInPlace,
    /// Swap the buffer's current backing for a fresh allocation and apply `preserve`.
    ///
    /// Caller must queue the old backing for seq-gated retention before
    /// returning the Lock pointer.
    Rename { preserve: PreserveKind },
}

/// Decide how to handle a `Lock` given the buffer's state and the caller's flags.
///
/// - `flags` is the raw `D3DLOCK_*` bitfield from the game.
/// - `_usage` is the buffer's `D3DUSAGE_*` bitfield captured at creation.
///   Currently unread — a plain Lock preserves contents regardless of
///   `D3DUSAGE_WRITEONLY` — but kept for caller symmetry and the test matrix.
/// - `logical_len` is the buffer's length in bytes (the D3D9-visible
///   size, not the `PageBox`-padded capacity).
/// - `offset_to_lock` / `size_to_lock` are the lock range. A
///   `size_to_lock` of 0 means "to end of buffer" per D3D9.
/// - `last_submit_seq` is the submit seq at which this buffer was
///   last referenced by a GPU-visible draw. Zero if never submitted.
/// - `coherent_seq` is the encoder thread's last retired submit seq.
///
/// Unknown high bits in `flags` are ignored here — the caller logs
/// them via `log_once_warn!`.
#[must_use]
pub fn plan_lock(
    flags: u32,
    _usage: u32,
    logical_len: u32,
    offset_to_lock: u32,
    size_to_lock: u32,
    last_submit_seq: u64,
    coherent_seq: u64,
) -> LockPlan {
    if flags & (D3DLOCK_NOOVERWRITE | D3DLOCK_READONLY) != 0 {
        return LockPlan::WriteInPlace;
    }
    if last_submit_seq <= coherent_seq {
        return LockPlan::WriteInPlace;
    }
    if flags & D3DLOCK_DISCARD != 0 {
        return LockPlan::Rename {
            preserve: PreserveKind::None,
        };
    }

    // `size_to_lock == 0` means "to end of buffer" per D3D9.
    let effective_size = if size_to_lock == 0 {
        logical_len.saturating_sub(offset_to_lock)
    } else {
        size_to_lock.min(logical_len.saturating_sub(offset_to_lock))
    };
    let whole_buffer = offset_to_lock == 0 && effective_size >= logical_len;
    if whole_buffer {
        // A plain (non-DISCARD) whole-buffer Lock must preserve the old bytes
        // even for a D3DUSAGE_WRITEONLY buffer: D3D9 does not discard on a plain
        // Lock, so the prior contents survive a contended rename. The "app
        // abandons old bytes" case is the explicit D3DLOCK_DISCARD branch above
        // (PreserveKind::None).
        return LockPlan::Rename {
            preserve: PreserveKind::Cpu,
        };
    }

    // Partial non-DISCARD non-NOOVERWRITE contended Lock. `plan_lock`
    // now serves only `Direct` (DYNAMIC) buffers, where the game opted
    // into the DISCARD/NOOVERWRITE timing contract — trust it and hand
    // back the existing pointer. (Non-DYNAMIC buffers are `Staged`: their
    // partial writes upload only the dirtied range to a separate device
    // buffer on Unlock and never reach `plan_lock`, so there is no
    // partial-rename race to guard against here.)
    LockPlan::WriteInPlace
}

/// How a VB/IB's CPU writes reach the GPU — chosen once at creation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferMapMode {
    /// Zero-copy: a single `bytesNoCopy` backing the GPU reads directly.
    ///
    /// The game manages write/draw timing via `DISCARD`/`NOOVERWRITE`.
    /// Reached only by `D3DPOOL_DEFAULT` + `D3DUSAGE_DYNAMIC` buffers — the
    /// per-frame UI batcher. This is the path `plan_lock` serves.
    Direct,
    /// Separate CPU staging + a persistent GPU device buffer.
    ///
    /// `Unlock` uploads only the dirtied range. Reached by everything else:
    /// any non-`DEFAULT` pool (regardless of usage) or any non-`DYNAMIC`
    /// buffer — typically static geometry, where renaming the whole backing
    /// on every Lock would allocate far more than a dirty-range upload.
    Staged,
}

/// Pick a buffer's map mode at creation from its pool and usage.
///
/// `Direct` (zero-copy) only when the buffer is `D3DPOOL_DEFAULT` *and*
/// `D3DUSAGE_DYNAMIC`; everything else is `Staged`. D3D9 forbids
/// `MANAGED + DYNAMIC`, so in practice `DYNAMIC` implies `DEFAULT`, but
/// keying on both is the exact rule and routes `MANAGED`/`SYSTEMMEM`
/// statics to `Staged` correctly.
#[must_use]
pub const fn classify_map_mode(usage: u32, pool: u32) -> BufferMapMode {
    if pool == D3DPOOL_DEFAULT && usage & D3DUSAGE_DYNAMIC != 0 {
        BufferMapMode::Direct
    } else {
        BufferMapMode::Staged
    }
}

#[cfg(test)]
mod tests;
