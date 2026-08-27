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
//! `Staged` buffers get their own two rules here. `records_dirty_range`
//! decides whether a `Lock` widens the pending-upload range at all, which
//! is where `D3DLOCK_READONLY` is honoured, and only in the pool that
//! gives it meaning. `may_trust_lock_bounds`
//! then decides whether the `[OffsetToLock, SizeToLock)` window a title
//! announces bounds the bytes it actually writes, and therefore whether
//! the Unlock upload can be narrowed to that window or has to carry the
//! whole buffer. It answers yes for an ordinary lock unless
//! `buffer.ignoreLockBounds` is set, because a whole-buffer range here is
//! not a wider memcpy but a rename; it answers no either way for the two
//! lock shapes that name no narrower window, `D3DLOCK_DISCARD` and a zero
//! `SizeToLock`.
//!
//! Side effects (allocate `PageBox`, sync memcpy preserve, queue
//! retention, bump perf counters) stay in `d3d9`; this module just
//! returns a verdict.

use mtld3d_types::{
    D3DLOCK_DISCARD, D3DLOCK_NOOVERWRITE, D3DLOCK_READONLY, D3DPOOL_DEFAULT, D3DPOOL_MANAGED,
    D3DUSAGE_DYNAMIC,
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

/// Whether a `Staged` buffer's `Lock` bounds may be taken as the extent of its writes.
///
/// A `Staged` buffer has a device buffer separate from its CPU staging and
/// uploads only the range recorded at `Lock`, so every byte a title writes
/// through the returned pointer without naming it is dropped and the GPU
/// reads whatever the device allocation happened to hold. A real D3D9
/// driver never noticed such a write, because the pointer it handed back
/// was into the single allocation the GPU read.
///
/// `true`, the default, means the announcement is taken at its word: it is
/// the dirty range D3D9 describes, the titles that overrun it are rare,
/// and widening is far more expensive here than the extra copy it looks
/// like. A whole-buffer range always overlaps a range a draw already read
/// this frame, so it takes the rename-at-overlap path, which costs a fresh
/// device buffer, a full device-to-device preserve copy, and retention
/// against `memory.vbibRetentionCapMB`, where reaching the cap forces a
/// synchronous mid-frame submit.
///
/// `ignore_configured` is `buffer.ignoreLockBounds`, for a title that
/// provably writes outside the window it names. It applies to every
/// `Staged` buffer, whatever its pool and usage: no pool's contract says
/// how far a write through the returned pointer may reach, so a title that
/// miscounts the window it announces miscounts it on a `MANAGED` buffer as
/// readily as on a `DEFAULT` one.
///
/// Two further terms hold in both positions, because neither names a
/// narrower window to disbelieve in the first place. `D3DLOCK_DISCARD`
/// abandons the whole buffer's contents by definition, and reaching it on
/// a `Staged` buffer means the flag was used outside the
/// `D3DUSAGE_DYNAMIC` it is defined for, which the caller warns about.
/// `SizeToLock == 0` is documented as locking the entire buffer, which
/// also settles the undocumented `(offset > 0, 0)` form: widening to
/// `[0, logical_len)` uploads the head rather than leaving it as the
/// previous upload left it.
///
/// - `flags` is the raw `D3DLOCK_*` bitfield from the game.
/// - `_usage` / `_pool` are the buffer's creation `D3DUSAGE_*` bitfield and
///   `D3DPOOL_*` value. Neither is read: the rule has no pool or usage
///   term. They stay so the caller passes what it has, and so the test
///   matrix can pin that independence over every pair reaching `Staged`.
/// - `size_to_lock` is the announced size, `0` meaning to end of buffer.
#[must_use]
pub const fn may_trust_lock_bounds(
    flags: u32,
    _usage: u32,
    _pool: u32,
    size_to_lock: u32,
    ignore_configured: bool,
) -> bool {
    !ignore_configured && size_to_lock != 0 && flags & D3DLOCK_DISCARD == 0
}

/// Whether a `Staged` buffer's `Lock` widens the range its `Unlock` will upload.
///
/// `D3DLOCK_READONLY` promises the caller will not write, and that promise
/// buys something only where a system-memory master copy exists whose
/// upload can be skipped, which is `D3DPOOL_MANAGED`: the pool is defined
/// by the runtime holding that copy and refreshing the device one from the
/// ranges the application dirties. An unmanaged pool has no second copy
/// and nothing to read back, so the flag carries no information there and
/// a write made under it has to reach the device buffer like any other.
///
/// `D3DLOCK_NO_DIRTY_UPDATE` is deliberately **not** honoured, and the
/// difference from READONLY is the whole point: it asks that the region
/// stay out of the dirty record, but it is not a promise that nothing was
/// written. Dropping the range would drop the write. That is safe only for
/// an implementation that uploads a managed buffer at draw time from a
/// standing whole-buffer dirty range, where the bytes ride along on the
/// next upload anyway. This one uploads at `Unlock` and clears the range,
/// so the write would simply be lost. The conformance suite pins that it
/// must not be: it fills a `MANAGED` buffer under no flags, refills it
/// under `NO_DIRTY_UPDATE` with no draw in between, and requires the draw
/// to show the second fill.
///
/// Returning `false` never loses a buffer's opening contents: creation
/// seeds a `Staged` buffer's range full, so the first `Unlock` carries
/// every byte however the title filled it, and a fill made entirely
/// through locks this rejects still reaches the GPU once.
///
/// - `flags` is the raw `D3DLOCK_*` bitfield from the game.
/// - `pool` is the buffer's creation `D3DPOOL_*` value.
#[must_use]
pub const fn records_dirty_range(flags: u32, pool: u32) -> bool {
    !(flags & D3DLOCK_READONLY != 0 && pool == D3DPOOL_MANAGED)
}

#[cfg(test)]
mod tests;
