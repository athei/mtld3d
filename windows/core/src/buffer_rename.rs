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
//!   memory-allocation-failure symptoms.
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
mod tests {
    use mtld3d_types::D3DUSAGE_WRITEONLY;

    use super::*;

    const DEFAULT_FLAGS: u32 = 0;
    const NO_USAGE: u32 = 0;
    const RETIRED_SEQ: u64 = 10;
    const IN_FLIGHT_SEQ: u64 = 20;
    const COHERENT: u64 = 15;
    const LEN: u32 = 4096;

    #[test]
    fn noncontended_lock_is_write_in_place() {
        assert_eq!(
            plan_lock(DEFAULT_FLAGS, NO_USAGE, LEN, 0, 1024, RETIRED_SEQ, COHERENT),
            LockPlan::WriteInPlace
        );
    }

    #[test]
    fn readonly_bypasses_rename_even_when_contended() {
        assert_eq!(
            plan_lock(
                D3DLOCK_READONLY,
                NO_USAGE,
                LEN,
                0,
                1024,
                IN_FLIGHT_SEQ,
                COHERENT
            ),
            LockPlan::WriteInPlace
        );
    }

    #[test]
    fn nooverwrite_bypasses_rename_even_when_contended() {
        assert_eq!(
            plan_lock(
                D3DLOCK_NOOVERWRITE,
                NO_USAGE,
                LEN,
                0,
                1024,
                IN_FLIGHT_SEQ,
                COHERENT
            ),
            LockPlan::WriteInPlace
        );
    }

    #[test]
    fn discard_contended_renames_with_no_preserve() {
        assert_eq!(
            plan_lock(
                D3DLOCK_DISCARD,
                NO_USAGE,
                LEN,
                0,
                1024,
                IN_FLIGHT_SEQ,
                COHERENT
            ),
            LockPlan::Rename {
                preserve: PreserveKind::None
            }
        );
    }

    /// Partial non-DISCARD non-NOOVERWRITE contended Locks are `WriteInPlace`, regardless of usage.
    ///
    /// `plan_lock` serves only `Direct` (DYNAMIC) buffers, which manage
    /// write/draw timing via DISCARD/NOOVERWRITE; non-DYNAMIC buffers are
    /// `Staged` and never reach here (their partial writes upload a dirty
    /// range instead).
    #[test]
    fn partial_contended_lock_is_write_in_place() {
        for usage in [NO_USAGE, D3DUSAGE_DYNAMIC, D3DUSAGE_WRITEONLY] {
            assert_eq!(
                plan_lock(
                    DEFAULT_FLAGS,
                    usage,
                    LEN,
                    256,
                    1024,
                    IN_FLIGHT_SEQ,
                    COHERENT
                ),
                LockPlan::WriteInPlace,
                "usage={usage:#x}"
            );
        }
    }

    /// Whole-buffer WRITEONLY contended still renames.
    ///
    /// The game has access to every byte; no-overlap is impossible to
    /// guarantee. A plain (non-DISCARD) whole-buffer Lock preserves the old
    /// bytes even for a WRITEONLY buffer — per the D3D9 lock model a plain
    /// Lock does not discard.
    #[test]
    fn writeonly_whole_buffer_contended_lock_renames_with_cpu_preserve() {
        assert_eq!(
            plan_lock(
                DEFAULT_FLAGS,
                D3DUSAGE_WRITEONLY,
                LEN,
                0,
                LEN,
                IN_FLIGHT_SEQ,
                COHERENT
            ),
            LockPlan::Rename {
                preserve: PreserveKind::Cpu
            }
        );
    }

    /// Whole-buffer non-WRITEONLY contended renames AND preserves.
    ///
    /// The game might read the whole buffer through the Lock pointer. Rare
    /// in practice — non-WRITEONLY VBs are usually static.
    #[test]
    fn plain_whole_buffer_contended_lock_renames_with_cpu_preserve() {
        assert_eq!(
            plan_lock(
                DEFAULT_FLAGS,
                NO_USAGE,
                LEN,
                0,
                LEN,
                IN_FLIGHT_SEQ,
                COHERENT
            ),
            LockPlan::Rename {
                preserve: PreserveKind::Cpu
            }
        );
    }

    #[test]
    fn writeonly_zero_size_is_whole_buffer() {
        // `size_to_lock == 0` means "to end of buffer" — from offset 0 that's the
        // whole buffer, which a plain Lock preserves (D3D9 doesn't discard).
        assert_eq!(
            plan_lock(
                DEFAULT_FLAGS,
                D3DUSAGE_WRITEONLY,
                LEN,
                0,
                0,
                IN_FLIGHT_SEQ,
                COHERENT
            ),
            LockPlan::Rename {
                preserve: PreserveKind::Cpu
            }
        );
    }

    /// Zero size from a nonzero offset reaches end of buffer but not the start.
    ///
    /// Still partial, so `WriteInPlace`.
    #[test]
    fn zero_size_with_nonzero_offset_is_partial_write_in_place() {
        assert_eq!(
            plan_lock(
                DEFAULT_FLAGS,
                D3DUSAGE_WRITEONLY,
                LEN,
                256,
                0,
                IN_FLIGHT_SEQ,
                COHERENT
            ),
            LockPlan::WriteInPlace
        );
    }

    #[test]
    fn never_submitted_is_write_in_place() {
        // `last_submit_seq == 0` against any `coherent_seq >= 0` is
        // retired — nothing in flight to race with.
        assert_eq!(
            plan_lock(DEFAULT_FLAGS, NO_USAGE, LEN, 0, 1024, 0, 0),
            LockPlan::WriteInPlace
        );
    }

    #[test]
    fn equal_seqs_are_retired() {
        // `last_submit_seq == coherent_seq` means the GPU has caught
        // up to this buffer's last submit; in-place is safe.
        assert_eq!(
            plan_lock(DEFAULT_FLAGS, NO_USAGE, LEN, 0, 1024, COHERENT, COHERENT),
            LockPlan::WriteInPlace
        );
    }

    #[test]
    fn nooverwrite_wins_over_discard_under_contention() {
        // NOOVERWRITE's disjoint-write promise makes in-place safe
        // even under contention; the DISCARD bit is ignored.
        assert_eq!(
            plan_lock(
                D3DLOCK_DISCARD | D3DLOCK_NOOVERWRITE,
                NO_USAGE,
                LEN,
                0,
                1024,
                IN_FLIGHT_SEQ,
                COHERENT
            ),
            LockPlan::WriteInPlace
        );
    }

    #[test]
    fn unknown_high_bits_do_not_affect_decision() {
        let unknown = 0x8000_0000;
        assert_eq!(
            plan_lock(
                unknown | D3DLOCK_DISCARD,
                NO_USAGE,
                LEN,
                0,
                1024,
                IN_FLIGHT_SEQ,
                COHERENT
            ),
            LockPlan::Rename {
                preserve: PreserveKind::None
            }
        );
        // No recognized flags + partial contended → WriteInPlace; the
        // unknown bit is ignored and doesn't perturb the decision.
        assert_eq!(
            plan_lock(unknown, NO_USAGE, LEN, 0, 1024, IN_FLIGHT_SEQ, COHERENT),
            LockPlan::WriteInPlace
        );
    }

    /// A lock range that runs past the buffer end clamps to the remaining bytes.
    ///
    /// Still partial, so `WriteInPlace`.
    #[test]
    fn size_clamped_to_buffer_end_is_partial_write_in_place() {
        assert_eq!(
            plan_lock(
                DEFAULT_FLAGS,
                D3DUSAGE_WRITEONLY,
                LEN,
                LEN - 256,
                1024,
                IN_FLIGHT_SEQ,
                COHERENT
            ),
            LockPlan::WriteInPlace
        );
    }

    // ── classify_map_mode ──

    use mtld3d_types::{D3DPOOL_MANAGED, D3DPOOL_SYSTEMMEM};

    #[test]
    fn default_dynamic_is_direct() {
        // The only zero-copy case: DEFAULT pool + DYNAMIC (the UI batcher).
        assert_eq!(
            classify_map_mode(D3DUSAGE_DYNAMIC, D3DPOOL_DEFAULT),
            BufferMapMode::Direct
        );
    }

    #[test]
    fn default_static_is_staged() {
        assert_eq!(
            classify_map_mode(NO_USAGE, D3DPOOL_DEFAULT),
            BufferMapMode::Staged
        );
    }

    #[test]
    fn default_writeonly_without_dynamic_is_staged() {
        // WRITEONLY alone doesn't make it Direct — DYNAMIC is required.
        assert_eq!(
            classify_map_mode(D3DUSAGE_WRITEONLY, D3DPOOL_DEFAULT),
            BufferMapMode::Staged
        );
    }

    #[test]
    fn managed_dynamic_is_staged() {
        // Non-DEFAULT pool → Staged regardless of usage (D3D9 forbids
        // MANAGED+DYNAMIC, but the rule keys on pool).
        assert_eq!(
            classify_map_mode(D3DUSAGE_DYNAMIC, D3DPOOL_MANAGED),
            BufferMapMode::Staged
        );
    }

    #[test]
    fn managed_static_is_staged() {
        assert_eq!(
            classify_map_mode(NO_USAGE, D3DPOOL_MANAGED),
            BufferMapMode::Staged
        );
    }

    #[test]
    fn systemmem_is_staged() {
        assert_eq!(
            classify_map_mode(D3DUSAGE_DYNAMIC, D3DPOOL_SYSTEMMEM),
            BufferMapMode::Staged
        );
    }
}
