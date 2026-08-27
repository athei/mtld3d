//! Unit tests for the VB/IB `Lock` planner.
//!
//! A matrix over lock flags, usage, lock range and submit-seq contention pins every arm of
//! `plan_lock`: `D3DLOCK_NOOVERWRITE` / `D3DLOCK_READONLY` and uncontended locks write in
//! place, `D3DLOCK_DISCARD` renames with no preserve, a contended whole-buffer lock renames
//! and preserves, and a contended partial lock stays in place so an append-only batcher is
//! not renamed per call. `classify_map_mode` pins `Direct` to `DEFAULT` plus `DYNAMIC` alone.
//! `may_trust_lock_bounds` is pinned over every pool/usage pair that reaches `Staged`, under
//! both `buffer.ignoreLockBounds` positions: off it trusts every announcement, on it trusts
//! none of them, `MANAGED` and `DYNAMIC` included. The two cases where there is no narrower
//! announcement to disbelieve (`D3DLOCK_DISCARD` and a zero `SizeToLock`) and the flags that
//! are no part of the rule are pinned in both positions too. `records_dirty_range` is pinned
//! per pool: `D3DLOCK_READONLY` suppresses the range only on `MANAGED`,
//! `D3DLOCK_NO_DIRTY_UPDATE` suppresses it nowhere, and any other lock
//! records its range in every pool.

use mtld3d_types::{D3DLOCK_NO_DIRTY_UPDATE, D3DUSAGE_WRITEONLY};

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

use mtld3d_types::{D3DPOOL_MANAGED, D3DPOOL_SCRATCH, D3DPOOL_SYSTEMMEM};

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

// ── may_trust_lock_bounds ──

/// The two `buffer.ignoreLockBounds` positions, named so the matrix rows read as prose.
const TRUST: bool = false;
const IGNORE: bool = true;
const SOME_SIZE: u32 = 1024;

/// Every pool/usage pair that reaches `Staged`, under both knob positions.
///
/// Off, the shipped default, every announcement binds. On, none does: the
/// rule has no pool or usage term, so `MANAGED` and `DYNAMIC` widen to the
/// whole buffer with everything else. The third column is the expected
/// answer with the knob on, written out per row rather than recomputed, so
/// a pool or usage carve-out reappearing in the predicate fails here; and
/// `classify_map_mode` is asserted per row so a row that stopped being
/// `Staged` fails here instead of quietly testing nothing.
#[test]
fn pool_usage_matrix_under_both_knob_positions() {
    const DYNAMIC_WRITEONLY: u32 = D3DUSAGE_DYNAMIC | D3DUSAGE_WRITEONLY;
    for (pool, usage, want_when_ignoring) in [
        // DEFAULT + DYNAMIC (with or without WRITEONLY) is `Direct`, so it
        // is absent: a `Direct` buffer shares its allocation with the GPU
        // and never consults a dirty range at all.
        (D3DPOOL_DEFAULT, NO_USAGE, false),
        (D3DPOOL_DEFAULT, D3DUSAGE_WRITEONLY, false),
        (D3DPOOL_MANAGED, NO_USAGE, false),
        (D3DPOOL_MANAGED, D3DUSAGE_WRITEONLY, false),
        (D3DPOOL_MANAGED, D3DUSAGE_DYNAMIC, false),
        (D3DPOOL_MANAGED, DYNAMIC_WRITEONLY, false),
        (D3DPOOL_SYSTEMMEM, NO_USAGE, false),
        (D3DPOOL_SYSTEMMEM, D3DUSAGE_WRITEONLY, false),
        (D3DPOOL_SYSTEMMEM, D3DUSAGE_DYNAMIC, false),
        (D3DPOOL_SYSTEMMEM, DYNAMIC_WRITEONLY, false),
        (D3DPOOL_SCRATCH, NO_USAGE, false),
        (D3DPOOL_SCRATCH, D3DUSAGE_WRITEONLY, false),
        (D3DPOOL_SCRATCH, D3DUSAGE_DYNAMIC, false),
        (D3DPOOL_SCRATCH, DYNAMIC_WRITEONLY, false),
    ] {
        assert_eq!(
            classify_map_mode(usage, pool),
            BufferMapMode::Staged,
            "row is not Staged: pool={pool} usage={usage:#x}"
        );
        assert!(
            may_trust_lock_bounds(DEFAULT_FLAGS, usage, pool, SOME_SIZE, TRUST),
            "knob off trusts every announcement: pool={pool} usage={usage:#x}"
        );
        assert_eq!(
            may_trust_lock_bounds(DEFAULT_FLAGS, usage, pool, SOME_SIZE, IGNORE),
            want_when_ignoring,
            "knob on: pool={pool} usage={usage:#x}"
        );
    }
}

/// A zero `SizeToLock` names no narrower window, in either knob position.
///
/// D3D9 documents `OffsetToLock` and `SizeToLock` both zero as locking the
/// entire buffer, so there is no announcement here to disbelieve and the
/// term is unconditional rather than part of the opt-in. Widening to
/// `[0, logical_len)` also settles the undocumented `(offset > 0, 0)` form,
/// which would otherwise upload `[offset, logical_len)` and leave the head
/// of the buffer carrying whatever the previous upload left.
#[test]
fn zero_size_to_lock_is_never_trusted() {
    for (usage, pool) in [
        (NO_USAGE, D3DPOOL_MANAGED),
        (D3DUSAGE_DYNAMIC, D3DPOOL_SYSTEMMEM),
        (NO_USAGE, D3DPOOL_DEFAULT),
    ] {
        for knob in [TRUST, IGNORE] {
            assert!(
                !may_trust_lock_bounds(DEFAULT_FLAGS, usage, pool, 0, knob),
                "pool={pool} usage={usage:#x} knob={knob}"
            );
        }
    }
}

/// `D3DLOCK_DISCARD` abandons the old bytes, in either knob position.
///
/// D3D9 documents the flag as discarding the whole buffer for a vertex or
/// index buffer, whatever range accompanies it, so like a zero
/// `SizeToLock` it leaves no announcement to disbelieve and the term is
/// unconditional. Reaching it on a `Staged` buffer means the title used
/// the flag outside the `D3DUSAGE_DYNAMIC` it is defined for, which
/// `vb_lock` warns about separately; taking the whole buffer is the
/// reading that matches what the flag claims.
#[test]
fn discard_always_voids_the_announcement() {
    for (usage, pool) in [
        (NO_USAGE, D3DPOOL_MANAGED),
        (D3DUSAGE_DYNAMIC, D3DPOOL_SYSTEMMEM),
        (D3DUSAGE_WRITEONLY, D3DPOOL_DEFAULT),
    ] {
        for knob in [TRUST, IGNORE] {
            assert!(
                !may_trust_lock_bounds(D3DLOCK_DISCARD, usage, pool, SOME_SIZE, knob),
                "pool={pool} usage={usage:#x} knob={knob}"
            );
        }
    }
}

/// Only `D3DLOCK_DISCARD` is part of the rule; the other flags leave it alone.
///
/// `D3DLOCK_READONLY` and `D3DLOCK_NO_DIRTY_UPDATE` decide whether a range
/// is recorded at all, which is `records_dirty_range`'s question, not this
/// one; an unrecognised high bit is ignored here as everywhere.
#[test]
fn unrelated_lock_flags_do_not_void_the_announcement() {
    const UNKNOWN_BIT: u32 = 0x8000_0000;
    for flags in [
        D3DLOCK_NOOVERWRITE,
        D3DLOCK_READONLY,
        D3DLOCK_NO_DIRTY_UPDATE,
        UNKNOWN_BIT,
    ] {
        for (usage, pool) in [
            (NO_USAGE, D3DPOOL_MANAGED),
            (D3DUSAGE_DYNAMIC, D3DPOOL_SYSTEMMEM),
            (D3DUSAGE_WRITEONLY, D3DPOOL_DEFAULT),
        ] {
            assert!(
                may_trust_lock_bounds(flags, usage, pool, SOME_SIZE, TRUST),
                "knob off: flags={flags:#x} pool={pool} usage={usage:#x}"
            );
            assert!(
                !may_trust_lock_bounds(flags, usage, pool, SOME_SIZE, IGNORE),
                "knob on: flags={flags:#x} pool={pool} usage={usage:#x}"
            );
        }
    }
}

// ── records_dirty_range ──

/// Every `D3DPOOL_*` value a `Staged` buffer can be created in.
const STAGED_POOLS: [u32; 4] = [
    D3DPOOL_MANAGED,
    D3DPOOL_DEFAULT,
    D3DPOOL_SYSTEMMEM,
    D3DPOOL_SCRATCH,
];

/// A lock that names neither flag records its range in every pool.
#[test]
fn a_plain_lock_records_its_range_in_every_pool() {
    for pool in STAGED_POOLS {
        for flags in [DEFAULT_FLAGS, D3DLOCK_NOOVERWRITE, D3DLOCK_DISCARD] {
            assert!(
                records_dirty_range(flags, pool),
                "pool={pool} flags={flags:#x}"
            );
        }
    }
}

/// `D3DLOCK_READONLY` suppresses the range only in `D3DPOOL_MANAGED`.
///
/// Managed is the pool that keeps a system-memory master copy, so skipping
/// the upload is what the promise buys. Elsewhere there is no second copy
/// to read back, the flag says nothing about what the device buffer needs,
/// and a title that writes under it anyway is still carried.
#[test]
fn read_only_is_honoured_only_by_the_managed_pool() {
    assert!(!records_dirty_range(D3DLOCK_READONLY, D3DPOOL_MANAGED));
    for pool in [D3DPOOL_DEFAULT, D3DPOOL_SYSTEMMEM, D3DPOOL_SCRATCH] {
        assert!(records_dirty_range(D3DLOCK_READONLY, pool), "pool={pool}");
    }
}

/// `D3DLOCK_NO_DIRTY_UPDATE` never suppresses the range, in any pool.
///
/// The flag asks that the region stay out of the dirty record, but it is
/// not a promise that nothing was written, so honouring it here would drop
/// the write: this path uploads at `Unlock` and clears the range, with no
/// later draw-time upload to carry the bytes. The conformance suite pins
/// it: a `MANAGED` buffer filled under no flags and refilled under this
/// flag, with no draw in between, must draw the second fill. Honouring the
/// flag drew the first.
#[test]
fn no_dirty_update_never_suppresses_the_range() {
    for pool in [
        D3DPOOL_DEFAULT,
        D3DPOOL_MANAGED,
        D3DPOOL_SYSTEMMEM,
        D3DPOOL_SCRATCH,
    ] {
        assert!(
            records_dirty_range(D3DLOCK_NO_DIRTY_UPDATE, pool),
            "pool={pool}"
        );
    }
}

/// With both flags set only the READONLY half suppresses, and only on `MANAGED`.
#[test]
fn read_only_and_no_dirty_update_together_suppress_only_on_managed() {
    const BOTH: u32 = D3DLOCK_READONLY | D3DLOCK_NO_DIRTY_UPDATE;
    assert!(!records_dirty_range(BOTH, D3DPOOL_MANAGED));
    for pool in [D3DPOOL_DEFAULT, D3DPOOL_SYSTEMMEM, D3DPOOL_SCRATCH] {
        assert!(records_dirty_range(BOTH, pool), "pool={pool}");
    }
}
