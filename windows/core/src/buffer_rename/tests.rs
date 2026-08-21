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
