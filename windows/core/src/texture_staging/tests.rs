//! Unit tests for the texture-staging decisions.
//!
//! Several cases per arm of `decide_lock_action`: flag priority, the uncontended shortcut,
//! `D3DLOCK_DISCARD` where it is honoured (a whole-mip Lock of a default-pool texture) and
//! where it is dropped (a partial Lock, the managed and system-memory pools), and whole-mip
//! renames preserving whichever pool the texture is in. The partial arms mostly resolve to
//! `WriteInPlace`; only an unaligned compressed rect forces a preserve, since the encoder widens
//! its read to the whole mip. The `honoured_lock_flags` cases pin the two conditions on their
//! own. The `texture_lock_offset` cases pin block-row arithmetic: a pixel-row index times a
//! block-row pitch runs past the end. The `staging_droppable_class` cases walk the pool, usage
//! and shape combinations, since every class outside the one that releases is a level whose only
//! copy of some byte is the staging.

use mtld3d_types::{D3DPOOL_MANAGED, D3DPOOL_SYSTEMMEM};

use super::*;

const DEFAULT_FLAGS: u32 = 0;
const NO_USAGE: u32 = 0;
const RETIRED_SEQ: u64 = 10;
const IN_FLIGHT_SEQ: u64 = 20;
const COHERENT: u64 = 15;

// 256×256 mip in pixel coords; uncompressed unless overridden.
const MIP_W: u32 = 256;
const MIP_H: u32 = 256;

fn rect(x: u32, y: u32, w: u32, h: u32) -> DirtyRect {
    DirtyRect { x, y, w, h }
}

fn full() -> DirtyRect {
    DirtyRect::full(MIP_W, MIP_H)
}

fn shape(block: (u32, u32)) -> MipShape {
    MipShape {
        mip_w: MIP_W,
        mip_h: MIP_H,
        block_w: block.0,
        block_h: block.1,
    }
}

fn decide(
    flags: u32,
    pool: u32,
    rect: Option<DirtyRect>,
    slot: u64,
    coh: u64,
    block: (u32, u32),
) -> LockAction {
    decide_lock_action(coh, slot, flags, pool, rect, shape(block))
}

// ── flag-priority arms (uncompressed) ──

#[test]
fn readonly_in_place_contended() {
    assert_eq!(
        decide(
            D3DLOCK_READONLY,
            D3DPOOL_DEFAULT,
            Some(rect(8, 8, 16, 16)),
            IN_FLIGHT_SEQ,
            COHERENT,
            (1, 1)
        ),
        LockAction::WriteInPlace
    );
}

#[test]
fn nooverwrite_in_place_contended() {
    assert_eq!(
        decide(
            D3DLOCK_NOOVERWRITE,
            D3DPOOL_DEFAULT,
            Some(rect(8, 8, 16, 16)),
            IN_FLIGHT_SEQ,
            COHERENT,
            (1, 1)
        ),
        LockAction::WriteInPlace
    );
}

#[test]
fn nooverwrite_wins_over_discard() {
    assert_eq!(
        decide(
            D3DLOCK_DISCARD | D3DLOCK_NOOVERWRITE,
            D3DPOOL_DEFAULT,
            None,
            IN_FLIGHT_SEQ,
            COHERENT,
            (1, 1)
        ),
        LockAction::WriteInPlace
    );
}

// ── seq arms ──

#[test]
fn never_uploaded_in_place() {
    assert_eq!(
        decide(
            DEFAULT_FLAGS,
            D3DPOOL_DEFAULT,
            Some(rect(8, 8, 16, 16)),
            0,
            0,
            (1, 1)
        ),
        LockAction::WriteInPlace
    );
}

#[test]
fn equal_seqs_in_place() {
    // slot == coh ⇒ retired ⇒ uncontended.
    assert_eq!(
        decide(
            DEFAULT_FLAGS,
            D3DPOOL_DEFAULT,
            Some(rect(8, 8, 16, 16)),
            COHERENT,
            COHERENT,
            (1, 1)
        ),
        LockAction::WriteInPlace
    );
}

#[test]
fn retired_slot_in_place_even_with_default_flags() {
    assert_eq!(
        decide(
            DEFAULT_FLAGS,
            D3DPOOL_DEFAULT,
            Some(rect(8, 8, 16, 16)),
            RETIRED_SEQ,
            COHERENT,
            (1, 1)
        ),
        LockAction::WriteInPlace
    );
}

#[test]
fn uncontended_whole_mip_discard_in_place() {
    // Nothing reads the staging, so even an honoured DISCARD keeps the box:
    // the game overwrites it in place and no allocation is needed.
    assert_eq!(
        decide(
            D3DLOCK_DISCARD,
            D3DPOOL_DEFAULT,
            None,
            RETIRED_SEQ,
            COHERENT,
            (1, 1)
        ),
        LockAction::WriteInPlace
    );
}

// ── DISCARD arms ──

#[test]
fn discard_full_mip_contended_freshbox_none() {
    assert_eq!(
        decide(
            D3DLOCK_DISCARD,
            D3DPOOL_DEFAULT,
            None,
            IN_FLIGHT_SEQ,
            COHERENT,
            (1, 1)
        ),
        LockAction::FreshBox {
            preserve: PreserveKind::None
        }
    );
}

#[test]
fn discard_full_rect_contended_freshbox_none() {
    // An explicit rect spanning the mip is a whole-mip Lock too.
    assert_eq!(
        decide(
            D3DLOCK_DISCARD,
            D3DPOOL_DEFAULT,
            Some(full()),
            IN_FLIGHT_SEQ,
            COHERENT,
            (1, 1)
        ),
        LockAction::FreshBox {
            preserve: PreserveKind::None
        }
    );
}

#[test]
fn discard_partial_contended_is_dropped() {
    // A partial DISCARD cannot be honoured: the bytes outside the rect stay
    // part of the level, so the Lock is served as a plain partial one.
    assert_eq!(
        decide(
            D3DLOCK_DISCARD,
            D3DPOOL_DEFAULT,
            Some(rect(8, 8, 16, 16)),
            IN_FLIGHT_SEQ,
            COHERENT,
            (1, 1)
        ),
        LockAction::WriteInPlace
    );
}

#[test]
fn discard_compressed_unaligned_freshbox_cpu() {
    // Dropped DISCARD, then the alignment guard: the encoder's full-mip
    // fallback reads outside the rect, so the old bytes are carried over.
    assert_eq!(
        decide(
            D3DLOCK_DISCARD,
            D3DPOOL_DEFAULT,
            Some(rect(2, 2, 13, 13)),
            IN_FLIGHT_SEQ,
            COHERENT,
            (4, 4)
        ),
        LockAction::FreshBox {
            preserve: PreserveKind::Cpu
        }
    );
}

#[test]
fn discard_whole_mip_managed_contended_freshbox_cpu() {
    // A managed level is served from its staging for its whole life, so
    // its DISCARD is dropped and the rename preserves.
    assert_eq!(
        decide(
            D3DLOCK_DISCARD,
            D3DPOOL_MANAGED,
            None,
            IN_FLIGHT_SEQ,
            COHERENT,
            (1, 1)
        ),
        LockAction::FreshBox {
            preserve: PreserveKind::Cpu
        }
    );
}

#[test]
fn discard_whole_mip_systemmem_contended_freshbox_cpu() {
    assert_eq!(
        decide(
            D3DLOCK_DISCARD,
            D3DPOOL_SYSTEMMEM,
            None,
            IN_FLIGHT_SEQ,
            COHERENT,
            (1, 1)
        ),
        LockAction::FreshBox {
            preserve: PreserveKind::Cpu
        }
    );
}

// ── whole-mip arms ──

#[test]
fn whole_mip_default_contended_freshbox_cpu() {
    assert_eq!(
        decide(
            DEFAULT_FLAGS,
            D3DPOOL_DEFAULT,
            Some(full()),
            IN_FLIGHT_SEQ,
            COHERENT,
            (1, 1)
        ),
        LockAction::FreshBox {
            preserve: PreserveKind::Cpu
        }
    );
}

#[test]
fn whole_mip_managed_contended_freshbox_cpu() {
    assert_eq!(
        decide(
            DEFAULT_FLAGS,
            D3DPOOL_MANAGED,
            Some(full()),
            IN_FLIGHT_SEQ,
            COHERENT,
            (1, 1)
        ),
        LockAction::FreshBox {
            preserve: PreserveKind::Cpu
        }
    );
}

#[test]
fn whole_mip_via_none_rect_freshbox_cpu() {
    assert_eq!(
        decide(
            DEFAULT_FLAGS,
            D3DPOOL_DEFAULT,
            None,
            IN_FLIGHT_SEQ,
            COHERENT,
            (1, 1)
        ),
        LockAction::FreshBox {
            preserve: PreserveKind::Cpu
        }
    );
}

// ── partial uncompressed: narrowing arms ──

#[test]
fn partial_default_contended_in_place() {
    // The headline narrowing: previously fresh+preserve_cpu, now
    // WriteInPlace under the no-overlap contract.
    assert_eq!(
        decide(
            DEFAULT_FLAGS,
            D3DPOOL_DEFAULT,
            Some(rect(8, 8, 16, 16)),
            IN_FLIGHT_SEQ,
            COHERENT,
            (1, 1)
        ),
        LockAction::WriteInPlace
    );
}

#[test]
fn partial_managed_contended_in_place() {
    assert_eq!(
        decide(
            DEFAULT_FLAGS,
            D3DPOOL_MANAGED,
            Some(rect(8, 8, 16, 16)),
            IN_FLIGHT_SEQ,
            COHERENT,
            (1, 1)
        ),
        LockAction::WriteInPlace
    );
}

// ── partial compressed: alignment guard arms ──

#[test]
fn compressed_aligned_partial_contended_in_place() {
    // Block-aligned partial (origin + size both multiples of 4)
    // — encoder won't fall back; narrowing applies.
    assert_eq!(
        decide(
            DEFAULT_FLAGS,
            D3DPOOL_DEFAULT,
            Some(rect(8, 8, 16, 16)),
            IN_FLIGHT_SEQ,
            COHERENT,
            (4, 4)
        ),
        LockAction::WriteInPlace
    );
}

#[test]
fn compressed_aligned_partial_managed_in_place() {
    assert_eq!(
        decide(
            DEFAULT_FLAGS,
            D3DPOOL_MANAGED,
            Some(rect(8, 8, 16, 16)),
            IN_FLIGHT_SEQ,
            COHERENT,
            (4, 4)
        ),
        LockAction::WriteInPlace
    );
}

#[test]
fn compressed_unaligned_origin_freshbox_cpu() {
    // origin off-grid (2, 2) → encoder fallback → must preserve.
    assert_eq!(
        decide(
            DEFAULT_FLAGS,
            D3DPOOL_DEFAULT,
            Some(rect(2, 2, 16, 16)),
            IN_FLIGHT_SEQ,
            COHERENT,
            (4, 4)
        ),
        LockAction::FreshBox {
            preserve: PreserveKind::Cpu
        }
    );
}

#[test]
fn compressed_unaligned_size_freshbox_cpu() {
    // size off-grid (13×13) and not reaching mip edge → fallback.
    assert_eq!(
        decide(
            DEFAULT_FLAGS,
            D3DPOOL_DEFAULT,
            Some(rect(8, 8, 13, 13)),
            IN_FLIGHT_SEQ,
            COHERENT,
            (4, 4)
        ),
        LockAction::FreshBox {
            preserve: PreserveKind::Cpu
        }
    );
}

#[test]
fn compressed_unaligned_managed_freshbox_cpu() {
    // The pool does not change the guard: the encoder fallback reads
    // outside-rect bytes, uninit there would corrupt the MTLTexture.
    assert_eq!(
        decide(
            DEFAULT_FLAGS,
            D3DPOOL_MANAGED,
            Some(rect(2, 2, 13, 13)),
            IN_FLIGHT_SEQ,
            COHERENT,
            (4, 4)
        ),
        LockAction::FreshBox {
            preserve: PreserveKind::Cpu
        }
    );
}

#[test]
fn compressed_partial_extends_to_mip_edge_in_place() {
    // The right-edge clause `r.x + r.w == mip_w` only matters when `r.w`
    // is not a multiple of `block_w`, which on a power-of-two mip with an
    // aligned origin needs a mip that is itself not block-aligned — and
    // such a rect is typically whole-mip, so the earlier whole-mip arm
    // claims it first. Exercising the clause therefore needs a
    // non-power-of-two mip and a strictly partial rect, driving the
    // helper directly so mip_w / mip_h can vary.
    //
    // 6×8 DXT mip, rect (0, 0, 6, 4): width 6 reaches mip_w=6 without
    // being a multiple of 4 (right-edge clause), height 4 is
    // block-aligned, and the rect covers only the top half — partial, so
    // it lands in the alignment-guard arm. Expect WriteInPlace.
    assert_eq!(
        decide_lock_action(
            COHERENT,
            IN_FLIGHT_SEQ,
            DEFAULT_FLAGS,
            D3DPOOL_DEFAULT,
            Some(DirtyRect {
                x: 0,
                y: 0,
                w: 6,
                h: 4,
            }),
            MipShape {
                mip_w: 6,
                mip_h: 8,
                block_w: 4,
                block_h: 4,
            },
        ),
        LockAction::WriteInPlace,
        "right-edge tolerance: r.x + r.w == mip_w with r.w % bw != 0"
    );
    // Height-edge mirror: 8×6 mip, rect covers right half of top
    // half. r.h=6 reaches mip_h=6 with non-multiple-of-4 height.
    assert_eq!(
        decide_lock_action(
            COHERENT,
            IN_FLIGHT_SEQ,
            DEFAULT_FLAGS,
            D3DPOOL_DEFAULT,
            Some(DirtyRect {
                x: 0,
                y: 0,
                w: 4,
                h: 6,
            }),
            MipShape {
                mip_w: 8,
                mip_h: 6,
                block_w: 4,
                block_h: 4,
            },
        ),
        LockAction::WriteInPlace,
        "bottom-edge tolerance: r.y + r.h == mip_h with r.h % bh != 0"
    );
}

#[test]
fn compressed_uncontended_unaligned_in_place() {
    // No in-flight read to race with — same outcome as today's
    // code. Encoder will still fall back, but no GPU work is
    // currently reading the staging.
    assert_eq!(
        decide(
            DEFAULT_FLAGS,
            D3DPOOL_DEFAULT,
            Some(rect(2, 2, 13, 13)),
            RETIRED_SEQ,
            COHERENT,
            (4, 4)
        ),
        LockAction::WriteInPlace
    );
}

// ── unknown bits / hygiene ──

#[test]
fn unknown_high_bits_pass_through() {
    let unknown = 0x8000_0000;
    assert_eq!(
        decide(
            unknown | D3DLOCK_DISCARD,
            D3DPOOL_DEFAULT,
            None,
            IN_FLIGHT_SEQ,
            COHERENT,
            (1, 1)
        ),
        LockAction::FreshBox {
            preserve: PreserveKind::None
        }
    );
    assert_eq!(
        decide(
            unknown,
            D3DPOOL_DEFAULT,
            Some(rect(8, 8, 16, 16)),
            IN_FLIGHT_SEQ,
            COHERENT,
            (1, 1)
        ),
        LockAction::WriteInPlace
    );
}

// ── honoured_lock_flags ──

#[test]
fn whole_mip_default_pool_keeps_discard() {
    let flags = D3DLOCK_DISCARD | mtld3d_types::D3DLOCK_NOSYSLOCK;
    assert_eq!(
        honoured_lock_flags(flags, D3DPOOL_DEFAULT, None, shape((1, 1))),
        flags
    );
    assert_eq!(
        honoured_lock_flags(flags, D3DPOOL_DEFAULT, Some(full()), shape((1, 1))),
        flags
    );
}

#[test]
fn partial_lock_drops_discard_and_keeps_the_rest() {
    let flags = D3DLOCK_DISCARD | D3DLOCK_NOOVERWRITE | mtld3d_types::D3DLOCK_NOSYSLOCK;
    assert_eq!(
        honoured_lock_flags(
            flags,
            D3DPOOL_DEFAULT,
            Some(rect(8, 8, 16, 16)),
            shape((1, 1))
        ),
        D3DLOCK_NOOVERWRITE | mtld3d_types::D3DLOCK_NOSYSLOCK
    );
}

#[test]
fn cpu_pools_drop_discard_even_whole_mip() {
    for pool in [
        D3DPOOL_MANAGED,
        D3DPOOL_SYSTEMMEM,
        mtld3d_types::D3DPOOL_SCRATCH,
    ] {
        assert_eq!(
            honoured_lock_flags(D3DLOCK_DISCARD, pool, None, shape((1, 1))),
            DEFAULT_FLAGS,
            "pool {pool}"
        );
    }
}

#[test]
fn flags_without_discard_pass_through_unchanged() {
    let flags = D3DLOCK_READONLY | mtld3d_types::D3DLOCK_NO_DIRTY_UPDATE;
    assert_eq!(
        honoured_lock_flags(
            flags,
            D3DPOOL_MANAGED,
            Some(rect(0, 0, 8, 8)),
            shape((1, 1))
        ),
        flags
    );
}

#[test]
fn a_rect_reaching_past_the_mip_is_whole() {
    // `parse_rect` clamps, so an oversized rect is the whole mip; the
    // tolerance keeps an unclamped one from being read as partial.
    assert!(is_whole_mip(
        Some(rect(0, 0, MIP_W + 8, MIP_H)),
        shape((1, 1))
    ));
    assert!(!is_whole_mip(Some(rect(1, 0, MIP_W, MIP_H)), shape((1, 1))));
    assert!(!is_whole_mip(
        Some(rect(0, 0, MIP_W - 1, MIP_H)),
        shape((1, 1))
    ));
}

// ── texture_lock_offset ──

fn off_rect(x: u32, y: u32, w: u32, h: u32) -> DirtyRect {
    DirtyRect { x, y, w, h }
}

#[test]
fn full_mip_lock_returns_zero_offset() {
    // None rect ⇒ full-mip lock; any pitch / block-shape returns 0.
    assert_eq!(texture_lock_offset(None, 1024, 1, 1, 4), 0);
    assert_eq!(texture_lock_offset(None, 2048, 4, 4, 8), 0);
}

#[test]
fn uncompressed_offset_matches_pixel_math() {
    // BGRA8: 1×1 block, 4 bytes per "block" (= 1 pixel). 256-wide mip,
    // pitch = 1024. Lock at row 5, x=8 → 5*1024 + 8*4 = 5152.
    let p = 1024;
    let off = texture_lock_offset(Some(off_rect(8, 5, 16, 16)), p, 1, 1, 4);
    assert_eq!(off, 5 * 1024 + 8 * 4);
}

#[test]
fn dxt1_offset_uses_block_rows_not_pixel_rows() {
    // 512×512 DXT1 mip: 128×128 blocks, 8 bytes per block, pitch =
    // 128 × 8 = 1024 bytes per block-row. Total staging = 131072.
    // Locking pixel-y=128 (= block-row 32) at pixel-x=256 (= block-col
    // 64) MUST yield offset = 32 * 1024 + 64 * 8 = 33280.
    //
    // A pixel-row × block-pitch formula would instead land at
    // 128 * 1024 = 131072 — exactly the Box's len, i.e. one-past-the-end,
    // so the game's first write through the Lock pointer runs off the
    // allocation. Pinning the block-coordinate formula here flips this
    // test red at that signature.
    let pitch = 1024; // bytes per block-row for 512-wide DXT1
    let off = texture_lock_offset(Some(off_rect(256, 128, 256, 64)), pitch, 4, 4, 8);
    assert_eq!(off, (128 / 4) * 1024 + (256 / 4) * 8);
    assert_eq!(off, 33280);
    // Sanity: the Box is 131072 bytes; the Lock window must end
    // strictly before that to be writable.
    let staging_len = 131_072;
    let locked_block_h = 64usize.div_ceil(4);
    let last_byte = off + locked_block_h * pitch as usize;
    assert!(last_byte <= staging_len);
}

#[test]
fn dxt5_offset_uses_block_rows_not_pixel_rows() {
    // 256×256 DXT5: 64×64 blocks, 16 bytes per block, pitch = 1024.
    // Lock at pixel-y=64 (block-row 16), pixel-x=32 (block-col 8) →
    // 16 * 1024 + 8 * 16 = 16512.
    let off = texture_lock_offset(Some(off_rect(32, 64, 64, 64)), 1024, 4, 4, 16);
    assert_eq!(off, (64 / 4) * 1024 + (32 / 4) * 16);
    assert_eq!(off, 16512);
}

#[test]
fn block_aligned_lock_origin_is_unchanged_under_division() {
    // When pixel coords are exact multiples of the block dims,
    // `r.y / block_h` and `r.x / block_w` lose nothing — the offset
    // matches what direct block-coord math would give.
    let pitch = 2048;
    let pixel = texture_lock_offset(Some(off_rect(16, 8, 16, 4)), pitch, 4, 4, 16);
    assert_eq!(pixel, (8 / 4) * 2048 + (16 / 4) * 16);
}

#[test]
fn a_plain_lock_of_a_released_level_reads_it_back() {
    assert!(released_level_lock_needs_readback(DEFAULT_FLAGS));
    assert!(released_level_lock_needs_readback(D3DLOCK_READONLY));
    assert!(released_level_lock_needs_readback(D3DLOCK_NOOVERWRITE));
    assert!(released_level_lock_needs_readback(
        mtld3d_types::D3DLOCK_NO_DIRTY_UPDATE
    ));
}

// ── staging_droppable_class ──

/// The only class that releases: default pool, no usage bit, plain 2D.
#[test]
fn plain_default_pool_texture_is_droppable() {
    assert!(staging_droppable_class(
        D3DPOOL_DEFAULT,
        NO_USAGE,
        TextureFlags::empty(),
        1
    ));
}

#[test]
fn discard_is_the_only_flag_that_skips_the_readback() {
    assert!(!released_level_lock_needs_readback(D3DLOCK_DISCARD));
    // Companion bits ride along with DISCARD without reviving the read.
    assert!(!released_level_lock_needs_readback(
        D3DLOCK_DISCARD | D3DLOCK_NOOVERWRITE
    ));
    assert!(!released_level_lock_needs_readback(
        D3DLOCK_DISCARD | mtld3d_types::D3DLOCK_NOSYSLOCK
    ));
}

#[test]
fn lockable_pools_are_never_droppable() {
    for pool in [
        D3DPOOL_MANAGED,
        D3DPOOL_SYSTEMMEM,
        mtld3d_types::D3DPOOL_SCRATCH,
    ] {
        assert!(
            !staging_droppable_class(pool, NO_USAGE, TextureFlags::empty(), 1),
            "pool {pool}"
        );
    }
}

#[test]
fn lockable_or_gpu_written_usage_is_never_droppable() {
    for usage in [
        D3DUSAGE_DYNAMIC,
        D3DUSAGE_RENDERTARGET,
        D3DUSAGE_DEPTHSTENCIL,
    ] {
        assert!(
            !staging_droppable_class(D3DPOOL_DEFAULT, usage, TextureFlags::empty(), 1),
            "usage {usage:#x}"
        );
    }
}

/// A volume texture is out of the class at any depth, a single slice included.
///
/// `depth` alone does not identify one: `CreateVolumeTexture` with a depth of
/// 1 is created 2D on both sides, and would otherwise pass the shape test the
/// way an ordinary 2D texture does.
#[test]
fn volume_textures_are_never_droppable() {
    for depth in [1, 2, 16] {
        assert!(
            !staging_droppable_class(
                D3DPOOL_DEFAULT,
                NO_USAGE,
                TextureFlags::VOLUME_TEXTURE,
                depth
            ),
            "depth {depth}"
        );
    }
}

#[test]
fn cube_offscreen_plain_and_depth_textures_are_never_droppable() {
    for flag in [
        TextureFlags::CUBE,
        TextureFlags::OFFSCREEN_PLAIN,
        TextureFlags::DEPTH_FORMAT,
    ] {
        assert!(
            !staging_droppable_class(D3DPOOL_DEFAULT, NO_USAGE, flag, 1),
            "flag {flag:?}"
        );
    }
}

/// A flag the class does not read leaves it alone.
#[test]
fn autogen_mipmap_alone_stays_droppable() {
    assert!(staging_droppable_class(
        D3DPOOL_DEFAULT,
        NO_USAGE,
        TextureFlags::AUTOGEN_MIPMAP,
        1
    ));
}

/// Multi-slice depth still fails the shape test on its own.
#[test]
fn multi_slice_depth_is_never_droppable() {
    assert!(!staging_droppable_class(
        D3DPOOL_DEFAULT,
        NO_USAGE,
        TextureFlags::empty(),
        4
    ));
}
