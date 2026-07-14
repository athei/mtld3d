//! Decision helper for per-mip texture-staging Lock handling.
//!
//! Mirrors `crate::buffer_rename::plan_lock`'s structure for textures.
//! Same well-behaved-game no-overlap contract VB/IB now relies on —
//! the locked sub-rect doesn't overlap with bytes any in-flight blit
//! reads. Same contract that non-persistent mapped-buffer APIs (e.g.
//! OpenGL `glBufferSubData`) make implicitly; UI atlas regen via
//! whole-mip Locks and DISCARD-heavy geometry texture uploads both
//! satisfy it in practice.
//!
//! Decision tree:
//! - `D3DLOCK_NOOVERWRITE` / `D3DLOCK_READONLY`, or uncontended
//!   (`last_submit_seq <= coherent_seq`): `WriteInPlace`.
//! - `D3DLOCK_DISCARD`: Rename, no preserve (game promised the old
//!   bytes are gone).
//! - Whole-mip contended: Rename. `D3DUSAGE_DYNAMIC` → no preserve;
//!   non-DYNAMIC → CPU-memcpy preserve (game might read all bytes
//!   through the Lock pointer — only possible when the Lock covers
//!   the full mip).
//! - Partial contended, compressed AND not block-aligned: Rename + Cpu
//!   preserve (always Cpu, even with DYNAMIC) — see
//!   `rect_block_aligned` for the formula and why preserve is forced.
//! - Partial contended, otherwise (uncompressed or block-aligned):
//!   `WriteInPlace`. Relies on the well-behaved-game no-overlap contract.
//!
//! Why DYNAMIC and not WRITEONLY: `plan_lock` keys the buffer
//! equivalent on `D3DUSAGE_WRITEONLY` because that's the
//! spec-documented "no readback" hint for `CreateVertexBuffer` /
//! `CreateIndexBuffer`. Microsoft does not document
//! `D3DUSAGE_WRITEONLY` for `CreateTexture`; `D3DUSAGE_DYNAMIC` is the
//! texture-side "frequently updated, no readback expected" hint
//! instead. It is also the prerequisite for legally passing
//! `D3DLOCK_DISCARD` on a texture, so games that care about the fast
//! path opt in via `DYNAMIC` either way.
//!
//! Why no GPU preserve path: `copyFromBuffer:toTexture:` only writes
//! the locked sub-rect, leaving prior `MTLTexture` pixels intact. The
//! GPU side preserves outside-rect pixels automatically. The Cpu
//! preserve only matters when the GAME reads bytes outside its
//! written rect through the Lock pointer (whole-mip locks), or when
//! the encoder's compressed-fallback is forced to read more bytes
//! than the rect (the alignment-guard arm).
//!
//! Side effects (allocate `PageBox`, sync memcpy preserve, queue
//! retention, bump perf counters) stay in `d3d9::texture`; this
//! module just returns a verdict.

use mtld3d_types::{D3DLOCK_DISCARD, D3DLOCK_NOOVERWRITE, D3DLOCK_READONLY, D3DUSAGE_DYNAMIC};

use crate::dirty_rect::DirtyRect;

/// What the caller should do with the old backing's contents on a rename.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreserveKind {
    /// No preserve needed.
    ///
    /// Either `D3DLOCK_DISCARD` was set, or the caller is a whole-mip
    /// `D3DUSAGE_DYNAMIC` Lock (the texture-side "no readback
    /// expected" hint; the encoder's blit only reads the locked rect).
    None,
    /// Caller must synchronously memcpy the old `PageBox` into the fresh allocation.
    ///
    /// The copy happens before returning the Lock pointer. Two cases:
    /// (a) whole-mip non-DYNAMIC contended (game might read all bytes
    /// through the pointer); (b) partial-but-unaligned compressed
    /// contended (encoder will fall back to a full-mip blit;
    /// outside-rect bytes must be valid).
    Cpu,
}

/// Verdict for a single `LockRect` call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockAction {
    /// Hand back a pointer into the existing `Arc<PageBox>`.
    ///
    /// Either uncontended, the caller promised no in-flight overlap
    /// (NOOVERWRITE / READONLY), or the partial sub-rect is small
    /// enough that the well-behaved-game contract holds.
    WriteInPlace,
    /// Swap the slot's `Arc<PageBox>` for a fresh allocation and apply `preserve`.
    ///
    /// Caller queues the old Arc clone for seq-gated retention via the
    /// encoder's `pending_blit_retention`.
    FreshBox { preserve: PreserveKind },
}

/// Geometry of a single mip.
///
/// The static-per-mip data `decide_lock_action` needs to classify a
/// Lock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MipShape {
    /// Mip pixel width.
    pub mip_w: u32,
    /// Mip pixel height.
    pub mip_h: u32,
    /// Format block width: `1` for uncompressed, `4` for DXT/BC.
    pub block_w: u32,
    /// Format block height: `1` for uncompressed, `4` for DXT/BC.
    pub block_h: u32,
}

/// Is the slot's last write still potentially being read by the GPU?
#[inline]
const fn is_in_flight(slot_last_submit_seq: u64, coherent_seq: u64) -> bool {
    slot_last_submit_seq > coherent_seq
}

/// Block-aligned in the sense the encoder requires.
///
/// The rect lands on the block grid, OR the rect's right/bottom edge
/// reaches the mip edge (the encoder tolerates that asymmetry because
/// the trailing blocks are partially-occupied at the mip boundary
/// anyway).
///
/// Mirrors the formula at `encoder.rs::run_texture_upload_blit` —
/// must stay in lock-step. If that file's check changes, this one
/// changes too. For uncompressed formats `block_w = block_h = 1`,
/// so the check trivially passes.
#[inline]
const fn rect_block_aligned(r: DirtyRect, shape: MipShape) -> bool {
    r.x.is_multiple_of(shape.block_w)
        && r.y.is_multiple_of(shape.block_h)
        && (r.w.is_multiple_of(shape.block_w) || r.x + r.w == shape.mip_w)
        && (r.h.is_multiple_of(shape.block_h) || r.y + r.h == shape.mip_h)
}

/// Decide the Lock action for a single mip.
///
/// - `coherent_seq` is the encoder thread's last retired submit seq.
/// - `slot_last_submit_seq` is the submit seq at which this mip's
///   staging was last referenced by a GPU-visible command. Zero if
///   never uploaded.
/// - `flags` is the raw `D3DLOCK_*` bitfield from the game.
/// - `usage` is the texture's `D3DUSAGE_*` bitfield captured at
///   `CreateTexture`.
/// - `rect` is the locked sub-rect. `None` ⇒ full-mip Lock.
/// - `shape` carries the mip + block dimensions (see [`MipShape`]).
///
/// Unknown flag bits are ignored here — the caller logs them via
/// `log_once_warn!`.
#[must_use]
pub const fn decide_lock_action(
    coherent_seq: u64,
    slot_last_submit_seq: u64,
    flags: u32,
    usage: u32,
    rect: Option<DirtyRect>,
    shape: MipShape,
) -> LockAction {
    if flags & (D3DLOCK_READONLY | D3DLOCK_NOOVERWRITE) != 0 {
        return LockAction::WriteInPlace;
    }
    if !is_in_flight(slot_last_submit_seq, coherent_seq) {
        return LockAction::WriteInPlace;
    }
    if flags & D3DLOCK_DISCARD != 0 {
        return LockAction::FreshBox {
            preserve: PreserveKind::None,
        };
    }

    // `>=` for tolerance; `parse_rect` already clamps so equality is
    // typical.
    let whole_mip = match rect {
        None => true,
        Some(r) => r.x == 0 && r.y == 0 && r.w >= shape.mip_w && r.h >= shape.mip_h,
    };
    if whole_mip {
        let preserve = if usage & D3DUSAGE_DYNAMIC != 0 {
            PreserveKind::None
        } else {
            PreserveKind::Cpu
        };
        return LockAction::FreshBox { preserve };
    }

    // Partial. Force the rename when the compressed alignment
    // formula would push the encoder into its full-mip-fallback path
    // — the GPU's read range becomes "all bytes" rather than the
    // rect, so we can't trust no-overlap and must preserve outside-
    // rect bytes regardless of DYNAMIC.
    if let Some(r) = rect
        && !rect_block_aligned(r, shape)
    {
        return LockAction::FreshBox {
            preserve: PreserveKind::Cpu,
        };
    }

    LockAction::WriteInPlace
}

/// Byte offset into a mip's staging Box for the start of the locked rect.
///
/// `pitch` is `mip_bytes_per_row` — for compressed formats this is
/// bytes-per-block-row, **not** bytes-per-pixel-row, so the rect's
/// pixel-space `x` and `y` must be converted to block coordinates before the
/// offset math. For uncompressed formats `block_w` and `block_h` are 1 and
/// `block_bytes == bytes_per_pixel`, so the same formula reduces to `r.y *
/// pitch + r.x * bpp`.
///
/// Returns `0` for a `None` rect (full-mip lock).
///
/// Kept as a pure function so the block-coordinate conversion has one home and
/// stays under test: a pixel-row index multiplied by a block-row pitch
/// overshoots a compressed mip's staging allocation by a factor of `block_h`
/// (4× for DXT) whenever `r.y > 0`, handing the game a Lock pointer past the
/// end of the allocation to write through.
#[must_use]
pub const fn texture_lock_offset(
    rect: Option<DirtyRect>,
    pitch: u32,
    block_w: u32,
    block_h: u32,
    block_bytes: u32,
) -> usize {
    match rect {
        Some(r) => {
            (r.y / block_h) as usize * pitch as usize
                + (r.x / block_w) as usize * block_bytes as usize
        }
        None => 0,
    }
}

#[cfg(test)]
mod tests {
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

    fn decide(
        flags: u32,
        usage: u32,
        rect: Option<DirtyRect>,
        slot: u64,
        coh: u64,
        block: (u32, u32),
    ) -> LockAction {
        decide_lock_action(
            coh,
            slot,
            flags,
            usage,
            rect,
            MipShape {
                mip_w: MIP_W,
                mip_h: MIP_H,
                block_w: block.0,
                block_h: block.1,
            },
        )
    }

    // ── flag-priority arms (uncompressed) ──

    #[test]
    fn readonly_in_place_contended() {
        assert_eq!(
            decide(
                D3DLOCK_READONLY,
                NO_USAGE,
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
                NO_USAGE,
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
                NO_USAGE,
                Some(rect(8, 8, 16, 16)),
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
                NO_USAGE,
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
                NO_USAGE,
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
                NO_USAGE,
                Some(rect(8, 8, 16, 16)),
                RETIRED_SEQ,
                COHERENT,
                (1, 1)
            ),
            LockAction::WriteInPlace
        );
    }

    // ── DISCARD arms ──

    #[test]
    fn discard_partial_contended_freshbox_none() {
        assert_eq!(
            decide(
                D3DLOCK_DISCARD,
                NO_USAGE,
                Some(rect(8, 8, 16, 16)),
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
    fn discard_full_mip_contended_freshbox_none() {
        assert_eq!(
            decide(
                D3DLOCK_DISCARD,
                NO_USAGE,
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
    fn discard_compressed_unaligned_freshbox_none() {
        // DISCARD short-circuits before the alignment check; bytes
        // don't matter when the game promised "throw it all out".
        assert_eq!(
            decide(
                D3DLOCK_DISCARD,
                NO_USAGE,
                Some(rect(2, 2, 13, 13)),
                IN_FLIGHT_SEQ,
                COHERENT,
                (4, 4)
            ),
            LockAction::FreshBox {
                preserve: PreserveKind::None
            }
        );
    }

    // ── whole-mip arms ──

    #[test]
    fn whole_mip_dynamic_contended_freshbox_none() {
        assert_eq!(
            decide(
                DEFAULT_FLAGS,
                D3DUSAGE_DYNAMIC,
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
    fn whole_mip_writeonly_alone_contended_freshbox_cpu() {
        // `D3DUSAGE_WRITEONLY` isn't documented for `CreateTexture`,
        // so we don't honour it as a no-readback hint here. Without
        // `D3DUSAGE_DYNAMIC` the lock falls into the preserve arm.
        // (Pin the contract — see module-doc.)
        use mtld3d_types::D3DUSAGE_WRITEONLY;
        assert_eq!(
            decide(
                DEFAULT_FLAGS,
                D3DUSAGE_WRITEONLY,
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
    fn whole_mip_default_contended_freshbox_cpu() {
        assert_eq!(
            decide(
                DEFAULT_FLAGS,
                NO_USAGE,
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
                NO_USAGE,
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
                NO_USAGE,
                Some(rect(8, 8, 16, 16)),
                IN_FLIGHT_SEQ,
                COHERENT,
                (1, 1)
            ),
            LockAction::WriteInPlace
        );
    }

    #[test]
    fn partial_dynamic_contended_in_place() {
        assert_eq!(
            decide(
                DEFAULT_FLAGS,
                D3DUSAGE_DYNAMIC,
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
                NO_USAGE,
                Some(rect(8, 8, 16, 16)),
                IN_FLIGHT_SEQ,
                COHERENT,
                (4, 4)
            ),
            LockAction::WriteInPlace
        );
    }

    #[test]
    fn compressed_aligned_partial_dynamic_in_place() {
        assert_eq!(
            decide(
                DEFAULT_FLAGS,
                D3DUSAGE_DYNAMIC,
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
                NO_USAGE,
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
                NO_USAGE,
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
    fn compressed_unaligned_dynamic_freshbox_cpu() {
        // DYNAMIC does NOT downgrade to None here — encoder
        // fallback reads outside-rect bytes; uninit there would
        // corrupt the MTLTexture. DYNAMIC is the game's promise,
        // not the GPU's.
        assert_eq!(
            decide(
                DEFAULT_FLAGS,
                D3DUSAGE_DYNAMIC,
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
                NO_USAGE,
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
                NO_USAGE,
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
                NO_USAGE,
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
                NO_USAGE,
                Some(rect(8, 8, 16, 16)),
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
                NO_USAGE,
                Some(rect(8, 8, 16, 16)),
                IN_FLIGHT_SEQ,
                COHERENT,
                (1, 1)
            ),
            LockAction::WriteInPlace
        );
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
}
