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

/// Whether a Lock of a released level has to read the level back from the GPU.
///
/// A default-pool level whose staging was released keeps its only copy on the
/// GPU, and D3D9 hands a `LockRect` the level's current contents. So the
/// released staging is re-created and refilled from the GPU before the pointer
/// goes out, unless the caller has declared the contents dead.
///
/// `D3DLOCK_DISCARD` is that declaration and the only flag that qualifies: it
/// is defined for a whole-level lock and promises the caller reads nothing it
/// did not just write. `D3DLOCK_READONLY` is the opposite promise, and
/// `D3DLOCK_NOOVERWRITE` constrains only where the caller writes, not what it
/// may read. D3D9 has no write-only lock flag: the "no readback expected" hint
/// for a texture is `D3DUSAGE_DYNAMIC` at create time, and a dynamic texture
/// keeps its staging for its whole life, so no usage bit reaches this decision.
#[must_use]
pub const fn released_level_lock_needs_readback(flags: u32) -> bool {
    flags & D3DLOCK_DISCARD == 0
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
mod tests;
