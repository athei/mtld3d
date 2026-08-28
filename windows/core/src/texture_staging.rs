//! Decision helpers for per-mip texture staging: the Lock action and the release class.
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
//! - `D3DLOCK_DISCARD` on a whole-mip Lock of a `D3DPOOL_DEFAULT`
//!   texture: Rename, no preserve (game promised the old bytes are
//!   gone). Every other DISCARD is dropped first, see
//!   [`honoured_lock_flags`].
//! - Whole-mip contended: Rename + CPU-memcpy preserve. The game may
//!   read every byte through the Lock pointer, and D3D9 hands it the
//!   level's current contents whatever the texture's usage says.
//! - Partial contended, compressed AND not block-aligned: Rename + Cpu
//!   preserve — see `rect_block_aligned` for the formula and why
//!   preserve is forced.
//! - Partial contended, otherwise (uncompressed or block-aligned):
//!   `WriteInPlace`. Relies on the well-behaved-game no-overlap contract.
//!
//! Why `D3DUSAGE_DYNAMIC` has no say: `plan_lock` keys the buffer
//! equivalent on `D3DUSAGE_WRITEONLY`, the spec-documented "no readback"
//! hint for `CreateVertexBuffer` / `CreateIndexBuffer`. `CreateTexture`
//! documents no such hint. DYNAMIC only makes a default-pool level
//! lockable and DISCARD legal on it; a plain Lock of a dynamic texture
//! still sees the level's contents, as D3D9 specifies for every lockable
//! resource. A game that locks a whole dynamic page and rewrites a few
//! blocks of it (a lightmap page under animated light styles) relies on
//! exactly that, so the whole-mip arm preserves regardless of usage.
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
//!
//! [`staging_droppable_class`] answers the other staging question, the
//! one asked after an upload rather than before a write: whether a
//! texture is of a class whose levels may release their staging at all.

use mtld3d_types::{
    D3DLOCK_DISCARD, D3DLOCK_NOOVERWRITE, D3DLOCK_READONLY, D3DPOOL_DEFAULT, D3DUSAGE_DEPTHSTENCIL,
    D3DUSAGE_DYNAMIC, D3DUSAGE_RENDERTARGET,
};

use crate::{dirty_rect::DirtyRect, texture_flags::TextureFlags};

/// What the caller should do with the old backing's contents on a rename.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreserveKind {
    /// No preserve needed.
    ///
    /// `D3DLOCK_DISCARD` on a whole-mip Lock of a default-pool texture:
    /// the game promised to rewrite every byte before it reads any.
    None,
    /// Caller must synchronously memcpy the old `PageBox` into the fresh allocation.
    ///
    /// The copy happens before returning the Lock pointer. Two cases:
    /// (a) whole-mip contended (game might read all bytes through the
    /// pointer); (b) partial-but-unaligned compressed contended (encoder
    /// will fall back to a full-mip blit; outside-rect bytes must be
    /// valid).
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
#[must_use]
pub const fn is_in_flight(slot_last_submit_seq: u64, coherent_seq: u64) -> bool {
    slot_last_submit_seq > coherent_seq
}

/// Whether `rect` covers the whole mip (`None` is a whole-mip Lock).
///
/// `>=` for tolerance; `parse_rect` already clamps so equality is typical.
#[inline]
#[must_use]
pub const fn is_whole_mip(rect: Option<DirtyRect>, shape: MipShape) -> bool {
    match rect {
        None => true,
        Some(r) => r.x == 0 && r.y == 0 && r.w >= shape.mip_w && r.h >= shape.mip_h,
    }
}

/// The `D3DLOCK_*` bits a Lock is served with: `flags` less a `D3DLOCK_DISCARD` it cannot honour.
///
/// DISCARD is honoured on a whole-mip Lock of a `D3DPOOL_DEFAULT` texture only,
/// the one shape where "the contents are dead" can be taken literally. The
/// staging is the CPU mirror of the whole level and a later re-upload (device
/// recreate, managed eviction) publishes all of it, so a partial DISCARD would
/// leave the bytes outside the rect undefined in what gets published. The
/// managed and system-memory pools are served from that mirror for their whole
/// life, so the same holds for every Lock of theirs. The game's writes then
/// land on preserved contents, which is what D3D9 hands a lock that carries no
/// usable discard.
#[must_use]
pub const fn honoured_lock_flags(
    flags: u32,
    pool: u32,
    rect: Option<DirtyRect>,
    shape: MipShape,
) -> u32 {
    if pool == D3DPOOL_DEFAULT && is_whole_mip(rect, shape) {
        flags
    } else {
        flags & !D3DLOCK_DISCARD
    }
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
/// - `flags` is the raw `D3DLOCK_*` bitfield from the game. A DISCARD the
///   Lock cannot honour is dropped here as well as in the caller (see
///   [`honoured_lock_flags`]), so a verdict never rests on one.
/// - `pool` is the texture's `D3DPOOL` captured at `CreateTexture`.
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
    pool: u32,
    rect: Option<DirtyRect>,
    shape: MipShape,
) -> LockAction {
    let flags = honoured_lock_flags(flags, pool, rect, shape);
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

    // The game may read every byte of the mip through the pointer, and
    // D3D9 hands it the current contents whatever the usage says.
    if is_whole_mip(rect, shape) {
        return LockAction::FreshBox {
            preserve: PreserveKind::Cpu,
        };
    }

    // Partial. Force the rename when the compressed alignment
    // formula would push the encoder into its full-mip-fallback path
    // — the GPU's read range becomes "all bytes" rather than the
    // rect, so we can't trust no-overlap and must preserve outside-
    // rect bytes.
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
/// did not just write. The caller passes the flags through
/// [`honoured_lock_flags`] first, so a DISCARD on a partial lock never reaches
/// this decision. `D3DLOCK_READONLY` is the opposite promise, and
/// `D3DLOCK_NOOVERWRITE` constrains only where the caller writes, not what it
/// may read. D3D9 has no write-only lock flag: the "no readback expected" hint
/// for a texture is `D3DUSAGE_DYNAMIC` at create time, and a dynamic texture
/// keeps its staging for its whole life, so no usage bit reaches this decision.
#[must_use]
pub const fn released_level_lock_needs_readback(flags: u32) -> bool {
    flags & D3DLOCK_DISCARD == 0
}

/// Whether a texture's class ever lets a level release its staging after an upload.
///
/// A default-pool texture without `D3DUSAGE_DYNAMIC` cannot be locked in D3D9,
/// and the runtime keeps no system-memory copy of it: once the GPU holds every
/// byte, ours is redundant, and keeping it doubles the footprint of every
/// streamed texture inside a 32-bit game.
///
/// Every other class keeps its staging, each for a reason of its own. The
/// lockable pools, `D3DUSAGE_DYNAMIC` and an offscreen-plain surface all hand
/// the game a pointer back into it. Render targets and depth textures are
/// written by the GPU, so no upload of ours ever makes the staging redundant.
/// Cubes and volumes are written and uploaded a whole level at a time by paths
/// that expect the level to be there, and a re-created level is sized as a
/// single 2D slice, which is short of a volume's box. `depth` does not identify
/// a volume on its own: a single-slice `CreateVolumeTexture` is 2D on both
/// sides, so the flag is what excludes it.
#[must_use]
pub const fn staging_droppable_class(
    pool: u32,
    usage: u32,
    flags: TextureFlags,
    depth: u32,
) -> bool {
    pool == D3DPOOL_DEFAULT
        && usage & (D3DUSAGE_DYNAMIC | D3DUSAGE_RENDERTARGET | D3DUSAGE_DEPTHSTENCIL) == 0
        && !flags.intersects(
            TextureFlags::CUBE
                .union(TextureFlags::OFFSCREEN_PLAIN)
                .union(TextureFlags::DEPTH_FORMAT)
                .union(TextureFlags::VOLUME_TEXTURE),
        )
        && depth <= 1
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
