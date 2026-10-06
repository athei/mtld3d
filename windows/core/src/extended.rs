//! What an extended (`D3D9Ex`) device decides differently from a plain one.
//!
//! An extended device is the same device created through `Direct3DCreate9Ex`.
//! It changes a handful of answers rather than adding a second contract: which
//! pools a create accepts, what a non-null `pSharedHandle` means, how `Reset`
//! treats the state it finds, and a few values only the extended entry points
//! report. The verdicts here take `extended` as an argument so one function
//! answers for both kinds of device.

use mtld3d_types::{
    D3DERR_INVALIDCALL, D3DERR_NOTAVAILABLE, D3DPOOL_DEFAULT, D3DPOOL_SYSTEMMEM,
    D3DPRESENT_DONOTFLIP, D3DPRESENT_DONOTWAIT, D3DPRESENT_FLIPRESTART, D3DPRESENT_FORCEIMMEDIATE,
    D3DPRESENT_HIDEOVERLAY, D3DPRESENT_LINEAR_CONTENT, D3DPRESENT_UPDATECOLORKEY,
    D3DPRESENT_UPDATEOVERLAYONLY, D3DPRESENT_VIDEO_RESTRICT_TO_MONITOR,
    D3DUSAGE_RESTRICT_SHARED_RESOURCE, D3DUSAGE_RESTRICT_SHARED_RESOURCE_DRIVER,
    D3DUSAGE_RESTRICTED_CONTENT, E_NOTIMPL,
};

/// The frame latency an extended device reports until the application sets one.
pub const DEFAULT_FRAME_LATENCY: u32 = 3;

/// The highest frame latency `SetMaximumFrameLatency` accepts.
pub const MAX_FRAME_LATENCY: u32 = 30;

/// The usage bits the extended surface creates accept, and nothing else.
///
/// `CreateRenderTargetEx`, `CreateOffscreenPlainSurfaceEx` and
/// `CreateDepthStencilSurfaceEx` take only the content-protection and
/// shared-resource restrictions; even the usage the base create implies
/// (`D3DUSAGE_RENDERTARGET`, `D3DUSAGE_DEPTHSTENCIL`) is refused there.
pub const EX_CREATE_USAGE: u32 = D3DUSAGE_RESTRICTED_CONTENT
    | D3DUSAGE_RESTRICT_SHARED_RESOURCE
    | D3DUSAGE_RESTRICT_SHARED_RESOURCE_DRIVER;

/// The kind of resource a create with a `pSharedHandle` argument makes.
#[derive(Debug, PartialEq, Eq)]
pub enum CreateKind {
    /// `CreateTexture`, with the `Levels` argument as the caller passed it.
    Texture {
        levels: u32,
    },
    CubeTexture,
    VolumeTexture,
    VertexBuffer,
    IndexBuffer,
    RenderTarget,
    DepthStencil,
    OffscreenPlain,
}

/// What a create does with its `pSharedHandle` argument.
#[derive(Debug, PartialEq, Eq)]
pub enum SharedHandleVerdict {
    /// The handle is null: the create proceeds as it always did.
    Proceed,
    /// The handle points at the application's pixels for the one level the create makes.
    ///
    /// An extended device reads `*pSharedHandle` as a pointer to tightly
    /// packed initial data for a single-level system-memory texture or
    /// offscreen plain surface.
    UserMemory,
    /// The create fails with the refusal's code.
    Refuse(SharedRefusal),
}

/// Why a create with a non-null `pSharedHandle` is refused.
#[derive(Debug, PartialEq, Eq)]
pub enum SharedRefusal {
    /// A plain device has neither shared resources nor user memory.
    PlainDevice,
    /// A shared `D3DPOOL_DEFAULT` resource, which this layer does not implement.
    SharedResource,
    /// User memory outside the one shape that takes it, or sharing outside the default pool.
    InvalidCall,
    /// A vertex or index buffer outside the default pool, which takes no user memory.
    BufferUserMemory,
}

impl SharedRefusal {
    /// The HRESULT the create answers with.
    #[must_use]
    pub const fn hresult(&self) -> i32 {
        match self {
            Self::PlainDevice => E_NOTIMPL,
            Self::SharedResource | Self::BufferUserMemory => D3DERR_NOTAVAILABLE,
            Self::InvalidCall => D3DERR_INVALIDCALL,
        }
    }

    /// A short account of the refusal for the create's log line.
    #[must_use]
    pub const fn reason(&self) -> &'static str {
        match self {
            Self::PlainDevice => "a plain device has no shared resources or user memory",
            Self::SharedResource => "shared resources are not implemented",
            Self::InvalidCall => {
                "user memory is a single-level D3DPOOL_SYSTEMMEM texture or offscreen plain \
                 surface, and sharing needs D3DPOOL_DEFAULT"
            }
            Self::BufferUserMemory => "a vertex or index buffer takes no user memory",
        }
    }
}

/// Decide what a create does with a `pSharedHandle` argument.
///
/// A null handle changes nothing on either kind of device. A plain device
/// refuses any other with `E_NOTIMPL`. An extended device reads it as user
/// memory for a single-level `D3DPOOL_SYSTEMMEM` 2D texture or offscreen plain
/// surface, as a shared resource for `D3DPOOL_DEFAULT`, which is refused with
/// `D3DERR_NOTAVAILABLE` because sharing across processes is not implemented,
/// and as an invalid call for every other pool; a vertex or index buffer
/// outside the default pool answers `D3DERR_NOTAVAILABLE` instead.
#[must_use]
pub const fn shared_handle_verdict(
    kind: &CreateKind,
    pool: u32,
    has_handle: bool,
    extended: bool,
) -> SharedHandleVerdict {
    if !has_handle {
        return SharedHandleVerdict::Proceed;
    }
    if !extended {
        return SharedHandleVerdict::Refuse(SharedRefusal::PlainDevice);
    }
    if pool == D3DPOOL_DEFAULT {
        return SharedHandleVerdict::Refuse(SharedRefusal::SharedResource);
    }
    match kind {
        CreateKind::Texture { levels: 1 } | CreateKind::OffscreenPlain
            if pool == D3DPOOL_SYSTEMMEM =>
        {
            SharedHandleVerdict::UserMemory
        }
        CreateKind::VertexBuffer | CreateKind::IndexBuffer => {
            SharedHandleVerdict::Refuse(SharedRefusal::BufferUserMemory)
        }
        CreateKind::Texture { .. }
        | CreateKind::CubeTexture
        | CreateKind::VolumeTexture
        | CreateKind::RenderTarget
        | CreateKind::DepthStencil
        | CreateKind::OffscreenPlain => SharedHandleVerdict::Refuse(SharedRefusal::InvalidCall),
    }
}

/// Whether `usage` is one the extended surface creates accept.
///
/// Only [`EX_CREATE_USAGE`] bits may be set, and either shared-resource
/// restriction also needs a `pSharedHandle` to restrict.
#[must_use]
pub const fn ex_create_usage_valid(usage: u32, has_shared_handle: bool) -> bool {
    if usage & !EX_CREATE_USAGE != 0 {
        return false;
    }
    let restricts_sharing =
        usage & (D3DUSAGE_RESTRICT_SHARED_RESOURCE | D3DUSAGE_RESTRICT_SHARED_RESOURCE_DRIVER) != 0;
    !restricts_sharing || has_shared_handle
}

/// Whether `ResetEx`'s display-mode argument agrees with its present parameters.
///
/// A fullscreen request names the mode and a windowed one names none, and a
/// named mode has the back buffer's size. `mode` is the `(width, height)` of
/// the `D3DDISPLAYMODEEX` argument, `None` when it is null.
#[must_use]
pub fn reset_ex_mode_valid(
    windowed: bool,
    mode: Option<(u32, u32)>,
    back_buffer: (u32, u32),
) -> bool {
    mode.map_or(windowed, |size| !windowed && size == back_buffer)
}

/// The frame latency `SetMaximumFrameLatency(requested)` stores, `None` for an invalid call.
///
/// Zero asks for the default, and a request past [`MAX_FRAME_LATENCY`] is
/// refused.
#[must_use]
pub const fn frame_latency(requested: u32) -> Option<u32> {
    match requested {
        0 => Some(DEFAULT_FRAME_LATENCY),
        value if value > MAX_FRAME_LATENCY => None,
        value => Some(value),
    }
}

/// The name of one `D3DPRESENT_*` flag bit, for the line that says it is not honoured.
#[must_use]
pub const fn present_flag_name(bit: u32) -> &'static str {
    match bit {
        D3DPRESENT_DONOTWAIT => "D3DPRESENT_DONOTWAIT",
        D3DPRESENT_LINEAR_CONTENT => "D3DPRESENT_LINEAR_CONTENT",
        D3DPRESENT_DONOTFLIP => "D3DPRESENT_DONOTFLIP",
        D3DPRESENT_FLIPRESTART => "D3DPRESENT_FLIPRESTART",
        D3DPRESENT_VIDEO_RESTRICT_TO_MONITOR => "D3DPRESENT_VIDEO_RESTRICT_TO_MONITOR",
        D3DPRESENT_UPDATEOVERLAYONLY => "D3DPRESENT_UPDATEOVERLAYONLY",
        D3DPRESENT_HIDEOVERLAY => "D3DPRESENT_HIDEOVERLAY",
        D3DPRESENT_UPDATECOLORKEY => "D3DPRESENT_UPDATECOLORKEY",
        D3DPRESENT_FORCEIMMEDIATE => "D3DPRESENT_FORCEIMMEDIATE",
        _ => "an unknown D3DPRESENT flag",
    }
}

/// Iterate the set bits of a `D3DPRESENT_*` flag word, lowest first.
pub fn present_flag_bits(flags: u32) -> impl Iterator<Item = u32> {
    (0..u32::BITS)
        .map(|shift| 1u32 << shift)
        .filter(move |bit| flags & bit != 0)
}

/// The rows of user memory an extended create reads, as the application packs them.
///
/// User memory carries no pitch: its rows are `width` texels (or blocks)
/// apart with no padding.
#[derive(Debug, PartialEq, Eq)]
pub struct PackedRows {
    /// Bytes in one row of texels, or one row of blocks for a block-compressed format.
    pub row_bytes: usize,
    /// Rows of texels, or rows of blocks for a block-compressed format.
    pub rows: usize,
}

impl PackedRows {
    /// The extent of a `width` x `height` level in a format of the given layout.
    ///
    /// `bytes_per_pixel` is zero for a block-compressed format, whose rows are
    /// rows of blocks, `block` giving a block's width, height and bytes.
    ///
    /// # Panics
    ///
    /// Never on the 32- and 64-bit targets this builds for, where every `u32`
    /// fits a `usize`.
    #[must_use]
    pub fn of_level(width: u32, height: u32, bytes_per_pixel: u32, block: (u32, u32, u32)) -> Self {
        let (block_width, block_height, bytes_per_block) = block;
        let to_usize = |value: u32| usize::try_from(value).expect("a u32 fits usize");
        if bytes_per_pixel == 0 {
            let blocks_across = width.div_ceil(block_width.max(1));
            let block_rows = height.div_ceil(block_height.max(1));
            Self {
                row_bytes: to_usize(blocks_across).saturating_mul(to_usize(bytes_per_block)),
                rows: to_usize(block_rows),
            }
        } else {
            Self {
                row_bytes: to_usize(width).saturating_mul(to_usize(bytes_per_pixel)),
                rows: to_usize(height),
            }
        }
    }

    /// Bytes the application's memory holds for the level.
    #[must_use]
    pub const fn total_bytes(&self) -> usize {
        self.row_bytes.saturating_mul(self.rows)
    }
}

/// Copy packed user-memory rows into a level whose rows are `dst_pitch` bytes apart.
///
/// Returns `false`, copying nothing, when either side is too short for the
/// rows or the pitch is narrower than a row.
#[must_use]
pub fn copy_packed_rows(src: &[u8], rows: &PackedRows, dst: &mut [u8], dst_pitch: usize) -> bool {
    if dst_pitch < rows.row_bytes || src.len() < rows.total_bytes() {
        return false;
    }
    let needed = match rows.rows {
        0 => 0,
        count => (count - 1)
            .saturating_mul(dst_pitch)
            .saturating_add(rows.row_bytes),
    };
    if dst.len() < needed {
        return false;
    }
    if rows.row_bytes == 0 {
        return true;
    }
    for (src_row, dst_row) in src
        .chunks_exact(rows.row_bytes)
        .take(rows.rows)
        .zip(dst.chunks_mut(dst_pitch))
    {
        dst_row[..rows.row_bytes].copy_from_slice(src_row);
    }
    true
}

/// What `GetAvailableTextureMem` reports from a `budget` with `used` bytes allocated.
///
/// An extended device reports the budget whatever is allocated, as a
/// driver model that pages default-pool memory does; a plain device takes
/// the default-pool bytes off it. The answer is capped at `u32::MAX`.
#[must_use]
pub fn available_texture_mem(budget: u64, used: u64, extended: bool) -> u32 {
    let available = if extended {
        budget
    } else {
        budget.saturating_sub(used)
    };
    u32::try_from(available.min(u64::from(u32::MAX))).unwrap_or(u32::MAX)
}

#[cfg(test)]
mod tests;
