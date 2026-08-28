use core::ffi::c_void;
use std::sync::{Arc, atomic::Ordering};

use mtld3d_core::{
    dirty_rect::{DirtyRect, clip_copy_region},
    ids::TextureId,
    page_box::PageBox,
    render_scale::RenderScale,
    staging_coverage::StagingCoverage,
    texture_staging::{LockAction, MipShape, PreserveKind, decide_lock_action},
};
use mtld3d_shared::{
    BlitTextureToBufferParams, InPtr, InPtrMut, MetalHandle, OutPtr, ValueIn,
    mtl::{PixelFormat, Swizzle, TextureUsage},
    mtl_handle::{MTLDeviceKind, MTLTextureKind},
};
use mtld3d_types::{
    D3DBOX, D3DFMT_A8R8G8B8, D3DFMT_R5G6B5, D3DFMT_UYVY, D3DFMT_X8R8G8B8, D3DFMT_YUY2,
    D3DLOCK_DISCARD, D3DLOCK_KNOWN_BITS, D3DLOCK_NO_DIRTY_UPDATE, D3DLOCK_READONLY, D3DLOCKED_BOX,
    D3DLOCKED_RECT, D3DPOOL_DEFAULT, D3DPOOL_MANAGED, D3DRECT, D3DRTYPE_CUBETEXTURE,
    D3DRTYPE_SURFACE, D3DRTYPE_VOLUME, D3DRTYPE_VOLUMETEXTURE, D3DSURFACE_DESC, D3DTEXF_LINEAR,
    D3DTEXF_NONE, D3DUSAGE_DYNAMIC, D3DVOLUME_DESC, Guid, IDirect3DCubeTexture9Vtbl,
    IDirect3DTexture9Vtbl, IDirect3DVolume9Vtbl, IDirect3DVolumeTexture9Vtbl,
    IID_IDIRECT3DBASETEXTURE9, IID_IDIRECT3DCUBETEXTURE9, IID_IDIRECT3DRESOURCE9,
    IID_IDIRECT3DTEXTURE9, IID_IDIRECT3DVOLUME9, IID_IDIRECT3DVOLUMETEXTURE9, IID_IUNKNOWN,
};

use super::{
    D3D_OK, D3DERR_INVALIDCALL, E_NOINTERFACE,
    com_ref::ComUnknown,
    device::DeviceInner,
    encoder::{FrameEncoder, TextureInfo, TextureUploadJob},
    null_out,
    private_data::PrivateDataStore,
    surface::{DcLockState, Direct3DSurface9},
    unix_call::unix_call,
};

/// Sub-target for texture-lifecycle probes.
///
/// Covers the Lock/Unlock dirty-flag set, bind-time `flush_dirty_mips`, and the
/// `EvictManagedResources` action. Mirrors `device.rs::TEX_TRACE_TARGET`; both
/// files key the same `RUST_LOG=mtld3d::d3d9::tex=trace` switch.
const TEX_TRACE_TARGET: &str = "mtld3d::d3d9::tex";

/// Cube-map face count; `D3DCUBEMAP_FACE_*` are `0..=5`.
pub const CUBE_FACE_COUNT: u32 = 6;

static DIRECT3D_TEXTURE9_VTBL: IDirect3DTexture9Vtbl = IDirect3DTexture9Vtbl {
    query_interface: texture_query_interface,
    add_ref: texture_add_ref,
    release: texture_release,
    get_device: texture_get_device,
    set_private_data: texture_set_private_data,
    get_private_data: texture_get_private_data,
    free_private_data: texture_free_private_data,
    set_priority: texture_set_priority,
    get_priority: texture_get_priority,
    pre_load: texture_pre_load,
    get_type: texture_get_type,
    set_lod: texture_set_lod,
    get_lod: texture_get_lod,
    get_level_count: texture_get_level_count,
    set_auto_gen_filter_type: texture_set_auto_gen_filter_type,
    get_auto_gen_filter_type: texture_get_auto_gen_filter_type,
    generate_mip_sub_levels: texture_generate_mip_sub_levels,
    get_level_desc: texture_get_level_desc,
    get_surface_level: texture_get_surface_level,
    lock_rect: texture_lock_rect,
    unlock_rect: texture_unlock_rect,
    add_dirty_rect: texture_add_dirty_rect,
};

bitflags::bitflags! {
    /// Boolean attributes of a texture, packed into one field.
    ///
    /// The packing keeps `TextureInner`/`TextureCreateInfo` under the bool-bag
    /// lint and tightens the surrounding structs' tail padding.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct TextureFlags: u8 {
        /// `D3DUSAGE_AUTOGENMIPMAP` requested AND the format supports it.
        ///
        /// Metal can't auto-generate compressed BC/DXT, so the flag is dropped
        /// in `device_create_texture` for those. When set, mip-0 uploads append
        /// a `BlitCommand::generate_mipmaps` to the frame's leading-blit list
        /// right after the mip-0 `CopyBufferToTexture`, and the COM
        /// `IDirect3DBaseTexture9::GenerateMipSubLevels` call pushes the same op
        /// explicitly. Also collapses the app-visible level count to 1.
        const AUTOGEN_MIPMAP = 1 << 1;
        /// Sampleable shadow-map texture.
        ///
        /// Created via
        /// `CreateTexture(format=D24X8, usage=D3DUSAGE_DEPTHSTENCIL)`.
        /// `LockRect` bails with INVALIDCALL, no staging is wired up, and
        /// `SetDepthStencilSurface` resolves through to this texture's Metal
        /// handle when one of its mip surfaces is bound.
        const DEPTH_FORMAT = 1 << 2;
        /// Cube-map texture.
        ///
        /// Backed by six Metal array slices when the pool is GPU-visible.
        const CUBE = 1 << 3;
        /// The texture behind a `CreateOffscreenPlainSurface` surface.
        ///
        /// D3D9 lets a game lock such a surface even in the default pool,
        /// so its staging is never released after an upload.
        const OFFSCREEN_PLAIN = 1 << 4;
        /// Created through `CreateVolumeTexture`, whatever its depth.
        ///
        /// Distinct from `Direct3DTexture9::is_volume` (`depth > 1`), which asks
        /// whether the *Metal* texture is 3D: a single-slice volume texture is
        /// created 2D on both sides yet still hands out `IDirect3DVolume9`
        /// sub-resources rather than surfaces. The cached sub-resource slots have
        /// to be freed as the kind they hold, so the container records which one
        /// that is.
        const VOLUME_TEXTURE = 1 << 5;
        /// System-memory resource: no Metal texture exists for it yet.
        ///
        /// Set at creation for the two CPU-only pools (`D3DPOOL_SYSTEMMEM` and
        /// `D3DPOOL_SCRATCH`). Nothing warms up an `MTLTexture` and nothing
        /// uploads a mip while it is set, so a texture the application only
        /// locks, copies from, or reads back into costs system memory alone.
        /// Binding one for sampling clears it ([`promote_to_gpu`]), because
        /// D3D9 does sample a bound system-memory texture.
        const CPU_ONLY = 1 << 6;
    }
}

/// A source image laid out like a mip level, for [`TextureInner::copy_bytes_to_staging_region`].
///
/// `bytes` at `pitch` bytes/row, `width` × `height` texels.
pub struct SourceImage<'a> {
    pub bytes: &'a [u8],
    pub pitch: usize,
    pub width: u32,
    pub height: u32,
}

/// State used only by cube textures.
///
/// Ordinary 2D and volume textures carry only the optional pointer in
/// `TextureInner`; their staging vectors and dirty-mask fast path are unchanged.
struct CubeStorage {
    staging: Vec<Arc<PageBox>>,
    dirty_masks: [u32; CUBE_FACE_COUNT as usize],
    current_lock_readonly: Vec<bool>,
    current_lock_no_dirty: Vec<bool>,
    last_submit_seq: Vec<u64>,
    was_uploaded: Vec<bool>,
    locked: Vec<bool>,
    update_dirty: Vec<Option<DirtyRect>>,
    dc_lock: DcLockState,
}

impl CubeStorage {
    fn new(staging: Vec<PageBox>, levels: usize, width: u32, height: u32) -> Self {
        let count = levels.saturating_mul(CUBE_FACE_COUNT as usize);
        debug_assert_eq!(staging.len(), count);
        Self {
            staging: staging.into_iter().map(Arc::new).collect(),
            dirty_masks: [0; CUBE_FACE_COUNT as usize],
            current_lock_readonly: vec![false; count],
            current_lock_no_dirty: vec![false; count],
            last_submit_seq: vec![0; count],
            was_uploaded: vec![false; count],
            locked: vec![false; count],
            update_dirty: (0..count)
                .map(|index| {
                    let level = index % levels.max(1);
                    Some(DirtyRect::full(
                        (width >> level).max(1),
                        (height >> level).max(1),
                    ))
                })
                .collect(),
            dc_lock: DcLockState::default(),
        }
    }
}

pub struct TextureInner {
    // Texture metadata (formerly outer-struct fields).
    texture_id: TextureId,
    device_handle: MetalHandle<MTLDeviceKind>,
    /// Opaque `DeviceInner*`.
    ///
    /// Kept as `u64` because `DeviceInner::from_ptr` takes a `u64` by
    /// convention.
    device_inner: u64,
    width: u32,
    height: u32,
    /// Slice count: 1 for ordinary 2D textures, >1 for a volume (3D) texture.
    ///
    /// Created via `CreateVolumeTexture`. Drives the `MTLTextureType3D`
    /// descriptor on the unix side and `LockBox` sizing.
    depth: u32,
    levels: u32,
    d3d_format: u32,
    metal_pixel_format: PixelFormat,
    /// Packed boolean attributes — see [`TextureFlags`].
    flags: TextureFlags,
    swizzle: Option<[Swizzle; 4]>,
    /// Metal usage bits for the backing texture.
    ///
    /// Empty for plain sampled textures, `RENDER_TARGET` for textures created
    /// with `D3DUSAGE_RENDERTARGET` — passed through to `CreateTextureParams`
    /// so the Metal texture is allocated with `MTLTextureUsage::RenderTarget`.
    usage_flags: TextureUsage,
    /// Raw D3D9 `D3DUSAGE_*` bits.
    ///
    /// Read by `lock_region_ptr` to detect the full-mip
    /// non-{DISCARD,READONLY,WRITEONLY} correctness gap.
    d3d_usage: u32,
    /// What the backing Metal texture is rasterized at relative to `width`/`height`.
    ///
    /// Non-identity only for a render-target or depth-stencil texture created
    /// at the reported back-buffer size, which is the game's main view and
    /// shares the back buffer's `render.scale`. `width`/`height` and every mip
    /// dimension stay logical, so `GetLevelDesc` and the game's coordinates are
    /// unaffected; only [`TextureInner::texture_info`] converts.
    render_scale: RenderScale,
    /// App-set `SetAutoGenFilterType` value, round-tripped by `GetAutoGenFilterType`.
    ///
    /// Metal's `generateMipmaps` is fixed-linear, so this is app-visible state
    /// only and does not change how the chain is generated. Defaults to
    /// `D3DTEXF_LINEAR`.
    autogen_filter_type: u32,

    /// App-set `SetLOD` value (the most-detailed mip the runtime may use).
    ///
    /// Round-tripped by `GetLOD`. D3D9 honours it only for `D3DPOOL_MANAGED`
    /// textures; other pools always report 0. Defaults to 0.
    lod: u32,

    /// Per-mip persistent staging.
    ///
    /// Each `Arc<PageBox>` holds the full mip bytes in a page-aligned +
    /// page-sized allocation so the encoder can wrap it via
    /// `newBufferWithBytesNoCopy:` (which on non-UMA Macs requires page
    /// alignment for both pointer and length). The game writes through the
    /// pointer returned by `lock_region_ptr`. At `Unlock`, the upload closure
    /// clones the `Arc` — refcount bump, no memcpy — and hands the pointer to
    /// the encoder thread. `lock_region_ptr` decides between `WriteInPlace`
    /// (cast `as_ptr()` to `*mut u8` even when retention queues hold clones —
    /// same primitive READONLY uses) and `FreshBox` (allocate + swap).
    staging: Vec<Arc<PageBox>>,
    mip_widths: Vec<u32>,
    mip_heights: Vec<u32>,
    mip_bytes_per_row: Vec<u32>,
    /// Source-format bytes per pixel.
    ///
    /// Zero for compressed formats (BC1/2/3), which fall back to full-mip
    /// upload with a `log_once_warn!`.
    bytes_per_pixel: u32,
    /// Format block geometry.
    ///
    /// For uncompressed formats: `(1, 1, bpp)` — the unified offset formula in
    /// `lock_region_ptr` reduces to the obvious `r.y * pitch + r.x * bpp`. For
    /// DXT (BC1/2/3): `(4, 4, 8 or 16)` — the formula correctly converts
    /// pixel-space rect coords to block-row/block-col before the offset math,
    /// so `Lock(rect{y:128})` on a 512×512 DXT1 mip returns `(128/4) * pitch` =
    /// block-row 32 rather than the buggy pixel-row 128 (which overshot the
    /// staging Box by 4×).
    block_w: u32,
    block_h: u32,
    block_bytes: u32,

    /// Per-mip "needs upload" mask, bit `level` set at non-READONLY `UnlockRect`.
    ///
    /// Cleared by `flush_dirty_mips` at bind time. The bind-time flush
    /// schedules an upload via `schedule_upload`. A mask (not a `Vec<bool>`)
    /// so the every-draw bind-time gate is a single load; level count is
    /// bounded by log2(max texture dim 16384) + 1 = 15 bits.
    dirty_mask: u32,
    /// Sub-rect a dirty 2D level's upload may narrow to (`None` = whole mip).
    ///
    /// A partial copy into a level whose staging was dropped after its
    /// upload re-creates the staging uninitialized outside the copied
    /// region; a whole-mip upload would then push that garbage over GPU
    /// content the copy never touched. Tracking the written union lets the
    /// flush upload only what the copies wrote. Whole-mip writes reset the
    /// entry to `None`.
    pending_upload_rects: Vec<Option<DirtyRect>>,
    /// Levels whose staging was released after their upload retired (bit N = level N).
    ///
    /// Only default-pool textures the game cannot lock qualify, see
    /// [`TextureInner::staging_droppable`]; `staging[N]` then holds the
    /// shared placeholder page until a write re-creates the level.
    dropped_staging: u32,
    /// Per-level union of the writes that landed since the level's staging was allocated.
    ///
    /// The staging can only be released once the GPU holds every byte of the
    /// level, which several partial writes reach as surely as one whole-level
    /// one, so the drop reads the accumulated union rather than the rect of a
    /// single upload. Empty for a texture whose class can never release its
    /// staging ([`TextureInner::staging_droppable_class`]), which pays neither
    /// the memory nor the bookkeeping.
    staging_coverage: Vec<StagingCoverage>,
    /// Levels whose Metal texture holds pixels the CPU staging does not (bit N = level N).
    ///
    /// Set when the GPU writes a lockable level with no CPU mirror: a
    /// `StretchRect` blit into a `D3DPOOL_DEFAULT` offscreen-plain surface. The
    /// next `LockRect` reads the level back into staging and clears the bit, so
    /// the two agree again.
    gpu_authoritative: u32,
    /// `LockRect(D3DLOCK_READONLY)` stash per mip.
    ///
    /// Suppresses the upload at `UnlockRect` so a game's read-only inspection
    /// of a static atlas never re-uploads.
    current_lock_readonly: Vec<bool>,
    /// `LockRect(D3DLOCK_NO_DIRTY_UPDATE)` stash per mip.
    ///
    /// Its `UnlockRect` must NOT add a source dirty rect, so a later
    /// `UpdateTexture` ignores it, per the D3D9 spec.
    current_lock_no_dirty: Vec<bool>,
    /// Per-mip submit seq of the most recent GPU-visible reference to this staging.
    ///
    /// Stamped by `schedule_upload` on the API thread. Compared against
    /// `DeviceInner::coherent_seq_arc()` by `lock_region_ptr` to decide whether
    /// to reuse the Box in place or allocate a fresh one — same mechanism VB/IB
    /// rename uses.
    last_submit_seq: Vec<u64>,
    /// Sticky "this mip has been uploaded at least once on *some* device."
    ///
    /// Survives cross-device migration where `last_submit_seq` is device-scoped
    /// and gets reset to 0 by `rehydrate_for_device`. `evict_mark_dirty` and
    /// `rehydrate_for_device` use this to decide which mips need a re-upload
    /// after recreate; `last_submit_seq` alone is insufficient because the seq
    /// counter belongs to a specific encoder thread and is meaningless across
    /// devices.
    was_uploaded: Vec<bool>,
    /// Lock/Unlock pairing assertion.
    ///
    /// Mismatches are non-fatal but loudly logged via `log_once_warn!` — real
    /// games don't trip this in practice.
    locked: Vec<bool>,
    /// D3D9 *source* dirty region per mip, used only by `UpdateTexture`/`UpdateSurface`.
    ///
    /// `None` = clean (a copy is a no-op), `Some(rect)` = the region modified
    /// since the last copy (the bounding box of every `AddDirtyRect` and
    /// non-`READONLY` `UnlockRect` since then). Created full-dirty;
    /// `UpdateTexture` copies only the dirty region and then clears it, so a
    /// second copy from a clean source does nothing. Distinct from
    /// `dirty_mask` (the GPU-upload-needed flag).
    update_dirty: Vec<Option<DirtyRect>>,
    /// The `D3DPOOL` this texture was created in.
    ///
    /// Drives device-refcount forwarding: every pool
    /// **except `D3DPOOL_MANAGED`** forwards one reference to the owning device
    /// for the texture's public lifetime (D3D9 child-refcount model). Managed
    /// textures outlive the device and migrate to the next one
    /// (`rehydrate_for_device`), so a device ref would pin the old device alive
    /// and break that handoff — they do not forward.
    d3d_pool: u32,
    /// App-set managed-resource priority, round-tripped by `GetPriority` / `SetPriority`.
    ///
    /// D3D9 only honours priority for `D3DPOOL_MANAGED` resources (it drives
    /// the resource manager's eviction order); for every other pool both
    /// accessors are fixed at `0`. Metal has no eviction-order hint, so this is
    /// app-visible state only and never acted upon.
    priority: u32,
    /// Lazily allocated cube-only staging and lock state.
    ///
    /// This replaces the former cube-only `DcLockState` storage position, so
    /// ordinary texture objects gain no inline face arrays or allocation.
    cube: Option<Box<CubeStorage>>,
    /// GUID-keyed application private data (`Set/Get/FreePrivateData`).
    ///
    /// Shared by the 2D, cube, and volume-texture vtbls (all
    /// `TextureInner`-backed); any stored `IUnknown` is released when this
    /// `TextureInner` drops.
    private_data: PrivateDataStore,
    /// Cached sub-resource COM wrappers, one raw pointer per sub-resource.
    ///
    /// D3D9 sub-resources have identity: repeated `GetSurfaceLevel(0)` hands
    /// back the same `IDirect3DSurface9*` (one reference stronger each time),
    /// and the same holds for `GetCubeMapSurface` and `GetVolumeLevel`. The
    /// slots hold `*mut Direct3DSurface9` as `u64` (matching
    /// `DeviceInner::implicit_render_target`), or `*mut Direct3DVolume9` when
    /// [`TextureFlags::VOLUME_TEXTURE`] is set. Indexed by mip level, or by
    /// [`TextureInner::cube_subresource_index`] for a cube map.
    ///
    /// The slot holds NO reference: the wrapper holds one on this texture, so
    /// counting back would be a cycle. It is instead an owning raw pointer freed
    /// by `finalize_texture`, which cannot run while any wrapper still holds a
    /// public or private reference here.
    ///
    /// Empty until the first getter hands a sub-resource out, so a texture a
    /// streaming engine only ever writes through `LockRect` pays nothing.
    subresources: Vec<u64>,
}

/// Locks taken on default-pool textures created without `D3DUSAGE_DYNAMIC`.
///
/// D3D9 rejects those; mtld3d serves them. The count tells whether a game
/// streams through that path, which the staging drop has to respect.
static DEFAULT_STATIC_LOCKS: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);

/// How many locks landed on default-pool textures without `D3DUSAGE_DYNAMIC`.
pub fn default_static_lock_count() -> u32 {
    DEFAULT_STATIC_LOCKS.load(Ordering::Relaxed)
}

/// The page every dropped staging level points at.
///
/// One shared page instead of a per-level allocation: the slot has to hold
/// an `Arc<PageBox>` so the accessors keep their shape, but nothing may
/// read it, and a write re-creates the level first.
fn dropped_staging_placeholder() -> Arc<PageBox> {
    static PLACEHOLDER: std::sync::LazyLock<Arc<PageBox>> =
        std::sync::LazyLock::new(|| Arc::new(PageBox::new_zeroed(1)));
    Arc::clone(&PLACEHOLDER)
}

impl TextureInner {
    /// D3DPOOL_* the texture was created in.
    pub const fn d3d_pool(&self) -> u32 {
        self.d3d_pool
    }

    /// D3DFMT_* the texture was created with.
    pub const fn d3d_format(&self) -> u32 {
        self.d3d_format
    }

    /// D3DUSAGE_* the texture was created with.
    pub const fn d3d_usage(&self) -> u32 {
        self.d3d_usage
    }

    /// Bytes of staging this texture still holds in the 32-bit address space.
    pub fn resident_staging_bytes(&self) -> u64 {
        self.staging
            .iter()
            .enumerate()
            .filter(|(level, _)| self.dropped_staging & (1u32 << level) == 0)
            .map(|(_, b)| b.logical_len() as u64)
            .sum()
    }

    /// Whether `level`'s staging is currently released (placeholder only).
    ///
    /// The staging warmup skips such levels: there is no backing worth
    /// wrapping, and every dropped level of every texture shares one page.
    pub const fn staging_is_dropped(&self, level: usize) -> bool {
        self.dropped_staging & (1u32 << level) != 0
    }

    /// Claim `level` for the GPU: its Metal texture holds pixels staging does not.
    ///
    /// The next `LockRect` on the level reads it back before handing out a
    /// staging pointer. A D3D9 mip chain tops out at 15 levels, so the mask
    /// covers every level a texture can carry; the bound is a guard, not a
    /// limit anything reaches.
    pub const fn mark_level_gpu_authoritative(&mut self, level: usize) {
        if level < u32::BITS as usize {
            self.gpu_authoritative |= 1u32 << level;
        }
    }

    /// Whether `level`'s Metal texture holds pixels its staging does not.
    pub const fn level_gpu_authoritative(&self, level: usize) -> bool {
        level < u32::BITS as usize && self.gpu_authoritative & (1u32 << level) != 0
    }

    /// Release the claim once staging matches the texture again.
    const fn clear_level_gpu_authoritative(&mut self, level: usize) {
        if level < u32::BITS as usize {
            self.gpu_authoritative &= !(1u32 << level);
        }
    }

    /// Whether `level`'s staging can go once its upload has retired.
    ///
    /// A default-pool texture without `D3DUSAGE_DYNAMIC` cannot be locked
    /// in D3D9, and the runtime keeps no system-memory copy of it: the GPU
    /// holds the only bytes. Keeping ours doubles the footprint of every
    /// streamed texture inside a 32-bit game. Render targets, depth
    /// textures, cubes and volumes keep theirs (their copies serve other
    /// paths); so do the lockable pools.
    fn staging_droppable(&self, level: usize) -> bool {
        self.staging_droppable_class()
            && self.dropped_staging & (1u32 << level) == 0
            && !self.locked[level]
    }

    /// Whether every texel of `level` has been written since its staging was allocated.
    ///
    /// A level that still holds bytes the GPU never received keeps its staging:
    /// releasing it would leave the level's only copy of them nowhere.
    fn staging_fully_written(&self, level: usize) -> bool {
        self.staging_coverage
            .get(level)
            .is_some_and(StagingCoverage::is_full)
    }

    /// The texture-wide half of [`Self::staging_droppable`]: pool, usage and shape.
    ///
    /// None of it changes after creation except the offscreen-plain mark, which
    /// only ever narrows the class, so a texture outside it at creation stays
    /// outside it and needs no coverage tracking at all.
    const fn staging_droppable_class(&self) -> bool {
        self.d3d_pool == D3DPOOL_DEFAULT
            && self.d3d_usage
                & (D3DUSAGE_DYNAMIC
                    | mtld3d_types::D3DUSAGE_RENDERTARGET
                    | mtld3d_types::D3DUSAGE_DEPTHSTENCIL)
                == 0
            && !self.flags.contains(TextureFlags::CUBE)
            && !self.flags.contains(TextureFlags::OFFSCREEN_PLAIN)
            && !self.flags.contains(TextureFlags::DEPTH_FORMAT)
            && self.depth <= 1
    }

    /// Mark the texture as the backing of an offscreen-plain surface.
    pub fn mark_offscreen_plain(&mut self) {
        self.flags |= TextureFlags::OFFSCREEN_PLAIN;
    }

    /// Release `level`'s staging; the in-flight upload keeps its own `Arc`.
    fn drop_staging(&mut self, level: usize) {
        self.staging[level] = dropped_staging_placeholder();
        self.dropped_staging |= 1u32 << level;
        self.reset_staging_coverage(level);
    }

    /// Forget what the level's staging held, because it no longer holds it.
    ///
    /// A released, re-created or unpreserved-renamed allocation carries none of
    /// the pixels the recorded rects describe, so the union starts empty again.
    fn reset_staging_coverage(&mut self, level: usize) {
        if let Some(coverage) = self.staging_coverage.get_mut(level) {
            coverage.reset();
        }
    }

    /// Give `level` a staging buffer again before a write lands in it.
    ///
    /// The fresh buffer holds no pixels: a full-level write follows in
    /// every path that calls this, and a partial one is logged, since the
    /// pixels outside its region then no longer match the GPU copy.
    fn ensure_staging(&mut self, level: usize) {
        if self.dropped_staging & (1u32 << level) == 0 {
            return;
        }
        let block_rows = self.mip_heights[level].div_ceil(self.block_h.max(1));
        let len = (self.mip_bytes_per_row[level] as usize).saturating_mul(block_rows as usize);
        self.staging[level] = Arc::new(new_uninit_page_box(len.max(1)));
        self.dropped_staging &= !(1u32 << level);
        self.reset_staging_coverage(level);
        mtld3d_shared::log_once_trace_by!(
            target: TEX_TRACE_TARGET,
            key: self.texture_id.raw(),
            "texture {:#x}: staging re-created for level {level} after a write",
            self.texture_id.raw()
        );
    }

    pub fn mip_width(&self, level: usize) -> u32 {
        self.mip_widths[level]
    }

    pub fn mip_height(&self, level: usize) -> u32 {
        self.mip_heights[level]
    }

    /// Base-level extent of the backing Metal texture.
    ///
    /// `mip_width`/`mip_height` report the logical size D3D9 answers with. A
    /// render-target or depth-stencil texture created at the reported
    /// back-buffer size is rasterized at `render.scale` of that, so a command
    /// that addresses the Metal texture directly (a full-surface blit, an
    /// attachment extent) has to be measured here instead.
    pub fn render_extent(&self) -> (u32, u32) {
        (
            self.render_scale.dimension(self.width),
            self.render_scale.dimension(self.height),
        )
    }

    pub fn mip_bytes_per_row(&self, level: usize) -> u32 {
        self.mip_bytes_per_row[level]
    }

    /// Total bytes the full mip chain occupies.
    ///
    /// Summed as `row_pitch * ceil(mip_h / block_h)` per level (the same
    /// slice-size formula `lock_box` uses). Drives `GetAvailableTextureMem`
    /// accounting for `D3DPOOL_DEFAULT` resources.
    pub fn allocated_bytes(&self) -> u64 {
        let bh = self.block_h.max(1);
        let one_face = (0..self.levels as usize)
            .map(|level| {
                let row_pitch = u64::from(self.mip_bytes_per_row[level]);
                let block_rows = u64::from(self.mip_heights[level].div_ceil(bh));
                row_pitch.saturating_mul(block_rows)
            })
            .sum::<u64>();
        if self.flags.contains(TextureFlags::CUBE) {
            one_face.saturating_mul(u64::from(CUBE_FACE_COUNT))
        } else {
            one_face
        }
    }

    /// True for `D3DPOOL_DEFAULT` textures.
    ///
    /// GPU-resident; counted against the `GetAvailableTextureMem` budget.
    pub const fn is_default_pool(&self) -> bool {
        self.d3d_pool == D3DPOOL_DEFAULT
    }

    /// True while this texture is system memory with no `MTLTexture`.
    ///
    /// See [`TextureFlags::CPU_ONLY`]. Every site that would hand the texture
    /// to the encoder checks it first: the create-time warmup, the upload
    /// flush, and the cross-device rehydrate.
    pub const fn is_cpu_only(&self) -> bool {
        self.flags.contains(TextureFlags::CPU_ONLY)
    }

    /// How many per-mip staging buffers to wrap in an `MTLBuffer` eagerly.
    ///
    /// Zero for a cube map, whose staging is face-major in the cube sidecar,
    /// and for a volume texture, whose box upload carries its own `Arc`;
    /// neither has a per-level wrapper for the encoder to reuse.
    pub fn staging_warmup_levels(&self) -> u32 {
        if self
            .flags
            .intersects(TextureFlags::CUBE.union(TextureFlags::VOLUME_TEXTURE))
        {
            return 0;
        }
        u32::try_from(self.staging.len()).expect("mip count fits u32")
    }

    /// Volume (`LockBox`) lock: a writable pointer into the level's single staging buffer.
    ///
    /// The D3D9 row/slice pitches come back with it. The encoder created a
    /// `MTLTextureType3D` texture for the same `(width, height, depth)`; the
    /// paired `UnlockBox` schedules a full-box upload of this staging into it.
    /// Returns `None` if the level is out of range or has no staging
    /// (depth-format volumes).
    pub fn lock_box(&self, level: usize) -> Option<(*mut u8, i32, i32)> {
        let staging = self.staging.get(level)?;
        // Block-aware pitches: the stored `mip_bytes_per_row` is the
        // block-aligned row pitch (one block-row of bytes), and a slice spans
        // `ceil(mip_h / block_h)` block-rows. For uncompressed formats
        // `block_h == 1`, so this reduces to `row_pitch * mip_h`.
        let row_pitch = *self.mip_bytes_per_row.get(level)?;
        let block_rows = self.mip_heights.get(level)?.div_ceil(self.block_h.max(1));
        let slice_pitch = row_pitch.saturating_mul(block_rows);
        // The caller writes at most `slice_pitch * depth` bytes, which is how
        // the buffer was sized; `as_ptr().cast_mut()` matches the in-place
        // write primitive the 2D `lock_region_ptr` uses.
        let ptr = staging.as_ptr().cast_mut();
        Some((
            ptr,
            i32::try_from(row_pitch).unwrap_or(i32::MAX),
            i32::try_from(slice_pitch).unwrap_or(i32::MAX),
        ))
    }

    pub const fn autogen_mipmap(&self) -> bool {
        self.flags.contains(TextureFlags::AUTOGEN_MIPMAP)
    }

    /// App-visible mip level count.
    ///
    /// An `AUTOGENMIPMAP` texture exposes a single level (0); the sub-levels
    /// are runtime-generated and not app-accessible, even though the backing
    /// Metal texture carries the full chain (`levels`). Non-autogen textures
    /// expose all `levels`.
    pub const fn app_level_count(&self) -> u32 {
        if self.flags.contains(TextureFlags::AUTOGEN_MIPMAP) {
            1
        } else {
            self.levels
        }
    }

    /// Raw `DeviceInner*` (as `u64`) recorded at create, or 0 if detached.
    pub const fn device_inner(&self) -> u64 {
        self.device_inner
    }

    /// How many sub-resource cache slots this texture addresses.
    ///
    /// `levels * 6` for a cube map (see
    /// [`Self::cube_subresource_index`]), else one per mip level. Sized from
    /// `levels` rather than [`Self::app_level_count`] because that is the widest
    /// range any of the three getters admits.
    const fn subresource_slot_count(&self) -> usize {
        if self.flags.contains(TextureFlags::CUBE) {
            (self.levels as usize).saturating_mul(CUBE_FACE_COUNT as usize)
        } else {
            self.levels as usize
        }
    }

    /// The cached sub-resource wrapper at `index`, or `0` when there is none yet.
    ///
    /// `index` is a mip level, or a [`Self::cube_subresource_index`] for a cube
    /// map. An out-of-range index answers `0`: the slots are unallocated until
    /// the first hand-out, which is the one case a caller sees before its own
    /// bounds check has anything to index.
    fn cached_subresource(&self, index: usize) -> u64 {
        self.subresources.get(index).copied().unwrap_or(0)
    }

    /// Record `ptr` as the cached sub-resource wrapper at `index`.
    ///
    /// Allocates the slot vector on first use. `index` has already been bounds
    /// checked by the getter against `levels`, which is what
    /// [`Self::subresource_slot_count`] covers.
    fn cache_subresource(&mut self, index: usize, ptr: u64) {
        if self.subresources.is_empty() {
            self.subresources = vec![0; self.subresource_slot_count()];
        }
        self.subresources[index] = ptr;
    }

    /// Take every cached sub-resource wrapper, leaving the slots empty.
    ///
    /// For the container's finalize, which owns them: taking the `Vec` hands
    /// ownership over in one move and leaves nothing behind that a later
    /// accessor could hand out again.
    fn take_subresources(&mut self) -> Vec<u64> {
        core::mem::take(&mut self.subresources)
    }

    /// Validate an `UpdateSurface` copy of `src`'s `src_level` into this texture's `dst_level`.
    ///
    /// The source is optionally a sub-rect `[l,t,r,b)`, and the destination
    /// origin is `dst_point` `(x,y)`. Returns false → INVALIDCALL when the rect
    /// is empty/inverted, the region does not fit the destination mip, or (for
    /// block-compressed formats) the origins/extents are not block-aligned (and
    /// not full). Enforces the D3D9 `UpdateSurface` block-alignment rules.
    pub fn update_region_valid(
        &self,
        dst_level: usize,
        src: &Self,
        src_level: usize,
        src_rect: Option<(i32, i32, i32, i32)>,
        dst_point: (i32, i32),
    ) -> bool {
        let (sw, sh) = (src.mip_width(src_level), src.mip_height(src_level));
        let (rx, ry, rw, rh) = match src_rect {
            None => (0u32, 0u32, sw, sh),
            Some((l, t, r, b)) => {
                if l < 0 || t < 0 || r <= l || b <= t {
                    return false;
                }
                (
                    l.cast_unsigned(),
                    t.cast_unsigned(),
                    (r - l).cast_unsigned(),
                    (b - t).cast_unsigned(),
                )
            }
        };
        if dst_point.0 < 0 || dst_point.1 < 0 {
            return false;
        }
        let (dx, dy) = (dst_point.0.cast_unsigned(), dst_point.1.cast_unsigned());
        let (dw, dh) = (self.mip_width(dst_level), self.mip_height(dst_level));
        // Region must lie inside both src and dst mips.
        if rx.saturating_add(rw) > sw
            || ry.saturating_add(rh) > sh
            || dx.saturating_add(rw) > dw
            || dy.saturating_add(rh) > dh
        {
            return false;
        }
        // Block-compressed: origins block-aligned; a non-block-aligned extent
        // is only allowed when it reaches the edge of BOTH the source and the
        // destination mip (a partial-block region that stops short of either
        // edge is rejected — e.g. a 2x2 region from a 2x2 src mip into a 4x4
        // dst mip is invalid even though it reaches the src edge).
        let (bw, bh) = (self.block_w.max(1), self.block_h.max(1));
        if bw > 1 || bh > 1 {
            let aligned = |v: u32, b: u32| v.is_multiple_of(b);
            if !aligned(rx, bw)
                || !aligned(ry, bh)
                || !aligned(dx, bw)
                || !aligned(dy, bh)
                || (!aligned(rw, bw) && (rx + rw != sw || dx + rw != dw))
                || (!aligned(rh, bh) && (ry + rh != sh || dy + rh != dh))
            {
                return false;
            }
        }
        true
    }

    /// Copy a sub-rectangle of `src`'s `src_level` staging into `dst_level`'s staging.
    ///
    /// Lands at `dst_point`, honouring `src_rect` (the `UpdateSurface` region
    /// relocation). `None` rect/`(0,0)` point copy the whole mip.
    /// `UpdateSurface` rejects a bad region up front via
    /// [`Self::update_region_valid`]; `UpdateTexture` pairs mips by size and
    /// has no such rejection, so the region is clipped to both levels here.
    /// Block-compressed formats copy whole blocks: the clip rounds the region
    /// out to the block grid and keeps it inside both mips.
    pub fn copy_sub_region_from(
        &mut self,
        dst_level: usize,
        src: &Self,
        src_level: usize,
        src_rect: Option<(i32, i32, i32, i32)>,
        dst_point: (i32, i32),
    ) -> bool {
        self.ensure_staging(dst_level);
        let (Some(dst_box), Some(src_box)) =
            (self.staging.get(dst_level), src.staging.get(src_level))
        else {
            return false;
        };
        let (sw, sh) = (src.mip_width(src_level), src.mip_height(src_level));
        let (rx, ry, rw, rh) = match src_rect {
            None => (0u32, 0u32, sw, sh),
            Some((l, t, r, b)) => {
                if l < 0 || t < 0 || r <= l || b <= t {
                    return false;
                }
                (
                    l.cast_unsigned(),
                    t.cast_unsigned(),
                    (r - l).cast_unsigned(),
                    (b - t).cast_unsigned(),
                )
            }
        };
        let (dx, dy) = (
            dst_point.0.max(0).cast_unsigned(),
            dst_point.1.max(0).cast_unsigned(),
        );
        let (dw, dh) = (self.mip_width(dst_level), self.mip_height(dst_level));
        let (bw, bh) = (self.block_w.max(1), self.block_h.max(1));
        // The copy has to sit inside both levels. `UpdateTexture` pairs source
        // and destination mips on the larger of width and height, so a source
        // whose two dimensions are transposed relative to the destination's
        // (4x2 into 2x4) arrives with an extent that overhangs one of them;
        // only the part the two levels share is defined. Clipping keeps the
        // memcpy inside both staging allocations and keeps the dirty region
        // this records inside the destination mip, which the upload blit's
        // copy region must never exceed.
        let Some((src_rect, dst_rect)) = clip_copy_region(
            DirtyRect {
                x: rx,
                y: ry,
                w: rw,
                h: rh,
            },
            (dx, dy),
            (sw, sh),
            (dw, dh),
            (bw, bh),
        ) else {
            return false;
        };
        let (rx, ry, rw, rh) = (src_rect.x, src_rect.y, src_rect.w, src_rect.h);
        let (dx, dy) = (dst_rect.x, dst_rect.y);
        let src_pitch = src.mip_bytes_per_row(src_level) as usize;
        let dst_pitch = self.mip_bytes_per_row(dst_level) as usize;
        // Bytes per block-column = row pitch / blocks per row.
        let src_blocks_per_row = sw.div_ceil(bw) as usize;
        if src_blocks_per_row == 0 {
            return false;
        }
        let block_bytes = src_pitch / src_blocks_per_row;
        let rblock_cols = rw.div_ceil(bw) as usize;
        let rblock_rows = rh.div_ceil(bh) as usize;
        let (src_col0, src_row0) = ((rx / bw) as usize, (ry / bh) as usize);
        let (dst_col0, dst_row0) = ((dx / bw) as usize, (dy / bh) as usize);
        let copy_bytes = rblock_cols * block_bytes;
        // A volume level is `depth` slices of `block_rows * pitch` bytes laid
        // out back to back (`lock_box` reports that slice pitch); a 2D or cube
        // level is the single-slice case. The rectangle applies to every
        // slice the two levels share.
        let src_slice = src_pitch * (sh.div_ceil(bh) as usize);
        let dst_slice = dst_pitch * (dh.div_ceil(bh) as usize);
        let depth = (src.depth >> src_level)
            .max(1)
            .min((self.depth >> dst_level).max(1)) as usize;
        for z in 0..depth {
            for br in 0..rblock_rows {
                let s_off = z * src_slice + (src_row0 + br) * src_pitch + src_col0 * block_bytes;
                let d_off = z * dst_slice + (dst_row0 + br) * dst_pitch + dst_col0 * block_bytes;
                if s_off + copy_bytes > src_box.logical_len()
                    || d_off + copy_bytes > dst_box.logical_len()
                {
                    return false;
                }
                // SAFETY: `s_off + copy_bytes <= src_box.logical_len()` (checked).
                let src_ptr = unsafe { src_box.as_ptr().add(s_off) };
                // SAFETY: `d_off + copy_bytes <= dst_box.logical_len()` (checked).
                let dst_ptr = unsafe { dst_box.as_ptr().cast_mut().add(d_off) };
                // SAFETY: both ranges are in-bounds (above) and `src`/`self` are
                // distinct textures with disjoint PageBox allocations.
                unsafe {
                    core::ptr::copy_nonoverlapping(src_ptr, dst_ptr, copy_bytes);
                }
            }
        }
        self.mark_written_region(
            dst_level,
            DirtyRect {
                x: dx,
                y: dy,
                w: rw,
                h: rh,
            },
        );
        true
    }

    /// Copy one cube face sub-rectangle between cube staging allocations.
    ///
    /// The region is clipped to both levels, so a caller that skipped
    /// [`Self::update_region_valid`] still cannot write past either face
    /// allocation. The source and destination textures are distinct, so their
    /// face allocations cannot overlap.
    pub fn copy_cube_sub_region_from(
        &mut self,
        dst_subresource: (u32, usize),
        src: &Self,
        src_face: u32,
        src_level: usize,
        src_rect: Option<(i32, i32, i32, i32)>,
        dst_point: (i32, i32),
    ) -> bool {
        let (dst_face, dst_level) = dst_subresource;
        let (Some(dst_index), Some(src_index)) = (
            self.cube_subresource_index(dst_face, dst_level),
            src.cube_subresource_index(src_face, src_level),
        ) else {
            return false;
        };
        let (Some(dst_cube), Some(src_cube)) = (self.cube.as_deref(), src.cube.as_deref()) else {
            return false;
        };
        let dst_box = &dst_cube.staging[dst_index];
        let src_box = &src_cube.staging[src_index];
        let (sw, sh) = (src.mip_width(src_level), src.mip_height(src_level));
        let (rx, ry, rw, rh) = match src_rect {
            None => (0u32, 0u32, sw, sh),
            Some((l, t, r, b)) => {
                if l < 0 || t < 0 || r <= l || b <= t {
                    return false;
                }
                (
                    l.cast_unsigned(),
                    t.cast_unsigned(),
                    (r - l).cast_unsigned(),
                    (b - t).cast_unsigned(),
                )
            }
        };
        let (dx, dy) = (
            dst_point.0.max(0).cast_unsigned(),
            dst_point.1.max(0).cast_unsigned(),
        );
        let (dw, dh) = (self.mip_width(dst_level), self.mip_height(dst_level));
        let (bw, bh) = (self.block_w.max(1), self.block_h.max(1));
        // Same clip as the 2D path, so the memcpy cannot run past either
        // face's staging allocation. Cube levels are square and the mip
        // pairing lines the two up, which makes this a guard rather than a
        // live correction.
        let Some((src_rect, dst_rect)) = clip_copy_region(
            DirtyRect {
                x: rx,
                y: ry,
                w: rw,
                h: rh,
            },
            (dx, dy),
            (sw, sh),
            (dw, dh),
            (bw, bh),
        ) else {
            return false;
        };
        let (rx, ry, rw, rh) = (src_rect.x, src_rect.y, src_rect.w, src_rect.h);
        let (dx, dy) = (dst_rect.x, dst_rect.y);
        let src_pitch = src.mip_bytes_per_row(src_level) as usize;
        let dst_pitch = self.mip_bytes_per_row(dst_level) as usize;
        let src_blocks_per_row = sw.div_ceil(bw) as usize;
        if src_blocks_per_row == 0 {
            return false;
        }
        let block_bytes = src_pitch / src_blocks_per_row;
        let rblock_cols = rw.div_ceil(bw) as usize;
        let rblock_rows = rh.div_ceil(bh) as usize;
        let (src_col0, src_row0) = ((rx / bw) as usize, (ry / bh) as usize);
        let (dst_col0, dst_row0) = ((dx / bw) as usize, (dy / bh) as usize);
        let copy_bytes = rblock_cols * block_bytes;
        for br in 0..rblock_rows {
            let s_off = (src_row0 + br) * src_pitch + src_col0 * block_bytes;
            let d_off = (dst_row0 + br) * dst_pitch + dst_col0 * block_bytes;
            if s_off + copy_bytes > src_box.logical_len()
                || d_off + copy_bytes > dst_box.logical_len()
            {
                return false;
            }
            // SAFETY: `s_off + copy_bytes <= src_box.logical_len()` (checked).
            let src_ptr = unsafe { src_box.as_ptr().add(s_off) };
            // SAFETY: `d_off + copy_bytes <= dst_box.logical_len()` (checked).
            let dst_ptr = unsafe { dst_box.as_ptr().cast_mut().add(d_off) };
            // SAFETY: both ranges are in-bounds and belong to distinct textures.
            unsafe { core::ptr::copy_nonoverlapping(src_ptr, dst_ptr, copy_bytes) };
        }
        self.mark_cube_dirty(dst_face, dst_level);
        true
    }

    /// Convert a sub-rectangle of `src`'s `src_level` staging into `dst_level`'s staging.
    ///
    /// Lands at `dst_point`, re-encoding each pixel from `src`'s D3D format to
    /// this texture's D3D format. This is the cross-format `StretchRect` path
    /// into an offscreen-plain destination: neither GPU path serves it — the
    /// 1:1 blit can't convert and the render-quad conversion needs a
    /// render-target destination — so the conversion runs on the CPU. Sources
    /// are the simple uncompressed RGB formats R5G6B5 / X8R8G8B8 / A8R8G8B8
    /// plus the packed YUV formats YUY2 / UYVY (decoded per macropixel, the
    /// CPU twin of the render quad's decode); destinations are the RGB three.
    /// Returns false for any other pair (caller falls back to a
    /// best-effort no-op) or an out-of-bounds region. Same-size only
    /// (`src_rect` extent equals the destination extent) — the caller rejects
    /// scaling upstream. Marks the written rectangle of `dst_level` dirty on
    /// success so a later `flush_dirty_mips` uploads the converted pixels.
    pub fn convert_sub_region_from(
        &mut self,
        dst_level: usize,
        src: &Self,
        src_level: usize,
        src_rect: Option<(i32, i32, i32, i32)>,
        dst_point: (i32, i32),
    ) -> bool {
        let (src_fmt, dst_fmt) = (src.d3d_format, self.d3d_format);
        let yuv_src = mtld3d_core::stretch_rect::is_packed_yuv(src_fmt);
        if !(is_convertible_rgb(src_fmt) || yuv_src) || !is_convertible_rgb(dst_fmt) {
            return false;
        }
        self.ensure_staging(dst_level);
        let (Some(dst_box), Some(src_box)) =
            (self.staging.get(dst_level), src.staging.get(src_level))
        else {
            return false;
        };
        let (sw, sh) = (src.mip_width(src_level), src.mip_height(src_level));
        let (rx, ry, rw, rh) = match src_rect {
            None => (0u32, 0u32, sw, sh),
            Some((l, t, r, b)) => {
                if l < 0 || t < 0 || r <= l || b <= t {
                    return false;
                }
                (
                    l.cast_unsigned(),
                    t.cast_unsigned(),
                    (r - l).cast_unsigned(),
                    (b - t).cast_unsigned(),
                )
            }
        };
        if rx + rw > sw || ry + rh > sh {
            return false;
        }
        let (dx, dy) = (
            dst_point.0.max(0).cast_unsigned(),
            dst_point.1.max(0).cast_unsigned(),
        );
        // The region has to land inside the destination mip: a row that ran
        // past its right edge would wrap into the next one, and the dirty
        // rectangle recorded below has to describe a real part of the level.
        if dx + rw > self.mip_width(dst_level) || dy + rh > self.mip_height(dst_level) {
            return false;
        }
        let src_pitch = src.mip_bytes_per_row(src_level) as usize;
        let dst_pitch = self.mip_bytes_per_row(dst_level) as usize;
        let (src_bpp, dst_bpp) = (rgb_bpp(src_fmt), rgb_bpp(dst_fmt));
        let (src_len, dst_len) = (src_box.logical_len(), dst_box.logical_len());
        let src_base = src_box.as_ptr();
        let dst_base = dst_box.as_ptr().cast_mut();
        for row in 0..rh {
            let s_row = (ry + row) as usize * src_pitch;
            let d_row = (dy + row) as usize * dst_pitch;
            for col in 0..rw {
                let sx = (rx + col) as usize;
                let d_off = d_row + (dx + col) as usize * dst_bpp;
                if d_off + dst_bpp > dst_len {
                    return false;
                }
                let rgba = if yuv_src {
                    // Packed 4:2:2: the pixel's macropixel is the 4 bytes at
                    // the even column; its parity picks the luma sample.
                    let s_off = s_row + (sx & !1) * 2;
                    if s_off + 4 > src_len {
                        return false;
                    }
                    // SAFETY: `s_off + 4 <= src_len` (checked), so this stays
                    // in-bounds of the source PageBox allocation.
                    let src_ptr = unsafe { src_base.add(s_off) };
                    // SAFETY: 4 bytes from `src_ptr`; `src`/`self` are distinct
                    // textures with disjoint PageBox allocations.
                    let mp = unsafe { std::slice::from_raw_parts(src_ptr, 4) };
                    let Some((r, g, b)) = mtld3d_core::stretch_rect::decode_packed_yuv(
                        src_fmt,
                        [mp[0], mp[1], mp[2], mp[3]],
                        sx & 1 == 1,
                    ) else {
                        return false;
                    };
                    (r, g, b, 0xff)
                } else {
                    let s_off = s_row + sx * src_bpp;
                    if s_off + src_bpp > src_len {
                        return false;
                    }
                    // SAFETY: `s_off + src_bpp <= src_len` (checked), so this stays
                    // in-bounds of the source PageBox allocation.
                    let src_ptr = unsafe { src_base.add(s_off) };
                    // SAFETY: `src_bpp` bytes from `src_ptr`; `src`/`self` are distinct
                    // textures with disjoint PageBox allocations.
                    let px = unsafe { std::slice::from_raw_parts(src_ptr, src_bpp) };
                    decode_rgb_pixel(src_fmt, px)
                };
                // SAFETY: `d_off + dst_bpp <= dst_len` (checked); in-bounds of dst.
                let dst_ptr = unsafe { dst_base.add(d_off) };
                // SAFETY: `dst_bpp` bytes from `dst_ptr`; disjoint from `src` as above.
                let out = unsafe { std::slice::from_raw_parts_mut(dst_ptr, dst_bpp) };
                encode_rgb_pixel(dst_fmt, rgba, out);
            }
        }
        self.mark_written_region(
            dst_level,
            DirtyRect {
                x: dx,
                y: dy,
                w: rw,
                h: rh,
            },
        );
        true
    }

    /// Copy a sub-rectangle of raw source bytes into `dst_level`'s CPU staging.
    ///
    /// The bytes are a standalone system-memory offscreen surface's backing,
    /// laid out like a mip: `src_pitch` bytes/row. This is the `UpdateSurface`
    /// path for a `D3DPOOL_SYSTEMMEM` offscreen-plain *source* surface (which is
    /// not texture-backed, so [`Self::copy_sub_region_from`] cannot serve it).
    /// Block-aware, mirroring `copy_sub_region_from`, down to marking only the
    /// written rectangle dirty. Returns false on a missing level or an
    /// out-of-bounds region.
    pub fn copy_bytes_to_staging_region(
        &mut self,
        dst_level: usize,
        src: &SourceImage<'_>,
        src_rect: Option<(i32, i32, i32, i32)>,
        dst_point: (i32, i32),
    ) -> bool {
        let &SourceImage {
            bytes: src_bytes,
            pitch: src_pitch,
            width: src_w,
            height: src_h,
        } = src;
        self.ensure_staging(dst_level);
        let Some(dst_box) = self.staging.get(dst_level) else {
            return false;
        };
        let (rx, ry, rw, rh) = match src_rect {
            None => (0u32, 0u32, src_w, src_h),
            Some((l, t, r, b)) => {
                if l < 0 || t < 0 || r <= l || b <= t {
                    return false;
                }
                (
                    l.cast_unsigned(),
                    t.cast_unsigned(),
                    (r - l).cast_unsigned(),
                    (b - t).cast_unsigned(),
                )
            }
        };
        if rx + rw > src_w || ry + rh > src_h {
            return false;
        }
        let (dx, dy) = (
            dst_point.0.max(0).cast_unsigned(),
            dst_point.1.max(0).cast_unsigned(),
        );
        // The region has to land inside the destination mip: a row that ran
        // past its right edge would wrap into the next one, and the dirty
        // rectangle recorded below has to describe a real part of the level.
        if dx + rw > self.mip_width(dst_level) || dy + rh > self.mip_height(dst_level) {
            return false;
        }
        let (bw, bh) = (self.block_w.max(1), self.block_h.max(1));
        let dst_pitch = self.mip_bytes_per_row(dst_level) as usize;
        let src_blocks_per_row = src_w.div_ceil(bw) as usize;
        if src_blocks_per_row == 0 || src_pitch == 0 {
            return false;
        }
        let block_bytes = src_pitch / src_blocks_per_row;
        let rblock_cols = rw.div_ceil(bw) as usize;
        let rblock_rows = rh.div_ceil(bh) as usize;
        let (src_col0, src_row0) = ((rx / bw) as usize, (ry / bh) as usize);
        let (dst_col0, dst_row0) = ((dx / bw) as usize, (dy / bh) as usize);
        let copy_bytes = rblock_cols * block_bytes;
        for br in 0..rblock_rows {
            let s_off = (src_row0 + br) * src_pitch + src_col0 * block_bytes;
            let d_off = (dst_row0 + br) * dst_pitch + dst_col0 * block_bytes;
            if s_off + copy_bytes > src_bytes.len() || d_off + copy_bytes > dst_box.logical_len() {
                return false;
            }
            // SAFETY: `d_off + copy_bytes <= dst_box.logical_len()` (checked); the
            // destination PageBox is distinct from the caller-owned `src_bytes`.
            let dst_ptr = unsafe { dst_box.as_ptr().cast_mut().add(d_off) };
            // SAFETY: `s_off + copy_bytes <= src_bytes.len()` (checked).
            let src_ptr = unsafe { src_bytes.as_ptr().add(s_off) };
            // SAFETY: both ranges are in-bounds (checked above).
            unsafe {
                core::ptr::copy_nonoverlapping(src_ptr, dst_ptr, copy_bytes);
            }
        }
        self.mark_written_region(
            dst_level,
            DirtyRect {
                x: dx,
                y: dy,
                w: rw,
                h: rh,
            },
        );
        true
    }

    /// Copy raw source bytes into one cube face's CPU staging allocation.
    ///
    /// This is the cube destination counterpart to
    /// [`Self::copy_bytes_to_staging_region`].
    pub fn copy_bytes_to_cube_staging_region(
        &mut self,
        dst_face: u32,
        dst_level: usize,
        src: &SourceImage<'_>,
        src_rect: Option<(i32, i32, i32, i32)>,
        dst_point: (i32, i32),
    ) -> bool {
        let Some(dst_index) = self.cube_subresource_index(dst_face, dst_level) else {
            return false;
        };
        let Some(dst_box) = self
            .cube
            .as_deref()
            .and_then(|cube| cube.staging.get(dst_index))
        else {
            return false;
        };
        let &SourceImage {
            bytes: src_bytes,
            pitch: src_pitch,
            width: src_w,
            height: src_h,
        } = src;
        let (rx, ry, rw, rh) = match src_rect {
            None => (0u32, 0u32, src_w, src_h),
            Some((l, t, r, b)) => {
                if l < 0 || t < 0 || r <= l || b <= t {
                    return false;
                }
                (
                    l.cast_unsigned(),
                    t.cast_unsigned(),
                    (r - l).cast_unsigned(),
                    (b - t).cast_unsigned(),
                )
            }
        };
        if rx + rw > src_w || ry + rh > src_h {
            return false;
        }
        let (dx, dy) = (
            dst_point.0.max(0).cast_unsigned(),
            dst_point.1.max(0).cast_unsigned(),
        );
        if dx + rw > self.mip_width(dst_level) || dy + rh > self.mip_height(dst_level) {
            return false;
        }
        let (bw, bh) = (self.block_w.max(1), self.block_h.max(1));
        let dst_pitch = self.mip_bytes_per_row(dst_level) as usize;
        let src_blocks_per_row = src_w.div_ceil(bw) as usize;
        if src_blocks_per_row == 0 || src_pitch == 0 {
            return false;
        }
        let block_bytes = src_pitch / src_blocks_per_row;
        let rblock_cols = rw.div_ceil(bw) as usize;
        let rblock_rows = rh.div_ceil(bh) as usize;
        let (src_col0, src_row0) = ((rx / bw) as usize, (ry / bh) as usize);
        let (dst_col0, dst_row0) = ((dx / bw) as usize, (dy / bh) as usize);
        let copy_bytes = rblock_cols * block_bytes;
        for br in 0..rblock_rows {
            let s_off = (src_row0 + br) * src_pitch + src_col0 * block_bytes;
            let d_off = (dst_row0 + br) * dst_pitch + dst_col0 * block_bytes;
            if s_off + copy_bytes > src_bytes.len() || d_off + copy_bytes > dst_box.logical_len() {
                return false;
            }
            // SAFETY: `d_off + copy_bytes <= dst_box.logical_len()` (checked).
            let dst_ptr = unsafe { dst_box.as_ptr().cast_mut().add(d_off) };
            // SAFETY: `s_off + copy_bytes <= src_bytes.len()` (checked).
            let src_ptr = unsafe { src_bytes.as_ptr().add(s_off) };
            // SAFETY: both ranges are in-bounds and cannot overlap.
            unsafe { core::ptr::copy_nonoverlapping(src_ptr, dst_ptr, copy_bytes) };
        }
        self.mark_cube_dirty(dst_face, dst_level);
        true
    }

    /// Fill a sub-rectangle of `level`'s CPU staging with a repeated `pixel`.
    ///
    /// The `ColorFill` path for a lockable `D3DPOOL_DEFAULT` offscreen-plain
    /// surface: its read-back is a `LockRect` into this CPU staging, which no
    /// path re-reads from the GPU, so the fill has to land here on the calling
    /// thread. One row is splatted at memcpy rate and the rest of the region
    /// copies from it. Uncompressed formats only (`pixel.len()` ==
    /// bytes/pixel; block-compressed `ColorFill` is rejected upstream).
    /// Returns false on a missing level or an out-of-bounds region.
    pub fn fill_staging_region(
        &mut self,
        level: usize,
        ox: u32,
        oy: u32,
        w: u32,
        h: u32,
        pixel: &[u8],
    ) -> bool {
        // A fill is a write: a level whose staging was never materialized (or
        // was dropped after its upload) gets one here, like any first write.
        self.ensure_staging(level);
        let Some(box_) = self.staging.get(level) else {
            return false;
        };
        let bpp = pixel.len();
        if bpp == 0 || w == 0 || h == 0 {
            return false;
        }
        let pitch = self.mip_bytes_per_row(level) as usize;
        let logical = box_.logical_len();
        let base = box_.as_ptr().cast_mut();
        let run = w as usize * bpp;
        for row in oy..oy.saturating_add(h) {
            let row_off = row as usize * pitch + ox as usize * bpp;
            if row_off + run > logical {
                return false;
            }
            // SAFETY: `row_off + run <= logical` (checked), so `row_off` is
            // within the allocation.
            let row_ptr = unsafe { base.add(row_off) };
            // SAFETY: `row_ptr..row_ptr+run` is in-bounds (above); D3D9 objects
            // are single-threaded, so the write has exclusive access and no
            // other slice aliases this run.
            let dst = unsafe { core::slice::from_raw_parts_mut(row_ptr, run) };
            mtld3d_core::convert::splat_pixel_pattern(dst, pixel);
        }
        true
    }

    /// Drop the link to the owning `DeviceInner` because the device is being released.
    ///
    /// Called by `device_release` rc==0 for every entry in `live_textures`.
    /// After this returns:
    ///   - `device_inner == 0` and `device_handle == 0` so no accessor
    ///     can dereference the freed `DeviceInner`;
    ///   - `last_submit_seq[level] == 0` because the seq counter is
    ///     scoped to the encoder we just shut down;
    ///   - `was_uploaded[level]` is preserved so a future
    ///     `rehydrate_for_device` on a new device knows which mips to
    ///     re-mark dirty.
    ///
    /// `dirty_mask` is left alone — it's still a "needs upload" hint.
    pub fn detach_from_device(&mut self) {
        self.device_inner = 0;
        self.device_handle = MetalHandle::NULL;
        for slot in &mut self.last_submit_seq {
            *slot = 0;
        }
        if let Some(cube) = self.cube.as_deref_mut() {
            for slot in &mut cube.last_submit_seq {
                *slot = 0;
            }
        }
        self.point_cached_surfaces_at(core::ptr::null_mut());
    }

    /// Repoint every cached sub-resource surface at `device_inner`.
    ///
    /// They outlive the app's last `Release` of them, so they travel with the
    /// container across both migration points a `D3DPOOL_MANAGED` texture sees
    /// (`detach_from_device` with null, `rehydrate_for_device` with the adopting
    /// device). Volume level shells hold no device pointer of their own; they
    /// read the container.
    fn point_cached_surfaces_at(&self, device_inner: *mut DeviceInner) {
        if self.flags.contains(TextureFlags::VOLUME_TEXTURE) {
            return;
        }
        for &slot in &self.subresources {
            // SAFETY: a non-zero slot is a live cached sub-resource surface this
            // texture owns (freed only by `finalize_texture`), and `device_inner`
            // is null or the live device this texture has just joined.
            unsafe { crate::surface::set_cached_surface_device(slot, device_inner) };
        }
    }

    /// Build a `TextureInfo` snapshot for upload closures and draw-time stage binding capture.
    pub fn texture_info(&self) -> TextureInfo {
        TextureInfo {
            texture_id: self.texture_id,
            // Render space: this snapshot is what creates and addresses the
            // Metal texture. `self.width`/`self.height` stay logical for
            // `GetLevelDesc` and for every rect the game supplies.
            width: self.render_scale.dimension(self.width),
            height: self.render_scale.dimension(self.height),
            depth: self.depth,
            levels: self.levels,
            pixel_format: self.metal_pixel_format,
            create_flags: {
                let mut flags = mtld3d_shared::mtl::TextureCreateFlags::empty();
                flags.set(
                    mtld3d_shared::mtl::TextureCreateFlags::HAS_SWIZZLE,
                    self.swizzle.is_some(),
                );
                flags.set(
                    mtld3d_shared::mtl::TextureCreateFlags::TYPE_3D,
                    self.depth > 1,
                );
                flags.set(
                    mtld3d_shared::mtl::TextureCreateFlags::TYPE_CUBE,
                    self.flags.contains(TextureFlags::CUBE),
                );
                flags
            },
            swizzle: self.swizzle.unwrap_or([Swizzle::Zero; 4]),
            usage_flags: self.usage_flags,
        }
    }

    /// Clone the staging `Arc` for this mip.
    ///
    /// Cheap (refcount bump) — used by the upload closure to keep the bytes
    /// alive until the encoder thread blits them to the texture, and by
    /// `push_texture_warmups` (device.rs) to populate
    /// `StagingWarmupEntry.keepalive` so the staging `MTLBuffer` wrapper
    /// survives a same-frame `texture_release`.
    pub fn staging_arc(&self, level: usize) -> Arc<PageBox> {
        Arc::clone(&self.staging[level])
    }

    /// Raw backing pointer of mip `level`'s staging `PageBox`.
    ///
    /// For `CreateTexture`-time staging-buffer warmup. Address stays valid
    /// until the next Lock(DISCARD) rename swaps the Arc.
    pub fn staging_backing_ptr(&self, level: usize) -> u64 {
        self.staging[level].as_ptr() as u64
    }

    /// Page-aligned length of mip `level`'s staging `PageBox`.
    ///
    /// Pairs with `staging_backing_ptr` for `BufferCreateDesc::length`.
    pub fn staging_backing_len(&self, level: usize) -> u64 {
        self.staging[level].len() as u64
    }

    const fn cube_subresource_index(&self, face: u32, level: usize) -> Option<usize> {
        if face >= CUBE_FACE_COUNT || level >= self.levels as usize {
            return None;
        }
        Some(face as usize * self.levels as usize + level)
    }

    pub fn cube_is_locked(&self, face: u32, level: usize) -> bool {
        let Some(index) = self.cube_subresource_index(face, level) else {
            return false;
        };
        self.cube.as_deref().is_some_and(|cube| cube.locked[index])
    }

    /// Whether any cube face subresource currently has an outstanding lock.
    #[must_use]
    pub fn cube_any_locked(&self) -> bool {
        self.cube
            .as_deref()
            .is_some_and(|cube| cube.locked.iter().any(|locked| *locked))
    }

    /// Return the CPU staging pointer and pitches for one cube subresource.
    ///
    /// Used by a parent-backed face surface's `GetDC` path. The pointer remains
    /// valid while the cube texture remains alive and the subresource is not
    /// renamed by a writable `LockRect`.
    pub fn cube_lock_box(&self, face: u32, level: usize) -> Option<(*mut u8, i32, i32)> {
        let index = self.cube_subresource_index(face, level)?;
        let staging = self.cube.as_deref()?.staging.get(index)?;
        let row_pitch = *self.mip_bytes_per_row.get(level)?;
        let block_rows = self.mip_heights.get(level)?.div_ceil(self.block_h.max(1));
        let slice_pitch = row_pitch.saturating_mul(block_rows);
        Some((
            staging.as_ptr().cast_mut(),
            i32::try_from(row_pitch).unwrap_or(i32::MAX),
            i32::try_from(slice_pitch).unwrap_or(i32::MAX),
        ))
    }

    /// Mark a cube face as modified through its surface interface.
    ///
    /// The next bind uploads the face, and a later `UpdateTexture` sees the
    /// subresource as source-dirty.
    pub fn mark_cube_surface_dirty(&mut self, face: u32, level: usize) {
        self.mark_cube_update_dirty(face, level, None);
        self.mark_cube_dirty(face, level);
    }

    fn cube_lock_region_ptr(
        &mut self,
        face: u32,
        level: usize,
        rect: Option<DirtyRect>,
        flags: u32,
    ) -> Option<(*mut u8, u32)> {
        let index = self.cube_subresource_index(face, level)?;
        let pitch = self.mip_bytes_per_row[level];
        let offset = mtld3d_core::texture_staging::texture_lock_offset(
            rect,
            pitch,
            self.block_w,
            self.block_h,
            self.block_bytes,
        );
        let staging_len = self.cube.as_deref()?.staging[index].logical_len();
        assert!(
            offset < staging_len,
            "cube LockRect offset {offset} >= face {face} level {level} length {staging_len}"
        );

        if flags & D3DLOCK_READONLY != 0 {
            let base = self.cube.as_deref()?.staging[index].as_ptr().cast_mut();
            // SAFETY: `offset` was checked against the logical staging length.
            return Some((unsafe { base.add(offset) }, pitch));
        }

        let coherent_seq = if self.device_inner == 0 {
            0
        } else {
            DeviceInner::from_ptr(self.device_inner)
                .upload_coherent_seq_arc()
                .load(Ordering::Acquire)
        };
        let last_submit_seq = self.cube.as_deref()?.last_submit_seq[index];
        let action = decide_lock_action(
            coherent_seq,
            last_submit_seq,
            flags,
            self.d3d_usage,
            rect,
            MipShape {
                mip_w: self.mip_widths[level],
                mip_h: self.mip_heights[level],
                block_w: self.block_w,
                block_h: self.block_h,
            },
        );
        let device_inner = self.device_inner;
        let cube = self.cube.as_deref_mut()?;
        let base = match action {
            LockAction::WriteInPlace => cube.staging[index].as_ptr().cast_mut(),
            LockAction::FreshBox { preserve } => {
                let mip_len = cube.staging[index].logical_len();
                let old = core::mem::replace(
                    &mut cube.staging[index],
                    Arc::new(new_uninit_page_box(mip_len)),
                );
                if preserve == PreserveKind::Cpu {
                    let dst = Arc::get_mut(&mut cube.staging[index])
                        .expect("fresh cube staging Arc is unique")
                        .as_mut_ptr();
                    // SAFETY: old and new cube staging allocations are disjoint
                    // and both contain `mip_len` logical bytes.
                    unsafe { core::ptr::copy_nonoverlapping(old.as_ptr(), dst, mip_len) };
                }
                if device_inner != 0 {
                    let perf = DeviceInner::from_ptr(device_inner).perf_mut();
                    match preserve {
                        PreserveKind::None => perf.bump_texture_discard(),
                        PreserveKind::Cpu => perf.bump_texture_preserve_cpu(),
                    }
                    perf.bump_texture_rename();
                }
                cube.last_submit_seq[index] = 0;
                Arc::get_mut(&mut cube.staging[index])
                    .expect("fresh cube staging Arc is unique")
                    .as_mut_ptr()
            }
        };
        // SAFETY: `offset` was checked against the logical staging length.
        Some((unsafe { base.add(offset) }, pitch))
    }

    fn cube_stash_lock(&mut self, face: u32, level: usize, read_only: bool, no_dirty: bool) {
        let index = self
            .cube_subresource_index(face, level)
            .expect("validated cube subresource");
        let cube = self.cube.as_deref_mut().expect("cube storage");
        cube.current_lock_readonly[index] = read_only;
        cube.current_lock_no_dirty[index] = no_dirty;
        cube.locked[index] = true;
    }

    fn cube_take_lock(&mut self, face: u32, level: usize) -> (bool, bool, bool, bool) {
        let index = self
            .cube_subresource_index(face, level)
            .expect("validated cube subresource");
        let cube = self.cube.as_deref_mut().expect("cube storage");
        let was_locked = core::mem::take(&mut cube.locked[index]);
        let read_only = core::mem::take(&mut cube.current_lock_readonly[index]);
        let no_dirty = core::mem::take(&mut cube.current_lock_no_dirty[index]);
        let was_uploaded = cube.was_uploaded[index];
        (read_only, no_dirty, was_locked, was_uploaded)
    }

    fn mark_cube_dirty(&mut self, face: u32, level: usize) {
        if level >= (self.levels as usize).min(32) || face >= CUBE_FACE_COUNT {
            return;
        }
        let cube = self.cube.as_deref_mut().expect("cube storage");
        cube.dirty_masks[face as usize] |= 1 << level;
        // Preserve the established single-load fast gate. Face selection is
        // deferred to `flush_dirty_mips_slow` after this aggregate bit fires.
        self.dirty_mask |= 1 << level;
    }

    fn mark_cube_update_dirty(&mut self, face: u32, level: usize, rect: Option<DirtyRect>) {
        let Some(index) = self.cube_subresource_index(face, level) else {
            return;
        };
        let add =
            rect.unwrap_or_else(|| DirtyRect::full(self.mip_width(level), self.mip_height(level)));
        let cube = self.cube.as_deref_mut().expect("cube storage");
        cube.update_dirty[index] = Some(cube.update_dirty[index].map_or(add, |cur| {
            let x = cur.x.min(add.x);
            let y = cur.y.min(add.y);
            let right = (cur.x + cur.w).max(add.x + add.w);
            let bottom = (cur.y + cur.h).max(add.y + add.h);
            DirtyRect {
                x,
                y,
                w: right - x,
                h: bottom - y,
            }
        }));
    }

    /// Return a pointer into the staging buffer for `LockRect`.
    ///
    /// READONLY is a fast-path: the game promised it won't write, so
    /// two readers (the game + any in-flight GPU blit sourcing from
    /// `pending_blit_retention`'s Arc clone) can share the same
    /// backing Box with no race. Return `as_ptr()` directly — no
    /// rename, no allocation, no preserve memcpy. The pointer is cast
    /// to `*mut u8` only to satisfy the shared signature; the lock
    /// contract forbids writes through it.
    ///
    /// Writable locks delegate the policy decision to
    /// `decide_lock_action` in `mtld3d-core` — same shape as
    /// `vertex_buffer::vb_lock` consumes `buffer_rename::plan_lock`.
    /// `WriteInPlace` returns `as_ptr() as *mut u8` (the same cast
    /// READONLY uses) and trusts the well-behaved-game no-overlap
    /// contract for partial sub-rects. `FreshBox { preserve }`
    /// allocates a fresh uninit Box and applies the requested preserve
    /// (CPU memcpy when the game might read outside the locked rect
    /// through the Lock pointer, or when the encoder's compressed
    /// full-mip-fallback would read outside-rect bytes; otherwise no
    /// preserve). The old `Arc<PageBox>` stays alive via
    /// `pending_blit_retention` until GPU retire.
    fn lock_region_ptr(
        &mut self,
        level: usize,
        rect: Option<DirtyRect>,
        flags: u32,
    ) -> (*mut u8, u32, usize) {
        if self.d3d_pool == D3DPOOL_DEFAULT && self.d3d_usage & D3DUSAGE_DYNAMIC == 0 {
            DEFAULT_STATIC_LOCKS.fetch_add(1, Ordering::Relaxed);
        }
        if self.dropped_staging & (1u32 << level) != 0 && rect.is_some() {
            mtld3d_shared::log_once_warn_by!(
                target: crate::LOG_TARGET,
                key: self.texture_id.raw(),
                "texture {:#x}: partial lock of level {level} after its staging was released; \
                 pixels outside the rect no longer match the GPU copy",
                self.texture_id.raw()
            );
        }
        self.ensure_staging(level);
        let pitch = self.mip_bytes_per_row[level];
        let offset = mtld3d_core::texture_staging::texture_lock_offset(
            rect,
            pitch,
            self.block_w,
            self.block_h,
            self.block_bytes,
        );
        // Invariant: the Lock pointer must land strictly within the
        // staging Box. We hand the game a raw `*mut c_void` through
        // `D3DLOCKED_RECT.bits`, so Rust's bounds-checking is no help
        // past this point — the game then writes via `rep movsd`
        // outside our control. A bad offset here would surface as
        // either an `0xC0000005` access violation (page unmapped) or
        // as snmalloc-metadata corruption on an unrelated free much
        // later. Catch at construction.
        // `logical_len`, not `len()` — the page-padded tail is owned by
        // the PageBox but contains no mip data, so the overshoot
        // assertion must trip on a write past the actual mip bytes
        // (otherwise DXT compressed-format offset bugs land as snmalloc
        // metadata corruption visible only at a much later free).
        let staging_len = self.staging[level].logical_len();
        assert!(
            offset < staging_len,
            "texture LockRect offset {offset} >= staging[{level}].logical_len() {staging_len} \
             (mip={mw}×{mh}, pitch={pitch}, block={bw}×{bh}×{bb}, rect={rect:?})",
            mw = self.mip_widths[level],
            mh = self.mip_heights[level],
            bw = self.block_w,
            bh = self.block_h,
            bb = self.block_bytes,
        );

        if flags & D3DLOCK_READONLY != 0 {
            // Shared-reader fast path. No rename, no preserve memcpy.
            // Caller must honour the READONLY contract.
            let base = self.staging[level].as_ptr().cast_mut();
            // SAFETY: `offset` is the byte offset of the locked sub-rect
            // within the staging mip, computed and bounds-checked by
            // `decide_lock_action`; the staging `PageBox` holds at least
            // `offset + locked_bytes` bytes.
            let ptr = unsafe { base.add(offset) };
            return (ptr, pitch, offset);
        }

        // `device_inner == 0` after `detach_from_device` — texture is
        // between devices (post-Release, pre-rehydrate). Treat as "no GPU
        // activity": next bind on a new device will rehydrate and
        // re-upload, so an in-place write here is sound.
        //
        // The staging Box is read only by the texture-upload blit, not by
        // draws. That blit lives in its own command buffer (committed
        // before the draw CB) which retires ~a frame before full-frame
        // completion, so compare against `upload_coherent_seq` — the
        // staging is free as soon as the upload retires.
        let coherent_seq = if self.device_inner == 0 {
            0
        } else {
            DeviceInner::from_ptr(self.device_inner)
                .upload_coherent_seq_arc()
                .load(Ordering::Acquire)
        };
        let action = decide_lock_action(
            coherent_seq,
            self.last_submit_seq[level],
            flags,
            self.d3d_usage,
            rect,
            MipShape {
                mip_w: self.mip_widths[level],
                mip_h: self.mip_heights[level],
                block_w: self.block_w,
                block_h: self.block_h,
            },
        );

        let base: *mut u8 = match action {
            LockAction::WriteInPlace => {
                // No rename, no preserve — same primitive as the
                // READONLY fast-path above. `PageBox` exposes only
                // raw-pointer accessors, so no Rust `&[u8]` borrow of
                // the bytes lives across this cast. Encoder closures
                // hold Arc clones to keep the staging alive while
                // they construct `newBufferWithBytesNoCopy:` MTLBuffer
                // wrappers; they never borrow the bytes themselves.
                // The GPU read happens at command-buffer execution
                // time, after the next submit retires; under the
                // well-behaved-game no-overlap contract the locked
                // sub-rect doesn't overlap any in-flight read range.
                // Same model `vb_lock` now uses (see `plan_lock` doc).
                self.staging[level].as_ptr().cast_mut()
            }
            LockAction::FreshBox { preserve } => {
                // Logical mip-byte length, not the page-padded total
                // — the memcpy below moves exactly the bytes the game
                // can read.
                let mip_len = self.staging[level].logical_len();
                let fresh = new_uninit_page_box(mip_len);
                let old = core::mem::replace(&mut self.staging[level], Arc::new(fresh));
                // `device_inner == 0` (between devices): no perf state to
                // bump — perf lives on the freed `DeviceInner`. Skip the
                // counters but still execute the rename + optional preserve.
                let dev_inner_raw = self.device_inner;
                let perf_attached = dev_inner_raw != 0;
                match preserve {
                    PreserveKind::None => {
                        // Explicit DISCARD or whole-mip DYNAMIC:
                        // game promised it won't read, and the
                        // encoder's blit only reads the locked rect.
                        // The fresh allocation carries none of the old
                        // pixels, so the written union starts over.
                        self.reset_staging_coverage(level);
                        if perf_attached {
                            DeviceInner::from_ptr(dev_inner_raw)
                                .perf_mut()
                                .bump_texture_discard();
                        }
                    }
                    PreserveKind::Cpu => {
                        // Game might read old bytes through the Lock
                        // pointer (whole-mip non-WRITEONLY), OR the
                        // encoder's compressed full-mip-fallback will
                        // read bytes outside the locked rect. Either
                        // way: carry the old bytes across synchronously
                        // on the API thread. The `old` Arc keeps the
                        // source bytes live for the duration of the
                        // copy.
                        if perf_attached {
                            DeviceInner::from_ptr(dev_inner_raw)
                                .perf_mut()
                                .bump_texture_preserve_cpu();
                        }
                        let dst = Arc::get_mut(&mut self.staging[level])
                            .expect("fresh Arc is unique")
                            .as_mut_ptr();
                        // SAFETY: `old` and `dst` are distinct `PageBox`
                        // allocations of `mip_len` bytes (logical mip
                        // size); ranges don't alias.
                        unsafe {
                            core::ptr::copy_nonoverlapping(old.as_ptr(), dst, mip_len);
                        }
                    }
                }
                if perf_attached {
                    DeviceInner::from_ptr(dev_inner_raw)
                        .perf_mut()
                        .bump_texture_rename();
                }
                // Mirror `vb_lock` — the fresh Arc has never been on
                // the GPU. Reset so a back-to-back Lock pre-Unlock
                // doesn't see the fresh Box as contended and force a
                // second rename.
                self.last_submit_seq[level] = 0;
                Arc::get_mut(&mut self.staging[level])
                    .expect("fresh Arc is unique")
                    .as_mut_ptr()
            }
        };

        // SAFETY: `offset` is the byte offset of the locked sub-rect
        // within the staging mip, computed and bounds-checked by
        // `decide_lock_action`; `base` is the staging-mip allocation
        // (either in-place or freshly renamed) and holds at least
        // `offset + locked_bytes` bytes.
        let ptr = unsafe { base.add(offset) };
        (ptr, pitch, offset)
    }

    fn stash_lock(&mut self, level: usize, read_only: bool, no_dirty: bool) {
        self.current_lock_readonly[level] = read_only;
        self.current_lock_no_dirty[level] = no_dirty;
        self.locked[level] = true;
    }

    /// Consume the state stashed by `stash_lock` at `UnlockRect` time.
    ///
    /// Returns `(was_read_only, was_properly_locked)`.
    fn take_lock(&mut self, level: usize) -> (bool, bool) {
        let was_locked = self.locked[level];
        self.locked[level] = false;
        let read_only = core::mem::take(&mut self.current_lock_readonly[level]);
        (read_only, was_locked)
    }

    /// Whether `level` is currently mapped (`LockRect` held).
    ///
    /// Used by an offscreen-plain surface's `UnlockRect` to reject an
    /// unlock-without-lock.
    pub fn is_level_locked(&self, level: usize) -> bool {
        self.locked.get(level).copied().unwrap_or(false)
    }

    /// Set `level`'s bit in `dirty_mask`; the next bind-time flush re-uploads the mip.
    ///
    /// An out-of-range `level` is ignored, mirroring the bounds tolerance
    /// the format-conversion call sites relied on when this was a
    /// `Vec<bool>` written through `get_mut`.
    pub fn mark_mip_dirty(&mut self, level: usize) {
        if level < (self.levels as usize).min(32) {
            self.dirty_mask |= 1 << level;
            if let Some(slot) = self.pending_upload_rects.get_mut(level) {
                *slot = None;
            }
            if let Some(coverage) = self.staging_coverage.get_mut(level) {
                coverage.mark_full();
            }
        }
    }

    /// Mark `level` dirty for only `rect` of it (see `pending_upload_rects`).
    ///
    /// A level already dirty for its whole mip stays whole; partial marks
    /// union.
    pub fn mark_mip_dirty_rect(&mut self, level: usize, rect: DirtyRect) {
        if level >= (self.levels as usize).min(32) {
            return;
        }
        let (level_w, level_h) = (self.mip_widths[level], self.mip_heights[level]);
        if let Some(coverage) = self.staging_coverage.get_mut(level) {
            coverage.add(rect, level_w, level_h);
        }
        let already_full = self.dirty_mask & (1 << level) != 0
            && self
                .pending_upload_rects
                .get(level)
                .copied()
                .flatten()
                .is_none();
        self.dirty_mask |= 1 << level;
        if already_full {
            return;
        }
        if let Some(slot) = self.pending_upload_rects.get_mut(level) {
            *slot = Some(slot.map_or(rect, |cur| {
                let x = cur.x.min(rect.x);
                let y = cur.y.min(rect.y);
                let right = (cur.x + cur.w).max(rect.x + rect.w);
                let bottom = (cur.y + cur.h).max(rect.y + rect.h);
                DirtyRect {
                    x,
                    y,
                    w: right - x,
                    h: bottom - y,
                }
            }));
        }
    }

    /// Mark the rectangle a staging write covered dirty, narrowing where it can.
    ///
    /// A write covering the whole destination level marks it whole; a partial
    /// one narrows the upload to the written union. Volume levels always mark
    /// whole: the upload path has no sub-rect form for them.
    fn mark_written_region(&mut self, level: usize, rect: DirtyRect) {
        let whole = rect.x == 0
            && rect.y == 0
            && rect.w >= self.mip_width(level)
            && rect.h >= self.mip_height(level);
        if whole || self.depth > 1 {
            self.mark_mip_dirty(level);
        } else {
            self.mark_mip_dirty_rect(level, rect);
        }
    }

    /// Union `rect` (the whole mip when `None`) into the source dirty region for `level`.
    ///
    /// Called on every `AddDirtyRect` and non-`READONLY` `UnlockRect`.
    pub fn mark_update_dirty(&mut self, level: usize, rect: Option<DirtyRect>) {
        if level >= self.update_dirty.len() {
            return;
        }
        let add =
            rect.unwrap_or_else(|| DirtyRect::full(self.mip_width(level), self.mip_height(level)));
        self.update_dirty[level] = Some(self.update_dirty[level].map_or(add, |cur| {
            let x = cur.x.min(add.x);
            let y = cur.y.min(add.y);
            let right = (cur.x + cur.w).max(add.x + add.w);
            let bottom = (cur.y + cur.h).max(add.y + add.h);
            DirtyRect {
                x,
                y,
                w: right - x,
                h: bottom - y,
            }
        }));
    }

    /// The source dirty region for `level` (`None` = clean).
    ///
    /// Read by `UpdateTexture` to copy only what changed.
    pub fn update_dirty_rect(&self, level: usize) -> Option<DirtyRect> {
        self.update_dirty.get(level).copied().flatten()
    }

    /// Source dirty region for one cube face mip, or `None` when clean.
    pub fn cube_update_dirty_rect(&self, face: u32, level: usize) -> Option<DirtyRect> {
        let index = self.cube_subresource_index(face, level)?;
        self.cube
            .as_deref()?
            .update_dirty
            .get(index)
            .copied()
            .flatten()
    }

    /// Clear every mip's source dirty region — done after a successful copy.
    pub fn clear_all_update_dirty(&mut self) {
        for d in &mut self.update_dirty {
            *d = None;
        }
    }

    /// Clear every cube face mip's source dirty region after `UpdateTexture`.
    pub fn clear_all_cube_update_dirty(&mut self) {
        if let Some(cube) = self.cube.as_deref_mut() {
            for d in &mut cube.update_dirty {
                *d = None;
            }
        }
    }

    /// The app-set `SetLOD` value (the most-detailed mip the runtime may use).
    pub const fn lod(&self) -> u32 {
        self.lod
    }
}

/// The simple uncompressed RGB formats the cross-format `StretchRect` converter handles.
///
/// The converter is the offscreen-plain destination path
/// ([`TextureInner::convert_sub_region_from`]). Any other pair takes the
/// best-effort no-op path (the HR still succeeds).
const fn is_convertible_rgb(d3d_format: u32) -> bool {
    matches!(
        d3d_format,
        D3DFMT_A8R8G8B8 | D3DFMT_X8R8G8B8 | D3DFMT_R5G6B5
    )
}

/// Bytes per pixel of an `is_convertible_rgb` format.
const fn rgb_bpp(d3d_format: u32) -> usize {
    match d3d_format {
        D3DFMT_R5G6B5 => 2,
        _ => 4, // A8R8G8B8 / X8R8G8B8
    }
}

/// Decode one `is_convertible_rgb` pixel from its little-endian bytes into `(r, g, b, a)`.
///
/// The 32bpp formats store `[B, G, R, A/X]`; R5G6B5 packs `RRRRR GGGGGG BBBBB`
/// into a little-endian `u16` (channels bit-replicated up to 8 bits). X8's
/// alpha reads as opaque. `px` is at least `rgb_bpp` long.
fn decode_rgb_pixel(d3d_format: u32, px: &[u8]) -> (u8, u8, u8, u8) {
    match d3d_format {
        D3DFMT_R5G6B5 => {
            let v = u16::from_le_bytes([px[0], px[1]]);
            let r5 = ((v >> 11) & 0x1f) as u8;
            let g6 = ((v >> 5) & 0x3f) as u8;
            let b5 = (v & 0x1f) as u8;
            (
                (r5 << 3) | (r5 >> 2),
                (g6 << 2) | (g6 >> 4),
                (b5 << 3) | (b5 >> 2),
                0xff,
            )
        }
        // X8R8G8B8: alpha byte is undefined; report opaque.
        D3DFMT_X8R8G8B8 => (px[2], px[1], px[0], 0xff),
        // A8R8G8B8 (only remaining is_convertible_rgb case).
        _ => (px[2], px[1], px[0], px[3]),
    }
}

/// Encode `(r, g, b, a)` into one `is_convertible_rgb` pixel's little-endian bytes.
///
/// Inverse of `decode_rgb_pixel`. X8's byte is written opaque. `out` is at
/// least `rgb_bpp` long.
fn encode_rgb_pixel(d3d_format: u32, (r, g, b, a): (u8, u8, u8, u8), out: &mut [u8]) {
    match d3d_format {
        D3DFMT_R5G6B5 => {
            let packed = (u16::from(r >> 3) << 11) | (u16::from(g >> 2) << 5) | u16::from(b >> 3);
            out[..2].copy_from_slice(&packed.to_le_bytes());
        }
        D3DFMT_X8R8G8B8 => {
            out[0] = b;
            out[1] = g;
            out[2] = r;
            out[3] = 0xff;
        }
        // A8R8G8B8 (only remaining is_convertible_rgb case).
        _ => {
            out[0] = b;
            out[1] = g;
            out[2] = r;
            out[3] = a;
        }
    }
}

/// Allocate `len` uninitialized bytes in a page-aligned `PageBox`.
///
/// Used by the `FreshBox` Lock path and by `CreateTexture` for initial
/// staging — the game writes the dirty rect before any GPU read, and
/// the blit upload only copies the dirty sub-rect, so untouched bytes
/// are never observed. On an initial Draw-before-Lock, the freshly-
/// created `MTLTexture` is the GPU-visible surface and is zeroed by
/// Metal; the staging `PageBox` is only read when an upload blit fires,
/// which requires a prior Lock write.
///
/// Page-aligned because the encoder wraps the staging via
/// `newBufferWithBytesNoCopy:`, which on non-UMA Macs (Intel/AMD)
/// rejects misaligned pointer or length. Apple Silicon tolerates the
/// misalignment in practice but documents the same contract.
pub fn new_uninit_page_box(len: usize) -> PageBox {
    PageBox::new_uninit(len)
}

/// Parameters for `Direct3DTexture9::new`.
///
/// Keeps the constructor signature from exploding into a dozen positional
/// arguments.
pub struct TextureCreateInfo {
    pub texture_id: TextureId,
    pub device_handle: MetalHandle<MTLDeviceKind>,
    pub device_inner: u64,
    pub width: u32,
    pub height: u32,
    /// 1 for 2D textures; >1 for a volume (3D) texture.
    pub depth: u32,
    pub levels: u32,
    pub d3d_format: u32,
    pub metal_pixel_format: PixelFormat,
    /// Packed boolean attributes — see [`TextureFlags`].
    pub flags: TextureFlags,
    pub swizzle: Option<[Swizzle; 4]>,
    pub usage_flags: TextureUsage,
    /// Raw D3D9 `D3DUSAGE_*` bits as passed to `CreateTexture`.
    ///
    /// Kept on `TextureInner` alongside the Metal `usage_flags` so
    /// `lock_region_ptr` can gate the non-WRITEONLY CPU-memcpy preserve on the
    /// contended-rename path.
    pub d3d_usage: u32,
    /// Scale for the backing Metal texture; see [`TextureInner::render_scale`].
    pub render_scale: RenderScale,
    /// The `D3DPOOL` the texture was created in.
    ///
    /// Drives device-refcount forwarding (see [`TextureInner::d3d_pool`]).
    pub d3d_pool: u32,
    pub bytes_per_pixel: u32,
    /// Format block geometry, populated from `FormatMapping` at create time.
    ///
    /// For uncompressed: `(1, 1, bytes_per_pixel)`. For DXT (BC1/2/3):
    /// `(4, 4, 8 or 16)`. Carried on `TextureInner` so `lock_region_ptr` can
    /// compute a correct sub-rect offset for compressed mips without
    /// re-resolving the format on every Lock.
    pub block_w: u32,
    pub block_h: u32,
    pub block_bytes: u32,
    /// Per-mip owned staging bytes.
    ///
    /// Page-aligned + page-sized so the encoder can wrap them via
    /// `newBufferWithBytesNoCopy:`. `Direct3DTexture9::new` wraps each entry in
    /// an `Arc<PageBox>` for refcount-based handoff to the encoder thread.
    /// Empty for depth-format textures (`LockRect` rejected upstream).
    pub staging: Vec<PageBox>,
    pub mip_widths: Vec<u32>,
    pub mip_heights: Vec<u32>,
    pub mip_bytes_per_row: Vec<u32>,
}

#[repr(C)]
pub struct Direct3DTexture9 {
    vtbl: *const IDirect3DTexture9Vtbl,
    refcount: u32,
    /// Device-internal "bound slot" refcount, kept in sync by `CachedComPtr<_, Bound>`.
    ///
    /// The wrapper is destroyed only when both `refcount` and
    /// `private_refcount` reach zero — the private count is a device-internal
    /// binding refcount distinct from the public `IUnknown` count.
    private_refcount: u32,
    inner: *mut TextureInner,
}

/// Build the shared `TextureInner` and register it with the owning device.
///
/// Used by both `Direct3DTexture9::new` (2D) and `Direct3DVolumeTexture9::new`
/// (3D) — the COM wrappers differ only in their vtable; the backing state is
/// identical (with `depth > 1` for volumes).
fn build_texture_inner(info: TextureCreateInfo) -> *mut TextureInner {
    // Depth-format textures carry an empty staging Vec (no CPU
    // upload path), so size the per-mip tracking arrays from
    // `levels` instead. For color textures the two are equal.
    let is_cube = info.flags.contains(TextureFlags::CUBE);
    let mip_count = if is_cube {
        0
    } else if info.flags.contains(TextureFlags::DEPTH_FORMAT) {
        info.levels as usize
    } else {
        info.staging.len()
    };
    let dev_ptr = info.device_inner;
    // The two system-memory pools get no Metal texture, so the residency bit
    // is derived from the pool here rather than restated at each Create*.
    let mut flags = info.flags;
    flags.set(
        TextureFlags::CPU_ONLY,
        mtld3d_core::pool::is_cpu_only(info.d3d_pool),
    );
    // A freshly created texture is fully dirty as an UpdateTexture source until
    // its first copy (per the D3D9 spec, a managed texture starts full-dirty).
    let update_dirty: Vec<Option<DirtyRect>> = (0..mip_count)
        .map(|i| {
            let w = info.mip_widths.get(i).copied().unwrap_or(info.width).max(1);
            let h = info
                .mip_heights
                .get(i)
                .copied()
                .unwrap_or(info.height)
                .max(1);
            Some(DirtyRect::full(w, h))
        })
        .collect();
    let (staging, cube): (Vec<Arc<PageBox>>, Option<Box<CubeStorage>>) = if is_cube {
        (
            Vec::new(),
            Some(Box::new(CubeStorage::new(
                info.staging,
                info.levels as usize,
                info.width,
                info.height,
            ))),
        )
    } else {
        (info.staging.into_iter().map(Arc::new).collect(), None)
    };
    let mut boxed = Box::new(TextureInner {
        texture_id: info.texture_id,
        device_handle: info.device_handle,
        device_inner: info.device_inner,
        width: info.width,
        height: info.height,
        depth: info.depth,
        levels: info.levels,
        d3d_format: info.d3d_format,
        metal_pixel_format: info.metal_pixel_format,
        flags,
        swizzle: info.swizzle,
        usage_flags: info.usage_flags,
        d3d_usage: info.d3d_usage,
        render_scale: info.render_scale,
        autogen_filter_type: D3DTEXF_LINEAR,
        lod: 0,
        d3d_pool: info.d3d_pool,
        priority: 0,
        staging,
        dropped_staging: 0,
        staging_coverage: Vec::new(),
        gpu_authoritative: 0,
        mip_widths: info.mip_widths,
        mip_heights: info.mip_heights,
        mip_bytes_per_row: info.mip_bytes_per_row,
        bytes_per_pixel: info.bytes_per_pixel,
        block_w: info.block_w,
        block_h: info.block_h,
        block_bytes: info.block_bytes,
        dirty_mask: 0,
        pending_upload_rects: vec![None; mip_count],
        current_lock_readonly: vec![false; mip_count],
        current_lock_no_dirty: vec![false; mip_count],
        last_submit_seq: vec![0; mip_count],
        was_uploaded: vec![false; mip_count],
        locked: vec![false; mip_count],
        update_dirty,
        cube,
        private_data: PrivateDataStore::default(),
        subresources: Vec::new(),
    });
    // Only a texture whose class can release its staging tracks what has been
    // written into it; every other one keeps an empty Vec.
    if boxed.staging_droppable_class() {
        boxed.staging_coverage = core::iter::repeat_with(StagingCoverage::new)
            .take(boxed.staging.len())
            .collect();
    }
    // A default-pool static texture's staging only carries writes to the
    // GPU, and a streaming engine creates far more textures than it ever
    // writes through this device. Release the pages now and let
    // `ensure_staging` materialize a level on its first write; a whole-level
    // upload then releases it again ([`schedule_upload`]).
    for level in 0..boxed.staging.len() {
        if boxed.staging_droppable(level) {
            boxed.drop_staging(level);
        }
    }
    let inner = Box::into_raw(boxed);
    DeviceInner::from_ptr(dev_ptr).register_texture(inner);
    inner
}

impl Direct3DTexture9 {
    pub fn new(info: TextureCreateInfo) -> Self {
        Self {
            vtbl: &raw const DIRECT3D_TEXTURE9_VTBL,
            refcount: 1,
            private_refcount: 0,
            inner: build_texture_inner(info),
        }
    }

    pub const fn vtbl(&self) -> &IDirect3DTexture9Vtbl {
        // SAFETY: `self.vtbl` is the `'static` `DIRECT3D_TEXTURE9_VTBL`
        // installed at `Self::new`.
        unsafe { &*self.vtbl }
    }

    pub fn inner(&self) -> &TextureInner {
        // SAFETY: `self.inner` was installed by `Self::new` as a
        // `Box::into_raw` and is dropped only in `tex_release` at refcount
        // zero, so it stays live for every live wrapper reference.
        unsafe { &*self.inner }
    }

    pub fn inner_mut(&mut self) -> &mut TextureInner {
        // SAFETY: see [`Self::inner`] — same `Box::into_raw` lifetime
        // contract; `&mut self` guarantees exclusive access.
        unsafe { &mut *self.inner }
    }

    /// Raw pointer to the per-resource `LockRect`/`GetDC` state shared by cube face shells.
    ///
    /// Called through a raw `*mut Direct3DTexture9` from a face surface, so it
    /// takes `&self` and points into the inner allocation: D3D9 objects are
    /// single-threaded, so no two faces touch it concurrently. A raw pointer
    /// (not `&mut`) so the caller can hold it alongside an unrelated borrow of
    /// the face surface.
    pub fn dc_lock_state_ptr(&self) -> *mut DcLockState {
        // SAFETY: `self.inner` is the live `Box::into_raw(TextureInner)` from
        // `Self::new`; single-threaded access makes the exclusive reborrow sound.
        let inner = unsafe { &mut *self.inner };
        let cube = inner
            .cube
            .as_deref_mut()
            .expect("cube face has cube storage");
        &raw mut cube.dc_lock
    }

    pub fn texture_id(&self) -> TextureId {
        self.inner().texture_id
    }

    pub fn d3d_format(&self) -> u32 {
        self.inner().d3d_format
    }

    /// D3DUSAGE_* flags the texture was created with (e.g. RENDERTARGET).
    pub fn d3d_usage(&self) -> u32 {
        self.inner().d3d_usage
    }

    /// D3DPOOL_* the texture was created in.
    pub fn d3d_pool(&self) -> u32 {
        self.inner().d3d_pool
    }

    pub fn metal_pixel_format(&self) -> PixelFormat {
        self.inner().metal_pixel_format
    }

    /// True for sampleable depth-format textures (shadow maps).
    ///
    /// `SetTexture` reads this to set the per-stage depth-sampler bit
    /// so the DXSO emitter picks the `depth2d<float>` MSL type for
    /// the bound slot.
    pub fn is_depth_format(&self) -> bool {
        self.inner().flags.contains(TextureFlags::DEPTH_FORMAT)
    }

    /// True when the backing Metal texture is `MTLTextureType3D`.
    ///
    /// Uses the same predicate as the unix-side create (`depth > 1`): a
    /// `CreateVolumeTexture` resource with a single depth slice is created
    /// as a plain 2D texture on both sides, so it must NOT set the
    /// per-stage volume-sampler bit either.
    pub fn is_volume(&self) -> bool {
        self.inner().depth > 1
    }

    pub fn is_cube(&self) -> bool {
        self.inner().flags.contains(TextureFlags::CUBE)
    }
}

// ── IUnknown ──

#[inline]
fn tex_timer(this: *mut c_void) -> mtld3d_core::perf::ApiTimer {
    use mtld3d_core::perf::{ApiCategory, ApiTimer};
    // SAFETY: vtable thunk; `this` is *mut Direct3DTexture9 per IDirect3DTexture9 ABI.
    let perf_ptr = (unsafe { InPtr::<Direct3DTexture9>::opt(this) })
        .map_or(core::ptr::null_mut(), |obj| {
            DeviceInner::perf_ptr_of(obj.inner().device_inner as *mut DeviceInner)
        });
    ApiTimer::start(perf_ptr, ApiCategory::Texture)
}

extern "system" fn texture_query_interface(
    this: *mut c_void,
    riid: *const Guid,
    ppv: *mut *mut c_void,
) -> i32 {
    let _timer = tex_timer(this);
    // The leaf interface follows the wrapper's kind; the 2D, cube and volume
    // textures share this vtable slot.
    // SAFETY: vtable `this` is the live texture wrapper for this call.
    let kind =
        unsafe { InPtr::<Direct3DTexture9>::opt(this) }.map(|obj| (obj.is_cube(), obj.is_volume()));
    let (leaf, name) = match kind {
        Some((true, _)) => (IID_IDIRECT3DCUBETEXTURE9, "IDirect3DCubeTexture9"),
        Some((false, true)) => (IID_IDIRECT3DVOLUMETEXTURE9, "IDirect3DVolumeTexture9"),
        _ => (IID_IDIRECT3DTEXTURE9, "IDirect3DTexture9"),
    };
    // SAFETY: vtable thunk; `this`, `riid` and `ppv` are the caller's per the
    // IUnknown::QueryInterface ABI.
    unsafe {
        crate::com_ref::com_query_interface(
            this,
            riid,
            ppv,
            &[
                IID_IUNKNOWN,
                IID_IDIRECT3DRESOURCE9,
                IID_IDIRECT3DBASETEXTURE9,
                leaf,
            ],
            texture_add_ref,
            name,
        )
    }
}

// Shared by the 2D and volume texture vtables: both wrappers have the identical
// `{ vtbl, refcount, private_refcount, inner: *mut TextureInner }` layout, so
// the engine treats either as `Direct3DTexture9` for refcount purposes.
extern "system" fn texture_add_ref(this: *mut c_void) -> u32 {
    let _timer = tex_timer(this);
    // SAFETY: IDirect3DTexture9/IDirect3DVolumeTexture9 IUnknown AddRef thunk;
    // the D3D9 ABI guarantees `this` is the live wrapper for the call.
    unsafe { crate::com_ref::com_add_ref::<Direct3DTexture9>(this) }
}

extern "system" fn texture_release(this: *mut c_void) -> u32 {
    let _timer = tex_timer(this);
    // SAFETY: IDirect3DTexture9/IDirect3DVolumeTexture9 IUnknown Release thunk;
    // the D3D9 ABI guarantees `this` is the live wrapper for the call.
    unsafe { crate::com_ref::com_release::<Direct3DTexture9>(this) }
}

/// Destroy a `Direct3DTexture9` wrapper once both `refcount` and `private_refcount` are zero.
///
/// Pushes the Metal-handle teardown to the encoder thread, deregisters from
/// the device's live-textures registry, then frees the inner + outer
/// allocations.
///
/// # Safety
/// `this` must point to a live `Direct3DTexture9` wrapper with both
/// counters at zero; caller must not access the wrapper afterwards.
unsafe fn finalize_texture(this: *mut Direct3DTexture9) {
    // SAFETY: caller asserts wrapper still live; both counters at zero
    // means no other reference can be outstanding.
    let obj = unsafe { &*this };
    let inner_ptr = obj.inner;
    let ti = obj.inner();
    let texture_id = ti.texture_id;
    let dev_inner_raw = ti.device_inner;
    // Mirror of the "target texture created" line: target textures are rare
    // and their release timing decides cross-pass data flow.
    if obj.d3d_usage() & (mtld3d_types::D3DUSAGE_RENDERTARGET | mtld3d_types::D3DUSAGE_DEPTHSTENCIL)
        != 0
    {
        log::debug!(
            target: crate::LOG_TARGET,
            "target texture destroyed: {texture_id:?} ptr={this:p}"
        );
    }

    // `device_inner == 0` after `detach_from_device` — the owning
    // device has already been released and torn down (its
    // `shutdown_cleanup` already drained the texture cache + freed
    // the matching `MTLTexture`). No closure to push, no live
    // registry to drop from. Just free the PE-side allocations.
    if dev_inner_raw != 0 {
        let dev = DeviceInner::from_ptr(dev_inner_raw);
        // Push cleanup closure to encoder thread — it owns the Metal handle
        dev.push_op(Box::new(move |enc| {
            enc.destroy_cached_texture(texture_id);
        }));
        // Drop from the live-textures registry before freeing the
        // inner Box so `evict_managed_resources` never sees a dangling
        // pointer.
        dev.deregister_texture(inner_ptr);
    }
    // Free the sub-resource wrappers this texture cached for `GetSurfaceLevel` /
    // `GetCubeMapSurface` / `GetVolumeLevel`. They survive their own last
    // `Release` so the getters keep handing back one identity, which makes the
    // container their single free site. Both of a cached wrapper's counters are
    // necessarily zero here: a live one holds a public or private reference on
    // this texture, and we would not be finalizing.
    // SAFETY: `inner_ptr` is the live `TextureInner` about to be freed.
    let ti_mut = unsafe { &mut *inner_ptr };
    let volume_levels = ti_mut.flags.contains(TextureFlags::VOLUME_TEXTURE);
    for slot in ti_mut.take_subresources() {
        if volume_levels {
            // SAFETY: a non-zero slot of a volume texture is a live cached
            // `Direct3DVolume9` shell this texture owns; freed exactly once here.
            unsafe { finalize_cached_volume(slot) };
        } else {
            // SAFETY: a non-zero slot is a live cached sub-resource surface this
            // texture owns; finalized exactly once here.
            unsafe { crate::surface::finalize_cached_surface(slot) };
        }
    }
    // A cube texture finalizing with a face's `GetDC` never released would
    // otherwise leak the memory DC + DIB held on the shared state; tear it down.
    // (The cube outlives every face referencing it, so this is the last owner.)
    if let Some(cube) = ti_mut.cube.as_deref_mut() {
        cube.dc_lock.teardown();
    }
    // SAFETY: both counters reached zero; `inner_ptr` is the original
    // `Box::into_raw(TextureInner)` from `Self::new` and no other
    // reference can survive.
    drop(unsafe { Box::from_raw(inner_ptr) });
    // SAFETY: both counters reached zero; `this` is the original
    // `Box::into_raw(Direct3DTexture9)` allocation.
    drop(unsafe { Box::from_raw(this) });
}

impl ComUnknown for Direct3DTexture9 {
    fn vtbl_add_ref(&self) -> unsafe extern "system" fn(*mut c_void) -> u32 {
        self.vtbl().add_ref
    }
    fn vtbl_release(&self) -> unsafe extern "system" fn(*mut c_void) -> u32 {
        self.vtbl().release
    }
    fn private_refcount_inc(&mut self) {
        self.private_refcount += 1;
    }
    unsafe fn private_refcount_dec_maybe_finalize(this: *mut Self) {
        // SAFETY: caller asserts `this` points to a live wrapper with
        // at least one private refcount outstanding.
        let obj = unsafe { &mut *this };
        obj.private_refcount -= 1;
        if obj.refcount == 0 && obj.private_refcount == 0 {
            // SAFETY: both counters reached zero — no other reference
            // can survive; finalize takes exclusive ownership.
            unsafe { finalize_texture(this) };
        }
    }
}

// SAFETY: `refcount_mut`/`private_refcount` expose this wrapper's own counters;
// `finalize` frees it exactly once when both reach zero. Shared by the 2D and
// volume texture vtables (identical layout) via the `texture_*` thunks.
unsafe impl crate::com_ref::ComChild for Direct3DTexture9 {
    fn refcount_mut(&mut self) -> &mut u32 {
        &mut self.refcount
    }
    fn blocks_reset_while_referenced(&self) -> bool {
        self.inner().is_default_pool()
    }
    fn private_refcount(&self) -> u32 {
        self.private_refcount
    }
    fn device_forward_target(&self) -> *mut c_void {
        let inner = self.inner();
        // Managed textures do not pin the device (they outlive it and migrate),
        // a detached texture (device_inner == 0, "between devices") has no
        // device to forward to, and a CPU-only cube shell has no device object
        // that must not keep the device alive.
        if inner.d3d_pool == mtld3d_types::D3DPOOL_MANAGED
            || inner.device_inner == 0
            || (inner.flags.contains(TextureFlags::CUBE)
                && matches!(
                    inner.d3d_pool,
                    mtld3d_types::D3DPOOL_SYSTEMMEM | mtld3d_types::D3DPOOL_SCRATCH
                ))
        {
            return core::ptr::null_mut();
        }
        self.owning_device()
    }
    fn owning_device(&self) -> *mut c_void {
        // Every pool answers here, including the managed and cube-shell cases
        // the forwarding target excludes: those opt out of pinning the device,
        // not out of having one. Null only once the device is gone, which
        // `detach_from_device` zeroes at teardown.
        let inner = self.inner();
        if inner.device_inner == 0 {
            return core::ptr::null_mut();
        }
        DeviceInner::from_ptr(inner.device_inner).device_wrapper()
    }
    unsafe fn finalize(this: *mut Self) {
        // SAFETY: forwarded from the engine — both counters are zero.
        unsafe { finalize_texture(this) };
    }
}

// ── IDirect3DResource9 ──

extern "system" fn texture_get_device(this: *mut c_void, device: *mut *mut c_void) -> i32 {
    let _timer = tex_timer(this);
    // SAFETY: IDirect3DTexture9 / IDirect3DVolumeTexture9 / IDirect3DCubeTexture9
    // GetDevice thunk: all three vtables share it, and their wrappers share a
    // layout, as the refcount thunks above already rely on. `device` is the
    // caller's out-param.
    unsafe { crate::com_ref::com_get_device::<Direct3DTexture9>(this, device) }
}

extern "system" fn texture_set_private_data(
    this: *mut c_void,
    guid: *const Guid,
    data: *const c_void,
    size: u32,
    flags: u32,
) -> i32 {
    let _timer = tex_timer(this);
    // SAFETY: vtable in-param; `guid` is *const Guid per IDirect3DResource9 ABI.
    let Some(guid) = (unsafe { InPtr::<Guid>::opt(guid.cast()) }) else {
        return D3DERR_INVALIDCALL;
    };
    // SAFETY: vtable thunk; `this` is *mut Direct3DTexture9 per ABI (shared by the
    // 2D/cube/volume-texture vtbls).
    let Some(mut obj) = (unsafe { InPtrMut::<Direct3DTexture9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let store = &mut obj.inner_mut().private_data;
    // SAFETY: `data`/`size`/`flags` are the caller-supplied payload; `set` validates.
    unsafe { store.set(&guid, data, size, flags) }
}

extern "system" fn texture_get_private_data(
    this: *mut c_void,
    guid: *const Guid,
    data: *mut c_void,
    size: *mut u32,
) -> i32 {
    let _timer = tex_timer(this);
    // SAFETY: vtable in-param; `guid` is *const Guid per IDirect3DResource9 ABI.
    let Some(guid) = (unsafe { InPtr::<Guid>::opt(guid.cast()) }) else {
        return D3DERR_INVALIDCALL;
    };
    // SAFETY: vtable thunk; `this` is *mut Direct3DTexture9 per ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DTexture9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    // SAFETY: `data`/`size` are caller out-params per the D3D9 ABI; `get` validates.
    unsafe { obj.inner().private_data.get(&guid, data, size) }
}

extern "system" fn texture_free_private_data(this: *mut c_void, guid: *const Guid) -> i32 {
    let _timer = tex_timer(this);
    // SAFETY: vtable in-param; `guid` is *const Guid per IDirect3DResource9 ABI.
    let Some(guid) = (unsafe { InPtr::<Guid>::opt(guid.cast()) }) else {
        return D3DERR_INVALIDCALL;
    };
    // SAFETY: vtable thunk; `this` is *mut Direct3DTexture9 per ABI.
    let Some(mut obj) = (unsafe { InPtrMut::<Direct3DTexture9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    obj.inner_mut().private_data.free(&guid)
}

// Priority is honoured only for `D3DPOOL_MANAGED` resources (D3D9 manager
// eviction order). For every other pool both accessors are fixed at `0`.
// Metal has no eviction-order hint, so the value is stored and round-tripped
// but never acted upon.
extern "system" fn texture_set_priority(this: *mut c_void, priority: u32) -> u32 {
    let _timer = tex_timer(this);
    // SAFETY: vtable thunk; `this` is *mut Direct3DTexture9 per IDirect3DTexture9 ABI.
    let Some(mut obj) = (unsafe { InPtrMut::<Direct3DTexture9>::opt(this) }) else {
        return 0;
    };
    let ti = obj.inner_mut();
    if ti.d3d_pool != mtld3d_types::D3DPOOL_MANAGED {
        return 0;
    }
    core::mem::replace(&mut ti.priority, priority)
}

extern "system" fn texture_get_priority(this: *mut c_void) -> u32 {
    let _timer = tex_timer(this);
    // SAFETY: vtable thunk; `this` is *mut Direct3DTexture9 per IDirect3DTexture9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DTexture9>::opt(this) }) else {
        return 0;
    };
    obj.inner().priority
}

extern "system" fn texture_pre_load(this: *mut c_void) {
    let _timer = tex_timer(this);
    // PreLoad is a hint to bring a managed texture into VRAM. Metal
    // has no equivalent (textures live in unified memory, the driver
    // resident-set is implicit), so this is an intentional no-op.
    // Logged once at info so it doesn't show up as a port candidate
    // in routine `RUST_LOG=warn` triage.
    mtld3d_shared::log_once_info!(
        target: crate::LOG_TARGET,
        "IDirect3DTexture9::PreLoad: no Metal analog, no-op"
    );
}

extern "system" fn texture_get_type(this: *mut c_void) -> u32 {
    let _timer = tex_timer(this);
    3 // D3DRTYPE_TEXTURE
}

// ── IDirect3DBaseTexture9 ──

extern "system" fn texture_set_lod(this: *mut c_void, lod: u32) -> u32 {
    let _timer = tex_timer(this);
    // SAFETY: vtable thunk; `this` is *mut Direct3DTexture9 per the ABI.
    let Some(mut obj) = (unsafe { InPtrMut::<Direct3DTexture9>::opt(this) }) else {
        return 0;
    };
    let ti = obj.inner_mut();
    // D3D9 honours SetLOD only for D3DPOOL_MANAGED textures (it picks the
    // most-detailed resident mip); other pools ignore it and report 0. The
    // accepted value is clamped to the last mip. Returns the PREVIOUS LOD.
    if ti.d3d_pool != D3DPOOL_MANAGED {
        return 0;
    }
    let prev = ti.lod;
    ti.lod = lod.min(ti.levels.saturating_sub(1));
    prev
}

extern "system" fn texture_get_lod(this: *mut c_void) -> u32 {
    let _timer = tex_timer(this);
    // SAFETY: vtable thunk; `this` is *mut Direct3DTexture9 per the ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DTexture9>::opt(this) }) else {
        return 0;
    };
    obj.inner().lod
}

extern "system" fn texture_get_level_count(this: *mut c_void) -> u32 {
    let _timer = tex_timer(this);
    // SAFETY: vtable thunk; `this` is *mut Direct3DTexture9 per IDirect3DTexture9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DTexture9>::opt(this) }) else {
        return 0;
    };
    obj.inner().app_level_count()
}

// AUTOGENMIPMAP filter type: Metal's `generateMipmaps` always uses
// linear filtering; we accept any D3DTEXF_* the game sets but the
// effective filter is always linear. Report that truthfully via Get,
// log once-by-value when Set asks for something other than LINEAR so
// "this game wanted point-filtered mipgen" is visible without spam.
extern "system" fn texture_set_auto_gen_filter_type(this: *mut c_void, filter_type: u32) -> i32 {
    let _timer = tex_timer(this);
    // D3DTEXF_NONE is not a valid autogen filter (the chain must be generated
    // with *some* filter). Metal's generateMipmaps is fixed-linear, so any
    // other value is stored as app-visible state but does not change the chain.
    if filter_type == D3DTEXF_NONE {
        return D3DERR_INVALIDCALL;
    }
    if filter_type != D3DTEXF_LINEAR {
        mtld3d_shared::log_once_warn_by!(target: crate::LOG_TARGET, key: u64::from(filter_type),
            "SetAutoGenFilterType({filter_type}): Metal generateMipmaps always uses linear, request stored but honoured as LINEAR"
        );
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DTexture9 per IDirect3DTexture9 ABI.
    let Some(mut obj) = (unsafe { InPtrMut::<Direct3DTexture9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    obj.inner_mut().autogen_filter_type = filter_type;
    0 // S_OK
}

extern "system" fn texture_get_auto_gen_filter_type(this: *mut c_void) -> u32 {
    let _timer = tex_timer(this);
    // SAFETY: vtable thunk; `this` is *mut Direct3DTexture9 per IDirect3DTexture9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DTexture9>::opt(this) }) else {
        return D3DTEXF_LINEAR;
    };
    obj.inner().autogen_filter_type
}

// Game-driven explicit mip regeneration. For an AUTOGENMIPMAP texture
// it pushes the same `run_generate_mipmaps` op the upload path uses.
// For a non-AUTOGENMIPMAP texture the D3D9 spec leaves it undefined —
// log once and do nothing.
extern "system" fn texture_generate_mip_sub_levels(this: *mut c_void) {
    let _timer = tex_timer(this);
    // SAFETY: vtable thunk; `this` is *mut Direct3DTexture9 per IDirect3DTexture9 ABI.
    let Some(mut obj) = (unsafe { InPtrMut::<Direct3DTexture9>::opt(this) }) else {
        return;
    };
    let ti = obj.inner_mut();
    if !ti.autogen_mipmap() {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "GenerateMipSubLevels on non-AUTOGENMIPMAP texture → no-op"
        );
        return;
    }
    let texture_id = ti.texture_id;
    if ti.device_inner == 0 {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "GenerateMipSubLevels on detached texture (device released) → no-op"
        );
        return;
    }
    let dev = DeviceInner::from_ptr(ti.device_inner);
    dev.push_op(Box::new(move |enc: &mut FrameEncoder| {
        enc.run_generate_mipmaps(texture_id);
    }));
}

// ── IDirect3DTexture9 ──

extern "system" fn texture_get_level_desc(
    this: *mut c_void,
    level: u32,
    desc: *mut D3DSURFACE_DESC,
) -> i32 {
    let _timer = tex_timer(this);
    // SAFETY: vtable thunk; `this` is *mut Direct3DTexture9 per IDirect3DTexture9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DTexture9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let ti = obj.inner();
    if level >= ti.app_level_count() || desc.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: `desc` is non-null (checked above) and per the D3D9 ABI
    // points to a writable `D3DSURFACE_DESC` slot owned by the caller.
    let out = unsafe { &mut *desc };
    out.format = ti.d3d_format;
    // A texture level is itself a surface, so its `D3DSURFACE_DESC.Type`
    // reports `D3DRTYPE_SURFACE` even though `GetType` on the container texture
    // returns `D3DRTYPE_TEXTURE`. Mirrors `surface_get_desc`.
    out.resource_type = D3DRTYPE_SURFACE;
    out.usage = ti.d3d_usage;
    out.pool = ti.d3d_pool;
    out.multi_sample_type = 0;
    out.multi_sample_quality = 0;
    out.width = ti.mip_width(level as usize);
    out.height = ti.mip_height(level as usize);
    0 // S_OK
}

extern "system" fn texture_get_surface_level(
    this: *mut c_void,
    level: u32,
    surface: *mut *mut c_void,
) -> i32 {
    let _timer = tex_timer(this);
    // SAFETY: vtable thunk; `this` is *mut Direct3DTexture9 per IDirect3DTexture9 ABI.
    let Some(mut obj) = (unsafe { InPtrMut::<Direct3DTexture9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    if level >= obj.inner().app_level_count() || surface.is_null() {
        null_out(surface);
        return D3DERR_INVALIDCALL;
    }
    let index = level as usize;
    let cached = obj.inner().cached_subresource(index); // last use of `obj` on this path
    if cached != 0 {
        // SAFETY: a non-zero slot is the live cached surface for this level, and
        // the `obj` borrow ended above, so the AddRef it forwards to this texture
        // does not alias it.
        unsafe { hand_back_cached_surface(cached, surface) };
        return 0; // S_OK
    }
    let device_inner = obj.inner().device_inner as *mut DeviceInner;
    let surf = Direct3DSurface9::new_texture_backed(device_inner, this.cast(), level);
    let surf_ptr = Box::into_raw(Box::new(surf));
    // The container owns the wrapper from here: every later call for this level
    // hands the same pointer back, and `finalize_texture` frees it.
    obj.inner_mut().cache_subresource(index, surf_ptr as u64); // last use of `obj`
    // The sub-surface's public refcount is shared with (forwards to) this
    // texture, so account for the reference the returned surface holds — D3D9's
    // GetSurfaceLevel AddRefs the container texture.
    // The `obj` borrow ended above, so this AddRef does not alias it.
    // SAFETY: `this` is the live parent texture for the call.
    unsafe { crate::com_ref::com_add_ref::<Direct3DTexture9>(this) };
    // SAFETY: vtable out-param; `surface` is *mut *mut c_void per IDirect3DTexture9 ABI.
    unsafe { OutPtr::write_opt(surface, surf_ptr.cast::<c_void>()) };
    0 // S_OK
}

/// Hand a cached sub-resource surface back to a getter's caller.
///
/// D3D9 sub-resource getters return the *same* object every call, one reference
/// stronger. The surface's own `AddRef` forwards to its container texture, so the
/// count the application observes is identical to the creating call's.
///
/// # Safety
/// `cached` must be a live `*mut Direct3DSurface9` from a `TextureInner`
/// sub-resource slot, and `out` a writable out-param per the D3D9 ABI.
unsafe fn hand_back_cached_surface(cached: u64, out: *mut *mut c_void) {
    let surf = cached as *mut Direct3DSurface9;
    // SAFETY: `surf` is the live cached surface wrapper per the contract.
    let add_ref = unsafe { (*surf).vtbl().add_ref };
    // SAFETY: `add_ref` is the surface's own IUnknown::AddRef thunk; `surf` is
    // its `this`.
    unsafe { add_ref(surf.cast::<c_void>()) };
    // SAFETY: `out` is the getter's writable out-param per the contract.
    unsafe { OutPtr::write_opt(out, surf.cast::<c_void>()) };
}

/// Read a GPU-authoritative level back into its CPU staging.
///
/// The read half of a `LockRect` on a level the GPU wrote with no CPU mirror
/// (a `StretchRect` blit into a `D3DPOOL_DEFAULT` offscreen plain). Flush the
/// frame so the write has landed, then blit the level into its staging through
/// the same `BlitTextureToBuffer` core `GetRenderTargetData` uses; a D3D9 Lock
/// of a GPU-written surface stalls on a real driver too. The claim is released
/// up front, so a level whose Metal handle or staging cannot serve the read
/// costs one warning rather than one per Lock, and keeps the bytes it has.
fn materialize_level_from_gpu(ti: &mut TextureInner, level: usize) {
    ti.clear_level_gpu_authoritative(level);
    let (width, height) = (ti.mip_width(level), ti.mip_height(level));
    let bytes_per_row = ti.mip_bytes_per_row(level);
    let block_rows = height.div_ceil(ti.block_h.max(1));
    let needed = (bytes_per_row as usize).saturating_mul(block_rows as usize);
    let device_inner_ptr = ti.device_inner;
    let texture_id = ti.texture_id;
    if device_inner_ptr == 0 || width == 0 || height == 0 || needed == 0 {
        return;
    }
    ti.ensure_staging(level);
    let (dst_ptr, dst_len) = {
        let page = &ti.staging[level];
        (page.as_ptr() as u64, page.len())
    };
    if dst_len < needed {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "texture {texture_id:#x}: level {level} staging is too small for a read-back \
             ({dst_len} < {needed}) → staging left as-is");
        return;
    }
    // The staging `PageBox` and the `DeviceInner` are distinct allocations, so
    // the raw-pointer borrow below never overlaps the slice read above.
    // SAFETY: `device_inner` is the `DeviceInner*` recorded at texture
    // creation (non-zero, checked); the device outlives every texture it owns.
    let dev = unsafe { &mut *(device_inner_ptr as *mut DeviceInner) };
    // The Metal texture lives encoder-side keyed by texture id, so resolve the
    // handle inside an op and read it back through an atomic slot once the
    // flush has drained the queue.
    let slot = Arc::new(core::sync::atomic::AtomicU64::new(0));
    let slot_op = Arc::clone(&slot);
    dev.push_op(Box::new(move |enc| {
        slot_op.store(enc.get_texture_handle_by_id(texture_id), Ordering::Release);
    }));
    dev.flush_current_frame_blocking();
    let handle = slot.load(Ordering::Acquire);
    if handle == 0 {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "texture {texture_id:#x}: no Metal handle for a level {level} read-back → \
             staging left as-is");
        return;
    }
    let mut params = BlitTextureToBufferParams {
        queue_handle: dev.queue_handle(),
        device_handle: dev.device_handle(),
        // SAFETY: `handle` is non-zero (checked above) and a live retained
        // `MTLTexture` handle from the encoder texture cache.
        tex_handle: unsafe { MetalHandle::<MTLTextureKind>::new(handle) },
        dst_ptr,
        dst_len: dst_len as u64,
        mip_level: u32::try_from(level).unwrap_or(0),
        origin_x: 0,
        origin_y: 0,
        width,
        height,
        bytes_per_row,
        // The texture's own extent, so the read-back resolve is a no-op: a
        // surface claimed this way is an offscreen plain, which never inherits
        // the back buffer's `render.scale`.
        source_width: ti.mip_width(0),
        source_height: ti.mip_height(0),
    };
    let status = unix_call(&mut params);
    if status != 0 {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "texture {texture_id:#x}: level {level} read-back BlitTextureToBuffer failed \
             status={status:#x} → staging left as-is");
    }
}

extern "system" fn texture_lock_rect(
    this: *mut c_void,
    level: u32,
    out_locked_rect: *mut D3DLOCKED_RECT,
    rect: *const c_void,
    flags: u32,
) -> i32 {
    let _timer = tex_timer(this);
    // SAFETY: vtable thunk; `this` is *mut Direct3DTexture9 per IDirect3DTexture9 ABI.
    // SAFETY: vtable `this` is the live cube wrapper for this call.
    let Some(mut obj) = (unsafe { InPtrMut::<Direct3DTexture9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let ti = obj.inner_mut();
    if level >= ti.app_level_count() || out_locked_rect.is_null() {
        return D3DERR_INVALIDCALL;
    }

    // Depth-format textures (sampleable shadow maps) have no CPU staging
    // path; the GPU is the sole writer and reader. D3D9 spec disallows
    // LockRect on D3DUSAGE_DEPTHSTENCIL textures unless the depth format
    // is one of the LOCKABLE variants — mtld3d doesn't expose those, so any
    // LockRect on a depth texture is a real error to surface.
    if ti.flags.contains(TextureFlags::DEPTH_FORMAT) {
        mtld3d_shared::log_once_warn!(
            target: crate::LOG_TARGET,
            "reject IDirect3DTexture9::LockRect on depth-format texture → INVALIDCALL"
        );
        return D3DERR_INVALIDCALL;
    }

    let level_u = level as usize;
    // D3D9 rejects re-locking an already-locked level with INVALIDCALL and
    // leaves the caller's D3DLOCKED_RECT untouched.
    // Checked before lock_region_ptr (which may rename) and the out-param write.
    if ti.locked[level_u] {
        return D3DERR_INVALIDCALL;
    }
    let level_u8 = u8::try_from(level).expect("D3D9 mip level ≤ 14");
    let mip_w = ti.mip_width(level_u);
    let mip_h = ti.mip_height(level_u);
    // DEFAULT-pool surfaces strictly validate a provided lock rect; the CPU
    // pools (SYSTEMMEM/MANAGED/SCRATCH) accept any rect, falling back to the
    // whole surface for a degenerate one.
    if ti.d3d_pool == D3DPOOL_DEFAULT {
        // SAFETY: `rect` is the *const RECT from the LockRect ABI; null → None
        // (whole surface, always valid).
        // SAFETY: `rect` is the caller's optional read-only RECT pointer.
        let provided = unsafe { ValueIn::<D3DRECT>::read_opt(rect) };
        // YUY2/UYVY are 2×1-macropixel packed formats in D3D9, but we map them to
        // a 1×1 RG8 surface (block_w/h stay 1 so the pitch/upload path is correct).
        // For DEFAULT-pool lock validation they nonetheless require 2-pixel X
        // alignment, so derive a YUV-aware block size
        // here without disturbing the stored block_w/h.
        let (vbw, vbh) = match ti.d3d_format {
            D3DFMT_YUY2 | D3DFMT_UYVY => (2, 1),
            _ => (ti.block_w, ti.block_h),
        };
        if provided.is_some_and(|r| !default_lock_rect_valid(&r, mip_w, mip_h, vbw, vbh)) {
            return D3DERR_INVALIDCALL;
        }
    }
    // A level the GPU wrote with no CPU mirror has to be read back before the
    // Lock hands out a pointer into staging, and before `lock_region_ptr` may
    // rename the box (a preserve then copies the fresh bytes). `D3DLOCK_DISCARD`
    // promises a whole-level overwrite, so it skips the stall.
    if flags & D3DLOCK_DISCARD == 0 && ti.level_gpu_authoritative(level_u) {
        materialize_level_from_gpu(ti, level_u);
    }
    let dirty_rect = parse_rect(rect, mip_w, mip_h);
    let read_only = flags & D3DLOCK_READONLY != 0;
    let no_dirty = flags & D3DLOCK_NO_DIRTY_UPDATE != 0;

    let (ptr, pitch, _offset) = ti.lock_region_ptr(level_u, dirty_rect, flags);
    ti.stash_lock(level_u, read_only, no_dirty);

    // SAFETY: `out_locked_rect` is non-null (checked above) and per the
    // D3D9 ABI points to a writable `D3DLOCKED_RECT` slot owned by the
    // caller.
    let out = unsafe { &mut *out_locked_rect };
    // D3DLOCKED_RECT.pitch is i32 by D3D9 spec but always non-negative —
    // bit-preserving cast.
    out.pitch = pitch.cast_signed();
    out.bits = ptr.cast::<c_void>();

    mtld3d_shared::crumb!(
        "api:tex_lock",
        (u64::from(level_u8) << 32) | u64::from(flags),
        ptr as usize as u64,
    );
    mtld3d_shared::crumb!(
        "tex_lock:geom",
        ti.texture_id.raw(),
        (u64::from(mip_w) << 32) | u64::from(mip_h),
    );

    let unknown = flags & !D3DLOCK_KNOWN_BITS;
    if unknown != 0 {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "texture_lock_rect: unrecognised D3DLOCK bits {unknown:#x} ignored");
    }

    0 // S_OK
}

extern "system" fn texture_unlock_rect(this: *mut c_void, level: u32) -> i32 {
    let _timer = tex_timer(this);
    let level_u8 = u8::try_from(level).expect("D3D9 mip level ≤ 14");
    mtld3d_shared::crumb!("api:tex_ulock", u64::from(level_u8));
    // SAFETY: vtable thunk; `this` is *mut Direct3DTexture9 per IDirect3DTexture9 ABI.
    let Some(mut obj) = (unsafe { InPtrMut::<Direct3DTexture9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let ti = obj.inner_mut();
    if level >= ti.app_level_count() {
        return D3DERR_INVALIDCALL;
    }
    let level_u = level as usize;
    let (read_only, was_locked) = ti.take_lock(level_u);
    if !was_locked {
        // A texture-level surface's (and IDirect3DTexture9::UnlockRect's)
        // Unlock-without-Lock / double-Unlock returns S_OK in D3D9 for a
        // D3DRTYPE_TEXTURE surface. (The offscreen-plain INVALIDCALL contract
        // lives on the standalone surface path in surface::surface_unlock_rect,
        // not here.)
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "texture_unlock_rect: Unlock without matching Lock (level={level_u}) → S_OK"
        );
        return 0;
    }
    // A READONLY lock wrote nothing, so it normally schedules no upload — EXCEPT
    // a mip's very FIRST lock, which must still upload its initial sysmem
    // contents once (D3D9 "managed textures are initially dirty"; a managed mip
    // filled only via a READONLY lock must still reach VRAM). A READONLY re-lock
    // of an already-uploaded mip stays a true no-op. Gating on `was_uploaded`
    // (not the pool) avoids clobbering GPU-autogenerated sub-mips — those are
    // never locked, so they keep their generated contents.
    if read_only && ti.was_uploaded[level_u] {
        return 0;
    }
    // A non-READONLY Unlock marks the whole mip dirty as an UpdateTexture source
    // (the game wrote it via Lock); a later UpdateTexture from this texture then
    // copies it and clears the dirty flag. A
    // D3DLOCK_NO_DIRTY_UPDATE lock is excluded — UpdateTexture must ignore it. A
    // READONLY first lock wrote nothing, so it adds no UpdateTexture dirty rect;
    // it only triggers the one-time initial upload below.
    if !read_only && !ti.current_lock_no_dirty[level_u] {
        ti.mark_update_dirty(level_u, None);
    }
    // Lazy upload: flag the mip dirty and return. Bind-time
    // `flush_dirty_mips` dispatches the actual upload via
    // `schedule_upload` — Unlock is now a single byte write, the
    // Box+Arc+Vec work happens at first bind after this Unlock.
    ti.mark_mip_dirty(level_u);
    let texture_id = ti.texture_id;
    let device_inner_ptr = ti.device_inner;
    mtld3d_shared::log_once_trace_by!(
        target: TEX_TRACE_TARGET, key: (texture_id.raw() << 8) | (level_u as u64 & 0xff),
        "tex {texture_id:#x} mip {level_u} dirty (deferred upload)"
    );
    // Force snapshot re-emit on the next draw: bind-time
    // `flush_dirty_mips` only runs when the API thread re-walks stage
    // bindings, which only happens when SnapshotDirty is non-empty.
    // Without this, an Unlock between two draws with otherwise-clean
    // state would leave the upload un-scheduled and the second draw
    // would sample stale GPU content. We don't check "is this texture
    // bound" here — any over-dirty just causes one redundant snapshot
    // re-emit, which the LastBoundCache + ScratchSlice cache dedup at
    // the encoder.
    if device_inner_ptr != 0 {
        // SAFETY: `device_inner` is the `DeviceInner*` recorded at
        // texture creation; the device outlives every texture it owns
        // (textures hold a refcount on the device via their COM ABI).
        let dev = unsafe { &mut *(device_inner_ptr as *mut DeviceInner) };
        dev.mark_snapshot_dirty_all();
    }
    0 // S_OK
}

extern "system" fn texture_add_dirty_rect(this: *mut c_void, rect: *const c_void) -> i32 {
    let _timer = tex_timer(this);
    // SAFETY: vtable thunk; `this` is *mut Direct3DTexture9 per the ABI. InPtrMut
    // so we can union the rect into the source dirty region.
    let Some(mut obj) = (unsafe { InPtrMut::<Direct3DTexture9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let ti = obj.inner_mut();
    let w = ti.mip_width(0);
    let h = ti.mip_height(0);
    // A non-NULL dirty rect must lie within the level-0 surface and be
    // non-empty/non-inverted, else INVALIDCALL. A NULL rect means "the whole
    // texture". Validated against sub-resource 0, per the D3D9 AddDirtyRect
    // validation rules.
    // SAFETY: `rect` is the *const RECT delivered by the game; null → None.
    let dirty = if let Some(r) = unsafe { ValueIn::<D3DRECT>::read_opt(rect) } {
        if r.x1 < 0
            || r.y1 < 0
            || r.x2 <= r.x1
            || r.y2 <= r.y1
            || r.x2.cast_unsigned() > w
            || r.y2.cast_unsigned() > h
        {
            return D3DERR_INVALIDCALL;
        }
        Some(DirtyRect {
            x: r.x1.cast_unsigned(),
            y: r.y1.cast_unsigned(),
            w: (r.x2 - r.x1).cast_unsigned(),
            h: (r.y2 - r.y1).cast_unsigned(),
        })
    } else {
        None
    };
    // AddDirtyRect probe (perf builds): does the game declare a usable changed
    // sub-region we could use to shrink the whole-mip preserve into a dirty-rect
    // snapshot upload? `partial` = the rect is narrower than the level-0 surface;
    // `area_bp` = its area in basis points of the mip (whole-mip / NULL = 10000).
    // Surfaces in the `AddDirtyRect` row of the Resources(textures) summary.
    let di = ti.device_inner;
    if di != 0 {
        let partial = dirty.is_some_and(|r| r.x > 0 || r.y > 0 || r.w < w || r.h < h);
        let area_bp = dirty.map_or(10000, |r| {
            // A dirty sub-rect has `r.w <= w` and `r.h <= h`, so the basis-point
            // ratio is at most 10000 and always fits `u32`; fall back to "whole
            // mip" (10000) on the impossible overflow rather than truncating.
            u32::try_from(
                (u64::from(r.w) * u64::from(r.h) * 10000) / (u64::from(w) * u64::from(h)).max(1),
            )
            .unwrap_or(10000)
        });
        DeviceInner::from_ptr(di)
            .perf_mut()
            .bump_texture_add_dirty_rect(partial, area_bp);
    }
    // Mark the source dirty region so a subsequent UpdateTexture re-copies it
    // (partial-rectangle tracking). This does NOT
    // schedule a GPU upload — UpdateTexture reads the CPU staging directly, and
    // an upload from un-Lock-written staging would clobber GPU-resident content.
    ti.mark_update_dirty(0, dirty);
    0 // S_OK
}

/// Whether a non-NULL `LockRect` rect is valid on a `D3DPOOL_DEFAULT` surface.
///
/// D3D9 validates a provided lock rect strictly on `D3DPOOL_DEFAULT`
/// surfaces: it must be in-bounds, non-empty/non-inverted,
/// and — for block-compressed formats — block-aligned (offsets on a block edge,
/// extents on a block edge or the surface edge). Returns false → INVALIDCALL.
/// SYSTEMMEM/MANAGED/SCRATCH surfaces accept any rect, and a NULL rect (whole
/// surface) is always valid, so this is only consulted for a non-NULL rect on a
/// DEFAULT-pool surface.
const fn default_lock_rect_valid(
    r: &D3DRECT,
    mip_w: u32,
    mip_h: u32,
    block_w: u32,
    block_h: u32,
) -> bool {
    if r.x1 < 0 || r.y1 < 0 || r.x2 <= r.x1 || r.y2 <= r.y1 {
        return false;
    }
    let x1 = r.x1.cast_unsigned();
    let y1 = r.y1.cast_unsigned();
    let x2 = r.x2.cast_unsigned();
    let y2 = r.y2.cast_unsigned();
    if x2 > mip_w || y2 > mip_h {
        return false;
    }
    x1.is_multiple_of(block_w)
        && y1.is_multiple_of(block_h)
        && (x2.is_multiple_of(block_w) || x2 == mip_w)
        && (y2.is_multiple_of(block_h) || y2 == mip_h)
}

/// Parse a `RECT*` passed by the game and clamp it to the mip dimensions.
///
/// `NULL` means "whole mip" → returns `None` so the caller substitutes
/// `DirtyRect::full(...)`. A non-null rect that is empty, inverted, or clamps to
/// zero area is loudly logged and likewise returns `None`.
fn parse_rect(rect: *const c_void, mip_w: u32, mip_h: u32) -> Option<DirtyRect> {
    // RECT and D3DRECT share the { left/x1, top/y1, right/x2, bottom/y2 }
    // i32 layout, so reusing D3DRECT here is safe for the RECT* the D3D9
    // Lock/AddDirtyRect APIs hand us. `ValueIn::read_opt` returns None on
    // null, which matches the spec's "NULL means whole mip" semantic.
    // SAFETY: `rect` is the *const c_void RECT* delivered by the game per
    // the IDirect3DTexture9 ABI; null is filtered.
    let r = unsafe { ValueIn::<D3DRECT>::read_opt(rect) }?;
    let x = r.x1.max(0).cast_unsigned();
    let y = r.y1.max(0).cast_unsigned();
    let x2 = r.x2.max(0).cast_unsigned();
    let y2 = r.y2.max(0).cast_unsigned();
    if x2 <= x || y2 <= y {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "texture: zero-area rect ({},{})-({},{}) → treating as full-mip",
            r.x1,
            r.y1,
            r.x2,
            r.y2
        );
        return None;
    }
    DirtyRect {
        x,
        y,
        w: x2 - x,
        h: y2 - y,
    }
    .clamp(mip_w, mip_h)
    .or_else(|| {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "texture: rect ({},{})-({},{}) clamped to zero on mip {mip_w}x{mip_h}",
            r.x1,
            r.y1,
            r.x2,
            r.y2
        );
        None
    })
}

/// Build the upload closure and push it onto the current frame's op list.
///
/// The closure holds an `Arc` clone of the staging mip plus a snapshot of
/// the D3D9 format / pitch / bpp — refcount bump, zero memcpy on the API
/// thread. The encoder thread runs it in order relative to draw closures.
///
/// Also stamps the current submit seq onto the mip's `last_submit_seq`
/// so a later `LockRect` can detect GPU-in-flight contention the same
/// way the VB/IB slab does. With lazy upload, this is the **sole**
/// stamp site for `last_submit_seq` — Unlock no longer stamps because
/// no upload is dispatched there. Stamping at the dispatch moment is
/// the correct semantic.
///
/// Caller passes `dev` explicitly to avoid lifting a second `&mut
/// DeviceInner` from `ti.device_inner` when a parent caller (e.g.
/// `snapshot_stage_bindings`) already holds one.
pub fn schedule_upload(ti: &mut TextureInner, dev: &mut DeviceInner, level: u32, rect: DirtyRect) {
    let level_u = level as usize;
    if ti.dropped_staging & (1u32 << level_u) != 0 {
        // Nothing to upload from: the level's bytes live on the GPU only. A
        // re-upload request (eviction, device rehydration) for it is moot.
        mtld3d_shared::log_once_warn_by!(
            target: crate::LOG_TARGET,
            key: ti.texture_id.raw(),
            "texture {:#x}: upload of level {level} requested after its staging was released; \
             the GPU copy stands",
            ti.texture_id.raw()
        );
        return;
    }
    // Volume (3D) textures upload `(depth >> level)` slices; 2D textures are
    // `depth == 1` (the encoder then keeps the untouched single-slice path).
    // `slice_pitch` is the box slice stride — `row_pitch * ceil(mip_h /
    // block_h)`, the same block-aware formula `lock_box` uses — and is only
    // consulted by the volume blit path.
    let block_rows = ti.mip_height(level_u).div_ceil(ti.block_h.max(1));
    let slice_pitch = ti.mip_bytes_per_row(level_u).saturating_mul(block_rows);
    let job = TextureUploadJob {
        info: ti.texture_info(),
        arc: ti.staging_arc(level_u),
        level,
        destination_slice: 0,
        staging_index: level_u,
        origin_x: rect.x,
        origin_y: rect.y,
        region_w: rect.w,
        region_h: rect.h,
        src_d3d_format: ti.d3d_format,
        src_pitch: ti.mip_bytes_per_row(level_u),
        bytes_per_pixel: ti.bytes_per_pixel,
        depth: (ti.depth >> level).max(1),
        slice_pitch,
    };
    let texture_id = ti.texture_id;
    let regen_mipmaps = ti.autogen_mipmap() && level == 0;
    ti.last_submit_seq[level as usize] = dev.current_seq();
    ti.was_uploaded[level as usize] = true;
    // Every byte of a level the game cannot lock again is now on the GPU: each
    // write since the staging was allocated was uploaded by the flush that
    // followed it, this one included, and together they cover the level. The
    // job holds its own `Arc` of the staging, so the texture's copy can go now.
    if ti.staging_droppable(level_u) && ti.staging_fully_written(level_u) {
        ti.drop_staging(level_u);
    }
    dev.push_op(Box::new(move |enc: &mut FrameEncoder| {
        enc.run_texture_upload(job);
        if regen_mipmaps {
            enc.run_generate_mipmaps(texture_id);
        }
    }));
}

fn schedule_cube_upload(
    ti: &mut TextureInner,
    dev: &mut DeviceInner,
    face: u32,
    level: u32,
    rect: DirtyRect,
) {
    let level_u = level as usize;
    let index = ti
        .cube_subresource_index(face, level_u)
        .expect("validated cube subresource");
    let block_rows = ti.mip_height(level_u).div_ceil(ti.block_h.max(1));
    let slice_pitch = ti.mip_bytes_per_row(level_u).saturating_mul(block_rows);
    let cube = ti.cube.as_deref_mut().expect("cube storage");
    let arc = Arc::clone(&cube.staging[index]);
    cube.last_submit_seq[index] = dev.current_seq();
    cube.was_uploaded[index] = true;
    let job = TextureUploadJob {
        info: ti.texture_info(),
        arc,
        level,
        destination_slice: face,
        staging_index: index,
        origin_x: rect.x,
        origin_y: rect.y,
        region_w: rect.w,
        region_h: rect.h,
        src_d3d_format: ti.d3d_format,
        src_pitch: ti.mip_bytes_per_row(level_u),
        bytes_per_pixel: ti.bytes_per_pixel,
        depth: 1,
        slice_pitch,
    };
    dev.push_op(Box::new(move |enc: &mut FrameEncoder| {
        enc.run_texture_upload(job);
    }));
}

/// Mark every previously-uploaded mip dirty for the next bind-time `flush_dirty_mips`.
///
/// That replays the staging upload. Render targets are skipped (RTs have no
/// Lock+Unlock staging — their content is GPU-rendered, so dropping their cache
/// entry would lose pixels irrecoverably). Returns `Some(texture_id)` when any
/// mip was marked so `evict_managed_resources` can enqueue a cache eviction;
/// `None` for unwritten textures and RTs.
pub fn evict_mark_dirty(ti: &mut TextureInner) -> Option<TextureId> {
    if ti.usage_flags.contains(TextureUsage::RENDER_TARGET) {
        return None;
    }
    if ti.flags.contains(TextureFlags::CUBE) {
        let mut had_uploads = false;
        for face in 0..CUBE_FACE_COUNT {
            for level in 0..ti.levels as usize {
                let index = ti
                    .cube_subresource_index(face, level)
                    .expect("validated cube subresource");
                if ti.cube.as_deref().expect("cube storage").was_uploaded[index] {
                    ti.mark_cube_dirty(face, level);
                    had_uploads = true;
                }
            }
        }
        return had_uploads.then_some(ti.texture_id);
    }
    let mut had_uploads = false;
    for level in 0..ti.levels as usize {
        if ti.was_uploaded[level] {
            ti.mark_mip_dirty(level);
            had_uploads = true;
        }
    }
    had_uploads.then_some(ti.texture_id)
}

/// Detect cross-device migration of a managed texture and prepare it for re-upload.
///
/// `D3DPOOL_MANAGED` textures survive a `Release` + `CreateDevice` (the game
/// keeps holding `IDirect3DTexture9`), so a `TextureInner` created on the old
/// device can be bound on a new one. The new device's `FrameEncoder` has an
/// empty texture cache and a fresh seq counter; the old MTL handles are gone
/// with the old device.
///
/// Without rehydration: the bind-time `flush_dirty_mips` finds nothing
/// dirty (the old device's upload completed cleanly), `get_or_create_texture`
/// cache-misses → creates a fresh empty `MTLTexture`, and the draw samples
/// zeros.
///
/// With rehydration: every previously-uploaded mip flips back to dirty,
/// `last_submit_seq` resets (it was scoped to the old device's encoder),
/// `device_handle` repoints to the new `MTLDevice`, and `flush_dirty_mips`
/// dispatches re-uploads against fresh `MTLTextures` on the right device.
///
/// Idempotent: returns immediately when `ti.device_inner` already matches
/// `dev`. Called from every bind site (draw + `StretchRect` + similar).
#[inline]
pub fn rehydrate_for_device(ti: &mut TextureInner, dev: &mut DeviceInner) {
    let dev_ptr = std::ptr::from_mut::<DeviceInner>(dev) as u64;
    if ti.device_inner == dev_ptr {
        return;
    }
    rehydrate_for_device_slow(ti, dev, dev_ptr);
}

/// Migration tail of [`rehydrate_for_device`], reached only on a device change.
///
/// Outlined `#[cold]` so the per-draw bind path inlines just the
/// same-device pointer compare above.
#[cold]
#[inline(never)]
fn rehydrate_for_device_slow(ti: &mut TextureInner, dev: &mut DeviceInner, dev_ptr: u64) {
    let texture_id = ti.texture_id;
    let mut levels_remarked: u32 = 0;
    if ti.flags.contains(TextureFlags::CUBE) {
        for face in 0..CUBE_FACE_COUNT {
            for level in 0..ti.levels as usize {
                let index = ti
                    .cube_subresource_index(face, level)
                    .expect("validated cube subresource");
                let cube = ti.cube.as_deref_mut().expect("cube storage");
                if cube.was_uploaded[index] {
                    cube.dirty_masks[face as usize] |= 1 << level;
                    ti.dirty_mask |= 1 << level;
                    levels_remarked += 1;
                }
                cube.last_submit_seq[index] = 0;
            }
        }
    } else {
        for level in 0..ti.levels as usize {
            if ti.was_uploaded[level] {
                ti.mark_mip_dirty(level);
                levels_remarked += 1;
            }
            // Old seq is from the old device's encoder counter, so it is meaningless.
            // on the new device. Zero it so `decide_lock_action` doesn't
            // misread "old huge seq vs new tiny coherent_seq" as GPU contention.
            ti.last_submit_seq[level] = 0;
        }
    }
    ti.device_inner = dev_ptr;
    ti.device_handle = dev.device_handle();
    ti.point_cached_surfaces_at(std::ptr::from_mut::<DeviceInner>(dev));
    dev.register_texture(std::ptr::from_mut::<TextureInner>(ti));
    // Seed the new device's encoder texture_cache with this texture's
    // info so the per-draw stage binding (which carries only
    // `texture_id`) resolves to a real Metal handle without needing
    // TextureInfo on the per-draw bump. The warmup is drained at
    // `run_frame` before any op processes — including the bind that
    // triggered this rehydrate call. A system-memory texture has no Metal
    // texture on any device, so it seeds nothing.
    if !ti.is_cpu_only() {
        dev.push_texture_warmup(ti.texture_info());
    }
    log::info!(
        target: TEX_TRACE_TARGET,
        "tex {texture_id:#x} rehydrated for new device (re-marked {levels_remarked} mips dirty)"
    );
}

/// End a system-memory texture's CPU-only phase; report whether it was in one.
///
/// D3D9 samples a `D3DPOOL_SYSTEMMEM` texture bound at a texture stage, so a
/// sampling bind is where the pool stops meaning "no GPU allocation". Clearing
/// the flag is all this does: the caller queues the `MTLTexture` create, and
/// every level the application has written is already marked dirty, so the
/// flush that follows the bind uploads them.
pub fn promote_to_gpu(ti: &mut TextureInner) -> bool {
    if !ti.is_cpu_only() {
        return false;
    }
    ti.flags.remove(TextureFlags::CPU_ONLY);
    true
}

/// Walk a texture's `dirty_mask` and dispatch a full-mip upload for every dirty level.
///
/// Called at bind time from `device.rs::snapshot_stage_bindings` (every Draw)
/// and `device.rs::device_stretch_rect` (`StretchRect` texture-source).
/// D3D9 is single-threaded per device, so the `&mut TextureInner` and
/// `&mut DeviceInner` here are sound — we hold them only for the
/// duration of this call. The `dev` parameter avoids lifting a second
/// `&mut DeviceInner` from `ti.device_inner` (which would alias the
/// caller's already-held `dev` borrow).
#[inline]
pub fn flush_dirty_mips(ti: &mut TextureInner, dev: &mut DeviceInner) {
    if ti.dirty_mask == 0 {
        return;
    }
    flush_dirty_mips_slow(ti, dev);
}

/// Upload tail of [`flush_dirty_mips`], reached only when some mip is dirty.
///
/// Outlined `#[cold]` so the per-draw bind path inlines just the
/// mask-is-zero gate above.
#[cold]
#[inline(never)]
fn flush_dirty_mips_slow(ti: &mut TextureInner, dev: &mut DeviceInner) {
    if ti.is_cpu_only() {
        // No `MTLTexture` to upload into yet. The bits stay set, so the
        // promotion a sampling bind performs uploads every level the
        // application has written by then.
        return;
    }
    if ti.flags.contains(TextureFlags::CUBE) {
        let mut dirty_count = 0u32;
        let mut regenerate_mipmaps = false;
        ti.dirty_mask = 0;
        for face in 0..CUBE_FACE_COUNT {
            let mut mask = {
                let cube = ti.cube.as_deref_mut().expect("cube storage");
                core::mem::take(&mut cube.dirty_masks[face as usize])
            };
            while mask != 0 {
                let level = mask.trailing_zeros();
                mask &= mask - 1;
                let level_u = level as usize;
                let rect = DirtyRect::full(ti.mip_widths[level_u], ti.mip_heights[level_u]);
                schedule_cube_upload(ti, dev, face, level, rect);
                regenerate_mipmaps |= ti.autogen_mipmap() && level == 0;
                dirty_count += 1;
            }
        }
        let texture_id = ti.texture_id;
        if regenerate_mipmaps {
            dev.push_op(Box::new(move |enc: &mut FrameEncoder| {
                enc.run_generate_mipmaps(texture_id);
            }));
        }
        mtld3d_shared::log_once_trace_by!(
            target: TEX_TRACE_TARGET, key: texture_id.raw(),
            "cube {texture_id:#x} flush dirty subresources={dirty_count}"
        );
        return;
    }
    let mut dirty_count: u32 = 0;
    let mut mask = ti.dirty_mask;
    ti.dirty_mask = 0;
    while mask != 0 {
        let level = mask.trailing_zeros();
        mask &= mask - 1;
        let level_u = level as usize;
        let rect = ti
            .pending_upload_rects
            .get_mut(level_u)
            .and_then(Option::take)
            .unwrap_or_else(|| DirtyRect::full(ti.mip_widths[level_u], ti.mip_heights[level_u]));
        schedule_upload(ti, dev, level, rect);
        dirty_count += 1;
    }
    let texture_id = ti.texture_id;
    mtld3d_shared::log_once_trace_by!(
        target: TEX_TRACE_TARGET, key: texture_id.raw(),
        "tex {texture_id:#x} flush dirty levels={dirty_count}"
    );
}

// ── IDirect3DVolumeTexture9 (volume / 3D textures) ──
//
// `Direct3DVolumeTexture9` has the SAME `#[repr(C)]` layout as
// `Direct3DTexture9` (vtbl ptr + refcount + private_refcount + inner ptr) and
// the same backing `TextureInner` (with `depth > 1`). Only the vtable differs,
// so the IUnknown / IDirect3DResource9 / IDirect3DBaseTexture9 thunks are
// reused verbatim, and `SetTexture`'s cast-to-`Direct3DTexture9` reads the
// shared `inner`/`texture_id` correctly. The 3D-specific tail is implemented
// here; `LockBox`/`UnlockBox` are real (a paired non-readonly `UnlockBox`
// schedules the box→3D upload — see `TextureInner::lock_box`), the rest are
// minimal.

static DIRECT3D_VOLUME_TEXTURE9_VTBL: IDirect3DVolumeTexture9Vtbl = IDirect3DVolumeTexture9Vtbl {
    query_interface: texture_query_interface,
    add_ref: texture_add_ref,
    release: texture_release,
    get_device: texture_get_device,
    set_private_data: texture_set_private_data,
    get_private_data: texture_get_private_data,
    free_private_data: texture_free_private_data,
    set_priority: texture_set_priority,
    get_priority: texture_get_priority,
    pre_load: texture_pre_load,
    get_type: volume_get_type,
    set_lod: texture_set_lod,
    get_lod: texture_get_lod,
    get_level_count: texture_get_level_count,
    set_auto_gen_filter_type: texture_set_auto_gen_filter_type,
    get_auto_gen_filter_type: texture_get_auto_gen_filter_type,
    generate_mip_sub_levels: texture_generate_mip_sub_levels,
    get_level_desc: volume_get_level_desc,
    get_volume_level: volume_get_volume_level,
    lock_box: volume_lock_box,
    unlock_box: volume_unlock_box,
    add_dirty_box: volume_add_dirty_box,
};

/// `IDirect3DVolumeTexture9` COM wrapper.
///
/// Layout-identical to `Direct3DTexture9` (see the module note above).
#[repr(C)]
pub struct Direct3DVolumeTexture9 {
    vtbl: *const IDirect3DVolumeTexture9Vtbl,
    refcount: u32,
    private_refcount: u32,
    inner: *mut TextureInner,
}

impl Direct3DVolumeTexture9 {
    pub fn new(info: TextureCreateInfo) -> Self {
        Self {
            vtbl: &raw const DIRECT3D_VOLUME_TEXTURE9_VTBL,
            refcount: 1,
            private_refcount: 0,
            inner: build_texture_inner(info),
        }
    }

    pub const fn inner(&self) -> &TextureInner {
        // SAFETY: `self.inner` is the `build_texture_inner` allocation, live
        // until `finalize_texture` at refcount zero.
        unsafe { &*self.inner }
    }
}

const extern "system" fn volume_get_type(_this: *mut c_void) -> u32 {
    D3DRTYPE_VOLUMETEXTURE
}

extern "system" fn volume_get_level_desc(this: *mut c_void, level: u32, desc: *mut c_void) -> i32 {
    if desc.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; volume-texture layout matches `Direct3DTexture9`,
    // so the cast reads the shared `inner` correctly.
    let Some(obj) = (unsafe { InPtr::<Direct3DTexture9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let inner = obj.inner();
    let lvl = level as usize;
    if lvl >= inner.levels as usize {
        return D3DERR_INVALIDCALL;
    }
    let volume_desc = D3DVOLUME_DESC {
        format: inner.d3d_format,
        resource_type: D3DRTYPE_VOLUME,
        usage: inner.d3d_usage,
        pool: inner.d3d_pool,
        width: inner.mip_width(lvl),
        height: inner.mip_height(lvl),
        depth: (inner.depth >> level).max(1),
    };
    // SAFETY: `desc` is a writable `D3DVOLUME_DESC` out-param per the ABI.
    unsafe { desc.cast::<D3DVOLUME_DESC>().write(volume_desc) };
    D3D_OK
}

extern "system" fn volume_get_volume_level(
    this: *mut c_void,
    level: u32,
    volume: *mut *mut c_void,
) -> i32 {
    if volume.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DTexture9 per the shared ABI.
    let Some(mut obj) = (unsafe { InPtrMut::<Direct3DTexture9>::opt(this) }) else {
        null_out(volume);
        return D3DERR_INVALIDCALL;
    };
    if level >= obj.inner().levels {
        null_out(volume);
        return D3DERR_INVALIDCALL;
    }
    let index = level as usize;
    let cached = obj.inner().cached_subresource(index); // last use of `obj` on this path
    if cached != 0 {
        let vol = cached as *mut c_void;
        // The same shell every call, one reference stronger: its `AddRef`
        // forwards to this texture, so the count the app observes is what the
        // creating call reported. The `obj` borrow ended above, so that forward
        // does not alias it.
        volume9_add_ref(vol);
        // SAFETY: `volume` is non-null per the check at entry; the app owns the slot.
        unsafe { *volume = vol };
        return D3D_OK;
    }
    let vol = Direct3DVolume9::new(this, level);
    // The container owns the shell from here: every later call for this level
    // hands the same pointer back, and `finalize_texture` frees it.
    obj.inner_mut().cache_subresource(index, vol as u64); // last use of `obj`
    texture_add_ref(this);
    // SAFETY: `volume` is non-null per the check above; the app owns the slot.
    unsafe { *volume = vol.cast::<c_void>() };
    D3D_OK
}

extern "system" fn volume_lock_box(
    this: *mut c_void,
    level: u32,
    locked_box: *mut D3DLOCKED_BOX,
    box_ptr: *const c_void,
    _flags: u32,
) -> i32 {
    if locked_box.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // A rejected lock hands back a null `pBits` (the caller may have seeded
    // the struct with garbage); the success path overwrites this.
    // SAFETY: `locked_box` is a writable `D3DLOCKED_BOX` out-param per the ABI.
    unsafe {
        locked_box.write(D3DLOCKED_BOX {
            row_pitch: 0,
            slice_pitch: 0,
            bits: core::ptr::null_mut(),
        });
    }
    // SAFETY: vtable thunk; volume layout matches `Direct3DTexture9`, so the
    // cast reads the shared `inner` correctly. InPtrMut so we can record the
    // per-level lock state so a double LockBox is rejected.
    let Some(mut obj) = (unsafe { InPtrMut::<Direct3DTexture9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let inner = obj.inner_mut();
    let lvl = level as usize;
    if lvl >= inner.levels as usize {
        return D3DERR_INVALIDCALL;
    }
    // Re-locking an already-mapped level is INVALIDCALL.
    if inner.locked[lvl] {
        return D3DERR_INVALIDCALL;
    }
    // A DEFAULT-pool volume is CPU-accessible only when it is DYNAMIC; the
    // same rule every 2D DEFAULT-pool texture lock applies.
    if inner.d3d_pool == D3DPOOL_DEFAULT && inner.d3d_usage & D3DUSAGE_DYNAMIC == 0 {
        return D3DERR_INVALIDCALL;
    }
    let Some((ptr, row_pitch, slice_pitch)) = inner.lock_box(lvl) else {
        return D3DERR_INVALIDCALL;
    };
    // An optional box must be a non-empty, in-bounds half-open region; unlike
    // 2D surfaces, volumes validate it strictly.
    // The returned pointer is then offset to the box origin.
    // SAFETY: `box_ptr` is the *const D3DBOX from the LockBox ABI; null → None.
    let offset = if let Some(b) = unsafe { ValueIn::<D3DBOX>::read_opt(box_ptr) } {
        let mip_w = inner.mip_width(lvl);
        let mip_h = inner.mip_height(lvl);
        let mip_d = (inner.depth >> lvl).max(1);
        if b.right <= b.left
            || b.bottom <= b.top
            || b.back <= b.front
            || b.right > mip_w
            || b.bottom > mip_h
            || b.back > mip_d
        {
            return D3DERR_INVALIDCALL;
        }
        // Block-compressed volumes (DXT/BC) require the box to land on the block
        // grid: offsets on a block edge, extents on a block edge or the mip edge
        // — mirroring `default_lock_rect_valid` / `update_region_valid`. Block
        // depth is always 1 for D3D9 BC/YUV, so no front/back block check.
        // Uncompressed formats have block_w == block_h == 1, so this is inert.
        let bw = inner.block_w.max(1);
        let bh = inner.block_h.max(1);
        if (bw > 1 || bh > 1)
            && (!b.left.is_multiple_of(bw)
                || !b.top.is_multiple_of(bh)
                || (!b.right.is_multiple_of(bw) && b.right != mip_w)
                || (!b.bottom.is_multiple_of(bh) && b.bottom != mip_h))
        {
            return D3DERR_INVALIDCALL;
        }
        // YUY2/UYVY are 2×1-macropixel packed formats mapped to a 1×1 RG8 texture
        // (block_w/h stay 1 for the upload path); LockBox still requires 2-pixel X
        // alignment of the box.
        if matches!(inner.d3d_format, D3DFMT_YUY2 | D3DFMT_UYVY)
            && (!b.left.is_multiple_of(2) || (!b.right.is_multiple_of(2) && b.right != mip_w))
        {
            return D3DERR_INVALIDCALL;
        }
        let row = usize::try_from(row_pitch).unwrap_or(0);
        let slice = usize::try_from(slice_pitch).unwrap_or(0);
        // Block-space offset: `row_pitch` is bytes-per-block-row for compressed
        // formats, so convert pixel coords to block coords before the multiply
        // (block_bytes == bytes_per_pixel for uncompressed, so unchanged there).
        (b.front as usize).saturating_mul(slice)
            + ((b.top / bh) as usize).saturating_mul(row)
            + ((b.left / bw) as usize).saturating_mul(inner.block_bytes as usize)
    } else {
        0
    };
    // Record the lock only after all validation passed, so a rejected LockBox
    // leaves the per-level state untouched.
    inner.stash_lock(lvl, false, false);
    // SAFETY: `offset` lands inside the level's allocation — the box is
    // validated above against the level dimensions and `lock_box` sized the
    // backing as `slice_pitch * depth`.
    let ptr = unsafe { ptr.add(offset) };
    // SAFETY: `locked_box` is a writable `D3DLOCKED_BOX` out-param per the ABI.
    unsafe {
        locked_box.write(D3DLOCKED_BOX {
            row_pitch,
            slice_pitch,
            bits: ptr.cast::<c_void>(),
        });
    }
    D3D_OK
}

extern "system" fn volume_unlock_box(this: *mut c_void, level: u32) -> i32 {
    // SAFETY: vtable thunk; volume layout matches `Direct3DTexture9`.
    let Some(mut obj) = (unsafe { InPtrMut::<Direct3DTexture9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let inner = obj.inner_mut();
    let lvl = level as usize;
    if lvl >= inner.levels as usize {
        return D3DERR_INVALIDCALL;
    }
    let (read_only, was_locked) = inner.take_lock(lvl);
    if !was_locked {
        // UnlockBox without a matching LockBox (or a double-Unlock) is INVALIDCALL.
        return D3DERR_INVALIDCALL;
    }
    if read_only {
        // A read-only lock wrote nothing, so there is nothing to upload.
        return D3D_OK;
    }
    // The written level is now an `UpdateTexture` source: a SYSTEMMEM volume
    // filled through LockBox and pushed into a DEFAULT-pool twin is the
    // standard way an engine uploads a colour-grading LUT, and UpdateTexture
    // copies only levels marked here. Volumes track dirtiness per whole
    // level (no sub-box), so the mark is the full mip.
    inner.mark_update_dirty(lvl, None);
    // Lazy box→3D upload, mirroring the 2D `texture_unlock_rect` path: mark the
    // level dirty so the next bind-time `flush_dirty_mips` dispatches
    // `schedule_upload` (the volume variant), which routes the whole staging
    // box through the encoder as a `depth`-slice `CopyBufferToTexture`. The
    // staging retains every byte the game wrote, so a full-box re-upload on
    // each Unlock subsumes any sub-box lock.
    inner.mark_mip_dirty(lvl);
    let device_inner_ptr = inner.device_inner();
    // `inner` is not used past this point, so lifting a `&mut DeviceInner` from
    // the recorded pointer does not alias it (distinct allocations anyway).
    if device_inner_ptr != 0 {
        // SAFETY: `device_inner` is the `DeviceInner*` recorded at texture
        // creation; the device outlives every texture it owns (textures hold a
        // device refcount via their COM ABI). Forces the next Draw to re-walk
        // stage bindings so `flush_dirty_mips` runs.
        let dev = unsafe { &mut *(device_inner_ptr as *mut DeviceInner) };
        dev.mark_snapshot_dirty_all();
    }
    D3D_OK
}

extern "system" fn volume_add_dirty_box(this: *mut c_void, _box: *const c_void) -> i32 {
    // SAFETY: vtable thunk; volume layout matches `Direct3DTexture9`.
    let Some(mut obj) = (unsafe { InPtrMut::<Direct3DTexture9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    // Same contract as `texture_add_dirty_rect`: a hint that the CPU bytes
    // changed, consumed by the next UpdateTexture from this volume. It never
    // schedules a GPU upload (the staging may not carry Lock-written bytes).
    // Volumes track the whole level, so the box itself is not recorded.
    obj.inner_mut().mark_update_dirty(0, None);
    D3D_OK
}

// ── IDirect3DVolume9 (one level of a volume texture) ──
//
// `GetVolumeLevel` hands back a shell that owns no pixels of its own: it names
// a level of the container and forwards `GetDesc`/`LockBox`/`UnlockBox` there,
// so a write through the level lands in the same staging the texture uploads
// from. It is a leaf object (never bound, no Metal backing, no device reference
// forwarded of its own), and it is container-cached like a `GetSurfaceLevel`
// surface: the same pointer every call, alive past its own last `Release`, and
// freed by `finalize_texture`.

static DIRECT3D_VOLUME9_VTBL: IDirect3DVolume9Vtbl = IDirect3DVolume9Vtbl {
    query_interface: volume9_query_interface,
    add_ref: volume9_add_ref,
    release: volume9_release,
    get_device: volume9_get_device,
    set_private_data: volume9_set_private_data,
    get_private_data: volume9_get_private_data,
    free_private_data: volume9_free_private_data,
    get_container: volume9_get_container,
    get_desc: volume9_get_desc,
    lock_box: volume9_lock_box,
    unlock_box: volume9_unlock_box,
};

/// `IDirect3DVolume9` COM wrapper for a single volume-texture level.
///
/// The shell owns no pixels: `LockBox`/`UnlockBox` go to the parent texture's
/// per-level path, so a write through the level lands in the same staging the
/// texture uploads from, and `GetDesc` reads the parent at call time.
///
/// D3D9 specifies that a volume level's refcount is **identical** to its parent
/// volume texture's, so `AddRef`/`Release` forward there and report its count;
/// `GetVolumeLevel` takes one parent reference on the app's behalf. Forwarding
/// (rather than an independent count) is load-bearing: an independent count
/// would let the app's `Release(volumeTexture)` free the texture while a level
/// reference is still held.
///
/// The container caches the shell for the lifetime of the texture, so
/// `GetVolumeLevel(n)` answers with one identity and the private data stored
/// through it round-trips; `finalize_texture` is the single free site. Its own
/// count then tracks only the references the application still holds, so a
/// stray extra `Release` is answered rather than underflowing the shared count.
#[repr(C)]
struct Direct3DVolume9 {
    vtbl: *const IDirect3DVolume9Vtbl,
    /// References handed out on this shell; zero means only the container holds it.
    refcount: u32,
    /// Parent `Direct3DVolumeTexture9` wrapper; `AddRef`/`Release` forward here.
    parent_texture: *mut c_void,
    /// Mip level of the parent this shell addresses.
    level: u32,
    /// GUID-keyed application private data (`Set/Get/FreePrivateData`).
    ///
    /// A volume is not an `IDirect3DResource9`, so this is the level's own store
    /// rather than the container's: `SetPrivateData` on a level and on its
    /// texture address different tables. Any stored `IUnknown` is released when
    /// the shell drops.
    private_data: PrivateDataStore,
}

impl Direct3DVolume9 {
    /// `parent_texture` is the owning `Direct3DVolumeTexture9*`.
    ///
    /// The caller must have already taken the parent reference this volume
    /// forwards; the shell starts with one reference of its own.
    fn new(parent_texture: *mut c_void, level: u32) -> *mut Self {
        Box::into_raw(Box::new(Self {
            vtbl: &raw const DIRECT3D_VOLUME9_VTBL,
            refcount: 1,
            parent_texture,
            level,
            private_data: PrivateDataStore::default(),
        }))
    }

    fn parent(&self) -> &Direct3DTexture9 {
        // SAFETY: every reference on the shell is forwarded to the parent, and
        // past the shell's own last one the parent owns the shell and frees it
        // in its finalize, so the wrapper behind `parent_texture` is live either
        // way.
        unsafe { &*self.parent_texture.cast::<Direct3DTexture9>() }
    }
}

extern "system" fn volume9_query_interface(
    this: *mut c_void,
    riid: *const Guid,
    ppv: *mut *mut c_void,
) -> i32 {
    // A volume is an `IUnknown` and an `IDirect3DVolume9`, nothing else: it is
    // not a resource (the parent texture is).
    // SAFETY: `riid` is the caller's read-only GUID pointer.
    let accepted = (unsafe { InPtr::<Guid>::opt(riid.cast()) })
        .is_some_and(|iid| matches!(*iid, IID_IUNKNOWN | IID_IDIRECT3DVOLUME9));
    if accepted && !ppv.is_null() {
        // SAFETY: validated writable out pointer.
        unsafe { *ppv = this };
        volume9_add_ref(this);
        return D3D_OK;
    }
    null_out(ppv);
    E_NOINTERFACE
}

extern "system" fn volume9_add_ref(this: *mut c_void) -> u32 {
    // SAFETY: IDirect3DVolume9 AddRef thunk; `this` is the live wrapper.
    let Some(mut obj) = (unsafe { InPtrMut::<Direct3DVolume9>::opt(this) }) else {
        return 0;
    };
    obj.refcount += 1;
    // Forward to the parent texture; its (shared) count is what D3D9 reports.
    texture_add_ref(obj.parent_texture)
}

extern "system" fn volume9_release(this: *mut c_void) -> u32 {
    // SAFETY: IDirect3DVolume9 Release thunk; `this` is the live wrapper.
    let Some(mut obj) = (unsafe { InPtrMut::<Direct3DVolume9>::opt(this) }) else {
        return 0;
    };
    // D3D9 tolerates Release past zero, which a container-cached shell makes
    // reachable: it outlives its own last release. Answer 0 without dropping a
    // container reference this shell never took.
    if obj.refcount == 0 {
        return 0;
    }
    obj.refcount -= 1;
    let parent = obj.parent_texture;
    // Reaching zero does NOT free the shell: the container caches it so
    // `GetVolumeLevel(n)` keeps one identity and the private data stored through
    // it survives, and `finalize_texture` frees it. Forward last, since that can
    // take the texture to zero and free the shell along with it, and report the
    // parent's (shared) count, which is what D3D9 answers for a sub-resource.
    texture_release(parent)
}

/// Free a container-cached `IDirect3DVolume9` shell at its container's teardown.
///
/// The volume counterpart of `crate::surface::finalize_cached_surface`: a level
/// shell is never freed by `Release`, so `finalize_texture` is its single free
/// site. Dropping the shell releases any `D3DSPD_IUNKNOWN` private-data object
/// it holds.
///
/// # Safety
/// `ptr` must be `0` or a live `*mut Direct3DVolume9` held in a `TextureInner`
/// sub-resource slot; after this returns the pointer is dangling and must not be
/// used again.
unsafe fn finalize_cached_volume(ptr: u64) {
    if ptr == 0 {
        return;
    }
    // SAFETY: `ptr` is the `Box::into_raw` allocation of `Direct3DVolume9::new`
    // and the caller guarantees no reference to the shell remains.
    drop(unsafe { Box::from_raw(ptr as *mut Direct3DVolume9) });
}

extern "system" fn volume9_get_device(this: *mut c_void, device: *mut *mut c_void) -> i32 {
    // The shell holds no device of its own, but it does hold the volume
    // texture that owns it, and that is the same device: a sub-resource and
    // its container cannot come from different ones.
    if device.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DVolume9 per IDirect3DVolume9 ABI.
    let parent = (unsafe { InPtr::<Direct3DVolume9>::opt(this) })
        .map_or(core::ptr::null_mut(), |v| v.parent_texture);
    if parent.is_null() {
        null_out(device);
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: `parent` is the live owning `Direct3DVolumeTexture9` this shell
    // forwards its refcount to.
    let dev_inner = (unsafe { InPtr::<Direct3DVolumeTexture9>::opt(parent) })
        .map_or(0, |t| t.inner().device_inner);
    let wrapper = if dev_inner == 0 {
        core::ptr::null_mut()
    } else {
        DeviceInner::from_ptr(dev_inner).device_wrapper()
    };
    if wrapper.is_null() {
        null_out(device);
        return D3DERR_INVALIDCALL;
    }
    crate::device::device_wrapper_add_ref(wrapper);
    // SAFETY: non-null (checked at entry) and writable per the ABI.
    unsafe { *device = wrapper };
    D3D_OK
}

extern "system" fn volume9_set_private_data(
    this: *mut c_void,
    guid: *const Guid,
    data: *const c_void,
    size: u32,
    flags: u32,
) -> i32 {
    // SAFETY: vtable in-param; `guid` is *const Guid per the D3D9 ABI.
    let Some(guid) = (unsafe { InPtr::<Guid>::opt(guid.cast()) }) else {
        return D3DERR_INVALIDCALL;
    };
    // SAFETY: IDirect3DVolume9 SetPrivateData thunk; `this` is the live wrapper.
    let Some(mut obj) = (unsafe { InPtrMut::<Direct3DVolume9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    // SAFETY: `data`/`size`/`flags` are the caller-supplied payload; `set` validates.
    unsafe { obj.private_data.set(&guid, data, size, flags) }
}

extern "system" fn volume9_get_private_data(
    this: *mut c_void,
    guid: *const Guid,
    data: *mut c_void,
    size: *mut u32,
) -> i32 {
    // SAFETY: vtable in-param; `guid` is *const Guid per the D3D9 ABI.
    let Some(guid) = (unsafe { InPtr::<Guid>::opt(guid.cast()) }) else {
        return D3DERR_INVALIDCALL;
    };
    // SAFETY: IDirect3DVolume9 GetPrivateData thunk; `this` is the live wrapper.
    let Some(obj) = (unsafe { InPtr::<Direct3DVolume9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    // SAFETY: `data`/`size` are caller out-params per the D3D9 ABI; `get` validates.
    unsafe { obj.private_data.get(&guid, data, size) }
}

extern "system" fn volume9_free_private_data(this: *mut c_void, guid: *const Guid) -> i32 {
    // SAFETY: vtable in-param; `guid` is *const Guid per the D3D9 ABI.
    let Some(guid) = (unsafe { InPtr::<Guid>::opt(guid.cast()) }) else {
        return D3DERR_INVALIDCALL;
    };
    // SAFETY: IDirect3DVolume9 FreePrivateData thunk; `this` is the live wrapper.
    let Some(mut obj) = (unsafe { InPtrMut::<Direct3DVolume9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    obj.private_data.free(&guid)
}

extern "system" fn volume9_get_container(
    this: *mut c_void,
    riid: *const Guid,
    container: *mut *mut c_void,
) -> i32 {
    if container.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is the live wrapper.
    let Some(obj) = (unsafe { InPtr::<Direct3DVolume9>::opt(this) }) else {
        null_out(container);
        return D3DERR_INVALIDCALL;
    };
    // `GetContainer` is `QueryInterface` against the parent volume texture,
    // so it answers the texture's own interface IIDs with an owned reference
    // and `E_NOINTERFACE` for anything else (a volume is not its own
    // container).
    // SAFETY: `riid` is a *const Guid per the QueryInterface ABI.
    let matches_texture = (unsafe { InPtr::<Guid>::opt(riid.cast()) }).is_some_and(|g| {
        let g = *g;
        g == IID_IUNKNOWN
            || g == IID_IDIRECT3DRESOURCE9
            || g == IID_IDIRECT3DBASETEXTURE9
            || g == IID_IDIRECT3DVOLUMETEXTURE9
    });
    if !matches_texture {
        null_out(container);
        return E_NOINTERFACE;
    }
    // `parent_texture` is the live owning `Direct3DVolumeTexture9`, kept alive
    // by the reference this volume forwards; the returned reference is the
    // caller's to release.
    texture_add_ref(obj.parent_texture);
    // SAFETY: `container` is non-null (checked) and a writable out-pointer.
    unsafe { *container = obj.parent_texture };
    D3D_OK
}

extern "system" fn volume9_get_desc(this: *mut c_void, desc: *mut D3DVOLUME_DESC) -> i32 {
    if desc.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: IDirect3DVolume9 GetDesc thunk; `this` is the live wrapper.
    let Some(obj) = (unsafe { InPtr::<Direct3DVolume9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let inner = obj.parent().inner();
    let lvl = obj.level as usize;
    // SAFETY: `desc` is non-null per the check above; D3DVOLUME_DESC is plain data.
    unsafe {
        desc.write(D3DVOLUME_DESC {
            format: inner.d3d_format,
            resource_type: D3DRTYPE_VOLUME,
            usage: inner.d3d_usage,
            pool: inner.d3d_pool,
            width: inner.mip_widths[lvl],
            height: inner.mip_heights[lvl],
            depth: (inner.depth >> obj.level).max(1),
        });
    }
    D3D_OK
}

extern "system" fn volume9_lock_box(
    this: *mut c_void,
    locked_box: *mut D3DLOCKED_BOX,
    box_ptr: *const c_void,
    flags: u32,
) -> i32 {
    // SAFETY: IDirect3DVolume9 LockBox thunk; `this` is the live wrapper.
    let Some(obj) = (unsafe { InPtr::<Direct3DVolume9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    // The parent's per-level lock: same staging, same contention rules, same
    // dirty marking on unlock, so a write through the level uploads exactly
    // like one through `IDirect3DVolumeTexture9::LockBox`.
    volume_lock_box(obj.parent_texture, obj.level, locked_box, box_ptr, flags)
}

extern "system" fn volume9_unlock_box(this: *mut c_void) -> i32 {
    // SAFETY: IDirect3DVolume9 UnlockBox thunk; `this` is the live wrapper.
    let Some(obj) = (unsafe { InPtr::<Direct3DVolume9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    volume_unlock_box(obj.parent_texture, obj.level)
}

static DIRECT3D_CUBE_TEXTURE9_VTBL: IDirect3DCubeTexture9Vtbl = IDirect3DCubeTexture9Vtbl {
    query_interface: texture_query_interface,
    add_ref: texture_add_ref,
    release: texture_release,
    get_device: texture_get_device,
    set_private_data: texture_set_private_data,
    get_private_data: texture_get_private_data,
    free_private_data: texture_free_private_data,
    set_priority: texture_set_priority,
    get_priority: texture_get_priority,
    pre_load: texture_pre_load,
    get_type: cube_get_type,
    set_lod: texture_set_lod,
    get_lod: texture_get_lod,
    get_level_count: texture_get_level_count,
    set_auto_gen_filter_type: texture_set_auto_gen_filter_type,
    get_auto_gen_filter_type: texture_get_auto_gen_filter_type,
    generate_mip_sub_levels: texture_generate_mip_sub_levels,
    get_level_desc: cube_get_level_desc,
    get_cube_map_surface: cube_get_cube_map_surface,
    lock_rect: cube_lock_rect,
    unlock_rect: cube_unlock_rect,
    add_dirty_rect: cube_add_dirty_rect,
};

/// `IDirect3DCubeTexture9` COM wrapper.
///
/// Layout-identical to `Direct3DTexture9` (see the module note above).
#[repr(C)]
pub struct Direct3DCubeTexture9 {
    vtbl: *const IDirect3DCubeTexture9Vtbl,
    refcount: u32,
    private_refcount: u32,
    inner: *mut TextureInner,
}

impl Direct3DCubeTexture9 {
    pub fn new(info: TextureCreateInfo) -> Self {
        Self {
            vtbl: &raw const DIRECT3D_CUBE_TEXTURE9_VTBL,
            refcount: 1,
            private_refcount: 0,
            inner: build_texture_inner(info),
        }
    }

    pub fn inner(&self) -> &TextureInner {
        // SAFETY: installed by `build_texture_inner` and owned for the live
        // wrapper lifetime, identical to `Direct3DTexture9`.
        unsafe { &*self.inner }
    }
}

const extern "system" fn cube_get_type(_this: *mut c_void) -> u32 {
    D3DRTYPE_CUBETEXTURE
}

// Cube faces share the parent texture and select their face and mip through the
// cube sidecar. The wrapper is layout-identical to `Direct3DTexture9`, so the
// delegated texture thunks and their casts are sound.
extern "system" fn cube_get_level_desc(this: *mut c_void, level: u32, desc: *mut c_void) -> i32 {
    // A cube level is a surface, so the delegated `D3DSURFACE_DESC.Type` of
    // `D3DRTYPE_SURFACE` is already correct — no per-level override.
    texture_get_level_desc(this, level, desc.cast::<D3DSURFACE_DESC>())
}

extern "system" fn cube_get_cube_map_surface(
    this: *mut c_void,
    face: u32,
    level: u32,
    surface: *mut *mut c_void,
) -> i32 {
    let _timer = tex_timer(this);
    if face >= CUBE_FACE_COUNT || surface.is_null() {
        null_out(surface);
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DCubeTexture9, layout-identical
    // to Direct3DTexture9.
    let Some(mut obj) = (unsafe { InPtrMut::<Direct3DTexture9>::opt(this) }) else {
        null_out(surface);
        return D3DERR_INVALIDCALL;
    };
    let ti = obj.inner();
    if level >= ti.app_level_count() {
        null_out(surface);
        return D3DERR_INVALIDCALL;
    }
    let Some(index) = ti.cube_subresource_index(face, level as usize) else {
        null_out(surface);
        return D3DERR_INVALIDCALL;
    };
    let cached = ti.cached_subresource(index); // last use of `obj` on this path
    if cached != 0 {
        // SAFETY: a non-zero slot is the live cached surface for this face and
        // level, and the `obj` borrow ended above, so the AddRef it forwards to
        // this cube does not alias it.
        unsafe { hand_back_cached_surface(cached, surface) };
        return 0;
    }
    let device_inner = ti.device_inner as *mut DeviceInner;
    let surf = Direct3DSurface9::new_cube_texture_backed(
        device_inner,
        this.cast::<Direct3DTexture9>(),
        face,
        level,
    );
    let surf_ptr = Box::into_raw(Box::new(surf));
    // The container owns the wrapper from here: every later call for this face
    // and level hands the same pointer back, and `finalize_texture` frees it.
    obj.inner_mut().cache_subresource(index, surf_ptr as u64); // last use of `obj`
    // The returned surface forwards its public reference to the parent cube,
    // matching ordinary texture-level surfaces.
    // SAFETY: `this` is the live cube parent and the surface owns the new ref.
    unsafe { crate::com_ref::com_add_ref::<Direct3DTexture9>(this) };
    // SAFETY: vtable out-param; `surface` is *mut *mut c_void per the ABI.
    unsafe { OutPtr::write_opt(surface, surf_ptr.cast::<c_void>()) };
    0
}

/// Lock one cube face subresource.
pub extern "system" fn cube_lock_rect(
    this: *mut c_void,
    face: u32,
    level: u32,
    locked_rect: *mut D3DLOCKED_RECT,
    rect: *const c_void,
    flags: u32,
) -> i32 {
    let _timer = tex_timer(this);
    if face >= CUBE_FACE_COUNT || locked_rect.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable `this` is the live cube wrapper for this call.
    let Some(mut obj) = (unsafe { InPtrMut::<Direct3DTexture9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let ti = obj.inner_mut();
    if level >= ti.app_level_count()
        || (ti.d3d_pool == D3DPOOL_DEFAULT && ti.d3d_usage & D3DUSAGE_DYNAMIC == 0)
    {
        return D3DERR_INVALIDCALL;
    }
    let level_u = level as usize;
    if ti.cube_is_locked(face, level_u) {
        return D3DERR_INVALIDCALL;
    }
    let mip_w = ti.mip_width(level_u);
    let mip_h = ti.mip_height(level_u);
    if ti.d3d_pool == D3DPOOL_DEFAULT {
        // SAFETY: `rect` is the caller's optional read-only RECT pointer.
        let provided = unsafe { ValueIn::<D3DRECT>::read_opt(rect) };
        if provided
            .is_some_and(|r| !default_lock_rect_valid(&r, mip_w, mip_h, ti.block_w, ti.block_h))
        {
            return D3DERR_INVALIDCALL;
        }
    }
    let dirty_rect = parse_rect(rect, mip_w, mip_h);
    let Some((ptr, pitch)) = ti.cube_lock_region_ptr(face, level_u, dirty_rect, flags) else {
        return D3DERR_INVALIDCALL;
    };
    ti.cube_stash_lock(
        face,
        level_u,
        flags & D3DLOCK_READONLY != 0,
        flags & D3DLOCK_NO_DIRTY_UPDATE != 0,
    );
    // SAFETY: checked non-null and points to a writable ABI out-param.
    let out = unsafe { &mut *locked_rect };
    out.pitch = pitch.cast_signed();
    out.bits = ptr.cast::<c_void>();
    D3D_OK
}

/// Unlock one cube face subresource.
pub extern "system" fn cube_unlock_rect(this: *mut c_void, face: u32, level: u32) -> i32 {
    if face >= CUBE_FACE_COUNT {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable `this` is the live cube wrapper for this call.
    let Some(mut obj) = (unsafe { InPtrMut::<Direct3DTexture9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let ti = obj.inner_mut();
    if level >= ti.app_level_count() {
        return D3DERR_INVALIDCALL;
    }
    let level_u = level as usize;
    let (read_only, no_dirty, was_locked, was_uploaded) = ti.cube_take_lock(face, level_u);
    if !was_locked {
        return D3D_OK;
    }
    if read_only && was_uploaded {
        return D3D_OK;
    }
    if !read_only && !no_dirty {
        ti.mark_cube_update_dirty(face, level_u, None);
    }
    ti.mark_cube_dirty(face, level_u);
    let device_inner = ti.device_inner;
    if device_inner != 0 {
        // SAFETY: the cube's device back-reference remains live while attached.
        unsafe { &mut *(device_inner as *mut DeviceInner) }.mark_snapshot_dirty_all();
    }
    D3D_OK
}

extern "system" fn cube_add_dirty_rect(this: *mut c_void, face: u32, rect: *const c_void) -> i32 {
    if face >= CUBE_FACE_COUNT {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable `this` is the live cube wrapper for this call.
    let Some(mut obj) = (unsafe { InPtrMut::<Direct3DTexture9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let ti = obj.inner_mut();
    let (w, h) = (ti.mip_width(0), ti.mip_height(0));
    // SAFETY: `rect` is the caller's optional read-only RECT pointer.
    let dirty = if let Some(r) = unsafe { ValueIn::<D3DRECT>::read_opt(rect) } {
        if r.x1 < 0
            || r.y1 < 0
            || r.x2 <= r.x1
            || r.y2 <= r.y1
            || r.x2.cast_unsigned() > w
            || r.y2.cast_unsigned() > h
        {
            return D3DERR_INVALIDCALL;
        }
        Some(DirtyRect {
            x: r.x1.cast_unsigned(),
            y: r.y1.cast_unsigned(),
            w: (r.x2 - r.x1).cast_unsigned(),
            h: (r.y2 - r.y1).cast_unsigned(),
        })
    } else {
        None
    };
    ti.mark_cube_update_dirty(face, 0, dirty);
    D3D_OK
}
