//! Owned frame operation data shared by API recording and native encoding.
//!
//! These Rust payloads describe local ownership. The PE/Unix transport uses explicit wire records.

use std::sync::Arc;

use mtld3d_shared::{
    MetalHandle,
    mtl::{PixelFormat, Swizzle, TextureCreateFlags, TextureUsage},
    mtl_handle::{CAMetalLayerKind, MTLDeviceKind, MTLTextureKind, NSViewKind},
    record_handle::DeviceRecordHandle,
};
use mtld3d_types::SAMPLER_STATE_COUNT;

use crate::{
    buffer_rename::BufferMapMode,
    dirty_rect::DirtyRect,
    draw_data::{CurrentSnapshotPtr, DrawOp, ScratchSlice},
    encoder_reply::{ReplyBool, ReplyU64},
    ids::{BufferId, ProgramId, TextureId},
    page_box::{PageBox, PageBoxRead},
    passes::BackbufferContents,
    perf::FramePerfPayload,
    present::LayerPacing,
    render_scale::RenderScale,
    scratch::ScratchArena,
    upload_redirty::{EmittedUpload, RedirtyQueue, RedirtySubresource},
};

bitflags::bitflags! {
    pub struct BindDepthOpFlags: u8 {
        const SAMPLEABLE = 1 << 0;
        const HAS_STENCIL = 1 << 1;
    }
}

bitflags::bitflags! {
    pub struct UploadTextureOpFlags: u8 {
        const ORDERED = 1 << 0;
        const REGENERATE_MIPMAPS = 1 << 1;
    }
}

bitflags::bitflags! {
    /// `StretchRect`-eligibility classification of a surface.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct StretchSurfaceFlags: u8 {
        /// The surface is a render target.
        ///
        /// Either a standalone backbuffer/RT, or a
        /// texture-level surface whose texture carries `D3DUSAGE_RENDERTARGET`.
        const IS_RENDER_TARGET = 1 << 0;
        /// The surface is a `CreateOffscreenPlainSurface(D3DPOOL_DEFAULT)` surface.
        ///
        /// A valid `StretchRect` destination, unlike an ordinary
        /// texture-level surface.
        const IS_OFFSCREEN_PLAIN_DEFAULT = 1 << 1;
        /// The surface is a standalone depth-stencil surface (`CreateDepthStencilSurface`).
        ///
        /// `StretchRect` allows only a 1:1
        /// depth→depth copy between two such surfaces.
        const IS_DEPTH_STENCIL = 1 << 2;
    }
}

#[derive(Clone)]
pub enum RtBinding {
    Backbuffer {
        handle: MetalHandle<MTLTextureKind>,
        /// Multisampled companion of the back buffer, NULL when there is none.
        msaa: MetalHandle<MTLTextureKind>,
        /// sRGB twin view of that companion, NULL whenever the companion is.
        msaa_srgb: MetalHandle<MTLTextureKind>,
        sample_count: u8,
        width: u32,
        height: u32,
    },
    /// A standalone `CreateRenderTarget` colour surface.
    ///
    /// `parent_texture` is null (so it is not texture-backed) but it
    /// carries its own persistent `metal_color_handle` distinct from the
    /// backbuffer, plus its own format and dimensions. Bound directly,
    /// unlike `Backbuffer`, the format is the surface's actual format, not
    /// the hard-wired backbuffer `Bgra8Unorm`.
    StandaloneColor {
        handle: MetalHandle<MTLTextureKind>,
        /// sRGB twin view of `handle`, or null when the format has none.
        ///
        /// Registered with the pass state when the target is bound, so a
        /// `D3DRS_SRGBWRITEENABLE` draw onto it attaches the twin.
        srgb: MetalHandle<MTLTextureKind>,
        /// Multisampled companion of the surface, NULL when there is none.
        msaa: MetalHandle<MTLTextureKind>,
        /// sRGB twin view of that companion, NULL whenever the companion is.
        msaa_srgb: MetalHandle<MTLTextureKind>,
        sample_count: u8,
        format: mtld3d_shared::mtl::PixelFormat,
        /// Whether the surface's D3D format has a real alpha channel.
        ///
        /// Carried separately because the Metal `format` can't distinguish
        /// X8R8G8B8 (no alpha) from A8R8G8B8 (both `Bgra8Unorm`). Feeds
        /// the pipeline snapshot's `COLOR_HAS_ALPHA` bit.
        has_alpha: bool,
        width: u32,
        height: u32,
    },
    Texture {
        info: TextureInfo,
        /// See `StandaloneColor::has_alpha`.
        has_alpha: bool,
        width: u32,
        height: u32,
        slice: u32,
        level: u32,
    },
}

/// `SetDepthStencilSurface` capture shape, owned by the operation pushed to the encoder thread.
///
/// `Lazy` defers the `MTLTexture` lookup to the encoder so a sampleable
/// shadow map's Metal handle is created (or reused from the cache) on
/// first bind, mirroring how `SetRenderTarget` handles texture-backed
/// render targets. `Eager` is the standalone-surface path
/// (`CreateDepthStencilSurface`) where the handle is known up-front.
#[derive(Clone)]
pub enum DepthBinding {
    None,
    /// A standalone depth surface: its Metal handle, the texture's real extent and its scale.
    Eager(
        MetalHandle<MTLTextureKind>,
        (u32, u32),
        crate::render_scale::RenderScale,
    ),
    /// A texture-backed depth surface: the parent's info, the mip level and the parent's scale.
    Lazy(TextureInfo, u32, crate::render_scale::RenderScale),
}

pub enum StretchKind {
    Texture(TextureInfo),
    Backbuffer(MetalHandle<MTLTextureKind>),
    /// A standalone depth-stencil surface's retained `Private` depth texture.
    DepthStencil(MetalHandle<MTLTextureKind>),
}

/// API-thread snapshot of a `StretchRect` source / destination surface.
///
/// `kind` carries enough info for the encoder operation to resolve the
/// underlying Metal texture handle without holding the surface pointer
/// (which may be released before the operation runs).
pub struct StretchSurfaceInfo {
    pub kind: StretchKind,
    /// Surface width as D3D9 reports it.
    pub width: u32,
    /// Surface height as D3D9 reports it.
    pub height: u32,
    /// Extent Metal allocated for the addressed subresource.
    ///
    /// `scale` of `width`/`height` for a surface or level 0, and Metal's own
    /// halving of the scaled base for a deeper level, which can differ from
    /// the scale of that level's reported size by a texel.
    pub texture_size: (u32, u32),
    /// What this endpoint's texture is rasterized at relative to `width`/`height`.
    ///
    /// Resolved on the API thread, where the backing resource is reachable, so
    /// the encoder-thread body can convert each endpoint without having to
    /// re-derive which surfaces `render.scale` applies to.
    pub scale: crate::render_scale::RenderScale,
    pub format: u32,
    pub mip_level: u32,
    /// Array slice the surface addresses within its backing texture.
    ///
    /// `Some(face)` is a cube face's `D3DCUBEMAP_FACES` index; `None` is every
    /// other surface kind, whose backing texture holds a single slice.
    pub slice: Option<u32>,
    /// D3DPOOL_* of the backing resource.
    ///
    /// `StretchRect` requires both surfaces in `D3DPOOL_DEFAULT`.
    pub pool: u32,
    /// Surface-kind classification.
    ///
    /// One of `IS_RENDER_TARGET` / `IS_OFFSCREEN_PLAIN_DEFAULT` /
    /// `IS_DEPTH_STENCIL`. See [`StretchSurfaceFlags`].
    pub flags: StretchSurfaceFlags,
    /// `Some(texture id)` when the backing texture carries `D3DUSAGE_AUTOGENMIPMAP`.
    ///
    /// A `StretchRect` or a `ColorFill` into level 0 must
    /// regenerate the mip chain afterwards, the same way a
    /// level-0 `UnlockRect` does.
    pub autogen_texture_id: Option<TextureId>,
    /// Multisampled companion of the surface's texture, or null.
    ///
    /// A multisampled source is read through the single-sample texture the
    /// resolve fills; a multisampled destination is written through this one
    /// by the render-quad path and resolved back at pass end.
    pub msaa: MetalHandle<MTLTextureKind>,
    /// sRGB twin view of that companion, or null whenever the companion is.
    pub msaa_srgb: MetalHandle<MTLTextureKind>,
    /// Sample count of the surface, 1 when it is single-sampled.
    pub sample_count: u8,
}

/// Render target 0 as the encoder binds it.
///
/// A parameter bag rather than eight positional arguments. `logical_size` is
/// the extent D3D9 reports, `size` the one Metal allocated for the bound
/// subresource, and `scale` what it is rasterized at; `msaa_texture`
/// is the multisampled companion the pass attaches, NULL for a single-sampled
/// target, and `sample_count` its count.
pub struct ColorRtBinding {
    pub texture: MetalHandle<MTLTextureKind>,
    pub msaa_texture: MetalHandle<MTLTextureKind>,
    /// sRGB twin view of `msaa_texture`, NULL whenever that is.
    pub msaa_srgb_texture: MetalHandle<MTLTextureKind>,
    pub sample_count: u8,
    pub logical_size: (u32, u32),
    /// Extent Metal allocated for the bound subresource.
    pub size: (u32, u32),
    pub format: PixelFormat,
    pub has_alpha: bool,
    pub scale: RenderScale,
    /// `(slice, level)` of the attachment.
    pub subresource: (u32, u32),
}

/// One side (source or destination) of a scaled blit, for `stretch_blit_scaled`.
pub struct BlitSide {
    pub handle: u64,
    pub rect: crate::stretch_rect::StretchRegion,
    pub dims: (u32, u32),
    /// Mip level of `handle` the blit reads or writes.
    pub mip: u32,
    /// Array slice of `handle` the blit reads or writes.
    ///
    /// `Some(face)` when the endpoint is one face of a cube map, `None` when
    /// its backing texture holds a single slice. The destination attaches the
    /// face as the colour attachment's slice; the source is sampled through a
    /// 2D view of it, since a `texturecube` binding would read face 0.
    pub slice: Option<u32>,
    /// Multisampled companion of `handle`, NULL when there is none.
    ///
    /// Read on the destination side only: the quad renders into the
    /// multisampled attachment and the pass resolves into `handle`. A
    /// multisampled *source* is read through `handle`, which the resolve has
    /// already filled.
    pub msaa: MetalHandle<MTLTextureKind>,
    /// sRGB twin view of that companion, NULL whenever the companion is.
    pub msaa_srgb: MetalHandle<MTLTextureKind>,
    /// Sample count of the destination, 1 when it is single-sampled.
    pub sample_count: u8,
}

/// One staging write-back into a colour texture that has to change size on the way in.
///
/// `GetDC` and `LockRect` hand their bytes out at the extent D3D9 reports, so
/// under a `render.scale` below 100% the page the caller wrote is larger than
/// the texture it belongs in. Serves the back buffer's `ReleaseDC` and a
/// lockable render target's `UnlockRect`. Built by `surface.rs` on the API
/// thread, which is where the device's scale and the surface's extent are both
/// reachable.
pub struct ResampledUpload {
    /// Destination colour `MTLTexture`.
    pub color_handle: u64,
    /// Metal format of the destination, and so of the staging source too.
    pub format: PixelFormat,
    /// Extent the `tight` rows describe, which is the extent D3D9 reports.
    pub logical: (u32, u32),
    /// Extent of the destination texture, at or below `logical`.
    pub texture: (u32, u32),
    /// Row stride of the source rows, which need not be the tight one.
    pub bytes_per_row: u32,
    /// Multisampled companion of the destination, null when single-sampled.
    pub msaa: MetalHandle<MTLTextureKind>,
    /// sRGB twin view of `msaa`, null whenever `msaa` is.
    pub msaa_srgb: MetalHandle<MTLTextureKind>,
    /// Sample count of the destination; 1 without a companion.
    pub sample_count: u8,
}

/// One `ColorFill` against a render-target texture, resolved on the API thread.
///
/// Built by `device_color_fill` and consumed by
/// `color_fill_target` on the encoder thread, which cannot
/// reach the device to re-derive the destination's scale or extent.
pub struct ColorFillTarget {
    /// Destination `MTLTexture`.
    pub texture: MetalHandle<MTLTextureKind>,
    /// Mip extent as D3D9 reports it.
    pub logical_size: (u32, u32),
    /// Extent Metal allocated for the destination subresource.
    pub texture_size: (u32, u32),
    /// Metal format of the destination as it was created on this device.
    pub format: PixelFormat,
    /// What the destination is rasterized at relative to `logical_size`.
    ///
    /// The fill rect converts from `logical_size` to `texture_size` through it.
    pub scale: RenderScale,
    /// `(array slice, mip level)` of the destination subresource.
    pub subresource: (u32, u32),
    /// Fill rect in D3D9 coordinates, as `(x, y, width, height)`.
    pub rect: (u32, u32, u32, u32),
    /// Fill colour, one `f32::to_bits` per channel in RGBA order.
    pub rgba: (u32, u32, u32, u32),
    /// Multisampled companion of the destination, null when single-sampled.
    pub msaa: MetalHandle<MTLTextureKind>,
    /// sRGB twin view of `msaa`, null whenever `msaa` is.
    pub msaa_srgb: MetalHandle<MTLTextureKind>,
    /// Sample count of the destination; 1 without a companion.
    pub sample_count: u8,
    /// True when the destination is level 0 of a `D3DUSAGE_AUTOGENMIPMAP` texture.
    ///
    /// The runtime owns that texture's mip chain, so the fill is followed by a
    /// regeneration from the level it just painted.
    pub regenerate_mipmaps: bool,
}

/// Parameter bag for `FrameEncoder::run_texture_upload`.
///
/// Built by `texture::schedule_upload` on the API thread and consumed by
/// the upload operation on the encoder thread; keeps the encoder method
/// signature to a single argument.
pub struct TextureUploadJob {
    pub info: TextureInfo,
    pub staging: PageBoxRead,
    pub level: u32,
    /// Destination array slice.
    ///
    /// Zero for ordinary textures; cube uploads use the face index.
    pub destination_slice: u32,
    /// Index in the texture's staging-buffer cache.
    ///
    /// Equal to `level` for ordinary textures and `face * levels + level` for
    /// cubes.
    pub staging_index: usize,
    pub origin_x: u32,
    pub origin_y: u32,
    pub region_w: u32,
    pub region_h: u32,
    pub src_d3d_format: u32,
    pub src_pitch: u32,
    pub bytes_per_pixel: u32,
    /// Slice count for this mip.
    ///
    /// 1 for a 2D texture, `(depth >> level)` (≥1) for a volume (3D)
    /// texture. Selects the 2D vs volume blit path in `run_texture_upload`.
    pub depth: u32,
    /// Byte stride between slices (the box slice pitch).
    ///
    /// Only read by the volume blit path; the 2D path derives its
    /// single-slice `bytes_per_image` from the region's block-row count
    /// instead.
    pub slice_pitch: u32,
    /// Where a job reports what became of it.
    ///
    /// The dirty state this upload was built from is already cleared when
    /// the encoder thread sees the job, so a decline that stayed on this
    /// thread would lose the region: `UnlockRect` publishes only the
    /// rectangle the game locked and nothing re-announces the rest. The
    /// queue carries the subresource and the rectangle back to the API
    /// thread, which marks them dirty again for the next bind, and carries
    /// the emitted answer back for the levels waiting to release their
    /// staging.
    pub redirty: Arc<RedirtyQueue>,
    /// Whether the level releases its staging once this upload is emitted.
    ///
    /// Set for a level of the staging-droppable class whose every texel this
    /// upload puts on the GPU. The release waits for the emitted answer: a
    /// level released at schedule time has no bytes left to retry from when
    /// the upload is declined downstream of the hand-off.
    pub release_staging: bool,
    /// Which of the level's scheduled uploads this one is.
    ///
    /// Travels back with the emitted answer so the release can tell whether a
    /// later upload of the level is still waiting for an answer of its own.
    pub upload_generation: u32,
}

/// Texture metadata captured from the API thread for deferred Metal creation.
#[derive(Clone)]
pub struct TextureInfo {
    pub texture_id: TextureId,
    /// `D3DFMT_*` the game created the texture with.
    ///
    /// Kept alongside `pixel_format` because the pair is what decides how an
    /// upload reaches the texture: the same `Bgra8Unorm` backs an
    /// `A8R8G8B8` source verbatim and a packed 16-bit source through the
    /// widening upload pass.
    pub d3d_format: u32,
    pub width: u32,
    pub height: u32,
    /// Slice count: 1 for 2D textures, >1 for a volume (3D) texture.
    pub depth: u32,
    pub levels: u32,
    pub pixel_format: PixelFormat,
    pub create_flags: TextureCreateFlags,
    pub swizzle: [Swizzle; 4],
    /// `TextureUsage` bits passed through to the unix side.
    ///
    /// The Metal texture is allocated with `RenderTarget` usage when the
    /// D3D9 texture was created with `D3DUSAGE_RENDERTARGET`.
    pub usage_flags: TextureUsage,
}

/// The Metal textures a standalone colour surface owns, for retirement.
///
/// A surface can carry up to four: the single-sample texture the D3D9
/// surface's identity is, its sRGB twin view, the multisampled companion the
/// passes attach, and that companion's own twin. Grouped so
/// `retire_color_target` takes one argument per surface
/// rather than one per view.
pub struct RetiredColorTarget {
    pub base: MetalHandle<MTLTextureKind>,
    pub srgb: MetalHandle<MTLTextureKind>,
    pub msaa: MetalHandle<MTLTextureKind>,
    pub msaa_srgb: MetalHandle<MTLTextureKind>,
}

/// One whole-level depth transfer between two depth textures.
///
/// `source_size` is the extent of the source level and `destination_size` that
/// of the destination's level 0. The two may differ, in which case each
/// destination texel takes the nearest source texel. A multisampled source
/// contributes sample zero, which is what the D3D9 RESZ hack and a
/// depth-to-depth `StretchRect` resolve deliver. The stencil plane travels
/// when both ends carry one.
pub struct DepthTransfer {
    pub source: MetalHandle<MTLTextureKind>,
    pub source_level: u32,
    pub source_size: (u32, u32),
    pub source_format: PixelFormat,
    pub source_samples: u8,
    pub destination: MetalHandle<MTLTextureKind>,
    pub destination_size: (u32, u32),
    pub destination_format: PixelFormat,
}

pub struct SetViewportOp {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub min_z: f32,
    pub max_z: f32,
}

pub struct SetVertexSamplerOp {
    pub slot: u8,
    pub state: [u32; SAMPLER_STATE_COUNT],
}

pub struct SetVertexTextureOp {
    pub slot: u8,
    pub id: Option<TextureId>,
}

pub struct BindDepthOp {
    pub binding: DepthBinding,
    pub sample_count: u8,
    pub flags: BindDepthOpFlags,
}

pub struct BindColorOp {
    pub slot: u8,
    pub info: RtBinding,
    pub scale: crate::render_scale::RenderScale,
}

pub struct GenerateMipmapsOrderedOp {
    pub old_id: TextureId,
}

pub struct UnbindExtraColorOp {
    pub slot: u8,
}

pub struct DestroyTextureOp {
    pub tex_id: TextureId,
}

pub struct ReadColorHandleOp {
    pub texture_id: TextureId,
    pub slot_op: ReplyU64,
}

pub struct NoteColorReadOp {
    pub src: MetalHandle<MTLTextureKind>,
}

pub struct ResolveDepthSurfaceOp {
    pub transfer: DepthTransfer,
}

pub struct StretchBlitOp {
    pub src_info: StretchSurfaceInfo,
    pub dst_info: StretchSurfaceInfo,
    pub src_region: crate::stretch_rect::StretchRegion,
    pub dst_region: crate::stretch_rect::StretchRegion,
    pub mip_level: u32,
    pub render_quad: bool,
    pub filter: u32,
}

pub struct ColorFillOp {
    pub kind: StretchKind,
    pub fill: ColorFillTarget,
}

pub struct CarryDepthOp {
    pub prev_id: TextureId,
    pub cur_id: TextureId,
    pub mip_w: u32,
    pub mip_h: u32,
}

pub struct ClearColorOp {
    pub r_bits: u32,
    pub g_bits: u32,
    pub b_bits: u32,
    pub a_bits: u32,
    pub srgb_write: bool,
}

pub struct ClearColorRectsOp {
    pub r_bits: u32,
    pub g_bits: u32,
    pub b_bits: u32,
    pub a_bits: u32,
    pub srgb_write: bool,
    pub rects: Vec<(i32, i32, i32, i32)>,
}

pub struct ClearDepthStencilRectsOp {
    pub depth: Option<u32>,
    pub stencil: Option<u32>,
    pub list: Vec<(i32, i32, i32, i32)>,
}

pub struct ClearDepthStencilOp {
    pub depth: Option<u32>,
    pub stencil: Option<u32>,
}

pub struct ResolveDynamicDepthOp {
    pub id: TextureId,
    pub info: TextureInfo,
}

pub struct ResolveDepthTextureOp {
    pub id: TextureId,
    pub w: u32,
    pub h: u32,
    pub format: mtld3d_shared::mtl::PixelFormat,
}

pub struct ReadDeviceBufferOp {
    pub done: ReplyBool,
    pub buffer_id: BufferId,
    pub dst_ptr: u64,
    pub dst_len: u64,
}

pub struct AdoptProgramOp {
    pub registration: u64,
}

pub struct RegisterProgramOp {
    pub shader_id: ProgramId,
    pub program: crate::dxso::DxsoProgram,
}

pub struct BeginVisibilityOp {
    pub generation: u64,
    pub c: Arc<crate::visibility::VisibilityQueryCore>,
}

pub struct EndVisibilityOp {
    pub generation: u64,
    pub core: Arc<crate::visibility::VisibilityQueryCore>,
}

pub struct RetireColorOp {
    pub retired: RetiredColorTarget,
}

pub struct RetireDepthOp {
    pub depth: MetalHandle<MTLTextureKind>,
}

pub struct UploadColorOp {
    pub color_handle: u64,
    pub bytes: ScratchSlice,
    pub width: u32,
    pub height: u32,
    pub src_stride: u32,
}

pub struct UploadResampledOp {
    pub target: ResampledUpload,
    pub bytes: ScratchSlice,
}

pub struct ReadTextureHandleOp {
    pub texture_id: TextureId,
    pub slot_op: ReplyU64,
}

pub struct GenerateMipmapsOp {
    pub texture_id: TextureId,
}

pub struct ReadTextureColorHandleOp {
    pub texture_id: TextureId,
    pub slot_op: ReplyU64,
}

pub struct UploadTextureAndMipsOp {
    pub job: TextureUploadJob,
    pub texture_id: TextureId,
    pub flags: UploadTextureOpFlags,
}

pub struct UploadTextureOp {
    pub job: TextureUploadJob,
}

pub struct SetDumpDrawOp {
    pub seq: u32,
}

impl TextureUploadJob {
    /// The subresource key this job's decline record is filed under.
    #[must_use]
    pub fn redirty_subresource(&self) -> RedirtySubresource {
        RedirtySubresource {
            texture_id: self.info.texture_id,
            index: u32::try_from(self.staging_index).unwrap_or(u32::MAX),
        }
    }

    /// The answer this job reports once its upload reaches the command stream.
    #[must_use]
    pub fn emitted_answer(&self) -> EmittedUpload {
        EmittedUpload {
            subresource: self.redirty_subresource(),
            level: self.level,
            generation: self.upload_generation,
            releases_staging: self.release_staging,
        }
    }

    /// The region this job was carrying, as a dirty rectangle.
    #[must_use]
    pub const fn redirty_rect(&self) -> DirtyRect {
        DirtyRect {
            x: self.origin_x,
            y: self.origin_y,
            w: self.region_w,
            h: self.region_h,
        }
    }
}

/// Discriminated union over the work the API thread queues for the encoder.
///
/// The hot per-draw path uses `Draw`, with no
/// per-op heap allocation. The latter carries changed state. Non-draw
/// operations carry typed, owned payloads and dispatch exhaustively.
///
/// See `windows/core/src/scratch.rs` for why hot payloads (snapshots,
/// const ranges, stage bindings) are pointers into the per-frame arena
/// rather than `Box<T>`.
pub enum Op {
    /// Apply a delta into the encoder-side VS programmable constant mirror.
    ///
    /// `data` is a scratch-allocated `[u8]` of `rows × 16` bytes starting
    /// at row `start_row`. Pushed by `SetVertexShaderConstantF` (and
    /// state-block-apply sites) on the API thread; consumed by `run_frame`
    /// which copies the bytes into `FrameEncoder::vs_constants_mirror`.
    SetVsConstRange {
        start_row: u16,
        rows: u16,
        data: ScratchSlice,
    },
    SetPsConstRange {
        start_row: u16,
        rows: u16,
        data: ScratchSlice,
    },
    /// Section delta into the FF VS const mirror.
    ///
    /// Pushed once per dirty `FfVsDirty` section from
    /// `emit_ff_vs_section_deltas`. Structurally identical to
    /// `SetVsConstRange` but routes to a separate mirror because FF and
    /// programmable VS feed different content into the same shader slot
    /// (slot 0 c-bank).
    SetFfVsConstRange {
        start_row: u16,
        rows: u16,
        data: ScratchSlice,
    },
    /// Install decoded native state before dependent draws.
    SetSnapshot(CurrentSnapshotPtr),
    /// Issue a draw using the current snapshot.
    Draw(DrawOp),
    SetViewport(
        #[cfg(windows)] SetViewportOp,
        #[cfg(not(windows))] Box<SetViewportOp>,
    ),
    SetVertexSampler(
        #[cfg(windows)] SetVertexSamplerOp,
        #[cfg(not(windows))] Box<SetVertexSamplerOp>,
    ),
    SetVertexTexture(
        #[cfg(windows)] SetVertexTextureOp,
        #[cfg(not(windows))] Box<SetVertexTextureOp>,
    ),
    BindDepth(
        #[cfg(windows)] BindDepthOp,
        #[cfg(not(windows))] Box<BindDepthOp>,
    ),
    BindColor(
        #[cfg(windows)] BindColorOp,
        #[cfg(not(windows))] Box<BindColorOp>,
    ),
    GenerateMipmapsOrdered(
        #[cfg(windows)] GenerateMipmapsOrderedOp,
        #[cfg(not(windows))] Box<GenerateMipmapsOrderedOp>,
    ),
    UnbindExtraColor(
        #[cfg(windows)] UnbindExtraColorOp,
        #[cfg(not(windows))] Box<UnbindExtraColorOp>,
    ),
    DestroyTexture(
        #[cfg(windows)] DestroyTextureOp,
        #[cfg(not(windows))] Box<DestroyTextureOp>,
    ),
    ReadColorHandle(
        #[cfg(windows)] ReadColorHandleOp,
        #[cfg(not(windows))] Box<ReadColorHandleOp>,
    ),
    NoteColorRead(
        #[cfg(windows)] NoteColorReadOp,
        #[cfg(not(windows))] Box<NoteColorReadOp>,
    ),
    ResolveDepthSurface(
        #[cfg(windows)] ResolveDepthSurfaceOp,
        #[cfg(not(windows))] Box<ResolveDepthSurfaceOp>,
    ),
    StretchBlit(
        #[cfg(windows)] StretchBlitOp,
        #[cfg(not(windows))] Box<StretchBlitOp>,
    ),
    ColorFill(
        #[cfg(windows)] ColorFillOp,
        #[cfg(not(windows))] Box<ColorFillOp>,
    ),
    CarryDepth(
        #[cfg(windows)] CarryDepthOp,
        #[cfg(not(windows))] Box<CarryDepthOp>,
    ),
    ClearColor(
        #[cfg(windows)] ClearColorOp,
        #[cfg(not(windows))] Box<ClearColorOp>,
    ),
    ClearColorRects(
        #[cfg(windows)] ClearColorRectsOp,
        #[cfg(not(windows))] Box<ClearColorRectsOp>,
    ),
    ClearDepthStencilRects(
        #[cfg(windows)] ClearDepthStencilRectsOp,
        #[cfg(not(windows))] Box<ClearDepthStencilRectsOp>,
    ),
    ClearDepthStencil(
        #[cfg(windows)] ClearDepthStencilOp,
        #[cfg(not(windows))] Box<ClearDepthStencilOp>,
    ),
    ResolveDynamicDepth(
        #[cfg(windows)] ResolveDynamicDepthOp,
        #[cfg(not(windows))] Box<ResolveDynamicDepthOp>,
    ),
    ResolveDepthTexture(
        #[cfg(windows)] ResolveDepthTextureOp,
        #[cfg(not(windows))] Box<ResolveDepthTextureOp>,
    ),
    ReadDeviceBuffer(
        #[cfg(windows)] ReadDeviceBufferOp,
        #[cfg(not(windows))] Box<ReadDeviceBufferOp>,
    ),
    RegisterProgram(
        #[cfg(windows)] RegisterProgramOp,
        #[cfg(not(windows))] Box<RegisterProgramOp>,
    ),
    AdoptProgram(
        #[cfg(windows)] AdoptProgramOp,
        #[cfg(not(windows))] Box<AdoptProgramOp>,
    ),
    BeginVisibility(
        #[cfg(windows)] BeginVisibilityOp,
        #[cfg(not(windows))] Box<BeginVisibilityOp>,
    ),
    EndVisibility(
        #[cfg(windows)] EndVisibilityOp,
        #[cfg(not(windows))] Box<EndVisibilityOp>,
    ),
    RetireColor(
        #[cfg(windows)] RetireColorOp,
        #[cfg(not(windows))] Box<RetireColorOp>,
    ),
    RetireDepth(
        #[cfg(windows)] RetireDepthOp,
        #[cfg(not(windows))] Box<RetireDepthOp>,
    ),
    UploadColor(
        #[cfg(windows)] UploadColorOp,
        #[cfg(not(windows))] Box<UploadColorOp>,
    ),
    UploadResampled(
        #[cfg(windows)] UploadResampledOp,
        #[cfg(not(windows))] Box<UploadResampledOp>,
    ),
    ReadTextureHandle(
        #[cfg(windows)] ReadTextureHandleOp,
        #[cfg(not(windows))] Box<ReadTextureHandleOp>,
    ),
    GenerateMipmaps(
        #[cfg(windows)] GenerateMipmapsOp,
        #[cfg(not(windows))] Box<GenerateMipmapsOp>,
    ),
    ReadTextureColorHandle(
        #[cfg(windows)] ReadTextureColorHandleOp,
        #[cfg(not(windows))] Box<ReadTextureColorHandleOp>,
    ),
    UploadTextureAndMips(
        #[cfg(windows)] UploadTextureAndMipsOp,
        #[cfg(not(windows))] Box<UploadTextureAndMipsOp>,
    ),
    UploadTexture(
        #[cfg(windows)] UploadTextureOp,
        #[cfg(not(windows))] Box<UploadTextureOp>,
    ),
    SetDumpDraw(
        #[cfg(windows)] SetDumpDrawOp,
        #[cfg(not(windows))] Box<SetDumpDrawOp>,
    ),
    /// Inline op-stream-ordered `Staged` VB/IB upload.
    ///
    /// Carries the transient `PageBox` snapshot of the bytes the game
    /// wrote between `Lock` and `Unlock` (taken on the API thread, no
    /// Metal thunk there). The encoder wraps it as a `bytesNoCopy` blit
    /// source and copies its range into the buffer's persistent `Private`
    /// device buffer via `frame_blit_commands` (a leading phase, before
    /// any draw). If a draw earlier this frame already read an overlapping
    /// region, the encoder first renames the device buffer so the earlier
    /// draw keeps its bytes, see `apply_stage_upload`.
    StageUpload {
        buffer_id: BufferId,
        page_box: PageBox,
        dst_offset: u32,
        size: u32,
    },
}

bitflags::bitflags! {
    /// Per-frame boolean state on [`FrameData`].
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct FrameDataFlags: u8 {
        /// `depth_texture` is a combined depth+stencil format.
        ///
        /// Forwarded to `PassState::reset_frame` so clear-quad pipelines
        /// match the pass.
        const DEPTH_HAS_STENCIL = 1 << 0;
        /// This frame is a mid-frame checkpoint rather than a user `Present`.
        ///
        /// The triggers are `LockRect` on the backbuffer and
        /// `GetRenderTargetData`. `submit()` honours the flag by zeroing the
        /// present-layer fields in `SubmitFrameParams` so the Metal side
        /// queues no present for it; a present still waiting for its drawable
        /// is copied into a slot rather than waited for. The command buffer
        /// still commits, so in-order queue execution makes the backbuffer
        /// texture safe to read from the subsequent readback-blit command
        /// buffer.
        const NO_PRESENT = 1 << 1;
        /// First frame of an F12 run: start the Metal GPU capture before it.
        ///
        /// Set by `frame_dump_present` on the frame it arms. The encoder
        /// drains the submit thread, starts the capture and runs every frame
        /// synchronously until `GPU_CAPTURE_STOP` so each `SubmitFrame`
        /// thunk falls inside the bracket.
        const GPU_CAPTURE_START = 1 << 2;
        /// Last frame of an F12 run: stop the Metal GPU capture after it.
        ///
        /// A frame swap that does not present moves this bit onto the
        /// continuation, so the capture ends with the piece the closing
        /// `Present` submits rather than with the process.
        const GPU_CAPTURE_STOP = 1 << 3;
        /// Internal packet footer contains diagnostic provenance inventories.
        ///
        /// The paired decoder consumes this transport bit before rendering.
        const VALIDATION_INVENTORY = 1 << 4;
    }
}

/// Warmup entry for a `MTLBuffer` wrap pre-registered by the API thread.
///
/// Registered at `CreateVertexBuffer` / `CreateIndexBuffer` time. The encoder
/// thread drains the queue into one batched `CreateBuffersBatch` thunk at the
/// head of `run_frame`, before the op loop, so subsequent draw operations
/// hit the `buffer_cache` instead of cache-missing on first bind.
#[derive(Clone, Copy)]
pub struct VbibWarmupEntry {
    pub buffer_id: BufferId,
    pub backing_ptr: u64,
    pub backing_len: u64,
    /// The backing allocation's identity (see `PageBox::generation`).
    pub backing_generation: u64,
    /// Decides the create path.
    ///
    /// `Direct` → one `bytesNoCopy` wrapper (today's zero-copy bind);
    /// `Staged` → a `StorageModePrivate` device buffer (the draw-bind target
    /// written by staging-upload blits).
    pub map_mode: BufferMapMode,
}

/// Warmup entry for a texture mip's staging `MTLBuffer` wrap.
///
/// Pushed at `CreateTexture` time alongside the `TextureInfo` warmup; drained
/// after `drain_texture_warmups` so the `texture_cache` entry exists. The
/// resulting handle lands in `TextureGpuState::mip_staging_buffers[level]`,
/// so the first `UnlockRect`-driven upload's `get_or_create_staging_buffer`
/// hits the cache instead of cache-missing.
///
/// Skipped for RT / depth / expansion-path textures, their staging
/// backing is never wrapped as a cached `MTLBuffer` here (RT/depth have no
/// upload staging path; the expansion path builds its own transient staging
/// buffer per upload before blitting).
///
/// `keepalive` holds the PE-side `Arc<PageBox>` so the staging
/// allocation outlives the API thread's `texture_release` (which drops
/// the original Arc from `TextureInner.staging` synchronously) and is
/// still valid when `drain_staging_warmups` wraps it via
/// `newBufferWithBytesNoCopy:`. After drain the clone moves into
/// `MipStagingBuffer.keepalive`.
pub struct StagingWarmupEntry {
    pub texture_id: TextureId,
    pub level: u32,
    pub backing_ptr: u64,
    pub backing_len: u64,
    pub keepalive: Arc<PageBox>,
}

pub struct FrameData {
    pub recorder: Option<crate::encoder_packet::FrameRecorder>,
    pub ops: Vec<Op>,
    /// Texture creates pushed by the API thread at `IDirect3DDevice9::CreateTexture` time.
    ///
    /// Drained into one batched `CreateTexturesBatch` thunk at the head of
    /// `run_frame` so the `MTLTexture` exists before any draw operation
    /// references it.
    pub pending_texture_warmups: Vec<TextureInfo>,
    /// VB/IB wraps pushed at `CreateVertexBuffer` / `CreateIndexBuffer` time.
    ///
    /// Drained alongside the texture warmups via one batched
    /// `CreateBuffersBatch` thunk.
    pub pending_buffer_warmups: Vec<VbibWarmupEntry>,
    /// Per-mip staging `MTLBuffer` wraps pushed alongside texture warmups.
    ///
    /// One per mip for non-RT / non-depth / non-expansion textures. Drained
    /// after `drain_texture_warmups` so the `texture_cache` slot already
    /// exists.
    pub pending_staging_warmups: Vec<StagingWarmupEntry>,
    pub device_handle: MetalHandle<MTLDeviceKind>,
    pub record_handle: DeviceRecordHandle,
    pub backbuffer_handle: MetalHandle<MTLTextureKind>,
    /// sRGB twin view of the back buffer; see `FrameInit`.
    pub backbuffer_srgb_handle: MetalHandle<MTLTextureKind>,
    /// Multisampled companion of `backbuffer_handle`, NULL when there is none.
    pub backbuffer_msaa_handle: MetalHandle<MTLTextureKind>,
    /// sRGB twin view of `backbuffer_msaa_handle`; see `FrameInit`.
    pub backbuffer_msaa_srgb_handle: MetalHandle<MTLTextureKind>,
    /// Sample count the frame's back buffer and default depth surface carry.
    pub backbuffer_sample_count: u8,
    pub layer_handle: MetalHandle<CAMetalLayerKind>,
    /// `NSView*` the layer was attached to.
    ///
    /// Forwarded to `SubmitFrameParams.present_view`: it names the attachment
    /// record `submit_frame` reads its display state from (the window's
    /// occlusion, the live EDR headroom, the present throttle, the geometry
    /// streak), all of which the unix side derives for that window alone.
    pub view_handle: MetalHandle<NSViewKind>,
    /// Logical back-buffer width, the resolution D3D9 reports.
    ///
    /// `render_scale` of this is the rasterized extent.
    pub backbuffer_width: u32,
    /// Logical back-buffer height. See `backbuffer_width`.
    pub backbuffer_height: u32,
    /// Fraction of the logical resolution the back buffer is rasterized at.
    ///
    /// Handed to `PassState::reset_frame`, which reconciles the two spaces.
    pub render_scale: RenderScale,
    /// Metal pixel format of the backbuffer.
    ///
    /// Always `Bgra8Unorm` in mtld3d today, `unix/unix/src/metal/texture.rs`
    /// always creates the backbuffer as `BGRA8Unorm`. Seeded into
    /// `PassState::reset_frame` so the initial pass's pipeline cache key has
    /// the right format before any `SetRenderTarget`.
    pub backbuffer_format: PixelFormat,
    /// Whether the back buffer starts the frame undefined; see `FrameInit`.
    pub backbuffer_contents: BackbufferContents,
    pub depth_texture: MetalHandle<MTLTextureKind>,
    /// Per-frame boolean state (`DEPTH_HAS_STENCIL` / `NO_PRESENT`).
    ///
    /// See [`FrameDataFlags`].
    pub flags: FrameDataFlags,
    /// All per-frame telemetry drained from `ApiPerfState` by `DeviceInner::present`.
    ///
    /// Plus the `present_block_cycles` field set on the *next* frame right
    /// after `send_frame` returns. See `crate::perf::FramePerfPayload`.
    pub perf: FramePerfPayload,
    /// Per-frame VB/IB backings + submit seqs queued for GPU-retire-gated destruction.
    ///
    /// Destroyed on the encoder thread; consumed in `begin_frame`.
    pub vbib_retentions: Vec<PendingVbibRetention>,
    /// Monotonic submit seq stamped by `DeviceInner::present` before the encoder handoff.
    ///
    /// Carried into `SubmitFrameParams` so the unix `addCompletedHandler`
    /// knows which seq to broadcast.
    pub submit_seq: u64,
    /// Raw pointer to the device's `Arc<AtomicU64>` coherent-seq.
    ///
    /// Stays valid for the device's lifetime (Arc is dropped after all frames
    /// drain). The completion block stores the retired seq via this
    /// pointer with Release ordering.
    pub coherent_seq_ptr: u64,
    /// Raw pointer to the device's `Arc<AtomicU64>` upload-coherent-seq.
    ///
    /// Same lifetime guarantee as `coherent_seq_ptr`. Forwarded verbatim
    /// into `SubmitFrameParams::upload_coherent_seq_ptr`; non-zero tells
    /// the unix side to split the ordered upload prefix into its own,
    /// earlier-retiring command buffer. 0 only before the frame is
    /// stamped (`FrameData::new` default); every submitted frame carries
    /// the real pointer.
    pub upload_coherent_seq_ptr: u64,
    /// Raw pointer to the device's `Arc<AtomicU64>` failed-submit seq.
    ///
    /// Same lifetime guarantee as `coherent_seq_ptr`. Forwarded verbatim
    /// into `SubmitFrameParams::failed_submit_seq_ptr`, which both
    /// completion handlers `fetch_max` when their command buffer aborts.
    /// 0 only before the frame is stamped.
    pub failed_submit_seq_ptr: u64,
    /// Raw pointer to the device's `Arc<AtomicU64>` VB/IB retained-bytes total.
    ///
    /// Same lifetime guarantee as `coherent_seq_ptr`. The encoder
    /// `fetch_add`/`fetch_sub`s it as `PageBox`es enter/leave retention.
    pub retained_bytes_ptr: u64,
    /// `Some(v)` if `IDirect3DDevice9::Reset` changed `PresentationInterval` since the last frame.
    ///
    /// Put here by `stamp_and_swap` on the frame it hands to the encoder, so
    /// the Present that follows the Reset is the one that carries it. The
    /// encoder applies it via `SetDisplaySyncEnabledParams` at the top of
    /// `run_frame` so the new pacing takes effect on this frame's
    /// `nextDrawable`, matching the spec's "next Present" timing rather than
    /// the previous behaviour of mutating the layer property synchronously
    /// from the API thread mid-frame. The ceiling it carries is the effective
    /// one, the interval's folded with `present.maxFps`.
    pub apply_pacing: Option<LayerPacing>,
    /// `Some(t)` if `SetGammaRamp` changed what the layer should carry since the last frame.
    ///
    /// `Some(Some(table))` applies that table, `Some(None)` removes the one
    /// the layer has. Put here by `stamp_and_swap`, like `apply_pacing`, and
    /// sent by the encoder at the top of `run_frame` so the ramp is live for
    /// this frame's own present rather than the one after it. A `Box` so a
    /// frame carrying no change costs a pointer.
    pub apply_gamma: Option<crate::gamma::Change>,
    /// API-thread bump arena.
    ///
    /// Used by `snapshot_shared` to allocate per-draw VS/PS constants +
    /// alpha-ref + fog-color bytes without per-draw `Vec::to_vec()` heap
    /// traffic, pointers handed across the channel via `ScratchSlice` stay
    /// valid until this `FrameData` is dropped after the encoder finishes
    /// draining `ops`. Separate from `FrameEncoder::scratch` (which the
    /// encoder thread uses for clear-pass constants etc.) so no two threads
    /// ever write the same arena.
    pub scratch: ScratchArena,
    /// Running per-frame total of bytes `Vec::push` memcpys when `ops` doubles its capacity.
    ///
    /// The API→encoder bridge counterpart to
    /// `PassState::cmd_vec_realloc_bytes`. `push_op` / `push_op_inline`
    /// check `len == capacity` before push and add
    /// `capacity × size_of::<Op>()` here on equality (= the bytes the
    /// imminent realloc memcpys). `peak_ops_count` in `DeviceInner` reserves
    /// the new frame's `ops` at the running peak, so steady-state should land
    /// at 0, non-zero signals a new variant or workload that perturbed the
    /// peak.
    pub op_vec_realloc_bytes: u64,
    pub replay_completion: Option<crate::encoder_packet::ReplayCompletion>,
}

/// The four GPU-fencing values `DeviceInner::stamp_and_swap` puts on a frame.
///
/// One bag rather than four positional `u64`s, because three of them are
/// raw pointers to device-owned atomics and swapping two at a call site
/// would compile silently. Every pointer is an `Arc<AtomicU64>` address
/// that stays valid for the device's lifetime.
pub struct SubmitFence {
    /// Monotonic seq of the frame being handed to the encoder.
    pub submit_seq: u64,
    /// Draw-command-buffer retirement counter.
    pub coherent_seq_ptr: u64,
    /// Upload-command-buffer retirement counter.
    pub upload_coherent_seq_ptr: u64,
    /// Highest seq whose command buffer the GPU aborted.
    pub failed_submit_seq_ptr: u64,
}

/// Parameter bag for `FrameData::new`.
///
/// Grouped so the constructor signature stays under clippy's
/// `too_many_arguments` threshold, pattern borrowed from `DeviceCreateInfo` /
/// `TextureCreateInfo`.
pub struct FrameInit {
    pub device_handle: MetalHandle<MTLDeviceKind>,
    pub record_handle: DeviceRecordHandle,
    pub backbuffer_handle: MetalHandle<MTLTextureKind>,
    /// sRGB twin view of the back buffer, attached under `D3DRS_SRGBWRITEENABLE`.
    pub backbuffer_srgb_handle: MetalHandle<MTLTextureKind>,
    /// Multisampled companion of the back buffer, NULL when it is single-sampled.
    pub backbuffer_msaa_handle: MetalHandle<MTLTextureKind>,
    /// sRGB twin view of that companion, attached under `D3DRS_SRGBWRITEENABLE`.
    pub backbuffer_msaa_srgb_handle: MetalHandle<MTLTextureKind>,
    /// Sample count of the back buffer and the frame's default depth surface.
    pub backbuffer_sample_count: u8,
    pub layer_handle: MetalHandle<CAMetalLayerKind>,
    pub view_handle: MetalHandle<NSViewKind>,
    /// The frame's **logical** back-buffer width, the one D3D9 reports.
    ///
    /// `render_scale` converts it to the rasterized extent; keeping the pair
    /// separate means every consumer states which space it wants.
    pub backbuffer_width: u32,
    /// The frame's **logical** back-buffer height. See `backbuffer_width`.
    pub backbuffer_height: u32,
    pub backbuffer_format: PixelFormat,
    /// Fraction of the logical resolution the back buffer is rasterized at.
    ///
    /// Forwarded to `PassState::reset_frame`, which is the single place the
    /// logical and render coordinate spaces are reconciled.
    pub render_scale: RenderScale,
    /// Whether the back buffer starts each frame undefined, from the swap effect.
    ///
    /// Forwarded to `PassState::reset_frame`: Rule A discards the back buffer
    /// on first use only under `D3DSWAPEFFECT_DISCARD`.
    pub backbuffer_contents: BackbufferContents,
    pub depth_texture: MetalHandle<MTLTextureKind>,
    /// `true` when the frame's default depth attachment is a combined depth+stencil format.
    ///
    /// That format is `Depth32Float_Stencil8`. Drives the clear-quad
    /// pipelines' depth/stencil attachment formats so they match the pass.
    pub depth_has_stencil: bool,
}

/// One VB/IB backing queued for seq-gated destruction.
///
/// Pushed by the API thread on Lock-rename and on VB/IB release; drained
/// into `FrameData` at `present()` and handed to the encoder for final
/// cleanup.
pub struct PendingVbibRetention {
    pub buffer_id: BufferId,
    pub page_box: PageBox,
    pub last_submit_seq: u64,
}

impl FrameData {
    #[must_use]
    pub const fn new(init: &FrameInit) -> Self {
        Self {
            recorder: None,
            replay_completion: None,
            ops: Vec::new(),
            pending_texture_warmups: Vec::new(),
            pending_buffer_warmups: Vec::new(),
            pending_staging_warmups: Vec::new(),
            device_handle: init.device_handle,
            record_handle: init.record_handle,
            backbuffer_handle: init.backbuffer_handle,
            backbuffer_srgb_handle: init.backbuffer_srgb_handle,
            backbuffer_msaa_handle: init.backbuffer_msaa_handle,
            backbuffer_msaa_srgb_handle: init.backbuffer_msaa_srgb_handle,
            backbuffer_sample_count: init.backbuffer_sample_count,
            layer_handle: init.layer_handle,
            view_handle: init.view_handle,
            backbuffer_width: init.backbuffer_width,
            backbuffer_height: init.backbuffer_height,
            backbuffer_format: init.backbuffer_format,
            render_scale: init.render_scale,
            backbuffer_contents: init.backbuffer_contents,
            depth_texture: init.depth_texture,
            flags: if init.depth_has_stencil {
                FrameDataFlags::DEPTH_HAS_STENCIL
            } else {
                FrameDataFlags::empty()
            },
            perf: FramePerfPayload::new(),
            vbib_retentions: Vec::new(),
            submit_seq: 0,
            coherent_seq_ptr: 0,
            upload_coherent_seq_ptr: 0,
            failed_submit_seq_ptr: 0,
            retained_bytes_ptr: 0,
            apply_pacing: None,
            apply_gamma: None,
            scratch: ScratchArena::new(),
            op_vec_realloc_bytes: 0,
        }
    }

    /// Mutable handle to the API-thread bump arena.
    ///
    /// Called by `snapshot_shared` to copy VS/PS constants + alpha-ref +
    /// fog-color bytes once per draw without `Vec::to_vec()`. The returned
    /// arena is cleared on `FrameData` drop (i.e. after the encoder finishes
    /// the frame), so pointers stay valid for the entire op-replay window.
    pub const fn scratch_mut(&mut self) -> &mut ScratchArena {
        &mut self.scratch
    }

    /// Number of ops queued in this frame so far.
    ///
    /// `stamp_and_swap` reads this on the outgoing frame to pre-size the
    /// incoming frame's ops Vec, eliminating the per-frame Vec doubling
    /// burden (which was statistically landing on Draw `push_op` calls after
    /// `Set*ConstRange` ops bumped the per-frame total by ~50%).
    #[must_use]
    pub const fn ops_len(&self) -> usize {
        match &self.recorder {
            Some(recorder) => recorder.len(),
            None => self.ops.len(),
        }
    }

    /// Pre-reserve `count` elements of capacity in the ops Vec.
    ///
    /// Called from `stamp_and_swap` with the previous frame's
    /// `ops_len()` so the new frame fills without any realloc in the
    /// common case where frame-to-frame op count is stable.
    #[cfg(not(windows))]
    pub fn reserve_ops(&mut self, count: usize) {
        self.ops.reserve(count);
    }

    /// Byte layout of the frame's back-buffer texture.
    ///
    /// Read by `GetFrontBufferData` to check its destination against the image
    /// it copies: the back buffer is created `Bgra8Unorm` whatever format the
    /// swap chain was asked for, so this is the layout, not the declared
    /// `D3DFMT_*`.
    #[must_use]
    pub const fn backbuffer_format(&self) -> PixelFormat {
        self.backbuffer_format
    }

    #[must_use]
    pub const fn perf(&self) -> &FramePerfPayload {
        &self.perf
    }

    pub const fn perf_mut(&mut self) -> &mut FramePerfPayload {
        &mut self.perf
    }

    pub const fn set_no_present(&mut self, no_present: bool) {
        // const fn: bitflags `.set()` isn't const, so union/difference (which
        // are) toggle the bit.
        self.flags = if no_present {
            self.flags.union(FrameDataFlags::NO_PRESENT)
        } else {
            self.flags.difference(FrameDataFlags::NO_PRESENT)
        };
    }

    /// Add GPU-capture marks (`GPU_CAPTURE_START` / `GPU_CAPTURE_STOP`) to the frame.
    pub const fn mark_gpu_capture(&mut self, marks: FrameDataFlags) {
        self.flags = self.flags.union(marks);
    }

    /// The GPU-capture marks the frame carries, if any.
    #[must_use]
    pub const fn gpu_capture_marks(&self) -> FrameDataFlags {
        self.flags
            .intersection(FrameDataFlags::GPU_CAPTURE_START.union(FrameDataFlags::GPU_CAPTURE_STOP))
    }

    /// Take the capture marks this frame hands to the frame replacing it.
    ///
    /// `submitted` says whether this frame still reaches the encoder. What
    /// the continuation inherits is cleared here, so exactly one of the two
    /// frames carries each mark.
    pub fn take_carried_capture_marks(&mut self, submitted: bool) -> FrameDataFlags {
        let marks = self.gpu_capture_marks();
        let (start, stop) = crate::present::carried_capture_marks(
            (
                marks.contains(FrameDataFlags::GPU_CAPTURE_START),
                marks.contains(FrameDataFlags::GPU_CAPTURE_STOP),
            ),
            submitted,
        );
        let mut carried = FrameDataFlags::empty();
        carried.set(FrameDataFlags::GPU_CAPTURE_START, start);
        carried.set(FrameDataFlags::GPU_CAPTURE_STOP, stop);
        self.flags = self.flags.difference(carried);
        carried
    }

    pub const fn set_submit_fence(&mut self, fence: &SubmitFence) {
        self.submit_seq = fence.submit_seq;
        self.coherent_seq_ptr = fence.coherent_seq_ptr;
        self.upload_coherent_seq_ptr = fence.upload_coherent_seq_ptr;
        self.failed_submit_seq_ptr = fence.failed_submit_seq_ptr;
    }

    pub const fn set_retained_bytes_ptr(&mut self, ptr: u64) {
        self.retained_bytes_ptr = ptr;
    }

    /// Record dirty API state directly into the data-only frame protocol.
    pub fn record_snapshot_delta(&mut self, delta: &crate::encoder_draw::SnapshotDelta<'_>) {
        let _ = self
            .recorder
            .get_or_insert_with(crate::encoder_packet::FrameRecorder::new)
            .record_snapshot_delta(&mut self.scratch, delta);
    }

    pub fn push_op(&mut self, op: Op) {
        let _ = self.try_push_op(op);
    }

    /// Record an operation and report a latched wire capture error.
    ///
    /// # Errors
    /// Returns allocation, size or malformed capture errors on the PE recording path.
    pub fn try_push_op(&mut self, op: Op) -> Result<(), mtld3d_shared::encoder_wire::WireError> {
        #[cfg(windows)]
        {
            self.recorder
                .get_or_insert_with(crate::encoder_packet::FrameRecorder::new)
                .try_record(&mut self.scratch, op)
        }
        #[cfg(not(windows))]
        {
            self.account_op_vec_realloc();
            self.ops.push(op);
            Ok(())
        }
    }

    /// Serialize a borrowed draw directly into the PE frame capture.
    #[cfg(windows)]
    pub fn record_draw(&mut self, draw: &crate::draw_data::DrawOp) {
        let _ = self
            .recorder
            .get_or_insert_with(crate::encoder_packet::FrameRecorder::new)
            .record_draw(&mut self.scratch, draw);
    }

    /// Retain an owned draw for native replay.
    #[cfg(not(windows))]
    pub fn record_draw(&mut self, draw: crate::draw_data::DrawOp) {
        self.push_op(Op::Draw(draw));
    }

    /// Record a constant delta directly on PE, or retain its operation on Unix.
    pub fn record_vs_constants(
        &mut self,
        start_row: u16,
        rows: u16,
        data: crate::draw_data::ScratchSlice,
    ) {
        #[cfg(windows)]
        {
            let _ = self
                .recorder
                .get_or_insert_with(crate::encoder_packet::FrameRecorder::new)
                .record_vs_constants(&mut self.scratch, start_row, rows, data);
        }
        #[cfg(not(windows))]
        self.push_op(Op::SetVsConstRange {
            start_row,
            rows,
            data,
        });
    }

    /// Record a constant delta directly on PE, or retain its operation on Unix.
    pub fn record_ps_constants(
        &mut self,
        start_row: u16,
        rows: u16,
        data: crate::draw_data::ScratchSlice,
    ) {
        #[cfg(windows)]
        {
            let _ = self
                .recorder
                .get_or_insert_with(crate::encoder_packet::FrameRecorder::new)
                .record_ps_constants(&mut self.scratch, start_row, rows, data);
        }
        #[cfg(not(windows))]
        self.push_op(Op::SetPsConstRange {
            start_row,
            rows,
            data,
        });
    }

    /// Record a constant delta directly on PE, or retain its operation on Unix.
    pub fn record_ff_vs_constants(
        &mut self,
        start_row: u16,
        rows: u16,
        data: crate::draw_data::ScratchSlice,
    ) {
        #[cfg(windows)]
        {
            let _ = self
                .recorder
                .get_or_insert_with(crate::encoder_packet::FrameRecorder::new)
                .record_ff_vs_constants(&mut self.scratch, start_row, rows, data);
        }
        #[cfg(not(windows))]
        self.push_op(Op::SetFfVsConstRange {
            start_row,
            rows,
            data,
        });
    }

    #[must_use]
    pub fn recording_error(&self) -> Option<mtld3d_shared::encoder_wire::WireError> {
        self.recorder
            .as_ref()
            .and_then(crate::encoder_packet::FrameRecorder::recording_error)
    }

    /// Push an `Op` variant directly.
    ///
    /// Used by the hot draw path (`emit_snapshot_deltas` + `Op::Draw`) so it
    /// can emit inline state-delta + draw variants without a payload allocation.
    pub fn push_op_inline(&mut self, op: Op) {
        self.push_op(op);
    }

    /// Add the old capacity's bytes to the per-frame counter before `Vec::push` reallocs.
    ///
    /// The imminent push is the one that trips `Vec::push`'s
    /// double-and-memcpy. Hot-path: a single `len == capacity`
    /// compare when no realloc fires. Mirrors the `emit_command`
    /// pattern in `PassState`.
    #[cfg(not(windows))]
    #[inline]
    const fn account_op_vec_realloc(&mut self) {
        if self.ops.len() == self.ops.capacity() {
            let bytes = (self.ops.capacity() as u64).saturating_mul(size_of::<Op>() as u64);
            self.op_vec_realloc_bytes = self.op_vec_realloc_bytes.saturating_add(bytes);
        }
    }

    /// Resident `Vec<Op>` capacity in bytes.
    ///
    /// Read by `stamp_and_swap` to seed the outgoing frame's
    /// `FramePerfPayload` so the `op_vec size` row in the per-frame allocator
    /// footprint reflects steady-state footprint paired with the realloc
    /// churn.
    #[must_use]
    pub const fn op_vec_capacity_bytes(&self) -> u64 {
        (self.ops.capacity() as u64).saturating_mul(size_of::<Op>() as u64)
    }

    /// Put a Reset's queued `PresentationInterval` change on this frame.
    ///
    /// Called from `stamp_and_swap` for the frame being handed to the
    /// encoder. `None` is the normal case and leaves the frame carrying
    /// nothing.
    pub const fn set_apply_pacing(&mut self, pacing: Option<LayerPacing>) {
        self.apply_pacing = pacing;
    }

    /// Put a queued gamma-ramp change on this frame.
    ///
    /// Called from `stamp_and_swap` for the frame being handed to the
    /// encoder, the same way the queued pacing is.
    pub fn set_apply_gamma(&mut self, change: Option<crate::gamma::Change>) {
        self.apply_gamma = change;
    }

    /// Drain the per-frame `Vec<Op>` realloc-byte counter into the caller and zero it.
    ///
    /// Called once per frame from `stamp_and_swap` so the outgoing frame
    /// ships its realloc total to the encoder via `FramePerfPayload`.
    pub const fn take_op_vec_realloc_bytes(&mut self) -> u64 {
        core::mem::replace(&mut self.op_vec_realloc_bytes, 0)
    }

    /// Queue a texture for eager `MTLTexture` creation at the head of the next `run_frame`.
    ///
    /// Called from `IDirect3DDevice9::CreateTexture` on the API thread.
    pub fn push_texture_warmup(&mut self, info: TextureInfo) {
        self.pending_texture_warmups.push(info);
    }

    /// Queue a VB/IB for eager `MTLBuffer` wrap at the head of the next `run_frame`.
    ///
    /// Called from `CreateVertexBuffer` / `CreateIndexBuffer` on the API
    /// thread.
    pub fn push_buffer_warmup(&mut self, entry: VbibWarmupEntry) {
        self.pending_buffer_warmups.push(entry);
    }

    /// Queue a texture-staging `MTLBuffer` wrap.
    ///
    /// Called per mip from `IDirect3DDevice9::CreateTexture` on the API
    /// thread for textures that go through the blit-upload path.
    pub fn push_staging_warmup(&mut self, entry: StagingWarmupEntry) {
        self.pending_staging_warmups.push(entry);
    }

    pub fn set_vbib_retentions(&mut self, retentions: Vec<PendingVbibRetention>) {
        self.vbib_retentions = retentions;
    }
}

/// Capture a typed operation without a transient allocation on the API runtime.
///
/// PE immediately serializes this value. Native decoding retains boxed payloads to keep its
/// queued operation enum compact.
#[cfg(windows)]
#[must_use]
pub const fn capture_op<T>(value: T) -> T {
    value
}

/// Box a decoded operation for the native encoder queue.
#[cfg(not(windows))]
#[must_use]
pub fn capture_op<T>(value: T) -> Box<T> {
    Box::new(value)
}
