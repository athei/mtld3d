//! Fixed metadata and immutable typed arrays retained by the publishing runtime.

use mtld3d_shared::{
    encoder_wire::WireError,
    frame_metadata::{ArrayRef, FrameMetadata},
};

use super::FrameRecorder;
use crate::{
    encoder_data::{FrameData, PendingVbibRetention, StagingWarmupEntry, VbibWarmupEntry},
    encoder_records::TextureRecord,
    guest_pages::{GuestPageDescriptor, GuestPageLease},
    perf::FramePerfPayload,
};

#[repr(C, align(8))]
pub struct BufferWarmupRecord {
    pub buffer_id: u64,
    pub backing_ptr: u64,
    pub backing_len: u64,
    pub backing_generation: u64,
    pub map_mode: u32,
    pub reserved: u32,
}

#[repr(C, align(8))]
pub struct StagingWarmupRecord {
    pub texture_id: u64,
    pub backing_ptr: u64,
    pub backing_len: u64,
    pub page: GuestPageDescriptor,
    pub level: u32,
    pub reserved: u32,
}

#[repr(C, align(8))]
pub struct VbibRetentionRecord {
    pub buffer_id: u64,
    pub last_submit_seq: u64,
    pub page: GuestPageDescriptor,
}

#[repr(C, align(8))]
pub struct LayerPacingRecord {
    pub layer: u64,
    pub display_sync: u32,
    pub max_fps: u32,
}
#[repr(C, align(8))]
pub struct GammaRecord {
    pub layer: u64,
    pub entries_ptr: u64,
    pub entries_len: u32,
    pub mode: u32,
}

// SAFETY: these canonical records contain only fixed integers and descriptors, and all offsets
// and sizes are asserted below. No field has invalid bit patterns or owns a Rust allocation.
unsafe impl crate::encoder_records::CommandRecord for BufferWarmupRecord {}
// SAFETY: the descriptor is seven u64 fields and all surrounding padding is explicit.
unsafe impl crate::encoder_records::CommandRecord for StagingWarmupRecord {}
// SAFETY: every byte belongs to initialized u64 fields, including the page descriptor.
unsafe impl crate::encoder_records::CommandRecord for VbibRetentionRecord {}
// SAFETY: this 16-byte integer record has no padding or invalid bit patterns.
unsafe impl crate::encoder_records::CommandRecord for LayerPacingRecord {}
// SAFETY: this 24-byte integer record has no padding or invalid bit patterns.
unsafe impl crate::encoder_records::CommandRecord for GammaRecord {}

/// The one fixed header lives in the existing frame arena.
pub struct MetadataStorage {
    header: u64,
}
impl MetadataStorage {
    #[must_use]
    pub const fn new() -> Self {
        Self { header: 0 }
    }
    pub const fn clear(&mut self) {
        self.header = 0;
    }
    /// Borrow the initialized header from its retained frame arena.
    ///
    /// # Safety
    ///
    /// The frame arena passed to seal remains alive and has not been cleared or reused for
    /// the returned borrow. The packet owner clears this token before returning that arena.
    #[must_use]
    pub const unsafe fn as_bytes(&self) -> &[u8] {
        if self.header == 0 {
            return &[];
        }
        // SAFETY: seal initializes every byte in the retained scratch allocation.
        unsafe { core::slice::from_raw_parts(self.header as *const u8, size_of::<FrameMetadata>()) }
    }
    /// Initialize one header in the existing frame arena without copying payload arrays.
    ///
    /// # Errors
    ///
    /// The current fixed header has no recoverable construction errors.
    pub fn seal(
        &mut self,
        frame: &mut FrameData,
        _recorder: &FrameRecorder,
    ) -> Result<(), WireError> {
        #[cfg(perf_tracking)]
        let perf = ArrayRef {
            address: core::ptr::from_ref(frame.perf_mut()) as u64,
            count: 1,
            reserved: 0,
        };
        #[cfg(not(perf_tracking))]
        let perf = ArrayRef::default();
        let value = FrameMetadata {
            backbuffer_handle: frame.backbuffer_handle.raw(),
            backbuffer_srgb_handle: frame.backbuffer_srgb_handle.raw(),
            backbuffer_msaa_handle: frame.backbuffer_msaa_handle.raw(),
            backbuffer_msaa_srgb_handle: frame.backbuffer_msaa_srgb_handle.raw(),
            layer_handle: frame.layer_handle.raw(),
            view_handle: frame.view_handle.raw(),
            depth_texture: frame.depth_texture.raw(),
            submit_seq: frame.submit_seq,
            perf,
            flags: u32::from(frame.flags.bits()),
            backbuffer_width: frame.backbuffer_width,
            backbuffer_height: frame.backbuffer_height,
            backbuffer_format: frame.backbuffer_format as u32,
            render_scale_percent: frame.render_scale.percent(),
            backbuffer_sample_count: u32::from(frame.backbuffer_sample_count),
            backbuffer_contents: frame.backbuffer_contents as u32,
            ..FrameMetadata::default()
        };
        let target = frame.scratch.alloc_uninit::<FrameMetadata>();
        // SAFETY: every scalar and explicit reserved field is initialized, and the arena retains it.
        unsafe {
            target.write(value);
        }
        self.header = target as u64;
        Ok(())
    }
}
impl Default for MetadataStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameRecorder {
    fn metadata_command<T: crate::encoder_records::CommandRecord>(
        &mut self,
        scratch: &mut crate::scratch::ScratchArena,
        tag: mtld3d_shared::encoder_protocol::EncoderOpcode,
        value: T,
    ) {
        if self.error.is_some() {
            return;
        }
        let result =
            self.slab
                .push_fixed_record(scratch, tag.into(), 0, size_of::<T>(), |destination| {
                    crate::encoder_records::write(destination, value)
                });
        match result {
            Ok(()) => self.count += 1,
            Err(error) => self.error = Some(error),
        }
    }
    pub fn capture_texture_warmup(
        &mut self,
        scratch: &mut crate::scratch::ScratchArena,
        info: &crate::encoder_data::TextureInfo,
    ) {
        self.metadata_command(
            scratch,
            mtld3d_shared::encoder_protocol::EncoderOpcode::WarmupTexture,
            TextureRecord::capture(info),
        );
    }
    pub fn capture_buffer_warmup(
        &mut self,
        scratch: &mut crate::scratch::ScratchArena,
        entry: VbibWarmupEntry,
    ) {
        self.metadata_command(
            scratch,
            mtld3d_shared::encoder_protocol::EncoderOpcode::WarmupBuffer,
            BufferWarmupRecord {
                buffer_id: entry.buffer_id.raw(),
                backing_ptr: entry.backing_ptr,
                backing_len: entry.backing_len,
                backing_generation: entry.backing_generation,
                map_mode: entry.map_mode as u32,
                reserved: 0,
            },
        );
    }
    pub fn capture_staging_warmup(
        &mut self,
        scratch: &mut crate::scratch::ScratchArena,
        entry: StagingWarmupEntry,
    ) {
        let lease = GuestPageLease::for_shared_pooled(entry.keepalive, &self.completion_pool);
        let page = lease.descriptor();
        self.pages.push(lease);
        self.metadata_command(
            scratch,
            mtld3d_shared::encoder_protocol::EncoderOpcode::WarmupStaging,
            StagingWarmupRecord {
                texture_id: entry.texture_id.raw(),
                backing_ptr: entry.backing_ptr,
                backing_len: entry.backing_len,
                page,
                level: entry.level,
                reserved: 0,
            },
        );
    }
    pub fn capture_vbib_retention(
        &mut self,
        scratch: &mut crate::scratch::ScratchArena,
        entry: PendingVbibRetention,
    ) {
        let lease = GuestPageLease::for_recyclable_pooled(
            entry.page_box,
            &self.completion_pool,
            self.pagebox_pool,
        );
        let page = lease.descriptor();
        self.pages.push(lease);
        self.metadata_command(
            scratch,
            mtld3d_shared::encoder_protocol::EncoderOpcode::RetainVbib,
            VbibRetentionRecord {
                buffer_id: entry.buffer_id.raw(),
                last_submit_seq: entry.last_submit_seq,
                page,
            },
        );
    }
    pub fn capture_pacing(
        &mut self,
        scratch: &mut crate::scratch::ScratchArena,
        layer: u64,
        pacing: crate::present::LayerPacing,
    ) {
        self.metadata_command(
            scratch,
            mtld3d_shared::encoder_protocol::EncoderOpcode::SetLayerPacing,
            LayerPacingRecord {
                layer,
                display_sync: u32::from(pacing.display_sync),
                max_fps: pacing.max_fps,
            },
        );
    }
    /// Capture a gamma change and retain its original LUT until replay finishes.
    ///
    /// # Panics
    ///
    /// Panics if the fixed LUT extent cannot fit the command's u32 count.
    pub fn capture_gamma(
        &mut self,
        scratch: &mut crate::scratch::ScratchArena,
        layer: u64,
        change: crate::gamma::Change,
    ) {
        let (mode, entries_ptr, entries_len) = match change {
            crate::gamma::Change::Remove => (1, 0, 0),
            crate::gamma::Change::Apply(table) => {
                let table = super::GammaTableOwner { table };
                let address = table.table.as_ptr() as u64;
                let length = u32::try_from(table.table.len()).expect("gamma LUT has fixed extent");
                self.gamma_tables.push(table);
                (2, address, length)
            }
        };
        self.metadata_command(
            scratch,
            mtld3d_shared::encoder_protocol::EncoderOpcode::SetGamma,
            GammaRecord {
                layer,
                entries_ptr,
                entries_len,
                mode,
            },
        );
    }
}

/// A validated borrow, bounded by the immutable packet's replay lifetime.
pub struct FrameView<'a> {
    header: &'a FrameMetadata,
}

impl<'a> FrameView<'a> {
    /// Borrow one matched producer's fixed header and optional telemetry.
    ///
    /// # Safety
    ///
    /// Header and every referenced initialized typed array are authentic, immutable and retained
    /// by the matched producer for this borrow. Numeric validation does not prove pointer validity.
    ///
    /// # Errors
    ///
    /// Rejects invalid scalar values, alignment or array extents.
    pub unsafe fn from_bytes(bytes: &'a [u8]) -> Result<Self, WireError> {
        if bytes.len() != size_of::<FrameMetadata>() || !(bytes.as_ptr() as usize).is_multiple_of(8)
        {
            return Err(WireError::InvalidValue);
        }
        // SAFETY: the contract retains initialized header bytes; this conversion checks
        // alignment without constructing a pointer with stronger alignment by casting.
        let (prefix, headers, suffix) = unsafe { bytes.align_to::<FrameMetadata>() };
        if !prefix.is_empty() || !suffix.is_empty() {
            return Err(WireError::InvalidValue);
        }
        let header = headers.first().ok_or(WireError::InvalidValue)?;
        if header.reserved != 0 {
            return Err(WireError::InvalidValue);
        }
        if mtld3d_shared::mtl::PixelFormat::from_repr(header.backbuffer_format).is_none()
            || header.backbuffer_contents > 1
            || header.backbuffer_sample_count > 255
            || header.render_scale_percent == 0
            || header.render_scale_percent > 100
            || u8::try_from(header.flags)
                .ok()
                .and_then(crate::encoder_data::FrameDataFlags::from_bits)
                .is_none()
        {
            return Err(WireError::InvalidValue);
        }
        let view = Self { header };
        view.perf()?;
        Ok(view)
    }
    /// Reborrow a header already checked for this packet.
    ///
    /// # Safety
    ///
    /// This exact immutable header and all its arrays passed `from_bytes` and remain retained.
    #[must_use]
    pub const unsafe fn from_validated_header(header: &'a FrameMetadata) -> Self {
        Self { header }
    }

    #[must_use]
    pub const fn backbuffer_handle(
        &self,
    ) -> mtld3d_shared::MetalHandle<mtld3d_shared::mtl_handle::MTLTextureKind> {
        // SAFETY: the matched producer supplies the live handle with its declared object kind.
        unsafe { mtld3d_shared::MetalHandle::new(self.header.backbuffer_handle) }
    }
    #[must_use]
    pub const fn backbuffer_srgb_handle(
        &self,
    ) -> mtld3d_shared::MetalHandle<mtld3d_shared::mtl_handle::MTLTextureKind> {
        // SAFETY: the matched producer supplies the live handle with its declared object kind.
        unsafe { mtld3d_shared::MetalHandle::new(self.header.backbuffer_srgb_handle) }
    }
    #[must_use]
    pub const fn backbuffer_msaa_handle(
        &self,
    ) -> mtld3d_shared::MetalHandle<mtld3d_shared::mtl_handle::MTLTextureKind> {
        // SAFETY: the matched producer supplies the live handle with its declared object kind.
        unsafe { mtld3d_shared::MetalHandle::new(self.header.backbuffer_msaa_handle) }
    }
    #[must_use]
    pub const fn backbuffer_msaa_srgb_handle(
        &self,
    ) -> mtld3d_shared::MetalHandle<mtld3d_shared::mtl_handle::MTLTextureKind> {
        // SAFETY: the matched producer supplies the live handle with its declared object kind.
        unsafe { mtld3d_shared::MetalHandle::new(self.header.backbuffer_msaa_srgb_handle) }
    }
    #[must_use]
    pub const fn layer_handle(
        &self,
    ) -> mtld3d_shared::MetalHandle<mtld3d_shared::mtl_handle::CAMetalLayerKind> {
        // SAFETY: the matched producer supplies the live handle with its declared object kind.
        unsafe { mtld3d_shared::MetalHandle::new(self.header.layer_handle) }
    }
    #[must_use]
    pub const fn view_handle(
        &self,
    ) -> mtld3d_shared::MetalHandle<mtld3d_shared::mtl_handle::NSViewKind> {
        // SAFETY: the matched producer supplies the live handle with its declared object kind.
        unsafe { mtld3d_shared::MetalHandle::new(self.header.view_handle) }
    }
    #[must_use]
    pub const fn depth_texture(
        &self,
    ) -> mtld3d_shared::MetalHandle<mtld3d_shared::mtl_handle::MTLTextureKind> {
        // SAFETY: the matched producer supplies the live handle with its declared object kind.
        unsafe { mtld3d_shared::MetalHandle::new(self.header.depth_texture) }
    }

    /// Read the corresponding validated frame value.
    ///
    /// # Panics
    ///
    /// Panics if an unsafe caller supplied a header that did not pass validation.
    #[must_use]
    pub const fn backbuffer_format(&self) -> mtld3d_shared::mtl::PixelFormat {
        mtld3d_shared::mtl::PixelFormat::from_repr(self.header.backbuffer_format)
            .expect("validated pixel format")
    }
    #[must_use]
    pub const fn render_scale(&self) -> crate::render_scale::RenderScale {
        crate::render_scale::RenderScale::from_percent(self.header.render_scale_percent)
    }
    #[must_use]
    pub const fn backbuffer_contents(&self) -> crate::passes::BackbufferContents {
        if self.header.backbuffer_contents == 0 {
            crate::passes::BackbufferContents::Undefined
        } else {
            crate::passes::BackbufferContents::Preserved
        }
    }
    /// Read the corresponding validated frame value.
    ///
    /// # Panics
    ///
    /// Panics if an unsafe caller supplied a header that did not pass validation.
    #[must_use]
    pub fn sample_count(&self) -> u8 {
        u8::try_from(self.header.backbuffer_sample_count).expect("validated sample count")
    }
    /// Read the corresponding validated frame value.
    ///
    /// # Panics
    ///
    /// Panics if an unsafe caller supplied a header that did not pass validation.
    #[must_use]
    pub fn flags(&self) -> crate::encoder_data::FrameDataFlags {
        crate::encoder_data::FrameDataFlags::from_bits_retain(
            u8::try_from(self.header.flags).expect("validated flags"),
        )
    }
    #[must_use]
    pub const fn header(&self) -> &'a FrameMetadata {
        self.header
    }
    /// Borrow the original source-clock telemetry retained by this frame.
    ///
    /// # Errors
    ///
    /// Rejects invalid telemetry alignment, count or extent.
    #[cfg(perf_tracking)]
    pub fn perf(&self) -> Result<Option<&'a FramePerfPayload>, WireError> {
        let source = &self.header.perf;
        let address = usize::try_from(source.address).map_err(|_| WireError::TooLarge)?;
        if source.reserved != 0
            || source.count != 1
            || address == 0
            || !address.is_multiple_of(align_of::<FramePerfPayload>())
            || address.checked_add(size_of::<FramePerfPayload>()).is_none()
        {
            return Err(WireError::InvalidValue);
        }
        // SAFETY: from_bytes grants this initialized retained canonical payload for 'a;
        // the fixed count, alignment and extent were checked before constructing its borrow.
        Ok(Some(unsafe { &*(address as *const FramePerfPayload) }))
    }

    /// PERF-disabled frames have no telemetry attachment.
    ///
    /// # Errors
    ///
    /// Rejects a nonempty attachment in this matched PERF-disabled build.
    #[cfg(not(perf_tracking))]
    pub const fn perf(&self) -> Result<Option<&'a FramePerfPayload>, WireError> {
        if self.header.perf.count != 0
            || self.header.perf.address != 0
            || self.header.perf.reserved != 0
        {
            return Err(WireError::InvalidValue);
        }
        Ok(None)
    }
}

const _: () = {
    assert!(size_of::<BufferWarmupRecord>() == 40);
    assert!(align_of::<BufferWarmupRecord>() == 8);
    assert!(core::mem::offset_of!(BufferWarmupRecord, buffer_id) == 0);
    assert!(core::mem::offset_of!(BufferWarmupRecord, backing_ptr) == 8);
    assert!(core::mem::offset_of!(BufferWarmupRecord, backing_len) == 16);
    assert!(core::mem::offset_of!(BufferWarmupRecord, backing_generation) == 24);
    assert!(core::mem::offset_of!(BufferWarmupRecord, map_mode) == 32);
    assert!(core::mem::offset_of!(BufferWarmupRecord, reserved) == 36);
    assert!(size_of::<StagingWarmupRecord>() == 88);
    assert!(align_of::<StagingWarmupRecord>() == 8);
    assert!(core::mem::offset_of!(StagingWarmupRecord, texture_id) == 0);
    assert!(core::mem::offset_of!(StagingWarmupRecord, backing_ptr) == 8);
    assert!(core::mem::offset_of!(StagingWarmupRecord, backing_len) == 16);
    assert!(core::mem::offset_of!(StagingWarmupRecord, page) == 24);
    assert!(core::mem::offset_of!(StagingWarmupRecord, level) == 80);
    assert!(core::mem::offset_of!(StagingWarmupRecord, reserved) == 84);
    assert!(size_of::<VbibRetentionRecord>() == 72);
    assert!(align_of::<VbibRetentionRecord>() == 8);
    assert!(core::mem::offset_of!(VbibRetentionRecord, buffer_id) == 0);
    assert!(core::mem::offset_of!(VbibRetentionRecord, last_submit_seq) == 8);
    assert!(core::mem::offset_of!(VbibRetentionRecord, page) == 16);
};

const _: () = {
    assert!(size_of::<LayerPacingRecord>() == 16);
    assert!(align_of::<LayerPacingRecord>() == 8);
    assert!(core::mem::offset_of!(LayerPacingRecord, layer) == 0);
    assert!(core::mem::offset_of!(LayerPacingRecord, display_sync) == 8);
    assert!(core::mem::offset_of!(LayerPacingRecord, max_fps) == 12);
    assert!(size_of::<GammaRecord>() == 24);
    assert!(align_of::<GammaRecord>() == 8);
    assert!(core::mem::offset_of!(GammaRecord, layer) == 0);
    assert!(core::mem::offset_of!(GammaRecord, entries_ptr) == 8);
    assert!(core::mem::offset_of!(GammaRecord, entries_len) == 16);
    assert!(core::mem::offset_of!(GammaRecord, mode) == 20);
};

#[cfg(test)]
mod tests;
