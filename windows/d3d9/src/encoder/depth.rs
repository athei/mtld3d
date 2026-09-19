//! Dynamic depth transfers, separated from ordinary color uploads.

use mtld3d_core::{
    depth_texture::{PackedDepth, PlaneLayout},
    page_box::{PageBox, PageBoxRead},
    storage_policy::buffer_storage_mode,
};
use mtld3d_shared::{
    BlitCommand, BlitCommandType, BufferCreateDesc, CopyBufferToTextureInfo, MetalHandle,
    mtl::{BufferKind, DestroyKind},
    mtl_handle::MTLBufferKind,
};

use super::{
    FrameEncoder, FrameEncoderFlags, PendingResourceRetention, TextureUploadJob,
    destroy_resources_bulk,
};

impl FrameEncoder {
    /// Convert one packed rectangle and enqueue both native planes atomically.
    pub fn run_depth_upload_blit(
        &mut self,
        job: &TextureUploadJob,
        texture: u64,
        format: &PackedDepth,
    ) -> bool {
        let Some(layout) = PlaneLayout::new(
            job.region_w,
            job.region_h,
            self.gpu_caps.min_linear_texture_align,
        ) else {
            log::error!(target: super::LOG_TARGET, "depth upload: plane geometry overflow");
            return false;
        };
        let Some(offset) = (job.origin_y as usize)
            .checked_mul(job.src_pitch as usize)
            .and_then(|n| n.checked_add(job.origin_x as usize * format.bytes_per_pixel()))
        else {
            log::error!(target: super::LOG_TARGET, "depth upload: packed offset overflow");
            return false;
        };
        let Some(source) = job.staging.backing().as_slice().get(offset..) else {
            log::error!(target: super::LOG_TARGET, "depth upload: packed offset outside staging");
            return false;
        };
        let mut depth = PageBox::new_zeroed(layout.depth_pitch * layout.height);
        let mut stencil = format
            .has_stencil()
            .then(|| PageBox::new_zeroed(layout.stencil_pitch * layout.height));
        if !layout.unpack(
            format,
            source,
            job.src_pitch as usize,
            depth.as_mut_slice(),
            stencil.as_mut().map_or(&mut [], PageBox::as_mut_slice),
        ) {
            log::error!(target: super::LOG_TARGET, "depth upload: packed source or native plane is too short");
            return false;
        }
        let mut pages = vec![(
            depth,
            layout.depth_pitch,
            BlitCommandType::CopyBufferToDepth,
        )];
        if let Some(stencil) = stencil {
            pages.push((
                stencil,
                layout.stencil_pitch,
                BlitCommandType::CopyBufferToStencil,
            ));
        }
        let descs: Vec<_> = pages
            .iter()
            .map(|(page, _, _)| BufferCreateDesc {
                backing_ptr: page.as_ptr() as u64,
                length: page.len() as u64,
                id: job.info.texture_id.raw(),
                storage_mode: buffer_storage_mode(self.gpu_caps.unified_memory),
                kind: BufferKind::Repack,
            })
            .collect();
        let mut handles = vec![MetalHandle::<MTLBufferKind>::NULL; pages.len()];
        let status = self.batch_create_buffers(&descs, &mut handles);
        if status != 0 || handles.iter().any(|h| h.is_null()) {
            let live: Vec<_> = handles
                .into_iter()
                .filter(|h| !h.is_null())
                .map(MetalHandle::raw)
                .collect();
            destroy_resources_bulk(DestroyKind::Buffer, &live);
            log::error!(target: super::LOG_TARGET, "depth upload: native plane allocation failed status={status:#x}");
            return false;
        }
        let ordered = self.pass_state.texture_written_by_blit_this_frame(
            // SAFETY: texture is the live destination resolved by the encoder cache.
            unsafe { MetalHandle::new(texture) },
        );
        if ordered {
            self.end_current_pass("dynamic_depth_upload");
        }
        for ((page, pitch, kind), handle) in pages.into_iter().zip(handles) {
            self.enqueue_notify_buffer_did_modify_range(handle.raw(), 0, page.len() as u64);
            let mut command = BlitCommand::copy_buffer_to_texture(&CopyBufferToTextureInfo {
                buffer_handle: handle.raw(),
                buffer_offset: 0,
                bytes_per_row: u32::try_from(pitch).expect("depth plane pitch fits u32"),
                texture_handle: texture,
                destination_slice: 0,
                mip_level: job.level,
                origin_x: job.origin_x,
                origin_y: job.origin_y,
                region_w: job.region_w,
                region_h: job.region_h,
                depth: 1,
                bytes_per_image: 0,
            });
            command.cmd = kind as u32;
            if ordered {
                self.pass_state.push_pending_leading_blit(command);
            } else {
                self.frame_blit_commands.push(command);
            }
            self.perf.bump_vbib_retained_add(page.len());
            self.add_retained_bytes(page.len());
            self.pending_resource_retention
                .push_back(PendingResourceRetention {
                    kind: DestroyKind::Buffer,
                    handle: handle.raw(),
                    page_box: Some(page),
                    staging_arc: None,
                    seq: self.current_submit_seq,
                    from_texture: true,
                });
        }
        if !ordered {
            self.flags.insert(FrameEncoderFlags::BLIT_CMDS_NEED_ENCODER);
        }
        self.current_blit_retention
            .push(PageBoxRead::new(std::sync::Arc::clone(
                job.staging.backing(),
            )));
        self.perf.bump_texture_blit_upload();
        true
    }
}

/// Four concrete kernels, owned by this device and initialized only when needed.
#[derive(Default)]
pub struct TransferState {
    pipelines: [MetalHandle<mtld3d_shared::mtl_handle::MTLComputePipelineStateKind>; 4],
}

impl TransferState {
    pub fn destroy(&mut self) {
        let handles: Vec<_> = self
            .pipelines
            .iter()
            .filter(|h| !h.is_null())
            .map(|h| h.raw())
            .collect();
        destroy_resources_bulk(DestroyKind::ComputePipeline, &handles);
        self.pipelines.fill(MetalHandle::NULL);
    }
}

impl FrameEncoder {
    /// Queue a common-plane transfer without making CPU staging authoritative.
    pub fn resolve_dynamic_depth(&mut self, destination: u64, info: &super::TextureInfo) {
        use mtld3d_shared::{
            CreateDepthTransferPipelineParams,
            mtl::{DepthTransferKind, PixelFormat},
            mtl_handle::MTLTextureKind,
        };
        let source = self.pass_state.current_depth_texture();
        if source.is_null() || destination == 0 {
            mtld3d_shared::log_once_warn!(target: super::LOG_TARGET, "dynamic depth transfer: missing endpoint");
            return;
        }
        let (width, height) = self.pass_state.current_depth_size();
        let source_format = if self.pass_state.current_depth_has_stencil() {
            PixelFormat::Depth32FloatStencil8
        } else {
            PixelFormat::Depth32Float
        };
        let samples = self.pass_state.current_depth_sample_count();
        let source_level = self.pass_state.current_depth_level();
        // SAFETY: the destination was resolved from this encoder's live texture cache.
        let destination = unsafe { MetalHandle::<MTLTextureKind>::new(destination) };
        self.pass_state.flush_pending_clears();
        self.pass_state.note_texture_read(source);
        self.end_current_pass("dynamic_depth_transfer");
        self.pass_state.note_ordered_texture_write(destination);
        if samples == 1
            && width == info.width
            && height == info.height
            && source_format == info.pixel_format
        {
            let mut command = BlitCommand::copy_texture_to_texture_full_mip(
                source.raw(),
                destination.raw(),
                0,
                width,
                height,
            );
            command.mip_level = source_level;
            self.pass_state.push_pending_leading_blit(command);
            return;
        }
        let stencil = source_format == PixelFormat::Depth32FloatStencil8
            && info.pixel_format == PixelFormat::Depth32FloatStencil8;
        let kind = match (samples > 1, stencil) {
            (false, false) => DepthTransferKind::Depth,
            (false, true) => DepthTransferKind::DepthStencil,
            (true, false) => DepthTransferKind::MultisampleDepth,
            (true, true) => DepthTransferKind::MultisampleDepthStencil,
        };
        let slot = &mut self.depth_transfer.pipelines[kind as usize];
        if slot.is_null() && (samples > 1 || width != info.width || height != info.height) {
            let mut params = CreateDepthTransferPipelineParams {
                device_handle: self.device_handle,
                pipeline_handle: MetalHandle::NULL,
                kind,
                pad: 0,
            };
            if crate::unix_call::unix_call(&mut params) != 0 || params.pipeline_handle.is_null() {
                log::error!(target: super::LOG_TARGET, "dynamic depth transfer: pipeline creation failed");
                return;
            }
            *slot = params.pipeline_handle;
        }
        let mut command = BlitCommand::copy_texture_to_texture_full_mip(
            source.raw(),
            destination.raw(),
            0,
            info.width,
            info.height,
        );
        command.mip_level = source_level;
        command.cmd = BlitCommandType::TransferDepth as u32;
        command.src_offset = slot.raw();
        self.pass_state.push_pending_leading_blit(command);
    }
}
