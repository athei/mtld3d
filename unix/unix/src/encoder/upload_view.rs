//! Borrowed upload inputs for direct command execution and owned recovery replay.

use std::sync::Arc;

use mtld3d_core::{
    dirty_rect::DirtyRect,
    encoder_data::{TextureInfo, TextureUploadJob},
    encoder_records::{TextureRecord, TextureUploadRecord},
    ids::TextureId,
    page_box::PageBoxRead,
    upload_redirty::{EmittedUpload, RedirtyQueue, RedirtySubresource},
};
use mtld3d_shared::{
    encoder_wire::WireError,
    mtl::{PixelFormat, Swizzle, TextureCreateFlags, TextureUsage},
};

pub(super) enum TextureView<'a> {
    Owned(&'a TextureInfo),
    Record(&'a TextureRecord),
}
impl TextureView<'_> {
    pub(super) fn validate(&self) -> Result<(), WireError> {
        if let Self::Record(r) = self {
            r.format()?;
            r.creation_flags()?;
            r.usage()?;
            r.channels()?;
        }
        Ok(())
    }
    pub(super) const fn texture_id(&self) -> TextureId {
        match self {
            Self::Owned(v) => v.texture_id,
            Self::Record(r) => r.texture_id(),
        }
    }
    pub(super) const fn d3d_format(&self) -> u32 {
        match self {
            Self::Owned(v) => v.d3d_format,
            Self::Record(r) => r.d3d_format,
        }
    }
    pub(super) const fn width(&self) -> u32 {
        match self {
            Self::Owned(v) => v.width,
            Self::Record(r) => r.width,
        }
    }
    pub(super) const fn height(&self) -> u32 {
        match self {
            Self::Owned(v) => v.height,
            Self::Record(r) => r.height,
        }
    }
    pub(super) const fn depth(&self) -> u32 {
        match self {
            Self::Owned(v) => v.depth,
            Self::Record(r) => r.depth,
        }
    }
    pub(super) const fn levels(&self) -> u32 {
        match self {
            Self::Owned(v) => v.levels,
            Self::Record(r) => r.levels,
        }
    }
    pub(super) fn pixel_format(&self) -> PixelFormat {
        match self {
            Self::Owned(v) => v.pixel_format,
            Self::Record(r) => r.format().expect("validated texture format"),
        }
    }
    pub(super) fn create_flags(&self) -> TextureCreateFlags {
        match self {
            Self::Owned(v) => v.create_flags,
            Self::Record(r) => r.creation_flags().expect("validated creation flags"),
        }
    }
    pub(super) fn usage_flags(&self) -> TextureUsage {
        match self {
            Self::Owned(v) => v.usage_flags,
            Self::Record(r) => r.usage().expect("validated texture usage"),
        }
    }
    pub(super) fn swizzle(&self) -> [Swizzle; 4] {
        match self {
            Self::Owned(v) => v.swizzle,
            Self::Record(r) => r.channels().expect("validated texture channels"),
        }
    }
    pub(super) fn to_owned(&self) -> TextureInfo {
        TextureInfo {
            texture_id: self.texture_id(),
            d3d_format: self.d3d_format(),
            width: self.width(),
            height: self.height(),
            depth: self.depth(),
            levels: self.levels(),
            pixel_format: self.pixel_format(),
            create_flags: self.create_flags(),
            usage_flags: self.usage_flags(),
            swizzle: self.swizzle(),
        }
    }
}

pub(super) enum UploadView<'a> {
    Owned(&'a TextureUploadJob),
    Record {
        record: &'a TextureUploadRecord,
        staging: &'a PageBoxRead,
        redirty: &'a Arc<RedirtyQueue>,
    },
}
impl UploadView<'_> {
    pub(super) const fn info(&self) -> TextureView<'_> {
        match self {
            Self::Owned(v) => TextureView::Owned(&v.info),
            Self::Record { record, .. } => TextureView::Record(&record.texture),
        }
    }
    pub(super) const fn staging(&self) -> &PageBoxRead {
        match self {
            Self::Owned(v) => &v.staging,
            Self::Record { staging, .. } => staging,
        }
    }
    pub(super) const fn redirty(&self) -> &Arc<RedirtyQueue> {
        match self {
            Self::Owned(v) => &v.redirty,
            Self::Record { redirty, .. } => redirty,
        }
    }
    pub(super) const fn level(&self) -> u32 {
        match self {
            Self::Owned(v) => v.level,
            Self::Record { record, .. } => record.level,
        }
    }
    pub(super) const fn destination_slice(&self) -> u32 {
        match self {
            Self::Owned(v) => v.destination_slice,
            Self::Record { record, .. } => record.destination_slice,
        }
    }
    pub(super) const fn staging_index(&self) -> usize {
        match self {
            Self::Owned(v) => v.staging_index,
            Self::Record { record, .. } => record.staging_index as usize,
        }
    }
    pub(super) const fn origin_x(&self) -> u32 {
        match self {
            Self::Owned(v) => v.origin_x,
            Self::Record { record, .. } => record.origin_x,
        }
    }
    pub(super) const fn origin_y(&self) -> u32 {
        match self {
            Self::Owned(v) => v.origin_y,
            Self::Record { record, .. } => record.origin_y,
        }
    }
    pub(super) const fn region_w(&self) -> u32 {
        match self {
            Self::Owned(v) => v.region_w,
            Self::Record { record, .. } => record.width,
        }
    }
    pub(super) const fn region_h(&self) -> u32 {
        match self {
            Self::Owned(v) => v.region_h,
            Self::Record { record, .. } => record.height,
        }
    }
    pub(super) const fn src_d3d_format(&self) -> u32 {
        match self {
            Self::Owned(v) => v.src_d3d_format,
            Self::Record { record, .. } => record.source_format,
        }
    }
    pub(super) const fn src_pitch(&self) -> u32 {
        match self {
            Self::Owned(v) => v.src_pitch,
            Self::Record { record, .. } => record.pitch,
        }
    }
    pub(super) const fn bytes_per_pixel(&self) -> u32 {
        match self {
            Self::Owned(v) => v.bytes_per_pixel,
            Self::Record { record, .. } => record.bytes_per_pixel,
        }
    }
    pub(super) const fn depth(&self) -> u32 {
        match self {
            Self::Owned(v) => v.depth,
            Self::Record { record, .. } => record.depth,
        }
    }
    pub(super) const fn slice_pitch(&self) -> u32 {
        match self {
            Self::Owned(v) => v.slice_pitch,
            Self::Record { record, .. } => record.slice_pitch,
        }
    }
    pub(super) const fn release_staging(&self) -> bool {
        match self {
            Self::Owned(v) => v.release_staging,
            Self::Record { record, .. } => record.release_staging != 0,
        }
    }
    pub(super) const fn upload_generation(&self) -> u32 {
        match self {
            Self::Owned(v) => v.upload_generation,
            Self::Record { record, .. } => record.upload_generation,
        }
    }
    pub(super) fn redirty_subresource(&self) -> RedirtySubresource {
        RedirtySubresource {
            texture_id: self.info().texture_id(),
            index: u32::try_from(self.staging_index()).unwrap_or(u32::MAX),
        }
    }
    pub(super) const fn redirty_rect(&self) -> DirtyRect {
        DirtyRect {
            x: self.origin_x(),
            y: self.origin_y(),
            w: self.region_w(),
            h: self.region_h(),
        }
    }
    pub(super) fn emitted_answer(&self) -> EmittedUpload {
        EmittedUpload {
            subresource: self.redirty_subresource(),
            level: self.level(),
            generation: self.upload_generation(),
            releases_staging: self.release_staging(),
        }
    }
    pub(super) const fn record_recovery(
        record: &TextureUploadRecord,
        info: TextureInfo,
        staging: PageBoxRead,
        redirty: Arc<RedirtyQueue>,
    ) -> TextureUploadJob {
        TextureUploadJob {
            info,
            staging,
            redirty,
            level: record.level,
            destination_slice: record.destination_slice,
            staging_index: record.staging_index as usize,
            origin_x: record.origin_x,
            origin_y: record.origin_y,
            region_w: record.width,
            region_h: record.height,
            src_d3d_format: record.source_format,
            src_pitch: record.pitch,
            bytes_per_pixel: record.bytes_per_pixel,
            depth: record.depth,
            slice_pitch: record.slice_pitch,
            release_staging: record.release_staging != 0,
            upload_generation: record.upload_generation,
        }
    }
}
