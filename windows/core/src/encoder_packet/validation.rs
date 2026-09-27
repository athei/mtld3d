//! Validate a complete operation stream before any ownership is adopted.

use super::{
    EncoderOpcode, GuestPageDescriptor, GuestQueryDescriptor, UploadFields, WireError, WireReader,
    WireValue, metadata::PacketInventory, read_scratch_slice,
};
use crate::encoder_data::{
    BindColorOp, BindDepthOp, CarryDepthOp, ClearColorOp, ClearDepthStencilOp, ColorFillOp,
    DestroyTextureOp, GenerateMipmapsOp, GenerateMipmapsOrderedOp, NoteColorReadOp,
    ResolveDepthSurfaceOp, ResolveDepthTextureOp, ResolveDynamicDepthOp, RetireColorOp,
    RetireDepthOp, SetDumpDrawOp, SetVertexSamplerOp, SetVertexTextureOp, SetViewportOp,
    StretchBlitOp, UnbindExtraColorOp,
};

pub(super) struct Validation<'a> {
    pub inventory: &'a mut PacketInventory,
    backings: rustc_hash::FxHashSet<(u64, u64)>,
}

impl<'a> Validation<'a> {
    pub fn new(inventory: &'a mut PacketInventory) -> Result<Self, WireError> {
        let mut backings = rustc_hash::FxHashSet::default();
        backings
            .try_reserve(inventory.backings.len())
            .map_err(|_| WireError::AllocationFailed)?;
        backings.extend(inventory.backings.iter().copied());
        Ok(Self {
            inventory,
            backings,
        })
    }

    pub fn operation(
        &mut self,
        tag: &EncoderOpcode,
        reader: &mut WireReader<'_>,
    ) -> Result<(), WireError> {
        match tag {
            EncoderOpcode::ReadColorHandle
            | EncoderOpcode::ReadTextureHandle
            | EncoderOpcode::ReadTextureColorHandle => {
                crate::ids::TextureId::read_wire(reader)?;
                let address = reader.u64()?;
                super::validate_range(address, 8, 8)?;
                take_exact(&mut self.inventory.replies_u64, address)?;
            }
            EncoderOpcode::ReadDeviceBuffer => {
                crate::ids::BufferId::read_wire(reader)?;
                let address = reader.u64()?;
                let length = reader.u64()?;
                let reply = reader.u64()?;
                super::validate_range(address, length, 1)?;
                super::validate_range(reply, 4, 4)?;
                take_exact(&mut self.inventory.readbacks, (address, length))?;
                take_exact(&mut self.inventory.replies_bool, reply)?;
            }
            EncoderOpcode::AdoptProgram => {
                let registration = reader.u64()?;
                let index = self
                    .inventory
                    .registrations
                    .iter()
                    .position(|value| *value == registration)
                    .ok_or(WireError::InvalidValue)?;
                self.inventory.registrations.swap_remove(index);
                if registration == 0 {
                    return Err(WireError::InvalidValue);
                }
            }
            EncoderOpcode::BeginVisibility | EncoderOpcode::EndVisibility => {
                reader.u64()?;
                let descriptor = GuestQueryDescriptor::read_wire(reader)?;
                take_exact(&mut self.inventory.queries, descriptor.wire_fields())?;
            }
            EncoderOpcode::UploadTextureAndMips | EncoderOpcode::UploadTexture => {
                if matches!(tag, EncoderOpcode::UploadTextureAndMips) {
                    crate::ids::TextureId::read_wire(reader)?;
                    crate::encoder_data::UploadTextureOpFlags::read_wire(reader)?;
                }
                let upload = UploadFields::read(reader)?;
                take_exact(&mut self.inventory.redirties, upload.redirty.wire_fields())?;
                let descriptor = GuestPageDescriptor::read_wire(reader)?;
                let fields = descriptor.wire_fields();
                if fields[6] == 0 {
                    return Err(WireError::InvalidValue);
                }
                take_exact(&mut self.inventory.pages, fields)?;
                upload.validate(fields[2])?;
            }
            EncoderOpcode::StageUpload => {
                crate::ids::BufferId::read_wire(reader)?;
                let destination = reader.u32()?;
                let size = reader.u32()?;
                let descriptor = GuestPageDescriptor::read_wire(reader)?;
                let fields = descriptor.wire_fields();
                if fields[6] != 0
                    || u64::from(size) > fields[2]
                    || destination.checked_add(size).is_none()
                {
                    return Err(WireError::InvalidValue);
                }
                take_exact(&mut self.inventory.pages, fields)?;
            }
            EncoderOpcode::SetViewport => {
                SetViewportOp::read_wire(reader)?;
            }
            EncoderOpcode::SetVertexSampler => {
                SetVertexSamplerOp::read_wire(reader)?;
            }
            EncoderOpcode::SetVertexTexture => {
                SetVertexTextureOp::read_wire(reader)?;
            }
            EncoderOpcode::BindDepth => {
                BindDepthOp::read_wire(reader)?;
            }
            EncoderOpcode::BindColor => {
                BindColorOp::read_wire(reader)?;
            }
            EncoderOpcode::GenerateMipmapsOrdered => {
                GenerateMipmapsOrderedOp::read_wire(reader)?;
            }
            EncoderOpcode::UnbindExtraColor => {
                UnbindExtraColorOp::read_wire(reader)?;
            }
            EncoderOpcode::DestroyTexture => {
                DestroyTextureOp::read_wire(reader)?;
            }
            EncoderOpcode::NoteColorRead => {
                NoteColorReadOp::read_wire(reader)?;
            }
            EncoderOpcode::ResolveDepthSurface => {
                ResolveDepthSurfaceOp::read_wire(reader)?;
            }
            EncoderOpcode::StretchBlit => {
                StretchBlitOp::read_wire(reader)?;
            }
            EncoderOpcode::ColorFill => {
                ColorFillOp::read_wire(reader)?;
            }
            EncoderOpcode::CarryDepth => {
                CarryDepthOp::read_wire(reader)?;
            }
            EncoderOpcode::ClearColor => {
                ClearColorOp::read_wire(reader)?;
            }
            EncoderOpcode::ClearDepthStencil => {
                ClearDepthStencilOp::read_wire(reader)?;
            }
            EncoderOpcode::ResolveDynamicDepth => {
                ResolveDynamicDepthOp::read_wire(reader)?;
            }
            EncoderOpcode::ResolveDepthTexture => {
                ResolveDepthTextureOp::read_wire(reader)?;
            }
            EncoderOpcode::RetireColor => {
                RetireColorOp::read_wire(reader)?;
            }
            EncoderOpcode::RetireDepth => {
                RetireDepthOp::read_wire(reader)?;
            }
            EncoderOpcode::GenerateMipmaps => {
                GenerateMipmapsOp::read_wire(reader)?;
            }
            EncoderOpcode::SetDumpDraw => {
                SetDumpDrawOp::read_wire(reader)?;
            }
            EncoderOpcode::SetVsConstRange
            | EncoderOpcode::SetPsConstRange
            | EncoderOpcode::SetFfVsConstRange => {
                let start = reader.u16()?;
                let rows = reader.u16()?;
                let data = read_scratch_slice(reader)?;
                if usize::from(start) + usize::from(rows) > crate::draw_data::CONSTANT_ROWS
                    || data.as_raw().1 < u32::from(rows) * 16
                {
                    return Err(WireError::InvalidValue);
                }
            }
            EncoderOpcode::Draw => {
                crate::encoder_draw::validate_wire_draw(reader, |address, length| {
                    if self.backings.contains(&(address, length)) {
                        Ok(())
                    } else {
                        Err(WireError::InvalidValue)
                    }
                })?;
            }
            EncoderOpcode::SetSnapshot => crate::encoder_draw::validate_wire_snapshot(reader)?,
            EncoderOpcode::ClearColorRects => {
                ClearColorOp::read_wire(reader)?;
                validate_rects(reader)?;
            }
            EncoderOpcode::ClearDepthStencilRects => {
                ClearDepthStencilOp::read_wire(reader)?;
                validate_rects(reader)?;
            }
            EncoderOpcode::UploadColor => {
                reader.u64()?;
                let bytes = read_scratch_slice(reader)?;
                reader.u32()?;
                let height = reader.u32()?;
                let stride = reader.u32()?;
                if u64::from(stride) * u64::from(height) > u64::from(bytes.as_raw().1) {
                    return Err(WireError::InvalidValue);
                }
            }
            EncoderOpcode::UploadResampled => {
                let target = crate::encoder_data::ResampledUpload::read_wire(reader)?;
                let bytes = read_scratch_slice(reader)?;
                if u64::from(target.bytes_per_row) * u64::from(target.logical.1)
                    > u64::from(bytes.as_raw().1)
                {
                    return Err(WireError::InvalidValue);
                }
            }
        }
        if !reader.is_empty() {
            return Err(WireError::InvalidValue);
        }
        Ok(())
    }

    pub fn finish(self) -> Result<(), WireError> {
        if !self.inventory.pages.is_empty()
            || !self.inventory.queries.is_empty()
            || !self.inventory.redirties.is_empty()
            || !self.inventory.registrations.is_empty()
            || !self.inventory.replies_u64.is_empty()
            || !self.inventory.replies_bool.is_empty()
            || !self.inventory.readbacks.is_empty()
        {
            return Err(WireError::InvalidValue);
        }
        Ok(())
    }
}

fn take_exact<T: PartialEq + Copy>(inventory: &mut Vec<T>, fields: T) -> Result<(), WireError> {
    let index = inventory
        .iter()
        .position(|value| value == &fields)
        .ok_or(WireError::InvalidValue)?;
    inventory.swap_remove(index);
    Ok(())
}

fn validate_rects(reader: &mut WireReader<'_>) -> Result<(), WireError> {
    let count = reader.u32()? as usize;
    if count > reader.remaining_len() / 16 {
        return Err(WireError::Truncated);
    }
    for _ in 0..count {
        <(i32, i32, i32, i32)>::read_wire(reader)?;
    }
    Ok(())
}
