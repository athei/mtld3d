//! Validate a complete operation stream before any ownership is adopted.

use super::{
    DrawReader, EncoderOpcode, GuestPageDescriptor, GuestQueryDescriptor, Op, QueryLeaseCache,
    ScratchArena, UploadFields, WireError, WireReader, WireValue, metadata::PacketInventory,
    read_operation,
};

pub(super) struct Validation {
    pub inventory: PacketInventory,
    backings: rustc_hash::FxHashSet<(u64, u64)>,
}

impl Validation {
    pub fn new(inventory: PacketInventory) -> Result<Self, WireError> {
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
        draws: &mut DrawReader,
        scratch: &mut ScratchArena,
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
            EncoderOpcode::SetVsConstRange
            | EncoderOpcode::SetPsConstRange
            | EncoderOpcode::SetFfVsConstRange
            | EncoderOpcode::Draw
            | EncoderOpcode::SetViewport
            | EncoderOpcode::SetVertexSampler
            | EncoderOpcode::SetVertexTexture
            | EncoderOpcode::BindDepth
            | EncoderOpcode::BindColor
            | EncoderOpcode::GenerateMipmapsOrdered
            | EncoderOpcode::UnbindExtraColor
            | EncoderOpcode::DestroyTexture
            | EncoderOpcode::NoteColorRead
            | EncoderOpcode::ResolveDepthSurface
            | EncoderOpcode::StretchBlit
            | EncoderOpcode::ColorFill
            | EncoderOpcode::CarryDepth
            | EncoderOpcode::ClearColor
            | EncoderOpcode::ClearColorRects
            | EncoderOpcode::ClearDepthStencilRects
            | EncoderOpcode::ClearDepthStencil
            | EncoderOpcode::ResolveDynamicDepth
            | EncoderOpcode::ResolveDepthTexture
            | EncoderOpcode::RetireColor
            | EncoderOpcode::RetireDepth
            | EncoderOpcode::UploadColor
            | EncoderOpcode::UploadResampled
            | EncoderOpcode::GenerateMipmaps
            | EncoderOpcode::SetDumpDraw
            | EncoderOpcode::SetSnapshot => {
                let mut queries = QueryLeaseCache::default();
                let op = read_operation(tag, reader, draws, scratch, &mut queries, &mut |_| {
                    Err(WireError::InvalidValue)
                })?;
                match &op {
                    Op::Draw(draw) => {
                        use crate::draw_data::{IndexSource, VertexSource};
                        if let VertexSource::Bound { first, extra, .. } = &draw.vertex_source {
                            for stream in core::iter::once(first).chain(extra.iter()) {
                                if !self.backings.contains(&(
                                    stream.backing_ptr as u64,
                                    stream.backing_len as u64,
                                )) {
                                    return Err(WireError::InvalidValue);
                                }
                            }
                        }
                        if let IndexSource::Bound {
                            backing_ptr,
                            backing_len,
                            ..
                        } = &draw.index_source
                            && !self
                                .backings
                                .contains(&(*backing_ptr as u64, *backing_len as u64))
                        {
                            return Err(WireError::InvalidValue);
                        }
                    }
                    Op::UploadColor(value)
                        if u64::from(value.src_stride) * u64::from(value.height)
                            > u64::from(value.bytes.as_raw().1) =>
                    {
                        return Err(WireError::InvalidValue);
                    }
                    Op::UploadResampled(value)
                        if u64::from(value.target.bytes_per_row)
                            * u64::from(value.target.logical.1)
                            > u64::from(value.bytes.as_raw().1) =>
                    {
                        return Err(WireError::InvalidValue);
                    }

                    _ => {}
                }
                if let Op::SetVsConstRange {
                    start_row,
                    rows,
                    data,
                }
                | Op::SetPsConstRange {
                    start_row,
                    rows,
                    data,
                }
                | Op::SetFfVsConstRange {
                    start_row,
                    rows,
                    data,
                } = op
                    && (usize::from(start_row)
                        .checked_add(usize::from(rows))
                        .is_none_or(|end| end > crate::draw_data::CONSTANT_ROWS)
                        || data.as_raw().1 < u32::from(rows) * 16)
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
