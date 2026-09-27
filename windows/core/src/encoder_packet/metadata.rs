//! Metadata fields accompanying an immutable recorded operation stream.

use mtld3d_shared::encoder_wire::{WireError, WireReader, WireWriter};

use super::FrameRecorder;
use crate::{
    encoder_data::{
        FrameData, FrameDataFlags, FrameInit, PendingVbibRetention, StagingWarmupEntry,
        VbibWarmupEntry,
    },
    encoder_value::WireValue,
    guest_pages::{GuestPageDescriptor, GuestPageLease},
};

/// Published address ranges and exact single-adoption descriptors.
pub struct PacketInventory {
    #[cfg(any(test, debug_assertions))]
    pub ranges: Vec<(u64, u64)>,
    #[cfg(any(test, debug_assertions))]
    pub pages: Vec<[u64; 7]>,
    #[cfg(any(test, debug_assertions))]
    pub queries: Vec<[u64; 2]>,
    pub registrations: Vec<u64>,
    #[cfg(any(test, debug_assertions))]
    pub redirties: Vec<[u64; 2]>,
    #[cfg(any(test, debug_assertions))]
    pub replies_u64: Vec<u64>,
    #[cfg(any(test, debug_assertions))]
    pub replies_bool: Vec<u64>,
    #[cfg(any(test, debug_assertions))]
    pub readbacks: Vec<(u64, u64)>,
    #[cfg(any(test, debug_assertions))]
    pub command_spans: Vec<(u64, u64)>,
    #[cfg(any(test, debug_assertions))]
    pub backings: Vec<(u64, u64)>,
}

pub fn write_metadata(
    frame: &mut FrameData,
    writer: &mut WireWriter<'_>,
    recorder: &mut FrameRecorder,
) -> Result<(), WireError> {
    write_metadata_with_inventory(frame, writer, recorder, cfg!(any(test, debug_assertions)))
}

// The explicit argument lets host tests exercise the release producer's identical wire shape.
fn write_metadata_with_inventory(
    frame: &mut FrameData,
    writer: &mut WireWriter<'_>,
    recorder: &mut FrameRecorder,
    has_inventory: bool,
) -> Result<(), WireError> {
    frame.device_handle.write_wire(writer)?;
    frame.record_handle.write_wire(writer)?;
    frame.backbuffer_handle.write_wire(writer)?;
    frame.backbuffer_srgb_handle.write_wire(writer)?;
    frame.backbuffer_msaa_handle.write_wire(writer)?;
    frame.backbuffer_msaa_srgb_handle.write_wire(writer)?;
    frame.backbuffer_sample_count.write_wire(writer)?;
    frame.layer_handle.write_wire(writer)?;
    frame.view_handle.write_wire(writer)?;
    frame.backbuffer_width.write_wire(writer)?;
    frame.backbuffer_height.write_wire(writer)?;
    frame.backbuffer_format.write_wire(writer)?;
    frame.render_scale.write_wire(writer)?;
    frame.backbuffer_contents.write_wire(writer)?;
    frame.depth_texture.write_wire(writer)?;
    let mut flags = frame.flags;
    flags.set(FrameDataFlags::VALIDATION_INVENTORY, has_inventory);
    flags.write_wire(writer)?;
    frame.perf.write_wire(writer)?;
    frame.submit_seq.write_wire(writer)?;
    frame.coherent_seq_ptr.write_wire(writer)?;
    frame.upload_coherent_seq_ptr.write_wire(writer)?;
    frame.failed_submit_seq_ptr.write_wire(writer)?;
    frame.retained_bytes_ptr.write_wire(writer)?;
    frame.apply_pacing.write_wire(writer)?;
    frame.apply_gamma.write_wire(writer)?;
    frame.op_vec_realloc_bytes.write_wire(writer)?;
    frame.pending_texture_warmups.write_wire(writer)?;
    u32::try_from(frame.pending_buffer_warmups.len())
        .map_err(|_| WireError::TooLarge)?
        .write_wire(writer)?;
    for entry in &frame.pending_buffer_warmups {
        entry.buffer_id.write_wire(writer)?;
        entry.backing_ptr.write_wire(writer)?;
        entry.backing_len.write_wire(writer)?;
        entry.backing_generation.write_wire(writer)?;
        entry.map_mode.write_wire(writer)?;
    }
    u32::try_from(frame.pending_staging_warmups.len())
        .map_err(|_| WireError::TooLarge)?
        .write_wire(writer)?;
    let mut staging = core::mem::take(&mut frame.pending_staging_warmups).into_iter();
    while let Some(entry) = staging.next() {
        let lease = GuestPageLease::for_shared_pooled(entry.keepalive, &recorder.completion_pool);
        let descriptor = lease.descriptor();
        recorder.pages.push(lease);
        let result = (|| {
            entry.texture_id.write_wire(writer)?;
            entry.level.write_wire(writer)?;
            entry.backing_ptr.write_wire(writer)?;
            entry.backing_len.write_wire(writer)?;
            descriptor.write_wire(writer)
        })();
        if let Err(error) = result {
            frame.pending_staging_warmups.extend(staging);
            return Err(error);
        }
    }
    u32::try_from(frame.vbib_retentions.len())
        .map_err(|_| WireError::TooLarge)?
        .write_wire(writer)?;
    let mut retained = core::mem::take(&mut frame.vbib_retentions).into_iter();
    while let Some(entry) = retained.next() {
        let lease = GuestPageLease::for_recyclable_pooled(
            entry.page_box,
            &recorder.completion_pool,
            recorder.pagebox_pool,
        );
        let descriptor = lease.descriptor();
        recorder.pages.push(lease);
        let result = (|| {
            entry.buffer_id.write_wire(writer)?;
            entry.last_submit_seq.write_wire(writer)?;
            descriptor.write_wire(writer)
        })();
        if let Err(error) = result {
            frame.vbib_retentions.extend(retained);
            return Err(error);
        }
    }
    if !has_inventory {
        writer.u32(0)?; // Scratch ranges.
        writer.u32(0)?; // Page provenance.
        writer.u32(0)?; // Query provenance.
        recorder.registrations.write_wire(writer)?;
        writer.u32(0)?; // Redirty provenance.
        writer.u32(0)?; // U64 reply provenance.
        writer.u32(0)?; // Boolean reply provenance.
        writer.u32(0)?; // Readback provenance.
        writer.u32(0)?; // Command provenance; the actual command table remains separate.
        writer.u32(0)?; // Buffer backing provenance.
        return Ok(());
    }
    u32::try_from(frame.scratch.allocation_ranges().count())
        .map_err(|_| WireError::TooLarge)?
        .write_wire(writer)?;
    for (address, length) in frame.scratch.allocation_ranges() {
        address.write_wire(writer)?;
        length.write_wire(writer)?;
    }
    u32::try_from(recorder.pages.len())
        .map_err(|_| WireError::TooLarge)?
        .write_wire(writer)?;
    for page in &recorder.pages {
        page.descriptor().wire_fields().write_wire(writer)?;
    }
    u32::try_from(recorder.queries.len())
        .map_err(|_| WireError::TooLarge)?
        .write_wire(writer)?;
    for query in &recorder.queries {
        query.descriptor().wire_fields().write_wire(writer)?;
    }
    recorder.registrations.write_wire(writer)?;
    u32::try_from(recorder.redirties.len())
        .map_err(|_| WireError::TooLarge)?
        .write_wire(writer)?;
    for redirty in &recorder.redirties {
        redirty.descriptor().wire_fields().write_wire(writer)?;
    }
    u32::try_from(recorder.replies_u64.len())
        .map_err(|_| WireError::TooLarge)?
        .write_wire(writer)?;
    for reply in &recorder.replies_u64 {
        reply.address().write_wire(writer)?;
    }
    u32::try_from(recorder.replies_bool.len())
        .map_err(|_| WireError::TooLarge)?
        .write_wire(writer)?;
    for reply in &recorder.replies_bool {
        reply.address().write_wire(writer)?;
    }
    u32::try_from(recorder.readbacks.len())
        .map_err(|_| WireError::TooLarge)?
        .write_wire(writer)?;
    for (address, length) in &recorder.readbacks {
        address.write_wire(writer)?;
        length.write_wire(writer)?;
    }
    u32::try_from(recorder.slab.ranges().count())
        .map_err(|_| WireError::TooLarge)?
        .write_wire(writer)?;
    for (address, length) in recorder.slab.ranges() {
        address.write_wire(writer)?;
        length.write_wire(writer)?;
    }
    u32::try_from(recorder.ranges.len())
        .map_err(|_| WireError::TooLarge)?
        .write_wire(writer)?;
    for (address, length) in &recorder.ranges {
        address.write_wire(writer)?;
        length.write_wire(writer)?;
    }
    Ok(())
}

/// Metadata parsed once, with guest ownership held back until complete validation.
pub(super) struct ParsedMetadata {
    #[cfg(any(test, debug_assertions))]
    pub has_inventory: bool,
    pub frame: FrameData,
    pub inventory: PacketInventory,
    staging: Vec<(crate::ids::TextureId, u32, u64, u64, GuestPageDescriptor)>,
    retained: Vec<(crate::ids::BufferId, u64, GuestPageDescriptor)>,
}

impl ParsedMetadata {
    /// Adopt descriptors only after all operation records and inventories have passed.
    ///
    /// # Safety
    /// Each descriptor is uniquely retained by the immutable packet until acknowledgment.
    pub unsafe fn adopt(mut self) -> Result<FrameData, WireError> {
        for (texture_id, level, backing_ptr, backing_len, descriptor) in self.staging {
            // SAFETY: whole-packet validation established this unique retained descriptor.
            let keepalive = unsafe { descriptor.adopt_shared()? };
            if keepalive.as_ptr() as u64 != backing_ptr || keepalive.len() as u64 != backing_len {
                return Err(WireError::InvalidValue);
            }
            self.frame.pending_staging_warmups.push(StagingWarmupEntry {
                texture_id,
                level,
                backing_ptr,
                backing_len,
                keepalive,
            });
        }
        for (buffer_id, last_submit_seq, descriptor) in self.retained {
            // SAFETY: whole-packet validation established this unique retained descriptor.
            let page_box = unsafe { descriptor.adopt_owned()? };
            self.frame.vbib_retentions.push(PendingVbibRetention {
                buffer_id,
                page_box,
                last_submit_seq,
            });
        }
        Ok(self.frame)
    }
}

/// Parse scalar metadata and descriptors without adopting or touching guest memory.
pub(super) fn read_metadata(reader: &mut WireReader<'_>) -> Result<ParsedMetadata, WireError> {
    #[cfg(any(test, debug_assertions))]
    let mut metadata_pages = Vec::new();
    let init = FrameInit {
        device_handle: WireValue::read_wire(reader)?,
        record_handle: WireValue::read_wire(reader)?,
        backbuffer_handle: WireValue::read_wire(reader)?,
        backbuffer_srgb_handle: WireValue::read_wire(reader)?,
        backbuffer_msaa_handle: WireValue::read_wire(reader)?,
        backbuffer_msaa_srgb_handle: WireValue::read_wire(reader)?,
        backbuffer_sample_count: WireValue::read_wire(reader)?,
        layer_handle: WireValue::read_wire(reader)?,
        view_handle: WireValue::read_wire(reader)?,
        backbuffer_width: WireValue::read_wire(reader)?,
        backbuffer_height: WireValue::read_wire(reader)?,
        backbuffer_format: WireValue::read_wire(reader)?,
        render_scale: WireValue::read_wire(reader)?,
        backbuffer_contents: WireValue::read_wire(reader)?,
        depth_texture: WireValue::read_wire(reader)?,
        depth_has_stencil: false,
    };
    let mut frame = FrameData::new(&init);
    frame.flags = WireValue::read_wire(reader)?;
    #[cfg(any(test, debug_assertions))]
    let has_inventory = frame.flags.contains(FrameDataFlags::VALIDATION_INVENTORY);
    frame.flags.remove(FrameDataFlags::VALIDATION_INVENTORY);
    frame.perf = WireValue::read_wire(reader)?;
    frame.submit_seq = WireValue::read_wire(reader)?;
    frame.coherent_seq_ptr = read_counter(reader)?;
    frame.upload_coherent_seq_ptr = read_counter(reader)?;
    frame.failed_submit_seq_ptr = read_counter(reader)?;
    frame.retained_bytes_ptr = read_counter(reader)?;
    frame.apply_pacing = WireValue::read_wire(reader)?;
    frame.apply_gamma = WireValue::read_wire(reader)?;
    frame.op_vec_realloc_bytes = WireValue::read_wire(reader)?;
    frame.pending_texture_warmups = WireValue::read_wire(reader)?;
    let buffers = bounded_count(reader, 33)?;
    frame
        .pending_buffer_warmups
        .try_reserve(buffers)
        .map_err(|_| WireError::AllocationFailed)?;
    for _ in 0..buffers {
        let entry = VbibWarmupEntry {
            buffer_id: WireValue::read_wire(reader)?,
            backing_ptr: reader.u64()?,
            backing_len: reader.u64()?,
            backing_generation: reader.u64()?,
            map_mode: WireValue::read_wire(reader)?,
        };
        validate_range(reader, entry.backing_ptr, entry.backing_len, 1)?;
        frame.pending_buffer_warmups.push(entry);
    }
    let staging = bounded_count(reader, 84)?;
    let mut staging_descriptors = Vec::new();
    staging_descriptors
        .try_reserve(staging)
        .map_err(|_| WireError::AllocationFailed)?;
    frame
        .pending_staging_warmups
        .try_reserve(staging)
        .map_err(|_| WireError::AllocationFailed)?;
    for _ in 0..staging {
        let texture_id = WireValue::read_wire(reader)?;
        let level = reader.u32()?;
        let backing_ptr = reader.u64()?;
        let backing_len = reader.u64()?;
        validate_range(reader, backing_ptr, backing_len, 1)?;
        let descriptor = GuestPageDescriptor::read_wire(reader)?;
        let fields = descriptor.wire_fields();
        if fields[6] != 0 {
            return Err(WireError::InvalidValue);
        }
        if fields[0] != backing_ptr || fields[1] != backing_len {
            return Err(WireError::InvalidValue);
        }
        #[cfg(any(test, debug_assertions))]
        if has_inventory {
            metadata_pages.push(fields);
        }
        staging_descriptors.push((texture_id, level, backing_ptr, backing_len, descriptor));
    }
    let retained = bounded_count(reader, 72)?;
    let mut retained_descriptors = Vec::new();
    retained_descriptors
        .try_reserve(retained)
        .map_err(|_| WireError::AllocationFailed)?;
    frame
        .vbib_retentions
        .try_reserve(retained)
        .map_err(|_| WireError::AllocationFailed)?;
    for _ in 0..retained {
        let buffer_id = WireValue::read_wire(reader)?;
        let last_submit_seq = reader.u64()?;
        let descriptor = GuestPageDescriptor::read_wire(reader)?;
        let fields = descriptor.wire_fields();
        if fields[6] != 0 {
            return Err(WireError::InvalidValue);
        }
        #[cfg(any(test, debug_assertions))]
        if has_inventory {
            metadata_pages.push(fields);
        }
        retained_descriptors.push((buffer_id, last_submit_seq, descriptor));
    }
    #[cfg(any(test, debug_assertions))]
    let inventory = read_inventory(reader, metadata_pages)?;
    #[cfg(not(any(test, debug_assertions)))]
    let inventory = read_inventory(reader)?;
    Ok(ParsedMetadata {
        #[cfg(any(test, debug_assertions))]
        has_inventory,
        frame,
        staging: staging_descriptors,
        retained: retained_descriptors,
        inventory,
    })
}

#[cfg(any(test, debug_assertions))]
fn read_inventory(
    reader: &mut WireReader<'_>,
    metadata_pages: Vec<[u64; 7]>,
) -> Result<PacketInventory, WireError> {
    let count = bounded_count(reader, 16)?;
    let mut ranges = Vec::new();
    ranges
        .try_reserve(count)
        .map_err(|_| WireError::AllocationFailed)?;
    for _ in 0..count {
        let address = reader.u64()?;
        let length = reader.u64()?;
        validate_range(reader, address, length, 1)?;
        ranges.push((address, length));
    }
    let mut pages = Vec::<[u64; 7]>::read_wire(reader)?;
    let queries = Vec::<[u64; 2]>::read_wire(reader)?;
    let registrations = Vec::<u64>::read_wire(reader)?;
    let redirties = Vec::<[u64; 2]>::read_wire(reader)?;
    let replies_u64 = Vec::<u64>::read_wire(reader)?;
    let replies_bool = Vec::<u64>::read_wire(reader)?;
    let readback_count = bounded_count(reader, 16)?;
    let mut readbacks = Vec::new();
    readbacks
        .try_reserve(readback_count)
        .map_err(|_| WireError::TooLarge)?;
    for _ in 0..readback_count {
        readbacks.push((u64::read_wire(reader)?, u64::read_wire(reader)?));
    }
    let command_spans = read_spans(reader)?;
    let backings = read_spans(reader)?;
    for fields in metadata_pages {
        let index = pages
            .iter()
            .position(|page| *page == fields)
            .ok_or(WireError::InvalidValue)?;
        pages.swap_remove(index);
    }
    Ok(PacketInventory {
        ranges,
        pages,
        queries,
        registrations,
        redirties,
        replies_u64,
        replies_bool,
        readbacks,
        command_spans,
        backings,
    })
}

#[cfg(not(any(test, debug_assertions)))]
fn read_inventory(reader: &mut WireReader<'_>) -> Result<PacketInventory, WireError> {
    Ok(PacketInventory {
        registrations: read_inventory_registrations(reader)?,
    })
}

#[cfg(any(test, not(debug_assertions)))]
fn read_inventory_registrations(reader: &mut WireReader<'_>) -> Result<Vec<u64>, WireError> {
    // Consume the diagnostic footer from a debug producer without allocating storage
    // for it. The actual command table and every owning descriptor are independent.
    discard_ranges(reader)?;
    discard_list::<[u64; 7]>(reader)?;
    discard_list::<[u64; 2]>(reader)?;
    let registrations = Vec::<u64>::read_wire(reader)?;
    discard_list::<[u64; 2]>(reader)?;
    discard_list::<u64>(reader)?;
    discard_list::<u64>(reader)?;
    discard_list::<[u64; 2]>(reader)?;
    discard_list::<[u64; 2]>(reader)?;
    discard_list::<[u64; 2]>(reader)?;
    Ok(registrations)
}

#[cfg(any(test, not(debug_assertions)))]
fn discard_ranges(reader: &mut WireReader<'_>) -> Result<(), WireError> {
    let count = bounded_count(reader, 16)?;
    for _ in 0..count {
        let address = reader.u64()?;
        let length = reader.u64()?;
        validate_range(reader, address, length, 1)?;
    }
    Ok(())
}

#[cfg(any(test, not(debug_assertions)))]
fn discard_list<T: WireValue>(reader: &mut WireReader<'_>) -> Result<(), WireError> {
    let count = bounded_count(reader, T::MIN_WIRE_BYTES)?;
    for _ in 0..count {
        T::read_wire(reader)?;
    }
    Ok(())
}

#[cfg(any(test, debug_assertions))]
fn read_spans(reader: &mut WireReader<'_>) -> Result<Vec<(u64, u64)>, WireError> {
    let count = bounded_count(reader, 16)?;
    let mut spans = Vec::new();
    spans
        .try_reserve(count)
        .map_err(|_| WireError::AllocationFailed)?;
    for _ in 0..count {
        spans.push((reader.u64()?, reader.u64()?));
    }
    Ok(spans)
}

fn bounded_count(reader: &mut WireReader<'_>, minimum_bytes: usize) -> Result<usize, WireError> {
    let count = usize::try_from(reader.u32()?).map_err(|_| WireError::TooLarge)?;
    if count > reader.remaining_len() / minimum_bytes {
        return Err(WireError::Truncated);
    }
    Ok(count)
}

fn read_counter(reader: &mut WireReader<'_>) -> Result<u64, WireError> {
    let address = reader.u64()?;
    if address != 0 {
        validate_range(reader, address, 8, 8)?;
    }
    Ok(address)
}

fn validate_range(
    reader: &WireReader<'_>,
    address: u64,
    length: u64,
    alignment: u64,
) -> Result<(), WireError> {
    if !reader.has_trusted_addresses()
        || address == 0
        || !address.is_multiple_of(alignment)
        || usize::try_from(address).is_err()
        || isize::try_from(length).is_err()
        || address.checked_add(length).is_none()
    {
        return Err(WireError::InvalidValue);
    }
    Ok(())
}

#[cfg(test)]
mod tests;
