//! Dirty draw state captured directly into the frame wire buffer.
//!
//! Structural values decode into native scratch. Immutable uniform bytes retain
//! their PE backing through submit replay; their addresses never transfer ownership.

use std::ptr::NonNull;

use mtld3d_shared::{
    VertexAttrDesc,
    encoder_wire::{WireError, WireReader, WireWriter},
};

use crate::{
    draw_data::{
        AttrSnapshot, CurrentSnapshot, CurrentSnapshotPtr, DepthStencilFlags, DrawOp, IndexSource,
        PsSource, PsSourcePtr, RenderStatePtr, RenderStateSnapshot, ScratchSlice, StageBinding,
        StageBindingsPtr, StreamBinding, VertexSource, VsSource, VsSourcePtr,
    },
    dxso::VariantKey,
    encoder_value::WireValue,
    scratch::ScratchArena,
};

#[cfg(test)]
mod tests;

/// Reserved upper bound for a delta with every structural field and byte binding.
///
/// The full-width declaration, stage and fixed-function fixture pins this limit.
pub const SNAPSHOT_DELTA_MAX_BYTES: usize = 4096;

/// API-owned shader keys needed to build subsequent dirty state.
///
/// Structural snapshots and uniform byte bindings are owned by the native decoder.
pub struct ApiSnapshotCache {
    pub vs: Option<VsSource>,
    pub ps: Option<PsSource>,
    pub variant: Option<VariantKey>,
    pub depth_stencil: DepthStencilFlags,
}

impl ApiSnapshotCache {
    pub const EMPTY: Self = Self {
        vs: None,
        ps: None,
        variant: None,
        depth_stencil: DepthStencilFlags::empty(),
    };
}

/// Borrowed declaration data emitted once when the declaration becomes dirty.
pub struct SnapshotAttributes<'a> {
    pub attrs: &'a [VertexAttrDesc],
    pub extents: &'a [u32; 16],
    pub used_streams: u16,
    pub vdecl_hash: u64,
}

/// Changes built by the API's existing dirty-state gates.
///
/// Byte bindings use outer `None` for unchanged and `Some(None)` for cleared.
/// Their order is VS constants, PS constants, alpha, fog, bump environment,
/// VS integer, VS boolean, PS integer, PS boolean, and per-draw VS uniforms.
#[derive(Default)]
pub struct SnapshotDelta<'a> {
    pub render_state: Option<&'a RenderStateSnapshot>,
    pub stages: Option<(u16, &'a [StageBinding])>,
    pub attrs: Option<SnapshotAttributes<'a>>,
    pub vs: Option<&'a VsSource>,
    pub ps: Option<&'a PsSource>,
    pub variant: Option<VariantKey>,
    pub bytes: [Option<Option<ScratchSlice>>; 10],
    pub depth_stencil: Option<DepthStencilFlags>,
}

/// Wire capture context for one frame. Failed writes poison subsequent records.
#[derive(Default)]
pub struct DrawWriter {
    poisoned: bool,
}

impl DrawWriter {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub const fn clear(&mut self) {
        self.poisoned = false;
    }

    /// Encode varying draw inputs.
    ///
    /// # Errors
    /// Returns malformed-field, size, allocation, or previous capture errors.
    pub fn encode_draw(
        &mut self,
        draw: &DrawOp,
        writer: &mut WireWriter<'_>,
    ) -> Result<(), WireError> {
        if self.poisoned {
            return Err(WireError::InvalidValue);
        }
        let result = write_draw(draw, writer);
        self.poisoned = result.is_err();
        result
    }

    /// Capture only changed state, directly from its API builder output.
    ///
    /// # Errors
    /// Returns malformed-field, size, allocation, or previous capture errors.
    pub fn encode_snapshot_delta(
        &mut self,
        delta: &SnapshotDelta<'_>,
        writer: &mut WireWriter<'_>,
    ) -> Result<(), WireError> {
        if self.poisoned {
            return Err(WireError::InvalidValue);
        }
        let result = write_snapshot_delta(delta, writer);
        self.poisoned = result.is_err();
        result
    }
}

fn write_snapshot_delta(
    delta: &SnapshotDelta<'_>,
    writer: &mut WireWriter<'_>,
) -> Result<(), WireError> {
    let mut mask = u32::from(delta.render_state.is_some())
        | (u32::from(delta.stages.is_some()) << 1)
        | (u32::from(delta.attrs.is_some()) << 2)
        | (u32::from(delta.vs.is_some()) << 3)
        | (u32::from(delta.ps.is_some()) << 4)
        | (u32::from(delta.variant.is_some()) << 5)
        | (u32::from(delta.depth_stencil.is_some()) << 16);
    for (index, value) in delta.bytes.iter().enumerate() {
        mask |= u32::from(value.is_some()) << (index + 6);
    }
    writer.u32(mask)?;
    if let Some(value) = delta.render_state {
        value.write_wire(writer)?;
    }
    if let Some((mask, values)) = delta.stages {
        if mask.count_ones() as usize != values.len() {
            return Err(WireError::InvalidValue);
        }
        writer.u16(mask)?;
        for value in values {
            value.write_wire(writer)?;
        }
    }
    if let Some(value) = &delta.attrs {
        if value.attrs.len() > 16 {
            return Err(WireError::InvalidValue);
        }
        writer.u32(u32::try_from(value.attrs.len()).map_err(|_| WireError::TooLarge)?)?;
        for attr in value.attrs {
            attr.write_wire(writer)?;
        }
        value.extents.write_wire(writer)?;
        value.used_streams.write_wire(writer)?;
        value.vdecl_hash.write_wire(writer)?;
    }
    if let Some(value) = delta.vs {
        value.write_wire(writer)?;
    }
    if let Some(value) = delta.ps {
        value.write_wire(writer)?;
    }
    if let Some(value) = delta.variant {
        value.write_wire(writer)?;
    }
    for value in delta.bytes.iter().flatten() {
        write_optional_bytes(*value, writer)?;
    }
    if let Some(value) = delta.depth_stencil {
        value.write_wire(writer)?;
    }
    Ok(())
}

/// Native decoder for one retained frame lease.
///
/// Decoded tokens borrow native scratch and leased PE byte ranges through replay.
pub struct DrawReader {
    current: CurrentSnapshot,
    poisoned: bool,
}

impl DrawReader {
    /// Establish the immutable backing contract for decoded byte ranges.
    ///
    /// # Safety
    /// Every encoded nonempty byte range must name initialized immutable storage
    /// retained through submit replay. Keep every arena passed to decoding alive
    /// and unchanged until all returned tokens and their copies are forgotten.
    /// The contract applies to every frame after `clear` as well.
    #[must_use]
    pub const unsafe fn new() -> Self {
        Self {
            current: CurrentSnapshot::EMPTY,
            poisoned: false,
        }
    }

    pub const fn clear(&mut self) {
        self.current = CurrentSnapshot::EMPTY;
        self.poisoned = false;
    }

    /// Decode a draw without copying guest bytes.
    ///
    /// # Errors
    /// Returns malformed-field or previous decode errors.
    pub fn decode_draw(&mut self, reader: &mut WireReader<'_>) -> Result<DrawOp, WireError> {
        if self.poisoned {
            return Err(WireError::InvalidValue);
        }
        let result = read_draw(reader);
        self.poisoned = result.is_err();
        result
    }

    /// Apply changed state and retain one immutable native snapshot for replay.
    ///
    /// # Errors
    /// Returns malformed-field or previous decode errors.
    pub fn decode_snapshot_delta(
        &mut self,
        reader: &mut WireReader<'_>,
        scratch: &mut ScratchArena,
    ) -> Result<CurrentSnapshotPtr, WireError> {
        if self.poisoned {
            return Err(WireError::InvalidValue);
        }
        let result = self.read_snapshot_delta(reader, scratch);
        self.poisoned = result.is_err();
        result
    }

    fn read_snapshot_delta(
        &mut self,
        reader: &mut WireReader<'_>,
        scratch: &mut ScratchArena,
    ) -> Result<CurrentSnapshotPtr, WireError> {
        let mask = reader.u32()?;
        if mask & !0x1ffff != 0 {
            return Err(WireError::InvalidValue);
        }
        if mask & 1 != 0 {
            let ptr = store(scratch, RenderStateSnapshot::read_wire(reader)?);
            // SAFETY: the frame owner retains initialized native scratch through replay.
            self.current.render_state = Some(unsafe { RenderStatePtr::new(ptr) });
        }
        if mask & 2 != 0 {
            let stage_mask = reader.u16()?;
            let values =
                read_arena_slice::<StageBinding>(reader, scratch, stage_mask.count_ones())?;
            // SAFETY: every mask-sized element is initialized in ascending stage order;
            // the frame retains this final arena storage unchanged through replay.
            self.current.stage_bindings =
                Some(unsafe { StageBindingsPtr::from_raw_parts(stage_mask, values) });
        }
        if mask & 4 != 0 {
            let count = reader.u32()?;
            if count > 16 {
                return Err(WireError::InvalidValue);
            }
            let ptr = read_arena_slice::<VertexAttrDesc>(reader, scratch, count)?;
            let extents = WireValue::read_wire(reader)?;
            let used_streams = WireValue::read_wire(reader)?;
            let vdecl_hash = WireValue::read_wire(reader)?;
            // SAFETY: the frame retains the initialized descriptor array through replay.
            self.current.attrs =
                Some(unsafe { AttrSnapshot::new(ptr, count, extents, used_streams, vdecl_hash) });
        }
        if mask & 8 != 0 {
            let ptr = store(scratch, VsSource::read_wire(reader)?);
            // SAFETY: the frame retains the initialized source in scratch through replay.
            self.current.vs = Some(unsafe { VsSourcePtr::new(ptr) });
        }
        if mask & 16 != 0 {
            let ptr = store(scratch, PsSource::read_wire(reader)?);
            // SAFETY: the frame retains the initialized source in scratch through replay.
            self.current.ps = Some(unsafe { PsSourcePtr::new(ptr) });
        }
        if mask & 32 != 0 {
            self.current.variant = Some(WireValue::read_wire(reader)?);
        }
        let byte_fields = [
            &mut self.current.vs_constants,
            &mut self.current.ps_constants,
            &mut self.current.alpha_ref_bytes,
            &mut self.current.fog_color_bytes,
            &mut self.current.bump_env_bytes,
            &mut self.current.vs_int_const_bytes,
            &mut self.current.vs_bool_const_bytes,
            &mut self.current.ps_int_const_bytes,
            &mut self.current.ps_bool_const_bytes,
            &mut self.current.vs_draw_bytes,
        ];
        for (index, value) in byte_fields.into_iter().enumerate() {
            if mask & (1 << (index + 6)) != 0 {
                *value = read_optional_bytes(reader)?;
            }
        }
        if mask & (1 << 16) != 0 {
            self.current.depth_stencil = WireValue::read_wire(reader)?;
        }
        // SAFETY: CurrentSnapshot contains only trivial-Drop tokens and scalar values;
        // native scratch and every borrowed PE range remain retained through replay.
        let ptr = unsafe { scratch.alloc_from(&self.current) };
        let ptr = NonNull::new(ptr).ok_or(WireError::InvalidValue)?;
        // SAFETY: the initialized snapshot and its referents remain live through replay.
        Ok(unsafe { CurrentSnapshotPtr::new(ptr) })
    }
}

// Decode directly into the final native array. A failed element leaves an unreachable
// initialized prefix, which needs no destruction and is reclaimed with the frame arena.
fn read_arena_slice<T: WireValue>(
    reader: &mut WireReader<'_>,
    scratch: &mut ScratchArena,
    count: u32,
) -> Result<NonNull<T>, WireError> {
    const {
        assert!(!std::mem::needs_drop::<T>());
        assert!(std::mem::align_of::<T>() <= 16);
    }
    if count == 0 {
        return Ok(NonNull::dangling());
    }
    let destination = scratch.alloc_uninit_slice::<T>(count as usize);
    for index in 0..count {
        let value = T::read_wire(reader)?;
        // SAFETY: index is strictly within the count aligned slots reserved above.
        let slot = unsafe { destination.add(index as usize) };
        // SAFETY: each exclusive slot is initialized exactly once before the
        // completed array is exposed to any reader.
        unsafe { slot.write(value) };
    }
    NonNull::new(destination).ok_or(WireError::InvalidValue)
}

fn store<T>(scratch: &mut ScratchArena, value: T) -> NonNull<T> {
    const {
        assert!(!std::mem::needs_drop::<T>());
    }
    let ptr = scratch.alloc_uninit::<T>();
    // SAFETY: the arena reserved aligned, exclusive space for one T.
    unsafe { ptr.write(value) };
    NonNull::new(ptr).expect("arena allocation is non-null")
}

fn write_optional_bytes(
    value: Option<ScratchSlice>,
    writer: &mut WireWriter<'_>,
) -> Result<(), WireError> {
    value.is_some().write_wire(writer)?;
    if let Some(value) = value {
        write_scratch_slice(value, writer)?;
    }
    Ok(())
}

fn read_optional_bytes(reader: &mut WireReader<'_>) -> Result<Option<ScratchSlice>, WireError> {
    if bool::read_wire(reader)? {
        read_scratch_slice(reader).map(Some)
    } else {
        Ok(None)
    }
}

/// Encode a borrowed byte range without copying its contents.
///
/// # Errors
///
/// Returns the wire writer's allocation or size error.
pub fn write_scratch_slice(
    value: ScratchSlice,
    writer: &mut WireWriter<'_>,
) -> Result<(), WireError> {
    let (address, length) = value.as_raw();
    writer.u64(address)?;
    writer.u32(length)
}

/// Decode a borrowed byte range from an explicitly trusted frame reader.
///
/// # Errors
///
/// Returns an error for truncated, untrusted, null or overflowing nonempty ranges.
pub fn read_scratch_slice(reader: &mut WireReader<'_>) -> Result<ScratchSlice, WireError> {
    let address = reader.u64()?;
    let length = reader.u32()?;
    if length == 0 {
        return Ok(ScratchSlice::EMPTY);
    }
    if !reader.permits_range(address, u64::from(length)) {
        return Err(WireError::InvalidValue);
    }
    let address = usize::try_from(address).map_err(|_| WireError::InvalidValue)?;
    address
        .checked_add(length as usize)
        .ok_or(WireError::InvalidValue)?;
    if usize::try_from(length).map_err(|_| WireError::TooLarge)? > isize::MAX as usize {
        return Err(WireError::InvalidValue);
    }
    let ptr = NonNull::new(address as *mut u8).ok_or(WireError::InvalidValue)?;
    // SAFETY: DrawReader construction requires immutable, retained storage for every wire range.
    Ok(unsafe { ScratchSlice::from_raw_parts(ptr, length) })
}

fn write_stream(value: &StreamBinding, writer: &mut WireWriter<'_>) -> Result<(), WireError> {
    value.stream.write_wire(writer)?;
    value.buffer_id.write_wire(writer)?;
    writer.u64(u64::try_from(value.backing_ptr).map_err(|_| WireError::TooLarge)?)?;
    writer.u64(u64::try_from(value.backing_len).map_err(|_| WireError::TooLarge)?)?;
    value.backing_generation.write_wire(writer)?;
    value.offset.write_wire(writer)?;
    value.stride.write_wire(writer)?;
    value.freq.write_wire(writer)
}

fn read_address(reader: &mut WireReader<'_>) -> Result<usize, WireError> {
    let address = reader.u64()?;
    if address != 0 && !reader.has_trusted_addresses() {
        return Err(WireError::InvalidValue);
    }
    usize::try_from(address).map_err(|_| WireError::InvalidValue)
}

fn read_stream(reader: &mut WireReader<'_>) -> Result<StreamBinding, WireError> {
    Ok(StreamBinding {
        stream: WireValue::read_wire(reader)?,
        buffer_id: WireValue::read_wire(reader)?,
        backing_ptr: read_address(reader)?,
        backing_len: usize::try_from(reader.u64()?).map_err(|_| WireError::TooLarge)?,
        backing_generation: WireValue::read_wire(reader)?,
        offset: WireValue::read_wire(reader)?,
        stride: WireValue::read_wire(reader)?,
        freq: WireValue::read_wire(reader)?,
    })
}

fn write_draw(draw: &DrawOp, writer: &mut WireWriter<'_>) -> Result<(), WireError> {
    draw.metal_prim.write_wire(writer)?;
    match &draw.vertex_source {
        VertexSource::Up {
            bytes,
            size,
            stride,
        } => {
            writer.u8(0)?;
            write_scratch_slice(*bytes, writer)?;
            size.write_wire(writer)?;
            stride.write_wire(writer)?;
        }
        VertexSource::Bound {
            first,
            extra,
            stream0_freq,
        } => {
            writer.u8(1)?;
            write_stream(first, writer)?;
            writer.u32(u32::try_from(extra.len()).map_err(|_| WireError::TooLarge)?)?;
            for stream in extra {
                write_stream(stream, writer)?;
            }
            stream0_freq.write_wire(writer)?;
        }
    }
    match &draw.index_source {
        IndexSource::None {
            start_vertex,
            vertex_count,
        } => {
            writer.u8(0)?;
            start_vertex.write_wire(writer)?;
            vertex_count.write_wire(writer)?;
        }
        IndexSource::Bound {
            buffer_id,
            backing_ptr,
            backing_len,
            backing_generation,
            offset,
            index_count,
            index_type,
            base_vertex,
        } => {
            writer.u8(1)?;
            buffer_id.write_wire(writer)?;
            writer.u64(u64::try_from(*backing_ptr).map_err(|_| WireError::TooLarge)?)?;
            writer.u64(u64::try_from(*backing_len).map_err(|_| WireError::TooLarge)?)?;
            backing_generation.write_wire(writer)?;
            offset.write_wire(writer)?;
            index_count.write_wire(writer)?;
            index_type.write_wire(writer)?;
            base_vertex.write_wire(writer)?;
        }
        IndexSource::Fan {
            start_vertex,
            primitive_count,
        } => {
            writer.u8(2)?;
            start_vertex.write_wire(writer)?;
            primitive_count.write_wire(writer)?;
        }
        IndexSource::Generated {
            data,
            index_count,
            index_type,
            min_vertex,
            max_vertex,
        } => {
            writer.u8(3)?;
            write_scratch_slice(*data, writer)?;
            index_count.write_wire(writer)?;
            index_type.write_wire(writer)?;
            min_vertex.write_wire(writer)?;
            max_vertex.write_wire(writer)?;
        }
        IndexSource::Up {
            bytes,
            index_count,
            index_type,
        } => {
            writer.u8(4)?;
            write_scratch_slice(*bytes, writer)?;
            index_count.write_wire(writer)?;
            index_type.write_wire(writer)?;
        }
    }
    Ok(())
}

fn read_draw(reader: &mut WireReader<'_>) -> Result<DrawOp, WireError> {
    let metal_prim = WireValue::read_wire(reader)?;
    let vertex_source = match reader.u8()? {
        0 => VertexSource::Up {
            bytes: read_scratch_slice(reader)?,
            size: WireValue::read_wire(reader)?,
            stride: WireValue::read_wire(reader)?,
        },
        1 => {
            let first = read_stream(reader)?;
            let count = reader.u32()?;
            if count > 15 {
                return Err(WireError::InvalidValue);
            }
            let mut extra = Vec::with_capacity(count as usize);
            for _ in 0..count {
                extra.push(read_stream(reader)?);
            }
            VertexSource::Bound {
                first,
                extra: extra.into_boxed_slice(),
                stream0_freq: WireValue::read_wire(reader)?,
            }
        }
        _ => return Err(WireError::InvalidValue),
    };
    let index_source = read_indices(reader)?;
    validate_draw(&vertex_source, &index_source)?;
    // Bound resources remain numeric descriptors here. Packet validation checks their
    // exact published identities before replay; only ScratchSlice forms borrowed bytes.
    Ok(DrawOp {
        metal_prim,
        vertex_source,
        index_source,
    })
}

fn read_indices(reader: &mut WireReader<'_>) -> Result<IndexSource, WireError> {
    Ok(match reader.u8()? {
        0 => IndexSource::None {
            start_vertex: WireValue::read_wire(reader)?,
            vertex_count: WireValue::read_wire(reader)?,
        },
        1 => IndexSource::Bound {
            buffer_id: WireValue::read_wire(reader)?,
            backing_ptr: read_address(reader)?,
            backing_len: usize::try_from(reader.u64()?).map_err(|_| WireError::TooLarge)?,
            backing_generation: WireValue::read_wire(reader)?,
            offset: WireValue::read_wire(reader)?,
            index_count: WireValue::read_wire(reader)?,
            index_type: WireValue::read_wire(reader)?,
            base_vertex: WireValue::read_wire(reader)?,
        },
        2 => IndexSource::Fan {
            start_vertex: WireValue::read_wire(reader)?,
            primitive_count: WireValue::read_wire(reader)?,
        },
        3 => IndexSource::Generated {
            data: read_scratch_slice(reader)?,
            index_count: WireValue::read_wire(reader)?,
            index_type: WireValue::read_wire(reader)?,
            min_vertex: WireValue::read_wire(reader)?,
            max_vertex: WireValue::read_wire(reader)?,
        },
        4 => IndexSource::Up {
            bytes: read_scratch_slice(reader)?,
            index_count: WireValue::read_wire(reader)?,
            index_type: WireValue::read_wire(reader)?,
        },
        _ => return Err(WireError::InvalidValue),
    })
}

macro_rules! source_codec {
    ($source:ty { $($tag:literal => $variant:ident { $($field:ident),+ $(,)? }),+ $(,)? }) => {
        impl WireValue for $source {
            fn write_wire(&self, writer: &mut WireWriter<'_>) -> Result<(), WireError> {
                match self { $(Self::$variant { $($field),+ } => {
                    writer.u8($tag)?;
                    $($field.write_wire(writer)?;)+
                }),+ }
                Ok(())
            }
            fn read_wire(reader: &mut WireReader<'_>) -> Result<Self, WireError> {
                match reader.u8()? {
                    $($tag => Ok(Self::$variant { $($field: WireValue::read_wire(reader)?,)+ }),)+
                    _ => Err(WireError::InvalidValue),
                }
            }
        }
    };
}

source_codec!(VsSource {
    0 => Programmable { vs_id, max_const_used, uses_rel_const, provided_input_mask,
        uses_int_const, uses_bool_const, clip_plane_count, sampler_kinds },
    1 => FixedFunction { key, max_row_count },
});

source_codec!(PsSource {
    0 => Programmable { ps_id, max_const_used, uses_bump_env, uses_int_const,
        uses_bool_const, color_out_mask },
    1 => FixedFunction { key, sampled_stage_mask, constant_rows },
});

fn validate_draw(vertices: &VertexSource, indices: &IndexSource) -> Result<(), WireError> {
    match vertices {
        VertexSource::Up { bytes, size, .. } => {
            if usize::try_from(*size).map_err(|_| WireError::TooLarge)? > bytes.as_slice().len() {
                return Err(WireError::InvalidValue);
            }
        }
        VertexSource::Bound { first, extra, .. } => {
            let mut used = 0u16;
            for stream in std::iter::once(first).chain(extra.iter()) {
                if stream.stream >= 16 || used & (1 << stream.stream) != 0 {
                    return Err(WireError::InvalidValue);
                }
                used |= 1 << stream.stream;
                validate_backing(stream.backing_ptr, stream.backing_len)?;
            }
        }
    }
    validate_indices(indices)
}

fn validate_indices(indices: &IndexSource) -> Result<(), WireError> {
    match indices {
        IndexSource::Bound {
            backing_ptr,
            backing_len,
            ..
        } => {
            validate_backing(*backing_ptr, *backing_len)?;
        }
        IndexSource::Up {
            bytes,
            index_count,
            index_type,
        } => {
            validate_index_bytes(*index_count, *index_type, bytes.as_slice().len())?;
        }
        IndexSource::Generated {
            data,
            index_count,
            index_type,
            min_vertex,
            max_vertex,
        } => {
            validate_index_bytes(*index_count, *index_type, data.as_raw().1 as usize)?;
            if min_vertex > max_vertex {
                return Err(WireError::InvalidValue);
            }
        }
        IndexSource::None { .. } | IndexSource::Fan { .. } => {}
    }
    Ok(())
}

fn validate_backing(address: usize, length: usize) -> Result<(), WireError> {
    // A staged WRITEONLY buffer can release its CPU allocation after upload. Its
    // zero address and generation still carry the padded GPU buffer length.
    // Bound descriptors stay numeric here; native staged buffers ignore absent
    // CPU backing and use their persistent GPU allocation.
    if length > isize::MAX as usize {
        return Err(WireError::InvalidValue);
    }
    address.checked_add(length).ok_or(WireError::InvalidValue)?;
    Ok(())
}

fn validate_index_bytes(
    count: u32,
    kind: mtld3d_shared::mtl::IndexType,
    length: usize,
) -> Result<(), WireError> {
    let stride = match kind {
        mtld3d_shared::mtl::IndexType::UInt16 => 2,
        mtld3d_shared::mtl::IndexType::UInt32 => 4,
    };
    let required = usize::try_from(count)
        .map_err(|_| WireError::TooLarge)?
        .checked_mul(stride)
        .ok_or(WireError::TooLarge)?;
    if required > length {
        return Err(WireError::InvalidValue);
    }
    Ok(())
}

/// Validate a draw without constructing stream storage or borrowing backing bytes.
///
/// # Errors
/// Rejects malformed fields, invalid byte extents, or a backing identity rejected by
/// the caller's retained-inventory check.
pub fn validate_wire_draw(
    reader: &mut WireReader<'_>,
    mut backing: impl FnMut(u64, u64) -> Result<(), WireError>,
) -> Result<(), WireError> {
    mtld3d_shared::mtl::PrimitiveType::read_wire(reader)?;
    match reader.u8()? {
        0 => {
            let bytes = read_scratch_slice(reader)?;
            let size = reader.u32()?;
            reader.u32()?;
            if size > bytes.as_raw().1 {
                return Err(WireError::InvalidValue);
            }
        }
        1 => {
            let mut used = 0u16;
            let mut stream = |reader: &mut WireReader<'_>| -> Result<(), WireError> {
                let value = read_stream(reader)?;
                if value.stream >= 16 || used & (1 << value.stream) != 0 {
                    return Err(WireError::InvalidValue);
                }
                used |= 1 << value.stream;
                validate_backing(value.backing_ptr, value.backing_len)?;
                backing(value.backing_ptr as u64, value.backing_len as u64)
            };
            stream(reader)?;
            let count = reader.u32()?;
            if count > 15 {
                return Err(WireError::InvalidValue);
            }
            for _ in 0..count {
                stream(reader)?;
            }
            reader.u32()?;
        }
        _ => return Err(WireError::InvalidValue),
    }
    let indices = read_indices(reader)?;
    validate_indices(&indices)?;
    if let IndexSource::Bound {
        backing_ptr,
        backing_len,
        ..
    } = indices
    {
        backing(backing_ptr as u64, backing_len as u64)?;
    }
    Ok(())
}

/// Validate dirty snapshot values without building native snapshots or allocating scratch.
///
/// # Errors
/// Rejects malformed masks, fields, or byte ranges outside the reader's retained inventory.
pub fn validate_wire_snapshot(reader: &mut WireReader<'_>) -> Result<(), WireError> {
    let mask = reader.u32()?;
    if mask & !0x1ffff != 0 {
        return Err(WireError::InvalidValue);
    }
    if mask & 1 != 0 {
        RenderStateSnapshot::read_wire(reader)?;
    }
    if mask & 2 != 0 {
        let stages = reader.u16()?;
        for _ in 0..stages.count_ones() {
            StageBinding::read_wire(reader)?;
        }
    }
    if mask & 4 != 0 {
        let count = reader.u32()?;
        if count > 16 {
            return Err(WireError::InvalidValue);
        }
        for _ in 0..count {
            VertexAttrDesc::read_wire(reader)?;
        }
        <[u32; 16]>::read_wire(reader)?;
        reader.u16()?;
        reader.u64()?;
    }
    if mask & 8 != 0 {
        VsSource::read_wire(reader)?;
    }
    if mask & 16 != 0 {
        PsSource::read_wire(reader)?;
    }
    if mask & 32 != 0 {
        VariantKey::read_wire(reader)?;
    }
    for index in 0..10 {
        if mask & (1 << (index + 6)) != 0 {
            read_optional_bytes(reader)?;
        }
    }
    if mask & (1 << 16) != 0 {
        DepthStencilFlags::read_wire(reader)?;
    }
    Ok(())
}
