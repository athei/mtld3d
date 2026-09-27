//! Fixed draw payloads. Every byte is an explicit little-endian integer field.

use std::ptr::NonNull;

use mtld3d_shared::{
    encoder_wire::WireError,
    mtl::{IndexType, PrimitiveType},
};

use crate::draw_data::{DrawOp, IndexSource, ScratchSlice, StreamBinding, VertexSource};

#[repr(C, align(8))]
struct DrawPrefix {
    primitive: u8,
    vertex_kind: u8,
    index_kind: u8,
    stream_count: u8,
    stride_or_frequency: u32,
    first_or_base: u32,
    count: u32,
}

#[repr(C, align(8))]
pub struct StreamRecord {
    pub buffer: u64,
    pub address: u64,
    pub length: u64,
    pub generation: u64,
    pub offset: u32,
    pub stride: u32,
    pub frequency: u32,
    pub stream: u8,
    pub reserved: [u8; 3],
}

#[repr(C, align(8))]
pub struct VertexBytes {
    pub address: u64,
    pub length: u32,
    pub size: u32,
}

#[repr(C, align(8))]
pub struct IndexBuffer {
    pub buffer: u64,
    pub address: u64,
    pub length: u64,
    pub generation: u64,
    pub offset: u32,
    pub kind: u8,
    pub reserved: [u8; 3],
}

#[repr(C, align(8))]
pub struct IndexBytes {
    pub address: u64,
    pub length: u32,
    pub maximum: u32,
    pub kind: u8,
    pub reserved: [u8; 7],
}

// The supported targets are little-endian; explicit fields cover the full record with no
// implicit padding. Integer-only records accept all input bit patterns before validation.
const _: () = {
    assert!(cfg!(target_endian = "little"));
    assert!(size_of::<DrawPrefix>() == 16 && align_of::<DrawPrefix>() == 8);
    assert!(std::mem::offset_of!(DrawPrefix, primitive) == 0);
    assert!(std::mem::offset_of!(DrawPrefix, vertex_kind) == 1);
    assert!(std::mem::offset_of!(DrawPrefix, index_kind) == 2);
    assert!(std::mem::offset_of!(DrawPrefix, stream_count) == 3);
    assert!(std::mem::offset_of!(DrawPrefix, stride_or_frequency) == 4);
    assert!(std::mem::offset_of!(DrawPrefix, first_or_base) == 8);
    assert!(std::mem::offset_of!(DrawPrefix, count) == 12);
    assert!(size_of::<StreamRecord>() == 48 && align_of::<StreamRecord>() == 8);
    assert!(std::mem::offset_of!(StreamRecord, buffer) == 0);
    assert!(std::mem::offset_of!(StreamRecord, address) == 8);
    assert!(std::mem::offset_of!(StreamRecord, length) == 16);
    assert!(std::mem::offset_of!(StreamRecord, generation) == 24);
    assert!(std::mem::offset_of!(StreamRecord, offset) == 32);
    assert!(std::mem::offset_of!(StreamRecord, stride) == 36);
    assert!(std::mem::offset_of!(StreamRecord, frequency) == 40);
    assert!(std::mem::offset_of!(StreamRecord, stream) == 44);
    assert!(std::mem::offset_of!(StreamRecord, reserved) == 45);
    assert!(size_of::<VertexBytes>() == 16 && align_of::<VertexBytes>() == 8);
    assert!(std::mem::offset_of!(VertexBytes, address) == 0);
    assert!(std::mem::offset_of!(VertexBytes, length) == 8);
    assert!(std::mem::offset_of!(VertexBytes, size) == 12);
    assert!(size_of::<IndexBuffer>() == 40 && align_of::<IndexBuffer>() == 8);
    assert!(std::mem::offset_of!(IndexBuffer, buffer) == 0);
    assert!(std::mem::offset_of!(IndexBuffer, address) == 8);
    assert!(std::mem::offset_of!(IndexBuffer, length) == 16);
    assert!(std::mem::offset_of!(IndexBuffer, generation) == 24);
    assert!(std::mem::offset_of!(IndexBuffer, offset) == 32);
    assert!(std::mem::offset_of!(IndexBuffer, kind) == 36);
    assert!(std::mem::offset_of!(IndexBuffer, reserved) == 37);
    assert!(size_of::<IndexBytes>() == 24 && align_of::<IndexBytes>() == 8);
    assert!(std::mem::offset_of!(IndexBytes, address) == 0);
    assert!(std::mem::offset_of!(IndexBytes, length) == 8);
    assert!(std::mem::offset_of!(IndexBytes, maximum) == 12);
    assert!(std::mem::offset_of!(IndexBytes, kind) == 16);
    assert!(std::mem::offset_of!(IndexBytes, reserved) == 17);
};

/// An integer-only draw record with explicitly initialized padding.
///
/// # Safety
///
/// All bit patterns are valid, and every byte is an explicit initialized field.
unsafe trait DrawPod {}
// SAFETY: the layout assertions cover integer-only records and explicit reserved bytes.
unsafe impl DrawPod for DrawPrefix {}
// SAFETY: the layout assertions cover integer-only records and explicit reserved bytes.
unsafe impl DrawPod for StreamRecord {}
// SAFETY: the layout assertions cover integer-only records and explicit reserved bytes.
unsafe impl DrawPod for VertexBytes {}
// SAFETY: the layout assertions cover integer-only records and explicit reserved bytes.
unsafe impl DrawPod for IndexBuffer {}
// SAFETY: the layout assertions cover integer-only records and explicit reserved bytes.
unsafe impl DrawPod for IndexBytes {}

fn put<T: DrawPod>(destination: &mut [u8], at: &mut usize, value: T) {
    let bytes = &mut destination[*at..*at + size_of::<T>()];
    // SAFETY: the checked destination extent fits T; DrawPod has no implicit padding
    // or ownership. Unaligned stores also support the generic test-record adapter.
    unsafe { bytes.as_mut_ptr().cast::<T>().write_unaligned(value) };
    *at += size_of::<T>();
}

const fn stream_record(value: &StreamBinding) -> StreamRecord {
    StreamRecord {
        buffer: value.buffer_id.raw(),
        address: value.backing_ptr as u64,
        length: value.backing_len as u64,
        generation: value.backing_generation,
        offset: value.offset,
        stride: value.stride,
        frequency: value.freq,
        stream: value.stream,
        reserved: [0; 3],
    }
}

pub(super) const fn payload_size(draw: &DrawOp) -> Result<usize, WireError> {
    let vertices = match &draw.vertex_source {
        VertexSource::Up { .. } => 16,
        VertexSource::Bound { extra, .. } => {
            if extra.len() > 15 {
                return Err(WireError::InvalidValue);
            }
            (1 + extra.len()) * 48
        }
    };
    let indices = match draw.index_source {
        IndexSource::None { .. } | IndexSource::Fan { .. } => 0,
        IndexSource::Bound { .. } => 40,
        IndexSource::Generated { .. } | IndexSource::Up { .. } => 24,
    };
    Ok(16 + vertices + indices)
}

pub(super) fn write_into(draw: &DrawOp, destination: &mut [u8]) -> Result<(), WireError> {
    if destination.len() != payload_size(draw)? {
        return Err(WireError::InvalidValue);
    }
    let (vertex_kind, stream_count, stride_or_frequency) = match &draw.vertex_source {
        VertexSource::Up { stride, .. } => (0, 0, *stride),
        VertexSource::Bound {
            extra,
            stream0_freq,
            ..
        } => (
            1,
            u8::try_from(extra.len() + 1).map_err(|_| WireError::InvalidValue)?,
            *stream0_freq,
        ),
    };
    let (index_kind, first_or_base, count) = match draw.index_source {
        IndexSource::None {
            start_vertex,
            vertex_count,
        } => (0, start_vertex, vertex_count),
        IndexSource::Bound {
            base_vertex,
            index_count,
            ..
        } => (1, base_vertex.cast_unsigned(), index_count),
        IndexSource::Fan {
            start_vertex,
            primitive_count,
        } => (2, start_vertex, primitive_count),
        IndexSource::Generated {
            min_vertex,
            index_count,
            ..
        } => (3, min_vertex, index_count),
        IndexSource::Up { index_count, .. } => (4, 0, index_count),
    };
    let mut at = 0;
    put(
        destination,
        &mut at,
        DrawPrefix {
            primitive: draw.metal_prim as u8,
            vertex_kind,
            index_kind,
            stream_count,
            stride_or_frequency,
            first_or_base,
            count,
        },
    );
    match &draw.vertex_source {
        VertexSource::Up { bytes, size, .. } => {
            let (address, length) = bytes.as_raw();
            put(
                destination,
                &mut at,
                VertexBytes {
                    address,
                    length,
                    size: *size,
                },
            );
        }
        VertexSource::Bound { first, extra, .. } => {
            put(destination, &mut at, stream_record(first));
            for value in extra {
                put(destination, &mut at, stream_record(&value));
            }
        }
    }
    match &draw.index_source {
        IndexSource::Bound {
            buffer_id,
            backing_ptr,
            backing_len,
            backing_generation,
            offset,
            index_type,
            ..
        } => put(
            destination,
            &mut at,
            IndexBuffer {
                buffer: buffer_id.raw(),
                address: *backing_ptr as u64,
                length: *backing_len as u64,
                generation: *backing_generation,
                offset: *offset,
                kind: *index_type as u8,
                reserved: [0; 3],
            },
        ),
        IndexSource::Generated {
            data,
            index_type,
            max_vertex,
            ..
        } => {
            let (address, length) = data.as_raw();
            put(
                destination,
                &mut at,
                IndexBytes {
                    address,
                    length,
                    maximum: *max_vertex,
                    kind: *index_type as u8,
                    reserved: [0; 7],
                },
            );
        }
        IndexSource::Up {
            bytes, index_type, ..
        } => {
            let (address, length) = bytes.as_raw();
            put(
                destination,
                &mut at,
                IndexBytes {
                    address,
                    length,
                    maximum: 0,
                    kind: *index_type as u8,
                    reserved: [0; 7],
                },
            );
        }
        IndexSource::None { .. } | IndexSource::Fan { .. } => {}
    }
    Ok(())
}

/// Borrowed fixed draw fields. The containing command allocation owns every record.
pub struct DrawView<'a> {
    prefix: &'a DrawPrefix,
    vertices: &'a [u8],
    indices: &'a [u8],
}

/// Vertex input borrowed directly from a fixed draw command.
pub enum VertexView<'a> {
    Up {
        record: &'a VertexBytes,
        stride: u32,
    },
    Bound {
        records: &'a [StreamRecord],
        stream0_freq: u32,
    },
}

/// Index input borrowed directly from a fixed draw command.
pub enum IndexView<'a> {
    None {
        start_vertex: u32,
        vertex_count: u32,
    },
    Bound {
        record: &'a IndexBuffer,
        index_count: u32,
        base_vertex: i32,
    },
    Fan {
        start_vertex: u32,
        primitive_count: u32,
    },
    Generated {
        record: &'a IndexBytes,
        index_count: u32,
        min_vertex: u32,
    },
    Up {
        record: &'a IndexBytes,
        index_count: u32,
    },
}

fn fixed_ref<T: DrawPod>(bytes: &[u8]) -> Result<&T, WireError> {
    // SAFETY: DrawPod permits every bit pattern; align_to returns only aligned complete values.
    let (prefix, values, suffix) = unsafe { bytes.align_to::<T>() };
    if !prefix.is_empty() || !suffix.is_empty() || values.len() != 1 {
        return Err(WireError::InvalidValue);
    }
    Ok(&values[0])
}

impl<'a> DrawView<'a> {
    /// Borrow a draw from its immutable aligned command payload.
    ///
    /// # Errors
    /// Returns an error for an invalid tag, record count, size or alignment.
    pub fn new(bytes: &'a [u8]) -> Result<Self, WireError> {
        let (prefix, rest) = bytes
            .split_at_checked(size_of::<DrawPrefix>())
            .ok_or(WireError::Truncated)?;
        let prefix: &DrawPrefix = fixed_ref(prefix)?;
        let vertex_bytes = match prefix.vertex_kind {
            0 if prefix.stream_count == 0 => size_of::<VertexBytes>(),
            1 if (1..=16).contains(&prefix.stream_count) => {
                usize::from(prefix.stream_count) * size_of::<StreamRecord>()
            }
            _ => return Err(WireError::InvalidValue),
        };
        let (vertices, indices) = rest
            .split_at_checked(vertex_bytes)
            .ok_or(WireError::Truncated)?;
        let value = Self {
            prefix,
            vertices,
            indices,
        };
        value.metal_primitive()?;
        value.vertices()?;
        value.indices()?;
        Ok(value)
    }

    /// Read the primitive type.
    ///
    /// # Errors
    ///
    /// Returns an error for an unknown primitive type.
    pub fn metal_primitive(&self) -> Result<PrimitiveType, WireError> {
        PrimitiveType::from_repr(u32::from(self.prefix.primitive)).ok_or(WireError::InvalidValue)
    }

    /// Borrow the vertex records.
    ///
    /// # Errors
    ///
    /// Returns an error for an incomplete or unaligned vertex record.
    pub fn vertices(&self) -> Result<VertexView<'a>, WireError> {
        if self.prefix.vertex_kind == 0 {
            Ok(VertexView::Up {
                record: fixed_ref(self.vertices)?,
                stride: self.prefix.stride_or_frequency,
            })
        } else {
            // SAFETY: StreamRecord has only integer fields and explicit initialized padding.
            let (prefix, records, suffix) = unsafe { self.vertices.align_to::<StreamRecord>() };
            if !prefix.is_empty() || !suffix.is_empty() {
                return Err(WireError::InvalidValue);
            }
            Ok(VertexView::Bound {
                records,
                stream0_freq: self.prefix.stride_or_frequency,
            })
        }
    }

    /// Borrow the index records.
    ///
    /// # Errors
    ///
    /// Returns an error for an incomplete, unaligned or unknown index record.
    pub fn indices(&self) -> Result<IndexView<'a>, WireError> {
        let first = self.prefix.first_or_base;
        let count = self.prefix.count;
        match self.prefix.index_kind {
            0 if self.indices.is_empty() => Ok(IndexView::None {
                start_vertex: first,
                vertex_count: count,
            }),
            1 => Ok(IndexView::Bound {
                record: fixed_ref(self.indices)?,
                index_count: count,
                base_vertex: first.cast_signed(),
            }),
            2 if self.indices.is_empty() => Ok(IndexView::Fan {
                start_vertex: first,
                primitive_count: count,
            }),
            3 => Ok(IndexView::Generated {
                record: fixed_ref(self.indices)?,
                index_count: count,
                min_vertex: first,
            }),
            4 => Ok(IndexView::Up {
                record: fixed_ref(self.indices)?,
                index_count: count,
            }),
            _ => Err(WireError::InvalidValue),
        }
    }
}

impl VertexBytes {
    /// Borrow the already retained UP capture for a native draw.
    ///
    /// # Panics
    /// Panics if a nonempty capture has a null address.
    ///
    /// # Safety
    /// The authentic packet must retain address..address+length as initialized immutable bytes.
    #[must_use]
    pub const unsafe fn bytes(&self) -> ScratchSlice {
        if self.length == 0 {
            return ScratchSlice::EMPTY;
        }
        let address = NonNull::new(self.address as *mut u8).expect("retained UP allocation");
        // SAFETY: the packet caller guarantees the exact immutable capture lifetime and extent.
        unsafe { ScratchSlice::from_raw_parts(address, self.length) }
    }
}

impl IndexBytes {
    /// Borrow the already retained index capture for a native draw.
    ///
    /// # Panics
    /// Panics if a nonempty capture has a null address.
    ///
    /// # Safety
    /// The authentic packet must retain address..address+length as initialized immutable bytes.
    #[must_use]
    pub const unsafe fn bytes(&self) -> ScratchSlice {
        if self.length == 0 {
            return ScratchSlice::EMPTY;
        }
        let address = NonNull::new(self.address as *mut u8).expect("retained index allocation");
        // SAFETY: the packet caller guarantees the exact immutable capture lifetime and extent.
        unsafe { ScratchSlice::from_raw_parts(address, self.length) }
    }

    /// Read the index element type.
    ///
    /// # Errors
    ///
    /// Returns an error for an unknown index element type.
    pub fn index_type(&self) -> Result<IndexType, WireError> {
        IndexType::from_repr(u32::from(self.kind)).ok_or(WireError::InvalidValue)
    }
}

impl IndexBuffer {
    /// Read the index element type.
    ///
    /// # Errors
    ///
    /// Returns an error for an unknown index element type.
    pub fn index_type(&self) -> Result<IndexType, WireError> {
        IndexType::from_repr(u32::from(self.kind)).ok_or(WireError::InvalidValue)
    }
}

/// One stream's input, borrowed from its command rather than rebuilt as an owned binding.
pub enum StreamViewFeed<'a> {
    Inline { stride: u32 },
    Buffer(&'a StreamRecord),
    Null,
}

impl<'a> VertexView<'a> {
    pub fn bindings(&self) -> core::slice::Iter<'a, StreamRecord> {
        match self {
            Self::Up { .. } => [].iter(),
            Self::Bound { records, .. } => records.iter(),
        }
    }

    #[must_use]
    pub fn feed(&self, stream: u32) -> StreamViewFeed<'a> {
        match self {
            Self::Up { stride, .. } if stream == 0 => StreamViewFeed::Inline { stride: *stride },
            Self::Up { .. } => StreamViewFeed::Null,
            Self::Bound { records, .. } => records
                .iter()
                .find(|record| u32::from(record.stream) == stream)
                .map_or(StreamViewFeed::Null, StreamViewFeed::Buffer),
        }
    }
}

/// Vertex layouts derived directly from borrowed command stream records.
#[must_use]
pub fn stream_layouts_view(
    source: &VertexView<'_>,
    attrs: &crate::draw_data::AttrSnapshot,
) -> [crate::pipeline_state::StreamLayout; mtld3d_types::MAX_STREAMS as usize] {
    use mtld3d_shared::mtl::VertexStepFunction;

    use crate::{
        pipeline_state::StreamLayout,
        streams::{bound_stream_layout, layout_stride},
    };
    crate::draw_data::stream_layouts_with(attrs, |stream, extent| match source.feed(stream) {
        StreamViewFeed::Inline { stride } => StreamLayout {
            stride: layout_stride(stride, extent),
            step: VertexStepFunction::PerVertex,
            step_rate: 1,
        },
        StreamViewFeed::Buffer(record) => {
            bound_stream_layout(record.stride, extent, record.frequency)
        }
        StreamViewFeed::Null => StreamLayout {
            stride: extent,
            step: VertexStepFunction::Constant,
            step_rate: 0,
        },
    })
}

#[cfg(test)]
mod tests;
