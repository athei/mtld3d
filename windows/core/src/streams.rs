//! D3D9 vertex-stream frequency: the `SetStreamSourceFreq` contract and what draws derive from it.
//!
//! A frequency word is `flags | count`. `D3DSTREAMSOURCE_INDEXEDDATA` marks a
//! per-vertex stream of an instanced draw and, on stream 0, carries the
//! instance count; `D3DSTREAMSOURCE_INSTANCEDATA` marks a per-instance stream
//! whose count is the number of instances that share one element. The
//! validation rules and the instance-count derivation follow the D3D9
//! runtime's observable behaviour.

use mtld3d_shared::{
    VertexAttrDesc,
    mtl::{VertexFormat, VertexStepFunction},
};
use mtld3d_types::{D3DSTREAMSOURCE_INDEXEDDATA, D3DSTREAMSOURCE_INSTANCEDATA, MAX_STREAMS};

use crate::pipeline_state::StreamLayout;

/// Mask of the count half of a frequency word.
///
/// Both flag bits sit above it; the runtime ignores bits 23..30.
pub const STREAM_FREQ_COUNT_MASK: u32 = 0x7F_FFFF;

/// Frequency of every stream on a fresh device: one element per vertex, no flags.
pub const STREAM_FREQ_DEFAULT: u32 = 1;

/// Why `SetStreamSourceFreq` rejects a call.
///
/// Each variant is a distinct `D3DERR_INVALIDCALL` reason the caller logs.
#[derive(Debug, PartialEq, Eq)]
pub enum StreamFreqError {
    /// `stream >= MaxStreams`.
    StreamOutOfRange,
    /// `D3DSTREAMSOURCE_INSTANCEDATA` on stream 0, the stream that carries vertices.
    InstanceDataOnStreamZero,
    /// Both flag bits set.
    BothFlags,
    /// A literal zero: neither a flag nor a count.
    Zero,
}

/// Validate a `SetStreamSourceFreq(stream, setting)` call.
///
/// Any flag with a zero count (`INDEXEDDATA | 0`, `INSTANCEDATA | 0`) is
/// accepted: the word is non-zero. A failed call leaves the stored state
/// untouched, which is the caller's job.
///
/// # Errors
///
/// The first rule the call breaks, in the order the runtime checks them.
pub const fn validate_stream_freq(stream: u32, setting: u32) -> Result<(), StreamFreqError> {
    if stream >= MAX_STREAMS {
        return Err(StreamFreqError::StreamOutOfRange);
    }
    let instanced = setting & D3DSTREAMSOURCE_INSTANCEDATA != 0;
    let indexed = setting & D3DSTREAMSOURCE_INDEXEDDATA != 0;
    if stream == 0 && instanced {
        return Err(StreamFreqError::InstanceDataOnStreamZero);
    }
    if instanced && indexed {
        return Err(StreamFreqError::BothFlags);
    }
    if setting == 0 {
        return Err(StreamFreqError::Zero);
    }
    Ok(())
}

/// Whether a frequency word marks a per-instance stream.
#[inline]
#[must_use]
pub const fn is_instance_data(setting: u32) -> bool {
    setting & D3DSTREAMSOURCE_INSTANCEDATA != 0
}

/// The count half of a frequency word.
#[inline]
#[must_use]
pub const fn stream_freq_count(setting: u32) -> u32 {
    setting & STREAM_FREQ_COUNT_MASK
}

/// Instances an indexed draw renders.
///
/// The count always comes from stream 0's frequency word, whether or not
/// stream 0 feeds the draw, and only applies when a stream the draw reads is
/// per-instance; otherwise the draw is a single instance no matter what
/// stream 0 says. `INDEXEDDATA | 0` is driver-defined on real hardware (one
/// instance, no instancing, or nothing); one instance is the choice here.
/// Non-indexed draws never instance and do not call this.
#[inline]
#[must_use]
pub const fn instance_count(stream0_freq: u32, any_used_stream_instanced: bool) -> u32 {
    if !any_used_stream_instanced {
        return 1;
    }
    let count = stream_freq_count(stream0_freq);
    if count == 0 { 1 } else { count }
}

/// The Metal step function and rate a stream's frequency word selects.
///
/// `INSTANCEDATA | n` advances one element every `n` instances; `n == 0`
/// never advances, which Metal spells as a `Constant` layout with rate 0
/// rather than `PerInstance` with rate 0. Everything else, `INDEXEDDATA`
/// included, is per-vertex.
#[inline]
#[must_use]
pub const fn stream_step(setting: u32) -> (VertexStepFunction, u32) {
    if !is_instance_data(setting) {
        return (VertexStepFunction::PerVertex, 1);
    }
    let rate = stream_freq_count(setting);
    if rate == 0 {
        (VertexStepFunction::Constant, 0)
    } else {
        (VertexStepFunction::PerInstance, rate)
    }
}

/// Bytes an instanced draw reads from a per-instance stream, from its offset.
///
/// `ceil(instances / rate) * stride`; a `Constant` stream reads one element.
/// Over-covers on overflow (`u32::MAX`), never under-covers: the value guards
/// a later overlapping upload against a draw still in flight.
#[must_use]
pub const fn instanced_stream_read_bytes(
    instances: u32,
    step: VertexStepFunction,
    rate: u32,
    stride: u32,
) -> u32 {
    let elements = match step {
        VertexStepFunction::Constant => 1,
        VertexStepFunction::PerVertex | VertexStepFunction::PerInstance => {
            if rate == 0 {
                1
            } else {
                instances.div_ceil(rate)
            }
        }
    };
    match elements.checked_mul(stride) {
        Some(bytes) => bytes,
        None => u32::MAX,
    }
}

/// The Metal layout stride before crossing attributes receive separate bindings.
///
/// A zero stride feeds one constant element. A nonzero stride is preserved;
/// [`remap_crossing_attributes`] moves offsets that do not fit into buffer bindings.
#[must_use]
pub const fn layout_stride(app_stride: u32, extent: u32) -> u32 {
    if app_stride == 0 {
        return extent;
    }
    app_stride
}

/// Source stream and byte advance of a Metal vertex buffer binding.
///
/// Copy because the eight-byte value initializes and indexes a fixed per-draw array.
#[derive(Clone, Copy)]
pub struct VertexFetchBinding {
    pub stream: u32,
    pub offset: u32,
}

/// Why a crossing attribute cannot use a second Metal buffer binding.
#[derive(Debug, PartialEq, Eq)]
pub enum VertexFetchError {
    StreamOutOfRange,
    AttributeWiderThanStride,
    UnalignedOffset,
    NoFreeSlot,
}

/// Remap crossing attributes to spare bindings without changing the vertex step.
///
/// Reserve all slots with ordinary attributes first, then assign each crossing
/// attribute an otherwise unused slot. Streams with only crossing attributes release
/// their original slot. The sixteen stream slots suffice for at most sixteen attributes
/// and cannot collide with uniforms. Bindings name original streams; layouts retain
/// their step function and instance rate. No vertex data or CPU backing is read here.
///
/// # Errors
///
/// An attribute wider than its stride, an unaligned advanced offset, or too many
/// attributes for the stream table. Discard outputs on error and log the unsupported draw.
///
/// # Panics
///
/// The internal stream-index conversion is asserted to fit in `u32`.
pub fn remap_crossing_attributes(
    attrs: &mut [VertexAttrDesc],
    layouts: &mut [StreamLayout; MAX_STREAMS as usize],
) -> Result<[VertexFetchBinding; MAX_STREAMS as usize], VertexFetchError> {
    let original = *layouts;
    let mut bindings = std::array::from_fn(|i| VertexFetchBinding {
        stream: u32::try_from(i).expect("stream index fits u32"),
        offset: 0,
    });
    let mut reserved = 0u16;
    for attr in attrs.iter() {
        let index =
            usize::try_from(attr.buffer_index).map_err(|_| VertexFetchError::StreamOutOfRange)?;
        let Some(layout) = original.get(index) else {
            return Err(VertexFetchError::StreamOutOfRange);
        };
        let width = vertex_format_bytes(attr.format);
        if width > layout.stride {
            return Err(VertexFetchError::AttributeWiderThanStride);
        }
        if attr
            .offset
            .checked_add(width)
            .is_some_and(|end| end <= layout.stride)
        {
            reserved |= 1 << index;
        } else if attr.offset % 4 != 0 {
            return Err(VertexFetchError::UnalignedOffset);
        }
    }
    for (index, layout) in layouts.iter_mut().enumerate() {
        if reserved & (1 << index) == 0 {
            *layout = StreamLayout::UNUSED;
        }
    }
    for attr in attrs.iter_mut() {
        let index =
            usize::try_from(attr.buffer_index).map_err(|_| VertexFetchError::StreamOutOfRange)?;
        if attr
            .offset
            .checked_add(vertex_format_bytes(attr.format))
            .is_some_and(|end| end <= original[index].stride)
        {
            continue;
        }
        let slot = (!reserved).trailing_zeros();
        if slot >= MAX_STREAMS {
            return Err(VertexFetchError::NoFreeSlot);
        }
        reserved |= 1 << slot;
        layouts[slot as usize] = original[index];
        bindings[slot as usize] = VertexFetchBinding {
            stream: attr.buffer_index,
            offset: attr.offset,
        };
        attr.buffer_index = slot;
        attr.offset = 0;
    }
    Ok(bindings)
}

/// Bytes required by an inline stream, including its last crossing attribute.
///
/// Keep the packed span where larger; checked arithmetic rejects impossible payloads
/// before the API borrows the user pointer.
#[must_use]
pub const fn inline_vertex_span(count: u32, stride: u32, extent: u32) -> Option<u32> {
    if count == 0 {
        return Some(0);
    }
    let Some(packed) = count.checked_mul(stride) else {
        return None;
    };
    let Some(last) = (count - 1).checked_mul(stride) else {
        return None;
    };
    let Some(end) = last.checked_add(extent) else {
        return None;
    };
    Some(if end > packed { end } else { packed })
}

/// The vertex buffer layout of a stream with a vertex buffer bound.
///
/// `app_stride` and `freq` are the stream's `SetStreamSource` stride and
/// `SetStreamSourceFreq` word, `extent` the span of the declaration elements
/// the shader consumes on it. A zero stride is D3D9's way of binding one
/// element to a whole draw: every vertex and instance reads the element at the
/// stream offset, whatever the frequency word says, which Metal spells as a
/// `Constant` layout (there is no zero-stride layout, and stepping such a
/// stream per vertex would fetch past the buffer's end). Any other stride
/// steps per the frequency word.
#[must_use]
pub const fn bound_stream_layout(app_stride: u32, extent: u32, freq: u32) -> StreamLayout {
    if app_stride == 0 {
        return StreamLayout {
            stride: extent,
            step: VertexStepFunction::Constant,
            step_rate: 0,
        };
    }
    let (step, step_rate) = stream_step(freq);
    StreamLayout {
        stride: layout_stride(app_stride, extent),
        step,
        step_rate,
    }
}

/// Storage width of the Metal vertex formats the declaration resolver emits.
const fn vertex_format_bytes(format: VertexFormat) -> u32 {
    match format {
        VertexFormat::Invalid => 0,
        VertexFormat::UChar4
        | VertexFormat::UChar4Normalized
        | VertexFormat::UChar4NormalizedBgra
        | VertexFormat::Short2
        | VertexFormat::UShort2Normalized
        | VertexFormat::Short2Normalized
        | VertexFormat::Half2
        | VertexFormat::Float => 4,
        VertexFormat::Short4
        | VertexFormat::UShort4Normalized
        | VertexFormat::Short4Normalized
        | VertexFormat::Half4
        | VertexFormat::Float2 => 8,
        VertexFormat::Float3 => 12,
        VertexFormat::Float4 => 16,
    }
}

#[cfg(test)]
mod tests;
