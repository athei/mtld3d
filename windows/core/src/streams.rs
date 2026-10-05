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
use xxhash_rust::xxh3::xxh3_64_with_seed;

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

/// The stride a stream's vertex buffer layout steps by.
///
/// The application's stride, or the extent of the declaration elements the
/// shader consumes on the stream when that stride is 0: the inline (UP) path
/// has no other span, and [`bound_stream_layout`] pairs it with a `Constant`
/// step. A stride shorter than the extent stays as it is; the draw fetches the
/// attributes that end past it through a [`CrossingFetch`].
#[must_use]
pub const fn layout_stride(app_stride: u32, extent: u32) -> u32 {
    if app_stride == 0 {
        return extent;
    }
    app_stride
}

/// Source stream and byte advance of a Metal vertex buffer binding.
pub struct VertexFetchBinding {
    pub stream: u8,
    pub offset: u32,
}

/// Why a crossing attribute cannot use a binding of its own.
#[derive(Debug, PartialEq, Eq)]
pub enum VertexFetchError {
    StreamOutOfRange,
    AttributeWiderThanStride,
    UnalignedOffset,
    OutsideBuffer,
    NoFreeSlot,
}

/// Remap crossing attributes to spare bindings without changing the vertex step.
///
/// An attribute crosses when it ends past its stream's stride. It moves to a
/// slot of its own at offset 0, with the stream's layout and a binding of the
/// same stream advanced by the attribute's offset, so vertex `i` reads
/// `base + i * stride + offset` as D3D9 addresses it. A stream keeps its own
/// slot while one of its attributes fits; when none does, its first crossing
/// attribute takes that slot. Every other crossing attribute takes the lowest
/// slot no stream the declaration reads owns. A draw carries at most 16
/// attributes (the declaration record's cap) and each stream it reads has one
/// of them, so the 16 stream slots never run out and never reach the uniform
/// slots above them.
///
/// The placement is a function of the attributes and the remapped layouts
/// alone: a stream's own slot holds its stride whenever it is used, and that
/// stride decides which of its attributes cross. The pipeline memo, which
/// compares snapshots without the attribute list, relies on it.
///
/// # Errors
///
/// An attribute wider than its stride, which no binding can fetch, a stream
/// index past the table, or more attributes than slots. The outputs are
/// unspecified on error.
///
/// # Panics
///
/// Never: the stream and slot indices it narrows to `u8` are below 16.
pub fn remap_crossing_attributes(
    attrs: &mut [VertexAttrDesc],
    layouts: &mut [StreamLayout; MAX_STREAMS as usize],
) -> Result<[VertexFetchBinding; MAX_STREAMS as usize], VertexFetchError> {
    let original = *layouts;
    let mut bindings = std::array::from_fn(|i| VertexFetchBinding {
        stream: u8::try_from(i).expect("stream index fits u8"),
        offset: 0,
    });
    let mut reserved = 0u16;
    let mut read = 0u16;
    for attr in attrs.iter() {
        let index =
            usize::try_from(attr.buffer_index).map_err(|_| VertexFetchError::StreamOutOfRange)?;
        let Some(layout) = original.get(index) else {
            return Err(VertexFetchError::StreamOutOfRange);
        };
        read |= 1 << index;
        let width = attr.format.byte_size();
        if width > layout.stride {
            return Err(VertexFetchError::AttributeWiderThanStride);
        }
        if attr
            .offset
            .checked_add(width)
            .is_some_and(|end| end <= layout.stride)
        {
            reserved |= 1 << index;
        }
    }
    for (index, layout) in layouts.iter_mut().enumerate() {
        if reserved & (1 << index) == 0 {
            *layout = StreamLayout::UNUSED;
        }
    }
    for attr in attrs.iter_mut() {
        let index = attr.buffer_index as usize;
        if attr
            .offset
            .checked_add(attr.format.byte_size())
            .is_some_and(|end| end <= original[index].stride)
        {
            continue;
        }
        let slot = if reserved & (1 << index) == 0 {
            attr.buffer_index
        } else {
            (!(reserved | read)).trailing_zeros()
        };
        if slot >= MAX_STREAMS {
            return Err(VertexFetchError::NoFreeSlot);
        }
        reserved |= 1 << slot;
        layouts[slot as usize] = original[index];
        bindings[slot as usize] = VertexFetchBinding {
            stream: u8::try_from(attr.buffer_index).expect("stream index checked above"),
            offset: attr.offset,
        };
        attr.buffer_index = slot;
        attr.offset = 0;
    }
    Ok(bindings)
}

/// The part of a stream offset below a multiple of 4, which a fetch moves into its attributes.
///
/// Metal ignores the low two bits of a vertex buffer offset bound for a
/// vertex descriptor, so a stream bound at such an offset binds at the
/// multiple of 4 below it and each of its attributes reads this many bytes
/// further on.
#[inline]
#[must_use]
pub const fn offset_shift(offset: u32) -> u32 {
    offset & 3
}

/// Per-stream [`offset_shift`]s of a draw, two bits per stream, stream `n` at bit `2 * n`.
///
/// `streams` yields each stream the draw reads from a vertex buffer with
/// that stream's offset. Zero when every offset is a multiple of 4.
#[must_use]
pub fn stream_shifts(streams: impl Iterator<Item = (u8, u32)>) -> u32 {
    streams.fold(0, |shifts, (stream, offset)| {
        shifts | (offset_shift(offset) << (2 * u32::from(stream)))
    })
}

/// The offset a slot binds: the stream offset rounded down to a multiple of 4, plus `advance`.
///
/// `advance` is a crossing attribute's offset, 0 on a stream's own slot.
/// Wraps on overflow; [`advanced_binding_offset`] checks the offsets that
/// can overflow before a draw binds them.
#[inline]
#[must_use]
pub const fn slot_binding_offset(stream_offset: u32, advance: u32) -> u32 {
    (stream_offset & !3).wrapping_add(advance)
}

/// The byte offset of an advanced binding, checked against what Metal accepts.
///
/// `base` is the stream's own offset into a buffer of `len` bytes and
/// `advance` the crossing attribute's offset; the binding starts at
/// [`slot_binding_offset`]. Metal takes a vertex buffer offset that is a
/// multiple of 4 and inside the buffer. The stream's rounded-down offset is
/// one, and so is every declaration element offset, so only a binding past
/// the buffer's end or an `advance` off a four-byte boundary is refused.
///
/// # Errors
///
/// [`VertexFetchError::UnalignedOffset`] or [`VertexFetchError::OutsideBuffer`].
pub fn advanced_binding_offset(base: u32, advance: u32, len: u64) -> Result<u32, VertexFetchError> {
    let offset = (base & !3)
        .checked_add(advance)
        .ok_or(VertexFetchError::OutsideBuffer)?;
    if !offset.is_multiple_of(4) {
        return Err(VertexFetchError::UnalignedOffset);
    }
    if u64::from(offset) >= len {
        return Err(VertexFetchError::OutsideBuffer);
    }
    Ok(offset)
}

/// The vertex fetch of a draw that cannot bind each stream once at its own offset.
///
/// Built for a draw with an attribute that ends past its stream's stride, or
/// with a stream offset off a four-byte boundary, from the resolved
/// attributes, the stream layouts and the per-stream [`stream_shifts`]; any
/// other draw never builds one. Holds the remapped attributes and layouts the
/// pipeline is built from and, per Metal slot, the stream and byte advance
/// the draw binds there (at [`slot_binding_offset`]). Every attribute read
/// from a shifted stream sits its stream's shift further into its slot, so
/// it may end up to 3 bytes past the stride, which Metal fetches as
/// addressed. It remembers what it was built from, so the next draw over the
/// same declaration record, layouts and shifts reuses it (see
/// [`Self::reuse_or_rebuild`]).
pub struct CrossingFetch {
    attrs: [VertexAttrDesc; MAX_STREAMS as usize],
    attr_count: u8,
    layouts: [StreamLayout; MAX_STREAMS as usize],
    bindings: [VertexFetchBinding; MAX_STREAMS as usize],
    /// Bit `n` set: Metal slot `n` is read by the remapped pipeline.
    used_slots: u16,
    /// The [`stream_shifts`] the attributes were moved by.
    shifts: u32,
    /// The declaration record's address, stream layouts and shifts this was built from.
    source: Option<(usize, [StreamLayout; MAX_STREAMS as usize], u32)>,
}

impl CrossingFetch {
    /// A fetch that reads nothing and remembers no source.
    #[must_use]
    pub fn empty() -> Self {
        const UNSET: VertexAttrDesc = VertexAttrDesc {
            attr_index: 0,
            buffer_index: 0,
            offset: 0,
            format: VertexFormat::Invalid,
        };
        Self {
            attrs: [UNSET; MAX_STREAMS as usize],
            attr_count: 0,
            layouts: [StreamLayout::UNUSED; MAX_STREAMS as usize],
            bindings: std::array::from_fn(|_| VertexFetchBinding {
                stream: 0,
                offset: 0,
            }),
            used_slots: 0,
            shifts: 0,
            source: None,
        }
    }

    /// Remap `attrs` over `layouts` and move them by `shifts` (see [`remap_crossing_attributes`]).
    ///
    /// # Errors
    ///
    /// The draw cannot be fetched this way; see [`remap_crossing_attributes`].
    pub fn new(
        attrs: &[VertexAttrDesc],
        layouts: &[StreamLayout; MAX_STREAMS as usize],
        shifts: u32,
    ) -> Result<Self, VertexFetchError> {
        let mut fetch = Self::empty();
        fetch.rebuild(attrs, layouts, shifts)?;
        Ok(fetch)
    }

    /// Reuse this fetch if it was built from `record`, `layouts` and `shifts`, else rebuild it.
    ///
    /// `record` is the address of the declaration record the attributes
    /// came from, which names one immutable list while a packet replays;
    /// [`Self::forget_source`] runs before the next packet, whose records
    /// may sit at the same addresses.
    ///
    /// # Errors
    ///
    /// As [`Self::new`]; the fetch then remembers no source.
    pub fn reuse_or_rebuild(
        &mut self,
        record: usize,
        attrs: &[VertexAttrDesc],
        layouts: &[StreamLayout; MAX_STREAMS as usize],
        shifts: u32,
    ) -> Result<(), VertexFetchError> {
        if self
            .source
            .as_ref()
            .is_some_and(|(built, built_layouts, built_shifts)| {
                *built == record && built_layouts == layouts && *built_shifts == shifts
            })
        {
            return Ok(());
        }
        self.source = None;
        self.rebuild(attrs, layouts, shifts)?;
        self.source = Some((record, *layouts, shifts));
        Ok(())
    }

    /// Forget what this fetch was built from, so the next draw builds it again.
    pub const fn forget_source(&mut self) {
        self.source = None;
    }

    /// Remap `attrs` over `layouts` into this fetch and move them by `shifts`.
    ///
    /// Which attributes cross is decided on the application's offsets; the
    /// shift then moves every attribute its slot's stream feeds.
    ///
    /// # Panics
    ///
    /// Never: the attribute count it narrows to `u8` is at most 16, checked first.
    fn rebuild(
        &mut self,
        attrs: &[VertexAttrDesc],
        layouts: &[StreamLayout; MAX_STREAMS as usize],
        shifts: u32,
    ) -> Result<(), VertexFetchError> {
        let Some(prefix) = self.attrs.get_mut(..attrs.len()) else {
            return Err(VertexFetchError::NoFreeSlot);
        };
        prefix.copy_from_slice(attrs);
        self.attr_count = u8::try_from(attrs.len()).expect("at most 16 attributes");
        self.layouts = *layouts;
        self.shifts = shifts;
        self.bindings = remap_crossing_attributes(prefix, &mut self.layouts)?;
        for attr in prefix.iter_mut() {
            let stream = self.bindings[attr.buffer_index as usize].stream;
            attr.offset += (shifts >> (2 * u32::from(stream))) & 3;
        }
        self.used_slots = (0..MAX_STREAMS)
            .filter(|&slot| self.layouts[slot as usize].is_used())
            .fold(0u16, |mask, slot| mask | (1 << slot));
        Ok(())
    }

    /// The remapped attributes the pipeline is built from.
    #[must_use]
    pub fn attrs(&self) -> &[VertexAttrDesc] {
        &self.attrs[..usize::from(self.attr_count)]
    }

    /// The remapped vertex buffer layouts, indexed by Metal slot.
    #[must_use]
    pub const fn layouts(&self) -> &[StreamLayout; MAX_STREAMS as usize] {
        &self.layouts
    }

    /// The `vdecl_hash` a pipeline snapshot of this fetch carries.
    ///
    /// The pipeline memo compares snapshots without the attribute list, and
    /// the shifts move attribute offsets that neither the declaration nor
    /// the layouts name, so a shifted fetch folds them into the declaration
    /// identity. Without shifts it is `vdecl_hash` unchanged.
    #[must_use]
    pub fn snapshot_vdecl_hash(&self, vdecl_hash: u64) -> u64 {
        if self.shifts == 0 {
            return vdecl_hash;
        }
        xxh3_64_with_seed(&self.shifts.to_le_bytes(), vdecl_hash)
    }

    /// Check every advanced binding of the `crossing` streams against what Metal accepts.
    ///
    /// `stream` gives a crossing stream's own offset and its buffer's length,
    /// `None` for a stream fed nothing (which steps by its extent and never
    /// crosses). Only a binding advanced past the stream offset is checked: a
    /// stream's own binding binds its offset rounded down to a multiple of 4,
    /// as every other draw's does when the offset is one.
    ///
    /// # Errors
    ///
    /// The first advanced offset [`advanced_binding_offset`] refuses.
    ///
    /// # Panics
    ///
    /// Never: a stream index from a 16-bit mask fits `u8`.
    pub fn check_advanced_offsets(
        &self,
        crossing: u16,
        stream: impl Fn(u32) -> Option<(u32, u64)>,
    ) -> Result<(), VertexFetchError> {
        let mut streams = crossing;
        while streams != 0 {
            let index = streams.trailing_zeros();
            streams &= streams - 1;
            let Some((base, len)) = stream(index) else {
                continue;
            };
            let index = u8::try_from(index).expect("a stream index below 16");
            for (_, advance) in self.slots_of(index).filter(|&(_, advance)| advance != 0) {
                advanced_binding_offset(base, advance, len)?;
            }
        }
        Ok(())
    }

    /// The Metal slots that read `stream`, each with its byte advance past the stream offset.
    pub fn slots_of(&self, stream: u8) -> impl Iterator<Item = (u32, u32)> + '_ {
        let mut used = self.used_slots;
        std::iter::from_fn(move || {
            while used != 0 {
                let slot = used.trailing_zeros();
                used &= used - 1;
                let binding = &self.bindings[slot as usize];
                if binding.stream == stream {
                    return Some((slot, binding.offset));
                }
            }
            None
        })
    }
}

/// Bytes required by an inline stream, including its last crossing attribute.
///
/// `(count - 1) * stride + extent` when that is past `count * stride`, else
/// `count * stride`. `None` on overflow.
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

/// The read-range size a crossing stream's draw records for rename-at-overlap.
///
/// `size` is the span of the packed elements, 0 meaning "to the end of the
/// buffer". A finite span grows by what the last element's crossing attribute
/// reads past its stride; the end-of-buffer marker already covers it.
#[must_use]
pub const fn crossing_read_size(size: u32, extent: u32, stride: u32) -> u32 {
    if size == 0 {
        return 0;
    }
    size.saturating_add(extent.saturating_sub(stride))
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

#[cfg(test)]
mod tests;
