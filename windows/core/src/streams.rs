//! D3D9 vertex-stream frequency: the `SetStreamSourceFreq` contract and what draws derive from it.
//!
//! A frequency word is `flags | count`. `D3DSTREAMSOURCE_INDEXEDDATA` marks a
//! per-vertex stream of an instanced draw and, on stream 0, carries the
//! instance count; `D3DSTREAMSOURCE_INSTANCEDATA` marks a per-instance stream
//! whose count is the number of instances that share one element. The
//! validation rules and the instance-count derivation follow the D3D9
//! runtime's observable behaviour.

use mtld3d_shared::mtl::VertexStepFunction;
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

/// The stride a stream's vertex buffer layout steps by.
///
/// The application's stride wins when it covers the extent of the declaration
/// elements the shader consumes on that stream (it can exceed it when the
/// vertex struct carries fields past them). A zero stride returns the extent:
/// the inline (UP) path has no other span, and [`bound_stream_layout`] pairs it
/// with a `Constant` step. A non-zero stride smaller than the consumed extent
/// means the shader reads an attribute past the end of each vertex, which
/// Metal rejects as a pipeline, so the layout is widened to the extent with a
/// warning; the affected draw fetches wrong data either way.
#[must_use]
pub fn layout_stride(app_stride: u32, extent: u32) -> u32 {
    if app_stride == 0 {
        return extent;
    }
    if app_stride < extent {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "stream stride {app_stride} below the consumed declaration extent {extent}; layout widened to the extent"
        );
        return extent;
    }
    app_stride
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
pub fn bound_stream_layout(app_stride: u32, extent: u32, freq: u32) -> StreamLayout {
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
