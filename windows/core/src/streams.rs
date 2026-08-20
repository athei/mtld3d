//! D3D9 vertex-stream frequency: the `SetStreamSourceFreq` contract and what draws derive from it.
//!
//! A frequency word is `flags | count`. `D3DSTREAMSOURCE_INDEXEDDATA` marks a
//! per-vertex stream of an instanced draw and, on stream 0, carries the
//! instance count; `D3DSTREAMSOURCE_INSTANCEDATA` marks a per-instance stream
//! whose count is the number of instances that share one element. The
//! validation rules and the instance-count derivation follow the D3D9
//! runtime as implemented by DXVK and wined3d, which agree on every rule.

use mtld3d_shared::mtl::VertexStepFunction;
use mtld3d_types::{D3DSTREAMSOURCE_INDEXEDDATA, D3DSTREAMSOURCE_INSTANCEDATA, MAX_STREAMS};

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validation_follows_the_runtime_rules() {
        assert_eq!(validate_stream_freq(0, 1), Ok(()));
        assert_eq!(validate_stream_freq(1, 2), Ok(()));
        assert_eq!(
            validate_stream_freq(MAX_STREAMS, 1),
            Err(StreamFreqError::StreamOutOfRange)
        );
        assert_eq!(
            validate_stream_freq(0, D3DSTREAMSOURCE_INSTANCEDATA | 1),
            Err(StreamFreqError::InstanceDataOnStreamZero)
        );
        assert_eq!(
            validate_stream_freq(
                1,
                D3DSTREAMSOURCE_INSTANCEDATA | D3DSTREAMSOURCE_INDEXEDDATA
            ),
            Err(StreamFreqError::BothFlags)
        );
        assert_eq!(validate_stream_freq(1, 0), Err(StreamFreqError::Zero));
        // A flag with a zero count is a non-zero word: accepted.
        assert_eq!(validate_stream_freq(1, D3DSTREAMSOURCE_INDEXEDDATA), Ok(()));
        assert_eq!(
            validate_stream_freq(1, D3DSTREAMSOURCE_INSTANCEDATA),
            Ok(())
        );
        assert_eq!(validate_stream_freq(0, D3DSTREAMSOURCE_INDEXEDDATA), Ok(()));
    }

    #[test]
    fn instance_count_reads_stream_zero_only_when_something_is_instanced() {
        assert_eq!(instance_count(D3DSTREAMSOURCE_INDEXEDDATA | 4, true), 4);
        // Stream 0 with a plain count and no flag still supplies the count.
        assert_eq!(instance_count(3, true), 3);
        // No per-instance stream in the draw: one instance regardless.
        assert_eq!(instance_count(D3DSTREAMSOURCE_INDEXEDDATA | 4, false), 1);
        // `INDEXEDDATA | 0` is driver-defined; one instance here.
        assert_eq!(instance_count(D3DSTREAMSOURCE_INDEXEDDATA, true), 1);
        // Bits above the count mask are ignored.
        assert_eq!(instance_count(0x3F80_0002, true), 2);
    }

    #[test]
    fn step_function_follows_the_flags() {
        assert_eq!(stream_step(1), (VertexStepFunction::PerVertex, 1));
        assert_eq!(
            stream_step(D3DSTREAMSOURCE_INDEXEDDATA | 7),
            (VertexStepFunction::PerVertex, 1)
        );
        assert_eq!(
            stream_step(D3DSTREAMSOURCE_INSTANCEDATA | 1),
            (VertexStepFunction::PerInstance, 1)
        );
        assert_eq!(
            stream_step(D3DSTREAMSOURCE_INSTANCEDATA | 3),
            (VertexStepFunction::PerInstance, 3)
        );
        assert_eq!(
            stream_step(D3DSTREAMSOURCE_INSTANCEDATA),
            (VertexStepFunction::Constant, 0)
        );
    }

    #[test]
    fn instanced_read_bytes_round_up_and_saturate() {
        assert_eq!(
            instanced_stream_read_bytes(4, VertexStepFunction::PerInstance, 1, 12),
            48
        );
        // 5 instances at rate 2 read 3 elements.
        assert_eq!(
            instanced_stream_read_bytes(5, VertexStepFunction::PerInstance, 2, 12),
            36
        );
        assert_eq!(
            instanced_stream_read_bytes(100, VertexStepFunction::Constant, 0, 12),
            12
        );
        assert_eq!(
            instanced_stream_read_bytes(u32::MAX, VertexStepFunction::PerInstance, 1, 16),
            u32::MAX
        );
    }
}
