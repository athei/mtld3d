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
