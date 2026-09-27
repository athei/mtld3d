use super::*;

#[test]
fn native_failure_is_deferred_until_explicit_observation() {
    let failure = AtomicI32::new(D3D_OK);
    let native = AtomicU32::new(0);
    assert_eq!(status(&failure, &native), Ok(()));
    native.store(1, Ordering::Release);
    assert_eq!(known_status(&failure), Ok(()));
    assert_eq!(failure.load(Ordering::Acquire), D3D_OK);
    assert_eq!(status(&failure, &native), Err(D3DERR_DEVICELOST));
    assert_eq!(known_status(&failure), Err(D3DERR_DEVICELOST));
}

#[test]
fn full_observation_and_later_errors_preserve_the_first_hresult() {
    for first in [E_OUTOFMEMORY, D3DERR_DEVICELOST] {
        let failure = AtomicI32::new(D3D_OK);
        let native = AtomicU32::new(1);
        assert_eq!(record_failure(&failure, first), first);
        assert_eq!(status(&failure, &native), Err(first));
        assert_eq!(record_failure(&failure, E_OUTOFMEMORY), first);
        assert_eq!(record_failure(&failure, D3DERR_DEVICELOST), first);
        assert_eq!(known_status(&failure), Err(first));
    }
}

#[test]
fn nonallocation_failures_normalize_to_device_lost() {
    for status in [D3D_OK, mtld3d_types::D3DERR_INVALIDCALL] {
        let failure = AtomicI32::new(D3D_OK);
        assert_eq!(record_failure(&failure, status), D3DERR_DEVICELOST);
        assert_eq!(known_status(&failure), Err(D3DERR_DEVICELOST));
    }
}

#[test]
fn known_failure_guard_stops_further_frame_capture() {
    use mtld3d_shared::encoder_protocol::EncoderOpcode;

    use crate::{encoder_packet::FrameRecorder, scratch::ScratchArena};

    let failure = AtomicI32::new(D3D_OK);
    let native = AtomicU32::new(1);
    let mut recorder = FrameRecorder::new();
    let mut scratch = ScratchArena::new();
    // A fresh native failure is not yet known on PE. This checks composition
    // with the real recorder, not the device's call-site or Present ordering.
    known_status(&failure).unwrap();
    recorder
        .record_constant_bytes(&mut scratch, EncoderOpcode::SetPsConstRange, 0, 1, &[0; 16])
        .unwrap();
    let committed = scratch.bytes_used();
    assert_eq!(status(&failure, &native), Err(D3DERR_DEVICELOST));
    let capture = known_status(&failure).map(|()| {
        recorder
            .record_constant_bytes(&mut scratch, EncoderOpcode::SetPsConstRange, 1, 1, &[0; 16])
            .unwrap();
    });
    assert_eq!(capture, Err(D3DERR_DEVICELOST));
    assert_eq!(recorder.len(), 1);
    assert_eq!(scratch.bytes_used(), committed);
    assert_eq!(recorder.recording_error(), None);
}
