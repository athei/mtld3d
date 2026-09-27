use mtld3d_core::{config::Mtld3dConfig, encoder_value::WireValue, gpu_caps::GpuCaps};
use mtld3d_shared::{encoder_runtime::CONFIG_RECORD, encoder_wire::FrameSlab};

use super::decode_settings;

#[test]
fn settings_reject_trailing_records_and_payload() {
    let mut slab = FrameSlab::new();
    slab.push_record(CONFIG_RECORD, |writer| {
        Mtld3dConfig::default().write_wire(writer)?;
        GpuCaps::apple_silicon_default().write_wire(writer)?;
        Some("/game/mtld3d_shaders.bin".to_owned()).write_wire(writer)
    })
    .unwrap();
    let (_, _, path) = decode_settings(slab.as_bytes()).unwrap();
    assert_eq!(
        path.unwrap(),
        std::path::PathBuf::from("/game/mtld3d_shaders.bin")
    );
    slab.push_record(CONFIG_RECORD, |_| Ok(())).unwrap();
    assert!(decode_settings(slab.as_bytes()).is_err());
    slab.clear();
    slab.push_record(CONFIG_RECORD, |writer| {
        Mtld3dConfig::default().write_wire(writer)?;
        GpuCaps::apple_silicon_default().write_wire(writer)?;
        Option::<String>::None.write_wire(writer)?;
        writer.u8(255)
    })
    .unwrap();
    assert!(decode_settings(slab.as_bytes()).is_err());
}

#[test]
fn truncated_settings_never_start_workers() {
    let mut slab = FrameSlab::new();
    slab.push_record(CONFIG_RECORD, |writer| {
        Mtld3dConfig::default().write_wire(writer)?;
        GpuCaps::apple_silicon_default().write_wire(writer)?;
        Option::<String>::None.write_wire(writer)
    })
    .unwrap();
    for end in 0..slab.as_bytes().len() {
        assert!(decode_settings(&slab.as_bytes()[..end]).is_err());
    }
}

#[test]
fn rejected_submission_never_claims_admission() {
    use mtld3d_shared::encoder_runtime::SubmitEncoderFrameParams;
    use mtld3d_types::D3DERR_INVALIDCALL;

    let mut params = SubmitEncoderFrameParams {
        runtime: 0,
        metadata_ptr: 0,
        operations_ptr: 0,
        completion: 0,
        metadata_len: 0,
        operations_len: 0,
        mode: 0,
        admitted: 99,
    };
    assert_eq!(
        super::submit_handler(std::ptr::from_mut(&mut params).cast()),
        D3DERR_INVALIDCALL
    );
    assert_eq!(params.admitted, 0);
}

#[test]
fn null_lifecycle_requests_report_failure() {
    use mtld3d_types::D3DERR_INVALIDCALL;

    assert_eq!(
        super::create_handler(std::ptr::null_mut()),
        D3DERR_INVALIDCALL
    );
    assert_eq!(
        super::destroy_handler(std::ptr::null_mut()),
        D3DERR_INVALIDCALL
    );
    assert_eq!(
        super::control_handler(std::ptr::null_mut()),
        D3DERR_INVALIDCALL
    );
}

#[test]
fn native_failure_mailbox_is_sticky() {
    use std::sync::atomic::{AtomicU32, Ordering};
    let failure = AtomicU32::new(0);
    let address = std::ptr::from_ref(&failure) as u64;
    // SAFETY: the local aligned atomic remains live through this publication.
    unsafe { super::publish_failure(address) };
    // SAFETY: the same atomic remains live for repeated publication.
    unsafe { super::publish_failure(address) };
    // SAFETY: zero explicitly requests no mailbox publication.
    unsafe { super::publish_failure(0) };
    assert_eq!(failure.load(Ordering::Acquire), 1);
}

#[test]
fn calibration_startup_does_not_wait_for_work_and_drop_joins() {
    use std::{sync::mpsc, time::Duration};
    let (release, blocked) = mpsc::channel();
    let (started, start) = mpsc::channel();
    let (finished, finish) = mpsc::channel();
    let owner = std::thread::spawn(move || {
        let calibration = super::CalibrationWorker::spawn(move || {
            blocked.recv().unwrap();
            1_000_000_000
        });
        started.send(()).unwrap();
        drop(calibration);
        finished.send(()).unwrap();
    });
    let startup = start.recv_timeout(Duration::from_secs(2));
    let prematurely_finished = finish.recv_timeout(Duration::from_millis(50)).is_ok();
    release.send(()).unwrap();
    finish.recv_timeout(Duration::from_secs(2)).unwrap();
    owner.join().unwrap();
    assert!(
        startup.is_ok(),
        "startup must return before calibration completes"
    );
    assert!(
        !prematurely_finished,
        "shutdown must retain and join calibration"
    );
}

#[test]
fn calibration_panic_publishes_failure_before_join() {
    let worker = super::CalibrationWorker::spawn(|| panic!("test calibration failure"));
    let mailbox = std::sync::Arc::clone(&worker.clock);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    while matches!(mailbox.get(), Ok(None)) && std::time::Instant::now() < deadline {
        std::thread::yield_now();
    }
    assert!(mailbox.get().is_err());
    drop(worker);
    assert_eq!(std::sync::Arc::strong_count(&mailbox), 1);
}
