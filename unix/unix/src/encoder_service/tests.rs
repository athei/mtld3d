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

struct RejectedSubmit {
    entered: std::sync::mpsc::Sender<String>,
}

impl crate::encoder::SubmitSpawner for RejectedSubmit {
    fn spawn(
        self,
        work: impl FnOnce() + Send + 'static,
    ) -> std::io::Result<std::thread::JoinHandle<()>> {
        self.entered
            .send(
                std::thread::current()
                    .name()
                    .unwrap_or("unnamed")
                    .to_owned(),
            )
            .expect("supervisor receives submit launch");
        drop(work);
        Err(std::io::Error::other("injected submit launch failure"))
    }
}

fn failed_startup_retires_started_workers(reject_submit: bool) {
    use std::{
        sync::{Arc, atomic::Ordering, mpsc},
        time::{Duration, Instant},
    };

    use mtld3d_core::shader_prewarm::PrewarmHandle;
    use mtld3d_shared::{MetalHandle, record_handle::DeviceRecordHandle};

    const DEADLINE: Duration = Duration::from_secs(5);
    let (release, blocked) = mpsc::channel();
    let (prewarm_done, prewarm_finished) = mpsc::channel();
    let (submit_entered, submit_start) = mpsc::channel();
    let (finished, finish) = mpsc::channel();
    let owner = std::thread::spawn(move || {
        let (started, starts) = mpsc::channel();
        let calibration_started = started.clone();
        let calibration = super::CalibrationWorker::spawn(move || {
            calibration_started.send(()).expect("calibration started");
            // A failed supervisor assertion cannot strand this worker.
            let _ = blocked.recv_timeout(DEADLINE);
            1_000_000_000
        });
        let clock_owner = Arc::downgrade(&calibration.clock);
        let native_clock = Arc::clone(&calibration.clock);
        let (prewarm, receiver) = PrewarmHandle::spawn(move |stop| {
            started.send(()).expect("prewarm started");
            let deadline = Instant::now() + DEADLINE;
            while !stop.load(Ordering::Acquire) && Instant::now() < deadline {
                std::thread::yield_now();
            }
            let canceled = stop.load(Ordering::Acquire);
            prewarm_done.send(canceled).expect("prewarm completed");
            None::<crate::encoder::WarmCache>
        });
        let result =
            super::EncoderService::finish_startup(calibration, prewarm, receiver, |receiver| {
                // Injection happens only after both earlier workers really started.
                starts.recv_timeout(DEADLINE).expect("first startup worker");
                starts
                    .recv_timeout(DEADLINE)
                    .expect("second startup worker");
                if !reject_submit {
                    drop(receiver);
                    drop(native_clock);
                    return Err(std::io::Error::other("injected encoder launch failure"));
                }
                let device = objc2_metal::MTLCreateSystemDefaultDevice()
                    .expect("Metal device for late startup failure");
                // SAFETY: device remains retained through the encoder startup result.
                let device_handle =
                    unsafe { MetalHandle::new(objc2::rc::Retained::as_ptr(&device) as u64) };
                let startup = super::EncoderStartup {
                    context: super::EncoderContext {
                        device_handle,
                        record_handle: DeviceRecordHandle::NULL,
                        coherent_seq_ptr: 0,
                        upload_coherent_seq_ptr: 0,
                        failed_submit_seq_ptr: 0,
                        retained_bytes_ptr: 0,
                    },
                    clocks: super::EncoderClocks {
                        native: native_clock,
                        #[cfg(perf_tracking)]
                        source: 0,
                    },
                };
                // The device is retained first, then the submit launch fails before replay.
                crate::encoder::EncoderThread::spawn_with_submit(
                    GpuCaps::apple_silicon_default(),
                    Arc::new(Mtld3dConfig::default()),
                    receiver,
                    None,
                    startup,
                    RejectedSubmit {
                        entered: submit_entered,
                    },
                )
            });
        let error = result.err().expect("injected startup failure");
        finished
            .send((error.to_string(), clock_owner.strong_count()))
            .expect("startup result");
    });

    let canceled = prewarm_finished.recv_timeout(DEADLINE);
    let returned_before_calibration = finish.try_recv();
    let waited_for_calibration =
        matches!(returned_before_calibration, Err(mpsc::TryRecvError::Empty));
    // Always release calibration before any supervisor assertion can unwind.
    let _ = release.send(());
    let result =
        returned_before_calibration.map_or_else(|_| finish.recv_timeout(DEADLINE).ok(), Some);
    if result.is_some() {
        owner.join().expect("startup owner finished");
    }
    assert_eq!(canceled, Ok(true), "failure cancels and retires prewarm");
    assert!(
        waited_for_calibration,
        "failure must join blocked calibration before returning"
    );
    let (error, owners) = result.expect("startup cleanup completes within its bound");
    assert_eq!(
        owners, 0,
        "calibration and encoder startup owners are retired"
    );
    if reject_submit {
        assert_eq!(
            submit_start.recv_timeout(DEADLINE).unwrap(),
            "mtld3d-encoder"
        );
        assert_eq!(error, "injected submit launch failure");
    } else {
        assert_eq!(error, "injected encoder launch failure");
    }
}

#[test]
fn encoder_spawn_failure_retires_started_service_workers() {
    failed_startup_retires_started_workers(false);
}

#[test]
fn submit_spawn_failure_joins_encoder_and_service_workers() {
    failed_startup_retires_started_workers(true);
}

#[test]
fn null_device_rejects_encoder_startup_before_submit_spawn() {
    use std::sync::{Arc, mpsc};

    use mtld3d_shared::{MetalHandle, record_handle::DeviceRecordHandle};

    let (prewarm, receiver) = mpsc::channel();
    drop(prewarm);
    let (entered, submit_start) = mpsc::channel();
    let startup = super::EncoderStartup {
        context: super::EncoderContext {
            device_handle: MetalHandle::NULL,
            record_handle: DeviceRecordHandle::NULL,
            coherent_seq_ptr: 0,
            upload_coherent_seq_ptr: 0,
            failed_submit_seq_ptr: 0,
            retained_bytes_ptr: 0,
        },
        clocks: super::EncoderClocks {
            native: Arc::new(mtld3d_shared::clock_calibration::ClockCalibration::new()),
            #[cfg(perf_tracking)]
            source: 0,
        },
    };
    let error = crate::encoder::EncoderThread::spawn_with_submit(
        GpuCaps::apple_silicon_default(),
        Arc::new(Mtld3dConfig::default()),
        receiver,
        None,
        startup,
        RejectedSubmit { entered },
    )
    .err()
    .expect("null device must fail startup");
    assert_eq!(error.to_string(), "encoder: missing Metal device");
    assert_eq!(
        submit_start.try_recv(),
        Err(mpsc::TryRecvError::Disconnected)
    );
}
