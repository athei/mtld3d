use std::{
    sync::{Arc, Weak, mpsc},
    thread,
    time::Duration,
};

use mtld3d_core::{async_compile::TicketSource, dxso::parse};

use super::*;

const WAIT: Duration = Duration::from_secs(5);

fn retained_job() -> (QueuedJob, Weak<DxsoProgram>) {
    let tokens = [
        0xFFFF_0200,
        0x0200_0001,
        0x800F_0800,
        0xA0E4_0000,
        0x0000_FFFF,
    ];
    let program = Arc::new(parse(&tokens).expect("valid shader"));
    let weak = Arc::downgrade(&program);
    let job = QueuedJob {
        job: CompileJob::Library(LibraryJob {
            input: LibraryInput::ProgrammablePs {
                ps_id: ProgramId::from_tokens(&tokens),
                program,
                variant: VariantKey::default(),
            },
            reference: ShaderRecordRef::new(CachedKind::Sm2Ps, 1),
            device: MetalHandle::NULL,
            persist: false,
            cache_path: None,
        }),
        enqueued_tsc: 0,
    };
    (job, weak)
}

/// Cancellation drops queued owners while a running job retains its owner through close and join.
#[test]
fn canceled_queue_and_blocked_worker_release_distinct_job_owners() {
    let queue = Arc::new(CompileQueue::new());
    let mut tickets = TicketSource::new();
    let running_ticket = tickets.issue();
    let (running, running_owner) = retained_job();
    queue.push(running_ticket, running);
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let worker_queue = Arc::clone(&queue);
    let worker = thread::spawn(move || {
        worker_main_with(&worker_queue, |ticket, job| {
            assert_eq!(ticket, running_ticket, "only the first job may execute");
            entered_tx.send(()).expect("test is waiting for entry");
            release_rx
                .recv_timeout(WAIT)
                .expect("release blocked build");
            drop(job);
            true
        });
    });
    entered_rx
        .recv_timeout(WAIT)
        .expect("worker took first job");
    let canceled_ticket = tickets.issue();
    let (canceled, canceled_owner) = retained_job();
    queue.push(canceled_ticket, canceled);
    assert_eq!(running_owner.strong_count(), 1);
    assert_eq!(canceled_owner.strong_count(), 1);
    let (ticket, pending) = queue.take_any().expect("unstarted job can be canceled");
    assert_eq!(ticket, canceled_ticket);
    drop(pending);
    assert!(
        canceled_owner.upgrade().is_none(),
        "canceled job drops its sole owner"
    );
    assert!(queue.take_any().is_none());
    queue.close();
    assert!(
        !worker.is_finished(),
        "closing cannot end an executing build"
    );
    assert_eq!(
        running_owner.strong_count(),
        1,
        "running build still owns its input"
    );
    let (joining_tx, joining_rx) = mpsc::channel();
    let (joined_tx, joined_rx) = mpsc::channel();
    let joiner = thread::spawn(move || {
        joining_tx.send(()).expect("join starts");
        worker.join().expect("compile worker exits after release");
        joined_tx.send(()).expect("join completion observed");
    });
    joining_rx.recv_timeout(WAIT).expect("join attempt started");
    assert!(
        joined_rx.try_recv().is_err(),
        "join cannot complete before build release"
    );
    release_tx.send(()).expect("worker still awaits release");
    joined_rx
        .recv_timeout(WAIT)
        .expect("join completes after release");
    joiner.join().expect("join observer exits");
    assert!(
        running_owner.upgrade().is_none(),
        "running job drops its sole owner"
    );
    assert!(
        canceled_owner.upgrade().is_none(),
        "no canceled owner was revived"
    );
}

fn pipeline_job(sibling_of: Option<u64>) -> QueuedJob {
    let snapshot = PipelineSnapshot {
        vs_fn: MetalHandle::NULL,
        ps_fn: MetalHandle::NULL,
        vdecl_hash: 0,
        stream_layouts: [mtld3d_core::pipeline_state::StreamLayout::UNUSED;
            mtld3d_types::MAX_STREAMS as usize],
        color_format: mtld3d_shared::mtl::PixelFormat::Bgra8Unorm,
        attach: mtld3d_core::pipeline_state::PipelineAttachFlags::empty(),
        rs: mtld3d_core::pipeline_state::PipelineRsBits::default(),
        extra: mtld3d_core::pipeline_state::ExtraColorAttachments::NONE,
        ps_color_out_mask: 1,
        sample_count: 1,
    };
    let shader = PairShaderId {
        is_programmable: true,
        hash: 1,
    };
    QueuedJob {
        job: CompileJob::Pipeline(Box::new(PipelineJob {
            snapshot,
            vertex_attrs: Vec::new(),
            identity: PipelineIdentity {
                shader_refs: None,
                vs: shader,
                ps: shader,
            },
            sibling_of,
            device: MetalHandle::NULL,
            persist: false,
            cache_path: None,
        })),
        enqueued_tsc: 0,
    }
}

/// A primary pipeline starts ahead of queued libraries; a no-colour sibling queues with them.
#[test]
fn push_sends_primary_pipelines_ahead_and_siblings_with_the_libraries() {
    let queue = CompileQueue::new();
    let mut tickets = TicketSource::new();
    let library = tickets.issue();
    let sibling = tickets.issue();
    let primary = tickets.issue();
    queue.push(library, retained_job().0);
    queue.push(sibling, pipeline_job(Some(0x1000)));
    queue.push(primary, pipeline_job(None));
    let order: Vec<JobTicket> = core::iter::from_fn(|| queue.take_any())
        .map(|(ticket, _)| ticket)
        .collect();
    assert_eq!(order, [primary, library, sibling]);
}

/// A draw through a `NULL` render target 0 is judged by its depth and stencil planes alone.
///
/// Its pass attaches no colour target, so the skip test answers yes once the
/// depth and stencil planes it draws into were cleared whole in this frame
/// and the one before, and no while either plane was cleared in one frame
/// only. A real render target 0 that no clear reached keeps every draw, its
/// depth planes rebuilt or not.
#[test]
fn a_null_render_target_0_is_judged_by_its_depth_and_stencil_planes() {
    use std::sync::atomic::AtomicU64;

    use mtld3d_core::{
        config::Mtld3dConfig,
        gpu_caps::GpuCaps,
        passes::{BackbufferContents, FrameReset},
        render_scale::RenderScale,
    };
    use mtld3d_shared::{
        clock_calibration::ClockCalibration, mtl::PixelFormat, record_handle::DeviceRecordHandle,
    };
    use objc2::rc::Retained;
    use objc2_metal::MTLCreateSystemDefaultDevice;

    use crate::encoder_service::EncoderContext;

    let Some(device) = MTLCreateSystemDefaultDevice() else {
        eprintln!("MTLCreateSystemDefaultDevice returned nil, skipping");
        return;
    };
    let coherent = AtomicU64::new(0);
    let retained = AtomicU64::new(0);
    let context = EncoderContext {
        // SAFETY: `device` stays retained for the whole test; the encoder takes a retain of its own.
        device_handle: unsafe { MetalHandle::new(Retained::as_ptr(&device) as u64) },
        record_handle: DeviceRecordHandle::NULL,
        coherent_seq_ptr: core::ptr::from_ref(&coherent) as u64,
        upload_coherent_seq_ptr: 0,
        failed_submit_seq_ptr: 0,
        retained_bytes_ptr: core::ptr::from_ref(&retained) as u64,
        paced_present_seq_ptr: 0,
        largest_free_word_ptr: 0,
        guest_page_bytes_ptr: 0,
        guest_pool_bytes_ptr: 0,
    };
    let mut enc = crate::encoder::FrameEncoder::new(
        GpuCaps::apple_silicon_default(),
        Arc::new(Mtld3dConfig::default()),
        None,
        Arc::new(ClockCalibration::new()),
        &context,
        crate::encoder::NativeSubmitSpawner,
    )
    .expect("an encoder on the default device");
    // SAFETY: tests; opaque attachment values, never dereferenced since nothing is submitted.
    let back_buffer = unsafe { MetalHandle::new(0x1000) };
    // SAFETY: as above.
    let depth = unsafe { MetalHandle::new(0x2000) };
    // SAFETY: as above.
    let colour = unsafe { MetalHandle::new(0x3000) };
    enc.pass_state.reset_frame(&FrameReset {
        backbuffer: back_buffer,
        backbuffer_srgb: MetalHandle::NULL,
        backbuffer_msaa: MetalHandle::NULL,
        backbuffer_msaa_srgb: MetalHandle::NULL,
        backbuffer_sample_count: 1,
        backbuffer_size: (640, 480),
        backbuffer_format: PixelFormat::Bgra8Unorm,
        backbuffer_contents: BackbufferContents::Preserved,
        depth_texture: depth,
        depth_size: (640, 480),
        depth_has_stencil: true,
        render_scale: RenderScale::IDENTITY,
        continues_frame: false,
    });
    let planes = ClearPlanes::DEPTH | ClearPlanes::STENCIL;
    enc.pass_state.set_color_render_target(
        MetalHandle::NULL,
        640,
        480,
        PixelFormat::R8Unorm,
        RenderScale::IDENTITY,
    );

    enc.cleared_targets.begin_frame();
    enc.cleared_targets.record(depth, 0, planes);
    assert!(!enc.targets_rebuilt(planes), "both planes cleared once");
    enc.cleared_targets.begin_frame();
    enc.cleared_targets.record(depth, 0, ClearPlanes::DEPTH);
    assert!(
        !enc.targets_rebuilt(planes),
        "the stencil plane cleared in one frame only"
    );
    assert!(
        enc.targets_rebuilt(ClearPlanes::DEPTH),
        "the depth plane cleared in both frames"
    );
    enc.cleared_targets.record(depth, 0, ClearPlanes::STENCIL);
    assert!(
        enc.targets_rebuilt(planes),
        "both planes cleared in both frames"
    );

    enc.pass_state.set_color_render_target(
        colour,
        640,
        480,
        PixelFormat::Bgra8Unorm,
        RenderScale::IDENTITY,
    );
    assert!(
        !enc.targets_rebuilt(planes),
        "a colour target no clear reached keeps the draw"
    );
    enc.shutdown_cleanup();
}
