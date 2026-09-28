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
