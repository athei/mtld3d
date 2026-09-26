use mtld3d_shared::{Command, MetalHandle, mtl::PixelFormat, mtl_handle::MTLTextureKind};

use super::{
    ClearHistory, ClearPlanes, CompileLanes, DeferredPipelineId, DeferredPipelines, DeferredState,
    FEED_MEMORY_FRAMES, JobTicket, LibrarySlot, TicketSource, mark_kept_reads, may_skip_draw,
};
use crate::{
    depth_stencil_state::DepthStencilSnapshot,
    passes::{BackbufferContents, FrameReset, PassState, UploadPassTarget},
    pipeline_state::PipelineAttachFlags,
    render_scale::RenderScale,
};

fn texture(raw: u64) -> MetalHandle<MTLTextureKind> {
    // SAFETY: the value is an opaque test identity; nothing dereferences it.
    unsafe { MetalHandle::new(raw) }
}

/// Urgent jobs start before normal ones, and each lane keeps its order.
#[test]
fn urgent_jobs_pop_first_in_order() {
    let mut tickets = TicketSource::new();
    let mut lanes = CompileLanes::new();
    let first = tickets.issue();
    let second = tickets.issue();
    let urgent = tickets.issue();
    lanes.push_normal(first, "first");
    lanes.push_normal(second, "second");
    lanes.push_urgent(urgent, "urgent");
    assert_eq!(lanes.len(), 3);
    assert_eq!(lanes.pop(), Some((urgent, "urgent")));
    assert_eq!(lanes.pop(), Some((first, "first")));
    assert_eq!(lanes.pop(), Some((second, "second")));
    assert_eq!(lanes.pop(), None);
    assert!(lanes.is_empty());
}

/// Promoting a queued normal job moves it ahead of every other normal job.
#[test]
fn promote_moves_a_normal_job_to_the_urgent_lane() {
    let mut tickets = TicketSource::new();
    let mut lanes = CompileLanes::new();
    let first = tickets.issue();
    let waited = tickets.issue();
    lanes.push_normal(first, 1);
    lanes.push_normal(waited, 2);
    assert!(lanes.promote(waited));
    assert!(lanes.promote(waited), "promoting twice keeps it urgent");
    assert_eq!(lanes.len(), 2, "promotion moves, it does not copy");
    assert_eq!(lanes.pop(), Some((waited, 2)));
    assert_eq!(lanes.pop(), Some((first, 1)));
}

/// A job a worker already took cannot be promoted or stolen.
#[test]
fn a_started_job_is_neither_promoted_nor_stolen() {
    let mut tickets = TicketSource::new();
    let mut lanes = CompileLanes::new();
    let started = tickets.issue();
    lanes.push_normal(started, ());
    assert_eq!(lanes.pop(), Some((started, ())));
    assert!(!lanes.promote(started));
    assert_eq!(lanes.steal(started, 0), None);
}

/// Stealing takes an unstarted urgent job by ticket and leaves the rest queued.
#[test]
fn steal_takes_only_the_named_urgent_job() {
    let mut tickets = TicketSource::new();
    let mut lanes = CompileLanes::new();
    let normal = tickets.issue();
    let urgent = tickets.issue();
    let other = tickets.issue();
    lanes.push_normal(normal, "normal");
    lanes.push_urgent(urgent, "urgent");
    lanes.push_urgent(other, "other");
    assert_eq!(
        lanes.steal(normal, 0),
        None,
        "a normal job is promoted first"
    );
    assert_eq!(lanes.steal(other, 0), Some("other"));
    assert_eq!(lanes.steal(other, 0), None, "a job leaves the lanes once");
    assert_eq!(lanes.steal_urgent(0), Some((urgent, "urgent")));
    assert_eq!(lanes.steal_urgent(0), None);
    assert_eq!(lanes.pop(), Some((normal, "normal")));
}

/// Every attached colour target and every plane the draw uses has to be regenerated.
#[test]
fn a_draw_skips_only_when_everything_it_depends_on_is_regenerated() {
    assert!(may_skip_draw(&[true], None, None));
    assert!(!may_skip_draw(&[false], None, None));
    assert!(
        !may_skip_draw(&[true, false], None, None),
        "every attached colour target counts"
    );
    assert!(may_skip_draw(&[true], Some(true), Some(true)));
    assert!(
        !may_skip_draw(&[true], Some(false), None),
        "a depth test against a kept attachment waits"
    );
    assert!(
        !may_skip_draw(&[true], Some(true), Some(false)),
        "a stencil test against a kept plane waits"
    );
    assert!(
        !may_skip_draw(&[false], Some(true), None),
        "a depth-only draw into a regenerated depth still feeds a kept colour target"
    );
}

/// One clear is not enough; a clear in each of two consecutive frames is.
#[test]
fn an_attachment_is_regenerated_after_two_consecutive_cleared_frames() {
    let mut history = ClearHistory::new();
    let rt = texture(0x100);
    history.record(rt, 0, ClearPlanes::COLOR);
    assert!(
        !history.regenerated(rt, 0, ClearPlanes::COLOR),
        "a first clear may open a one-off render"
    );
    history.begin_frame();
    assert!(!history.regenerated(rt, 0, ClearPlanes::COLOR));
    history.record(rt, 0, ClearPlanes::COLOR);
    history.record(rt, 0, ClearPlanes::COLOR);
    assert!(history.regenerated(rt, 0, ClearPlanes::COLOR));
    history.begin_frame();
    assert!(
        !history.regenerated(rt, 0, ClearPlanes::COLOR),
        "not before this frame's own clear"
    );
    history.record(rt, 0, ClearPlanes::COLOR);
    assert!(history.regenerated(rt, 0, ClearPlanes::COLOR));
}

/// A frame without the clear breaks the streak and the entry is dropped.
#[test]
fn a_skipped_clear_restarts_the_streak() {
    let mut history = ClearHistory::new();
    let rt = texture(0x100);
    history.record(rt, 0, ClearPlanes::COLOR);
    history.begin_frame();
    history.begin_frame();
    history.record(rt, 0, ClearPlanes::COLOR);
    assert!(!history.regenerated(rt, 0, ClearPlanes::COLOR));
    history.begin_frame();
    history.record(rt, 0, ClearPlanes::COLOR);
    assert!(history.regenerated(rt, 0, ClearPlanes::COLOR));
    history.begin_frame();
    history.begin_frame();
    assert!(history.is_empty(), "stale attachments are forgotten");
}

/// Planes and subresources are tracked apart, and a null texture never counts.
#[test]
fn planes_and_subresources_are_separate() {
    let mut history = ClearHistory::new();
    let ds = texture(0x200);
    for _ in 0..2 {
        history.begin_frame();
        history.record(ds, 0, ClearPlanes::DEPTH);
        history.record(ds, 1, ClearPlanes::DEPTH | ClearPlanes::STENCIL);
        history.record(MetalHandle::NULL, 0, ClearPlanes::COLOR);
    }
    assert!(history.regenerated(ds, 0, ClearPlanes::DEPTH));
    assert!(
        !history.regenerated(ds, 0, ClearPlanes::STENCIL),
        "a depth-only clear leaves stencil kept"
    );
    assert!(history.regenerated(ds, 1, ClearPlanes::STENCIL));
    assert!(!history.regenerated(ds, 2, ClearPlanes::DEPTH));
    assert!(!history.regenerated(MetalHandle::NULL, 0, ClearPlanes::COLOR));
}

/// A destroyed texture's address names a new one, which inherits none of its clears.
#[test]
fn a_forgotten_texture_starts_over_at_its_address() {
    let mut history = ClearHistory::new();
    let rt = texture(0x300);
    history.record(rt, 0, ClearPlanes::COLOR);
    history.begin_frame();
    history.record(rt, 0, ClearPlanes::COLOR);
    assert!(history.regenerated(rt, 0, ClearPlanes::COLOR));
    history.forget(rt);
    history.record(rt, 0, ClearPlanes::COLOR);
    assert!(
        !history.regenerated(rt, 0, ClearPlanes::COLOR),
        "a one-off target at a reused address is not rebuilt every frame"
    );
    assert!(history.tracks(rt));
}

/// A `Reset` forgets every texture at once.
#[test]
fn clearing_forgets_every_texture() {
    let mut history = ClearHistory::new();
    let rt = texture(0x300);
    let ds = texture(0x400);
    for _ in 0..2 {
        history.begin_frame();
        history.record(rt, 0, ClearPlanes::COLOR);
        history.record(ds, 0, ClearPlanes::DEPTH);
    }
    history.mark_feeds_persistent(rt);
    history.clear();
    assert!(history.is_empty());
    assert!(!history.regenerated(rt, 0, ClearPlanes::COLOR));
    assert!(!history.regenerated(ds, 0, ClearPlanes::DEPTH));
    assert!(!history.feeds_persistent(rt));
}

/// A read into kept content holds for many frames after it, then lapses.
#[test]
fn feeding_kept_content_lasts_the_feed_memory_then_lapses() {
    let mut history = ClearHistory::new();
    let scratch = texture(0x500);
    assert!(!history.feeds_persistent(scratch));
    assert!(!history.tracks(scratch));
    assert!(
        history.mark_feeds_persistent(scratch),
        "a first mark is new"
    );
    assert!(!history.mark_feeds_persistent(scratch), "a repeat is not");
    assert!(history.feeds_persistent(scratch));
    assert!(!history.tracks(scratch), "a read is not a clear");
    for _ in 0..FEED_MEMORY_FRAMES {
        history.begin_frame();
    }
    assert!(
        history.feeds_persistent(scratch),
        "a periodic kept read keeps its mark between reads"
    );
    history.begin_frame();
    assert!(!history.feeds_persistent(scratch));
    assert!(history.is_empty(), "and the texture is forgotten");
    assert!(!history.mark_feeds_persistent(MetalHandle::NULL));
    assert!(history.is_empty());
}

/// A pass state recording texture binds, on a discard-effect back buffer.
fn recording_passes() -> PassState {
    let mut passes = PassState::new();
    passes.reset_frame(&FrameReset {
        backbuffer: texture(0x1000),
        backbuffer_srgb: MetalHandle::NULL,
        backbuffer_msaa: MetalHandle::NULL,
        backbuffer_msaa_srgb: MetalHandle::NULL,
        backbuffer_sample_count: 1,
        backbuffer_size: (64, 64),
        backbuffer_format: PixelFormat::Bgra8Unorm,
        backbuffer_contents: BackbufferContents::Undefined,
        depth_texture: MetalHandle::NULL,
        depth_size: (0, 0),
        depth_has_stencil: false,
        render_scale: RenderScale::IDENTITY,
        continues_frame: false,
    });
    passes.record_pass_reads(true);
    passes
}

/// A history in which each of `targets` was cleared in this frame and the one before.
fn rebuilt(targets: &[MetalHandle<MTLTextureKind>]) -> ClearHistory {
    let mut history = ClearHistory::new();
    for _ in 0..2 {
        history.begin_frame();
        for &target in targets {
            history.record(target, 0, ClearPlanes::COLOR);
        }
    }
    history
}

/// Bind `target` as render target 0 and sample `read` in the pass that opens.
fn sample_into(passes: &mut PassState, target: MetalHandle<MTLTextureKind>, read: u64) {
    passes.set_color_render_target(
        target,
        64,
        64,
        PixelFormat::Bgra8Unorm,
        RenderScale::IDENTITY,
    );
    passes.emit_command(Command::set_fragment_texture(read, 0));
}

/// Record a draw in the open pass that overwrites the stencil plane it attaches.
fn write_stencil(passes: &mut PassState) {
    passes.note_draw_depth_stencil(
        &DepthStencilSnapshot::stencil_overwrite(),
        PipelineAttachFlags::HAS_DEPTH | PipelineAttachFlags::HAS_STENCIL,
    );
}

/// A kept pass marks the scratch target it samples; a rebuilt pass marks nothing.
#[test]
fn a_kept_pass_marks_what_it_samples() {
    let scratch = texture(0x2000);
    let kept = texture(0x3000);
    let mut history = rebuilt(&[scratch]);
    let mut passes = recording_passes();
    sample_into(&mut passes, texture(0x1000), scratch.raw());
    mark_kept_reads(&passes, &mut history);
    assert!(
        !history.feeds_persistent(scratch),
        "the back buffer is rebuilt every frame"
    );
    sample_into(&mut passes, kept, scratch.raw());
    mark_kept_reads(&passes, &mut history);
    assert!(history.feeds_persistent(scratch));
}

/// An upload pass spliced in ahead of the application passes does not shift a recorded read.
#[test]
fn an_upload_pass_spliced_ahead_leaves_recorded_reads_on_their_pass() {
    let scratch = texture(0x2000);
    let kept = texture(0x3000);
    let mut history = rebuilt(&[scratch]);
    let mut passes = recording_passes();
    sample_into(&mut passes, kept, scratch.raw());
    // The upload writes the scratch target, which is rebuilt: were the read
    // shifted onto it, the kept pass's read would go unmarked.
    passes.push_upload_pass(
        &UploadPassTarget {
            texture: scratch,
            subresource: (0, 0),
            size: (64, 64),
            format: PixelFormat::Bgra8Unorm,
            rect: (0, 0, 64, 64),
        },
        &[Command::set_fragment_texture(0, 0)],
        Vec::new(),
    );
    assert_eq!(passes.passes().len(), 2, "the upload pass went in first");
    mark_kept_reads(&passes, &mut history);
    assert!(history.feeds_persistent(scratch));
}

/// A chain of scratch targets feeding a kept one is marked whole in one submission.
#[test]
fn a_chain_of_scratch_targets_is_marked_in_one_submission() {
    let first = texture(0x2000);
    let second = texture(0x2100);
    let kept = texture(0x3000);
    let mut history = rebuilt(&[first, second]);
    let mut passes = recording_passes();
    sample_into(&mut passes, second, first.raw());
    sample_into(&mut passes, kept, second.raw());
    mark_kept_reads(&passes, &mut history);
    assert!(history.feeds_persistent(second));
    assert!(
        history.feeds_persistent(first),
        "marking the second makes the pass writing it kept, and it read the first"
    );
}

/// A pass writing retained stencil keeps the scratch texture that controls its fragments.
#[test]
fn retained_stencil_marks_its_sampled_source() {
    let scratch = texture(0x2000);
    let depth = texture(0x4000);
    let mut history = rebuilt(&[scratch]);
    history.record(depth, 0, ClearPlanes::DEPTH);
    history.begin_frame();
    history.record(scratch, 0, ClearPlanes::COLOR);
    history.record(depth, 0, ClearPlanes::DEPTH);
    let mut passes = recording_passes();
    passes.set_depth_stencil_attachment(depth, (64, 64), true, true);
    sample_into(&mut passes, texture(0x1000), scratch.raw());
    write_stencil(&mut passes);
    assert!(history.regenerated(depth, 0, ClearPlanes::DEPTH));
    assert!(!history.regenerated(depth, 0, ClearPlanes::STENCIL));
    mark_kept_reads(&passes, &mut history);
    assert!(
        history.feeds_persistent(scratch),
        "the stencil plane is retained even though depth and color are rebuilt"
    );
}

/// Absent, inactive and fully rebuilt stencil leave scratch producers skippable.
#[test]
fn regenerated_depth_and_stencil_leave_sampled_sources_skippable() {
    let scratch = texture(0x2000);
    let depth = texture(0x4000);
    for (has_stencil, writes_stencil) in [(false, false), (true, false), (true, true)] {
        let mut history = rebuilt(&[scratch]);
        let planes = if writes_stencil {
            ClearPlanes::DEPTH | ClearPlanes::STENCIL
        } else {
            ClearPlanes::DEPTH
        };
        history.record(depth, 1, planes);
        history.begin_frame();
        history.record(scratch, 0, ClearPlanes::COLOR);
        history.record(depth, 1, planes);
        let mut passes = recording_passes();
        passes.set_depth_stencil_attachment_level(depth, 1, (64, 64), true, has_stencil);
        sample_into(&mut passes, texture(0x1000), scratch.raw());
        if writes_stencil {
            write_stencil(&mut passes);
        }
        mark_kept_reads(&passes, &mut history);
        assert!(!history.feeds_persistent(scratch));
    }
}

/// A stencil clear on a different mip does not regenerate the attached plane.
#[test]
fn retained_stencil_is_tracked_at_the_attached_mip() {
    let scratch = texture(0x2000);
    let depth = texture(0x4000);
    let mut history = rebuilt(&[scratch]);
    for _ in 0..2 {
        history.begin_frame();
        history.record(scratch, 0, ClearPlanes::COLOR);
        history.record(depth, 1, ClearPlanes::DEPTH);
        history.record(depth, 0, ClearPlanes::STENCIL);
    }
    let mut passes = recording_passes();
    passes.set_depth_stencil_attachment_level(depth, 1, (64, 64), true, true);
    sample_into(&mut passes, texture(0x1000), scratch.raw());
    write_stencil(&mut passes);
    mark_kept_reads(&passes, &mut history);
    assert!(history.feeds_persistent(scratch));
}

/// The oldest urgent jobs are left to the idle workers, and only the rest may be stolen.
#[test]
fn stealing_leaves_the_idle_workers_their_jobs() {
    let mut tickets = TicketSource::new();
    let mut lanes = CompileLanes::new();
    let (first, second, third) = (tickets.issue(), tickets.issue(), tickets.issue());
    lanes.push_urgent(first, "first");
    lanes.push_urgent(second, "second");
    lanes.push_urgent(third, "third");
    assert_eq!(lanes.steal(first, 2), None, "an idle worker takes it next");
    assert_eq!(lanes.steal(second, 2), None, "so does the other one");
    assert_eq!(
        lanes.steal_urgent(3),
        None,
        "every job has a worker waiting for it"
    );
    assert_eq!(lanes.steal_urgent(2), Some((third, "third")));
    assert_eq!(
        lanes.pop(),
        Some((first, "first")),
        "the workers' jobs stay queued"
    );
    assert_eq!(lanes.steal(second, 0), Some("second"));
}

/// Deferred records over plain numbers: functions and pipelines are `u32`, templates `&str`.
const fn records() -> DeferredPipelines<u32, u32, &'static str> {
    DeferredPipelines::new()
}

fn libraries(vs: LibrarySlot<u32>, ps: LibrarySlot<u32>) -> DeferredState<u32, u32> {
    DeferredState::Libraries { vs, ps }
}

/// Queue the pipeline of every record whose libraries are in, as `ticket`.
fn queue_pipelines(deferred: &mut DeferredPipelines<u32, u32, &'static str>, ticket: JobTicket) {
    deferred.advance(|_, _, _| DeferredState::Pipeline(ticket));
}

/// A placeholder carries the top bit no handle has, and round-trips its record.
#[test]
fn a_placeholder_never_equals_a_real_handle() {
    let mut deferred = records();
    let mut tickets = TicketSource::new();
    let id = deferred.defer(
        DeferredState::Pipeline(tickets.issue()),
        |_| false,
        || "draw",
    );
    let raw = id.placeholder();
    assert!(DeferredPipelineId::is_placeholder(raw));
    assert_eq!(
        DeferredPipelineId::from_placeholder(raw).map(|id| id.placeholder()),
        Some(raw)
    );
    // The highest user-space address a macOS process maps.
    let real = 0x0000_7FFF_FFFF_FFFF_u64;
    assert!(!DeferredPipelineId::is_placeholder(real));
    assert!(DeferredPipelineId::from_placeholder(real).is_none());
    assert!(DeferredPipelineId::from_placeholder(0).is_none());
}

/// Libraries landing in either order both lead to the pipeline build and its answer.
#[test]
fn libraries_landing_in_either_order_reach_the_pipeline() {
    for vs_first in [true, false] {
        let mut tickets = TicketSource::new();
        let (vs, ps, pipeline) = (tickets.issue(), tickets.issue(), tickets.issue());
        let mut deferred = records();
        let id = deferred.defer(
            libraries(LibrarySlot::Pending(vs), LibrarySlot::Pending(ps)),
            |_| false,
            || "draw",
        );
        assert_eq!(deferred.pending_tickets(), [vs, ps]);
        let (first, second) = if vs_first { (vs, ps) } else { (ps, vs) };
        deferred.on_library(first, Some(7));
        let mut asked = 0;
        deferred.advance(|_, _, _| {
            asked += 1;
            DeferredState::Failed
        });
        assert_eq!(asked, 0, "one library is still building");
        deferred.on_library(second, Some(7));
        deferred.advance(|template, vs_fn, ps_fn| {
            assert_eq!((*template, vs_fn, ps_fn), ("draw", 7, 7));
            DeferredState::Pipeline(pipeline)
        });
        assert_eq!(deferred.pending_tickets(), [pipeline]);
        assert_eq!(deferred.answer(&id), None, "no answer while it builds");
        deferred.on_pipeline(pipeline, Some(42));
        assert!(deferred.pending_tickets().is_empty());
        assert_eq!(deferred.answer(&id), Some(42));
    }
}

/// A failed library fails the record, and so does a failed pipeline.
#[test]
fn a_failed_library_or_pipeline_fails_the_record() {
    let mut tickets = TicketSource::new();
    let (vs, ps, pipeline) = (tickets.issue(), tickets.issue(), tickets.issue());
    let mut deferred = records();
    let by_library = deferred.defer(
        libraries(LibrarySlot::Ready(1), LibrarySlot::Pending(ps)),
        |_| false,
        || "library",
    );
    let by_pipeline = deferred.defer(DeferredState::Pipeline(pipeline), |_| false, || "pipeline");
    deferred.on_library(vs, None);
    assert_eq!(
        deferred.pending_tickets(),
        [ps, pipeline],
        "an unrelated failure changes nothing"
    );
    deferred.on_library(ps, None);
    deferred.on_pipeline(pipeline, None);
    assert!(deferred.pending_tickets().is_empty());
    assert_eq!(deferred.answer(&by_library), None);
    assert_eq!(deferred.answer(&by_pipeline), None);
}

/// Records waiting on one ticket all advance when it lands.
#[test]
fn records_sharing_a_ticket_advance_together() {
    let mut tickets = TicketSource::new();
    let (ps, pipeline) = (tickets.issue(), tickets.issue());
    let mut deferred = records();
    let first = deferred.defer(
        libraries(LibrarySlot::Ready(1), LibrarySlot::Pending(ps)),
        |_| false,
        || "first",
    );
    let second = deferred.defer(
        libraries(LibrarySlot::Ready(2), LibrarySlot::Pending(ps)),
        |_| false,
        || "second",
    );
    assert_eq!(deferred.pending_tickets(), [ps], "one ticket, named once");
    deferred.on_library(ps, Some(3));
    queue_pipelines(&mut deferred, pipeline);
    deferred.on_pipeline(pipeline, Some(9));
    assert_eq!(deferred.answer(&first), Some(9));
    assert_eq!(deferred.answer(&second), Some(9));
    let mut ready = Vec::new();
    deferred.for_each_ready(|template, pipeline| ready.push((*template, pipeline)));
    assert_eq!(ready, [("first", 9), ("second", 9)]);
}

/// Consecutive identical draws share a record; a different template or state gets its own.
#[test]
fn consecutive_identical_draws_share_one_record() {
    let mut tickets = TicketSource::new();
    let (ps, other) = (tickets.issue(), tickets.issue());
    let mut deferred = records();
    let state = || libraries(LibrarySlot::Ready(1), LibrarySlot::Pending(ps));
    let first = deferred.defer(state(), |_| false, || "draw");
    let mut built = false;
    let again = deferred.defer(
        state(),
        |template| *template == "draw",
        || {
            built = true;
            "draw"
        },
    );
    assert!(!built, "a reused record builds no template");
    assert_eq!(first.placeholder(), again.placeholder());
    let other_template = deferred.defer(state(), |template| *template == "other", || "other");
    assert_ne!(first.placeholder(), other_template.placeholder());
    let other_state = deferred.defer(
        libraries(LibrarySlot::Ready(1), LibrarySlot::Pending(other)),
        |_| true,
        || "other",
    );
    assert_ne!(other_template.placeholder(), other_state.placeholder());
}

/// A lost pipeline is retried from its template, a lost library fails, and the rest stay.
#[test]
fn lost_tickets_retry_pipelines_and_fail_libraries() {
    let mut tickets = TicketSource::new();
    let (built, stuck, library, other, again) = (
        tickets.issue(),
        tickets.issue(),
        tickets.issue(),
        tickets.issue(),
        tickets.issue(),
    );
    let mut deferred = records();
    let done = deferred.defer(DeferredState::Pipeline(built), |_| false, || "done");
    let waiting = deferred.defer(DeferredState::Pipeline(stuck), |_| false, || "waiting");
    let by_library = deferred.defer(
        libraries(LibrarySlot::Ready(1), LibrarySlot::Pending(library)),
        |_| false,
        || "library",
    );
    let unaffected = deferred.defer(DeferredState::Pipeline(other), |_| false, || "other");
    deferred.on_pipeline(built, Some(5));
    let mut retried = Vec::new();
    deferred.retry_lost(&[stuck, library], |template| {
        retried.push(*template);
        DeferredState::Pipeline(again)
    });
    assert_eq!(retried, ["waiting"], "only the lost pipeline is retried");
    assert_eq!(deferred.pending_tickets(), [again, other]);
    deferred.on_pipeline(again, Some(6));
    assert_eq!(deferred.answer(&done), Some(5));
    assert_eq!(deferred.answer(&waiting), Some(6));
    assert_eq!(deferred.answer(&by_library), None);
    assert_eq!(deferred.answer(&unaffected), None, "still building");
    deferred.clear();
    assert!(deferred.is_empty());
    assert_eq!(
        deferred.answer(&done),
        None,
        "a cleared record names nothing"
    );
}
