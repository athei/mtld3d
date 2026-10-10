//! Unit tests for the perf counters, the bottleneck classifier and the summary layout.
//!
//! Built only under `perf_tracking`, these pin arithmetic that otherwise fails silently:
//! the API-to-encoder counter drain with its per-frame reset, window accumulation keeping
//! encoder CPU separate from submit-thread drawable wait, exclusive-time accounting for
//! nested timers (self-times partition the outer span), every `Bottleneck::classify`
//! branch, a golden snapshot of the summary grid in both plain and ANSI form, and a
//! golden snapshot of the `perf-kv` line with its key-naming contract.

use rustc_hash::FxHashSet;

use super::*;

/// Convert cycles of the tests' fixed 1 GHz clock to milliseconds.
///
/// The summary renders through this instead of the calibrated
/// `mtld3d_shared::tsc::cycles_to_ms`, so a fixture's cycles are nanoseconds
/// on every host and the goldens below hold on a 24 MHz counter as on a 1 GHz
/// one. Only rendering uses this clock: a value converted on the ingest side
/// through `ns_to_cycles` or `secs_to_cycles` is in host ticks, not test
/// nanoseconds, so a test that feeds nanoseconds through ingest and checks the
/// rendered milliseconds depends on the host's counter rate.
pub(super) fn cycles_to_ms(cycles: u64) -> f64 {
    mtld3d_shared::tsc::u64_to_f64_exact(cycles) / 1e6
}

pub(super) const fn sample(enc_cyc: u64, drawable_wait: u64) -> FrameSample {
    FrameSample {
        counters: FrameCounters::new(),
        timing: FrameTiming::new(),
        enc: EncoderFrameCounters {
            drawable_wait_cycles: drawable_wait,
            ..EncoderFrameCounters::new()
        },
        passes: 0,
        commands: 0,
        draws: 0,
        scratch_small_blocks: 0,
        scratch_oversized_blocks: 0,
        scratch_bytes: 0,
        cmd_vec_capacity_bytes: 0,
        vbib_retention_depth: 0,
        vbib_retained_bytes: 0,
        pending_blit_retention_depth: 0,
        tex_staging_retained_bytes: 0,
        cmd_vec_realloc_bytes: 0,
        pagebox_allocs: 0,
        pagebox_alloc_bytes: 0,
        pagebox_frees: 0,
        pagebox_free_bytes: 0,
        pagebox_uncached_allocs: 0,
        pagebox_pool_bytes: 0,
        outside_d3d9: 0,
        api_cyc: 0,
        api_work: 0,
        enc_cyc,
        submit_status: 0,
    }
}

/// Encoder-thread CPU (`enc_work`) is just `enc_cyc` (op + finalize).
///
/// `drawable_wait` is submit-thread time tracked independently — it is
/// NOT subtracted from the encoder bucket (the submit moved off-thread).
/// Verify `PerfWindow::accumulate` keeps the two separate.
#[test]
fn perf_window_enc_work_identity() {
    let mut w = PerfWindow::new();
    w.accumulate(&sample(1000, 400));
    w.accumulate(&sample(1500, 800));
    assert_eq!(w.enc_cyc.sum, 2500);
    assert_eq!(w.drawable_wait.sum, 1200);
    assert_eq!(w.enc_work.sum, 2500);
    assert_eq!(w.enc_work.sum, w.enc_cyc.sum);
}

/// A large submit-thread `drawable_wait` never reduces the encoder-thread CPU bucket.
///
/// They are independent threads now.
#[test]
fn perf_window_enc_work_independent_of_drawable_wait() {
    let mut w = PerfWindow::new();
    w.accumulate(&sample(100, 5000));
    assert_eq!(w.enc_work.sum, 100);
    assert_eq!(w.drawable_wait.sum, 5000);
}

/// A sample with the submit thread's execute and its wait for the previous present.
const fn sample_submit(enc_cyc: u64, submit_exec: u64, present_wait: u64) -> FrameSample {
    let mut s = sample(enc_cyc, 0);
    s.enc.submit_exec_cycles = submit_exec;
    s.enc.present_wait_cycles = present_wait;
    s
}

/// The wait for the previous present is submit-thread time, never encoder CPU.
#[test]
fn perf_window_enc_work_independent_of_present_wait() {
    let mut w = PerfWindow::new();
    w.accumulate(&sample_submit(100, 6000, 5000));
    assert_eq!(w.enc_work.sum, 100);
    assert_eq!(w.present_wait.sum, 5000);
    assert_eq!(w.submit_exec.sum, 6000);
}

/// `Encode+commit`'s peak is the execute less the present wait, on any one frame.
#[test]
fn perf_window_encode_commit_excludes_present_wait() {
    let mut w = PerfWindow::new();
    w.accumulate(&sample_submit(0, 1000, 400));
    w.accumulate(&sample_submit(0, 900, 100));
    assert_eq!(w.encode_commit.max, 800);
    assert_eq!(w.submit_exec.sum, 1900);
}

/// Snapshots carry across the per-frame reset and clear once sampled.
#[test]
fn snapshots_survive_begin_frame_until_sampled() {
    let mut state = EncoderPerfState::new();
    state.bump_snapshot();
    state.bump_snapshot();
    state.bump_slot_wait();
    state.begin_frame(&FramePerfPayload::default());
    assert_eq!(
        state.enc.snapshots, 2,
        "the barrier's bumps outlive the reset"
    );
    assert_eq!(state.enc.slot_waits, 1, "and so does the wait it counted");
    assert_eq!(state.enc.op_cycles, 0, "every other counter was reset");
}

/// Every submission folded before a sample adds its execute, its split and its GPU time.
///
/// All of it outlives the per-frame reset, like the snapshots: a report
/// folded between a summary and the next frame still counts, and the
/// sample that takes it clears it.
#[test]
fn submit_timings_fold_adds_and_survives_until_sampled() {
    let frame = CommandBufferRole::Frame as usize;
    let upload = CommandBufferRole::Upload as usize;
    let present = CommandBufferRole::Present as usize;
    let mut first = SubmitTimings::new();
    first.leading_blits_ns = 1_000;
    first.passes_ns = 2_000;
    first.commit_ns = 3_000;
    first.gpu[frame].ns = 4_000_000;
    first.gpu[frame].buffers = 1;
    let mut second = SubmitTimings::new();
    second.passes_ns = 5_000;
    second.gpu[frame].ns = 6_000_000;
    second.gpu[frame].buffers = 1;
    second.gpu[present].ns = 1_000_000;
    second.gpu[present].buffers = 2;

    let mut state = EncoderPerfState::new();
    state.fold_submit_timings(&first, 7_000);
    state.fold_submit_timings(&second, 8_000);
    assert_eq!(state.enc.submit_exec_cycles, 15_000, "the executes add");
    assert_eq!(state.enc.submit_blits_cycles, ns_to_cycles(1_000));
    let passes = ns_to_cycles(2_000) + ns_to_cycles(5_000);
    assert_eq!(state.enc.submit_passes_cycles, passes, "the splits add");
    assert_eq!(state.enc.submit_commit_cycles, ns_to_cycles(3_000));
    let frame_cycles = ns_to_cycles(4_000_000) + ns_to_cycles(6_000_000);
    assert_eq!(state.enc.gpu_cycles[frame], frame_cycles, "GPU time adds");
    assert_eq!(state.enc.gpu_cycles[upload], 0);
    assert_eq!(state.enc.gpu_cycles[present], ns_to_cycles(1_000_000));
    assert_eq!(state.enc.gpu_buffers, [2, 0, 2]);

    state.begin_frame(&FramePerfPayload::default());
    assert_eq!(
        state.enc.gpu_cycles[frame], frame_cycles,
        "sticky across the reset"
    );
    assert_eq!(state.enc.gpu_buffers, [2, 0, 2]);
    assert_eq!(
        state.enc.submit_passes_cycles, passes,
        "and so is the split"
    );
    assert_eq!(state.enc.submit_exec_cycles, 15_000);

    let sampled = state.take_sample(&sample_caches(), &[], 0, 0);
    assert_eq!(sampled.enc.submit_passes_cycles, passes);
    assert_eq!(sampled.enc.gpu_buffers, [2, 0, 2]);
    assert_eq!(state.enc.submit_exec_cycles, 0, "the sample clears it");
    assert_eq!(state.enc.submit_passes_cycles, 0);
    assert_eq!(state.enc.gpu_buffers, [0; CommandBufferRole::COUNT]);
}

/// Both waits add over a sample's submissions.
///
/// Two present-bearing submissions returning in one frame each report the
/// drawable waits committed before them, so neither wait may be dropped.
#[test]
fn submit_waits_add_over_a_samples_submissions() {
    let mut state = EncoderPerfState::new();
    state.add_submit_wait_nanos(3_000_000, 1_000_000);
    state.add_submit_wait_nanos(2_000_000, 2_000_000);
    assert_eq!(
        state.enc.drawable_wait_cycles,
        ns_to_cycles(3_000_000) + ns_to_cycles(2_000_000)
    );
    assert_eq!(
        state.enc.present_wait_cycles,
        ns_to_cycles(1_000_000) + ns_to_cycles(2_000_000)
    );
}

/// The submit thread's timings reach the window when a blocking flush drains them.
///
/// The order a game that flushes every frame produces: an async present,
/// the flush's barrier draining that present's submission before the
/// flush's own frame begins, the flush's synchronous submit, then the next
/// async present. The present's passes land in the flush's sample with the
/// flush's own, and the next frame reports nothing twice.
#[test]
fn a_drained_present_submission_reaches_the_flush_sample() {
    let timed = || {
        let mut payload = FramePerfPayload::new();
        payload.counters.timed = 1;
        payload
    };
    let caches = sample_caches();
    let mut state = EncoderPerfState::new();
    let mut window = PerfWindow::new();

    // The async present: its submission returns after its own summary.
    state.begin_frame(&timed());
    window.accumulate(&state.take_sample(&caches, &[], 0, 0));

    // The flush: the barrier drains the present's submission first.
    let mut present = SubmitTimings::new();
    present.passes_ns = 2_000_000;
    present.commit_ns = 100_000;
    state.add_submit_wait_nanos(4_000_000, 500_000);
    state.fold_submit_timings(&present, 3_000_000);
    state.begin_frame(&timed());
    let mut flush = SubmitTimings::new();
    flush.passes_ns = 100_000;
    state.add_submit_wait_nanos(0, 0);
    state.fold_submit_timings(&flush, 200_000);
    let flushed = state.take_sample(&caches, &[], 0, 0);
    window.accumulate(&flushed);

    // The next async present, with nothing returned yet.
    state.begin_frame(&timed());
    let next = state.take_sample(&caches, &[], 0, 0);
    window.accumulate(&next);

    let passes = ns_to_cycles(2_000_000) + ns_to_cycles(100_000);
    assert_eq!(flushed.enc.submit_passes_cycles, passes);
    assert_eq!(
        window.submit_passes.sum, passes,
        "the present's passes reach the window"
    );
    assert_eq!(window.submit_exec.sum, 3_200_000);
    assert_eq!(window.submit_commit.sum, ns_to_cycles(100_000));
    assert_eq!(window.present_wait.sum, ns_to_cycles(500_000));
    assert_eq!(window.drawable_wait.sum, ns_to_cycles(4_000_000));
    assert_eq!(
        next.enc.submit_passes_cycles, 0,
        "nothing is reported twice"
    );
    assert_eq!(next.enc.submit_exec_cycles, 0);
}

/// The encode children and their residual partition `Encode+commit` on every frame.
#[test]
fn perf_window_submit_children_leave_a_residual() {
    let mut w = PerfWindow::new();
    let mut s = sample_submit(0, 1_000, 100);
    s.enc.submit_blits_cycles = 200;
    s.enc.submit_passes_cycles = 300;
    s.enc.submit_commit_cycles = 100;
    s.enc.gpu_cycles = [700, 200, 100];
    w.accumulate(&s);
    let mut over = sample_submit(0, 500, 0);
    // Children over their parent, as a sync submit with no execute reports.
    over.enc.submit_passes_cycles = 900;
    w.accumulate(&over);
    assert_eq!(
        w.submit_resid.max, 300,
        "900 - 200 - 300 - 100, and never negative"
    );
    assert_eq!(w.submit_passes.sum, 1_200);
    assert_eq!(w.gpu[CommandBufferRole::Frame as usize].sum, 700);
}

/// A synchronous submit behind a barrier adds to the async submission the barrier drained.
///
/// Parent and children add together, so `Encode+commit` and its children
/// describe the same two submissions and the residual is theirs.
#[test]
fn barrier_submit_adds_parent_and_children_together() {
    let mut async_timings = SubmitTimings::new();
    async_timings.leading_blits_ns = 4_000;
    async_timings.passes_ns = 9_000;
    async_timings.commit_ns = 2_000;
    let mut sync_timings = SubmitTimings::new();
    sync_timings.passes_ns = 1_000;

    let mut state = EncoderPerfState::new();
    state.fold_submit_timings(&async_timings, 900_000);
    state.fold_submit_timings(&sync_timings, 300_000);
    assert_eq!(state.enc.submit_exec_cycles, 1_200_000, "both executes");
    let blits = ns_to_cycles(4_000);
    let passes = ns_to_cycles(9_000) + ns_to_cycles(1_000);
    let commit = ns_to_cycles(2_000);
    assert_eq!(state.enc.submit_blits_cycles, blits);
    assert_eq!(state.enc.submit_passes_cycles, passes);
    assert_eq!(state.enc.submit_commit_cycles, commit);

    let mut w = PerfWindow::new();
    let mut s = sample(0, 0);
    s.enc = state.enc;
    w.accumulate(&s);
    assert_eq!(w.submit_exec.sum, 1_200_000);
    assert_eq!(w.submit_resid.max, 1_200_000 - blits - passes - commit);
}

/// Upload outcome totals partition successful renames and survive only their reporting window.
#[test]
fn staged_upload_outcomes_partition_bytes_and_reset_per_device() {
    let mut first = EncoderPerfState::new();
    let second = EncoderPerfState::new();
    for (length, offset, size) in [(16_384, 0, 16_384), (32_768, 64, 48), (16_384, 0, 16_368)] {
        if crate::buffer_rename::stage_upload_needs_preserve(length, offset, size) {
            first.bump_vbib_preserve_gpu(length);
        } else {
            first.bump_vbib_full_upload_skip(length);
        }
        first.bump_vbib_mid_pass_reorder();
    }
    first.bump_vbib_reorder_alloc_failure();
    assert_eq!(first.enc.vbib_mid_pass_reorders, 3);
    assert_eq!(first.enc.vbib_full_upload_skips, 1);
    assert_eq!(first.enc.vbib_full_upload_skip_bytes, 16_384);
    assert_eq!(first.enc.vbib_preserve_gpu_bytes, 49_152);
    assert_eq!(first.enc.vbib_reorder_alloc_failures, 1);
    assert_eq!(second.enc.vbib_mid_pass_reorders, 0);
    assert_eq!(second.enc.vbib_full_upload_skips, 0);
    assert_eq!(second.enc.vbib_full_upload_skip_bytes, 0);
    assert_eq!(second.enc.vbib_preserve_gpu_bytes, 0);
    assert_eq!(second.enc.vbib_reorder_alloc_failures, 0);

    let mut frame = sample(0, 0);
    frame.enc = first.enc;
    let mut window = PerfWindow::new();
    window.accumulate(&frame);
    first.begin_frame(&FramePerfPayload::new());
    assert_eq!(first.enc.vbib_mid_pass_reorders, 0);
    assert_eq!(first.enc.vbib_full_upload_skips, 0);
    assert_eq!(first.enc.vbib_full_upload_skip_bytes, 0);
    assert_eq!(first.enc.vbib_preserve_gpu_bytes, 0);
    assert_eq!(first.enc.vbib_reorder_alloc_failures, 0);
    frame.enc = first.enc;
    window.accumulate(&frame);
    assert_eq!(window.vbib_mid_pass_reorders.sum, 3);
    assert_eq!(window.vbib_full_upload_skips.sum, 1);
    assert_eq!(window.vbib_full_upload_skip_bytes.sum, 16_384);
    assert_eq!(window.vbib_preserve_gpu_bytes.sum, 49_152);
    assert_eq!(window.vbib_reorder_alloc_failures.sum, 1);
    window.reset();
    assert_eq!(window.vbib_mid_pass_reorders.sum, 0);
    assert_eq!(window.vbib_full_upload_skips.sum, 0);
    assert_eq!(window.vbib_full_upload_skip_bytes.sum, 0);
    assert_eq!(window.vbib_preserve_gpu_bytes.sum, 0);
    assert_eq!(window.vbib_reorder_alloc_failures.sum, 0);
}

/// Byte totals keep their width on PE32 and saturate instead of wrapping.
#[test]
fn staged_upload_byte_totals_are_wide_and_saturating() {
    let mut state = EncoderPerfState::new();
    state.bump_vbib_full_upload_skip(1_u64 << 32);
    state.bump_vbib_preserve_gpu(1_u64 << 32);
    assert_eq!(state.enc.vbib_full_upload_skip_bytes, 1_u64 << 32);
    assert_eq!(state.enc.vbib_preserve_gpu_bytes, 1_u64 << 32);
    state.bump_vbib_full_upload_skip(u64::MAX);
    state.bump_vbib_preserve_gpu(u64::MAX);
    assert_eq!(state.enc.vbib_full_upload_skip_bytes, u64::MAX);
    assert_eq!(state.enc.vbib_preserve_gpu_bytes, u64::MAX);
}

/// `ApiPerfState::drain_into_payload` moves counters to the payload and zeroes the source.
///
/// First-frame `frame_total` must be 0 (no predecessor TSC yet);
/// subsequent drains report a delta.
#[test]
fn api_perf_drain_moves_and_resets() {
    let mut api = ApiPerfState::new();
    api.add_api_cycles(ApiCategory::VertexBuffer, 1234);
    api.bump_vb_rename();
    api.bump_vb_rename();
    api.bump_vbib_preserve_cpu();
    api.bump_vbib_write_in_place_contended();

    let mut p = FramePerfPayload::new();
    api.drain_into_payload(&mut p, true);

    assert_eq!(
        p.counters.api_cycles_by_category[ApiCategory::VertexBuffer as usize],
        1234
    );
    assert_eq!(
        p.counters.api_call_counts_by_category[ApiCategory::VertexBuffer as usize],
        1
    );
    assert_eq!(p.counters.vb_rename, 2);
    assert_eq!(p.counters.vbib_preserve_cpu, 1);
    assert_eq!(p.counters.vbib_write_in_place_contended, 1);
    assert_eq!(
        p.timing.frame_total_cycles, 0,
        "first frame has no predecessor"
    );

    // Source must be zeroed.
    assert_eq!(
        api.counters.api_cycles_by_category[ApiCategory::VertexBuffer as usize],
        0
    );
    assert_eq!(api.counters.vb_rename, 0);
    assert_eq!(api.counters.vbib_preserve_cpu, 0);
    assert_eq!(api.counters.vbib_write_in_place_contended, 0);

    // Second drain should report a real frame_total (tsc moved).
    let mut p2 = FramePerfPayload::new();
    api.drain_into_payload(&mut p2, true);
    assert!(
        p2.timing.frame_total_cycles > 0,
        "second drain must see a non-zero TSC delta"
    );
}

/// `EncoderPerfState::begin_frame` seeds per-frame encoder counters from the payload.
///
/// Resets the encoder-side counters without clobbering running totals
/// (`vbib_retained_bytes`).
#[test]
fn encoder_begin_frame_seeds_from_payload() {
    let mut enc = EncoderPerfState::new();
    enc.bump_vbib_retained_add(4096);
    enc.bump_buffer_destroy();
    enc.bump_texture_destroy();

    let mut payload = FramePerfPayload::new();
    payload.counters.vb_rename = 7;
    payload.set_present_block_cycles(42);
    payload.timing.frame_total_cycles = 1000;

    enc.begin_frame(&payload);

    assert_eq!(enc.counters.vb_rename, 7);
    assert_eq!(enc.timing.present_block_cycles, 42);
    assert_eq!(enc.timing.frame_total_cycles, 1000);
    assert_eq!(
        enc.enc.buffer_destroys, 0,
        "encoder-side counter reset per frame"
    );
    assert_eq!(enc.enc.texture_destroys, 0);
    assert_eq!(
        enc.vbib_retained_bytes, 4096,
        "running totals survive begin_frame"
    );
}

/// Encoder stall is the first signal.
///
/// If the API thread is blocked on the backpressure channel for > 15% of
/// the frame, classify by which encoder sub-bucket dominates.
#[test]
fn bottleneck_encoder_gpu_when_drawable_wait_dominates() {
    // frame=10ms, present_block=2ms (20%), gpu_wait=4ms, enc_cpu=1ms.
    let bn = Bottleneck::classify(10.0, 0.5, 0.5, 1.0, 4.0, 2.0);
    assert_eq!(bn, Bottleneck::EncoderGpu);
}

#[test]
fn bottleneck_encoder_cpu_when_enc_cpu_dominates() {
    // frame=10ms, present_block=3ms (30%), enc_cpu=5ms, gpu_wait=0.5ms.
    let bn = Bottleneck::classify(10.0, 1.0, 1.0, 5.0, 0.5, 3.0);
    assert_eq!(bn, Bottleneck::EncoderCpu);
}

/// API thread keeps up with the encoder (present stall is small).
///
/// The pacing side is the API thread, so the tie-break is D3D9 vs game
/// code.
#[test]
fn bottleneck_api_d3d9_when_api_work_leads() {
    // present_block tiny, api_work > outside.
    let bn = Bottleneck::classify(10.0, 6.0, 3.0, 0.5, 0.3, 0.2);
    assert_eq!(bn, Bottleneck::ApiD3d9);
}

#[test]
fn bottleneck_api_game_when_outside_leads() {
    let bn = Bottleneck::classify(10.0, 2.0, 7.0, 0.5, 0.3, 0.2);
    assert_eq!(bn, Bottleneck::ApiGame);
}

/// `Balanced` only when all four buckets are within 20 % of the per-bucket mean.
///
/// No clear winner.
#[test]
fn bottleneck_balanced_when_buckets_even() {
    // frame=10ms, each bucket ≈ 2.5ms, present_block below encoder
    // threshold.
    let bn = Bottleneck::classify(10.0, 2.5, 2.5, 2.5, 2.5, 0.5);
    assert_eq!(bn, Bottleneck::Balanced);
}

/// ANSI on and ANSI off must emit the same visible layout.
///
/// Stripping CSI escapes from the ANSI output must yield the exact
/// plain output — if that fails, escapes are leaking into cells
/// whose widths are format-padded.
#[test]
fn summary_ansi_off_matches_stripped_ansi_on() {
    let w = sample_window();
    let caches = sample_caches();
    let plain = Summary::render_with_ansi(&w, &caches, 2.01, false);
    let colored = Summary::render_with_ansi(&w, &caches, 2.01, true);
    let stripped = strip_ansi(&colored);
    assert_eq!(
        plain, stripped,
        "ANSI off output must equal the ANSI-on output with escapes stripped"
    );
}

#[test]
fn staged_upload_summary_marks_saturated_derived_counts() {
    for saturated_input in 0..4 {
        let mut window = sample_window();
        match saturated_input {
            0 => window.vbib_mid_pass_reorders.max = u64::from(u32::MAX),
            1 => window.vbib_full_upload_skips.max = u64::from(u32::MAX),
            2 => window.vbib_mid_pass_reorders.sum = u64::MAX,
            _ => window.vbib_full_upload_skips.sum = u64::MAX,
        }
        let summary = Summary::render_with_ansi(&window, &sample_caches(), 2.01, false);
        assert!(summary.contains("  GPU copy  count=saturated bytes=49152"));
        let kv = render_kv(&window, &window, &sample_caches(), 2.01).finish();
        assert!(
            !kv.contains(" vbib_gpu_copy_total="),
            "an unknown count is left out"
        );
    }
}

/// Golden-string snapshot pinning the column grid.
///
/// Every cell in the layout lands at a fixed column (`LABEL_W`,
/// `AUX_COL`, `DESC_COL`, `PEAK_COL` for tree rows; `RES_WINDOW_COL` /
/// `RES_PEAK_COL` / `RES_COMMENT_COL` for Resources rows). If a future
/// edit drifts any cell, this fails with a diff that points at the exact
/// row/col.
#[test]
fn summary_golden_layout() {
    let w = sample_window();
    let caches = sample_caches();
    let got = Summary::render_with_ansi(&w, &caches, 2.01, false);
    let want = concat!(
        "── perf  window=2.01s  frames=1  timed=1  bottleneck=ENCODER (GPU) ──\n",
        "reset epochs=0..0; inverse/upload counts are interval totals (not cumulative)\n",
        "buckets: api_d3d9=2.80  api_outside=3.00  enc_work=1.50  submit_work=0.10  gpu_wait=6.00  (ms/frame, avg)\n",
        "\n",
        "API thread             10.00 ms                                             peak 10.00 ms\n",
        "├─ D3D9 calls           4.00 ms   ( 40.0 %)           263 calls             peak  4.00 ms\n",
        "│  ├─ Device            0.30 ms   (       123)                              peak  0.30 ms\n",
        "│  │  ├─ Frame          0.04 ms   (         3)                              peak  0.04 ms\n",
        "│  │  │  ├─ Send stall  3.20 ms                       encoder backpressure  peak  3.20 ms\n",
        "│  │  │  └─ other       0.00 ms                       non-blocking body     peak  0.00 ms\n",
        "│  │  ├─ Draws          0.12 ms   (1200 ns/draw)                            peak  0.12 ms\n",
        "│  │  │  ├─ snapshot    0.09 ms   ( 900 ns/draw)      read+stamp state      peak  0.09 ms\n",
        "│  │  │  │  ├─ stages   0.02 ms   ( 200 ns/draw)      tex/samp/TSS walk     peak  0.02 ms\n",
        "│  │  │  │  ├─ c_ff     0.02 ms   ( 200 ns/draw)      FF VS+PS const build  peak  0.02 ms\n",
        "│  │  │  │  ├─ c_pr     0.01 ms   ( 100 ns/draw)      programmable consts   peak  0.01 ms\n",
        "│  │  │  │  ├─ keys     0.02 ms   ( 200 ns/draw)      VDECL+RS+variant+srcs peak  0.02 ms\n",
        "│  │  │  │  │  ├─ decl  0.00 ms   (   0 ns/draw)      attrs + FF VS layout  peak  0.00 ms\n",
        "│  │  │  │  │  ├─ rs    0.01 ms   ( 100 ns/draw)      render-state struct   peak  0.01 ms\n",
        "│  │  │  │  │  ├─ rtds  0.00 ms   (   0 ns/draw)      depth/stencil flags   peak  0.00 ms\n",
        "│  │  │  │  │  ├─ var   0.00 ms   (   0 ns/draw)      pipeline variant key  peak  0.00 ms\n",
        "│  │  │  │  │  ├─ vs    0.00 ms   (   0 ns/draw)      FF VS key / VS id     peak  0.00 ms\n",
        "│  │  │  │  │  ├─ ps    0.00 ms   (   0 ns/draw)      FF PS key / PS id     peak  0.00 ms\n",
        "│  │  │  │  │  └─ rest  0.01 ms   ( 100 ns/draw)      keys less its parts   peak  0.01 ms\n",
        "│  │  │  │  ├─ bumps    0.01 ms   ( 100 ns/draw)      scratch+cache+wrapper peak  0.01 ms\n",
        "│  │  │  │  └─ resid    0.01 ms   ( 100 ns/draw)      uninstrumented noise  peak  0.01 ms\n",
        "│  │  │  └─ push_op     0.02 ms   ( 200 ns/draw)      inline Op::Draw push  peak  0.02 ms\n",
        "│  │  ├─ RenderState    0.06 ms   (2000 ns/call)                            peak  0.06 ms\n",
        "│  │  ├─ TexStageState  0.03 ms   (1667 ns/call)                            peak  0.03 ms\n",
        "│  │  ├─ SamplerState   0.02 ms   (1667 ns/call)                            peak  0.02 ms\n",
        "│  │  ├─ ShaderConst    0.02 ms   (2500 ns/call)                            peak  0.02 ms\n",
        "│  │  ├─ Bind           0.01 ms   (1667 ns/call)                            peak  0.01 ms\n",
        "│  │  │  ├─ Texture     0.00 ms   (         2)        Set/GetTexture        peak  0.00 ms\n",
        "│  │  │  ├─ Buffer      0.00 ms   (         1)        VB/IB/StreamFreq      peak  0.00 ms\n",
        "│  │  │  ├─ Shader      0.00 ms   (         1)        VDecl/VS/PS/FVF       peak  0.00 ms\n",
        "│  │  │  ├─ RtDs        0.00 ms   (         1)        RT + DepthStencil     peak  0.00 ms\n",
        "│  │  │  ├─ FfFixed     0.00 ms   (         1)        xform/material/light  peak  0.00 ms\n",
        "│  │  │  └─ VpScissor   0.00 ms   (         0)        viewport + scissor    peak  0.00 ms\n",
        "│  │  ├─ StateBlock     0.00 ms   (         0)                              peak  0.00 ms\n",
        "│  │  └─ Misc           0.00 ms   (         0)                              peak  0.00 ms\n",
        "│  ├─ VertexBuffer      0.20 ms   (        88)                              peak  0.20 ms\n",
        "│  ├─ IndexBuffer       0.08 ms   (        32)                              peak  0.08 ms\n",
        "│  ├─ Texture           0.00 ms   (         0)                              peak  0.00 ms\n",
        "│  ├─ Surface           0.04 ms   (        20)                              peak  0.04 ms\n",
        "│  │  ├─ LockRect       0.02 ms   (         2)        map/rename/readback   peak  0.02 ms\n",
        "│  │  ├─ UnlockRect     0.01 ms   (         2)        unmap + upload queue  peak  0.01 ms\n",
        "│  │  ├─ GetDC          0.01 ms   (         1)        DIB + memory DC       peak  0.01 ms\n",
        "│  │  ├─ ReleaseDC      0.00 ms   (         0)        DC write-back         peak  0.00 ms\n",
        "│  │  └─ Misc           0.00 ms   (        15)        getters + IUnknown    peak  0.00 ms\n",
        "│  ├─ Query             0.00 ms   (         0)                              peak  0.00 ms\n",
        "│  │  └─ Wait for GPU   0.00 ms                       waitUntilCompleted    peak  0.00 ms\n",
        "│  ├─ StateBlock        0.00 ms   (         0)                              peak  0.00 ms\n",
        "│  ├─ VertexDecl        0.00 ms   (         0)                              peak  0.00 ms\n",
        "│  ├─ VertexShader      0.00 ms   (         0)                              peak  0.00 ms\n",
        "│  └─ PixelShader       0.00 ms   (         0)                              peak  0.00 ms\n",
        "└─ Outside d3d9         3.00 ms   ( 30.0 %)           game code             peak  3.00 ms\n",
        "\n",
        "Encoder thread          1.70 ms\n",
        "├─ Closures (op)        1.40 ms   ( 82.4 %)           D3D9→Metal translate  peak  1.40 ms\n",
        "│  ├─ resolve           0.30 ms   (3000 ns/draw)      tex + shader libs     peak  0.30 ms\n",
        "│  │  ├─ consts         0.15 ms   (1500 ns/draw)      VS/PS/FF const copy   peak  0.15 ms\n",
        "│  │  ├─ skip           0.09 ms   ( 900 ns/draw)      skip-set check        peak  0.09 ms\n",
        "│  │  ├─ lookup         0.03 ms   ( 300 ns/draw)      tex + lib probes      peak  0.03 ms\n",
        "│  │  └─ resid          0.03 ms   ( 300 ns/draw)      deref + glue          peak  0.03 ms\n",
        "│  ├─ pipeline          0.40 ms   (4000 ns/draw)      pipeline+depth+cull   peak  0.40 ms\n",
        "│  ├─ state             0.10 ms   (1000 ns/draw)      pass open + binds     peak  0.10 ms\n",
        "│  ├─ probe             0.20 ms   (2000 ns/draw)      decal/caster diag     peak  0.20 ms\n",
        "│  ├─ samplers          0.15 ms   (1500 ns/draw)      texture+sampler bind  peak  0.15 ms\n",
        "│  ├─ binds             0.20 ms   (2000 ns/draw)      consts + VB/IB + draw peak  0.20 ms\n",
        "│  │  ├─ cbind          0.12 ms   (1200 ns/draw)      const memcmp+bytes    peak  0.12 ms\n",
        "│  │  ├─ vbib           0.05 ms   ( 500 ns/draw)      VB/IB wrap + notify   peak  0.05 ms\n",
        "│  │  ├─ draw           0.02 ms   ( 200 ns/draw)      draw cmd emit         peak  0.02 ms\n",
        "│  │  └─ resid          0.01 ms   ( 100 ns/draw)      deref + glue          peak  0.01 ms\n",
        "│  ├─ tex_raw           0.01 ms   ( 100 ns/draw)      texture blit upload   peak  0.01 ms\n",
        "│  ├─ stage_up          0.01 ms   ( 100 ns/draw)      VB/IB staged upload   peak  0.01 ms\n",
        "│  ├─ const_rng         0.01 ms   ( 100 ns/draw)      VS/PS/FF const copy   peak  0.01 ms\n",
        "│  └─ resid             0.02 ms   ( 200 ns/draw)      other ops + noise     peak  0.02 ms\n",
        "├─ Finalize             0.10 ms   (  5.9 %)           passes + descriptors  peak  0.10 ms\n",
        "└─ Submit stall         0.20 ms   ( 11.8 %)           submit backpressure   peak  0.20 ms\n",
        "\n",
        "Submit thread           6.10 ms                                             peak  6.10 ms\n",
        "├─ Encode+commit        0.10 ms                       command-walk → Metal  peak  0.10 ms\n",
        "│  ├─ leading blits     0.01 ms                       frame-leading blits   peak  0.01 ms\n",
        "│  ├─ passes            0.05 ms                       render-pass replay    peak  0.05 ms\n",
        "│  ├─ commit            0.02 ms                       handlers + commit     peak  0.02 ms\n",
        "│  └─ resid             0.02 ms                       setup, settle, thunk  peak  0.02 ms\n",
        "└─ Present wait         6.00 ms                       prior present commit  peak  6.00 ms\n",
        "\n",
        "Present thread          6.00 ms                                             peak  6.00 ms\n",
        "├─ Drawable wait        6.00 ms                       nextDrawable GPU+comp peak  6.00 ms\n",
        "├─ Snapshots                      (         1)        presented from a copy\n",
        "└─ Slot waits                     (         0)        copy waited for a present\n",
        "\n",
        "GPU (CBs may overlap)   4.50 ms                       frame/upload/present\n",
        "├─ Frame CBs            4.00 ms   (         1)        render passes\n",
        "├─ Upload CBs           0.30 ms   (         1)        uploads + blits\n",
        "└─ Present CBs          0.20 ms   (         1)        drawable present\n",
        "\n",
        "Frame total            10.00 ms                                             peak 10.00 ms\n",
        "submit_status=0x0   (API, Encoder, Submit, Present run in parallel; frame_total ≥ max(api_cpu, enc_cpu, submit_cpu + present_wait, gpu_wait))\n",
        "\n",
        "Resources (VB/IB)  — window totals; depth/retention averaged\n",
        "rename      VB=12     IB=3          peak/frame VB=12  IB=3      API: PageBox alloc on contended Lock(DISCARD) or whole-buffer Lock\n",
        "  discards  VB=10     IB=3                                      API: rename, no preserve (DISCARD or whole-buffer WRITEONLY)\n",
        "  preserve  2                                                   API: rename + sync memcpy (whole-buffer non-WRITEONLY contended — game may read back)\n",
        "  bytes     720 KB                  peak/frame 720 KB           API: fresh PageBox bytes behind rename (16 KiB-padded; what the allocator serves)\n",
        "in-place    3                                                   API: contended partial Lock handed back live (kept divergence; no rename, no stall)\n",
        "staging up  4                                                   encoder: Staged (non-DYNAMIC) dirty-range upload blits — separate-staging path; high here with rename≈0 is the goal\n",
        "reorder     3                                                   encoder: successful overlap renames (full skip + GPU copy)\n",
        "  full skip count=1 bytes=16384                                 encoder: exact allocation overwritten; preservation omitted\n",
        "  GPU copy  count=2 bytes=49152                                 encoder: partial/padded upload; queued preservation bytes\n",
        "  allocfail 1                                                   encoder: fresh allocation failed; excluded from reorder\n",
        "destroys    1                                                   encoder: MTLBuffer wrappers freed (VB/IB cache renames, Lock-rename intake, visibility-pool eviction)\n",
        "ret cap     drain=2 submit=1        peak/frame submit=1         API: VB/IB retention cap hit before a rename alloc (drain=cheap, submit=GPU wait)\n",
        "retention   depth= 6.0  3.5 MB avg  peak depth=6   3.5 MB       encoder: shared PageBox queue (VB/IB renames + texture-blit padded staging + visibility pool)\n",
        "pool        hit=14 miss=1 (93.3%)   recycled=14  672 KB         API pops a warm same-size PageBox; both runtimes park retired ones (memory.pageboxPoolCapMB, 0 = off)\n",
        "  parked    1.0 MB avg              peak 1.0 MB                 bytes held in the pool for reuse (already retired; excluded from retention above)\n",
        "\n",
        "Resources (textures)  — same layout as VB/IB; n/a rows omitted\n",
        "rename      2                                                   API: fresh staging Arc on contended LockRect\n",
        "  discards  1                                                   API: rename, no preserve (whole-level DISCARD on a DEFAULT-pool texture)\n",
        "  preserve  1                       peak/frame 1                API: rename + sync memcpy (whole-level non-DISCARD contended, or an unaligned compressed rect)\n",
        "in-place    0                                                   API: contended partial Lock handed back live (kept divergence; no rename, no stall)\n",
        "pool        hit=7 miss=1 (87.5%)                                API: staging pops a warm same-size PageBox; last owners park retired staging\n",
        "snapshot    2 6 KB                  full=1                      API: partial uploads copying their box, for levels a partial Lock renamed (full = arena declined)\n",
        "uploads     2                                                   encoder: total texture uploads (raw + padded + pass)\n",
        "  raw       2                                                   encoder: blit; source = cached bytesNoCopy wrapper (cheap)\n",
        "  padded    0                                                   encoder: blit; source repacked on the CPU into a transient buffer (alloc + memcpy + extra unix_call)\n",
        "  pass      0                                                   encoder: render pass; staging read by a fragment function (16-bit widening, sub-alignment pitch)\n",
        "reorder     1                                                   encoder: MTLTexture rename-at-overlap (upload hit a texture sampled earlier this frame)\n",
        "destroys    1                                                   encoder: MTLTexture freed + texture-staging MTLBuffer wrappers freed (rename + padded + staging release + texture release)\n",
        "retention   depth= 0.0  0 KB avg    peak 0 KB                   encoder: blit source staging Arcs (separate from VB/IB retention; MTLTexture handles in destroys)\n",
        "wrapped     4.0 MB                                              encoder: staging under cached per-level bytesNoCopy wrappers at the summary (pins its guest pages)\n",
        "  churn     new=3 retired=2         peak/frame new=3            encoder: wrappers created, and queued for destroy (backing change, staging release, texture release)\n",
        "dirtyrect   4                       3 partial, 25% mip          API: AddDirtyRect calls; partial = sub-region narrower than mip (feeds dirty-rect snapshot upload decision)\n",
        "\n",
        "Caches (live sizes at summary emit)\n",
        "  textures=48     pipelines=12     samplers=6     programs=8\n",
        "  libs=8        depth_states=4\n",
        "\n",
        "Commands / passes (raw window totals)\n",
        "  passes=4         commands=140         draws=100\n",
        "  pipeline memo  97 / 100  (97.0%)  recent-draw resolve elided\n",
        "  fan generated  0         indexed / oversized fans rewritten per draw (slow path; 0 is the goal)\n",
        "  up indexed     3         DrawIndexedPrimitiveUP draws (inline indices copied into the upload ring)\n",
        "  up oversized   2         UP draws past the 4 KiB inline limit (vertices copied into the upload ring)\n",
        "\n",
        "Keys gating  — redundant snapshot dirty-marks elided (skips/calls); higher = more `keys` work avoided\n",
        "  SetTexture               0 / 0         (  0.0%)\n",
        "  SetRenderState           0 / 0         (  0.0%)\n",
        "  SetTexStageState         0 / 0         (  0.0%)\n",
        "  SetFvf                   0 / 0         (  0.0%)\n",
        "  SetVertexDecl            0 / 0         (  0.0%)\n",
        "  SetVertexShader          0 / 0         (  0.0%)\n",
        "  SetPixelShader           0 / 0         (  0.0%)\n",
        "  SetVsConst               0 / 0         (  0.0%)\n",
        "  SetPsConst               0 / 0         (  0.0%)\n",
        "\n",
        "Snapshot rebuilds  draws that rebuilt each section, as a share of draw calls; ns per rebuild where `keys` times it, from 1 in 16 rebuilding draws\n",
        "  any section             40 / 100       ( 40.0%)\n",
        "  RS                      10             ( 10.0%)   1000 ns/rebuild\n",
        "  STAGES                  30             ( 30.0%)\n",
        "  RT_DS                    0             (  0.0%)\n",
        "  VDECL                    0             (  0.0%)\n",
        "  VARIANT                  0             (  0.0%)\n",
        "  VS_SOURCE               20             ( 20.0%)\n",
        "  PS_SOURCE               20             ( 20.0%)\n",
        "  VS_CONST                25             ( 25.0%)\n",
        "  PS_CONST                 0             (  0.0%)\n",
        "  ALPHA_REF                0             (  0.0%)\n",
        "  FOG_COLOR                0             (  0.0%)\n",
        "  BUMP_ENV                 0             (  0.0%)\n",
        "  VS_CONST_I               0             (  0.0%)\n",
        "  VS_DRAW                  0             (  0.0%)\n",
        "  VS_CONST_B               0             (  0.0%)\n",
        "  PS_CONST_I               0             (  0.0%)\n",
        "  PS_CONST_B               0             (  0.0%)\n",
        "inverse-view interval epoch=0 builds=0 bypass=0 hit=0 recompute=0 enabled-hit=n/a saturated=false\n",
        "\n",
        "Per-frame allocator footprint  scratch as small/oversized blocks; op_vec + cmd_vec each split into size + realloc\n",
        "scratch     24.0/0.0  2.0 MB avg    peak 24/0   peak 2.0 MB     per-frame bump arena (VS/PS constants + UP vertices); cleared at begin_frame\n",
        "op_vec                                                          API→encoder: Vec<Op> shipped via sync_channel(1); encoder drains and translates each Op\n",
        "  size      72 KB avg               peak 72 KB                  high-water-reserved Vec<Op> capacity per frame (peak_ops_count)\n",
        "  realloc   32 KB avg               peak 32 KB                  Vec<Op> doubling memcpy on push_op (target ≈ 0 with peak_ops_count reserve)\n",
        "cmd_vec                                                         encoder→unix: Vec<Command> shipped via SubmitCommandBuffer; unix dispatches each Command to a Metal encoder\n",
        "  size      64 KB avg               peak 64 KB                  submitted payload Vec<Command> capacity (excludes idle pool and other payloads)\n",
        "  realloc   192 KB avg              peak 192 KB                 Vec<Command> potential growth copies on emit_command (excludes initial allocations)\n",
        "pagebox     alloc=17  4.8 MB        free=15  4.6 MB             window totals of PageBox allocs/frees reaching either runtime's allocator (fresh pages fault on first touch)\n",
        "  uncached  1 (5.9%)                peak 1/frame                allocs over 1 MiB: past snmalloc's per-thread budget, so commit in / decommit out every time\n",
        "faults      minflt=4200  majflt=3   4200.0 min/frame            process-wide getrusage delta this window (all threads); zero-fill faults on fresh pages land here\n",
        "footprint   3072.0 MB                                           process phys_footprint at the summary, the Metal HUD's app memory (Wine and the game included)\n",
        "unix heap   96.0 MB                                             mtld3d.so's snmalloc committed at the summary (its Rust heap; framework malloc not included)\n",
        "malloc      512.0 MB                                            whole-process malloc in use at the summary (Wine's unix side and the frameworks, Metal's objects included)\n",
        "metal alloc 1536.0 MB                                           MTLDevice currentAllocatedSize at the summary (every Metal allocation of the process)",
    );
    assert_eq!(got, want, "perf summary drifted — diff above");
}

/// Golden snapshot of the `perf-kv v1` line on the grid golden's fixture.
///
/// Pins every key, its order and its value format. Each value is the one
/// the grid golden above shows for the same window, at the kv line's own
/// precision.
#[test]
fn kv_golden_line() {
    let got = render_kv(&sample_window(), &sample_window(), &sample_caches(), 2.01).finish();
    let want = concat!(
        "perf-kv v1 window_s=2.010 frames=1 timed_frames=1",
        " frame_ms=10.000 frame_peak_ms=10.000 api_d3d9_ms=2.800 api_d3d9_peak_ms=2.800",
        " api_outside_ms=3.000 api_outside_peak_ms=3.000 enc_work_ms=1.500",
        " enc_work_peak_ms=1.500 submit_work_ms=0.100 submit_work_peak_ms=0.100",
        " gpu_wait_ms=6.000 gpu_wait_peak_ms=6.000 api_calls_ms=4.000",
        " api_calls_peak_ms=4.000 api_calls_total=263 api_device_ms=0.300",
        " api_device_peak_ms=0.300 api_device_calls_total=123 api_vertex_buffer_ms=0.200",
        " api_vertex_buffer_peak_ms=0.200 api_vertex_buffer_calls_total=88",
        " api_index_buffer_ms=0.080 api_index_buffer_peak_ms=0.080",
        " api_index_buffer_calls_total=32 api_texture_ms=0.000 api_texture_peak_ms=0.000",
        " api_texture_calls_total=0 api_surface_ms=0.040 api_surface_peak_ms=0.040",
        " api_surface_calls_total=20 api_query_ms=0.000 api_query_peak_ms=0.000",
        " api_query_calls_total=0 api_state_block_ms=0.000 api_state_block_peak_ms=0.000",
        " api_state_block_calls_total=0 api_vertex_decl_ms=0.000",
        " api_vertex_decl_peak_ms=0.000 api_vertex_decl_calls_total=0",
        " api_vertex_shader_ms=0.000 api_vertex_shader_peak_ms=0.000",
        " api_vertex_shader_calls_total=0 api_pixel_shader_ms=0.000",
        " api_pixel_shader_peak_ms=0.000 api_pixel_shader_calls_total=0",
        " query_wait_ms=0.000 query_wait_peak_ms=0.000 dev_frame_ms=0.040",
        " dev_frame_peak_ms=0.040 dev_frame_calls_total=3 dev_draws_ms=0.120",
        " dev_draws_peak_ms=0.120 dev_draws_calls_total=100 dev_render_state_ms=0.060",
        " dev_render_state_peak_ms=0.060 dev_render_state_calls_total=30",
        " dev_tex_stage_state_ms=0.030 dev_tex_stage_state_peak_ms=0.030",
        " dev_tex_stage_state_calls_total=18 dev_sampler_state_ms=0.020",
        " dev_sampler_state_peak_ms=0.020 dev_sampler_state_calls_total=12",
        " dev_shader_const_ms=0.020 dev_shader_const_peak_ms=0.020",
        " dev_shader_const_calls_total=8 dev_bind_ms=0.010 dev_bind_peak_ms=0.010",
        " dev_bind_calls_total=6 dev_state_block_ms=0.000 dev_state_block_peak_ms=0.000",
        " dev_state_block_calls_total=0 dev_misc_ms=0.000 dev_misc_peak_ms=0.000",
        " dev_misc_calls_total=0 present_stall_ms=3.200 present_stall_peak_ms=3.200",
        " dev_frame_other_ms=0.000 dev_frame_other_peak_ms=0.000 draw_snapshot_ms=0.090",
        " draw_snapshot_peak_ms=0.090 draw_snapshot_stages_ms=0.020",
        " draw_snapshot_stages_peak_ms=0.020 draw_snapshot_c_ff_ms=0.020",
        " draw_snapshot_c_ff_peak_ms=0.020 draw_snapshot_c_pr_ms=0.010",
        " draw_snapshot_c_pr_peak_ms=0.010 draw_snapshot_keys_ms=0.020",
        " draw_snapshot_keys_peak_ms=0.020 draw_snapshot_bumps_ms=0.010",
        " draw_snapshot_bumps_peak_ms=0.010 draw_snapshot_resid_ms=0.010",
        " draw_snapshot_resid_peak_ms=0.010",
        " draw_snapshot_keys_vdecl_ms=0.000 draw_snapshot_keys_vdecl_peak_ms=0.000",
        " draw_snapshot_keys_rs_ms=0.010 draw_snapshot_keys_rs_peak_ms=0.010",
        " draw_snapshot_keys_rt_ds_ms=0.000 draw_snapshot_keys_rt_ds_peak_ms=0.000",
        " draw_snapshot_keys_variant_ms=0.000 draw_snapshot_keys_variant_peak_ms=0.000",
        " draw_snapshot_keys_vs_source_ms=0.000",
        " draw_snapshot_keys_vs_source_peak_ms=0.000",
        " draw_snapshot_keys_ps_source_ms=0.000",
        " draw_snapshot_keys_ps_source_peak_ms=0.000 draw_snapshot_keys_resid_ms=0.010",
        " draw_snapshot_keys_resid_peak_ms=0.010 draw_snapshot_rebuild_draws_total=40",
        " draw_snapshot_sampled_draws_total=20 draw_snapshot_sampled_rebuild_vdecl_total=0",
        " draw_snapshot_sampled_rebuild_rs_total=5 draw_snapshot_sampled_rebuild_rt_ds_total=0",
        " draw_snapshot_sampled_rebuild_variant_total=0",
        " draw_snapshot_sampled_rebuild_vs_source_total=0",
        " draw_snapshot_sampled_rebuild_ps_source_total=0",
        " draw_snapshot_rebuild_rs_total=10 draw_snapshot_rebuild_stages_total=30",
        " draw_snapshot_rebuild_rt_ds_total=0 draw_snapshot_rebuild_vdecl_total=0",
        " draw_snapshot_rebuild_variant_total=0 draw_snapshot_rebuild_vs_source_total=20",
        " draw_snapshot_rebuild_ps_source_total=20",
        " draw_snapshot_rebuild_vs_const_total=25 draw_snapshot_rebuild_ps_const_total=0",
        " draw_snapshot_rebuild_alpha_ref_total=0 draw_snapshot_rebuild_fog_color_total=0",
        " draw_snapshot_rebuild_bump_env_total=0 draw_snapshot_rebuild_vs_const_i_total=0",
        " draw_snapshot_rebuild_vs_draw_total=0 draw_snapshot_rebuild_vs_const_b_total=0",
        " draw_snapshot_rebuild_ps_const_i_total=0",
        " draw_snapshot_rebuild_ps_const_b_total=0",
        " draw_push_op_ms=0.020",
        " draw_push_op_peak_ms=0.020 bind_texture_ms=0.004 bind_texture_peak_ms=0.004",
        " bind_texture_calls_total=2 bind_buffer_ms=0.002 bind_buffer_peak_ms=0.002",
        " bind_buffer_calls_total=1 bind_shader_ms=0.002 bind_shader_peak_ms=0.002",
        " bind_shader_calls_total=1 bind_rt_ds_ms=0.001 bind_rt_ds_peak_ms=0.001",
        " bind_rt_ds_calls_total=1 bind_ff_fixed_ms=0.001 bind_ff_fixed_peak_ms=0.001",
        " bind_ff_fixed_calls_total=1 bind_view_scissor_ms=0.000",
        " bind_view_scissor_peak_ms=0.000 bind_view_scissor_calls_total=0",
        " surf_lock_rect_ms=0.020 surf_lock_rect_peak_ms=0.020",
        " surf_lock_rect_calls_total=2 surf_unlock_rect_ms=0.010",
        " surf_unlock_rect_peak_ms=0.010 surf_unlock_rect_calls_total=2",
        " surf_get_dc_ms=0.010 surf_get_dc_peak_ms=0.010 surf_get_dc_calls_total=1",
        " surf_release_dc_ms=0.000 surf_release_dc_peak_ms=0.000",
        " surf_release_dc_calls_total=0 surf_misc_ms=0.000 surf_misc_peak_ms=0.000",
        " surf_misc_calls_total=15 enc_ms=1.700 enc_peak_ms=1.700 enc_op_ms=1.400",
        " enc_op_peak_ms=1.400 enc_op_resolve_ms=0.300 enc_op_resolve_peak_ms=0.300",
        " enc_op_pipeline_ms=0.400 enc_op_pipeline_peak_ms=0.400 enc_op_state_ms=0.100",
        " enc_op_state_peak_ms=0.100 enc_op_probe_ms=0.200 enc_op_probe_peak_ms=0.200",
        " enc_op_samplers_ms=0.150 enc_op_samplers_peak_ms=0.150 enc_op_binds_ms=0.200",
        " enc_op_binds_peak_ms=0.200 enc_op_tex_raw_ms=0.010",
        " enc_op_tex_raw_peak_ms=0.010 enc_op_stage_up_ms=0.010",
        " enc_op_stage_up_peak_ms=0.010 enc_op_const_rng_ms=0.010",
        " enc_op_const_rng_peak_ms=0.010 enc_op_resolve_consts_ms=0.150",
        " enc_op_resolve_consts_peak_ms=0.150 enc_op_resolve_skip_ms=0.090",
        " enc_op_resolve_skip_peak_ms=0.090 enc_op_resolve_lookup_ms=0.030",
        " enc_op_resolve_lookup_peak_ms=0.030 enc_op_binds_cbind_ms=0.120",
        " enc_op_binds_cbind_peak_ms=0.120 enc_op_binds_vbib_ms=0.050",
        " enc_op_binds_vbib_peak_ms=0.050 enc_op_binds_draw_ms=0.020",
        " enc_op_binds_draw_peak_ms=0.020 enc_op_resolve_resid_ms=0.030",
        " enc_op_resolve_resid_peak_ms=0.030 enc_op_binds_resid_ms=0.010",
        " enc_op_binds_resid_peak_ms=0.010 enc_op_resid_ms=0.020",
        " enc_op_resid_peak_ms=0.020 enc_finalize_ms=0.100 enc_finalize_peak_ms=0.100",
        " enc_submit_stall_ms=0.200 enc_submit_stall_peak_ms=0.200 submit_ms=6.100",
        " submit_peak_ms=6.100 submit_blits_ms=0.010 submit_blits_peak_ms=0.010",
        " submit_passes_ms=0.050 submit_passes_peak_ms=0.050 submit_commit_ms=0.020",
        " submit_commit_peak_ms=0.020 submit_resid_ms=0.020 submit_resid_peak_ms=0.020",
        " present_wait_ms=6.000 present_wait_peak_ms=6.000 snapshots_total=1",
        " slot_waits_total=0 gpu_ms=4.500 gpu_frame_ms=4.000 gpu_frame_cbs_total=1",
        " gpu_upload_ms=0.300 gpu_upload_cbs_total=1 gpu_present_ms=0.200",
        " gpu_present_cbs_total=1 vb_rename_total=12 ib_rename_total=3",
        " vb_discard_total=10 ib_discard_total=3 vbib_preserve_cpu_total=2",
        " vbib_rename_bytes_total=737280 vbib_in_place_total=3",
        " vbib_staging_uploads_total=4 vbib_reorder_total=3 vbib_full_skip_total=1",
        " vbib_full_skip_bytes_total=16384 vbib_gpu_copy_total=2",
        " vbib_gpu_copy_bytes_total=49152 vbib_alloc_fail_total=1 vbib_destroy_total=1",
        " vbib_ret_cap_drain_total=2 vbib_ret_cap_submit_total=1",
        " vbib_retention_peak_count=6 vbib_retained_bytes=3670016 vbib_pool_hit_total=14",
        " vbib_pool_miss_total=1 pagebox_pool_recycled_total=14",
        " pagebox_pool_recycled_bytes_total=688128 pagebox_pool_parked_bytes=1048576",
        " pe_pagebox_pool_recycled_total=0 pe_pagebox_pool_recycled_bytes_total=0",
        " pe_pagebox_pool_parked_bytes=0",
        " tex_rename_total=2 tex_discard_total=1 tex_pool_hit_total=7 tex_pool_miss_total=1",
        " tex_preserve_cpu_total=1",
        " tex_in_place_total=0 tex_snapshot_total=2 tex_snapshot_bytes_total=6144",
        " tex_snapshot_full_total=1 tex_uploads_total=2 tex_uploads_raw_total=2",
        " tex_uploads_padded_total=0 tex_uploads_pass_total=0 tex_reorder_total=1",
        " tex_destroy_total=1 tex_retention_peak_count=0 tex_staging_retained_bytes=0",
        " tex_staging_wrapped_bytes=4194304 tex_wrapper_create_total=3",
        " tex_wrapper_retire_total=2",
        " tex_dirtyrect_calls_total=4 tex_dirtyrect_partial_total=3",
        " cache_textures_count=48 cache_pipelines_count=12 cache_samplers_count=6",
        " cache_programs_count=8 cache_libs_count=8 cache_depth_states_count=4",
        " metal_capture_layer_count=0",
        " metal_allocated_bytes=1610612736",
        " passes_total=4 commands_total=140 draws_total=100 pipeline_memo_hits_total=97",
        " pipeline_memo_calls_total=100 fan_generated_total=0 draw_unpinned_total=0",
        " up_indexed_total=3 up_oversized_total=2 keys_set_texture_calls_total=0",
        " keys_set_texture_skips_total=0 keys_set_render_state_calls_total=0",
        " keys_set_render_state_skips_total=0 keys_set_tex_stage_state_calls_total=0",
        " keys_set_tex_stage_state_skips_total=0 keys_set_fvf_calls_total=0",
        " keys_set_fvf_skips_total=0 keys_set_vertex_decl_calls_total=0",
        " keys_set_vertex_decl_skips_total=0 keys_set_vertex_shader_calls_total=0",
        " keys_set_vertex_shader_skips_total=0 keys_set_pixel_shader_calls_total=0",
        " keys_set_pixel_shader_skips_total=0 keys_set_vs_const_calls_total=0",
        " keys_set_vs_const_skips_total=0 keys_set_ps_const_calls_total=0",
        " keys_set_ps_const_skips_total=0 inverse_bypass_total=0 inverse_hit_total=0",
        " inverse_recompute_total=0 scratch_small_peak_count=24",
        " scratch_oversized_peak_count=0 scratch_bytes=2097152",
        " op_vec_capacity_bytes=73728 op_vec_realloc_bytes_total=32768",
        " cmd_vec_capacity_bytes=65536 cmd_vec_realloc_bytes_total=196608",
        " pagebox_alloc_total=17 pagebox_alloc_bytes_total=4980736 pagebox_free_total=15",
        " pagebox_free_bytes_total=4849664 pagebox_uncached_total=1",
        " pe_pagebox_alloc_total=0 pe_pagebox_alloc_bytes_total=0 pe_pagebox_free_total=0",
        " pe_pagebox_free_bytes_total=0 pe_pagebox_uncached_total=0",
        " faults_minor_total=4200 faults_major_total=3",
        " process_footprint_bytes=3221225472 unix_heap_bytes=100663296",
        " malloc_in_use_bytes=536870912",
    );
    assert_eq!(got, want, "perf-kv line drifted");
}

/// A window closed before its memory sample prints `n/a` in the five rows and has no keys.
///
/// The grid and the line are otherwise the goldens above, cell for cell: the
/// `n/a` cells keep the comment column, and nothing else moves.
#[test]
fn an_unsampled_window_prints_n_a_and_leaves_the_memory_keys_out() {
    let w = sample_window();
    let unsampled = CacheSizes {
        memory: None,
        ..sample_caches()
    };
    let grid = Summary::render_with_ansi(&w, &unsampled, 2.01, false);
    let mut want = Summary::render_with_ansi(&w, &sample_caches(), 2.01, false);
    for (sampled, missing) in [
        ("\nwrapped     4.0 MB    ", "\nwrapped     n/a       "),
        ("\nfootprint   3072.0 MB ", "\nfootprint   n/a       "),
        ("\nunix heap   96.0 MB   ", "\nunix heap   n/a       "),
        ("\nmalloc      512.0 MB  ", "\nmalloc      n/a       "),
        ("\nmetal alloc 1536.0 MB ", "\nmetal alloc n/a       "),
    ] {
        assert!(want.contains(sampled), "the sampled grid has {sampled:?}");
        want = want.replace(sampled, missing);
    }
    assert_eq!(grid, want, "only the five memory cells change");

    let kv = render_kv(&w, &w, &unsampled, 2.01).finish();
    let mut want = render_kv(&w, &w, &sample_caches(), 2.01).finish();
    for key in [
        " tex_staging_wrapped_bytes=4194304",
        " metal_allocated_bytes=1610612736",
        " process_footprint_bytes=3221225472",
        " unix_heap_bytes=100663296",
        " malloc_in_use_bytes=536870912",
    ] {
        assert!(want.contains(key), "the sampled line has {key:?}");
        want = want.replace(key, "");
    }
    assert_eq!(kv, want, "only the five memory keys are left out");
}

/// A window measured under Metal's capture layer says so in the grid's header and on the line.
///
/// Everything else is the goldens above: one line follows the header, and
/// the flag key reads 1 where it otherwise reads 0.
#[test]
fn a_capture_layer_window_says_so_in_the_header_and_the_kv_line() {
    let w = sample_window();
    let captured = CacheSizes {
        capture_layer: true,
        ..sample_caches()
    };
    let grid = Summary::render_with_ansi(&w, &captured, 2.01, false);
    let plain = Summary::render_with_ansi(&w, &sample_caches(), 2.01, false);
    let note =
        "Metal capture layer loaded (MTL_CAPTURE_ENABLED): submit and GPU rows include its cost\n";
    let header_end = plain.find('\n').expect("the grid has a header line") + 1;
    let mut want = plain.clone();
    want.insert_str(header_end, note);
    assert_eq!(grid, want, "one line after the header, nothing else");
    assert!(
        !plain.contains("capture layer"),
        "a window without it has no such line"
    );

    let kv = render_kv(&w, &w, &captured, 2.01).finish();
    let want = render_kv(&w, &w, &sample_caches(), 2.01).finish().replace(
        " metal_capture_layer_count=0",
        " metal_capture_layer_count=1",
    );
    assert_eq!(kv, want, "only the flag key changes");
}

/// Over two unequal frames, `_ms` averages, `_peak_ms` takes the worst frame and `_total` sums.
///
/// A one-frame window cannot tell the three apart. Gauges take the peak,
/// and the fault keys stay out of a window that measured no faults.
#[test]
fn kv_aggregates_average_peak_and_total_over_frames() {
    let mut light = sample(0, 0);
    light.timing.frame_total_cycles = 4_000_000;
    light.counters.vb_rename = 1;
    light.counters.api_call_counts_by_category[ApiCategory::Device as usize] = 10;
    light.draws = 100;
    light.vbib_retained_bytes = 1_000;
    let mut heavy = sample(0, 0);
    heavy.timing.frame_total_cycles = 8_000_000;
    heavy.counters.api_call_counts_by_category[ApiCategory::Device as usize] = 30;
    heavy.draws = 300;
    heavy.vbib_retained_bytes = 3_000;
    let mut w = PerfWindow::new();
    w.accumulate(&light);
    w.accumulate(&heavy);
    let line = render_kv(&w, &w, &sample_caches(), 2.0).finish();
    let expected = [
        "perf-kv v1 window_s=2.000 frames=2 ".to_owned(),
        format!(" frame_ms={:.3} ", cycles_to_ms(12_000_000) / 2.0),
        format!(" frame_peak_ms={:.3} ", cycles_to_ms(8_000_000)),
        " api_device_calls_total=40 ".to_owned(),
        " vb_rename_total=1 ".to_owned(),
        " draws_total=400 ".to_owned(),
        " vbib_retained_bytes=3000 ".to_owned(),
    ];
    for pair in &expected {
        assert!(line.contains(pair.as_str()), "missing {pair:?} in {line}");
    }
    assert!(!line.contains("faults_"), "no fault sample, no fault keys");
}

/// Every key is a unit-suffixed `[a-z0-9_]+` name that appears once.
///
/// Runs on the full line a window logs, compilation keys included, so a
/// key added anywhere is held to the naming contract `docs/ARCHITECTURE.md`
/// states: the suffix names the unit, and the value format follows it
/// (three decimals for `_ms`, `_avg_ms` and `_peak_ms`, a bare integer
/// otherwise).
#[test]
fn kv_keys_are_unique_and_unit_suffixed() {
    const FLOAT_SUFFIXES: [&str; 1] = ["_ms"];
    const INTEGER_SUFFIXES: [&str; 3] = ["_total", "_bytes", "_count"];
    let mut kv = render_kv(&sample_window(), &sample_window(), &sample_caches(), 2.01);
    compilation::CompilationPerf::new().append_kv(&mut kv);
    let line = kv.finish();
    assert!(
        !line.contains('\x1b') && !line.contains('\n'),
        "one plain line: {line:?}"
    );
    let mut fields = line.split(' ');
    assert_eq!(
        fields.by_ref().take(5).collect::<Vec<_>>(),
        [
            "perf-kv",
            "v1",
            "window_s=2.010",
            "frames=1",
            "timed_frames=1"
        ]
    );
    let mut seen = FxHashSet::default();
    for field in fields {
        let (key, value) = field
            .split_once('=')
            .unwrap_or_else(|| panic!("{field:?} is not key=value"));
        assert!(
            !key.is_empty()
                && key
                    .bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_'),
            "{key:?} is not [a-z0-9_]+"
        );
        let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
        if FLOAT_SUFFIXES.iter().any(|suffix| key.ends_with(suffix)) {
            let (whole, fraction) = value
                .split_once('.')
                .unwrap_or_else(|| panic!("{key}={value} lacks three decimals"));
            assert!(
                digits(whole) && digits(fraction) && fraction.len() == 3,
                "{key}={value} is not a plain three-decimal number"
            );
        } else if INTEGER_SUFFIXES.iter().any(|suffix| key.ends_with(suffix)) {
            assert!(digits(value), "{key}={value} is not a bare integer");
        } else {
            panic!("{key} carries no unit suffix");
        }
        assert!(seen.insert(key), "{key} repeats");
    }
}

/// Sanity snapshot of the rendered summary.
///
/// The summary contains the section headers and the bottleneck label for
/// a GPU-bound frame. Keeps the layout from silently losing a section in
/// future refactors without locking the entire multi-line string.
#[test]
fn summary_contains_expected_sections() {
    let w = sample_window();
    let caches = sample_caches();
    let out = Summary::render_with_ansi(&w, &caches, 2.01, false);
    for expected in [
        "── perf  window=2.01s",
        "bottleneck=ENCODER (GPU)",
        "API thread",
        "├─ D3D9 calls",
        "├─ Send stall",
        "└─ Outside d3d9",
        "Encoder thread",
        "├─ Closures (op)",
        "├─ Finalize",
        "└─ Submit stall",
        "Submit thread",
        "├─ Encode+commit",
        "│  ├─ leading blits",
        "│  ├─ passes",
        "│  ├─ commit",
        "│  └─ resid",
        "└─ Present wait",
        "Present thread",
        "├─ Drawable wait",
        "├─ Snapshots",
        "└─ Slot waits",
        "GPU (CBs may overlap)",
        "├─ Frame CBs",
        "├─ Upload CBs",
        "└─ Present CBs",
        "Frame total",
        "submit_status=0x0",
        "Resources (VB/IB)",
        "Resources (textures)",
        "Caches",
        "Commands / passes",
        "Keys gating",
    ] {
        assert!(
            out.contains(expected),
            "summary missing {expected:?}:\n{out}"
        );
    }
}

fn sample_window() -> PerfWindow {
    // Construct a window populated with a single synthetic frame
    // whose shape classifies as ENCODER (GPU): drawable_wait
    // dominates, present_block is large.
    let mut w = PerfWindow::new();
    let mut cats = [0u64; ApiCategory::COUNT];
    let mut calls = [0u32; ApiCategory::COUNT];
    cats[ApiCategory::Device as usize] = 300_000;
    cats[ApiCategory::VertexBuffer as usize] = 200_000;
    cats[ApiCategory::IndexBuffer as usize] = 80_000;
    cats[ApiCategory::Surface as usize] = 40_000;
    calls[ApiCategory::Device as usize] = 123;
    calls[ApiCategory::VertexBuffer as usize] = 88;
    calls[ApiCategory::IndexBuffer as usize] = 32;
    calls[ApiCategory::Surface as usize] = 20;
    // Decompose Device into sub-buckets; must sum to cats[Device]
    // = 300_000 to mirror the production invariant. Calls sum to
    // calls[Device] = 123.
    let mut dsub = [0u64; DeviceSubCategory::COUNT];
    let mut dcalls = [0u32; DeviceSubCategory::COUNT];
    // Values picked so the milliseconds round cleanly at `{:.2}`:
    // multiples of 10_000 cycles only. Sum = 300_000 to match
    // `cats[Device]`.
    dsub[DeviceSubCategory::Frame as usize] = 40_000;
    dsub[DeviceSubCategory::Draws as usize] = 120_000;
    dsub[DeviceSubCategory::RenderState as usize] = 60_000;
    dsub[DeviceSubCategory::TexStageState as usize] = 30_000;
    dsub[DeviceSubCategory::SamplerState as usize] = 20_000;
    dsub[DeviceSubCategory::ShaderConst as usize] = 20_000;
    dsub[DeviceSubCategory::Bind as usize] = 10_000;
    dsub[DeviceSubCategory::StateBlock as usize] = 0;
    dsub[DeviceSubCategory::Misc as usize] = 0;
    dcalls[DeviceSubCategory::Frame as usize] = 3;
    dcalls[DeviceSubCategory::Draws as usize] = 100;
    dcalls[DeviceSubCategory::RenderState as usize] = 30;
    dcalls[DeviceSubCategory::TexStageState as usize] = 18;
    dcalls[DeviceSubCategory::SamplerState as usize] = 12;
    dcalls[DeviceSubCategory::ShaderConst as usize] = 8;
    dcalls[DeviceSubCategory::Bind as usize] = 6;
    dcalls[DeviceSubCategory::StateBlock as usize] = 0;
    dcalls[DeviceSubCategory::Misc as usize] = 0;
    // Decompose the Bind device-sub (10_000 cyc, 6 calls) into
    // BindSubCategory rows. Sums must match the parent exactly —
    // every BindSubCategory site uses `bind_timer`, no escape.
    // Values are multiples of 1_000 cycles to round cleanly at
    // `{:.2}`.
    let mut bsub = [0u64; BindSubCategory::COUNT];
    let mut bcalls = [0u32; BindSubCategory::COUNT];
    bsub[BindSubCategory::Texture as usize] = 4_000;
    bsub[BindSubCategory::Buffer as usize] = 2_000;
    bsub[BindSubCategory::Shader as usize] = 2_000;
    bsub[BindSubCategory::RtDs as usize] = 1_000;
    bsub[BindSubCategory::FfFixed as usize] = 1_000;
    bsub[BindSubCategory::ViewScissor as usize] = 0;
    bcalls[BindSubCategory::Texture as usize] = 2;
    bcalls[BindSubCategory::Buffer as usize] = 1;
    bcalls[BindSubCategory::Shader as usize] = 1;
    bcalls[BindSubCategory::RtDs as usize] = 1;
    bcalls[BindSubCategory::FfFixed as usize] = 1;
    bcalls[BindSubCategory::ViewScissor as usize] = 0;
    // Decompose the Surface category (40_000 cyc, 20 calls) into
    // SurfaceSubCategory rows. Sums must match `cats[Surface]` exactly —
    // every surface thunk tags a variant via `surf_timer`, with `Misc`
    // as the catch-all. Multiples of 10_000 cycles round cleanly at
    // `{:.2}`.
    let mut ssub = [0u64; SurfaceSubCategory::COUNT];
    let mut scalls = [0u32; SurfaceSubCategory::COUNT];
    ssub[SurfaceSubCategory::LockRect as usize] = 20_000;
    ssub[SurfaceSubCategory::UnlockRect as usize] = 10_000;
    ssub[SurfaceSubCategory::GetDc as usize] = 10_000;
    ssub[SurfaceSubCategory::ReleaseDc as usize] = 0;
    ssub[SurfaceSubCategory::Misc as usize] = 0;
    scalls[SurfaceSubCategory::LockRect as usize] = 2;
    scalls[SurfaceSubCategory::UnlockRect as usize] = 2;
    scalls[SurfaceSubCategory::GetDc as usize] = 1;
    scalls[SurfaceSubCategory::ReleaseDc as usize] = 0;
    scalls[SurfaceSubCategory::Misc as usize] = 15;
    // 40 of the 100 draw calls rebuilt a section and 20 of those were timed.
    // Of the six sections the `keys` bucket times only RS carries cycles:
    // 5 000 sampled, scaled to 10 000, so `keys` 20 000 splits into RS
    // 10 000 (1000 ns per each of its 10 rebuilds) and a `rest` of 10 000.
    let mut rebuilds = [0u32; SnapshotSection::SLOTS];
    rebuilds[SnapshotSection::Rs as usize] = 10;
    rebuilds[SnapshotSection::Stages as usize] = 30;
    rebuilds[SnapshotSection::VsSource as usize] = 20;
    rebuilds[SnapshotSection::PsSource as usize] = 20;
    rebuilds[SnapshotSection::VsConst as usize] = 25;
    let mut section_cycles = [0u64; SnapshotSection::SLOTS];
    section_cycles[SnapshotSection::Rs as usize] = 5_000;
    let mut sampled_rebuilds = [0u32; SnapshotSection::SLOTS];
    sampled_rebuilds[SnapshotSection::Rs as usize] = 5;
    let s = FrameSample {
        counters: FrameCounters {
            timed: 1,
            reset_epoch: 0,
            reset_epoch_saturated: 0,
            inverse_view: [0; 3],
            inverse_view_saturated: 0,
            api_cycles_by_category: cats,
            api_call_counts_by_category: calls,
            vb_rename: 12,
            ib_rename: 3,
            // 15 renames of a 48 KiB-average buffer: 720 KB of fresh
            // PageBox pages this frame. Exercises the `  bytes` row.
            vbib_rename_bytes: 737_280,
            // Pool fixture: 14 of 15 rename allocs served warm
            // (93.3%), one fell through to the allocator.
            vbib_pool_hits: 14,
            vbib_pool_misses: 1,
            vb_discards: 10,
            ib_discards: 3,
            // Two whole-buffer non-WRITEONLY contended Locks took the
            // CPU-memcpy preserve path. Surfaces in the `preserve` row.
            // `rename = discards + preserve_cpu` holds (12 = 10 + 2 for VB).
            vbib_preserve_cpu: 2,
            // Three contended partial Locks took the kept-divergence
            // in-place arm. Outside the `rename = discards +
            // preserve_cpu` partition: no rename was allocated.
            vbib_write_in_place_contended: 3,
            // Two cheap-tier recoveries (drain) and one heavy
            // (submit) — exercises the row formatting on both halves.
            retention_cap_drain: 2,
            retention_cap_submit: 1,
            // 2 renames: 1 was DISCARD/WRITEONLY (no preserve),
            // 1 needed CPU memcpy preserve. Invariant
            // `rename = discards + preserve_cpu` holds.
            texture_renames: 2,
            texture_discards: 1,
            texture_preserve_cpu: 1,
            texture_write_in_place_contended: 0,
            // Staging pool fixture: 7 of 8 staging allocations served
            // warm (87.5%), one fell through to the allocator.
            texture_pool_hits: 7,
            texture_pool_misses: 1,
            // AddDirtyRect probe fixture: 4 calls, 3 with a usable
            // sub-region; area sum 10000 bp ⇒ mean coverage 25% of the mip.
            texture_add_dirty_calls: 4,
            texture_add_dirty_partial: 3,
            texture_add_dirty_area_bp: 10_000,
            query_wait_cycles: 0,
            device_sub_cycles: dsub,
            device_sub_calls: dcalls,
            bind_sub_cycles: bsub,
            bind_sub_calls: bcalls,
            surface_sub_cycles: ssub,
            surface_sub_calls: scalls,
            keys_gate_calls: [0; KeysGate::COUNT],
            keys_gate_skips: [0; KeysGate::COUNT],
            snapshot_rebuilds: rebuilds,
            snapshot_sampled_rebuilds: sampled_rebuilds,
            snapshot_rebuild_draws: 40,
            // Half the rebuilding draws were timed, so the sampled section
            // cycles below count twice.
            snapshot_sampled_draws: 20,
            // snapshot dominates the Draws bucket; split inside it
            // is stages 20 + c_ff 20 + c_pr 10 + keys 20 + bumps 10
            // + leftover 10 = 90. push_op trails. Every component is
            // a multiple of 10_000 so its milliseconds stay off the
            // {:.2} rounding boundary that values like 5_000 or 25_000
            // fall on.
            draw_snapshot_cycles: 90_000,
            draw_snapshot_stages_cycles: 20_000,
            draw_snapshot_c_ff_cycles: 20_000,
            draw_snapshot_c_pr_cycles: 10_000,
            draw_snapshot_keys_cycles: 20_000,
            draw_snapshot_bumps_cycles: 10_000,
            draw_push_op_cycles: 20_000,
            draw_snapshot_section_cycles: section_cycles,
            // The sampled draws spent 10 000 in `keys`, 5 000 of them in RS,
            // so `rest` scales to 10 000.
            draw_snapshot_keys_sampled_cycles: 10_000,
            // `d3d9.dll`'s traffic, which the window keeps apart under the
            // `pe_pagebox_*` keys and the grid adds to the encoder's own.
            pagebox_alloc_bytes: 0,
            pagebox_free_bytes: 0,
            pool_recycled_bytes: 0,
            pool_parked_bytes: 0,
            pagebox_allocs: 0,
            pagebox_frees: 0,
            pagebox_uncached_allocs: 0,
            pool_recycled: 0,
            // Two partial uploads snapshotted 6 KiB between them; one was declined.
            texture_snapshot_bytes: 6144,
            texture_snapshot_uploads: 2,
            texture_snapshot_full: 1,
        },
        timing: FrameTiming {
            present_block_cycles: 3_200_000,
            frame_total_cycles: 10_000_000,
            // peak_ops_count × size_of::<Op>() — peak ~1000 ops at
            // ~72 B/Op rounds to ~72 KB. Pick 72 KB exactly so the
            // golden assertion pins the row and `format_kb_pair`
            // renders cleanly.
            op_vec_capacity_bytes: 72 * 1024,
            // Steady-state target is 0; pick a small non-zero value
            // so the realloc row's number column is exercised by the
            // golden assertion.
            op_vec_realloc_bytes: 32 * 1024,
        },
        enc: EncoderFrameCounters {
            buffer_destroys: 1,
            texture_destroys: 1,
            // The drain parked 14 retired boxes (672 KB) in the pool
            // this frame, matching the 14 hits above.
            pagebox_pool_recycled: 14,
            pagebox_pool_recycled_bytes: 688_128,
            vbib_staging_uploads: 4,
            vbib_mid_pass_reorders: 3,
            vbib_full_upload_skips: 1,
            vbib_full_upload_skip_bytes: 16_384,
            vbib_preserve_gpu_bytes: 49_152,
            vbib_reorder_alloc_failures: 1,
            texture_blit_uploads: 2,
            texture_blit_padded_uploads: 0,
            texture_expand_uploads: 0,
            texture_gpu_renames: 1,
            staging_wrapper_creates: 3,
            staging_wrapper_retires: 2,
            op_cycles: 1_400_000,
            // Decompose op_cyc 1.40M into the nine phases (six draw phases sum
            // 1.35M + three non-draw phases sum 0.03M = 1.38M) so the golden
            // pins each "Closures (op)" sub-row; resid = 0.02M. Multiples of
            // 10_000 cyc round cleanly at {:.2}.
            op_sub_cycles: [
                300_000, 400_000, 100_000, 200_000, 150_000, 200_000, 10_000, 10_000, 10_000,
            ],
            // resolve(300k) split 150/90/30 → resid 30k; binds(200k) split
            // 120/50/20 → resid 10k. Multiples of 10k round cleanly at {:.2}.
            op_sub_detail: [150_000, 90_000, 30_000, 120_000, 50_000, 20_000],
            // 97 of 100 pipeline resolves served from the memo → 97.0%.
            pipeline_memo_hits: 97,
            fan_generated: 0,
            draw_unpinned: 0,
            // Three indexed UP draws and two oversized inline vertex streams
            // went through the unix upload ring.
            up_indexed: 3,
            up_vertex_oversized: 2,
            pipeline_memo_calls: 100,
            submit_cycles: 300_000,
            // Presenter: the last present waited 6.0M for its drawable.
            drawable_wait_cycles: 6_000_000,
            // Submit thread: total execute 6.1M = encode+commit 0.1M +
            // present wait 6.0M.
            submit_exec_cycles: 6_100_000,
            // 0.2M backpressure stall → Finalize 0.10M, stall 0.20M.
            submit_stall_cycles: 200_000,
            present_wait_cycles: 6_000_000,
            // Encode+commit 0.1M = blits 0.01M + passes 0.05M + commit
            // 0.02M + resid 0.02M.
            submit_blits_cycles: 10_000,
            submit_passes_cycles: 50_000,
            submit_commit_cycles: 20_000,
            // One buffer of each role finished: frame 4.0M, upload 0.3M,
            // present 0.2M, 4.5M in all.
            gpu_cycles: [4_000_000, 300_000, 200_000],
            gpu_buffers: [1, 1, 1],
            // One read-back in the window presented its frame from a copy,
            // and no copy waited for a slot.
            snapshots: 1,
            slot_waits: 0,
        },
        passes: 4,
        commands: 140,
        draws: 100,
        scratch_small_blocks: 24,
        scratch_oversized_blocks: 0,
        scratch_bytes: 2 * 1024 * 1024,
        // ~140 commands at 32 B each ≈ 4.4 KB live; round to one
        // 64 KB pool entry to keep the resident-footprint cell
        // non-zero in the golden assertion.
        cmd_vec_capacity_bytes: 64 * 1024,
        // Retention in MB range exercises the format_kb_pair MB
        // branch and the longest peak cell the Resources grid
        // ever renders — the golden assertion below pins the
        // column gap so future edits don't re-regress the butting
        // bug where "peak N.N MBlive queue..." ran together.
        vbib_retention_depth: 6,
        vbib_retained_bytes: 3_670_016,
        pending_blit_retention_depth: 0,
        tex_staging_retained_bytes: 0,
        cmd_vec_realloc_bytes: 196_608,
        // Allocator-visible PageBox traffic: the 15 renames (720 KB)
        // plus one 48 KB creation on the alloc side; 14 retired boxes
        // freed. Exercises both cells of the `pagebox` footprint row.
        //
        // Plus one 4 MiB texture mip-staging box allocated and freed
        // this frame. Nothing pools that producer, and 4 MiB is past
        // snmalloc's per-thread budget, so it is the single
        // `uncached` hit the child row reports.
        pagebox_allocs: 17,
        pagebox_alloc_bytes: 4_980_736,
        pagebox_frees: 15,
        pagebox_free_bytes: 4_849_664,
        pagebox_uncached_allocs: 1,
        // 1 MB parked in the recycle pool at frame end; exercises the
        // MB branch of the `  parked` row.
        pagebox_pool_bytes: 1_048_576,
        // Encoder thread = op (Closures) + finalize. The unix
        // command-walk + present moved to the submit thread.
        outside_d3d9: 3_000_000,
        api_cyc: 4_000_000,
        api_work: 2_800_000,
        enc_cyc: 1_700_000,
        submit_status: 0,
    };
    w.accumulate(&s);
    // Emit-time fields (not part of accumulate): the once-per-window
    // fault sample delta, as `log_frame_summary` would set it.
    w.faults_window = Some(TaskFaults {
        minor: 4200,
        major: 3,
    });
    w
}

const fn sample_caches() -> CacheSizes {
    CacheSizes {
        textures: 48,
        pipelines: 12,
        samplers: 6,
        programs: 8,
        libs: 8,
        depth_states: 4,
        scratch_small_blocks: 24,
        scratch_oversized_blocks: 0,
        scratch_bytes: 2 * 1024 * 1024,
        cmd_vec_capacity_bytes: 64 * 1024,
        pending_blit_retention_depth: 0,
        pending_resource_retention_depth: 1,
        pagebox_pool_bytes: 1_048_576,
        memory: Some(MemoryGauges {
            process_footprint: 3 << 30,
            metal_allocated: 3 << 29,
            unix_heap: 96 << 20,
            malloc_in_use: 512 << 20,
            staging_wrapped: 4 << 20,
        }),
        capture_layer: false,
    }
}

/// Strip ANSI CSI sequences (ESC `[` … final-byte) from a string.
///
/// Implemented locally so the test suite doesn't grow a
/// `strip-ansi-escapes` dep.
fn strip_ansi(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == 0x1b && i + 1 < bytes.len() && bytes[i + 1] == b'[' {
            i += 2;
            while i < bytes.len() {
                let c = bytes[i];
                i += 1;
                if (0x40..=0x7e).contains(&c) {
                    break;
                }
            }
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8(out).expect("ANSI-strip produced invalid UTF-8")
}

// Exclusive-time accounting: a nested timer's interval must land in
// its own bucket only, never double-counted into the delegating
// parent's bucket too (the Surface→Texture LockRect case). The
// invariant is `Σ self_time == outermost elapsed`.
#[test]
fn exclusive_exit_single_timer_books_full_elapsed() {
    // No children, outermost (depth 0 after): self == elapsed,
    // accumulator resets to 0.
    let (self_time, restored) = exclusive_exit(200, 0, 0, 0);
    assert_eq!(self_time, 200);
    assert_eq!(restored, 0);
}

#[test]
fn exclusive_exit_nested_pair_no_double_count() {
    // Outer (Surface) elapsed 200 wraps inner (Texture) elapsed 50.
    // Inner drops first: depth back to 1 (parent still live), no
    // children of its own → self 50, hands its 50 up to the parent
    // accumulator (saved 0 + 50).
    let (inner_self, inner_restored) = exclusive_exit(50, 0, 0, 1);
    assert_eq!(inner_self, 50);
    assert_eq!(inner_restored, 50);
    // Outer drops: its children == the 50 the inner handed up;
    // self 150; outermost → reset to 0.
    let (outer_self, outer_restored) = exclusive_exit(200, inner_restored, 0, 0);
    assert_eq!(outer_self, 150);
    assert_eq!(outer_restored, 0);
    // No double-count: the two self-times partition the outer span.
    assert_eq!(inner_self + outer_self, 200);
}

#[test]
fn exclusive_exit_two_siblings_under_parent() {
    // Parent (elapsed 100) contains two sequential children (30, 40).
    // c1 drops, hands 30 up (parent acc 0+30).
    let (c1_self, acc) = exclusive_exit(30, 0, 0, 1);
    // c2 starts with the parent acc saved (30), drops, hands 40 up.
    let (c2_self, acc) = exclusive_exit(40, 0, acc, 1);
    // Parent drops: children == 70; self == 30; outermost resets.
    let (p_self, p_restored) = exclusive_exit(100, acc, 0, 0);
    assert_eq!((c1_self, c2_self, p_self), (30, 40, 30));
    assert_eq!(p_restored, 0);
    assert_eq!(c1_self + c2_self + p_self, 100);
}

#[test]
fn exclusive_exit_saturates_when_children_exceed_elapsed() {
    // TSC noise can make a nested delta momentarily exceed the
    // parent's measured span; self_time clamps to 0, never wraps.
    let (self_time, _) = exclusive_exit(10, 25, 0, 1);
    assert_eq!(self_time, 0);
}

/// A final Release may destroy its device inside a child Release's timer.
#[test]
fn api_timer_storage_outlives_nested_device_release() {
    let mut storage = ApiPerfStorage::new();
    let weak = Rc::downgrade(&storage.state);
    let outer = ApiTimer::new(Some(&storage), ApiCategory::Texture, true);
    let inner = ApiTimer::new(Some(&storage), ApiCategory::Device, true);
    storage.state_mut().bump_texture_rename();
    assert_eq!(storage.state.borrow().timer_depth, 2);
    drop(storage);
    assert_eq!(weak.strong_count(), 2);
    drop(inner);
    {
        let retained = weak.upgrade().expect("outer timer owns the state");
        let state = retained.borrow();
        assert_eq!(state.timer_depth, 1);
        assert_eq!(state.counters.texture_renames, 1);
        assert_eq!(
            state.counters.api_call_counts_by_category[ApiCategory::Device as usize],
            1
        );
    }
    drop(outer);
    assert!(weak.upgrade().is_none(), "last timer frees counter storage");
}

#[test]
fn api_timer_storage_outlives_direct_device_release() {
    let storage = ApiPerfStorage::new();
    let weak = Rc::downgrade(&storage.state);
    let timer = ApiTimer::new(Some(&storage), ApiCategory::Device, true);
    drop(storage);
    assert_eq!(weak.strong_count(), 1);
    drop(timer);
    assert!(weak.upgrade().is_none());
}

#[test]
fn api_timer_nested_writeback_balances_depth_and_counts() {
    let storage = ApiPerfStorage::new();
    let outer = ApiTimer::new(Some(&storage), ApiCategory::Surface, true);
    let inner = ApiTimer::new(Some(&storage), ApiCategory::Texture, true);
    drop(inner);
    drop(outer);
    let state = storage.state.borrow();
    assert_eq!(state.timer_depth, 0);
    assert_eq!(state.active_child_cycles, 0);
    assert_eq!(
        state.counters.api_call_counts_by_category[ApiCategory::Surface as usize],
        1
    );
    assert_eq!(
        state.counters.api_call_counts_by_category[ApiCategory::Texture as usize],
        1
    );
}

#[test]
fn api_timer_disabled_does_not_retain_storage() {
    // Unit tests install no logger, so the runtime gate remains disabled.
    assert!(!perf_enabled());
    let storage = ApiPerfStorage::new();
    let weak = Rc::downgrade(&storage.state);
    let timer = ApiTimer::start_device(Some(&storage), DeviceSubCategory::Misc);
    assert_eq!(storage.state.borrow().timer_depth, 0);
    assert_eq!(weak.strong_count(), 1);
    drop(storage);
    assert!(weak.upgrade().is_none());
    drop(timer);
}

#[test]
fn api_perf_storage_without_timers_is_reclaimed() {
    let storage = ApiPerfStorage::new();
    let weak = Rc::downgrade(&storage.state);
    drop(storage);
    assert!(weak.upgrade().is_none());
}

#[test]
fn inverse_counts_partition_builds_and_preserve_reset_intervals() {
    use mtld3d_types::{D3DMATRIX, D3DRS_CLIPPLANEENABLE, render_state_defaults};

    use crate::vs_draw::{InverseViewUse, VsDrawState};
    let mut api = ApiPerfState::new();
    let mut cache = VsDrawState::new();
    let mut rs = render_state_defaults();
    let _ = cache.build_bytes(&rs, 1.0f32.to_bits(), &D3DMATRIX::IDENTITY, &[]);
    api.record_inverse_view(cache.last_use());
    rs[D3DRS_CLIPPLANEENABLE as usize] = 1;
    for _ in 0..2 {
        let _ = cache.build_bytes(&rs, 1.0f32.to_bits(), &D3DMATRIX::IDENTITY, &[]);
        api.record_inverse_view(cache.last_use());
    }
    let mut payload = FramePerfPayload::new();
    api.drain_into_payload(&mut payload, true);
    assert_eq!(payload.counters.inverse_view, [1, 1, 1]);
    assert_eq!(api.counters.inverse_view, [0; 3]);
    let mut old = sample(0, 0);
    old.counters = payload.counters;
    let mut w = PerfWindow::new();
    w.accumulate(&old);
    // Reset flushes old counters before replacing the cache and advancing epoch.
    cache = VsDrawState::new();
    api.advance_reset_epoch();
    let _ = cache.build_bytes(&rs, 1.0f32.to_bits(), &D3DMATRIX::IDENTITY, &[]);
    assert!(matches!(cache.last_use(), InverseViewUse::Recompute));
    api.record_inverse_view(cache.last_use());
    api.drain_into_payload(&mut payload, true);
    let mut new = sample(0, 0);
    new.counters = payload.counters;
    w.accumulate(&new);
    assert_eq!(w.inverse_epochs.len(), 2);
    assert_eq!(w.inverse_epochs[0].counts, [1, 1, 1]);
    assert_eq!(w.inverse_epochs[1].counts, [0, 0, 1]);
    let out = Summary::render_with_ansi(&w, &sample_caches(), 5.0, false);
    assert!(
        out.contains(
            "reset epochs=0..1; inverse/upload counts are interval totals (not cumulative)"
        )
    );
    assert!(out.contains("epoch=0 builds=3 bypass=1 hit=1 recompute=1 enabled-hit=50.0%"));
    assert!(out.contains("epoch=1 builds=1 bypass=0 hit=0 recompute=1 enabled-hit=0.0%"));
    let mut other = ApiPerfState::new();
    other.drain_into_payload(&mut payload, true);
    assert_eq!(payload.counters.reset_epoch, 0);
    assert_eq!(payload.counters.inverse_view, [0; 3]);
    let mut other_window = PerfWindow::new();
    other_window.accumulate(&sample(0, 0));
    assert_eq!(other_window.inverse_epochs[0].counts, [0; 3]);
    w.reset();
    assert!(w.inverse_epochs.is_empty());
}

#[test]
fn inverse_rates_are_undefined_for_empty_or_saturated_counts() {
    use crate::vs_draw::InverseViewUse;
    let mut api = ApiPerfState::new();
    api.record_inverse_view(&InverseViewUse::Bypass);
    let mut payload = FramePerfPayload::new();
    api.drain_into_payload(&mut payload, true);
    let mut s = sample(0, 0);
    s.counters = payload.counters;
    let mut w = PerfWindow::new();
    w.accumulate(&s);
    let out = Summary::render_with_ansi(&w, &sample_caches(), 5.0, false);
    assert!(out.contains("enabled-hit=n/a saturated=false"));
    api.counters.inverse_view[1] = u64::MAX;
    api.record_inverse_view(&InverseViewUse::Hit);
    assert!(api.counters.inverse_view_saturated != 0);
    assert_eq!(api.counters.inverse_view[1], u64::MAX);
    api.drain_into_payload(&mut payload, true);
    s.counters = payload.counters;
    w.accumulate(&s);
    let out = Summary::render_with_ansi(&w, &sample_caches(), 5.0, false);
    assert!(out.contains("enabled-hit=n/a saturated=true"));
    // Aggregation overflow is also visible even if each frame fit individually.
    w.reset();
    s.counters.inverse_view_saturated = 0;
    w.accumulate(&s);
    w.accumulate(&s);
    assert!(w.inverse_epochs[0].saturated);
    api.reset_epoch = u64::MAX;
    api.advance_reset_epoch();
    api.drain_into_payload(&mut payload, true);
    assert_eq!(payload.counters.reset_epoch, u64::MAX);
    assert!(payload.counters.reset_epoch_saturated != 0);
}

#[test]
fn dispatch_trace_excludes_logger_and_separates_shader_validation_from_compile() {
    use mtld3d_shared::Thunks;

    assert_eq!(unix_dispatch_kind(Thunks::WriteLog as u32), None);
    assert_eq!(
        unix_dispatch_kind(Thunks::SubmitEncoderFrame as u32),
        Some("enqueue")
    );
    // Shader validation and resource creation remain necessary synchronous calls.
    for thunk in [
        Thunks::CreateShaderProgram,
        Thunks::CreateEncoder,
        Thunks::EncoderControl,
        Thunks::CreateBackbuffer,
        Thunks::DestroyEncoder,
    ] {
        assert_eq!(unix_dispatch_kind(thunk as u32), Some("sync"));
    }
}

#[test]
fn native_decode_cycles_extend_operations_once_and_reset_next_frame() {
    let mut state = EncoderPerfState::new();
    let payload = FramePerfPayload::new();
    state.begin_frame(&payload);
    state.set_op_cycles(100);
    state.set_submit_cycles(25);
    state.add_op_cycles(40);
    assert_eq!(state.enc.op_cycles, 140);
    assert_eq!(state.enc.op_cycles + state.enc.submit_cycles, 165);
    assert_eq!(
        state.timing.frame_total_cycles,
        payload.timing.frame_total_cycles
    );
    state.begin_frame(&payload);
    assert_eq!(state.enc.op_cycles, 0);
    assert_eq!(state.enc.submit_cycles, 0);
}

/// Every rebuilding draw counts once per dirty section, and about one per sample period is timed.
#[test]
fn snapshot_rebuilds_count_each_section_and_sample_one_in_the_period() {
    let mut state = ApiPerfState::new();
    let dirty = SnapshotSection::Rs.bit() | SnapshotSection::PsConstB.bit() | (1 << 31);
    let draws = 16_000u32;
    let mut sampled = 0u32;
    for _ in 0..draws {
        sampled += u32::from(state.record_snapshot_rebuild(dirty));
    }
    let c = &state.counters;
    assert_eq!(c.snapshot_rebuild_draws, draws);
    assert_eq!(c.snapshot_rebuilds[SnapshotSection::Rs as usize], draws);
    assert_eq!(
        c.snapshot_rebuilds[SnapshotSection::PsConstB as usize],
        draws
    );
    assert_eq!(c.snapshot_rebuilds[SnapshotSection::Stages as usize], 0);
    assert_eq!(c.snapshot_sampled_draws, sampled);
    let expected = draws / KEYS_SECTION_SAMPLE_PERIOD;
    assert!(
        sampled.abs_diff(expected) < expected / 10,
        "sampled {sampled} of {draws}, expected about {expected}"
    );
}

/// Sampled cycles scale by all rebuilding draws over the sampled ones, and to zero with none.
#[test]
fn sampled_cycles_scale_to_every_rebuilding_draw() {
    assert_eq!(scale_sampled(5_000, 40, 20), 10_000);
    assert_eq!(scale_sampled(5_000, 40, 0), 0);
    assert_eq!(scale_sampled(u64::MAX, u64::MAX, 1), u64::MAX);
}

/// One frame's section counters for the sampled-section tests.
const fn sampled_frame(
    rebuild_draws: u32,
    sampled_draws: u32,
    rs: (u32, u64),
    keys: (u64, u64),
) -> FrameSample {
    let mut frame = sample(0, 0);
    let c = &mut frame.counters;
    c.snapshot_rebuild_draws = rebuild_draws;
    c.snapshot_sampled_draws = sampled_draws;
    c.snapshot_rebuilds[SnapshotSection::Rs as usize] = rebuild_draws;
    c.snapshot_sampled_rebuilds[SnapshotSection::Rs as usize] = rs.0;
    c.draw_snapshot_section_cycles[SnapshotSection::Rs as usize] = rs.1;
    c.draw_snapshot_keys_cycles = keys.0;
    c.draw_snapshot_keys_sampled_cycles = keys.1;
    frame
}

/// Frames with different sampling ratios scale by the window's ratio of sums, not per frame.
#[test]
fn sampled_sections_scale_by_the_window_ratio_across_frames() {
    let mut w = PerfWindow::new();
    // 20 of 40 draws sampled, then 10 of 100.
    w.accumulate(&sampled_frame(40, 20, (20, 5_000), (20_000, 8_000)));
    w.accumulate(&sampled_frame(100, 10, (10, 1_000), (30_000, 3_000)));
    // 6 000 sampled RS cycles over 30 of 140 draws.
    assert_eq!(
        w.draw_snapshot_section_scaled(SnapshotSection::Rs as usize)
            .sum,
        28_000
    );
    // Sampled keys 11 000 less sampled RS 6 000, scaled the same way.
    assert_eq!(w.draw_snapshot_keys_resid_sum(), 23_333);
    // Cost per rebuild divides by the sampled rebuilds of RS, 30.
    assert_eq!(
        w.draw_snapshot_section_per_rebuild(SnapshotSection::Rs as usize),
        Some(200)
    );
    // The per-frame peak is that frame's own scaled estimate.
    assert_eq!(
        w.draw_snapshot_section[SnapshotSection::Rs as usize].max,
        10_000
    );
    assert_eq!(w.draw_snapshot_keys_leftover.max, 20_000);
    assert_eq!(
        w.draw_snapshot_section_per_rebuild(SnapshotSection::Vdecl as usize),
        None
    );
}

/// Scaled sections above the parent `keys` leave `rest` measured, not clamped to zero.
///
/// The section timers' cost lands in the sampled draws only, so the scaled
/// sections can exceed the `keys` row taken over all draws; `rest` comes
/// from the sampled `keys` and still reads what the sampled draws spent
/// outside the sections.
#[test]
fn sections_above_keys_keep_the_rest_from_the_sampled_draws() {
    let mut w = PerfWindow::new();
    w.accumulate(&sampled_frame(64, 4, (4, 4_000), (5_000, 6_000)));
    let sections = w
        .draw_snapshot_section_scaled(SnapshotSection::Rs as usize)
        .sum;
    assert_eq!(sections, 64_000);
    assert!(sections > w.draw_snapshot_keys.sum);
    assert_eq!(w.draw_snapshot_keys_resid_sum(), 32_000);
    let grid = Summary::render_with_ansi(&w, &sample_caches(), 2.0, false);
    assert!(grid.contains("└─ rest"), "the rest row renders: {grid}");
}

/// About one frame in the period is timed, with gaps that never repeat on a fixed beat.
#[test]
fn frames_are_timed_about_one_in_the_period() {
    let mut state = ApiPerfState::new();
    let mut payload = FramePerfPayload::new();
    let frames = 32_000u32;
    let mut timed = 0u32;
    let period = u32::from(FRAME_SAMPLE_PERIOD);
    let mut by_phase = [0u32; FRAME_SAMPLE_PERIOD as usize];
    for frame in 0..frames {
        state.drain_into_payload(&mut payload, true);
        if payload.counters.timed != 0 {
            timed += 1;
            by_phase[(frame % period) as usize] += 1;
        }
    }
    let expected = frames / u32::from(FRAME_SAMPLE_PERIOD);
    assert!(
        timed.abs_diff(expected) < expected / 10,
        "{timed} of {frames} timed"
    );
    for (phase, &n) in by_phase.iter().enumerate() {
        assert!(
            n > expected / 32,
            "frames at position {phase} of 16 timed {n} times"
        );
    }
}

/// An untimed frame hands out no draw-phase target, and the encoder follows the frame's flag.
#[test]
fn untimed_frames_run_no_phase_timers_on_either_side() {
    let mut storage = ApiPerfStorage::new();
    storage.state.borrow_mut().untimed_frames = 3;
    assert!(storage.as_ptr().is_null());
    storage.state.borrow_mut().untimed_frames = 0;
    assert!(!storage.as_ptr().is_null());

    let mut enc = EncoderPerfState::new();
    let mut payload = FramePerfPayload::new();
    payload.counters.timed = 0;
    enc.begin_frame(&payload);
    assert!(enc.op_sub_cycles_ptr(OpSub::Binds).is_null());
    assert!(enc.op_sub_detail_ptr(OpSubDetail::BDraw).is_null());
    payload.counters.timed = 1;
    enc.begin_frame(&payload);
    assert!(!enc.op_sub_cycles_ptr(OpSub::Binds).is_null());
    assert!(!enc.op_sub_detail_ptr(OpSubDetail::BDraw).is_null());
}

/// In an untimed frame a timer only counts its call; a `Frame` call's timer reads the clock.
#[test]
fn untimed_frame_counts_every_call_and_times_only_the_frame_calls() {
    let storage = ApiPerfStorage::new();
    storage.state.borrow_mut().untimed_frames = 5;
    let counted = ApiTimer::frame_gated(Some(&storage), ApiCategory::Device, false);
    assert!(!counted.clock, "an untimed frame reads no clock");
    assert_eq!(storage.state.borrow().timer_depth, 0);
    drop(counted);
    let frame = ApiTimer::frame_gated(Some(&storage), ApiCategory::Device, true);
    assert!(frame.clock, "a Frame call reads the clock in every frame");
    assert_eq!(storage.state.borrow().timer_depth, 1);
    drop(frame);
    let state = storage.state.borrow();
    assert_eq!(
        state.counters.api_call_counts_by_category[ApiCategory::Device as usize],
        2,
        "both calls counted"
    );
    assert_eq!(state.timer_depth, 0);
}

/// A mid-frame flush keeps the frame's flag; only the presenting drain draws the next gap.
#[test]
fn a_flush_keeps_the_frame_flag_and_a_present_moves_on() {
    let mut state = ApiPerfState::new();
    let mut payload = FramePerfPayload::new();
    state.untimed_frames = 0;
    for _ in 0..3 {
        state.drain_into_payload(&mut payload, false);
        assert_eq!(
            payload.counters.timed, 1,
            "every piece of a timed frame is timed"
        );
        assert_eq!(state.untimed_frames, 0);
    }
    state.untimed_frames = 4;
    state.drain_into_payload(&mut payload, false);
    assert_eq!(payload.counters.timed, 0);
    assert_eq!(
        state.untimed_frames, 4,
        "a flush does not count down the gap"
    );
    state.drain_into_payload(&mut payload, true);
    assert_eq!(payload.counters.timed, 0);
    assert_eq!(state.untimed_frames, 3, "the present does");
}

/// Every frame's counts reach the count window; only timed frames reach the timer window.
#[test]
fn every_frame_folds_its_counts_and_only_timed_frames_their_timers() {
    let mut enc = EncoderPerfState::new();
    for timed in [1, 0, 0, 1, 0] {
        let mut frame = sample(100, 0);
        frame.counters.timed = timed;
        frame.counters.vb_rename = 2;
        frame.draws = 7;
        enc.fold(&frame);
    }
    assert_eq!(enc.count_window.frames, 5);
    assert_eq!(enc.perf_window.frames, 2);
    assert_eq!(enc.count_window.vb_rename.sum, 10);
    assert_eq!(enc.count_window.draws.sum, 35);
    assert_eq!(
        enc.perf_window.enc_cyc.sum, 200,
        "the timers of the two timed frames"
    );
    let line = render_kv(&enc.perf_window, &enc.count_window, &sample_caches(), 2.0).finish();
    assert!(line.contains(" frames=5 timed_frames=2 "), "{line}");
    assert!(line.contains(" vb_rename_total=10 "), "{line}");
}

/// A window with no timed frame still reports its counts, with its timer rows at zero.
#[test]
fn a_window_without_timed_frames_reports_counts_and_zero_timers() {
    let mut count = PerfWindow::new();
    let mut frame = sample(100, 0);
    frame.counters.vb_rename = 3;
    count.accumulate(&frame);
    let timed = PerfWindow::new();
    let line = render_kv(&timed, &count, &sample_caches(), 2.0).finish();
    assert!(line.contains(" frames=1 timed_frames=0 "), "{line}");
    assert!(line.contains(" vb_rename_total=3 "), "{line}");
    assert!(line.contains(" frame_ms=0.000 "), "{line}");
    let grid = Summary::render_windows(&timed, &count, &sample_caches(), 2.0, false);
    assert!(grid.contains("frames=1  timed=0"), "{grid}");
}

/// The grid takes its counts and the peaks of the every-frame timers from every frame.
///
/// The per-call ratios stay with the timed frames, whose time they divide.
#[test]
fn grid_counts_and_every_frame_peaks_come_from_every_frame() {
    let device = ApiCategory::Device as usize;
    let render_state = DeviceSubCategory::RenderState as usize;
    let timed_frame = {
        let mut frame = sample(100, 0);
        frame.counters.timed = 1;
        frame.counters.api_call_counts_by_category[device] = 10;
        frame.counters.device_sub_calls[render_state] = 10;
        frame.counters.device_sub_cycles[render_state] = 1_000;
        frame.enc.slot_waits = 1;
        frame.timing.frame_total_cycles = 1_000_000;
        frame.timing.present_block_cycles = 100_000;
        frame
    };
    let untimed_frame = {
        let mut frame = sample(100, 0);
        frame.counters.api_call_counts_by_category[device] = 30;
        frame.counters.device_sub_calls[render_state] = 30;
        frame.enc.slot_waits = 4;
        frame.timing.frame_total_cycles = 9_000_000;
        frame.timing.present_block_cycles = 7_000_000;
        frame
    };
    let mut enc = EncoderPerfState::new();
    enc.fold(&timed_frame);
    enc.fold(&untimed_frame);
    let grid = Summary::render_windows(
        &enc.perf_window,
        &enc.count_window,
        &sample_caches(),
        2.0,
        false,
    );
    let row = |label: &str| {
        grid.lines()
            .find(|line| line.starts_with(label))
            .unwrap_or_else(|| panic!("no {label} row in\n{grid}"))
            .to_owned()
    };
    assert!(row("├─ D3D9 calls").contains("40 calls"), "{grid}");
    assert!(row("│  ├─ Device").contains("(        40)"), "{grid}");
    assert!(row("└─ Slot waits").contains("(         5)"), "{grid}");
    assert!(
        row("│  │  ├─ RenderState").contains("( 100 ns/call)"),
        "the ratio divides the timed frame's time by its own calls: {grid}"
    );
    let peak = |ns: u64| format!("peak {:>5.2} ms", cycles_to_ms(ns));
    assert!(row("API thread").contains(&peak(9_000_000)), "{grid}");
    assert!(row("Frame total").contains(&peak(9_000_000)), "{grid}");
    assert!(
        row("│  │  │  ├─ Send stall").contains(&peak(7_000_000)),
        "{grid}"
    );

    let line = render_kv(&enc.perf_window, &enc.count_window, &sample_caches(), 2.0).finish();
    assert!(line.contains(" api_calls_total=40 "), "{line}");
    assert!(line.contains(" slot_waits_total=5 "), "{line}");
    let kv_peak = |ns: u64| format!("{:.3}", cycles_to_ms(ns));
    assert!(
        line.contains(&format!(" frame_peak_ms={} ", kv_peak(9_000_000))),
        "{line}"
    );
    assert!(
        line.contains(&format!(" present_stall_peak_ms={} ", kv_peak(7_000_000))),
        "{line}"
    );
    assert!(
        line.contains(&format!(" frame_ms={} ", kv_peak(1_000_000))),
        "the time stays with the timed frame: {line}"
    );
}

/// The frame picks follow the seed alone: the draw picks between presents leave them be.
#[test]
fn frame_picks_depend_only_on_the_seed_and_the_frame_count() {
    let picks = |seed: u32, draws_per_frame: u32| {
        let mut state = ApiPerfState::with_frame_seed(seed);
        let mut payload = FramePerfPayload::new();
        (0..512)
            .map(|_| {
                for _ in 0..draws_per_frame {
                    state.record_snapshot_rebuild(SnapshotSection::Rs.bit());
                }
                state.drain_into_payload(&mut payload, true);
                payload.counters.timed != 0
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(
        picks(7, 0),
        picks(7, 37),
        "draws do not move the frame picks"
    );
    assert_ne!(picks(7, 0), picks(8, 0), "another seed picks other frames");
    assert_eq!(
        ApiPerfState::with_frame_seed(0).frame_sample_state,
        1,
        "a zero seed is made odd, since xorshift stays at 0"
    );
}

/// `d3d9.dll`'s page-box and pool traffic rides the payload and joins the encoder's own.
///
/// The `PageBox` counters and the pool the game recycles through belong to
/// `d3d9.dll`, while the summary runs in `mtld3d.so`. The API thread
/// carries the delta since its previous drain (none on the first, which has
/// no baseline) and the parked gauge; the sample adds them to the encoder's.
#[test]
fn pe_pagebox_traffic_rides_the_payload_into_the_window() {
    const fn volume(allocs: u64, frees: u64, uncached_allocs: u64) -> PageBoxVolume {
        PageBoxVolume {
            allocs,
            alloc_bytes: allocs * 16_384,
            frees,
            free_bytes: frees * 16_384,
            uncached_allocs,
        }
    }
    const fn pool(recycled: u64, parked_bytes: u64) -> PoolTraffic {
        PoolTraffic {
            recycled,
            recycled_bytes: recycled * 32_768,
            parked_bytes,
        }
    }
    let mut api = ApiPerfState::new();
    let mut payload = FramePerfPayload::new();
    api.drain_into_payload(&mut payload, true);
    api.carry_pagebox_traffic(&mut payload, volume(100, 40, 3), pool(10, 4_096));
    assert_eq!(
        payload.counters.pagebox_allocs, 0,
        "the first drain has no baseline"
    );
    assert_eq!(payload.counters.pool_recycled, 0);
    assert_eq!(
        payload.counters.pool_parked_bytes, 4_096,
        "the gauge needs none"
    );

    api.drain_into_payload(&mut payload, true);
    api.carry_pagebox_traffic(&mut payload, volume(130, 55, 4), pool(4_016, 8_192));
    let carried = &payload.counters;
    assert_eq!(carried.pagebox_allocs, 30);
    assert_eq!(carried.pagebox_alloc_bytes, 30 * 16_384);
    assert_eq!(carried.pagebox_frees, 15);
    assert_eq!(carried.pagebox_free_bytes, 15 * 16_384);
    assert_eq!(carried.pagebox_uncached_allocs, 1);
    assert_eq!(carried.pool_recycled, 4_006);
    assert_eq!(carried.pool_recycled_bytes, 4_006 * 32_768);
    assert_eq!(carried.pool_parked_bytes, 8_192);

    let mut enc = EncoderPerfState::new();
    enc.begin_frame(&payload);
    enc.bump_pagebox_pool_recycled(16_384);
    let mut caches = sample_caches();
    caches.pagebox_pool_bytes = 1_000;
    // A fresh encoder has no baseline of its own, so its runtime adds no
    // `PageBox` traffic to this sample.
    let frame = enc.take_sample(&caches, &[], 0, 0);
    assert_eq!(frame.pagebox_allocs, 0, "the sample's own runtime only");
    assert_eq!(frame.pagebox_pool_bytes, 1_000);

    let mut window = PerfWindow::new();
    window.accumulate(&frame);
    assert_eq!(
        window.pagebox_pool_recycled.sum, 1,
        "the encoder's pool only"
    );
    assert_eq!(window.pagebox_allocs.sum, 0);
    assert_eq!(window.pe_pagebox_allocs.sum, 30);
    assert_eq!(window.pe_pagebox_alloc_bytes.sum, 30 * 16_384);
    assert_eq!(window.pe_pagebox_frees.sum, 15);
    assert_eq!(window.pe_pagebox_uncached_allocs.sum, 1);
    assert_eq!(window.pe_pagebox_pool_recycled.sum, 4_006);
    assert_eq!(window.pe_pagebox_pool_recycled_bytes.sum, 4_006 * 32_768);
    assert_eq!(window.pe_pagebox_pool_bytes.max, 8_192);
    assert_eq!(
        window.pagebox_pool_bytes_both.max,
        1_000 + 8_192,
        "both pools parked"
    );
    assert_eq!(window.pagebox_uncached_both.max, 1);

    // The old keys keep the encoder's runtime; `d3d9.dll`'s get keys of their own.
    let kv = render_kv(&window, &window, &caches, 2.0).finish();
    for pair in [
        " pagebox_alloc_total=0 ",
        " pagebox_pool_recycled_total=1 ",
        " pagebox_pool_parked_bytes=1000 ",
        " pe_pagebox_alloc_total=30 ",
        " pe_pagebox_free_total=15 ",
        " pe_pagebox_uncached_total=1",
        " pe_pagebox_pool_recycled_total=4006 ",
        " pe_pagebox_pool_parked_bytes=8192 ",
    ] {
        assert!(kv.contains(pair), "{pair} missing from {kv}");
    }
    // The grid's rows show the two runtimes together.
    let grid = Summary::render_with_ansi(&window, &caches, 2.0, false);
    assert!(grid.contains("recycled=4007 "), "{grid}");
    assert!(grid.contains("alloc=30 "), "{grid}");
    assert!(grid.contains("free=15 "), "{grid}");
}
