//! Unit tests for the perf counters, the bottleneck classifier and the summary layout.
//!
//! Built only under `perf_tracking`, these pin arithmetic that otherwise fails silently:
//! the API-to-encoder counter drain with its per-frame reset, window accumulation keeping
//! encoder CPU separate from submit-thread drawable wait, exclusive-time accounting for
//! nested timers (self-times partition the outer span), every `Bottleneck::classify`
//! branch, and a golden snapshot of the summary grid in both plain and ANSI form.

use super::*;

const fn sample(enc_cyc: u64, drawable_wait: u64) -> FrameSample {
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

    let mut p = FramePerfPayload::new();
    api.drain_into_payload(&mut p);

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

    // Second drain should report a real frame_total (tsc moved).
    let mut p2 = FramePerfPayload::new();
    api.drain_into_payload(&mut p2);
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
    let plain = Summary::render_with_ansi(&w, &caches, 5.01, false);
    let colored = Summary::render_with_ansi(&w, &caches, 5.01, true);
    let stripped = strip_ansi(&colored);
    assert_eq!(
        plain, stripped,
        "ANSI off output must equal the ANSI-on output with escapes stripped"
    );
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
    let got = Summary::render_with_ansi(&w, &caches, 5.01, false);
    let want = concat!(
        "── perf  window=5.01s  frames=1  bottleneck=ENCODER (GPU) ──\n",
        "buckets: api_d3d9=2.80  api_outside=3.00  enc_work=1.50  submit_work=0.10  gpu_wait=6.00  (ms/frame, avg)\n",
        "\n",
        "API thread             10.00 ms                                             peak 10.00 ms\n",
        "├─ D3D9 calls           4.00 ms   ( 40.0 %)           243 calls             peak  4.00 ms\n",
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
        "│  ├─ Surface           0.00 ms   (         0)                              peak  0.00 ms\n",
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
        "└─ Drawable wait        6.00 ms                       nextDrawable GPU+comp peak  6.00 ms\n",
        "\n",
        "Frame total            10.00 ms                                             peak 10.00 ms\n",
        "submit_status=0x0   (API, Encoder, Submit run in parallel; frame_total ≥ max(api_cpu, enc_cpu, submit_cpu + gpu_wait))\n",
        "\n",
        "Resources (VB/IB)  — window totals; depth/retention averaged\n",
        "rename      VB=12     IB=3          peak/frame VB=12  IB=3      API: PageBox alloc on contended Lock(DISCARD) or whole-buffer Lock\n",
        "  discards  VB=10     IB=3                                      API: rename, no preserve (DISCARD or whole-buffer WRITEONLY)\n",
        "  preserve  2                                                   API: rename + sync memcpy (whole-buffer non-WRITEONLY contended — game may read back)\n",
        "  bytes     720 KB                  peak/frame 720 KB           API: fresh PageBox bytes behind rename (16 KiB-padded; what the allocator serves)\n",
        "staging up  0                                                   encoder: Staged (non-DYNAMIC) dirty-range upload blits — separate-staging path; high here with rename≈0 is the goal\n",
        "reorder     0                                                   encoder: rename-at-overlap (upload hit a just-drawn region; rare)\n",
        "destroys    1                                                   encoder: MTLBuffer wrappers freed (VB/IB cache renames, Lock-rename intake, visibility-pool eviction)\n",
        "ret cap     drain=2 submit=1        peak/frame submit=1         API: VB/IB retention cap hit before a rename alloc (drain=cheap, submit=GPU wait)\n",
        "retention   depth= 6.0  3.5 MB avg  peak depth=6   3.5 MB       encoder: shared PageBox queue (VB/IB renames + texture-blit padded staging + visibility pool)\n",
        "pool        hit=14 miss=1 (93.3%)   recycled=14  672 KB         API pops a warm same-size PageBox; encoder parks retired ones (memory.pageboxPoolCapMB, 0 = off)\n",
        "  parked    1.0 MB avg              peak 1.0 MB                 bytes held in the pool for reuse (already retired; excluded from retention above)\n",
        "\n",
        "Resources (textures)  — same layout as VB/IB; n/a rows omitted\n",
        "rename      2                                                   API: fresh staging Arc on contended LockRect\n",
        "  discards  1                                                   API: rename, no preserve (DISCARD or D3DUSAGE_DYNAMIC)\n",
        "  preserve  1                       peak/frame 1                API: rename + sync memcpy (non-DISCARD non-DYNAMIC contended)\n",
        "uploads     2                                                   encoder: total texture uploads (raw + padded)\n",
        "  raw       2                                                   encoder: blit; source = cached bytesNoCopy wrapper (cheap)\n",
        "  padded    0                                                   encoder: blit; source repacked into transient buffer (alloc + memcpy + extra unix_call)\n",
        "reorder     1                                                   encoder: MTLTexture rename-at-overlap (upload hit a texture sampled earlier this frame)\n",
        "destroys    1                                                   encoder: MTLTexture freed + texture-staging MTLBuffer wrappers freed (rename + padded + texture release)\n",
        "retention   depth= 0.0  0 KB avg    peak 0 KB                   encoder: blit source staging Arcs (separate from VB/IB retention; MTLTexture handles in destroys)\n",
        "dirtyrect   4                       3 partial, 25% mip          API: AddDirtyRect calls; partial = sub-region narrower than mip (feeds dirty-rect snapshot upload decision)\n",
        "\n",
        "Caches (live sizes at summary emit)\n",
        "  textures=48     pipelines=12     samplers=6     programs=8\n",
        "  libs=8        depth_states=4\n",
        "\n",
        "Commands / passes (raw window totals)\n",
        "  passes=4         commands=140         draws=100\n",
        "  pipeline memo  97 / 100  (97.0%)  consecutive-draw resolve elided\n",
        "  fan generated  0         indexed / oversized fans rewritten per draw (slow path; 0 is the goal)\n",
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
        "Per-frame allocator footprint  scratch as small/oversized blocks; op_vec + cmd_vec each split into size + realloc\n",
        "scratch     24.0/0.0  2.0 MB avg    peak 24/0   peak 2.0 MB     per-frame bump arena (VS/PS constants + UP vertices); cleared at begin_frame\n",
        "op_vec                                                          API→encoder: Vec<Op> shipped via sync_channel(1); encoder drains and translates each Op\n",
        "  size      72 KB avg               peak 72 KB                  high-water-reserved Vec<Op> capacity per frame (peak_ops_count)\n",
        "  realloc   32 KB avg               peak 32 KB                  Vec<Op> doubling memcpy on push_op (target ≈ 0 with peak_ops_count reserve)\n",
        "cmd_vec                                                         encoder→unix: Vec<Command> shipped via SubmitCommandBuffer; unix dispatches each Command to a Metal encoder\n",
        "  size      64 KB avg               peak 64 KB                  pool-resident Vec<Command> capacity across frames (recycled, not freed)\n",
        "  realloc   192 KB avg              peak 192 KB                 Vec<Command> doubling memcpy on emit_command (target ≈ 0 with Pass::commands pool)\n",
        "pagebox     alloc=17  4.8 MB        free=15  4.6 MB             window totals of PageBox allocs/frees reaching the global allocator (fresh pages fault on first touch)\n",
        "  uncached  1 (5.9%)                peak 1/frame                allocs over 1 MiB: past snmalloc's per-thread budget, so commit in / decommit out every time\n",
        "faults      minflt=4200  majflt=3   4200.0 min/frame            process-wide getrusage delta this window (all threads); zero-fill faults on fresh pages land here",
    );
    assert_eq!(got, want, "perf summary drifted — diff above");
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
    let out = Summary::render_with_ansi(&w, &caches, 5.01, false);
    for expected in [
        "── perf  window=5.01s",
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
        "└─ Drawable wait",
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
    calls[ApiCategory::Device as usize] = 123;
    calls[ApiCategory::VertexBuffer as usize] = 88;
    calls[ApiCategory::IndexBuffer as usize] = 32;
    // Decompose Device into sub-buckets; must sum to cats[Device]
    // = 300_000 to mirror the production invariant. Calls sum to
    // calls[Device] = 123.
    let mut dsub = [0u64; DeviceSubCategory::COUNT];
    let mut dcalls = [0u32; DeviceSubCategory::COUNT];
    // Values picked so `cycles / tsc_hz * 1e3` rounds cleanly at
    // `{:.2}` even with the small jitter in runtime calibration —
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
    // `{:.2}` under runtime tsc_hz calibration.
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
    let s = FrameSample {
        counters: FrameCounters {
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
            keys_gate_calls: [0; KeysGate::COUNT],
            keys_gate_skips: [0; KeysGate::COUNT],
            // snapshot dominates the Draws bucket; split inside it
            // is stages 20 + c_ff 20 + c_pr 10 + keys 20 + bumps 10
            // + leftover 10 = 90. push_op trails. Every component is
            // a multiple of 10_000 so `cycles / tsc_hz * 1e3` rounds
            // identically across calibration jitter (the underlying
            // tsc_hz wobble of ±few ppm only flips rounding for
            // values like 5_000 or 25_000 that fall on the {:.2}
            // boundary).
            draw_snapshot_cycles: 90_000,
            draw_snapshot_stages_cycles: 20_000,
            draw_snapshot_c_ff_cycles: 20_000,
            draw_snapshot_c_pr_cycles: 10_000,
            draw_snapshot_keys_cycles: 20_000,
            draw_snapshot_bumps_cycles: 10_000,
            draw_push_op_cycles: 20_000,
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
            vbib_staging_uploads: 0,
            vbib_mid_pass_reorders: 0,
            texture_blit_uploads: 2,
            texture_blit_padded_uploads: 0,
            texture_gpu_renames: 1,
            op_cycles: 1_400_000,
            // Decompose op_cyc 1.40M into the nine phases (six draw phases sum
            // 1.35M + three non-draw phases sum 0.03M = 1.38M) so the golden
            // pins each "Closures (op)" sub-row; resid = 0.02M. Multiples of
            // 10_000 cyc round cleanly at {:.2} under tsc jitter.
            op_sub_cycles: [
                300_000, 400_000, 100_000, 200_000, 150_000, 200_000, 10_000, 10_000, 10_000,
            ],
            // resolve(300k) split 150/90/30 → resid 30k; binds(200k) split
            // 120/50/20 → resid 10k. Multiples of 10k round cleanly at {:.2}.
            op_sub_detail: [150_000, 90_000, 30_000, 120_000, 50_000, 20_000],
            // 97 of 100 pipeline resolves served from the memo → 97.0%.
            pipeline_memo_hits: 97,
            fan_generated: 0,
            pipeline_memo_calls: 100,
            submit_cycles: 300_000,
            // Submit thread: total execute 6.1M = encode+commit 0.1M +
            // drawable wait 6.0M.
            drawable_wait_cycles: 6_000_000,
            submit_exec_cycles: 6_100_000,
            // 0.2M backpressure stall → Finalize 0.10M, stall 0.20M.
            submit_stall_cycles: 200_000,
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
    w.minor_faults_window = 4200;
    w.major_faults_window = 3;
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
