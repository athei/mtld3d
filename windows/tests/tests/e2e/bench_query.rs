//! Frame time when every frame waits on the EVENT query of the frame before it.
//!
//! Guards the cost of an EVENT-query throttle. World of Warcraft's "Reduce
//! Input Lag" option (`gxFixLag`, on by default) issues an EVENT query each
//! frame and spins on `GetData(D3DGETDATA_FLUSH)` until the previous one
//! answers, to keep the CPU from running ahead of the GPU. When that answer
//! waits for the GPU to retire the frame, the CPU and GPU stop overlapping:
//! 3.3.5a loses about 40% of its frame rate that way, with unchanged GPU time
//! per frame. The built-in `wow` profile therefore sets
//! `query.eventImmediate=true` (and `query.flushImmediate=true`), and a
//! change that makes the throttled frame slower, the poll dearer or the
//! answer late again shows here first.
//!
//! Both tests draw the same light frame: about 300 textured `vs_2_0`/`ps_2_0`
//! draws in ten material runs, then two small flare quads, each inside an
//! occlusion query that the next frame reads with `GetData(0)`, no FLUSH, as
//! the game reads its lens-flare queries. At the end of every frame, before
//! `Present`, the frame issues its EVENT query and then polls the previous
//! frame's with `D3DGETDATA_FLUSH` until it answers `S_OK`, counting the
//! calls and timing, with the benchmarks' `rdtsc` clock (`TscClock`), the span
//! from that query's `Issue` to the answer. The span includes the frame
//! between them, so under the `wow` keys, where the first poll answers, it
//! is about one frame; on the spec path it runs to the GPU's retirement of
//! the frame the query was issued in.
//!
//! `query_poll_wow` passes the built-in `wow` profile's settings, read from
//! `mtld3d-core` so the benchmark follows the profile, as harness
//! configuration: the profile matches `WoW.exe` by its version strings and
//! never this executable. Under it the first poll answers, so its poll
//! counts are exact and any other count means the immediate answer is
//! gone; on the spec path a cheaper poll spins more often, so there they
//! are context only. `query_poll_spec` pins both keys to their
//! defaults, stated explicitly so a suite-wide `BENCH_CONFIG` cannot move
//! them. `query.flushImmediate` changes only an occlusion read that passes
//! the FLUSH flag, which this frame never does; it is set in the `wow` test
//! only so the test runs what the profile runs.

use std::time::{Duration, Instant, SystemTime};

use mtld3d_tests::{
    Harness, HarnessConfig, IndexBuffer, MemorySample, PixelShader, Query, Texture, VertexBuffer,
    VertexDeclaration, VertexShader,
};
use mtld3d_types::{
    D3D_OK, D3DCLEAR_TARGET, D3DCLEAR_ZBUFFER, D3DFMT_D24S8, D3DFMT_INDEX16, D3DGETDATA_FLUSH,
    D3DISSUE_BEGIN, D3DISSUE_END, D3DPOOL_MANAGED, D3DPRESENT_INTERVAL_IMMEDIATE,
    D3DPT_TRIANGLELIST, D3DQUERYTYPE_EVENT, D3DQUERYTYPE_OCCLUSION, D3DUSAGE_WRITEONLY, S_FALSE,
};

use crate::bench::{
    Class, Direction, FrameClock, FrameStats, FrameWork, IDENTITY_ROWS, LayerLog, Metrics, Model,
    STRIDE, TEXTURED_DECL, TscClock, Value, grid, material_ps, material_vs, memory_section,
    nearest_rank, ok, pattern_texture, ratio, world_rows, write_report,
};

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
/// Material pairs, each with a texture of its own; a run draws one.
const MATERIALS: u32 = 8;
const RUNS: u32 = 10;
const DRAWS_PER_RUN: u32 = 30;
/// Occlusion queries per frame, each around one flare quad drawn after the scene.
const FLARES: usize = 2;
/// Frames of queries in flight: a frame's queries are read by the next frame.
const QUERY_SETS: usize = 2;
const WARM_UP_FRAMES: u32 = 60;
/// The measured phase is at least this many frames and at least [`MIN_MEASURED`] long.
///
/// The duration floor puts one whole window of a `PERF=1` build's summary
/// inside the measured frames.
const MEASURED_FRAMES: usize = 600;
const MIN_MEASURED: Duration = Duration::from_secs(12);
/// The longest one EVENT query may be polled before the benchmark fails.
const POLL_LIMIT: Duration = Duration::from_secs(5);
/// The shortest frame the sample buffers are sized for, so they never grow while measuring.
const FRAME_FLOOR: Duration = Duration::from_micros(50);
/// The two query keys of the `wow` profile at their defaults, the D3D9 answers.
const SPEC_KEYS: &str = "query.flushImmediate=false;query.eventImmediate=false";

/// How the EVENT polls are answered, which decides how a comparison reads the poll counts.
enum Answering {
    /// At once: the first poll answers, and any other count means that stopped.
    Immediate,
    /// From GPU retirement: a cheaper poll spins more often, so the counts are context.
    Retirement,
}

/// The throttled frame under the built-in `wow` profile's settings.
#[test]
#[ignore = "benchmark, run by `make bench`"]
fn query_poll_wow() {
    let wow = mtld3d_core::app_profile::builtin("wow").expect("the wow profile ships");
    poll("query_poll_wow", wow.settings(), &Answering::Immediate);
}

/// The throttled frame with both query keys at their defaults.
#[test]
#[ignore = "benchmark, run by `make bench`"]
fn query_poll_spec() {
    poll("query_poll_spec", SPEC_KEYS, &Answering::Retirement);
}

/// Warm up, time the throttled frames under `keys`, and write the report `name`.
fn poll(name: &str, keys: &'static str, answering: &Answering) {
    let h = Harness::create(&HarnessConfig {
        width: WIDTH,
        height: HEIGHT,
        depth_format: Some(D3DFMT_D24S8),
        presentation_interval: D3DPRESENT_INTERVAL_IMMEDIATE,
        config_entries: keys,
        ..HarnessConfig::default()
    });
    let tsc = TscClock::calibrated();
    let since = SystemTime::now();
    let mut scene = Scene::new(&h);
    let mut frame = 0;
    for _ in 0..WARM_UP_FRAMES {
        assert!(h.pump(), "WM_QUIT during warm-up");
        scene.read_occlusion(frame);
        scene.draw(frame);
        scene.throttle(frame);
        ok(h.present(), "Present");
        frame += 1;
    }
    let log = LayerLog::find(since);
    let warm = MemorySample::now();

    let from = log.mark();
    let capacity = usize::try_from(MIN_MEASURED.as_micros() / FRAME_FLOOR.as_micros())
        .expect("frame capacity fits usize")
        .max(MEASURED_FRAMES);
    let mut clock = FrameClock::start(&tsc, capacity);
    let mut polls = Vec::with_capacity(capacity);
    let mut latencies = Vec::with_capacity(capacity);
    let (mut ready, mut pending) = (0, 0);
    while clock.frames() < MEASURED_FRAMES || clock.elapsed() < MIN_MEASURED {
        assert!(h.pump(), "WM_QUIT during the measured frames");
        let occlusion = scene.read_occlusion(frame);
        ready += occlusion.ready;
        pending += occlusion.pending;
        scene.draw(frame);
        let answer = scene
            .throttle(frame)
            .expect("the previous frame issued its EVENT query");
        polls.push(answer.polls);
        latencies.push(tsc.duration(answer.latency));
        clock.present(&h);
        frame += 1;
    }
    let to = log.mark();
    let end = MemorySample::now();

    let stats = clock.stats();
    let work = clock.work_stats();
    let latency = FrameStats::of(&latencies);
    polls.sort_unstable();
    let frames = polls.len();
    let polls_total: u64 = polls.iter().sum();
    let (polls_p50, polls_p99, polls_max) = (
        polls[nearest_rank(frames, 50)],
        polls[nearest_rank(frames, 99)],
        polls[frames - 1],
    );
    let body = format!(
        "shape: back buffer {WIDTH}x{HEIGHT} X8R8G8B8 + D24S8; {scene_draws} scene draws in \
         {RUNS} material runs + {FLARES} flare draws in occlusion queries per frame\n\
         harness configuration, over MTLD3D_CONFIG: {keys}\n\
         per frame: Issue(END) of this frame's EVENT query, then GetData(FLUSH) on the \
         previous frame's until S_OK, then Present; each frame reads the previous frame's \
         occlusion queries once with GetData(0)\n\
         warm-up: {WARM_UP_FRAMES} frames\n\
         measured: {frames} frames in {elapsed:.2?} (at least {MEASURED_FRAMES} frames \
         and {MIN_MEASURED:?})\n\
         frame time (Present to Present): {row}\n\
         API work (Present return to Present call, the polls included): {work}\n\
         EVENT Issue to S_OK, one frame between: {latency}\n\
         EVENT polls per frame: mean {polls_mean:.2}  p50 {polls_p50}  p99 {polls_p99}  \
         max {polls_max}  total {polls_total}\n\
         occlusion reads one frame later: {ready} ready, {pending} still pending\n\
         {memory}{perf}",
        scene_draws = RUNS * DRAWS_PER_RUN,
        elapsed = clock.elapsed(),
        row = stats.row(),
        work = work.row(),
        latency = latency.row(),
        polls_mean = mean(polls_total, frames),
        memory = memory_section(&warm, &end),
        perf = log.perf_rows(from, to).section(),
    );

    let count = |n: usize| Value::Count(u64::try_from(n).expect("a count fits u64"));
    let mut metrics = Metrics::new(name, &h, &tsc);
    metrics.frame_rows("frame", &stats);
    metrics.frame_rows("api", &work);
    metrics.frame_rows("event.latency", &latency);
    let typical = || match answering {
        Answering::Immediate => Class::Exact,
        Answering::Retirement => Class::Info,
    };
    for (row, value, class) in [
        ("polls.p50", polls_p50, typical()),
        ("polls.p99", polls_p99, typical()),
        ("polls.max", polls_max, Class::Info),
        ("polls.total", polls_total, Class::Info),
    ] {
        metrics.metric(row, Value::Count(value), Direction::Lower, class);
    }
    metrics.metric(
        "occlusion.ready",
        count(ready),
        Direction::Higher,
        Class::Info,
    );
    metrics.metric(
        "occlusion.pending",
        count(pending),
        Direction::Lower,
        Class::Info,
    );
    metrics.metric(
        "measured.frames",
        count(frames),
        Direction::Higher,
        Class::Info,
    );
    metrics.metric(
        "measured.ms",
        Value::Ms(clock.elapsed()),
        Direction::Lower,
        Class::Info,
    );
    metrics.memory(&warm, &end);
    // Answered at once, every frame polls once and issues the same calls;
    // answered from retirement, the poll count and so the query calls vary.
    let frame_work = match answering {
        Answering::Immediate => FrameWork::Fixed,
        Answering::Retirement => FrameWork::Varying,
    };
    metrics.perf(&log.perf_kv_last_full(from, to), &frame_work);
    write_report(&metrics, &log, &body);
}

/// `total / count`, for a report row.
fn mean(total: u64, count: usize) -> f64 {
    let high = u32::try_from(total >> 32).expect("the high half fits u32");
    let low = u32::try_from(total & 0xFFFF_FFFF).expect("the low half fits u32");
    let total = f64::from(high).mul_add(4_294_967_296.0, f64::from(low));
    total / f64::from(u32::try_from(count).expect("frame count fits u32"))
}

/// How one frame's reads of the previous frame's occlusion queries answered.
struct OcclusionReads {
    ready: usize,
    pending: usize,
}

/// How the previous frame's EVENT query answered.
struct Answer {
    /// `GetData` calls up to and including the one that answered `S_OK`.
    polls: u64,
    /// From the return of that query's `Issue` to the return of the `S_OK`, in `rdtsc` ticks.
    latency: u64,
}

/// The frame's device objects and its queries.
struct Scene<'h> {
    h: &'h Harness,
    vb: VertexBuffer<'h>,
    ib: IndexBuffer<'h>,
    decl: VertexDeclaration<'h>,
    materials: Vec<(VertexShader<'h>, PixelShader<'h>, Texture<'h>)>,
    vertex_count: u32,
    triangles: u32,
    /// One EVENT query per frame in flight, by frame parity.
    events: [Query<'h>; QUERY_SETS],
    /// When each EVENT query was last issued, an `rdtsc` reading.
    issued: [Option<u64>; QUERY_SETS],
    /// The flares' occlusion queries, one set per frame in flight.
    occlusion: [[Query<'h>; FLARES]; QUERY_SETS],
    /// Whether each set of occlusion queries has been issued.
    occlusion_issued: [bool; QUERY_SETS],
}

impl<'h> Scene<'h> {
    fn new(h: &'h Harness) -> Self {
        let query = |kind| {
            h.create_query(kind)
                .expect("EVENT and OCCLUSION queries are supported")
        };
        let (vertices, indices) = grid(4);
        let vertex_count = u32::try_from(vertices.len()).expect("mesh fits u32");
        let triangles = u32::try_from(indices.len() / 3).expect("mesh fits u32");
        let vb = h.create_vertex_buffer(
            vertex_count * STRIDE,
            D3DUSAGE_WRITEONLY,
            0,
            D3DPOOL_MANAGED,
        );
        vb.lock(0, 0, 0).write(&vertices);
        let ib = h.create_index_buffer(
            triangles * 6,
            D3DUSAGE_WRITEONLY,
            D3DFMT_INDEX16,
            D3DPOOL_MANAGED,
        );
        ib.lock(0, 0, 0).write(&indices);
        let materials = (0..MATERIALS)
            .map(|at| {
                let shade = ratio(at, MATERIALS);
                (
                    h.create_vertex_shader(&material_vs(&Model::Sm2, shade * 0.125)),
                    h.create_pixel_shader(&material_ps(
                        &Model::Sm2,
                        [0.0, shade * 0.25, 0.0, 0.0],
                        false,
                    )),
                    pattern_texture(h, 0xFF30_6090 ^ (at * 0x0011_0B05)),
                )
            })
            .collect();
        let scene = Self {
            h,
            vb,
            ib,
            decl: h.create_vertex_declaration(&TEXTURED_DECL),
            materials,
            vertex_count,
            triangles,
            events: [(); QUERY_SETS].map(|()| query(D3DQUERYTYPE_EVENT)),
            issued: [None; QUERY_SETS],
            occlusion: [(); QUERY_SETS]
                .map(|()| [(); FLARES].map(|()| query(D3DQUERYTYPE_OCCLUSION))),
            occlusion_issued: [false; QUERY_SETS],
        };
        ok(h.set_vertex_declaration(&scene.decl), "declaration");
        ok(h.set_stream_source(0, &scene.vb, 0, STRIDE), "stream");
        ok(h.set_indices(&scene.ib), "indices");
        ok(
            h.set_vertex_shader_constant_f(0, &IDENTITY_ROWS),
            "view-projection",
        );
        ok(h.set_pixel_shader_constant_f(0, &[1.0; 4]), "tint");
        scene
    }

    /// Read the previous frame's occlusion queries once each, without FLUSH.
    fn read_occlusion(&self, frame: u32) -> OcclusionReads {
        let set = set_of(frame + 1);
        let mut reads = OcclusionReads {
            ready: 0,
            pending: 0,
        };
        if !self.occlusion_issued[set] {
            return reads;
        }
        for query in &self.occlusion[set] {
            match query.data_u32(0) {
                (D3D_OK, _) => reads.ready += 1,
                (S_FALSE, _) => reads.pending += 1,
                (hr, _) => panic!("occlusion GetData(0): 0x{hr:08X}"),
            }
        }
        reads
    }

    /// Draw the frame: the scene's material runs, then the flares inside this frame's queries.
    fn draw(&mut self, frame: u32) {
        let h = self.h;
        ok(h.begin_scene(), "BeginScene");
        ok(
            h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, 0xFF20_3040, 1.0, 0),
            "clear",
        );
        for run in 0..RUNS {
            let (vs, ps, texture) =
                &self.materials[usize::try_from(run % MATERIALS).expect("fits usize")];
            ok(h.set_vertex_shader(vs), "VS");
            ok(h.set_pixel_shader(ps), "PS");
            ok(h.set_texture(0, texture), "texture");
            for at in 0..DRAWS_PER_RUN {
                let spot = run * DRAWS_PER_RUN + at;
                self.mesh(
                    0.08,
                    ratio(spot % 20, 20).mul_add(1.8, -0.95),
                    ratio(spot / 20 % 15, 15).mul_add(1.8, -0.95),
                    0.5,
                );
            }
        }
        let set = set_of(frame);
        for (at, query) in self.occlusion[set].iter().enumerate() {
            let offset = ratio(u32::try_from(at).expect("flare index fits u32"), 1);
            ok(query.issue(D3DISSUE_BEGIN), "occlusion Issue(BEGIN)");
            self.mesh(0.05, offset.mul_add(0.8, -0.4), 0.6, 0.2);
            ok(query.issue(D3DISSUE_END), "occlusion Issue(END)");
        }
        self.occlusion_issued[set] = true;
        ok(h.end_scene(), "EndScene");
    }

    /// Issue this frame's EVENT query, then poll the previous frame's until it answers.
    ///
    /// `None` when the previous frame issued none, which only the first
    /// warm-up frame meets.
    ///
    /// # Panics
    /// Panics if a poll fails or the query has not answered within [`POLL_LIMIT`].
    fn throttle(&mut self, frame: u32) -> Option<Answer> {
        let set = set_of(frame);
        ok(self.events[set].issue(D3DISSUE_END), "EVENT Issue(END)");
        self.issued[set] = Some(TscClock::now());

        let previous = set_of(frame + 1);
        let issued = self.issued[previous]?;
        let event = &self.events[previous];
        let started = Instant::now();
        let mut polls = 0;
        loop {
            polls += 1;
            match event.data_u32(D3DGETDATA_FLUSH) {
                (D3D_OK, signalled) => {
                    let latency = TscClock::now().saturating_sub(issued);
                    assert_eq!(
                        signalled, 1,
                        "an EVENT query that answers S_OK reports TRUE"
                    );
                    return Some(Answer { polls, latency });
                }
                (S_FALSE, _) => assert!(
                    started.elapsed() < POLL_LIMIT,
                    "the previous frame's EVENT query has not answered after {POLL_LIMIT:?}"
                ),
                (hr, _) => panic!("EVENT GetData(FLUSH): 0x{hr:08X}"),
            }
        }
    }

    /// One mesh draw scaled by `scale` at clip-space `(x, y, z)`.
    fn mesh(&self, scale: f32, x: f32, y: f32, z: f32) {
        let h = self.h;
        ok(
            h.set_vertex_shader_constant_f(4, &world_rows(scale, x, y, z)),
            "world",
        );
        ok(
            h.draw_indexed_primitive(
                D3DPT_TRIANGLELIST,
                0,
                0,
                self.vertex_count,
                0,
                self.triangles,
            ),
            "draw",
        );
    }
}

/// The query set of `frame`, by parity.
fn set_of(frame: u32) -> usize {
    usize::try_from(frame).expect("frame fits usize") % QUERY_SETS
}
