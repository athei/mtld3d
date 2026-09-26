//! Vertex and index buffer lock churn at World of Warcraft 1.12's rates, timed per lock.
//!
//! Guards the layer's dynamic-buffer paths against regressions in time and
//! address space: the in-place write a `D3DLOCK_NOOVERWRITE` append gets,
//! the rename a `D3DLOCK_DISCARD` at a ring's start gets, with the old
//! backing kept until the GPU retires the frames that read it, the rename
//! and synchronous copy of the old contents a whole-buffer lock without
//! `DISCARD` gets while the GPU may still read the buffer, the staging and
//! upload a static buffer's `Lock(0)` rewrite goes through, the rename of
//! its device buffer when a rewrite lands after a draw that read it in the
//! same frame, and the page-box pool all of those allocate from and give
//! back to.
//!
//! The rates are the game's, read from the layer's `PERF=1` summary over
//! busy frames: 460 to 980 vertex-buffer locks a frame, nearly all of them
//! `NOOVERWRITE` appends to dynamic rings with one `DISCARD` when a ring
//! wraps, about 350 index-buffer locks, about 0.8 buffer renames and 0.76
//! preserving copies of about 2.1 MB a frame. One frame here is about 720
//! vertex-buffer and 350 index-buffer locks: three dynamic vertex rings
//! (370 particle appends of 2 or 8 quads, 290 UI appends of 2 quads and 60
//! model appends of an 81-vertex grid), one dynamic index ring the UI and
//! the models append to, two static buffers rewritten whole and the first
//! of them rewritten and drawn a second time after its draw, and on three
//! frames of four one whole-buffer lock without flags of a 2.7 MiB dynamic
//! buffer the frame has already drawn from, so the lock always finds it in
//! use and renames it with a preserving copy. The rings wrap about once in
//! 16 to 20 frames each, so renames run near one a frame. Every lock is
//! followed by a draw of what it wrote.
//!
//! The test times each `Lock`, the copy into it and its `Unlock` with the
//! benchmarks' `rdtsc` clock (`TscClock`), so the lock rows need no
//! `PERF=1` build. One such call is short enough that its single time moves
//! with the machine, so each kind is compared by its mean time per lock
//! over whole frames (see `CallTimes`), with the slowest
//! single lock reported beside it. `lock.*` is the `NOOVERWRITE` appends
//! alone; every other kind has rows of its own.

use std::time::{Duration, SystemTime};

use mtld3d_tests::{
    Harness, HarnessConfig, IndexBuffer, MemorySample, TexturedVertex, VertexBuffer,
};
use mtld3d_types::{
    D3DCLEAR_TARGET, D3DCULL_NONE, D3DFMT_INDEX16, D3DFVF_DIFFUSE, D3DFVF_TEX1, D3DFVF_XYZ,
    D3DLOCK_DISCARD, D3DLOCK_NOOVERWRITE, D3DPOOL_DEFAULT, D3DPOOL_MANAGED,
    D3DPRESENT_INTERVAL_IMMEDIATE, D3DPT_TRIANGLELIST, D3DRS_CULLMODE, D3DRS_LIGHTING,
    D3DRS_ZENABLE, D3DTA_DIFFUSE, D3DTOP_SELECTARG1, D3DTSS_COLORARG1, D3DTSS_COLOROP,
    D3DUSAGE_DYNAMIC, D3DUSAGE_WRITEONLY,
};

use crate::bench::{
    CallTimes, Class, Direction, FrameClock, FrameWork, LayerLog, Metrics, STRIDE, TscClock, Value,
    grid, memory_section, ok, ratio, write_report,
};

/// The back buffer, about the size of a windowed game.
const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
/// Particle appends a frame, alternately [`PARTICLE_SMALL`] and [`PARTICLE_LARGE`] quads.
const PARTICLE_LOCKS: u32 = 370;
const PARTICLE_SMALL: u32 = 2;
const PARTICLE_LARGE: u32 = 8;
/// UI appends a frame, [`UI_QUADS`] quads each, with their indices in the index ring.
const UI_LOCKS: u32 = 290;
const UI_QUADS: u32 = 2;
/// Model appends a frame, one [`MODEL_GRID`]-square grid each, with its indices.
const MODEL_LOCKS: u32 = 60;
const MODEL_GRID: u16 = 8;
/// Vertices each dynamic vertex ring holds, sized so each wraps about once in 16 to 20 frames.
const PARTICLE_RING: u32 = 174_000;
const UI_RING: u32 = 43_680;
const MODEL_RING: u32 = 87_360;
/// 16-bit indices the dynamic index ring holds.
const INDEX_RING: u32 = 524_288;
/// Quads in each static buffer, rewritten whole every frame.
const STATIC_QUADS: u32 = 340;
/// The whole-buffer-locked dynamic buffer: [`CACHE_CHUNKS`] chunks of [`CHUNK_QUADS`] quads.
const CACHE_CHUNKS: u32 = 44;
const CHUNK_QUADS: u32 = 448;
/// Distinct contents each kind of write cycles through.
const VARIANTS: u32 = 8;
const WARM_UP_FRAMES: u32 = 60;
/// The measured phase is at least this many frames and at least [`MIN_MEASURED`] long.
const MEASURED_FRAMES: usize = 600;
const MIN_MEASURED: Duration = Duration::from_secs(12);
const FVF: u32 = D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1;

/// Dynamic ring appends, wraps, static rewrites and preserving locks, repeatedly: warm up, time.
#[test]
#[ignore = "benchmark, run by `make bench`"]
fn dynamic_buffer_churn() {
    let h = Harness::create(&HarnessConfig {
        width: WIDTH,
        height: HEIGHT,
        presentation_interval: D3DPRESENT_INTERVAL_IMMEDIATE,
        ..HarnessConfig::default()
    });
    let tsc = TscClock::calibrated();
    let since = SystemTime::now();
    let started = TscClock::now();
    let mut scene = Scene::new(&h);
    for tick in 0..WARM_UP_FRAMES {
        assert!(h.pump(), "WM_QUIT during warm-up");
        scene.render(tick);
        ok(h.present(), "Present");
    }
    let log = LayerLog::find(since);
    let warm_up = TscClock::since(started);
    let warm = MemorySample::now();
    scene.times = LockTimes::default();

    let from = log.mark();
    let mut clock = FrameClock::start(MEASURED_FRAMES * 4);
    let mut tick = WARM_UP_FRAMES;
    while clock.frames() < MEASURED_FRAMES || clock.elapsed() < MIN_MEASURED {
        assert!(h.pump(), "WM_QUIT during the measured frames");
        scene.render(tick);
        clock.present(&h);
        tick += 1;
    }
    let to = log.mark();
    let end = MemorySample::now();

    let stats = clock.stats();
    let work = clock.work_stats();
    let times = &scene.times;
    let body = format!(
        "shape: back buffer {WIDTH}x{HEIGHT} X8R8G8B8, fixed-function quads\n\
         per frame: {vb} vertex-buffer locks ({PARTICLE_LOCKS} particle, {UI_LOCKS} UI and \
         {MODEL_LOCKS} model appends, 2 static rewrites, a second rewrite of one after its \
         draw, a preserving whole-buffer lock on 3 frames of 4) and {ib} index-buffer \
         appends, each followed by a draw of what it wrote\n\
         rings: particle {particle} KiB, UI {ui} KiB, model {model} KiB, index {index} KiB; \
         whole-buffer-locked dynamic buffer {cache} KiB, static buffers {statics} KiB each\n\
         warm-up: {WARM_UP_FRAMES} frames in {warm_up:.2?}\n\
         measured: {frames} frames in {elapsed:.2?} (at least {MEASURED_FRAMES} frames \
         and {MIN_MEASURED:?}), {wraps} ring wraps, {preserves} preserving locks\n\
         frame time (Present to Present): {row}\n\
         API work (Present return to Present call): {work_row}\n\
         NOOVERWRITE append (Lock, copy, Unlock): {append_row}\n\
         DISCARD at a ring's start: {wrap_row}\n\
         whole-buffer lock of a buffer in use: {preserve_row}\n\
         static Lock(0) rewrite: {rewrite_row}\n\
         static rewrite after a draw of it in the frame: {overlap_row}\n{memory}{perf}",
        vb = PARTICLE_LOCKS + UI_LOCKS + MODEL_LOCKS + 3,
        ib = UI_LOCKS + MODEL_LOCKS,
        particle = PARTICLE_RING * STRIDE / 1024,
        ui = UI_RING * STRIDE / 1024,
        model = MODEL_RING * STRIDE / 1024,
        index = INDEX_RING * 2 / 1024,
        cache = CACHE_CHUNKS * CHUNK_QUADS * 6 * STRIDE / 1024,
        statics = STATIC_QUADS * 6 * STRIDE / 1024,
        frames = stats.frames,
        elapsed = clock.elapsed(),
        wraps = times.wrap.calls(),
        preserves = times.preserve.calls(),
        row = stats.row(),
        work_row = work.row(),
        append_row = times.append.row(),
        wrap_row = times.wrap.row(),
        preserve_row = times.preserve.row(),
        rewrite_row = times.rewrite.row(),
        overlap_row = times.overlap.row(),
        memory = memory_section(&warm, &end),
        perf = log.perf_rows(from, to).section(),
    );
    let mut metrics = Metrics::new("buffers", &h, &tsc);
    metrics.frame_rows("frame", &stats);
    metrics.frame_rows("api", &work);
    metrics.call_rows("lock", &times.append);
    metrics.call_rows("lock.discard", &times.wrap);
    metrics.call_rows("lock.preserve", &times.preserve);
    metrics.call_rows("lock.static", &times.rewrite);
    metrics.call_rows("lock.overlap", &times.overlap);
    // The test's own lock counts, fixed by the frame count: context, not a comparison.
    for (name, value) in [
        ("locks.discard", times.wrap.calls()),
        ("locks.preserve", times.preserve.calls()),
    ] {
        metrics.metric(name, Value::Count(value), Direction::Lower, Class::Info);
    }
    metrics.metric(
        "measured.frames",
        Value::Count(u64::try_from(stats.frames).expect("frame count fits u64")),
        Direction::Higher,
        Class::Info,
    );
    metrics.metric(
        "warmup.ms",
        Value::Ms(warm_up),
        Direction::Lower,
        Class::Info,
    );
    metrics.metric(
        "measured.ms",
        Value::Ms(clock.elapsed()),
        Direction::Lower,
        Class::Info,
    );
    metrics.memory(&warm, &end);
    // The frames lock the same kinds at the same rates, but a ring's wrap or
    // a preserving lock falls on some frames and not others, so a window's
    // per-frame counts depend on where it starts.
    metrics.perf(&log.perf_kv_last_full(from, to), &FrameWork::Varying);
    write_report(&metrics, &log, &body);
}

/// The time of every lock the measured frames take, by kind.
#[derive(Default)]
struct LockTimes {
    /// `NOOVERWRITE` appends to the rings, vertex and index.
    append: CallTimes,
    /// `DISCARD` locks at a ring's start, one per wrap.
    wrap: CallTimes,
    /// Whole-buffer locks without flags of the buffer the frame already drew from.
    preserve: CallTimes,
    /// Whole-buffer `Lock(0)` rewrites of the static buffers.
    rewrite: CallTimes,
    /// The second rewrite of a static buffer, after a draw of it in the same frame.
    overlap: CallTimes,
}

impl LockTimes {
    /// Count a ring lock: a wrap when it discarded, an append otherwise.
    fn ring(&mut self, flags: u32, took: Duration) {
        if flags == D3DLOCK_DISCARD {
            self.wrap.add(took);
        } else {
            self.append.add(took);
        }
    }

    /// End a frame for every kind.
    fn end_frame(&mut self) {
        for calls in [
            &mut self.append,
            &mut self.wrap,
            &mut self.preserve,
            &mut self.rewrite,
            &mut self.overlap,
        ] {
            calls.end_frame();
        }
    }
}

/// A dynamic vertex buffer appended to until full, then discarded and started again.
struct VertexRing<'h> {
    vb: VertexBuffer<'h>,
    /// Vertices the buffer holds.
    capacity: u32,
    cursor: u32,
}

impl<'h> VertexRing<'h> {
    fn new(h: &'h Harness, capacity: u32) -> Self {
        Self {
            vb: h.create_vertex_buffer(
                capacity * STRIDE,
                D3DUSAGE_DYNAMIC | D3DUSAGE_WRITEONLY,
                FVF,
                D3DPOOL_DEFAULT,
            ),
            capacity,
            cursor: 0,
        }
    }

    /// Write `data` after what the ring holds, or at its start when it does not fit.
    ///
    /// Returns the first vertex written, the lock flags and the lock's time.
    fn append(&mut self, data: &[TexturedVertex]) -> (u32, u32, Duration) {
        let count = u32::try_from(data.len()).expect("vertex count fits u32");
        let flags = if self.cursor + count > self.capacity {
            self.cursor = 0;
            D3DLOCK_DISCARD
        } else {
            D3DLOCK_NOOVERWRITE
        };
        let first = self.cursor;
        self.cursor += count;
        let took = write_vertices(&self.vb, first * STRIDE, count * STRIDE, data, flags);
        (first, flags, took)
    }
}

/// A dynamic index buffer of 16-bit indices, appended to like a [`VertexRing`].
struct IndexRing<'h> {
    ib: IndexBuffer<'h>,
    cursor: u32,
}

impl IndexRing<'_> {
    /// Write `data` after what the ring holds, or at its start when it does not fit.
    ///
    /// Returns the first index written, the lock flags and the lock's time.
    fn append(&mut self, data: &[u16]) -> (u32, u32, Duration) {
        let count = u32::try_from(data.len()).expect("index count fits u32");
        let flags = if self.cursor + count > INDEX_RING {
            self.cursor = 0;
            D3DLOCK_DISCARD
        } else {
            D3DLOCK_NOOVERWRITE
        };
        let first = self.cursor;
        self.cursor += count;
        let started = TscClock::now();
        let mut lock = self.ib.lock(first * 2, count * 2, flags);
        lock.write(data);
        let hr = lock.unlock();
        let took = TscClock::since(started);
        ok(hr, "IndexBuffer Unlock");
        (first, flags, took)
    }
}

/// The contents each kind of write cycles through, built once.
struct Contents {
    particle_small: Vec<Vec<TexturedVertex>>,
    particle_large: Vec<Vec<TexturedVertex>>,
    ui: Vec<Vec<TexturedVertex>>,
    ui_indices: Vec<u16>,
    models: Vec<Vec<TexturedVertex>>,
    model_indices: Vec<u16>,
    statics: Vec<Vec<TexturedVertex>>,
    chunks: Vec<Vec<TexturedVertex>>,
}

impl Contents {
    fn new() -> Self {
        let variants = |make: &dyn Fn(u32) -> Vec<TexturedVertex>| -> Vec<Vec<TexturedVertex>> {
            (0..VARIANTS).map(make).collect()
        };
        let (grid_vertices, model_indices) = grid(MODEL_GRID);
        let quads = u16::try_from(UI_QUADS).expect("UI quads fit u16");
        Self {
            particle_small: variants(&|seed| quad_list(PARTICLE_SMALL, seed, 0.01)),
            particle_large: variants(&|seed| quad_list(PARTICLE_LARGE, seed + 3, 0.02)),
            ui: variants(&|seed| quad_corners(UI_QUADS, seed + 5, 0.03)),
            ui_indices: (0..quads)
                .flat_map(|quad| {
                    let base = quad * 4;
                    [base, base + 1, base + 2, base + 2, base + 1, base + 3]
                })
                .collect(),
            models: variants(&|seed| {
                let x = ratio(seed * 23 % 100, 100).mul_add(1.6, -0.9);
                let y = ratio(seed * 41 % 100, 100).mul_add(1.6, -0.9);
                grid_vertices
                    .iter()
                    .map(|vertex| TexturedVertex {
                        x: vertex.x.mul_add(0.08, x),
                        y: vertex.y.mul_add(0.08, y),
                        z: 0.5,
                        color: 0xFF60_A040,
                        u: vertex.u,
                        v: vertex.v,
                    })
                    .collect()
            }),
            model_indices,
            statics: variants(&|seed| quad_list(STATIC_QUADS, seed + 7, 0.01)),
            chunks: (0..CACHE_CHUNKS)
                .map(|chunk| quad_list(CHUNK_QUADS, chunk * 3, 0.008))
                .collect(),
        }
    }
}

/// Every buffer the frame locks and draws, created once, and the lock times.
struct Scene<'h> {
    h: &'h Harness,
    particles: VertexRing<'h>,
    ui: VertexRing<'h>,
    models: VertexRing<'h>,
    indices: IndexRing<'h>,
    /// A managed and a default-pool static buffer, both rewritten whole every frame.
    statics: [VertexBuffer<'h>; 2],
    /// The dynamic buffer locked whole, without flags, on three frames of four.
    cache: VertexBuffer<'h>,
    contents: Contents,
    times: LockTimes,
}

impl<'h> Scene<'h> {
    fn new(h: &'h Harness) -> Self {
        let contents = Contents::new();
        let static_bytes = STATIC_QUADS * 6 * STRIDE;
        let statics = [D3DPOOL_MANAGED, D3DPOOL_DEFAULT]
            .map(|pool| h.create_vertex_buffer(static_bytes, D3DUSAGE_WRITEONLY, FVF, pool));
        let cache = h.create_vertex_buffer(
            CACHE_CHUNKS * CHUNK_QUADS * 6 * STRIDE,
            D3DUSAGE_DYNAMIC | D3DUSAGE_WRITEONLY,
            FVF,
            D3DPOOL_DEFAULT,
        );
        let whole: Vec<TexturedVertex> = contents.chunks.iter().flatten().copied().collect();
        write_vertices(&cache, 0, 0, &whole, D3DLOCK_DISCARD);
        ok(h.set_fvf(FVF), "SetFVF");
        ok(h.set_render_state(D3DRS_LIGHTING, 0), "lighting off");
        ok(h.set_render_state(D3DRS_ZENABLE, 0), "depth off");
        ok(h.set_render_state(D3DRS_CULLMODE, D3DCULL_NONE), "cull off");
        ok(
            h.set_texture_stage_state(0, D3DTSS_COLOROP, D3DTOP_SELECTARG1),
            "colour op",
        );
        ok(
            h.set_texture_stage_state(0, D3DTSS_COLORARG1, D3DTA_DIFFUSE),
            "colour arg",
        );
        Self {
            h,
            particles: VertexRing::new(h, PARTICLE_RING),
            ui: VertexRing::new(h, UI_RING),
            models: VertexRing::new(h, MODEL_RING),
            indices: IndexRing {
                ib: h.create_index_buffer(
                    INDEX_RING * 2,
                    D3DUSAGE_DYNAMIC | D3DUSAGE_WRITEONLY,
                    D3DFMT_INDEX16,
                    D3DPOOL_DEFAULT,
                ),
                cursor: 0,
            },
            statics,
            cache,
            contents,
            times: LockTimes::default(),
        }
    }

    /// One frame up to its `Present`.
    fn render(&mut self, tick: u32) {
        let h = self.h;
        ok(h.begin_scene(), "BeginScene");
        ok(h.clear(D3DCLEAR_TARGET, 0xFF10_1820, 1.0, 0), "clear");
        // Drawn first, so the whole-buffer lock later in the frame finds it in use.
        ok(
            h.set_stream_source(0, &self.cache, 0, STRIDE),
            "cache stream",
        );
        draw_list(h, (tick % CACHE_CHUNKS) * CHUNK_QUADS * 6, CHUNK_QUADS);
        self.draw_models(tick);
        self.draw_particles(tick);
        self.draw_ui(tick);
        self.rewrite_statics(tick);
        if tick % 4 != 3 {
            self.rewrite_cache(tick);
        }
        ok(h.end_scene(), "EndScene");
        self.times.end_frame();
    }

    /// The model appends: a grid into the model ring and its indices into the index ring.
    fn draw_models(&mut self, tick: u32) {
        let h = self.h;
        ok(
            h.set_stream_source(0, &self.models.vb, 0, STRIDE),
            "model stream",
        );
        ok(h.set_indices(&self.indices.ib), "SetIndices");
        let indices = &self.contents.model_indices;
        let triangles = u32::try_from(indices.len() / 3).expect("triangle count fits u32");
        for model in 0..MODEL_LOCKS {
            let data = &self.contents.models[variant(tick + model)];
            let (first, flags, took) = self.models.append(data);
            self.times.ring(flags, took);
            let (start, flags, took) = self.indices.append(indices);
            self.times.ring(flags, took);
            draw_indexed(h, first, data.len(), start, triangles);
        }
    }

    /// The particle appends, alternately small and large batches, drawn unindexed.
    fn draw_particles(&mut self, tick: u32) {
        let h = self.h;
        ok(
            h.set_stream_source(0, &self.particles.vb, 0, STRIDE),
            "particle stream",
        );
        for batch in 0..PARTICLE_LOCKS {
            let (set, quads) = if batch % 2 == 0 {
                (&self.contents.particle_small, PARTICLE_SMALL)
            } else {
                (&self.contents.particle_large, PARTICLE_LARGE)
            };
            let (first, flags, took) = self.particles.append(&set[variant(tick + batch)]);
            self.times.ring(flags, took);
            draw_list(h, first, quads);
        }
    }

    /// The UI appends: two quads into the UI ring and their indices into the index ring.
    fn draw_ui(&mut self, tick: u32) {
        let h = self.h;
        ok(h.set_stream_source(0, &self.ui.vb, 0, STRIDE), "UI stream");
        let indices = &self.contents.ui_indices;
        for batch in 0..UI_LOCKS {
            let data = &self.contents.ui[variant(tick + batch)];
            let (first, flags, took) = self.ui.append(data);
            self.times.ring(flags, took);
            let (start, flags, took) = self.indices.append(indices);
            self.times.ring(flags, took);
            draw_indexed(h, first, data.len(), start, UI_QUADS * 2);
        }
    }

    /// Rewrite both static buffers whole with `Lock(0)`, then draw each.
    ///
    /// The first is then rewritten and drawn again: its upload lands after
    /// a draw in the same frame that read the old contents.
    fn rewrite_statics(&mut self, tick: u32) {
        let h = self.h;
        for (at, vb) in (0..).zip(&self.statics) {
            let data = &self.contents.statics[variant(tick + at)];
            self.times.rewrite.add(write_vertices(vb, 0, 0, data, 0));
            ok(h.set_stream_source(0, vb, 0, STRIDE), "static stream");
            draw_list(h, 0, STATIC_QUADS);
        }
        let vb = &self.statics[0];
        let data = &self.contents.statics[variant(tick + 2)];
        self.times.overlap.add(write_vertices(vb, 0, 0, data, 0));
        ok(h.set_stream_source(0, vb, 0, STRIDE), "static stream");
        draw_list(h, 0, STATIC_QUADS);
    }

    /// Lock the in-use dynamic buffer whole without flags, rewrite its first chunk, draw it.
    fn rewrite_cache(&mut self, tick: u32) {
        let h = self.h;
        let data = &self.contents.chunks[slot(tick % CACHE_CHUNKS)];
        self.times
            .preserve
            .add(write_vertices(&self.cache, 0, 0, data, 0));
        ok(
            h.set_stream_source(0, &self.cache, 0, STRIDE),
            "cache stream",
        );
        draw_list(h, 0, CHUNK_QUADS);
    }
}

/// Lock `size` bytes of `vb` at `offset` with `flags`, copy `data` to the start, unlock, timed.
///
/// A `size` of 0 locks the whole buffer, as D3D9 defines it.
fn write_vertices(
    vb: &VertexBuffer<'_>,
    offset: u32,
    size: u32,
    data: &[TexturedVertex],
    flags: u32,
) -> Duration {
    let started = TscClock::now();
    let mut lock = vb.lock(offset, size, flags);
    lock.write(data);
    let hr = lock.unlock();
    let took = TscClock::since(started);
    ok(hr, "VertexBuffer Unlock");
    took
}

/// Draw `quads` quads of six vertices each from the bound stream, starting at vertex `first`.
fn draw_list(h: &Harness, first: u32, quads: u32) {
    ok(
        h.draw_primitive(D3DPT_TRIANGLELIST, first, quads * 2),
        "DrawPrimitive",
    );
}

/// Draw `triangles` from the bound index buffer at `start`, its vertices from `first` on.
fn draw_indexed(h: &Harness, first: u32, vertices: usize, start: u32, triangles: u32) {
    ok(
        h.draw_indexed_primitive(
            D3DPT_TRIANGLELIST,
            i32::try_from(first).expect("base vertex fits i32"),
            0,
            u32::try_from(vertices).expect("vertex count fits u32"),
            start,
            triangles,
        ),
        "DrawIndexedPrimitive",
    );
}

/// `count` squares of edge `size` as a triangle list, six vertices each, scattered by `seed`.
fn quad_list(count: u32, seed: u32, size: f32) -> Vec<TexturedVertex> {
    (0..count)
        .flat_map(|quad| {
            let [a, b, c, d] = corners(quad, seed, size);
            [a, b, c, c, b, d]
        })
        .collect()
}

/// The same squares as four corners each, for an index list of `0 1 2 2 1 3` per quad.
fn quad_corners(count: u32, seed: u32, size: f32) -> Vec<TexturedVertex> {
    (0..count)
        .flat_map(|quad| corners(quad, seed, size))
        .collect()
}

/// The corners of the `quad`-th square of edge `size` in clip space, placed by `seed`.
fn corners(quad: u32, seed: u32, size: f32) -> [TexturedVertex; 4] {
    let x = ratio((quad * 17 + seed * 5) % 100, 100).mul_add(1.8, -0.95);
    let y = ratio((quad * 29 + seed * 3) % 100, 100).mul_add(1.8, -0.95);
    let color = 0xC000_0000 | (seed * 0x0015_2B41 + quad * 0x0003_0507) & 0x00FF_FFFF;
    let corner = |dx: f32, dy: f32| TexturedVertex {
        x: dx.mul_add(size, x),
        y: dy.mul_add(-size, y),
        z: 0.5,
        color,
        u: dx,
        v: dy,
    };
    [
        corner(0.0, 0.0),
        corner(1.0, 0.0),
        corner(0.0, 1.0),
        corner(1.0, 1.0),
    ]
}

/// Which of the [`VARIANTS`] contents the write numbered `n` uses.
fn variant(n: u32) -> usize {
    slot(n % VARIANTS)
}

/// `value` as an index into one of the scene's lists.
fn slot(value: u32) -> usize {
    usize::try_from(value).expect("a list index fits usize")
}
