//! Texture streaming at World of Warcraft 1.12's zone-load rates, timed per `LockRect`.
//!
//! Guards three things: single `LockRect` calls that take far longer than
//! the rest, the stall a frame shows when one does (issue #340), the upload
//! paths a managed texture's levels, a `D3DPOOL_SYSTEMMEM` to
//! `D3DPOOL_DEFAULT` `UpdateTexture` and a whole-level lock of a texture the
//! frame already drew with go through, and how the 32-bit address space
//! fares while thousands of textures are created and released. On i686 the
//! largest free region at the end is the number to watch: a game loading
//! zones dies when no region is large enough for the next allocation, long
//! before committed memory runs out.
//!
//! The rates are the game's, read from the layer's `PERF=1` summary: 1 to 5
//! `LockRect` a frame in the open world, one of them without flags on a
//! texture the GPU is still using, and 17 to 23 a frame during a zone load
//! while the live texture set grows from about 200 to about 4400. Here a
//! warm-up first loads [`LIVE_TEXTURES`] textures, [`LOAD_PER_FRAME`] a
//! frame, then every frame creates [`NEW_PER_FRAME`] managed textures of a
//! twelve-step mix of DXT1, DXT3 and A8R8G8B8 from 64 to 512 texels with
//! full mip chains, locks and fills every level (about 16 `LockRect`),
//! draws once with each, and releases the oldest to keep the live set at
//! [`LIVE_TEXTURES`]. It then issues an event query and asks for it once
//! with `D3DGETDATA_FLUSH`, redraws [`REVISITS`] of the live textures,
//! rewrites level 0 of two system-memory textures and copies each to its
//! default-pool twin with `UpdateTexture`, and locks the whole of level 0
//! of a managed atlas the frame drew with first, without flags, so the lock
//! always finds its upload in flight and renames it with a preserving copy.
//!
//! The test times each `CreateTexture` and each `LockRect` call alone with
//! the benchmarks' `rdtsc` clock (`TscClock`), so those rows need no
//! `PERF=1` build. One such call is short enough that its single time moves
//! with the machine, so each kind is compared by its mean time per call
//! over whole frames (see `CallTimes`), with the slowest single
//! call reported beside it; a spike is a single `LockRect` over twice that
//! median and over 50 us, since the stalls a game shows take milliseconds.

use std::{
    collections::VecDeque,
    time::{Duration, SystemTime},
};

use mtld3d_tests::{
    Harness, HarnessConfig, MemorySample, Query, Texture, TexturedVertex, VertexBuffer,
};
use mtld3d_types::{
    D3D_OK, D3DCLEAR_TARGET, D3DCULL_NONE, D3DFMT_A8R8G8B8, D3DFMT_DXT1, D3DFMT_DXT3,
    D3DFVF_DIFFUSE, D3DFVF_TEX1, D3DFVF_XYZ, D3DGETDATA_FLUSH, D3DISSUE_END, D3DPOOL_DEFAULT,
    D3DPOOL_MANAGED, D3DPOOL_SYSTEMMEM, D3DPRESENT_INTERVAL_IMMEDIATE, D3DPT_TRIANGLELIST,
    D3DQUERYTYPE_EVENT, D3DRS_CULLMODE, D3DRS_LIGHTING, D3DRS_ZENABLE, D3DTA_TEXTURE,
    D3DTOP_SELECTARG1, D3DTSS_COLORARG1, D3DTSS_COLOROP, D3DUSAGE_WRITEONLY, S_FALSE,
};

use crate::bench::{
    CallTimes, Class, Direction, FrameClock, FrameWork, LayerLog, Metrics, STRIDE, TscClock, Value,
    memory_section, nanos, ok, ratio, write_report,
};

/// The back buffer, about the size of a windowed game.
const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
/// Textures kept alive once the warm-up has loaded them.
const LIVE_TEXTURES: usize = 2000;
/// Textures the warm-up creates, fills and draws per frame until the live set is full.
const LOAD_PER_FRAME: usize = 100;
/// Textures each measured frame creates, and so releases.
const NEW_PER_FRAME: u32 = 2;
/// Live textures each frame draws again, walking the set.
const REVISITS: u32 = 32;
/// The formats and edges new textures cycle through.
const KINDS: [(u32, u32); 12] = [
    (D3DFMT_DXT1, 256),
    (D3DFMT_DXT1, 64),
    (D3DFMT_A8R8G8B8, 64),
    (D3DFMT_DXT3, 128),
    (D3DFMT_DXT1, 128),
    (D3DFMT_DXT1, 512),
    (D3DFMT_A8R8G8B8, 128),
    (D3DFMT_DXT3, 64),
    (D3DFMT_DXT1, 256),
    (D3DFMT_DXT3, 256),
    (D3DFMT_A8R8G8B8, 64),
    (D3DFMT_DXT1, 128),
];
/// The system-memory sources `UpdateTexture` copies to default-pool twins each frame.
const UPDATES: [(u32, u32); 2] = [(D3DFMT_A8R8G8B8, 128), (D3DFMT_DXT1, 256)];
/// Edge of the managed A8R8G8B8 atlas locked whole every frame.
const ATLAS_EDGE: u32 = 256;
/// Rows of the atlas each whole-level lock rewrites.
const ATLAS_ROWS: usize = 16;
/// Screen slots the draws cycle through, an 8x8 grid of quads.
const SLOTS: u32 = 64;
/// Bytes of fill pattern, enough for the largest level at the largest offset.
const PATTERN_BYTES: usize = 320 * 1024;
const WARM_UP_FRAMES: u32 = 60;
/// The measured phase is at least this many frames and at least [`MIN_MEASURED`] long.
const MEASURED_FRAMES: usize = 600;
const MIN_MEASURED: Duration = Duration::from_secs(12);
const FVF: u32 = D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1;

/// Create, fill, draw and release textures every frame against a full live set: warm up, time.
#[test]
#[ignore = "benchmark, run by `make bench`"]
fn texture_streaming() {
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
    while scene.live.len() < LIVE_TEXTURES {
        assert!(h.pump(), "WM_QUIT during the load");
        scene.load();
        ok(h.present(), "Present");
    }
    for tick in 0..WARM_UP_FRAMES {
        assert!(h.pump(), "WM_QUIT during warm-up");
        scene.render(tick);
        ok(h.present(), "Present");
    }
    let log = LayerLog::find(since);
    let warm_up = TscClock::since(started);
    let warm = MemorySample::now();
    scene.times = Times::default();

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
    let (spikes, limit) = times.lock.spikes();
    let per_frame = |count: u64| {
        let count = u32::try_from(count).expect("count fits u32");
        let frames = u32::try_from(stats.frames).expect("frame count fits u32");
        f64::from(count) / f64::from(frames)
    };
    let body = format!(
        "shape: back buffer {WIDTH}x{HEIGHT} X8R8G8B8, fixed-function textured quads; \
         {LIVE_TEXTURES} live managed textures from a {kinds}-step mix of DXT1, DXT3 and \
         A8R8G8B8, 64 to 512 texels, full mip chains\n\
         per frame: {NEW_PER_FRAME} textures created, every level locked and filled, drawn \
         once and the oldest released; an event query read once with D3DGETDATA_FLUSH; \
         {REVISITS} live textures redrawn; {updates} system-memory level-0 rewrites, each \
         copied by UpdateTexture and drawn; one whole-level lock without flags of an atlas \
         drawn earlier in the frame\n\
         warm-up: {LIVE_TEXTURES} textures loaded {LOAD_PER_FRAME} a frame, then \
         {WARM_UP_FRAMES} frames, in {warm_up:.2?}\n\
         measured: {frames} frames in {elapsed:.2?} (at least {MEASURED_FRAMES} frames \
         and {MIN_MEASURED:?}), {locks_per_frame:.1} timed LockRect a frame, event query \
         complete at its one read on {complete} frames\n\
         frame time (Present to Present): {row}\n\
         API work (Present return to Present call): {work_row}\n\
         LockRect of a new texture's level (the call alone): {lock_row}\n\
         single LockRect calls over 2x the median and 50 us ({limit} ns): {spikes}\n\
         whole-level LockRect of the in-use atlas (the call alone): {preserve_row}\n\
         CreateTexture: {create_row}\n{memory}{perf}",
        kinds = KINDS.len(),
        updates = UPDATES.len(),
        frames = stats.frames,
        elapsed = clock.elapsed(),
        locks_per_frame = per_frame(times.lock.calls()),
        complete = times.complete,
        row = stats.row(),
        work_row = work.row(),
        lock_row = times.lock.row(),
        limit = limit.as_nanos(),
        preserve_row = times.preserve.row(),
        create_row = times.create.row(),
        memory = memory_section(&warm, &end),
        perf = log.perf_rows(from, to).section(),
    );
    let mut metrics = Metrics::new("streaming", &h, &tsc);
    metrics.frame_rows("frame", &stats);
    metrics.frame_rows("api", &work);
    metrics.call_rows("lockrect", &times.lock);
    metrics.metric(
        "lockrect.spikes",
        Value::Count(u64::try_from(spikes).expect("count fits u64")),
        Direction::Lower,
        Class::Spikes,
    );
    metrics.metric(
        "lockrect.spike_limit",
        Value::Ns(nanos(limit)),
        Direction::Lower,
        Class::Info,
    );
    metrics.call_rows("lockrect.preserve", &times.preserve);
    metrics.call_rows("create", &times.create);
    for (name, value) in [
        ("lockrect.count", times.lock.calls()),
        (
            "live_textures",
            u64::try_from(scene.live.len()).expect("count fits u64"),
        ),
        (
            "measured.frames",
            u64::try_from(stats.frames).expect("count fits u64"),
        ),
    ] {
        metrics.metric(name, Value::Count(value), Direction::Higher, Class::Info);
    }
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

/// The measured frames' timings.
#[derive(Default)]
struct Times {
    /// Every `LockRect` call on a new texture's levels, the call alone.
    lock: CallTimes,
    /// Every whole-level `LockRect` of the in-use atlas, the call alone.
    preserve: CallTimes,
    /// Every `CreateTexture` call.
    create: CallTimes,
    /// Frames whose event query was complete at its one `D3DGETDATA_FLUSH` read.
    complete: usize,
}

impl Times {
    /// End a frame for every kind of call.
    fn end_frame(&mut self) {
        for calls in [&mut self.lock, &mut self.preserve, &mut self.create] {
            calls.end_frame();
        }
    }
}

/// A system-memory texture and the default-pool twin `UpdateTexture` copies it to.
struct Update<'h> {
    format: u32,
    edge: u32,
    source: Texture<'h>,
    target: Texture<'h>,
}

/// Every resource the frame uses and the live set it streams through.
struct Scene<'h> {
    h: &'h Harness,
    /// Live managed textures, oldest first.
    live: VecDeque<Texture<'h>>,
    /// Textures created so far, which picks each new one's kind and fill.
    created: u32,
    updates: Vec<Update<'h>>,
    atlas: Texture<'h>,
    /// One quad per screen slot, six vertices each.
    quads: VertexBuffer<'h>,
    query: Query<'h>,
    pattern: Vec<u8>,
    times: Times,
}

impl<'h> Scene<'h> {
    fn new(h: &'h Harness) -> Self {
        let pattern = (0..PATTERN_BYTES)
            .map(|at| {
                let at = u32::try_from(at).expect("pattern offset fits u32");
                u8::try_from(at.wrapping_mul(0x9E37_79B1) >> 24).expect("top byte fits u8")
            })
            .collect();
        let quads =
            h.create_vertex_buffer(SLOTS * 6 * STRIDE, D3DUSAGE_WRITEONLY, FVF, D3DPOOL_MANAGED);
        let vertices: Vec<TexturedVertex> = (0..SLOTS).flat_map(slot_quad).collect();
        let mut lock = quads.lock(0, 0, 0);
        lock.write(&vertices);
        ok(lock.unlock(), "VertexBuffer Unlock");
        let mut scene = Self {
            h,
            live: VecDeque::with_capacity(LIVE_TEXTURES + 1),
            created: 0,
            updates: Vec::new(),
            atlas: h.create_texture(
                ATLAS_EDGE,
                ATLAS_EDGE,
                1,
                0,
                D3DFMT_A8R8G8B8,
                D3DPOOL_MANAGED,
            ),
            quads,
            query: h
                .create_query(D3DQUERYTYPE_EVENT)
                .expect("EVENT query is supported"),
            pattern,
            times: Times::default(),
        };
        fill(
            &scene.atlas,
            (D3DFMT_A8R8G8B8, ATLAS_EDGE),
            &scene.pattern,
            0,
            &mut CallTimes::default(),
        );
        for (format, edge) in UPDATES {
            let create = |pool| h.create_texture(edge, edge, 0, 0, format, pool);
            let update = Update {
                format,
                edge,
                source: create(D3DPOOL_SYSTEMMEM),
                target: create(D3DPOOL_DEFAULT),
            };
            fill(
                &update.source,
                (format, edge),
                &scene.pattern,
                1,
                &mut CallTimes::default(),
            );
            ok(
                h.update_texture_hr(&update.source, &update.target),
                "UpdateTexture",
            );
            scene.updates.push(update);
        }
        ok(h.set_fvf(FVF), "SetFVF");
        ok(h.set_render_state(D3DRS_LIGHTING, 0), "lighting off");
        ok(h.set_render_state(D3DRS_ZENABLE, 0), "depth off");
        ok(h.set_render_state(D3DRS_CULLMODE, D3DCULL_NONE), "cull off");
        ok(
            h.set_texture_stage_state(0, D3DTSS_COLOROP, D3DTOP_SELECTARG1),
            "colour op",
        );
        ok(
            h.set_texture_stage_state(0, D3DTSS_COLORARG1, D3DTA_TEXTURE),
            "colour arg",
        );
        ok(
            h.set_stream_source(0, &scene.quads, 0, STRIDE),
            "quad stream",
        );
        scene
    }

    /// One warm-up frame of the load: [`LOAD_PER_FRAME`] new textures, then the event query.
    fn load(&mut self) {
        let h = self.h;
        ok(h.begin_scene(), "BeginScene");
        ok(h.clear(D3DCLEAR_TARGET, 0xFF10_1820, 1.0, 0), "clear");
        for _ in 0..LOAD_PER_FRAME.min(LIVE_TEXTURES - self.live.len()) {
            self.stream_in();
        }
        self.flush_query();
        ok(h.end_scene(), "EndScene");
        self.times.end_frame();
    }

    /// One measured-shape frame up to its `Present`.
    fn render(&mut self, tick: u32) {
        let h = self.h;
        ok(h.begin_scene(), "BeginScene");
        ok(h.clear(D3DCLEAR_TARGET, 0xFF10_1820, 1.0, 0), "clear");
        // Drawn first: the draw uploads last frame's rewrite, so the lock
        // below finds that upload in flight.
        self.draw(&self.atlas, tick);
        for _ in 0..NEW_PER_FRAME {
            self.stream_in();
            if self.live.len() > LIVE_TEXTURES {
                self.live.pop_front();
            }
        }
        self.flush_query();
        let live = u32::try_from(self.live.len()).expect("live count fits u32");
        for visit in 0..REVISITS {
            let at = (tick * REVISITS + visit) % live;
            self.draw(&self.live[slot(at)], visit);
        }
        for (at, update) in (0..).zip(&self.updates) {
            let (row_bytes, rows) = level_rows(update.format, update.edge);
            let offset = pattern_offset(tick + at);
            let mut lock = update.source.lock_rect(0, 0);
            lock.write_u8_rect(
                row_bytes,
                rows,
                &self.pattern[offset..offset + row_bytes * rows],
            );
            ok(lock.unlock(), "system-memory UnlockRect");
            ok(
                h.update_texture_hr(&update.source, &update.target),
                "UpdateTexture",
            );
            self.draw(&update.target, tick + at);
        }
        let started = TscClock::now();
        let mut lock = self.atlas.lock_rect(0, 0);
        self.times.preserve.add(TscClock::since(started));
        let (row_bytes, _) = level_rows(D3DFMT_A8R8G8B8, ATLAS_EDGE);
        let offset = pattern_offset(tick);
        lock.write_u8_rect(
            row_bytes,
            ATLAS_ROWS,
            &self.pattern[offset..offset + row_bytes * ATLAS_ROWS],
        );
        ok(lock.unlock(), "atlas UnlockRect");
        ok(h.end_scene(), "EndScene");
        self.times.end_frame();
    }

    /// Create the next texture of the mix, fill every level, draw once with it, keep it live.
    fn stream_in(&mut self) {
        let h = self.h;
        let serial = self.created;
        self.created += 1;
        let kinds = u32::try_from(KINDS.len()).expect("kind count fits u32");
        let (format, edge) = KINDS[slot(serial % kinds)];
        let started = TscClock::now();
        let texture = h.create_texture(edge, edge, 0, 0, format, D3DPOOL_MANAGED);
        self.times.create.add(TscClock::since(started));
        fill(
            &texture,
            (format, edge),
            &self.pattern,
            serial,
            &mut self.times.lock,
        );
        self.draw(&texture, serial);
        self.live.push_back(texture);
    }

    /// One quad textured with `texture`, in the screen slot `n` picks.
    fn draw(&self, texture: &Texture<'_>, n: u32) {
        let h = self.h;
        ok(h.set_texture(0, texture), "SetTexture");
        ok(
            h.draw_primitive(D3DPT_TRIANGLELIST, (n % SLOTS) * 6, 2),
            "DrawPrimitive",
        );
    }

    /// Issue the event query and read it once with `D3DGETDATA_FLUSH`, as a loading game does.
    fn flush_query(&mut self) {
        ok(self.query.issue(D3DISSUE_END), "Issue(END)");
        let (hr, _) = self.query.data_u32(D3DGETDATA_FLUSH);
        assert!(hr == D3D_OK || hr == S_FALSE, "GetData(FLUSH): 0x{hr:08X}");
        self.times.complete += usize::from(hr == D3D_OK);
    }
}

/// Lock and fill every level of `texture`, of `(format, edge)`, from `pattern`.
///
/// Each `LockRect` call's time, the call alone, goes to `times`; `seed`
/// picks where in `pattern` the fill starts.
fn fill(
    texture: &Texture<'_>,
    (format, edge): (u32, u32),
    pattern: &[u8],
    seed: u32,
    times: &mut CallTimes,
) {
    let offset = pattern_offset(seed);
    for level in 0..texture.level_count() {
        let (row_bytes, rows) = level_rows(format, edge >> level);
        let started = TscClock::now();
        let mut lock = texture.lock_rect(level, 0);
        times.add(TscClock::since(started));
        lock.write_u8_rect(row_bytes, rows, &pattern[offset..offset + row_bytes * rows]);
        ok(lock.unlock(), "UnlockRect");
    }
}

/// Bytes in one row of `LockRect`'s layout, and the rows, of an `edge`-square level of `format`.
///
/// A block-compressed row is one row of 4x4 blocks, and a level smaller
/// than a block still holds one.
///
/// # Panics
/// Panics on a format the stream does not mix.
fn level_rows(format: u32, edge: u32) -> (usize, usize) {
    let edge = usize::try_from(edge.max(1)).expect("edge fits usize");
    let blocks = edge.div_ceil(4);
    match format {
        D3DFMT_DXT1 => (blocks * 8, blocks),
        D3DFMT_DXT3 => (blocks * 16, blocks),
        D3DFMT_A8R8G8B8 => (edge * 4, edge),
        other => panic!("the stream mixes no format {other:#x}"),
    }
}

/// The quad of screen slot `at`, one cell of an 8x8 grid in clip space, as six vertices.
fn slot_quad(at: u32) -> [TexturedVertex; 6] {
    let x = ratio(at % 8, 8).mul_add(2.0, -1.0);
    let y = ratio(at / 8, 8).mul_add(-2.0, 1.0);
    let corner = |dx: f32, dy: f32| TexturedVertex {
        x: dx.mul_add(0.24, x),
        y: dy.mul_add(-0.24, y),
        z: 0.5,
        color: 0xFFFF_FFFF,
        u: dx,
        v: dy,
    };
    let (top_left, top_right) = (corner(0.0, 0.0), corner(1.0, 0.0));
    let (bottom_left, bottom_right) = (corner(0.0, 1.0), corner(1.0, 1.0));
    [
        top_left,
        top_right,
        bottom_left,
        bottom_left,
        top_right,
        bottom_right,
    ]
}

/// Where in the fill pattern the texture or rewrite numbered `n` starts reading.
fn pattern_offset(n: u32) -> usize {
    slot(n % 16) * 4096
}

/// `value` as an index into one of the scene's lists.
fn slot(value: u32) -> usize {
    usize::try_from(value).expect("a list index fits usize")
}
