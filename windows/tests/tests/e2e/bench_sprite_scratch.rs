//! A sprite scratch texture rewritten between the draws that sample it, at a busy town's rate.
//!
//! Guards what the layer pays for a texture an application rewrites after
//! drawing with it in the same frame. The 2D game of issue #1114 draws each
//! character's sprite by locking a sprite-sized rectangle at `(0, 0)` of
//! one 512x512 A4R4G4B4 managed texture, writing it and drawing it, so
//! every sprite after a frame's first is an upload into a texture an
//! earlier draw of the frame sampled. The layer cannot write that texture
//! in place, since the upload runs ahead of the draws, so it renames it:
//! the next draws sample another version, and that version receives the
//! texels the upload leaves alone. The game's frames do this between 5 and
//! about 268 times, the most in a crowded town; this frame does it
//! [`SPRITES`] times, each sprite one of the sizes a GPU capture of the
//! game shows, between 10x20 and 129x112.
//!
//! The renames show in the frame time (the encoder thread builds them) and
//! in the layer's own counters, read from its `perf-kv` line: `tex_reorder`
//! counts the renames, `tex_destroy` the textures destroyed, and
//! `enc_op` and `gpu_upload` the encoder's and the upload command buffer's
//! time. A rename that reuses a retired version destroys nothing, and one
//! that copies only what changed since that version retired copies a corner
//! of the level rather than all of it.

use std::time::SystemTime;

use mtld3d_tests::{Harness, HarnessConfig, MemorySample, Texture, TexturedVertex, VertexBuffer};
use mtld3d_types::{
    D3DCLEAR_TARGET, D3DCULL_NONE, D3DFMT_A4R4G4B4, D3DFVF_DIFFUSE, D3DFVF_TEX1, D3DFVF_XYZ,
    D3DPOOL_MANAGED, D3DPRESENT_INTERVAL_IMMEDIATE, D3DPT_TRIANGLELIST, D3DRS_CULLMODE,
    D3DRS_LIGHTING, D3DRS_ZENABLE, D3DTA_TEXTURE, D3DTOP_SELECTARG1, D3DTSS_COLORARG1,
    D3DTSS_COLOROP, D3DUSAGE_WRITEONLY,
};

use crate::bench::{
    Class, Direction, FrameClock, FrameWork, LayerLog, Metrics, STRIDE, TscClock, Value,
    memory_section, ok, ratio, write_report,
};

/// The back buffer, about the size of a windowed game.
const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
/// Edge of the sprite scratch texture.
const SCRATCH_EDGE: u32 = 512;
/// Sprites written and drawn per frame: the most a frame of a crowded town showed.
const SPRITES: u32 = 268;
/// Sprite sizes, cycled through: those of the uploads a GPU capture of the game recorded.
const SIZES: [(u32, u32); 12] = [
    (109, 112),
    (45, 36),
    (79, 75),
    (11, 20),
    (61, 23),
    (25, 28),
    (23, 30),
    (129, 112),
    (10, 20),
    (64, 48),
    (96, 80),
    (32, 40),
];
/// Screen slots the sprites are drawn into, a 20x14 grid of quads.
const COLUMNS: u32 = 20;
const ROWS: u32 = 14;
const SLOTS: u32 = COLUMNS * ROWS;
/// Bytes of fill pattern, enough for the largest sprite at the largest offset.
const PATTERN_BYTES: usize = 64 * 1024;
const WARM_UP_FRAMES: u32 = 60;
/// The measured phase is at least this many frames and at least one perf window long.
const MEASURED_FRAMES: usize = 300;
const FVF: u32 = D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1;

const _: () = assert!(SPRITES <= SLOTS, "one slot per sprite");

/// Rewrite a sprite scratch between the draws that sample it, every frame: warm up, time.
#[test]
#[ignore = "benchmark, run by `make bench`"]
fn sprite_scratch() {
    // Before the device: its creation logs, so the layer log is written after
    // this mark whatever the benchmark's warm-up logs.
    let tsc = TscClock::calibrated();
    let since = SystemTime::now();
    // Before the interface: what this benchmark adds to the address space
    // is measured from here, whatever an earlier one in the process left.
    let before = MemorySample::now();
    let h = Harness::create(&HarnessConfig {
        width: WIDTH,
        height: HEIGHT,
        presentation_interval: D3DPRESENT_INTERVAL_IMMEDIATE,
        ..HarnessConfig::default()
    });
    let started = TscClock::now();
    let mut scene = Scene::new(&h);
    for _ in 0..WARM_UP_FRAMES {
        assert!(h.pump(), "WM_QUIT during warm-up");
        scene.render();
        ok(h.present(), "Present");
    }
    let log = LayerLog::find(since);
    let warm_up = TscClock::since(started);
    let warm = MemorySample::now();

    let start = log.start_span(started, || {
        assert!(h.pump(), "WM_QUIT outside the measured frames");
        scene.render();
        ok(h.present(), "Present");
    });
    let mut clock = FrameClock::start(MEASURED_FRAMES * 4);
    while clock.frames() < MEASURED_FRAMES || clock.elapsed() < start.length() {
        assert!(h.pump(), "WM_QUIT during the measured frames");
        scene.render();
        clock.present(&h);
    }
    let end = MemorySample::now();
    let span = start.end(&log, || {
        assert!(h.pump(), "WM_QUIT outside the measured frames");
        scene.render();
        ok(h.present(), "Present");
    });

    let stats = clock.stats();
    let work = clock.work_stats();
    let body = format!(
        "shape: back buffer {WIDTH}x{HEIGHT} X8R8G8B8, fixed-function textured quads; one \
         {SCRATCH_EDGE}x{SCRATCH_EDGE} A4R4G4B4 managed texture\n\
         per frame: {SPRITES} sprites, each a LockRect of a rectangle at (0, 0) of one of \
         {sizes} sizes from 10x20 to 129x112, its rows written, UnlockRect, and one draw \
         sampling the texture, so every sprite after the first renames it\n\
         warm-up: {WARM_UP_FRAMES} frames in {warm_up:.2?}\n\
         measured: {frames} frames in {elapsed:.2?} (at least {MEASURED_FRAMES} frames \
         and {length:?}, {start})\n\
         frame time (Present to Present): {row}\n\
         API work (Present return to Present call): {work_row}\n{memory}{perf}",
        sizes = SIZES.len(),
        frames = stats.frames,
        elapsed = clock.elapsed(),
        row = stats.row(),
        work_row = work.row(),
        memory = memory_section(&before, &warm, &end),
        start = span.start(),
        length = span.length(),
        perf = span.perf_rows(&log).section(),
    );
    let mut metrics = Metrics::new("sprite_scratch", &h, &tsc);
    metrics.frame_rows("frame", &stats);
    metrics.frame_rows("api", &work);
    metrics.metric(
        "measured.frames",
        Value::Count(u64::try_from(stats.frames).expect("count fits u64")),
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
    metrics.memory(&before, &warm, &end);
    metrics.meta("window_s", &span.window_s());
    // Every frame writes the same sprites in the same order.
    metrics.perf(&span.perf_kv(&log), &FrameWork::Fixed);
    write_report(&metrics, &log, &body);
}

/// The scratch texture, the quads the sprites are drawn into and the bytes they are filled with.
struct Scene<'h> {
    h: &'h Harness,
    scratch: Texture<'h>,
    /// One quad per screen slot, six vertices each; kept alive for the stream binding.
    quads: VertexBuffer<'h>,
    pattern: Vec<u8>,
    /// Sprites written so far, which picks each one's fill.
    written: usize,
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
        let scene = Self {
            h,
            scratch: h.create_texture(
                SCRATCH_EDGE,
                SCRATCH_EDGE,
                1,
                0,
                D3DFMT_A4R4G4B4,
                D3DPOOL_MANAGED,
            ),
            quads,
            pattern,
            written: 0,
        };
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
        ok(h.set_texture(0, &scene.scratch), "SetTexture");
        scene
    }

    /// One frame up to its `Present`: every sprite written at `(0, 0)`, then drawn.
    fn render(&mut self) {
        let h = self.h;
        ok(h.begin_scene(), "BeginScene");
        ok(h.clear(D3DCLEAR_TARGET, 0xFF10_1820, 1.0, 0), "clear");
        for sprite in 0..SPRITES {
            let (width, height) = SIZES[self.written % SIZES.len()];
            let row_bytes = width as usize * 2;
            let bytes = row_bytes * height as usize;
            let offset = (self.written * 4099) % (PATTERN_BYTES - bytes);
            let rect = [
                0,
                0,
                i32::try_from(width).expect("sprite width fits i32"),
                i32::try_from(height).expect("sprite height fits i32"),
            ];
            let mut lock = self.scratch.lock_rect_partial(0, &rect, 0);
            lock.write_u8_rect(
                row_bytes,
                height as usize,
                &self.pattern[offset..offset + bytes],
            );
            ok(lock.unlock(), "sprite UnlockRect");
            ok(
                h.draw_primitive(D3DPT_TRIANGLELIST, sprite * 6, 2),
                "DrawPrimitive",
            );
            self.written += 1;
        }
        ok(h.end_scene(), "EndScene");
    }
}

/// Slot `at`'s quad, sampling the scratch's top-left 128 texels, where the sprites land.
fn slot_quad(at: u32) -> [TexturedVertex; 6] {
    let x = ratio(at % COLUMNS, COLUMNS).mul_add(2.0, -1.0);
    let y = ratio(at / COLUMNS, ROWS).mul_add(-2.0, 1.0);
    let (w, h) = (2.0 / ratio(COLUMNS, 1), 2.0 / ratio(ROWS, 1));
    let span = ratio(128, SCRATCH_EDGE);
    let corner = |dx: f32, dy: f32| TexturedVertex {
        x: dx.mul_add(w, x),
        y: dy.mul_add(-h, y),
        z: 0.5,
        color: 0xFFFF_FFFF,
        u: dx * span,
        v: dy * span,
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
