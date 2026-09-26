//! A synthetic frame shaped like World of Warcraft 1.12's busy frame, timed over many frames.
//!
//! The shape and the call rates come from the layer's `PERF=1` summary
//! windows over the game's busy frames and a Metal capture of one of them
//! (2026-09): 1072 to 1177 draws, exactly five passes, 10.6k to 12.2k D3D9
//! calls a frame. One frame here is 1104 draws in those five passes: the
//! scene, 860 draws into a full-size A8R8G8B8 render-target texture over the
//! device's D24S8 depth; three one-draw glow passes at a quarter of the size
//! on each axis, each sampling the previous target with four taps while the
//! full-size depth stays bound; and the back buffer, one composite of the
//! scene and the glow followed by 240 fixed-function UI quads with one or two
//! DXT3 textures each, then `Present`.
//!
//! What sets this frame apart from the 3.3.5a one in `bench_frame_shape.rs` is
//! the fixed-function share. 44 % of the scene uses the fixed vertex pipeline
//! and the fixed texture stages, lit, fogged and in five blend modes, some of
//! them alpha-tested; 38 % are terrain chunks, the fixed vertex pipeline
//! feeding a `ps_2_0` program that blends three layer textures through two
//! alpha maps; 13 % are `vs_2_0` models over the fixed texture stages; 5 % are
//! `vs_2_0` with `ps_2_0`. Per draw the binds follow the game's: about one
//! `SetTexture`, most of them rebinding what is bound, two transform, material
//! or light calls, 0.4 render-state and 0.7 texture-stage-state changes, a
//! redundant `SetFVF` or `SetVertexDeclaration`, and few shader changes. Per
//! frame, 590 vertex-buffer and 350 index-buffer locks append to two dynamic
//! rings with `D3DLOCK_NOOVERWRITE`, the first lock of each frame starting
//! the ring over with `D3DLOCK_DISCARD`; a managed UI texture the previous
//! frame still samples is locked whole without `D3DLOCK_DISCARD`, so the
//! layer preserves it; a dynamic one is locked with it; an EVENT query is
//! issued and polled with `D3DGETDATA_FLUSH`; and there are 8 render-target
//! and 8 depth-stencil binds. That comes to about 10.5k D3D9 calls a frame,
//! counting each `Lock` and `Unlock` pair as two calls.
//!
//! Every program and every fixed-function variant is first drawn in the
//! warm-up, so the measured frames compile nothing. The metrics file carries
//! one `shape` record per pass, computed from the constants below, and the
//! count of every kind of D3D9 call one measured frame made. The draw count
//! of that frame is checked against the shapes, which checks the benchmark
//! against itself; how many passes the layer makes of the frame is its
//! `PERF=1` summary's to say.

use core::{cell::Cell, ffi::c_void, fmt::Write as _};
use std::time::{Duration, Instant, SystemTime};

use mtld3d_tests::{
    Harness, HarnessConfig, IndexBuffer, MemorySample, PixelShader, Query, Surface, Texture,
    TexturedVertex, VertexBuffer, VertexDeclaration, VertexShader,
};
use mtld3d_types::{
    D3DBLEND_DESTCOLOR, D3DBLEND_INVSRCALPHA, D3DBLEND_ONE, D3DBLEND_SRCALPHA, D3DBLEND_ZERO,
    D3DCLEAR_STENCIL, D3DCLEAR_TARGET, D3DCLEAR_ZBUFFER, D3DCMP_GREATEREQUAL, D3DCMP_LESSEQUAL,
    D3DCOLORVALUE, D3DCULL_CCW, D3DCULL_NONE, D3DDECL_END, D3DDECLTYPE_FLOAT2, D3DDECLTYPE_FLOAT3,
    D3DDECLUSAGE_NORMAL, D3DDECLUSAGE_POSITION, D3DDECLUSAGE_TEXCOORD, D3DFMT_A8R8G8B8,
    D3DFMT_D24S8, D3DFMT_DXT3, D3DFMT_INDEX16, D3DFOG_LINEAR, D3DFOG_NONE, D3DFVF_DIFFUSE,
    D3DFVF_NORMAL, D3DFVF_TEX1, D3DFVF_TEXCOUNT_SHIFT, D3DFVF_XYZ, D3DGETDATA_FLUSH, D3DISSUE_END,
    D3DLIGHT_POINT, D3DLIGHT9, D3DLOCK_DISCARD, D3DLOCK_NOOVERWRITE, D3DMATERIAL9, D3DPOOL_DEFAULT,
    D3DPOOL_MANAGED, D3DPRESENT_INTERVAL_IMMEDIATE, D3DPT_TRIANGLELIST, D3DQUERYTYPE_EVENT,
    D3DRS_ALPHABLENDENABLE, D3DRS_ALPHAFUNC, D3DRS_ALPHAREF, D3DRS_ALPHATESTENABLE, D3DRS_AMBIENT,
    D3DRS_CULLMODE, D3DRS_DESTBLEND, D3DRS_FOGCOLOR, D3DRS_FOGENABLE, D3DRS_FOGEND, D3DRS_FOGSTART,
    D3DRS_FOGTABLEMODE, D3DRS_FOGVERTEXMODE, D3DRS_LIGHTING, D3DRS_NORMALIZENORMALS,
    D3DRS_SRCBLEND, D3DRS_ZENABLE, D3DRS_ZFUNC, D3DRS_ZWRITEENABLE, D3DSAMP_ADDRESSU,
    D3DSAMP_ADDRESSV, D3DSAMP_MAGFILTER, D3DSAMP_MINFILTER, D3DTA_CURRENT, D3DTA_DIFFUSE,
    D3DTA_TEXTURE, D3DTADDRESS_CLAMP, D3DTADDRESS_WRAP, D3DTEXF_LINEAR, D3DTOP_DISABLE,
    D3DTOP_MODULATE, D3DTOP_MODULATE2X, D3DTOP_SELECTARG1, D3DTOP_SELECTARG2, D3DTS_PROJECTION,
    D3DTS_VIEW, D3DTS_WORLD, D3DTSS_ALPHAARG1, D3DTSS_ALPHAARG2, D3DTSS_ALPHAOP, D3DTSS_COLORARG1,
    D3DTSS_COLORARG2, D3DTSS_COLOROP, D3DTSS_TEXCOORDINDEX, D3DUSAGE_DYNAMIC,
    D3DUSAGE_RENDERTARGET, D3DUSAGE_WRITEONLY, D3DVECTOR, D3DVERTEXELEMENT9, S_FALSE,
};

use crate::bench::{
    Class, Direction, FrameClock, FrameWork, IDENTITY_ROWS, LayerLog, Metrics, PassShape, STRIDE,
    TEXTURED_DECL, TscClock, Value, def, element, grid, memory_section, ok, pattern_texture, ratio,
    transform, world_rows, write_report,
};

/// The back buffer and the scene target, the size of a windowed game.
const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
/// The glow targets, a quarter of the scene on each axis, as the game's are.
const GLOW_WIDTH: u32 = WIDTH / 4;
const GLOW_HEIGHT: u32 = HEIGHT / 4;
/// The configuration the game runs with.
///
/// The built-in `wow` profile sets exactly these two keys, but it matches the
/// game's executable and version strings, never this benchmark's, so they are
/// passed here instead. They go after the suite-wide configuration and win
/// over a `BENCH_CONFIG` entry for the same key.
const GAME_CONFIG: &str = "query.flushImmediate=true;query.eventImmediate=true";

/// Terrain chunks, 38 % of the game's scene draws: fixed vertex pipeline, `ps_2_0` terrain.
const TERRAIN_DRAWS: u32 = 327;
/// Chunks in a run that shares one program, its three layers and its two alpha maps.
const TERRAIN_RUN: u32 = 16;
/// The runs that share one set of three layer textures before the next set.
const RUNS_PER_LAYER_SET: u32 = 4;
/// The textures a terrain draw samples: three 256x256 layers and two alpha maps.
const TERRAIN_TEXTURES: u32 = 5;
/// Fixed-function models, 44 % of the scene: lit, fogged, both fixed pipelines.
const FF_MODEL_DRAWS: u32 = 378;
/// The fixed-function draws at the head of each section that come from static buffers.
///
/// The rest write their vertices and indices into the dynamic rings first.
const FF_STATIC_PER_SECTION: u32 = 7;
/// Consecutive fixed-function draws that share a texture and a blend mode.
const MODEL_GROUP: u32 = 8;
/// `vs_2_0` models over the fixed texture stages, 13 % of the scene.
const CHARACTER_DRAWS: u32 = 112;
/// Consecutive `vs_2_0` model draws that share a texture.
const CHARACTER_GROUP: u32 = 7;
/// `vs_2_0` with `ps_2_0` (liquids), 5 % of the scene, in one run per program pair.
const LIQUID_DRAWS: u32 = 43;
const LIQUID_RUNS: u32 = 2;
/// Textures a liquid draw samples.
const LIQUID_TEXTURES: u32 = 2;
/// Fixed-function stretches of the scene, each followed by a run of `vs_2_0` models.
///
/// Interleaving them is what puts the game's rare `SetVertexShader` calls
/// (0.015 a draw) between the fixed-function and programmable models.
const SECTIONS: u32 = 4;
const SCENE_DRAWS: u32 = TERRAIN_DRAWS + FF_MODEL_DRAWS + CHARACTER_DRAWS + LIQUID_DRAWS;
/// The glow passes, one draw each.
const GLOW_PASSES: u32 = 3;
/// UI quads per frame, one draw each, all fixed function.
const UI_QUADS: u32 = 240;
/// Consecutive UI quads of one element, sharing its blend mode and stage operations.
const UI_ELEMENT: u32 = 10;
/// Every this many UI quads, one modulates a second texture on stage 1.
const UI_MASK_EVERY: u32 = 8;
/// The composite draws the scene and the glow.
const COMPOSITE_TEXTURES: u32 = 2;
/// Draws per frame: the scene, the glow passes, the composite and the UI.
const DRAWS_PER_FRAME: u32 = SCENE_DRAWS + GLOW_PASSES + 1 + UI_QUADS;

/// Terrain chunks per row of the screen, and the chunk copies in the terrain vertex buffer.
const TERRAIN_COLUMNS: u32 = 18;
const TILE_CHUNKS: u32 = 16;
/// Quads per edge of a terrain chunk's grid, and what that makes.
const TERRAIN_GRID: u32 = 8;
const TERRAIN_VERTS: u32 = (TERRAIN_GRID + 1) * (TERRAIN_GRID + 1);
const TERRAIN_TRIANGLES: u32 = TERRAIN_GRID * TERRAIN_GRID * 2;
/// Quads per edge of every model's grid, and what that makes.
const MODEL_GRID: u32 = 4;
const MODEL_VERTS: u32 = (MODEL_GRID + 1) * (MODEL_GRID + 1);
const MODEL_TRIANGLES: u32 = MODEL_GRID * MODEL_GRID * 2;
/// Depth of the terrain, behind every model.
const TERRAIN_DEPTH: f32 = 0.9;

/// Terrain vertex: position, normal, the layers' tiled coordinate and the alpha maps' one.
const TERRAIN_FVF: u32 = D3DFVF_XYZ | D3DFVF_NORMAL | (2 << D3DFVF_TEXCOUNT_SHIFT);
const TERRAIN_STRIDE: u32 = 40;
/// Model vertex: position, normal, one texture coordinate, for the fixed and `vs_2_0` models.
const MODEL_FVF: u32 = D3DFVF_XYZ | D3DFVF_NORMAL | D3DFVF_TEX1;
const MODEL_STRIDE: u32 = 32;
/// UI vertex: [`TexturedVertex`], [`STRIDE`] bytes.
const UI_FVF: u32 = D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1;
/// Bytes of the dynamic vertex ring, which holds a frame's appends with room to spare.
const RING_VERTEX_BYTES: u32 = 512 * 1024;
/// Bytes of the dynamic index ring.
const RING_INDEX_BYTES: u32 = 128 * 1024;

const LAYER_SETS: u32 = 2;
const LAYER_EDGE: u32 = 256;
const ALPHA_MAPS: u32 = 8;
const ALPHA_EDGE: u32 = 64;
const MODEL_TEXTURES: u32 = 16;
const UI_TEXTURES: u32 = 8;
/// Edge of the DXT3 UI textures.
const UI_EDGE: u32 = 64;
/// The managed font atlas, and the glyph cell of it rewritten every frame.
const FONT_EDGE: u32 = 256;
const GLYPH_EDGE: u32 = 16;
/// The dynamic minimap texture, rewritten whole every frame.
const MINIMAP_EDGE: u32 = 128;
const TERRAIN_PROGRAMS: u32 = 3;

/// Linear vertex fog over the scene's depth range.
const FOG_START: f32 = 0.3;
const FOG_END: f32 = 1.4;
/// The render states the scene starts from, set at the start of every frame.
const SCENE_STATES: [(u32, u32); 17] = [
    (D3DRS_ZENABLE, 1),
    (D3DRS_ZWRITEENABLE, 1),
    (D3DRS_ZFUNC, D3DCMP_LESSEQUAL),
    (D3DRS_CULLMODE, D3DCULL_CCW),
    (D3DRS_LIGHTING, 1),
    (D3DRS_AMBIENT, 0xFF40_4048),
    (D3DRS_NORMALIZENORMALS, 1),
    (D3DRS_FOGENABLE, 1),
    (D3DRS_FOGCOLOR, 0xFF50_6070),
    (D3DRS_FOGVERTEXMODE, D3DFOG_LINEAR),
    (D3DRS_FOGTABLEMODE, D3DFOG_NONE),
    (D3DRS_FOGSTART, FOG_START.to_bits()),
    (D3DRS_FOGEND, FOG_END.to_bits()),
    (D3DRS_ALPHABLENDENABLE, 0),
    (D3DRS_ALPHATESTENABLE, 0),
    (D3DRS_ALPHAFUNC, D3DCMP_GREATEREQUAL),
    (D3DRS_ALPHAREF, 0x80),
];
/// The `vs_2_0` models' light rows `c90..c92`: direction to the light, its colour, ambient.
const LIGHT_ROWS: [f32; 12] = [
    0.0, 0.0, -1.0, 0.0, //
    0.8, 0.8, 0.7, 1.0, //
    0.25, 0.25, 0.3, 0.0,
];
/// The first light row the `vs_2_0` programs read.
const LIGHT_ROW: u32 = 90;
/// The fixed-function models' blend modes, one per group of [`MODEL_GROUP`] draws in turn.
const BLEND_MODES: [Blend; 5] = [
    // Opaque.
    Blend {
        alpha_test: 0,
        enabled: 0,
        src: D3DBLEND_ONE,
        dst: D3DBLEND_ZERO,
        z_write: 1,
        cull: D3DCULL_CCW,
        lighting: 1,
        alpha_op: D3DTOP_MODULATE,
    },
    // Alpha-keyed foliage.
    Blend {
        alpha_test: 1,
        enabled: 0,
        src: D3DBLEND_ONE,
        dst: D3DBLEND_ZERO,
        z_write: 1,
        cull: D3DCULL_NONE,
        lighting: 1,
        alpha_op: D3DTOP_SELECTARG1,
    },
    // Alpha-blended.
    Blend {
        alpha_test: 1,
        enabled: 1,
        src: D3DBLEND_SRCALPHA,
        dst: D3DBLEND_INVSRCALPHA,
        z_write: 0,
        cull: D3DCULL_NONE,
        lighting: 1,
        alpha_op: D3DTOP_MODULATE,
    },
    // Additive, unlit.
    Blend {
        alpha_test: 0,
        enabled: 1,
        src: D3DBLEND_SRCALPHA,
        dst: D3DBLEND_ONE,
        z_write: 0,
        cull: D3DCULL_NONE,
        lighting: 0,
        alpha_op: D3DTOP_MODULATE,
    },
    // Modulating.
    Blend {
        alpha_test: 0,
        enabled: 1,
        src: D3DBLEND_DESTCOLOR,
        dst: D3DBLEND_ZERO,
        z_write: 0,
        cull: D3DCULL_CCW,
        lighting: 1,
        alpha_op: D3DTOP_MODULATE,
    },
];
/// The index of the additive, unlit mode in [`BLEND_MODES`], whose draws select the texture.
const ADDITIVE: usize = 3;
/// The model declaration of the `vs_2_0` programs: the [`MODEL_FVF`] layout.
const MODEL_DECL: [D3DVERTEXELEMENT9; 4] = [
    element(0, D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_POSITION),
    element(12, D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_NORMAL),
    element(24, D3DDECLTYPE_FLOAT2, D3DDECLUSAGE_TEXCOORD),
    D3DDECL_END,
];
const WARM_UP_FRAMES: u32 = 60;
/// The measured phase is at least this many frames and at least [`MIN_MEASURED`] long.
///
/// The duration floor is what puts one whole five-second window of a
/// `PERF=1` build's summary inside the measured frames.
const MEASURED_FRAMES: usize = 600;
const MIN_MEASURED: Duration = Duration::from_secs(12);
/// The longest the last frame's EVENT query may stay pending before the benchmark fails.
const EVENT_DEADLINE: Duration = Duration::from_secs(5);

/// One busy frame, repeatedly: warm up, then time the frames.
#[test]
#[ignore = "benchmark, run by `make bench`"]
fn wow_112_busy_frame() {
    let h = Harness::create(&HarnessConfig {
        width: WIDTH,
        height: HEIGHT,
        depth_format: Some(D3DFMT_D24S8),
        presentation_interval: D3DPRESENT_INTERVAL_IMMEDIATE,
        config_entries: GAME_CONFIG,
        ..HarnessConfig::default()
    });
    let tsc = TscClock::calibrated();
    let since = SystemTime::now();
    let started = TscClock::now();
    let frame = Frame::new(&h);
    for tick in 0..WARM_UP_FRAMES {
        assert!(h.pump(), "WM_QUIT during warm-up");
        frame.render(tick);
        frame.count_present();
        ok(h.present(), "Present");
    }
    let log = LayerLog::find(since);
    let warm_up = TscClock::since(started);
    let warm = MemorySample::now();

    let from = log.mark();
    let pending_from = frame.calls.pending_polls.get();
    let mut clock = FrameClock::start(MEASURED_FRAMES * 4);
    let mut tick = WARM_UP_FRAMES;
    let mut one_frame = Vec::new();
    while clock.frames() < MEASURED_FRAMES || clock.elapsed() < MIN_MEASURED {
        assert!(h.pump(), "WM_QUIT during the measured frames");
        let before = one_frame.is_empty().then(|| frame.calls.rows());
        frame.render(tick);
        frame.count_present();
        clock.present(&h);
        if let Some(before) = before {
            one_frame = per_frame(&before, &frame.calls.rows());
        }
        tick += 1;
    }
    let to = log.mark();
    let end = MemorySample::now();
    let pending = frame.calls.pending_polls.get() - pending_from;

    let draws = one_frame
        .iter()
        .find(|row| row.0 == "draw")
        .map_or(0, |row| row.1);
    assert_eq!(
        draws, DRAWS_PER_FRAME,
        "the benchmark draws what its own pass shapes say"
    );
    let total: u32 = one_frame.iter().map(|row| row.1 * row.2).sum();

    let stats = clock.stats();
    let work = clock.work_stats();
    let per_draw_ns = work.mean.as_nanos() / u128::from(DRAWS_PER_FRAME);
    let mut mix = String::new();
    for (name, count, _) in &one_frame {
        let _ = writeln!(
            mix,
            "  {name:<24} {count:>6}  {rate:>5.3} per draw",
            rate = f64::from(*count) / f64::from(DRAWS_PER_FRAME),
        );
    }
    let body = format!(
        "shape: scene {WIDTH}x{HEIGHT} A8R8G8B8 target texture + D24S8; {GLOW_PASSES} glow \
         targets {GLOW_WIDTH}x{GLOW_HEIGHT}; back buffer {WIDTH}x{HEIGHT} X8R8G8B8; {GAME_CONFIG}\n\
         per frame: {DRAWS_PER_FRAME} draws in 5 passes (scene {SCENE_DRAWS}: terrain \
         {TERRAIN_DRAWS} FF VS + ps_2_0, FF models {FF_MODEL_DRAWS}, vs_2_0 + FF PS \
         {CHARACTER_DRAWS}, vs_2_0 + ps_2_0 {LIQUID_DRAWS}; glow {GLOW_PASSES}; composite 1 + \
         UI {UI_QUADS})\n\
         programs: {TERRAIN_PROGRAMS} terrain ps_2_0, {SECTIONS} model vs_2_0, {LIQUID_RUNS} \
         liquid pairs, a glow pair and a composite ps_2_0; fixed function in {modes} blend modes, \
         one or two lights, fog\n\
         warm-up: {WARM_UP_FRAMES} frames in {warm_up:.2?}\n\
         measured: {frames} frames in {elapsed:.2?} (at least {MEASURED_FRAMES} frames \
         and {MIN_MEASURED:?})\n\
         frame time (Present to Present): {row}\n\
         API work (Present return to Present call): {work_row}\n\
         API work per draw: {per_draw_ns} ns (mean API work over {DRAWS_PER_FRAME} draws)\n\
         D3D9 calls in one measured frame: {total} ({calls_per_draw:.2} per draw; a lock \
         and a scene count two calls, the redundant SetTexture row none)\n{mix}\
         EVENT polls that found the query pending: {pending} over the measured frames\n\
         {memory}{perf}{warm_up_compiles}",
        modes = BLEND_MODES.len(),
        frames = stats.frames,
        elapsed = clock.elapsed(),
        row = stats.row(),
        work_row = work.row(),
        calls_per_draw = f64::from(total) / f64::from(DRAWS_PER_FRAME),
        memory = memory_section(&warm, &end),
        perf = log.perf_rows(from, to).section(),
        warm_up_compiles = log
            .first_window_rows(to)
            .map_or_else(String::new, |rows| format!(
                "perf: this device's first window, its warm-up compiles\n{rows}"
            )),
    );
    let mut metrics = Metrics::new("wow112", &h, &tsc);
    metrics.frame_rows("frame", &stats);
    metrics.frame_rows("api", &work);
    metrics.metric(
        "warmup.ms",
        Value::Ms(warm_up),
        Direction::Lower,
        Class::Info,
    );
    metrics.metric(
        "measured.frames",
        Value::Count(u64::try_from(stats.frames).expect("frame count fits u64")),
        Direction::Higher,
        Class::Info,
    );
    metrics.metric(
        "measured.ms",
        Value::Ms(clock.elapsed()),
        Direction::Lower,
        Class::Info,
    );
    metrics.metric(
        "query.pending_polls",
        Value::Count(pending),
        Direction::Lower,
        Class::Noisy,
    );
    for (name, count, _) in &one_frame {
        metrics.metric(
            &format!("calls.{name}"),
            Value::Count(u64::from(*count)),
            Direction::Lower,
            Class::Info,
        );
    }
    metrics.metric(
        "calls.total",
        Value::Count(u64::from(total)),
        Direction::Lower,
        Class::Info,
    );
    metrics.memory(&warm, &end);
    metrics.perf(&log.perf_kv_last_full(from, to), &FrameWork::Fixed);
    metrics.shapes(&pass_shapes());
    write_report(&metrics, &log, &body);
}

/// The five passes of one frame, in the order [`Frame::render`] draws them.
///
/// A draw's textures are the ones its program or its enabled stages sample:
/// five for a terrain chunk, two for a liquid, the composite and a masked UI
/// quad, and one for every other draw. The fixed vertex pipeline serves the
/// terrain, the fixed-function models and the UI; the fixed texture stages
/// serve the fixed-function models, the `vs_2_0` models and the UI.
///
/// # Panics
/// Panics if the passes do not add up to [`DRAWS_PER_FRAME`].
fn pass_shapes() -> Vec<PassShape> {
    let mut passes = vec![PassShape {
        width: WIDTH,
        height: HEIGHT,
        draws: SCENE_DRAWS,
        ff_vs: TERRAIN_DRAWS + FF_MODEL_DRAWS,
        ff_ps: FF_MODEL_DRAWS + CHARACTER_DRAWS,
        textures: TERRAIN_DRAWS * TERRAIN_TEXTURES
            + FF_MODEL_DRAWS
            + CHARACTER_DRAWS
            + LIQUID_DRAWS * LIQUID_TEXTURES,
    }];
    passes.extend((0..GLOW_PASSES).map(|_| PassShape {
        width: GLOW_WIDTH,
        height: GLOW_HEIGHT,
        draws: 1,
        ff_vs: 0,
        ff_ps: 0,
        textures: 1,
    }));
    passes.push(PassShape {
        width: WIDTH,
        height: HEIGHT,
        draws: 1 + UI_QUADS,
        ff_vs: UI_QUADS,
        ff_ps: UI_QUADS,
        textures: COMPOSITE_TEXTURES + UI_QUADS + UI_QUADS / UI_MASK_EVERY,
    });
    let draws: u32 = passes.iter().map(|pass| pass.draws).sum();
    assert_eq!(draws, DRAWS_PER_FRAME, "the pass shapes cover every draw");
    passes
}

/// The rows of `after` less those of `before`: what the frame between them called.
fn per_frame(
    before: &[(&'static str, u32, u32)],
    after: &[(&'static str, u32, u32)],
) -> Vec<(&'static str, u32, u32)> {
    before
        .iter()
        .zip(after)
        .map(|(before, after)| (after.0, after.1 - before.1, after.2))
        .collect()
}

/// A fixed-function model group's blend mode: its render states and stage 0's alpha operation.
struct Blend {
    alpha_test: u32,
    enabled: u32,
    src: u32,
    dst: u32,
    z_write: u32,
    cull: u32,
    lighting: u32,
    alpha_op: u32,
}

/// How many D3D9 calls of each kind the frame has made, counted where it makes them.
#[derive(Default)]
struct Calls {
    draw: Cell<u32>,
    set_texture: Cell<u32>,
    /// `SetTexture` calls that bound the texture already bound on that stage.
    set_texture_redundant: Cell<u32>,
    set_transform: Cell<u32>,
    set_material: Cell<u32>,
    set_light: Cell<u32>,
    light_enable: Cell<u32>,
    set_render_state: Cell<u32>,
    set_texture_stage_state: Cell<u32>,
    set_sampler_state: Cell<u32>,
    set_pixel_shader: Cell<u32>,
    set_vertex_shader: Cell<u32>,
    set_vs_const: Cell<u32>,
    set_ps_const: Cell<u32>,
    set_fvf: Cell<u32>,
    set_vertex_declaration: Cell<u32>,
    set_stream_source: Cell<u32>,
    set_indices: Cell<u32>,
    /// `Lock` and `Unlock` pairs on the dynamic vertex ring.
    vb_lock: Cell<u32>,
    /// `Lock` and `Unlock` pairs on the dynamic index ring.
    ib_lock: Cell<u32>,
    /// `LockRect` and `UnlockRect` pairs.
    lock_rect: Cell<u32>,
    set_render_target: Cell<u32>,
    set_depth_stencil: Cell<u32>,
    clear: Cell<u32>,
    query_issue: Cell<u32>,
    /// The `GetData` poll that found the event done, one a frame.
    query_get_data: Cell<u32>,
    /// `BeginScene` and `EndScene` pairs.
    scene: Cell<u32>,
    present: Cell<u32>,
    /// `GetData` polls that found the event still pending, which vary from run to run.
    pending_polls: Cell<u64>,
}

impl Calls {
    /// Every counter as `(name, count, D3D9 calls one count stands for)`.
    fn rows(&self) -> Vec<(&'static str, u32, u32)> {
        [
            ("draw", &self.draw, 1),
            ("set_texture", &self.set_texture, 1),
            ("set_texture_redundant", &self.set_texture_redundant, 0),
            ("set_transform", &self.set_transform, 1),
            ("set_material", &self.set_material, 1),
            ("set_light", &self.set_light, 1),
            ("light_enable", &self.light_enable, 1),
            ("set_render_state", &self.set_render_state, 1),
            ("set_texture_stage_state", &self.set_texture_stage_state, 1),
            ("set_sampler_state", &self.set_sampler_state, 1),
            ("set_pixel_shader", &self.set_pixel_shader, 1),
            ("set_vertex_shader", &self.set_vertex_shader, 1),
            ("set_vs_const_f", &self.set_vs_const, 1),
            ("set_ps_const_f", &self.set_ps_const, 1),
            ("set_fvf", &self.set_fvf, 1),
            ("set_vertex_declaration", &self.set_vertex_declaration, 1),
            ("set_stream_source", &self.set_stream_source, 1),
            ("set_indices", &self.set_indices, 1),
            ("vb_lock", &self.vb_lock, 2),
            ("ib_lock", &self.ib_lock, 2),
            ("lock_rect", &self.lock_rect, 2),
            ("set_render_target", &self.set_render_target, 1),
            ("set_depth_stencil", &self.set_depth_stencil, 1),
            ("clear", &self.clear, 1),
            ("query_issue", &self.query_issue, 1),
            ("query_get_data", &self.query_get_data, 1),
            ("scene", &self.scene, 2),
            ("present", &self.present, 1),
        ]
        .into_iter()
        .map(|(name, count, calls)| (name, count.get(), calls))
        .collect()
    }
}

/// An offscreen colour target that later passes sample.
struct Target<'h> {
    texture: Texture<'h>,
    surface: Surface<'h>,
}

/// Every resource the frame uses, created once, and the counts of the calls it makes.
struct Frame<'h> {
    h: &'h Harness,
    back_buffer: Surface<'h>,
    depth: Surface<'h>,
    scene: Target<'h>,
    glow: [Target<'h>; 2],
    /// [`LAYER_SETS`] sets of three terrain layers.
    layers: Vec<Texture<'h>>,
    alpha_maps: Vec<Texture<'h>>,
    model_textures: Vec<Texture<'h>>,
    ui_textures: Vec<Texture<'h>>,
    /// The second texture of a masked UI quad.
    ui_mask: Texture<'h>,
    /// The managed font atlas, locked whole every frame while the last frame still samples it.
    font: Texture<'h>,
    /// The dynamic minimap, locked with `D3DLOCK_DISCARD` every frame.
    minimap: Texture<'h>,
    /// Two glyph cells, one written into the font atlas each frame.
    glyphs: Vec<u32>,
    /// Two minimap images one above the other, a window of which is written each frame.
    minimap_texels: Vec<u32>,
    terrain_vb: VertexBuffer<'h>,
    terrain_ib: IndexBuffer<'h>,
    /// The static model mesh, for the static fixed-function, `vs_2_0` and liquid draws.
    model_vb: VertexBuffer<'h>,
    model_ib: IndexBuffer<'h>,
    /// What each ring-fed fixed-function draw writes into the rings.
    model_vertices: Vec<[f32; 8]>,
    model_indices: Vec<u16>,
    ring_vb: VertexBuffer<'h>,
    ring_ib: IndexBuffer<'h>,
    /// The next free byte of each ring; zero makes the next append start it over.
    vertex_at: Cell<u32>,
    index_at: Cell<u32>,
    /// One quad's indices, for the glow passes, the composite and the UI.
    quad_ib: IndexBuffer<'h>,
    screen_quad: VertexBuffer<'h>,
    model_decl: VertexDeclaration<'h>,
    textured_decl: VertexDeclaration<'h>,
    terrain_ps: Vec<PixelShader<'h>>,
    /// One per section's run.
    model_vs: Vec<VertexShader<'h>>,
    /// One pair per liquid run.
    liquid: Vec<(VertexShader<'h>, PixelShader<'h>)>,
    glow_vs: VertexShader<'h>,
    glow_ps: PixelShader<'h>,
    composite_ps: PixelShader<'h>,
    event: Query<'h>,
    /// Whether the event has been issued once, so it has something to answer.
    issued: Cell<bool>,
    /// The texture bound on each stage the frame uses, to count redundant binds.
    bound: [Cell<*mut c_void>; 5],
    calls: Calls,
}

impl<'h> Frame<'h> {
    fn new(h: &'h Harness) -> Self {
        let target = |width, height| {
            let texture = h.create_texture(
                width,
                height,
                1,
                D3DUSAGE_RENDERTARGET,
                D3DFMT_A8R8G8B8,
                D3DPOOL_DEFAULT,
            );
            let surface = texture.surface_level(0);
            Target { texture, surface }
        };
        let (terrain_vertices, terrain_indices) = terrain_buffers(h);
        let (model_vertices, model_indices) = model_mesh();
        Self {
            h,
            back_buffer: h.back_buffer(0),
            depth: h
                .depth_stencil_surface()
                .expect("the device has an auto depth-stencil"),
            scene: target(WIDTH, HEIGHT),
            glow: [0, 1].map(|_| target(GLOW_WIDTH, GLOW_HEIGHT)),
            layers: (0..LAYER_SETS * 3)
                .map(|at| layer_texture(h, 0xFF30_5020 + at * 0x000A_0B09))
                .collect(),
            alpha_maps: (0..ALPHA_MAPS).map(|at| alpha_map(h, at)).collect(),
            model_textures: (0..MODEL_TEXTURES)
                .map(|at| pattern_texture(h, 0xFF30_5020 + at * 0x0007_0B05))
                .collect(),
            ui_textures: (0..UI_TEXTURES).map(|at| dxt3_texture(h, at)).collect(),
            ui_mask: dxt3_texture(h, UI_TEXTURES),
            font: font_atlas(h),
            minimap: h.create_texture(
                MINIMAP_EDGE,
                MINIMAP_EDGE,
                1,
                D3DUSAGE_DYNAMIC,
                D3DFMT_A8R8G8B8,
                D3DPOOL_DEFAULT,
            ),
            glyphs: (0..GLYPH_EDGE * GLYPH_EDGE * 2)
                .map(|at| {
                    if at % 3 == 0 {
                        0x00FF_FFFF
                    } else {
                        0xFFFF_FFFF
                    }
                })
                .collect(),
            minimap_texels: (0..MINIMAP_EDGE * MINIMAP_EDGE * 2)
                .map(|at| 0xFF20_4020 | ((at / MINIMAP_EDGE) % 64) << 17 | (at % 64) << 1)
                .collect(),
            terrain_vb: terrain_vertices,
            terrain_ib: terrain_indices,
            model_vb: static_buffer(h, &model_vertices),
            model_ib: static_indices(h, &model_indices),
            model_vertices,
            model_indices,
            ring_vb: h.create_vertex_buffer(
                RING_VERTEX_BYTES,
                D3DUSAGE_DYNAMIC | D3DUSAGE_WRITEONLY,
                0,
                D3DPOOL_DEFAULT,
            ),
            ring_ib: h.create_index_buffer(
                RING_INDEX_BYTES,
                D3DUSAGE_DYNAMIC | D3DUSAGE_WRITEONLY,
                D3DFMT_INDEX16,
                D3DPOOL_DEFAULT,
            ),
            vertex_at: Cell::new(0),
            index_at: Cell::new(0),
            quad_ib: static_indices(h, &[0, 1, 2, 2, 1, 3]),
            screen_quad: screen_quad(h),
            model_decl: h.create_vertex_declaration(&MODEL_DECL),
            textured_decl: h.create_vertex_declaration(&TEXTURED_DECL),
            terrain_ps: (0..TERRAIN_PROGRAMS)
                .map(|at| h.create_pixel_shader(&terrain_ps(ratio(at, 50))))
                .collect(),
            model_vs: (0..SECTIONS)
                .map(|at| h.create_vertex_shader(&model_vs(ratio(at, 1000))))
                .collect(),
            liquid: (0..LIQUID_RUNS)
                .map(|at| {
                    (
                        h.create_vertex_shader(&liquid_vs(ratio(at + 1, 100))),
                        h.create_pixel_shader(&liquid_ps(ratio(at, 20))),
                    )
                })
                .collect(),
            glow_vs: h.create_vertex_shader(&glow_vs()),
            glow_ps: h.create_pixel_shader(&glow_ps()),
            composite_ps: h.create_pixel_shader(&composite_ps()),
            event: h
                .create_query(D3DQUERYTYPE_EVENT)
                .expect("EVENT queries are supported"),
            issued: Cell::new(false),
            bound: Default::default(),
            calls: Calls::default(),
        }
    }

    /// One whole frame up to its `Present`, `tick` animating positions and texture updates.
    fn render(&self, tick: u32) {
        let h = self.h;
        self.poll_event();
        self.update_textures(tick);
        self.vertex_at.set(0);
        self.index_at.set(0);
        bump(&self.calls.scene);
        ok(h.begin_scene(), "BeginScene");
        // The frame opens on the default target before the world renderer takes over.
        self.target(&self.back_buffer);
        self.scene_start();
        self.terrain(tick);
        for section in 0..SECTIONS {
            self.doodads(tick, section);
            self.models(tick, section);
        }
        self.liquids(tick);
        self.glow();
        self.composite();
        self.ui(tick);
        ok(h.end_scene(), "EndScene");
        bump(&self.calls.query_issue);
        ok(self.event.issue(D3DISSUE_END), "EVENT Issue");
        self.issued.set(true);
    }

    /// Count the `Present` the caller is about to make.
    fn count_present(&self) {
        bump(&self.calls.present);
    }

    /// Spin on the last frame's event with `D3DGETDATA_FLUSH` until it answers.
    ///
    /// # Panics
    /// Panics if the event is still pending after [`EVENT_DEADLINE`], so a
    /// layer that never answers fails the benchmark instead of hanging it.
    fn poll_event(&self) {
        if !self.issued.get() {
            return;
        }
        let started = Instant::now();
        let mut polls = 0_u64;
        loop {
            let (hr, _) = self.event.data_u32(D3DGETDATA_FLUSH);
            if hr != S_FALSE {
                ok(hr, "EVENT GetData");
                break;
            }
            polls += 1;
            let pending = &self.calls.pending_polls;
            pending.set(pending.get() + 1);
            assert!(
                started.elapsed() < EVENT_DEADLINE,
                "the EVENT query is still pending after {polls} polls over {EVENT_DEADLINE:?}"
            );
        }
        bump(&self.calls.query_get_data);
    }

    /// Rewrite a glyph of the font atlas without `D3DLOCK_DISCARD`, and the whole minimap with it.
    fn update_textures(&self, tick: u32) {
        let glyph = usize::try_from(GLYPH_EDGE).expect("glyph edge fits usize");
        let at = glyph * glyph * slot(tick % 2);
        bump(&self.calls.lock_rect);
        self.font.lock_rect(0, 0).write_u32_rect(
            glyph,
            glyph,
            &self.glyphs[at..at + glyph * glyph],
        );
        let edge = usize::try_from(MINIMAP_EDGE).expect("minimap edge fits usize");
        let at = edge * slot(tick % MINIMAP_EDGE);
        bump(&self.calls.lock_rect);
        self.minimap.lock_rect(0, D3DLOCK_DISCARD).write_u32_rect(
            edge,
            edge,
            &self.minimap_texels[at..at + edge * edge],
        );
    }

    /// Bind the scene target, clear it and set the render states the scene starts from.
    fn scene_start(&self) {
        let h = self.h;
        self.target(&self.scene.surface);
        // The terrain renderer binds the same target again.
        self.target(&self.scene.surface);
        bump(&self.calls.clear);
        ok(
            h.clear(
                D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER | D3DCLEAR_STENCIL,
                0xFF50_6070,
                1.0,
                0,
            ),
            "scene clear",
        );
        for (state, value) in SCENE_STATES {
            self.rs(state, value);
        }
        self.xform(D3DTS_VIEW, &IDENTITY_ROWS);
        self.xform(D3DTS_PROJECTION, &IDENTITY_ROWS);
    }

    /// The terrain chunks, in runs that share a program and five textures.
    fn terrain(&self, tick: u32) {
        self.vs(None);
        self.fvf(TERRAIN_FVF);
        // The program reads the layers' coordinate in t0 and the alpha maps' in t1.
        self.tss(0, D3DTSS_TEXCOORDINDEX, 0);
        self.tss(1, D3DTSS_TEXCOORDINDEX, 1);
        for stage in 0..TERRAIN_TEXTURES {
            let address = if stage < 3 {
                D3DTADDRESS_WRAP
            } else {
                D3DTADDRESS_CLAMP
            };
            self.samp(stage, D3DSAMP_MINFILTER, D3DTEXF_LINEAR);
            self.samp(stage, D3DSAMP_MAGFILTER, D3DTEXF_LINEAR);
            self.samp(stage, D3DSAMP_ADDRESSU, address);
            self.samp(stage, D3DSAMP_ADDRESSV, address);
        }
        self.light(0, &sun());
        self.light_enable(0, true);
        self.stream(&self.terrain_vb, TERRAIN_STRIDE);
        self.indices(&self.terrain_ib);
        for chunk in 0..TERRAIN_DRAWS {
            if chunk % TERRAIN_RUN == 0 {
                self.terrain_run(chunk / TERRAIN_RUN);
            }
            self.fvf(TERRAIN_FVF);
            self.tex(0, self.layer(chunk / TERRAIN_RUN, 0));
            self.stream(&self.terrain_vb, TERRAIN_STRIDE);
            self.xform(D3DTS_WORLD, &chunk_world(chunk, tick));
            let base = (chunk % TILE_CHUNKS) * TERRAIN_VERTS;
            self.draw(
                i32::try_from(base).expect("terrain base vertex fits i32"),
                TERRAIN_VERTS,
                0,
                TERRAIN_TRIANGLES,
            );
        }
    }

    /// A terrain run's program, its constants, its five textures and its material.
    fn terrain_run(&self, run: u32) {
        self.ps(Some(&self.terrain_ps[slot(run % TERRAIN_PROGRAMS)]));
        let shade = ratio(run % 8, 8);
        self.ps_const(0, &[shade.mul_add(0.2, 0.8), 1.0, 0.9, 1.0]);
        self.ps_const(1, &[4.0, 4.0, 1.0, 1.0]);
        for layer in 0..3 {
            self.tex(layer, self.layer(run, layer));
        }
        let pair = (run % 4) * 2;
        self.tex(3, &self.alpha_maps[slot(pair)]);
        self.tex(4, &self.alpha_maps[slot(pair + 1)]);
        self.material(&material(run % 16));
    }

    /// Layer `layer` of the set terrain run `run` draws with.
    fn layer(&self, run: u32, layer: u32) -> &Texture<'h> {
        let set = (run / RUNS_PER_LAYER_SET) % LAYER_SETS;
        &self.layers[slot(set * 3 + layer)]
    }

    /// A section's fixed-function models, in groups that share a texture and a blend mode.
    fn doodads(&self, tick: u32, section: u32) {
        let (first, end) = split(FF_MODEL_DRAWS, SECTIONS, section);
        self.vs(None);
        self.fvf(MODEL_FVF);
        self.rs(D3DRS_FOGENABLE, 1);
        self.tss(0, D3DTSS_COLORARG1, D3DTA_TEXTURE);
        self.tss(0, D3DTSS_COLORARG2, D3DTA_DIFFUSE);
        for draw in first..end {
            let local = draw - first;
            let group = section * 16 + local / MODEL_GROUP;
            if local % MODEL_GROUP == 0 {
                self.doodad_group(group);
            }
            self.doodad(tick, draw, group, local < FF_STATIC_PER_SECTION);
        }
    }

    /// A fixed-function group's blend mode, alpha stage and sun.
    fn doodad_group(&self, group: u32) {
        let mode = &BLEND_MODES[slot(group) % BLEND_MODES.len()];
        self.ps(None);
        self.rs(D3DRS_ALPHATESTENABLE, mode.alpha_test);
        self.rs(D3DRS_ALPHABLENDENABLE, mode.enabled);
        self.rs(D3DRS_SRCBLEND, mode.src);
        self.rs(D3DRS_DESTBLEND, mode.dst);
        self.rs(D3DRS_ZWRITEENABLE, mode.z_write);
        self.rs(D3DRS_CULLMODE, mode.cull);
        self.rs(D3DRS_LIGHTING, mode.lighting);
        self.tss(0, D3DTSS_ALPHAOP, mode.alpha_op);
        self.tss(0, D3DTSS_ALPHAARG1, D3DTA_TEXTURE);
        self.tss(0, D3DTSS_ALPHAARG2, D3DTA_DIFFUSE);
        self.tss(1, D3DTSS_COLOROP, D3DTOP_DISABLE);
        self.light(0, &sun());
    }

    /// One fixed-function model: from the static buffers, or written into the rings first.
    fn doodad(&self, tick: u32, draw: u32, group: u32, from_static: bool) {
        let (base, start) = if from_static {
            self.stream(&self.model_vb, MODEL_STRIDE);
            self.indices(&self.model_ib);
            (0, 0)
        } else {
            let base = self.append_vertices(MODEL_STRIDE, &self.model_vertices);
            let start = self.append_indices(&self.model_indices);
            self.stream(&self.ring_vb, MODEL_STRIDE);
            self.indices(&self.ring_ib);
            (base, start)
        };
        self.fvf(MODEL_FVF);
        self.tex(0, &self.model_textures[slot(group % MODEL_TEXTURES)]);
        let op = if slot(group) % BLEND_MODES.len() == ADDITIVE {
            D3DTOP_SELECTARG1
        } else if draw % 4 == 3 {
            D3DTOP_MODULATE2X
        } else {
            D3DTOP_MODULATE
        };
        self.tss(0, D3DTSS_COLOROP, op);
        let (scale, x, y, z) = model_place(draw, tick);
        self.xform(D3DTS_WORLD, &ff_world(scale, x, y, z));
        self.material(&material(draw % 16));
        self.light(1, &local_light(x, y, z));
        self.light_enable(1, !draw.is_multiple_of(3));
        self.draw(base, MODEL_VERTS, start, MODEL_TRIANGLES);
    }

    /// A section's run of `vs_2_0` models over the fixed texture stages.
    fn models(&self, tick: u32, section: u32) {
        let (first, end) = split(CHARACTER_DRAWS, SECTIONS, section);
        self.vs(Some(&self.model_vs[slot(section)]));
        self.ps(None);
        self.decl(&self.model_decl);
        self.vs_const(0, &IDENTITY_ROWS);
        self.vs_const(LIGHT_ROW, &LIGHT_ROWS);
        self.stream(&self.model_vb, MODEL_STRIDE);
        self.indices(&self.model_ib);
        // The programs write no fog, so the fixed fog is off under them.
        self.rs(D3DRS_FOGENABLE, 0);
        self.rs(D3DRS_ALPHATESTENABLE, 1);
        self.rs(D3DRS_ALPHABLENDENABLE, 0);
        self.rs(D3DRS_ZWRITEENABLE, 1);
        self.rs(D3DRS_CULLMODE, D3DCULL_NONE);
        self.tss(0, D3DTSS_COLOROP, D3DTOP_MODULATE);
        self.tss(0, D3DTSS_COLORARG1, D3DTA_TEXTURE);
        self.tss(0, D3DTSS_COLORARG2, D3DTA_DIFFUSE);
        self.tss(0, D3DTSS_ALPHAOP, D3DTOP_SELECTARG1);
        self.tss(0, D3DTSS_ALPHAARG1, D3DTA_TEXTURE);
        self.tss(1, D3DTSS_COLOROP, D3DTOP_DISABLE);
        for draw in first..end {
            let group = draw / CHARACTER_GROUP;
            if (draw - first) % CHARACTER_GROUP == 0 {
                let op = if group.is_multiple_of(2) {
                    D3DTOP_MODULATE
                } else {
                    D3DTOP_MODULATE2X
                };
                self.tss(0, D3DTSS_COLOROP, op);
            }
            self.tex(0, &self.model_textures[slot((group + 5) % MODEL_TEXTURES)]);
            self.decl(&self.model_decl);
            // The world rows, then up to three 4-row bone matrices the program does not read.
            let (scale, x, y, z) = model_place(draw + FF_MODEL_DRAWS, tick);
            let mut rows = [0.25_f32; 64];
            rows[..16].copy_from_slice(&world_rows(scale, x, y, z));
            let count = 16 * (1 + slot(draw % 4));
            self.vs_const(4, &rows[..count]);
            self.stream(&self.model_vb, MODEL_STRIDE);
            self.draw(0, MODEL_VERTS, 0, MODEL_TRIANGLES);
        }
    }

    /// The liquids: `vs_2_0` with `ps_2_0`, blended, one run per program pair.
    fn liquids(&self, tick: u32) {
        for run in 0..LIQUID_RUNS {
            let (first, end) = split(LIQUID_DRAWS, LIQUID_RUNS, run);
            let (vs, ps) = &self.liquid[slot(run)];
            let water = &self.model_textures[slot(2 + run)];
            self.vs(Some(vs));
            self.ps(Some(ps));
            self.decl(&self.model_decl);
            self.vs_const(0, &IDENTITY_ROWS);
            self.vs_const(LIGHT_ROW, &LIGHT_ROWS);
            self.ps_const(0, &[0.9, 0.95, 1.0, 0.7]);
            self.tex(0, water);
            self.tex(1, &self.model_textures[9]);
            self.stream(&self.model_vb, MODEL_STRIDE);
            self.indices(&self.model_ib);
            self.rs(D3DRS_ALPHABLENDENABLE, 1);
            self.rs(D3DRS_SRCBLEND, D3DBLEND_SRCALPHA);
            self.rs(D3DRS_DESTBLEND, D3DBLEND_INVSRCALPHA);
            self.rs(D3DRS_ZWRITEENABLE, 0);
            self.rs(D3DRS_ALPHATESTENABLE, 0);
            self.rs(D3DRS_CULLMODE, D3DCULL_NONE);
            for draw in first..end {
                self.ps(Some(ps));
                self.tex(0, water);
                self.decl(&self.model_decl);
                let (scale, x, y, z) = model_place(draw + FF_MODEL_DRAWS + CHARACTER_DRAWS, tick);
                self.vs_const(4, &world_rows(scale * 2.0, x, y, z));
                self.draw(0, MODEL_VERTS, 0, MODEL_TRIANGLES);
            }
        }
        self.rs(D3DRS_ZWRITEENABLE, 1);
        self.rs(D3DRS_ALPHABLENDENABLE, 0);
    }

    /// Three one-draw glow passes at a quarter size, each sampling the previous target.
    fn glow(&self) {
        self.vs(Some(&self.glow_vs));
        self.ps(Some(&self.glow_ps));
        self.decl(&self.textured_decl);
        self.stream(&self.screen_quad, STRIDE);
        self.indices(&self.quad_ib);
        self.rs(D3DRS_ZENABLE, 0);
        self.rs(D3DRS_FOGENABLE, 0);
        self.rs(D3DRS_ALPHABLENDENABLE, 0);
        self.rs(D3DRS_ALPHATESTENABLE, 0);
        self.rs(D3DRS_CULLMODE, D3DCULL_NONE);
        self.samp(0, D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP);
        self.samp(0, D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP);
        let passes = [
            (&self.glow[0], &self.scene.texture),
            (&self.glow[1], &self.glow[0].texture),
            (&self.glow[0], &self.glow[1].texture),
        ];
        for (pass, (target, source)) in (0..GLOW_PASSES).zip(passes) {
            // The full-size scene depth stays bound under the quarter-size target.
            self.target(&target.surface);
            self.tex(0, source);
            self.vs_const(0, &taps(pass));
            self.ps_const(0, &[0.25; 4]);
            self.draw(0, 4, 0, 2);
        }
    }

    /// The composite of the scene and the glow onto the back buffer.
    fn composite(&self) {
        self.target(&self.back_buffer);
        self.ps(Some(&self.composite_ps));
        self.tex(0, &self.scene.texture);
        self.tex(1, &self.glow[0].texture);
        self.vs_const(0, &[0.0; 16]);
        self.ps_const(0, &[0.6, 0.6, 0.6, 0.0]);
        self.draw(0, 4, 0, 2);
    }

    /// The UI: fixed-function quads written into the vertex ring, one draw each.
    fn ui(&self, tick: u32) {
        // The UI renderer binds its target again.
        self.target(&self.back_buffer);
        self.vs(None);
        self.ps(None);
        self.fvf(UI_FVF);
        self.rs(D3DRS_LIGHTING, 0);
        self.rs(D3DRS_ALPHABLENDENABLE, 1);
        self.rs(D3DRS_SRCBLEND, D3DBLEND_SRCALPHA);
        self.rs(D3DRS_DESTBLEND, D3DBLEND_INVSRCALPHA);
        self.rs(D3DRS_ALPHATESTENABLE, 1);
        self.tss(0, D3DTSS_COLOROP, D3DTOP_MODULATE);
        self.tss(0, D3DTSS_COLORARG1, D3DTA_TEXTURE);
        self.tss(0, D3DTSS_COLORARG2, D3DTA_DIFFUSE);
        self.tss(0, D3DTSS_ALPHAOP, D3DTOP_MODULATE);
        self.tss(0, D3DTSS_ALPHAARG1, D3DTA_TEXTURE);
        self.tss(0, D3DTSS_ALPHAARG2, D3DTA_DIFFUSE);
        self.tss(1, D3DTSS_COLOROP, D3DTOP_DISABLE);
        self.tss(1, D3DTSS_COLORARG1, D3DTA_TEXTURE);
        self.tss(1, D3DTSS_COLORARG2, D3DTA_CURRENT);
        // A masked quad enables stage 1's colour, and its alpha may not stay disabled then.
        self.tss(1, D3DTSS_ALPHAOP, D3DTOP_SELECTARG2);
        // The UI vertex has one coordinate, which stage 1 reads too.
        self.tss(1, D3DTSS_TEXCOORDINDEX, 0);
        self.samp(0, D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP);
        self.samp(0, D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP);
        self.xform(D3DTS_VIEW, &IDENTITY_ROWS);
        self.xform(D3DTS_PROJECTION, &IDENTITY_ROWS);
        self.stream(&self.ring_vb, STRIDE);
        self.indices(&self.quad_ib);
        for quad in 0..UI_QUADS {
            if quad % UI_ELEMENT == 0 {
                self.ui_element(quad / UI_ELEMENT);
            }
            self.ui_quad(tick, quad);
        }
    }

    /// A UI element's blend mode and stage operations: text, icons or frames.
    fn ui_element(&self, element: u32) {
        let dst = if element % 4 == 3 {
            D3DBLEND_ONE
        } else {
            D3DBLEND_INVSRCALPHA
        };
        self.rs(D3DRS_SRCBLEND, D3DBLEND_SRCALPHA);
        self.rs(D3DRS_DESTBLEND, dst);
        let (color, alpha) = match element % 3 {
            0 => (D3DTOP_SELECTARG2, D3DTOP_MODULATE),
            1 => (D3DTOP_MODULATE, D3DTOP_MODULATE),
            _ => (D3DTOP_SELECTARG1, D3DTOP_SELECTARG1),
        };
        self.tss(0, D3DTSS_COLOROP, color);
        self.tss(0, D3DTSS_ALPHAOP, alpha);
    }

    /// One UI quad: its vertices into the ring, its placement, its one or two textures.
    fn ui_quad(&self, tick: u32, quad: u32) {
        let element = quad / UI_ELEMENT;
        let base = self.append_vertices(STRIDE, &ui_vertices(quad));
        self.fvf(UI_FVF);
        self.stream(&self.ring_vb, STRIDE);
        let drift = ratio(tick % 32, 32) * 0.002;
        self.xform(D3DTS_WORLD, &ff_world(1.0, drift, 0.0, 0.0));
        let texture = if quad == 0 {
            &self.minimap
        } else if element.is_multiple_of(3) {
            &self.font
        } else {
            &self.ui_textures[slot(element % UI_TEXTURES)]
        };
        self.tex(0, texture);
        let masked = quad % UI_MASK_EVERY == UI_MASK_EVERY - 1;
        if masked {
            self.tex(1, &self.ui_mask);
            self.tss(1, D3DTSS_COLOROP, D3DTOP_MODULATE);
        }
        self.draw(base, 4, 0, 2);
        if masked {
            self.tss(1, D3DTSS_COLOROP, D3DTOP_DISABLE);
        }
    }

    /// Append `data` to the vertex ring at a multiple of `stride`; its base vertex.
    ///
    /// The first append of a frame starts the ring over with
    /// `D3DLOCK_DISCARD`; every other one takes `D3DLOCK_NOOVERWRITE`.
    fn append_vertices<T: Copy>(&self, stride: u32, data: &[T]) -> i32 {
        let bytes = u32::try_from(size_of_val(data)).expect("an append fits u32");
        let start = self.vertex_at.get().div_ceil(stride) * stride;
        assert!(
            start + bytes <= RING_VERTEX_BYTES,
            "the vertex ring holds a frame"
        );
        let flags = if start == 0 {
            D3DLOCK_DISCARD
        } else {
            D3DLOCK_NOOVERWRITE
        };
        bump(&self.calls.vb_lock);
        self.ring_vb.lock(start, bytes, flags).write(data);
        self.vertex_at.set(start + bytes);
        i32::try_from(start / stride).expect("base vertex fits i32")
    }

    /// Append `data` to the index ring, as [`Self::append_vertices`] does; its start index.
    fn append_indices(&self, data: &[u16]) -> u32 {
        let bytes = u32::try_from(size_of_val(data)).expect("an append fits u32");
        let start = self.index_at.get();
        assert!(
            start + bytes <= RING_INDEX_BYTES,
            "the index ring holds a frame"
        );
        let flags = if start == 0 {
            D3DLOCK_DISCARD
        } else {
            D3DLOCK_NOOVERWRITE
        };
        bump(&self.calls.ib_lock);
        self.ring_ib.lock(start, bytes, flags).write(data);
        self.index_at.set(start + bytes);
        start / 2
    }

    /// `SetRenderTarget(0, surface)`, then the scene depth, which every pass keeps.
    fn target(&self, surface: &Surface<'_>) {
        bump(&self.calls.set_render_target);
        ok(self.h.set_render_target(0, surface), "SetRenderTarget");
        bump(&self.calls.set_depth_stencil);
        ok(
            self.h.set_depth_stencil_surface(&self.depth),
            "SetDepthStencilSurface",
        );
    }

    fn tex(&self, stage: u32, texture: &Texture<'_>) {
        bump(&self.calls.set_texture);
        let bound = &self.bound[slot(stage)];
        if bound.get() == texture.as_ptr() {
            bump(&self.calls.set_texture_redundant);
        }
        bound.set(texture.as_ptr());
        ok(self.h.set_texture(stage, texture), "SetTexture");
    }

    fn rs(&self, state: u32, value: u32) {
        bump(&self.calls.set_render_state);
        ok(self.h.set_render_state(state, value), "SetRenderState");
    }

    fn tss(&self, stage: u32, key: u32, value: u32) {
        bump(&self.calls.set_texture_stage_state);
        ok(
            self.h.set_texture_stage_state(stage, key, value),
            "SetTextureStageState",
        );
    }

    fn samp(&self, stage: u32, key: u32, value: u32) {
        bump(&self.calls.set_sampler_state);
        ok(
            self.h.set_sampler_state(stage, key, value),
            "SetSamplerState",
        );
    }

    fn xform(&self, state: u32, matrix: &[f32; 16]) {
        bump(&self.calls.set_transform);
        ok(self.h.set_transform(state, matrix), "SetTransform");
    }

    fn material(&self, material: &D3DMATERIAL9) {
        bump(&self.calls.set_material);
        ok(self.h.set_material(material), "SetMaterial");
    }

    fn light(&self, index: u32, light: &D3DLIGHT9) {
        bump(&self.calls.set_light);
        ok(self.h.set_light(index, light), "SetLight");
    }

    fn light_enable(&self, index: u32, enable: bool) {
        bump(&self.calls.light_enable);
        ok(self.h.light_enable(index, enable), "LightEnable");
    }

    fn vs(&self, shader: Option<&VertexShader<'_>>) {
        bump(&self.calls.set_vertex_shader);
        let hr = shader.map_or_else(
            || self.h.clear_vertex_shader(),
            |shader| self.h.set_vertex_shader(shader),
        );
        ok(hr, "SetVertexShader");
    }

    fn ps(&self, shader: Option<&PixelShader<'_>>) {
        bump(&self.calls.set_pixel_shader);
        let hr = shader.map_or_else(
            || self.h.clear_pixel_shader(),
            |shader| self.h.set_pixel_shader(shader),
        );
        ok(hr, "SetPixelShader");
    }

    fn vs_const(&self, start: u32, rows: &[f32]) {
        bump(&self.calls.set_vs_const);
        ok(
            self.h.set_vertex_shader_constant_f(start, rows),
            "SetVertexShaderConstantF",
        );
    }

    fn ps_const(&self, start: u32, rows: &[f32]) {
        bump(&self.calls.set_ps_const);
        ok(
            self.h.set_pixel_shader_constant_f(start, rows),
            "SetPixelShaderConstantF",
        );
    }

    fn fvf(&self, fvf: u32) {
        bump(&self.calls.set_fvf);
        ok(self.h.set_fvf(fvf), "SetFVF");
    }

    fn decl(&self, decl: &VertexDeclaration<'_>) {
        bump(&self.calls.set_vertex_declaration);
        ok(self.h.set_vertex_declaration(decl), "SetVertexDeclaration");
    }

    fn stream(&self, vb: &VertexBuffer<'_>, stride: u32) {
        bump(&self.calls.set_stream_source);
        ok(
            self.h.set_stream_source(0, vb, 0, stride),
            "SetStreamSource",
        );
    }

    fn indices(&self, ib: &IndexBuffer<'_>) {
        bump(&self.calls.set_indices);
        ok(self.h.set_indices(ib), "SetIndices");
    }

    /// An indexed triangle-list draw from the bound stream and indices.
    fn draw(&self, base: i32, vertices: u32, start: u32, triangles: u32) {
        bump(&self.calls.draw);
        ok(
            self.h
                .draw_indexed_primitive(D3DPT_TRIANGLELIST, base, 0, vertices, start, triangles),
            "DrawIndexedPrimitive",
        );
    }
}

/// The terrain vertex buffer, [`TILE_CHUNKS`] chunks one after another, and their index list.
fn terrain_buffers(h: &Harness) -> (VertexBuffer<'_>, IndexBuffer<'_>) {
    let (cells, indices) = grid(u16::try_from(TERRAIN_GRID).expect("terrain grid fits u16"));
    let mut vertices: Vec<[f32; 10]> = Vec::new();
    for chunk in 0..TILE_CHUNKS {
        for (at, cell) in (0..).zip(&cells) {
            let height = ratio((at * 7 + chunk * 3) % 11, 11) * 0.01;
            vertices.push([
                cell.x,
                cell.y,
                height,
                0.0,
                0.0,
                -1.0,
                cell.u * 4.0,
                cell.v * 4.0,
                cell.u,
                cell.v,
            ]);
        }
    }
    (static_buffer(h, &vertices), static_indices(h, &indices))
}

/// The model mesh: a grid of normals facing the viewer, and its index list.
fn model_mesh() -> (Vec<[f32; 8]>, Vec<u16>) {
    let (cells, indices) = grid(u16::try_from(MODEL_GRID).expect("model grid fits u16"));
    let vertices = cells
        .iter()
        .map(|cell| [cell.x, cell.y, cell.z, 0.0, 0.0, -1.0, cell.u, cell.v])
        .collect();
    (vertices, indices)
}

/// A managed, write-only vertex buffer holding `vertices`.
fn static_buffer<'h, T: Copy>(h: &'h Harness, vertices: &[T]) -> VertexBuffer<'h> {
    let bytes = u32::try_from(size_of_val(vertices)).expect("vertex bytes fit u32");
    let vb = h.create_vertex_buffer(bytes, D3DUSAGE_WRITEONLY, 0, D3DPOOL_MANAGED);
    vb.lock(0, 0, 0).write(vertices);
    vb
}

/// A managed, write-only 16-bit index buffer holding `indices`.
fn static_indices<'h>(h: &'h Harness, indices: &[u16]) -> IndexBuffer<'h> {
    let bytes = u32::try_from(size_of_val(indices)).expect("index bytes fit u32");
    let ib = h.create_index_buffer(bytes, D3DUSAGE_WRITEONLY, D3DFMT_INDEX16, D3DPOOL_MANAGED);
    ib.lock(0, 0, 0).write(indices);
    ib
}

/// A full-target quad in clip space, texture coordinates top-down.
fn screen_quad(h: &Harness) -> VertexBuffer<'_> {
    let corner = |x: f32, y: f32| TexturedVertex {
        x,
        y,
        z: 0.5,
        color: 0xFFFF_FFFF,
        u: f32::midpoint(x, 1.0),
        v: (1.0 - y) / 2.0,
    };
    static_buffer(
        h,
        &[
            corner(-1.0, 1.0),
            corner(1.0, 1.0),
            corner(-1.0, -1.0),
            corner(1.0, -1.0),
        ],
    )
}

/// A managed `edge`x`edge` A8R8G8B8 texture, one level, `texel` giving each texel by position.
fn filled_texture(h: &Harness, edge: u32, texel: impl Fn(u32, u32) -> u32) -> Texture<'_> {
    let texture = h.create_texture(edge, edge, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
    let texels: Vec<u32> = (0..edge * edge)
        .map(|at| texel(at % edge, at / edge))
        .collect();
    let side = usize::try_from(edge).expect("texture edge fits usize");
    texture.lock_rect(0, 0).write_u32_rect(side, side, &texels);
    texture
}

/// A 256x256 terrain layer: a checker of `color` and a darker shade.
fn layer_texture(h: &Harness, color: u32) -> Texture<'_> {
    filled_texture(h, LAYER_EDGE, |col, row| {
        if (col / 16 + row / 16) % 2 == 0 {
            color
        } else {
            (color >> 1) & 0x7F7F_7F7F | 0xFF00_0000
        }
    })
}

/// A 64x64 alpha map: white, with the blend weight in alpha, a gradient that differs per map.
fn alpha_map(h: &Harness, seed: u32) -> Texture<'_> {
    filled_texture(h, ALPHA_EDGE, |col, row| {
        ((col * 4 + row * 2 + seed * 29) % 256) << 24 | 0x00FF_FFFF
    })
}

/// The managed font atlas: glyph-like cells, their alpha keyed on and off.
fn font_atlas(h: &Harness) -> Texture<'_> {
    filled_texture(h, FONT_EDGE, |col, row| {
        if (col % 8 + row % 8) % 5 == 0 {
            0x00FF_FFFF
        } else {
            0xFFFF_FFFF
        }
    })
}

/// A managed 64x64 DXT3 UI texture, one level, its colours and alpha rows set by `seed`.
fn dxt3_texture(h: &Harness, seed: u32) -> Texture<'_> {
    let texture = h.create_texture(UI_EDGE, UI_EDGE, 1, 0, D3DFMT_DXT3, D3DPOOL_MANAGED);
    let blocks = UI_EDGE / 4;
    let mut bytes = Vec::new();
    for block in 0..blocks * blocks {
        // Explicit alpha, four bits a texel: the first block row of every
        // four is half transparent, still over the UI's alpha reference.
        let alpha: u8 = if (block / blocks + seed).is_multiple_of(4) {
            0x99
        } else {
            0xFF
        };
        bytes.extend_from_slice(&[alpha; 8]);
        let red = u16::try_from((seed * 5 + block) % 32).expect("a 5-bit channel fits u16");
        let light = red << 11 | 0x07E0 | 0x0010;
        let dark = light >> 2 & 0x39E7;
        bytes.extend_from_slice(&light.to_le_bytes());
        bytes.extend_from_slice(&dark.to_le_bytes());
        bytes.extend_from_slice(&0x1B1B_E4E4_u32.to_le_bytes());
    }
    let row_bytes = usize::try_from(blocks * 16).expect("block row fits usize");
    let rows = usize::try_from(blocks).expect("block rows fit usize");
    texture
        .lock_rect(0, 0)
        .write_u8_rect(row_bytes, rows, &bytes);
    texture
}

/// The four vertices of UI quad `quad`, in clip space, in a grid over the screen.
fn ui_vertices(quad: u32) -> [TexturedVertex; 4] {
    let left = ratio(quad % 24, 24).mul_add(1.95, -0.98);
    let top = ratio(quad / 24, 10).mul_add(-1.9, 0.96);
    let corner = |dx: f32, dy: f32| TexturedVertex {
        x: dx.mul_add(0.075, left),
        y: dy.mul_add(-0.17, top),
        z: 0.5,
        color: 0xE0FF_FFFF,
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

/// The fixed-function world matrix of terrain chunk `chunk`, the chunks tiling the screen.
fn chunk_world(chunk: u32, tick: u32) -> [f32; 16] {
    let edge = ratio(2, TERRAIN_COLUMNS);
    let drift = ratio(tick % 64, 64) * 0.004;
    let x = ratio(chunk % TERRAIN_COLUMNS, TERRAIN_COLUMNS).mul_add(2.0, -1.0) - drift;
    let y = ratio(chunk / TERRAIN_COLUMNS + 1, TERRAIN_COLUMNS).mul_add(-2.0, 1.0);
    ff_world(edge, x, y, TERRAIN_DEPTH)
}

/// Scale and clip-space position of the `at`-th model in frame `tick`.
fn model_place(at: u32, tick: u32) -> (f32, f32, f32, f32) {
    let drift = ratio(tick % 64, 64) * 0.01;
    (
        ratio(at % 5, 5).mul_add(0.05, 0.08),
        ratio((at * 37) % 100, 100).mul_add(1.8, -0.95) + drift,
        ratio((at * 61) % 100, 100).mul_add(1.8, -0.95),
        ratio((at * 13) % 97, 97).mul_add(0.7, 0.1),
    )
}

/// The fixed-function world matrix (row vectors): scale in x and y, then a translation.
const fn ff_world(scale: f32, x: f32, y: f32, z: f32) -> [f32; 16] {
    [
        scale, 0.0, 0.0, 0.0, //
        0.0, scale, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, //
        x, y, z, 1.0,
    ]
}

/// The four tap offsets `c0..c3` of glow pass `pass`, spreading wider each pass.
fn taps(pass: u32) -> [f32; 16] {
    let step = ratio(pass + 1, GLOW_WIDTH);
    [
        -step, -step, 0.0, 0.0, //
        step, -step, 0.0, 0.0, //
        -step, step, 0.0, 0.0, //
        step, step, 0.0, 0.0,
    ]
}

/// The material of the `shade`-th of sixteen shades, lit by the sun and the ambient.
fn material(shade: u32) -> D3DMATERIAL9 {
    let level = ratio(shade, 16);
    D3DMATERIAL9 {
        diffuse: rgb(level.mul_add(0.4, 0.6), 0.9, level.mul_add(-0.3, 0.9)),
        ambient: rgb(0.5, 0.5, 0.5),
        specular: D3DCOLORVALUE::default(),
        emissive: rgb(0.05, 0.05, 0.05),
        power: 0.0,
    }
}

/// The directional sun, shining into the screen onto normals that face the viewer.
fn sun() -> D3DLIGHT9 {
    D3DLIGHT9 {
        diffuse: rgb(0.9, 0.85, 0.75),
        ambient: rgb(0.2, 0.2, 0.25),
        ..D3DLIGHT9::default()
    }
}

/// A model's local point light, just in front of it.
fn local_light(x: f32, y: f32, z: f32) -> D3DLIGHT9 {
    D3DLIGHT9 {
        type_: D3DLIGHT_POINT,
        diffuse: rgb(1.0, 0.7, 0.4),
        position: D3DVECTOR {
            x: x + 0.05,
            y: y + 0.05,
            z: z - 0.1,
        },
        range: 0.6,
        attenuation0: 1.0,
        attenuation1: 2.0,
        ..D3DLIGHT9::default()
    }
}

/// An opaque colour.
const fn rgb(red: f32, green: f32, blue: f32) -> D3DCOLORVALUE {
    D3DCOLORVALUE {
        r: red,
        g: green,
        b: blue,
        a: 1.0,
    }
}

/// The `ps_2_0` terrain program: three layers blended through two alpha maps, lit.
///
/// `t0` is the layers' tiled coordinate and `t1` the alpha maps'; `v0` is
/// the fixed vertex pipeline's lit diffuse. `def c7` is `tint` in `x`, so
/// distinct tints are distinct programs.
#[rustfmt::skip]
fn terrain_ps(tint: f32) -> Vec<u32> {
    let mut tokens = vec![
        0xFFFF_0200,                           // ps_2_0
        0x0200_001F, 0x8000_0000, 0xB00F_0000, // dcl t0
        0x0200_001F, 0x8000_0000, 0xB00F_0001, // dcl t1
        0x0200_001F, 0x8000_0000, 0x900F_0000, // dcl v0
    ];
    for sampler in 0..5 {
        tokens.extend_from_slice(&[0x0200_001F, 0x9000_0000, 0xA00F_0800 + sampler]); // dcl_2d sN
    }
    tokens.extend_from_slice(&def(0xA00F_0007, [tint, 0.0, 0.0, 0.0])); // def c7
    tokens.extend_from_slice(&[
        0x0300_0042, 0x800F_0000, 0xB0E4_0000, 0xA0E4_0800, // texld r0, t0, s0
        0x0300_0042, 0x800F_0001, 0xB0E4_0000, 0xA0E4_0801, // texld r1, t0, s1
        0x0300_0042, 0x800F_0002, 0xB0E4_0000, 0xA0E4_0802, // texld r2, t0, s2
        0x0300_0042, 0x800F_0003, 0xB0E4_0001, 0xA0E4_0803, // texld r3, t1, s3
        0x0300_0042, 0x800F_0004, 0xB0E4_0001, 0xA0E4_0804, // texld r4, t1, s4
        0x0400_0012, 0x800F_0005, 0x80FF_0003, 0x80E4_0001, 0x80E4_0000, // lrp r5, r3.w, r1, r0
        0x0400_0012, 0x800F_0000, 0x80FF_0004, 0x80E4_0002, 0x80E4_0005, // lrp r0, r4.w, r2, r5
        0x0300_0005, 0x800F_0000, 0x80E4_0000, 0x90E4_0000,              // mul r0, r0, v0
        0x0400_0004, 0x800F_0000, 0x80E4_0000, 0xA0E4_0000, 0xA0E4_0007, // mad r0, r0, c0, c7
        0x0200_0001, 0x800F_0800, 0x80E4_0000,                           // mov oC0, r0
        0x0000_FFFF,
    ]);
    tokens
}

/// The `vs_2_0` model program, whose output the fixed texture stages take.
///
/// `oPos` is `c0..c3` applied to `c4..c7` applied to the position (see
/// [`transform`]); `oD0` is the sun from `c90..c92` on the normal; `oT0` is
/// the coordinate shifted by `def c95`, which is `variant` in `x`.
#[rustfmt::skip]
fn model_vs(variant: f32) -> Vec<u32> {
    let mut tokens = vec![
        0xFFFE_0200,                           // vs_2_0
        0x0200_001F, 0x8000_0000, 0x900F_0000, // dcl_position v0
        0x0200_001F, 0x8000_0003, 0x900F_0001, // dcl_normal v1
        0x0200_001F, 0x8000_0005, 0x900F_0002, // dcl_texcoord0 v2
    ];
    tokens.extend_from_slice(&def(0xA00F_005F, [variant, 0.0, 0.0, 0.0])); // def c95
    tokens.extend_from_slice(&transform(0xC000_0000)); // oPos
    tokens.extend_from_slice(&[
        0x0300_0008, 0x8001_0001, 0x90E4_0001, 0xA0E4_005A,              // dp3 r1.x, v1, c90
        0x0300_000B, 0x8001_0001, 0x8000_0001, 0xA055_005F,              // max r1.x, r1.x, c95.y
        0x0400_0004, 0xD00F_0000, 0x8000_0001, 0xA0E4_005B, 0xA0E4_005C, // mad oD0, r1.x, c91, c92
        0x0300_0002, 0xE00F_0000, 0x90E4_0002, 0xA0E4_005F,              // add oT0, v2, c95
        0x0000_FFFF,
    ]);
    tokens
}

/// The liquids' `vs_2_0` program: two scrolled coordinates and the ambient row as colour.
///
/// `def c95` is `variant` in `x`, the scroll, and a half in `y`, the second
/// coordinate's scale.
#[rustfmt::skip]
fn liquid_vs(variant: f32) -> Vec<u32> {
    let mut tokens = vec![
        0xFFFE_0200,                           // vs_2_0
        0x0200_001F, 0x8000_0000, 0x900F_0000, // dcl_position v0
        0x0200_001F, 0x8000_0005, 0x900F_0001, // dcl_texcoord0 v1
    ];
    tokens.extend_from_slice(&def(0xA00F_005F, [variant, 0.5, 0.0, 0.0])); // def c95
    tokens.extend_from_slice(&transform(0xC000_0000)); // oPos
    tokens.extend_from_slice(&[
        0x0300_0002, 0xE00F_0000, 0x90E4_0001, 0xA0E4_005F, // add oT0, v1, c95
        0x0300_0005, 0xE00F_0001, 0x90E4_0001, 0xA055_005F, // mul oT1, v1, c95.y
        0x0200_0001, 0xD00F_0000, 0xA0E4_005C,              // mov oD0, c92
        0x0000_FFFF,
    ]);
    tokens
}

/// The liquids' `ps_2_0` program: two textures multiplied, times the colour, then `mad` with `c0`.
///
/// `def c7` is `tint` in `x`, so distinct tints are distinct programs.
#[rustfmt::skip]
fn liquid_ps(tint: f32) -> Vec<u32> {
    let mut tokens = vec![
        0xFFFF_0200,                           // ps_2_0
        0x0200_001F, 0x8000_0000, 0xB00F_0000, // dcl t0
        0x0200_001F, 0x8000_0000, 0xB00F_0001, // dcl t1
        0x0200_001F, 0x8000_0000, 0x900F_0000, // dcl v0
        0x0200_001F, 0x9000_0000, 0xA00F_0800, // dcl_2d s0
        0x0200_001F, 0x9000_0000, 0xA00F_0801, // dcl_2d s1
    ];
    tokens.extend_from_slice(&def(0xA00F_0007, [tint, 0.0, 0.0, 0.0])); // def c7
    tokens.extend_from_slice(&[
        0x0300_0042, 0x800F_0000, 0xB0E4_0000, 0xA0E4_0800,              // texld r0, t0, s0
        0x0300_0042, 0x800F_0001, 0xB0E4_0001, 0xA0E4_0801,              // texld r1, t1, s1
        0x0300_0005, 0x800F_0000, 0x80E4_0000, 0x80E4_0001,              // mul r0, r0, r1
        0x0300_0005, 0x800F_0000, 0x80E4_0000, 0x90E4_0000,              // mul r0, r0, v0
        0x0400_0004, 0x800F_0000, 0x80E4_0000, 0xA0E4_0000, 0xA0E4_0007, // mad r0, r0, c0, c7
        0x0200_0001, 0x800F_0800, 0x80E4_0000,                           // mov oC0, r0
        0x0000_FFFF,
    ]);
    tokens
}

/// The glow's `vs_2_0` program: the screen quad as is, its coordinate plus the taps `c0..c3`.
#[rustfmt::skip]
fn glow_vs() -> Vec<u32> {
    vec![
        0xFFFE_0200,                                        // vs_2_0
        0x0200_001F, 0x8000_0000, 0x900F_0000,              // dcl_position v0
        0x0200_001F, 0x8000_0005, 0x900F_0001,              // dcl_texcoord0 v1
        0x0200_0001, 0xC00F_0000, 0x90E4_0000,              // mov oPos, v0
        0x0300_0002, 0xE00F_0000, 0x90E4_0001, 0xA0E4_0000, // add oT0, v1, c0
        0x0300_0002, 0xE00F_0001, 0x90E4_0001, 0xA0E4_0001, // add oT1, v1, c1
        0x0300_0002, 0xE00F_0002, 0x90E4_0001, 0xA0E4_0002, // add oT2, v1, c2
        0x0300_0002, 0xE00F_0003, 0x90E4_0001, 0xA0E4_0003, // add oT3, v1, c3
        0x0000_FFFF,
    ]
}

/// The glow's `ps_2_0` program: four taps of `s0`, summed and weighted by `c0`.
#[rustfmt::skip]
fn glow_ps() -> Vec<u32> {
    vec![
        0xFFFF_0200,                                        // ps_2_0
        0x0200_001F, 0x8000_0000, 0xB00F_0000,              // dcl t0
        0x0200_001F, 0x8000_0000, 0xB00F_0001,              // dcl t1
        0x0200_001F, 0x8000_0000, 0xB00F_0002,              // dcl t2
        0x0200_001F, 0x8000_0000, 0xB00F_0003,              // dcl t3
        0x0200_001F, 0x9000_0000, 0xA00F_0800,              // dcl_2d s0
        0x0300_0042, 0x800F_0000, 0xB0E4_0000, 0xA0E4_0800, // texld r0, t0, s0
        0x0300_0042, 0x800F_0001, 0xB0E4_0001, 0xA0E4_0800, // texld r1, t1, s0
        0x0300_0042, 0x800F_0002, 0xB0E4_0002, 0xA0E4_0800, // texld r2, t2, s0
        0x0300_0042, 0x800F_0003, 0xB0E4_0003, 0xA0E4_0800, // texld r3, t3, s0
        0x0300_0002, 0x800F_0000, 0x80E4_0000, 0x80E4_0001, // add r0, r0, r1
        0x0300_0002, 0x800F_0002, 0x80E4_0002, 0x80E4_0003, // add r2, r2, r3
        0x0300_0002, 0x800F_0000, 0x80E4_0000, 0x80E4_0002, // add r0, r0, r2
        0x0300_0005, 0x800F_0000, 0x80E4_0000, 0xA0E4_0000, // mul r0, r0, c0
        0x0200_0001, 0x800F_0800, 0x80E4_0000,              // mov oC0, r0
        0x0000_FFFF,
    ]
}

/// The composite's `ps_2_0` program: the scene in `s0` plus the glow in `s1` weighted by `c0`.
#[rustfmt::skip]
fn composite_ps() -> Vec<u32> {
    vec![
        0xFFFF_0200,                                                     // ps_2_0
        0x0200_001F, 0x8000_0000, 0xB00F_0000,                           // dcl t0
        0x0200_001F, 0x9000_0000, 0xA00F_0800,                           // dcl_2d s0
        0x0200_001F, 0x9000_0000, 0xA00F_0801,                           // dcl_2d s1
        0x0300_0042, 0x800F_0000, 0xB0E4_0000, 0xA0E4_0800,              // texld r0, t0, s0
        0x0300_0042, 0x800F_0001, 0xB0E4_0000, 0xA0E4_0801,              // texld r1, t0, s1
        0x0400_0004, 0x800F_0000, 0x80E4_0001, 0xA0E4_0000, 0x80E4_0000, // mad r0, r1, c0, r0
        0x0200_0001, 0x800F_0800, 0x80E4_0000,                           // mov oC0, r0
        0x0000_FFFF,
    ]
}

/// The first and one-past-last of `total` items in part `index` of `parts` near-equal parts.
const fn split(total: u32, parts: u32, index: u32) -> (u32, u32) {
    (total * index / parts, total * (index + 1) / parts)
}

/// `value` as an index into one of the frame's resource lists.
fn slot(value: u32) -> usize {
    usize::try_from(value).expect("a resource index fits usize")
}

fn bump(counter: &Cell<u32>) {
    counter.set(counter.get() + 1);
}
