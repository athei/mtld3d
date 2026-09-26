//! A synthetic frame shaped like World of Warcraft 3.3.5a's busy frame, timed over many frames.
//!
//! The shape follows a frame the game drew under the layer, taken from the
//! layer's F12 `[dump]` of one 2026-09 session (three dumped frames, each
//! the same 1323 draws), and the call mix follows a busy five-second window
//! of the layer's `PERF=1` summary from another session of the same build
//! (601 frames of about 1420 draws and ten passes each).
//!
//! One frame is 1323 draws in ten render passes. Five shadow-cascade passes
//! of 8, 120, 158, 382 and 143 caster draws render into one shared 2048x2048
//! A8R8G8B8 colour target with colour writes off, each beside a D24X8 depth
//! texture of its own; binding the targets resets the scissor rectangle, so
//! each pass clears both over the whole target and then draws under a
//! scissor band of its own. The targets are bound through the back buffer
//! the way the game does, six target binds a cascade. About a third of the
//! casters sample a DXT texture and `texkill` on its alpha. The scene pass
//! on the back buffer, which the frame clears before the cascades, is 411
//! draws in the dumped frame's order: opaque runs, a block alternating a
//! source-alpha-times-zero pair with an inverse-source-alpha-plus-one pair,
//! three fixed-function sky draws in a viewport of their own depth range,
//! long opaque runs, middle runs of which 14 draws also sample a DXT1
//! detail texture on stage 1, and a blended tail of 146 draws cycling an
//! alpha-blended pair with two additive ones, the last of them particles
//! from a dynamic vertex buffer. Its 408 programmable draws sample a DXT1,
//! DXT5, DXT3 or (four of them) uncompressed texture on stage 0 and four
//! cascade depth textures on stages 4 to 7, read as hardware shadow maps
//! (stage 4 alternates between the two nearest cascades), 2056 textures in
//! all with the sky's, through 27 pairs of 17 vertex and 17 pixel programs
//! in 214 runs; 179 draws blend, 104 of them additively. A 1:1 `StretchRect`
//! copies the back buffer; three one-draw glow passes at a quarter of its
//! size ping-pong between two targets beside a 2048x2048 D24S8 surface,
//! through the fixed-function vertex pipeline and four-tap pixel programs;
//! the UI pass on the back buffer is the glow composite, 95 programmable
//! quads, each written into the dynamic buffer with its own
//! `D3DLOCK_NOOVERWRITE` lock, and two fixed-function minimap draws in a
//! viewport of their own, then `Present`. Where D24X8 depth textures are
//! not offered, the cascades render into D24X8 surfaces and the scene
//! samples a stand-in DXT texture on those stages instead; the metrics file
//! says which with `meta <bench> depth_path d24x8` or `dxt_standin`.
//!
//! Per draw the frame makes about 0.64 `SetTexture`, 0.63
//! `SetVertexShaderConstantF`, 0.12 `SetPixelShaderConstantF`, 0.29
//! `SetRenderState`, 0.23 `SetSamplerState`, 1.01 stream and index binds
//! and 0.17 and 0.12 vertex and pixel program binds (the game's window:
//! 0.65, 0.65, 0.14, 0.28, 0.25, 1.08, 0.15 and 0.15), and per frame 44
//! target binds, as the game does, about 40 declaration binds and about
//! 150 buffer locks, with no queries and no `SetFVF`. Programs,
//! declarations and render states are set only when they change; the blend
//! factors change while blending is off, so opaque draws carry stale ones.
//! Every program is created up front and first drawn in the warm-up, so the
//! measured frames compile nothing.
//!
//! The metrics file carries one `shape` record per pass, computed from the
//! constants and the run list below rather than read back from the layer.

use std::{cell::Cell, time::SystemTime};

use mtld3d_tests::{
    Harness, HarnessConfig, IndexBuffer, MemorySample, PixelShader, Surface, Texture,
    TexturedVertex, VertexBuffer, VertexDeclaration, VertexShader,
};
use mtld3d_types::{
    D3D_OK, D3DBLEND_INVSRCALPHA, D3DBLEND_ONE, D3DBLEND_SRCALPHA, D3DBLEND_ZERO, D3DCLEAR_TARGET,
    D3DCLEAR_ZBUFFER, D3DCMP_ALWAYS, D3DCMP_LESSEQUAL, D3DCULL_CW, D3DCULL_NONE, D3DFMT_A4R4G4B4,
    D3DFMT_A8R8G8B8, D3DFMT_D24S8, D3DFMT_D24X8, D3DFMT_DXT1, D3DFMT_DXT3, D3DFMT_DXT5,
    D3DFMT_INDEX16, D3DFMT_X8R8G8B8, D3DLOCK_DISCARD, D3DLOCK_NOOVERWRITE, D3DPOOL_DEFAULT,
    D3DPOOL_MANAGED, D3DPRESENT_INTERVAL_IMMEDIATE, D3DPT_TRIANGLELIST, D3DRECT,
    D3DRS_ALPHABLENDENABLE, D3DRS_ALPHATESTENABLE, D3DRS_COLORWRITEENABLE, D3DRS_CULLMODE,
    D3DRS_DESTBLEND, D3DRS_LIGHTING, D3DRS_SCISSORTESTENABLE, D3DRS_SRCBLEND, D3DRS_ZENABLE,
    D3DRS_ZFUNC, D3DRS_ZWRITEENABLE, D3DRTYPE_TEXTURE, D3DSAMP_ADDRESSU, D3DSAMP_ADDRESSV,
    D3DSAMP_MAGFILTER, D3DSAMP_MINFILTER, D3DSAMP_MIPFILTER, D3DTA_CURRENT, D3DTA_DIFFUSE,
    D3DTA_TEXTURE, D3DTADDRESS_CLAMP, D3DTADDRESS_WRAP, D3DTEXF_LINEAR, D3DTEXF_NONE,
    D3DTEXF_POINT, D3DTOP_DISABLE, D3DTOP_MODULATE, D3DTOP_SELECTARG1, D3DTS_PROJECTION,
    D3DTS_VIEW, D3DTS_WORLD, D3DTSS_COLORARG1, D3DTSS_COLORARG2, D3DTSS_COLOROP,
    D3DTSS_TEXCOORDINDEX, D3DUSAGE_DEPTHSTENCIL, D3DUSAGE_DYNAMIC, D3DUSAGE_RENDERTARGET,
    D3DUSAGE_WRITEONLY, D3DVIEWPORT9,
};

use crate::bench::{
    Class, Direction, FrameClock, FrameWork, IDENTITY_ROWS, LayerLog, Metrics, Model, PassShape,
    STRIDE, TEXTURED_DECL, TscClock, Value, def, grid, material_ps, material_vs, memory_section,
    ok, pattern_texture, ratio, rs, transform, world_rows, write_report,
};

/// The back buffer, about the size of a windowed game.
const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
/// Edge of the cascades' shared colour target and of each cascade's depth texture.
const SHADOW_EDGE: u32 = 2048;
const CASCADES: usize = 5;
/// Each cascade's caster runs in draw order: caster vertex program, caster pixel program, draws.
///
/// Pixel program [`TEXTURED_CASTER`] samples a texture and `texkill`s on its
/// alpha; the other writes a constant colour the masked colour writes drop.
/// The runs are the dumped frame's, program for program.
const CASCADE_RUNS: [&[(usize, usize, u32)]; CASCADES] = [
    &[(0, 0, 6), (1, 0, 1), (1, 1, 1)],
    &[
        (2, 0, 1),
        (2, 1, 18),
        (2, 0, 8),
        (0, 0, 53),
        (1, 0, 17),
        (0, 1, 15),
        (1, 1, 8),
    ],
    &[
        (2, 0, 2),
        (2, 1, 18),
        (2, 0, 8),
        (0, 0, 67),
        (1, 0, 28),
        (0, 1, 20),
        (1, 1, 15),
    ],
    &[
        (2, 0, 8),
        (2, 1, 18),
        (2, 0, 29),
        (0, 0, 163),
        (1, 0, 53),
        (0, 1, 71),
        (1, 1, 40),
    ],
    &[(2, 0, 6), (2, 1, 18), (2, 0, 34), (0, 0, 50), (0, 1, 35)],
];
/// The caster pixel program that samples a texture.
const TEXTURED_CASTER: usize = 1;
/// Each cascade's scissor band in its 2048x2048 targets, the dumped frame's.
const CASCADE_SCISSOR: [D3DRECT; CASCADES] = [
    rect(0, 773, 2048, 1275),
    rect(0, 773, 2048, 1275),
    rect(0, 897, 2048, 1151),
    rect(0, 986, 2048, 1062),
    rect(658, 1022, 1390, 1026),
];
/// Edge of the glow targets, a quarter of the back buffer's.
const GLOW_WIDTH: u32 = WIDTH / 4;
const GLOW_HEIGHT: u32 = HEIGHT / 4;
const SCENE_VS: usize = 17;
const SCENE_PS: usize = 17;
/// The scene's program pairs as (vertex, pixel) program indices, the dumped frame's 27.
const SCENE_PAIRS: [(usize, usize); 27] = [
    (0, 0),
    (0, 1),
    (1, 0),
    (2, 1),
    (2, 0),
    (3, 2),
    (3, 3),
    (4, 4),
    (4, 5),
    (5, 4),
    (5, 5),
    (6, 6),
    (6, 7),
    (7, 7),
    (8, 8),
    (9, 9),
    (9, 10),
    (9, 11),
    (9, 12),
    (10, 13),
    (11, 14),
    (12, 15),
    (13, 6),
    (14, 15),
    (15, 16),
    (10, 15),
    (16, 4),
];
/// The pixel programs that also sample a DXT1 detail texture on stage 1.
const DETAIL_PS: [usize; 6] = [8, 9, 10, 11, 12, 14];
/// The pairs whose draws alpha-test.
const ALPHA_TESTED_PAIRS: [usize; 5] = [20, 21, 23, 24, 25];
/// The alpha-blended pair that still writes depth.
const DEPTH_WRITING_ALPHA_PAIR: usize = 24;
/// The opaque runs that open the scene: pair, draws.
const HEAD_RUNS: [(usize, u32); 11] = [
    (0, 4),
    (2, 1),
    (0, 1),
    (2, 3),
    (0, 2),
    (2, 1),
    (1, 1),
    (2, 1),
    (1, 1),
    (0, 1),
    (1, 3),
];
/// The pairs of the modulate block: source alpha times zero, then inverse source alpha plus one.
const MODULATE_PAIR: usize = 3;
const INVERSE_ADD_PAIR: usize = 5;
/// The modulate block: alternations of its two pairs, then draws of the second with blending off.
const MODULATE_BLOCKS: [(u32, u32); 3] = [(5, 4), (7, 7), (2, 3)];
/// The long opaque runs after the sky: pair, draws.
const LONG_RUNS: [(usize, u32); 6] = [(7, 19), (8, 55), (9, 5), (10, 15), (11, 16), (12, 56)];
/// The runs between the long ones and the blended tail: pair, draws, alpha-blended.
const MIDDLE_RUNS: [(usize, u32, bool); 20] = [
    (13, 1, false),
    (14, 2, false),
    (15, 3, false),
    (16, 4, false),
    (17, 1, false),
    (18, 3, false),
    (19, 1, false),
    (7, 1, true),
    (20, 1, true),
    (7, 1, false),
    (8, 2, false),
    (7, 1, false),
    (8, 1, false),
    (7, 3, false),
    (9, 1, false),
    (12, 2, false),
    (8, 4, false),
    (12, 1, false),
    (19, 1, false),
    (7, 1, false),
];
/// The blended tail's draws before its closing runs, one run each.
///
/// They cycle through three pairs, an alpha-blended one, an alpha-tested
/// additive one and an additive particle one, with every third cycle
/// leaving out the second; the first and third pairs change after
/// [`TAIL_SWITCH`] draws.
const TAIL_DRAWS: u32 = 143;
const TAIL_CYCLE: [usize; 8] = [0, 1, 2, 0, 1, 2, 0, 2];
const TAIL_PAIRS: [[usize; 3]; 2] = [[11, 21, 22], [7, 21, 26]];
const TAIL_SWITCH: u32 = 37;
/// Tail draws that use [`DEPTH_WRITING_ALPHA_PAIR`] instead of their cycle's pair.
const TAIL_DEPTH_WRITING: [u32; 2] = [29, 142];
/// The runs that close the tail: pair, additive (else alpha-blended).
const TAIL_END: [(usize, bool); 3] = [(25, true), (23, false), (25, true)];
/// Scene draws, counted from the first, whose stage-0 texture is DXT3.
const DXT3_DRAWS: [u32; 12] = [70, 71, 75, 99, 100, 105, 159, 160, 161, 162, 163, 252];
/// Scene draws whose stage-0 texture is uncompressed: two A8R8G8B8, then two A4R4G4B4.
const UNCOMPRESSED_DRAWS: [u32; 4] = [83, 143, 294, 407];
/// The scene draws before the long runs, and the first of the blended tail.
const SCENE_OPENING: u32 = 61;
const SCENE_TAIL: u32 = 265;
const SKY_DRAWS: u32 = 3;
/// Programmable UI quads before the minimap, and after it.
const UI_BEFORE_MINIMAP: u32 = 11;
const UI_AFTER_MINIMAP: u32 = 84;
/// The quads after the minimap drawn with the second UI pixel program.
const UI_SECOND_PS: [u32; 2] = [72, 76];
const MINIMAP_DRAWS: u32 = 2;
/// Quads in one particle draw.
const PARTICLE_QUADS: u32 = 16;
/// Distinct particle vertex sets, picked by run and frame.
const PARTICLE_SETS: u32 = 8;
/// Vertices the dynamic buffer holds; a frame's particles and UI quads fit.
const RING_VERTICES: u32 = 6144;
const SCENE_TEXTURES: u32 = 32;
const DXT3_TEXTURES: u32 = 4;
/// Edge of every DXT texture, in texels.
const DXT_EDGE: u32 = 256;
const UI_TEXTURES: u32 = 8;
/// Grid sizes of the static meshes, one vertex buffer each.
const MESH_GRIDS: [u16; 4] = [4, 6, 8, 10];
const WARM_UP_FRAMES: u32 = 60;
/// The measured phase is at least this many frames and at least one perf window long.
const MEASURED_FRAMES: usize = 600;
/// The scene's viewport; the sky draws alone use the far end of the depth range.
const SCENE_VIEWPORT: D3DVIEWPORT9 = viewport(0, 0, WIDTH, HEIGHT, 0.0, 0.94);
const SKY_VIEWPORT: D3DVIEWPORT9 = viewport(0, 0, WIDTH, HEIGHT, 0.999_023_44, 1.0);
const UI_VIEWPORT: D3DVIEWPORT9 = viewport(0, 0, WIDTH, HEIGHT, 0.0, 1.0);
const MINIMAP_VIEWPORT: D3DVIEWPORT9 = viewport(1128, 18, 128, 128, 0.0, 1.0);
const SHADOW_VIEWPORT: D3DVIEWPORT9 = viewport(0, 0, SHADOW_EDGE, SHADOW_EDGE, 0.0, 1.0);
const GLOW_VIEWPORT: D3DVIEWPORT9 = viewport(0, 0, GLOW_WIDTH, GLOW_HEIGHT, 0.0, 1.0);
/// The scissor rectangle the scene sets with the test off, and the full one the sky draws under.
const SCENE_SCISSOR: D3DRECT = rect(411, 0, 869, 361);
const FULL_SCISSOR: D3DRECT = rect(0, 0, WIDTH.cast_signed(), HEIGHT.cast_signed());
const GLOW_SCISSOR: D3DRECT = rect(0, 0, GLOW_WIDTH.cast_signed(), GLOW_HEIGHT.cast_signed());

/// One busy frame, repeatedly: warm up, then time the frames.
#[test]
#[ignore = "benchmark, run by `make bench`"]
fn wow_335a_busy_frame() {
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
        depth_format: Some(D3DFMT_D24X8),
        presentation_interval: D3DPRESENT_INTERVAL_IMMEDIATE,
        ..HarnessConfig::default()
    });
    let started = TscClock::now();
    let frame = Frame::new(&h);
    for tick in 0..WARM_UP_FRAMES {
        assert!(h.pump(), "WM_QUIT during warm-up");
        frame.render(tick);
        ok(h.present(), "Present");
    }
    let log = LayerLog::find(since);
    let warm_up = TscClock::since(started);
    let warm = MemorySample::now();

    let mut tick = WARM_UP_FRAMES;
    let start = log.start_span(started, || {
        assert!(h.pump(), "WM_QUIT outside the measured frames");
        frame.render(tick);
        ok(h.present(), "Present");
        tick += 1;
    });
    let mut clock = FrameClock::start(MEASURED_FRAMES * 4);
    while clock.frames() < MEASURED_FRAMES || clock.elapsed() < start.length() {
        assert!(h.pump(), "WM_QUIT during the measured frames");
        frame.render(tick);
        clock.present(&h);
        tick += 1;
    }
    let end = MemorySample::now();
    let span = start.end(&log, || {
        assert!(h.pump(), "WM_QUIT outside the measured frames");
        frame.render(tick);
        ok(h.present(), "Present");
        tick += 1;
    });

    let stats = clock.stats();
    let work = clock.work_stats();
    let shadow = if frame.shadows[0].texture.is_some() {
        "D24X8 textures, sampled as shadow maps"
    } else {
        "D24X8 surfaces, D24X8 depth textures not offered"
    };
    let body = format!(
        "shape: back buffer {WIDTH}x{HEIGHT} X8R8G8B8 + D24X8; {CASCADES} cascade depth \
         targets {SHADOW_EDGE}x{SHADOW_EDGE} {shadow}, beside one shared A8R8G8B8 colour \
         target with colour writes off\n\
         per frame: {draws} draws (casters {casters} in {CASCADES} passes, scene {scene} with \
         {SKY_DRAWS} fixed-function sky draws, {detail} detail-textured draws and {particles} \
         particle draws, glow 3 in 3 \
         passes at {GLOW_WIDTH}x{GLOW_HEIGHT}, UI {ui}), one StretchRect, {locks} dynamic-buffer \
         locks\n\
         programs: 3 caster VS + 2 caster PS, {SCENE_VS} scene VS + {SCENE_PS} scene PS in \
         {pairs} pairs over {runs} runs, 2 glow PS + 1 composite PS, 1 UI VS + 2 UI PS; fixed \
         function for the sky, the glow and composite vertices and the minimap\n\
         warm-up: {WARM_UP_FRAMES} frames in {warm_up:.2?}\n\
         measured: {frames} frames in {elapsed:.2?} (at least {MEASURED_FRAMES} frames \
         and {length:?}, {start})\n\
         frame time (Present to Present): {row}\n\
         API work (Present return to Present call): {work}\n{memory}{perf}{warm_up_compiles}",
        draws = DRAWS_PER_FRAME,
        casters = (0..CASCADES).map(cascade_draws).sum::<u32>(),
        scene = SCENE_DRAWS,
        particles = particle_draws(),
        detail = scene_runs()
            .iter()
            .filter(|run| run.detail())
            .map(|run| run.draws)
            .sum::<u32>(),
        ui = UI_DRAWS,
        locks = particle_draws() + UI_BEFORE_MINIMAP + UI_AFTER_MINIMAP,
        pairs = SCENE_PAIRS.len(),
        runs = scene_runs().iter().filter(|run| run.pair.is_some()).count(),
        frames = stats.frames,
        elapsed = clock.elapsed(),
        row = stats.row(),
        work = work.row(),
        memory = memory_section(&before, &warm, &end),
        start = span.start(),
        length = span.length(),
        perf = span.perf_rows(&log).section(),
        warm_up_compiles = log
            .first_window_rows(span.to())
            .map_or_else(String::new, |rows| format!(
                "perf: this device's first window, its warm-up compiles\n{rows}"
            )),
    );
    let mut metrics = Metrics::new("frame_shape", &h, &tsc);
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
    let depth_path = if frame.shadows[0].texture.is_some() {
        "d24x8"
    } else {
        "dxt_standin"
    };
    metrics.meta("depth_path", depth_path);
    metrics.memory(&before, &warm, &end);
    metrics.perf(&span.perf_kv(&log), &FrameWork::Fixed);
    metrics.meta("backbuffer", &format!("{WIDTH}x{HEIGHT}"));
    metrics.shapes(&pass_shapes());
    write_report(&metrics, &log, &body);
}

/// The ten passes of one frame, in the order [`Frame::render`] draws them.
///
/// A draw's textures are the ones its programs or stages sample: one for a
/// textured caster and none for the others, five for a programmable scene
/// draw and six for a detail-textured one, one for a sky draw but the
/// middle one, four for a glow draw, two for the glow composite and the
/// first minimap draw, and one for every other UI draw. The fixed-function
/// vertex draws are the sky, the glow draws, the composite and the minimap;
/// the fixed-function pixel draws are the sky and the minimap.
///
/// # Panics
/// Panics if the passes do not add up to [`DRAWS_PER_FRAME`].
fn pass_shapes() -> Vec<PassShape> {
    let mut passes: Vec<PassShape> = CASCADE_RUNS
        .iter()
        .map(|runs| PassShape {
            width: SHADOW_EDGE,
            height: SHADOW_EDGE,
            draws: runs.iter().map(|&(_, _, draws)| draws).sum(),
            ff_vs: 0,
            ff_ps: 0,
            textures: runs
                .iter()
                .filter(|&&(_, ps, _)| ps == TEXTURED_CASTER)
                .map(|&(_, _, draws)| draws)
                .sum(),
        })
        .collect();

    let scene_textures = scene_runs()
        .iter()
        .map(|run| match run.pair {
            None => run.draws - 1,
            Some(_) => run.draws * (5 + u32::from(run.detail())),
        })
        .sum();
    passes.push(PassShape {
        width: WIDTH,
        height: HEIGHT,
        draws: SCENE_DRAWS,
        ff_vs: SKY_DRAWS,
        ff_ps: SKY_DRAWS,
        textures: scene_textures,
    });
    passes.extend((0..3).map(|_| PassShape {
        width: GLOW_WIDTH,
        height: GLOW_HEIGHT,
        draws: 1,
        ff_vs: 1,
        ff_ps: 0,
        textures: 4,
    }));
    let quads = UI_BEFORE_MINIMAP + UI_AFTER_MINIMAP;
    passes.push(PassShape {
        width: WIDTH,
        height: HEIGHT,
        draws: UI_DRAWS,
        ff_vs: 1 + MINIMAP_DRAWS,
        ff_ps: MINIMAP_DRAWS,
        textures: 2 + quads + 3,
    });

    let draws: u32 = passes.iter().map(|pass| pass.draws).sum();
    assert_eq!(draws, DRAWS_PER_FRAME, "the pass shapes cover every draw");
    passes
}

/// Draws per frame: the casters, the scene, the glow and the UI.
const DRAWS_PER_FRAME: u32 = {
    let mut draws = 0;
    let mut cascade = 0;
    while cascade < CASCADES {
        draws += cascade_draws(cascade);
        cascade += 1;
    }
    draws + SCENE_DRAWS + 3 + UI_DRAWS
};

/// The scene's draws: the opening runs, the sky, the long and middle runs and the tail.
const SCENE_DRAWS: u32 = {
    let mut draws = SKY_DRAWS + TAIL_DRAWS;
    let mut at = 0;
    while at < TAIL_END.len() {
        draws += 1;
        at += 1;
    }
    at = 0;
    while at < HEAD_RUNS.len() {
        draws += HEAD_RUNS[at].1;
        at += 1;
    }
    at = 0;
    while at < MODULATE_BLOCKS.len() {
        draws += MODULATE_BLOCKS[at].0 * 2 + MODULATE_BLOCKS[at].1;
        at += 1;
    }
    at = 0;
    while at < LONG_RUNS.len() {
        draws += LONG_RUNS[at].1;
        at += 1;
    }
    at = 0;
    while at < MIDDLE_RUNS.len() {
        draws += MIDDLE_RUNS[at].1;
        at += 1;
    }
    draws
};

/// The UI pass's draws: the glow composite, the programmable quads and the minimap.
const UI_DRAWS: u32 = 1 + UI_BEFORE_MINIMAP + MINIMAP_DRAWS + UI_AFTER_MINIMAP;

/// How a scene run blends.
enum Blend {
    /// Blending off, the factors left as the last blended run set them.
    Opaque,
    /// Source alpha times zero.
    Modulate,
    /// Inverse source alpha plus one.
    InverseAdd,
    /// Source alpha over inverse source alpha.
    Alpha,
    /// Source alpha plus one.
    Add,
}

impl Blend {
    const fn enabled(&self) -> bool {
        !matches!(self, Self::Opaque)
    }

    /// The source and destination factors, which an opaque run leaves stale.
    const fn factors(&self) -> Option<(u32, u32)> {
        match self {
            Self::Opaque => None,
            Self::Modulate => Some((D3DBLEND_SRCALPHA, D3DBLEND_ZERO)),
            Self::InverseAdd => Some((D3DBLEND_INVSRCALPHA, D3DBLEND_ONE)),
            Self::Alpha => Some((D3DBLEND_SRCALPHA, D3DBLEND_INVSRCALPHA)),
            Self::Add => Some((D3DBLEND_SRCALPHA, D3DBLEND_ONE)),
        }
    }

    /// Whether the run's draws write depth.
    const fn writes_depth(&self, pair: Option<usize>) -> bool {
        match self {
            Self::Opaque | Self::Modulate | Self::InverseAdd => true,
            Self::Alpha | Self::Add => matches!(pair, Some(DEPTH_WRITING_ALPHA_PAIR)),
        }
    }
}

/// A scene run: consecutive draws that share one program pair and one blend.
struct SceneRun {
    /// The run's entry of [`SCENE_PAIRS`], or `None` for the fixed-function sky.
    pair: Option<usize>,
    draws: u32,
    blend: Blend,
    /// Whether the draws are particles, quads written into the dynamic buffer.
    particles: bool,
}

impl SceneRun {
    const fn alpha_test(&self) -> bool {
        match self.pair {
            None => true,
            Some(pair) => {
                let mut at = 0;
                while at < ALPHA_TESTED_PAIRS.len() {
                    if ALPHA_TESTED_PAIRS[at] == pair {
                        return true;
                    }
                    at += 1;
                }
                false
            }
        }
    }

    /// Whether the run samples a detail texture on stage 1.
    fn detail(&self) -> bool {
        self.pair
            .is_some_and(|pair| DETAIL_PS.contains(&SCENE_PAIRS[pair].1))
    }
}

/// The scene's runs in draw order, following the dumped frame's.
///
/// Opaque runs open it, then the modulate block, the fixed-function sky,
/// the long opaque runs, the middle runs with the detail-textured ones among
/// them, and the blended tail.
fn scene_runs() -> Vec<SceneRun> {
    let run = |pair, draws, blend| SceneRun {
        pair: Some(pair),
        draws,
        blend,
        particles: false,
    };
    let mut runs: Vec<SceneRun> = HEAD_RUNS
        .iter()
        .map(|&(pair, draws)| run(pair, draws, Blend::Opaque))
        .collect();
    for &(cycles, opaque) in &MODULATE_BLOCKS {
        for _ in 0..cycles {
            runs.push(run(MODULATE_PAIR, 1, Blend::Modulate));
            runs.push(run(INVERSE_ADD_PAIR, 1, Blend::InverseAdd));
        }
        runs.push(run(INVERSE_ADD_PAIR, opaque, Blend::Opaque));
    }
    runs.push(SceneRun {
        pair: None,
        draws: SKY_DRAWS,
        blend: Blend::Alpha,
        particles: false,
    });
    runs.extend(
        LONG_RUNS
            .iter()
            .map(|&(pair, draws)| run(pair, draws, Blend::Opaque)),
    );
    runs.extend(MIDDLE_RUNS.iter().map(|&(pair, draws, alpha)| {
        run(
            pair,
            draws,
            if alpha { Blend::Alpha } else { Blend::Opaque },
        )
    }));
    for at in 0..TAIL_DRAWS {
        if TAIL_DEPTH_WRITING.contains(&at) {
            runs.push(run(DEPTH_WRITING_ALPHA_PAIR, 1, Blend::Alpha));
            continue;
        }
        let pairs = &TAIL_PAIRS[usize::from(at >= TAIL_SWITCH)];
        let role = TAIL_CYCLE[slot(at) % TAIL_CYCLE.len()];
        let pair = pairs[role];
        runs.push(match role {
            0 => run(pair, 1, Blend::Alpha),
            1 => run(pair, 1, Blend::Add),
            _ => SceneRun {
                particles: true,
                ..run(pair, 1, Blend::Add)
            },
        });
    }
    runs.extend(
        TAIL_END.iter().map(|&(pair, additive)| {
            run(pair, 1, if additive { Blend::Add } else { Blend::Alpha })
        }),
    );
    runs
}

fn particle_draws() -> u32 {
    scene_runs()
        .iter()
        .filter(|run| run.particles)
        .map(|run| run.draws)
        .sum()
}

/// Draws of cascade `cascade`.
const fn cascade_draws(cascade: usize) -> u32 {
    let runs = CASCADE_RUNS[cascade];
    let mut draws = 0;
    let mut at = 0;
    while at < runs.len() {
        draws += runs[at].2;
        at += 1;
    }
    draws
}

/// The cull mode of scene draw `draw` of a run with `pair`.
///
/// The opening draws cull clockwise; two draws in nine of the long and middle
/// runs cull nothing; the tail culls nothing but its closing additive pair.
const fn scene_cull(draw: u32, pair: Option<usize>) -> u32 {
    let none = if draw < SCENE_OPENING {
        false
    } else if draw < SCENE_TAIL {
        matches!(draw % 9, 4 | 5)
    } else {
        !matches!(pair, Some(25))
    };
    if none { D3DCULL_NONE } else { D3DCULL_CW }
}

/// The declaration each scene vertex program reads its vertices through, one of three.
///
/// The opening's programs share one, the middle runs' another, and the long
/// runs' and the tail's the third, but for the tail's first particle program.
const SCENE_DECL: [usize; SCENE_VS] = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 1, 2, 1, 1, 1];

/// The address mode scene pair `pair` samples its stage-0 texture with.
///
/// The tail's alpha-tested additive pair clamps; every other pair wraps.
const fn scene_wrap(pair: usize) -> u32 {
    if pair == 21 {
        D3DTADDRESS_CLAMP
    } else {
        D3DTADDRESS_WRAP
    }
}

/// The index buffer a draw reads.
#[derive(PartialEq, Eq)]
enum Indices {
    Mesh,
    Quads,
}

/// The scene's program, declaration and sampler binds, so each is set only when it changes.
#[derive(Default)]
struct SceneBinds {
    vs: Option<usize>,
    ps: Option<usize>,
    decl: Option<usize>,
    wrap: Option<u32>,
    nearest: Option<bool>,
    indices: Option<Indices>,
}

/// The render states set so far this frame, so a state is set only when its value changes.
struct States<'h> {
    h: &'h Harness,
    set: Vec<(u32, u32)>,
}

impl<'h> States<'h> {
    fn new(h: &'h Harness) -> Self {
        Self {
            h,
            set: Vec::with_capacity(16),
        }
    }

    fn set(&mut self, state: u32, value: u32) {
        match self.set.iter_mut().find(|(known, _)| *known == state) {
            Some((_, current)) if *current == value => return,
            Some((_, current)) => *current = value,
            None => self.set.push((state, value)),
        }
        rs(self.h, state, value);
    }

    /// A scene run's blend, depth-write and alpha-test states.
    fn run(&mut self, run: &SceneRun) {
        let blend = &run.blend;
        self.set(D3DRS_ALPHABLENDENABLE, u32::from(blend.enabled()));
        if let Some((src, dst)) = blend.factors() {
            self.set(D3DRS_SRCBLEND, src);
            self.set(D3DRS_DESTBLEND, dst);
        }
        self.set(D3DRS_ZWRITEENABLE, u32::from(blend.writes_depth(run.pair)));
        self.set(D3DRS_ALPHATESTENABLE, u32::from(run.alpha_test()));
    }
}

/// A static mesh: its own vertex buffer and its range of the shared index buffer.
struct Mesh<'h> {
    vb: VertexBuffer<'h>,
    vertices: u32,
    start_index: u32,
    triangles: u32,
}

/// A cascade's depth target, and the texture behind it when it can be sampled.
struct Shadow<'h> {
    texture: Option<Texture<'h>>,
    surface: Surface<'h>,
}

/// An offscreen colour target that later passes sample.
struct Target<'h> {
    texture: Texture<'h>,
    surface: Surface<'h>,
}

/// A dynamic vertex buffer filled front to back with `NOOVERWRITE` locks, `DISCARD` when it wraps.
struct Ring<'h> {
    vb: VertexBuffer<'h>,
    /// The first vertex not yet written since the last `DISCARD`.
    cursor: Cell<u32>,
}

impl Ring<'_> {
    /// Make the next write the frame's first, which discards the buffer.
    fn restart(&self) {
        self.cursor.set(RING_VERTICES);
    }

    /// Write `vertices` behind the last write, or at the start after a `DISCARD`, and return where.
    fn write(&self, vertices: &[TexturedVertex]) -> u32 {
        let count = u32::try_from(vertices.len()).expect("a ring write fits u32");
        let at = self.cursor.get();
        let (start, flags) = if at + count > RING_VERTICES {
            (0, D3DLOCK_DISCARD)
        } else {
            (at, D3DLOCK_NOOVERWRITE)
        };
        self.vb
            .lock(start * STRIDE, count * STRIDE, flags)
            .write(vertices);
        self.cursor.set(start + count);
        start
    }
}

/// Every resource the frame uses, created once.
struct Frame<'h> {
    h: &'h Harness,
    back_buffer: Surface<'h>,
    scene_depth: Surface<'h>,
    /// The colour target every cascade renders beside, which their draws never write.
    shadow_color: Target<'h>,
    /// One depth target per cascade.
    shadows: Vec<Shadow<'h>>,
    scene_copy: Target<'h>,
    glow: [Target<'h>; 2],
    /// The depth-stencil surface bound beside the glow targets, which no glow draw tests.
    glow_depth: Surface<'h>,
    /// DXT1 and DXT5 textures, alternating.
    textures: Vec<Texture<'h>>,
    dxt3_textures: Vec<Texture<'h>>,
    /// The scene's uncompressed stage-0 textures, one per entry of [`UNCOMPRESSED_DRAWS`].
    uncompressed: [Texture<'h>; 4],
    ui_textures: Vec<Texture<'h>>,
    meshes: Vec<Mesh<'h>>,
    mesh_ib: IndexBuffer<'h>,
    /// Quad indices `0 1 2 2 1 3` repeated, for the particles, the UI and the screen quad.
    quad_ib: IndexBuffer<'h>,
    screen_quad: VertexBuffer<'h>,
    ring: Ring<'h>,
    particle_sets: Vec<Vec<TexturedVertex>>,
    /// Four vertices per programmable UI quad, in draw order.
    ui_quads: Vec<TexturedVertex>,
    /// Three separately created declarations of the one [`TexturedVertex`] layout.
    decls: [VertexDeclaration<'h>; 3],
    runs: Vec<SceneRun>,
    caster_vs: [VertexShader<'h>; 3],
    /// The constant-colour caster program, then the textured one.
    caster_ps: [PixelShader<'h>; 2],
    scene_vs: Vec<VertexShader<'h>>,
    scene_ps: Vec<PixelShader<'h>>,
    /// The downsampling glow program, then the blurring one.
    glow_ps: [PixelShader<'h>; 2],
    composite_ps: PixelShader<'h>,
    ui_vs: VertexShader<'h>,
    ui_ps: [PixelShader<'h>; 2],
}

impl<'h> Frame<'h> {
    fn new(h: &'h Harness) -> Self {
        let back_buffer = h.back_buffer(0);
        let scene_depth = h
            .depth_stencil_surface()
            .expect("the device has an auto depth-stencil");
        let depth_textures = h.check_device_format(
            D3DFMT_X8R8G8B8,
            D3DUSAGE_DEPTHSTENCIL,
            D3DRTYPE_TEXTURE,
            D3DFMT_D24X8,
        ) == D3D_OK;
        let shadows = (0..CASCADES).map(|_| shadow(h, depth_textures)).collect();
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
        let textures = (0..SCENE_TEXTURES)
            .map(|at| {
                let format = if at % 2 == 0 {
                    D3DFMT_DXT1
                } else {
                    D3DFMT_DXT5
                };
                dxt_texture(h, format, at)
            })
            .collect();
        let dxt3_textures = (0..DXT3_TEXTURES)
            .map(|at| dxt_texture(h, D3DFMT_DXT3, at + SCENE_TEXTURES))
            .collect();
        let uncompressed = [
            pattern_texture(h, 0xFF60_4020),
            pattern_texture(h, 0xFF20_6040),
            a4r4g4b4_texture(h, 0xF8A4),
            a4r4g4b4_texture(h, 0xF4A8),
        ];
        let ui_textures = (0..UI_TEXTURES)
            .map(|at| pattern_texture(h, 0x80C0_A080 + at * 0x0003_0507))
            .collect();
        let (meshes, mesh_ib) = meshes(h);
        let quads = UI_BEFORE_MINIMAP + UI_AFTER_MINIMAP;
        Self {
            h,
            back_buffer,
            scene_depth,
            shadow_color: target(SHADOW_EDGE, SHADOW_EDGE),
            shadows,
            scene_copy: target(WIDTH, HEIGHT),
            glow: [0, 1].map(|_| target(GLOW_WIDTH, GLOW_HEIGHT)),
            glow_depth: h.create_depth_stencil_surface(SHADOW_EDGE, SHADOW_EDGE, D3DFMT_D24S8),
            textures,
            dxt3_textures,
            uncompressed,
            ui_textures,
            meshes,
            mesh_ib,
            quad_ib: quad_indices(h),
            screen_quad: screen_quad(h),
            ring: Ring {
                vb: h.create_vertex_buffer(
                    RING_VERTICES * STRIDE,
                    D3DUSAGE_DYNAMIC | D3DUSAGE_WRITEONLY,
                    0,
                    D3DPOOL_DEFAULT,
                ),
                cursor: Cell::new(RING_VERTICES),
            },
            particle_sets: (0..PARTICLE_SETS)
                .map(|set| quads_at(PARTICLE_QUADS, set * 11, 0.03))
                .collect(),
            ui_quads: quads_at(quads, 7, 0.04),
            decls: [0, 1, 2].map(|_| h.create_vertex_declaration(&TEXTURED_DECL)),
            runs: scene_runs(),
            caster_vs: [0.0, 1.0e-4, 2.0e-4]
                .map(|variant| h.create_vertex_shader(&caster_vs(variant))),
            caster_ps: [
                h.create_pixel_shader(&caster_ps()),
                h.create_pixel_shader(&textured_caster_ps()),
            ],
            scene_vs: (0..SCENE_VS)
                .map(|at| {
                    let variant = shade(at, SCENE_VS) * 1.0e-3;
                    h.create_vertex_shader(&material_vs(&Model::Sm2, variant))
                })
                .collect(),
            scene_ps: (0..SCENE_PS)
                .map(|at| {
                    let shade = shade(at, SCENE_PS);
                    let tint = [shade * 0.1, 0.05, shade.mul_add(-0.1, 0.1), 0.0];
                    h.create_pixel_shader(&receiver_ps(tint, DETAIL_PS.contains(&at)))
                })
                .collect(),
            glow_ps: [1.0 / 640.0, 1.0 / 160.0]
                .map(|spread| h.create_pixel_shader(&glow_ps(spread))),
            composite_ps: h.create_pixel_shader(&composite_ps()),
            ui_vs: h.create_vertex_shader(&material_vs(&Model::Sm3, 2.0e-3)),
            ui_ps: [[0.0, 0.0, 0.0, 0.25], [0.1, 0.1, 0.0, 0.0]]
                .map(|tint| h.create_pixel_shader(&material_ps(&Model::Sm3, tint, false))),
        }
    }

    /// One whole frame up to its `Present`, `tick` animating the per-draw constants.
    fn render(&self, tick: u32) {
        let h = self.h;
        let mut states = States::new(h);
        ok(h.begin_scene(), "BeginScene");
        self.ring.restart();
        ok(h.set_viewport(&SCENE_VIEWPORT), "frame viewport");
        ok(
            h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, 0, 1.0, 0),
            "back buffer clear",
        );
        self.cascades(&mut states, tick);
        self.scene(&mut states, tick);
        ok(
            h.stretch_rect(&self.back_buffer, &self.scene_copy.surface, D3DTEXF_NONE),
            "back-buffer copy",
        );
        self.glow(&mut states);
        self.ui(&mut states);
        ok(h.end_scene(), "EndScene");
    }

    /// The five caster passes, one depth texture each beside the shared colour target.
    fn cascades(&self, states: &mut States<'_>, tick: u32) {
        let h = self.h;
        // The scene sampled the cascades on stages 4 to 7; they are depth targets now.
        for stage in 4..8 {
            ok(h.clear_texture(stage), "unbind a shadow map");
        }
        states.set(D3DRS_ALPHABLENDENABLE, 0);
        states.set(D3DRS_ALPHATESTENABLE, 0);
        states.set(D3DRS_ZENABLE, 1);
        states.set(D3DRS_ZWRITEENABLE, 1);
        states.set(D3DRS_ZFUNC, D3DCMP_LESSEQUAL);
        states.set(D3DRS_CULLMODE, D3DCULL_NONE);
        states.set(D3DRS_COLORWRITEENABLE, 0);
        states.set(D3DRS_SCISSORTESTENABLE, 1);
        ok(
            h.set_vertex_declaration(&self.decls[0]),
            "caster declaration",
        );
        ok(h.set_indices(&self.mesh_ib), "caster indices");
        let mut first = 0;
        for cascade in 0..CASCADES {
            self.cascade(tick, cascade, first);
            first += cascade_draws(cascade);
        }
    }

    /// One caster pass; `first` numbers its draws after the earlier cascades'.
    fn cascade(&self, tick: u32, cascade: usize, first: u32) {
        let h = self.h;
        let shadow = &self.shadows[cascade].surface;
        // The game reaches each cascade's targets through the back buffer.
        ok(h.set_render_target(0, &self.back_buffer), "back buffer");
        ok(h.set_depth_stencil_surface(shadow), "cascade depth");
        ok(
            h.set_render_target(0, &self.shadow_color.surface),
            "caster target",
        );
        ok(h.set_depth_stencil_surface(shadow), "cascade depth");
        ok(h.set_viewport(&SHADOW_VIEWPORT), "cascade viewport");
        ok(
            h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, 0xFFFF_FFFF, 1.0, 0),
            "cascade clear",
        );
        ok(
            h.set_scissor_rect(&CASCADE_SCISSOR[cascade]),
            "cascade band",
        );
        let reach = ratio(u32::try_from(cascade).expect("cascade fits u32") + 1, 5);
        let mut view = IDENTITY_ROWS;
        view[0] = reach;
        view[5] = reach;
        ok(
            h.set_vertex_shader_constant_f(0, &view),
            "cascade view-projection",
        );
        let mut vs_bound = None;
        let mut ps_bound = None;
        let mut at = first;
        for &(vs, ps, draws) in CASCADE_RUNS[cascade] {
            if vs_bound != Some(vs) {
                ok(h.set_vertex_shader(&self.caster_vs[vs]), "caster VS");
                vs_bound = Some(vs);
            }
            let textured = ps == TEXTURED_CASTER;
            if ps_bound != Some(ps) {
                ok(h.set_pixel_shader(&self.caster_ps[ps]), "caster PS");
                ok(
                    h.set_pixel_shader_constant_f(0, &[1.0; 4]),
                    "caster PS constant",
                );
                ps_bound = Some(ps);
                if textured {
                    for (state, value) in [
                        (D3DSAMP_MINFILTER, D3DTEXF_LINEAR),
                        (D3DSAMP_MAGFILTER, D3DTEXF_LINEAR),
                        (D3DSAMP_MIPFILTER, D3DTEXF_NONE),
                    ] {
                        ok(h.set_sampler_state(0, state, value), "caster sampler");
                    }
                }
            }
            for draw in 0..draws {
                self.caster(tick, at, draw, textured);
                at += 1;
            }
        }
        ok(h.set_render_target(0, &self.back_buffer), "back buffer");
        ok(h.set_depth_stencil_surface(shadow), "cascade depth");
    }

    /// Caster draw `at` of the frame, the `draw`-th of its run.
    ///
    /// Two consecutive draws of a run are one model's batches and share its world rows.
    fn caster(&self, tick: u32, at: u32, draw: u32, textured: bool) {
        let h = self.h;
        if textured {
            ok(
                h.set_texture(0, &self.textures[slot((at * 5) % SCENE_TEXTURES)]),
                "caster texture",
            );
        }
        let mesh = &self.meshes[slot(at % 4)];
        ok(h.set_stream_source(0, &mesh.vb, 0, STRIDE), "caster stream");
        if draw.is_multiple_of(2) {
            let (scale, x, y, z) = placement(at, tick);
            ok(
                h.set_vertex_shader_constant_f(4, &world_rows(scale, x, y, z)),
                "caster world",
            );
        }
        draw_mesh(h, mesh);
    }

    /// The scene on the back buffer: the frame's runs in order.
    fn scene(&self, states: &mut States<'_>, tick: u32) {
        let h = self.h;
        ok(h.set_render_target(0, &self.back_buffer), "scene target");
        ok(
            h.set_depth_stencil_surface(&self.scene_depth),
            "scene depth",
        );
        ok(h.set_scissor_rect(&SCENE_SCISSOR), "scene scissor");
        ok(h.set_viewport(&SCENE_VIEWPORT), "scene viewport");
        states.set(D3DRS_COLORWRITEENABLE, 0xF);
        states.set(D3DRS_SCISSORTESTENABLE, 0);
        states.set(D3DRS_LIGHTING, 0);
        ok(
            h.set_vertex_shader_constant_f(0, &IDENTITY_ROWS),
            "view-projection",
        );
        ok(h.set_transform(D3DTS_VIEW, &IDENTITY_ROWS), "FF view");
        ok(
            h.set_transform(D3DTS_PROJECTION, &IDENTITY_ROWS),
            "FF projection",
        );
        for stage in 4..8 {
            for (state, value) in [
                (D3DSAMP_MINFILTER, D3DTEXF_LINEAR),
                (D3DSAMP_MAGFILTER, D3DTEXF_LINEAR),
                (D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP),
                (D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP),
            ] {
                ok(h.set_sampler_state(stage, state, value), "shadow sampler");
            }
            ok(
                h.set_texture(stage, self.shadow_map(slot(stage - 3))),
                "shadow map",
            );
        }
        // The cascades left the first declaration and the mesh indices bound.
        let mut binds = SceneBinds {
            decl: Some(0),
            nearest: Some(false),
            indices: Some(Indices::Mesh),
            ..SceneBinds::default()
        };
        let mut draw = 0;
        for (index, run) in self.runs.iter().enumerate() {
            let Some(pair) = run.pair else {
                self.sky(states, tick, run, &mut binds, &mut draw);
                continue;
            };
            // Stage 4 holds the nearest cascade for five runs in fifteen.
            self.bind_run(states, run, pair, (index / 5) % 3 == 0, &mut binds);
            for _ in 0..run.draws {
                states.set(D3DRS_CULLMODE, scene_cull(draw, run.pair));
                self.scene_draw(tick, draw, run, &mut binds);
                draw += 1;
            }
        }
    }

    /// What the scene samples for cascade `cascade`: its depth texture, or a stand-in.
    fn shadow_map(&self, cascade: usize) -> &Texture<'h> {
        self.shadows[cascade]
            .texture
            .as_ref()
            .unwrap_or(&self.textures[cascade])
    }

    /// Bind what changed of a programmable run's programs, declaration, samplers and states.
    fn bind_run(
        &self,
        states: &mut States<'_>,
        run: &SceneRun,
        pair: usize,
        nearest: bool,
        binds: &mut SceneBinds,
    ) {
        let h = self.h;
        let (vs, ps) = SCENE_PAIRS[pair];
        if binds.vs != Some(vs) {
            ok(h.set_vertex_shader(&self.scene_vs[vs]), "scene VS");
            binds.vs = Some(vs);
            let decl = SCENE_DECL[vs];
            if binds.decl != Some(decl) {
                ok(
                    h.set_vertex_declaration(&self.decls[decl]),
                    "scene declaration",
                );
                binds.decl = Some(decl);
            }
        }
        if binds.ps != Some(ps) {
            ok(h.set_pixel_shader(&self.scene_ps[ps]), "scene PS");
            let shade = shade(ps, SCENE_PS);
            ok(
                h.set_pixel_shader_constant_f(0, &[1.0 - shade, 0.8, shade, 1.0]),
                "material tint",
            );
            binds.ps = Some(ps);
        }
        let wrap = scene_wrap(pair);
        if binds.wrap != Some(wrap) {
            ok(h.set_sampler_state(0, D3DSAMP_ADDRESSU, wrap), "address U");
            ok(h.set_sampler_state(0, D3DSAMP_ADDRESSV, wrap), "address V");
            binds.wrap = Some(wrap);
        }
        if binds.nearest != Some(nearest) {
            ok(
                h.set_texture(4, self.shadow_map(usize::from(!nearest))),
                "near shadow map",
            );
            binds.nearest = Some(nearest);
        }
        states.run(run);
    }

    /// Programmable scene draw `draw`: stage-0 texture, stream, world rows, draw.
    fn scene_draw(&self, tick: u32, draw: u32, run: &SceneRun, binds: &mut SceneBinds) {
        let h = self.h;
        ok(h.set_texture(0, self.scene_texture(draw)), "scene texture");
        if run.detail() {
            // Even entries are DXT1.
            let detail = &self.textures[slot((draw * 2) % SCENE_TEXTURES)];
            ok(h.set_texture(1, detail), "detail texture");
        }
        if run.particles {
            let set = &self.particle_sets[slot((draw + tick) % PARTICLE_SETS)];
            let base = self.ring.write(set);
            ok(
                h.set_stream_source(0, &self.ring.vb, 0, STRIDE),
                "particle stream",
            );
            self.bind_indices(binds, Indices::Quads);
            ok(
                h.set_vertex_shader_constant_f(4, &IDENTITY_ROWS),
                "particle world",
            );
            ok(
                h.draw_indexed_primitive(
                    D3DPT_TRIANGLELIST,
                    i32::try_from(base).expect("ring vertex fits i32"),
                    0,
                    PARTICLE_QUADS * 4,
                    0,
                    PARTICLE_QUADS * 2,
                ),
                "particle draw",
            );
            return;
        }
        let mesh = &self.meshes[slot(draw % 4)];
        ok(h.set_stream_source(0, &mesh.vb, 0, STRIDE), "scene stream");
        self.bind_indices(binds, Indices::Mesh);
        // The world rows, then up to three rows of per-draw extras
        // (bones, texture transforms) the programs do not read.
        let (scale, x, y, z) = placement(draw, tick);
        let mut rows = [0.25_f32; 64];
        rows[..16].copy_from_slice(&world_rows(scale, x, y, z));
        let count = 16 * (1 + slot(draw % 4));
        ok(
            h.set_vertex_shader_constant_f(4, &rows[..count]),
            "scene world",
        );
        draw_mesh(h, mesh);
    }

    /// The stage-0 texture of scene draw `draw`.
    ///
    /// DXT1 and DXT5 but for the dumped frame's DXT3 and uncompressed draws;
    /// one draw in five reuses the texture of the draw before it.
    fn scene_texture(&self, draw: u32) -> &Texture<'h> {
        if let Some(at) = UNCOMPRESSED_DRAWS.iter().position(|&known| known == draw) {
            return &self.uncompressed[at];
        }
        if DXT3_DRAWS.contains(&draw) {
            return &self.dxt3_textures[slot(draw % DXT3_TEXTURES)];
        }
        &self.textures[slot(((draw - draw / 5) * 7) % SCENE_TEXTURES)]
    }

    fn bind_indices(&self, binds: &mut SceneBinds, indices: Indices) {
        if binds.indices.as_ref() == Some(&indices) {
            return;
        }
        let ib = match indices {
            Indices::Mesh => &self.mesh_ib,
            Indices::Quads => &self.quad_ib,
        };
        ok(self.h.set_indices(ib), "SetIndices");
        binds.indices = Some(indices);
    }

    /// The three fixed-function sky draws, scissor-tested at the far end of the depth range.
    fn sky(
        &self,
        states: &mut States<'_>,
        tick: u32,
        run: &SceneRun,
        binds: &mut SceneBinds,
        draw: &mut u32,
    ) {
        let h = self.h;
        states.set(D3DRS_SCISSORTESTENABLE, 1);
        ok(h.set_scissor_rect(&FULL_SCISSOR), "sky scissor");
        ok(h.set_viewport(&SKY_VIEWPORT), "sky viewport");
        ok(h.clear_vertex_shader(), "sky VS");
        ok(h.clear_pixel_shader(), "sky PS");
        binds.vs = None;
        binds.ps = None;
        if binds.decl != Some(0) {
            ok(h.set_vertex_declaration(&self.decls[0]), "sky declaration");
            binds.decl = Some(0);
        }
        states.run(run);
        states.set(D3DRS_CULLMODE, D3DCULL_NONE);
        ok(
            h.set_texture_stage_state(0, D3DTSS_COLORARG2, D3DTA_DIFFUSE),
            "sky arg 2",
        );
        for at in 0..run.draws {
            // The middle draw is untextured and takes the vertex colour alone.
            if at == 1 {
                ok(h.clear_texture(0), "untextured sky");
                ok(
                    h.set_texture_stage_state(0, D3DTSS_COLORARG1, D3DTA_DIFFUSE),
                    "sky arg 1",
                );
                ok(
                    h.set_texture_stage_state(0, D3DTSS_COLOROP, D3DTOP_SELECTARG1),
                    "sky op",
                );
            } else {
                ok(
                    h.set_texture(0, &self.textures[slot(at + 8)]),
                    "sky texture",
                );
                if at == 2 {
                    ok(
                        h.set_texture_stage_state(0, D3DTSS_COLORARG1, D3DTA_TEXTURE),
                        "sky arg 1",
                    );
                }
                ok(
                    h.set_texture_stage_state(0, D3DTSS_COLOROP, D3DTOP_MODULATE),
                    "sky op",
                );
            }
            let (scale, x, y, z) = placement(*draw, tick);
            ok(
                h.set_transform(D3DTS_WORLD, &ff_world(scale, x, y, z)),
                "sky world",
            );
            let mesh = &self.meshes[slot(at)];
            ok(h.set_stream_source(0, &mesh.vb, 0, STRIDE), "sky stream");
            self.bind_indices(binds, Indices::Mesh);
            draw_mesh(h, mesh);
            *draw += 1;
        }
        states.set(D3DRS_SCISSORTESTENABLE, 0);
        ok(h.set_viewport(&SCENE_VIEWPORT), "scene viewport");
    }

    /// Three one-draw glow passes, each sampling on four stages the target the one before wrote.
    ///
    /// The targets are bound as the game binds them, the back buffer and its
    /// depth coming and going between the first pass and the second.
    fn glow(&self, states: &mut States<'_>) {
        let h = self.h;
        states.set(D3DRS_ZWRITEENABLE, 0);
        states.set(D3DRS_ZFUNC, D3DCMP_ALWAYS);
        states.set(D3DRS_ALPHABLENDENABLE, 0);
        states.set(D3DRS_ALPHATESTENABLE, 0);
        states.set(D3DRS_CULLMODE, D3DCULL_NONE);
        ok(h.clear_vertex_shader(), "glow VS");
        ok(h.set_vertex_declaration(&self.decls[0]), "glow declaration");
        ok(h.set_transform(D3DTS_WORLD, &IDENTITY_ROWS), "glow world");
        ok(
            h.set_stream_source(0, &self.screen_quad, 0, STRIDE),
            "glow stream",
        );
        ok(h.set_indices(&self.quad_ib), "glow indices");
        for stage in 0..4 {
            for (state, value) in [
                (D3DSAMP_MINFILTER, D3DTEXF_LINEAR),
                (D3DSAMP_MAGFILTER, D3DTEXF_LINEAR),
                (D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP),
                (D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP),
            ] {
                ok(h.set_sampler_state(stage, state, value), "glow sampler");
            }
        }
        let passes = [
            (&self.glow[0], &self.scene_copy.texture, 0),
            (&self.glow[1], &self.glow[0].texture, 1),
            (&self.glow[0], &self.glow[1].texture, 1),
        ];
        for (at, (target, source, program)) in passes.into_iter().enumerate() {
            ok(h.set_render_target(0, &target.surface), "glow target");
            ok(h.set_depth_stencil_surface(&self.glow_depth), "glow depth");
            ok(h.set_scissor_rect(&GLOW_SCISSOR), "glow scissor");
            ok(h.set_viewport(&GLOW_VIEWPORT), "glow viewport");
            if at < 2 {
                ok(h.set_pixel_shader(&self.glow_ps[program]), "glow PS");
            }
            ok(h.set_pixel_shader_constant_f(0, &[0.25; 4]), "glow weight");
            for stage in 0..4 {
                ok(h.set_texture(stage, source), "glow source");
            }
            ok(
                h.draw_indexed_primitive(D3DPT_TRIANGLELIST, 0, 0, 4, 0, 2),
                "glow draw",
            );
            if at == 0 {
                ok(h.set_render_target(0, &self.back_buffer), "back buffer");
                ok(
                    h.set_depth_stencil_surface(&self.scene_depth),
                    "scene depth",
                );
                ok(h.set_render_target(0, &target.surface), "glow target");
                ok(h.set_depth_stencil_surface(&self.glow_depth), "glow depth");
            }
        }
        ok(h.set_render_target(0, &self.back_buffer), "back buffer");
        ok(
            h.set_depth_stencil_surface(&self.scene_depth),
            "scene depth",
        );
    }

    /// The UI pass: the glow composite, the programmable quads with the minimap among them.
    fn ui(&self, states: &mut States<'_>) {
        let h = self.h;
        ok(h.set_scissor_rect(&FULL_SCISSOR), "UI scissor");
        ok(h.set_viewport(&UI_VIEWPORT), "UI viewport");
        ok(h.set_pixel_shader(&self.composite_ps), "composite PS");
        ok(
            h.set_pixel_shader_constant_f(0, &[0.5; 4]),
            "composite weight",
        );
        ok(
            h.set_texture(0, &self.scene_copy.texture),
            "composite scene",
        );
        ok(h.set_texture(1, &self.glow[0].texture), "composite glow");
        ok(
            h.draw_indexed_primitive(D3DPT_TRIANGLELIST, 0, 0, 4, 0, 2),
            "glow composite",
        );

        ok(h.set_vertex_shader(&self.ui_vs), "UI VS");
        ok(
            h.set_vertex_shader_constant_f(0, &IDENTITY_ROWS),
            "UI projection",
        );
        ok(
            h.set_vertex_shader_constant_f(4, &IDENTITY_ROWS),
            "UI world",
        );
        ok(
            h.set_stream_source(0, &self.ring.vb, 0, STRIDE),
            "UI stream",
        );
        states.set(D3DRS_ALPHABLENDENABLE, 1);
        states.set(D3DRS_ALPHATESTENABLE, 1);
        let mut ps_bound = None;
        for quad in 0..UI_BEFORE_MINIMAP {
            self.ui_quad(states, quad, &mut ps_bound);
        }
        self.minimap();
        ok(h.set_viewport(&UI_VIEWPORT), "UI viewport");
        ok(h.set_vertex_shader(&self.ui_vs), "UI VS");
        ok(
            h.set_stream_source(0, &self.ring.vb, 0, STRIDE),
            "UI stream",
        );
        ps_bound = None;
        for quad in UI_BEFORE_MINIMAP..UI_BEFORE_MINIMAP + UI_AFTER_MINIMAP {
            self.ui_quad(states, quad, &mut ps_bound);
        }
    }

    /// Programmable UI quad `quad`, written into the dynamic buffer by a lock of its own.
    fn ui_quad(&self, states: &mut States<'_>, quad: u32, ps_bound: &mut Option<usize>) {
        let h = self.h;
        let second =
            quad >= UI_BEFORE_MINIMAP && UI_SECOND_PS.contains(&(quad - UI_BEFORE_MINIMAP));
        let ps = usize::from(second);
        if *ps_bound != Some(ps) {
            ok(h.set_pixel_shader(&self.ui_ps[ps]), "UI PS");
            ok(h.set_pixel_shader_constant_f(0, &[1.0; 4]), "UI tint");
            *ps_bound = Some(ps);
        }
        let additive = quad % 12 == 11;
        states.set(D3DRS_SRCBLEND, D3DBLEND_SRCALPHA);
        states.set(
            D3DRS_DESTBLEND,
            if additive {
                D3DBLEND_ONE
            } else {
                D3DBLEND_INVSRCALPHA
            },
        );
        states.set(
            D3DRS_CULLMODE,
            if quad % 7 == 3 {
                D3DCULL_NONE
            } else {
                D3DCULL_CW
            },
        );
        let at = slot(quad * 4);
        let base = self.ring.write(&self.ui_quads[at..at + 4]);
        // Four quads in five draw game art, the fifth a pattern point-sampled like text.
        let (texture, filter) = if quad.is_multiple_of(5) {
            (
                &self.ui_textures[slot((quad / 5) % UI_TEXTURES)],
                D3DTEXF_POINT,
            )
        } else {
            (
                &self.textures[slot((quad * 3) % SCENE_TEXTURES)],
                D3DTEXF_LINEAR,
            )
        };
        ok(h.set_texture(0, texture), "UI texture");
        ok(
            h.set_sampler_state(0, D3DSAMP_MINFILTER, filter),
            "UI filter",
        );
        ok(
            h.draw_indexed_primitive(
                D3DPT_TRIANGLELIST,
                i32::try_from(base).expect("ring vertex fits i32"),
                0,
                4,
                0,
                2,
            ),
            "UI draw",
        );
    }

    /// The two fixed-function minimap draws: a map under a two-stage mask, then its border.
    fn minimap(&self) {
        let h = self.h;
        ok(h.set_viewport(&MINIMAP_VIEWPORT), "minimap viewport");
        ok(h.clear_vertex_shader(), "minimap VS");
        ok(h.clear_pixel_shader(), "minimap PS");
        ok(
            h.set_stream_source(0, &self.screen_quad, 0, STRIDE),
            "minimap stream",
        );
        ok(
            h.set_transform(D3DTS_WORLD, &IDENTITY_ROWS),
            "minimap world",
        );
        for (stage, op, arg2) in [
            (0, D3DTOP_MODULATE, D3DTA_DIFFUSE),
            (1, D3DTOP_MODULATE, D3DTA_CURRENT),
        ] {
            ok(
                h.set_texture_stage_state(stage, D3DTSS_COLOROP, op),
                "minimap op",
            );
            ok(
                h.set_texture_stage_state(stage, D3DTSS_COLORARG1, D3DTA_TEXTURE),
                "minimap arg 1",
            );
            ok(
                h.set_texture_stage_state(stage, D3DTSS_COLORARG2, arg2),
                "minimap arg 2",
            );
        }
        ok(
            h.set_texture_stage_state(1, D3DTSS_TEXCOORDINDEX, 0),
            "minimap mask coordinates",
        );
        ok(h.set_texture(0, &self.ui_textures[0]), "minimap");
        ok(h.set_texture(1, &self.textures[3]), "minimap mask");
        ok(
            h.draw_indexed_primitive(D3DPT_TRIANGLELIST, 0, 0, 4, 0, 2),
            "minimap draw",
        );
        ok(
            h.set_texture_stage_state(1, D3DTSS_COLOROP, D3DTOP_DISABLE),
            "minimap border op",
        );
        ok(h.set_texture(0, &self.textures[5]), "minimap border");
        ok(
            h.draw_indexed_primitive(D3DPT_TRIANGLELIST, 0, 0, 4, 0, 2),
            "minimap border draw",
        );
    }
}

/// A cascade's depth target: a D24X8 texture's level when `sampled`, else a D24X8 surface.
fn shadow(h: &Harness, sampled: bool) -> Shadow<'_> {
    if sampled {
        let texture = h.create_texture(
            SHADOW_EDGE,
            SHADOW_EDGE,
            1,
            D3DUSAGE_DEPTHSTENCIL,
            D3DFMT_D24X8,
            D3DPOOL_DEFAULT,
        );
        let surface = texture.surface_level(0);
        Shadow {
            texture: Some(texture),
            surface,
        }
    } else {
        Shadow {
            texture: None,
            surface: h.create_depth_stencil_surface(SHADOW_EDGE, SHADOW_EDGE, D3DFMT_D24X8),
        }
    }
}

/// A managed 256x256 DXT1, DXT3 or DXT5 texture, one level, of blocks checkered by `seed`.
///
/// The colour endpoints keep the four-colour block mode; the DXT3 and DXT5
/// blocks of every third column carry an alpha below one half, which the
/// textured casters discard.
fn dxt_texture(h: &Harness, format: u32, seed: u32) -> Texture<'_> {
    let texture = h.create_texture(DXT_EDGE, DXT_EDGE, 1, 0, format, D3DPOOL_MANAGED);
    let blocks = DXT_EDGE / 4;
    let bright = 0x8000 | u16::try_from((seed * 0x0923) & 0x7FFF).expect("masked to 15 bits");
    let dark = bright >> 1;
    let mut bytes = Vec::new();
    for row in 0..blocks {
        for column in 0..blocks {
            if format == D3DFMT_DXT3 {
                // Explicit four-bit alpha, below one half in every third column.
                let alpha = if column.is_multiple_of(3) { 0x77 } else { 0xFF };
                bytes.extend_from_slice(&[alpha; 8]);
            }
            if format == D3DFMT_DXT5 {
                // Endpoints 0xFF and 0x40; index 1 on every texel selects 0x40.
                let low = column.is_multiple_of(3);
                bytes.extend_from_slice(&[0xFF, 0x40]);
                bytes.extend_from_slice(&if low {
                    [0x49, 0x92, 0x24, 0x49, 0x92, 0x24]
                } else {
                    [0; 6]
                });
            }
            let indices: u32 = if (row + column).is_multiple_of(2) {
                0
            } else {
                0x5555_5555
            };
            bytes.extend_from_slice(&bright.to_le_bytes());
            bytes.extend_from_slice(&dark.to_le_bytes());
            bytes.extend_from_slice(&indices.to_le_bytes());
        }
    }
    let rows = slot(blocks);
    texture
        .lock_rect(0, 0)
        .write_u8_rect(bytes.len() / rows, rows, &bytes);
    texture
}

/// A managed 64x64 A4R4G4B4 checker of `color` and its inverse, one level.
fn a4r4g4b4_texture(h: &Harness, color: u16) -> Texture<'_> {
    const EDGE: u32 = 64;
    let texture = h.create_texture(EDGE, EDGE, 1, 0, D3DFMT_A4R4G4B4, D3DPOOL_MANAGED);
    let bytes: Vec<u8> = (0..EDGE * EDGE)
        .flat_map(|at| {
            let texel = if (at / 8 + at / (8 * EDGE)).is_multiple_of(2) {
                color
            } else {
                !color | 0xF000
            };
            texel.to_le_bytes()
        })
        .collect();
    let edge = slot(EDGE);
    texture
        .lock_rect(0, 0)
        .write_u8_rect(edge * 2, edge, &bytes);
    texture
}

/// The static meshes, each in a vertex buffer of its own, and the index buffer they share.
fn meshes(h: &Harness) -> (Vec<Mesh<'_>>, IndexBuffer<'_>) {
    let mut all_indices = Vec::new();
    let mut meshes = Vec::new();
    for n in MESH_GRIDS {
        let (vertices, indices) = grid(n);
        let bytes = u32::try_from(vertices.len()).expect("mesh fits u32") * STRIDE;
        let vb = h.create_vertex_buffer(bytes, D3DUSAGE_WRITEONLY, 0, D3DPOOL_MANAGED);
        vb.lock(0, 0, 0).write(&vertices);
        meshes.push(Mesh {
            vb,
            vertices: u32::try_from(vertices.len()).expect("mesh fits u32"),
            start_index: u32::try_from(all_indices.len()).expect("index count fits u32"),
            triangles: u32::try_from(indices.len() / 3).expect("triangle count fits u32"),
        });
        all_indices.extend_from_slice(&indices);
    }
    let bytes = u32::try_from(all_indices.len() * 2).expect("index bytes fit u32");
    let ib = h.create_index_buffer(bytes, D3DUSAGE_WRITEONLY, D3DFMT_INDEX16, D3DPOOL_MANAGED);
    ib.lock(0, 0, 0).write(&all_indices);
    (meshes, ib)
}

/// Quad indices for [`PARTICLE_QUADS`] quads of four vertices each.
fn quad_indices(h: &Harness) -> IndexBuffer<'_> {
    let indices: Vec<u16> = (0..u16::try_from(PARTICLE_QUADS).expect("quad count fits u16"))
        .flat_map(|quad| {
            let base = quad * 4;
            [base, base + 1, base + 2, base + 2, base + 1, base + 3]
        })
        .collect();
    let bytes = u32::try_from(indices.len() * 2).expect("index bytes fit u32");
    let ib = h.create_index_buffer(bytes, D3DUSAGE_WRITEONLY, D3DFMT_INDEX16, D3DPOOL_MANAGED);
    ib.lock(0, 0, 0).write(&indices);
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
    let vertices = [
        corner(-1.0, 1.0),
        corner(1.0, 1.0),
        corner(-1.0, -1.0),
        corner(1.0, -1.0),
    ];
    let vb = h.create_vertex_buffer(4 * STRIDE, D3DUSAGE_WRITEONLY, 0, D3DPOOL_MANAGED);
    vb.lock(0, 0, 0).write(&vertices);
    vb
}

/// `count` small squares of edge `size` in clip space, scattered by `seed`.
fn quads_at(count: u32, seed: u32, size: f32) -> Vec<TexturedVertex> {
    let mut vertices = Vec::new();
    for quad in 0..count {
        let x = ratio((quad * 17 + seed * 5) % 100, 100).mul_add(1.8, -0.95);
        let y = ratio((quad * 29 + seed * 3) % 100, 100).mul_add(1.8, -0.95);
        let corner = |dx: f32, dy: f32| TexturedVertex {
            x: dx.mul_add(size, x),
            y: dy.mul_add(-size, y),
            z: 0.1,
            color: 0xC0FF_FFFF,
            u: dx,
            v: dy,
        };
        vertices.extend_from_slice(&[
            corner(0.0, 0.0),
            corner(1.0, 0.0),
            corner(0.0, 1.0),
            corner(1.0, 1.0),
        ]);
    }
    vertices
}

/// Scale and clip-space position of the `at`-th mesh in frame `tick`.
fn placement(at: u32, tick: u32) -> (f32, f32, f32, f32) {
    let drift = ratio(tick % 64, 64) * 0.01;
    (
        ratio(at % 5, 5).mul_add(0.1, 0.3),
        ratio((at * 37) % 100, 100).mul_add(1.7, -1.0) + drift,
        ratio((at * 61) % 100, 100).mul_add(1.7, -1.0),
        ratio((at * 13) % 97, 97).mul_add(0.8, 0.1),
    )
}

/// The fixed-function world matrix (row vectors) that [`world_rows`] states as constant rows.
const fn ff_world(scale: f32, x: f32, y: f32, z: f32) -> [f32; 16] {
    [
        scale, 0.0, 0.0, 0.0, //
        0.0, scale, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, //
        x, y, z, 1.0,
    ]
}

const fn viewport(x: u32, y: u32, width: u32, height: u32, min_z: f32, max_z: f32) -> D3DVIEWPORT9 {
    D3DVIEWPORT9 {
        x,
        y,
        width,
        height,
        min_z,
        max_z,
    }
}

/// `left`/`top`/`right`/`bottom` in target coordinates.
const fn rect(x1: i32, y1: i32, x2: i32, y2: i32) -> D3DRECT {
    D3DRECT { x1, y1, x2, y2 }
}

/// A caster vertex program: the scene transform, nudged by `def c95` so the three are distinct.
///
/// It passes the texture coordinate on for the textured caster program.
#[rustfmt::skip]
fn caster_vs(variant: f32) -> Vec<u32> {
    let mut tokens = vec![
        0xFFFE_0200,                           // vs_2_0
        0x0200_001F, 0x8000_0000, 0x900F_0000, // dcl_position v0
        0x0200_001F, 0x8000_0005, 0x900F_0001, // dcl_texcoord0 v1
    ];
    tokens.extend_from_slice(&def(0xA00F_005F, [0.0, 0.0, variant, 0.0]));
    // The world rows land in r0; nudge it before view-projection.
    let mut body = transform(0xC000_0000).to_vec(); // oPos
    body.splice(16..16, [0x0300_0002, 0x800F_0000, 0x80E4_0000, 0xA0E4_005F]); // add r0, r0, c95
    tokens.extend_from_slice(&body);
    tokens.extend_from_slice(&[
        0x0200_0001, 0xE00F_0000, 0x90E4_0001, // mov oT0, v1
        0x0000_FFFF,
    ]);
    tokens
}

/// `ps_2_0 { def c7, 1, 1, 1, 1; mov oC0, c7 }`, the untextured caster colour, masked off.
#[rustfmt::skip]
fn caster_ps() -> Vec<u32> {
    let mut tokens = vec![0xFFFF_0200]; // ps_2_0
    tokens.extend_from_slice(&def(0xA00F_0007, [1.0; 4]));
    tokens.extend_from_slice(&[
        0x0200_0001, 0x800F_0800, 0xA0E4_0007, // mov oC0, c7
        0x0000_FFFF,
    ]);
    tokens
}

/// The textured caster: `texld r0, t0, s0`, then `texkill` where the alpha is below one half.
#[rustfmt::skip]
fn textured_caster_ps() -> Vec<u32> {
    let mut tokens = vec![
        0xFFFF_0200,                           // ps_2_0
        0x0200_001F, 0x8000_0000, 0xB00F_0000, // dcl t0
        0x0200_001F, 0x9000_0000, 0xA00F_0800, // dcl_2d s0
    ];
    tokens.extend_from_slice(&def(0xA00F_0001, [-0.5; 4])); // def c1
    tokens.extend_from_slice(&[
        0x0300_0042, 0x800F_0000, 0xB0E4_0000, 0xA0E4_0800, // texld r0, t0, s0
        0x0300_0002, 0x800F_0001, 0x80FF_0000, 0xA0E4_0001, // add r1, r0.wwww, c1
        0x0100_0041, 0x800F_0001,                           // texkill r1
        0x0200_0001, 0x800F_0800, 0x80E4_0000,              // mov oC0, r0
        0x0000_FFFF,
    ]);
    tokens
}

/// A scene pixel program: the stage-0 texture times the four shadow maps on stages 4 to 7.
///
/// `mad oC0, r0, c0, c7` finishes it, `def c7` being `tint`, so distinct
/// tints are distinct programs. A depth texture on stages 4 to 7 is read as
/// a hardware shadow comparison against the coordinate's `z`. A `detail`
/// program also multiplies in the texture on stage 1.
#[rustfmt::skip]
fn receiver_ps(tint: [f32; 4], detail: bool) -> Vec<u32> {
    let mut tokens = vec![
        0xFFFF_0200,                           // ps_2_0
        0x0200_001F, 0x8000_0000, 0xB00F_0000, // dcl t0
        0x0200_001F, 0x9000_0000, 0xA00F_0800, // dcl_2d s0
    ];
    if detail {
        tokens.extend_from_slice(&[0x0200_001F, 0x9000_0000, 0xA00F_0801]); // dcl_2d s1
    }
    for sampler in 4..8 {
        tokens.extend_from_slice(&[0x0200_001F, 0x9000_0000, 0xA00F_0800 | sampler]); // dcl_2d s4..s7
    }
    tokens.extend_from_slice(&def(0xA00F_0007, tint)); // def c7
    tokens.extend_from_slice(&[0x0300_0042, 0x800F_0000, 0xB0E4_0000, 0xA0E4_0800]); // texld r0, t0, s0
    for (register, sampler) in (1..5).zip(4..8) {
        // texld r1..r4, t0, s4..s7
        tokens.extend_from_slice(&[0x0300_0042, 0x800F_0000 | register, 0xB0E4_0000, 0xA0E4_0800 | sampler]);
    }
    tokens.extend_from_slice(&[
        0x0300_0005, 0x800F_0001, 0x80E4_0001, 0x80E4_0002,              // mul r1, r1, r2
        0x0300_0005, 0x800F_0003, 0x80E4_0003, 0x80E4_0004,              // mul r3, r3, r4
        0x0300_0005, 0x800F_0001, 0x80E4_0001, 0x80E4_0003,              // mul r1, r1, r3
        0x0300_0005, 0x800F_0000, 0x80E4_0000, 0x80E4_0001,              // mul r0, r0, r1
    ]);
    if detail {
        tokens.extend_from_slice(&[
            0x0300_0042, 0x800F_0005, 0xB0E4_0000, 0xA0E4_0801, // texld r5, t0, s1
            0x0300_0005, 0x800F_0000, 0x80E4_0000, 0x80E4_0005, // mul r0, r0, r5
        ]);
    }
    tokens.extend_from_slice(&[
        0x0400_0004, 0x800F_0000, 0x80E4_0000, 0xA0E4_0000, 0xA0E4_0007, // mad r0, r0, c0, c7
        0x0200_0001, 0x800F_0800, 0x80E4_0000,                           // mov oC0, r0
        0x0000_FFFF,
    ]);
    tokens
}

/// A glow program: four taps of one source bound on stages 0 to 3, `spread` apart, times `c0`.
#[rustfmt::skip]
fn glow_ps(spread: f32) -> Vec<u32> {
    let mut tokens = vec![
        0xFFFF_0200,                           // ps_2_0
        0x0200_001F, 0x8000_0000, 0xB00F_0000, // dcl t0
    ];
    for sampler in 0..4 {
        tokens.extend_from_slice(&[0x0200_001F, 0x9000_0000, 0xA00F_0800 | sampler]); // dcl_2d s0..s3
    }
    tokens.extend_from_slice(&def(0xA00F_0004, [spread, 0.0, 0.0, 0.0])); // def c4
    tokens.extend_from_slice(&def(0xA00F_0005, [0.0, spread, 0.0, 0.0])); // def c5
    tokens.extend_from_slice(&def(0xA00F_0006, [spread, spread, 0.0, 0.0])); // def c6
    tokens.extend_from_slice(&[
        0x0300_0002, 0x800F_0004, 0xB0E4_0000, 0xA0E4_0004, // add r4, t0, c4
        0x0300_0002, 0x800F_0005, 0xB0E4_0000, 0xA0E4_0005, // add r5, t0, c5
        0x0300_0002, 0x800F_0006, 0xB0E4_0000, 0xA0E4_0006, // add r6, t0, c6
        0x0300_0042, 0x800F_0000, 0xB0E4_0000, 0xA0E4_0800, // texld r0, t0, s0
        0x0300_0042, 0x800F_0001, 0x80E4_0004, 0xA0E4_0801, // texld r1, r4, s1
        0x0300_0042, 0x800F_0002, 0x80E4_0005, 0xA0E4_0802, // texld r2, r5, s2
        0x0300_0042, 0x800F_0003, 0x80E4_0006, 0xA0E4_0803, // texld r3, r6, s3
        0x0300_0002, 0x800F_0000, 0x80E4_0000, 0x80E4_0001, // add r0, r0, r1
        0x0300_0002, 0x800F_0002, 0x80E4_0002, 0x80E4_0003, // add r2, r2, r3
        0x0300_0002, 0x800F_0000, 0x80E4_0000, 0x80E4_0002, // add r0, r0, r2
        0x0300_0005, 0x800F_0000, 0x80E4_0000, 0xA0E4_0000, // mul r0, r0, c0
        0x0200_0001, 0x800F_0800, 0x80E4_0000,              // mov oC0, r0
        0x0000_FFFF,
    ]);
    tokens
}

/// The glow composite: the scene copy on stage 0 plus `c0` times the glow on stage 1.
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

/// Draw a static mesh from the bound stream and the shared index buffer.
fn draw_mesh(h: &Harness, mesh: &Mesh<'_>) {
    ok(
        h.draw_indexed_primitive(
            D3DPT_TRIANGLELIST,
            0,
            0,
            mesh.vertices,
            mesh.start_index,
            mesh.triangles,
        ),
        "mesh draw",
    );
}

/// Program `at` of `of` as a fraction, which sets it apart from its siblings.
fn shade(at: usize, of: usize) -> f32 {
    ratio(
        u32::try_from(at).expect("program index fits u32"),
        u32::try_from(of).expect("program count fits u32"),
    )
}

/// `value` as an index into one of the frame's resource lists.
fn slot(value: u32) -> usize {
    usize::try_from(value).expect("a resource index fits usize")
}
