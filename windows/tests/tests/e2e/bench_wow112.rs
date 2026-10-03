//! A synthetic frame shaped like World of Warcraft 1.12's busy frame, timed over many frames.
//!
//! The shape comes from the layer's `[dump]` of a busy frame and the Metal
//! capture taken with it (WoW-14623 on a `PERF=1` build of main at 66b9a279,
//! 2026-10; frame 3 of the three dumped), and the call rates from the
//! perf window the game wrote next. One frame is 5,243 draws in five passes
//! at half the game's 2624x1696 on each axis: the scene, 5,091 draws into a 1312x848 A8R8G8B8
//! render-target texture over a D24S8 depth surface of its own; three
//! one-draw glow passes at a quarter of that size, ping-ponging between two
//! targets with the scene depth left bound, each a fixed-function vertex
//! stage feeding a pixel program that takes four taps of one texture bound
//! on stages 0 to 3; and the back buffer over the device's own depth, a
//! composite of the scene and the glow followed by 148 interface draws.
//!
//! The scene is built from run tables in the constants below, walked in the
//! game's order: a sky of fixed-function vertices under three pixel
//! programs and the fixed stages; the opaque world, 3,163 programmable draws
//! in 20 vertex-program stretches whose draws alternate opaque runs with
//! alpha-tested ones and turn culling off now and then; the transparent
//! world, 699 draws whose vertex program changes about every other draw,
//! additive, alpha-blended, modulating and colour-masked; the effects, 1,134
//! fixed-function particles and text quads written into the dynamic vertex
//! ring; and the sun, one colour-masked draw inside an occlusion query and
//! its flare. 30 vertex and 11 pixel programs, 885 textures in the game's
//! format mix, one texture a draw. As in the game, whose dump shows 1.6 %
//! of its scene draws changing shaders or render states without changing
//! texture, a mesh or particle draw that starts an object or changes shaders
//! or render states also changes texture. The tables are tuned so that the draws,
//! fixed-function shares, textures per draw, blend, alpha-test, depth-write,
//! cull and colour-mask shares, the shader, texture, blend, alpha-test and
//! cull switches and the distinct shaders and textures of every pass land
//! close to the game's, which `make bench-shape` checks against the dump.
//!
//! The calls between draws follow the game's own state cache: a render
//! state, shader, texture, sampler state or index buffer is set only when it
//! changes, while `SetVertexDeclaration` and `SetStreamSource` come with
//! every draw, as the game's do. An object, a run of draws of one mesh, sets
//! its world transform and uploads its vertex constants in three or four
//! calls; seven in ten of the draws that continue it set the world again,
//! and a particle run starts an emitter, with its own transform, every other
//! draw. Every 28th object binds a second declaration of the same layout, so
//! about 1.5 % of the declaration sets change it, as in the game. A
//! programmable draw sets one pixel-shader row, most of them the value
//! already bound. Geometry is indexed triangle lists from 64 static meshes
//! in the game's index-count buckets, and from one 2.75 MiB dynamic vertex
//! ring: particles append what they draw with `D3DLOCK_NOOVERWRITE`, text
//! reserves 2048 or 4096 vertices a draw, as the game's capture shows, and
//! writes its quads into the reservation in one or two locks, and the ring
//! starts over with `D3DLOCK_DISCARD` when it fills and at the start of
//! each frame, about 12 times a frame. Interface quads are strips whose four
//! indices go into a 256 KiB dynamic index ring with `D3DLOCK_NOOVERWRITE`.
//! Per frame there are also 16 render-target and depth-stencil binds, 15
//! viewports, three clears, the occlusion query's `GetDataSize`, `GetData`
//! with `D3DGETDATA_FLUSH` and `Issue` pair, and one `LockRect` of a managed
//! texture no draw samples.
//!
//! Three inputs are estimates, since neither the dump nor the capture records
//! them: how many constant registers each upload writes (four rows of world
//! matrix, two of light and one of ambient per object, one more on every
//! fifth, a pixel row), how much of each text reservation one `Lock` covers,
//! and which call is the fourth query call of the game's perf window, where
//! the dump shows three: it is taken to be `GetDataSize`.
//!
//! Every program and every fixed-function variant is first drawn in the
//! warm-up, so the measured frames compile nothing: their perf window counts
//! no pipeline builds and no misses (`perf.comp_*` in the metrics file). The
//! metrics file carries one `shape` record per pass with its state mix,
//! computed from the run tables; the last warm-up frame records what the
//! wrappers actually set and is checked against them, and the first measured
//! frame's draw count against the shapes, which checks the benchmark against
//! itself.

use core::{
    cell::{Cell, RefCell},
    ffi::c_void,
    fmt::Write as _,
};
use std::{
    collections::BTreeSet,
    time::{Duration, Instant, SystemTime},
};

use mtld3d_tests::{
    Harness, HarnessConfig, IndexBuffer, MemorySample, PixelShader, Query, Surface, Texture,
    TexturedVertex, VertexBuffer, VertexDeclaration, VertexShader,
};
use mtld3d_types::{
    D3DBLEND_DESTCOLOR, D3DBLEND_INVSRCALPHA, D3DBLEND_ONE, D3DBLEND_SRCALPHA, D3DBLEND_SRCCOLOR,
    D3DBLEND_ZERO, D3DCLEAR_TARGET, D3DCLEAR_ZBUFFER, D3DCMP_ALWAYS, D3DCMP_GREATEREQUAL,
    D3DCMP_LESSEQUAL, D3DCULL_CW, D3DCULL_NONE, D3DDECL_END, D3DDECLTYPE_FLOAT2,
    D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_NORMAL, D3DDECLUSAGE_POSITION, D3DDECLUSAGE_TEXCOORD,
    D3DFMT_A1R5G5B5, D3DFMT_A4R4G4B4, D3DFMT_A8R8G8B8, D3DFMT_D24S8, D3DFMT_DXT1, D3DFMT_DXT3,
    D3DFMT_DXT5, D3DFMT_INDEX16, D3DFMT_R5G6B5, D3DFOG_LINEAR, D3DFOG_NONE, D3DGETDATA_FLUSH,
    D3DISSUE_BEGIN, D3DISSUE_END, D3DLOCK_DISCARD, D3DLOCK_NOOVERWRITE, D3DPOOL_DEFAULT,
    D3DPOOL_MANAGED, D3DPRESENT_INTERVAL_IMMEDIATE, D3DPT_TRIANGLELIST, D3DPT_TRIANGLESTRIP,
    D3DQUERYTYPE_OCCLUSION, D3DRS_ALPHABLENDENABLE, D3DRS_ALPHAFUNC, D3DRS_ALPHAREF,
    D3DRS_ALPHATESTENABLE, D3DRS_COLORWRITEENABLE, D3DRS_CULLMODE, D3DRS_DESTBLEND, D3DRS_FOGCOLOR,
    D3DRS_FOGENABLE, D3DRS_FOGEND, D3DRS_FOGSTART, D3DRS_FOGTABLEMODE, D3DRS_FOGVERTEXMODE,
    D3DRS_LIGHTING, D3DRS_SRCBLEND, D3DRS_ZENABLE, D3DRS_ZFUNC, D3DRS_ZWRITEENABLE,
    D3DSAMP_ADDRESSU, D3DSAMP_ADDRESSV, D3DSAMP_MAGFILTER, D3DSAMP_MINFILTER, D3DTA_CURRENT,
    D3DTA_DIFFUSE, D3DTA_TEXTURE, D3DTADDRESS_CLAMP, D3DTADDRESS_WRAP, D3DTEXF_LINEAR,
    D3DTOP_DISABLE, D3DTOP_MODULATE, D3DTOP_SELECTARG1, D3DTOP_SELECTARG2, D3DTS_PROJECTION,
    D3DTS_TEXTURE0, D3DTS_VIEW, D3DTS_WORLD, D3DTSS_ALPHAARG1, D3DTSS_ALPHAARG2, D3DTSS_ALPHAOP,
    D3DTSS_COLORARG1, D3DTSS_COLORARG2, D3DTSS_COLOROP, D3DTSS_TEXCOORDINDEX,
    D3DTSS_TEXTURETRANSFORMFLAGS, D3DTTFF_COUNT2, D3DTTFF_DISABLE, D3DUSAGE_DYNAMIC,
    D3DUSAGE_RENDERTARGET, D3DUSAGE_WRITEONLY, D3DVERTEXELEMENT9, D3DVIEWPORT9, S_FALSE,
    render_state_defaults, sampler_state_defaults, texture_stage_state_defaults,
};

use crate::bench::{
    Class, Direction, FrameClock, FrameWork, IDENTITY_ROWS, LayerLog, Metrics, Model, PassShape,
    PassState, STRIDE, TEXTURED_DECL, TscClock, Value, def, element, material_ps, memory_section,
    ok, ratio, transform, world_rows, write_report,
};

/// The back buffer and the scene target: the game's 2624x1696, halved on each axis.
const WIDTH: u32 = 1312;
const HEIGHT: u32 = 848;
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

/// The texture ids of the pools, laid out one after another.
///
/// The transparent world's pool starts inside the opaque world's, sharing
/// its last [`SHARED_WORLD`] textures, as the game's frame shares textures
/// between the two. The render targets come after the pools' textures.
const SKY_FIRST: u16 = 0;
const SKY_COUNT: u16 = 29;
const WORLD_FIRST: u16 = SKY_FIRST + SKY_COUNT;
const WORLD_COUNT: u16 = 738;
const SHARED_WORLD: u16 = 53;
const TRANS_FIRST: u16 = WORLD_FIRST + WORLD_COUNT - SHARED_WORLD;
const TRANS_COUNT: u16 = 120;
const PARTICLE_FIRST: u16 = TRANS_FIRST + TRANS_COUNT;
const PARTICLE_COUNT: u16 = 47;
const FONT_FIRST: u16 = PARTICLE_FIRST + PARTICLE_COUNT;
const FONT_COUNT: u16 = 3;
const SUN_TEXTURE: u16 = FONT_FIRST + FONT_COUNT;
const UI_FIRST: u16 = SUN_TEXTURE + 1;
const UI_COUNT: u16 = 61;
const MINIMAP_TEXTURE: u16 = UI_FIRST + UI_COUNT;
const MINIMAP_MASK: u16 = MINIMAP_TEXTURE + 1;
const ICON_TEXTURE: u16 = MINIMAP_MASK + 1;
/// A managed texture no draw samples, rewritten every frame as a streamed one would be.
const STREAM_TEXTURE: u16 = ICON_TEXTURE + 1;
const TEXTURE_COUNT: u16 = STREAM_TEXTURE + 1;
const SCENE_TEXTURE: u16 = TEXTURE_COUNT;
const GLOW_TEXTURES: [u16; 2] = [TEXTURE_COUNT + 1, TEXTURE_COUNT + 2];

/// The textures created for each id range, in the formats and sizes the game's frame binds.
///
/// Each range cycles through its formats and edges by id. The world's
/// range gives the opaque world's bind mix: R5G6B5 and DXT1 about equally,
/// then DXT3, A8R8G8B8 and DXT5.
const TEXTURE_SETS: [TextureSet; 11] = [
    TextureSet {
        first: SKY_FIRST,
        count: SKY_COUNT,
        formats: &[D3DFMT_DXT1, D3DFMT_A8R8G8B8, D3DFMT_DXT3, D3DFMT_DXT5],
        edges: &[128, 64],
        aspect: 1,
    },
    TextureSet {
        first: WORLD_FIRST,
        count: WORLD_COUNT,
        formats: &[
            D3DFMT_DXT1,
            D3DFMT_R5G6B5,
            D3DFMT_DXT1,
            D3DFMT_R5G6B5,
            D3DFMT_DXT1,
            D3DFMT_R5G6B5,
            D3DFMT_DXT3,
            D3DFMT_DXT1,
            D3DFMT_R5G6B5,
            D3DFMT_DXT1,
            D3DFMT_R5G6B5,
            D3DFMT_DXT1,
            D3DFMT_R5G6B5,
            D3DFMT_DXT3,
            D3DFMT_DXT1,
            D3DFMT_R5G6B5,
            D3DFMT_A8R8G8B8,
            D3DFMT_R5G6B5,
            D3DFMT_DXT1,
            D3DFMT_DXT5,
        ],
        edges: &[64, 128, 64, 32],
        aspect: 1,
    },
    TextureSet {
        first: WORLD_FIRST + WORLD_COUNT,
        count: TRANS_FIRST + TRANS_COUNT - (WORLD_FIRST + WORLD_COUNT),
        formats: &[
            D3DFMT_DXT5,
            D3DFMT_DXT3,
            D3DFMT_DXT1,
            D3DFMT_A1R5G5B5,
            D3DFMT_DXT5,
            D3DFMT_DXT1,
        ],
        edges: &[64, 128],
        aspect: 1,
    },
    TextureSet {
        first: PARTICLE_FIRST,
        count: PARTICLE_COUNT,
        formats: &[
            D3DFMT_DXT5,
            D3DFMT_DXT5,
            D3DFMT_DXT3,
            D3DFMT_DXT5,
            D3DFMT_DXT1,
        ],
        edges: &[64, 32],
        aspect: 1,
    },
    // The fonts: 256x1024 A4R4G4B4 atlases.
    TextureSet {
        first: FONT_FIRST,
        count: FONT_COUNT,
        formats: &[D3DFMT_A4R4G4B4],
        edges: &[256],
        aspect: 4,
    },
    TextureSet {
        first: SUN_TEXTURE,
        count: 1,
        formats: &[D3DFMT_A8R8G8B8],
        edges: &[256],
        aspect: 1,
    },
    TextureSet {
        first: UI_FIRST,
        count: UI_COUNT,
        formats: &[D3DFMT_DXT3, D3DFMT_DXT3, D3DFMT_DXT1, D3DFMT_A8R8G8B8],
        edges: &[32, 64, 128],
        aspect: 1,
    },
    TextureSet {
        first: MINIMAP_TEXTURE,
        count: 1,
        formats: &[D3DFMT_A8R8G8B8],
        edges: &[256],
        aspect: 1,
    },
    TextureSet {
        first: MINIMAP_MASK,
        count: 1,
        formats: &[D3DFMT_DXT3],
        edges: &[64],
        aspect: 1,
    },
    TextureSet {
        first: ICON_TEXTURE,
        count: 1,
        formats: &[D3DFMT_DXT3],
        edges: &[32],
        aspect: 1,
    },
    TextureSet {
        first: STREAM_TEXTURE,
        count: 1,
        formats: &[D3DFMT_DXT1],
        edges: &[STREAM_EDGE],
        aspect: 1,
    },
];
/// Edge of the streamed texture.
const STREAM_EDGE: u32 = 64;

/// The pools the run tables draw textures from.
///
/// A pool hands out one texture a run of draws, the run lengths cycling
/// through `runs`. A mesh or particle draw that starts an object or changes
/// shaders or render states ends the run early, so the world's runs of 12
/// draws only end that way. A revisiting pool hands out each texture for two
/// runs close together, as the game's frame binds most of its textures twice
/// or more, never the same one twice in a row.
const SKY: Pool = Pool {
    slot: 0,
    first: SKY_FIRST,
    count: SKY_COUNT,
    runs: &[1, 1, 1, 2],
    revisit: true,
};
const WORLD: Pool = Pool {
    slot: 1,
    first: WORLD_FIRST,
    count: WORLD_COUNT,
    runs: &[12],
    revisit: true,
};
const TRANS: Pool = Pool {
    slot: 2,
    first: TRANS_FIRST,
    count: TRANS_COUNT,
    runs: &[1, 2, 1, 1, 2],
    revisit: true,
};
const PARTICLES: Pool = Pool {
    slot: 3,
    first: PARTICLE_FIRST,
    count: PARTICLE_COUNT,
    runs: &[2],
    revisit: true,
};
const FONTS: Pool = Pool {
    slot: 4,
    first: FONT_FIRST,
    count: FONT_COUNT,
    runs: &[4],
    revisit: true,
};
/// The interface's text, which changes font more often than the scene's.
const UI_FONTS: Pool = Pool {
    slot: 4,
    first: FONT_FIRST,
    count: FONT_COUNT,
    runs: &[1],
    revisit: true,
};
/// The interface's textures, each new until the pool wraps.
const UI: Pool = Pool {
    slot: 5,
    first: UI_FIRST,
    count: UI_COUNT,
    runs: &[2, 1, 2, 2, 1, 3],
    revisit: false,
};
const POOL_SLOTS: usize = 6;

/// The sky's first draws, blended, before the sky dome proper.
const LEAD_A: Material = Material {
    blend: [1, D3DBLEND_SRCALPHA, D3DBLEND_INVSRCALPHA],
    alpha: [1, 1],
    cull: D3DCULL_CW,
    z_write: 0,
    color_mask: 0xF,
    ops: [D3DTOP_MODULATE, D3DTOP_MODULATE],
    feed: Feed::Mesh,
    textures: Textures::Pool(&SKY),
};
const LEAD_B: Material = Material {
    blend: [1, D3DBLEND_SRCALPHA, D3DBLEND_ONE],
    cull: D3DCULL_NONE,
    textures: Textures::Off,
    ..LEAD_A
};
const LEAD_C: Material = Material {
    cull: D3DCULL_NONE,
    ..LEAD_A
};
/// The sky dome and the stars: opaque, with stale blend factors left from the lead.
const DOME: Material = Material {
    blend: [0, D3DBLEND_SRCALPHA, D3DBLEND_INVSRCALPHA],
    alpha: [0, 1],
    cull: D3DCULL_CW,
    z_write: 1,
    color_mask: 0xF,
    ops: [D3DTOP_MODULATE, D3DTOP_SELECTARG1],
    feed: Feed::Mesh,
    textures: Textures::Pool(&SKY),
};
/// The sky glow, whose draws alternate between two blends.
const GLOW_A: Material = Material {
    blend: [1, D3DBLEND_SRCALPHA, D3DBLEND_ZERO],
    ..DOME
};
const GLOW_B: Material = Material {
    blend: [1, D3DBLEND_INVSRCALPHA, D3DBLEND_ONE],
    ..DOME
};
const CLOUDS: Material = Material {
    blend: [0, D3DBLEND_INVSRCALPHA, D3DBLEND_ONE],
    ..DOME
};
/// The opaque world, with the blend factors the game leaves set while blending is off.
const OPAQUE: Material = Material {
    blend: [0, D3DBLEND_INVSRCALPHA, D3DBLEND_ONE],
    alpha: [0, 224],
    cull: D3DCULL_CW,
    z_write: 1,
    color_mask: 0xF,
    ops: [D3DTOP_MODULATE, D3DTOP_SELECTARG1],
    feed: Feed::Mesh,
    textures: Textures::Pool(&WORLD),
};
const OPAQUE_OPEN: Material = Material {
    cull: D3DCULL_NONE,
    ..OPAQUE
};
/// Alpha-tested foliage and fences: the stretch's keyed pixel program, no culling.
const KEYED: Material = Material {
    alpha: [1, 224],
    cull: D3DCULL_NONE,
    ..OPAQUE
};
const KEYED_CULLED: Material = Material {
    cull: D3DCULL_CW,
    ..KEYED
};
const ADD: Material = Material {
    blend: [1, D3DBLEND_SRCALPHA, D3DBLEND_ONE],
    alpha: [1, 1],
    cull: D3DCULL_CW,
    z_write: 0,
    color_mask: 0xF,
    ops: [D3DTOP_MODULATE, D3DTOP_MODULATE],
    feed: Feed::Mesh,
    textures: Textures::Pool(&TRANS),
};
const ADD_OPEN: Material = Material {
    cull: D3DCULL_NONE,
    ..ADD
};
const BLEND: Material = Material {
    blend: [1, D3DBLEND_SRCALPHA, D3DBLEND_INVSRCALPHA],
    ..ADD
};
const BLEND_OPEN: Material = Material {
    cull: D3DCULL_NONE,
    ..BLEND
};
const MODULATE: Material = Material {
    blend: [1, D3DBLEND_DESTCOLOR, D3DBLEND_SRCCOLOR],
    ..ADD
};
/// Depth-only draws: colour writes off, depth writes on.
const MASKED: Material = Material {
    blend: [0, D3DBLEND_SRCALPHA, D3DBLEND_ONE],
    alpha: [0, 112],
    z_write: 1,
    color_mask: 0,
    ..ADD
};
const MASKED_MODULATE: Material = Material {
    blend: [0, D3DBLEND_DESTCOLOR, D3DBLEND_SRCCOLOR],
    ..MASKED
};
/// Alpha-blended without the alpha test, writing depth.
const GLASS: Material = Material {
    blend: [1, D3DBLEND_SRCALPHA, D3DBLEND_INVSRCALPHA],
    alpha: [0, 112],
    z_write: 1,
    ..ADD
};
/// Fixed-function particles, appended to the vertex ring.
const SPARKS: Material = Material {
    blend: [1, D3DBLEND_SRCALPHA, D3DBLEND_ONE],
    alpha: [1, 1],
    cull: D3DCULL_NONE,
    z_write: 0,
    color_mask: 0xF,
    ops: [D3DTOP_MODULATE, D3DTOP_MODULATE],
    feed: Feed::Sparks,
    textures: Textures::Pool(&PARTICLES),
};
const SMOKE: Material = Material {
    blend: [1, D3DBLEND_SRCALPHA, D3DBLEND_INVSRCALPHA],
    ..SPARKS
};
/// Floating text in the world: font quads in a reservation of the vertex ring, writing depth.
const TEXT: Material = Material {
    blend: [1, D3DBLEND_SRCALPHA, D3DBLEND_INVSRCALPHA],
    z_write: 1,
    ops: [D3DTOP_SELECTARG2, D3DTOP_MODULATE],
    feed: Feed::Glyphs,
    textures: Textures::Pool(&FONTS),
    ..SPARKS
};
/// The sun's visibility test: no texture, no colour, inside the occlusion query.
const SUN_TEST: Material = Material {
    blend: [0, D3DBLEND_SRCALPHA, D3DBLEND_ONE],
    alpha: [0, 1],
    cull: D3DCULL_CW,
    z_write: 1,
    color_mask: 0,
    feed: Feed::Occlusion,
    textures: Textures::Off,
    ..SPARKS
};
const SUN: Material = Material {
    textures: Textures::Fixed(&[SUN_TEXTURE]),
    ..SPARKS
};
/// A glow pass: the pixel program over four stages, its textures given by the pass.
const GLOW: Material = Material {
    blend: [0, D3DBLEND_SRCALPHA, D3DBLEND_ONE],
    alpha: [0, 1],
    cull: D3DCULL_NONE,
    z_write: 0,
    color_mask: 0xF,
    ops: [D3DTOP_MODULATE, D3DTOP_MODULATE],
    feed: Feed::Screen,
    textures: Textures::Off,
};
const COMPOSITE: Material = Material {
    textures: Textures::Fixed(&[SCENE_TEXTURE, GLOW_TEXTURES[0]]),
    ..GLOW
};
const UI_QUAD: Material = Material {
    blend: [1, D3DBLEND_SRCALPHA, D3DBLEND_INVSRCALPHA],
    alpha: [1, 1],
    cull: D3DCULL_CW,
    z_write: 0,
    color_mask: 0xF,
    ops: [D3DTOP_MODULATE, D3DTOP_MODULATE],
    feed: Feed::Strip,
    textures: Textures::Pool(&UI),
};
const UI_ADD: Material = Material {
    blend: [1, D3DBLEND_SRCALPHA, D3DBLEND_ONE],
    ..UI_QUAD
};
const UI_TEXT: Material = Material {
    cull: D3DCULL_NONE,
    ops: [D3DTOP_SELECTARG2, D3DTOP_MODULATE],
    feed: Feed::Glyphs,
    textures: Textures::Pool(&UI_FONTS),
    ..UI_QUAD
};
/// The minimap: its image and its round mask, in a viewport of its own.
const MINIMAP: Material = Material {
    cull: D3DCULL_NONE,
    feed: Feed::Minimap,
    textures: Textures::Fixed(&[MINIMAP_TEXTURE, MINIMAP_MASK]),
    ..UI_QUAD
};
/// A programmable icon over the minimap, alpha-tested, in a viewport whose depth is cleared.
const ICON: Material = Material {
    blend: [0, D3DBLEND_SRCALPHA, D3DBLEND_INVSRCALPHA],
    alpha: [1, 224],
    ops: [D3DTOP_MODULATE, D3DTOP_SELECTARG1],
    feed: Feed::Icon,
    textures: Textures::Fixed(&[ICON_TEXTURE]),
    ..UI_QUAD
};

/// The sky: `(material of even draws, material of odd draws, pixel program, draws)`.
///
/// Stars (program 9) break up the dome; the glow (program 5) alternates its
/// two blends every draw between runs of clouds (program 7).
const SKY_RUNS: [(&Material, &Material, Option<u8>, u32); 24] = [
    (&LEAD_A, &LEAD_A, None, 1),
    (&LEAD_B, &LEAD_B, None, 1),
    (&LEAD_C, &LEAD_C, None, 1),
    (&DOME, &DOME, None, 4),
    (&DOME, &DOME, Some(9), 1),
    (&DOME, &DOME, None, 1),
    (&DOME, &DOME, Some(9), 3),
    (&DOME, &DOME, None, 2),
    (&DOME, &DOME, Some(9), 1),
    (&DOME, &DOME, None, 1),
    (&DOME, &DOME, Some(9), 1),
    (&DOME, &DOME, None, 9),
    (&DOME, &DOME, Some(9), 1),
    (&DOME, &DOME, None, 1),
    (&DOME, &DOME, Some(9), 1),
    (&DOME, &DOME, None, 1),
    (&DOME, &DOME, Some(9), 1),
    (&DOME, &DOME, None, 2),
    (&GLOW_A, &GLOW_B, Some(5), 10),
    (&CLOUDS, &CLOUDS, Some(7), 8),
    (&GLOW_A, &GLOW_B, Some(5), 14),
    (&CLOUDS, &CLOUDS, Some(7), 7),
    (&GLOW_A, &GLOW_B, Some(5), 16),
    (&CLOUDS, &CLOUDS, Some(7), 5),
];
/// The sky's first draws, which the game draws under a viewport of their own depth range.
const SKY_LEAD: u32 = 3;
const SKY_DRAWS: u32 = sky_draws();
/// The opaque world: `(vertex program, draws, opaque pixel program, keyed pixel program)`.
///
/// The game's stretches in its order: a few fixed-function pixel stages
/// first, then terrain-like stretches under two programs, then the four
/// stretches that draw most of the world, and a short tail.
const STRETCHES: [(u8, u32, Option<u8>, Option<u8>); 20] = [
    (0, 1, None, None),
    (1, 11, None, None),
    (2, 5, None, None),
    (3, 3, None, None),
    (4, 4, None, None),
    (5, 21, None, None),
    (6, 12, Some(4), Some(2)),
    (7, 48, Some(4), Some(2)),
    (8, 1, Some(4), Some(4)),
    (9, 258, Some(0), Some(1)),
    (10, 278, Some(0), Some(1)),
    (11, 46, Some(0), Some(1)),
    (12, 474, Some(0), Some(1)),
    (13, 1977, Some(0), Some(1)),
    (14, 1, Some(10), Some(10)),
    (15, 4, Some(6), Some(6)),
    (16, 2, Some(6), Some(6)),
    (17, 9, Some(6), Some(6)),
    (18, 7, Some(6), Some(6)),
    (1, 1, None, None),
];
const OPAQUE_DRAWS: u32 = stretch_draws();
/// The opaque world's batches, cycled across the stretches: opaque runs between keyed ones.
const BATCH: [(&Material, u32); 11] = [
    (&OPAQUE, 5),
    (&KEYED, 1),
    (&OPAQUE, 4),
    (&OPAQUE_OPEN, 1),
    (&OPAQUE, 3),
    (&KEYED, 2),
    (&OPAQUE, 6),
    (&KEYED_CULLED, 1),
    (&OPAQUE, 3),
    (&OPAQUE_OPEN, 1),
    (&OPAQUE, 3),
];
/// The transparent world, cycled: `(vertex program, pixel program, material, draws)`.
const TRANS_RUNS: [(u8, u8, &Material, u32); 22] = [
    (13, 1, &ADD, 3),
    (19, 2, &ADD, 2),
    (20, 1, &BLEND_OPEN, 2),
    (21, 2, &BLEND, 2),
    (22, 3, &MODULATE, 2),
    (13, 0, &MASKED_MODULATE, 1),
    (23, 2, &ADD, 2),
    (12, 0, &MASKED, 3),
    (13, 0, &GLASS, 2),
    (24, 1, &BLEND_OPEN, 2),
    (25, 8, &ADD_OPEN, 2),
    (26, 1, &ADD, 2),
    (19, 3, &MODULATE, 2),
    (20, 2, &ADD, 2),
    (27, 1, &BLEND, 2),
    (12, 8, &BLEND_OPEN, 2),
    (28, 1, &ADD, 2),
    (9, 3, &MODULATE, 2),
    (10, 2, &ADD, 3),
    (29, 4, &ADD, 1),
    (11, 1, &ADD, 2),
    (14, 2, &BLEND, 2),
];
const TRANS_DRAWS: u32 = 699;
/// The effects, cycled: particles between runs of floating text.
const FX_RUNS: [(&Material, u32); 10] = [
    (&SPARKS, 5),
    (&TEXT, 4),
    (&SPARKS, 6),
    (&SMOKE, 1),
    (&TEXT, 4),
    (&SPARKS, 5),
    (&TEXT, 3),
    (&SPARKS, 4),
    (&SMOKE, 2),
    (&TEXT, 4),
];
const FX_DRAWS: u32 = 1134;
/// The sun's two draws.
const SUN_DRAWS: u32 = 2;
/// The scene's last two draws: the sun's visibility test inside the query, then its flare.
const SUN_RUNS: [&Material; SUN_DRAWS as usize] = [&SUN_TEST, &SUN];
const SCENE_DRAWS: u32 = SKY_DRAWS + OPAQUE_DRAWS + TRANS_DRAWS + FX_DRAWS + SUN_DRAWS;
/// The interface after the composite: `(material, draws)`, in the game's order.
const UI_RUNS: [(&Material, u32); 32] = [
    (&COMPOSITE, 1),
    (&UI_QUAD, 10),
    (&UI_TEXT, 1),
    (&UI_QUAD, 10),
    (&UI_TEXT, 2),
    (&MINIMAP, 1),
    (&UI_QUAD, 5),
    (&ICON, 1),
    (&UI_QUAD, 13),
    (&UI_TEXT, 1),
    (&UI_QUAD, 3),
    (&UI_TEXT, 1),
    (&UI_QUAD, 1),
    (&UI_ADD, 1),
    (&UI_QUAD, 2),
    (&UI_ADD, 1),
    (&UI_QUAD, 8),
    (&UI_TEXT, 1),
    (&UI_QUAD, 5),
    (&UI_TEXT, 1),
    (&UI_QUAD, 32),
    (&UI_TEXT, 1),
    (&UI_QUAD, 2),
    (&UI_ADD, 1),
    (&UI_TEXT, 1),
    (&UI_QUAD, 19),
    (&UI_ADD, 1),
    (&UI_QUAD, 9),
    (&UI_TEXT, 2),
    (&UI_QUAD, 9),
    (&UI_TEXT, 2),
    (&UI_QUAD, 1),
];
const UI_DRAWS: u32 = ui_draws();
/// The glow passes, one draw each.
const GLOW_PASSES: u32 = 3;
/// Draws per frame: the scene, the glow passes and the back buffer's.
const DRAWS_PER_FRAME: u32 = SCENE_DRAWS + UI_DRAWS + GLOW_PASSES;

/// Vertex programs: the scene's 30, then the minimap icon's.
const SCENE_VS: u8 = 30;
const ICON_VS: u8 = SCENE_VS;
/// Pixel programs: the scene's 11, then the two glow programs and the composite.
const SCENE_PS: u8 = 11;
const GLOW_PS: [u8; 2] = [SCENE_PS, SCENE_PS + 1];
const COMPOSITE_PS: u8 = SCENE_PS + 2;

/// Index counts of the static draws, cycled: the game's buckets from 6 to 3,600.
const INDEX_COUNTS: [u32; 32] = [
    60, 180, 546, 180, 18, 546, 180, 60, 1650, 546, 180, 60, 546, 180, 18, 60, 546, 180, 3600, 60,
    180, 546, 1650, 180, 60, 18, 546, 180, 60, 546, 1650, 6,
];
/// Quads of the particle draws, cycled.
const SPARK_QUADS: [u32; 10] = [1, 2, 3, 1, 4, 2, 10, 3, 1, 5];
/// Quads of the text draws, cycled.
const GLYPH_QUADS: [u32; 8] = [2, 8, 1, 16, 4, 3, 26, 6];
/// Vertices a text draw reserves in the ring, cycled, about the game's mix of the two sizes.
const GLYPH_RESERVE: [u32; 7] = [4096, 2048, 4096, 2048, 4096, 2048, 2048];
/// Draws per object of the static draws, cycled.
const OBJECT_RUNS: [u32; 5] = [4, 3, 5, 3, 4];
/// Every this many continuing programmable draws sends the view rows again, unchanged.
const VIEW_ROWS_EVERY: u32 = 5;
/// Every this many objects of the static draws changes the fog colour.
const FOG_EVERY: u32 = 3;
/// Every this many objects changes the pixel-shader row the programmable draws send.
const TINT_EVERY: u32 = 3;
/// A texture whose id is a multiple of this clamps; every other one wraps.
const CLAMP_EVERY: u16 = 7;

/// Quads per edge of a static mesh, and what that makes.
const GRID: u32 = 30;
const MESH_VERTS: u32 = (GRID + 1) * (GRID + 1);
/// The static meshes, one picked per object.
const MESHES: u32 = 64;
/// Model vertex: position, normal, one texture coordinate.
const MODEL_DECL: [D3DVERTEXELEMENT9; 4] = [
    element(0, D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_POSITION),
    element(12, D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_NORMAL),
    element(24, D3DDECLTYPE_FLOAT2, D3DDECLUSAGE_TEXCOORD),
    D3DDECL_END,
];
const MODEL_STRIDE: u32 = 32;
/// The game's dynamic vertex buffer, 2.75 MiB.
const RING_VERTEX_BYTES: u32 = 2_883_584;
/// The dynamic index ring the interface's strips write into.
const RING_INDEX_BYTES: u32 = 256 * 1024;
/// The most quads one dynamic draw writes, and the static quad list's capacity.
const MAX_QUADS: u32 = 32;

/// The viewport depth ranges the game's scene draws under.
const SKY_DEPTH: (f32, f32) = (0.975, 0.98);
const WORLD_DEPTH: (f32, f32) = (0.0, 0.94);
const SUN_DEPTH: (f32, f32) = (0.995, 1.0);
/// The minimap's and its icon's viewports, at half the game's coordinates.
const MINIMAP_RECT: [u32; 4] = [1155, 21, 140, 139];
const ICON_RECT: [u32; 4] = [1208, 71, 37, 37];
/// The render states the material wrappers track, in [`Material`] order.
const TRACKED: [u32; 8] = [
    D3DRS_ALPHABLENDENABLE,
    D3DRS_SRCBLEND,
    D3DRS_DESTBLEND,
    D3DRS_ALPHATESTENABLE,
    D3DRS_ALPHAREF,
    D3DRS_CULLMODE,
    D3DRS_ZWRITEENABLE,
    D3DRS_COLORWRITEENABLE,
];
/// The stage states [`Material::ops`] sets, in order.
const OPS: [u32; 2] = [D3DTSS_COLOROP, D3DTSS_ALPHAOP];
/// The fog colours the objects cycle through.
const FOG_COLORS: [u32; 4] = [0xFF50_6070, 0xFF48_5868, 0xFF58_6878, 0xFF40_5060];
/// The `vs_2_0` programs' light rows `c90..c91`: direction to the light and its colour.
const LIGHT_ROWS: [f32; 8] = [
    0.0, 0.0, -1.0, 0.0, //
    0.8, 0.8, 0.7, 1.0,
];
/// The first light row the `vs_2_0` programs read.
const LIGHT_ROW: u32 = 90;
/// The ambient row `c92`, sent with every object.
const AMBIENT_ROW: u32 = 92;
/// A row the programs do not read, sent on every [`EXTRA_EVERY`]th object.
const EXTRA_ROW: u32 = 93;
const EXTRA_EVERY: u32 = 5;
/// Of every ten continuing static draws, those that set the world transform again.
const WORLD_AGAIN: [bool; 10] = [
    false, true, true, false, true, true, false, true, true, true,
];
/// A particle run starts an emitter, with its own world transform, every this many draws.
///
/// The draws between continue the emitter and set no transform.
const EMITTER_DRAWS: u32 = 2;
/// Every this many objects, a static draw binds the second model declaration.
///
/// It has the same layout as the first, so the draw pays for a declaration
/// change, as the game's model formats alternate, without a new program.
const SECOND_DECL_EVERY: u32 = 28;
/// Linear vertex fog over the scene's depth range.
const FOG_START: f32 = 0.3;
const FOG_END: f32 = 1.4;
const WARM_UP_FRAMES: u32 = 60;
/// The measured phase is at least this many frames and at least one perf window long.
const MEASURED_FRAMES: usize = 600;
/// The longest the occlusion query's flushed read may stay pending before the benchmark fails.
const QUERY_DEADLINE: Duration = Duration::from_secs(5);

/// One busy frame, repeatedly: warm up, then time the frames.
#[test]
#[ignore = "benchmark, run by `make bench`"]
fn wow_112_busy_frame() {
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
        depth_format: Some(D3DFMT_D24S8),
        presentation_interval: D3DPRESENT_INTERVAL_IMMEDIATE,
        config_entries: GAME_CONFIG,
        ..HarnessConfig::default()
    });
    let started = TscClock::now();
    let frame = Frame::new(&h);
    for tick in 0..WARM_UP_FRAMES {
        assert!(h.pump(), "WM_QUIT during warm-up");
        // The last warm-up frame records what the wrappers set; every frame
        // makes the same calls, so it stands for the measured ones.
        frame.recorder.on.set(tick + 1 == WARM_UP_FRAMES);
        frame.render(tick);
        frame.count_present();
        ok(h.present(), "Present");
    }
    frame.recorder.on.set(false);
    let shapes = pass_shapes();
    assert_eq!(
        frame.recorder.rows(),
        shapes.iter().map(row).collect::<Vec<_>>(),
        "the frame sets what its run tables say"
    );
    let log = LayerLog::find(since);
    let warm_up = TscClock::since(started);
    let warm = MemorySample::now();

    let mut tick = WARM_UP_FRAMES;
    let start = log.start_span(started, || {
        assert!(h.pump(), "WM_QUIT outside the measured frames");
        frame.render(tick);
        frame.count_present();
        ok(h.present(), "Present");
        tick += 1;
    });
    let pending_from = frame.calls.pending_polls.get();
    let mut clock = FrameClock::start(MEASURED_FRAMES * 4);
    let mut one_frame = Vec::new();
    while clock.frames() < MEASURED_FRAMES || clock.elapsed() < start.length() {
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
    let end = MemorySample::now();
    let pending = frame.calls.pending_polls.get() - pending_from;
    let span = start.end(&log, || {
        assert!(h.pump(), "WM_QUIT outside the measured frames");
        frame.render(tick);
        frame.count_present();
        ok(h.present(), "Present");
        tick += 1;
    });

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
    let mut passes = String::new();
    for shape in &shapes {
        let _ = writeln!(passes, "  {}", row(shape));
    }
    let body = format!(
        "shape: scene {WIDTH}x{HEIGHT} A8R8G8B8 target texture + D24S8 surface; {GLOW_PASSES} \
         glow passes {GLOW_WIDTH}x{GLOW_HEIGHT}; back buffer {WIDTH}x{HEIGHT} X8R8G8B8 + D24S8; \
         {GAME_CONFIG}\n\
         source: WoW-14623 frame 3 of its F12 dump and Metal capture (main 66b9a279, \
         2026-10-02), call rates from its next PERF=1 window\n\
         per frame: {DRAWS_PER_FRAME} draws in 5 passes (scene {SCENE_DRAWS}: sky {SKY_DRAWS}, \
         opaque world {OPAQUE_DRAWS}, transparent world {TRANS_DRAWS}, effects {FX_DRAWS}, sun 2; \
         glow {GLOW_PASSES}; back buffer {UI_DRAWS}: composite 1 + interface)\n\
         programs: {SCENE_VS} scene vs_2_0, {SCENE_PS} scene ps_2_0, two glow ps_2_0, a \
         composite ps_2_0 and an icon vs_2_0; {TEXTURE_COUNT} textures\n\
         estimates: constant rows per upload (4 world, 2 light and 1 ambient per object, 1 \
         more on every {EXTRA_EVERY}th, 1 pixel row), the share of a text reservation each \
         lock covers, and `GetDataSize` as the game's fourth query call\n\
         passes (from the run tables, matched by what the last warm-up frame set):\n{passes}\
         warm-up: {WARM_UP_FRAMES} frames in {warm_up:.2?}\n\
         measured: {frames} frames in {elapsed:.2?} (at least {MEASURED_FRAMES} frames \
         and {length:?}, {start})\n\
         frame time (Present to Present): {row}\n\
         API work (Present return to Present call): {work_row}\n\
         API work per draw: {per_draw_ns} ns (mean API work over {DRAWS_PER_FRAME} draws)\n\
         D3D9 calls in one measured frame: {total} ({calls_per_draw:.2} per draw; a lock \
         and a scene count two calls, the counted-only rows none)\n{mix}\
         occlusion reads that found the query pending: {pending} over the measured frames\n\
         {memory}{perf}{warm_up_compiles}",
        frames = stats.frames,
        elapsed = clock.elapsed(),
        row = stats.row(),
        work_row = work.row(),
        calls_per_draw = f64::from(total) / f64::from(DRAWS_PER_FRAME),
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
    metrics.memory(&before, &warm, &end);
    metrics.meta("window_s", &span.window_s());
    metrics.perf(&span.perf_kv(&log), &FrameWork::Fixed);
    metrics.meta("backbuffer", &format!("{WIDTH}x{HEIGHT}"));
    metrics.shapes(&shapes);
    write_report(&metrics, &log, &body);
}

/// The five passes of one frame, in the order [`Frame::render`] draws them, from the run tables.
///
/// # Panics
/// Panics if the passes do not add up to [`DRAWS_PER_FRAME`].
fn pass_shapes() -> Vec<PassShape> {
    let mut passes = Vec::new();
    let mut add = |width, height, steps: &[Step]| {
        let mut tally = Tally::new(width, height);
        for step in steps {
            tally.add(Look::of(step));
        }
        passes.push(tally.finish());
    };
    add(WIDTH, HEIGHT, &scene_steps());
    for step in glow_steps() {
        add(GLOW_WIDTH, GLOW_HEIGHT, &[step]);
    }
    add(WIDTH, HEIGHT, &ui_steps());
    let draws: u32 = passes.iter().map(|pass| pass.draws).sum();
    assert_eq!(draws, DRAWS_PER_FRAME, "the pass shapes cover every draw");
    passes
}

/// A pass shape as one line, for comparing what was set with what the tables say.
fn row(pass: &PassShape) -> String {
    let mut line = format!(
        "{}x{} draws={} ff_vs={} ff_ps={} textures={}",
        pass.width, pass.height, pass.draws, pass.ff_vs, pass.ff_ps, pass.textures
    );
    if let Some(state) = &pass.state {
        state.write_keys(&mut line);
    }
    line
}

/// The scene's draws: the sky, the opaque world, the transparent world, the effects, the sun.
fn scene_steps() -> Vec<Step> {
    let mut build = Builder::new();
    for (even, odd, ps, draws) in SKY_RUNS {
        for at in 0..draws {
            let material = if at % 2 == 0 { even } else { odd };
            build.push(material, None, ps, at == 0);
        }
    }
    let mut batch = 0;
    let mut left = BATCH[0].1;
    for (vs, draws, opaque_ps, keyed_ps) in STRETCHES {
        for _ in 0..draws {
            let material = BATCH[batch].0;
            let ps = if material.alpha[0] == 0 {
                opaque_ps
            } else {
                keyed_ps
            };
            build.push(material, Some(vs), ps, left == BATCH[batch].1);
            left -= 1;
            if left == 0 {
                batch = (batch + 1) % BATCH.len();
                left = BATCH[batch].1;
            }
        }
    }
    for (vs, ps, material, draws) in TRANS_RUNS.iter().cycle() {
        let world = slot(SKY_DRAWS + OPAQUE_DRAWS + TRANS_DRAWS);
        for at in 0..*draws {
            if build.steps.len() < world {
                build.push(material, Some(*vs), Some(*ps), at == 0);
            }
        }
        if build.steps.len() == world {
            break;
        }
    }
    let effects = slot(SCENE_DRAWS - SUN_DRAWS);
    for (material, draws) in FX_RUNS.iter().cycle() {
        for at in 0..*draws {
            if build.steps.len() < effects {
                build.push(material, None, None, at == 0);
            }
        }
        if build.steps.len() == effects {
            break;
        }
    }
    for material in SUN_RUNS {
        build.push(material, None, None, true);
    }
    assert_eq!(
        build.steps.len(),
        slot(SCENE_DRAWS),
        "the scene tables add up"
    );
    build.steps
}

/// The three glow passes: the scene into the first target, then back and forth.
fn glow_steps() -> [Step; 3] {
    let sources = [SCENE_TEXTURE, GLOW_TEXTURES[0], GLOW_TEXTURES[1]];
    let programs = [GLOW_PS[0], GLOW_PS[1], GLOW_PS[1]];
    core::array::from_fn(|pass| Step {
        material: &GLOW,
        vs: None,
        ps: Some(programs[pass]),
        textures: [Some(sources[pass]); 4],
        object: Some(u32::try_from(pass).expect("a pass index fits u32")),
        size: 1,
        reserve: 0,
        view_rows: false,
        world: true,
        run: true,
    })
}

/// The back buffer's draws: the composite, then the interface.
fn ui_steps() -> Vec<Step> {
    let mut build = Builder::new();
    for (material, draws) in UI_RUNS {
        for at in 0..draws {
            let (vs, ps) = match material.feed {
                Feed::Screen => (None, Some(COMPOSITE_PS)),
                Feed::Icon => (Some(ICON_VS), None),
                _ => (None, None),
            };
            build.push(material, vs, ps, at == 0);
        }
    }
    build.steps
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

/// The textures one id range is created with.
struct TextureSet {
    first: u16,
    count: u16,
    formats: &'static [u32],
    edges: &'static [u32],
    /// Height over width.
    aspect: u32,
}

/// A range of texture ids the run tables hand out, one a run of draws.
struct Pool {
    /// The pool's place in a [`Builder`]'s cursors, shared by pools that never meet in a pass.
    slot: usize,
    first: u16,
    count: u16,
    runs: &'static [u32],
    revisit: bool,
}

/// The render states every draw of a material sets, and where its geometry and textures come from.
struct Material {
    /// `ALPHABLENDENABLE`, `SRCBLEND` and `DESTBLEND`.
    blend: [u32; 3],
    /// `ALPHATESTENABLE` and `ALPHAREF`.
    alpha: [u32; 2],
    cull: u32,
    z_write: u32,
    color_mask: u32,
    /// Stage 0's `COLOROP` and `ALPHAOP` when the pixel stage is fixed function.
    ops: [u32; 2],
    feed: Feed,
    textures: Textures,
}

impl Material {
    /// Whether a draw of `other` sets the same render and stage states as one of this.
    fn same_state(&self, other: &Self) -> bool {
        self.blend == other.blend
            && self.alpha == other.alpha
            && self.cull == other.cull
            && self.z_write == other.z_write
            && self.color_mask == other.color_mask
            && self.ops == other.ops
    }
}

/// Where a draw's vertices come from.
enum Feed {
    /// A static mesh, indexed triangles.
    Mesh,
    /// Particle quads appended to the vertex ring, sized to what they draw.
    Sparks,
    /// Text quads in a reservation of the vertex ring, written in one or two locks.
    Glyphs,
    /// Particle quads drawn inside the occlusion query.
    Occlusion,
    /// A full-target strip of four vertices and four ring indices.
    Screen,
    /// An interface strip of four vertices and four ring indices.
    Strip,
    /// An interface strip in the minimap's viewport.
    Minimap,
    /// A static quad in the icon's viewport, after a depth clear.
    Icon,
}

/// Where a draw's textures come from.
enum Textures {
    Off,
    Pool(&'static Pool),
    /// Fixed ids, stage 0 first.
    Fixed(&'static [u16]),
}

/// One draw of a pass, as the run tables lay it out.
struct Step {
    material: &'static Material,
    /// The vertex program, `None` for the fixed-function pipeline.
    vs: Option<u8>,
    /// The pixel program, `None` for the fixed texture stages.
    ps: Option<u8>,
    textures: [Option<u16>; 4],
    /// The object a draw starts, numbered within its pass; `None` continues the one before.
    object: Option<u32>,
    /// Index count of a mesh draw, quad count of a dynamic one.
    size: u32,
    /// Vertices a text draw reserves in the ring.
    reserve: u32,
    /// A continuing programmable draw that sends the view rows again, unchanged.
    view_rows: bool,
    /// The draw sets the world transform, as every draw but some continuing static ones do.
    world: bool,
    /// The first draw of a table run, where a fixed-function draw sets its stage operations.
    run: bool,
}

/// Lays out a pass's steps, handing out textures, sizes and object numbers as its runs go.
struct Builder {
    steps: Vec<Step>,
    /// Each pool's run count and the draws left in its current run.
    pools: [(u32, u32); POOL_SLOTS],
    meshes: u32,
    objects: u32,
    /// Static draws left in the current object, and the objects of static draws so far.
    object_left: u32,
    mesh_objects: u32,
    /// Continuing programmable draws so far, which decide when the view rows go again.
    continued: u32,
    /// Continuing static draws so far, which decide when the world transform goes again.
    continued_meshes: u32,
    sparks: u32,
    /// Draws of the current particle run so far, which decide when an emitter starts.
    emitter_draws: u32,
    glyphs: u32,
}

impl Builder {
    const fn new() -> Self {
        Self {
            steps: Vec::new(),
            pools: [(0, 0); POOL_SLOTS],
            meshes: 0,
            objects: 0,
            object_left: 0,
            mesh_objects: 0,
            continued: 0,
            continued_meshes: 0,
            sparks: 0,
            emitter_draws: 0,
            glyphs: 0,
        }
    }

    /// The next texture of `pool`: a run's texture until the run is spent.
    fn texture(&mut self, pool: &Pool) -> u16 {
        let (run, left) = &mut self.pools[pool.slot];
        if *left == 0 {
            *run += 1;
            *left = pool.runs[slot(*run) % pool.runs.len()];
        }
        *left -= 1;
        let at = if pool.revisit {
            *run / 2 + 2 * (*run % 2)
        } else {
            *run
        };
        pool.first + u16::try_from(at % u32::from(pool.count)).expect("a pool index fits u16")
    }

    const fn next_object(&mut self) -> u32 {
        self.objects += 1;
        self.objects - 1
    }

    fn push(&mut self, material: &'static Material, vs: Option<u8>, ps: Option<u8>, run: bool) {
        let restates = self.steps.last().is_none_or(|last| {
            last.vs != vs || last.ps != ps || !last.material.same_state(material)
        });
        let (object, size, reserve) = match material.feed {
            Feed::Mesh => {
                let starts = self.object_left == 0;
                if starts {
                    self.object_left = OBJECT_RUNS[slot(self.mesh_objects) % OBJECT_RUNS.len()];
                    self.mesh_objects += 1;
                }
                self.object_left -= 1;
                let size = INDEX_COUNTS[slot(self.meshes) % INDEX_COUNTS.len()];
                self.meshes += 1;
                (starts.then(|| self.next_object()), size, 0)
            }
            Feed::Sparks | Feed::Occlusion => {
                let size = SPARK_QUADS[slot(self.sparks) % SPARK_QUADS.len()];
                self.sparks += 1;
                if run {
                    self.emitter_draws = 0;
                }
                let starts = !matches!(material.feed, Feed::Sparks)
                    || self.emitter_draws.is_multiple_of(EMITTER_DRAWS);
                self.emitter_draws += 1;
                (starts.then(|| self.next_object()), size, 0)
            }
            Feed::Glyphs => {
                let at = slot(self.glyphs);
                self.glyphs += 1;
                (
                    Some(self.next_object()),
                    GLYPH_QUADS[at % GLYPH_QUADS.len()],
                    GLYPH_RESERVE[at % GLYPH_RESERVE.len()],
                )
            }
            Feed::Screen | Feed::Strip | Feed::Minimap | Feed::Icon => {
                (Some(self.next_object()), 1, 0)
            }
        };
        // As in the game, a mesh or emitter changes its texture whenever it
        // starts an object or changes shaders or render states, so those
        // changes land on draws that rebind a texture anyway.
        let fresh =
            matches!(material.feed, Feed::Mesh | Feed::Sparks) && (object.is_some() || restates);
        let mut textures = [None; 4];
        match material.textures {
            Textures::Off => {}
            Textures::Pool(pool) => {
                if fresh {
                    self.pools[pool.slot].1 = 0;
                }
                textures[0] = Some(self.texture(pool));
            }
            Textures::Fixed(ids) => {
                for (bound, id) in textures.iter_mut().zip(ids) {
                    *bound = Some(*id);
                }
            }
        }
        let view_rows = object.is_none() && vs.is_some() && {
            self.continued += 1;
            self.continued.is_multiple_of(VIEW_ROWS_EVERY)
        };
        // A continuing static draw sets the world again in a pattern; a continuing particle never.
        let world = object.is_some()
            || (matches!(material.feed, Feed::Mesh) && {
                self.continued_meshes += 1;
                WORLD_AGAIN[slot(self.continued_meshes) % WORLD_AGAIN.len()]
            });
        self.steps.push(Step {
            material,
            vs,
            ps,
            textures,
            object,
            size,
            reserve,
            view_rows,
            world,
            run,
        });
    }
}

/// What one draw had bound, in the terms of a `shape` record.
struct Look {
    vs: Option<u8>,
    ps: Option<u8>,
    textures: [Option<u16>; 4],
    blend: [u32; 3],
    alpha: [u32; 2],
    cull: u32,
    z_write: u32,
    color_mask: u32,
}

impl Look {
    /// What the run tables say `step` binds.
    const fn of(step: &Step) -> Self {
        let material = step.material;
        Self {
            vs: step.vs,
            ps: step.ps,
            textures: step.textures,
            blend: material.blend,
            alpha: material.alpha,
            cull: material.cull,
            z_write: material.z_write,
            color_mask: material.color_mask,
        }
    }
}

/// One pass's shape, counted draw by draw.
struct Tally {
    shape: PassShape,
    last: Option<Look>,
    vs: BTreeSet<u8>,
    ps: BTreeSet<u8>,
    textures: BTreeSet<u16>,
}

impl Tally {
    const fn new(width: u32, height: u32) -> Self {
        Self {
            shape: PassShape {
                width,
                height,
                draws: 0,
                ff_vs: 0,
                ff_ps: 0,
                textures: 0,
                state: Some(PassState {
                    blend: 0,
                    atest: 0,
                    zwrite_off: 0,
                    cull_none: 0,
                    cmask0: 0,
                    vs_sw: 0,
                    ps_sw: 0,
                    tex_sw: 0,
                    blend_sw: 0,
                    atest_sw: 0,
                    cull_sw: 0,
                    vs_n: 0,
                    ps_n: 0,
                    tex_n: 0,
                }),
            },
            last: None,
            vs: BTreeSet::new(),
            ps: BTreeSet::new(),
            textures: BTreeSet::new(),
        }
    }

    fn add(&mut self, look: Look) {
        let shape = &mut self.shape;
        shape.draws += 1;
        shape.ff_vs += u32::from(look.vs.is_none());
        shape.ff_ps += u32::from(look.ps.is_none());
        for texture in look.textures.iter().flatten() {
            shape.textures += 1;
            self.textures.insert(*texture);
        }
        self.vs.extend(look.vs);
        self.ps.extend(look.ps);
        let state = shape.state.as_mut().expect("a tally counts the state mix");
        state.blend += u32::from(look.blend[0] != 0);
        state.atest += u32::from(look.alpha[0] != 0);
        state.zwrite_off += u32::from(look.z_write == 0);
        state.cull_none += u32::from(look.cull == D3DCULL_NONE);
        state.cmask0 += u32::from(look.color_mask == 0);
        if let Some(last) = &self.last {
            state.vs_sw += u32::from(last.vs != look.vs);
            state.ps_sw += u32::from(last.ps != look.ps);
            state.tex_sw += u32::from(last.textures[0] != look.textures[0]);
            state.blend_sw += u32::from(last.blend != look.blend);
            state.atest_sw += u32::from(last.alpha != look.alpha);
            state.cull_sw += u32::from(last.cull != look.cull);
        }
        self.last = Some(look);
    }

    fn finish(mut self) -> PassShape {
        let count = |set: usize| u32::try_from(set).expect("a distinct count fits u32");
        if let Some(state) = self.shape.state.as_mut() {
            state.vs_n = count(self.vs.len());
            state.ps_n = count(self.ps.len());
            state.tex_n = count(self.textures.len());
        }
        self.shape
    }
}

/// What the wrappers set at each draw of one frame, while switched on.
struct Recorder {
    on: Cell<bool>,
    passes: RefCell<Vec<Tally>>,
}

impl Recorder {
    /// Start a pass of `width` x `height`.
    fn pass(&self, width: u32, height: u32) {
        if self.on.get() {
            self.passes.borrow_mut().push(Tally::new(width, height));
        }
    }

    fn draw(&self, look: impl FnOnce() -> Look) {
        if self.on.get() {
            self.passes
                .borrow_mut()
                .last_mut()
                .expect("a draw follows a pass start")
                .add(look());
        }
    }

    /// The recorded passes as [`row`] lines.
    fn rows(&self) -> Vec<String> {
        self.passes
            .take()
            .into_iter()
            .map(|tally| row(&tally.finish()))
            .collect()
    }
}

/// How many D3D9 calls of each kind the frame has made, counted where it makes them.
#[derive(Default)]
struct Calls {
    draw: Cell<u32>,
    set_texture: Cell<u32>,
    set_transform: Cell<u32>,
    set_render_state: Cell<u32>,
    set_texture_stage_state: Cell<u32>,
    set_sampler_state: Cell<u32>,
    set_pixel_shader: Cell<u32>,
    set_vertex_shader: Cell<u32>,
    set_vs_const: Cell<u32>,
    set_ps_const: Cell<u32>,
    set_vertex_declaration: Cell<u32>,
    set_stream_source: Cell<u32>,
    set_indices: Cell<u32>,
    /// `Lock` and `Unlock` pairs on the dynamic vertex ring.
    vb_lock: Cell<u32>,
    /// The vertex-ring locks that started the ring over with `D3DLOCK_DISCARD`.
    vb_discard: Cell<u32>,
    /// `Lock` and `Unlock` pairs on the dynamic index ring.
    ib_lock: Cell<u32>,
    /// `LockRect` and `UnlockRect` pairs.
    lock_rect: Cell<u32>,
    set_render_target: Cell<u32>,
    set_depth_stencil: Cell<u32>,
    set_viewport: Cell<u32>,
    clear: Cell<u32>,
    query_issue: Cell<u32>,
    /// The flushed `GetData` that found the occlusion query answered, one a frame.
    query_get_data: Cell<u32>,
    /// `GetDataSize` before the read, one a frame.
    query_data_size: Cell<u32>,
    /// `BeginScene` and `EndScene` pairs.
    scene: Cell<u32>,
    present: Cell<u32>,
    /// `GetData` polls that found the query still pending, which vary from run to run.
    pending_polls: Cell<u64>,
}

impl Calls {
    /// Every counter as `(name, count, D3D9 calls one count stands for)`.
    fn rows(&self) -> Vec<(&'static str, u32, u32)> {
        [
            ("draw", &self.draw, 1),
            ("set_texture", &self.set_texture, 1),
            ("set_transform", &self.set_transform, 1),
            ("set_render_state", &self.set_render_state, 1),
            ("set_texture_stage_state", &self.set_texture_stage_state, 1),
            ("set_sampler_state", &self.set_sampler_state, 1),
            ("set_pixel_shader", &self.set_pixel_shader, 1),
            ("set_vertex_shader", &self.set_vertex_shader, 1),
            ("set_vs_const_f", &self.set_vs_const, 1),
            ("set_ps_const_f", &self.set_ps_const, 1),
            ("set_vertex_declaration", &self.set_vertex_declaration, 1),
            ("set_stream_source", &self.set_stream_source, 1),
            ("set_indices", &self.set_indices, 1),
            ("vb_lock", &self.vb_lock, 2),
            ("vb_discard", &self.vb_discard, 0),
            ("ib_lock", &self.ib_lock, 2),
            ("lock_rect", &self.lock_rect, 2),
            ("set_render_target", &self.set_render_target, 1),
            ("set_depth_stencil", &self.set_depth_stencil, 1),
            ("set_viewport", &self.set_viewport, 1),
            ("clear", &self.clear, 1),
            ("query_issue", &self.query_issue, 1),
            ("query_get_data", &self.query_get_data, 1),
            ("query_data_size", &self.query_data_size, 1),
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

/// A static mesh's buffers.
struct Mesh<'h> {
    vertices: VertexBuffer<'h>,
    indices: IndexBuffer<'h>,
}

/// Every resource the frame uses, created once, what it last set, and the calls it makes.
struct Frame<'h> {
    h: &'h Harness,
    back_buffer: Surface<'h>,
    /// The device's own depth, which the back buffer passes keep.
    back_depth: Surface<'h>,
    /// The scene's depth, which the glow passes keep bound under their smaller targets.
    scene_depth: Surface<'h>,
    /// The scene target, then the two glow targets.
    targets: [Target<'h>; 3],
    /// The pools' textures, by id.
    textures: Vec<Texture<'h>>,
    meshes: Vec<Mesh<'h>>,
    /// The icon's quad, in the model layout.
    icon: Mesh<'h>,
    /// Quad-list indices for the particle and text quads.
    quad_list: IndexBuffer<'h>,
    ring_vb: VertexBuffer<'h>,
    ring_ib: IndexBuffer<'h>,
    /// The next free byte of each ring; zero makes the next vertex append start the ring over.
    vertex_at: Cell<u32>,
    index_at: Cell<u32>,
    /// The quads the particle and text draws write, a small cluster the world transform places.
    quads: Vec<TexturedVertex>,
    /// One interface quad, and the full-target quad of the glow and the composite.
    ui_quad: [TexturedVertex; 4],
    screen_quad: [TexturedVertex; 4],
    /// The streamed texture's two images, one written each frame.
    stream_blocks: Vec<u8>,
    model_decl: VertexDeclaration<'h>,
    second_decl: VertexDeclaration<'h>,
    textured_decl: VertexDeclaration<'h>,
    vs: Vec<VertexShader<'h>>,
    ps: Vec<PixelShader<'h>>,
    occlusion: Query<'h>,
    scene: Vec<Step>,
    glow: [Step; 3],
    ui: Vec<Step>,
    /// The bound programs, `None` for fixed function.
    vs_bound: Cell<Option<u8>>,
    ps_bound: Cell<Option<u8>>,
    /// The texture bound on each of the stages the frame uses, and its address mode.
    ///
    /// These and the cached states below start at the D3D9 defaults, so each
    /// always equals the device's state and a value equal to it is never sent.
    textures_bound: [Cell<Option<u16>>; 4],
    address_bound: [Cell<u32>; 4],
    /// The [`TRACKED`] render states as last set.
    states: [Cell<u32>; 8],
    /// `COLOROP` and `ALPHAOP` of stages 0 and 1 as last set.
    ops_bound: [[Cell<u32>; 2]; 2],
    indices_bound: Cell<*mut c_void>,
    fog_bound: Cell<u32>,
    /// The object the current draws belong to.
    object: Cell<u32>,
    calls: Calls,
    recorder: Recorder,
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
        let mut textures = Vec::new();
        for set in &TEXTURE_SETS {
            assert_eq!(
                usize::from(set.first),
                textures.len(),
                "the texture sets follow one another"
            );
            for at in 0..set.count {
                let index = slot(u32::from(at));
                let edge = set.edges[index % set.edges.len()];
                let format = set.formats[index % set.formats.len()];
                textures.push(filled_texture(
                    h,
                    format,
                    edge,
                    edge * set.aspect,
                    u32::from(set.first + at),
                ));
            }
        }
        assert_eq!(
            textures.len(),
            usize::from(TEXTURE_COUNT),
            "every id has a texture"
        );
        let (mesh_vertices, mesh_indices) = mesh();
        let quad_list: Vec<u16> = (0..u16::try_from(MAX_QUADS).expect("quads fit u16"))
            .flat_map(|quad| {
                let first = quad * 4;
                [first, first + 1, first + 2, first + 2, first + 1, first + 3]
            })
            .collect();
        let render = render_state_defaults();
        let sampler = sampler_state_defaults();
        let icon_vertices: [[f32; 8]; 4] = [
            [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 1.0],
        ];
        let mut stream_blocks = dxt1_bytes(STREAM_EDGE, STREAM_EDGE, 0);
        stream_blocks.extend(dxt1_bytes(STREAM_EDGE, STREAM_EDGE, 1));
        Self {
            h,
            back_buffer: h.back_buffer(0),
            back_depth: h
                .depth_stencil_surface()
                .expect("the device has an auto depth-stencil"),
            scene_depth: h.create_depth_stencil_surface(WIDTH, HEIGHT, D3DFMT_D24S8),
            targets: [
                target(WIDTH, HEIGHT),
                target(GLOW_WIDTH, GLOW_HEIGHT),
                target(GLOW_WIDTH, GLOW_HEIGHT),
            ],
            textures,
            meshes: (0..MESHES)
                .map(|_| Mesh {
                    vertices: static_buffer(h, &mesh_vertices),
                    indices: static_indices(h, &mesh_indices),
                })
                .collect(),
            icon: Mesh {
                vertices: static_buffer(h, &icon_vertices),
                indices: static_indices(h, &[0, 1, 2, 2, 1, 3]),
            },
            quad_list: static_indices(h, &quad_list),
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
            quads: cluster(),
            ui_quad: quad(0.0, 1.0, 0.0, 1.0, 0xE0FF_FFFF),
            screen_quad: quad(-1.0, 1.0, 1.0, -1.0, 0xFFFF_FFFF),
            stream_blocks,
            model_decl: h.create_vertex_declaration(&MODEL_DECL),
            second_decl: h.create_vertex_declaration(&MODEL_DECL),
            textured_decl: h.create_vertex_declaration(&TEXTURED_DECL),
            vs: (0..=SCENE_VS)
                .map(|at| h.create_vertex_shader(&model_vs(ratio(u32::from(at) + 1, 100))))
                .collect(),
            ps: (0..SCENE_PS)
                .map(|at| {
                    let tint = ratio(u32::from(at), 32);
                    h.create_pixel_shader(&material_ps(&Model::Sm2, [tint, 0.0, 0.0, 0.0], false))
                })
                .chain([
                    h.create_pixel_shader(&glow_ps(0.25)),
                    h.create_pixel_shader(&glow_ps(0.3)),
                    h.create_pixel_shader(&composite_ps()),
                ])
                .collect(),
            occlusion: h
                .create_query(D3DQUERYTYPE_OCCLUSION)
                .expect("occlusion queries are supported"),
            scene: scene_steps(),
            glow: glow_steps(),
            ui: ui_steps(),
            vs_bound: Cell::new(None),
            ps_bound: Cell::new(None),
            textures_bound: Default::default(),
            address_bound: [0; 4].map(|_| Cell::new(sampler[slot(D3DSAMP_ADDRESSU)])),
            states: TRACKED.map(|state| Cell::new(render[slot(state)])),
            ops_bound: [0, 1].map(|stage| {
                let defaults = texture_stage_state_defaults(stage);
                OPS.map(|op| Cell::new(defaults[slot(op)]))
            }),
            indices_bound: Cell::new(core::ptr::null_mut()),
            fog_bound: Cell::new(render[slot(D3DRS_FOGCOLOR)]),
            object: Cell::new(0),
            calls: Calls::default(),
            recorder: Recorder {
                on: Cell::new(false),
                passes: RefCell::new(Vec::new()),
            },
        }
    }

    /// One whole frame up to its `Present`, `tick` animating positions and the streamed texture.
    fn render(&self, tick: u32) {
        let h = self.h;
        self.stream(tick);
        self.vertex_at.set(0);
        self.index_at.set(0);
        bump(&self.calls.scene);
        ok(h.begin_scene(), "BeginScene");

        self.target(&self.targets[0].surface, &self.scene_depth);
        self.recorder.pass(WIDTH, HEIGHT);
        self.viewport([0, 0, WIDTH, HEIGHT], (0.0, 1.0));
        self.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER);
        self.viewport([0, 0, WIDTH, HEIGHT], WORLD_DEPTH);
        self.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER);
        self.viewport([0, 0, WIDTH, HEIGHT], SKY_DEPTH);
        self.scene_start();
        for (at, step) in self.scene.iter().enumerate() {
            if at == slot(SKY_LEAD) || at == slot(SKY_DRAWS) {
                self.viewport([0, 0, WIDTH, HEIGHT], WORLD_DEPTH);
            }
            self.step(step, tick);
        }

        self.glow_passes(tick);

        self.target(&self.back_buffer, &self.back_depth);
        self.recorder.pass(WIDTH, HEIGHT);
        self.viewport([0, 0, WIDTH, HEIGHT], (0.0, 1.0));
        self.ui_start();
        for step in &self.ui {
            self.step(step, tick);
        }
        ok(h.end_scene(), "EndScene");
    }

    /// Count the `Present` the caller is about to make.
    fn count_present(&self) {
        bump(&self.calls.present);
    }

    /// Rewrite the streamed texture, which no draw samples, with one of its two images.
    fn stream(&self, tick: u32) {
        let rows = slot(STREAM_EDGE / 4);
        let row_bytes = rows * 8;
        let at = row_bytes * rows * slot(tick % 2);
        bump(&self.calls.lock_rect);
        self.textures[usize::from(STREAM_TEXTURE)]
            .lock_rect(0, 0)
            .write_u8_rect(
                row_bytes,
                rows,
                &self.stream_blocks[at..at + row_bytes * rows],
            );
    }

    /// The render states, stage arguments, samplers and transforms the scene starts from.
    fn scene_start(&self) {
        self.rs(D3DRS_ZENABLE, 1);
        self.rs(D3DRS_ZFUNC, D3DCMP_LESSEQUAL);
        self.rs(D3DRS_ALPHAFUNC, D3DCMP_GREATEREQUAL);
        self.rs(D3DRS_LIGHTING, 0);
        self.rs(D3DRS_FOGENABLE, 1);
        self.rs(D3DRS_FOGVERTEXMODE, D3DFOG_LINEAR);
        self.rs(D3DRS_FOGTABLEMODE, D3DFOG_NONE);
        self.rs(D3DRS_FOGSTART, FOG_START.to_bits());
        self.rs(D3DRS_FOGEND, FOG_END.to_bits());
        self.stage_arguments();
        self.xform(D3DTS_VIEW, &IDENTITY_ROWS);
        self.xform(D3DTS_PROJECTION, &IDENTITY_ROWS);
    }

    /// Stage arguments and filters every pass shares, set at its start.
    fn stage_arguments(&self) {
        self.tss(0, D3DTSS_COLORARG1, D3DTA_TEXTURE);
        self.tss(0, D3DTSS_COLORARG2, D3DTA_DIFFUSE);
        self.tss(0, D3DTSS_ALPHAARG1, D3DTA_TEXTURE);
        self.tss(0, D3DTSS_ALPHAARG2, D3DTA_DIFFUSE);
        self.tss(1, D3DTSS_COLORARG1, D3DTA_TEXTURE);
        self.tss(1, D3DTSS_COLORARG2, D3DTA_CURRENT);
        for stage in 0..2 {
            self.samp(stage, D3DSAMP_MINFILTER, D3DTEXF_LINEAR);
            self.samp(stage, D3DSAMP_MAGFILTER, D3DTEXF_LINEAR);
        }
    }

    /// The three glow passes, binding targets in the game's order, 12 binds in all.
    fn glow_passes(&self, tick: u32) {
        // `None` is the back buffer over its own depth, a target index one over the scene's.
        let binds: [&[Option<usize>]; 3] =
            [&[None, Some(1)], &[None, Some(1), Some(2)], &[Some(1)]];
        for (pass, binds) in binds.iter().enumerate() {
            for bind in *binds {
                match bind {
                    None => self.target(&self.back_buffer, &self.back_depth),
                    Some(at) => self.target(&self.targets[*at].surface, &self.scene_depth),
                }
            }
            self.recorder.pass(GLOW_WIDTH, GLOW_HEIGHT);
            self.viewport([0, 0, GLOW_WIDTH, GLOW_HEIGHT], (0.0, 1.0));
            if pass == 0 {
                self.glow_start();
            }
            self.step(&self.glow[pass], tick);
        }
    }

    /// The states the glow passes and the composite share: no depth test, no fog, four taps.
    ///
    /// The fixed vertex stage feeds all four stages the screen quad's
    /// coordinate, each shifted by a texture transform, so one texture on
    /// four stages gives the pixel program its four taps.
    fn glow_start(&self) {
        self.rs(D3DRS_ZFUNC, D3DCMP_ALWAYS);
        self.rs(D3DRS_FOGENABLE, 0);
        self.xform(D3DTS_WORLD, &IDENTITY_ROWS);
        for stage in 0..4 {
            let step = ratio(stage + 1, GLOW_WIDTH);
            let mut shift = IDENTITY_ROWS;
            shift[8] = if stage % 2 == 0 { -step } else { step };
            shift[9] = if stage < 2 { -step } else { step };
            self.tss(stage, D3DTSS_TEXCOORDINDEX, 0);
            self.tss(stage, D3DTSS_TEXTURETRANSFORMFLAGS, D3DTTFF_COUNT2);
            self.xform(D3DTS_TEXTURE0 + stage, &shift);
        }
    }

    /// The back buffer's states: the stages back to one coordinate each, untransformed.
    fn ui_start(&self) {
        for stage in 0..4 {
            self.tss(stage, D3DTSS_TEXTURETRANSFORMFLAGS, D3DTTFF_DISABLE);
        }
        self.tss(1, D3DTSS_TEXCOORDINDEX, 0);
        self.stage_arguments();
    }

    /// One draw: its programs, states, textures, constants and geometry.
    fn step(&self, step: &Step, tick: u32) {
        let material = step.material;
        self.vs(step.vs);
        self.ps(step.ps);
        let current = step.object.unwrap_or_else(|| self.object.get());
        self.decl(match material.feed {
            Feed::Mesh if current.is_multiple_of(SECOND_DECL_EVERY) => &self.second_decl,
            Feed::Mesh | Feed::Icon => &self.model_decl,
            _ => &self.textured_decl,
        });
        let tracked = [
            material.blend[0],
            material.blend[1],
            material.blend[2],
            material.alpha[0],
            material.alpha[1],
            material.cull,
            material.z_write,
            material.color_mask,
        ];
        for (at, value) in tracked.into_iter().enumerate() {
            self.state(at, value);
        }
        if step.ps.is_none() {
            for (which, value) in material.ops.into_iter().enumerate() {
                // The game sets a fixed-function run's operations whether or not they changed.
                if step.run {
                    self.ops_bound[0][which].set(value);
                    self.tss(0, OPS[which], value);
                } else {
                    self.op(0, which, value);
                }
            }
            if step.run {
                self.ops_bound[1][0].set(D3DTOP_DISABLE);
                self.tss(1, D3DTSS_COLOROP, D3DTOP_DISABLE);
            }
            let stage1 = if step.textures[1].is_some() {
                D3DTOP_MODULATE
            } else {
                D3DTOP_DISABLE
            };
            self.op(1, 0, stage1);
        }
        for (stage, texture) in (0..).zip(step.textures) {
            self.tex(stage, texture);
        }
        if let Some(object) = step.object {
            self.object.set(object);
            self.start_object(step, object);
        } else if step.view_rows {
            self.vs_const(0, &IDENTITY_ROWS);
        }
        let object = self.object.get();
        let world = match material.feed {
            Feed::Screen => IDENTITY_ROWS,
            Feed::Strip | Feed::Minimap | Feed::Icon => ui_world(object, tick),
            _ => {
                let (scale, x, y, z) = place(object, tick);
                ff_world(scale, x, y, z)
            }
        };
        if step.world {
            self.xform(D3DTS_WORLD, &world);
        }
        if step.ps.is_some() {
            let tint = ratio(object / TINT_EVERY % 8, 8);
            self.ps_const(0, &[tint.mul_add(0.2, 0.8), 1.0, 0.9, 1.0]);
        }
        match material.feed {
            Feed::Mesh => self.mesh_draw(step, object),
            Feed::Sparks => self.ring_draw(step.size, step.size * 4, false),
            Feed::Glyphs => {
                // The game binds the text's index buffer with every text draw.
                self.indices_bound.set(core::ptr::null_mut());
                self.ring_draw(step.size, step.reserve, object % 2 == 1);
            }
            Feed::Occlusion => self.occlusion_test(step),
            Feed::Screen | Feed::Strip => self.strip_draw(material),
            Feed::Minimap => {
                self.viewport(MINIMAP_RECT, (0.0, 1.0));
                self.strip_draw(material);
                self.viewport([0, 0, WIDTH, HEIGHT], (0.0, 1.0));
            }
            Feed::Icon => {
                self.viewport(ICON_RECT, (0.0, 1.0));
                self.clear(D3DCLEAR_ZBUFFER);
                self.stream_source(&self.icon.vertices, MODEL_STRIDE);
                self.indices(&self.icon.indices);
                self.draw(D3DPT_TRIANGLELIST, 0, 4, 0, 2);
                self.viewport([0, 0, WIDTH, HEIGHT], (0.0, 1.0));
            }
        }
    }

    /// An object's first draw: its vertex constants, and for a static draw its fog colour.
    fn start_object(&self, step: &Step, object: u32) {
        if step.vs.is_some() {
            let (scale, x, y, z) = place(object, 0);
            self.vs_const(4, &world_rows(scale, x, y, z));
            let mut light = LIGHT_ROWS;
            light[4] = ratio(object % 8, 8).mul_add(0.2, 0.7);
            self.vs_const(LIGHT_ROW, &light);
            let ambient = ratio(object % 16, 16).mul_add(0.1, 0.2);
            self.vs_const(AMBIENT_ROW, &[ambient, ambient, 0.3, 0.0]);
            if object.is_multiple_of(EXTRA_EVERY) {
                self.vs_const(EXTRA_ROW, &[ratio(object % 7, 7), 0.0, 0.0, 0.0]);
            }
        }
        if matches!(step.material.feed, Feed::Mesh) {
            // The game sets the view with each model it draws; it never changes here.
            self.xform(D3DTS_VIEW, &IDENTITY_ROWS);
        }
        if matches!(step.material.feed, Feed::Mesh) && object.is_multiple_of(FOG_EVERY) {
            let color = FOG_COLORS[slot(object / FOG_EVERY) % FOG_COLORS.len()];
            if self.fog_bound.replace(color) != color {
                self.rs(D3DRS_FOGCOLOR, color);
            }
        }
    }

    /// A static draw: the object's mesh, `step.size` of its indices.
    fn mesh_draw(&self, step: &Step, object: u32) {
        let mesh = &self.meshes[slot(object % MESHES)];
        self.stream_source(&mesh.vertices, MODEL_STRIDE);
        if step.object.is_some() {
            // The game binds an object's index buffer whether or not it is bound.
            self.indices_bound.set(core::ptr::null_mut());
        }
        self.indices(&mesh.indices);
        self.draw(D3DPT_TRIANGLELIST, 0, MESH_VERTS, 0, step.size / 3);
    }

    /// `quads` particle or text quads written into `reserve` vertices of the ring.
    ///
    /// A `split` write takes two locks, the second covering the rest of the reservation.
    fn ring_draw(&self, quads: u32, reserve: u32, split: bool) {
        let vertices = &self.quads[..slot(quads * 4)];
        let base = self.append(vertices, reserve, split);
        self.stream_source(&self.ring_vb, STRIDE);
        self.indices(&self.quad_list);
        self.draw(D3DPT_TRIANGLELIST, base, quads * 4, 0, quads * 2);
    }

    /// The sun's test: read last frame's result, then draw inside a new query.
    ///
    /// # Panics
    /// Panics if the query's data is not one `u32`, or if the flushed read is
    /// still pending after [`QUERY_DEADLINE`], so a layer that never answers
    /// fails the benchmark instead of hanging it.
    fn occlusion_test(&self, step: &Step) {
        // The game's perf window counts four query calls a frame where its
        // dump shows three, `GetData`, `Issue` and `Issue`; the fourth is an
        // entry the dump does not log. Any single call fits: `GetDataSize`,
        // `GetType`, `GetDevice`, an `Issue` on an event query, or a `GetData`
        // on a query never issued. Every game window counts exactly 4.0 a
        // frame, which rules out `AddRef` and `Release`, since they come in
        // pairs. An issued event query would be polled, and under
        // `query.eventImmediate` each `GetData(EVENT)` writes a line to the
        // dump; none appears, which rules out the event `Issue`. `GetType`,
        // `GetDevice` and a `GetData` on a query never issued stay possible
        // and the data cannot tell them apart, so `GetDataSize` stands for
        // all of them: like them, it answers without touching the GPU.
        bump(&self.calls.query_data_size);
        assert_eq!(
            self.occlusion.data_size(),
            4,
            "an occlusion query's data size"
        );
        let started = Instant::now();
        let mut polls = 0_u64;
        loop {
            let (hr, _) = self.occlusion.data_u32(D3DGETDATA_FLUSH);
            if hr != S_FALSE {
                ok(hr, "occlusion GetData");
                break;
            }
            polls += 1;
            let pending = &self.calls.pending_polls;
            pending.set(pending.get() + 1);
            assert!(
                started.elapsed() < QUERY_DEADLINE,
                "the occlusion query is still pending after {polls} polls over {QUERY_DEADLINE:?}"
            );
        }
        bump(&self.calls.query_get_data);
        bump(&self.calls.query_issue);
        ok(
            self.occlusion.issue(D3DISSUE_BEGIN),
            "occlusion Issue BEGIN",
        );
        self.viewport([0, 0, WIDTH, HEIGHT], SUN_DEPTH);
        self.ring_draw(step.size, step.size * 4, false);
        bump(&self.calls.query_issue);
        ok(self.occlusion.issue(D3DISSUE_END), "occlusion Issue END");
        self.viewport([0, 0, WIDTH, HEIGHT], SUN_DEPTH);
    }

    /// A strip of four ring vertices and four ring indices: an interface quad or a full target.
    fn strip_draw(&self, material: &Material) {
        let vertices = if matches!(material.feed, Feed::Screen) {
            &self.screen_quad
        } else {
            &self.ui_quad
        };
        let base = self.append(vertices, 4, false);
        let start = self.append_indices(&[0, 1, 2, 3]);
        self.stream_source(&self.ring_vb, STRIDE);
        self.indices(&self.ring_ib);
        self.draw(D3DPT_TRIANGLESTRIP, base, 4, start, 2);
    }

    /// Write `vertices` into a `reserve`-vertex reservation of the ring; its base vertex.
    ///
    /// The first reservation of a frame, and one that does not fit in what
    /// is left, starts the ring over with `D3DLOCK_DISCARD`; every other one
    /// takes `D3DLOCK_NOOVERWRITE`.
    fn append(&self, vertices: &[TexturedVertex], reserve: u32, split: bool) -> i32 {
        let bytes = reserve * STRIDE;
        let mut start = self.vertex_at.get();
        if start + bytes > RING_VERTEX_BYTES {
            start = 0;
        }
        let mut flags = D3DLOCK_NOOVERWRITE;
        if start == 0 {
            flags = D3DLOCK_DISCARD;
            bump(&self.calls.vb_discard);
        }
        if split {
            let first = vertices.len() / 2;
            let first_bytes = u32::try_from(first).expect("a lock fits u32") * STRIDE;
            bump(&self.calls.vb_lock);
            self.ring_vb
                .lock(start, first_bytes, flags)
                .write(&vertices[..first]);
            bump(&self.calls.vb_lock);
            self.ring_vb
                .lock(
                    start + first_bytes,
                    bytes - first_bytes,
                    D3DLOCK_NOOVERWRITE,
                )
                .write(&vertices[first..]);
        } else {
            bump(&self.calls.vb_lock);
            self.ring_vb.lock(start, bytes, flags).write(vertices);
        }
        self.vertex_at.set(start + bytes);
        i32::try_from(start / STRIDE).expect("base vertex fits i32")
    }

    /// Append `data` to the index ring with `D3DLOCK_NOOVERWRITE`; its start index.
    ///
    /// The ring starts over at every frame without `D3DLOCK_DISCARD`: a
    /// frame writes the same indices at the same places as the one before,
    /// so what the GPU may still read is never changed, and the ring is
    /// never renamed, as the game's is about once in 90 frames.
    fn append_indices(&self, data: &[u16]) -> u32 {
        let bytes = u32::try_from(size_of_val(data)).expect("an append fits u32");
        let start = self.index_at.get();
        assert!(
            start + bytes <= RING_INDEX_BYTES,
            "the index ring holds a frame"
        );
        bump(&self.calls.ib_lock);
        self.ring_ib
            .lock(start, bytes, D3DLOCK_NOOVERWRITE)
            .write(data);
        self.index_at.set(start + bytes);
        start / 2
    }

    /// `SetRenderTarget(0, surface)`, then `depth`.
    fn target(&self, surface: &Surface<'_>, depth: &Surface<'_>) {
        bump(&self.calls.set_render_target);
        ok(self.h.set_render_target(0, surface), "SetRenderTarget");
        bump(&self.calls.set_depth_stencil);
        ok(
            self.h.set_depth_stencil_surface(depth),
            "SetDepthStencilSurface",
        );
    }

    fn viewport(&self, [x, y, width, height]: [u32; 4], (min_z, max_z): (f32, f32)) {
        bump(&self.calls.set_viewport);
        ok(
            self.h.set_viewport(&D3DVIEWPORT9 {
                x,
                y,
                width,
                height,
                min_z,
                max_z,
            }),
            "SetViewport",
        );
    }

    fn clear(&self, flags: u32) {
        bump(&self.calls.clear);
        ok(self.h.clear(flags, 0, 1.0, 0), "Clear");
    }

    /// Bind `texture` on `stage` if another is bound, and its address mode if that differs.
    fn tex(&self, stage: u32, texture: Option<u16>) {
        let bound = &self.textures_bound[slot(stage)];
        if bound.get() == texture {
            return;
        }
        bound.set(texture);
        bump(&self.calls.set_texture);
        let Some(id) = texture else {
            ok(self.h.clear_texture(stage), "SetTexture(null)");
            return;
        };
        ok(self.h.set_texture(stage, self.texture(id)), "SetTexture");
        let address = if id % CLAMP_EVERY == 0 {
            D3DTADDRESS_CLAMP
        } else {
            D3DTADDRESS_WRAP
        };
        if self.address_bound[slot(stage)].replace(address) != address {
            self.samp(stage, D3DSAMP_ADDRESSU, address);
            self.samp(stage, D3DSAMP_ADDRESSV, address);
        }
    }

    /// The texture of `id`: a pool's, or a render target's.
    fn texture(&self, id: u16) -> &Texture<'h> {
        self.textures
            .get(usize::from(id))
            .unwrap_or_else(|| &self.targets[usize::from(id - TEXTURE_COUNT)].texture)
    }

    /// Set the [`TRACKED`] render state at `at` if it holds another value.
    ///
    /// Turning the alpha test on sends its function again too, as the game does.
    fn state(&self, at: usize, value: u32) {
        if self.states[at].replace(value) != value {
            self.rs(TRACKED[at], value);
            if TRACKED[at] == D3DRS_ALPHATESTENABLE && value != 0 {
                self.rs(D3DRS_ALPHAFUNC, D3DCMP_GREATEREQUAL);
            }
        }
    }

    /// Set stage `stage`'s `COLOROP` (`which` 0) or `ALPHAOP` (1) if it holds another value.
    fn op(&self, stage: u32, which: usize, value: u32) {
        if self.ops_bound[slot(stage)][which].replace(value) != value {
            self.tss(stage, OPS[which], value);
        }
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

    /// Bind vertex program `program` (`None` for fixed function) if another is bound.
    fn vs(&self, program: Option<u8>) {
        if self.vs_bound.replace(program) == program {
            return;
        }
        bump(&self.calls.set_vertex_shader);
        let hr = program.map_or_else(
            || self.h.clear_vertex_shader(),
            |at| self.h.set_vertex_shader(&self.vs[usize::from(at)]),
        );
        ok(hr, "SetVertexShader");
    }

    /// Bind pixel program `program` (`None` for the fixed stages) if another is bound.
    fn ps(&self, program: Option<u8>) {
        if self.ps_bound.replace(program) == program {
            return;
        }
        bump(&self.calls.set_pixel_shader);
        let hr = program.map_or_else(
            || self.h.clear_pixel_shader(),
            |at| self.h.set_pixel_shader(&self.ps[usize::from(at)]),
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

    fn decl(&self, decl: &VertexDeclaration<'_>) {
        bump(&self.calls.set_vertex_declaration);
        ok(self.h.set_vertex_declaration(decl), "SetVertexDeclaration");
    }

    fn stream_source(&self, vb: &VertexBuffer<'_>, stride: u32) {
        bump(&self.calls.set_stream_source);
        ok(
            self.h.set_stream_source(0, vb, 0, stride),
            "SetStreamSource",
        );
    }

    /// `SetIndices(ib)` if another index buffer is bound.
    fn indices(&self, ib: &IndexBuffer<'_>) {
        if self.indices_bound.replace(ib.as_ptr()) == ib.as_ptr() {
            return;
        }
        bump(&self.calls.set_indices);
        ok(self.h.set_indices(ib), "SetIndices");
    }

    /// An indexed draw from the bound stream and indices, recorded while the recorder is on.
    fn draw(&self, primitive: u32, base: i32, vertices: u32, start: u32, primitives: u32) {
        bump(&self.calls.draw);
        self.recorder.draw(|| Look {
            vs: self.vs_bound.get(),
            ps: self.ps_bound.get(),
            textures: self.textures_bound.each_ref().map(Cell::get),
            blend: [
                self.states[0].get(),
                self.states[1].get(),
                self.states[2].get(),
            ],
            alpha: [self.states[3].get(), self.states[4].get()],
            cull: self.states[5].get(),
            z_write: self.states[6].get(),
            color_mask: self.states[7].get(),
        });
        ok(
            self.h
                .draw_indexed_primitive(primitive, base, 0, vertices, start, primitives),
            "DrawIndexedPrimitive",
        );
    }
}

/// The static mesh: a [`GRID`]x[`GRID`] grid of normals facing the viewer, and its index list.
fn mesh() -> (Vec<[f32; 8]>, Vec<u16>) {
    let edge = u16::try_from(GRID).expect("mesh grid fits u16");
    let step = 1.0 / f32::from(edge);
    let mut vertices = Vec::new();
    for row in 0..=edge {
        for col in 0..=edge {
            let (u, v) = (f32::from(col) * step, f32::from(row) * step);
            vertices.push([u, v, 0.0, 0.0, 0.0, -1.0, u, v]);
        }
    }
    let mut indices = Vec::new();
    for row in 0..edge {
        for col in 0..edge {
            let top = row * (edge + 1) + col;
            let bottom = top + edge + 1;
            indices.extend_from_slice(&[top, bottom, top + 1, top + 1, bottom, bottom + 1]);
        }
    }
    (vertices, indices)
}

/// [`MAX_QUADS`] small quads in a cluster around the origin, for particles and text.
fn cluster() -> Vec<TexturedVertex> {
    (0..MAX_QUADS)
        .flat_map(|at| {
            let x = ratio(at % 8, 8) - 0.5;
            let y = ratio(at / 8, 8) - 0.5;
            quad(x, x + 0.1, y + 0.1, y, 0xC0FF_FFFF)
        })
        .collect()
}

/// A quad from `left` to `right` and `top` to `bottom`, as a strip, texture coordinates top-down.
fn quad(left: f32, right: f32, top: f32, bottom: f32, color: u32) -> [TexturedVertex; 4] {
    let corner = |x: f32, y: f32, u: f32, v: f32| TexturedVertex {
        x,
        y,
        z: 0.5,
        color,
        u,
        v,
    };
    [
        corner(left, top, 0.0, 0.0),
        corner(right, top, 1.0, 0.0),
        corner(left, bottom, 0.0, 1.0),
        corner(right, bottom, 1.0, 1.0),
    ]
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

/// A managed `width`x`height` texture of `format`, one level, filled from `seed`.
fn filled_texture(h: &Harness, format: u32, width: u32, height: u32, seed: u32) -> Texture<'_> {
    let texture = h.create_texture(width, height, 1, 0, format, D3DPOOL_MANAGED);
    let (row_bytes, rows) = match format {
        D3DFMT_DXT1 => (width / 4 * 8, height / 4),
        D3DFMT_DXT3 | D3DFMT_DXT5 => (width / 4 * 16, height / 4),
        D3DFMT_A8R8G8B8 => (width * 4, height),
        _ => (width * 2, height),
    };
    let bytes = if format == D3DFMT_DXT1 {
        dxt1_bytes(width, height, seed)
    } else {
        noise(row_bytes * rows, seed)
    };
    texture
        .lock_rect(0, 0)
        .write_u8_rect(slot(row_bytes), slot(rows), &bytes);
    texture
}

/// DXT1 blocks for a `width`x`height` level: two colours from `seed`, no transparent texels.
fn dxt1_bytes(width: u32, height: u32, seed: u32) -> Vec<u8> {
    (0..width / 4 * (height / 4))
        .flat_map(|block| {
            let red = u16::try_from((seed * 5 + block) % 32).expect("a 5-bit channel fits u16");
            let light = red << 11 | 0x07E0 | 0x0010;
            let dark = light >> 2 & 0x39E7;
            let [l0, l1] = light.to_le_bytes();
            let [d0, d1] = dark.to_le_bytes();
            [l0, l1, d0, d1, 0xE4, 0xE4, 0x1B, 0x1B]
        })
        .collect()
}

/// `count` bytes of a fixed pseudo-random sequence started from `seed`.
fn noise(count: u32, seed: u32) -> Vec<u8> {
    let mut state = seed.wrapping_mul(0x9E37_79B9) | 1;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state.to_le_bytes()[3]
        })
        .collect()
}

/// Scale and clip-space position of object `object` in frame `tick`.
fn place(object: u32, tick: u32) -> (f32, f32, f32, f32) {
    let drift = ratio(tick % 64, 64) * 0.01;
    (
        ratio(object % 5, 5).mul_add(0.04, 0.05),
        ratio((object * 37) % 100, 100).mul_add(1.8, -0.95) + drift,
        ratio((object * 61) % 100, 100).mul_add(1.8, -0.95),
        ratio((object * 13) % 97, 97).mul_add(0.7, 0.1),
    )
}

/// The world matrix of interface quad `object`, the quads in a grid over the screen.
fn ui_world(object: u32, tick: u32) -> [f32; 16] {
    let drift = ratio(tick % 32, 32) * 0.002;
    let x = ratio(object % 24, 24).mul_add(1.95, -0.98) + drift;
    let y = ratio(object / 24 % 10, 10).mul_add(-1.9, 0.79);
    let mut world = ff_world(0.075, x, y, 0.0);
    world[5] = 0.17;
    world
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

/// The `vs_2_0` model program, whose output the fixed stages or a pixel program take.
///
/// `oPos` is `c0..c3` applied to `c4..c7` applied to the position (see
/// [`transform`]); `oD0` is the light `c90..c91` on the normal plus the
/// ambient `c92`; `oT0` is the coordinate shifted by `def c95`, which is
/// `variant` in `x`, so distinct variants are distinct programs.
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

/// A glow `ps_2_0` program: one tap from each of `s0..s3`, summed and weighted.
///
/// `def c1` is `weight`, so distinct weights are distinct programs.
#[rustfmt::skip]
fn glow_ps(weight: f32) -> Vec<u32> {
    let mut tokens = vec![
        0xFFFF_0200,                           // ps_2_0
        0x0200_001F, 0x8000_0000, 0xB00F_0000, // dcl t0
        0x0200_001F, 0x8000_0000, 0xB00F_0001, // dcl t1
        0x0200_001F, 0x8000_0000, 0xB00F_0002, // dcl t2
        0x0200_001F, 0x8000_0000, 0xB00F_0003, // dcl t3
        0x0200_001F, 0x9000_0000, 0xA00F_0800, // dcl_2d s0
        0x0200_001F, 0x9000_0000, 0xA00F_0801, // dcl_2d s1
        0x0200_001F, 0x9000_0000, 0xA00F_0802, // dcl_2d s2
        0x0200_001F, 0x9000_0000, 0xA00F_0803, // dcl_2d s3
    ];
    tokens.extend_from_slice(&def(0xA00F_0001, [weight; 4])); // def c1
    tokens.extend_from_slice(&[
        0x0300_0042, 0x800F_0000, 0xB0E4_0000, 0xA0E4_0800, // texld r0, t0, s0
        0x0300_0042, 0x800F_0001, 0xB0E4_0001, 0xA0E4_0801, // texld r1, t1, s1
        0x0300_0042, 0x800F_0002, 0xB0E4_0002, 0xA0E4_0802, // texld r2, t2, s2
        0x0300_0042, 0x800F_0003, 0xB0E4_0003, 0xA0E4_0803, // texld r3, t3, s3
        0x0300_0002, 0x800F_0000, 0x80E4_0000, 0x80E4_0001, // add r0, r0, r1
        0x0300_0002, 0x800F_0002, 0x80E4_0002, 0x80E4_0003, // add r2, r2, r3
        0x0300_0002, 0x800F_0000, 0x80E4_0000, 0x80E4_0002, // add r0, r0, r2
        0x0300_0005, 0x800F_0000, 0x80E4_0000, 0xA0E4_0001, // mul r0, r0, c1
        0x0200_0001, 0x800F_0800, 0x80E4_0000,              // mov oC0, r0
        0x0000_FFFF,
    ]);
    tokens
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

/// The sky's draws, summed from [`SKY_RUNS`].
const fn sky_draws() -> u32 {
    let mut draws = 0;
    let mut at = 0;
    while at < SKY_RUNS.len() {
        draws += SKY_RUNS[at].3;
        at += 1;
    }
    draws
}

/// The opaque world's draws, summed from [`STRETCHES`].
const fn stretch_draws() -> u32 {
    let mut draws = 0;
    let mut at = 0;
    while at < STRETCHES.len() {
        draws += STRETCHES[at].1;
        at += 1;
    }
    draws
}

/// The back buffer's draws, summed from [`UI_RUNS`].
const fn ui_draws() -> u32 {
    let mut draws = 0;
    let mut at = 0;
    while at < UI_RUNS.len() {
        draws += UI_RUNS[at].1;
        at += 1;
    }
    draws
}

/// `value` as an index into one of the frame's lists.
fn slot(value: u32) -> usize {
    usize::try_from(value).expect("an index fits usize")
}

fn bump(counter: &Cell<u32>) {
    counter.set(counter.get() + 1);
}
