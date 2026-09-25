//! A synthetic frame shaped like World of Warcraft 3.3.5a's busy frame, timed over many frames.
//!
//! One frame is about 1400 draws in nine render passes: four shadow-cascade
//! passes of depth-only draws, each into a 2048x2048 depth target of its own
//! (INTZ, sampled by the scene's receivers, or D24S8 where INTZ is not
//! offered), the scene on the back buffer (900 draws in 36 material runs,
//! two particle batches among them), three one-draw glow passes ping-ponging
//! between two offscreen targets, and a UI pass of 300 small alpha-blended
//! quads, then `Present`. The full back-buffer `StretchRect` between the
//! scene and the glow is a blit, not a pass. Each cascade clears and fills
//! its own target, as the game's do, so the layer keeps the four passes
//! apart; cascades sharing one target through viewport quadrants would be
//! joined into one depth-only pass.
//!
//! Per draw the bind mix follows the game's: about one `SetTexture`, one
//! `SetVertexShaderConstantF` of 4 to 16 rows, a fifth of a
//! `SetPixelShaderConstantF`, and 0.4 render-state changes, among them blend
//! toggles that leave stale blend factors set while blending is off. Vertex
//! and index buffers come from a handful of static buffers plus one dynamic
//! vertex buffer locked with `D3DLOCK_DISCARD` four times a frame, and the
//! scene cycles three vertex declarations that describe identical
//! attributes. Every program (32 programmable scene pairs across `ps_2_0`
//! and `ps_3_0`, four fixed-function scene stages, two caster pairs, the
//! glow pair and two UI stages) is created up front and first drawn in the
//! warm-up, so the measured frames compile nothing.

use std::time::{Duration, Instant};

use mtld3d_tests::{
    Harness, HarnessConfig, IndexBuffer, PixelShader, Surface, Texture, TexturedVertex,
    VertexBuffer, VertexDeclaration, VertexShader,
};
use mtld3d_types::{
    D3D_OK, D3DBLEND_INVSRCALPHA, D3DBLEND_ONE, D3DBLEND_SRCALPHA, D3DCLEAR_STENCIL,
    D3DCLEAR_TARGET, D3DCLEAR_ZBUFFER, D3DCMP_ALWAYS, D3DCMP_LESSEQUAL, D3DCULL_CCW, D3DCULL_NONE,
    D3DDECL_END, D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_POSITION, D3DFMT_A8R8G8B8, D3DFMT_D24S8,
    D3DFMT_INDEX16, D3DFMT_INTZ, D3DFMT_X8R8G8B8, D3DFVF_DIFFUSE, D3DFVF_TEX1, D3DFVF_XYZ,
    D3DLOCK_DISCARD, D3DPOOL_DEFAULT, D3DPOOL_MANAGED, D3DPRESENT_INTERVAL_IMMEDIATE,
    D3DPT_TRIANGLELIST, D3DRS_ALPHABLENDENABLE, D3DRS_ALPHAREF, D3DRS_ALPHATESTENABLE,
    D3DRS_BLENDFACTOR, D3DRS_COLORWRITEENABLE, D3DRS_CULLMODE, D3DRS_DEPTHBIAS, D3DRS_DESTBLEND,
    D3DRS_LIGHTING, D3DRS_SLOPESCALEDEPTHBIAS, D3DRS_SRCBLEND, D3DRS_ZENABLE, D3DRS_ZFUNC,
    D3DRS_ZWRITEENABLE, D3DRTYPE_TEXTURE, D3DSAMP_MAGFILTER, D3DSAMP_MINFILTER, D3DTA_DIFFUSE,
    D3DTA_TEXTURE, D3DTEXF_LINEAR, D3DTEXF_NONE, D3DTOP_ADD, D3DTOP_MODULATE, D3DTOP_MODULATE2X,
    D3DTOP_SELECTARG1, D3DTS_PROJECTION, D3DTS_VIEW, D3DTS_WORLD, D3DTSS_COLORARG1,
    D3DTSS_COLORARG2, D3DTSS_COLOROP, D3DUSAGE_DEPTHSTENCIL, D3DUSAGE_DYNAMIC,
    D3DUSAGE_RENDERTARGET, D3DUSAGE_WRITEONLY, D3DVERTEXELEMENT9,
};

use crate::bench::{
    FrameClock, IDENTITY_ROWS, LayerLog, Model, STRIDE, TEXTURED_DECL, def, element, grid,
    material_ps, material_vs, ok, pattern_texture, ratio, transform, world_rows, write_report,
};

/// The back buffer, about the size of a windowed game.
const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
/// Edge of each cascade's shadow depth target.
const SHADOW_EDGE: u32 = 2048;
const CASCADES: u32 = 4;
const CASTERS_PER_CASCADE: u32 = 50;
/// Scene material runs, one per material, each [`DRAWS_PER_RUN`] draws long.
const SCENE_RUNS: u32 = 36;
const DRAWS_PER_RUN: u32 = 25;
/// The runs a particle batch from the dynamic buffer goes in front of.
const PARTICLE_RUNS: [u32; 2] = [12, 24];
const PARTICLE_QUADS: u32 = 64;
/// UI quads per frame, written into the dynamic buffer in batches of [`UI_BATCH`].
const UI_QUADS: u32 = 300;
const UI_BATCH: u32 = 150;
const SM2_MATERIALS: u32 = 24;
const SM3_MATERIALS: u32 = 8;
/// The colour operations of the fixed-function scene materials, one material each.
const FF_OPS: [u32; 4] = [
    D3DTOP_MODULATE,
    D3DTOP_SELECTARG1,
    D3DTOP_ADD,
    D3DTOP_MODULATE2X,
];
/// Grid sizes of the static meshes, one vertex buffer each.
const MESH_GRIDS: [u16; 4] = [4, 6, 8, 10];
const SCENE_TEXTURES: u32 = 16;
const UI_TEXTURES: u32 = 8;
const WARM_UP_FRAMES: u32 = 60;
/// The measured phase is at least this many frames and at least [`MIN_MEASURED`] long.
///
/// The duration floor is what puts one whole five-second window of a
/// `PERF=1` build's summary inside the measured frames: the first window
/// written after they start began before them.
const MEASURED_FRAMES: usize = 600;
const MIN_MEASURED: Duration = Duration::from_secs(12);
const FVF: u32 = D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1;

/// One busy frame, repeatedly: warm up, then time the frames.
#[test]
#[ignore = "benchmark, run by `make bench`"]
fn wow_335a_busy_frame() {
    let h = Harness::create(&HarnessConfig {
        width: WIDTH,
        height: HEIGHT,
        depth_format: Some(D3DFMT_D24S8),
        presentation_interval: D3DPRESENT_INTERVAL_IMMEDIATE,
        ..HarnessConfig::default()
    });
    let log = LayerLog::find();
    let created = log.mark();
    let started = Instant::now();
    let frame = Frame::new(&h);
    for tick in 0..WARM_UP_FRAMES {
        assert!(h.pump(), "WM_QUIT during warm-up");
        frame.render(tick);
        ok(h.present(), "Present");
    }
    let warm_up = started.elapsed();

    let from = log.mark();
    let mut clock = FrameClock::start(MEASURED_FRAMES * 4);
    let mut tick = WARM_UP_FRAMES;
    while clock.frames() < MEASURED_FRAMES || clock.elapsed() < MIN_MEASURED {
        assert!(h.pump(), "WM_QUIT during the measured frames");
        frame.render(tick);
        clock.present(&h);
        tick += 1;
    }
    let to = log.mark();

    let stats = clock.stats();
    let shadow = if frame.shadows[0].texture.is_some() {
        "INTZ, sampled by the receivers"
    } else {
        "D24S8, INTZ not offered"
    };
    let body = format!(
        "shape: back buffer {WIDTH}x{HEIGHT} X8R8G8B8 + D24S8; {CASCADES} shadow depth \
         targets {SHADOW_EDGE}x{SHADOW_EDGE} {shadow}\n\
         per frame: {draws} draws (casters {casters} in {CASCADES} passes, scene {scene}, \
         glow 3 in 3 passes, UI {ui}), one StretchRect, {locks} DISCARD locks\n\
         programs: {SM2_MATERIALS} ps_2_0 + {SM3_MATERIALS} ps_3_0 scene pairs, {ff} \
         fixed-function scene stages, 2 caster pairs, 1 glow pair, 2 UI stages\n\
         warm-up: {WARM_UP_FRAMES} frames in {warm_up:.2?}\n\
         measured: {frames} frames in {elapsed:.2?} (at least {MEASURED_FRAMES} frames \
         and {MIN_MEASURED:?})\n\
         frame time (Present to Present): {row}\n\
         API work (Present return to Present call): {work}\n{perf}{warm_up_compiles}",
        draws = DRAWS_PER_FRAME,
        casters = CASCADES * CASTERS_PER_CASCADE,
        scene = SCENE_RUNS * DRAWS_PER_RUN + 2,
        ui = UI_QUADS + 1,
        locks = 2 + UI_QUADS / UI_BATCH,
        ff = FF_OPS.len(),
        frames = stats.frames,
        elapsed = clock.elapsed(),
        row = stats.row(),
        work = clock.work_stats().row(),
        perf = log.perf_rows(from, to).section(),
        warm_up_compiles =
            log.compilation_rows(created, to)
                .first()
                .map_or_else(String::new, |rows| format!(
                    "perf: first window after device creation, its warm-up compiles\n{rows}"
                ),),
    );
    write_report("frame_shape", &log, &body);
}

/// A shadow depth target: an INTZ texture's level when `intz`, else a D24S8 surface.
fn shadow(h: &Harness, intz: bool) -> Shadow<'_> {
    if intz {
        let texture = h.create_texture(
            SHADOW_EDGE,
            SHADOW_EDGE,
            1,
            D3DUSAGE_DEPTHSTENCIL,
            D3DFMT_INTZ,
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
            surface: h.create_depth_stencil_surface(SHADOW_EDGE, SHADOW_EDGE, D3DFMT_D24S8),
        }
    }
}

/// Draws per frame: casters, scene with its particle batches, glow, the glow composite and UI.
const DRAWS_PER_FRAME: u32 =
    CASCADES * CASTERS_PER_CASCADE + SCENE_RUNS * DRAWS_PER_RUN + 2 + 3 + 1 + UI_QUADS;

/// A scene material: a programmable pair, or a fixed-function colour operation.
enum Material<'h> {
    Programmable {
        vs: VertexShader<'h>,
        ps: PixelShader<'h>,
        tint: [f32; 4],
        /// The cascade whose shadow map a receiver samples on stage 1.
        shadow_map: Option<u32>,
    },
    Fixed(u32),
}

/// A static mesh: its own vertex buffer and its range of the shared index buffer.
struct Mesh<'h> {
    vb: VertexBuffer<'h>,
    vertices: u32,
    start_index: u32,
    triangles: u32,
}

/// A cascade's shadow depth target, and the texture behind it when it can be sampled.
struct Shadow<'h> {
    texture: Option<Texture<'h>>,
    surface: Surface<'h>,
}

/// An offscreen colour target that later passes sample.
struct Target<'h> {
    texture: Texture<'h>,
    surface: Surface<'h>,
}

/// Every resource the frame uses, created once.
struct Frame<'h> {
    h: &'h Harness,
    back_buffer: Surface<'h>,
    scene_depth: Surface<'h>,
    /// One depth target per cascade.
    shadows: Vec<Shadow<'h>>,
    /// The colour target the depth-only cascades render beside, which their draws never write.
    placeholder: Surface<'h>,
    scene_copy: Target<'h>,
    glow: [Target<'h>; 2],
    textures: Vec<Texture<'h>>,
    ui_textures: Vec<Texture<'h>>,
    meshes: Vec<Mesh<'h>>,
    mesh_ib: IndexBuffer<'h>,
    /// Quad indices `0 1 2 2 1 3` repeated, for the particles, the UI and the screen quad.
    quad_ib: IndexBuffer<'h>,
    screen_quad: VertexBuffer<'h>,
    dynamic: VertexBuffer<'h>,
    /// Three separately created declarations of the one [`TexturedVertex`] layout.
    decls: [VertexDeclaration<'h>; 3],
    position_decl: VertexDeclaration<'h>,
    materials: Vec<Material<'h>>,
    /// The opaque caster, then the alpha-tested one that samples its texture.
    casters: [(VertexShader<'h>, PixelShader<'h>); 2],
    glow_program: (VertexShader<'h>, PixelShader<'h>),
}

impl<'h> Frame<'h> {
    fn new(h: &'h Harness) -> Self {
        let back_buffer = h.back_buffer(0);
        let scene_depth = h
            .depth_stencil_surface()
            .expect("the device has an auto depth-stencil");
        let intz = h.check_device_format(
            D3DFMT_X8R8G8B8,
            D3DUSAGE_DEPTHSTENCIL,
            D3DRTYPE_TEXTURE,
            D3DFMT_INTZ,
        ) == D3D_OK;
        let shadows = (0..CASCADES).map(|_| shadow(h, intz)).collect();
        let placeholder = h.create_render_target(SHADOW_EDGE, SHADOW_EDGE, D3DFMT_A8R8G8B8);
        let target = |width, height, format| {
            let texture = h.create_texture(
                width,
                height,
                1,
                D3DUSAGE_RENDERTARGET,
                format,
                D3DPOOL_DEFAULT,
            );
            let surface = texture.surface_level(0);
            Target { texture, surface }
        };
        let scene_copy = target(WIDTH, HEIGHT, D3DFMT_X8R8G8B8);
        let glow = [0, 1].map(|_| target(WIDTH / 2, HEIGHT / 2, D3DFMT_A8R8G8B8));
        let textures = (0..SCENE_TEXTURES)
            .map(|at| pattern_texture(h, 0xFF10_2030 + at * 0x0009_0B07))
            .collect();
        let ui_textures = (0..UI_TEXTURES)
            .map(|at| pattern_texture(h, 0x80C0_A080 + at * 0x0003_0507))
            .collect();
        let (meshes, mesh_ib) = meshes(h);
        let decls = [0, 1, 2].map(|_| h.create_vertex_declaration(&TEXTURED_DECL));
        let position_decl = h.create_vertex_declaration(&POSITION_DECL);
        Self {
            h,
            back_buffer,
            scene_depth,
            shadows,
            placeholder,
            scene_copy,
            glow,
            textures,
            ui_textures,
            meshes,
            mesh_ib,
            quad_ib: quad_indices(h),
            screen_quad: screen_quad(h),
            dynamic: h.create_vertex_buffer(
                UI_BATCH * 4 * STRIDE,
                D3DUSAGE_DYNAMIC | D3DUSAGE_WRITEONLY,
                FVF,
                D3DPOOL_DEFAULT,
            ),
            decls,
            position_decl,
            materials: materials(h),
            casters: [
                (
                    h.create_vertex_shader(&caster_vs(false)),
                    h.create_pixel_shader(&caster_ps()),
                ),
                (
                    h.create_vertex_shader(&caster_vs(true)),
                    h.create_pixel_shader(&material_ps(&Model::Sm2, [0.0; 4], false)),
                ),
            ],
            glow_program: (
                h.create_vertex_shader(&glow_vs()),
                h.create_pixel_shader(&material_ps(&Model::Sm2, [0.0, 0.0, 0.0, 0.5], false)),
            ),
        }
    }

    /// One whole frame up to its `Present`, `tick` animating the per-draw constants.
    fn render(&self, tick: u32) {
        let h = self.h;
        ok(h.begin_scene(), "BeginScene");
        self.cascades(tick);
        self.scene(tick);
        ok(
            h.stretch_rect(&self.back_buffer, &self.scene_copy.surface, D3DTEXF_NONE),
            "back-buffer copy",
        );
        self.glow();
        self.ui(tick);
        ok(h.end_scene(), "EndScene");
    }

    /// Four depth-only caster passes, one shadow target each.
    fn cascades(&self, tick: u32) {
        let h = self.h;
        // The receivers left a shadow map on stage 1; it is a depth target now.
        ok(h.clear_texture(1), "unbind the shadow map");
        ok(h.set_render_target(0, &self.placeholder), "caster target");
        rs(h, D3DRS_ALPHABLENDENABLE, 0);
        rs(h, D3DRS_ZENABLE, 1);
        rs(h, D3DRS_ZWRITEENABLE, 1);
        rs(h, D3DRS_ZFUNC, D3DCMP_LESSEQUAL);
        rs(h, D3DRS_CULLMODE, D3DCULL_NONE);
        rs(h, D3DRS_COLORWRITEENABLE, 0);
        rs(h, D3DRS_ALPHAREF, 0x80);
        ok(h.set_indices(&self.mesh_ib), "SetIndices");
        ok(
            h.set_vertex_shader_constant_f(0, &IDENTITY_ROWS),
            "caster view-projection",
        );
        for (cascade, shadow) in (0..CASCADES).zip(&self.shadows) {
            ok(h.set_depth_stencil_surface(&shadow.surface), "shadow depth");
            ok(h.clear(D3DCLEAR_ZBUFFER, 0, 1.0, 0), "cascade depth clear");
            rs(
                h,
                D3DRS_DEPTHBIAS,
                (1.0e-4 * ratio(cascade + 1, 1)).to_bits(),
            );
            rs(
                h,
                D3DRS_SLOPESCALEDEPTHBIAS,
                (ratio(cascade + 1, 2)).to_bits(),
            );
            for draw in 0..CASTERS_PER_CASCADE {
                self.caster(tick, cascade, draw);
            }
            rs(h, D3DRS_ALPHATESTENABLE, 0);
        }
        rs(h, D3DRS_DEPTHBIAS, 0);
        rs(h, D3DRS_SLOPESCALEDEPTHBIAS, 0);
        rs(h, D3DRS_COLORWRITEENABLE, 0xF);
    }

    /// One caster draw: opaque for the first half of a cascade, alpha-tested for the second.
    fn caster(&self, tick: u32, cascade: u32, draw: u32) {
        let h = self.h;
        let tested = draw >= CASTERS_PER_CASCADE / 2;
        if draw == 0 || draw == CASTERS_PER_CASCADE / 2 {
            let (vs, ps) = &self.casters[usize::from(tested)];
            ok(h.set_vertex_shader(vs), "caster VS");
            ok(h.set_pixel_shader(ps), "caster PS");
            ok(
                h.set_pixel_shader_constant_f(0, &[1.0; 4]),
                "caster PS constant",
            );
            if tested {
                rs(h, D3DRS_ALPHATESTENABLE, 1);
            }
        }
        if tested {
            ok(
                h.set_vertex_declaration(&self.decls[0]),
                "caster declaration",
            );
            ok(
                h.set_texture(0, &self.textures[slot(draw % SCENE_TEXTURES)]),
                "caster texture",
            );
        } else {
            ok(
                h.set_vertex_declaration(&self.position_decl),
                "caster declaration",
            );
        }
        let at = cascade * CASTERS_PER_CASCADE + draw;
        let mesh = &self.meshes[slot(at % 4)];
        ok(h.set_stream_source(0, &mesh.vb, 0, STRIDE), "caster stream");
        let (scale, x, y, z) = placement(at, tick);
        ok(
            h.set_vertex_shader_constant_f(4, &world_rows(scale, x, y, z)),
            "caster world",
        );
        draw_mesh(h, mesh);
    }

    /// The scene: 36 material runs of 25 draws on the back buffer.
    fn scene(&self, tick: u32) {
        let h = self.h;
        ok(h.set_render_target(0, &self.back_buffer), "scene target");
        ok(
            h.set_depth_stencil_surface(&self.scene_depth),
            "scene depth",
        );
        ok(
            h.clear(
                D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER | D3DCLEAR_STENCIL,
                0xFF20_3040,
                1.0,
                0,
            ),
            "scene clear",
        );
        ok(
            h.set_vertex_shader_constant_f(0, &IDENTITY_ROWS),
            "view-projection",
        );
        ok(h.set_transform(D3DTS_VIEW, &IDENTITY_ROWS), "FF view");
        ok(
            h.set_transform(D3DTS_PROJECTION, &IDENTITY_ROWS),
            "FF projection",
        );
        rs(h, D3DRS_LIGHTING, 0);
        rs(h, D3DRS_SRCBLEND, D3DBLEND_SRCALPHA);
        rs(h, D3DRS_DESTBLEND, D3DBLEND_INVSRCALPHA);
        for stage in 0..2 {
            ok(
                h.set_sampler_state(stage, D3DSAMP_MINFILTER, D3DTEXF_LINEAR),
                "min filter",
            );
            ok(
                h.set_sampler_state(stage, D3DSAMP_MAGFILTER, D3DTEXF_LINEAR),
                "mag filter",
            );
        }
        let mut blend = false;
        let mut cull = false;
        for draw in 0..SCENE_RUNS * DRAWS_PER_RUN {
            if draw % DRAWS_PER_RUN == 0 {
                let run = draw / DRAWS_PER_RUN;
                if PARTICLE_RUNS.contains(&run) {
                    self.particles(tick, run, blend);
                }
                ok(h.set_indices(&self.mesh_ib), "SetIndices");
                // Coprime with the run count, so every material comes up once, shuffled.
                self.bind_material(&self.materials[slot((run * 7) % SCENE_RUNS)]);
            }
            let material = &self.materials[slot((draw / DRAWS_PER_RUN * 7) % SCENE_RUNS)];
            self.scene_draw(tick, draw, material);
            match draw % 5 {
                1 => {
                    blend = !blend;
                    rs(h, D3DRS_ALPHABLENDENABLE, u32::from(blend));
                }
                // With blending off the blend factors are stale: changing
                // them must not change what is drawn.
                3 if !blend && (draw / 5) % 2 == 0 => {
                    let src = [D3DBLEND_SRCALPHA, D3DBLEND_ONE][slot(draw / 10 % 2)];
                    rs(h, D3DRS_SRCBLEND, src);
                }
                3 if !blend => rs(h, D3DRS_BLENDFACTOR, 0xFF00_0000 | (draw * 0x0001_0203)),
                3 => {
                    cull = !cull;
                    rs(
                        h,
                        D3DRS_CULLMODE,
                        if cull { D3DCULL_CCW } else { D3DCULL_NONE },
                    );
                }
                _ => {}
            }
        }
        rs(h, D3DRS_ALPHABLENDENABLE, 0);
        rs(h, D3DRS_CULLMODE, D3DCULL_NONE);
    }

    /// Bind a material's programs, or its fixed-function stage.
    fn bind_material(&self, material: &Material<'_>) {
        let h = self.h;
        match material {
            Material::Programmable {
                vs,
                ps,
                tint,
                shadow_map,
            } => {
                ok(h.set_vertex_shader(vs), "material VS");
                ok(h.set_pixel_shader(ps), "material PS");
                ok(h.set_pixel_shader_constant_f(0, tint), "material tint");
                if let Some(cascade) = shadow_map {
                    let cascade = &self.shadows[slot(*cascade)];
                    let shadow = cascade.texture.as_ref().unwrap_or(&self.textures[0]);
                    ok(h.set_texture(1, shadow), "shadow map");
                }
            }
            Material::Fixed(op) => {
                ok(h.clear_vertex_shader(), "fixed-function VS");
                ok(h.clear_pixel_shader(), "fixed-function PS");
                ok(h.set_fvf(FVF), "SetFVF");
                ok(
                    h.set_texture_stage_state(0, D3DTSS_COLOROP, *op),
                    "colour op",
                );
                ok(
                    h.set_texture_stage_state(0, D3DTSS_COLORARG1, D3DTA_TEXTURE),
                    "arg 1",
                );
                ok(
                    h.set_texture_stage_state(0, D3DTSS_COLORARG2, D3DTA_DIFFUSE),
                    "arg 2",
                );
            }
        }
    }

    /// One scene draw: texture, stream, world constants (or transform), draw.
    fn scene_draw(&self, tick: u32, draw: u32, material: &Material<'_>) {
        let h = self.h;
        ok(
            h.set_texture(0, &self.textures[slot((draw * 7) % SCENE_TEXTURES)]),
            "scene texture",
        );
        let mesh = &self.meshes[slot(draw % 4)];
        ok(h.set_stream_source(0, &mesh.vb, 0, STRIDE), "scene stream");
        let (scale, x, y, z) = placement(draw, tick);
        if matches!(material, Material::Programmable { .. }) {
            ok(
                h.set_vertex_declaration(&self.decls[slot(draw % 3)]),
                "scene declaration",
            );
            // The world rows, then up to three rows of per-draw extras
            // (bones, texture transforms) the programs do not read.
            let mut rows = [0.25_f32; 64];
            rows[..16].copy_from_slice(&world_rows(scale, x, y, z));
            let count = 16 * (1 + slot(draw % 4));
            ok(
                h.set_vertex_shader_constant_f(4, &rows[..count]),
                "scene world",
            );
        } else {
            ok(
                h.set_transform(D3DTS_WORLD, &ff_world(scale, x, y, z)),
                "FF world",
            );
        }
        if draw.is_multiple_of(5) {
            let shade = ratio(draw % 50, 50);
            ok(
                h.set_pixel_shader_constant_f(0, &[shade, 1.0 - shade, 0.5, 1.0]),
                "per-draw tint",
            );
        }
        draw_mesh(h, mesh);
    }

    /// A particle batch: one `DISCARD` lock of the dynamic buffer and one additive draw.
    fn particles(&self, tick: u32, run: u32, blend: bool) {
        let h = self.h;
        self.dynamic.lock(0, 0, D3DLOCK_DISCARD).write(&quads(
            PARTICLE_QUADS,
            run + tick % 16,
            0.03,
        ));
        ok(h.clear_vertex_shader(), "particle VS");
        ok(h.clear_pixel_shader(), "particle PS");
        ok(h.set_fvf(FVF), "particle FVF");
        ok(
            h.set_texture_stage_state(0, D3DTSS_COLOROP, D3DTOP_MODULATE),
            "particle op",
        );
        ok(
            h.set_transform(D3DTS_WORLD, &IDENTITY_ROWS),
            "particle world",
        );
        ok(h.set_texture(0, &self.textures[3]), "particle texture");
        rs(h, D3DRS_ALPHABLENDENABLE, 1);
        rs(h, D3DRS_SRCBLEND, D3DBLEND_ONE);
        rs(h, D3DRS_DESTBLEND, D3DBLEND_ONE);
        rs(h, D3DRS_ZWRITEENABLE, 0);
        ok(
            h.set_stream_source(0, &self.dynamic, 0, STRIDE),
            "particle stream",
        );
        ok(h.set_indices(&self.quad_ib), "particle indices");
        ok(
            h.draw_indexed_primitive(
                D3DPT_TRIANGLELIST,
                0,
                0,
                PARTICLE_QUADS * 4,
                0,
                PARTICLE_QUADS * 2,
            ),
            "particle draw",
        );
        rs(h, D3DRS_ZWRITEENABLE, 1);
        rs(h, D3DRS_SRCBLEND, D3DBLEND_SRCALPHA);
        rs(h, D3DRS_DESTBLEND, D3DBLEND_INVSRCALPHA);
        rs(h, D3DRS_ALPHABLENDENABLE, u32::from(blend));
    }

    /// Three one-draw glow passes, each sampling the target the one before wrote.
    fn glow(&self) {
        let h = self.h;
        let (vs, ps) = &self.glow_program;
        ok(h.set_vertex_shader(vs), "glow VS");
        ok(h.set_pixel_shader(ps), "glow PS");
        ok(h.set_vertex_declaration(&self.decls[0]), "glow declaration");
        ok(
            h.set_stream_source(0, &self.screen_quad, 0, STRIDE),
            "glow stream",
        );
        ok(h.set_indices(&self.quad_ib), "glow indices");
        rs(h, D3DRS_ZENABLE, 0);
        let passes = [
            (&self.glow[0], &self.scene_copy),
            (&self.glow[1], &self.glow[0]),
            (&self.glow[0], &self.glow[1]),
        ];
        for (target, source) in passes {
            ok(h.set_render_target(0, &target.surface), "glow target");
            ok(h.set_texture(0, &source.texture), "glow source");
            ok(h.set_pixel_shader_constant_f(0, &[0.5; 4]), "glow weight");
            ok(
                h.draw_indexed_primitive(D3DPT_TRIANGLELIST, 0, 0, 4, 0, 2),
                "glow draw",
            );
        }
    }

    /// The UI pass: the glow composite, then 300 alpha-blended quads from the dynamic buffer.
    fn ui(&self, tick: u32) {
        let h = self.h;
        ok(h.set_render_target(0, &self.back_buffer), "UI target");
        rs(h, D3DRS_ZENABLE, 1);
        rs(h, D3DRS_ZWRITEENABLE, 0);
        rs(h, D3DRS_ZFUNC, D3DCMP_ALWAYS);
        rs(h, D3DRS_ALPHABLENDENABLE, 1);
        rs(h, D3DRS_SRCBLEND, D3DBLEND_ONE);
        rs(h, D3DRS_DESTBLEND, D3DBLEND_ONE);
        ok(
            h.set_texture(0, &self.glow[0].texture),
            "glow composite source",
        );
        ok(
            h.draw_indexed_primitive(D3DPT_TRIANGLELIST, 0, 0, 4, 0, 2),
            "glow composite",
        );

        ok(h.clear_vertex_shader(), "UI VS");
        ok(h.clear_pixel_shader(), "UI PS");
        ok(h.set_fvf(FVF), "UI FVF");
        ok(h.set_transform(D3DTS_WORLD, &IDENTITY_ROWS), "UI world");
        ok(
            h.set_stream_source(0, &self.dynamic, 0, STRIDE),
            "UI stream",
        );
        let mut additive = false;
        for batch in 0..UI_QUADS / UI_BATCH {
            self.dynamic.lock(0, 0, D3DLOCK_DISCARD).write(&quads(
                UI_BATCH,
                batch * 7 + tick % 4,
                0.04,
            ));
            for quad in 0..UI_BATCH {
                let at = batch * UI_BATCH + quad;
                if at.is_multiple_of(10) {
                    additive = !additive;
                    let (src, dst) = if additive {
                        (D3DBLEND_SRCALPHA, D3DBLEND_ONE)
                    } else {
                        (D3DBLEND_SRCALPHA, D3DBLEND_INVSRCALPHA)
                    };
                    rs(h, D3DRS_SRCBLEND, src);
                    rs(h, D3DRS_DESTBLEND, dst);
                }
                if at.is_multiple_of(50) {
                    let op = [D3DTOP_MODULATE, D3DTOP_SELECTARG1][slot(at / 50 % 2)];
                    ok(
                        h.set_texture_stage_state(0, D3DTSS_COLOROP, op),
                        "UI colour op",
                    );
                }
                ok(
                    h.set_texture(0, &self.ui_textures[slot((at * 3) % UI_TEXTURES)]),
                    "UI texture",
                );
                let base = i32::try_from(quad * 4).expect("UI base vertex fits i32");
                ok(
                    h.draw_indexed_primitive(D3DPT_TRIANGLELIST, base, 0, 4, 0, 2),
                    "UI draw",
                );
            }
        }
        rs(h, D3DRS_ALPHABLENDENABLE, 0);
        rs(h, D3DRS_ZFUNC, D3DCMP_LESSEQUAL);
        rs(h, D3DRS_ZWRITEENABLE, 1);
    }
}

/// The opaque casters' declaration: the position alone, over the same stride.
const POSITION_DECL: [D3DVERTEXELEMENT9; 2] = [
    element(0, D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_POSITION),
    D3DDECL_END,
];

/// The 36 scene materials: `ps_2_0` pairs (every other one a receiver), `ps_3_0` pairs, then FF.
fn materials(h: &Harness) -> Vec<Material<'_>> {
    let programmable = |model: &Model, at: u32, shadow_map: Option<u32>| {
        let shade = ratio(at, SM2_MATERIALS + SM3_MATERIALS);
        let tint = [shade * 0.1, 0.05, shade.mul_add(-0.1, 0.1), 0.0];
        Material::Programmable {
            vs: h.create_vertex_shader(&material_vs(model, shade * 1.0e-3)),
            ps: h.create_pixel_shader(&material_ps(model, tint, shadow_map.is_some())),
            tint: [1.0 - shade, 0.8, shade, 1.0],
            shadow_map,
        }
    };
    let mut materials: Vec<Material<'_>> = (0..SM2_MATERIALS)
        .map(|at| programmable(&Model::Sm2, at, (at % 2 == 1).then_some(at / 2 % CASCADES)))
        .collect();
    materials
        .extend((0..SM3_MATERIALS).map(|at| programmable(&Model::Sm3, SM2_MATERIALS + at, None)));
    materials.extend(FF_OPS.iter().map(|&op| Material::Fixed(op)));
    materials
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

/// Quad indices for [`UI_BATCH`] quads of four vertices each.
fn quad_indices(h: &Harness) -> IndexBuffer<'_> {
    let indices: Vec<u16> = (0..u16::try_from(UI_BATCH).expect("UI batch fits u16"))
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
    let vb = h.create_vertex_buffer(4 * STRIDE, D3DUSAGE_WRITEONLY, FVF, D3DPOOL_MANAGED);
    vb.lock(0, 0, 0).write(&vertices);
    vb
}

/// `count` small squares of edge `size` in clip space, scattered by `seed`.
fn quads(count: u32, seed: u32, size: f32) -> Vec<TexturedVertex> {
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

/// A caster vertex program: the scene transform, nudged by `def c95` so the two are distinct.
///
/// The alpha-tested caster also passes its texture coordinate on.
#[rustfmt::skip]
fn caster_vs(tested: bool) -> Vec<u32> {
    let mut tokens = vec![
        0xFFFE_0200,                           // vs_2_0
        0x0200_001F, 0x8000_0000, 0x900F_0000, // dcl_position v0
    ];
    if tested {
        tokens.extend_from_slice(&[0x0200_001F, 0x8000_0005, 0x900F_0001]); // dcl_texcoord0 v1
    }
    tokens.extend_from_slice(&def(0xA00F_005F, [0.0, 0.0, if tested { 1.0e-4 } else { 0.0 }, 0.0]));
    // The world rows land in r0; nudge it before view-projection.
    let mut body = transform(0xC000_0000).to_vec(); // oPos
    body.splice(16..16, [0x0300_0002, 0x800F_0000, 0x80E4_0000, 0xA0E4_005F]); // add r0, r0, c95
    tokens.extend_from_slice(&body);
    if tested {
        tokens.extend_from_slice(&[0x0200_0001, 0xE00F_0000, 0x90E4_0001]); // mov oT0, v1
    }
    tokens.push(0x0000_FFFF);
    tokens
}

/// `ps_2_0 { def c7, 1, 1, 1, 1; mov oC0, c7 }`, the opaque caster's colour, which is masked off.
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

/// `vs_2_0 { mov oPos, v0; mov oT0, v1 }`, the screen quad of the glow passes.
#[rustfmt::skip]
fn glow_vs() -> Vec<u32> {
    vec![
        0xFFFE_0200,                           // vs_2_0
        0x0200_001F, 0x8000_0000, 0x900F_0000, // dcl_position v0
        0x0200_001F, 0x8000_0005, 0x900F_0001, // dcl_texcoord0 v1
        0x0200_0001, 0xC00F_0000, 0x90E4_0000, // mov oPos, v0
        0x0200_0001, 0xE00F_0000, 0x90E4_0001, // mov oT0, v1
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

/// `value` as an index into one of the frame's resource lists.
fn slot(value: u32) -> usize {
    usize::try_from(value).expect("a resource index fits usize")
}

fn rs(h: &Harness, state: u32, value: u32) {
    ok(h.set_render_state(state, value), "SetRenderState");
}
