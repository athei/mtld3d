//! Frame-time spikes from pixel shaders that first appear mid-game, with the shader cache off.
//!
//! Every measured frame draws a fixed base workload and then creates and
//! draws never-seen programmable pixel shaders: `K` on the back buffer, one
//! quad each, and, in the variant that asks for it, one more into an
//! offscreen target that is cleared once and never again, so its pass loads
//! what the previous frame left and the draw must keep its content. Each
//! shader's bytecode is unique (its `def c7` carries a running count and a
//! per-run salt), so neither the layer nor Metal's own compiler cache can
//! have seen it, on this run or an earlier one. The interface is created
//! with `shaderCache.enable=false` on top of whatever the suite-wide
//! configuration carries, which is how `make bench BENCH_CONFIG=...` tries
//! other options against the same frames.
//!
//! The measured frames are the `MEASURED_FRAMES` that introduce shaders.
//! Base frames follow until [`IDLE_TAIL`] has passed without a new shader
//! and the whole span has run [`MIN_SPAN`], so a `PERF=1` build writes at
//! least one whole summary window, and any account of compiles that waits
//! for a quiet spell has had one; the report copies the Compilation rows of
//! every window in the span, which is where a layer that skips draws whose
//! pipeline is not ready says how many it skipped and how long installs
//! took. Nothing the measured frames draw is there to be checked, so the
//! check changes nothing they measure: after the settle frames, verification
//! frames clear a probe target and draw every new shader into a 4x4 cell of
//! its own, and a readback says whether each cell shows the shader's
//! colour, until every one does or [`VERIFY_FRAMES`] have gone by. A layer
//! that skips a draw for a pipeline still building has had the settle
//! frames to build it, so a cell left black is a shader that was never
//! drawn. `K` and the offscreen shader are fixed per test; `make bench
//! FILTER=<test name>` picks one.

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use mtld3d_tests::{
    Harness, HarnessConfig, IndexBuffer, PixelShader, Surface, Texture, VertexBuffer,
    VertexDeclaration, VertexShader,
};
use mtld3d_types::{
    D3DCLEAR_TARGET, D3DCLEAR_ZBUFFER, D3DFMT_D24S8, D3DFMT_INDEX16, D3DFMT_X8R8G8B8,
    D3DLOCK_READONLY, D3DPOOL_DEFAULT, D3DPOOL_MANAGED, D3DPOOL_SYSTEMMEM,
    D3DPRESENT_INTERVAL_IMMEDIATE, D3DPT_TRIANGLELIST, D3DRS_ZENABLE, D3DUSAGE_RENDERTARGET,
    D3DUSAGE_WRITEONLY,
};

use crate::bench::{
    FrameClock, IDENTITY_ROWS, LayerLog, Model, STRIDE, TEXTURED_DECL, grid, material_ps,
    material_vs, ok, pattern_texture, ratio, world_rows, write_report,
};

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
/// Edge of the probe target and of the uncleared offscreen target.
const TARGET_EDGE: u32 = 512;
/// Edge of one shader's probe cell, in pixels.
const CELL_EDGE: u32 = 4;
const CELLS_PER_ROW: u32 = TARGET_EDGE / CELL_EDGE;
/// Draws of the base program each frame, so a frame without a new shader is not empty.
const BASE_DRAWS: u32 = 50;
const WARM_UP_FRAMES: u32 = 30;
const MEASURED_FRAMES: u32 = 200;
/// Time without a new shader before the run ends.
const IDLE_TAIL: Duration = Duration::from_secs(2);
/// The least time from the first measured frame to the end of the run.
const MIN_SPAN: Duration = Duration::from_secs(12);
/// The most verification frames before a black cell counts as never drawn.
const VERIFY_FRAMES: u32 = 100;

/// One new pixel shader per frame on the back buffer.
#[test]
#[ignore = "benchmark, run by `make bench`"]
fn one_new_shader_per_frame() {
    stutter("shader_stutter_k1", 1, false);
}

/// Two new pixel shaders per frame on the back buffer and one into an uncleared offscreen target.
#[test]
#[ignore = "benchmark, run by `make bench`"]
fn two_new_shaders_per_frame_and_one_offscreen() {
    stutter("shader_stutter_k2_offscreen", 2, true);
}

/// Run the measured frames, `per_frame` new shaders each, and write the report `name`.
fn stutter(name: &str, per_frame: u32, offscreen: bool) {
    let h = Harness::create(&HarnessConfig {
        width: WIDTH,
        height: HEIGHT,
        depth_format: Some(D3DFMT_D24S8),
        presentation_interval: D3DPRESENT_INTERVAL_IMMEDIATE,
        config_entries: "shaderCache.enable=false",
        ..HarnessConfig::default()
    });
    let since = SystemTime::now();
    let mut bench = Stutter::new(&h);
    for _ in 0..WARM_UP_FRAMES {
        assert!(h.pump(), "WM_QUIT during warm-up");
        bench.base_frame();
        ok(h.end_scene(), "EndScene");
        ok(h.present(), "Present");
    }

    let log = LayerLog::find(since);
    let from = log.mark();
    let started = Instant::now();
    let mut clock = FrameClock::start(usize::try_from(MEASURED_FRAMES).expect("fits usize"));
    for frame in 0..MEASURED_FRAMES {
        assert!(h.pump(), "WM_QUIT during the measured frames");
        bench.base_frame();
        bench.introduce(frame, per_frame, offscreen);
        ok(h.end_scene(), "EndScene");
        clock.present(&h);
    }
    let introduced = Instant::now();
    let mut settle_frames = 0_u32;
    while introduced.elapsed() < IDLE_TAIL || started.elapsed() < MIN_SPAN {
        assert!(h.pump(), "WM_QUIT while the frames settle");
        bench.base_frame();
        ok(h.end_scene(), "EndScene");
        ok(h.present(), "Present");
        settle_frames += 1;
    }
    let to = log.mark();
    let span = started.elapsed();
    let verified = bench.verify();

    let stats = clock.stats();
    let spikes = clock.over(stats.p50 * 2);
    let mut compiles = String::new();
    for rows in log.compilation_rows(from, to) {
        compiles.push_str("perf: window in the span, its compiles\n");
        compiles.push_str(&rows);
    }
    let body = format!(
        "shape: back buffer {WIDTH}x{HEIGHT}, {BASE_DRAWS} base draws per frame, \
         shaderCache.enable=false\n\
         new pixel shaders per frame: {per_frame} on the back buffer{offscreen}\n\
         warm-up: {WARM_UP_FRAMES} frames without new shaders\n\
         measured: {frames} frames in {elapsed:.2?}, {shaders} new shaders; then \
         {settle_frames} settle frames, {span:.2?} in all\n\
         frame time (Present to Present): {row}\n\
         API work (Present return to Present call): {work}\n\
         frames over 2x the median ({limit:.3} ms): {spikes}\n\
         {verified}{perf}{compiles}",
        offscreen = if offscreen {
            " + 1 into an offscreen target never cleared after the first frame"
        } else {
            ""
        },
        frames = stats.frames,
        shaders = bench.shaders.len(),
        elapsed = clock.elapsed(),
        row = stats.row(),
        work = clock.work_stats().row(),
        limit = (stats.p50 * 2).as_secs_f64() * 1e3,
        perf = log.perf_rows(from, to).section(),
    );
    write_report(name, &log, &body);
}

/// The benchmark's device objects and the shaders it has introduced.
struct Stutter<'h> {
    h: &'h Harness,
    back_buffer: Surface<'h>,
    /// The texture behind [`Self::offscreen`], held so the target outlives the frames.
    _offscreen_texture: Texture<'h>,
    offscreen: Surface<'h>,
    /// The texture behind [`Self::probe`], held so the target outlives the frames.
    _probe_texture: Texture<'h>,
    probe: Surface<'h>,
    texture: Texture<'h>,
    vb: VertexBuffer<'h>,
    ib: IndexBuffer<'h>,
    decl: VertexDeclaration<'h>,
    vs: VertexShader<'h>,
    base_ps: PixelShader<'h>,
    vertex_count: u32,
    triangles: u32,
    /// Kept alive to the end, as a game keeps what it has loaded.
    shaders: Vec<PixelShader<'h>>,
    salt: u16,
}

impl<'h> Stutter<'h> {
    fn new(h: &'h Harness) -> Self {
        let target = || {
            h.create_texture(
                TARGET_EDGE,
                TARGET_EDGE,
                1,
                D3DUSAGE_RENDERTARGET,
                D3DFMT_X8R8G8B8,
                D3DPOOL_DEFAULT,
            )
        };
        let (offscreen_texture, probe_texture) = (target(), target());
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
        let bench = Self {
            h,
            back_buffer: h.back_buffer(0),
            offscreen: offscreen_texture.surface_level(0),
            _offscreen_texture: offscreen_texture,
            probe: probe_texture.surface_level(0),
            _probe_texture: probe_texture,
            texture: pattern_texture(h, 0xFF40_8020),
            vb,
            ib,
            decl: h.create_vertex_declaration(&TEXTURED_DECL),
            vs: h.create_vertex_shader(&material_vs(&Model::Sm2, 0.0)),
            base_ps: h.create_pixel_shader(&material_ps(&Model::Sm2, [0.0; 4], false)),
            vertex_count,
            triangles,
            shaders: Vec::new(),
            salt: salt(),
        };
        ok(h.set_vertex_declaration(&bench.decl), "declaration");
        ok(h.set_stream_source(0, &bench.vb, 0, STRIDE), "stream");
        ok(h.set_indices(&bench.ib), "indices");
        ok(h.set_texture(0, &bench.texture), "texture");
        ok(h.set_vertex_shader(&bench.vs), "VS");
        ok(
            h.set_vertex_shader_constant_f(0, &IDENTITY_ROWS),
            "view-projection",
        );
        ok(h.set_pixel_shader_constant_f(0, &[1.0; 4]), "tint");
        ok(h.set_render_state(D3DRS_ZENABLE, 0), "depth off");
        ok(h.set_render_target(0, &bench.offscreen), "offscreen target");
        ok(
            h.clear(D3DCLEAR_TARGET, 0xFF00_0000, 1.0, 0),
            "offscreen clear, once",
        );
        ok(h.set_render_target(0, &bench.back_buffer), "back buffer");
        bench
    }

    /// Open a frame: clear, and the base draws on the back buffer.
    fn base_frame(&self) {
        let h = self.h;
        ok(h.begin_scene(), "BeginScene");
        ok(
            h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, 0xFF20_3040, 1.0, 0),
            "clear",
        );
        ok(h.set_pixel_shader(&self.base_ps), "base PS");
        for at in 0..BASE_DRAWS {
            self.draw(0.1, spot(at));
        }
    }

    /// Create this frame's new shaders and draw each once on the back buffer, or offscreen.
    fn introduce(&mut self, frame: u32, per_frame: u32, offscreen: bool) {
        let h = self.h;
        for at in 0..per_frame {
            let ps = self.new_shader();
            ok(h.set_pixel_shader(&ps), "new PS");
            self.draw(0.1, spot(BASE_DRAWS + at + frame % 7));
            self.shaders.push(ps);
        }
        if offscreen {
            let ps = self.new_shader();
            ok(h.set_render_target(0, &self.offscreen), "offscreen target");
            ok(h.set_pixel_shader(&ps), "offscreen PS");
            self.draw(0.1, spot(frame % 40));
            ok(h.set_render_target(0, &self.back_buffer), "back buffer");
            self.shaders.push(ps);
        }
    }

    /// Draw every new shader into its cleared probe cell until each cell shows it, and report.
    fn verify(&self) -> String {
        let h = self.h;
        let mut attempts = 0;
        let missing = loop {
            attempts += 1;
            assert!(h.pump(), "WM_QUIT while verifying");
            ok(h.begin_scene(), "BeginScene");
            ok(h.set_render_target(0, &self.probe), "probe target");
            ok(h.clear(D3DCLEAR_TARGET, 0xFF00_0000, 1.0, 0), "probe clear");
            ok(h.set_pixel_shader_constant_f(0, &[0.0; 4]), "probe tint");
            for (at, ps) in self.shaders.iter().enumerate() {
                let cell = u32::try_from(at).expect("shader count fits u32");
                ok(h.set_pixel_shader(ps), "probe PS");
                let edge = ratio(CELL_EDGE * 2, TARGET_EDGE);
                let (col, row) = (cell % CELLS_PER_ROW, cell / CELLS_PER_ROW);
                self.draw(
                    edge,
                    (
                        ratio(col, CELLS_PER_ROW).mul_add(2.0, -1.0),
                        ratio(row + 1, CELLS_PER_ROW).mul_add(-2.0, 1.0),
                    ),
                );
            }
            ok(h.set_pixel_shader_constant_f(0, &[1.0; 4]), "tint");
            ok(h.set_render_target(0, &self.back_buffer), "back buffer");
            ok(h.end_scene(), "EndScene");
            ok(h.present(), "Present");
            let missing = self.missing_cells();
            if missing == 0 || attempts == VERIFY_FRAMES {
                break missing;
            }
        };
        format!(
            "drawn (readback of a probe target cleared every verification frame): {drawn} of \
             {total} new shaders show their colour after {attempts} verification frame(s), \
             {missing} never drawn\n",
            total = self.shaders.len(),
            drawn = self.shaders.len() - missing,
        )
    }

    /// How many new shaders' probe cells are still the clear colour.
    fn missing_cells(&self) -> usize {
        let h = self.h;
        let sysmem = h.create_offscreen_plain_surface(
            TARGET_EDGE,
            TARGET_EDGE,
            D3DFMT_X8R8G8B8,
            D3DPOOL_SYSTEMMEM,
        );
        ok(
            h.get_render_target_data_hr(&self.probe, &sysmem),
            "probe readback",
        );
        let locked = sysmem.lock_rect(D3DLOCK_READONLY);
        let pitch = usize::try_from(locked.pitch()).expect("positive pitch") / 4;
        let edge = usize::try_from(TARGET_EDGE).expect("target edge fits usize");
        let pixels = locked.as_u32(pitch * edge);
        let centre = |index: u32| {
            usize::try_from(index * CELL_EDGE + CELL_EDGE / 2).expect("pixel fits usize")
        };
        (0..self.shaders.len())
            .filter(|&at| {
                let cell = u32::try_from(at).expect("shader count fits u32");
                let (x, y) = (centre(cell % CELLS_PER_ROW), centre(cell / CELLS_PER_ROW));
                (pixels[y * pitch + x] >> 16) & 0xFF != 0xFF
            })
            .count()
    }

    /// The next never-seen pixel shader.
    ///
    /// Its `def c7` is full red, the running count and the run's salt: with
    /// `c0` at zero the shader writes red, and the other two lanes make it
    /// unique without reaching an 8-bit step.
    fn new_shader(&self) -> PixelShader<'h> {
        let count = u16::try_from(self.shaders.len() + 1).expect("new shader count fits u16");
        let tint = [
            1.0,
            f32::from(count) * 1.0e-6,
            f32::from(self.salt) * 1.0e-8,
            0.0,
        ];
        self.h
            .create_pixel_shader(&material_ps(&Model::Sm2, tint, false))
    }

    /// One mesh draw scaled by `scale` at clip-space `(x, y)`.
    fn draw(&self, scale: f32, (x, y): (f32, f32)) {
        let h = self.h;
        ok(
            h.set_vertex_shader_constant_f(4, &world_rows(scale, x, y, 0.5)),
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

/// Where the `at`-th small quad of a frame goes, in clip space.
fn spot(at: u32) -> (f32, f32) {
    (
        ratio(at % 20, 20).mul_add(1.8, -0.95),
        ratio(at / 20 % 20, 20).mul_add(1.8, -0.95),
    )
}

/// A per-run value that keeps this run's shaders unlike any earlier run's.
fn salt() -> u16 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.subsec_nanos());
    u16::try_from(nanos % 65_521).expect("a remainder below 65521 fits u16")
}
