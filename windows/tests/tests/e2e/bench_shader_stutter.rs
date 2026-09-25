//! Frame-time spikes from pixel shaders that first appear mid-game, with the shader cache off.
//!
//! Every measured frame draws a fixed base workload and then creates and
//! draws never-seen programmable pixel shaders: `K` on the back buffer, one
//! quad each, and, in the variant that asks for it, one more into an
//! offscreen target that is not cleared that frame, so its pass loads what
//! the previous frame left. Each shader's bytecode is unique (its `def c7`
//! carries a running count and a per-run salt), so neither the layer nor
//! Metal's own compiler cache can have seen it, on this run or an earlier
//! one. The interface is created with `shaderCache.enable=false` on top of
//! whatever the suite-wide configuration carries, which is how `make bench
//! BENCH_CONFIG=...` tries other options against the same frames.
//!
//! A layer may skip a draw whose pipeline is not ready yet, so the report
//! also says when each new shader's draws started to land. Every new shader
//! keeps drawing into a 4x4 cell of its own in the offscreen probe target,
//! which is cleared once and blended additively one red step per draw that
//! lands, for [`LAND_WINDOW`] after it appears: in its first frame and then
//! at most once every [`PROBE_INTERVAL`], so the count stays below 255 at
//! any frame rate. The new shaders' back-buffer draws blend the same way,
//! so the probe draws use the same pipeline. One readback at the end counts
//! the draws that landed; the ones before them were skipped, assuming a
//! shader is skipped only until it first lands.
//!
//! The measured frames are the `MEASURED_FRAMES` that introduce shaders.
//! Base frames follow, still drawing the probes, until every probe window
//! has closed, [`IDLE_TAIL`] has passed without a new shader, and the whole
//! span has run [`MIN_SPAN`], so a `PERF=1` build writes at least one whole
//! summary window and any summary of compiles that waits for a quiet spell
//! has had one. `K` and the offscreen shader are fixed per test; `make bench
//! FILTER=<test name>` picks one.

use core::fmt::Write as _;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use mtld3d_tests::{
    Harness, HarnessConfig, IndexBuffer, PixelShader, Surface, Texture, VertexBuffer,
    VertexDeclaration, VertexShader,
};
use mtld3d_types::{
    D3DBLEND_ONE, D3DCLEAR_TARGET, D3DCLEAR_ZBUFFER, D3DFMT_D24S8, D3DFMT_INDEX16, D3DFMT_X8R8G8B8,
    D3DLOCK_READONLY, D3DPOOL_DEFAULT, D3DPOOL_MANAGED, D3DPOOL_SYSTEMMEM,
    D3DPRESENT_INTERVAL_IMMEDIATE, D3DPT_TRIANGLELIST, D3DRS_ALPHABLENDENABLE, D3DRS_DESTBLEND,
    D3DRS_SRCBLEND, D3DRS_ZENABLE, D3DUSAGE_RENDERTARGET, D3DUSAGE_WRITEONLY,
};

use crate::bench::{
    FrameClock, IDENTITY_ROWS, LayerLog, Model, STRIDE, TEXTURED_DECL, grid, material_ps,
    material_vs, ok, pattern_texture, ratio, world_rows, write_report,
};

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
/// Edge of the probe target, which is also the uncleared offscreen target.
const PROBE_EDGE: u32 = 512;
/// Edge of one shader's probe cell, in pixels.
const CELL_EDGE: u32 = 4;
const CELLS_PER_ROW: u32 = PROBE_EDGE / CELL_EDGE;
/// Draws of the base program each frame, so a frame without a new shader is not empty.
const BASE_DRAWS: u32 = 50;
const WARM_UP_FRAMES: u32 = 30;
const MEASURED_FRAMES: u32 = 200;
/// How long a new shader keeps drawing into its probe cell.
const LAND_WINDOW: Duration = Duration::from_secs(1);
/// The least time between two probe draws of one shader after its first frame.
const PROBE_INTERVAL: Duration = Duration::from_millis(5);
/// The most probe draws of one shader, below the 255 steps a cell's red channel holds.
const MAX_PROBE_DRAWS: usize = 250;
/// Time without a new shader before the run ends.
const IDLE_TAIL: Duration = Duration::from_secs(2);
/// The least time from the first measured frame to the end of the run.
const MIN_SPAN: Duration = Duration::from_secs(12);

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
    let mut bench = Stutter::new(&h);
    for _ in 0..WARM_UP_FRAMES {
        assert!(h.pump(), "WM_QUIT during warm-up");
        bench.begin_frame();
        bench.end_frame();
        ok(h.present(), "Present");
    }

    let log = LayerLog::find();
    let from = log.mark();
    let started = Instant::now();
    let mut clock = FrameClock::start(usize::try_from(MEASURED_FRAMES).expect("fits usize"));
    for _ in 0..MEASURED_FRAMES {
        assert!(h.pump(), "WM_QUIT during the measured frames");
        bench.begin_frame();
        bench.introduce(per_frame, offscreen);
        bench.end_frame();
        clock.present(&h);
    }
    let introduced = Instant::now();
    let mut settle_frames = 0_u32;
    while bench.probing() || introduced.elapsed() < IDLE_TAIL || started.elapsed() < MIN_SPAN {
        assert!(h.pump(), "WM_QUIT while the probes settle");
        bench.begin_frame();
        bench.end_frame();
        ok(h.present(), "Present");
        settle_frames += 1;
    }
    let to = log.mark();
    let landing = bench.landing();

    let stats = clock.stats();
    let spikes = clock.over(stats.p50 * 2);
    let mut compiles = String::new();
    for rows in log.compilation_rows(from, to) {
        let _ = write!(compiles, "perf: window in the span, its compiles\n{rows}");
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
         {landing}{perf}{compiles}",
        offscreen = if offscreen {
            " + 1 into an offscreen target not cleared that frame"
        } else {
            ""
        },
        frames = stats.frames,
        shaders = bench.probes.len(),
        elapsed = clock.elapsed(),
        span = started.elapsed(),
        row = stats.row(),
        work = clock.work_stats().row(),
        limit = (stats.p50 * 2).as_secs_f64() * 1e3,
        perf = log.perf_rows(from, to).section(),
    );
    write_report(name, &log, &body);
}

/// A new shader and the frames its probe draws went out in.
struct Probe<'h> {
    ps: PixelShader<'h>,
    born: Instant,
    /// Frame numbers of its probe draws, the first being the frame it appeared in.
    frames: Vec<u32>,
    last: Instant,
}

impl Probe<'_> {
    /// Whether the probe still draws in a frame that starts at `now`.
    fn open(&self, now: Instant) -> bool {
        now - self.born < LAND_WINDOW && self.frames.len() < MAX_PROBE_DRAWS
    }
}

/// The benchmark's device objects, its probes, and the start time of every frame.
struct Stutter<'h> {
    h: &'h Harness,
    back_buffer: Surface<'h>,
    /// The texture behind [`Self::probe_target`], held so the target outlives the frames.
    _probe_texture: Texture<'h>,
    probe_target: Surface<'h>,
    texture: Texture<'h>,
    vb: VertexBuffer<'h>,
    ib: IndexBuffer<'h>,
    decl: VertexDeclaration<'h>,
    vs: VertexShader<'h>,
    base_ps: PixelShader<'h>,
    vertex_count: u32,
    triangles: u32,
    probes: Vec<Probe<'h>>,
    /// Probes created this frame, which draw whatever the interval says.
    fresh: usize,
    frame_starts: Vec<Instant>,
    salt: u16,
}

impl<'h> Stutter<'h> {
    fn new(h: &'h Harness) -> Self {
        let probe_texture = h.create_texture(
            PROBE_EDGE,
            PROBE_EDGE,
            1,
            D3DUSAGE_RENDERTARGET,
            D3DFMT_X8R8G8B8,
            D3DPOOL_DEFAULT,
        );
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
            probe_target: probe_texture.surface_level(0),
            _probe_texture: probe_texture,
            texture: pattern_texture(h, 0xFF40_8020),
            vb,
            ib,
            decl: h.create_vertex_declaration(&TEXTURED_DECL),
            vs: h.create_vertex_shader(&material_vs(&Model::Sm2, 0.0)),
            base_ps: h.create_pixel_shader(&material_ps(&Model::Sm2, [0.0; 4], false)),
            vertex_count,
            triangles,
            probes: Vec::new(),
            fresh: 0,
            frame_starts: Vec::new(),
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
        ok(
            h.set_render_state(D3DRS_SRCBLEND, D3DBLEND_ONE),
            "additive source",
        );
        ok(
            h.set_render_state(D3DRS_DESTBLEND, D3DBLEND_ONE),
            "additive destination",
        );
        ok(h.set_render_target(0, &bench.probe_target), "probe target");
        ok(
            h.clear(D3DCLEAR_TARGET, 0xFF00_0000, 1.0, 0),
            "probe clear, once",
        );
        ok(h.set_render_target(0, &bench.back_buffer), "back buffer");
        bench
    }

    /// Open a frame: clear, and the base draws on the back buffer.
    fn begin_frame(&mut self) {
        let h = self.h;
        self.frame_starts.push(Instant::now());
        ok(h.begin_scene(), "BeginScene");
        ok(
            h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, 0xFF20_3040, 1.0, 0),
            "clear",
        );
        ok(h.set_pixel_shader(&self.base_ps), "base PS");
        for at in 0..BASE_DRAWS {
            let x = ratio(at % 20, 20).mul_add(1.8, -0.95);
            let y = ratio(at / 20 % 20, 20).mul_add(1.8, -0.95);
            self.draw(0.1, x, y);
        }
    }

    /// Create this frame's new shaders and draw each once on the back buffer, or offscreen.
    fn introduce(&mut self, per_frame: u32, offscreen: bool) {
        let h = self.h;
        rs(h, D3DRS_ALPHABLENDENABLE, 1);
        for at in 0..per_frame {
            let ps = self.new_shader();
            ok(h.set_pixel_shader(&ps), "new PS");
            let slot = BASE_DRAWS + at + self.frame() % 7;
            let x = ratio(slot % 20, 20).mul_add(1.8, -0.95);
            let y = ratio(slot / 20 % 20, 20).mul_add(1.8, -0.95);
            self.draw(0.1, x, y);
            self.adopt(ps);
        }
        if offscreen {
            // Its first probe draw is its draw into the uncleared target.
            let ps = self.new_shader();
            self.adopt(ps);
        }
        rs(h, D3DRS_ALPHABLENDENABLE, 0);
    }

    /// Draw the open probes into the probe target, then close the frame.
    fn end_frame(&mut self) {
        let h = self.h;
        let now = Instant::now();
        let frame = self.frame();
        let first_fresh = self.probes.len() - self.fresh;
        let due: Vec<usize> = (0..self.probes.len())
            .filter(|&at| {
                let probe = &self.probes[at];
                at >= first_fresh || (probe.open(now) && now - probe.last >= PROBE_INTERVAL)
            })
            .collect();
        self.fresh = 0;
        if !due.is_empty() {
            ok(h.set_render_target(0, &self.probe_target), "probe target");
            ok(h.set_pixel_shader_constant_f(0, &[0.0; 4]), "probe tint");
            rs(h, D3DRS_ALPHABLENDENABLE, 1);
            for at in due {
                ok(h.set_pixel_shader(&self.probes[at].ps), "probe PS");
                let cell = u32::try_from(at).expect("probe count fits u32");
                let (col, row) = (cell % CELLS_PER_ROW, cell / CELLS_PER_ROW);
                let edge = ratio(CELL_EDGE * 2, PROBE_EDGE);
                self.draw(
                    edge,
                    ratio(col, CELLS_PER_ROW).mul_add(2.0, -1.0),
                    ratio(row + 1, CELLS_PER_ROW).mul_add(-2.0, 1.0),
                );
                let probe = &mut self.probes[at];
                probe.frames.push(frame);
                probe.last = now;
            }
            rs(h, D3DRS_ALPHABLENDENABLE, 0);
            ok(h.set_pixel_shader_constant_f(0, &[1.0; 4]), "tint");
            ok(h.set_render_target(0, &self.back_buffer), "back buffer");
        }
        ok(h.end_scene(), "EndScene");
    }

    /// Whether any probe still draws.
    fn probing(&self) -> bool {
        let now = Instant::now();
        self.probes.iter().any(|probe| probe.open(now))
    }

    /// Read the probe target back and account for when each shader's draws landed.
    fn landing(&self) -> String {
        let h = self.h;
        let sysmem = h.create_offscreen_plain_surface(
            PROBE_EDGE,
            PROBE_EDGE,
            D3DFMT_X8R8G8B8,
            D3DPOOL_SYSTEMMEM,
        );
        ok(
            h.get_render_target_data_hr(&self.probe_target, &sysmem),
            "probe readback",
        );
        let locked = sysmem.lock_rect(D3DLOCK_READONLY);
        let pitch = usize::try_from(locked.pitch()).expect("positive pitch") / 4;
        let edge = usize::try_from(PROBE_EDGE).expect("probe edge fits usize");
        let pixels = locked.as_u32(pitch * edge);
        let mut at_once = 0;
        let mut never = 0;
        let mut overcounted = 0;
        let mut skipped_frames = Vec::new();
        let mut skipped_times = Vec::new();
        for (at, probe) in self.probes.iter().enumerate() {
            let cell = u32::try_from(at).expect("probe count fits u32");
            let centre = |index: u32| {
                usize::try_from(index * CELL_EDGE + CELL_EDGE / 2).expect("pixel fits usize")
            };
            let (x, y) = (centre(cell % CELLS_PER_ROW), centre(cell / CELLS_PER_ROW));
            let landed = usize::try_from((pixels[y * pitch + x] >> 16) & 0xFF).expect("byte");
            let drawn = probe.frames.len();
            if landed == 0 {
                never += 1;
            } else if landed > drawn {
                overcounted += 1;
            } else if landed == drawn {
                at_once += 1;
            } else {
                let skipped = drawn - landed;
                let (first, landing) = (probe.frames[0], probe.frames[skipped]);
                skipped_frames.push(landing - first);
                skipped_times
                    .push(self.frame_starts[slot(landing)] - self.frame_starts[slot(first)]);
            }
        }
        drop(locked);
        let late = skipped_frames.len();
        let mut report = format!(
            "landing (readback of the probe cells): {total} new shaders: {at_once} drawn the \
             first time, {late} skipped then drawn, {never} never drawn within {LAND_WINDOW:?}\
             {overcounted}\n",
            total = self.probes.len(),
            overcounted = if overcounted > 0 {
                format!(", {overcounted} with more draws landed than issued")
            } else {
                String::new()
            },
        );
        if late > 0 {
            skipped_frames.sort_unstable();
            skipped_times.sort_unstable();
            let pick = |percent: usize| (late * percent / 100).min(late - 1);
            let _ = writeln!(
                report,
                "frames to land (skipped then drawn): p50 {} p90 {} max {}; time to land: \
                 p50 {:.3} ms p90 {:.3} ms max {:.3} ms",
                skipped_frames[pick(50)],
                skipped_frames[pick(90)],
                skipped_frames[late - 1],
                skipped_times[pick(50)].as_secs_f64() * 1e3,
                skipped_times[pick(90)].as_secs_f64() * 1e3,
                skipped_times[late - 1].as_secs_f64() * 1e3,
            );
        }
        report
    }

    /// The number of the frame being recorded.
    fn frame(&self) -> u32 {
        u32::try_from(self.frame_starts.len() - 1).expect("frame count fits u32")
    }

    /// The next never-seen pixel shader.
    ///
    /// Its `def c7` is one red step, the running count and the run's salt:
    /// with `c0` at zero the shader writes that one step, and the other two
    /// lanes make it unique without reaching an 8-bit step.
    fn new_shader(&self) -> PixelShader<'h> {
        let count = u16::try_from(self.probes.len() + 1).expect("new shader count fits u16");
        let tint = [
            1.0 / 255.0,
            f32::from(count) * 1.0e-6,
            f32::from(self.salt) * 1.0e-8,
            0.0,
        ];
        self.h
            .create_pixel_shader(&material_ps(&Model::Sm2, tint, false))
    }

    /// Start probing `ps` from this frame on.
    fn adopt(&mut self, ps: PixelShader<'h>) {
        let now = Instant::now();
        self.probes.push(Probe {
            ps,
            born: now,
            frames: Vec::new(),
            last: now,
        });
        self.fresh += 1;
    }

    /// One mesh draw scaled by `scale` at clip-space `(x, y)`.
    fn draw(&self, scale: f32, x: f32, y: f32) {
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

/// A per-run value that keeps this run's shaders unlike any earlier run's.
fn salt() -> u16 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.subsec_nanos());
    u16::try_from(nanos % 65_521).expect("a remainder below 65521 fits u16")
}

/// `value` as an index into the frame list.
fn slot(value: u32) -> usize {
    usize::try_from(value).expect("a frame number fits usize")
}

fn rs(h: &Harness, state: u32, value: u32) {
    ok(h.set_render_state(state, value), "SetRenderState");
}
