//! What one COM call into the layer costs the API thread, per kind of call.
//!
//! Guards the per-call cost of the device's entry points. In real gameplay
//! the game's own thread is the bottleneck, and every call it makes into the
//! layer (a state change, a constant upload, a bind, a draw, a buffer lock)
//! is paid on that thread, so a few nanoseconds more per call is frame time
//! a frame benchmark would spread too thin to see. This one isolates each
//! kind: a batch is [`CALLS`] calls of one kind between two `Present`s,
//! timed with `Instant` (`QueryPerformanceCounter`) around the calls alone,
//! and the figure a kind reports is the median batch's nanoseconds per
//! iteration.
//!
//! State calls alternate between two values, so that none is a redundant
//! set the layer may drop, except in the `_same` kinds, which repeat the
//! bound value and so measure the redundant path. Every batch ends on the
//! state the others start from. `set_texture_stage_state` and
//! `set_transform_world` run with both shaders cleared, the fixed-function
//! pipeline bound, as a fixed-function title such as World of Warcraft 1.12
//! makes them, so the layer pays the fixed-function state work it may skip
//! while shaders are bound; their `_shader_bound` twins run the same calls
//! under the programmable pair, which is that skipped path. The draws are
//! one small triangle each into a 64x64 back buffer; `draw_clean` changes
//! nothing between draws,
//! `draw_dirty_rs` pairs each draw with one `SetRenderState` change, and
//! `draw_ff_transform` pairs each fixed-function draw with one world
//! `SetTransform`, so for those, and for the lock kind (a
//! `D3DLOCK_NOOVERWRITE` `Lock` of 64 bytes, a 64-byte write and the
//! `Unlock`), the figure is per pair. `Present` between batches hands each
//! batch to the encoder, and the layer's encoding of a batch overlaps the
//! next; encoder back-pressure reaches the figures only through the
//! untimed `Present`, so they are the API thread's own cost per call.
//!
//! The kinds run interleaved, one batch of each per round, so a drift in
//! the machine's speed over the run touches every kind alike. The rounds
//! run until there are at least [`ROUNDS`] of them and [`MIN_MEASURED`]
//! has passed, which puts one whole window of a `PERF=1` build's summary
//! in the measured span.

use core::fmt::Write as _;
use std::time::{Duration, Instant, SystemTime};

use mtld3d_tests::{
    Harness, HarnessConfig, IndexBuffer, MemorySample, PixelShader, Texture, TexturedVertex,
    VertexBuffer, VertexDeclaration, VertexShader,
};
use mtld3d_types::{
    D3DCULL_CCW, D3DCULL_NONE, D3DFMT_INDEX16, D3DFVF_DIFFUSE, D3DFVF_TEX1, D3DFVF_XYZ,
    D3DLOCK_NOOVERWRITE, D3DPOOL_DEFAULT, D3DPOOL_MANAGED, D3DPRESENT_INTERVAL_IMMEDIATE,
    D3DPT_TRIANGLELIST, D3DRS_CULLMODE, D3DRS_LIGHTING, D3DRS_ZENABLE, D3DSAMP_MAGFILTER,
    D3DTEXF_LINEAR, D3DTEXF_POINT, D3DTOP_ADD, D3DTOP_MODULATE, D3DTS_PROJECTION, D3DTS_VIEW,
    D3DTS_WORLD, D3DTSS_COLOROP, D3DUSAGE_DYNAMIC, D3DUSAGE_WRITEONLY,
};

use crate::bench::{
    Class, Direction, FrameWork, IDENTITY_ROWS, LayerLog, Metrics, Model, STRIDE, TEXTURED_DECL,
    Value, material_ps, material_vs, memory_section, nearest_rank, ok, pattern_texture, world_rows,
    write_report,
};

/// Edge of the back buffer, which every draw lands in.
const EDGE: u32 = 64;
/// Calls (or pairs) in one timed batch. Even, so an alternation ends on the value it started from.
const CALLS: u32 = 20_000;
const _: () = assert!(
    CALLS.is_multiple_of(2),
    "an alternation ends where it began"
);
/// The least number of measured rounds, one batch of every kind each.
const ROUNDS: usize = 15;
const MIN_MEASURED: Duration = Duration::from_secs(12);
/// Untimed rounds first, so every pipeline the batches need is built before the timing.
const WARM_UP_ROUNDS: usize = 2;
/// Bytes of the dynamic vertex buffer the lock kind cycles through.
const DYNAMIC_BYTES: u32 = 64 * 1024;
/// Bytes one lock covers and writes.
const LOCK_BYTES: u32 = 64;
const FVF: u32 = D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1;
/// The two cull modes the render-state kinds alternate, the second the bound one.
///
/// The triangle winds clockwise, so neither culls it.
const CULL: [u32; 2] = [D3DCULL_CCW, D3DCULL_NONE];
/// The two colour operations of stage 0, the second the default.
const COLOR_OPS: [u32; 2] = [D3DTOP_ADD, D3DTOP_MODULATE];
/// The two magnification filters of sampler 0, the second the default.
const MAG_FILTERS: [u32; 2] = [D3DTEXF_LINEAR, D3DTEXF_POINT];
/// The two world matrices, the second the bound identity.
const WORLDS: [[f32; 16]; 2] = [
    [
        1.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, //
        0.1, 0.1, 0.0, 1.0,
    ],
    IDENTITY_ROWS,
];
/// Two sets of sixteen constant rows the constant kinds alternate.
const ROWS: [[f32; 64]; 2] = [[0.25; 64], [0.5; 64]];
/// The first register the constant kinds write, above every register the programs read.
const FIRST_FREE_REGISTER: u32 = 8;
/// What the lock kind writes into each locked span.
const PAYLOAD: [u32; 16] = [0x3F80_0000; 16];

/// One kind of call the benchmark times.
enum Kind {
    SetRenderState,
    SetRenderStateSame,
    SetTextureStageState,
    SetTextureStageStateShaderBound,
    SetSamplerState,
    SetTexture,
    SetTextureSame,
    SetTransformWorld,
    SetTransformWorldShaderBound,
    SetVsConstantF4,
    SetVsConstantF16,
    SetPsConstantF4,
    SetStreamSource,
    SetDeclFvf,
    DrawClean,
    DrawDirtyRs,
    DrawFfTransform,
    VbLockNooverwrite,
}

/// Every kind, in the order a round runs them and the report lists them.
const KINDS: [Kind; 18] = [
    Kind::SetRenderState,
    Kind::SetRenderStateSame,
    Kind::SetTextureStageState,
    Kind::SetTextureStageStateShaderBound,
    Kind::SetSamplerState,
    Kind::SetTexture,
    Kind::SetTextureSame,
    Kind::SetTransformWorld,
    Kind::SetTransformWorldShaderBound,
    Kind::SetVsConstantF4,
    Kind::SetVsConstantF16,
    Kind::SetPsConstantF4,
    Kind::SetStreamSource,
    Kind::SetDeclFvf,
    Kind::DrawClean,
    Kind::DrawDirtyRs,
    Kind::DrawFfTransform,
    Kind::VbLockNooverwrite,
];

impl Kind {
    /// The name the report and the metric carry.
    const fn name(&self) -> &'static str {
        match self {
            Self::SetRenderState => "set_render_state",
            Self::SetRenderStateSame => "set_render_state_same",
            Self::SetTextureStageState => "set_texture_stage_state",
            Self::SetTextureStageStateShaderBound => "set_texture_stage_state_shader_bound",
            Self::SetSamplerState => "set_sampler_state",
            Self::SetTexture => "set_texture",
            Self::SetTextureSame => "set_texture_same",
            Self::SetTransformWorld => "set_transform_world",
            Self::SetTransformWorldShaderBound => "set_transform_world_shader_bound",
            Self::SetVsConstantF4 => "set_vs_constant_f_4",
            Self::SetVsConstantF16 => "set_vs_constant_f_16",
            Self::SetPsConstantF4 => "set_ps_constant_f_4",
            Self::SetStreamSource => "set_stream_source",
            Self::SetDeclFvf => "set_decl_fvf",
            Self::DrawClean => "draw_clean",
            Self::DrawDirtyRs => "draw_dirty_rs",
            Self::DrawFfTransform => "draw_ff_transform",
            Self::VbLockNooverwrite => "vb_lock_nooverwrite_64",
        }
    }

    /// COM calls one iteration makes: two for the pairs, one otherwise.
    ///
    /// The lock kind's pair is the `Lock` and the `Unlock`.
    const fn calls(&self) -> u64 {
        match self {
            Self::DrawDirtyRs | Self::DrawFfTransform | Self::VbLockNooverwrite => 2,
            _ => 1,
        }
    }
}

/// Every kind's batches, timed round by round.
#[test]
#[ignore = "benchmark, run by `make bench`"]
fn api_call_cost() {
    let h = Harness::create(&HarnessConfig {
        width: EDGE,
        height: EDGE,
        presentation_interval: D3DPRESENT_INTERVAL_IMMEDIATE,
        ..HarnessConfig::default()
    });
    let since = SystemTime::now();
    let bench = Bench::new(&h);
    for _ in 0..WARM_UP_ROUNDS {
        for kind in &KINDS {
            bench.frame(kind);
        }
    }
    let log = LayerLog::find(since);
    let warm = MemorySample::now();

    let from = log.mark();
    let started = Instant::now();
    let mut samples: Vec<Vec<f64>> = KINDS.iter().map(|_| Vec::new()).collect();
    let mut rounds = 0;
    while rounds < ROUNDS || started.elapsed() < MIN_MEASURED {
        for (kind, samples) in KINDS.iter().zip(&mut samples) {
            let batch = bench.frame(kind);
            samples.push(batch.as_secs_f64() * 1e9 / f64::from(CALLS));
        }
        rounds += 1;
    }
    let measured = started.elapsed();
    let to = log.mark();
    let end = MemorySample::now();

    let rounds_u64 = u64::try_from(rounds).expect("round count fits u64");
    let calls_total: u64 = KINDS
        .iter()
        .map(|kind| kind.calls() * u64::from(CALLS) * rounds_u64)
        .sum();
    let mut table = String::new();
    let mut metrics = Metrics::new("api_call_cost", &h);
    for (kind, samples) in KINDS.iter().zip(&mut samples) {
        samples.sort_unstable_by(f64::total_cmp);
        let median = samples[nearest_rank(samples.len(), 50)];
        let unit = if kind.calls() == 2 { "pair" } else { "call" };
        let _ = writeln!(
            table,
            "  {name:<24} median {median:>8.1} ns/{unit}  min {min:>8.1}  max {max:>8.1}",
            name = kind.name(),
            min = samples[0],
            max = samples[samples.len() - 1],
        );
        metrics.metric(
            &format!("ns_per_call.{}", kind.name()),
            Value::Ns(median),
            Direction::Lower,
            Class::Time,
        );
    }
    let body = format!(
        "shape: back buffer {EDGE}x{EDGE}, one small triangle per draw, vs_2_0/ps_2_0 \
         (fixed function for draw_ff_transform)\n\
         batches: {CALLS} calls (pairs where marked) of one kind between two Presents, \
         timed around the calls alone; median, min and max batch per kind\n\
         warm-up: {WARM_UP_ROUNDS} rounds; measured: {rounds} rounds of {kinds} kinds in \
         {measured:.2?} (at least {ROUNDS} rounds and {MIN_MEASURED:?}), \
         {calls_total} COM calls timed\n\
         {table}{memory}{perf}",
        kinds = KINDS.len(),
        memory = memory_section(&warm, &end),
        perf = log.perf_rows(from, to).section(),
    );
    metrics.metric(
        "calls_total",
        Value::Count(calls_total),
        Direction::Higher,
        Class::Info,
    );
    metrics.metric(
        "rounds",
        Value::Count(rounds_u64),
        Direction::Higher,
        Class::Info,
    );
    metrics.metric(
        "measured.ms",
        Value::Ms(measured),
        Direction::Lower,
        Class::Info,
    );
    metrics.memory(&warm, &end);
    // Each frame is one batch of one call kind, so a window's per-frame
    // counts depend on where in the rotation it starts and ends.
    metrics.perf(&log.perf_kv_last_full(from, to), &FrameWork::Varying);
    write_report(&metrics, &log, &body);
}

/// The device objects the batches bind and draw with.
struct Bench<'h> {
    h: &'h Harness,
    /// The bound stream source.
    vb: VertexBuffer<'h>,
    /// A second buffer of the same triangle, for the stream-source kind to alternate to.
    other_vb: VertexBuffer<'h>,
    ib: IndexBuffer<'h>,
    decl: VertexDeclaration<'h>,
    vs: VertexShader<'h>,
    ps: PixelShader<'h>,
    /// The bound texture of stage 0.
    texture: Texture<'h>,
    other_texture: Texture<'h>,
    /// A dynamic buffer the lock kind writes and nothing draws from.
    dynamic: VertexBuffer<'h>,
}

impl<'h> Bench<'h> {
    fn new(h: &'h Harness) -> Self {
        let triangle = [(0.0, 0.0), (0.0, 0.05), (0.05, 0.0)].map(|(x, y)| TexturedVertex {
            x,
            y,
            z: 0.5,
            color: 0xFFFF_FFFF,
            u: x,
            v: y,
        });
        let buffer = || {
            let vb = h.create_vertex_buffer(3 * STRIDE, D3DUSAGE_WRITEONLY, 0, D3DPOOL_MANAGED);
            vb.lock(0, 0, 0).write(&triangle);
            vb
        };
        let (vb, other_vb) = (buffer(), buffer());
        let ib = h.create_index_buffer(6, D3DUSAGE_WRITEONLY, D3DFMT_INDEX16, D3DPOOL_MANAGED);
        ib.lock(0, 0, 0).write(&[0u16, 1, 2]);
        let bench = Self {
            h,
            vb,
            other_vb,
            ib,
            decl: h.create_vertex_declaration(&TEXTURED_DECL),
            vs: h.create_vertex_shader(&material_vs(&Model::Sm2, 0.0)),
            ps: h.create_pixel_shader(&material_ps(&Model::Sm2, [0.0; 4], false)),
            texture: pattern_texture(h, 0xFF40_8020),
            other_texture: pattern_texture(h, 0xFF20_4080),
            dynamic: h.create_vertex_buffer(
                DYNAMIC_BYTES,
                D3DUSAGE_DYNAMIC | D3DUSAGE_WRITEONLY,
                0,
                D3DPOOL_DEFAULT,
            ),
        };
        ok(h.set_vertex_declaration(&bench.decl), "declaration");
        ok(h.set_stream_source(0, &bench.vb, 0, STRIDE), "stream");
        ok(h.set_indices(&bench.ib), "indices");
        ok(h.set_texture(0, &bench.texture), "texture");
        ok(h.set_vertex_shader(&bench.vs), "VS");
        ok(h.set_pixel_shader(&bench.ps), "PS");
        ok(
            h.set_vertex_shader_constant_f(0, &IDENTITY_ROWS),
            "view-projection",
        );
        ok(
            h.set_vertex_shader_constant_f(4, &world_rows(1.0, 0.0, 0.0, 0.0)),
            "world rows",
        );
        ok(h.set_pixel_shader_constant_f(0, &[1.0; 4]), "tint");
        ok(h.set_render_state(D3DRS_CULLMODE, CULL[1]), "cull mode");
        ok(h.set_render_state(D3DRS_LIGHTING, 0), "lighting off");
        ok(h.set_render_state(D3DRS_ZENABLE, 0), "depth off");
        for state in [D3DTS_WORLD, D3DTS_VIEW, D3DTS_PROJECTION] {
            ok(h.set_transform(state, &IDENTITY_ROWS), "transform");
        }
        bench
    }

    /// One frame: a timed batch of `kind` inside a scene, then `Present`.
    fn frame(&self, kind: &Kind) -> Duration {
        let h = self.h;
        assert!(h.pump(), "WM_QUIT during the batches");
        ok(h.begin_scene(), "BeginScene");
        let batch = self.batch(kind);
        ok(h.end_scene(), "EndScene");
        ok(h.present(), "Present");
        batch
    }

    /// [`CALLS`] iterations of `kind`, timed around the iterations alone.
    fn batch(&self, kind: &Kind) -> Duration {
        let h = self.h;
        let rows = |count: usize| {
            let end = count * 4;
            [&ROWS[0][..end], &ROWS[1][..end]]
        };
        let stage_state = || {
            timed(|odd| {
                ok(
                    h.set_texture_stage_state(0, D3DTSS_COLOROP, COLOR_OPS[usize::from(odd)]),
                    "SetTextureStageState",
                );
            })
        };
        let world = || {
            timed(|odd| {
                ok(
                    h.set_transform(D3DTS_WORLD, &WORLDS[usize::from(odd)]),
                    "SetTransform",
                );
            })
        };
        match kind {
            Kind::SetRenderState => timed(|odd| {
                ok(
                    h.set_render_state(D3DRS_CULLMODE, CULL[usize::from(odd)]),
                    "SetRenderState",
                );
            }),
            Kind::SetRenderStateSame => timed(|_| {
                ok(
                    h.set_render_state(D3DRS_CULLMODE, CULL[1]),
                    "SetRenderState",
                );
            }),
            Kind::SetTextureStageState => self.fixed_function(stage_state),
            Kind::SetTextureStageStateShaderBound => stage_state(),
            Kind::SetSamplerState => timed(|odd| {
                ok(
                    h.set_sampler_state(0, D3DSAMP_MAGFILTER, MAG_FILTERS[usize::from(odd)]),
                    "SetSamplerState",
                );
            }),
            Kind::SetTexture => {
                let textures = [&self.other_texture, &self.texture];
                timed(|odd| ok(h.set_texture(0, textures[usize::from(odd)]), "SetTexture"))
            }
            Kind::SetTextureSame => timed(|_| ok(h.set_texture(0, &self.texture), "SetTexture")),
            Kind::SetTransformWorld => self.fixed_function(world),
            Kind::SetTransformWorldShaderBound => world(),
            Kind::SetVsConstantF4 => {
                let rows = rows(4);
                timed(|odd| {
                    ok(
                        h.set_vertex_shader_constant_f(FIRST_FREE_REGISTER, rows[usize::from(odd)]),
                        "SetVertexShaderConstantF",
                    );
                })
            }
            Kind::SetVsConstantF16 => {
                let rows = rows(16);
                timed(|odd| {
                    ok(
                        h.set_vertex_shader_constant_f(FIRST_FREE_REGISTER, rows[usize::from(odd)]),
                        "SetVertexShaderConstantF",
                    );
                })
            }
            Kind::SetPsConstantF4 => {
                let rows = rows(4);
                timed(|odd| {
                    ok(
                        h.set_pixel_shader_constant_f(FIRST_FREE_REGISTER, rows[usize::from(odd)]),
                        "SetPixelShaderConstantF",
                    );
                })
            }
            Kind::SetStreamSource => {
                let buffers = [&self.other_vb, &self.vb];
                timed(|odd| {
                    ok(
                        h.set_stream_source(0, buffers[usize::from(odd)], 0, STRIDE),
                        "SetStreamSource",
                    );
                })
            }
            Kind::SetDeclFvf => timed(|odd| {
                if odd {
                    ok(h.set_vertex_declaration(&self.decl), "SetVertexDeclaration");
                } else {
                    ok(h.set_fvf(FVF), "SetFVF");
                }
            }),
            Kind::DrawClean => timed(|_| self.draw()),
            Kind::DrawDirtyRs => timed(|odd| {
                ok(
                    h.set_render_state(D3DRS_CULLMODE, CULL[usize::from(odd)]),
                    "SetRenderState",
                );
                self.draw();
            }),
            Kind::DrawFfTransform => self.fixed_function(|| {
                timed(|odd| {
                    ok(
                        h.set_transform(D3DTS_WORLD, &WORLDS[usize::from(odd)]),
                        "SetTransform",
                    );
                    self.draw();
                })
            }),
            Kind::VbLockNooverwrite => {
                let mut offset = 0;
                timed(|_| {
                    self.dynamic
                        .lock(offset, LOCK_BYTES, D3DLOCK_NOOVERWRITE)
                        .write(&PAYLOAD);
                    offset = (offset + LOCK_BYTES) % DYNAMIC_BYTES;
                })
            }
        }
    }

    /// Run `batch` with both shaders cleared, the fixed-function pipeline bound, then rebind them.
    fn fixed_function(&self, batch: impl FnOnce() -> Duration) -> Duration {
        let h = self.h;
        ok(h.clear_vertex_shader(), "fixed-function VS");
        ok(h.clear_pixel_shader(), "fixed-function PS");
        let batch = batch();
        ok(h.set_vertex_shader(&self.vs), "VS");
        ok(h.set_pixel_shader(&self.ps), "PS");
        batch
    }

    /// The triangle, drawn indexed from the bound streams.
    fn draw(&self) {
        ok(
            self.h
                .draw_indexed_primitive(D3DPT_TRIANGLELIST, 0, 0, 3, 0, 1),
            "DrawIndexedPrimitive",
        );
    }
}

/// Run `body` [`CALLS`] times and return how long the calls took.
///
/// `body` is told whether the iteration is odd, which selects the second of
/// an alternating pair; the last iteration is odd.
fn timed(mut body: impl FnMut(bool)) -> Duration {
    let started = Instant::now();
    for at in 0..CALLS {
        body(at % 2 == 1);
    }
    started.elapsed()
}
