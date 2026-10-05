//! What a draw with an attribute past its stream's stride costs against one that fits.
//!
//! Both kinds of each pair draw one small triangle under the same
//! declaration, a position at byte 0 and a colour at byte 28, and the same
//! shaders. The `fits` kinds step 32 bytes, so the colour lies inside each
//! vertex; the `crossing` kinds step 16, so the colour lies in the next
//! vertex and the draw fetches it through a binding of its own. The `bound`
//! kinds draw from a vertex buffer, the `up` kinds through `DrawPrimitiveUP`,
//! whose crossing copy also zero-fills the last vertex's colour.
//!
//! A kind runs [`BLOCK_FRAMES`] frames of [`DRAWS`] draws in a row, so the
//! encoder works on frames of that kind alone; the last [`MEASURED_FRAMES`]
//! are measured. `api` is the draws' own time on the API thread; `frame` is
//! the time from the first draw to the return of the frame's `Present`,
//! which waits for the encoder to take the frame before it, so with draws
//! this small it is the slower of the API thread and the encoder, per draw.
//! The kinds run interleaved, one block of each per round.

use core::fmt::Write as _;
use std::time::SystemTime;

use mtld3d_tests::{
    Harness, HarnessConfig, PixelShader, VertexBuffer, VertexDeclaration, VertexShader,
};
use mtld3d_types::{
    D3DCULL_NONE, D3DPOOL_MANAGED, D3DPRESENT_INTERVAL_IMMEDIATE, D3DPT_TRIANGLELIST,
    D3DRS_CULLMODE, D3DRS_ZENABLE, D3DUSAGE_WRITEONLY,
};

use crate::{
    bench::{Class, Direction, LayerLog, Metrics, TscClock, Value, nearest_rank, ok, write_report},
    streams::{PS_DIFFUSE, VS_POS_COLOR_TEXCOORD, crossing_color_elements},
};

/// Edge of the back buffer, which every draw lands in.
const EDGE: u32 = 64;
/// Draws in one frame.
const DRAWS: u32 = 10_000;
/// Frames of one kind in a row.
const BLOCK_FRAMES: usize = 4;
/// The last frames of a block that are measured: the frame before each has the same kind.
const MEASURED_FRAMES: usize = 2;
/// The least number of measured rounds, one block of every kind each.
const ROUNDS: usize = 15;
/// Untimed rounds first, so every pipeline the kinds need is built before the timing.
const WARM_UP_ROUNDS: usize = 2;

/// A position and a colour field, padded to the 32-byte stride of the `fits` kinds.
#[repr(C)]
#[derive(Clone, Copy)]
struct WideVertex {
    x: f32,
    y: f32,
    z: f32,
    pad: [u32; 4],
    color: u32,
}

/// A position and a colour field in 16 bytes, the stride of the `crossing` kinds.
#[repr(C)]
#[derive(Clone, Copy)]
struct NarrowVertex {
    x: f32,
    y: f32,
    z: f32,
    color: u32,
}

/// The triangle's corners.
const CORNERS: [(f32, f32); 3] = [(0.0, 0.0), (0.0, 0.05), (0.05, 0.0)];

/// One kind of draw the benchmark times.
enum Kind {
    BoundFits,
    BoundCrossing,
    UpFits,
    UpCrossing,
}

/// Every kind, in the order a round runs them and the report lists them.
const KINDS: [Kind; 4] = [
    Kind::BoundFits,
    Kind::BoundCrossing,
    Kind::UpFits,
    Kind::UpCrossing,
];

impl Kind {
    /// The name the report and the metrics carry.
    const fn name(&self) -> &'static str {
        match self {
            Self::BoundFits => "bound_fits",
            Self::BoundCrossing => "bound_crossing",
            Self::UpFits => "up_fits",
            Self::UpCrossing => "up_crossing",
        }
    }
}

/// Every kind's frames, block by block.
#[test]
#[ignore = "benchmark, run by `make bench`"]
fn crossing_stride_draws() {
    let tsc = TscClock::calibrated();
    let since = SystemTime::now();
    let h = Harness::create(&HarnessConfig {
        width: EDGE,
        height: EDGE,
        presentation_interval: D3DPRESENT_INTERVAL_IMMEDIATE,
        ..HarnessConfig::default()
    });
    let bench = Bench::new(&h);
    for _ in 0..WARM_UP_ROUNDS {
        for kind in &KINDS {
            bench.block(kind);
        }
    }
    let log = LayerLog::find(since);
    let mut api: Vec<Vec<f64>> = KINDS.iter().map(|_| Vec::new()).collect();
    let mut frame: Vec<Vec<f64>> = KINDS.iter().map(|_| Vec::new()).collect();
    for _ in 0..ROUNDS {
        for ((kind, api), frame) in KINDS.iter().zip(&mut api).zip(&mut frame) {
            for (draws, whole) in bench.block(kind) {
                api.push(TscClock::ticks_ns(draws) / f64::from(DRAWS));
                frame.push(TscClock::ticks_ns(whole) / f64::from(DRAWS));
            }
        }
    }

    let mut table = String::new();
    let mut metrics = Metrics::new("crossing_stride_draws", &h, &tsc);
    for ((kind, api), frame) in KINDS.iter().zip(&mut api).zip(&mut frame) {
        for (row, samples) in [("api", api), ("frame", frame)] {
            samples.sort_unstable_by(f64::total_cmp);
            let median = samples[nearest_rank(samples.len(), 50)];
            let _ = writeln!(
                table,
                "  {name:<16} {row:<6} median {median:>8.1} ns/draw  min {min:>8.1}  max {max:>8.1}",
                name = kind.name(),
                min = samples[0],
                max = samples[samples.len() - 1],
            );
            metrics.metric(
                &format!("{row}_ns_per_draw.{}", kind.name()),
                Value::Ns(median),
                Direction::Lower,
                Class::Time,
            );
        }
    }
    let body = format!(
        "shape: back buffer {EDGE}x{EDGE}, one small triangle per draw, vs_2_0/ps_2_0, \
         colour at byte 28 of a 32-byte (fits) or 16-byte (crossing) vertex\n\
         frames: {DRAWS} draws each, {BLOCK_FRAMES} of one kind in a row, the last \
         {MEASURED_FRAMES} measured; api is the draws alone, frame the first draw to the \
         return of Present; medians over {ROUNDS} rounds after {WARM_UP_ROUNDS} warm-up \
         rounds\n{table}",
    );
    write_report(&metrics, &log, &body);
}

/// The device objects the kinds draw with.
struct Bench<'h> {
    h: &'h Harness,
    wide_vb: VertexBuffer<'h>,
    narrow_vb: VertexBuffer<'h>,
    _decl: VertexDeclaration<'h>,
    _vs: VertexShader<'h>,
    _ps: PixelShader<'h>,
    wide: [WideVertex; 3],
    narrow: [NarrowVertex; 3],
}

impl<'h> Bench<'h> {
    fn new(h: &'h Harness) -> Self {
        let wide = CORNERS.map(|(x, y)| WideVertex {
            x,
            y,
            z: 0.5,
            pad: [0; 4],
            color: 0xFF00_FF00,
        });
        let narrow = CORNERS.map(|(x, y)| NarrowVertex {
            x,
            y,
            z: 0.5,
            color: 0xFF00_FF00,
        });
        let wide_vb = h.create_vertex_buffer(3 * 32, D3DUSAGE_WRITEONLY, 0, D3DPOOL_MANAGED);
        wide_vb.lock(0, 0, 0).write(&wide);
        // A fourth vertex carries the third one's colour.
        let narrow_vb = h.create_vertex_buffer(4 * 16, D3DUSAGE_WRITEONLY, 0, D3DPOOL_MANAGED);
        narrow_vb
            .lock(0, 0, 0)
            .write(&[narrow[0], narrow[1], narrow[2], narrow[2]]);
        let decl = h.create_vertex_declaration(&crossing_color_elements());
        let vs = h.create_vertex_shader(&VS_POS_COLOR_TEXCOORD);
        let ps = h.create_pixel_shader(&PS_DIFFUSE);
        ok(h.set_vertex_declaration(&decl), "declaration");
        ok(h.set_vertex_shader(&vs), "VS");
        ok(h.set_pixel_shader(&ps), "PS");
        ok(
            h.set_render_state(D3DRS_CULLMODE, D3DCULL_NONE),
            "cull mode",
        );
        ok(h.set_render_state(D3DRS_ZENABLE, 0), "depth off");
        Self {
            h,
            wide_vb,
            narrow_vb,
            _decl: decl,
            _vs: vs,
            _ps: ps,
            wide,
            narrow,
        }
    }

    /// [`BLOCK_FRAMES`] frames of `kind`; the measured ones' draw and frame times, in ticks.
    fn block(&self, kind: &Kind) -> Vec<(u64, u64)> {
        let h = self.h;
        match kind {
            Kind::BoundFits => ok(h.set_stream_source(0, &self.wide_vb, 0, 32), "stream"),
            Kind::BoundCrossing => ok(h.set_stream_source(0, &self.narrow_vb, 0, 16), "stream"),
            Kind::UpFits | Kind::UpCrossing => {}
        }
        let mut measured = Vec::with_capacity(MEASURED_FRAMES);
        for at in 0..BLOCK_FRAMES {
            assert!(h.pump(), "WM_QUIT during the blocks");
            ok(h.begin_scene(), "BeginScene");
            let started = TscClock::now();
            for _ in 0..DRAWS {
                let hr = match kind {
                    Kind::BoundFits | Kind::BoundCrossing => {
                        h.draw_primitive(D3DPT_TRIANGLELIST, 0, 1)
                    }
                    Kind::UpFits => h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &self.wide),
                    Kind::UpCrossing => h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &self.narrow),
                };
                ok(hr, "draw");
            }
            let drawn = TscClock::now();
            ok(h.end_scene(), "EndScene");
            ok(h.present(), "Present");
            let presented = TscClock::now();
            if at >= BLOCK_FRAMES - MEASURED_FRAMES {
                measured.push((drawn - started, presented - started));
            }
        }
        measured
    }
}
