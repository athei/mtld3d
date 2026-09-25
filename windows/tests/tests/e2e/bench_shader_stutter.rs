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
//! `K` and the offscreen shader are fixed per test; `make bench
//! FILTER=<test name>` picks one.

use std::time::{SystemTime, UNIX_EPOCH};

use mtld3d_tests::{Harness, HarnessConfig, PixelShader};
use mtld3d_types::{
    D3DCLEAR_TARGET, D3DCLEAR_ZBUFFER, D3DFMT_A8R8G8B8, D3DFMT_D24S8, D3DFMT_INDEX16,
    D3DPOOL_DEFAULT, D3DPOOL_MANAGED, D3DPRESENT_INTERVAL_IMMEDIATE, D3DPT_TRIANGLELIST,
    D3DRS_ZENABLE, D3DUSAGE_RENDERTARGET, D3DUSAGE_WRITEONLY,
};

use crate::bench::{
    FrameClock, IDENTITY_ROWS, LayerLog, Model, STRIDE, TEXTURED_DECL, grid, material_ps,
    material_vs, ok, pattern_texture, ratio, world_rows, write_report,
};

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
/// Edge of the offscreen target the extra shader draws into.
const OFFSCREEN_EDGE: u32 = 512;
/// Draws of the base program each frame, so a frame without a new shader is not empty.
const BASE_DRAWS: u32 = 50;
const WARM_UP_FRAMES: u32 = 30;
const MEASURED_FRAMES: u32 = 200;

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
    let back_buffer = h.back_buffer(0);
    let target_texture = h.create_texture(
        OFFSCREEN_EDGE,
        OFFSCREEN_EDGE,
        1,
        D3DUSAGE_RENDERTARGET,
        D3DFMT_A8R8G8B8,
        D3DPOOL_DEFAULT,
    );
    let target = target_texture.surface_level(0);
    let texture = pattern_texture(&h, 0xFF40_8020);
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
    let decl = h.create_vertex_declaration(&TEXTURED_DECL);
    let vs = h.create_vertex_shader(&material_vs(&Model::Sm2, 0.0));
    let base_ps = h.create_pixel_shader(&material_ps(&Model::Sm2, [0.0; 4], false));

    ok(h.set_vertex_declaration(&decl), "declaration");
    ok(h.set_stream_source(0, &vb, 0, STRIDE), "stream");
    ok(h.set_indices(&ib), "indices");
    ok(h.set_texture(0, &texture), "texture");
    ok(h.set_vertex_shader(&vs), "VS");
    ok(
        h.set_vertex_shader_constant_f(0, &IDENTITY_ROWS),
        "view-projection",
    );
    ok(h.set_pixel_shader_constant_f(0, &[1.0; 4]), "tint");
    ok(h.set_render_state(D3DRS_ZENABLE, 0), "depth off");
    ok(h.set_render_target(0, &target), "offscreen target");
    ok(
        h.clear(D3DCLEAR_TARGET, 0xFF00_0000, 1.0, 0),
        "offscreen clear, once",
    );
    ok(h.set_render_target(0, &back_buffer), "back buffer");

    let draw = |at: u32| {
        let x = ratio(at % 20, 20).mul_add(1.8, -0.95);
        let y = ratio(at / 20 % 20, 20).mul_add(1.8, -0.95);
        ok(
            h.set_vertex_shader_constant_f(4, &world_rows(0.1, x, y, 0.5)),
            "world",
        );
        ok(
            h.draw_indexed_primitive(D3DPT_TRIANGLELIST, 0, 0, vertex_count, 0, triangles),
            "draw",
        );
    };
    let base_frame = || {
        ok(h.begin_scene(), "BeginScene");
        ok(
            h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, 0xFF20_3040, 1.0, 0),
            "clear",
        );
        ok(h.set_pixel_shader(&base_ps), "base PS");
        for at in 0..BASE_DRAWS {
            draw(at);
        }
    };
    for _ in 0..WARM_UP_FRAMES {
        assert!(h.pump(), "WM_QUIT during warm-up");
        base_frame();
        ok(h.end_scene(), "EndScene");
        ok(h.present(), "Present");
    }

    let salt = salt();
    // Kept alive to the end, as a game keeps what it has loaded.
    let mut created: Vec<PixelShader<'_>> = Vec::new();
    let log = LayerLog::find();
    let from = log.mark();
    let mut clock = FrameClock::start(usize::try_from(MEASURED_FRAMES).expect("fits usize"));
    let mut count = 0_u16;
    for frame in 0..MEASURED_FRAMES {
        assert!(h.pump(), "WM_QUIT during the measured frames");
        base_frame();
        for at in 0..per_frame {
            let ps = h.create_pixel_shader(&unique_ps(&mut count, salt));
            ok(h.set_pixel_shader(&ps), "new PS");
            draw(BASE_DRAWS + at + frame % 7);
            created.push(ps);
        }
        if offscreen {
            let ps = h.create_pixel_shader(&unique_ps(&mut count, salt));
            ok(h.set_render_target(0, &target), "offscreen target");
            ok(h.set_pixel_shader(&ps), "offscreen PS");
            draw(frame % 40);
            ok(h.set_render_target(0, &back_buffer), "back buffer");
            created.push(ps);
        }
        ok(h.end_scene(), "EndScene");
        clock.present(&h);
    }
    let to = log.mark();

    let stats = clock.stats();
    let spikes = clock.over(stats.p50 * 2);
    let body = format!(
        "shape: back buffer {WIDTH}x{HEIGHT}, {BASE_DRAWS} base draws per frame, \
         shaderCache.enable=false\n\
         new pixel shaders per frame: {per_frame} on the back buffer{offscreen}\n\
         warm-up: {WARM_UP_FRAMES} frames without new shaders\n\
         measured: {frames} frames in {elapsed:.2?}, {shaders} new shaders\n\
         frame time (Present to Present): {row}\n\
         API work (Present return to Present call): {work}\n\
         frames over 2x the median ({limit:.3} ms): {spikes}\n{perf}",
        offscreen = if offscreen {
            " + 1 into an offscreen target not cleared that frame"
        } else {
            ""
        },
        frames = stats.frames,
        shaders = created.len(),
        elapsed = clock.elapsed(),
        row = stats.row(),
        work = clock.work_stats().row(),
        limit = (stats.p50 * 2).as_secs_f64() * 1e3,
        perf = log.perf_rows(from, to).section(),
    );
    write_report(name, &log, &body);
}

/// The next never-seen pixel shader: its `def c7` is the running `count` and the run's `salt`.
fn unique_ps(count: &mut u16, salt: u16) -> Vec<u32> {
    *count += 1;
    let tint = [
        f32::from(*count) * 1.0e-4,
        f32::from(salt) * 1.0e-6,
        0.0,
        0.0,
    ];
    material_ps(&Model::Sm2, tint, false)
}

/// A per-run value that keeps this run's shaders unlike any earlier run's.
fn salt() -> u16 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.subsec_nanos());
    u16::try_from(nanos % 65_521).expect("a remainder below 65521 fits u16")
}
