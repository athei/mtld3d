//! Follow-up reads for a GPU read-back that came back wrong.
//!
//! A test that renders, copies the result somewhere readable and reads it on
//! the CPU learns one thing from a wrong value: some stage between the pass and
//! the read lost the data. [`assert_or_reread`] asks the layer twice more
//! before it panics, so the failure names the stage. A read that is accepted
//! returns before either follow-up is built, so a passing test does no GPU
//! work it did not do before. [`assert_or_reread_and_probe`] adds a third
//! observation after those two, such as [`multisampled_depth_sample_zero`],
//! which says whether a multisampled depth attachment holds the pass.

use std::{thread, time::Duration};

use mtld3d_types::{
    D3DCLEAR_ZBUFFER, D3DFMT_A8R8G8B8, D3DFMT_INTZ, D3DFVF_DIFFUSE, D3DFVF_TEX1, D3DFVF_XYZ,
    D3DPOOL_DEFAULT, D3DPT_TRIANGLELIST, D3DRS_POINTSIZE, D3DRS_ZENABLE, D3DSAMP_ADDRESSU,
    D3DSAMP_ADDRESSV, D3DSAMP_MAGFILTER, D3DSAMP_MINFILTER, D3DTADDRESS_CLAMP, D3DTEXF_POINT,
    D3DUSAGE_DEPTHSTENCIL, D3DUSAGE_RENDERTARGET,
};

use crate::{Harness, Rgba8, TexturedVertex, resource::Texture};

/// How long the frame that carried the pass gets to finish before the plain re-read.
///
/// Nothing in D3D9 waits for one command buffer by name, and a wait that
/// trusts the queue's order is worth nothing in the one case the re-read is
/// for. The frames these tests submit run in a few milliseconds.
const SETTLE: Duration = Duration::from_millis(100);

/// The colour the depth probe clears its sampling target to; not a grey, so a missed draw shows.
const PROBE_CLEAR: u32 = 0xFF00_00FF;

/// The depth the probe's INTZ holds before the RESZ, so a RESZ that never ran reads as this.
const PROBE_PRIMER: f32 = 0.5;

/// A clip-space quad over the whole sampling target with the INTZ mapped onto it.
const PROBE_QUAD: [TexturedVertex; 6] = [
    probe_vertex(-1.0, 1.0, 0.0, 0.0),
    probe_vertex(1.0, 1.0, 1.0, 0.0),
    probe_vertex(-1.0, -1.0, 0.0, 1.0),
    probe_vertex(1.0, 1.0, 1.0, 0.0),
    probe_vertex(1.0, -1.0, 1.0, 1.0),
    probe_vertex(-1.0, -1.0, 0.0, 1.0),
];

/// One CPU observation of a GPU result, judged by the test that made it.
pub struct Reading {
    accepted: bool,
    shown: String,
}

impl Reading {
    /// What the test read, as the panic message shows it, and the test's verdict on it.
    #[must_use]
    pub const fn described(shown: String, accepted: bool) -> Self {
        Self { accepted, shown }
    }
}

/// Accept `first`, or read twice more and panic with all three readings.
///
/// `plain` reads the destination of the copy again and must not repeat the
/// copy. It has to reach the GPU: a second `GetRenderTargetData` does, since
/// every call flushes the open frame and blits the texture in a command buffer
/// it waits for, while a second `LockRect` of a texture the first lock read
/// back does not, because the first lock moved authority to the CPU staging.
/// `recopy` repeats the copy out of the same source without drawing, then
/// reads. Both run a tenth of a second after the first read at the earliest, so
/// the frame that carried the pass has had time to finish.
///
/// What the three readings say, with `A` the command buffer that carried the
/// copy and `B` the first read-back's own, committed after `A` on one queue:
///
/// - `plain` accepted: the destination holds the pass output now and did not
///   when `B` ran, so `B` ran ahead of `A`.
/// - `plain` rejected, `recopy` accepted: the destination never received the
///   pass output although the source of the copy holds it, so the first copy
///   ran ahead of the pass that feeds it, or was dropped. The two are one
///   outcome here.
/// - both rejected: the source of the copy does not hold the pass output. For
///   a `StretchRect` out of a multisampled surface that source is the
///   single-sampled resolve twin, because a repeated `StretchRect` finds no
///   pass of its own frame to hang a resolve on and copies the twin as the
///   earlier frame left it, so a lost pass and a lost resolve are one outcome.
///   For a RESZ write it is the multisampled depth attachment itself, which
///   the second transfer reads again without any draw.
///
/// # Panics
/// Panics when `first` was rejected, after both follow-up reads.
#[track_caller]
pub fn assert_or_reread(
    h: &Harness,
    context: &str,
    expected: &str,
    first: &Reading,
    plain: impl FnOnce() -> Reading,
    recopy: impl FnOnce() -> Reading,
) {
    reread_or_panic(h, context, expected, first, plain, recopy, || None);
}

/// [`assert_or_reread`], with one more observation after both follow-up reads.
///
/// `probe` runs last, after `recopy`, so it may change any device state the
/// two follow-ups read; its text is the fourth line of the panic message.
///
/// # Panics
/// Panics when `first` was rejected, after both follow-up reads and the probe.
#[track_caller]
pub fn assert_or_reread_and_probe(
    h: &Harness,
    context: &str,
    expected: &str,
    first: &Reading,
    plain: impl FnOnce() -> Reading,
    recopy: impl FnOnce() -> Reading,
    probe: impl FnOnce() -> String,
) {
    reread_or_panic(h, context, expected, first, plain, recopy, || Some(probe()));
}

/// Sample zero of the bound multisampled depth surface at one pixel, as a RESZ reads it.
///
/// For a test whose colour read-back went wrong after a depth-tested pass: it
/// says whether the depth attachment holds what the pass wrote. A fresh INTZ
/// texture is bound as depth and cleared to a primer depth, the surface goes
/// back in its place, and a RESZ (`D3DRS_POINTSIZE` set to the RESZ code with
/// the INTZ at stage 0) copies sample zero into the INTZ; the INTZ is then drawn over a
/// single-sampled target with fixed-function sampling, which returns the
/// stored depth in every colour channel, and `at` is read back. `cleared` is
/// the depth the pass cleared to and `drawn` the depth its draw wrote at
/// `at`; the text names which of them, or the primer, the eight-bit read
/// matches.
///
/// It leaves depth testing off, no depth surface bound and its own target at
/// slot 0, so it runs after every other read of the failure. A failed call
/// ends the probe with the step and its `HRESULT` instead of panicking, so the
/// readings taken before it still reach the report.
#[must_use]
pub fn multisampled_depth_sample_zero(
    h: &Harness,
    at: (u32, u32),
    cleared: f32,
    drawn: f32,
) -> String {
    match sample_zero_read(h, at) {
        Ok(pixel) => describe_depth_read(pixel, at, cleared, drawn),
        Err(step) => format!("not read: {step}"),
    }
}

/// The body of [`assert_or_reread`] and [`assert_or_reread_and_probe`].
#[track_caller]
fn reread_or_panic(
    h: &Harness,
    context: &str,
    expected: &str,
    first: &Reading,
    plain: impl FnOnce() -> Reading,
    recopy: impl FnOnce() -> Reading,
    probe: impl FnOnce() -> Option<String>,
) {
    if first.accepted {
        return;
    }
    thread::sleep(SETTLE);
    let plain = plain();
    let recopy = recopy();
    let probe = probe().map_or_else(String::new, |text| {
        format!("\n  depth probe:         {text}")
    });
    let stage = match (plain.accepted, recopy.accepted) {
        (true, true) => {
            "the destination is right when read again with no new copy, so the first read-back \
             ran ahead of the command buffer that carried the copy"
        }
        (true, false) => {
            "the destination is right when read again with no new copy, so the first read-back \
             ran ahead of the command buffer that carried the copy; the repeated copy then \
             delivered a wrong value, which is a second fault"
        }
        (false, true) => {
            "the destination stayed wrong and a repeated copy is right, so the source holds the \
             pass output and the first copy ran ahead of the pass or was dropped"
        }
        (false, false) => {
            "wrong throughout, so the source of the copy does not hold the pass output: the \
             pass, or the resolve that belongs to it, was lost"
        }
    };
    panic!(
        "{context}\n  expected:            {expected}\n  first read:          {}\n  \
         re-read, no copy:    {}\n  after a second copy: {}\n  reading: {stage}\n  device: {}{probe}",
        first.shown,
        plain.shown,
        recopy.shown,
        h.adapter_description(),
    );
}

/// One corner of [`PROBE_QUAD`].
const fn probe_vertex(x: f32, y: f32, u: f32, v: f32) -> TexturedVertex {
    TexturedVertex {
        x,
        y,
        z: 0.5,
        color: 0xFFFF_FFFF,
        u,
        v,
    }
}

/// RESZ the bound depth surface into a fresh INTZ texture and read it back at `at`.
///
/// The error names the step that failed, with its `HRESULT` where it has one.
fn sample_zero_read(h: &Harness, at: (u32, u32)) -> Result<u32, String> {
    let Some(depth) = h.depth_stencil_surface() else {
        return Err("no depth surface is bound".to_owned());
    };
    let (hr, desc) = depth.desc();
    succeeded(hr, "GetDesc on the depth surface")?;
    let size = (desc.width, desc.height);
    let intz = probe_texture(h, size, D3DUSAGE_DEPTHSTENCIL, D3DFMT_INTZ)?;
    let target = probe_texture(h, size, D3DUSAGE_RENDERTARGET, D3DFMT_A8R8G8B8)?;
    // Prime the INTZ, so a RESZ that never reaches it reads as the primer.
    let colour = h.render_target(0);
    succeeded(
        h.set_render_target(0, &target.surface_level(0)),
        "bind the sampling target to prime the INTZ",
    )?;
    succeeded(
        h.set_depth_stencil_surface(&intz.surface_level(0)),
        "bind the INTZ as depth",
    )?;
    succeeded(
        h.clear(D3DCLEAR_ZBUFFER, 0, PROBE_PRIMER, 0),
        "clear the INTZ to the primer",
    )?;
    succeeded(h.set_render_target(0, &colour), "rebind the render target")?;
    // Priming bound the INTZ as depth; the surface under test goes back.
    succeeded(
        h.set_depth_stencil_surface(&depth),
        "rebind the depth surface",
    )?;
    drop(colour);
    drop(depth);
    succeeded(h.set_texture(0, &intz), "bind the INTZ at stage 0")?;
    succeeded(
        h.set_render_state(D3DRS_POINTSIZE, 0x7fa0_5000),
        "the RESZ write",
    )?;
    succeeded(
        h.set_render_target(0, &target.surface_level(0)),
        "bind the sampling target",
    )?;
    succeeded(h.clear_depth_stencil_surface(), "unbind the depth surface")?;
    succeeded(h.set_render_state(D3DRS_ZENABLE, 0), "depth test off")?;
    h.select_texture_stage(0);
    for (state, value) in [
        (D3DSAMP_MINFILTER, D3DTEXF_POINT),
        (D3DSAMP_MAGFILTER, D3DTEXF_POINT),
        (D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP),
        (D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP),
    ] {
        succeeded(h.set_sampler_state(0, state, value), "SetSamplerState")?;
    }
    succeeded(
        h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1),
        "SetFVF",
    )?;
    succeeded(h.begin_scene(), "BeginScene")?;
    let cleared = h.clear_target(PROBE_CLEAR);
    let drawn = h.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &PROBE_QUAD);
    let ended = h.end_scene();
    succeeded(cleared, "clear the sampling target")?;
    succeeded(drawn, "draw the INTZ")?;
    succeeded(ended, "EndScene")?;
    Ok(h.read_pixel(at.0, at.1))
}

/// A one-level `D3DPOOL_DEFAULT` texture for the depth probe.
fn probe_texture(
    h: &Harness,
    (width, height): (u32, u32),
    usage: u32,
    format: u32,
) -> Result<Texture<'_>, String> {
    let (hr, ptr) = h.try_create_texture(width, height, 1, usage, format, D3DPOOL_DEFAULT);
    succeeded(hr, "CreateTexture")?;
    if ptr.is_null() {
        return Err(format!(
            "CreateTexture({width}x{height}, format {format:#x}) returned null"
        ));
    }
    Ok(Texture::from_raw(ptr))
}

/// `Ok` for `D3D_OK`, otherwise the step and its `HRESULT`.
fn succeeded(hr: i32, step: &str) -> Result<(), String> {
    if hr == 0 {
        Ok(())
    } else {
        Err(format!("{step} failed: {hr:#010x}"))
    }
}

/// Name which depth the eight-bit INTZ read at `at` matches.
fn describe_depth_read(pixel: u32, at: (u32, u32), cleared: f32, drawn: f32) -> String {
    const TOLERANCE: f32 = 2.0 / 255.0;
    let read = Rgba8::from_pixel(pixel);
    let (x, y) = at;
    if read.r.abs_diff(read.g) > 2 || read.r.abs_diff(read.b) > 2 {
        return format!(
            "the INTZ draw did not land at {x},{y}: the sampling target reads {read:?}, not a grey"
        );
    }
    let depth = f32::from(read.r) / 255.0;
    let verdict = if (depth - PROBE_PRIMER).abs() <= TOLERANCE {
        format!(
            "the primer's {PROBE_PRIMER}: the RESZ did not reach the INTZ, so this says nothing"
        )
    } else if (depth - drawn).abs() <= TOLERANCE {
        format!(
            "the draw's {drawn}: the depth attachment holds the pass, so the colour side lost it"
        )
    } else if (depth - cleared).abs() <= TOLERANCE {
        format!(
            "the clear's {cleared} without the draw's {drawn}: the clear reached the depth \
             attachment and the draw's depth write did not"
        )
    } else {
        format!(
            "neither the clear's {cleared} nor the draw's {drawn}: the depth attachment does not \
             hold the pass, its clear included"
        )
    };
    format!(
        "sample zero at {x},{y} reads {} of 255 (depth {depth:.3}), {verdict}",
        read.r
    )
}
