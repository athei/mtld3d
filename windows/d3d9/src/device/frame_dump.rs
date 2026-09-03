//! Per-draw D3D9 state dump for a short run of frames, armed by F12 (see `crate::capture`).
//!
//! The silent-write audit reports the states a game sets that we never
//! read; it cannot tell whether a consumed state produced the pass the game
//! meant. This dump lists, for [`FrameDump::FRAMES`] consecutive frames,
//! every render-target and depth-stencil bind, viewport and scissor change,
//! clear, copy, query and draw with the states that decide pass shape
//! (depth, stencil, blend, cull, colour mask, alpha test, bias), the bound
//! shaders and the bound textures. It logs at info level, so a play session
//! needs no environment change: press F12, read the log.
//!
//! The same press captures the same frames into a Metal GPU trace, which
//! holds everything the dump deliberately leaves out (pipeline state,
//! bindings, buffer and texture bytes, shader source, load/store actions).
//! The two name each other: the frame end line carries the label of the
//! frame's command buffer, and every dumped draw sits in a `draw N` debug
//! group in the trace.

use core::ffi::c_void;

use log::info;
use mtld3d_types::{
    D3DRS_ALPHABLENDENABLE, D3DRS_ALPHAFUNC, D3DRS_ALPHAREF, D3DRS_ALPHATESTENABLE, D3DRS_BLENDOP,
    D3DRS_BLENDOPALPHA, D3DRS_CCW_STENCILFAIL, D3DRS_CCW_STENCILFUNC, D3DRS_CCW_STENCILPASS,
    D3DRS_CCW_STENCILZFAIL, D3DRS_COLORWRITEENABLE, D3DRS_COLORWRITEENABLE1,
    D3DRS_COLORWRITEENABLE2, D3DRS_COLORWRITEENABLE3, D3DRS_CULLMODE, D3DRS_DEPTHBIAS,
    D3DRS_DESTBLEND, D3DRS_DESTBLENDALPHA, D3DRS_SCISSORTESTENABLE, D3DRS_SEPARATEALPHABLENDENABLE,
    D3DRS_SLOPESCALEDEPTHBIAS, D3DRS_SRCBLEND, D3DRS_SRCBLENDALPHA, D3DRS_STENCILENABLE,
    D3DRS_STENCILFAIL, D3DRS_STENCILFUNC, D3DRS_STENCILMASK, D3DRS_STENCILPASS, D3DRS_STENCILREF,
    D3DRS_STENCILWRITEMASK, D3DRS_STENCILZFAIL, D3DRS_TWOSIDEDSTENCILMODE, D3DRS_ZENABLE,
    D3DRS_ZFUNC, D3DRS_ZWRITEENABLE,
};

use super::{DepthBinding, DeviceInner, RtBinding};
use crate::{
    LOG_TARGET,
    draw::{PsSource, VsSource},
    encoder::FrameDataFlags,
};

/// Label a surface pointer by its backing texture for a dump event.
///
/// Dump events name surfaces by texture id so they cross-reference the
/// per-draw lines, which print the same ids; the raw COM pointer alone
/// cannot be matched to anything.
pub fn surface_label(surface: *mut c_void) -> String {
    if surface.is_null() {
        return String::from("null");
    }
    let surf = surface.cast::<crate::surface::Direct3DSurface9>();
    // SAFETY: callers pass a live IDirect3DSurface9 the game handed to this
    // API call; the dump reads it on the API thread that owns it.
    let surf = unsafe { &*surf };
    let parent = surf.parent_texture();
    if parent.is_null() {
        return format!(
            "standalone {:#x} {}x{}",
            surf.standalone_format(),
            surf.standalone_width(),
            surf.standalone_height()
        );
    }
    // SAFETY: a surface keeps a reference on its parent texture for its
    // whole lifetime, so the parent is live while the surface is.
    let tex = unsafe { &*parent };
    format!(
        "{:?}/{:#x} l{}",
        tex.texture_id(),
        tex.d3d_format(),
        surf.mip_level()
    )
}

/// Whether the dump is running and how many draws it has listed.
pub struct FrameDump {
    /// The current frame is being dumped.
    pub active: bool,
    /// Draws listed so far in this frame.
    pub draws: u32,
    /// Frames still to dump after the current one.
    ///
    /// A press captures a short run of consecutive frames rather than a
    /// single one, so cross-frame resource flow (a depth texture written in
    /// frame N and consumed in frame N+1) is visible in one capture.
    pub frames_remaining: u32,
}

impl FrameDump {
    pub const IDLE: Self = Self {
        active: false,
        draws: 0,
        frames_remaining: 0,
    };

    /// Consecutive frames one F12 press dumps and captures.
    pub const FRAMES: u32 = 3;
}

impl DeviceInner {
    /// Whether a dump is currently running.
    ///
    /// For callers outside the device module that want to skip building an
    /// event string when no dump is armed.
    #[must_use]
    pub const fn frame_dump_active(&self) -> bool {
        self.frame_dump.active
    }

    /// Close the dumped frame at `Present` and arm the next one if F12 asked.
    ///
    /// `seq` is the `submit_seq` of the frame just sent; the unix side labels
    /// its command buffer `mtld3d-frame-{seq:#x}`, so the frame end line
    /// names the node a GPU trace shows it under. A mid-frame flush inside a
    /// dumped frame submits its own command buffer with its own seq; only
    /// the closing one is named here.
    ///
    /// A press starts a [`FrameDump::FRAMES`]-frame run; a run already going
    /// continues until its remaining-frame count is spent. The frame armed
    /// here is the one about to be built, so it also receives the GPU
    /// capture marks of its position in the run: the first frame starts the
    /// capture, the last one stops it, and the trace covers exactly the
    /// dumped frames.
    pub fn frame_dump_present(&mut self, arm_next: bool, seq: u64) {
        let carried = if self.frame_dump.active {
            info!(
                target: LOG_TARGET,
                "[dump] frame end: {} draws, command buffer mtld3d-frame-{seq:#x}",
                self.frame_dump.draws
            );
            self.frame_dump.frames_remaining
        } else {
            0
        };
        let remaining = if arm_next { FrameDump::FRAMES } else { carried };
        self.frame_dump = FrameDump {
            active: remaining > 0,
            draws: 0,
            frames_remaining: remaining.saturating_sub(1),
        };
        if self.frame_dump.active {
            let index = FrameDump::FRAMES - remaining + 1;
            info!(
                target: LOG_TARGET,
                "[dump] frame start ({index} of {})",
                FrameDump::FRAMES
            );
            let (start, stop) = mtld3d_core::present::capture_marks(index, FrameDump::FRAMES);
            let mut marks = FrameDataFlags::empty();
            if start {
                marks.insert(FrameDataFlags::GPU_CAPTURE_START);
            }
            if stop {
                marks.insert(FrameDataFlags::GPU_CAPTURE_STOP);
            }
            self.current_frame.mark_gpu_capture(marks);
        }
    }
}

impl DeviceInner {
    /// Log one bind or clear event while a dump is running.
    pub fn frame_dump_event(&self, what: &str) {
        if self.frame_dump.active {
            info!(target: LOG_TARGET, "[dump] {what}");
        }
    }

    /// Labels for the bound colour RT0 and depth-stencil, for dump lines.
    #[must_use]
    pub fn frame_dump_target_labels(&self) -> (String, String) {
        let rt = match &self.last_color_rt_binding {
            Some((RtBinding::Backbuffer { width, height, .. }, _)) => {
                format!("backbuffer {width}x{height}")
            }
            Some((
                RtBinding::StandaloneColor {
                    format,
                    width,
                    height,
                    ..
                },
                _,
            )) => format!("surface {format:?} {width}x{height}"),
            Some((
                RtBinding::Texture {
                    info,
                    width,
                    height,
                    slice,
                    level,
                    ..
                },
                _,
            )) => format!(
                "texture {:?} {:?} {width}x{height} slice={slice} level={level}",
                info.texture_id, info.pixel_format
            ),
            None => String::from("none"),
        };
        let ds = match &self.last_depth_binding {
            Some((DepthBinding::Lazy(info, level), ..)) => {
                format!("{:?} level={level}", info.texture_id)
            }
            Some((DepthBinding::Eager(..), ..)) => String::from("default"),
            Some((DepthBinding::None, ..)) | None => String::from("none"),
        };
        (rt, ds)
    }

    /// Log the assembled draw state; called once per draw while a dump runs.
    ///
    /// Also tags the draw for the encoder, which wraps its Metal draw in a
    /// `draw {seq}` debug group; the closure op lands ahead of the draw op
    /// in the same op stream, so the tag reaches `emit_draw` with the draw.
    pub fn frame_dump_draw(&mut self) {
        let seq = self.frame_dump.draws;
        self.frame_dump.draws += 1;
        self.push_op(Box::new(move |enc| enc.set_dump_draw(seq)));
        let vp = self.viewport();
        let rs = |i: u32| self.render_state(i as usize);

        let (rt, ds) = self.frame_dump_target_labels();
        let vs = self.snapshot_cache.vs.map_or_else(
            || String::from("none"),
            |p| match p.as_ref() {
                VsSource::Programmable { vs_id, .. } => format!("{vs_id:?}"),
                VsSource::FixedFunction { .. } => String::from("ff"),
            },
        );
        let ps = self.snapshot_cache.ps.map_or_else(
            || String::from("none"),
            |p| match p.as_ref() {
                PsSource::Programmable { ps_id, .. } => format!("{ps_id:?}"),
                PsSource::FixedFunction { .. } => String::from("ff"),
            },
        );

        let bindings = self.stage_bindings();
        let bound = bindings.bound_mask();
        let depth = bindings.depth_sampler_mask();
        let fetch = bindings.depth_fetch_mask();
        let mut textures = String::new();
        for stage in 0..16usize {
            if bound & (1u16 << stage) == 0 {
                continue;
            }
            let tex = bindings.texture(stage);
            if tex.is_null() {
                continue;
            }
            // SAFETY: a bound stage holds a live texture reference until it is
            // rebound or the texture is released, both on this thread.
            let tex = unsafe { &*tex };
            let kind = if fetch & (1u16 << stage) != 0 {
                "F"
            } else if depth & (1u16 << stage) != 0 {
                "D"
            } else {
                ""
            };
            let inner = tex.inner();
            let _ = std::fmt::Write::write_fmt(
                &mut textures,
                format_args!(
                    " s{stage}={:?}/{:#x}/{}x{}{kind}",
                    tex.texture_id(),
                    tex.d3d_format(),
                    inner.mip_width(0),
                    inner.mip_height(0)
                ),
            );
        }
        // Vertex fetch slots (`D3DVERTEXTEXTURESAMPLER0..3`): bound through
        // `SetTexture(257..)`, consumed by `vs_3_0` `texldl`, and invisible in
        // the fragment-stage list above.
        for slot in 0..4usize {
            let tex = self.vertex_texture(slot);
            if tex.is_null() {
                continue;
            }
            // SAFETY: a bound vertex slot holds a live texture reference until
            // it is rebound or the texture is released, both on this thread.
            let tex = unsafe { &*tex };
            let inner = tex.inner();
            let _ = std::fmt::Write::write_fmt(
                &mut textures,
                format_args!(
                    " vt{slot}={:?}/{:#x}/{}x{}",
                    tex.texture_id(),
                    tex.d3d_format(),
                    inner.mip_width(0),
                    inner.mip_height(0)
                ),
            );
        }

        info!(
            target: LOG_TARGET,
            "[dump] draw {seq}: rt={rt} ds={ds}/{:#x} vs={vs} ps={ps} \
             z=[{},{},{}] blend=[{},{},{},{} sep={} {},{},{}] cull={} cw=[{:#x},{:#x},{:#x},{:#x}] \
             alpha=[{},{},{}] stencil=[{},{},{:#x},{:#x},{:#x} {},{},{} two={} ccw={},{},{},{}] \
             bias=[{:#x},{:#x}] vp={},{}+{}x{} scissor={} tex=[{}]",
            self.snapshot_cache.depth_stencil.bits(),
            rs(D3DRS_ZENABLE),
            rs(D3DRS_ZWRITEENABLE),
            rs(D3DRS_ZFUNC),
            rs(D3DRS_ALPHABLENDENABLE),
            rs(D3DRS_SRCBLEND),
            rs(D3DRS_DESTBLEND),
            rs(D3DRS_BLENDOP),
            rs(D3DRS_SEPARATEALPHABLENDENABLE),
            rs(D3DRS_SRCBLENDALPHA),
            rs(D3DRS_DESTBLENDALPHA),
            rs(D3DRS_BLENDOPALPHA),
            rs(D3DRS_CULLMODE),
            rs(D3DRS_COLORWRITEENABLE),
            rs(D3DRS_COLORWRITEENABLE1),
            rs(D3DRS_COLORWRITEENABLE2),
            rs(D3DRS_COLORWRITEENABLE3),
            rs(D3DRS_ALPHATESTENABLE),
            rs(D3DRS_ALPHAFUNC),
            rs(D3DRS_ALPHAREF),
            rs(D3DRS_STENCILENABLE),
            rs(D3DRS_STENCILFUNC),
            rs(D3DRS_STENCILREF),
            rs(D3DRS_STENCILMASK),
            rs(D3DRS_STENCILWRITEMASK),
            rs(D3DRS_STENCILFAIL),
            rs(D3DRS_STENCILZFAIL),
            rs(D3DRS_STENCILPASS),
            rs(D3DRS_TWOSIDEDSTENCILMODE),
            rs(D3DRS_CCW_STENCILFUNC),
            rs(D3DRS_CCW_STENCILFAIL),
            rs(D3DRS_CCW_STENCILZFAIL),
            rs(D3DRS_CCW_STENCILPASS),
            rs(D3DRS_DEPTHBIAS),
            rs(D3DRS_SLOPESCALEDEPTHBIAS),
            vp.x,
            vp.y,
            vp.width,
            vp.height,
            rs(D3DRS_SCISSORTESTENABLE),
            textures.trim_start(),
        );
        // Draws that fetch raw depth reconstruct positions from shared PS
        // constants (screen scale, linearization); print the window those
        // shaders read so wrong uploads are visible next to the draw.
        if fetch != 0 {
            let c = self.shader_bindings().ps_constants_copy();
            info!(
                target: LOG_TARGET,
                "[dump] draw {seq} psc: c66={:?} c72={:?} c73={:?} c77={:?} c78={:?}",
                c[66],
                c[72],
                c[73],
                c[77],
                c[78]
            );
        }
    }
}
