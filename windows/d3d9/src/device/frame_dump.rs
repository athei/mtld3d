//! One-frame per-draw state dump, armed by Ctrl+Shift+D (see `crate::capture`).
//!
//! The silent-write audit reports the states a game sets that we never
//! read; it cannot tell whether a consumed state produced the pass the game
//! meant. This dump lists, for exactly one frame, every render-target and
//! depth-stencil bind, every clear and every draw with the states that
//! decide pass shape (depth, stencil, blend, cull, colour mask, alpha test,
//! bias), the bound shaders and the bound textures. It logs at info level,
//! so a play session needs no environment change: press Ctrl+Shift+D, read the log.

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
};

/// Whether the dump is running and how many draws it has listed.
pub struct FrameDump {
    /// The current frame is being dumped.
    pub active: bool,
    /// Draws listed so far in this frame.
    pub draws: u32,
}

impl FrameDump {
    pub const IDLE: Self = Self {
        active: false,
        draws: 0,
    };
}

impl DeviceInner {
    /// Close the dumped frame at `Present` and arm the next one if the chord asked.
    pub fn frame_dump_present(&mut self, arm_next: bool) {
        if self.frame_dump.active {
            info!(
                target: LOG_TARGET,
                "[dump] frame end: {} draws",
                self.frame_dump.draws
            );
        }
        self.frame_dump = FrameDump {
            active: arm_next,
            draws: 0,
        };
        if arm_next {
            info!(target: LOG_TARGET, "[dump] frame start (Ctrl+Shift+D)");
        }
    }

    /// Log one bind or clear event while a dump is running.
    pub fn frame_dump_event(&self, what: &str) {
        if self.frame_dump.active {
            info!(target: LOG_TARGET, "[dump] {what}");
        }
    }

    /// Log the assembled draw state; called once per draw while a dump runs.
    pub fn frame_dump_draw(&mut self) {
        let seq = self.frame_dump.draws;
        self.frame_dump.draws += 1;
        let rs = |i: u32| self.render_state(i as usize);

        let rt = match &self.last_color_rt_binding {
            Some(RtBinding::Backbuffer { width, height, .. }) => {
                format!("backbuffer {width}x{height}")
            }
            Some(RtBinding::StandaloneColor {
                format,
                width,
                height,
                ..
            }) => format!("surface {format:?} {width}x{height}"),
            Some(RtBinding::Texture {
                info,
                width,
                height,
                slice,
                level,
                ..
            }) => format!(
                "texture {:?} {:?} {width}x{height} slice={slice} level={level}",
                info.texture_id, info.pixel_format
            ),
            None => String::from("none"),
        };
        let ds = match &self.last_depth_binding {
            Some((DepthBinding::Lazy(info, level), _, _)) => {
                format!("{:?} level={level}", info.texture_id)
            }
            Some((DepthBinding::Eager(_), _, _)) => String::from("default"),
            Some((DepthBinding::None, _, _)) | None => String::from("none"),
        };
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

        info!(
            target: LOG_TARGET,
            "[dump] draw {seq}: rt={rt} ds={ds}/{:#x} vs={vs} ps={ps} \
             z=[{},{},{}] blend=[{},{},{},{} sep={} {},{},{}] cull={} cw=[{:#x},{:#x},{:#x},{:#x}] \
             alpha=[{},{},{}] stencil=[{},{},{:#x},{:#x},{:#x} {},{},{} two={} ccw={},{},{},{}] \
             bias=[{:#x},{:#x}] scissor={} tex=[{}]",
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
            rs(D3DRS_SCISSORTESTENABLE),
            textures.trim_start(),
        );
    }
}
