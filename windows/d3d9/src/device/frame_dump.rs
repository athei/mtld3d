//! One-frame per-draw state dump, armed by Ctrl+Shift+D (see `crate::capture`).
//!
//! The silent-write audit reports the states a game sets that we never
//! read; it cannot tell whether a consumed state produced the pass the game
//! meant. This dump lists, for exactly one frame, every render-target and
//! depth-stencil bind, every clear and every draw with the states that
//! decide pass shape (depth, stencil, blend, cull, colour mask, alpha test,
//! bias), the bound shaders and the bound textures. It logs at info level,
//! so a play session needs no environment change: press Ctrl+Shift+D, read the log.

use core::ffi::c_void;

use log::info;
use mtld3d_shared::MetalHandle;
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

/// One texture the dump reads back and summarizes at frame end.
pub struct DumpReadback {
    id: mtld3d_core::ids::TextureId,
    d3d_format: u32,
    width: u32,
    height: u32,
}

/// Whether the dump is running and how many draws it has listed.
pub struct FrameDump {
    /// The current frame is being dumped.
    pub active: bool,
    /// Draws listed so far in this frame.
    pub draws: u32,
    /// Frames still to dump after the current one.
    ///
    /// The chord captures a short run of consecutive frames rather than a
    /// single one, so cross-frame resource flow (a depth texture written in
    /// frame N and consumed in frame N+1) is visible in one capture.
    pub frames_remaining: u32,
    /// Textures to read back and summarize at frame end.
    ///
    /// Every depth texture a draw sampled and every extra render target
    /// bound above slot 0: their contents decide occlusion-style effects,
    /// and only a readback can say whether the bytes are sane.
    pub readback: Vec<DumpReadback>,
}

impl FrameDump {
    pub const IDLE: Self = Self {
        active: false,
        draws: 0,
        frames_remaining: 0,
        readback: Vec::new(),
    };

    /// Consecutive frames one Ctrl+Shift+D press captures.
    pub const FRAMES: u32 = 3;
}

impl DeviceInner {
    /// Whether a one-frame dump is currently running.
    ///
    /// For callers outside the device module that want to skip building an
    /// event string when no dump is armed.
    #[must_use]
    pub const fn frame_dump_active(&self) -> bool {
        self.frame_dump.active
    }

    /// Close the dumped frame at `Present` and arm the next one if the chord asked.
    ///
    /// A chord press starts a [`FrameDump::FRAMES`]-frame capture; a capture
    /// already running continues until its remaining-frame count is spent.
    pub fn frame_dump_present(&mut self, arm_next: bool) {
        let carried = if self.frame_dump.active {
            self.frame_dump_run_readbacks();
            info!(
                target: LOG_TARGET,
                "[dump] frame end: {} draws",
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
            readback: Vec::new(),
        };
        if self.frame_dump.active {
            info!(
                target: LOG_TARGET,
                "[dump] frame start ({} of {})",
                FrameDump::FRAMES - remaining + 1,
                FrameDump::FRAMES
            );
        }
    }

    /// Queue a texture for the frame-end readback while a dump is running.
    ///
    /// Deduplicates by id; anything past a small cap is dropped so a frame
    /// binding many depth textures cannot stall the present for long.
    pub fn frame_dump_note_readback(&mut self, tex: &crate::texture::Direct3DTexture9) {
        const CAP: usize = 14;
        if !self.frame_dump.active {
            return;
        }
        let id = tex.texture_id();
        if self.frame_dump.readback.len() >= CAP
            || self.frame_dump.readback.iter().any(|r| r.id == id)
        {
            return;
        }
        let inner = tex.inner();
        self.frame_dump.readback.push(DumpReadback {
            id,
            d3d_format: tex.d3d_format(),
            width: inner.mip_width(0),
            height: inner.mip_height(0),
        });
    }

    /// Read back and summarize every noted texture; called at frame end.
    ///
    /// Each texture is resolved to its Metal handle through a frame op, the
    /// frame is flushed to the GPU, and the pixels are copied to a host
    /// buffer. Four-byte formats only: the depth formats and R32F log float
    /// statistics (garbage shows as absurd ranges or NaN counts), everything
    /// else logs a few raw pixels.
    fn frame_dump_run_readbacks(&mut self) {
        use core::sync::atomic::{AtomicU64, Ordering};
        use std::sync::Arc;

        let targets = std::mem::take(&mut self.frame_dump.readback);
        for t in &targets {
            let label = format!("{:?}/{:#x} {}x{}", t.id, t.d3d_format, t.width, t.height);
            // The depth fourccs read back as their Depth32Float substrate;
            // `bytes_per_pixel` calls them 0 because they have no lock pitch.
            let is_depth = matches!(
                t.d3d_format,
                mtld3d_types::D3DFMT_INTZ | mtld3d_types::D3DFMT_DF24 | mtld3d_types::D3DFMT_DF16
            );
            let bpp = if is_depth {
                4
            } else {
                mtld3d_core::format::map_d3d_format(t.d3d_format).map_or(0, |m| m.bytes_per_pixel())
            };
            if !matches!(bpp, 2 | 4) || t.width == 0 || t.height == 0 {
                info!(target: LOG_TARGET, "[dump] readback {label}: skipped ({bpp} bytes/px)");
                continue;
            }
            let slot = Arc::new(AtomicU64::new(0));
            let slot_op = Arc::clone(&slot);
            let snap_slot = Arc::new(AtomicU64::new(0));
            let snap_slot_op = Arc::clone(&snap_slot);
            let id = t.id;
            self.push_op(Box::new(move |enc| {
                let h = enc.get_texture_handle_by_id(id);
                if h != 0 {
                    // SAFETY: `h` is a live retained MTLTexture handle from
                    // the encoder texture cache.
                    enc.note_color_read_back(unsafe { MetalHandle::new(h) });
                    // A depth attachment sampled by draws is read through a
                    // snapshot copy; its contents are what those draws saw.
                    snap_slot_op.store(enc.depth_snapshot_handle(h), Ordering::Release);
                }
                slot_op.store(h, Ordering::Release);
            }));
            self.flush_current_frame_blocking();
            for (h, what) in [
                (slot.load(Ordering::Acquire), ""),
                (snap_slot.load(Ordering::Acquire), " snapshot"),
            ] {
                if h == 0 {
                    if what.is_empty() {
                        info!(target: LOG_TARGET, "[dump] readback {label}: no Metal texture");
                    }
                    continue;
                }
                let bytes_per_row = t.width * bpp;
                let len = bytes_per_row as usize * t.height as usize;
                let mut buf = vec![0u8; len];
                let hr = super::blit_handle_to_systemmem(
                    self,
                    // SAFETY: `h` is non-zero (checked above) and a live retained
                    // MTLTexture handle from the encoder texture cache.
                    unsafe { MetalHandle::new(h) },
                    buf.as_mut_ptr() as u64,
                    len as u64,
                    t.width,
                    t.height,
                    bytes_per_row,
                );
                if hr != 0 {
                    info!(
                        target: LOG_TARGET,
                        "[dump] readback {label}{what}: blit failed {hr:#x}"
                    );
                    continue;
                }
                let full = format!("{label}{what}");
                if bpp == 2 {
                    frame_dump_log_readback_f16(&full, &buf);
                } else {
                    frame_dump_log_readback(&full, t, &buf);
                }
            }
        }
    }
}

/// Decode one IEEE half-precision value to `f32`.
///
/// The readback path is diagnostics-only, so a plain bit-level expansion
/// (sign, 5-bit exponent, 10-bit mantissa, subnormals flushed through the
/// scale) beats pulling in a half-float dependency.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = f32::from(u8::from(bits & 0x8000 != 0)).mul_add(-2.0, 1.0);
    let exp = i32::from((bits >> 10) & 0x1F);
    let mant = f32::from(bits & 0x3FF);
    match exp {
        0 => sign * mant * 2f32.powi(-24),
        0x1F => {
            if mant == 0.0 {
                sign * f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => sign * (1024.0 + mant) * 2f32.powi(exp - 25),
    }
}

/// Log the statistics line for one read-back 2-byte (`R16F`) texture.
fn frame_dump_log_readback_f16(label: &str, buf: &[u8]) {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut nan = 0u32;
    let mut finite = 0u32;
    for c in buf.chunks_exact(2) {
        let v = f16_to_f32(u16::from_le_bytes([c[0], c[1]]));
        if v.is_nan() {
            nan += 1;
            continue;
        }
        finite += 1;
        min = min.min(v);
        max = max.max(v);
        sum += f64::from(v);
    }
    let mean = sum / f64::from(finite.max(1));
    info!(
        target: LOG_TARGET,
        "[dump] readback {label}: f16 min={min:.6} max={max:.6} mean={mean:.6} nan={nan}"
    );
}

/// Log the statistics line for one read-back texture.
fn frame_dump_log_readback(label: &str, t: &DumpReadback, buf: &[u8]) {
    let is_float = matches!(
        t.d3d_format,
        mtld3d_types::D3DFMT_R32F
            | mtld3d_types::D3DFMT_INTZ
            | mtld3d_types::D3DFMT_DF24
            | mtld3d_types::D3DFMT_DF16
    );
    let px = |x: u32, y: u32| {
        let idx = (y * t.width + x) as usize * 4;
        u32::from_le_bytes([buf[idx], buf[idx + 1], buf[idx + 2], buf[idx + 3]])
    };
    let (cx, cy) = (t.width / 2, t.height / 2);
    if is_float {
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut sum = 0.0f64;
        let mut nan = 0u32;
        let mut finite = 0u32;
        for c in buf.chunks_exact(4) {
            let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            if v.is_nan() {
                nan += 1;
                continue;
            }
            finite += 1;
            min = min.min(v);
            max = max.max(v);
            sum += f64::from(v);
        }
        let mean = sum / f64::from(finite.max(1));
        info!(
            target: LOG_TARGET,
            "[dump] readback {label}: f32 min={min:.6} max={max:.6} mean={mean:.6} nan={nan} \
             center={:.6} q1={:.6} q3={:.6}",
            f32::from_bits(px(cx, cy)),
            f32::from_bits(px(t.width / 4, t.height / 4)),
            f32::from_bits(px(3 * t.width / 4, 3 * t.height / 4))
        );
    } else {
        info!(
            target: LOG_TARGET,
            "[dump] readback {label}: raw center={:#010x} tl={:#010x} tr={:#010x} \
             bl={:#010x} br={:#010x}",
            px(cx, cy),
            px(t.width / 4, t.height / 4),
            px(3 * t.width / 4, t.height / 4),
            px(t.width / 4, 3 * t.height / 4),
            px(3 * t.width / 4, 3 * t.height / 4)
        );
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
            Some((DepthBinding::Eager(..), _, _)) => String::from("default"),
            Some((DepthBinding::None, _, _)) | None => String::from("none"),
        };
        (rt, ds)
    }

    /// Log the assembled draw state; called once per draw while a dump runs.
    pub fn frame_dump_draw(&mut self) {
        let seq = self.frame_dump.draws;
        self.frame_dump.draws += 1;
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
        let mut note: Vec<*const crate::texture::Direct3DTexture9> = Vec::new();
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
            if !kind.is_empty() {
                note.push(std::ptr::from_ref(tex));
            }
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
        for tex in note {
            // SAFETY: collected above from live bound stages on this thread;
            // nothing between there and here can release them.
            self.frame_dump_note_readback(unsafe { &*tex });
        }
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
