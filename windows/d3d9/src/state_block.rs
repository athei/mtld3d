//! `IDirect3DStateBlock9` — two creation paths, one vtable.
//!
//! `CreateStateBlock(D3DSBT_ALL)` takes an immediate full-device
//! snapshot via `StateSnapshot::capture_from`. `BeginStateBlock` /
//! `EndStateBlock` produces a *recorded* block: the COM setters
//! between Begin and End divert into `RecordingStateBlock::record`
//! instead of mutating the live device, so a subsequent `Apply()` only
//! replays the states the game explicitly set while recording. Full
//! D3D9 recording semantics — matches the replay pattern `WoW` uses to
//! pre-bake a portrait render state once at init and swap into it each
//! portrait frame.
//!
//! The filtered types (`D3DSBT_PIXELSTATE` / `D3DSBT_VERTEXSTATE`) snapshot
//! the whole device like `D3DSBT_ALL`, but record their [`StateBlockType`] so
//! `Apply` writes back only the vertex- or pixel-pipeline slice — the
//! membership of each state follows the [`StateBlockType`] predicates, which
//! classify each state into the vertex- or pixel-pipeline group per the
//! D3D9 state-block spec.

use core::ffi::c_void;

use log::warn;
use mtld3d_core::ff_state::FfStateSnapshot;
use mtld3d_shared::{InPtr, VtableThis};
use mtld3d_types::{
    D3DLIGHT9, D3DMATERIAL9, D3DMATRIX, D3DVIEWPORT9, Guid, IDirect3DStateBlock9Vtbl,
    RENDER_STATE_COUNT, SAMPLER_STATE_COUNT, StateBlockType,
};

use super::{
    D3D_OK, D3DERR_INVALIDCALL, E_NOINTERFACE, LOG_TARGET,
    com_ref::CachedComPtr,
    device::{DeviceInner, Direct3DDevice9},
    index_buffer::Direct3DIndexBuffer9,
    pixel_shader::Direct3DPixelShader9,
    shader_bindings::{BOOL_CONSTANT_COUNT, INT_CONSTANT_ROWS},
    stage_bindings::STAGE_COUNT,
    texture::Direct3DTexture9,
    vertex_buffer::Direct3DVertexBuffer9,
    vertex_decl::Direct3DVertexDeclaration9,
    vertex_shader::Direct3DVertexShader9,
};

static DIRECT3D_STATE_BLOCK9_VTBL: IDirect3DStateBlock9Vtbl = IDirect3DStateBlock9Vtbl {
    query_interface: sb_query_interface,
    add_ref: sb_add_ref,
    release: sb_release,
    get_device: sb_get_device,
    capture: sb_capture,
    apply: sb_apply,
};

/// One state-change operation recorded between `BeginStateBlock` and `EndStateBlock`.
///
/// COM-object variants carry their own [`CachedComPtr`] which Releases
/// on Drop; `Apply` replays each op by calling the corresponding live
/// `DeviceInner` setter, lending the stored pointer via
/// [`CachedComPtr::raw`] — the live setter does its own AddRef/Release
/// dance, so refcounts stay balanced.
pub enum StateOp {
    RenderState {
        state: u32,
        value: u32,
    },
    SamplerState {
        sampler: u32,
        type_: u32,
        value: u32,
    },
    TextureStageState {
        stage: u32,
        type_: u32,
        value: u32,
    },
    Transform {
        state: u32,
        matrix: D3DMATRIX,
    },
    Material(D3DMATERIAL9),
    Light {
        index: u32,
        light: D3DLIGHT9,
    },
    LightEnable {
        index: u32,
        enable: bool,
    },
    Viewport(D3DVIEWPORT9),
    ScissorRect([u32; 4]),
    Fvf(u32),
    Texture {
        stage: u32,
        /// Null for SetTexture(stage, NULL).
        tex: CachedComPtr<Direct3DTexture9>,
    },
    VertexDeclaration(CachedComPtr<Direct3DVertexDeclaration9>),
    VertexShader(CachedComPtr<Direct3DVertexShader9>),
    VertexShaderConstantF {
        start: u32,
        values: Vec<[f32; 4]>,
    },
    StreamSource {
        /// Null for SetStreamSource(stream, NULL, ..).
        ///
        /// Only stream 0 reaches the recorder — higher streams are
        /// rejected upstream with INVALIDCALL, same as the live setter.
        vb: CachedComPtr<Direct3DVertexBuffer9>,
        offset: u32,
        stride: u32,
    },
    Indices(CachedComPtr<Direct3DIndexBuffer9>),
    PixelShader(CachedComPtr<Direct3DPixelShader9>),
    PixelShaderConstantF {
        start: u32,
        values: Vec<[f32; 4]>,
    },
    VertexShaderConstantI {
        start: u32,
        values: Vec<[i32; 4]>,
    },
    VertexShaderConstantB {
        start: u32,
        values: Vec<i32>,
    },
    PixelShaderConstantI {
        start: u32,
        values: Vec<[i32; 4]>,
    },
    PixelShaderConstantB {
        start: u32,
        values: Vec<i32>,
    },
}

/// Ordered log of state-change ops recorded inside a `BeginStateBlock` / `EndStateBlock` pair.
///
/// `Apply` replays each op onto the device in order.
pub struct RecordingStateBlock {
    ops: Vec<StateOp>,
}

impl RecordingStateBlock {
    pub const fn new() -> Self {
        Self { ops: Vec::new() }
    }

    /// Append `op` to the recording.
    ///
    /// COM-object variants own their [`CachedComPtr`]; the caller adopts
    /// the pointer (bumping the refcount) before recording, and `Drop` on
    /// the recorded op releases it.
    pub fn record(&mut self, op: StateOp) {
        self.ops.push(op);
    }

    /// Refresh every op's payload from the current device state.
    ///
    /// Used by `IDirect3DStateBlock9::Capture` on a recorded block —
    /// games that pre-record a replayable state block at init, then call
    /// `Capture` per frame to re-snapshot the live values into that block
    /// before `Apply`ing it elsewhere. Only the *states already in the
    /// recording* are touched; Capture never adds new ops.
    ///
    /// For COM-object ops, we Release the currently-held ref and
    /// `AddRef` the device's current binding so the refcount invariant
    /// (block holds one ref per COM-object op until Drop) is preserved.
    fn capture_from(&mut self, dev: &DeviceInner) {
        for op in &mut self.ops {
            match op {
                StateOp::RenderState { state, value } => {
                    *value = dev.render_state(*state as usize);
                }
                StateOp::SamplerState {
                    sampler,
                    type_,
                    value,
                } => {
                    *value = dev
                        .stage_bindings()
                        .sampler_state(*sampler as usize, *type_ as usize);
                }
                StateOp::TextureStageState {
                    stage,
                    type_,
                    value,
                } => {
                    *value = dev
                        .ff_state()
                        .texture_stage_state(*stage as usize, *type_ as usize);
                }
                StateOp::Transform { state, matrix } => {
                    if let Some(m) = dev.ff_state().transform(*state).copied() {
                        *matrix = m;
                    }
                }
                StateOp::Material(m) => {
                    *m = *dev.ff_state().material();
                }
                StateOp::Light { index, light } => {
                    // A recorded Light op implies the slot was defined, so this
                    // resolves; leave the recorded value untouched otherwise.
                    if let Some(l) = dev.ff_state().get_light_at(*index) {
                        *light = l;
                    }
                }
                StateOp::LightEnable { index, enable } => {
                    *enable = dev.ff_state().is_light_enabled_at(*index);
                }
                StateOp::Viewport(v) => {
                    *v = dev.viewport();
                }
                StateOp::ScissorRect(r) => {
                    *r = dev.scissor_rect();
                }
                StateOp::Fvf(fvf) => {
                    *fvf = dev.fvf_field();
                }
                StateOp::Texture { stage, tex } => {
                    let live = dev.stage_bindings().texture(*stage as usize);
                    if tex.raw() != live {
                        // SAFETY: `live` came from the device's stage-binding
                        // slot, so it is null or a live IDirect3DTexture9
                        // whose AddRef/Release thunks remain callable for
                        // the duration of the recording.
                        *tex = unsafe { CachedComPtr::adopt(live) };
                    }
                }
                StateOp::VertexDeclaration(cur) => {
                    let live = dev.vertex_decl();
                    if cur.raw() != live {
                        // SAFETY: `live` from device's vertex_decl slot.
                        *cur = unsafe { CachedComPtr::adopt(live) };
                    }
                }
                StateOp::VertexShader(cur) => {
                    let live = dev.shader_bindings().vertex_shader();
                    if cur.raw() != live {
                        // SAFETY: `live` from device's shader-binding slot.
                        *cur = unsafe { CachedComPtr::adopt(live) };
                    }
                }
                StateOp::VertexShaderConstantF { start, values } => {
                    let copy = dev.shader_bindings().vs_constants_copy();
                    let s = *start as usize;
                    let end = (s + values.len()).min(copy.len());
                    if s < end {
                        values[..end - s].copy_from_slice(&copy[s..end]);
                    }
                }
                StateOp::StreamSource { vb, offset, stride } => {
                    let live = dev.bound_buffers().vertex_buffer();
                    if vb.raw() != live {
                        // SAFETY: `live` from device's bound vertex-buffer slot.
                        *vb = unsafe { CachedComPtr::adopt(live) };
                    }
                    *offset = dev.bound_buffers().vb_offset();
                    *stride = dev.bound_buffers().vb_stride();
                }
                StateOp::Indices(cur) => {
                    let live = dev.bound_buffers().index_buffer();
                    if cur.raw() != live {
                        // SAFETY: `live` from device's bound index-buffer slot.
                        *cur = unsafe { CachedComPtr::adopt(live) };
                    }
                }
                StateOp::PixelShader(cur) => {
                    let live = dev.shader_bindings().pixel_shader();
                    if cur.raw() != live {
                        // SAFETY: `live` from device's shader-binding slot.
                        *cur = unsafe { CachedComPtr::adopt(live) };
                    }
                }
                StateOp::PixelShaderConstantF { start, values } => {
                    let copy = dev.shader_bindings().ps_constants_copy();
                    let s = *start as usize;
                    let end = (s + values.len()).min(copy.len());
                    if s < end {
                        values[..end - s].copy_from_slice(&copy[s..end]);
                    }
                }
                StateOp::VertexShaderConstantI { start, values } => {
                    refresh_constant_rows(
                        values,
                        *start,
                        &dev.shader_bindings().vs_constants_i_copy(),
                    );
                }
                StateOp::VertexShaderConstantB { start, values } => {
                    refresh_constant_rows(
                        values,
                        *start,
                        &dev.shader_bindings().vs_constants_b_copy(),
                    );
                }
                StateOp::PixelShaderConstantI { start, values } => {
                    refresh_constant_rows(
                        values,
                        *start,
                        &dev.shader_bindings().ps_constants_i_copy(),
                    );
                }
                StateOp::PixelShaderConstantB { start, values } => {
                    refresh_constant_rows(
                        values,
                        *start,
                        &dev.shader_bindings().ps_constants_b_copy(),
                    );
                }
            }
        }
    }

    fn apply_to(&self, dev: &mut DeviceInner) {
        for op in &self.ops {
            match op {
                StateOp::RenderState { state, value } => {
                    dev.set_render_state(*state as usize, *value);
                }
                StateOp::SamplerState {
                    sampler,
                    type_,
                    value,
                } => {
                    dev.stage_bindings_mut().set_sampler_state(
                        *sampler as usize,
                        *type_ as usize,
                        *value,
                    );
                }
                StateOp::TextureStageState {
                    stage,
                    type_,
                    value,
                } => {
                    dev.ff_state_mut().set_texture_stage_state(
                        *stage as usize,
                        *type_ as usize,
                        *value,
                    );
                }
                StateOp::Transform { state, matrix } => {
                    dev.ff_state_mut().set_transform(*state, matrix);
                }
                StateOp::Material(m) => {
                    dev.ff_state_mut().set_material(m);
                }
                StateOp::Light { index, light } => {
                    dev.ff_state_mut().set_light_at(*index, light);
                }
                StateOp::LightEnable { index, enable } => {
                    dev.ff_state_mut().set_light_enabled_at(*index, *enable);
                }
                StateOp::Viewport(v) => {
                    dev.set_viewport(*v);
                }
                StateOp::ScissorRect(r) => {
                    dev.set_scissor_rect(*r);
                }
                StateOp::Fvf(fvf) => {
                    dev.set_fvf_field(*fvf);
                }
                StateOp::Texture { stage, tex } => {
                    dev.stage_bindings_mut()
                        .replace_texture(*stage as usize, tex.raw());
                }
                StateOp::VertexDeclaration(decl) => {
                    // D3D9 restores the vertex declaration on Apply only when the
                    // block captured a non-NULL one. A NULL-captured vdecl leaves
                    // the device's current declaration untouched — so a Capture
                    // taken while no decl is bound does not clobber a decl set
                    // afterwards.
                    if !decl.raw().is_null() {
                        dev.replace_vertex_decl(decl.raw());
                    }
                }
                StateOp::VertexShader(vs) => {
                    dev.shader_bindings_mut().replace_vertex_shader(vs.raw());
                }
                StateOp::VertexShaderConstantF { start, values } => {
                    dev.shader_bindings_mut().write_vs_constants(*start, values);
                    crate::device::propagate_vs_const_delta(dev, *start, values);
                }
                StateOp::StreamSource { vb, offset, stride } => {
                    dev.bound_buffers_mut()
                        .replace_vertex_buffer(vb.raw(), *offset, *stride);
                }
                StateOp::Indices(ib) => {
                    dev.bound_buffers_mut().replace_index_buffer(ib.raw());
                }
                StateOp::PixelShader(ps) => {
                    dev.shader_bindings_mut().replace_pixel_shader(ps.raw());
                }
                StateOp::PixelShaderConstantF { start, values } => {
                    dev.shader_bindings_mut().write_ps_constants(*start, values);
                    crate::device::propagate_ps_const_delta(dev, *start, values);
                }
                // Int/bool constants are stored only — no encoder mirror to
                // propagate into (see `device_set_vertex_shader_constant_i`).
                StateOp::VertexShaderConstantI { start, values } => {
                    dev.shader_bindings_mut()
                        .write_vs_constants_i(*start, values);
                }
                StateOp::VertexShaderConstantB { start, values } => {
                    dev.shader_bindings_mut()
                        .write_vs_constants_b(*start, values);
                }
                StateOp::PixelShaderConstantI { start, values } => {
                    dev.shader_bindings_mut()
                        .write_ps_constants_i(*start, values);
                }
                StateOp::PixelShaderConstantB { start, values } => {
                    dev.shader_bindings_mut()
                        .write_ps_constants_b(*start, values);
                }
            }
        }
        // apply_to bypasses the COM Set* thunks (which dirty per-call)
        // and writes the inner mutators directly. Mark dirty once at
        // the end so the next draw re-emits all snapshot pieces.
        dev.mark_snapshot_dirty_all();
    }
}

#[repr(C)]
pub struct Direct3DStateBlock9 {
    vtbl: *const IDirect3DStateBlock9Vtbl,
    refcount: u32,
    inner: *mut StateBlockInner,
}

impl Direct3DStateBlock9 {
    /// Capture a fresh state block of the given `D3DSTATEBLOCKTYPE` from `device_obj`.
    ///
    /// Snapshots the full device regardless of type; the stored
    /// [`StateBlockType`] decides which slice `Apply` writes back. Rejects an
    /// unrecognised type with `D3DERR_INVALIDCALL`.
    pub fn capture(device_obj: *mut Direct3DDevice9, type_: u32) -> Result<Self, i32> {
        let Some(block_type) = StateBlockType::from_d3dsbt(type_) else {
            warn!(target: LOG_TARGET, "reject CreateStateBlock(type={type_}) → INVALIDCALL");
            return Err(D3DERR_INVALIDCALL);
        };
        // SAFETY: `device_obj` is the caller-supplied `Direct3DDevice9*`
        // that originated `CreateStateBlock`; D3D9 ABI guarantees it is
        // a live wrapper.
        let d = unsafe { &*device_obj };
        let dev = d.inner();
        let fvf = d.fvf();
        let inner = Box::into_raw(Box::new(StateBlockInner {
            device: device_obj,
            body: StateBlockBody::Snapshot(Box::new(StateSnapshot::capture_from(
                dev, fvf, block_type,
            ))),
        }));
        Ok(Self {
            vtbl: &raw const DIRECT3D_STATE_BLOCK9_VTBL,
            refcount: 1,
            inner,
        })
    }

    /// Wrap a `RecordingStateBlock` produced by `EndStateBlock` in a COM-addressable state block.
    ///
    /// `Apply()` replays the recorded ops onto the device.
    pub fn from_recording(
        device_obj: *mut Direct3DDevice9,
        recording: RecordingStateBlock,
    ) -> Self {
        let inner = Box::into_raw(Box::new(StateBlockInner {
            device: device_obj,
            body: StateBlockBody::Recorded(recording),
        }));
        Self {
            vtbl: &raw const DIRECT3D_STATE_BLOCK9_VTBL,
            refcount: 1,
            inner,
        }
    }
}

struct StateBlockInner {
    device: *mut Direct3DDevice9,
    body: StateBlockBody,
}

enum StateBlockBody {
    /// Full snapshot from `CreateStateBlock` / `Capture`.
    ///
    /// `Apply` overwrites every capturable state.
    Snapshot(Box<StateSnapshot>),
    /// Ordered op log from `BeginStateBlock` / `EndStateBlock`.
    ///
    /// `Apply` replays only the states the game explicitly set.
    Recorded(RecordingStateBlock),
}

struct StateSnapshot {
    /// Which slice of the captured state `apply_to` writes back.
    ///
    /// `Vertex` / `Pixel` snapshots still record every field (capture is
    /// type-agnostic); the filter is applied only at `Apply` time.
    block_type: StateBlockType,
    fvf: u32,
    render_states: [u32; RENDER_STATE_COUNT],
    sampler_states: [[u32; SAMPLER_STATE_COUNT]; STAGE_COUNT],
    ff: FfStateSnapshot,
    bound_textures: [CachedComPtr<Direct3DTexture9>; STAGE_COUNT],
    bound_vertex_shader: CachedComPtr<Direct3DVertexShader9>,
    bound_pixel_shader: CachedComPtr<Direct3DPixelShader9>,
    /// Vertex declaration + index buffer round-trip like the bound shaders.
    ///
    /// Captured with a public `AddRef` (default `Owned` marker), released on
    /// drop. The device's own slots hold a separate `Bound` ref, so the
    /// snapshot keeps the object alive even after the app releases its ref.
    bound_vertex_decl: CachedComPtr<Direct3DVertexDeclaration9>,
    bound_index_buffer: CachedComPtr<Direct3DIndexBuffer9>,
    vs_constants: Box<[[f32; 4]; 256]>,
    ps_constants: Box<[[f32; 4]; 256]>,
    vs_constants_i: Box<[[i32; 4]; INT_CONSTANT_ROWS]>,
    vs_constants_b: Box<[i32; BOOL_CONSTANT_COUNT]>,
    ps_constants_i: Box<[[i32; 4]; INT_CONSTANT_ROWS]>,
    ps_constants_b: Box<[i32; BOOL_CONSTANT_COUNT]>,
}

impl StateSnapshot {
    fn capture_from(dev: &DeviceInner, fvf: u32, block_type: StateBlockType) -> Self {
        let bound_textures = core::array::from_fn(|i| {
            let ptr = dev.stage_bindings().texture(i);
            // SAFETY: `ptr` comes from the device's stage-binding slot,
            // which is null or a live IDirect3DTexture9.
            unsafe { CachedComPtr::adopt(ptr) }
        });
        // SAFETY: `vs` from device's vertex-shader slot.
        let bound_vertex_shader =
            unsafe { CachedComPtr::adopt(dev.shader_bindings().vertex_shader()) };
        // SAFETY: `ps` from device's pixel-shader slot.
        let bound_pixel_shader =
            unsafe { CachedComPtr::adopt(dev.shader_bindings().pixel_shader()) };
        // SAFETY: `decl` from the device's vertex-declaration slot — null or a
        // live IDirect3DVertexDeclaration9 whose AddRef/Release stay callable.
        let bound_vertex_decl = unsafe { CachedComPtr::adopt(dev.vertex_decl()) };
        // SAFETY: `ib` from the device's bound index-buffer slot.
        let bound_index_buffer = unsafe { CachedComPtr::adopt(dev.bound_buffers().index_buffer()) };

        Self {
            block_type,
            fvf,
            render_states: *dev.render_states(),
            sampler_states: capture_sampler_states(dev),
            ff: FfStateSnapshot::from(dev.ff_state()),
            bound_textures,
            bound_vertex_shader,
            bound_pixel_shader,
            bound_vertex_decl,
            bound_index_buffer,
            vs_constants: Box::new(dev.shader_bindings().vs_constants_copy()),
            ps_constants: Box::new(dev.shader_bindings().ps_constants_copy()),
            vs_constants_i: Box::new(dev.shader_bindings().vs_constants_i_copy()),
            vs_constants_b: Box::new(dev.shader_bindings().vs_constants_b_copy()),
            ps_constants_i: Box::new(dev.shader_bindings().ps_constants_i_copy()),
            ps_constants_b: Box::new(dev.shader_bindings().ps_constants_b_copy()),
        }
    }

    /// Write the snapshot back into the device.
    ///
    /// The `replace_*` helpers run the full refcount swap dance, so the
    /// snapshot's own `CachedComPtr` keeps its ref and stays valid for a
    /// subsequent Apply.
    fn apply_to(&self, dev: &mut DeviceInner, device_wrapper: &Direct3DDevice9) {
        let block_type = self.block_type;

        // FVF (vertex pipeline). Restore first: the field-only `set_fvf` has no
        // decl side-effect, so the decl restore below lands the captured pair.
        if block_type.includes_vertex_pipeline() {
            device_wrapper.set_fvf(self.fvf);
        }

        // Render states — per-index membership. `0u32..` yields the D3DRS index
        // as a u32 without a fallible width conversion.
        for (rs, &v) in (0u32..).zip(self.render_states.iter()) {
            if block_type.includes_render_state(rs) {
                dev.set_render_state(rs as usize, v);
            }
        }

        // Sampler states — per-type membership, every stage.
        for stage in 0..STAGE_COUNT {
            for (samp_ty, &val) in (0u32..).zip(self.sampler_states[stage].iter()) {
                if block_type.includes_sampler_state(samp_ty) {
                    dev.stage_bindings_mut()
                        .set_sampler_state(stage, samp_ty as usize, val);
                }
            }
        }

        // Fixed-function: transforms + material are ALL-only, lights are
        // vertex-pipeline, texture-stage states split per index.
        self.ff.restore_filtered(dev.ff_state_mut(), block_type);

        // Float shader constants (+ encoder delta). Gate the write and its
        // delta propagation together so we never propagate an unwritten range.
        if block_type.includes_vertex_pipeline() {
            dev.shader_bindings_mut()
                .write_vs_constants(0, self.vs_constants.as_ref());
            crate::device::propagate_vs_const_delta(dev, 0, self.vs_constants.as_ref());
        }
        if block_type.includes_pixel_pipeline() {
            dev.shader_bindings_mut()
                .write_ps_constants(0, self.ps_constants.as_ref());
            crate::device::propagate_ps_const_delta(dev, 0, self.ps_constants.as_ref());
        }
        // Int/bool constants are stored only — no encoder mirror to
        // propagate into (see `device_set_vertex_shader_constant_i`).
        if block_type.includes_vertex_pipeline() {
            dev.shader_bindings_mut()
                .write_vs_constants_i(0, self.vs_constants_i.as_ref());
            dev.shader_bindings_mut()
                .write_vs_constants_b(0, self.vs_constants_b.as_ref());
        }
        if block_type.includes_pixel_pipeline() {
            dev.shader_bindings_mut()
                .write_ps_constants_i(0, self.ps_constants_i.as_ref());
            dev.shader_bindings_mut()
                .write_ps_constants_b(0, self.ps_constants_b.as_ref());
        }

        // Bound textures + index buffer are D3DSBT_ALL-only — a filtered block
        // leaves them at their live values.
        if matches!(block_type, StateBlockType::All) {
            for (i, tex) in self.bound_textures.iter().enumerate() {
                dev.stage_bindings_mut().replace_texture(i, tex.raw());
            }
            dev.bound_buffers_mut()
                .replace_index_buffer(self.bound_index_buffer.raw());
        }

        // Vertex shader + declaration are vertex-pipeline; pixel shader is
        // pixel-pipeline.
        if block_type.includes_vertex_pipeline() {
            dev.shader_bindings_mut()
                .replace_vertex_shader(self.bound_vertex_shader.raw());
            // D3D9 restores the vertex declaration only when the block captured a
            // non-NULL one; the vertex shader, by contrast, applies even when
            // NULL (unbinds).
            if !self.bound_vertex_decl.raw().is_null() {
                dev.replace_vertex_decl(self.bound_vertex_decl.raw());
            }
        }
        if block_type.includes_pixel_pipeline() {
            dev.shader_bindings_mut()
                .replace_pixel_shader(self.bound_pixel_shader.raw());
        }

        // apply_to bypasses the COM Set* thunks (which dirty per-call) and
        // writes the inner mutators directly; for a filtered block this
        // over-marks (re-emits unchanged pieces) but never under-marks. Mark
        // dirty once so the next draw re-emits the snapshot pieces.
        dev.mark_snapshot_dirty_all();
    }
}

/// Refresh a recorded constant range `values` from the live device mirror `src`.
///
/// The range starts at register `start` and is clamped to the mirror's
/// length. Shared by the integer/boolean constant `Capture` arms; the
/// float arms inline the identical clamp against their own mirror.
fn refresh_constant_rows<T: Copy>(values: &mut [T], start: u32, src: &[T]) {
    let s = start as usize;
    let end = (s + values.len()).min(src.len());
    if s < end {
        values[..end - s].copy_from_slice(&src[s..end]);
    }
}

fn capture_sampler_states(dev: &DeviceInner) -> [[u32; SAMPLER_STATE_COUNT]; STAGE_COUNT] {
    let mut out = [[0u32; SAMPLER_STATE_COUNT]; STAGE_COUNT];
    for (stage, row) in out.iter_mut().enumerate() {
        *row = dev.stage_bindings().sampler_states(stage);
    }
    out
}

// ── Vtable implementations ──

#[inline]
fn sb_timer(this: *mut c_void) -> mtld3d_core::perf::ApiTimer {
    use mtld3d_core::perf::{ApiCategory, ApiTimer};
    // SAFETY: vtable thunk; `this` is *mut Direct3DStateBlock9 per the
    // IDirect3DStateBlock9 ABI. `opt` filters null, forwarding a null
    // `perf_ptr` instead of dereferencing.
    let Some(obj) = (unsafe { InPtr::<Direct3DStateBlock9>::opt(this) }) else {
        return ApiTimer::start(core::ptr::null_mut(), ApiCategory::StateBlock);
    };
    // SAFETY: `obj.inner` was installed by `new`/`from_recording` as a
    // `Box::into_raw` and lives until `sb_release` at refcount zero.
    let sb_inner = unsafe { &*obj.inner };
    let dev_ptr = if sb_inner.device.is_null() {
        core::ptr::null_mut()
    } else {
        // SAFETY: `sb_inner.device` was stamped at the state block's
        // creation from a live `Direct3DDevice9*`; non-null here.
        unsafe { (*sb_inner.device).inner_ptr() }
    };
    ApiTimer::start(
        crate::device::DeviceInner::perf_ptr_of(dev_ptr),
        ApiCategory::StateBlock,
    )
}

extern "system" fn sb_query_interface(
    this: *mut c_void,
    _riid: *const Guid,
    _ppv: *mut *mut c_void,
) -> i32 {
    let _timer = sb_timer(this);
    mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "stub IDirect3DStateBlock9::QueryInterface → E_NOINTERFACE");
    E_NOINTERFACE
}

extern "system" fn sb_add_ref(this: *mut c_void) -> u32 {
    let _timer = sb_timer(this);
    // SAFETY: IDirect3DStateBlock9 IUnknown AddRef thunk; the D3D9 ABI
    // guarantees `this` is the live wrapper for the call.
    unsafe { crate::com_ref::com_add_ref::<Direct3DStateBlock9>(this) }
}

extern "system" fn sb_release(this: *mut c_void) -> u32 {
    let _timer = sb_timer(this);
    // SAFETY: IDirect3DStateBlock9 IUnknown Release thunk; the D3D9 ABI
    // guarantees `this` is the live wrapper for the call.
    unsafe { crate::com_ref::com_release::<Direct3DStateBlock9>(this) }
}

/// Destroy a `Direct3DStateBlock9` wrapper once its refcount has reached zero.
///
/// # Safety
/// `this` must point to a live `Direct3DStateBlock9` wrapper at refcount zero;
/// caller must not access the wrapper afterwards.
unsafe fn finalize_state_block(this: *mut Direct3DStateBlock9) {
    // SAFETY: refcount reached zero; `(*this).inner` is the original
    // `Box::into_raw(StateBlockInner)` from `Self::new` and no other reference
    // can survive a zero refcount.
    let inner = unsafe { (*this).inner };
    // SAFETY: as above — sole owner of the inner allocation.
    drop(unsafe { Box::from_raw(inner) });
    // SAFETY: refcount reached zero; `this` is the original
    // `Box::into_raw(Direct3DStateBlock9)` allocation.
    drop(unsafe { Box::from_raw(this) });
}

// SAFETY: `refcount_mut` exposes this wrapper's own counter; `finalize` frees it
// exactly once at refcount zero. State blocks have no bound-slot (private)
// refcount and do not forward to the device in this revision.
unsafe impl crate::com_ref::ComChild for Direct3DStateBlock9 {
    fn refcount_mut(&mut self) -> &mut u32 {
        &mut self.refcount
    }
    fn device_forward_target(&self) -> *mut c_void {
        // `StateBlockInner::device` is the owning `Direct3DDevice9`* wrapper.
        // SAFETY: `inner` is the live `Box::into_raw(StateBlockInner)`, valid for
        // every live wrapper reference.
        unsafe { (*self.inner).device.cast::<c_void>() }
    }
    unsafe fn finalize(this: *mut Self) {
        // SAFETY: forwarded from the engine — refcount is zero.
        unsafe { finalize_state_block(this) };
    }
}

extern "system" fn sb_get_device(this: *mut c_void, _device: *mut *mut c_void) -> i32 {
    let _timer = sb_timer(this);
    mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "stub IDirect3DStateBlock9::GetDevice → INVALIDCALL");
    D3DERR_INVALIDCALL
}

extern "system" fn sb_capture(this: *mut c_void) -> i32 {
    let _timer = sb_timer(this);
    // SAFETY: IDirect3DStateBlock9 IUnknown thunk; D3D9 ABI guarantees `this` is *mut.
    let mut wrap = unsafe { VtableThis::<Direct3DStateBlock9>::new(this) };
    let obj: &mut Direct3DStateBlock9 = &mut wrap;
    // SAFETY: `obj.inner` was installed by `Self::new` /
    // `from_recording` as a `Box::into_raw` and lives until `sb_release`
    // at refcount zero.
    let inner = unsafe { &mut *obj.inner };
    if inner.device.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: `inner.device` was stamped at creation from a live
    // `Direct3DDevice9*`; non-null here.
    let device_obj = unsafe { &*inner.device };
    let dev = device_obj.inner();
    // Capture during an open BeginStateBlock recording is INVALIDCALL.
    if dev.is_state_block_recording() {
        return D3DERR_INVALIDCALL;
    }
    match &mut inner.body {
        StateBlockBody::Snapshot(snap) => {
            // Re-snapshot the whole device but keep the block's filter type so
            // the next Apply writes back the same slice. Assignment drops the
            // previous snapshot (releases its AddRef'd slots).
            let block_type = snap.block_type;
            **snap = StateSnapshot::capture_from(dev, device_obj.fvf(), block_type);
        }
        StateBlockBody::Recorded(rec) => {
            rec.capture_from(dev);
        }
    }
    D3D_OK
}

extern "system" fn sb_apply(this: *mut c_void) -> i32 {
    let _timer = sb_timer(this);
    // SAFETY: IDirect3DStateBlock9 IUnknown thunk; D3D9 ABI guarantees `this` is *mut.
    let mut wrap = unsafe { VtableThis::<Direct3DStateBlock9>::new(this) };
    let obj: &mut Direct3DStateBlock9 = &mut wrap;
    // SAFETY: `obj.inner` was installed by `Self::new` /
    // `from_recording` as a `Box::into_raw` and lives until `sb_release`
    // at refcount zero.
    let inner = unsafe { &*obj.inner };
    if inner.device.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: `inner.device` was stamped at creation from a live
    // `Direct3DDevice9*`; non-null here.
    let device_obj = unsafe { &mut *inner.device };
    let dev = device_obj.inner();
    // Apply during an open BeginStateBlock recording is INVALIDCALL — and must
    // NOT mutate live render state.
    if dev.is_state_block_recording() {
        return D3DERR_INVALIDCALL;
    }
    match &inner.body {
        StateBlockBody::Snapshot(snap) => snap.apply_to(dev, device_obj),
        StateBlockBody::Recorded(rec) => rec.apply_to(dev),
    }
    D3D_OK
}
