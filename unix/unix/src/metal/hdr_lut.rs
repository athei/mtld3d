//! `color.hdr.lut`: the HDR present's inverse tone map baked into a 3D lookup table.
//!
//! The analytic HDR present stage runs BT.2446 Method A in `ICtCp` on every
//! pixel: a luminance curve, two PQ encodes, a PQ decode and four matrix
//! products, about twenty transcendentals. Its input is the game's 8-bit
//! sRGB-encoded back buffer and its only other inputs are the display peak
//! and the guest's gamma ramp, both of which move rarely. So the whole map is
//! a function of the encoded colour alone between two such changes, and it is
//! smooth enough that a [`LATTICE`]³ table, square-root spaced and sampled
//! with trilinear filtering, reproduces it below one just-noticeable
//! difference (`command/tests.rs` measures `ΔE_ITP` over every 8-bit input).
//!
//! The table is baked on the GPU, by a compute kernel in the present library
//! that calls the same `bt2446_ictcp` and `apply_gamma_ramp` the analytic
//! stage calls, into the present command buffer ahead of the pass that reads
//! it. Metal orders the two within the buffer. A bake is one lattice of
//! threads, a seventh of the pixels of a 1741x1132 frame, and runs only when
//! the peak, the ramp's revision or the layer changes.
//!
//! Everything lives on the device's record: the table, the bake pipeline and
//! the sampling pipeline, built on the first HDR present with the key on, so a
//! session without the key compiles nothing new.

use std::sync::{Mutex, PoisonError};

use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLDevice, MTLLibrary, MTLPixelFormat, MTLRenderPipelineState, MTLResource, MTLSize,
    MTLStorageMode, MTLTexture, MTLTextureDescriptor, MTLTextureType, MTLTextureUsage,
};

use crate::LOG_TARGET;

/// Lattice points per axis of the table.
///
/// Measured over all 2²⁴ 8-bit inputs against the analytic stage, worst
/// `ΔE_ITP` (BT.2124; 1.0 = one just-noticeable difference): 33 points give 0.96
/// at peak 2 and 1.93 at peak 8, 65 give 0.32 and 0.97, 97 give 0.30 and 0.62.
/// 65 is the smallest that stays below one JND at both, at 65³ × 8 B ≈ 2.2 MiB.
pub const LATTICE: usize = 65;

/// Kernel bake flag: the guest's gamma ramp is bound and applies before the decode.
const BAKE_RAMP: u32 = 1;

/// What a baked table was evaluated for; a change of any field re-bakes it.
#[derive(Clone, Copy, PartialEq, Eq)]
struct LutKey {
    peak_bits: u32,
    gamma_layer: usize,
    gamma_revision: u64,
}

/// One device's table and the two pipelines that bake and sample it.
#[derive(Default)]
pub struct HdrLut {
    state: Mutex<LutState>,
}

/// The device's lookup-table objects, each owning one canonical retain.
///
/// Raw pointers keep the record `Send`: every access is under the mutex, on
/// the presenter thread or in device teardown, and a command buffer that
/// encoded against an object retains it for itself, so a release here never
/// pulls an object out from under the GPU.
struct LutState {
    texture: *mut ProtocolObject<dyn MTLTexture>,
    bake: *mut ProtocolObject<dyn MTLComputePipelineState>,
    sample: *mut ProtocolObject<dyn MTLRenderPipelineState>,
    key: Option<LutKey>,
    /// Set once building either pipeline or the texture failed.
    ///
    /// The frame then takes the analytic stage for the rest of the session,
    /// warned once.
    failed: bool,
}

impl Default for LutState {
    fn default() -> Self {
        Self {
            texture: core::ptr::null_mut(),
            bake: core::ptr::null_mut(),
            sample: core::ptr::null_mut(),
            key: None,
            failed: false,
        }
    }
}

// SAFETY: see the type's doc: the pointers own retains reached only under the
// mutex, and Metal objects may be used and released from any thread.
unsafe impl Send for LutState {}

impl Drop for LutState {
    fn drop(&mut self) {
        // SAFETY: each pointer is null or owns the retain `Retained::into_raw`
        // gave it, and nothing else holds a copy.
        drop(unsafe { Retained::from_raw(self.texture) });
        // SAFETY: as above.
        drop(unsafe { Retained::from_raw(self.bake) });
        // SAFETY: as above.
        drop(unsafe { Retained::from_raw(self.sample) });
    }
}

/// A table ready to sample, and the pipeline that samples it onto an `RGBA16Float` target.
pub struct PreparedLut {
    pub texture: Retained<ProtocolObject<dyn MTLTexture>>,
    pub pipeline: u64,
}

impl HdrLut {
    /// The table for `peak` and the layer's current ramp, baking it into `cmd_buf` when stale.
    ///
    /// `None` when a pipeline or the texture could not be built, or the bake
    /// could not be encoded; the caller then runs the analytic stage, which
    /// produces the same frame at its usual cost.
    pub fn prepare(
        &self,
        cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
        uniforms: [f32; 4],
        peak: f32,
        gamma_layer: usize,
    ) -> Option<PreparedLut> {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if state.failed {
            return None;
        }
        let device = cmd_buf.device();
        if state.texture.is_null() && !build(&mut state, &device) {
            state.failed = true;
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET,
                "present: color.hdr.lut could not build its table or pipelines; HDR presents \
                 run the per-pixel tone map instead"
            );
            return None;
        }
        let key = LutKey {
            peak_bits: peak.to_bits(),
            gamma_layer,
            gamma_revision: if gamma_layer == 0 {
                0
            } else {
                super::gamma::revision(gamma_layer)
            },
        };
        if state.key != Some(key) {
            state.key = None;
            if !bake(&state, cmd_buf, uniforms, gamma_layer) {
                return None;
            }
            state.key = Some(key);
            log::debug!(
                target: LOG_TARGET,
                "present: HDR lookup table baked for peak {peak:.3} (ramp revision {})",
                key.gamma_revision
            );
        }
        // SAFETY: the state holds the texture's canonical retain; the returned
        // retain keeps this borrow independent of the lock.
        let texture = unsafe { Retained::retain(state.texture) }?;
        Some(PreparedLut {
            texture,
            pipeline: state.sample as u64,
        })
    }
}

/// Build the table texture and both pipelines into `state`; `false` if any is refused.
fn build(state: &mut LutState, device: &ProtocolObject<dyn MTLDevice>) -> bool {
    let Some(library) = super::present::ensure_library(device) else {
        return false;
    };
    let function = |name: &str| library.newFunctionWithName(&NSString::from_str(name));
    let (Some(kernel), Some(vs), Some(ps)) = (
        function("mtld3d_hdr_lut_bake"),
        function("mtld3d_present_vs"),
        function("mtld3d_present_ps_hdr_lut"),
    ) else {
        return false;
    };
    let Ok(bake) = device.newComputePipelineStateWithFunction_error(&kernel) else {
        return false;
    };
    let Some(sample) = super::present::build_pipeline(
        device,
        &vs,
        &ps,
        MTLPixelFormat::RGBA16Float,
        "mtld3d-present-pipeline-hdr-lut",
    ) else {
        return false;
    };
    let desc = MTLTextureDescriptor::new();
    desc.setTextureType(MTLTextureType::Type3D);
    desc.setPixelFormat(MTLPixelFormat::RGBA16Float);
    // SAFETY: scalar property writes on an owned descriptor; the lattice is
    // far inside every device's 3D texture limit.
    unsafe { desc.setWidth(LATTICE) };
    // SAFETY: as above.
    unsafe { desc.setHeight(LATTICE) };
    // SAFETY: as above.
    unsafe { desc.setDepth(LATTICE) };
    desc.setStorageMode(MTLStorageMode::Private);
    desc.setUsage(MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite);
    let Some(texture) = super::texture::new_texture(device, &desc, "mtld3d-hdr-lut") else {
        return false;
    };
    texture.setLabel(Some(&NSString::from_str("mtld3d-hdr-lut")));
    state.texture = Retained::into_raw(texture);
    state.bake = Retained::into_raw(bake);
    state.sample = Retained::into_raw(sample);
    true
}

/// Encode one bake of the table for `uniforms` and the layer's ramp.
fn bake(
    state: &LutState,
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    uniforms: [f32; 4],
    gamma_layer: usize,
) -> bool {
    let Some(enc) = cmd_buf.computeCommandEncoder() else {
        return false;
    };
    enc.setLabel(Some(&NSString::from_str("mtld3d-hdr-lut-bake")));
    // SAFETY: `build` stored live retains in both pointers, held under the lock.
    let pipeline = unsafe { &*state.bake };
    // SAFETY: as above.
    let texture = unsafe { &*state.texture };
    enc.setComputePipelineState(pipeline);
    // SAFETY: objc2 typed bindings; every pointer is a live stack slot or the
    // texture the state holds, and the encoder copies the bytes before
    // returning.
    unsafe { enc.setTexture_atIndex(Some(texture), 0) };
    // SAFETY: as above.
    unsafe {
        enc.setBytes_length_atIndex(
            core::ptr::NonNull::from(&uniforms).cast(),
            core::mem::size_of_val(&uniforms),
            0,
        );
    }
    let flags = if gamma_layer != 0 && super::gamma::bind_compute(&enc, gamma_layer) {
        BAKE_RAMP
    } else {
        // The kernel reads slot 1 only under `BAKE_RAMP`, but every declared
        // buffer argument gets a binding.
        // SAFETY: as above; four bytes of a stack slot, copied.
        unsafe {
            enc.setBytes_length_atIndex(
                core::ptr::NonNull::from(&0u64).cast(),
                8,
                super::gamma::FRAGMENT_BUFFER_INDEX,
            );
        }
        0
    };
    // SAFETY: as above.
    unsafe {
        enc.setBytes_length_atIndex(core::ptr::NonNull::from(&flags).cast(), 4, 2);
    }
    let grid = MTLSize {
        width: LATTICE,
        height: LATTICE,
        depth: LATTICE,
    };
    let group = MTLSize {
        width: 4,
        height: 4,
        depth: 4,
    };
    enc.dispatchThreads_threadsPerThreadgroup(grid, group);
    enc.endEncoding();
    true
}
