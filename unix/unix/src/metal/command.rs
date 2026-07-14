use core::ffi::c_void;
use std::{
    collections::BTreeMap,
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

use block2::RcBlock;
use log::error;
use mtld3d_shared::{
    BlitCommand, BlitCommandType, Command, CommandType, MetalHandle, PassDescriptor,
    SubmitFrameParams,
    mtl::{CullMode, IndexType, LoadAction, PrimitiveType, StoreAction, VisibilityResultMode},
    mtl_handle::{
        MTLBufferKind, MTLCommandQueueKind, MTLDepthStencilStateKind, MTLDeviceKind,
        MTLRenderPipelineStateKind, MTLSamplerStateKind, MTLTextureKind,
    },
    perf::CycleSetTimer,
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSRange;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLCullMode, MTLDepthClipMode, MTLDevice, MTLIndexType, MTLLoadAction, MTLOrigin,
    MTLPixelFormat, MTLPrimitiveType, MTLRenderCommandEncoder, MTLRenderPassDescriptor,
    MTLResource, MTLResourceOptions, MTLScissorRect, MTLSize, MTLStoreAction, MTLTexture,
    MTLViewport, MTLVisibilityResultMode,
};
use objc2_quartz_core::CAMetalDrawable;

use crate::{LOG_TARGET, metal::handle::IntoRetained};

/// `Retained<ProtocolObject<dyn MTLCommandBuffer>>` is not `Send`/`Sync` in objc2.
///
/// Apple does not categorically mark its APIs thread-safe. The operations
/// we perform on this handle from the completion-handler thread
/// (`Retained` drop = refcount decrement) and from the wait thread
/// (`clone` = refcount increment, `waitUntilCompleted`) are all documented
/// thread-safe by Apple. Wrap and assert.
struct PendingCmdBuf(Retained<ProtocolObject<dyn MTLCommandBuffer>>);
// allow: chosen narrow exception. The structural alternatives — `SendWrapper`
// (panics on cross-thread access, but Metal's completion-handler runs on its
// own thread) or storing as `usize` + `Retained::retain(ptr)` on every access
// (multiplies the unsafe-block count across the file for no safety gain) —
// both make the code worse. The `unsafe impl Send`/`Sync` here is correct per
// Apple's documented thread-safety for the three ops we actually perform
// (`clone` = refcount inc, `Drop` = refcount dec, `waitUntilCompleted` = block).
// SAFETY rationale lives in the doc comment on `PendingCmdBuf` above.
// SAFETY: see the `PendingCmdBuf` doc comment above — `clone`, `Drop`,
// and `waitUntilCompleted` are documented thread-safe by Apple for
// `MTLCommandBuffer`, which is the only API surface we touch.
#[allow(clippy::non_send_fields_in_send_ty)]
unsafe impl Send for PendingCmdBuf {}
// SAFETY: as above.
unsafe impl Sync for PendingCmdBuf {}

/// Registry of in-flight `MTLCommandBuffer`s keyed by `submit_seq`.
///
/// `submit_frame` inserts before `commit()`; the
/// `addCompletedHandler` block removes after the GPU retires;
/// `wait_for_gpu_retire` looks up by range to do a kernel-blocked
/// wait. The retain held here is what keeps the cmdbuf addressable
/// after `commit()` returns ownership to Metal — Metal's queue keeps
/// its own refcount, but we need a stable pointer to call
/// `waitUntilCompleted` on. The completion handler always removes,
/// so the map size is bounded by in-flight frames.
static PENDING_CMDBUFS: Mutex<BTreeMap<u64, PendingCmdBuf>> = Mutex::new(BTreeMap::new());

/// Block until `coherent_seq >= target_seq` by calling `waitUntilCompleted`.
///
/// The wait targets the registered cmdbuf for the smallest in-flight seq
/// ≥ target. Metal's queue is in-order, so waiting on that one implicitly
/// waits on every earlier one too. After the wait we `fetch_max` the
/// atomic ourselves: the completion handler may not have fired yet (it
/// runs on Metal's own dispatch queue), and our caller needs to observe
/// `coherent_seq >= target_seq` on return.
pub fn wait_for_gpu_retire(target_seq: u64, coherent_seq_ptr: u64) {
    if coherent_seq_ptr == 0 || target_seq == 0 {
        return;
    }
    // SAFETY: PE-side `Arc<AtomicU64>::as_ptr()` was handed across; the Arc
    // outlives every in-flight command buffer that references it (device
    // teardown drains pending cmdbufs before dropping the Arc).
    let atomic = unsafe { &*(coherent_seq_ptr as *const AtomicU64) };
    if atomic.load(Ordering::Acquire) >= target_seq {
        return;
    }
    let cmdbuf = {
        // Lock dropped before `waitUntilCompleted`; the completion
        // handler removes from the same map and would deadlock if we
        // held the lock across the kernel sleep.
        let map = PENDING_CMDBUFS.lock().unwrap();
        map.range(target_seq..).next().map(|(_, cb)| cb.0.clone())
    };
    let Some(cmdbuf) = cmdbuf else {
        // Either the handler raced ahead of us (already removed) or
        // the caller passed a target the encoder never submitted.
        // Either way, re-check the atomic and trust it.
        return;
    };
    mtld3d_shared::crumb!("gpuretirebeg", target_seq, atomic.load(Ordering::Acquire));
    cmdbuf.waitUntilCompleted();
    mtld3d_shared::crumb!("gpuretireend", target_seq);
    atomic.fetch_max(target_seq, Ordering::Release);
}

// SubmitFrame breadcrumb probes via `mtld3d_shared::crumb!()`. Each
// probe fires *before* the Metal/objc operation it precedes, so on a
// crash the most-recent trail entry uniquely identifies the next call
// site — used to localise `unix_call(SubmitFrame) → status=0xc0000005`
// SIGSEGVs (Wine's unix-call shim translates a unix-side SIGSEGV into
// that PE status, so the PE error log never names the actual crash
// site). When `cfg(mtld3d_crumb)` is off the probes compile to nothing.

/// Origin of a `encode_leading_blits` invocation.
///
/// Used as the bracket label in trace probes (`blit[frame-leading/3]: …`,
/// `blit[pass2/0]: …`). `Display` formats only when the trace macro
/// fires, so the empty-args case allocates nothing.
#[derive(Clone, Copy)]
enum BlitSite {
    FrameLeading,
    Pass(usize),
}

impl core::fmt::Display for BlitSite {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::FrameLeading => f.write_str("frame-leading"),
            Self::Pass(idx) => write!(f, "pass{idx}"),
        }
    }
}

/// Processes a frame into one `MTLCommandBuffer`.
///
/// Encodes each `PassDescriptor` as a distinct `MTLRenderCommandEncoder`
/// with its own attachments and load actions, optionally blits the
/// backbuffer to the drawable, and commits.
pub fn submit_frame(params: &mut SubmitFrameParams) -> bool {
    params.drawable_wait_tsc = 0;
    mtld3d_shared::crumb!("submit:enter", params.queue_handle.raw(), params.pass_count);
    mtld3d_shared::crumb!("submit:queueret", params.queue_handle.raw());
    let Some(queue) = params.queue_handle.into_retained() else {
        error!(target: LOG_TARGET, "submit_frame: queue retain failed (handle={:#x})", params.queue_handle);
        return false;
    };

    mtld3d_shared::crumb!("submit:cmdbuf");
    let Some(cmd_buf) = queue.commandBuffer() else {
        error!(target: LOG_TARGET, "submit_frame: commandBuffer() returned nil");
        return false;
    };
    {
        let label =
            objc2_foundation::NSString::from_str(&format!("mtld3d-frame-{:#x}", params.submit_seq));
        cmd_buf.setLabel(Some(&label));
    }

    if params.blit_commands_ptr != 0 && params.blit_command_count > 0 {
        // SAFETY: PE supplied `blit_commands_ptr` as a `[BlitCommand; count]`
        // valid for the call duration per the SubmitFrame wire contract.
        let blits = unsafe {
            core::slice::from_raw_parts(
                params.blit_commands_ptr as *const BlitCommand,
                params.blit_command_count as usize,
            )
        };
        if params.upload_coherent_seq_ptr != 0 {
            // Separate-upload-CB path: encode the frame-leading blits
            // into their OWN command buffer committed *before* the draw
            // `cmd_buf`. Metal's queue is
            // in-order, so the partial order "all frame-leading blits
            // before all passes" is preserved (same-frame draws still see
            // the uploaded texels) — but this CB retires as soon as the
            // blits finish, ~a frame before the draw CB, and its
            // completion handler advances the PE-side
            // `upload_coherent_seq`. That lets the next frame's texture
            // `LockRect` observe the staging as retired and write in place
            // instead of renaming + memcpying. The draw `cmd_buf` keeps
            // its own `coherent_seq` handler (below) for VB/IB + draws.
            if !submit_upload_cmd_buf(
                &queue,
                blits,
                params.blit_commands_need_encoder != 0,
                params.submit_seq,
                params.upload_coherent_seq_ptr,
            ) {
                return false;
            }
        } else if !encode_leading_blits(
            &cmd_buf,
            blits,
            params.blit_commands_need_encoder != 0,
            BlitSite::FrameLeading,
        ) {
            return false;
        }
    }

    if params.passes_ptr != 0 && params.pass_count > 0 {
        // SAFETY: PE supplied `passes_ptr` as a `[PassDescriptor; pass_count]`
        // valid for the call duration per the SubmitFrame wire contract.
        let passes = unsafe {
            core::slice::from_raw_parts(
                params.passes_ptr as *const PassDescriptor,
                params.pass_count as usize,
            )
        };
        for (pass_idx, pass) in passes.iter().enumerate() {
            if !encode_pass(&cmd_buf, pass, pass_idx) {
                return false;
            }
        }
    }

    // Present: blit backbuffer → drawable
    if !params.present_layer.is_null() {
        mtld3d_shared::crumb!("submit:layerret", params.present_layer.raw());
        let Some(layer) =
            crate::metal::handle::IntoRetainedLayer::into_retained(params.present_layer)
        else {
            cmd_buf.commit();
            return true;
        };
        mtld3d_shared::crumb!("submit:texret", params.present_texture.raw());
        let Some(present_texture) = params.present_texture.into_retained() else {
            cmd_buf.commit();
            return true;
        };

        let drawable_opt = if super::macdrv::window_occluded() {
            // Window fully occluded: the compositor isn't recycling drawables,
            // so `nextDrawable` would block its full timeout for nothing that
            // reaches the screen. Skip the acquire entirely — the command
            // buffer still commits below (the frame's render work executes and
            // the coherent sequence advances), so the pipeline never
            // back-pressures and the guest's render loop keeps running.
            mtld3d_shared::crumb!("submit:occluded-skip", params.present_layer.raw());
            None
        } else {
            mtld3d_shared::crumb!("submit:nextdraw", params.present_layer.raw());
            let drawable = {
                let _wait = CycleSetTimer::start(&raw mut params.drawable_wait_tsc);
                layer.nextDrawable()
            };
            if drawable.is_none() {
                // Visible, yet no drawable within the timeout — a rare
                // compositor stall, or an occlusion signal that hasn't
                // propagated yet. The frame is dropped (committed below without
                // a present); surface the otherwise-silent ~1s stall.
                mtld3d_shared::crumb!(
                    "submit:nodrawable",
                    params.present_layer.raw(),
                    params.drawable_wait_tsc,
                );
            }
            // A nil drawable means `nextDrawable` exhausted its timeout;
            // self-dump the ring on the onset and on recovery so an
            // intermittent stall is captured in the log without manual timing.
            mtld3d_shared::crumb::dump_on_stall_edge(drawable.is_none());
            drawable
        };
        if let Some(drawable) = drawable_opt {
            let drawable_texture = drawable.texture();

            // HDR present: when the layer was configured for EDR at
            // attach (RGBA16Float + ExtendedLinearDisplayP3 + wantsEDR),
            // the drawable expects *linear* float values — a raw blit
            // copy of the game's gamma-encoded BGRA8 backbuffer into
            // an RGBA16Float drawable reinterprets the bytes and
            // produces magenta noise. So once we're on the HDR layer
            // we're committed to running the present shader.
            //
            // Feed the *live* dynamic headroom directly into the
            // shader, with no bootstrap and no latch. When `current >
            // 1.0` the panel is in EDR mode and the BT.2446 curve
            // boosts the midtones to fill that range. When `current ==
            // 1.0` the panel has no EDR headroom right now — either
            // macOS hasn't promoted the screen yet (early frames) or
            // brightness/thermal state physically rules it out for the
            // session. In that case the shader short-circuits to a
            // sRGB→linear pass-through (see `hdr_present.rs`), which
            // writes correct SDR-equivalent values into the
            // ExtendedLinear layer instead of crushing the image with
            // an over-headroom BT.2446 boost. macOS global-scales
            // content that exceeds the current EDR ceiling
            // (multiplies every pixel by `current_max /
            // requested_peak`), so any peak > current is a guaranteed
            // visible regression — the OS clamps and dims the entire
            // image. Following the live ceiling avoids that entirely.
            let presented = if super::macdrv::hdr_active() {
                let view_ptr = params.present_view.raw() as *mut c_void;
                let current = super::macdrv::poll_current_headroom(view_ptr);
                super::macdrv::log_headroom_change_if_any(current, view_ptr);
                encode_hdr_present(&cmd_buf, &present_texture, &drawable_texture, current)
            } else {
                false
            };
            if !presented && let Some(blit) = cmd_buf.blitCommandEncoder() {
                let label = objc2_foundation::NSString::from_str("mtld3d-present-blit");
                blit.setLabel(Some(&label));
                let width = present_texture.width();
                let height = present_texture.height();

                mtld3d_shared::crumb!(
                    "submit:pblit",
                    params.present_texture.raw(),
                    (width << 32) | height,
                );
                // SAFETY: objc2 typed binding; both textures are non-nil
                // retained protocol objects valid for the call.
                unsafe {
                    blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                        &present_texture,
                        0, 0,
                        MTLOrigin { x: 0, y: 0, z: 0 },
                        MTLSize { width, height, depth: 1 },
                        &drawable_texture,
                        0, 0,
                        MTLOrigin { x: 0, y: 0, z: 0 },
                    );
                }

                blit.endEncoding();
            }

            mtld3d_shared::crumb!("submit:present", params.drawable_wait_tsc);
            // Throttle presents to `1/panel_max_hz` when the guest asked
            // for vsync (PE-side `D3DPRESENT_INTERVAL_*` mapping). On a
            // ProMotion panel the system adapts the panel rate to whatever
            // sub-max cadence we sustain under the cap, so fractional
            // production rates display at their actual rate. `0.0` means
            // free-run (D3DPRESENT_INTERVAL_IMMEDIATE) — drop the throttle.
            let drawable_obj = ProtocolObject::from_ref(&*drawable);
            let min_duration = super::macdrv::min_present_duration_sec();
            if min_duration > 0.0 {
                cmd_buf.presentDrawable_afterMinimumDuration(drawable_obj, min_duration);
            } else {
                cmd_buf.presentDrawable(drawable_obj);
            }
        }
    }

    // Register an addCompletedHandler that bumps the PE-side
    // `coherent_seq` atomic when this frame retires on the GPU. The
    // block runs on a Metal-internal dispatch thread. `fetch_max` makes
    // out-of-order retirement safe. Consumers on the encoder thread
    // read the atomic directly to drain retention queues. The same
    // handler also removes our `PENDING_CMDBUFS` entry — the registry
    // keeps the cmdbuf reachable for `wait_for_gpu_retire`'s
    // `waitUntilCompleted` call until Metal signals completion.
    if params.coherent_seq_ptr != 0 && params.submit_seq > 0 {
        let atomic_ptr = usize::try_from(params.coherent_seq_ptr)
            .expect("PE wire pointer fits host address space (unix is 64-bit)");
        let seq = params.submit_seq;
        let handler = RcBlock::new(
            move |_cb: core::ptr::NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                // SAFETY: the PE side allocated an `Arc<AtomicU64>` and
                // passed its pointer. The Arc is kept alive for the
                // device's lifetime, and all command buffers that reference
                // it are drained on device teardown before the Arc drops.
                let atomic = unsafe { &*(atomic_ptr as *const AtomicU64) };
                atomic.fetch_max(seq, Ordering::Release);
                mtld3d_shared::crumb!("submit:retire", seq);
                let _ = PENDING_CMDBUFS.lock().unwrap().remove(&seq);
            },
        );
        // SAFETY: objc2 typed binding; `handler` is kept alive on the stack
        // until `commit()` below, at which point Metal has retained the block.
        unsafe { cmd_buf.addCompletedHandler(RcBlock::as_ptr(&handler)) };

        // Register the cmdbuf for `wait_for_gpu_retire` lookups before
        // committing. Cloning a `Retained` is a refcount bump.
        PENDING_CMDBUFS
            .lock()
            .unwrap()
            .insert(seq, PendingCmdBuf(cmd_buf.clone()));
    }

    mtld3d_shared::crumb!("submit:commit");
    cmd_buf.commit();
    mtld3d_shared::crumb!("submit:done");
    true
}

/// Encode the frame-leading (texture-upload) blits into a dedicated command buffer.
///
/// It is committed *before* the draw CB. Its completion handler
/// `fetch_max`es `submit_seq` into the PE-side `upload_coherent_seq`
/// atomic, so the next frame's contended texture `LockRect` can observe
/// the upload as retired and write in place instead of renaming +
/// memcpying. Because Metal's queue is in-order and this CB is committed
/// before the draw CB, the uploads still finish before any same-frame
/// draw samples them.
///
/// Deliberately NOT registered in `PENDING_CMDBUFS`: nothing ever waits
/// on an upload seq via `wait_for_gpu_retire`, and the draw CB retiring
/// (which *is* registered, under the same `submit_seq`) already implies
/// this earlier-committed CB retired — registering it too would put two
/// command buffers under one key. Returns `false` if command-buffer
/// creation or blit encoding failed.
fn submit_upload_cmd_buf(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    blits: &[BlitCommand],
    need_encoder: bool,
    submit_seq: u64,
    upload_coherent_seq_ptr: u64,
) -> bool {
    mtld3d_shared::crumb!("submit:upcmdbuf");
    let Some(upload_cb) = queue.commandBuffer() else {
        error!(target: LOG_TARGET, "submit_frame: upload commandBuffer() returned nil");
        return false;
    };
    {
        let label = objc2_foundation::NSString::from_str(&format!("mtld3d-upload-{submit_seq:#x}"));
        upload_cb.setLabel(Some(&label));
    }
    if !encode_leading_blits(&upload_cb, blits, need_encoder, BlitSite::FrameLeading) {
        return false;
    }
    if submit_seq > 0 {
        let atomic_ptr = usize::try_from(upload_coherent_seq_ptr)
            .expect("PE wire pointer fits host address space (unix is 64-bit)");
        let seq = submit_seq;
        let handler = RcBlock::new(
            move |_cb: core::ptr::NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                // SAFETY: the PE side allocated an `Arc<AtomicU64>` and
                // passed its pointer. The Arc is kept alive for the
                // device's lifetime, and all command buffers that
                // reference it are drained on device teardown before the
                // Arc drops.
                let atomic = unsafe { &*(atomic_ptr as *const AtomicU64) };
                atomic.fetch_max(seq, Ordering::Release);
                mtld3d_shared::crumb!("submit:upretire", seq);
            },
        );
        // SAFETY: objc2 typed binding; Metal copies the block on
        // `addCompletedHandler`, so the local `handler` may drop after.
        unsafe { upload_cb.addCompletedHandler(RcBlock::as_ptr(&handler)) };
    }
    mtld3d_shared::crumb!("submit:upcommit");
    upload_cb.commit();
    true
}

/// HDR present pass: the game's `BGRA8` backbuffer onto the drawable's `RGBA16Float` surface.
///
/// Rendered via a fullscreen triangle that sRGB-decodes each sample and
/// multiplies by the EDR boost factor.
///
/// Returns `false` (with a once-warn at the call site of
/// `ensure_resources`) if pipeline creation failed; the caller falls
/// back to the SDR blit-present so the frame still surfaces.
fn encode_hdr_present(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    src: &ProtocolObject<dyn MTLTexture>,
    dst: &ProtocolObject<dyn MTLTexture>,
    peak: f32,
) -> bool {
    let device = cmd_buf.device();
    let Some(resources) = super::hdr_present::ensure_resources(&device) else {
        return false;
    };
    // Pick the pass-through pipeline when the panel reports no EDR
    // headroom this frame, BT.2446 otherwise. The two pipelines share
    // the vertex stage and the sRGB EOTF; pass-through skips the
    // BT.2446 math and requires no uniforms. See `hdr_present.rs` for
    // the per-pipeline rationale.
    let pipeline_handle = if peak <= 1.0 {
        resources.pipeline_passthrough
    } else {
        resources.pipeline_bt2446
    };
    // SAFETY: pipeline_handle is a previously-retained MTLRenderPipelineState address.
    let Some(pipeline) =
        (unsafe { MetalHandle::<MTLRenderPipelineStateKind>::new(pipeline_handle) })
            .into_retained()
    else {
        return false;
    };

    let pass_desc = MTLRenderPassDescriptor::new();
    // SAFETY: `colorAttachments()` returns a non-null descriptor array;
    // subscript 0 is always valid.
    let color0 = unsafe { pass_desc.colorAttachments().objectAtIndexedSubscript(0) };
    color0.setTexture(Some(dst));
    color0.setLoadAction(MTLLoadAction::DontCare); // fullscreen triangle covers every pixel
    color0.setStoreAction(MTLStoreAction::Store);

    let Some(enc) = cmd_buf.renderCommandEncoderWithDescriptor(&pass_desc) else {
        return false;
    };
    let label = objc2_foundation::NSString::from_str("mtld3d-present-pass");
    enc.setLabel(Some(&label));
    enc.setRenderPipelineState(&pipeline);
    // SAFETY: objc2 typed binding; `src` is a retained `MTLTexture` live
    // for the call.
    unsafe {
        enc.setFragmentTexture_atIndex(Some(src), 0);
    }
    if peak > 1.0 {
        // Fragment uniform block consumed by the BT.2446 pipeline:
        // { float l_hdr_nits; float p_hdr; float log2_p_hdr;
        //   float inv_p_minus_one; } — 16 bytes. MSL alignment for
        // `constant T&` requires 16-byte alignment; a stack array of
        // four f32 is naturally aligned and fits.
        //
        // BT.2446-A takes the target peak in nits, not a multiplier;
        // Apple anchors scRGB 1.0 = 100 nits, so L_hdr = peak × 100.
        // `p_hdr`, `log2(p_hdr)` and `1 / (p_hdr - 1)` only depend on
        // `l_hdr_nits`, so we pre-compute them once per frame on the
        // CPU instead of re-deriving them in every fragment.
        let l_hdr_nits = peak * 100.0;
        let p_hdr = 32.0_f32.mul_add((l_hdr_nits / 10000.0).powf(1.0 / 2.4), 1.0);
        let log2_p_hdr = p_hdr.log2();
        let inv_p_minus_one = 1.0 / (p_hdr - 1.0);
        let uniforms: [f32; 4] = [l_hdr_nits, p_hdr, log2_p_hdr, inv_p_minus_one];
        // SAFETY: `&uniforms` is a fresh stack reference; the raw pointer is
        // non-null by construction.
        let uniforms_ptr = unsafe {
            core::ptr::NonNull::new_unchecked(
                core::ptr::from_ref(&uniforms).cast::<c_void>().cast_mut(),
            )
        };
        // SAFETY: objc2 typed binding; `uniforms_ptr` borrows the stack
        // slot for the duration of this call, and the encoder copies before
        // returning.
        unsafe {
            enc.setFragmentBytes_length_atIndex(uniforms_ptr, core::mem::size_of_val(&uniforms), 0);
        }
    }
    // SAFETY: objc2 typed binding; pipeline is bound above; no buffer args.
    unsafe {
        enc.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 3);
    }
    enc.endEncoding();
    true
}

/// Pixel formats Apple lists as valid arguments to `blit.generateMipmapsForTexture`.
///
/// Color-renderable and color-filterable. Compressed (BC*) and
/// depth/stencil formats are excluded by Metal at runtime; PE-side
/// `device_create_texture` already drops the autogen flag for
/// `fmt.is_compressed()`, so this guard is defensive against future
/// format additions.
const fn pixel_format_supports_mipgen(fmt: MTLPixelFormat) -> bool {
    matches!(
        fmt,
        MTLPixelFormat::A8Unorm
            | MTLPixelFormat::R8Unorm
            | MTLPixelFormat::R8Snorm
            | MTLPixelFormat::R16Unorm
            | MTLPixelFormat::R16Snorm
            | MTLPixelFormat::R16Float
            | MTLPixelFormat::R32Float
            | MTLPixelFormat::RG8Unorm
            | MTLPixelFormat::RG8Snorm
            | MTLPixelFormat::RG16Unorm
            | MTLPixelFormat::RG16Snorm
            | MTLPixelFormat::RG16Float
            | MTLPixelFormat::RG32Float
            | MTLPixelFormat::RGBA8Unorm
            | MTLPixelFormat::RGBA8Unorm_sRGB
            | MTLPixelFormat::RGBA8Snorm
            | MTLPixelFormat::BGRA8Unorm
            | MTLPixelFormat::BGRA8Unorm_sRGB
            | MTLPixelFormat::RGBA16Unorm
            | MTLPixelFormat::RGBA16Snorm
            | MTLPixelFormat::RGBA16Float
            | MTLPixelFormat::RGBA32Float
            | MTLPixelFormat::RGB10A2Unorm
    )
}

/// Replay the frame's leading blit commands inside a single `MTLBlitCommandEncoder`.
///
/// Runs before any render pass. Preserves ordering between
/// `CopyTextureToTexture` (preserve) and `CopyBufferToTexture` (sub-rect
/// upload) the PE side emits — preserve blits targeting a given texture
/// must precede any sub-rect upload blits targeting that same texture.
fn encode_leading_blits(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    blits: &[BlitCommand],
    needs_encoder: bool,
    site: BlitSite,
) -> bool {
    let to_usize =
        |v: u64| usize::try_from(v).expect("PE wire u64 fits unix host usize (unix is 64-bit)");
    mtld3d_shared::crumb!("blit:enter", blits.len() as u64, u64::from(needs_encoder),);
    // `needs_encoder` is set on the PE side whenever an encoder-bound
    // command (CopyBuffer/Texture variants) was emitted. Without it
    // we'd have to scan the blit list to know whether to create the
    // blit encoder; the PE side already knows, so just trust the flag.
    // Pure-notify frames skip encoder creation entirely.
    let blit = if needs_encoder {
        if let Some(b) = cmd_buf.blitCommandEncoder() {
            let label =
                objc2_foundation::NSString::from_str(&format!("mtld3d-leading-blits-{site}"));
            b.setLabel(Some(&label));
            Some(b)
        } else {
            error!(
                target: LOG_TARGET,
                "encode_leading_blits: blitCommandEncoder() returned nil (count={})",
                blits.len(),
            );
            return false;
        }
    } else {
        None
    };

    for (i, cmd) in blits.iter().enumerate() {
        mtld3d_shared::crumb!("blit:cmd", u64::from(cmd.cmd), i as u64);
        match BlitCommandType::from_repr(cmd.cmd) {
            Some(BlitCommandType::NotifyBufferDidModifyRange) => {
                // CPU-side flag-set on `MTLBuffer`, not an encoder
                // call. Safe to interleave with open encoder commands;
                // also safe outside any encoder.
                // SAFETY: cmd.src_handle is a previously-retained MTLBuffer address.
                let Some(buffer) =
                    (unsafe { MetalHandle::<MTLBufferKind>::new(cmd.src_handle) }).into_retained()
                else {
                    error!(
                        target: LOG_TARGET,
                        "encode_leading_blits: notify buffer retain failed (handle={:#x})",
                        cmd.src_handle,
                    );
                    continue;
                };
                mtld3d_shared::crumb!("blit:modifyrange", cmd.src_handle);
                buffer.didModifyRange(NSRange {
                    location: to_usize(cmd.src_offset),
                    length: to_usize(cmd.byte_size),
                });
            }
            Some(BlitCommandType::CopyBufferToTexture) => {
                let blit = blit.as_ref().expect("non-notify command requires encoder");
                // SAFETY: cmd.src_handle is a previously-retained MTLBuffer address.
                let Some(buffer) =
                    (unsafe { MetalHandle::<MTLBufferKind>::new(cmd.src_handle) }).into_retained()
                else {
                    error!(
                        target: LOG_TARGET,
                        "encode_leading_blits: buffer retain failed (handle={:#x})",
                        cmd.src_handle,
                    );
                    continue;
                };
                // SAFETY: cmd.dst_handle is a previously-retained MTLTexture address.
                let Some(texture) =
                    (unsafe { MetalHandle::<MTLTextureKind>::new(cmd.dst_handle) }).into_retained()
                else {
                    error!(
                        target: LOG_TARGET,
                        "encode_leading_blits: dst texture retain failed (handle={:#x})",
                        cmd.dst_handle,
                    );
                    continue;
                };
                mtld3d_shared::crumb!("blit:buf2tex", cmd.src_handle, cmd.dst_handle);
                // `depth` is the slice count (1 for a 2D texture, >1 for a
                // volume/3D texture) and `bytes_per_image` the per-slice byte
                // stride. For the 2D hot path the PE side passes `depth == 1`
                // and `bytes_per_image == bytes_per_row * region_h`, exactly
                // the values this call computed implicitly before the fields
                // existed — so the 2D copy is byte-identical.
                // SAFETY: objc2 typed binding; `buffer` and `texture` are
                // retained Metal objects live for the call; geometry is
                // bounds-checked by the PE side via the wire contract.
                unsafe {
                    blit.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                        &buffer,
                        to_usize(cmd.src_offset),
                        to_usize(cmd.bytes_per_row),
                        cmd.bytes_per_image as usize,
                        MTLSize {
                            width: cmd.region_w as usize,
                            height: cmd.region_h as usize,
                            depth: cmd.depth as usize,
                        },
                        &texture,
                        0,
                        cmd.mip_level as usize,
                        MTLOrigin {
                            x: cmd.origin_x as usize,
                            y: cmd.origin_y as usize,
                            z: 0,
                        },
                    );
                }
            }
            Some(BlitCommandType::CopyTextureToTexture) => {
                let blit = blit.as_ref().expect("non-notify command requires encoder");
                // SAFETY: cmd.src_handle is a previously-retained MTLTexture address.
                let Some(src) =
                    (unsafe { MetalHandle::<MTLTextureKind>::new(cmd.src_handle) }).into_retained()
                else {
                    error!(
                        target: LOG_TARGET,
                        "encode_leading_blits: src texture retain failed (handle={:#x})",
                        cmd.src_handle,
                    );
                    continue;
                };
                // SAFETY: cmd.dst_handle is a previously-retained MTLTexture address.
                let Some(dst) =
                    (unsafe { MetalHandle::<MTLTextureKind>::new(cmd.dst_handle) }).into_retained()
                else {
                    error!(
                        target: LOG_TARGET,
                        "encode_leading_blits: dst texture retain failed (handle={:#x})",
                        cmd.dst_handle,
                    );
                    continue;
                };
                // Source origin lives in `origin_x`/`origin_y`;
                // destination origin is packed into `dst_offset` as
                // `(dst_y as u64) << 32 | dst_x as u64`. Full-mip
                // preserve sets all of these to 0, so existing callers
                // are unaffected.
                let dst_x = (cmd.dst_offset & 0xFFFF_FFFF) as usize;
                let dst_y = ((cmd.dst_offset >> 32) & 0xFFFF_FFFF) as usize;
                mtld3d_shared::crumb!("blit:tex2tex", cmd.src_handle, cmd.dst_handle);
                // SAFETY: objc2 typed binding; `src`/`dst` are retained Metal
                // textures live for the call; geometry comes from a packed
                // PE-side `BlitCommand` per the wire contract.
                unsafe {
                    blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                        &src,
                        0,
                        cmd.mip_level as usize,
                        MTLOrigin {
                            x: cmd.origin_x as usize,
                            y: cmd.origin_y as usize,
                            z: 0,
                        },
                        MTLSize {
                            width: cmd.region_w as usize,
                            height: cmd.region_h as usize,
                            depth: 1,
                        },
                        &dst,
                        0,
                        cmd.mip_level as usize,
                        MTLOrigin { x: dst_x, y: dst_y, z: 0 },
                    );
                }
            }
            Some(BlitCommandType::CopyBufferToBuffer) => {
                let blit = blit.as_ref().expect("non-notify command requires encoder");
                // SAFETY: cmd.src_handle is a previously-retained MTLBuffer address.
                let Some(src) =
                    (unsafe { MetalHandle::<MTLBufferKind>::new(cmd.src_handle) }).into_retained()
                else {
                    error!(
                        target: LOG_TARGET,
                        "encode_leading_blits: src buffer retain failed (handle={:#x})",
                        cmd.src_handle,
                    );
                    continue;
                };
                // SAFETY: cmd.dst_handle is a previously-retained MTLBuffer address.
                let Some(dst) =
                    (unsafe { MetalHandle::<MTLBufferKind>::new(cmd.dst_handle) }).into_retained()
                else {
                    error!(
                        target: LOG_TARGET,
                        "encode_leading_blits: dst buffer retain failed (handle={:#x})",
                        cmd.dst_handle,
                    );
                    continue;
                };
                mtld3d_shared::crumb!("blit:buf2buf", cmd.src_handle, cmd.dst_handle);
                // SAFETY: objc2 typed binding; `src`/`dst` are retained
                // `MTLBuffer`s live for the call; sizes are PE-side bounded.
                unsafe {
                    blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                        &src,
                        to_usize(cmd.src_offset),
                        &dst,
                        to_usize(cmd.dst_offset),
                        to_usize(cmd.byte_size),
                    );
                }
            }
            Some(BlitCommandType::GenerateMipmaps) => {
                let blit = blit.as_ref().expect("non-notify command requires encoder");
                // SAFETY: cmd.dst_handle is a previously-retained MTLTexture address.
                let Some(texture) =
                    (unsafe { MetalHandle::<MTLTextureKind>::new(cmd.dst_handle) }).into_retained()
                else {
                    error!(
                        target: LOG_TARGET,
                        "encode_leading_blits: mipgen texture retain failed (handle={:#x})",
                        cmd.dst_handle,
                    );
                    continue;
                };
                if texture.mipmapLevelCount() <= 1 {
                    continue;
                }
                if !pixel_format_supports_mipgen(texture.pixelFormat()) {
                    mtld3d_shared::log_once_warn_by!(
                        target: crate::LOG_TARGET,
                        key: texture.pixelFormat().0 as u64,
                        "encode_leading_blits: pixel format {:?} not supported by Metal generateMipmaps — skipped",
                        texture.pixelFormat()
                    );
                    continue;
                }
                mtld3d_shared::crumb!("blit:mipgen", cmd.dst_handle);
                blit.generateMipmapsForTexture(&texture);
            }
            None => {
                mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
                    "encode_leading_blits: unknown BlitCommandType {t} → skipped", t = cmd.cmd
                );
            }
        }
    }

    if let Some(blit) = blit {
        mtld3d_shared::crumb!("blit:endenc");
        blit.endEncoding();
    }
    true
}

fn encode_pass(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    pass: &PassDescriptor,
    pass_idx: usize,
) -> bool {
    let to_usize =
        |v: u64| usize::try_from(v).expect("PE wire u64 fits unix host usize (unix is 64-bit)");
    let to_u32 =
        |v: u64| u32::try_from(v).expect("PE wire u64 low-half fits u32 by packing contract");
    mtld3d_shared::crumb!("pass:enter", pass_idx as u64, pass.command_count);
    // Per-pass leading blits: a `StretchRect` between two D3D9 draws
    // queues a `BlitCommand` against the *next* pass to open, so it
    // orders correctly between the source pass's draws and this pass's
    // draws. Runs in its own `MTLBlitCommandEncoder` before the render
    // encoder begins.
    if pass.leading_blits_ptr != 0 && pass.leading_blits_count > 0 {
        // SAFETY: PE supplied `leading_blits_ptr` as a `[BlitCommand; n]`
        // valid for the call duration per the PassDescriptor wire contract.
        let blits = unsafe {
            core::slice::from_raw_parts(
                pass.leading_blits_ptr as *const BlitCommand,
                pass.leading_blits_count as usize,
            )
        };
        if !encode_leading_blits(
            cmd_buf,
            blits,
            pass.leading_blits_need_encoder != 0,
            BlitSite::Pass(pass_idx),
        ) {
            return false;
        }
    }

    // Blit-only trailing pass: synthesised by the PE side when a
    // StretchRect lands after the last draw of the frame. The leading
    // blits have already run above; there's nothing else to do.
    if pass.color_texture.is_null() && pass.command_count == 0 {
        mtld3d_shared::crumb!("pass:blitonly", pass_idx as u64);
        return true;
    }

    // Render-pass dimensions, captured from the bound attachment textures.
    // Metal requires every `setScissorRect:` to satisfy `x+width ≤ passW` and
    // `y+height ≤ passH` (the pass extent is the minimum over its attachments).
    // A D3D9 app can leave a larger viewport/scissor set when it switches to a
    // smaller render target, so we clamp the scissor to these below — exceeding
    // them is a hard validation error with the debug layer and out-of-bounds
    // (heap-corrupting) behaviour without it.
    let mut rt_width = usize::MAX;
    let mut rt_height = usize::MAX;
    let rp_desc = MTLRenderPassDescriptor::new();
    // Color attachment is optional — Rule G on the PE side strips the
    // color attachment from clear-only passes whose color is wasted
    // (cascade depth-clear sub-passes where the cascade color is just
    // a placeholder). `color_texture == 0` here means "depth-only
    // render pass", which Metal accepts as long as the depth (or
    // stencil) attachment is set.
    if !pass.color_texture.is_null() {
        mtld3d_shared::crumb!("pass:colorret", pass.color_texture.raw());
        let Some(texture) = pass.color_texture.into_retained() else {
            error!(target: LOG_TARGET, "encode_pass: color texture retain failed (handle={:#x})", pass.color_texture);
            return false;
        };
        rt_width = rt_width.min(texture.width());
        rt_height = rt_height.min(texture.height());
        // SAFETY: `colorAttachments()` returns a non-null descriptor array;
        // subscript 0 is always valid.
        let color0 = unsafe { rp_desc.colorAttachments().objectAtIndexedSubscript(0) };
        color0.setTexture(Some(&texture));
        color0.setStoreAction(map_store_action(pass.color_store_action));
        match pass.color_load_action {
            LoadAction::Clear => {
                color0.setLoadAction(MTLLoadAction::Clear);
                color0.setClearColor(objc2_metal::MTLClearColor {
                    red: f64::from(f32::from_bits(pass.clear_r)),
                    green: f64::from(f32::from_bits(pass.clear_g)),
                    blue: f64::from(f32::from_bits(pass.clear_b)),
                    alpha: f64::from(f32::from_bits(pass.clear_a)),
                });
            }
            LoadAction::Load => color0.setLoadAction(MTLLoadAction::Load),
            LoadAction::DontCare => color0.setLoadAction(MTLLoadAction::DontCare),
        }
    } else if pass.depth_texture.is_null() {
        // No color AND no depth attachment with a non-zero command
        // count would be an empty render encoder targeting nothing —
        // shouldn't happen, but bail rather than ask Metal to build a
        // pass descriptor with no attachments.
        error!(
            target: LOG_TARGET,
            "encode_pass[{pass_idx}]: color=0 + depth=0 with cmds={} — skipping",
            pass.command_count,
        );
        return true;
    }

    if !pass.depth_texture.is_null() {
        mtld3d_shared::crumb!("pass:depthret", pass.depth_texture.raw());
        let depth_tex = pass.depth_texture.into_retained();
        if depth_tex.is_none() {
            error!(
                target: LOG_TARGET,
                "encode_pass: depth texture retain failed (handle={:#x})",
                pass.depth_texture,
            );
        }
        if let Some(depth_tex) = depth_tex {
            rt_width = rt_width.min(depth_tex.width());
            rt_height = rt_height.min(depth_tex.height());
            let depth_attach = rp_desc.depthAttachment();
            depth_attach.setTexture(Some(&depth_tex));
            depth_attach.setStoreAction(map_store_action(pass.depth_store_action));
            match pass.depth_load_action {
                LoadAction::Clear => {
                    depth_attach.setLoadAction(MTLLoadAction::Clear);
                    depth_attach.setClearDepth(f64::from(f32::from_bits(pass.depth_clear_value)));
                }
                LoadAction::Load => depth_attach.setLoadAction(MTLLoadAction::Load),
                LoadAction::DontCare => depth_attach.setLoadAction(MTLLoadAction::DontCare),
            }

            let fmt = depth_tex.pixelFormat();
            if fmt == MTLPixelFormat::Depth32Float_Stencil8 {
                let stencil_attach = rp_desc.stencilAttachment();
                stencil_attach.setTexture(Some(&depth_tex));
                // Stencil shares the depth attachment's storage on
                // `Depth32Float_Stencil8`, so the store action mirrors
                // depth — flipping one without the other would either
                // be a Metal validation error or a redundant store.
                stencil_attach.setStoreAction(map_store_action(pass.depth_store_action));
                match pass.depth_load_action {
                    LoadAction::Clear => {
                        stencil_attach.setLoadAction(MTLLoadAction::Clear);
                        stencil_attach.setClearStencil(0);
                    }
                    LoadAction::Load => stencil_attach.setLoadAction(MTLLoadAction::Load),
                    LoadAction::DontCare => {
                        stencil_attach.setLoadAction(MTLLoadAction::DontCare);
                    }
                }
            }
        }
    }

    if !pass.visibility_result_buffer.is_null() {
        mtld3d_shared::crumb!("pass:visret", pass.visibility_result_buffer.raw());
        let vis_buf = pass.visibility_result_buffer.into_retained();
        match vis_buf {
            Some(buf) => rp_desc.setVisibilityResultBuffer(Some(&buf)),
            None => error!(
                target: LOG_TARGET,
                "encode_pass: visibility result buffer retain failed (handle={:#x})",
                pass.visibility_result_buffer,
            ),
        }
    }

    mtld3d_shared::crumb!("pass:rendenc", pass_idx as u64);
    let Some(encoder) = cmd_buf.renderCommandEncoderWithDescriptor(&rp_desc) else {
        error!(
            target: LOG_TARGET,
            "encode_pass: renderCommandEncoderWithDescriptor returned nil (color={:#x}, depth={:#x}, load={:?}, cmds={})",
            pass.color_texture,
            pass.depth_texture,
            pass.color_load_action,
            pass.command_count,
        );
        return false;
    };
    {
        let label = objc2_foundation::NSString::from_str(&format!("mtld3d-pass-{pass_idx}"));
        encoder.setLabel(Some(&label));
    }

    if pass.commands_ptr != 0 && pass.command_count > 0 {
        // SAFETY: PE supplied `commands_ptr` as a `[Command; command_count]`
        // valid for the call duration per the PassDescriptor wire contract.
        let commands = unsafe {
            core::slice::from_raw_parts(
                pass.commands_ptr as *const Command,
                pass.command_count as usize,
            )
        };

        for (i, cmd) in commands.iter().enumerate() {
            mtld3d_shared::crumb!("pass:cmd", u64::from(cmd.cmd), i as u64);
            match CommandType::from_repr(cmd.cmd) {
                Some(CommandType::SetRenderPipelineState) => {
                    // SAFETY: cmd.param_b is a previously-retained MTLRenderPipelineState address.
                    let Some(pipeline) =
                        (unsafe { MetalHandle::<MTLRenderPipelineStateKind>::new(cmd.param_b) })
                            .into_retained()
                    else {
                        continue;
                    };
                    encoder.setRenderPipelineState(&pipeline);
                }
                Some(CommandType::SetViewport) => {
                    let height =
                        u32::try_from(cmd.param_b & 0xFFFF_FFFF).expect("masked to 32 bits");
                    let min_z_bits = u32::try_from(cmd.param_b >> 32).expect("u64 >> 32 fits u32");
                    let min_z = f32::from_bits(min_z_bits);
                    let vp_x = u32::try_from(cmd.param_c & 0xFFFF_FFFF).expect("masked to 32 bits");
                    let max_z_bits = u32::try_from(cmd.param_c >> 32).expect("u64 >> 32 fits u32");
                    let max_z = f32::from_bits(max_z_bits);
                    let vp_y = u32::try_from(cmd.param_d).expect("viewport y packed as u32");
                    let viewport = MTLViewport {
                        originX: f64::from(vp_x),
                        originY: f64::from(vp_y),
                        width: f64::from(cmd.param_a),
                        height: f64::from(height),
                        znear: f64::from(min_z),
                        zfar: f64::from(max_z),
                    };
                    encoder.setViewport(viewport);
                }
                Some(CommandType::SetVertexBytes) => {
                    let ptr = core::ptr::NonNull::new(cmd.param_b as *mut c_void);
                    if let Some(ptr) = ptr {
                        // SAFETY: objc2 typed binding; `ptr` is non-null per
                        // the `Some` branch and `length` matches the PE-side
                        // buffer; encoder copies bytes synchronously.
                        unsafe {
                            encoder.setVertexBytes_length_atIndex(
                                ptr,
                                to_usize(cmd.param_c),
                                cmd.param_a as usize,
                            );
                        }
                    }
                }
                Some(CommandType::DrawPrimitives) => {
                    let prim_type = mtl_primitive_type_or_fallback(cmd.param_a, "DrawPrimitives");
                    // SAFETY: objc2 typed binding; pipeline and resources
                    // already bound by prior commands in the same pass.
                    unsafe {
                        encoder.drawPrimitives_vertexStart_vertexCount(
                            prim_type,
                            to_usize(cmd.param_b),
                            to_usize(cmd.param_c),
                        );
                    }
                }
                Some(CommandType::SetDepthStencilState) => {
                    // SAFETY: cmd.param_b is a previously-retained MTLDepthStencilState address.
                    let Some(state) =
                        (unsafe { MetalHandle::<MTLDepthStencilStateKind>::new(cmd.param_b) })
                            .into_retained()
                    else {
                        continue;
                    };
                    encoder.setDepthStencilState(Some(&state));
                }
                Some(CommandType::SetCullMode) => {
                    let mode = match CullMode::from_repr(cmd.param_a) {
                        Some(CullMode::None) => MTLCullMode::None,
                        Some(CullMode::Front) => MTLCullMode::Front,
                        Some(CullMode::Back) => MTLCullMode::Back,
                        None => {
                            mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
                                "SetCullMode: raw={} unmapped → MTLCullMode::None",
                                cmd.param_a
                            );
                            MTLCullMode::None
                        }
                    };
                    encoder.setCullMode(mode);
                }
                Some(CommandType::SetDepthBias) => {
                    // PE side already scales `depth_bias` to the active
                    // depth format's ULP via
                    // `mtld3d_core::convert::d3d_depth_bias_to_metal`,
                    // so pass straight through. D3D9 has no clamp
                    // analog — hardcode 0.0.
                    let depth_bias = f32::from_bits(cmd.param_a);
                    let slope_scale = f32::from_bits(to_u32(cmd.param_b));
                    encoder.setDepthBias_slopeScale_clamp(depth_bias, slope_scale, 0.0);
                }
                Some(CommandType::SetDepthClipMode) => {
                    let mode = if cmd.param_a != 0 {
                        MTLDepthClipMode::Clip
                    } else {
                        MTLDepthClipMode::Clamp
                    };
                    encoder.setDepthClipMode(mode);
                }
                Some(CommandType::SetFragmentTexture) => {
                    // SAFETY: cmd.param_b is a previously-retained MTLTexture address.
                    let Some(tex) = (unsafe { MetalHandle::<MTLTextureKind>::new(cmd.param_b) })
                        .into_retained()
                    else {
                        continue;
                    };
                    // SAFETY: objc2 typed binding; `tex` is retained for the
                    // duration of the binding (encoder retains the texture).
                    unsafe {
                        encoder.setFragmentTexture_atIndex(Some(&tex), cmd.param_a as usize);
                    }
                }
                Some(CommandType::SetFragmentSamplerState) => {
                    // SAFETY: cmd.param_b is a previously-retained MTLSamplerState address.
                    let Some(sampler) =
                        (unsafe { MetalHandle::<MTLSamplerStateKind>::new(cmd.param_b) })
                            .into_retained()
                    else {
                        continue;
                    };
                    // SAFETY: objc2 typed binding; `sampler` is retained for
                    // the duration of the binding.
                    unsafe {
                        encoder
                            .setFragmentSamplerState_atIndex(Some(&sampler), cmd.param_a as usize);
                    }
                }
                Some(CommandType::SetVertexBytesAt) => {
                    let ptr = cmd.param_b as *const core::ffi::c_void;
                    if ptr.is_null() {
                        continue;
                    }
                    let length = to_usize(cmd.param_c);
                    // SAFETY: non-null branch above guarantees `ptr` is non-null.
                    let nn = unsafe { core::ptr::NonNull::new_unchecked(ptr.cast_mut()) };
                    // SAFETY: objc2 typed binding; encoder copies bytes synchronously.
                    unsafe {
                        encoder.setVertexBytes_length_atIndex(nn, length, cmd.param_a as usize);
                    }
                }
                Some(CommandType::SetFragmentBytesAt) => {
                    let ptr = cmd.param_b as *const core::ffi::c_void;
                    if ptr.is_null() {
                        continue;
                    }
                    let length = to_usize(cmd.param_c);
                    // SAFETY: non-null branch above guarantees `ptr` is non-null.
                    let nn = unsafe { core::ptr::NonNull::new_unchecked(ptr.cast_mut()) };
                    // SAFETY: objc2 typed binding; encoder copies bytes synchronously.
                    unsafe {
                        encoder.setFragmentBytes_length_atIndex(nn, length, cmd.param_a as usize);
                    }
                }
                Some(CommandType::SetScissorRect) => {
                    let req_width = (cmd.param_c >> 32) as usize;
                    let req_height = (cmd.param_c & 0xFFFF_FFFF) as usize;
                    // Clamp to the render-pass extent: a stale viewport/scissor
                    // from a larger render target would otherwise exceed the
                    // bound attachment (Metal validation error / OOB without the
                    // debug layer). Origin past the edge collapses the rect to
                    // empty rather than wrapping negative.
                    let x = (cmd.param_a as usize).min(rt_width);
                    let y = to_usize(cmd.param_b).min(rt_height);
                    let rect = MTLScissorRect {
                        x,
                        y,
                        width: req_width.min(rt_width - x),
                        height: req_height.min(rt_height - y),
                    };
                    encoder.setScissorRect(rect);
                }
                Some(CommandType::SetVertexBuffer) => {
                    // SAFETY: cmd.param_b is a previously-retained MTLBuffer address.
                    let Some(buffer) =
                        (unsafe { MetalHandle::<MTLBufferKind>::new(cmd.param_b) }).into_retained()
                    else {
                        continue;
                    };
                    // SAFETY: objc2 typed binding; `buffer` is retained for
                    // the duration of the binding (encoder retains).
                    unsafe {
                        encoder.setVertexBuffer_offset_atIndex(
                            Some(&buffer),
                            to_usize(cmd.param_c),
                            cmd.param_a as usize,
                        );
                    }
                }
                Some(CommandType::DrawIndexedPrimitives) => {
                    let prim_type =
                        mtl_primitive_type_or_fallback(cmd.param_a, "DrawIndexedPrimitives");
                    // SAFETY: cmd.param_b is a previously-retained MTLBuffer address.
                    let Some(index_buffer) =
                        (unsafe { MetalHandle::<MTLBufferKind>::new(cmd.param_b) }).into_retained()
                    else {
                        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
                            "DrawIndexedPrimitives: index buffer retain failed — draw skipped"
                        );
                        continue;
                    };
                    let index_count = (cmd.param_d >> 8) as usize;
                    let index_type_raw = (cmd.param_d & 0xFF) as u32;
                    let index_type = match IndexType::from_repr(index_type_raw) {
                        Some(IndexType::UInt16) => MTLIndexType::UInt16,
                        Some(IndexType::UInt32) => MTLIndexType::UInt32,
                        None => {
                            mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
                                "DrawIndexedPrimitives: MTLIndexType raw={index_type_raw} unmapped → UInt16"
                            );
                            MTLIndexType::UInt16
                        }
                    };
                    // param_c packs (index_buffer_offset << 32) | (base_vertex as u32).
                    // Low-half extraction must mask explicitly — `to_u32` would
                    // panic on any non-zero offset. The sign of base_vertex is
                    // recovered via the u32→i32 bitcast then widened to isize.
                    let offset = (cmd.param_c >> 32) as usize;
                    let base_vertex_u32 =
                        u32::try_from(cmd.param_c & 0xFFFF_FFFF).expect("masked to 32 bits");
                    let base_vertex = isize::try_from(base_vertex_u32.cast_signed())
                        .expect("i32 fits isize on 64-bit unix");
                    // SAFETY: objc2 typed binding; `index_buffer` is retained
                    // for the call; index count/offset come from the PE-side
                    // packed `param_c`/`param_d` per the wire contract.
                    unsafe {
                        encoder.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount_baseVertex_baseInstance(
                            prim_type,
                            index_count,
                            index_type,
                            &index_buffer,
                            offset,
                            1,
                            base_vertex,
                            0,
                        );
                    }
                }
                Some(CommandType::DrawIndexedPrimitivesUp) => {
                    let prim_type =
                        mtl_primitive_type_or_fallback(cmd.param_a, "DrawIndexedPrimitivesUp");
                    let Some(ptr) = core::ptr::NonNull::new(cmd.param_b as *mut c_void) else {
                        continue;
                    };
                    let byte_len = to_usize(cmd.param_c);
                    let index_count = (cmd.param_d >> 8) as usize;
                    let index_type_raw = (cmd.param_d & 0xFF) as u32;
                    let index_type = match IndexType::from_repr(index_type_raw) {
                        Some(IndexType::UInt16) => MTLIndexType::UInt16,
                        Some(IndexType::UInt32) => MTLIndexType::UInt32,
                        None => {
                            mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
                                "DrawIndexedPrimitivesUp: MTLIndexType raw={index_type_raw} unmapped → UInt16"
                            );
                            MTLIndexType::UInt16
                        }
                    };
                    // Metal has no inline-index draw, so copy the scratch index
                    // bytes into a transient buffer. Metal retains buffers a draw
                    // references until the command buffer completes, so releasing
                    // our handle after encoding is safe.
                    let device = cmd_buf.device();
                    // SAFETY: `ptr` is non-null (checked) and the PE scratch arena
                    // holds `byte_len` readable index bytes for the frame; Metal
                    // copies them into the new buffer.
                    let index_buffer = unsafe {
                        device.newBufferWithBytes_length_options(
                            ptr,
                            byte_len,
                            MTLResourceOptions::StorageModeShared,
                        )
                    };
                    let Some(index_buffer) = index_buffer else {
                        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
                            "DrawIndexedPrimitivesUp: transient index buffer alloc failed — draw skipped"
                        );
                        continue;
                    };
                    // SAFETY: objc2 typed binding; `index_buffer` is retained for
                    // the call; inline UP indices are absolute (base vertex 0,
                    // single instance).
                    unsafe {
                        encoder.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount_baseVertex_baseInstance(
                            prim_type,
                            index_count,
                            index_type,
                            &index_buffer,
                            0,
                            1,
                            0,
                            0,
                        );
                    }
                }
                Some(CommandType::SetVisibilityResultMode) => {
                    let mode_raw = cmd.param_a;
                    let mode = match VisibilityResultMode::from_repr(mode_raw) {
                        Some(VisibilityResultMode::Disabled) => MTLVisibilityResultMode::Disabled,
                        Some(VisibilityResultMode::Boolean) => MTLVisibilityResultMode::Boolean,
                        Some(VisibilityResultMode::Counting) => MTLVisibilityResultMode::Counting,
                        None => {
                            mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
                                "SetVisibilityResultMode: raw={mode_raw} unmapped → Disabled"
                            );
                            MTLVisibilityResultMode::Disabled
                        }
                    };
                    encoder.setVisibilityResultMode_offset(mode, to_usize(cmd.param_b));
                }
                Some(CommandType::SetBlendColor) => {
                    let red = f32::from_bits(cmd.param_a);
                    let green = f32::from_bits(to_u32(cmd.param_b));
                    let blue = f32::from_bits(to_u32(cmd.param_c));
                    let alpha = f32::from_bits(to_u32(cmd.param_d));
                    encoder.setBlendColorRed_green_blue_alpha(red, green, blue, alpha);
                }
                None => {
                    mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "unknown command type {t}", t = cmd.cmd);
                }
            }
        }
    }

    mtld3d_shared::crumb!("pass:endenc", pass_idx as u64);
    encoder.endEncoding();
    true
}

/// Translate a wire `StoreAction` to the corresponding `MTLStoreAction`.
///
/// Trivial today (two variants); centralised so MSAA-resolve variants
/// land in one place when we wire MSAA.
const fn map_store_action(s: StoreAction) -> MTLStoreAction {
    match s {
        StoreAction::Store => MTLStoreAction::Store,
        StoreAction::DontCare => MTLStoreAction::DontCare,
    }
}

/// Decode a wire `PrimitiveType` u32 into `MTLPrimitiveType`.
///
/// Fallback is `Triangle` so an unmapped code doesn't drop the draw
/// silently — the warn fires once per call site, and the pipeline still
/// renders something visible that makes the miswiring obvious.
fn mtl_primitive_type_or_fallback(raw: u32, site: &str) -> MTLPrimitiveType {
    match PrimitiveType::from_repr(raw) {
        Some(PrimitiveType::Point) => MTLPrimitiveType::Point,
        Some(PrimitiveType::Line) => MTLPrimitiveType::Line,
        Some(PrimitiveType::LineStrip) => MTLPrimitiveType::LineStrip,
        Some(PrimitiveType::Triangle) => MTLPrimitiveType::Triangle,
        Some(PrimitiveType::TriangleStrip) => MTLPrimitiveType::TriangleStrip,
        None => {
            mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "{site}: MTLPrimitiveType raw={raw} unmapped → Triangle");
            MTLPrimitiveType::Triangle
        }
    }
}

/// Arguments for `blit_texture_to_buffer`.
///
/// Grouped so the function's argument list stays under the clippy
/// threshold.
pub struct BlitArgs {
    pub queue_handle: MetalHandle<MTLCommandQueueKind>,
    pub device_handle: MetalHandle<MTLDeviceKind>,
    pub tex_handle: MetalHandle<MTLTextureKind>,
    pub dst_ptr: u64,
    pub dst_len: u64,
    pub mip_level: u32,
    pub origin_x: u32,
    pub origin_y: u32,
    pub width: u32,
    pub height: u32,
    pub bytes_per_row: u32,
}

/// Synchronous texture→buffer readback into PE-addressable memory.
///
/// Wraps the caller's page-aligned `dst_ptr / dst_len` via
/// `newBufferWithBytesNoCopy:length:options:deallocator:` (Shared), blits
/// the source texture sub-rect at `mip_level` into it at `bytes_per_row`
/// stride, commits, and waits for completion. On return the caller's
/// memory holds the pixels. Ordering against a prior `submit_frame` call
/// on the same `queue_handle` is guaranteed by Metal's in-order queue
/// execution — this command buffer will not start until the previously
/// committed render command buffer has finished.
pub fn blit_texture_to_buffer(args: &BlitArgs) -> bool {
    use core::{ffi::c_void, ptr::NonNull};

    let to_usize =
        |v: u64| usize::try_from(v).expect("PE wire u64 fits unix host usize (unix is 64-bit)");
    let BlitArgs {
        queue_handle,
        device_handle,
        tex_handle,
        dst_ptr,
        dst_len,
        mip_level,
        origin_x,
        origin_y,
        width,
        height,
        bytes_per_row,
    } = *args;

    if dst_ptr == 0 || dst_len == 0 || width == 0 || height == 0 {
        error!(target: LOG_TARGET, "blit_texture_to_buffer: invalid args");
        return false;
    }
    let Some(queue) = queue_handle.into_retained() else {
        error!(target: LOG_TARGET, "blit_texture_to_buffer: queue retain failed");
        return false;
    };
    let Some(device) = device_handle.into_retained() else {
        error!(target: LOG_TARGET, "blit_texture_to_buffer: device retain failed");
        return false;
    };
    let Some(texture) = tex_handle.into_retained() else {
        error!(target: LOG_TARGET, "blit_texture_to_buffer: texture retain failed");
        return false;
    };

    let Some(ptr) = NonNull::new(dst_ptr as *mut c_void) else {
        error!(target: LOG_TARGET, "blit_texture_to_buffer: null dst_ptr");
        return false;
    };
    // Managed + `synchronizeResource:` so the GPU→CPU readback works on
    // non-UMA Macs (Intel/AMD): the blit writes into VRAM, then the
    // synchronize copies VRAM back to the wrapped PE pages before the
    // CPU read on `waitUntilCompleted` return. On UMA the storage mode
    // collapses to Shared semantics and synchronize is a no-op, so
    // there's no Apple-Silicon overhead.
    // SAFETY: `ptr` is the PE-supplied dst pointer (non-null by the check
    // above); `dst_len` matches its allocation; deallocator is None so the
    // PE allocation is never freed by Metal.
    let Some(dst_buffer) = (unsafe {
        device.newBufferWithBytesNoCopy_length_options_deallocator(
            ptr,
            to_usize(dst_len),
            MTLResourceOptions::StorageModeManaged,
            None,
        )
    }) else {
        error!(target: LOG_TARGET, "blit_texture_to_buffer: newBufferWithBytesNoCopy failed");
        return false;
    };
    {
        let label = objc2_foundation::NSString::from_str("mtld3d-readback");
        dst_buffer.setLabel(Some(&label));
    }

    let Some(cmd_buf) = queue.commandBuffer() else {
        error!(target: LOG_TARGET, "blit_texture_to_buffer: commandBuffer() nil");
        return false;
    };
    {
        let label = objc2_foundation::NSString::from_str("mtld3d-readback");
        cmd_buf.setLabel(Some(&label));
    }
    let Some(blit) = cmd_buf.blitCommandEncoder() else {
        error!(target: LOG_TARGET, "blit_texture_to_buffer: blitCommandEncoder() nil");
        return false;
    };
    {
        let label = objc2_foundation::NSString::from_str("mtld3d-readback-blit");
        blit.setLabel(Some(&label));
    }

    let bytes_per_image = (bytes_per_row as usize) * (height as usize);
    // SAFETY: objc2 typed binding; `texture`/`dst_buffer` are retained Metal
    // objects live for the call; geometry is caller-bounded.
    unsafe {
        blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
            &texture,
            0,
            mip_level as usize,
            MTLOrigin {
                x: origin_x as usize,
                y: origin_y as usize,
                z: 0,
            },
            MTLSize {
                width: width as usize,
                height: height as usize,
                depth: 1,
            },
            &dst_buffer,
            0,
            bytes_per_row as usize,
            bytes_per_image,
        );
        blit.synchronizeResource(ProtocolObject::from_ref(&*dst_buffer));
    }

    blit.endEncoding();
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();
    // dst_buffer drops here — Metal wrapper released, caller's memory
    // untouched (deallocator was None).
    true
}
