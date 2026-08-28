use core::ffi::c_void;
use std::{
    collections::BTreeMap,
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

use block2::RcBlock;
use log::{debug, error, trace};
use mtld3d_shared::{
    BlitCommand, BlitCommandType, Command, CommandType, ExtraColorDesc, MetalHandle,
    NullTextureKind, PassDescriptor, SubmitFrameParams,
    mtl::{
        CullMode, IndexType, LoadAction, PixelFormat, PrimitiveType, StoreAction,
        VisibilityResultMode,
    },
    mtl_handle::{
        MTLBufferKind, MTLCommandQueueKind, MTLDepthStencilStateKind, MTLDeviceKind,
        MTLRenderPipelineStateKind, MTLSamplerStateKind, MTLTextureKind,
    },
    perf::NanosSetTimer,
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSRange;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLClearColor, MTLCommandBuffer, MTLCommandBufferStatus,
    MTLCommandEncoder, MTLCommandQueue, MTLCullMode, MTLDevice, MTLDrawable, MTLIndexType,
    MTLLoadAction, MTLOrigin, MTLPixelFormat, MTLPrimitiveType, MTLRenderCommandEncoder,
    MTLRenderPassDescriptor, MTLResource, MTLResourceOptions, MTLScissorRect, MTLSize,
    MTLStoreAction, MTLTexture, MTLViewport, MTLVisibilityResultMode,
};
use objc2_metal_fx::MTLFXSpatialScalerColorProcessingMode;
use objc2_quartz_core::CAMetalDrawable;

use crate::{
    LOG_TARGET,
    metal::{handle::IntoRetained, null_texture, texture::mtl_pixel_format},
};

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

/// Log sub-target of the presented-cadence probe.
///
/// Inherits `mtld3d::unix` filters by prefix; `mtld3d::unix::present=trace`
/// turns on the per-frame rows without touching anything else.
const PRESENT_LOG_TARGET: &str = "mtld3d::unix::present";

/// Host time (ns) at which the previous drawable reached the screen.
///
/// Half of the presented-cadence probe, see [`register_presented_probe`].
/// Atomics because Metal runs the presented handler on its own thread.
static LAST_PRESENTED_NS: AtomicU64 = AtomicU64::new(0);
/// Exponential running average of the presented interval, ns; 0 = unseeded.
static TYPICAL_PRESENTED_NS: AtomicU64 = AtomicU64::new(0);
/// A presented interval above this is a pause, not a hitch; it reseeds.
const PRESENTED_MAX_INTERVAL_NS: u64 = 500_000_000;
/// Minimum excess over the typical presented interval for a hitch, ns.
///
/// Applied on top of the 1.5x ratio, so jitter on a fast panel stays quiet
/// while one dropped refresh at 120 Hz (8.3 ms to 16.6 ms) registers.
const PRESENTED_MIN_EXCESS_NS: u64 = 3_000_000;

/// Presented-cadence probe: when each frame actually reached the screen.
///
/// Registers a presented handler on the drawable. The handler reads
/// `presentedTime`, the compositor's host time for the frame hitting the
/// panel, keeps an exponential running typical interval, and logs one
/// debug line when an interval exceeds 1.5x the typical one by at least
/// `PRESENTED_MIN_EXCESS_NS`: the interval, the typical interval, the
/// frame's sequence number and its `nextDrawable` wait. Together with the
/// PE-side `frame hitch` line (Present-call cadence on the API thread) it
/// separates a stalled game thread from a frame that was produced on time
/// and displayed late. One block allocation per frame while the target is
/// at debug or below; the caller skips the registration otherwise, and the
/// line only forms on a hitch.
fn register_presented_probe(
    drawable: &ProtocolObject<dyn CAMetalDrawable>,
    seq: u64,
    drawable_wait_ns: u64,
) {
    let handler = RcBlock::new(
        move |d_ptr: core::ptr::NonNull<ProtocolObject<dyn MTLDrawable>>| {
            // SAFETY: Metal invokes the block with the presented drawable;
            // the pointer is valid for the handler's duration.
            let d = unsafe { d_ptr.as_ref() };
            let now_ns = super::macdrv::host_seconds_to_ns(d.presentedTime());
            if now_ns == 0 {
                // Never presented (drawable dropped): leave the chain alone.
                return;
            }
            let last_ns = LAST_PRESENTED_NS.swap(now_ns, Ordering::AcqRel);
            if last_ns == 0 || now_ns <= last_ns {
                return;
            }
            let interval_ns = now_ns - last_ns;
            // Per-frame timeline at trace, the presented-side twin of the
            // PE `present:` row; `t_us` is the absolute host time so rows
            // can be aligned across threads.
            trace!(
                target: PRESENT_LOG_TARGET,
                "presented: seq={seq} interval_us={} wait_us={} t_us={}",
                interval_ns / 1000,
                drawable_wait_ns / 1000,
                now_ns / 1000,
            );
            if interval_ns > PRESENTED_MAX_INTERVAL_NS {
                TYPICAL_PRESENTED_NS.store(0, Ordering::Relaxed);
                return;
            }
            let typical_ns = TYPICAL_PRESENTED_NS.load(Ordering::Relaxed);
            let next_typical = if typical_ns == 0 {
                interval_ns
            } else {
                typical_ns - typical_ns / 16 + interval_ns / 16
            };
            TYPICAL_PRESENTED_NS.store(next_typical, Ordering::Relaxed);
            let hitch = typical_ns != 0
                && interval_ns * 2 > typical_ns * 3
                && interval_ns > typical_ns + PRESENTED_MIN_EXCESS_NS;
            if hitch {
                debug!(
                    target: PRESENT_LOG_TARGET,
                    "presented hitch: presented interval {} us (typical {} us) seq={seq} next_drawable_wait_us={}",
                    interval_ns / 1000,
                    typical_ns / 1000,
                    drawable_wait_ns / 1000,
                );
            }
        },
    );
    // SAFETY: objc2 typed binding; Metal copies (retains) the block on
    // registration, so the stack `handler` may drop when this returns.
    unsafe { drawable.addPresentedHandler(RcBlock::as_ptr(&handler)) };
}

/// Block until `coherent_seq >= target_seq` by calling `waitUntilCompleted`.
///
/// The wait targets the registered cmdbuf for the smallest in-flight seq
/// ≥ target. Metal's queue is in-order, so waiting on that one implicitly
/// waits on every earlier one too. After the wait we `fetch_max` the
/// atomic ourselves: the completion handler may not have fired yet (it
/// runs on Metal's own dispatch queue), and our caller needs to observe
/// `coherent_seq >= target_seq` on return.
///
/// Bumping `coherent_seq` by hand is also why this has to inspect the
/// status: a command buffer the GPU killed is finished, so the bump is
/// right, but recording it as retired and nothing else would hide the
/// abort from the PE-side upload recovery that the completion handlers
/// feed. `failed_submit_seq_ptr` gets the same `fetch_max` they do.
pub fn wait_for_gpu_retire(target_seq: u64, coherent_seq_ptr: u64, failed_submit_seq_ptr: u64) {
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
    // Record the abort before the retirement bump: both stores are
    // `Release`, so a PE-side `Acquire` load of `coherent_seq` that sees
    // this seq is guaranteed to see the failure too.
    if let Some((code, desc)) = record_failed_submit(&cmdbuf, target_seq, failed_submit_seq_ptr) {
        mtld3d_shared::crumb!("gpuretirecberr", target_seq);
        mtld3d_shared::log_once_warn_by!(
            target: LOG_TARGET,
            key: code,
            "wait_for_gpu_retire: command buffer for frame seq={target_seq} failed on the \
             GPU (code {code}: {desc}); everything it carried was discarded",
        );
    }
    atomic.fetch_max(target_seq, Ordering::Release);
}

/// `fetch_max` an aborted command buffer's seq into the PE-side failed-submit counter.
///
/// Returns the Metal error's `(code, localizedDescription)` when the
/// command buffer failed, so the caller can name it in its own tripwire,
/// and `None` when it completed normally. A `failed_submit_seq_ptr` of 0
/// (a frame stamped before the atomic was wired) records nothing and
/// still reports the error.
fn record_failed_submit(
    cb: &ProtocolObject<dyn MTLCommandBuffer>,
    seq: u64,
    failed_submit_seq_ptr: u64,
) -> Option<(u64, String)> {
    if cb.status() != MTLCommandBufferStatus::Error {
        return None;
    }
    if failed_submit_seq_ptr != 0 {
        // SAFETY: the PE side allocated an `Arc<AtomicU64>` and passed its
        // pointer. The Arc is kept alive for the device's lifetime, and all
        // command buffers that reference it are drained on device teardown
        // before the Arc drops.
        let atomic = unsafe { &*(failed_submit_seq_ptr as *const AtomicU64) };
        atomic.fetch_max(seq, Ordering::Release);
    }
    Some(cb.error().map_or_else(
        || (0, String::new()),
        |e| {
            (
                e.code().unsigned_abs() as u64,
                e.localizedDescription().to_string(),
            )
        },
    ))
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
    params.drawable_wait_ns = 0;
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
                params.failed_submit_seq_ptr,
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
            // Re-point the drawable at the layer's backing store before
            // acquiring one. A window resize changes the layer under us and
            // `drawableSize` does not follow on its own, so without this the
            // frames between the resize and the guest's own reaction would
            // be composited at the old size, which means rescaled.
            super::macdrv::sync_drawable_size(&layer);
            mtld3d_shared::crumb!("submit:nextdraw", params.present_layer.raw());
            let drawable = {
                let _wait = NanosSetTimer::start(&raw mut params.drawable_wait_ns);
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
                    params.drawable_wait_ns,
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
            // The back buffer is the grid we rasterized on; the drawable is
            // the layer's own surface. Whatever the two sizes are, present
            // resolves them here — nothing downstream can, since the
            // compositor sees only a finished drawable.
            let device = cmd_buf.device();
            let geometry = PresentGeometry {
                src: (present_texture.width(), present_texture.height()),
                dst: (drawable_texture.width(), drawable_texture.height()),
            };
            // An enlargement the geometry has not settled on yet takes the
            // shader rather than building a scaler for a size that is about
            // to change again. See `SETTLED_PRESENTS`.
            let route = match present_route(
                geometry.src,
                geometry.dst,
                super::upscale::is_available(&device),
            ) {
                PresentRoute::Upscale if !present_geometry_settled(geometry) => {
                    PresentRoute::Stretch
                }
                route => route,
            };
            let hdr = super::macdrv::hdr_active();

            let presented = if hdr {
                // Reads what the main thread last published and queues the next
                // refresh when due. Deriving it here would mean walking
                // NSView.window on this thread, which is what used to crash
                // inside AppKit a few seconds after a zone transition.
                let current = super::macdrv::current_headroom();
                match route {
                    PresentRoute::Upscale => encode_hdr_present_upscaled(
                        &cmd_buf,
                        &present_texture,
                        &drawable_texture,
                        current,
                    ),
                    // The tone-map pass samples through `filter::linear`, so
                    // one encode covers both an exact present and a
                    // minification.
                    PresentRoute::Copy | PresentRoute::Stretch => {
                        encode_hdr_present(&cmd_buf, &present_texture, &drawable_texture, current)
                    }
                }
            } else {
                match route {
                    // Extents match: the blit below is exact and cheaper than
                    // a render pass.
                    PresentRoute::Copy => false,
                    // A scaler Metal declines after `is_available` said yes
                    // still has to write every drawable pixel, so it falls
                    // through to the stretch rather than to the blit.
                    PresentRoute::Upscale => {
                        super::upscale::encode(
                            &cmd_buf,
                            &device,
                            &present_texture,
                            &drawable_texture,
                            MTLFXSpatialScalerColorProcessingMode::Perceptual,
                        ) || encode_present_copy(&cmd_buf, &present_texture, &drawable_texture)
                    }
                    PresentRoute::Stretch => {
                        encode_present_copy(&cmd_buf, &present_texture, &drawable_texture)
                    }
                }
            };
            if !presented {
                if hdr {
                    // No blit fallback on the HDR layer: a `copyFromTexture`
                    // from the BGRA8 backbuffer into an RGBA16Float drawable
                    // is invalid API use, so Metal kills the command buffer
                    // and the drawable is presented with nothing written,
                    // which reads as magenta noise. A defined black frame is
                    // the only correct fallback here.
                    mtld3d_shared::log_once_warn!(
                        target: LOG_TARGET,
                        "present: HDR present pass failed to encode {}x{} → {}x{}; \
                         presenting a cleared drawable instead",
                        geometry.src.0, geometry.src.1, geometry.dst.0, geometry.dst.1,
                    );
                    clear_drawable(&cmd_buf, &drawable_texture);
                } else {
                    encode_present_blit(
                        &cmd_buf,
                        &present_texture,
                        &drawable_texture,
                        route,
                        params.present_texture.raw(),
                    );
                }
            }

            mtld3d_shared::crumb!("submit:present", params.drawable_wait_ns);
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
            // Debug and trace output only, so the per-frame block allocation
            // and handler registration are skipped when the target is off.
            if log::log_enabled!(target: PRESENT_LOG_TARGET, log::Level::Debug) {
                register_presented_probe(&drawable, params.submit_seq, params.drawable_wait_ns);
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
        let failed_seq_ptr = params.failed_submit_seq_ptr;
        let handler = RcBlock::new(
            move |cb_ptr: core::ptr::NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                // Tripwire: a command buffer the GPU rejected discards every
                // encode it carried, but a queued `presentDrawable` still
                // fires, so the drawable reaches the screen with undefined
                // contents (magenta on the RGBA16Float HDR layer). There is
                // no way to un-queue the present at this point; what this
                // buys is that the otherwise silent one-frame flash leaves a
                // log line naming the actual GPU error. Keyed per error code
                // so distinct failure kinds each surface once.
                //
                // SAFETY: Metal invokes the block with the completed command
                // buffer; the pointer is valid for the handler's duration.
                let cb = unsafe { cb_ptr.as_ref() };
                // The failure is recorded before the retirement bump below:
                // both stores are `Release`, so a PE-side `Acquire` load of
                // `coherent_seq` that observes this seq observes the failure
                // too, and the upload recovery never reads a stale "clean".
                if let Some((code, desc)) = record_failed_submit(cb, seq, failed_seq_ptr) {
                    mtld3d_shared::crumb!("submit:cberr", seq);
                    mtld3d_shared::log_once_warn_by!(
                        target: LOG_TARGET,
                        key: code,
                        "submit_frame: command buffer for frame seq={seq} failed on the GPU \
                         (code {code}: {desc}); its rendering was discarded, so a queued \
                         present showed undefined memory",
                    );
                }
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
    failed_submit_seq_ptr: u64,
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
        let failed_seq_ptr = failed_submit_seq_ptr;
        let handler = RcBlock::new(
            move |cb_ptr: core::ptr::NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                // Tripwire: this command buffer carries every frame-leading
                // blit, so an abort here loses texture uploads and `Staged`
                // VB/IB dirty-range copies rather than rendering. Nothing
                // in the API stream ever re-announces them, so the loss is
                // permanent unless the PE side replays it: record the seq
                // before the retirement bump (both `Release`, so a reader
                // that sees the retirement sees the failure) and let the
                // encoder's upload-recovery queues re-issue.
                //
                // SAFETY: Metal invokes the block with the completed command
                // buffer; the pointer is valid for the handler's duration.
                let cb = unsafe { cb_ptr.as_ref() };
                if let Some((code, desc)) = record_failed_submit(cb, seq, failed_seq_ptr) {
                    mtld3d_shared::crumb!("submit:upcberr", seq);
                    mtld3d_shared::log_once_warn_by!(
                        target: LOG_TARGET,
                        key: code,
                        "submit_frame: upload command buffer for frame seq={seq} failed on \
                         the GPU (code {code}: {desc}); every texture and VB/IB upload it \
                         carried was discarded and will be re-issued",
                    );
                }
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

/// How present resolves the back buffer onto the drawable.
///
/// The three arms are the three things Metal can do here, in preference
/// order for the geometry that selects them.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PresentRoute {
    /// Extents match: a 1:1 blit, exact and costing no render pass.
    Copy,
    /// The drawable is larger in both axes and this GPU has `MetalFX`.
    ///
    /// An edge-aware upscale, materially sharper than a bilinear magnify.
    Upscale,
    /// Any other geometry.
    ///
    /// The present shader's filtered stretch, the only route that covers
    /// every drawable pixel at any ratio.
    Stretch,
}

/// One present's source and destination extents.
///
/// Only ever compared, never measured against, so the axes stay in the
/// tuples the Metal texture accessors hand back.
#[derive(Clone, Copy, PartialEq, Eq)]
struct PresentGeometry {
    src: (usize, usize),
    dst: (usize, usize),
}

/// Consecutive presents at one geometry before `MetalFX` is worth building.
///
/// The drawable follows the layer every present, while the back buffer only
/// follows the guest's `WM_SIZE`, so a window being dragged larger produces a
/// *different* enlargement on almost every frame. Each one would be a fresh
/// `newSpatialScalerWithDevice`, an expensive create that the scaler cache
/// then keeps for the life of the process: a hitch per frame of the drag, and
/// a leak that outlives it. Waiting for the geometry to hold still spends a
/// scaler only on sizes the game settled on.
///
/// **Transience, not ratio, is the discriminator.** The tempting alternative
/// is to skip the scaler for enlargements too small to see, but the two cases
/// overlap: a live drag was measured producing ratios up to `1.004`
/// (`2474x1546 → 2484x1552`) while `render.scale = 0.99` asks for `1.0098`.
/// No threshold separates those without being a coincidence.
///
/// Half a second is long enough that a pause mid-drag rarely reaches it, and
/// short enough to be invisible: the frames before it present through the
/// shader's bilinear stretch, and they are frames right after a resize, a
/// `Reset`, or device creation, which are about to change again anyway.
const SETTLED_PRESENTS: u32 = 30;

/// Metal's `setVertexBytes`/`setFragmentBytes` payload cap in bytes.
///
/// A larger payload (a `DrawPrimitiveUP` vertex stream past ~200 vertices)
/// must ride a transient `MTLBuffer` instead; the validation layer rejects
/// the `setBytes` call outright ("length must be <= 4096").
const SET_BYTES_MAX: usize = 4096;

/// Advance the settle counter for `geometry`, and say whether it has settled.
///
/// A change of geometry restarts the count, so "settled" means
/// [`SETTLED_PRESENTS`] presents in a row at the same pair rather than that
/// many presents in total. `seen` is the caller's state so this stays a pure
/// function of it.
fn geometry_settled(seen: &mut Option<(PresentGeometry, u32)>, geometry: PresentGeometry) -> bool {
    match seen {
        Some((last, streak)) if *last == geometry => {
            *streak = streak.saturating_add(1);
            *streak >= SETTLED_PRESENTS
        }
        _ => {
            *seen = Some((geometry, 1));
            SETTLED_PRESENTS <= 1
        }
    }
}

/// [`geometry_settled`] against the encoder thread's own running count.
fn present_geometry_settled(geometry: PresentGeometry) -> bool {
    /// Last present geometry and how many consecutive presents have used it.
    ///
    /// Only `submit_frame` touches this, and only from the encoder thread;
    /// the mutex is for the `static`, not for contention.
    static SEEN: Mutex<Option<(PresentGeometry, u32)>> = Mutex::new(None);

    SEEN.lock()
        .is_ok_and(|mut seen| geometry_settled(&mut seen, geometry))
}

/// Pick the present route for one frame's geometry.
///
/// `MTLBlitCommandEncoder` only copies 1:1 and `MTLFXSpatialScaler` only
/// enlarges, so anything else is the shader's. A drawable larger in one
/// axis and smaller in the other is a stretch, not an upscale: the scaler
/// rejects that pair, and routing it to the blit would leave the axis where
/// the drawable is larger unwritten.
///
/// Whether an enlargement is *worth* a scaler is a separate question, and
/// deliberately not asked here: see [`SETTLED_PRESENTS`].
const fn present_route(
    src: (usize, usize),
    dst: (usize, usize),
    metalfx_available: bool,
) -> PresentRoute {
    if src.0 == dst.0 && src.1 == dst.1 {
        PresentRoute::Copy
    } else if metalfx_available && src.0 <= dst.0 && src.1 <= dst.1 {
        PresentRoute::Upscale
    } else {
        PresentRoute::Stretch
    }
}

/// The 1:1 present blit, and the last resort when a shader route failed to encode.
///
/// SDR only: the drawable and the backbuffer are both `BGRA8Unorm`, so the
/// copy is well-formed. On the HDR layer the caller clears the drawable
/// instead, because a cross-format `copyFromTexture` into `RGBA16Float` is
/// invalid API use that kills the command buffer.
///
/// The copy extent is clamped to the smaller texture in each axis. On the
/// `Copy` route that changes nothing (the extents are equal); on any other
/// route it is what keeps a source larger than the drawable from being an
/// out-of-bounds copy. A clamped copy cannot fill a larger drawable, so the
/// margin is cleared first, because undefined drawable memory reads as
/// noise, and on an `RGBA16Float` layer that noise is magenta.
fn encode_present_blit(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    src: &ProtocolObject<dyn MTLTexture>,
    drawable: &ProtocolObject<dyn MTLTexture>,
    route: PresentRoute,
    src_handle: u64,
) {
    if route != PresentRoute::Copy {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "present: {}x{} → {}x{} needed a resample and none could be encoded; \
             the frame is copied 1:1 into the corner of a cleared drawable",
            src.width(), src.height(), drawable.width(), drawable.height(),
        );
        clear_drawable(cmd_buf, drawable);
    }
    let Some(blit) = cmd_buf.blitCommandEncoder() else {
        return;
    };
    let label = objc2_foundation::NSString::from_str("mtld3d-present-blit");
    blit.setLabel(Some(&label));
    let width = src.width().min(drawable.width());
    let height = src.height().min(drawable.height());

    mtld3d_shared::crumb!("submit:pblit", src_handle, (width << 32) | height);
    // SAFETY: objc2 typed binding; both textures are non-nil retained
    // protocol objects valid for the call, and the extent is clamped to
    // both above.
    unsafe {
        blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
            src,
            0, 0,
            MTLOrigin { x: 0, y: 0, z: 0 },
            MTLSize { width, height, depth: 1 },
            drawable,
            0, 0,
            MTLOrigin { x: 0, y: 0, z: 0 },
        );
    }

    blit.endEncoding();
}

/// Clear the drawable to opaque black with an empty render pass.
///
/// Reached only when a shader route failed to encode, which for a
/// process-lifetime pipeline means it will fail every frame. Black is not a
/// correct frame, but it is a defined one.
fn clear_drawable(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    drawable: &ProtocolObject<dyn MTLTexture>,
) {
    let pass_desc = MTLRenderPassDescriptor::new();
    // SAFETY: `colorAttachments()` returns a non-null descriptor array;
    // subscript 0 is always valid.
    let color0 = unsafe { pass_desc.colorAttachments().objectAtIndexedSubscript(0) };
    color0.setTexture(Some(drawable));
    color0.setLoadAction(MTLLoadAction::Clear);
    color0.setClearColor(MTLClearColor {
        red: 0.0,
        green: 0.0,
        blue: 0.0,
        alpha: 1.0,
    });
    color0.setStoreAction(MTLStoreAction::Store);
    if let Some(enc) = cmd_buf.renderCommandEncoderWithDescriptor(&pass_desc) {
        let label = objc2_foundation::NSString::from_str("mtld3d-present-clear");
        enc.setLabel(Some(&label));
        enc.endEncoding();
    }
}

/// HDR present of a `render.scale`-d frame: tone-map at render size, then upscale.
///
/// The two operations do not commute in the obvious direction. The drawable is
/// `RGBA16Float` and the tone map has to produce it, so `MTLFXSpatialScaler`
/// cannot be the last step *on the back buffer* — but it can be the last step
/// on the tone map's output. Running the present pass into a scratch texture
/// the size of the back buffer and handing that to the scaler in
/// `ColorProcessingMode::HDR` gets both: the frame is tone-mapped, and it is
/// enlarged by the same edge-aware upscaler SDR gets rather than by the present
/// shader's own bilinear sample.
///
/// `HDR` is the mode built for exactly this input — extended-range linear
/// values past `1.0`, which is what the present shader emits (`1.0` = SDR paper
/// white). `MetalFX` applies its own reversible tone map internally to work in
/// `[0, 1]`.
///
/// Keeping the scratch at *render* resolution rather than drawable resolution
/// is what makes this cheap: the ICtCp/PQ math runs over fewer pixels than it
/// does at scale 1.0, and `MetalFX` replaces a full-resolution shader pass.
///
/// Returns `false` when the scratch or the scaler is unavailable, which the
/// caller answers by tone-mapping straight to the drawable — softer, but a
/// frame. Every probe runs before the tone-map pass is encoded, because a
/// decline afterwards would strand it: the present fallback blit is a 1:1 copy
/// and cannot resample. The GPU-capability check comes first of all, so a
/// machine without `MetalFX` never allocates the float scratch it could not
/// consume.
fn encode_hdr_present_upscaled(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    src: &ProtocolObject<dyn MTLTexture>,
    drawable: &ProtocolObject<dyn MTLTexture>,
    peak: f32,
) -> bool {
    let device = cmd_buf.device();
    let width = u32::try_from(src.width()).unwrap_or(u32::MAX);
    let height = u32::try_from(src.height()).unwrap_or(u32::MAX);
    let scratch = if super::upscale::is_available(&device) {
        super::upscale::scratch_target(&device, width, height, PixelFormat::Rgba16Float)
    } else {
        None
    };
    let Some(scratch) = scratch.filter(|scratch| {
        super::upscale::can_scale(
            &device,
            scratch,
            drawable,
            MTLFXSpatialScalerColorProcessingMode::HDR,
        )
    }) else {
        // Unreachable in practice: `render.scale` is held at 1.0 whenever the
        // GPU has no MetalFX (`AttachMetalLayerParams::metalfx_available`), so
        // a scaled frame implies a working scaler.
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "present: no MetalFX HDR upscale for {width}x{height} — the frame is stretched by \
             the present shader instead and will look softer"
        );
        return encode_hdr_present(cmd_buf, src, drawable, peak);
    };

    encode_hdr_present(cmd_buf, src, &scratch, peak)
        && super::upscale::encode(
            cmd_buf,
            &device,
            &scratch,
            drawable,
            MTLFXSpatialScalerColorProcessingMode::HDR,
        )
}

/// HDR present pass: the game's `BGRA8` backbuffer onto an `RGBA16Float` surface.
///
/// Rendered via a fullscreen triangle that sRGB-decodes each sample and
/// multiplies by the EDR boost factor.
///
/// `dst` is the drawable at the default scale, and the render-resolution
/// scratch [`encode_hdr_present_upscaled`] hands to `MetalFX` otherwise. Both
/// are `RGBA16Float`, which is what the present pipelines are built against.
///
/// Returns `false` (with an error at the call site of `ensure_resources`)
/// if pipeline creation failed; the caller falls back to the blit-present
/// so the frame still surfaces.
fn encode_hdr_present(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    src: &ProtocolObject<dyn MTLTexture>,
    dst: &ProtocolObject<dyn MTLTexture>,
    peak: f32,
) -> bool {
    let device = cmd_buf.device();
    let Some(resources) = super::present::ensure_resources(&device) else {
        return false;
    };
    // Pick the pass-through pipeline when the panel reports no EDR
    // headroom this frame, BT.2446 otherwise. The two pipelines share
    // the vertex stage and the sRGB EOTF; pass-through skips the
    // BT.2446 math and requires no uniforms. See `present.rs` for
    // the per-pipeline rationale.
    let (pipeline_handle, uniforms) = if peak <= 1.0 {
        (resources.passthrough, None)
    } else {
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
        (
            resources.bt2446,
            Some([l_hdr_nits, p_hdr, log2_p_hdr, inv_p_minus_one]),
        )
    };
    encode_present_pass(cmd_buf, src, dst, pipeline_handle, uniforms)
}

/// SDR present pass: the game's back buffer onto a same-format drawable, resampled.
///
/// The route for every SDR geometry a blit and `MetalFX` cannot serve: a
/// minification, a mixed-axis change, or a GPU with no `MetalFX` at all.
/// The fragment stage is a plain sample, so an exact-extent call through
/// here is bit-identical to the blit; it is the resample that needs the
/// render pass.
///
/// Returns `false` (with an error at the call site of `ensure_resources`)
/// if pipeline creation failed.
fn encode_present_copy(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    src: &ProtocolObject<dyn MTLTexture>,
    dst: &ProtocolObject<dyn MTLTexture>,
) -> bool {
    let device = cmd_buf.device();
    let Some(resources) = super::present::ensure_resources(&device) else {
        return false;
    };
    encode_present_pass(cmd_buf, src, dst, resources.copy, None)
}

/// Encode one present pass: a fullscreen triangle sampling `src` across `dst`.
///
/// Shared by every shader-driven route. `uniforms` is the BT.2446 fragment
/// block at buffer slot 0; the copy and pass-through pipelines declare no
/// uniforms and pass `None`.
fn encode_present_pass(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    src: &ProtocolObject<dyn MTLTexture>,
    dst: &ProtocolObject<dyn MTLTexture>,
    pipeline_handle: u64,
    uniforms: Option<[f32; 4]>,
) -> bool {
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
    if let Some(uniforms) = uniforms {
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

/// Why Metal would refuse a texture-to-texture blit.
///
/// `copyFromTexture:` validates every one of these itself: under
/// `MTL_DEBUG_LAYER` a violation aborts the process, and without the layer
/// the copy is undefined. The blit encoder tests them first and skips the
/// copy, so a caller that sends a mismatched pair loses one copy rather than
/// the process.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum CopyRejectReason {
    /// The pixel formats are neither equal nor a linear/sRGB twin pair.
    FormatMismatch,
    /// The sample counts differ, which a blit copy cannot resolve.
    SampleCountMismatch,
    /// The source texture has no such mip level.
    SourceLevelMissing,
    /// The destination texture has no such mip level.
    DestinationLevelMissing,
    /// The region leaves the addressed source mip level.
    SourceRegionOutOfBounds,
    /// The region leaves the addressed destination mip level.
    DestinationRegionOutOfBounds,
}

impl CopyRejectReason {
    /// Stable `u64` key so `log_once_warn_by!` fires once per reason.
    ///
    /// Keying on the discriminant keeps the reasons distinct instead of
    /// collapsing every later rejection into the first one seen.
    const fn key(self) -> u64 {
        self as u64
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::FormatMismatch => "source and destination pixel formats are incompatible",
            Self::SampleCountMismatch => "source and destination sample counts differ",
            Self::SourceLevelMissing => "the source has no such mip level",
            Self::DestinationLevelMissing => "the destination has no such mip level",
            Self::SourceRegionOutOfBounds => "the region leaves the source mip level",
            Self::DestinationRegionOutOfBounds => "the region leaves the destination mip level",
        }
    }
}

/// One end of a texture-to-texture blit, as the live `MTLTexture` describes it.
///
/// `width` and `height` are the base level's, `level` the mip level the copy
/// addresses and `levels` the texture's level count, so the addressed extent
/// is derived here rather than passed in alongside them.
struct CopyEndpoint {
    pixel_format: MTLPixelFormat,
    sample_count: usize,
    width: usize,
    height: usize,
    level: usize,
    levels: usize,
    origin_x: usize,
    origin_y: usize,
}

impl CopyEndpoint {
    /// Extent of the addressed mip level, or `None` when it does not exist.
    const fn level_extent(&self) -> Option<(usize, usize)> {
        if self.level >= self.levels {
            return None;
        }
        let w = self.width >> self.level;
        let h = self.height >> self.level;
        Some((if w == 0 { 1 } else { w }, if h == 0 { 1 } else { h }))
    }
}

impl core::fmt::Display for CopyEndpoint {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{:?} samples={} level={}/{} origin={},{} size={}x{}",
            self.pixel_format,
            self.sample_count,
            self.level,
            self.levels,
            self.origin_x,
            self.origin_y,
            self.width,
            self.height
        )
    }
}

/// The sRGB-encoded twin of a pixel format the wire enum covers.
fn srgb_twin_of(format: MTLPixelFormat) -> Option<MTLPixelFormat> {
    let raw = u32::try_from(format.0).ok()?;
    Some(mtl_pixel_format(PixelFormat::from_repr(raw)?.srgb_twin()?))
}

/// Whether `copyFromTexture:` accepts a copy between these two pixel formats.
///
/// Equal formats always. A linear format and its sRGB twin are the one
/// unequal pair Metal accepts, being two encodings of one base format, and
/// mtld3d creates an sRGB view next to every colour texture that has one.
fn pixel_formats_copy_compatible(src: MTLPixelFormat, dst: MTLPixelFormat) -> bool {
    if src == dst {
        return true;
    }
    if srgb_twin_of(src) == Some(dst) {
        return true;
    }
    srgb_twin_of(dst) == Some(src)
}

/// Whether a region placed at `origin` stays inside `extent`.
const fn region_fits(
    origin: (usize, usize),
    region: (usize, usize),
    extent: (usize, usize),
) -> bool {
    match (
        origin.0.checked_add(region.0),
        origin.1.checked_add(region.1),
    ) {
        (Some(right), Some(bottom)) => right <= extent.0 && bottom <= extent.1,
        _ => false,
    }
}

/// Reject a texture-to-texture copy `copyFromTexture:` would not accept.
///
/// `None` means the pair is copyable. One region serves both ends because a
/// blit copy cannot resize, so it is bounds-checked against each of them.
fn copy_texture_reject(
    src: &CopyEndpoint,
    dst: &CopyEndpoint,
    region_w: usize,
    region_h: usize,
) -> Option<CopyRejectReason> {
    if !pixel_formats_copy_compatible(src.pixel_format, dst.pixel_format) {
        return Some(CopyRejectReason::FormatMismatch);
    }
    if src.sample_count != dst.sample_count {
        return Some(CopyRejectReason::SampleCountMismatch);
    }
    let Some(src_extent) = src.level_extent() else {
        return Some(CopyRejectReason::SourceLevelMissing);
    };
    let Some(dst_extent) = dst.level_extent() else {
        return Some(CopyRejectReason::DestinationLevelMissing);
    };
    let region = (region_w, region_h);
    if !region_fits((src.origin_x, src.origin_y), region, src_extent) {
        return Some(CopyRejectReason::SourceRegionOutOfBounds);
    }
    if !region_fits((dst.origin_x, dst.origin_y), region, dst_extent) {
        return Some(CopyRejectReason::DestinationRegionOutOfBounds);
    }
    None
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
                        to_usize(cmd.dst_offset),
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
                let src_endpoint = CopyEndpoint {
                    pixel_format: src.pixelFormat(),
                    sample_count: src.sampleCount(),
                    width: src.width(),
                    height: src.height(),
                    level: cmd.mip_level as usize,
                    levels: src.mipmapLevelCount(),
                    origin_x: cmd.origin_x as usize,
                    origin_y: cmd.origin_y as usize,
                };
                let dst_endpoint = CopyEndpoint {
                    pixel_format: dst.pixelFormat(),
                    sample_count: dst.sampleCount(),
                    width: dst.width(),
                    height: dst.height(),
                    level: cmd.dst_mip_level as usize,
                    levels: dst.mipmapLevelCount(),
                    origin_x: dst_x,
                    origin_y: dst_y,
                };
                let region_w = cmd.region_w;
                let region_h = cmd.region_h;
                if let Some(reason) = copy_texture_reject(
                    &src_endpoint,
                    &dst_endpoint,
                    region_w as usize,
                    region_h as usize,
                ) {
                    let reason_text = reason.as_str();
                    let src_handle = cmd.src_handle;
                    let dst_handle = cmd.dst_handle;
                    mtld3d_shared::log_once_warn_by!(
                        target: crate::LOG_TARGET,
                        key: reason.key(),
                        "encode_leading_blits: {reason_text}, copy skipped. \
                         src handle={src_handle:#x} {src_endpoint}, \
                         dst handle={dst_handle:#x} {dst_endpoint}, \
                         region {region_w}x{region_h}"
                    );
                    continue;
                }
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
                        cmd.dst_mip_level as usize,
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
            pass.leading_blits_need_encoder(),
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
        // A 3D texture addresses its slices through `depthPlane`; `slice`
        // stays 0 there and selects the array/cube face everywhere else.
        // The PE side packs both into one subresource field because the
        // texture's own type is what decides which one Metal wants.
        if texture.textureType() == objc2_metal::MTLTextureType::Type3D {
            color0.setDepthPlane(pass.color_slice() as usize);
        } else {
            color0.setSlice(pass.color_slice() as usize);
        }
        color0.setLevel(pass.color_level() as usize);
        rt_width = rt_width.min((texture.width() >> pass.color_level()).max(1));
        rt_height = rt_height.min((texture.height() >> pass.color_level()).max(1));
        if !pass.color_resolve_texture.is_null() {
            mtld3d_shared::crumb!("pass:colorres", pass.color_resolve_texture.raw());
            let Some(resolve) = pass.color_resolve_texture.into_retained() else {
                error!(target: LOG_TARGET, "encode_pass: colour resolve texture retain failed (handle={:#x})", pass.color_resolve_texture);
                return false;
            };
            color0.setResolveTexture(Some(&resolve));
            // The resolve target is the same D3D9 surface without
            // multisampling, so it takes the attachment's own subresource.
            color0.setResolveSlice(pass.color_slice() as usize);
            color0.setResolveLevel(pass.color_level() as usize);
        }
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
    } else if pass.depth_texture.is_null() && !pass.extra_color.iter().any(ExtraColorDesc::is_bound)
    {
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

    // Render targets 1..3. Same slice/level/load/store handling as attachment
    // 0; the clear colour is the shared one. A stripped slot is simply unbound.
    for (i, extra) in pass.extra_color.iter().enumerate() {
        if !extra.is_bound() {
            continue;
        }
        mtld3d_shared::crumb!("pass:extraret", extra.texture.raw());
        let Some(texture) = extra.texture.into_retained() else {
            error!(target: LOG_TARGET, "encode_pass: color texture {} retain failed (handle={:#x})", i + 1, extra.texture);
            return false;
        };
        // SAFETY: `colorAttachments()` returns a non-null descriptor array;
        // subscripts 1..=3 are within Metal's colour attachment count.
        let color = unsafe { rp_desc.colorAttachments().objectAtIndexedSubscript(i + 1) };
        color.setTexture(Some(&texture));
        color.setSlice(extra.slice() as usize);
        color.setLevel(extra.level() as usize);
        rt_width = rt_width.min((texture.width() >> extra.level()).max(1));
        rt_height = rt_height.min((texture.height() >> extra.level()).max(1));
        if !extra.resolve_texture.is_null() {
            let Some(resolve) = extra.resolve_texture.into_retained() else {
                error!(target: LOG_TARGET, "encode_pass: colour {} resolve texture retain failed (handle={:#x})", i + 1, extra.resolve_texture);
                return false;
            };
            color.setResolveTexture(Some(&resolve));
            color.setResolveSlice(extra.slice() as usize);
            color.setResolveLevel(extra.level() as usize);
        }
        color.setStoreAction(map_store_action(extra.store_action));
        match extra.load_action {
            LoadAction::Clear => {
                color.setLoadAction(MTLLoadAction::Clear);
                color.setClearColor(objc2_metal::MTLClearColor {
                    red: f64::from(f32::from_bits(pass.clear_r)),
                    green: f64::from(f32::from_bits(pass.clear_g)),
                    blue: f64::from(f32::from_bits(pass.clear_b)),
                    alpha: f64::from(f32::from_bits(pass.clear_a)),
                });
            }
            LoadAction::Load => color.setLoadAction(MTLLoadAction::Load),
            LoadAction::DontCare => color.setLoadAction(MTLLoadAction::DontCare),
        }
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
            let level = pass.depth_level();
            rt_width = rt_width.min((depth_tex.width() >> level).max(1));
            rt_height = rt_height.min((depth_tex.height() >> level).max(1));
            let depth_attach = rp_desc.depthAttachment();
            depth_attach.setTexture(Some(&depth_tex));
            depth_attach.setLevel(level as usize);
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
                stencil_attach.setLevel(level as usize);
                // Stencil shares the depth attachment's storage on
                // `Depth32Float_Stencil8`, so the store action mirrors
                // depth — flipping one without the other would either
                // be a Metal validation error or a redundant store.
                stencil_attach.setStoreAction(map_store_action(pass.depth_store_action));
                match pass.stencil_load_action {
                    LoadAction::Clear => {
                        stencil_attach.setLoadAction(MTLLoadAction::Clear);
                        stencil_attach.setClearStencil(pass.stencil_clear_value);
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
                        let length = to_usize(cmd.param_c);
                        if length > SET_BYTES_MAX {
                            // `setVertexBytes` caps at 4 KiB; a UP draw with a
                            // larger inline vertex payload rides a transient
                            // buffer instead. Metal retains buffers a draw
                            // references until the command buffer completes,
                            // so releasing our handle after encoding is safe.
                            let device = cmd_buf.device();
                            // SAFETY: `ptr` is non-null (checked) and the PE
                            // scratch arena holds `length` readable bytes for
                            // the frame; Metal copies them into the new buffer.
                            let vertex_buffer = unsafe {
                                device.newBufferWithBytes_length_options(
                                    ptr,
                                    length,
                                    MTLResourceOptions::StorageModeShared,
                                )
                            };
                            let Some(vertex_buffer) = vertex_buffer else {
                                mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
                                    "SetVertexBytes: transient vertex buffer alloc failed ({length} B) — bind skipped"
                                );
                                continue;
                            };
                            // SAFETY: objc2 typed binding; `vertex_buffer` is
                            // retained for the call.
                            unsafe {
                                encoder.setVertexBuffer_offset_atIndex(
                                    Some(&vertex_buffer),
                                    0,
                                    cmd.param_a as usize,
                                );
                            }
                            continue;
                        }
                        // SAFETY: objc2 typed binding; `ptr` is non-null per
                        // the `Some` branch and `length` matches the PE-side
                        // buffer; encoder copies bytes synchronously.
                        unsafe {
                            encoder.setVertexBytes_length_atIndex(
                                ptr,
                                length,
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
                Some(CommandType::SetVertexTexture) => {
                    // SAFETY: cmd.param_b is a previously-retained MTLTexture address.
                    let Some(tex) = (unsafe { MetalHandle::<MTLTextureKind>::new(cmd.param_b) })
                        .into_retained()
                    else {
                        continue;
                    };
                    // SAFETY: objc2 typed binding; `tex` is retained for the
                    // duration of the binding (encoder retains the texture).
                    unsafe {
                        encoder.setVertexTexture_atIndex(Some(&tex), cmd.param_a as usize);
                    }
                }
                Some(CommandType::SetVertexSamplerState) => {
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
                        encoder.setVertexSamplerState_atIndex(Some(&sampler), cmd.param_a as usize);
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
                Some(CommandType::SetFragmentNullTexture) => {
                    let Some(kind) = NullTextureKind::from_repr(to_u32(cmd.param_b)) else {
                        mtld3d_shared::log_once_warn!(
                            target: LOG_TARGET,
                            "null texture: unknown kind {}; leaving the slot unbound",
                            cmd.param_b,
                        );
                        continue;
                    };
                    let device = cmd_buf.device();
                    let Some(null) = null_texture::ensure(&device) else {
                        continue;
                    };
                    // SAFETY: the handle came from `null_texture::create`'s
                    // `Retained::into_raw`, alive for the process lifetime.
                    let Some(tex) =
                        (unsafe { MetalHandle::<MTLTextureKind>::new(null.texture(kind)) })
                            .into_retained()
                    else {
                        continue;
                    };
                    // SAFETY: as above, for the shared default sampler.
                    let Some(sampler) =
                        (unsafe { MetalHandle::<MTLSamplerStateKind>::new(null.sampler()) })
                            .into_retained()
                    else {
                        continue;
                    };
                    // SAFETY: objc2 typed binding; both are retained for the
                    // binding's duration.
                    unsafe {
                        encoder.setFragmentTexture_atIndex(Some(&tex), cmd.param_a as usize);
                    }
                    // SAFETY: objc2 typed binding; retained for the duration.
                    unsafe {
                        encoder
                            .setFragmentSamplerState_atIndex(Some(&sampler), cmd.param_a as usize);
                    }
                }
                Some(CommandType::SetVertexNullTexture) => {
                    let Some(kind) = NullTextureKind::from_repr(to_u32(cmd.param_b)) else {
                        mtld3d_shared::log_once_warn!(
                            target: LOG_TARGET,
                            "null texture: unknown kind {}; leaving the vertex slot unbound",
                            cmd.param_b,
                        );
                        continue;
                    };
                    let device = cmd_buf.device();
                    let Some(null) = null_texture::ensure(&device) else {
                        continue;
                    };
                    // SAFETY: the handle came from `null_texture::create`'s
                    // `Retained::into_raw`, alive for the process lifetime.
                    let Some(tex) =
                        (unsafe { MetalHandle::<MTLTextureKind>::new(null.texture(kind)) })
                            .into_retained()
                    else {
                        continue;
                    };
                    // SAFETY: as above, for the shared default sampler.
                    let Some(sampler) =
                        (unsafe { MetalHandle::<MTLSamplerStateKind>::new(null.sampler()) })
                            .into_retained()
                    else {
                        continue;
                    };
                    // SAFETY: objc2 typed binding; both are retained for the
                    // binding's duration.
                    unsafe {
                        encoder.setVertexTexture_atIndex(Some(&tex), cmd.param_a as usize);
                    }
                    // SAFETY: objc2 typed binding; retained for the duration.
                    unsafe {
                        encoder.setVertexSamplerState_atIndex(Some(&sampler), cmd.param_a as usize);
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
                Some(CommandType::SetFragmentBuffer) => {
                    // SAFETY: cmd.param_b is a previously-retained MTLBuffer address.
                    let Some(buffer) =
                        (unsafe { MetalHandle::<MTLBufferKind>::new(cmd.param_b) }).into_retained()
                    else {
                        continue;
                    };
                    // SAFETY: objc2 typed binding; `buffer` is retained for
                    // the duration of the binding (encoder retains).
                    unsafe {
                        encoder.setFragmentBuffer_offset_atIndex(
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
                    let (index_count, index_type_raw, instance_count) =
                        Command::unpack_indexed_draw_counts(cmd.param_d);
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
                    // for the call; the counts and offset come from the PE-side
                    // packed `param_c`/`param_d` per the wire contract.
                    unsafe {
                        encoder.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount_baseVertex_baseInstance(
                            prim_type,
                            to_usize(u64::from(index_count)),
                            index_type,
                            &index_buffer,
                            offset,
                            to_usize(u64::from(instance_count.max(1))),
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
                    let (index_count, index_type_raw, instance_count) =
                        Command::unpack_indexed_draw_counts(cmd.param_d);
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
                    // the call; inline UP indices are absolute (base vertex 0).
                    unsafe {
                        encoder.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount_baseVertex_baseInstance(
                            prim_type,
                            to_usize(u64::from(index_count)),
                            index_type,
                            &index_buffer,
                            0,
                            to_usize(u64::from(instance_count.max(1))),
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
                Some(CommandType::SetStencilReference) => {
                    encoder.setStencilReferenceValue(cmd.param_a);
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
const fn map_store_action(s: StoreAction) -> MTLStoreAction {
    match s {
        StoreAction::Store => MTLStoreAction::Store,
        StoreAction::DontCare => MTLStoreAction::DontCare,
        StoreAction::MultisampleResolve => MTLStoreAction::MultisampleResolve,
        StoreAction::StoreAndMultisampleResolve => MTLStoreAction::StoreAndMultisampleResolve,
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
    /// Full logical width the sub-rect coordinates are measured in.
    ///
    /// See `BlitTextureToBufferParams::source_width`. Differs from the source
    /// texture's own width only under a non-default `render.scale`.
    pub source_width: u32,
    /// Full logical height the sub-rect coordinates are measured in.
    pub source_height: u32,
}

/// Resolve a render-resolution source up to the size the caller's coordinates assume.
///
/// Returns `Some(resolved)` when a resolve happened, `None` to read the source
/// as-is, which is both the default-scale path and the fallback when the
/// resolve cannot be set up. Reading as-is after a declined resolve yields a
/// smaller image than requested, so it warns rather than failing the call: a
/// wrong-sized readback is recoverable for the game, a failed `LockRect` often
/// is not.
fn resolve_readback_source(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    device: &ProtocolObject<dyn MTLDevice>,
    texture: &ProtocolObject<dyn MTLTexture>,
    source_width: u32,
    source_height: u32,
) -> Option<Retained<ProtocolObject<dyn MTLTexture>>> {
    if source_width == 0 || source_height == 0 {
        return None;
    }
    let (tex_w, tex_h) = (texture.width(), texture.height());
    if tex_w == source_width as usize && tex_h == source_height as usize {
        return None;
    }
    let resolved = encode_readback_resolve(cmd_buf, device, texture, source_width, source_height);
    if resolved.is_none() {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "readback: could not resolve a {tex_w}x{tex_h} frame up to {source_width}x\
             {source_height}; the caller gets render-resolution pixels"
        );
    }
    resolved
}

/// Resample `src` to `out_w` x `out_h` for a CPU readback, returning the target.
///
/// `GetRenderTargetData`, a back-buffer `LockRect` and `GetDC` all owe the game
/// pixels at the resolution D3D9 reports, but under `render.scale` the back
/// buffer is rasterized smaller. The present pass resamples it into a scratch
/// texture of the reported size, encoded onto `cmd_buf` ahead of the caller's
/// blit encoder so the resolve and the readback are one command buffer and one
/// wait.
///
/// The present pass rather than the `MTLFXSpatialScaler` the display path runs:
/// the scaler writes an opaque alpha, and a game reading the back buffer back
/// is owed the alpha it drew. A plain filtered resample carries all four
/// channels, and reproduces the source exactly wherever the source is flat,
/// which is where a readback is compared against a known colour.
///
/// Returns `None` when the scratch or the pipeline is unavailable, leaving the
/// caller to read `src` directly.
fn encode_readback_resolve(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    device: &ProtocolObject<dyn MTLDevice>,
    src: &ProtocolObject<dyn MTLTexture>,
    out_w: u32,
    out_h: u32,
) -> Option<Retained<ProtocolObject<dyn MTLTexture>>> {
    // The resolve target has to match the source's format, and the only source
    // that ever needs resolving is the back buffer, which is pinned to
    // `BGRA8Unorm`. Gating on that declines rather than guessing if it ever
    // does vary.
    if src.pixelFormat() != MTLPixelFormat::BGRA8Unorm {
        return None;
    }
    let target = super::upscale::scratch_target(device, out_w, out_h, PixelFormat::Bgra8Unorm);
    let Some(target) = target else {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "readback resolve target {out_w}x{out_h} could not be created; readback reads \
             the render-resolution frame instead and will be the wrong size"
        );
        return None;
    };
    if !encode_present_copy(cmd_buf, src, &target) {
        return None;
    }
    Some(target)
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
        source_width,
        source_height,
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
    // Under `render.scale` the source is rasterized smaller than the resolution
    // the caller's coordinates are in, so resolve it up first, on this same
    // command buffer and ahead of the blit encoder: the resolve opens a render
    // pass of its own and Metal allows one encoder at a time. Sizes match at
    // the default scale and this is skipped.
    let source = resolve_readback_source(&cmd_buf, &device, &texture, source_width, source_height);
    let texture = source.as_deref().unwrap_or(&*texture);

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
            texture,
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

#[cfg(test)]
mod tests;
