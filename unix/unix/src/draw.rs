//! Shared draw-path closure body for all four `IDirect3DDevice9::Draw*` entry points.
//!
//! Those four are `DrawPrimitive`, `DrawIndexedPrimitive`, `DrawPrimitiveUP`,
//! `DrawIndexedPrimitiveUP`. Each entry point is a thin wrapper that
//! snapshots D3D9 state into a `DrawContext` on the API thread and hands the
//! context to `emit_draw` on the encoder thread.

use log::{Level, log_enabled};
use mtld3d_core::{
    async_compile::{ClearPlanes, DeferredState, JobTicket, LibrarySlot, Resolution},
    convert::{d3d_depth_bias_to_clip, d3d_to_metal_cull, d3d_to_metal_fill},
    depth_stencil_state::STENCIL_MASK_BITS,
    dirty_range::{indexed_vb_range_lower_bound, nonindexed_vb_range},
    draw_data::{FixedPsSource, FixedVsSource, ProgrammablePsSource, ProgrammableVsSource},
    dxso::{VariantFlags, VariantKey, bound_sampler_type},
    encoder_draw::draw_record::{
        DrawView, IndexView, StreamViewFeed as VertexFeed, VertexView, stream_layouts_view,
    },
    ids::BufferId,
    passes::{
        NULL_TEXTURE_SAMPLER_SENTINEL, Rt0DropCandidate, VertexBufferBind,
        null_texture_tex_sentinel, sampler_cache_key,
    },
    perf::{CycleAddTimer, OpSub, OpSubDetail},
    pipeline_state::{ExtraColorAttachments, PipelineAttachFlags, PipelineSnapshot},
    streams::{instance_count, instanced_stream_read_bytes, is_instance_data},
    vs_draw::{MAX_CLIP_PLANES, VS_DRAW_BYTES, VsDrawState},
};
use mtld3d_shared::{
    Command, MetalHandle, VertexAttrDesc,
    encoder_wire::WireError,
    mtl::{
        IndexType, PS_BOOL_CONST_SLOT, PS_DRAW_SLOT, PS_INT_CONST_SLOT, PS_LOD_BIAS_SLOT,
        PrimitiveType, SET_BYTES_MAX, VS_BOOL_CONST_SLOT, VS_DRAW_SLOT, VS_FLOAT_CONST_SLOT,
        VS_INT_CONST_SLOT, VS_POS_FIXUP_SLOT, VertexStepFunction,
    },
    mtl_handle::MTLFunctionKind,
};
use mtld3d_types::{D3DCMP_ALWAYS, D3DCMP_NEVER, D3DMATRIX, render_state_defaults};

/// `VsDraw` bytes for the default render states.
///
/// The fallback bind when a snapshot reaches `emit_draw` without its own
/// (never expected; warned once).
static VS_DRAW_DEFAULT: std::sync::LazyLock<[u8; VS_DRAW_BYTES]> = std::sync::LazyLock::new(|| {
    VsDrawState::new().build_bytes(
        &render_state_defaults(),
        render_state_defaults()[mtld3d_types::D3DRS_POINTSIZE as usize],
        &D3DMATRIX::IDENTITY,
        &[[0.0; 4]; MAX_CLIP_PLANES],
    )
});

use super::encoder::{FrameEncoder, STAGE_COUNT, StageLibHandles};

/// Sub-target for the per-`(VS, PS, state)` diagnostic from the depth-bias site below.
///
/// Sits under `mtld3d::d3d9::*` like the other diag probes;
/// `RUST_LOG=mtld3d::d3d9::decal=trace` opts in without flipping the
/// broader d3d9 logger.
const DECAL_TRACE_TARGET: &str = "mtld3d::d3d9::decal";

/// Per-unique-caster trace target.
///
/// One row per `(depth_tex, vs_hash, ps_hash, alpha_func, alpha_ref_bits,
/// depth_write, blend_enable, cull_mode)` tuple for draws that target a
/// sampleable depth attachment (cascade shadow map). Opt in with
/// `RUST_LOG=mtld3d::d3d9::caster=trace`; `log_once_trace_by!` keeps cost
/// at one cached atomic load when not enabled. Built for diffing
/// caster-pipeline state between two GPU captures when the visual shadow
/// flickers across runs.
const CASTER_TRACE_TARGET: &str = "mtld3d::d3d9::caster";

pub use mtld3d_core::draw_data::{
    CurrentSnapshot, CurrentSnapshotPtr, DepthStencilFlags, NULL_STREAM_ZEROS, PsKey, PsSourceView,
    RenderStateSnapshot, ScratchSlice, ShaderRef, StageBindingsPtr, VsSourceView,
    arena_alloc_bytes, null_texture_kind,
};

/// Close the `draw N` debug group `emit_draw` opened for a dumped draw.
fn close_dump_group(enc: &mut FrameEncoder, dump_draw: Option<u32>) {
    if dump_draw.is_some() {
        enc.emit_command(Command::pop_debug_group());
    }
}

/// Encoder-thread draw dispatch.
///
/// Pulls the cumulative state from `enc.current_snapshot` (updated by
/// the draw op before dispatch) and combines it with `draw`'s
/// per-call varying parameters (primitive type + vertex/index source).
/// UP spans borrow the API frame capture until native submit replay completes.
/// The depth and stencil planes a draw tests or writes, as a pending build's skip needs them.
fn planes_used(
    render_state: &RenderStateSnapshot,
    target_planes: PipelineAttachFlags,
) -> ClearPlanes {
    let mut planes = ClearPlanes::empty();
    planes.set(
        ClearPlanes::DEPTH,
        target_planes.contains(PipelineAttachFlags::HAS_DEPTH)
            && render_state.depth_stencil_state.depth_enable != 0,
    );
    planes.set(
        ClearPlanes::STENCIL,
        target_planes.contains(PipelineAttachFlags::HAS_STENCIL)
            && render_state.depth_stencil_state.stencil_enable != 0,
    );
    planes
}

/// The libraries of a draw whose first probe found one of them unbuilt.
///
/// Out of line, so the draw that finds both built pays for nothing here.
/// Both stages resolve before any decision, so a draw missing both queues
/// both builds at once. A pending build is then skipped
/// (`FrameEncoder::skip_pending_draw`), deferred to the submission when
/// `may_defer` holds, and waited for otherwise; `None` drops the draw. A
/// deferred draw gets null handles for the libraries still building, and
/// the encoder keeps what they wait for
/// (`FrameEncoder::note_pending_libraries`), so the pipeline resolve that
/// follows binds a placeholder instead.
#[cold]
#[inline(never)]
fn resolve_libraries_slow(
    enc: &mut FrameEncoder,
    shaders: &ShaderRef<'_>,
    planes: ClearPlanes,
    may_defer: bool,
) -> Option<(StageLibHandles, StageLibHandles)> {
    // A build finished since the last frame began may be the one this draw
    // needs; the probes below see it once installed.
    enc.drain_compile_results();
    let mut waited = false;
    loop {
        let vs_resolved = enc.resolve_vs_library(shaders.vs);
        let Some(vs) = library_slot(&vs_resolved) else {
            let dk = shaders.vs.disk_key();
            mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "draw dropped: the VS library failed to build");
            mtld3d_shared::log_once_trace_by!(
                target: crate::LOG_TARGET,
                key: dk,
                "drop: VS {dk:#x} did not resolve",
            );
            return None;
        };
        let ps_resolved = enc.resolve_ps_library(shaders.ps, shaders.variant);
        let Some(ps) = library_slot(&ps_resolved) else {
            let dk = shaders.ps.disk_key(shaders.variant);
            mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "draw dropped: the PS library failed to build");
            mtld3d_shared::log_once_trace_by!(
                target: crate::LOG_TARGET,
                key: dk,
                "drop: PS {dk:#x} did not resolve",
            );
            return None;
        };
        let handles = |resolved: &Resolution<StageLibHandles>| match resolved {
            Resolution::Ready(handles) => *handles,
            Resolution::Pending(_) | Resolution::Failed => StageLibHandles {
                library: MetalHandle::NULL,
                func: MetalHandle::NULL,
            },
        };
        if let (LibrarySlot::Ready(_), LibrarySlot::Ready(_)) = (&vs, &ps) {
            return Some((handles(&vs_resolved), handles(&ps_resolved)));
        }
        if waited {
            mtld3d_shared::log_once_warn!(
                target: crate::LOG_TARGET,
                "draw dropped: its shader library was still unbuilt after waiting for it"
            );
            return None;
        }
        if enc.skip_pending_draw(planes) {
            return None;
        }
        if may_defer {
            let pending = (handles(&vs_resolved), handles(&ps_resolved));
            enc.note_pending_libraries(vs, ps);
            return Some(pending);
        }
        let tickets: Vec<JobTicket> = [vs, ps]
            .into_iter()
            .filter_map(|slot| match slot {
                LibrarySlot::Pending(ticket) => Some(ticket),
                LibrarySlot::Ready(_) => None,
            })
            .collect();
        enc.wait_for_compiles(&tickets);
        waited = true;
    }
}

/// A library resolve as the draw records it; `None` for a failed one.
const fn library_slot(
    resolved: &Resolution<StageLibHandles>,
) -> Option<LibrarySlot<MetalHandle<MTLFunctionKind>>> {
    match resolved {
        Resolution::Ready(handles) => Some(LibrarySlot::Ready(handles.func)),
        Resolution::Pending(ticket) => Some(LibrarySlot::Pending(*ticket)),
        Resolution::Failed => None,
    }
}

/// What a draw's pipeline resolve may change on its way to a pipeline it can bind.
struct PipelineRetry<'a> {
    snapshot: &'a mut PipelineSnapshot,
    rt0_drop: &'a mut bool,
    extra: ExtraColorAttachments,
    attrs: &'a [VertexAttrDesc],
    shaders: &'a ShaderRef<'a>,
    planes: ClearPlanes,
}

/// The pipeline of a draw whose first resolve was pending or failed.
///
/// Out of line for the same reason as [`resolve_libraries_slow`]. A pending
/// build is skipped, or bound as a placeholder whose build the submission
/// waits for; the answer is then the placeholder. A draw leaving render
/// target 0 out waits instead, since the pass it opens is fixed by then and
/// its failed no-colour pipeline retries with render target 0 attached.
/// `None` drops the draw.
#[cold]
#[inline(never)]
fn resolve_pipeline_slow(
    enc: &mut FrameEncoder,
    first: Resolution<u64>,
    retry: PipelineRetry<'_>,
) -> Option<u64> {
    let PipelineRetry {
        snapshot,
        rt0_drop,
        extra,
        attrs,
        shaders,
        planes,
    } = retry;
    let mut resolved = first;
    if matches!(resolved, Resolution::Pending(_)) {
        // As for the libraries: install what finished, then look again.
        enc.drain_compile_results();
        resolved = enc.get_or_create_pipeline(snapshot, attrs, shaders);
    }
    let mut waited = false;
    loop {
        match resolved {
            Resolution::Ready(handle) => return Some(handle),
            Resolution::Pending(ticket) => {
                if waited {
                    mtld3d_shared::log_once_warn!(
                        target: crate::LOG_TARGET,
                        "draw dropped: its pipeline was still unbuilt after waiting for it"
                    );
                    return None;
                }
                if enc.skip_pending_draw(planes) {
                    return None;
                }
                if !*rt0_drop {
                    let id = enc.defer_pipeline(
                        DeferredState::Pipeline(ticket),
                        snapshot,
                        attrs,
                        shaders,
                    );
                    return Some(id.placeholder());
                }
                enc.wait_for_compiles(&[ticket]);
                waited = true;
            }
            Resolution::Failed if *rt0_drop => {
                // The pass keeps render target 0 instead, so the draw still
                // runs, at render target 0's extent, with the pipeline that
                // declares it.
                mtld3d_shared::log_once_warn!(
                    target: crate::LOG_TARGET,
                    "no-colour pipeline for a draw leaving a 1x1 render target 0 out failed: \
                     drawing with render target 0 attached"
                );
                *rt0_drop = false;
                waited = false;
                snapshot
                    .attach
                    .insert(PipelineAttachFlags::HAS_COLOR_OUTPUT);
                snapshot.extra = extra;
            }
            Resolution::Failed => {
                // Pipeline build failed, e.g. a vertex-declaration/shader
                // attribute mismatch (a shader reads `v0` the bound decl
                // never supplies) or a shader that did not compile. Drop the
                // draw, mirroring the VS/PS resolve-failure drops: a render
                // pass that issues `drawPrimitives` with no pipeline bound is
                // undefined in Metal and faults hard at submit (a
                // process-killing SIGSEGV with no recovery), so the draw must
                // never be emitted.
                mtld3d_shared::log_once_warn!(
                    target: crate::LOG_TARGET,
                    "draw dropped: pipeline creation failed (no pipeline bound)"
                );
                return None;
            }
        }
        resolved = enc.get_or_create_pipeline(snapshot, attrs, shaders);
    }
}

/// Execute a draw directly from its retained command record.
///
/// # Safety
/// The view must belong to the authentic admitted packet. Its captured bytes and
/// backing allocations remain immutable and retained until submit completion. The
/// encoder snapshot cache must name initialized snapshots retained by that packet.
///
/// # Errors
/// Rejects malformed draw fields before changing encoder state.
pub unsafe fn emit_draw(enc: &mut FrameEncoder, draw: &DrawView<'_>) -> Result<(), WireError> {
    let metal_prim = draw.metal_primitive()?;
    let vertex_source = draw.vertices()?;
    let index_source = draw.indices()?;
    match &index_source {
        IndexView::Bound { record, .. } => {
            record.index_type()?;
        }
        IndexView::Up { record, .. } | IndexView::Generated { record, .. } => {
            record.index_type()?;
        }
        _ => {}
    }
    emit_draw_view(enc, metal_prim, &vertex_source, &index_source);
    Ok(())
}

fn emit_draw_view(
    enc: &mut FrameEncoder,
    metal_prim: PrimitiveType,
    vertex_source: &VertexView<'_>,
    index_source: &IndexView<'_>,
) {
    // Taken up front: a draw dropped below must not leave its frame-dump
    // index for the next draw to wear.
    let dump_draw = enc.take_dump_draw();
    // Per-draw cost breakdown: six `CycleAddTimer` scopes tile `emit_draw`
    // end to end so the perf summary's "Closures (op)" row decomposes into
    // resolve / pipeline / state / probe / samplers / binds. Each guard holds
    // only a raw counter pointer (no borrow of `enc`), so the measured region
    // reborrows `enc` freely; the explicit `drop` closes one phase before the
    // next begins, and a draw-drop `return` folds the open phase in on the way
    // out. All no-ops unless perf tracking is on.
    let t_resolve = CycleAddTimer::start(enc.op_sub_cycles_ptr(OpSub::Resolve));
    // Lifetime-launder the scratch-resident snapshot ptr off `enc` so
    // the rest of emit_draw can freely reborrow `&mut enc`. SAFETY:
    // the pointee lives in `NativeFrame::scratch`, retained through
    // submission or failure quarantine. A snapshot command in this
    // frame installed the pointer before this draw.
    let snap_ptr = enc
        .current_snapshot_ptr()
        .expect("emit_draw: snapshot not supplied")
        .as_ptr();
    // SAFETY: snap_ptr is non-null (NonNull invariant) and points to
    // a live CurrentSnapshot in NativeFrame::scratch. Its frame owner
    // retains the allocation through submission or failure quarantine,
    // beyond every `enc` reborrow below.
    let snap: &CurrentSnapshot = unsafe { &*snap_ptr };
    // Every Option must be Some by the time a Draw runs — the API
    // thread populates every field before queuing the changed snapshot.
    let render_state: &RenderStateSnapshot = snap
        .render_state
        .as_ref()
        .expect("emit_draw: render_state not populated")
        .as_ref();
    let stage_bindings: &StageBindingsPtr = snap
        .stage_bindings
        .as_ref()
        .expect("emit_draw: stage_bindings not populated");
    let attrs = snap.attrs.expect("emit_draw: attrs not populated");
    let vs: VsSourceView<'_> = snap
        .vs
        .as_ref()
        .expect("emit_draw: vs not populated")
        .as_ref();
    let ps: PsSourceView<'_> = snap
        .ps
        .as_ref()
        .expect("emit_draw: ps not populated")
        .as_ref();
    let variant = snap.variant.expect("emit_draw: variant not populated");
    // The snapshot records what the app bound; the pass records what Metal
    // will accept. A depth surface that disagrees with render target 0 on
    // sample count is dropped at pass open, and a pipeline built for that pass
    // must declare neither a depth nor a stencil format or Metal rejects the
    // draw.
    let pass_binds_depth = enc.pass_binds_depth();
    let mut target_planes = PipelineAttachFlags::empty();
    target_planes.set(
        PipelineAttachFlags::HAS_DEPTH,
        pass_binds_depth && snap.depth_stencil.contains(DepthStencilFlags::HAS_DEPTH),
    );
    target_planes.set(
        PipelineAttachFlags::HAS_STENCIL,
        pass_binds_depth && snap.depth_stencil.contains(DepthStencilFlags::HAS_STENCIL),
    );
    let has_depth = target_planes.contains(PipelineAttachFlags::HAS_DEPTH);
    let has_stencil = target_planes.contains(PipelineAttachFlags::HAS_STENCIL);
    // Bit `i` set ⇒ the pixel shader writes `oCi`; the FF PS writes one output.
    let ps_color_out_mask = match ps {
        PsSourceView::Programmable(ProgrammablePsSource { color_out_mask, .. }) => *color_out_mask,
        PsSourceView::FixedFunction(_) => 1,
    };
    // Whether this draw leaves render target 0 out of its pass, so the depth
    // surface sets the extent (`PassState::rt0_drop_candidate`). Only asked
    // here: the pass is told right before it opens, because the path between
    // can still end the pass or return. A draw that writes render target 0
    // stops at the first test and any target other than a 1x1 one at the
    // candidate's first compare.
    let mut rt0_drop = !render_state.pipeline_rs.writes_rt0(ps_color_out_mask)
        && has_depth
        && match enc.rt0_drop_candidate() {
            Rt0DropCandidate::Yes => true,
            Rt0DropCandidate::No => false,
            Rt0DropCandidate::ScaledDepth => {
                mtld3d_shared::log_once_warn!(
                    target: crate::LOG_TARGET,
                    "render target 0 is a 1x1 target left unwritten over a larger depth surface \
                     that render.scale reduces: drawing at render target 0's extent"
                );
                false
            }
        };
    // A draw that can write nothing is left out before it emits or resolves
    // anything, so the pass list and `last_bound` stay exactly as they were
    // and no bind of it reads a texture. `PassState::skip_dead_draw` owns the
    // conditions. Occlusion queries are ordered by their own ops and a RESZ
    // depth transfer by the `POINTSIZE` write that queued it, so neither
    // moves.
    if enc.skip_dead_draw(
        &render_state.pipeline_rs,
        &render_state.depth_stencil_state,
        ps_color_out_mask,
        target_planes,
    ) {
        return;
    }
    // Stages the bound pixel shader declares a sampler for. Every texture,
    // sampler and per-slot bias bind below is confined to this mask: a stage
    // the game bound a texture to that the shader never samples has no
    // argument in the emitted function, so binding it only adds encoder work.
    let ps_sampled_mask = match ps {
        PsSourceView::Programmable(ProgrammablePsSource { ps_id, .. }) => {
            enc.ps_declared_samplers(*ps_id).mask()
        }
        PsSourceView::FixedFunction(FixedPsSource {
            sampled_stage_mask, ..
        }) => *sampled_stage_mask,
    };
    // `D3DSAMP_MIPMAPLODBIAS` has no Metal sampler equivalent, so the bias
    // reaches the GPU as a fragment uniform the sample sites read. Resolving
    // it here keeps every draw that leaves the state at its zero default on
    // the unbiased shader with nothing extra bound. A bias on a stage this
    // shader does not sample reaches no sample site, so it neither mints the
    // biased variant nor binds the table.
    // Under a reduced `render.scale` the sampler picks its mip from the
    // render grid, coarser than the presented size warrants; `render.lodBias`
    // adds the compensating `log2(scale)` to every stage the shader samples,
    // on top of the game's own bias. Zero at the identity and for a target
    // the scale does not reach, so the default path is untouched.
    let scale_bias = if enc.config().render_lod_bias {
        enc.target_scale().lod_bias()
    } else {
        0.0
    };
    let mut lod_bias = [0.0f32; mtld3d_core::sampler_state::LOD_BIAS_SLOTS];
    let mut any_lod_bias = false;
    for (stage_u32, b) in stage_bindings {
        if (ps_sampled_mask & !variant.fetch4_mask) & (1u16 << stage_u32) == 0 {
            continue;
        }
        let bias = mtld3d_core::sampler_state::lod_bias(&b.sampler_state) + scale_bias;
        if mtld3d_core::sampler_state::lod_bias_active(bias) {
            lod_bias[stage_u32 as usize] = bias;
            any_lod_bias = true;
        }
    }
    // `D3DRS_SRGBWRITEENABLE` picks the pass's colour attachment views, so it
    // has to reach the pass state before anything this draw emits opens a
    // pass. A change ends the current one, since the views are frozen at
    // pass open.
    enc.set_srgb_write_enabled(variant.flags.contains(VariantFlags::SRGB_WRITE));
    // The render pass decides which colour outputs the PS may export, and
    // that is only known here on the encoder thread. Patch a PS-only copy so
    // the VS key, which shares `variant`, is untouched; the FF PS writes one
    // output and keeps the default so its library index never fragments.
    let extra_attachments = enc.current_extra_color_attachments();
    let mut ps_variant = match ps {
        PsSourceView::Programmable(_) => VariantKey {
            color_out_mask: extra_attachments.present_mask << 1,
            ..variant
        },
        PsSourceView::FixedFunction(_) => variant,
    };
    // Coverage consumes the fragment alpha instead of applying ALPHAFUNC.
    // Resolve from the current target so switching back to one sample restores
    // alpha testing without requiring another render-state write.
    if render_state
        .pipeline_rs
        .alpha_to_coverage(enc.current_color_sample_count())
    {
        ps_variant.alpha_func = 0;
    }
    // Point sprites only exist on point primitives: the API thread raises the
    // flag from `D3DRS_POINTSPRITEENABLE` alone, so every other primitive
    // drops it here and keeps its non-sprite library.
    if metal_prim != PrimitiveType::Point {
        ps_variant.flags.remove(VariantFlags::POINT_SPRITE);
    }
    // With sRGB attachment views bound, Metal applies the linear → sRGB OETF
    // after the blender, which is the D3D9 order. The in-shader encode is the
    // fallback for a colour target with no sRGB Metal view, and it must not
    // run as well or the colour is encoded twice.
    if enc.color_attachment_is_srgb() {
        ps_variant.flags.remove(VariantFlags::SRGB_WRITE);
    }
    // A fragment function declaring a depth output against a pass with no
    // depth attachment is a Metal pipeline error, so a programmable PS drops
    // its depth export when no depth buffer is bound (D3D9 discards the
    // write). The FF PS never writes depth and keeps the default key.
    if matches!(ps, PsSourceView::Programmable(_)) {
        ps_variant.flags.set(
            VariantFlags::NO_DEPTH_ATTACHMENT,
            !snap.depth_stencil.contains(DepthStencilFlags::HAS_DEPTH),
        );
    }
    // Both emitters honour the bias, so the flag rides on the shared PS key.
    ps_variant.flags.set(VariantFlags::LOD_BIAS, any_lod_bias);
    // `vPos` is the rasterized pixel coordinate. Into a target rasterized
    // below the resolution D3D9 reports, a shader that declares the register
    // reads it through the `PsDraw` uniform so it stays in the reported space;
    // only such a shader pays for the variant and the bind.
    //
    // The variant flag and the uniform the variant reads are one value,
    // built here from one read of this encoder's bound-target scale: the
    // scale belongs to the device whose encoder this is, and a shader that
    // took the flag can only ever be bound the bytes that came with it.
    let vpos_target_scale = enc.target_scale();
    let ps_draw_bytes = (!vpos_target_scale.is_identity()
        && matches!(ps, PsSourceView::Programmable(ProgrammablePsSource { ps_id, .. }) if enc.ps_reads_vpos(*ps_id)))
    .then(|| mtld3d_core::ps_draw::build_ps_draw_bytes(vpos_target_scale));
    ps_variant
        .flags
        .set(VariantFlags::VPOS_SCALE, ps_draw_bytes.is_some());
    // `D3DRS_MULTISAMPLEMASK`: Metal has no pipeline-state sample mask, so a
    // narrowed mask becomes a `[[sample_mask]]` output in a pixel-shader
    // variant. The API thread already resolved the state against the bound
    // target, so a full mask (the default) never mints a variant.
    if render_state.sample_mask != mtld3d_core::multisample::SAMPLE_MASK_ALL {
        ps_variant.flags.insert(VariantFlags::SAMPLE_MASK);
        ps_variant.sample_mask = render_state.sample_mask;
    }
    // Programmable VS/PS: snapshot from the encoder-side mirror (kept
    // in sync via `Op::Set{Vs,Ps}ConstRange` deltas). FF: symmetric —
    // snapshot from `ff_vs_constants_mirror` (kept in sync via
    // `Op::SetFfVsConstRange`). Each path's `*_const_scratch` bumps
    // fresh scratch bytes per "mirror epoch" so Metal's submit-time
    // `setVertexBytes` copy sees a stable per-draw payload; a shared
    // mirror would let a later draw's constants bleed into an in-flight
    // one.
    let t_consts = CycleAddTimer::start(enc.op_sub_detail_ptr(OpSubDetail::RConsts));
    let vs_constants = match vs {
        VsSourceView::Programmable(value) => {
            let rows = if value.uses_rel_const() {
                enc.vs_constants_populated_rows()
            } else {
                value.max_const_used
            };
            enc.vs_const_scratch(rows)
        }
        VsSourceView::FixedFunction(FixedVsSource { max_row_count, .. }) => {
            enc.ff_vs_const_scratch(*max_row_count)
        }
    };
    let ps_constants = match ps {
        PsSourceView::Programmable(ProgrammablePsSource { max_const_used, .. }) => {
            enc.ps_const_scratch(*max_const_used)
        }
        PsSourceView::FixedFunction(FixedPsSource { constant_rows, .. }) if *constant_rows != 0 => {
            snap.ps_constants.unwrap_or(ScratchSlice::EMPTY)
        }
        PsSourceView::FixedFunction(_) => ScratchSlice::EMPTY,
    };
    let alpha_ref_slice = snap.alpha_ref_bytes.unwrap_or(ScratchSlice::EMPTY);
    let fog_color_slice = snap.fog_color_bytes.unwrap_or(ScratchSlice::EMPTY);
    // SM1 texbem bump-env uniform (slot 12). Only the bound PS knowing it uses
    // a bem-family op pulls the slice — every other draw skips it entirely.
    let ps_uses_bump_env = matches!(ps, PsSourceView::Programmable(value) if value.uses_bump_env());
    let bump_env_slice = if ps_uses_bump_env {
        snap.bump_env_bytes.unwrap_or(ScratchSlice::EMPTY)
    } else {
        ScratchSlice::EMPTY
    };
    // VS integer constants (vertex slot 14). Only a VS that reads a dynamic
    // integer constant pulls the slice; every other draw skips it entirely.
    let vs_uses_int_const =
        matches!(vs, VsSourceView::Programmable(value) if value.uses_int_const());
    let vs_int_const_slice = if vs_uses_int_const {
        snap.vs_int_const_bytes.unwrap_or(ScratchSlice::EMPTY)
    } else {
        ScratchSlice::EMPTY
    };
    // VS boolean constants (vertex slot 26), gated the same way.
    let vs_uses_bool_const =
        matches!(vs, VsSourceView::Programmable(value) if value.uses_bool_const());
    let vs_bool_const_slice = if vs_uses_bool_const {
        snap.vs_bool_const_bytes.unwrap_or(ScratchSlice::EMPTY)
    } else {
        ScratchSlice::EMPTY
    };
    // PS integer / boolean constants (fragment slots 11 / 10), gated by the
    // bound PS the same way.
    let ps_uses_int_const =
        matches!(ps, PsSourceView::Programmable(value) if value.uses_int_const());
    let ps_int_const_slice = if ps_uses_int_const {
        snap.ps_int_const_bytes.unwrap_or(ScratchSlice::EMPTY)
    } else {
        ScratchSlice::EMPTY
    };
    let ps_uses_bool_const =
        matches!(ps, PsSourceView::Programmable(value) if value.uses_bool_const());
    let ps_bool_const_slice = if ps_uses_bool_const {
        snap.ps_bool_const_bytes.unwrap_or(ScratchSlice::EMPTY)
    } else {
        ScratchSlice::EMPTY
    };
    drop(t_consts);

    // 1. Lazily create each bound Metal texture; collect handles. Upload
    //    closures for dirty mips were pushed at `UnlockRect` time and run
    //    earlier in this frame's op list, so by the time we get here the
    //    texture contents are already in place.
    let t_lookup = CycleAddTimer::start(enc.op_sub_detail_ptr(OpSubDetail::RLookup));
    let mut stage_texture_handles: [u64; STAGE_COUNT] = [0; STAGE_COUNT];
    for (stage, b) in stage_bindings {
        if ps_sampled_mask & (1u16 << stage) == 0 {
            // No sampler argument at this slot. Resolving the handle anyway
            // would put a texture no draw reads into the alias check below,
            // where a stage that happens to hold the pass's depth attachment
            // costs a depth copy.
            continue;
        }
        // D3DSAMP_SRGBTEXTURE binds the texture's eager sRGB twin view so
        // the hardware decodes sRGB→linear at sample time. The handle value
        // itself carries the choice, so the `last_bound` dedup below re-emits
        // the bind whenever a stage flips the state on an unchanged texture.
        stage_texture_handles[stage as usize] =
            if mtld3d_core::sampler_state::srgb_texture_enabled(&b.sampler_state) {
                enc.get_texture_handle_by_id_srgb(b.texture_id)
            } else {
                enc.get_texture_sample_handle_by_id(b.texture_id)
            };
    }
    // A draw that samples the bound depth attachment reads a copy of it: Metal
    // forbids reading an attachment of the running pass. D3D9 permits the
    // bind (scene depth as both depth test and position source) with the
    // values as of the last write, which is what the copy holds.
    let depth_attachment = enc.current_depth_texture();
    if !depth_attachment.is_null() && stage_texture_handles.contains(&depth_attachment.raw()) {
        let snapshot = enc.depth_snapshot_for_sampling();
        if snapshot != 0 {
            for handle in &mut stage_texture_handles {
                if *handle == depth_attachment.raw() {
                    *handle = snapshot;
                }
            }
        }
    }
    if !depth_attachment.is_null() && render_state.depth_enable() && render_state.depth_write() {
        enc.bump_depth_write_epoch();
    }
    drop(t_lookup);

    // 2. Resolve the VS and PS libraries independently. The encoder owns
    //    the parsed programs (populated by register_program ops at
    //    CreateShader); `resolve_*_library` handles cache hit/miss + emit
    //    + compile behind the scenes.
    // `debug.skipShaders` bisection: drop this draw if either stage's content
    // hash is in the skip set. Each value is a `pair_id().hash` u64 (the value
    // the per-pass debug log prints as `VS ff 0xN` / `VS prog 0xN` / `PS ff
    // 0xN` / `PS prog 0xN`), stable across frames unlike an index-based skip.
    // The hash (`disk_key`) is computed only when the set is armed — empty in
    // normal play — so the hot path pays nothing.
    let t_keys = CycleAddTimer::start(enc.op_sub_detail_ptr(OpSubDetail::RKeys));
    let skip_set = enc.config().skip_shaders.as_slice();
    if !skip_set.is_empty() {
        let (vs_h, ps_h) = (vs.disk_key(), ps.disk_key(ps_variant));
        // Also match the raw content-hash program ids: they are what the
        // frame dump prints per draw and what names the dumped bytecode
        // files, so a dump line's id can go straight into the skip list
        // without a debug-log run to harvest variant-mixed disk keys.
        let vs_raw = match vs {
            VsSourceView::Programmable(ProgrammableVsSource { vs_id, .. }) => vs_id.raw(),
            VsSourceView::FixedFunction(_) => 0,
        };
        let ps_raw = match ps {
            PsSourceView::Programmable(ProgrammablePsSource { ps_id, .. }) => ps_id.raw(),
            PsSourceView::FixedFunction(_) => 0,
        };
        if skip_set.contains(&vs_h)
            || skip_set.contains(&ps_h)
            || (vs_raw != 0 && skip_set.contains(&vs_raw))
            || (ps_raw != 0 && skip_set.contains(&ps_raw))
        {
            mtld3d_shared::log_once_warn_by!(
                target: crate::LOG_TARGET,
                key: vs_h ^ ps_h,
                "debug.skipShaders: dropping draw with VS {vs_h:#x}/{vs_raw:#x} \
                 PS {ps_h:#x}/{ps_raw:#x}"
            );
            return;
        }
    }
    drop(t_keys);
    // 2. Resolve the VS and PS libraries. The hot path is a borrow-probe of
    //    the source-keyed index (no per-draw content hash, no clone); the
    //    slow path owns the disk-key hash, warm-cache bridge and enqueue.
    //    An unbuilt or failed stage sends the draw there; it installs
    //    finished builds before probing again and queues both stages before
    //    deciding whether to wait, defer or skip the draw.
    let t_lookup = CycleAddTimer::start(enc.op_sub_detail_ptr(OpSubDetail::RLookup));
    let libraries = match enc.lookup_vs_library(vs) {
        Some(Some(vs_handles)) => match enc.lookup_ps_library(ps, ps_variant) {
            Some(Some(ps_handles)) => Some((*vs_handles, *ps_handles)),
            None | Some(None) => None,
        },
        None | Some(None) => None,
    };
    let Some((vs_handles, ps_handles)) = libraries.or_else(|| {
        resolve_libraries_slow(
            enc,
            &ShaderRef {
                vs,
                ps,
                variant: ps_variant,
            },
            planes_used(render_state, target_planes),
            !rt0_drop,
        )
    }) else {
        return;
    };
    drop(t_lookup);
    let shaders = ShaderRef {
        vs,
        ps,
        variant: ps_variant,
    };
    enc.maybe_log_pass_shader(shaders, stage_bindings);
    // One vertex buffer layout per stream the declaration reads: stride and
    // step function from the binding (a zero stride is one constant element,
    // the rest step per the stream's `SetStreamSourceFreq`), a constant zero
    // feed where nothing is bound. Part of the pipeline identity.
    let layouts = stream_layouts_view(vertex_source, &attrs);
    enc.maybe_emit_draw_trace(
        shaders,
        metal_prim,
        vertex_source,
        index_source,
        layouts[0].stride,
    );
    drop(t_resolve);

    let t_pipeline = CycleAddTimer::start(enc.op_sub_cycles_ptr(OpSub::Pipeline));
    let attrs_ref = attrs.as_slice();
    // Instances of an indexed draw: stream 0's frequency count, but only when
    // a stream this draw reads is per-instance; non-indexed draws never
    // instance (D3D9 ignores the frequency state for them).
    let instances = match (vertex_source, index_source) {
        (_, IndexView::None { .. } | IndexView::Fan { .. }) | (VertexView::Up { .. }, _) => 1,
        (VertexView::Bound { stream0_freq, .. }, _) => {
            let any_instanced = vertex_source.bindings().any(|b| {
                attrs.used_streams() & (1 << b.stream) != 0 && is_instance_data(b.frequency)
            });
            instance_count(*stream0_freq, any_instanced)
        }
    };
    let vdecl_hash = attrs.vdecl_hash();
    let alpha_ref_bytes = alpha_ref_slice.as_slice();
    let fog_color_bytes = fog_color_slice.as_slice();
    let bump_env_bytes = bump_env_slice.as_slice();

    // 3. Pipeline + depth state + cull.
    let color_format = enc.current_color_format();
    let mut attach = target_planes | PipelineAttachFlags::HAS_COLOR_OUTPUT;
    // Carry the bound RT's D3D "has alpha" bit so destination-alpha blend
    // factors clamp on alpha-less targets (X8R8G8B8 shares `Bgra8Unorm` with
    // A8R8G8B8, so the color format alone can't distinguish them).
    attach.set(
        PipelineAttachFlags::COLOR_HAS_ALPHA,
        enc.current_color_rt_has_alpha(),
    );
    let mut pipeline_snapshot = PipelineSnapshot {
        vs_fn: vs_handles.func,
        ps_fn: ps_handles.func,
        vdecl_hash,
        stream_layouts: layouts,
        color_format,
        attach,
        rs: render_state.pipeline_rs,
        extra: extra_attachments,
        ps_color_out_mask,
        sample_count: enc.current_color_sample_count(),
    };
    if rt0_drop {
        pipeline_snapshot.remove_color_output();
    }
    let pipeline = match enc.get_or_create_pipeline(&pipeline_snapshot, attrs_ref, &shaders) {
        Resolution::Ready(handle) => handle,
        first @ (Resolution::Pending(_) | Resolution::Failed) => {
            let retry = PipelineRetry {
                snapshot: &mut pipeline_snapshot,
                rt0_drop: &mut rt0_drop,
                extra: extra_attachments,
                attrs: attrs_ref,
                shaders: &shaders,
                planes: planes_used(render_state, target_planes),
            };
            let Some(handle) = resolve_pipeline_slow(enc, first, retry) else {
                return;
            };
            handle
        }
    };
    let depth_stencil = render_state
        .depth_stencil_state
        .gated_on_stencil_attachment(has_stencil);
    let depth_state = if has_depth {
        enc.get_or_create_depth_stencil(&depth_stencil, true)
    } else {
        0
    };
    let metal_cull = d3d_to_metal_cull(u32::from(render_state.cull_mode));
    drop(t_pipeline);

    let t_state = CycleAddTimer::start(enc.op_sub_cycles_ptr(OpSub::State));
    // After every return and every call that can end the pass, so the pass
    // opened or continued next is the one the decision describes.
    enc.set_rt0_dropped(rt0_drop);
    enc.begin_render_pass_if_needed();
    debug_assert_eq!(
        enc.rt0_dropped(),
        !pipeline_snapshot.has_color_output(),
        "the pass leaves render target 0 out exactly when the pipeline declares no colour"
    );
    // Tag the pass with "this draw wants to write color" iff
    // COLORWRITEENABLE is non-zero. When every draw in the pass closes
    // with this still false, Rule H strips the color attachment + swaps
    // the bound pipeline to the no-color variant. Must run after
    // begin_render_pass_if_needed so the tag lands on the right pass.
    enc.note_draw_color_write_mask(u32::from(!pipeline_snapshot.writes_no_color()));
    // Tag the pass with what the draw does to its depth-stencil attachment:
    // a pass none of whose draws test or write it may discard its loads,
    // and a stencil write keeps the stencil plane's stores.
    enc.note_draw_depth_stencil(&depth_stencil, target_planes);
    enc.emit_scissor(
        render_state.scissor_test_enable(),
        render_state.scissor_rect.map(u32::from),
    );
    if enc.last_bound().pipeline_changed(pipeline) {
        enc.emit_command(Command::set_render_pipeline_state(pipeline));
    }
    if depth_state != 0 && enc.last_bound().depth_stencil_changed(depth_state) {
        enc.emit_command(Command::set_depth_stencil_state(depth_state));
    }
    if enc.last_bound().cull_mode_changed(metal_cull) {
        enc.emit_command(Command::set_cull_mode(metal_cull));
    }

    enc.emit_triangle_fill_mode(d3d_to_metal_fill(u32::from(render_state.fill_mode)));

    // `D3DRS_DEPTHBIAS` and `D3DRS_SLOPESCALEDEPTHBIAS`, applied as the game
    // set them. The constant term goes to the vertex shader through
    // `pos_fixup` (emitted below): Metal's own constant bias scales with the
    // depth's exponent on a float depth buffer, D3D9's does not. Only the
    // slope term, which Metal applies unscaled, stays on `setDepthBias`,
    // routed through `LastBoundCache` so it re-binds only when it changes.
    let (min_z, max_z) = enc.viewport_depth_range();
    let depth_bias = d3d_depth_bias_to_clip(render_state.depth_bias, min_z, max_z);
    let slope_scale = f32::from_bits(render_state.slope_scale_depth_bias);
    if enc.last_bound().depth_bias_changed(0.0, slope_scale) {
        enc.emit_command(Command::set_depth_bias(0.0, slope_scale));
    }

    // D3D9 depth-clamps (skips z-clip on) pre-transformed (XYZRHW) geometry
    // while the depth test is inactive; everything else z-clips. Both
    // conjuncts are load-bearing (the D3D9 depth-clamp rule: depth-clamp ⇔
    // depth test inactive AND geometry pre-transformed):
    //  - Case 1: RHW quads spanning z in [-0.5, 1.5] with no depth surface
    //    (or ZENABLE off) draw in full — Metal's always-clip default discards
    //    their outer columns;
    //  - Case 2: the SAME ZENABLE=FALSE state with a regular VS quad still
    //    z-clips (so "depth test inactive" alone is wrong);
    //  - Case 3: RHW with the test LIVE stays clipped under any
    //    D3DRS_CLIPPING value (so "RHW" alone is wrong — clamping every
    //    RHW draw unconditionally would wrongly bypass clipping here).
    // The RHW-with-bound-VS bypass resolves pre-transformed draws to a
    // FixedFunction source, so the FF key's has_rhw covers every RHW draw.
    // Realized as a VS-side clamp selected per-draw through `pos_fixup.z`
    // (emitted below with the half-pixel fixup) rather than
    // `MTLDepthClipMode::Clamp`: encoder-level clamp is not honoured by
    // every Metal device (a GitHub runner's paravirtual GPU clips
    // regardless), while the clamp in the FF vertex shader behaves
    // identically on all of them.
    let position_transformed =
        matches!(vs, VsSourceView::FixedFunction(FixedVsSource { key, .. }) if key.has_rhw());
    let depth_clamp_z = position_transformed && !(has_depth && render_state.depth_enable());
    drop(t_state);

    // Diagnostic probe: per-(VS, PS, state) decal + caster trace rows. Zero
    // cost at the default log level via the explicit `log_enabled!` gate
    // below — it skips the whole key build, not just the trace emit.
    // `RUST_LOG=mtld3d::d3d9::decal=trace` / `…::caster=trace` opt in.
    let t_probe = CycleAddTimer::start(enc.op_sub_cycles_ptr(OpSub::Probe));
    // `note_caster_draw` self-gates on `mtld3d::d3d9::cascade` and the
    // session's sampleable-depth set, so it stays unconditional (one cached
    // atomic load when that probe is off) — this keeps the cascade summary's
    // caster-write counters correct regardless of the trace gate below.
    let depth_tex = enc.current_depth_texture();
    enc.note_caster_draw(depth_tex);
    // Everything else here only feeds the decal/caster trace rows. Gate the
    // whole key+message build on those two targets so a default-log-level
    // draw pays nothing — in particular it skips the `pair_id` content hash
    // the keys would otherwise recompute (an Xxh3 for FF / programmable-PS).
    if log_enabled!(target: DECAL_TRACE_TARGET, Level::Trace)
        || log_enabled!(target: CASTER_TRACE_TARGET, Level::Trace)
    {
        // Gated path: the content hashes are computed here (only when a trace
        // target is on), not on the hot path.
        let vs_hash = vs.disk_key();
        let ps_hash = ps.disk_key(variant);
        let pair_key = vs_hash ^ ps_hash.rotate_left(1);
        // Bake the discriminating render-state bits into the dedup
        // key so a shader pair re-used in distinct (ZFUNC, ZW, AB)
        // configurations produces one trace row per configuration
        // instead of collapsing.
        let state_bits = (u64::from(render_state.depth_stencil_state.depth_func) << 2)
            | (u64::from(render_state.depth_write()) << 1)
            | u64::from(render_state.blend_enable());
        let probe_key = pair_key ^ (state_bits << 58);
        let pass_idx = enc.current_pass_index();
        let alpha_func = variant.alpha_func;
        mtld3d_shared::log_once_trace_by!(
            target: DECAL_TRACE_TARGET,
            key: probe_key,
            "decal: pass={pass_idx} VS prog {vs_hash:#018x} PS prog {ps_hash:#018x} \
             rs[Z={z} ZW={zw} AB={ab} zf={zf} bias={bias:#010x} slope={slope:#010x}] \
             blend[src={src} dst={dst} op={op}] at={alpha_func} \
             applied_clip={depth_bias:e} slope_metal={slope_scale:.3}",
            z = u32::from(render_state.depth_enable()),
            zw = u32::from(render_state.depth_write()),
            ab = u32::from(render_state.blend_enable()),
            zf = render_state.depth_stencil_state.depth_func,
            bias = render_state.depth_bias,
            slope = render_state.slope_scale_depth_bias,
            src = render_state.pipeline_rs.src_blend,
            dst = render_state.pipeline_rs.dst_blend,
            op = render_state.pipeline_rs.blend_op,
        );

        // Caster probe: one row per unique caster-draw signature on a
        // sampleable shadow map (cascade depth attachment). Built to
        // diff caster pipeline state between two captures when tree
        // self-shadow flickers. Combines alpha-test (the hypothesised
        // failure mode for foliage casters), depth-write / blend, and
        // bias to flag any frame-to-frame drift. Self-filters on the
        // session's sampleable-depth set — the same handles
        // `note_caster_draw` above counts — so it fires only on draws
        // into cascade textures, not the main scene depth.
        if !depth_tex.is_null() && enc.is_depth_handle_sampleable(depth_tex) {
            // 32-bit alpha-ref f32 mantissa-truncated into the key; full
            // bits go into the message so legitimate frame-to-frame
            // ref changes show as distinct rows.
            let alpha_ref_bits: u32 = alpha_ref_bytes
                .get(..4)
                .map_or(0, |b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]));
            let caster_state_bits = (u64::from(alpha_func) << 56)
                | (u64::from(render_state.depth_write()) << 55)
                | (u64::from(render_state.blend_enable()) << 54)
                | (u64::from(render_state.cull_mode & 0x3) << 52)
                | u64::from(alpha_ref_bits);
            let caster_key = pair_key ^ caster_state_bits ^ depth_tex.raw().rotate_left(8);
            mtld3d_shared::log_once_trace_by!(
                target: CASTER_TRACE_TARGET,
                key: caster_key,
                "caster: depth=0x{dt:x} VS prog {vs_hash:#018x} PS prog {ps_hash:#018x} \
                 at={alpha_func} aref={aref:#010x} zw={zw} ze={ze} ab={ab} \
                 cull={cull} bias={bias:#010x} slope={slope:#010x}",
                dt = depth_tex,
                aref = alpha_ref_bits,
                ze = u32::from(render_state.depth_enable()),
                zw = u32::from(render_state.depth_write()),
                ab = u32::from(render_state.blend_enable()),
                cull = render_state.cull_mode,
                bias = render_state.depth_bias,
                slope = render_state.slope_scale_depth_bias,
            );
        }
    }
    drop(t_probe);

    let t_samplers = CycleAddTimer::start(enc.op_sub_cycles_ptr(OpSub::Samplers));
    // D3DRS_STENCILREF drives Metal's per-encoder stencil reference. Only
    // meaningful while the stencil test is on, and unlike the blend color it
    // is emitted on every change (including back to zero) so a pass that
    // lowers the reference mid-encoder doesn't keep comparing against the
    // previous one.
    let stencil_ref = render_state.stencil_ref & STENCIL_MASK_BITS;
    if render_state.depth_stencil_state.stencil_enable != 0
        && has_stencil
        && enc.last_bound().stencil_reference_changed(stencil_ref)
    {
        enc.emit_command(Command::set_stencil_reference(stencil_ref));
    }

    // D3DRS_BLENDFACTOR drives Metal's per-encoder constant blend
    // color. A fresh encoder blends with zero, not with the D3D9 default
    // opaque white, and the cache starts each pass at that zero, so the
    // first draw of a pass emits the factor it blends with. Every change
    // is emitted, including back to the default, so a pass that restores
    // white mid-encoder doesn't keep blending with the previous factor.
    if enc
        .last_bound()
        .blend_color_changed(render_state.blend_factor)
    {
        let [r, g, b, a] = mtld3d_core::convert::d3dcolor_to_rgba_f32(render_state.blend_factor);
        enc.emit_command(Command::set_blend_color(r, g, b, a));
    }

    // 4. Texture + sampler binds, one pair per valid stage. Depth-bound
    //    slots (sampleable shadow maps) need the `compareFunction =
    //    LessEqual` sampler variant so MSL `sample_compare` returns the
    //    D3D9 hardware-shadow PCF result; the bit mirrors the emitter's
    //    `depth_sampler_mask` so the call site and the sampler state
    //    can't drift.
    let depth_mask = variant.depth_sampler_mask;
    // Raw-fetch depth slots (INTZ/DF24/DF16) are depth textures but are read
    // with a plain `.sample()`, which requires a NON-comparison sampler — so
    // exclude them from the compare-sampler set.
    let fetch_mask = variant.depth_fetch_mask;
    let mut bound_mask: u16 = 0;
    for (stage_u32, b) in stage_bindings {
        if ps_sampled_mask & (1u16 << stage_u32) == 0 {
            continue;
        }
        let handle = stage_texture_handles[stage_u32 as usize];
        if handle == 0 {
            mtld3d_shared::log_once_warn_by!(target: crate::LOG_TARGET,
                key: b.texture_id.raw(),
                "draw: stage {stage_u32} bound to {:?} but its texture handle is 0 — bind \
                 skipped, an unbound declared sampler reads opaque black",
                b.texture_id
            );
            continue;
        }
        let bit = 1u16 << stage_u32;
        bound_mask |= bit;
        let is_compare = (depth_mask & bit) != 0 && (fetch_mask & bit) == 0;
        let is_fetch = (fetch_mask & bit) != 0;
        let sampler = enc.get_or_create_sampler(stage_u32, &b.sampler_state, is_compare, is_fetch);
        if enc.last_bound().fragment_texture_changed(stage_u32, handle) {
            enc.emit_command(Command::set_fragment_texture(handle, stage_u32));
        }
        // A sampler state the device declined to create is handle 0, which the
        // unix side replaces with the default sampler; the cache records the
        // default's sentinel so the slot is neither left unbound against a
        // fresh cache nor deduped against a later real sampler.
        if enc
            .last_bound()
            .fragment_sampler_changed(stage_u32, sampler_cache_key(sampler))
        {
            enc.emit_command(Command::set_fragment_sampler_state(sampler, stage_u32));
        }
    }

    // A pixel shader may declare a sampler the game bound no texture to. D3D9
    // requires that sample to read opaque black, and Metal requires every
    // declared `[[texture(n)]]` argument to be bound, so bind the shared 1×1
    // black texture (of the declared type) plus a default sampler to each such
    // slot. Only the programmable path can declare-without-binding; the FF PS
    // only declares samplers for stages it actually samples a bound texture on.
    if let PsSourceView::Programmable(ProgrammablePsSource { ps_id, .. }) = ps {
        let decls = enc.ps_declared_samplers(*ps_id);
        let mut unbound = decls.unbound(bound_mask);
        while unbound != 0 {
            let stage = unbound.trailing_zeros();
            unbound &= unbound - 1;
            // The fragment function types every slot from the bound texture,
            // not from its `dcl_<dim>`, so the fallback follows the same rule:
            // an unbound slot has no mask bit set and takes the 2D black
            // texture whatever the shader declared.
            let slot = u16::try_from(stage).expect("declared sampler slot is below STAGE_COUNT");
            let kind = null_texture_kind(bound_sampler_type(variant, slot));
            let tex_sentinel = null_texture_tex_sentinel(kind as u64);
            if enc
                .last_bound()
                .fragment_texture_changed(stage, tex_sentinel)
            {
                enc.emit_command(Command::set_fragment_null_texture(kind, stage));
            }
            // The null command also binds a default sampler; record a sentinel
            // so a later real sampler bind to this slot is not deduped away.
            enc.last_bound()
                .fragment_sampler_changed(stage, NULL_TEXTURE_SAMPLER_SENTINEL);
        }
    }
    // Vertex texture fetch: bind each slot the VS declares a sampler for
    // (`vs_3_0`, at most four). The bindings mirror `SetTexture` /
    // `SetSamplerState` on `D3DVERTEXTEXTURESAMPLER0..3` and live on the
    // encoder rather than the per-draw snapshot; a declared slot the game
    // never bound gets the shared black fallback, as on the fragment side.
    if let VsSourceView::Programmable(ProgrammableVsSource {
        vs_id,
        sampler_kinds,
        ..
    }) = vs
    {
        let decls = enc.ps_declared_samplers(*vs_id);
        let mut mask = decls.unbound(0) & 0xF;
        while mask != 0 {
            let slot = mask.trailing_zeros();
            mask &= mask - 1;
            let (id, ss) = enc.vertex_binding(slot as usize);
            let handle = id.map_or(0, |id| {
                if mtld3d_core::sampler_state::srgb_texture_enabled(&ss) {
                    enc.get_texture_handle_by_id_srgb(id)
                } else {
                    enc.get_texture_sample_handle_by_id(id)
                }
            });
            if handle == 0 {
                // The emitter types the argument from `sampler_kinds`, so the
                // black fallback must carry the same kind: an unbound slot is
                // 2D there, and a slot whose texture has no handle yet keeps
                // the kind the key already compiled for.
                let kind = null_texture_kind(
                    sampler_kinds
                        .kind(u16::try_from(slot).expect("vertex fetch slot is below four")),
                );
                let tex_sentinel = null_texture_tex_sentinel(kind as u64);
                if enc.last_bound().vertex_texture_changed(slot, tex_sentinel) {
                    enc.emit_command(Command::set_vertex_null_texture(kind, slot));
                }
                enc.last_bound()
                    .vertex_sampler_changed(slot, NULL_TEXTURE_SAMPLER_SENTINEL);
            } else {
                // Slot indices 16..19: past the fragment-stage memo range,
                // so vertex samplers never collide with a stage's memo entry.
                let sampler = enc.get_or_create_sampler(16 + slot, &ss, false, false);
                if enc.last_bound().vertex_texture_changed(slot, handle) {
                    enc.emit_command(Command::set_vertex_texture(handle, slot));
                }
                if enc
                    .last_bound()
                    .vertex_sampler_changed(slot, sampler_cache_key(sampler))
                {
                    enc.emit_command(Command::set_vertex_sampler_state(sampler, slot));
                }
            }
        }
    }
    drop(t_samplers);

    let t_binds = CycleAddTimer::start(enc.op_sub_cycles_ptr(OpSub::Binds));
    let t_cbind = CycleAddTimer::start(enc.op_sub_detail_ptr(OpSubDetail::BCbind));
    // 5. Shader constants (VS float slot / PS slot 15), alpha-ref float (PS
    //    slot 14), fog color (PS slot 13). VS/PS dedup uses snapshot tokens: when
    //    the same constants re-bind draw-after-draw (FF pass with one CB
    //    update at the head, or shadow-cast pass with shared light constants)
    //    we skip the `setBytes` command. The bytes already live in the owning
    //    frame or encoder arena, so the Metal command takes their `(ptr, len)`
    //    without an additional copy into a binding cache.
    if enc.vs_constants_changed(vs_constants) {
        let (p, n) = vs_constants.as_raw();
        enc.emit_command(Command::set_vertex_bytes_at(p, n, VS_FLOAT_CONST_SLOT));
    }
    // Half-pixel rasterization fixup (VS pos-fixup slot). Every DXSO/FF vertex
    // shader declares `constant float4 &pos_fixup` there and shifts
    // clip-space position half a pixel right/down so on-boundary geometry
    // lands on the D3D9 window→NDC reference.
    // `(1/vp_w, -1/vp_h, depth_clamp_z, render_scale, depth_bias)`, the
    // `PosFixup` struct of `mtld3d_core::dxso::emit::POS_FIXUP_MSL`;
    // the `.z` lane selects the FF RHW epilogue's depth clamp (the D3D9
    // depth-clamp rule, see `depth_clamp_z` above) and the `.w` lane carries
    // render pixels per logical pixel, which the point-size epilogue applies
    // to a size D3D9 states in the logical space. The viewport dims are
    // already in the bound target's space, so `.xy` needs no conversion.
    // The last lane is `D3DRS_DEPTHBIAS` as the clip-space offset resolved
    // above. Deduped so it only re-emits when the viewport dims, the clamp
    // predicate, the bound target's scale or the bias change (rare).
    let (_, _, vp_w, vp_h) = enc.effective_viewport();
    // Viewport dims fit u16 in practice; convert without an `as`-cast
    // precision-loss lint (same idiom as `encoder.rs`).
    let to_f = |v: u32| f32::from(u16::try_from(v).unwrap_or(u16::MAX));
    let pos_fixup: [f32; 5] = [
        1.0 / to_f(vp_w.max(1)),
        -1.0 / to_f(vp_h.max(1)),
        f32::from(u8::from(depth_clamp_z)),
        enc.target_scale().factor(),
        depth_bias,
    ];
    // SAFETY: `[f32; 5]` is POD with no padding; reinterpreting the array as
    // 20 contiguous bytes is sound and the borrow is local to this scope.
    let pos_fixup_bytes =
        unsafe { core::slice::from_raw_parts(pos_fixup.as_ptr().cast::<u8>(), 20) };
    if enc.last_bound().vs_pos_fixup_changed(pos_fixup_bytes) {
        let ptr = enc.alloc_scratch(pos_fixup_bytes);
        enc.emit_command(Command::set_vertex_bytes_at(ptr, 20, VS_POS_FIXUP_SLOT));
    }
    // Per-draw `VsDraw` uniform (point size state). Every vertex shader
    // declares it, so it must be bound before the first draw of a pass; the
    // API thread rebuilds the bytes only when a point render state changes
    // and the dedup skips the bind when they match the last ones emitted.
    let vs_draw = snap.vs_draw_bytes.unwrap_or(ScratchSlice::EMPTY);
    let vs_draw_bytes = vs_draw.as_slice();
    if vs_draw_bytes.is_empty() {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "draw: VsDraw uniform not populated; binding the render-state defaults"
        );
        if enc.last_bound().vs_draw_changed(&*VS_DRAW_DEFAULT) {
            let ptr = enc.alloc_scratch(&*VS_DRAW_DEFAULT);
            enc.emit_command(Command::set_vertex_bytes_at(
                ptr,
                u32::try_from(VS_DRAW_BYTES).expect("32 fits u32"),
                VS_DRAW_SLOT,
            ));
        }
    } else if enc.last_bound().vs_draw_changed(vs_draw_bytes) {
        let (p, n) = vs_draw.as_raw();
        enc.emit_command(Command::set_vertex_bytes_at(p, n, VS_DRAW_SLOT));
    }
    if enc.ps_constants_changed(ps_constants) {
        let (p, n) = ps_constants.as_raw();
        enc.emit_command(Command::set_fragment_bytes_at(p, n, 15));
    }
    // The alpha-reference scalar (slot 14) is read only by a comparison that
    // has a reference to compare against: the two constant results compile to
    // `true` / `false` and leave the argument untouched, and a disabled test
    // emits none. Asking the shader rather than trusting the byte buffer to be
    // empty keeps the bind in step with what the emitter produced.
    let alpha_func = u32::from(ps_variant.alpha_func);
    let alpha_ref_read =
        alpha_func != 0 && alpha_func != D3DCMP_ALWAYS && alpha_func != D3DCMP_NEVER;
    if alpha_ref_read
        && !alpha_ref_bytes.is_empty()
        && enc.last_bound().ps_alpha_ref_changed(alpha_ref_bytes)
    {
        let (p, n) = alpha_ref_slice.as_raw();
        enc.emit_command(Command::set_fragment_bytes_at(p, n, 14));
    }
    if !fog_color_bytes.is_empty() && enc.last_bound().ps_fog_color_changed(fog_color_bytes) {
        let (p, n) = fog_color_slice.as_raw();
        enc.emit_command(Command::set_fragment_bytes_at(p, n, 13));
    }
    if !bump_env_bytes.is_empty() && enc.last_bound().ps_bump_env_changed(bump_env_bytes) {
        let (p, n) = bump_env_slice.as_raw();
        enc.emit_command(Command::set_fragment_bytes_at(p, n, 12));
    }
    // Per-slot LOD bias. Bound only for a draw whose shader declares the
    // uniform; the binding then persists on the encoder, so a later biased
    // draw carrying the same table skips the re-bind.
    if any_lod_bias && let Some(ptr) = enc.alloc_lod_bias_if_changed(&lod_bias) {
        enc.emit_command(Command::set_fragment_bytes_at(
            ptr,
            u32::try_from(mtld3d_core::sampler_state::LOD_BIAS_BYTES)
                .expect("LOD bias uniform is 256 bytes"),
            PS_LOD_BIAS_SLOT,
        ));
    }
    // The render scale behind a scaled `vPos` read. Bound only for a draw
    // whose shader took the variant, from the bytes that decided the flag;
    // the encoder dedups it, and the scale changes only with the bound target.
    if let Some(draw_bytes) = ps_draw_bytes
        && enc.last_bound().ps_draw_changed(&draw_bytes)
    {
        let ptr = enc.alloc_scratch(&draw_bytes);
        enc.emit_command(Command::set_fragment_bytes_at(
            ptr,
            u32::try_from(mtld3d_core::ps_draw::PS_DRAW_BYTES).expect("16 fits u32"),
            PS_DRAW_SLOT,
        ));
    }
    // VS integer constants — bound only for the rare shader that reads a
    // dynamic integer constant. Re-bound unconditionally (no dedup): such
    // draws are infrequent and the payload is a fixed 256 B.
    if !vs_int_const_slice.as_slice().is_empty() {
        let (p, n) = vs_int_const_slice.as_raw();
        enc.emit_command(Command::set_vertex_bytes_at(p, n, VS_INT_CONST_SLOT));
    }
    // VS boolean constants: a 4-byte bitmask, same rare-draw policy.
    if !vs_bool_const_slice.as_slice().is_empty() {
        let (p, n) = vs_bool_const_slice.as_raw();
        enc.emit_command(Command::set_vertex_bytes_at(p, n, VS_BOOL_CONST_SLOT));
    }
    // PS integer / boolean constants: the fragment-side twins, same policy.
    if !ps_int_const_slice.as_slice().is_empty() {
        let (p, n) = ps_int_const_slice.as_raw();
        enc.emit_command(Command::set_fragment_bytes_at(p, n, PS_INT_CONST_SLOT));
    }
    if !ps_bool_const_slice.as_slice().is_empty() {
        let (p, n) = ps_bool_const_slice.as_raw();
        enc.emit_command(Command::set_fragment_bytes_at(p, n, PS_BOOL_CONST_SLOT));
    }
    drop(t_cbind);

    // 6. Bind the vertex streams. Wraps each bound VB's `PageBox` in an
    //    MTLBuffer lazily — the cache hits after the first draw post-rename
    //    and churns only when the game renames.
    let t_vbib = CycleAddTimer::start(enc.op_sub_detail_ptr(OpSubDetail::BVbib));
    match vertex_source {
        VertexView::Up { record, .. } => {
            let scratch_ptr = record.address;
            let size = record.size;
            if usize::try_from(size).is_ok_and(|size| size > SET_BYTES_MAX) {
                enc.bump_up_vertex_oversized();
            }
            enc.emit_command(Command::set_vertex_bytes(scratch_ptr, size, 0));
            // Inline slot-0 bind clobbers the real Metal vertex-buffer
            // binding; drop the cached bound-VB so the next bound draw
            // re-emits its `setVertexBuffer` instead of reading these bytes.
            enc.last_bound().invalidate_vertex_buffer();
        }
        VertexView::Bound { .. } => {
            for b in vertex_source.bindings() {
                let slot = u32::from(b.stream);
                let layout = layouts[b.stream as usize];
                if !layout.is_used() {
                    // Bound but not read by the declaration's consumed
                    // attributes: nothing to bind.
                    continue;
                }
                let (buffer_handle, staged) = enc.ensure_vbib_mtl_buffer(
                    BufferId::from_raw(b.buffer),
                    b.address,
                    b.length,
                    b.generation,
                );
                if buffer_handle == 0 {
                    mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "draw dropped: ensure_vbib_mtl_buffer returned 0 for VB");
                    mtld3d_shared::log_once_trace_by!(
                        target: crate::LOG_TARGET,
                        key: b.buffer,
                        "drop: VB buffer {:#x} (stream {slot}) wrap failed",
                        b.buffer,
                    );
                    return;
                }
                let bind = enc.last_bound().vertex_buffer_changed(
                    slot,
                    buffer_handle,
                    b.offset,
                    b.generation,
                );
                if bind == VertexBufferBind::ReusedHandle {
                    // The dedup would have kept the wrapper this address used
                    // to name bound; that wrapper was destroyed inside this
                    // pass, which the retention schedule is meant to rule out.
                    mtld3d_shared::log_once_warn!(
                        target: crate::LOG_TARGET,
                        "vertex buffer handle {buffer_handle:#x} reused within a pass for \
                         buffer {:#x} generation {}: rebinding instead of deduplicating",
                        b.buffer,
                        b.generation
                    );
                }
                let vb_emitted = bind != VertexBufferBind::Same;
                if vb_emitted {
                    enc.emit_command(Command::set_vertex_buffer(buffer_handle, b.offset, slot));
                }
                // Only a staged buffer takes staging uploads, so only its
                // ranges are ever asked about.
                if !staged {
                    continue;
                }
                // Record this draw's VB read range so a later overlapping
                // staging upload renames instead of corrupting this draw.
                // Tighten it past the whole-buffer fallback so a disjoint
                // later upload to the same buffer doesn't force a needless
                // rename: non-indexed draws give the exact vertex span;
                // indexed draws tighten only the lower bound by `base_vertex`
                // (the upper bound needs the max index value, which we don't
                // scan, so it stays at end-of-buffer); a per-instance stream
                // reads one element per `step_rate` instances. All may
                // over-cover but never under-cover; overflow falls back to the
                // whole tail. `None` = the draw reads nothing → record nothing.
                let logical_len = u32::try_from(b.length).unwrap_or(u32::MAX);
                let read_range = match (layout.step, index_source) {
                    (
                        VertexStepFunction::PerVertex,
                        IndexView::None {
                            start_vertex,
                            vertex_count,
                        },
                    ) => nonindexed_vb_range(b.offset, layout.stride, *start_vertex, *vertex_count),
                    (
                        VertexStepFunction::PerVertex,
                        IndexView::Bound {
                            base_vertex,
                            index_count,
                            ..
                        },
                    ) => indexed_vb_range_lower_bound(
                        b.offset,
                        layout.stride,
                        *base_vertex,
                        *index_count,
                    ),
                    // A fan over the shared pattern reads `primitive_count + 2`
                    // vertices from its start, like a non-indexed draw.
                    (
                        VertexStepFunction::PerVertex,
                        IndexView::Fan {
                            start_vertex,
                            primitive_count,
                        },
                    ) => nonindexed_vb_range(
                        b.offset,
                        layout.stride,
                        *start_vertex,
                        primitive_count.saturating_add(2),
                    ),
                    // A generated index list knows exactly which vertices it
                    // references, so the range is as tight as a non-indexed
                    // draw's.
                    (
                        VertexStepFunction::PerVertex,
                        IndexView::Generated {
                            min_vertex, record, ..
                        },
                    ) => nonindexed_vb_range(
                        b.offset,
                        layout.stride,
                        *min_vertex,
                        record.maximum - min_vertex + 1,
                    ),
                    // `Up` indices only ever pair with `VertexView::Up`,
                    // never a bound VB, so this arm is unreachable in
                    // practice; record no read range.
                    (VertexStepFunction::PerVertex, IndexView::Up { .. }) => None,
                    (VertexStepFunction::PerInstance | VertexStepFunction::Constant, _) => Some((
                        b.offset,
                        instanced_stream_read_bytes(
                            instances,
                            layout.step,
                            layout.step_rate,
                            layout.stride,
                        ),
                    )),
                };
                if let Some((range_off, range_size)) = read_range {
                    enc.note_buffer_draw_range(b.buffer, range_off, range_size, logical_len);
                }
            }
        }
    }
    // Streams the declaration reads with nothing bound: feed zeros inline
    // under the constant layout built above. The inline bind clobbers that
    // slot's real Metal binding, so forget it in the cache too.
    let mut null_streams = attrs.used_streams();
    while null_streams != 0 {
        let stream = null_streams.trailing_zeros();
        null_streams &= null_streams - 1;
        if !matches!(vertex_source.feed(stream), VertexFeed::Null) {
            continue;
        }
        let extent = layouts[stream as usize].stride;
        let len = u32::try_from(NULL_STREAM_ZEROS.len()).expect("4 KiB fits u32");
        if extent > len {
            mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
                "stream {stream} is unbound and its declaration extent {extent} exceeds the zero feed; clamped"
            );
        }
        enc.emit_command(Command::set_vertex_bytes_at(
            NULL_STREAM_ZEROS.as_ptr() as u64,
            extent.min(len),
            stream,
        ));
        enc.last_bound().invalidate_vertex_buffer_slot(stream);
    }
    drop(t_vbib);

    // 7. Emit the draw call.
    // Debug-build invariant: the per-draw dedup cache must match what was
    // actually emitted onto the encoder before the draw consumes it — catches
    // any cached-slot bind that bypassed its `last_bound` gate.
    #[cfg(debug_assertions)]
    enc.debug_assert_cache_in_sync();
    let t_draw = CycleAddTimer::start(enc.op_sub_detail_ptr(OpSubDetail::BDraw));
    // The generated-index fan is the slow path (per-draw rewrite plus an
    // upload-ring copy); the PERF grid counts it as a tripwire.
    if matches!(index_source, IndexView::Generated { .. }) {
        enc.bump_fan_generated();
    }
    // While the F12 dump runs, the Metal draw sits in a `draw N` debug group
    // so the trace node and the `[dump] draw N` line name each other.
    if let Some(index) = dump_draw {
        enc.emit_command(Command::push_debug_group(index));
    }
    let verts = match *index_source {
        IndexView::None {
            start_vertex,
            vertex_count,
        } => {
            enc.emit_command(Command::draw_primitives(
                metal_prim,
                start_vertex,
                vertex_count,
            ));
            vertex_count
        }
        IndexView::Bound {
            record,
            index_count,
            base_vertex,
        } => {
            let buffer_id = BufferId::from_raw(record.buffer);
            let backing_ptr = record.address;
            let backing_len = record.length;
            let backing_generation = record.generation;
            let offset = record.offset;
            let index_type = record.index_type().expect("validated index type");
            let (buffer_handle, staged) =
                enc.ensure_vbib_mtl_buffer(buffer_id, backing_ptr, backing_len, backing_generation);
            if buffer_handle == 0 {
                mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "draw dropped: ensure_vbib_mtl_buffer returned 0 for IB");
                mtld3d_shared::log_once_trace_by!(
                    target: crate::LOG_TARGET,
                    key: buffer_id.raw(),
                    "drop: IB buffer {:#x} wrap failed",
                    buffer_id.raw(),
                );
                close_dump_group(enc, dump_draw);
                return;
            }
            // Record a staged IB's read range so a later overlapping
            // staging upload renames instead of corrupting this draw.
            // Exact — `[offset, offset + index_count × index_size)`.
            if staged {
                let index_size: u32 = match index_type {
                    IndexType::UInt16 => 2,
                    IndexType::UInt32 => 4,
                };
                let read_bytes = index_count.saturating_mul(index_size);
                let logical_len = u32::try_from(backing_len).unwrap_or(u32::MAX);
                enc.note_buffer_draw_range(buffer_id.raw(), offset, read_bytes, logical_len);
            }
            enc.emit_command(Command::draw_indexed_primitives(
                metal_prim,
                index_count,
                index_type,
                buffer_handle,
                offset,
                base_vertex,
                instances,
            ));
            index_count
        }
        IndexView::Fan {
            start_vertex,
            primitive_count,
        } => {
            // The shared pattern is relative to the fan's first vertex;
            // `start_vertex` becomes Metal's base vertex. The device routes
            // only fans the pattern can address here, so the narrowing and
            // the buffer are expected to succeed; both failures drop the
            // draw loudly rather than draw the wrong triangles.
            let Ok(base_vertex) = i32::try_from(start_vertex) else {
                mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
                    "draw dropped: triangle fan start vertex {start_vertex} exceeds the base vertex range");
                close_dump_group(enc, dump_draw);
                return;
            };
            let buffer_handle = enc.fan_index_buffer(primitive_count);
            if buffer_handle == 0 {
                mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
                    "draw dropped: the shared triangle fan index buffer could not be created");
                close_dump_group(enc, dump_draw);
                return;
            }
            let index_count = primitive_count * 3;
            enc.emit_command(Command::draw_indexed_primitives(
                metal_prim,
                index_count,
                IndexType::UInt16,
                buffer_handle,
                0,
                base_vertex,
                instances,
            ));
            index_count
        }
        IndexView::Up {
            record,
            index_count,
        } => {
            let index_type = record.index_type().expect("validated index type");
            // The retained API capture stays live through submit replay, which
            // copies indices into the Metal upload ring. No intermediate native
            // CPU allocation is needed.
            enc.bump_up_indexed();
            let (scratch_ptr, byte_len) = (record.address, record.length);
            enc.emit_command(Command::draw_indexed_primitives_up(
                metal_prim,
                index_count,
                index_type,
                scratch_ptr,
                byte_len,
                instances,
            ));
            index_count
        }
        IndexView::Generated {
            record,
            index_count,
            ..
        } => {
            let index_type = record.index_type().expect("validated index type");
            // The list is already in the frame arena the unix side reads at
            // replay time, so it goes to the same inline-index draw form
            // without a second copy. The vertices were bound above: the
            // caller's buffers, or `VertexView::Up` bytes.
            let (index_ptr, byte_len) = (record.address, record.length);
            enc.emit_command(Command::draw_indexed_primitives_up(
                metal_prim,
                index_count,
                index_type,
                index_ptr,
                byte_len,
                instances,
            ));
            index_count
        }
    };
    close_dump_group(enc, dump_draw);
    drop(t_draw);
    drop(t_binds);

    enc.bump_pair_stats(
        shaders,
        verts,
        variant.alpha_func,
        u32::from(render_state.cull_mode),
    );
}
