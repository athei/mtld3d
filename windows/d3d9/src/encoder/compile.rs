//! Shader-library and render-pipeline builds on worker threads.
//!
//! A draw that names a library or pipeline the encoder has never built used
//! to build it on the encoder thread, which stalled the frame for the
//! length of a Metal compile. The encoder now does only the probe, the
//! content key, the warm-cache bridge and the enqueue; a pool of
//! `mtld3d-compile` threads does the MSL emission, the Metal compile or
//! pipeline build and the cache append, and the encoder installs what they
//! hand back at the top of its next frame or draw. The draws that need a
//! build still in flight are left out of the frame when that loses nothing a
//! later frame does not redraw, and wait for the build otherwise
//! (`mtld3d_core::async_compile::may_skip_draw`).

use std::{
    sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError, mpsc},
    thread,
    time::{Duration, Instant},
};

use log::{Level, debug, error, log_enabled, trace, warn};
use mtld3d_core::{
    async_compile::{
        ClearPlanes, CompileLanes, JobTicket, Resolution, mark_kept_reads, may_skip_draw,
    },
    build_index::BuildLookup,
    dxso::{
        DxsoProgram, FfPsKey, FfVsKey, LOG_TARGET as MSL_TRACE_TARGET, VariantKey, VsSamplerKinds,
        emit_ps_ff_named, emit_ps_programmable_named, emit_vs_ff_named, emit_vs_programmable_named,
    },
    ids::ProgramId,
    perf::{
        PairShaderId,
        compilation::{Identity as CompileIdentity, Kind as CompileKind},
    },
    pipeline_state::{self, PipelineBuildInputs, PipelineKey, PipelineSnapshot},
    shader_cache::{self, CachedKind, PipelineRecipe, ShaderRecordRef},
    shader_compile_stats::CompileBucket,
};
use mtld3d_shared::{
    MetalHandle, VertexAttrDesc,
    mtl::StageTag,
    mtl_handle::{MTLDeviceKind, MTLRenderPipelineStateKind, MTLTextureKind},
    perf::{NanosSetTimer, PipelineTimings, ShaderTimings},
    tsc::{rdtsc, secs_to_cycles},
};

use super::{
    FrameEncoder, FrameEncoderFlags, LOG_TARGET, StageLibHandles, compile_stage_library,
    open_or_create_cache_file,
};
use crate::{
    draw::{PsSource, ShaderRef, VsSource},
    unix_call::unix_call,
};

/// Worker threads each encoder builds with.
///
/// A burst of first-seen shaders at a scene change is what they absorb; four
/// keep a burst's libraries compiling side by side while leaving cores for
/// the API, encoder and submit threads.
const COMPILE_WORKERS: usize = 4;

/// Seconds a build may stay in flight before the encoder warns that it looks stuck.
const STALLED_BUILD_SECS: u64 = 5;

/// Stack reserved for each compile worker: 1 MiB, half the thread default.
///
/// A 32-bit guest's address space is what runs out first, and four workers
/// at the default 2 MiB would reserve 8 MiB of it per device. Wine raises
/// every thread's stack reservation to at least 1 MiB, so asking for less
/// would change nothing but what this constant claims; this asks for the
/// floor explicitly, and four workers cost 4 MiB per device. The stack holds
/// the MSL emission and the PE half of the thunk; the `unix_call` itself
/// runs on Wine's kernel stack.
const COMPILE_WORKER_STACK: usize = 1024 * 1024;

/// The jobs waiting for a worker, and the workers' wake-up.
///
/// Shared by the encoder that queues and steals and the workers that pop.
/// The lock is held for a lane operation and never across a build.
pub struct CompileQueue {
    state: Mutex<QueueState>,
    ready: Condvar,
}

struct QueueState {
    lanes: CompileLanes<QueuedJob>,
    /// Set once at encoder teardown; a worker that finds the lanes empty then exits.
    closed: bool,
}

impl CompileQueue {
    pub const fn new() -> Self {
        Self {
            state: Mutex::new(QueueState {
                lanes: CompileLanes::new(),
                closed: false,
            }),
            ready: Condvar::new(),
        }
    }

    fn lock(&self) -> MutexGuard<'_, QueueState> {
        // Nothing panics while holding the lock (the builds run outside it),
        // and a panic aborts the process anyway, so a poisoned guard still
        // holds consistent lanes.
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn push(&self, ticket: JobTicket, job: QueuedJob) {
        self.lock().lanes.push_normal(ticket, job);
        self.ready.notify_one();
    }

    fn promote(&self, ticket: JobTicket) -> bool {
        self.lock().lanes.promote(ticket)
    }

    fn steal(&self, ticket: JobTicket) -> Option<QueuedJob> {
        self.lock().lanes.steal(ticket)
    }

    fn steal_urgent(&self) -> Option<(JobTicket, QueuedJob)> {
        self.lock().lanes.steal_urgent()
    }

    fn take_any(&self) -> Option<(JobTicket, QueuedJob)> {
        self.lock().lanes.pop()
    }

    /// Wake every worker to exit once the lanes are empty.
    pub fn close(&self) {
        self.lock().closed = true;
        self.ready.notify_all();
    }

    /// The next job for a worker, blocking while there is none; `None` once closed.
    fn next_for_worker(&self) -> Option<(JobTicket, QueuedJob)> {
        let mut state = self.lock();
        loop {
            if let Some(job) = state.lanes.pop() {
                return Some(job);
            }
            if state.closed {
                return None;
            }
            state = self
                .ready
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }
}

/// Start the encoder's compile workers; answers how many started.
///
/// Called from the encoder thread's startup, never from `DllMain`. Each
/// worker is detached: like the submit thread it is never joined (Wine can
/// fail the wait on its handle), and it exits when the queue closes at
/// encoder teardown or the encoder drops the result channel.
pub fn spawn_workers(queue: &Arc<CompileQueue>, results: &mpsc::Sender<CompileResult>) -> usize {
    let mut started = 0;
    for _ in 0..COMPILE_WORKERS {
        let queue = Arc::clone(queue);
        let results = results.clone();
        match thread::Builder::new()
            .name("mtld3d-compile".into())
            .stack_size(COMPILE_WORKER_STACK)
            .spawn(move || worker_main(&queue, &results))
        {
            Ok(_detached) => started += 1,
            Err(e) => {
                mtld3d_shared::log_once_warn!(
                    target: LOG_TARGET,
                    "encoder: spawning a compile worker failed, building with fewer: {e}"
                );
            }
        }
    }
    started
}

fn worker_main(queue: &CompileQueue, results: &mpsc::Sender<CompileResult>) {
    mtld3d_shared::crumb::init();
    while let Some((ticket, job)) = queue.next_for_worker() {
        if results.send(run_job(ticket, job, true)).is_err() {
            break;
        }
    }
}

/// One build and the moment it was queued.
struct QueuedJob {
    job: CompileJob,
    /// TSC reading at enqueue, for the enqueue-to-install latency.
    enqueued_tsc: u64,
}

enum CompileJob {
    Library(LibraryJob),
    /// Boxed: a pipeline job carries a whole snapshot, several times a library job's size.
    Pipeline(Box<PipelineJob>),
}

/// Everything a worker needs to emit, compile and persist one stage library.
struct LibraryJob {
    input: LibraryInput,
    /// The on-disk identity, which also names the MSL entry point.
    reference: ShaderRecordRef,
    device: MetalHandle<MTLDeviceKind>,
    /// Whether to append the record to the shader cache.
    persist: bool,
}

/// The emitter input of one library, and the index key its outcome lands under.
enum LibraryInput {
    ProgrammableVs {
        vs_id: ProgramId,
        program: Arc<DxsoProgram>,
        provided_input_mask: u16,
        clip_plane_count: u8,
        sampler_kinds: VsSamplerKinds,
    },
    FixedFunctionVs {
        key: FfVsKey,
    },
    ProgrammablePs {
        ps_id: ProgramId,
        program: Arc<DxsoProgram>,
        variant: VariantKey,
    },
    FixedFunctionPs {
        key: FfPsKey,
        variant: VariantKey,
    },
}

impl LibraryInput {
    const fn is_vertex(&self) -> bool {
        matches!(
            self,
            Self::ProgrammableVs { .. } | Self::FixedFunctionVs { .. }
        )
    }

    const fn stage(&self) -> &'static str {
        if self.is_vertex() { "VS" } else { "PS" }
    }

    const fn is_programmable(&self) -> bool {
        matches!(
            self,
            Self::ProgrammableVs { .. } | Self::ProgrammablePs { .. }
        )
    }
}

/// Everything a worker needs to build and persist one render pipeline.
struct PipelineJob {
    snapshot: PipelineSnapshot,
    vertex_attrs: Vec<VertexAttrDesc>,
    /// The two shader records the recipe names; `None` when a shader has no cache kind.
    shader_refs: Option<(ShaderRecordRef, ShaderRecordRef)>,
    vs: PairShaderId,
    ps: PairShaderId,
    /// The with-colour pipeline this one is the no-colour sibling of, as a raw handle.
    sibling_of: Option<u64>,
    device: MetalHandle<MTLDeviceKind>,
    persist: bool,
}

/// A finished build on its way back to the encoder.
pub struct CompileResult {
    ticket: JobTicket,
    enqueued_tsc: u64,
    /// Built by a worker rather than by the encoder while it waited.
    on_worker: bool,
    outcome: Outcome,
}

enum Outcome {
    Library(LibraryOutcome),
    /// Boxed for the same reason as [`CompileJob::Pipeline`].
    Pipeline(Box<PipelineOutcome>),
}

struct LibraryOutcome {
    input: LibraryInput,
    reference: ShaderRecordRef,
    handles: Option<StageLibHandles>,
    /// Whether the emitter produced MSL, the success of the emission row.
    emitted: bool,
    bucket: Option<CompileBucket>,
    /// Emission through compile, the duration the compile summary counts.
    elapsed: Duration,
    total_ns: u64,
    emit_ns: u64,
    persist_ns: u64,
    native: ShaderTimings,
    persist_error: Option<std::io::Error>,
}

struct PipelineOutcome {
    key: PipelineKey,
    snapshot: PipelineSnapshot,
    vs: PairShaderId,
    ps: PairShaderId,
    sibling_of: Option<u64>,
    handle: Option<MetalHandle<MTLRenderPipelineStateKind>>,
    total_ns: u64,
    persist_ns: u64,
    native: PipelineTimings,
    persist_error: Option<std::io::Error>,
}

fn run_job(ticket: JobTicket, queued: QueuedJob, on_worker: bool) -> CompileResult {
    let QueuedJob { job, enqueued_tsc } = queued;
    let outcome = match job {
        CompileJob::Library(job) => Outcome::Library(build_library(job)),
        CompileJob::Pipeline(job) => Outcome::Pipeline(Box::new(build_pipeline(*job))),
    };
    CompileResult {
        ticket,
        enqueued_tsc,
        on_worker,
        outcome,
    }
}

fn build_library(job: LibraryJob) -> LibraryOutcome {
    let LibraryJob {
        input,
        reference,
        device,
        persist,
    } = job;
    let mut total_ns = 0;
    let mut emit_ns = 0;
    let mut persist_ns = 0;
    let mut native = ShaderTimings::new();
    let mut persist_error = None;
    let mut elapsed = Duration::ZERO;
    let mut emitted = false;
    let total_timer = NanosSetTimer::start(&raw mut total_ns);
    let entry_name = reference.kind().entry_name(reference.key());
    let started = Instant::now();
    let emission = NanosSetTimer::start(&raw mut emit_ns);
    let (msl, bucket) = match &input {
        LibraryInput::ProgrammableVs {
            program,
            provided_input_mask,
            clip_plane_count,
            sampler_kinds,
            ..
        } => (
            emit_vs_programmable_named(
                program,
                &entry_name,
                *provided_input_mask,
                *clip_plane_count,
                *sampler_kinds,
            )
            .map_err(|e| error!(target: LOG_TARGET, "emit_vs_programmable failed: {e:?}"))
            .ok(),
            CompileBucket::from_sm_major(program.major),
        ),
        LibraryInput::FixedFunctionVs { key } => {
            mtld3d_shared::crumb!("ffvs:emit", reference.key(), u64::from(key.tex_coord_count));
            (
                Some(emit_vs_ff_named(key, &entry_name)),
                Some(CompileBucket::Ff),
            )
        }
        LibraryInput::ProgrammablePs {
            program, variant, ..
        } => (
            emit_ps_programmable_named(program, *variant, &entry_name)
                .map_err(|e| error!(target: LOG_TARGET, "emit_ps_programmable failed: {e:?}"))
                .ok(),
            CompileBucket::from_sm_major(program.major),
        ),
        LibraryInput::FixedFunctionPs { key, variant } => (
            Some(emit_ps_ff_named(key, *variant, &entry_name)),
            Some(CompileBucket::Ff),
        ),
    };
    drop(emission);
    let handles = msl.and_then(|msl| {
        emitted = true;
        let stage = input.stage();
        if log_enabled!(target: MSL_TRACE_TARGET, Level::Trace) {
            let tag = reference_tag(reference);
            trace!(target: MSL_TRACE_TARGET, "── {stage} MSL {tag} ──\n{msl}\n── /{stage} MSL {tag} ──");
        }
        let stage_tag = if input.is_vertex() {
            StageTag::Vertex
        } else {
            StageTag::Fragment
        };
        let handles = compile_stage_library(device, stage_tag, &msl, &entry_name, &mut native)?;
        elapsed = started.elapsed();
        if persist {
            let _persist = NanosSetTimer::start(&raw mut persist_ns);
            let retained = match &input {
                LibraryInput::ProgrammableVs {
                    program,
                    provided_input_mask,
                    clip_plane_count,
                    sampler_kinds,
                    ..
                } => Some(shader_cache::ShaderSource::vertex(
                    program,
                    *provided_input_mask,
                    *clip_plane_count,
                    *sampler_kinds,
                )),
                LibraryInput::ProgrammablePs {
                    program, variant, ..
                } => Some(shader_cache::ShaderSource::pixel(program, *variant)),
                LibraryInput::FixedFunctionVs { .. } | LibraryInput::FixedFunctionPs { .. } => {
                    None
                }
            };
            let entry =
                shader_cache::CacheEntry::new(reference.kind(), reference.key(), msl, retained);
            persist_error = open_or_create_cache_file()
                .and_then(|writer| writer.append_shader(&entry))
                .err();
        }
        Some(handles)
    });
    drop(total_timer);
    LibraryOutcome {
        input,
        reference,
        handles,
        emitted,
        bucket,
        elapsed,
        total_ns,
        emit_ns,
        persist_ns,
        native,
        persist_error,
    }
}

fn build_pipeline(job: PipelineJob) -> PipelineOutcome {
    let PipelineJob {
        snapshot,
        vertex_attrs,
        shader_refs,
        vs,
        ps,
        sibling_of,
        device,
        persist,
    } = job;
    let mut total_ns = 0;
    let mut persist_ns = 0;
    let mut persist_error = None;
    let total = NanosSetTimer::start(&raw mut total_ns);
    let key = pipeline_state::key_from_snapshot(&snapshot, &vertex_attrs);
    // One wire layout per used stream; lives until the synchronous thunk
    // below has read it.
    let vertex_layouts = pipeline_state::vertex_layouts_from_snapshot(&snapshot);
    let mut params = pipeline_state::params_from_snapshot(&PipelineBuildInputs {
        snapshot: &snapshot,
        vertex_attrs: &vertex_attrs,
        vertex_layouts: &vertex_layouts,
        device_handle: device,
    });
    let status = unix_call(&mut params);
    let pipeline = params.pipeline_handle;
    let native = params.timings.into_inner();
    debug!(
        target: LOG_TARGET,
        "encoder: live CreateRenderPipeline status={status:#x} sibling={}",
        sibling_of.is_some()
    );
    let success = status == 0 && !pipeline.is_null();
    if success && persist {
        let _persist = NanosSetTimer::start(&raw mut persist_ns);
        if let Some((vs_ref, ps_ref)) = shader_refs {
            let recipe = PipelineRecipe::from_snapshot(vs_ref, ps_ref, &snapshot, &vertex_attrs);
            persist_error = open_or_create_cache_file()
                .and_then(|writer| writer.append_pipeline(&recipe))
                .err();
        } else {
            mtld3d_shared::log_once_warn_by!(
                target: LOG_TARGET,
                key: snapshot.vdecl_hash,
                "shader_cache: pipeline has an unsupported shader reference, recipe skipped"
            );
        }
    }
    drop(total);
    PipelineOutcome {
        key,
        snapshot,
        vs,
        ps,
        sibling_of,
        handle: success.then_some(pipeline),
        total_ns,
        persist_ns,
        native,
        persist_error,
    }
}

/// `prog 0x…` / `ff 0x…` for a shader record, the tag the trace dumps and warnings use.
fn reference_tag(reference: ShaderRecordRef) -> String {
    PairShaderId {
        is_programmable: !matches!(reference.kind(), CachedKind::FfVs | CachedKind::FfPs),
        hash: reference.key(),
    }
    .tag()
}

/// The endpoints of one `StretchRect`, for `note_stretch_copy`.
pub struct StretchCopyTargets {
    pub src: MetalHandle<MTLTextureKind>,
    pub dst: MetalHandle<MTLTextureKind>,
    /// Slice in the low half, level in the high half, as colour clears are recorded.
    pub dst_subresource: u32,
    /// The copy covers the whole of a colour destination.
    pub whole_color_dst: bool,
}

/// How the cold half of a library resolve ended on the encoder thread.
enum Begun {
    /// The warm cache already holds the library.
    Bridged(StageLibHandles),
    /// A worker is building it under this ticket, queued by this miss.
    Queued(JobTicket),
    /// A worker was already building it under this ticket.
    Pending(JobTicket),
    /// It cannot be built at all.
    Failed,
}

impl FrameEncoder {
    /// Resolve the VS library for a draw.
    ///
    /// Hot path: borrow-probe the source-keyed index (`ff_vs_libs` /
    /// `prog_vs_libs`), `FxHash` + exact `Eq`, no per-draw content hash,
    /// no clone. VS variants share one `MTLLibrary`, so the index key
    /// excludes `variant`. On a miss (about once per shader) the cold half
    /// computes the `disk_key`, answers from the warm cache when it can and
    /// otherwise queues the build and records the key as pending. A failure
    /// is recorded under the key, so it costs one build and its log lines
    /// however often the key is drawn. Programs register before the first
    /// draw that names them and are never removed, so a missing program is
    /// as final as a rejected one.
    #[inline]
    pub fn resolve_vs_library(&mut self, source: &VsSource) -> Resolution<StageLibHandles> {
        let known = match source {
            VsSource::FixedFunction { key, .. } => self.ff_vs_libs.lookup(key),
            VsSource::Programmable {
                vs_id,
                provided_input_mask,
                clip_plane_count,
                sampler_kinds,
                ..
            } => self.prog_vs_libs.lookup(&(
                *vs_id,
                *provided_input_mask,
                *clip_plane_count,
                *sampler_kinds,
            )),
        };
        match known {
            BuildLookup::Ready(handles) => return Resolution::Ready(handles),
            BuildLookup::Failed => return Resolution::Failed,
            BuildLookup::Unknown => {}
        }
        self.resolve_vs_library_miss(source)
    }

    /// The cold half of [`Self::resolve_vs_library`], out of line so a hit pays nothing for it.
    #[cold]
    #[inline(never)]
    fn resolve_vs_library_miss(&mut self, source: &VsSource) -> Resolution<StageLibHandles> {
        let mut miss_ns = 0;
        let miss = NanosSetTimer::start(&raw mut miss_ns);
        let begun = self.begin_vs_library(source);
        let resolution = match begun {
            Begun::Bridged(handles) => {
                self.record_vs_library(source, Some(handles));
                Resolution::Ready(handles)
            }
            Begun::Failed => {
                warn!(
                    target: LOG_TARGET,
                    "encoder: VS library {} failed to build, its draws are dropped without another attempt",
                    vs_source_tag(source)
                );
                self.record_vs_library(source, None);
                Resolution::Failed
            }
            Begun::Queued(ticket) | Begun::Pending(ticket) => Resolution::Pending(ticket),
        };
        drop(miss);
        self.perf.compilation_mut().note_miss(miss_ns);
        resolution
    }

    fn record_vs_library(&mut self, source: &VsSource, outcome: Option<StageLibHandles>) {
        match source {
            VsSource::FixedFunction { key, .. } => self.ff_vs_libs.record(key.clone(), outcome),
            VsSource::Programmable {
                vs_id,
                provided_input_mask,
                clip_plane_count,
                sampler_kinds,
                ..
            } => self.prog_vs_libs.record(
                (
                    *vs_id,
                    *provided_input_mask,
                    *clip_plane_count,
                    *sampler_kinds,
                ),
                outcome,
            ),
        }
    }

    /// Cold half of [`Self::resolve_vs_library`]: the index missed.
    ///
    /// Computes the Xxh3 `disk_key` (the only content hash, about once per
    /// shader), bridges the warm-loaded disk-keyed `lib_cache`, and queues
    /// the build otherwise. Every `VsKey` variant of a shader maps to the
    /// same `disk_key`.
    fn begin_vs_library(&mut self, source: &VsSource) -> Begun {
        let disk_key = source.disk_key();
        let (kind, program) = match source {
            VsSource::Programmable { vs_id, .. } => {
                let Some(program) = self.program_cache.get(vs_id) else {
                    error!(target: LOG_TARGET, "VS {vs_id:#x} missing from program_cache");
                    return Begun::Failed;
                };
                (
                    CachedKind::from_programmable(program.major, false),
                    Some(Arc::clone(program)),
                )
            }
            VsSource::FixedFunction { .. } => (Some(CachedKind::FfVs), None),
        };
        // `lib_cache` owns every library built from here, and device
        // teardown destroys what it holds. A shader with no cache kind would
        // build into the non-owning indexes alone and outlive its device, so
        // it is not built; the parser admits no such model today.
        let Some(kind) = kind else {
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET,
                "VS {disk_key:#x}: shader model has no cache kind, not compiled"
            );
            return Begun::Failed;
        };
        let reference = ShaderRecordRef::new(kind, disk_key);
        if let Some(&ticket) = self.pending_libs.get(&reference) {
            return Begun::Pending(ticket);
        }
        if let Some(&handles) = self.lib_cache.get(&reference) {
            return Begun::Bridged(handles);
        }
        let input = match (source, program) {
            (
                VsSource::Programmable {
                    vs_id,
                    provided_input_mask,
                    clip_plane_count,
                    sampler_kinds,
                    ..
                },
                Some(program),
            ) => LibraryInput::ProgrammableVs {
                vs_id: *vs_id,
                program,
                provided_input_mask: *provided_input_mask,
                clip_plane_count: *clip_plane_count,
                sampler_kinds: *sampler_kinds,
            },
            (VsSource::FixedFunction { key, .. }, _) => {
                LibraryInput::FixedFunctionVs { key: key.clone() }
            }
            (VsSource::Programmable { .. }, None) => return Begun::Failed,
        };
        Begun::Queued(self.enqueue_library(input, reference))
    }

    /// Resolve the PS library for a draw.
    ///
    /// Hot path: borrow-probe the source-keyed index. PS MSL depends on
    /// `variant`, so the key folds it in: `ff_ps_libs` nests
    /// `FfPsKey → variant → handles` (borrow the `FfPsKey`, no clone),
    /// `prog_ps_libs` uses a `(ProgramId, VariantKey)` `Copy` tuple. A miss
    /// takes the same cold half as the vertex stage.
    #[inline]
    pub fn resolve_ps_library(
        &mut self,
        source: &PsSource,
        variant: VariantKey,
    ) -> Resolution<StageLibHandles> {
        let known = match source {
            PsSource::FixedFunction { key, .. } => self
                .ff_ps_libs
                .get(key)
                .map_or(BuildLookup::Unknown, |variants| variants.lookup(&variant)),
            PsSource::Programmable { ps_id, .. } => self.prog_ps_libs.lookup(&(*ps_id, variant)),
        };
        match known {
            BuildLookup::Ready(handles) => return Resolution::Ready(handles),
            BuildLookup::Failed => return Resolution::Failed,
            BuildLookup::Unknown => {}
        }
        self.resolve_ps_library_miss(source, variant)
    }

    /// The cold half of [`Self::resolve_ps_library`], out of line so a hit pays nothing for it.
    #[cold]
    #[inline(never)]
    fn resolve_ps_library_miss(
        &mut self,
        source: &PsSource,
        variant: VariantKey,
    ) -> Resolution<StageLibHandles> {
        let mut miss_ns = 0;
        let miss = NanosSetTimer::start(&raw mut miss_ns);
        let begun = self.begin_ps_library(source, variant);
        let resolution = match begun {
            Begun::Bridged(handles) => {
                self.record_ps_library(source, variant, Some(handles));
                Resolution::Ready(handles)
            }
            Begun::Failed => {
                warn!(
                    target: LOG_TARGET,
                    "encoder: PS library {} failed to build, its draws are dropped without another attempt",
                    ps_source_tag(source, variant)
                );
                self.record_ps_library(source, variant, None);
                Resolution::Failed
            }
            Begun::Queued(ticket) | Begun::Pending(ticket) => Resolution::Pending(ticket),
        };
        drop(miss);
        self.perf.compilation_mut().note_miss(miss_ns);
        resolution
    }

    fn record_ps_library(
        &mut self,
        source: &PsSource,
        variant: VariantKey,
        outcome: Option<StageLibHandles>,
    ) {
        match source {
            PsSource::FixedFunction { key, .. } => {
                self.ff_ps_libs
                    .entry(key.clone())
                    .or_default()
                    .record(variant, outcome);
            }
            PsSource::Programmable { ps_id, .. } => {
                self.prog_ps_libs.record((*ps_id, variant), outcome);
            }
        }
    }

    /// Cold half of [`Self::resolve_ps_library`]; the `disk_key` folds in `variant`.
    fn begin_ps_library(&mut self, source: &PsSource, variant: VariantKey) -> Begun {
        let disk_key = source.disk_key(variant);
        let (kind, program) = match source {
            PsSource::Programmable { ps_id, .. } => {
                let Some(program) = self.program_cache.get(ps_id) else {
                    error!(target: LOG_TARGET, "PS {ps_id:#x} missing from program_cache");
                    return Begun::Failed;
                };
                (
                    CachedKind::from_programmable(program.major, true),
                    Some(Arc::clone(program)),
                )
            }
            PsSource::FixedFunction { .. } => (Some(CachedKind::FfPs), None),
        };
        // See `begin_vs_library`: a library no cache kind owns would outlive
        // its device.
        let Some(kind) = kind else {
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET,
                "PS {disk_key:#x}: shader model has no cache kind, not compiled"
            );
            return Begun::Failed;
        };
        let reference = ShaderRecordRef::new(kind, disk_key);
        if let Some(&ticket) = self.pending_libs.get(&reference) {
            return Begun::Pending(ticket);
        }
        if let Some(&handles) = self.lib_cache.get(&reference) {
            return Begun::Bridged(handles);
        }
        let input = match (source, program) {
            (PsSource::Programmable { ps_id, .. }, Some(program)) => LibraryInput::ProgrammablePs {
                ps_id: *ps_id,
                program,
                variant,
            },
            (PsSource::FixedFunction { key, .. }, _) => LibraryInput::FixedFunctionPs {
                key: key.clone(),
                variant,
            },
            (PsSource::Programmable { .. }, None) => return Begun::Failed,
        };
        Begun::Queued(self.enqueue_library(input, reference))
    }

    fn enqueue_library(&mut self, input: LibraryInput, reference: ShaderRecordRef) -> JobTicket {
        let job = LibraryJob {
            input,
            reference,
            device: self.device_handle,
            persist: self.cache_persists(),
        };
        let ticket = self.enqueue(CompileJob::Library(job));
        self.pending_libs.insert(reference, ticket);
        ticket
    }

    /// Look up or queue an `MTLRenderPipelineState` for the given pipeline state snapshot.
    ///
    /// Translation from D3D9 state to Metal enums happens in
    /// `mtld3d_core::pipeline_state`; the per-field invariant test there
    /// guards against "classified Consumed but value silently dropped".
    pub fn get_or_create_pipeline(
        &mut self,
        snapshot: &PipelineSnapshot,
        vertex_attrs: &[VertexAttrDesc],
        shaders: &ShaderRef<'_>,
    ) -> Resolution<u64> {
        self.perf.bump_pipeline_memo_call();
        // L0 memo: a draw whose pipeline snapshot is identical to the
        // previous one returns the cached handle without rebuilding the
        // `PipelineKey` (its D3D→Metal translations) or probing
        // `pipeline_cache`. It also skips the no-color twin's second resolve
        // below. A successful sibling mapping is process-lifetime. Only
        // built primaries are memoised: a pending or failing snapshot goes
        // on to `resolve_pipeline`, whose cache answers pending or failed on
        // the probe. The `match` copies the handle out so the memo borrow
        // ends before the `&mut perf` bump.
        let memo_hit = match &self.last_pipeline_memo {
            Some((prev, handle)) if *prev == *snapshot => Some(*handle),
            _ => None,
        };
        if let Some(handle) = memo_hit {
            self.perf.bump_pipeline_memo_hit();
            return Resolution::Ready(handle);
        }
        let with_color = match self.resolve_pipeline(snapshot, vertex_attrs, shaders, None) {
            Resolution::Ready(handle) => handle,
            Resolution::Pending(ticket) => return Resolution::Pending(ticket),
            Resolution::Failed => return Resolution::Failed,
        };
        // Dual-build for zero-mask draws: queue the matching no-color
        // variant up-front so pass-finalisation (Rule H) can swap to it
        // retroactively if every draw in the pass had `mask == 0`.
        // Rule H keeps color when there is no depth attachment. Its unused
        // sibling would have no attachments, which Mac2 Metal rejects.
        // A successful sibling mapping stays valid as long as the pipeline
        // cache, so an L0 miss can reuse it without rebuilding the alternate
        // snapshot and key. The sibling builds asynchronously and nothing
        // waits for it: until its mapping lands (at install, or on an L0 miss
        // that finds it built), Rule H keeps the pass's color. A failed
        // sibling leaves no mapping, and `resolve_pipeline` answers failed
        // from its cache without another build.
        if snapshot.has_depth()
            && snapshot.writes_no_color()
            && snapshot.has_color_output()
            && !self.no_color_pipeline_alt.contains_key(&with_color.raw())
        {
            // No-color twin: same identity except the attach flag (and no
            // render targets 1..3, which Rule H strips together with target
            // 0). Explicit `.clone()` because PipelineSnapshot is not Copy;
            // fires on L0 misses until the sibling has a mapping.
            let mut alt = snapshot.clone();
            alt.remove_color_output();
            if let Resolution::Ready(no_color) =
                self.resolve_pipeline(&alt, vertex_attrs, shaders, Some(with_color.raw()))
            {
                self.no_color_pipeline_alt
                    .insert(with_color.raw(), no_color);
            }
        }
        self.last_pipeline_memo = Some((snapshot.clone(), with_color.raw()));
        Resolution::Ready(with_color.raw())
    }

    fn resolve_pipeline(
        &mut self,
        snapshot: &PipelineSnapshot,
        vertex_attrs: &[VertexAttrDesc],
        shaders: &ShaderRef<'_>,
        sibling_of: Option<u64>,
    ) -> Resolution<MetalHandle<MTLRenderPipelineStateKind>> {
        let key = pipeline_state::key_from_snapshot(snapshot, vertex_attrs);
        match self.pipeline_cache.lookup(&key) {
            BuildLookup::Ready(handle) => return Resolution::Ready(handle),
            BuildLookup::Failed => return Resolution::Failed,
            BuildLookup::Unknown => {}
        }
        if let Some(&ticket) = self.pending_pipelines.get(&key) {
            return Resolution::Pending(ticket);
        }
        let mut miss_ns = 0;
        let miss = NanosSetTimer::start(&raw mut miss_ns);
        let job = PipelineJob {
            snapshot: snapshot.clone(),
            vertex_attrs: vertex_attrs.to_vec(),
            shader_refs: self.pipeline_shader_refs(shaders),
            vs: PairShaderId {
                is_programmable: matches!(shaders.vs, VsSource::Programmable { .. }),
                hash: shaders.vs.disk_key(),
            },
            ps: PairShaderId {
                is_programmable: matches!(shaders.ps, PsSource::Programmable { .. }),
                hash: shaders.ps.disk_key(shaders.variant),
            },
            sibling_of,
            device: self.device_handle,
            persist: self.cache_persists(),
        };
        let ticket = self.enqueue(CompileJob::Pipeline(Box::new(job)));
        self.pending_pipelines.insert(key, ticket);
        drop(miss);
        self.perf.compilation_mut().note_miss(miss_ns);
        Resolution::Pending(ticket)
    }

    fn pipeline_shader_refs(
        &self,
        shaders: &ShaderRef<'_>,
    ) -> Option<(ShaderRecordRef, ShaderRecordRef)> {
        let vs_kind = match shaders.vs {
            VsSource::FixedFunction { .. } => CachedKind::FfVs,
            VsSource::Programmable { vs_id, .. } => {
                let major = self.program_cache.get(vs_id)?.major;
                CachedKind::from_programmable(major, false)?
            }
        };
        let ps_kind = match shaders.ps {
            PsSource::FixedFunction { .. } => CachedKind::FfPs,
            PsSource::Programmable { ps_id, .. } => {
                let major = self.program_cache.get(ps_id)?.major;
                CachedKind::from_programmable(major, true)?
            }
        };
        Some((
            ShaderRecordRef::new(vs_kind, shaders.vs.disk_key()),
            ShaderRecordRef::new(ps_kind, shaders.ps.disk_key(shaders.variant)),
        ))
    }

    /// Whether a build queued now appends its record to the shader cache.
    ///
    /// Only after prewarm validated the file, and never once a write has
    /// failed or the cache is off.
    const fn cache_persists(&self) -> bool {
        self.flags.contains(FrameEncoderFlags::CACHE_READY)
            && !self.flags.contains(FrameEncoderFlags::CACHE_DISABLED)
    }

    fn enqueue(&mut self, job: CompileJob) -> JobTicket {
        let ticket = self.compile_tickets.issue();
        let enqueued_tsc = rdtsc();
        self.compile_in_flight.insert(ticket, enqueued_tsc);
        self.compile_queue
            .push(ticket, QueuedJob { job, enqueued_tsc });
        let pending = self.compile_in_flight.len();
        self.perf.compilation_mut().note_pending(pending);
        ticket
    }

    /// Install every build the workers have finished, without waiting.
    ///
    /// Called at `begin_frame` and by a draw whose probe found a build in
    /// flight, before it decides, so a finished build serves the next draw
    /// that needs it and a draw whose builds are done never pays for the
    /// check. Installing is bookkeeping only; the slow work happened on the
    /// worker.
    #[inline]
    pub fn drain_compile_results(&mut self) {
        if !self.compile_in_flight.is_empty() {
            self.drain_compile_results_pending();
        }
    }

    /// Warn once when a build has been in flight for longer than any compile takes.
    ///
    /// Called at `begin_frame`, and costs nothing while no build is in
    /// flight. A compiler service that hangs would otherwise leave the draws
    /// that need its build out of every frame with only the one info line
    /// that announced the first skip.
    pub fn check_stalled_compiles(&self) {
        let Some(&oldest) = self.compile_in_flight.values().min() else {
            return;
        };
        if rdtsc().saturating_sub(oldest) > secs_to_cycles(STALLED_BUILD_SECS) {
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET,
                "encoder: a shader or pipeline build has been in flight for over \
                 {STALLED_BUILD_SECS} s; the draws that need it are left out or wait \
                 until it lands"
            );
        }
    }

    #[cold]
    #[inline(never)]
    fn drain_compile_results_pending(&mut self) {
        while let Ok(result) = self.compile_results.try_recv() {
            self.install_compile(result, false);
        }
    }

    /// Whether a draw whose build is in flight is left out of this frame.
    ///
    /// Only under `shader.asyncCompile`, never while an occlusion query
    /// counts, and only when [`Self::targets_rebuilt`] holds for the bound
    /// targets and the depth and stencil planes in `planes`, the ones the
    /// draw tests or writes.
    /// Reached only from a pending resolve, so none of it costs a draw whose
    /// builds are done. A skip is counted and logged
    /// once; the caller drops the draw.
    ///
    /// The occlusion guard keeps a counted draw in its count. A depth-only
    /// draw left out before the query begins (a depth prepass) still makes
    /// the counted draws after it pass depth tests they would have failed,
    /// so that frame's count is high, never low: the application sees the
    /// object as more visible for one frame.
    pub fn skip_pending_draw(&mut self, planes: ClearPlanes) -> bool {
        if !self.flags.contains(FrameEncoderFlags::ASYNC_COMPILE)
            || self.visibility.active_count() != 0
            || !self.targets_rebuilt(planes)
        {
            return false;
        }
        mtld3d_shared::log_once_info!(
            target: LOG_TARGET,
            "encoder: leaving draws out of their frame while their shader or pipeline builds \
             (shader.asyncCompile); they appear once the build lands"
        );
        self.compile_stats.record_skipped_draw();
        self.perf.compilation_mut().note_skipped_draw();
        true
    }

    /// Whether everything a draw into the bound targets lands in is rebuilt every frame.
    ///
    /// Each colour target the pass attaches, and the depth and stencil
    /// planes named in `planes`, has to be rebuilt every frame and read only
    /// by work that is rebuilt every frame too. A colour target is rebuilt
    /// when it is the back buffer under the discard swap effect, which
    /// starts every frame undefined, or when a whole clear reached it in
    /// this frame and the one before; a depth or stencil plane only by the
    /// clears. It is read into kept content when one of the last
    /// `FEED_MEMORY_FRAMES` copied out of it with a `StretchRect` into a
    /// target that is not rebuilt, or sampled it in a pass whose own targets
    /// are not rebuilt ([`Self::note_frame_reads`]). A read-back to system
    /// memory is not such a read ([`Self::note_copy_source`] says why).
    ///
    /// Not covered: the first frame such a kept read happens in. Its marks
    /// are made when the frame is submitted, so a draw left out of a scratch
    /// target in that frame is missing from what the read makes.
    fn targets_rebuilt(&self, planes: ClearPlanes) -> bool {
        let history = &self.cleared_targets;
        let mut color = [false; 4];
        let mut count = 0;
        for (slot, (texture, subresource)) in color
            .iter_mut()
            .zip(self.pass_state.attached_color_targets())
        {
            *slot = (self.pass_state.is_discarded_back_buffer(texture)
                || history.regenerated(texture, subresource, ClearPlanes::COLOR))
                && !history.feeds_persistent(texture);
            count += 1;
        }
        let depth_texture = self.pass_state.current_depth_texture();
        let depth_level = self.pass_state.current_depth_level();
        let plane_rebuilt = |plane| {
            history.regenerated(depth_texture, depth_level, plane)
                && !history.feeds_persistent(depth_texture)
        };
        let depth = planes
            .contains(ClearPlanes::DEPTH)
            .then(|| plane_rebuilt(ClearPlanes::DEPTH));
        let stencil = planes
            .contains(ClearPlanes::STENCIL)
            .then(|| plane_rebuilt(ClearPlanes::STENCIL));
        may_skip_draw(&color[..count], depth, stencil)
    }

    /// Mark the textures the passes of this submission read into kept targets.
    ///
    /// Called once per submission, after its last pass closed and before
    /// any pass rule removes or merges a pass, so the recorded bind-to-pass
    /// indices still name the passes they were recorded against. The pass
    /// state records every texture bind with its pass while
    /// `shader.asyncCompile` is on (one push per bind command, never per
    /// draw); `mtld3d_core::async_compile::mark_kept_reads` judges them.
    /// A frame's marks are made at its end, so a kept read protects the
    /// frames after the one it first happens in.
    pub fn note_frame_reads(&mut self) {
        mark_kept_reads(&self.pass_state, &mut self.cleared_targets);
        self.pass_state.clear_pass_reads();
    }

    /// Account a `StretchRect` from `copy.src` into `copy.dst`.
    ///
    /// A copy covering a whole colour target rebuilds it as a clear does, so
    /// it counts toward the target's two-frame streak. The source is marked
    /// as feeding kept content unless the destination is itself rebuilt
    /// every frame and read by nothing kept, since only then is a draw
    /// missing from the source gone again the next frame. A no-op unless
    /// `shader.asyncCompile` is on.
    pub fn note_stretch_copy(&mut self, copy: &StretchCopyTargets) {
        if !self.flags.contains(FrameEncoderFlags::ASYNC_COMPILE) {
            return;
        }
        let dst = self.pass_state.identity_of(copy.dst);
        if copy.whole_color_dst {
            self.cleared_targets
                .record(dst, copy.dst_subresource, ClearPlanes::COLOR);
        }
        let dst_rebuilt = (self.pass_state.is_discarded_back_buffer(dst)
            || self
                .cleared_targets
                .regenerated(dst, copy.dst_subresource, ClearPlanes::COLOR))
            && !self.cleared_targets.feeds_persistent(dst);
        if !dst_rebuilt {
            self.note_copy_source(copy.src);
        }
    }

    /// Mark `texture` as copied into content the application may keep.
    ///
    /// For the source of a `StretchRect` into a kept target. A read-back to
    /// system memory marks nothing: the mark only protects frames after the
    /// read, and a read-back repeated every frame hands the application a
    /// fresh copy each time, so a draw missing from one is missing from one
    /// frame's copy only, while marking would keep every draw into a target
    /// the application reads back each frame (a probe, a picking buffer)
    /// from ever being left out. A one-off read-back, a screenshot, sees the
    /// frame as it was drawn, skipped draws included, which is the residual
    /// of the first frame of any read.
    fn note_copy_source(&mut self, texture: MetalHandle<MTLTextureKind>) {
        if self.flags.contains(FrameEncoderFlags::ASYNC_COMPILE) {
            let identity = self.pass_state.identity_of(texture);
            self.cleared_targets.mark_feeds_persistent(identity);
        }
    }

    /// Block until the builds behind `tickets` are installed, doing the work where it can.
    ///
    /// Each queued job is moved to the urgent lane; one no worker has
    /// started yet is taken back and built on this thread, since the
    /// encoder would only sit idle waiting for it. A job a worker already
    /// runs is waited for, installing whatever else finishes meanwhile.
    pub fn wait_for_compiles(&mut self, tickets: &[JobTicket]) {
        let mut wait_ns = 0;
        let timer = NanosSetTimer::start(&raw mut wait_ns);
        let mut stolen = 0u64;
        for &ticket in tickets {
            if self.compile_in_flight.contains_key(&ticket) {
                self.compile_queue.promote(ticket);
            }
        }
        while tickets
            .iter()
            .any(|ticket| self.compile_in_flight.contains_key(ticket))
        {
            let unstarted = tickets
                .iter()
                .find_map(|&ticket| self.compile_queue.steal(ticket).map(|job| (ticket, job)))
                .or_else(|| self.compile_queue.steal_urgent());
            if let Some((ticket, job)) = unstarted {
                stolen += 1;
                let result = run_job(ticket, job, false);
                self.install_compile(result, tickets.contains(&ticket));
                continue;
            }
            if let Ok(result) = self.compile_results.recv() {
                let waited = tickets.contains(&result.ticket);
                self.install_compile(result, waited);
            } else {
                self.abandon_lost_builds();
                break;
            }
        }
        drop(timer);
        self.perf
            .compilation_mut()
            .note_urgent_wait(wait_ns, stolen);
    }

    /// Wait until no build is in flight, before a `Reset` or teardown touches the caches.
    ///
    /// `cancel_unstarted` drops the jobs no worker has started instead of
    /// building them, for a teardown whose caches no draw will read again;
    /// a `Reset` keeps its caches, so it builds them. A job a worker runs
    /// is always waited for, since its handles have to reach the cache that
    /// destroys them.
    pub fn finish_compiles(&mut self, cancel_unstarted: bool) {
        while !self.compile_in_flight.is_empty() {
            if let Some((ticket, job)) = self.compile_queue.take_any() {
                if cancel_unstarted {
                    self.compile_in_flight.remove(&ticket);
                } else {
                    let result = run_job(ticket, job, false);
                    self.install_compile(result, true);
                }
                continue;
            }
            if let Ok(result) = self.compile_results.recv() {
                self.install_compile(result, false);
            } else {
                self.abandon_lost_builds();
            }
        }
    }

    /// Forget the builds lost with every compile worker, and build on the encoder from now on.
    ///
    /// The result channel only closes once no worker is left, so a job one
    /// of them had taken will never report. Its keys leave the pending maps,
    /// so they are unknown again rather than pending for good, and the skip
    /// is turned off, so the next draw that needs one queues it afresh and
    /// waits for it; with no worker to take it, that wait builds it inline
    /// on the encoder thread.
    #[cold]
    fn abandon_lost_builds(&mut self) {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "encoder: every compile worker is gone with {} build(s) in flight; \
             building on the encoder thread from now on",
            self.compile_in_flight.len()
        );
        self.compile_in_flight.clear();
        self.pending_libs.clear();
        self.pending_pipelines.clear();
        self.flags.remove(FrameEncoderFlags::ASYNC_COMPILE);
    }

    /// Record one finished build: index, owning cache, counters, and the cache latch.
    fn install_compile(&mut self, result: CompileResult, waited: bool) {
        let CompileResult {
            ticket,
            enqueued_tsc,
            on_worker,
            outcome,
        } = result;
        self.compile_in_flight.remove(&ticket);
        self.perf
            .compilation_mut()
            .note_install(enqueued_tsc, rdtsc());
        match outcome {
            Outcome::Library(outcome) => {
                if on_worker && !waited && outcome.handles.is_some() {
                    self.compile_stats.record_async_compile();
                }
                self.install_library(outcome);
            }
            Outcome::Pipeline(outcome) => self.install_pipeline(*outcome),
        }
    }

    fn install_library(&mut self, outcome: LibraryOutcome) {
        let LibraryOutcome {
            input,
            reference,
            handles,
            emitted,
            bucket,
            elapsed,
            total_ns,
            emit_ns,
            persist_ns,
            native,
            persist_error,
        } = outcome;
        if let Some(error) = persist_error {
            self.disable_cache_after_write_failure(&error);
        }
        let stage = input.stage();
        if let Some(handles) = handles {
            self.lib_cache.insert(reference, handles);
            if let Some(bucket) = bucket {
                self.compile_stats.record(bucket, elapsed);
            }
        } else {
            warn!(
                target: LOG_TARGET,
                "encoder: {stage} library {} failed to build, its draws are dropped without another attempt",
                reference_tag(reference)
            );
        }
        let device = self.device_handle.raw();
        let seq = self.current_submit_seq;
        let shader = PairShaderId {
            is_programmable: input.is_programmable(),
            hash: reference.key(),
        };
        let identity = || CompileIdentity::Shader {
            device,
            stage,
            shader,
        };
        let (total_kind, emit_kind) = if input.is_vertex() {
            (CompileKind::ShaderVs, CompileKind::EmitVs)
        } else {
            (CompileKind::ShaderPs, CompileKind::EmitPs)
        };
        let success = handles.is_some();
        let cache_disabled = self.flags.contains(FrameEncoderFlags::CACHE_DISABLED);
        let perf = self.perf.compilation_mut();
        perf.record(total_kind, total_ns, success, seq, identity);
        perf.record(emit_kind, emit_ns, emitted, seq, identity);
        perf.shader_parts(&native, success, seq, identity);
        if persist_ns != 0 {
            perf.record(
                CompileKind::CacheWrite,
                persist_ns,
                !cache_disabled,
                seq,
                identity,
            );
        }
        self.pending_libs.remove(&reference);
        match input {
            LibraryInput::FixedFunctionVs { key } => self.ff_vs_libs.record(key, handles),
            LibraryInput::ProgrammableVs {
                vs_id,
                provided_input_mask,
                clip_plane_count,
                sampler_kinds,
                ..
            } => self.prog_vs_libs.record(
                (vs_id, provided_input_mask, clip_plane_count, sampler_kinds),
                handles,
            ),
            LibraryInput::FixedFunctionPs { key, variant } => self
                .ff_ps_libs
                .entry(key)
                .or_default()
                .record(variant, handles),
            LibraryInput::ProgrammablePs { ps_id, variant, .. } => {
                self.prog_ps_libs.record((ps_id, variant), handles);
            }
        }
    }

    fn install_pipeline(&mut self, outcome: PipelineOutcome) {
        let PipelineOutcome {
            key,
            snapshot,
            vs,
            ps,
            sibling_of,
            handle,
            total_ns,
            persist_ns,
            native,
            persist_error,
        } = outcome;
        if let Some(error) = persist_error {
            self.disable_cache_after_write_failure(&error);
        }
        let success = handle.is_some();
        let sibling = sibling_of.is_some();
        let device = self.device_handle.raw();
        let seq = self.current_submit_seq;
        let cache_disabled = self.flags.contains(FrameEncoderFlags::CACHE_DISABLED);
        let identity = || CompileIdentity::Pipeline {
            device,
            vs,
            ps,
            snapshot: Box::new(snapshot.clone()),
            sibling,
        };
        let perf = self.perf.compilation_mut();
        perf.record(
            if sibling {
                CompileKind::Sibling
            } else {
                CompileKind::Pipeline
            },
            total_ns,
            success,
            seq,
            identity,
        );
        perf.record(
            CompileKind::PipelinePreparation,
            native.preparation_ns,
            success || native.build_ns != 0,
            seq,
            identity,
        );
        if native.build_ns != 0 {
            perf.record(
                CompileKind::PipelineBuild,
                native.build_ns,
                success,
                seq,
                identity,
            );
        }
        if persist_ns != 0 {
            perf.record(
                CompileKind::PipelineCacheWrite,
                persist_ns,
                !cache_disabled,
                seq,
                identity,
            );
        }
        if let Some(handle) = handle {
            if let Some(primary) = sibling_of {
                self.no_color_pipeline_alt.insert(primary, handle);
            }
        } else {
            error!(target: LOG_TARGET, "encoder: CreateRenderPipeline failed");
            let consequence = if sibling {
                "its passes keep their color attachment"
            } else {
                "its draws are dropped"
            };
            warn!(
                target: LOG_TARGET,
                "encoder: render pipeline (VS {}, PS {}, sibling={sibling}) failed to build, {consequence} without another attempt",
                vs.tag(),
                ps.tag()
            );
        }
        self.pending_pipelines.remove(&key);
        self.pipeline_cache.record(key, handle);
    }

    /// Latch the shader cache off after a worker's append failed.
    fn disable_cache_after_write_failure(&mut self, error: &std::io::Error) {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "shader_cache: write mtld3d_shaders.bin failed, cache disabled: {error}"
        );
        self.flags.insert(FrameEncoderFlags::CACHE_DISABLED);
    }

    /// Remember that the colour targets the next pass attaches were cleared whole.
    pub fn note_color_targets_cleared(&mut self) {
        for (texture, subresource) in self.pass_state.attached_color_targets() {
            self.cleared_targets
                .record(texture, subresource, ClearPlanes::COLOR);
        }
    }

    /// Remember that `planes` of the bound depth-stencil attachment were cleared whole.
    pub fn note_depth_stencil_cleared(&mut self, planes: ClearPlanes) {
        self.cleared_targets.record(
            self.pass_state.current_depth_texture(),
            self.pass_state.current_depth_level(),
            planes,
        );
    }
}

/// `prog 0x…` / `ff 0x…` for a vertex-shader source, keyed as its library is.
fn vs_source_tag(source: &VsSource) -> String {
    PairShaderId {
        is_programmable: matches!(source, VsSource::Programmable { .. }),
        hash: source.disk_key(),
    }
    .tag()
}

/// `prog 0x…` / `ff 0x…` for a pixel-shader source and variant, keyed as its library is.
fn ps_source_tag(source: &PsSource, variant: VariantKey) -> String {
    PairShaderId {
        is_programmable: matches!(source, PsSource::Programmable { .. }),
        hash: source.disk_key(variant),
    }
    .tag()
}
