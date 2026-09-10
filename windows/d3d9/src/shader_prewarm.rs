//! Shader and render-pipeline cache prewarm thread.
//!
//! Spawned once at `CreateDevice`, reads `<host-exe-dir>/mtld3d_shaders.bin`,
//! recreates every valid shader library and render pipeline, and ships the
//! resulting device-local handles to the encoder over a dedicated one-shot
//! `PrewarmSender` channel. The encoder blocks on that channel before draining
//! any `EncoderMessage`, so live miss-compiles cannot duplicate prewarm work.

use std::{
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use log::{error, info};
use mtld3d_core::{
    perf::{
        PairShaderId,
        compilation::{Identity as CompileIdentity, Kind as CompileKind},
    },
    pipeline_state::{self, PipelineBuildInputs},
    shader_cache::{self, CacheLoad, CachedKind, ShaderRecordRef},
    shader_compile_stats::{CompileBucket, Snapshot, format_summary},
};
use mtld3d_shared::{
    MetalHandle,
    mtl::StageTag,
    mtl_handle::{MTLDeviceKind, MTLRenderPipelineStateKind},
    perf::NanosSetTimer,
};
use rustc_hash::FxHashMap;

use crate::{
    LOG_TARGET,
    encoder::{PrewarmSender, WarmCache, compile_stage_library, shader_cache_path},
    unix_call::unix_call,
};

/// Lifetime handle for the prewarm thread.
///
/// Held by `DeviceInner` and cancelled at `device_release` so a long Metal
/// `newLibraryWithSource:` in flight can't issue `unix_call`s concurrently
/// with `shutdown_cleanup` (Metal's device-internal locks would serialise
/// the two and stretch shutdown into the seconds).
pub struct PrewarmHandle {
    stop: Arc<AtomicBool>,
    join: Option<JoinHandle<()>>,
}

impl PrewarmHandle {
    /// Set the stop flag and wait for the prewarm thread to finish.
    ///
    /// The prewarm loop checks the flag between compiles, so the wait
    /// does not start another shader or pipeline compile after cancellation.
    /// An in-flight Metal call or cache operation must still finish. Idempotent.
    pub fn cancel_and_join(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(j) = self.join.take() {
            // Don't call `j.join()`. On long sessions Wine reports
            // `STATUS_INVALID_HANDLE` for the prewarm thread's Win32
            // handle (mechanism not yet identified —
            // `server/thread.c:1141` `wait_on_handles` ->
            // `get_handle_obj` NULL -> "os error 6"). std's
            // `JoinHandle::join` panics on `WAIT_FAILED`, and
            // `panic = "abort"` makes `catch_unwind` a no-op.
            // `is_finished` reads the std Packet `Arc` strong count
            // (handle-independent); Drop CloseHandles the
            // possibly-invalid handle silently.
            while !j.is_finished() {
                thread::sleep(Duration::from_millis(1));
            }
            drop(j);
        }
    }
}

/// Spawn the pre-warm thread for one `CreateDevice` call.
///
/// Each call spawns its own thread because each device gets a distinct
/// `EncoderThread` whose `cache_ready` must be flipped via its own
/// `PrewarmSender`. `MTLLibrary` handles compiled for one `MTLDevice` would
/// not be valid on another, so per-device runs are also correct (no shared
/// cross-device state).
pub fn spawn(
    device_handle: MetalHandle<MTLDeviceKind>,
    sender: PrewarmSender,
    shader_cache: bool,
) -> PrewarmHandle {
    let stop = Arc::new(AtomicBool::new(false));
    let stop_for_thread = stop.clone();
    let join = thread::Builder::new()
        .name("mtld3d-shader-prewarm".into())
        .spawn(move || run(device_handle, sender, &stop_for_thread, shader_cache))
        .ok();
    PrewarmHandle { stop, join }
}

/// The pre-warm body; `shader_cache` is the interface's `shaderCache.enable`.
fn run(
    device_handle: MetalHandle<MTLDeviceKind>,
    sender: PrewarmSender,
    stop: &AtomicBool,
    shader_cache: bool,
) {
    if !shader_cache {
        info!(
            target: LOG_TARGET,
            "shader_cache: shaderCache.enable = false, skipping pre-warm"
        );
        sender.send(WarmCache::empty());
        return;
    }
    let started = Instant::now();

    let Some(path) = shader_cache_path() else {
        sender.send(WarmCache::empty());
        return;
    };

    let records = match shader_cache::load(&path) {
        Ok(CacheLoad::Missing) => {
            sender.send(WarmCache::empty());
            return;
        }
        Ok(CacheLoad::InvalidatedVersion(header)) => {
            info!(
                target: LOG_TARGET,
                "shader_cache: cache format {} / shader schema {} is stale, wiped mtld3d_shaders.bin",
                header.format_version,
                header.shader_schema_version,
            );
            sender.send(WarmCache::empty());
            return;
        }
        Ok(CacheLoad::InvalidatedWrongMagic) => {
            info!(
                target: LOG_TARGET,
                "shader_cache: wrong magic in mtld3d_shaders.bin, wiped"
            );
            sender.send(WarmCache::empty());
            return;
        }
        Ok(CacheLoad::Current(records)) => records,
        Err(e) => {
            info!(
                target: LOG_TARGET,
                "shader_cache: read mtld3d_shaders.bin failed, cache disabled: {e}"
            );
            sender.send_disabled();
            return;
        }
    };

    let mut compilation = mtld3d_core::perf::compilation::CompilationPerf::new();
    let mut libraries = FxHashMap::default();
    let mut counts = [0u32; 4];
    let mut duration_ns = [0u64; 4];

    for entry in &records.shaders {
        if stop.load(Ordering::Acquire) {
            break;
        }
        let stage = stage_for_kind(entry.kind);
        let entry_name = entry.kind.entry_name(entry.key);
        let started = Instant::now();
        let mut timings = mtld3d_shared::perf::ShaderTimings::new();
        let handles =
            compile_stage_library(device_handle, stage, &entry.msl, &entry_name, &mut timings);
        let elapsed = started.elapsed();
        compilation.shader_parts(&timings, handles.is_some(), 0, || {
            mtld3d_core::perf::compilation::Identity::Prewarm {
                device: device_handle.raw(),
                kind: entry.kind,
                key: entry.key,
            }
        });
        let Some(handles) = handles else {
            continue;
        };
        let idx = bucket_index(entry.kind.compile_bucket());
        counts[idx] += 1;
        duration_ns[idx] += u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX);
        libraries.insert(ShaderRecordRef::new(entry.kind, entry.key), handles);
    }

    let total: u32 = counts.iter().sum();
    let cached = u32::try_from(libraries.len()).unwrap_or(u32::MAX);
    let mut pipelines: FxHashMap<
        mtld3d_core::pipeline_state::PipelineKey,
        MetalHandle<MTLRenderPipelineStateKind>,
    > = FxHashMap::default();
    let mut primary_candidates = Vec::new();

    if !stop.load(Ordering::Acquire) {
        for recipe in &records.pipelines {
            if stop.load(Ordering::Acquire) {
                break;
            }
            let Some(vs) = libraries.get(&recipe.vs()) else {
                mtld3d_shared::log_once_warn_by!(
                    target: LOG_TARGET,
                    key: recipe.disk_key(),
                    "shader_cache: pipeline recipe skipped after VS prewarm failure"
                );
                continue;
            };
            let Some(ps) = libraries.get(&recipe.ps()) else {
                mtld3d_shared::log_once_warn_by!(
                    target: LOG_TARGET,
                    key: recipe.disk_key(),
                    "shader_cache: pipeline recipe skipped after PS prewarm failure"
                );
                continue;
            };
            let snapshot = recipe.resolve(vs.func, ps.func);
            let key = pipeline_state::key_from_snapshot(&snapshot);
            if pipelines.contains_key(&key) {
                continue;
            }
            let mut total_ns = 0;
            let timer = NanosSetTimer::start(&raw mut total_ns);
            let vertex_layouts = pipeline_state::vertex_layouts_from_snapshot(&snapshot);
            let mut params = pipeline_state::params_from_snapshot(&PipelineBuildInputs {
                snapshot: &snapshot,
                vertex_attrs: recipe.vertex_attrs(),
                vertex_layouts: &vertex_layouts,
                device_handle,
            });
            let status = unix_call(&mut params);
            let pipeline = params.pipeline_handle;
            let timings = params.timings.into_inner();
            drop(timer);
            let success = status == 0 && !pipeline.is_null();
            record_pipeline(
                &mut compilation,
                &PipelineMeasurement {
                    device: device_handle,
                    vs: recipe.vs(),
                    ps: recipe.ps(),
                    snapshot: &snapshot,
                    total_ns,
                    timings: &timings,
                    success,
                },
            );
            if !success {
                error!(target: LOG_TARGET, "shader_cache: pipeline prewarm failed");
                continue;
            }
            if snapshot.writes_no_color() && snapshot.has_color_output() {
                primary_candidates.push((snapshot.clone(), pipeline.raw()));
            }
            pipelines.insert(key, pipeline);
        }
    }

    let mut no_color_siblings = Vec::new();
    for (mut snapshot, primary) in primary_candidates {
        snapshot
            .attach
            .remove(mtld3d_core::pipeline_state::PipelineAttachFlags::HAS_COLOR_OUTPUT);
        snapshot.extra = mtld3d_core::pipeline_state::ExtraColorAttachments::NONE;
        let key = pipeline_state::key_from_snapshot(&snapshot);
        if let Some(&sibling) = pipelines.get(&key) {
            no_color_siblings.push((primary, sibling));
        }
    }

    if records.needs_compaction && !stop.load(Ordering::Acquire) {
        rewrite_as_bundle(&path);
    }

    let pipeline_count = pipelines.len();
    compilation.log_startup(device_handle.raw());
    if total > 0 {
        let snap = Snapshot {
            counts,
            duration_ns,
        };
        info!(target: LOG_TARGET, "{}", format_summary(&snap, "pre-warmed", cached));
    }
    info!(
        target: LOG_TARGET,
        "shader_cache: pre-warmed {pipeline_count} render pipelines, {} no-color mappings; \
         startup {:.3}s, cancelled={}",
        no_color_siblings.len(),
        started.elapsed().as_secs_f64(),
        stop.load(Ordering::Acquire),
    );
    sender.send(WarmCache {
        libraries: libraries.into_iter().collect(),
        pipelines: pipelines.into_iter().collect(),
        no_color_siblings,
    });
}

/// Replace `path` with one Bundle containing the latest valid records.
///
/// `shader_cache::compact` rereads while holding the sidecar lock and renames a
/// temporary into place. Best-effort: any I/O failure logs once and leaves the
/// original file untouched. The next launch tries again.
fn rewrite_as_bundle(path: &Path) {
    match shader_cache::compact(path) {
        Ok(Some((records, len))) => info!(
            target: LOG_TARGET,
            "shader_cache: compacted {records} records into one Bundle ({len} bytes)"
        ),
        Ok(None) => {}
        Err(e) => mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "shader_cache: compaction of {} failed → leaving original: {e}",
            path.display()
        ),
    }
}

struct PipelineMeasurement<'a> {
    device: MetalHandle<MTLDeviceKind>,
    vs: ShaderRecordRef,
    ps: ShaderRecordRef,
    snapshot: &'a mtld3d_core::pipeline_state::PipelineSnapshot,
    total_ns: u64,
    timings: &'a mtld3d_shared::perf::PipelineTimings,
    success: bool,
}

fn record_pipeline(
    compilation: &mut mtld3d_core::perf::compilation::CompilationPerf,
    measurement: &PipelineMeasurement<'_>,
) {
    let sibling = !measurement.snapshot.has_color_output();
    let identity = || CompileIdentity::Pipeline {
        device: measurement.device.raw(),
        vs: PairShaderId {
            is_programmable: measurement.vs.kind().is_programmable(),
            hash: measurement.vs.key(),
        },
        ps: PairShaderId {
            is_programmable: measurement.ps.kind().is_programmable(),
            hash: measurement.ps.key(),
        },
        snapshot: Box::new(measurement.snapshot.clone()),
        sibling,
    };
    compilation.record(
        if sibling {
            CompileKind::Sibling
        } else {
            CompileKind::Pipeline
        },
        measurement.total_ns,
        measurement.success,
        0,
        identity,
    );
    compilation.record(
        CompileKind::PipelinePreparation,
        measurement.timings.preparation_ns,
        measurement.success || measurement.timings.build_ns != 0,
        0,
        identity,
    );
    if measurement.timings.build_ns != 0 {
        compilation.record(
            CompileKind::PipelineBuild,
            measurement.timings.build_ns,
            measurement.success,
            0,
            identity,
        );
    }
}

const fn stage_for_kind(kind: CachedKind) -> StageTag {
    match kind {
        CachedKind::FfVs | CachedKind::Sm1Vs | CachedKind::Sm2Vs | CachedKind::Sm3Vs => {
            StageTag::Vertex
        }
        CachedKind::FfPs | CachedKind::Sm1Ps | CachedKind::Sm2Ps | CachedKind::Sm3Ps => {
            StageTag::Fragment
        }
    }
}

const fn bucket_index(bucket: CompileBucket) -> usize {
    match bucket {
        CompileBucket::Ff => 0,
        CompileBucket::Sm1 => 1,
        CompileBucket::Sm2 => 2,
        CompileBucket::Sm3 => 3,
    }
}
