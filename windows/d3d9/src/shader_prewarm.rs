//! Shader and render-pipeline cache prewarm thread.
//!
//! Spawned once at `CreateDevice`, reads `<host-exe-dir>/mtld3d_shaders.bin`,
//! recreates every valid shader library and render pipeline, and ships the
//! resulting device-local handles to the encoder over a dedicated one-shot
//! completion channel. The encoder blocks on that channel before draining
//! any `EncoderMessage`, so live miss-compiles cannot duplicate prewarm work.

use std::{
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::Receiver,
    },
    time::Instant,
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
    shader_prewarm::PrewarmHandle,
    startup_work,
};
use mtld3d_shared::{
    MetalHandle,
    mtl::StageTag,
    mtl_handle::{MTLDeviceKind, MTLRenderPipelineStateKind},
    perf::NanosSetTimer,
};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    LOG_TARGET,
    encoder::{StageLibHandles, WarmCache, compile_stage_library, shader_cache_path},
    unix_call::unix_call,
};

/// Start prewarm for one device and return its startup barrier.
///
/// Every device owns its libraries and pipelines. The worker's result reaches
/// the encoder before frames, and an unavailable worker disables cache writes.
pub fn spawn(
    device_handle: MetalHandle<MTLDeviceKind>,
    shader_cache: bool,
) -> (PrewarmHandle, Receiver<Option<WarmCache>>) {
    PrewarmHandle::spawn(move |stop| run(device_handle, stop, shader_cache))
}

/// The pre-warm body; `shader_cache` is the interface's `shaderCache.enable`.
fn run(
    device_handle: MetalHandle<MTLDeviceKind>,
    stop: &AtomicBool,
    shader_cache: bool,
) -> Option<WarmCache> {
    if !shader_cache {
        info!(
            target: LOG_TARGET,
            "shader_cache: shaderCache.enable = false, skipping pre-warm"
        );
        return Some(WarmCache::empty());
    }
    let started = Instant::now();

    let Some(path) = shader_cache_path() else {
        return Some(WarmCache::empty());
    };

    let mut records = match shader_cache::load(&path) {
        Ok(CacheLoad::Missing) => {
            return Some(WarmCache::empty());
        }
        Ok(CacheLoad::InvalidatedVersion(header)) => {
            info!(
                target: LOG_TARGET,
                "shader_cache: cache format {} / shader schema {} is stale, wiped mtld3d_shaders.bin",
                header.format_version,
                header.shader_schema_version,
            );
            return Some(WarmCache::empty());
        }
        Ok(CacheLoad::InvalidatedWrongMagic) => {
            info!(
                target: LOG_TARGET,
                "shader_cache: wrong magic in mtld3d_shaders.bin, wiped"
            );
            return Some(WarmCache::empty());
        }
        Ok(CacheLoad::Current(records)) => records,
        Err(e) => {
            info!(
                target: LOG_TARGET,
                "shader_cache: read mtld3d_shaders.bin failed, cache disabled: {e}"
            );
            return None;
        }
    };

    let mut compilation = mtld3d_core::perf::compilation::CompilationPerf::new();
    let mut libraries = FxHashMap::default();
    let mut counts = [0u32; 4];
    let mut duration_ns = [0u64; 4];

    let mut refreshed = 0u32;
    let mut ready_shaders = Vec::with_capacity(records.shaders.len());
    for entry in &mut records.shaders {
        if stop.load(Ordering::Acquire) {
            break;
        }
        match entry.refresh_msl() {
            Ok(true) => {
                refreshed += 1;
                records.needs_compaction = true;
                if let Err(error) = shader_cache::CacheWriter::open(&path)
                    .and_then(|writer| writer.append_shader(entry))
                {
                    mtld3d_shared::log_once_warn!(
                        target: LOG_TARGET,
                        "shader_cache: persisting regenerated MSL failed: {error}"
                    );
                }
            }
            Ok(false) => {}
            Err(error) => {
                mtld3d_shared::log_once_warn_by!(
                    target: LOG_TARGET,
                    key: entry.key,
                    "shader_cache: regenerating {:?} {:#x} failed: {error}", entry.kind, entry.key
                );
                continue;
            }
        }
        ready_shaders.push(&*entry);
    }

    // Regeneration and persistence keep disk order. Native calls share only the
    // device and immutable source; all admitted calls finish before PSO work.
    let compiled = startup_work::map(&ready_shaders, stop, |entry| {
        compile_library(device_handle, entry)
    });
    for (index, result) in compiled {
        let entry = ready_shaders[index];
        let LibraryCompilation {
            handles,
            timings,
            elapsed_ns,
        } = result;
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
        duration_ns[idx] += elapsed_ns;
        libraries.insert(ShaderRecordRef::new(entry.kind, entry.key), handles);
    }

    if refreshed != 0 {
        info!(target: LOG_TARGET, "shader_cache: regenerated MSL for {refreshed} retained DXSO variants");
    }
    let total: u32 = counts.iter().sum();
    let cached = u32::try_from(libraries.len()).unwrap_or(u32::MAX);
    let mut pipelines: FxHashMap<
        mtld3d_core::pipeline_state::PipelineKey,
        MetalHandle<MTLRenderPipelineStateKind>,
    > = FxHashMap::default();
    let mut primary_candidates = Vec::new();
    let mut ready_pipelines = Vec::new();
    let mut scheduled = FxHashSet::default();

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
            let key = pipeline_state::key_from_snapshot(&snapshot, recipe.vertex_attrs());
            // One attempt per resolved key in this startup, including failures.
            // Retained recipes remain available for a later startup or live miss.
            if scheduled.insert(key) {
                ready_pipelines.push((recipe, snapshot));
            }
        }
    }
    let compiled = startup_work::map(&ready_pipelines, stop, |(recipe, snapshot)| {
        compile_pipeline(device_handle, recipe, snapshot)
    });
    for (index, result) in compiled {
        let (recipe, snapshot) = &ready_pipelines[index];
        let PipelineCompilation {
            pipeline,
            timings,
            total_ns,
            success,
        } = result;
        record_pipeline(
            &mut compilation,
            &PipelineMeasurement {
                device: device_handle,
                vs: recipe.vs(),
                ps: recipe.ps(),
                snapshot,
                total_ns,
                timings: &timings,
                success,
            },
        );
        if !success {
            error!(target: LOG_TARGET, "shader_cache: pipeline prewarm failed");
            continue;
        }
        if snapshot.has_depth() && snapshot.writes_no_color() && snapshot.has_color_output() {
            primary_candidates.push((snapshot.clone(), recipe.vertex_attrs(), pipeline.raw()));
        }
        pipelines.insert(
            pipeline_state::key_from_snapshot(snapshot, recipe.vertex_attrs()),
            pipeline,
        );
    }

    let mut no_color_siblings = Vec::new();
    for (mut snapshot, vertex_attrs, primary) in primary_candidates {
        snapshot
            .attach
            .remove(mtld3d_core::pipeline_state::PipelineAttachFlags::HAS_COLOR_OUTPUT);
        snapshot.extra = mtld3d_core::pipeline_state::ExtraColorAttachments::NONE;
        let key = pipeline_state::key_from_snapshot(&snapshot, vertex_attrs);
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
    Some(WarmCache {
        libraries: libraries.into_iter().collect(),
        pipelines: pipelines.into_iter().collect(),
        no_color_siblings,
    })
}

/// One native shader result, retained until the coordinator records it.
struct LibraryCompilation {
    handles: Option<StageLibHandles>,
    timings: mtld3d_shared::perf::ShaderTimings,
    elapsed_ns: u64,
}

fn compile_library(
    device: MetalHandle<MTLDeviceKind>,
    entry: &shader_cache::CacheEntry,
) -> LibraryCompilation {
    let entry_name = entry.kind.entry_name(entry.key);
    let started = Instant::now();
    let mut timings = mtld3d_shared::perf::ShaderTimings::new();
    let handles = compile_stage_library(
        device,
        stage_for_kind(entry.kind),
        &entry.msl,
        &entry_name,
        &mut timings,
    );
    LibraryCompilation {
        handles,
        timings,
        elapsed_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
    }
}

/// One native PSO result, retained until the coordinator records it.
struct PipelineCompilation {
    pipeline: MetalHandle<MTLRenderPipelineStateKind>,
    timings: mtld3d_shared::perf::PipelineTimings,
    total_ns: u64,
    success: bool,
}

fn compile_pipeline(
    device_handle: MetalHandle<MTLDeviceKind>,
    recipe: &shader_cache::PipelineRecipe,
    snapshot: &pipeline_state::PipelineSnapshot,
) -> PipelineCompilation {
    let mut total_ns = 0;
    let timer = NanosSetTimer::start(&raw mut total_ns);
    let vertex_layouts = pipeline_state::vertex_layouts_from_snapshot(snapshot);
    let mut params = pipeline_state::params_from_snapshot(&PipelineBuildInputs {
        snapshot,
        vertex_attrs: recipe.vertex_attrs(),
        vertex_layouts: &vertex_layouts,
        device_handle,
    });
    let status = unix_call(&mut params);
    let pipeline = params.pipeline_handle;
    let timings = params.timings.into_inner();
    drop(timer);
    PipelineCompilation {
        pipeline,
        timings,
        total_ns,
        success: status == 0 && !pipeline.is_null(),
    }
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
