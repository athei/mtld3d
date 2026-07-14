//! Shader-cache pre-warm thread.
//!
//! Spawned once at `CreateDevice`, reads `<host-exe-dir>/mtld3d_shaders.bin`,
//! calls `CompileShaderLibrary` for every cached MSL entry, and ships the
//! resulting `MTLLibrary` handles to the encoder over a dedicated one-shot
//! `PrewarmSender` channel. Fires after the encoder thread is up; the
//! encoder blocks on that channel before draining any `EncoderMessage`, so
//! live miss-compiles can never race the prewarm and duplicate work that's
//! about to land in `lib_cache`.

use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use log::info;
use mtld3d_core::{
    shader_cache::{self, CacheEntry, CachedKind, SHADER_CACHE_SCHEMA_VERSION},
    shader_compile_stats::{CompileBucket, Snapshot, format_summary},
};
use mtld3d_shared::{MetalHandle, mtl::StageTag, mtl_handle::MTLDeviceKind};

use crate::{
    LOG_TARGET,
    encoder::{
        PrewarmSender, StageLibHandles, compile_stage_library, shader_cache_enabled,
        shader_cache_path,
    },
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
    /// is bounded by one in-flight `compile_stage_library`
    /// (~tens-to-hundreds of ms). Idempotent.
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
pub fn spawn(device_handle: MetalHandle<MTLDeviceKind>, sender: PrewarmSender) -> PrewarmHandle {
    let stop = Arc::new(AtomicBool::new(false));
    let stop_for_thread = stop.clone();
    let join = thread::Builder::new()
        .name("mtld3d-shader-prewarm".into())
        .spawn(move || run(device_handle, sender, &stop_for_thread))
        .ok();
    PrewarmHandle { stop, join }
}

fn run(device_handle: MetalHandle<MTLDeviceKind>, sender: PrewarmSender, stop: &AtomicBool) {
    if !shader_cache_enabled() {
        info!(
            target: LOG_TARGET,
            "shader_cache: shaderCache.enable = false, skipping pre-warm"
        );
        sender.send(Vec::new());
        return;
    }

    let Some(path) = shader_cache_path() else {
        sender.send(Vec::new());
        return;
    };

    let bytes = match fs::read(&path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            // Cold start — no file yet.
            sender.send(Vec::new());
            return;
        }
        Err(e) => {
            // File exists but we couldn't read it (permission, I/O, …).
            // Don't attempt `remove_file` — whatever blocked the read
            // likely blocks the delete too, and a half-failed wipe
            // leaves the encoder appending past foreign content.
            // Disable cache writes for the session instead so the
            // existing file stays untouched.
            info!(
                target: LOG_TARGET,
                "shader_cache: read mtld3d_shaders.bin failed → cache disabled: {e}"
            );
            sender.send_disabled();
            return;
        }
    };

    match shader_cache::read_header(&bytes) {
        Ok(schema) if schema == SHADER_CACHE_SCHEMA_VERSION => {}
        Ok(other) => {
            // Schema bump → wipe and rebuild from scratch.
            let _ = fs::remove_file(&path);
            info!(
                target: LOG_TARGET,
                "shader_cache: schema {other} != current {SHADER_CACHE_SCHEMA_VERSION}, wiped mtld3d_shaders.bin"
            );
            sender.send(Vec::new());
            return;
        }
        Err(_) => {
            // Foreign magic at our path. Wipe so the encoder doesn't
            // later append fresh chunks past the foreign content (the
            // file's `open_or_create_cache_file` only writes a header
            // when the file is absent).
            let _ = fs::remove_file(&path);
            info!(
                target: LOG_TARGET,
                "shader_cache: wrong magic in mtld3d_shaders.bin, wiped"
            );
            sender.send(Vec::new());
            return;
        }
    }

    let (entries, needs_compaction) = shader_cache::read_records(&bytes);

    // Dedup by `disk_key` up front so both the compile loop and the
    // compaction rewrite consume one canonical sequence — without this
    // we'd pay `newLibraryWithSource:` cost N times for a key that
    // appeared in both the bundle and a later append.
    let mut deduped: Vec<CacheEntry> = Vec::with_capacity(entries.len());
    let mut seen = HashSet::with_capacity(entries.len());
    for entry in entries {
        if seen.insert(entry.key) {
            deduped.push(entry);
        }
    }

    // Atomic rewrite into one dense Bundle chunk if the file isn't
    // already in that shape. Must complete before `sender.send` because
    // the encoder opens the file for append only after the prewarm
    // payload arrives — the rename therefore always lands before any
    // new append-handle is opened against this path.
    if needs_compaction && !deduped.is_empty() {
        rewrite_as_bundle(&path, &deduped);
    }

    let mut warm: Vec<(u64, StageLibHandles)> = Vec::with_capacity(deduped.len());
    let mut counts = [0u32; 4];
    let mut duration_ns = [0u64; 4];

    for entry in &deduped {
        if stop.load(Ordering::Acquire) {
            break;
        }
        let stage = stage_for_kind(entry.kind);
        let entry_name = entry.kind.entry_name(entry.key);
        let started = Instant::now();
        let Some(handles) = compile_stage_library(device_handle, stage, &entry.msl, &entry_name)
        else {
            continue;
        };
        let elapsed = started.elapsed();
        let idx = bucket_index(entry.kind.compile_bucket());
        counts[idx] += 1;
        // u128 nanos → u64: saturates at ~584 years; pre-warm batch fits easily.
        duration_ns[idx] += u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX);
        warm.push((entry.key, handles));
    }

    let total: u32 = counts.iter().sum();
    let cached = u32::try_from(warm.len()).unwrap_or(u32::MAX);
    sender.send(warm);
    if total > 0 {
        let snap = Snapshot {
            counts,
            duration_ns,
        };
        info!(target: LOG_TARGET, "{}", format_summary(&snap, "pre-warmed", cached));
    }
}

/// Replace `path` with a fresh file containing one Bundle chunk holding every entry.
///
/// Written to `<path>.tmp` first and then atomically renamed over the
/// original. Best-effort: any I/O failure logs once and leaves the original
/// file untouched (worst case is a missed size optimisation; the next launch
/// tries again).
fn rewrite_as_bundle(path: &Path, entries: &[CacheEntry]) {
    let mut buf = Vec::new();
    shader_cache::write_header(&mut buf);
    shader_cache::write_bundle(&mut buf, entries);

    let tmp: PathBuf = {
        let mut p = path.as_os_str().to_owned();
        p.push(".tmp");
        PathBuf::from(p)
    };
    if let Err(e) = fs::write(&tmp, &buf) {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "shader_cache: compaction write to {} failed → leaving original: {e}",
            tmp.display()
        );
        return;
    }
    if let Err(e) = fs::rename(&tmp, path) {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "shader_cache: compaction rename {} → {} failed → leaving original: {e}",
            tmp.display(),
            path.display()
        );
        let _ = fs::remove_file(&tmp);
        return;
    }
    info!(
        target: LOG_TARGET,
        "shader_cache: compacted {} entries into one Bundle ({} bytes)",
        entries.len(),
        buf.len()
    );
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
