use log::{info, warn};
use mtld3d_shared::{MetalHandle, mtl_handle::MTLDeviceKind};
use objc2_foundation::{NSString, NSURL};
use objc2_metal::{MTLCaptureDescriptor, MTLCaptureDestination, MTLCaptureManager};

use crate::{LOG_TARGET, metal::handle::IntoRetained};

/// Where a trace goes when the process has no log directory yet.
const FALLBACK_PATH: &str = "/tmp/mtld3d_capture.gputrace";

/// Begin a Metal GPU frame capture.
///
/// The capture object is the device passed in (covers all command queues
/// on it). Output is a `.gputrace` document next to the process's log file,
/// numbered per press (`log_file::next_trace_path`), or at `FALLBACK_PATH`
/// when no log location was named; openable in Xcode or with `gpudebug`.
///
/// Apple gates this on `MTL_CAPTURE_ENABLED=1` at process launch; when
/// the env is unset `startCaptureWithDescriptor` returns an error which
/// we surface as a single warn (doesn't repeat per attempt at this site,
/// but the user-visible action, the Ctrl+Shift+P hotkey, already
/// self-rate-limits to one press).
pub fn start_capture(device_handle: MetalHandle<MTLDeviceKind>) {
    // SAFETY: `sharedCaptureManager` is an always-live process-wide singleton
    // per the Metal capture API; the typed objc2 binding requires `unsafe`
    // because the trait method is marked so.
    let manager = unsafe { MTLCaptureManager::sharedCaptureManager() };
    if manager.isCapturing() {
        warn!(target: LOG_TARGET, "start_capture: a capture is already in progress, ignoring");
        return;
    }
    let Some(device) = device_handle.into_retained() else {
        warn!(target: LOG_TARGET, "start_capture: device_handle is null, cannot capture");
        return;
    };

    let desc = MTLCaptureDescriptor::new();
    // SAFETY: `device` is a freshly retained `MTLDevice` we just decoded; the
    // setter only borrows the object for the duration of the call.
    unsafe { desc.setCaptureObject(Some(device.as_ref())) };
    desc.setDestination(MTLCaptureDestination::GPUTraceDocument);

    let path = crate::log_file::next_trace_path().map_or_else(
        || FALLBACK_PATH.to_owned(),
        |p| p.to_string_lossy().into_owned(),
    );
    // Trace path must not exist; remove a stale one from a prior run.
    let _ = std::fs::remove_dir_all(&path);
    let _ = std::fs::remove_file(&path);

    let path_ns = NSString::from_str(&path);
    let url = NSURL::fileURLWithPath(&path_ns);
    desc.setOutputURL(Some(&url));

    match manager.startCaptureWithDescriptor_error(&desc) {
        Ok(()) => {
            info!(target: LOG_TARGET, "started GPU capture → {path}");
            *CURRENT_PATH.lock().expect("capture path mutex poisoned") = Some(path);
        }
        Err(err) => {
            let msg = err.localizedDescription();
            warn!(
                target: LOG_TARGET,
                "start_capture failed: {msg} (is MTL_CAPTURE_ENABLED=1 set in the launch env?)"
            );
        }
    }
}

/// Whether Metal's capture layer is loaded in this process.
///
/// Metal loads the layer at launch when `MTL_CAPTURE_ENABLED=1` is in the
/// environment (or the app bundle asks for it), and only then supports a
/// trace document as a destination, so this asks Metal rather than parsing
/// the variable. The layer sits under every Metal call whether or not a
/// capture runs: on `make bench`, with it the submit thread's pass replay
/// took 2.9 to 4.3 times as long in `wow_112_busy_frame`.
pub fn capture_layer_loaded() -> bool {
    // SAFETY: `sharedCaptureManager` is an always-live process-wide singleton.
    let manager = unsafe { MTLCaptureManager::sharedCaptureManager() };
    manager.supportsDestination(MTLCaptureDestination::GPUTraceDocument)
}

/// Log once per process whether the Metal capture layer and the Metal HUD are on.
///
/// Both change what a perf log measures, and neither is the layer's own
/// setting, so the log has to say which a run had. The capture state is
/// Metal's answer ([`capture_layer_loaded`]); the HUD's is the
/// `MTL_HUD_ENABLED` variable as the process got it, since a layer's own HUD
/// properties are set per layer and later. In a `PERF=1` build a loaded
/// capture layer also gets a warning, because every submit and GPU row of the
/// perf summary then includes its cost.
pub fn log_metal_tools() {
    let capture = capture_layer_loaded();
    let capture_env = std::env::var("MTL_CAPTURE_ENABLED");
    let hud_env = std::env::var("MTL_HUD_ENABLED");
    let shown = |value: &Result<String, std::env::VarError>| match value {
        Ok(value) => format!("{value:?}"),
        Err(std::env::VarError::NotPresent) => "unset".to_owned(),
        Err(std::env::VarError::NotUnicode(_)) => "not UTF-8".to_owned(),
    };
    mtld3d_shared::log_once_info!(
        target: LOG_TARGET,
        "metal tools: capture layer {} (MTL_CAPTURE_ENABLED={}), MTL_HUD_ENABLED={}",
        if capture { "loaded" } else { "not loaded" },
        shown(&capture_env),
        shown(&hud_env),
    );
    if cfg!(perf_tracking) && capture {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "metal tools: the perf summary's submit and GPU rows include the Metal capture \
             layer's cost; unset MTL_CAPTURE_ENABLED outside a capture session"
        );
    }
}

/// End the in-progress capture. No-op if none was started.
pub fn stop_capture() {
    // SAFETY: `sharedCaptureManager` is an always-live process-wide singleton.
    let manager = unsafe { MTLCaptureManager::sharedCaptureManager() };
    if !manager.isCapturing() {
        return;
    }
    manager.stopCapture();
    let path = CURRENT_PATH
        .lock()
        .expect("capture path mutex poisoned")
        .take();
    info!(
        target: LOG_TARGET,
        "stopped GPU capture → {}",
        path.as_deref().unwrap_or(FALLBACK_PATH)
    );
}

/// The path of the capture in progress, for the stop line.
static CURRENT_PATH: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);
