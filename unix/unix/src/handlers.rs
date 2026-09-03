use core::ffi::c_void;

use log::{debug, error, info, warn};
use mtld3d_shared::{
    AttachMetalLayerParams, BlitTextureToBufferParams, BufferCreateDesc,
    CompileShaderLibraryParams, CreateBackbufferParams, CreateBuffersBatchParams,
    CreateColorTargetParams, CreateCommandQueueParams, CreateDepthStencilStateParams,
    CreateDepthTextureParams, CreateRenderPipelineParams, CreateSamplerStateParams,
    CreateTextureSliceViewParams, CreateTexturesBatchParams, DestroyCommandQueueParams,
    DestroyResourcesBulkParams, EnsureBlitPipelineParams, EnsureClearQuadPipelineParams,
    GetDeviceInfoParams, GetTaskFaultsParams, InPtr, InPtrMut, MetalHandle, OpenLogParams,
    SetCursorOverlayParams, SetDisplaySyncEnabledParams, StartGpuCaptureParams, SubmitFrameParams,
    TextureCreateDesc, VertexAttrDesc, VertexBufferLayoutDesc, WaitForGpuRetireParams,
    WriteLogParams, identity,
    mtl::{DestroyKind, QuadPipelineKind},
    mtl_handle::{MTLBufferKind, MTLTextureKind},
};

use crate::{LOG_TARGET, metal, metal::handle::IntoRetained};

const STATUS_SUCCESS: i32 = 0;
// NTSTATUS bit-pattern reinterpret for `unix_call` return; see d3d9/lib.rs
// for the matching pattern on HRESULT.
const STATUS_UNSUCCESSFUL: i32 = 0xC000_0001_u32.cast_signed();

/// One-shot logger init.
///
/// d3d9.dll dispatches this as its first thunk after it has wired up its
/// own PE-side `env_logger`. `mtld3d_shared` owns the init policy; this
/// handler just forwards to it so all three cdylibs stay byte-identical.
pub extern "C" fn init_logger_handler(_args: *mut c_void) -> i32 {
    // The PE side can replay this first-thunk init (a second `Direct3DCreate9`
    // re-runs it), so the one-time process setup runs under a single `Once`
    // here rather than each callee carrying its own idempotency flag.
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        // Every line goes to the process's log file once `OpenLog` names
        // it; the file sink keeps the lines logged before that.
        mtld3d_shared::init_logger_to(Box::new(crate::log_file::FileSink));
        log_identity();
        // Latch the unix-side perf-tracking gate (`PERF_TRACKING_ENABLED`)
        // from `RUST_LOG`. Per-cdylib because each cdylib has its own
        // `log` statics; d3d9.dll latches its own copy in `init_logger`.
        metal::init_tracking_enabled();
        // Map the shared crash crumb (cfg-gated no-op in production) and
        // install the always-on signal handler.
        mtld3d_shared::crumb::init();
        mtld3d_shared::crumb::set_write_sink(crate::log_file::write_bytes);
        crate::crash::install();
        // Declare to macOS that we're a latency-critical game, not idle UI, so
        // it keeps the process out of App Nap / display throttling and the
        // compositor keeps cycling the layer even when the on-screen scene is
        // static.
        metal::declare_latency_critical_activity();
    });
    STATUS_SUCCESS
}

/// `WriteLog`: the sink behind the PE-side logger.
///
/// Writes the formatted line into this process's log file, next to the unix
/// side's own lines; the PE side has no usable standard handles of its own
/// when a launcher spawned the game.
pub extern "C" fn write_log_handler(args: *mut c_void) -> i32 {
    // SAFETY: unix-call handler params; PE side passes *mut WriteLogParams.
    let Some(params) = (unsafe { InPtrMut::<WriteLogParams>::opt(args) }) else {
        return -1;
    };
    if params.ptr == 0 || params.len == 0 {
        return STATUS_SUCCESS;
    }
    // SAFETY: PE supplied `ptr`/`len` as a byte slice valid for the call
    // duration; the pointer is non-zero per the check above.
    let bytes =
        unsafe { core::slice::from_raw_parts(params.ptr as *const u8, params.len as usize) };
    crate::log_file::write_all(bytes);
    STATUS_SUCCESS
}

/// `OpenLog`: where this process's log file and GPU traces go, once per process.
///
/// The lines logged since `InitLogger` wait in the file sink's backlog for
/// this, so the PE side sends it before it starts its own log thread. The
/// file itself appears with the first line written after this.
pub extern "C" fn open_log_handler(args: *mut c_void) -> i32 {
    // SAFETY: unix-call handler params; PE side passes *mut OpenLogParams.
    let Some(params) = (unsafe { InPtrMut::<OpenLogParams>::opt(args) }) else {
        return -1;
    };
    if params.dir_ptr == 0 || params.dir_len == 0 || params.stem_ptr == 0 {
        // The PE side found no usable location; its warn said why.
        crate::log_file::fall_back_to_stderr();
        return STATUS_SUCCESS;
    }
    // SAFETY: PE supplied `dir_ptr`/`dir_len` as a byte slice valid for the
    // call duration; the pointer is non-zero per the check above.
    let dir = unsafe {
        core::slice::from_raw_parts(params.dir_ptr as *const u8, params.dir_len as usize)
    };
    // SAFETY: same contract for `stem_ptr`/`stem_len`.
    let stem = unsafe {
        core::slice::from_raw_parts(params.stem_ptr as *const u8, params.stem_len as usize)
    };
    let (Ok(dir), Ok(stem)) = (core::str::from_utf8(dir), core::str::from_utf8(stem)) else {
        warn!(target: LOG_TARGET, "OpenLog: the location is not UTF-8, logging to stderr");
        crate::log_file::fall_back_to_stderr();
        return STATUS_UNSUCCESSFUL;
    };
    let path = crate::log_file::open(dir, stem);
    info!(target: LOG_TARGET, "log file: {}", path.display());
    STATUS_SUCCESS
}

/// Name this build in the log, as the first line the unix side emits.
///
/// [`identity::BUILD`] says which release the source came from; the image ID is
/// the Mach-O `LC_UUID` the linker assigned, which names this exact binary and
/// picks the `.dSYM` that symbolicates it out of the release's debug archive.
fn log_identity() {
    let id = identity::image_id();
    let id = id.as_deref().unwrap_or("no-image-id");
    let build = identity::BUILD;
    info!(target: LOG_TARGET, "mtld3d.so {build} {id} initialized");
}

pub extern "C" fn get_device_info_handler(args: *mut c_void) -> i32 {
    // SAFETY: unix-call handler params; PE side passes *mut GetDeviceInfoParams.
    let Some(mut params) = (unsafe { InPtrMut::<GetDeviceInfoParams>::opt(args) }) else {
        return -1;
    };

    if let Some((name, registry_id, caps)) = metal::default_device_info() {
        params.registry_id = registry_id;
        params.caps = caps;

        if params.name_ptr != 0 && params.name_buf_len > 0 {
            let buf_len =
                usize::try_from(params.name_buf_len).expect("name buf len fits host address space");
            let name_bytes = name.as_bytes();
            let copy_len = name_bytes.len().min(buf_len - 1);

            // SAFETY: PE side supplied `name_ptr`/`name_buf_len` as a writable
            // `u8` buffer it owns for the unix-call duration; `buf_len` fits its
            // allocation per the wire contract.
            let buf =
                unsafe { core::slice::from_raw_parts_mut(params.name_ptr as *mut u8, buf_len) };
            buf[..copy_len].copy_from_slice(&name_bytes[..copy_len]);
            buf[copy_len] = 0;
            params.name_len = u64::try_from(copy_len).expect("name copy len fits u64");
        }
    }

    STATUS_SUCCESS
}

pub extern "C" fn create_command_queue_handler(args: *mut c_void) -> i32 {
    // SAFETY: unix-call handler params; PE side passes *mut CreateCommandQueueParams.
    let Some(mut params) = (unsafe { InPtrMut::<CreateCommandQueueParams>::opt(args) }) else {
        return -1;
    };
    let params: &mut CreateCommandQueueParams = &mut params;

    if let Some(caps) = metal::create_command_queue() {
        params.device_handle = caps.device_handle;
        params.queue_handle = caps.queue_handle;
        params.unified_memory = u32::from(caps.unified_memory);
        params.min_linear_texture_align = caps.min_linear_texture_align;
        info!(
            target: LOG_TARGET,
            "created Metal device + command queue (unified_memory={}, min_linear_texture_align={})",
            caps.unified_memory, caps.min_linear_texture_align,
        );
        STATUS_SUCCESS
    } else {
        error!(target: LOG_TARGET, "failed to create Metal device/command queue");
        STATUS_UNSUCCESSFUL
    }
}

pub extern "C" fn attach_metal_layer_handler(args: *mut c_void) -> i32 {
    // SAFETY: unix-call handler params; PE side passes *mut AttachMetalLayerParams.
    let Some(mut params) = (unsafe { InPtrMut::<AttachMetalLayerParams>::opt(args) }) else {
        return -1;
    };

    let request = metal::LayerAttachRequest {
        hwnd: params.hwnd,
        width: params.width,
        height: params.height,
        pacing: metal::PresentPacing {
            vsync_requested: params.display_sync_enabled != 0,
            max_fps: params.max_fps,
        },
        hdr_enable: params.hdr_enable != 0,
        color_space: params.color_space,
        backing_scale_sink_ptr: params.backing_scale_ptr,
        software_cursor: params.software_cursor,
    };
    if let Some((view, layer, caps)) = metal::attach_metal_layer(params.device_handle, request) {
        params.view_handle = view;
        params.layer_handle = layer;
        params.backing_scale = caps.backing_scale;
        params.software_cursor_active = u32::from(caps.software_cursor_active);
        params.metalfx_available = u32::from(metal::upscale_is_supported(params.device_handle));
        info!(
            target: LOG_TARGET,
            "attached Metal layer {}x{} (vsync {}, maxFps {})",
            params.width,
            params.height,
            if params.display_sync_enabled != 0 { "on" } else { "off" },
            params.max_fps
        );
        STATUS_SUCCESS
    } else {
        params.view_handle = MetalHandle::NULL;
        params.layer_handle = MetalHandle::NULL;
        params.backing_scale = 1;
        params.software_cursor_active = 0;
        error!(
            target: LOG_TARGET,
            "failed to attach Metal layer (hwnd=0x{:x})",
            params.hwnd
        );
        STATUS_UNSUCCESSFUL
    }
}

/// `SetCursorOverlay`: the software cursor's wanted sprite and visibility.
///
/// Runs on the API thread, so it only validates, hands the sprite bytes over
/// to be copied when one came along, stores the wanted state and queues the
/// main-thread apply. Nothing here waits on `AppKit`.
pub extern "C" fn set_cursor_overlay_handler(args: *mut c_void) -> i32 {
    // SAFETY: unix-call handler params; PE side passes *mut SetCursorOverlayParams.
    let Some(params) = (unsafe { InPtr::<SetCursorOverlayParams>::opt(args) }) else {
        return -1;
    };
    if params.hash == 0 {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "SetCursorOverlay: hash 0 names no sprite → ignored"
        );
        return STATUS_UNSUCCESSFUL;
    }
    let pixels = if params.pixels_ptr == 0 {
        None
    } else {
        let expected = u64::from(params.width) * u64::from(params.height) * 4;
        if params.width == 0
            || params.height == 0
            || u64::from(params.pixels_len) != expected
            || !(1..=8).contains(&params.scale)
        {
            warn!(
                target: LOG_TARGET,
                "SetCursorOverlay: rejected sprite {}x{} scale={} len={} (expected {expected} bytes)",
                params.width, params.height, params.scale, params.pixels_len,
            );
            return STATUS_UNSUCCESSFUL;
        }
        // SAFETY: PE supplied `pixels_ptr`/`pixels_len` as a BGRA byte slice
        // valid for the call duration; the pointer is non-zero per the branch
        // and the length was just checked against the sprite's geometry.
        Some(unsafe {
            core::slice::from_raw_parts(params.pixels_ptr as *const u8, params.pixels_len as usize)
        })
    };
    metal::set_cursor_overlay(&params, pixels);
    STATUS_SUCCESS
}

pub extern "C" fn set_display_sync_enabled_handler(args: *mut c_void) -> i32 {
    // SAFETY: unix-call handler params; PE side passes *const SetDisplaySyncEnabledParams.
    let Some(params) = (unsafe { InPtr::<SetDisplaySyncEnabledParams>::opt(args.cast()) }) else {
        return -1;
    };
    metal::set_display_sync_enabled(
        params.layer_handle,
        &metal::PresentPacing {
            vsync_requested: params.display_sync_enabled != 0,
            max_fps: params.max_fps,
        },
    );
    STATUS_SUCCESS
}

pub extern "C" fn wait_for_gpu_retire_handler(args: *mut c_void) -> i32 {
    // SAFETY: unix-call handler params; PE side passes *const WaitForGpuRetireParams.
    let Some(params) = (unsafe { InPtr::<WaitForGpuRetireParams>::opt(args.cast()) }) else {
        return -1;
    };
    metal::wait_for_gpu_retire(
        params.target_seq,
        params.coherent_seq_ptr,
        params.failed_submit_seq_ptr,
    );
    STATUS_SUCCESS
}

pub extern "C" fn start_gpu_capture_handler(args: *mut c_void) -> i32 {
    // SAFETY: unix-call handler params; PE side passes *const StartGpuCaptureParams.
    let Some(params) = (unsafe { InPtr::<StartGpuCaptureParams>::opt(args.cast()) }) else {
        return -1;
    };
    metal::start_capture(params.device_handle);
    STATUS_SUCCESS
}

pub extern "C" fn stop_gpu_capture_handler(_args: *mut c_void) -> i32 {
    metal::stop_capture();
    STATUS_SUCCESS
}

/// Report the process-wide page-fault counters from `getrusage`.
///
/// One cheap libc call; the PE side gates it to once per perf summary
/// window. `ru_minflt` / `ru_majflt` are cumulative since process start,
/// so the caller deltas consecutive samples.
pub extern "C" fn get_task_faults_handler(args: *mut c_void) -> i32 {
    // SAFETY: unix-call handler params; PE side passes *mut GetTaskFaultsParams.
    let Some(mut params) = (unsafe { InPtrMut::<GetTaskFaultsParams>::opt(args) }) else {
        return -1;
    };
    // SAFETY: `rusage` is plain old data; all-zero is a valid initial value
    // for an out-parameter the kernel overwrites.
    let mut usage: libc::rusage = unsafe { core::mem::zeroed() };
    // SAFETY: `usage` is a live, writable rusage and RUSAGE_SELF is a valid
    // `who` selector.
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, &raw mut usage) };
    if rc != 0 {
        return STATUS_UNSUCCESSFUL;
    }
    // c_long on macOS is i64; a negative count never occurs, but saturate
    // to 0 rather than wrap if the kernel ever reports one.
    params.minor_faults = u64::try_from(usage.ru_minflt).unwrap_or(0);
    params.major_faults = u64::try_from(usage.ru_majflt).unwrap_or(0);
    STATUS_SUCCESS
}

pub extern "C" fn destroy_command_queue_handler(args: *mut c_void) -> i32 {
    // SAFETY: unix-call handler params; PE side passes *const DestroyCommandQueueParams.
    let Some(params) = (unsafe { InPtr::<DestroyCommandQueueParams>::opt(args.cast()) }) else {
        return -1;
    };
    metal::destroy_command_queue(
        params.device_handle,
        params.queue_handle,
        params.view_handle,
        params.backbuffer_handle,
        params.pipeline_handle,
        params.depth_texture_handle,
    );
    info!(target: LOG_TARGET, "destroyed Metal device + command queue");
    STATUS_SUCCESS
}

pub extern "C" fn create_backbuffer_handler(args: *mut c_void) -> i32 {
    // SAFETY: unix-call handler params; PE side passes *mut CreateBackbufferParams.
    let Some(mut params) = (unsafe { InPtrMut::<CreateBackbufferParams>::opt(args) }) else {
        return -1;
    };
    let params: &mut CreateBackbufferParams = &mut params;

    let Some((handle, srgb_handle)) = metal::create_backbuffer(
        params.device_handle,
        params.queue_handle,
        params.width,
        params.height,
    ) else {
        error!(target: LOG_TARGET, "failed to create backbuffer");
        return STATUS_UNSUCCESSFUL;
    };
    let msaa = metal::create_msaa_companion(
        params.device_handle,
        params.width,
        params.height,
        mtld3d_shared::mtl::PixelFormat::Bgra8Unorm,
        params.sample_count,
        "mtld3d-backbuffer-msaa",
    );
    if params.sample_count > 1 && msaa.is_none() {
        error!(
            target: LOG_TARGET,
            "failed to create {}x multisampled backbuffer companion", params.sample_count
        );
        return STATUS_UNSUCCESSFUL;
    }
    params.texture_handle = handle;
    // SAFETY: `create_backbuffer` transfers a retain into `srgb_handle`
    // (0 when the format has no sRGB twin).
    params.srgb_texture_handle = unsafe { MetalHandle::<MTLTextureKind>::new(srgb_handle) };
    let (msaa_handle, msaa_srgb_handle) = msaa.unwrap_or((MetalHandle::NULL, 0));
    params.msaa_texture_handle = msaa_handle;
    // SAFETY: `create_msaa_companion` transfers a retain into
    // `msaa_srgb_handle` (0 when the companion has no sRGB twin).
    params.msaa_srgb_texture_handle =
        unsafe { MetalHandle::<MTLTextureKind>::new(msaa_srgb_handle) };
    // debug, not info: it fires per-frame during a Reset-driven
    // window drag. The CreateDevice + AttachMetalLayer info
    // lines already cover the boot-time milestone.
    debug!(
        target: LOG_TARGET,
        "created backbuffer {}x{} samples={}",
        params.width, params.height, params.sample_count
    );
    STATUS_SUCCESS
}

pub extern "C" fn create_render_pipeline_handler(args: *mut c_void) -> i32 {
    // SAFETY: unix-call handler params; PE side passes *mut CreateRenderPipelineParams.
    let Some(mut params) = (unsafe { InPtrMut::<CreateRenderPipelineParams>::opt(args) }) else {
        return -1;
    };
    let params: &mut CreateRenderPipelineParams = &mut params;

    let attrs = if params.vertex_attr_count == 0 || params.vertex_attrs_ptr == 0 {
        &[][..]
    } else {
        // SAFETY: PE supplied `vertex_attrs_ptr` as the address of a
        // `[VertexAttrDesc; vertex_attr_count]` valid for the call duration.
        unsafe {
            core::slice::from_raw_parts(
                params.vertex_attrs_ptr as *const VertexAttrDesc,
                params.vertex_attr_count as usize,
            )
        }
    };
    let layouts = if params.vertex_layout_count == 0 || params.vertex_layouts_ptr == 0 {
        &[][..]
    } else {
        // SAFETY: PE supplied `vertex_layouts_ptr` as the address of a
        // `[VertexBufferLayoutDesc; vertex_layout_count]` valid for the call
        // duration.
        unsafe {
            core::slice::from_raw_parts(
                params.vertex_layouts_ptr as *const VertexBufferLayoutDesc,
                params.vertex_layout_count as usize,
            )
        }
    };

    if let Some(handle) = metal::create_render_pipeline(params, attrs, layouts) {
        params.pipeline_handle = handle;
        STATUS_SUCCESS
    } else {
        error!(target: LOG_TARGET, "failed to create render pipeline");
        STATUS_UNSUCCESSFUL
    }
}

pub extern "C" fn ensure_clear_quad_pipeline_handler(args: *mut c_void) -> i32 {
    // SAFETY: unix-call handler params; PE side passes *mut EnsureClearQuadPipelineParams.
    let Some(mut params) = (unsafe { InPtrMut::<EnsureClearQuadPipelineParams>::opt(args) }) else {
        return -1;
    };
    let params: &mut EnsureClearQuadPipelineParams = &mut params;
    if let Some(handle) = metal::ensure_clear_quad_pipeline(params) {
        params.pipeline_handle = handle;
        STATUS_SUCCESS
    } else {
        error!(target: LOG_TARGET, "failed to ensure clear-quad pipeline");
        STATUS_UNSUCCESSFUL
    }
}

pub extern "C" fn ensure_blit_pipeline_handler(args: *mut c_void) -> i32 {
    // SAFETY: unix-call handler params; PE side passes *mut EnsureBlitPipelineParams.
    let Some(mut params) = (unsafe { InPtrMut::<EnsureBlitPipelineParams>::opt(args) }) else {
        return -1;
    };
    let params: &mut EnsureBlitPipelineParams = &mut params;
    let resolved = match params.quad_kind {
        QuadPipelineKind::StretchBlit => metal::ensure_blit_pipeline(params),
        QuadPipelineKind::TextureUpload => metal::ensure_upload_pipeline(params),
    };
    if let Some(handle) = resolved {
        params.pipeline_handle = handle;
        STATUS_SUCCESS
    } else {
        error!(target: LOG_TARGET, "failed to ensure {:?} quad pipeline", params.quad_kind);
        STATUS_UNSUCCESSFUL
    }
}

pub extern "C" fn create_texture_slice_view_handler(args: *mut c_void) -> i32 {
    // SAFETY: unix-call handler params; PE side passes *mut CreateTextureSliceViewParams.
    let Some(mut params) = (unsafe { InPtrMut::<CreateTextureSliceViewParams>::opt(args) }) else {
        return -1;
    };
    let params: &mut CreateTextureSliceViewParams = &mut params;
    if let Some(handle) = metal::create_texture_slice_view(params) {
        params.view_handle = handle;
        STATUS_SUCCESS
    } else {
        error!(target: LOG_TARGET, "failed to create a 2D view of texture slice {}", params.slice);
        STATUS_UNSUCCESSFUL
    }
}

pub extern "C" fn compile_shader_library_handler(args: *mut c_void) -> i32 {
    // SAFETY: unix-call handler params; PE side passes *mut CompileShaderLibraryParams.
    let Some(mut params) = (unsafe { InPtrMut::<CompileShaderLibraryParams>::opt(args) }) else {
        return -1;
    };

    if params.msl_ptr == 0 || params.msl_len == 0 {
        warn!(target: LOG_TARGET, "CompileShaderLibrary: empty source");
        return STATUS_UNSUCCESSFUL;
    }
    if params.entry_ptr == 0 || params.entry_len == 0 {
        warn!(target: LOG_TARGET, "CompileShaderLibrary: empty entry name");
        return STATUS_UNSUCCESSFUL;
    }

    // SAFETY: PE supplied `msl_ptr`/`msl_len` as an MSL source slice valid for
    // the call duration; the pointer is non-zero per the length check above.
    let bytes = unsafe {
        core::slice::from_raw_parts(params.msl_ptr as *const u8, params.msl_len as usize)
    };
    let Ok(src) = core::str::from_utf8(bytes) else {
        warn!(target: LOG_TARGET, "CompileShaderLibrary: invalid UTF-8");
        return STATUS_UNSUCCESSFUL;
    };

    // SAFETY: PE supplied `entry_ptr`/`entry_len` as an entry-name slice valid
    // for the call duration; non-zero per the length check above.
    let entry_bytes = unsafe {
        core::slice::from_raw_parts(params.entry_ptr as *const u8, params.entry_len as usize)
    };
    let Ok(entry) = core::str::from_utf8(entry_bytes) else {
        warn!(target: LOG_TARGET, "CompileShaderLibrary: invalid UTF-8 in entry name");
        return STATUS_UNSUCCESSFUL;
    };

    match metal::compile_shader_library(params.device_handle, src, params.stage_tag, entry) {
        Some((lib, func)) => {
            params.library_handle = lib;
            params.fn_handle = func;
            STATUS_SUCCESS
        }
        None => STATUS_UNSUCCESSFUL,
    }
}

pub extern "C" fn submit_frame_handler(args: *mut c_void) -> i32 {
    // SAFETY: unix-call handler params; PE side passes *mut SubmitFrameParams.
    let Some(mut params) = (unsafe { InPtrMut::<SubmitFrameParams>::opt(args) }) else {
        return -1;
    };

    if metal::submit_frame(&mut params) {
        STATUS_SUCCESS
    } else {
        STATUS_UNSUCCESSFUL
    }
}

pub extern "C" fn blit_texture_to_buffer_handler(args: *mut c_void) -> i32 {
    // SAFETY: unix-call handler params; PE side passes *const BlitTextureToBufferParams.
    let Some(params) = (unsafe { InPtr::<BlitTextureToBufferParams>::opt(args.cast()) }) else {
        return -1;
    };
    let blit_args = metal::BlitArgs {
        queue_handle: params.queue_handle,
        device_handle: params.device_handle,
        tex_handle: params.tex_handle,
        dst_ptr: params.dst_ptr,
        dst_len: params.dst_len,
        mip_level: params.mip_level,
        slice: params.slice,
        origin_x: params.origin_x,
        origin_y: params.origin_y,
        width: params.width,
        height: params.height,
        bytes_per_row: params.bytes_per_row,
        source_width: params.source_width,
        source_height: params.source_height,
        block_height: params.block_height,
    };
    if metal::blit_texture_to_buffer(&blit_args) {
        STATUS_SUCCESS
    } else {
        STATUS_UNSUCCESSFUL
    }
}

pub extern "C" fn create_depth_texture_handler(args: *mut c_void) -> i32 {
    // SAFETY: unix-call handler params; PE side passes *mut CreateDepthTextureParams.
    let Some(mut params) = (unsafe { InPtrMut::<CreateDepthTextureParams>::opt(args) }) else {
        return -1;
    };
    let params: &mut CreateDepthTextureParams = &mut params;

    if let Some(handle) = metal::create_depth_texture(
        params.device_handle,
        params.width,
        params.height,
        params.pixel_format,
        params.sample_count,
    ) {
        params.texture_handle = handle;
        STATUS_SUCCESS
    } else {
        error!(target: LOG_TARGET, "failed to create depth texture");
        STATUS_UNSUCCESSFUL
    }
}

pub extern "C" fn create_color_target_handler(args: *mut c_void) -> i32 {
    // SAFETY: unix-call handler params; PE side passes *mut CreateColorTargetParams.
    let Some(mut params) = (unsafe { InPtrMut::<CreateColorTargetParams>::opt(args) }) else {
        return -1;
    };
    let params: &mut CreateColorTargetParams = &mut params;

    let Some((handle, srgb_handle)) = metal::create_color_target(
        params.device_handle,
        params.width,
        params.height,
        params.pixel_format,
    ) else {
        error!(target: LOG_TARGET, "failed to create color target texture");
        return STATUS_UNSUCCESSFUL;
    };
    let msaa = metal::create_msaa_companion(
        params.device_handle,
        params.width,
        params.height,
        params.pixel_format,
        params.sample_count,
        "mtld3d-color-target-msaa",
    );
    if params.sample_count > 1 && msaa.is_none() {
        error!(
            target: LOG_TARGET,
            "failed to create {}x multisampled color target companion", params.sample_count
        );
        return STATUS_UNSUCCESSFUL;
    }
    params.texture_handle = handle;
    // SAFETY: `create_color_target` transfers a retain into `srgb_handle`
    // (0 when the format has no sRGB twin).
    params.srgb_texture_handle = unsafe { MetalHandle::<MTLTextureKind>::new(srgb_handle) };
    let (msaa_handle, msaa_srgb_handle) = msaa.unwrap_or((MetalHandle::NULL, 0));
    params.msaa_texture_handle = msaa_handle;
    // SAFETY: `create_msaa_companion` transfers a retain into
    // `msaa_srgb_handle` (0 when the companion has no sRGB twin).
    params.msaa_srgb_texture_handle =
        unsafe { MetalHandle::<MTLTextureKind>::new(msaa_srgb_handle) };
    STATUS_SUCCESS
}

pub extern "C" fn create_depth_stencil_state_handler(args: *mut c_void) -> i32 {
    // SAFETY: unix-call handler params; PE side passes *mut CreateDepthStencilStateParams.
    let Some(mut params) = (unsafe { InPtrMut::<CreateDepthStencilStateParams>::opt(args) }) else {
        return -1;
    };
    let params: &mut CreateDepthStencilStateParams = &mut params;

    if let Some(handle) = metal::create_depth_stencil_state(params) {
        params.state_handle = handle;
        STATUS_SUCCESS
    } else {
        error!(target: LOG_TARGET, "failed to create depth stencil state");
        STATUS_UNSUCCESSFUL
    }
}

pub extern "C" fn create_textures_batch_handler(args: *mut c_void) -> i32 {
    // SAFETY: unix-call handler params; PE side passes *mut CreateTexturesBatchParams.
    let Some(params) = (unsafe { InPtrMut::<CreateTexturesBatchParams>::opt(args) }) else {
        return -1;
    };
    if params.count == 0 {
        return STATUS_SUCCESS;
    }
    let Some(device) = params.device_handle.into_retained() else {
        error!(
            target: LOG_TARGET,
            "create_textures_batch: device_handle={:#x} reject",
            params.device_handle
        );
        return STATUS_UNSUCCESSFUL;
    };
    // SAFETY: PE supplied `descs_ptr` as a `[TextureCreateDesc; count]` valid
    // for the call duration per the wire contract.
    let descs = unsafe {
        core::slice::from_raw_parts(
            params.descs_ptr as *const TextureCreateDesc,
            params.count as usize,
        )
    };
    // SAFETY: PE allocates a `[MetalHandle<MTLTextureKind>; count]` slice
    // and hands its raw pointer here; the layout is wire-compatible with
    // `u64` (`#[repr(transparent)]`).
    let handles = unsafe {
        core::slice::from_raw_parts_mut(
            params.handles_out_ptr as *mut MetalHandle<MTLTextureKind>,
            params.count as usize,
        )
    };
    // SAFETY: same wire contract as `handles_out_ptr` — a caller-owned
    // `[MetalHandle<MTLTextureKind>; count]` slice for the sRGB twin views.
    let srgb_handles = unsafe {
        core::slice::from_raw_parts_mut(
            params.srgb_handles_out_ptr as *mut MetalHandle<MTLTextureKind>,
            params.count as usize,
        )
    };
    let mut any_failed = false;
    for ((desc, slot), srgb_slot) in descs.iter().zip(handles.iter_mut()).zip(srgb_handles) {
        if let Some((handle, srgb_handle)) = metal::create_texture(&device, desc) {
            // SAFETY: `create_texture` returns the raw u64s of freshly
            // retained MTLTextures; adopt them as the canonical typed handles.
            *slot = unsafe { MetalHandle::<MTLTextureKind>::new(handle) };
            // SAFETY: as above; 0 (no twin) adopts as NULL.
            *srgb_slot = unsafe { MetalHandle::<MTLTextureKind>::new(srgb_handle) };
        } else {
            *slot = MetalHandle::NULL;
            *srgb_slot = MetalHandle::NULL;
            any_failed = true;
            error!(
                target: LOG_TARGET,
                "failed to create texture tex_id={:#x}",
                desc.tex_id
            );
        }
    }
    if any_failed {
        STATUS_UNSUCCESSFUL
    } else {
        STATUS_SUCCESS
    }
}

pub extern "C" fn create_sampler_state_handler(args: *mut c_void) -> i32 {
    // SAFETY: unix-call handler params; PE side passes *mut CreateSamplerStateParams.
    let Some(mut params) = (unsafe { InPtrMut::<CreateSamplerStateParams>::opt(args) }) else {
        return -1;
    };
    let params: &mut CreateSamplerStateParams = &mut params;

    if let Some(handle) = metal::create_sampler_state(params) {
        params.sampler_handle = handle;
        STATUS_SUCCESS
    } else {
        error!(target: LOG_TARGET, "failed to create sampler state");
        STATUS_UNSUCCESSFUL
    }
}

pub extern "C" fn create_buffers_batch_handler(args: *mut c_void) -> i32 {
    // SAFETY: unix-call handler params; PE side passes *mut CreateBuffersBatchParams.
    let Some(params) = (unsafe { InPtrMut::<CreateBuffersBatchParams>::opt(args) }) else {
        return -1;
    };
    if params.count == 0 {
        return STATUS_SUCCESS;
    }
    let Some(device) = params.device_handle.into_retained() else {
        error!(
            target: LOG_TARGET,
            "create_buffers_batch: device_handle={:#x} reject",
            params.device_handle
        );
        return STATUS_UNSUCCESSFUL;
    };
    // SAFETY: PE supplied `descs_ptr` as a `[BufferCreateDesc; count]` valid
    // for the call duration per the wire contract.
    let descs = unsafe {
        core::slice::from_raw_parts(
            params.descs_ptr as *const BufferCreateDesc,
            params.count as usize,
        )
    };
    // SAFETY: PE allocates a `[MetalHandle<MTLBufferKind>; count]` slice
    // and hands its raw pointer here; wire-compatible with `u64`.
    let handles = unsafe {
        core::slice::from_raw_parts_mut(
            params.handles_out_ptr as *mut MetalHandle<MTLBufferKind>,
            params.count as usize,
        )
    };
    let mut any_failed = false;
    // `metal::create_buffer` logs the precise reason (length=0, backing_ptr=0,
    // newBufferWithBytesNoCopy nil) before returning None.
    for (desc, slot) in descs.iter().zip(handles.iter_mut()) {
        if let Some(handle) = metal::create_buffer(&device, desc) {
            // SAFETY: `create_buffer` returns the raw u64 of a freshly
            // retained MTLBuffer; adopt it as canonical.
            *slot = unsafe { MetalHandle::<MTLBufferKind>::new(handle) };
        } else {
            *slot = MetalHandle::NULL;
            any_failed = true;
        }
    }
    if any_failed {
        STATUS_UNSUCCESSFUL
    } else {
        STATUS_SUCCESS
    }
}

pub extern "C" fn destroy_resources_bulk_handler(args: *mut c_void) -> i32 {
    // SAFETY: unix-call handler params; PE side passes *const DestroyResourcesBulkParams.
    let Some(params) = (unsafe { InPtr::<DestroyResourcesBulkParams>::opt(args.cast()) }) else {
        return -1;
    };
    if params.count == 0 {
        return STATUS_SUCCESS;
    }
    // SAFETY: PE supplied `handles_ptr` as a `[u64; count]` valid for the
    // call duration; the handles are read-only here.
    let slice = unsafe {
        core::slice::from_raw_parts(params.handles_ptr as *const u64, params.count as usize)
    };
    match params.kind {
        DestroyKind::Buffer => {
            for &h in slice {
                metal::destroy_buffer(h);
            }
        }
        DestroyKind::Texture => {
            for &h in slice {
                metal::destroy_texture(h);
            }
        }
        DestroyKind::RenderPipeline => {
            for &h in slice {
                metal::destroy_render_pipeline(h);
            }
        }
        DestroyKind::ShaderLibrary => {
            for &h in slice {
                metal::destroy_library(h);
            }
        }
        DestroyKind::ShaderFunction => {
            for &h in slice {
                metal::destroy_function(h);
            }
        }
        DestroyKind::SamplerState => {
            for &h in slice {
                metal::destroy_sampler_state(h);
            }
        }
        DestroyKind::DepthStencilState => {
            for &h in slice {
                metal::destroy_depth_stencil_state(h);
            }
        }
    }
    STATUS_SUCCESS
}
