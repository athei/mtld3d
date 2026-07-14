use core::ffi::c_void;

use log::{debug, error, info, warn};
use mtld3d_shared::{
    AttachMetalLayerParams, BlitTextureToBufferParams, BufferCreateDesc,
    CompileShaderLibraryParams, CreateBackbufferParams, CreateBuffersBatchParams,
    CreateColorTargetParams, CreateCommandQueueParams, CreateDepthStencilStateParams,
    CreateDepthTextureParams, CreateRenderPipelineParams, CreateSamplerStateParams,
    CreateTexturesBatchParams, DestroyCommandQueueParams, DestroyResourcesBulkParams,
    EnsureBlitPipelineParams, EnsureClearQuadPipelineParams, GetDeviceInfoParams,
    GetPrimaryDisplayModeParams, InPtr, InPtrMut, MetalHandle, SetDisplaySyncEnabledParams,
    SetLayerDrawableSizeParams, StartGpuCaptureParams, SubmitFrameParams, TextureCreateDesc,
    VertexAttrDesc, WaitForGpuRetireParams,
    mtl::DestroyKind,
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
        mtld3d_shared::init_logger();
        // Latch the unix-side perf-tracking gate (`PERF_TRACKING_ENABLED`)
        // from `RUST_LOG`. Per-cdylib because each cdylib has its own
        // `log` statics; d3d9.dll latches its own copy in `init_logger`.
        metal::init_tracking_enabled();
        // Map the shared crash crumb (cfg-gated no-op in production) and
        // install the always-on signal handler.
        mtld3d_shared::crumb::init();
        crate::crash::install();
        // Declare to macOS that we're a latency-critical game, not idle UI, so
        // it keeps the process out of App Nap / display throttling and the
        // compositor keeps cycling the layer even when the on-screen scene is
        // static.
        metal::declare_latency_critical_activity();
    });
    STATUS_SUCCESS
}

pub extern "C" fn get_device_info_handler(args: *mut c_void) -> i32 {
    // SAFETY: unix-call handler params; PE side passes *mut GetDeviceInfoParams.
    let Some(mut params) = (unsafe { InPtrMut::<GetDeviceInfoParams>::opt(args) }) else {
        return -1;
    };

    if let Some((name, registry_id)) = metal::default_device_info() {
        params.registry_id = registry_id;

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

    let pacing = metal::PresentPacing {
        vsync_requested: params.display_sync_enabled != 0,
        max_fps: params.max_fps,
    };
    let hdr_enable = params.hdr_enable != 0;
    if let Some((view, layer, caps)) = metal::attach_metal_layer(
        params.device_handle,
        params.hwnd,
        params.width,
        params.height,
        pacing,
        hdr_enable,
        params.color_space,
    ) {
        params.view_handle = view;
        params.layer_handle = layer;
        params.backing_scale = caps.backing_scale;
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
        error!(
            target: LOG_TARGET,
            "failed to attach Metal layer (hwnd=0x{:x})",
            params.hwnd
        );
        STATUS_UNSUCCESSFUL
    }
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

pub extern "C" fn set_layer_drawable_size_handler(args: *mut c_void) -> i32 {
    // SAFETY: unix-call handler params; PE side passes *const SetLayerDrawableSizeParams.
    let Some(params) = (unsafe { InPtr::<SetLayerDrawableSizeParams>::opt(args.cast()) }) else {
        return -1;
    };
    metal::set_layer_drawable_size(params.layer_handle, params.width, params.height);
    STATUS_SUCCESS
}

pub extern "C" fn wait_for_gpu_retire_handler(args: *mut c_void) -> i32 {
    // SAFETY: unix-call handler params; PE side passes *const WaitForGpuRetireParams.
    let Some(params) = (unsafe { InPtr::<WaitForGpuRetireParams>::opt(args.cast()) }) else {
        return -1;
    };
    metal::wait_for_gpu_retire(params.target_seq, params.coherent_seq_ptr);
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

pub extern "C" fn get_primary_display_mode_handler(args: *mut c_void) -> i32 {
    // SAFETY: unix-call handler params; PE side passes *mut GetPrimaryDisplayModeParams.
    let Some(mut params) = (unsafe { InPtrMut::<GetPrimaryDisplayModeParams>::opt(args) }) else {
        return -1;
    };
    let (w, h, hz) = metal::get_primary_display_mode();
    params.width = w;
    params.height = h;
    params.refresh_hz = hz;
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

    if let Some(handle) =
        metal::create_backbuffer(params.device_handle, params.width, params.height)
    {
        params.texture_handle = handle;
        // debug, not info — fires per-frame during a Reset-driven
        // window drag. The CreateDevice + AttachMetalLayer info
        // lines already cover the boot-time milestone.
        debug!(
            target: LOG_TARGET,
            "created backbuffer {}x{}",
            params.width, params.height
        );
        STATUS_SUCCESS
    } else {
        error!(target: LOG_TARGET, "failed to create backbuffer");
        STATUS_UNSUCCESSFUL
    }
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

    if let Some(handle) = metal::create_render_pipeline(params, attrs) {
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
    if let Some(handle) = metal::ensure_blit_pipeline(params) {
        params.pipeline_handle = handle;
        STATUS_SUCCESS
    } else {
        error!(target: LOG_TARGET, "failed to ensure blit pipeline");
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
        origin_x: params.origin_x,
        origin_y: params.origin_y,
        width: params.width,
        height: params.height,
        bytes_per_row: params.bytes_per_row,
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

    if let Some(handle) = metal::create_color_target(
        params.device_handle,
        params.width,
        params.height,
        params.pixel_format,
    ) {
        params.texture_handle = handle;
        STATUS_SUCCESS
    } else {
        error!(target: LOG_TARGET, "failed to create color target texture");
        STATUS_UNSUCCESSFUL
    }
}

pub extern "C" fn create_depth_stencil_state_handler(args: *mut c_void) -> i32 {
    // SAFETY: unix-call handler params; PE side passes *mut CreateDepthStencilStateParams.
    let Some(mut params) = (unsafe { InPtrMut::<CreateDepthStencilStateParams>::opt(args) }) else {
        return -1;
    };
    let params: &mut CreateDepthStencilStateParams = &mut params;

    if let Some(handle) = metal::create_depth_stencil_state(
        params.device_handle,
        params.depth_test_enable,
        params.depth_write_enable,
        params.depth_compare_func,
        params.id,
    ) {
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
    let mut any_failed = false;
    for (desc, slot) in descs.iter().zip(handles.iter_mut()) {
        if let Some(handle) = metal::create_texture(&device, desc) {
            // SAFETY: `create_texture` returns the raw u64 of a freshly
            // retained MTLTexture; adopt it as the canonical typed handle.
            *slot = unsafe { MetalHandle::<MTLTextureKind>::new(handle) };
        } else {
            *slot = MetalHandle::NULL;
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
