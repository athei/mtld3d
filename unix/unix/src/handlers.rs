use core::ffi::c_void;
use std::sync::Arc;

use log::{debug, error, info, warn};
use mtld3d_shared::{
    AttachMetalLayerParams, BlitTextureToBufferParams, CreateBackbufferParams,
    CreateColorTargetParams, CreateCommandQueueParams, CreateDepthTextureParams,
    DestroyCommandQueueParams, DestroyResourcesBulkParams, DetachMetalLayerParams,
    GetDeviceInfoParams, InPtr, InPtrMut, MetalHandle, OpenLogParams, SetCursorOverlayParams,
    SetPresentWaitPolicyParams, WriteLogParams, identity,
    mtl::{CursorOverlayFlags, DestroyKind},
    mtl_handle::MTLTextureKind,
    record_handle::DeviceRecordHandle,
};
use objc2_core_foundation::kCFRunLoopCommonModes;

use crate::{LOG_TARGET, metal};

const STATUS_SUCCESS: i32 = 0;
// NTSTATUS bit-pattern reinterpret for `unix_call` return; see d3d9/lib.rs
// for the matching pattern on HRESULT.
const STATUS_UNSUCCESSFUL: i32 = 0xC000_0001_u32.cast_signed();

/// One-shot logger init.
///
/// d3d9.dll dispatches this as its first thunk after it has wired up its
/// own PE-side `env_logger`. `mtld3d_shared` owns the init policy; this
/// handler just forwards to it so all three cdylibs stay byte-identical.
pub extern "C" fn init_logger_handler(args: *mut c_void) -> i32 {
    // The PE side can replay this first-thunk init (a second `Direct3DCreate9`
    // re-runs it), so the one-time process setup runs under a single `Once`
    // here rather than each callee carrying its own idempotency flag.
    static INIT: std::sync::Once = std::sync::Once::new();
    // SAFETY: the dispatcher supplies this request's borrowed parameter record.
    let Some(params) = (unsafe { InPtr::<mtld3d_shared::InitLoggerParams>::opt(args) }) else {
        return STATUS_UNSUCCESSFUL;
    };
    let filter = if params.filter_len == 0 {
        None
    } else {
        if params.filter_ptr == 0
            || params
                .filter_ptr
                .checked_add(u64::from(params.filter_len))
                .is_none()
        {
            return STATUS_UNSUCCESSFUL;
        }
        // SAFETY: the PE caller retains the byte-aligned UTF-8 buffer through this call.
        let bytes = unsafe {
            std::slice::from_raw_parts(params.filter_ptr as *const u8, params.filter_len as usize)
        };
        let Ok(filter) = std::str::from_utf8(bytes) else {
            return STATUS_UNSUCCESSFUL;
        };
        Some(filter)
    };
    INIT.call_once(|| {
        // Every line goes to the process's log file once `OpenLog` names
        // it; the file sink keeps the lines logged before that.
        mtld3d_shared::init_logger_to_filter(Box::new(crate::log_file::FileSink), filter);
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
/// file itself appears with the first line written after this. The thunk
/// also carries `debug.mainThreadChecker`, acted on here because this is the
/// first thunk that runs with the configuration resolved.
pub extern "C" fn open_log_handler(args: *mut c_void) -> i32 {
    // SAFETY: unix-call handler params; PE side passes *mut OpenLogParams.
    let Some(params) = (unsafe { InPtrMut::<OpenLogParams>::opt(args) }) else {
        return -1;
    };
    // Ahead of the location checks: the checker gates AppKit use for the
    // rest of the process whether or not this process gets a log file.
    if params.main_thread_checker != 0 {
        crate::main_thread_checker::load();
    }
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
    let Some(image) = core_foundation_image() else {
        warn!(target: LOG_TARGET, "CoreFoundation image identity unavailable");
        return;
    };
    let base = image.base();
    let uuid = image.uuid().unwrap_or("unavailable");
    match image.path() {
        Some(path) => info!(target: LOG_TARGET,
            "CoreFoundation path=\"{}\" base={base:#x} uuid={uuid}",
            path.display().to_string().escape_debug()),
        None => warn!(target: LOG_TARGET,
            "CoreFoundation path=unavailable base={base:#x} uuid={uuid}"),
    }
    if image.uuid().is_none() {
        warn!(target: LOG_TARGET, "CoreFoundation image UUID unavailable");
    }
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
    } else {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "GetDeviceInfo: no Metal device; the adapter reports an empty name, a zero \
             registry id and no capability bits"
        );
    }

    STATUS_SUCCESS
}

/// The record a thunk names, or `None` after a warning.
///
/// A null or unknown handle is a device whose creation failed or one already
/// destroyed; the caller returns without touching Metal, as it did when the
/// device was looked up by queue address.
fn device_record(handle: DeviceRecordHandle, thunk: &str) -> Option<Arc<metal::DeviceRecord>> {
    let record = borrow_device_record(handle);
    if record.is_none() {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "{thunk}: no device record for handle {handle:#x}; the call is dropped",
        );
    }
    record
}

/// The record a handle names, with no verdict on a miss.
///
/// The PE lifecycle keeps the named record live through each boundary call.
fn borrow_device_record(handle: DeviceRecordHandle) -> Option<Arc<metal::DeviceRecord>> {
    // SAFETY: the PE side passes back a handle `CreateCommandQueue` produced
    // and keeps it until its `DestroyCommandQueue`, which is the one caller
    // that consumes it.
    unsafe { metal::DeviceRecord::borrow(handle) }
}

pub extern "C" fn create_command_queue_handler(args: *mut c_void) -> i32 {
    // SAFETY: unix-call handler params; PE side passes *mut CreateCommandQueueParams.
    let Some(mut params) = (unsafe { InPtrMut::<CreateCommandQueueParams>::opt(args) }) else {
        return -1;
    };
    let params: &mut CreateCommandQueueParams = &mut params;

    let gate = if params.gate_file_ptr == 0 || params.gate_file_len == 0 {
        None
    } else {
        // SAFETY: PE supplied `gate_file_ptr`/`gate_file_len` as a byte slice
        // valid for the call duration; the pointer is non-zero per the check.
        let bytes = unsafe {
            core::slice::from_raw_parts(
                params.gate_file_ptr as *const u8,
                params.gate_file_len as usize,
            )
        };
        core::str::from_utf8(bytes).map_or_else(
            |_| {
                warn!(target: LOG_TARGET, "CreateCommandQueue: the gate path is not UTF-8, no gate");
                None
            },
            |path| Some(std::path::PathBuf::from(path)),
        )
    };
    if let Some(caps) = metal::create_command_queue(gate) {
        params.device_handle = caps.device_handle;
        params.record_handle = caps.record_handle;
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
        cursor_kick_sink_ptr: params.cursor_kick_ptr,
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
            "attached Metal layer {}x{} on window {:#x} (vsync {}, maxFps {})",
            params.width,
            params.height,
            params.hwnd,
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

/// `DetachMetalLayer`: retire one view's attachment record, then the view.
///
/// The order is the one `DestroyCommandQueue` uses: the record goes out of
/// the registry before the view it names is retired, so the process-lifetime
/// observers never walk a view that is being released. Sent by a `Reset` that
/// retargets the device at another window, which attaches a view there
/// straight after; the retired view stays kept for a device that comes back
/// to its window.
pub extern "C" fn detach_metal_layer_handler(args: *mut c_void) -> i32 {
    // SAFETY: unix-call handler params; PE side passes *const DetachMetalLayerParams.
    let Some(params) = (unsafe { InPtr::<DetachMetalLayerParams>::opt(args.cast()) }) else {
        return -1;
    };
    let record = metal::detach_metal_layer(params.view_handle);
    metal::retire_metal_view(params.view_handle, record.as_deref());
    info!(
        target: LOG_TARGET,
        "detached Metal layer (view {:#x})",
        params.view_handle.raw(),
    );
    STATUS_SUCCESS
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
    let hardware = params.flags.contains(CursorOverlayFlags::HARDWARE);
    if params.hash == 0 && !hardware {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "SetCursorOverlay: hash 0 names no sprite → ignored"
        );
        return STATUS_UNSUCCESSFUL;
    }
    let pixels = if params.pixels_ptr == 0 {
        None
    } else {
        let expected = u64::from(params.width)
            .checked_mul(u64::from(params.height))
            .and_then(|pixels| pixels.checked_mul(4));
        if params.width == 0
            || params.height == 0
            || Some(u64::from(params.pixels_len)) != expected
            || !(1..=8).contains(&params.scale)
        {
            warn!(
                target: LOG_TARGET,
                "SetCursorOverlay: rejected sprite {}x{} scale={} len={} (expected {expected:?} bytes)",
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
    if metal::set_cursor_overlay(&params, pixels) {
        STATUS_SUCCESS
    } else {
        STATUS_UNSUCCESSFUL
    }
}

pub extern "C" fn set_present_wait_policy_handler(args: *mut c_void) -> i32 {
    // SAFETY: unix-call handler params; PE side passes *const SetPresentWaitPolicyParams.
    let Some(params) = (unsafe { InPtr::<SetPresentWaitPolicyParams>::opt(args.cast()) }) else {
        return -1;
    };
    let Some(record) = device_record(params.record_handle, "SetPresentWaitPolicy") else {
        // The policy is a hint for the presents of one device, so a handle
        // that names no device has no present to apply it to and the caller
        // has nothing to do differently.
        return STATUS_SUCCESS;
    };
    metal::set_wait_policy(record.present(), params.policy);
    STATUS_SUCCESS
}

/// Sample process-wide faults once per enabled encoder summary window.
pub fn task_faults() -> mtld3d_core::perf::TaskFaults {
    // SAFETY: rusage is plain data initialized before the kernel fills it.
    let mut usage: libc::rusage = unsafe { core::mem::zeroed() };
    // SAFETY: usage is a live writable output and RUSAGE_SELF is valid.
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, &raw mut usage) };
    mtld3d_core::perf::TaskFaults {
        minor: if rc == 0 {
            u64::try_from(usage.ru_minflt).unwrap_or(0)
        } else {
            0
        },
        major: if rc == 0 {
            u64::try_from(usage.ru_majflt).unwrap_or(0)
        } else {
            0
        },
    }
}

pub extern "C" fn destroy_command_queue_handler(args: *mut c_void) -> i32 {
    // SAFETY: unix-call handler params; PE side passes *const DestroyCommandQueueParams.
    let Some(params) = (unsafe { InPtr::<DestroyCommandQueueParams>::opt(args.cast()) }) else {
        return -1;
    };
    // SAFETY: the handle is the one `CreateCommandQueue` produced for this
    // device, and this is the only thunk that gives it back; no later thunk
    // may name it.
    let Some(record) = (unsafe { metal::DeviceRecord::consume(params.record_handle) }) else {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "DestroyCommandQueue: no device record for handle {:#x}; nothing is released",
            params.record_handle,
        );
        return STATUS_SUCCESS;
    };
    metal::destroy_command_queue(
        params.device_handle,
        &record,
        params.view_handle,
        params.backbuffer_handle,
        params.depth_texture_handle,
    );
    info!(target: LOG_TARGET, "destroyed Metal device + command queue");
    STATUS_SUCCESS
}

pub extern "C" fn create_backbuffer_handler(args: *mut c_void) -> i32 {
    // SAFETY: unix-call handler params; PE side passes *mut CreateBackbufferParams.
    let Some(mut params) = (unsafe { InPtrMut::<CreateBackbufferParams>::opt(args) }) else {
        error!(target: LOG_TARGET, "CreateBackbuffer: null params pointer");
        return -1;
    };
    let params: &mut CreateBackbufferParams = &mut params;
    let Some(record) = device_record(params.record_handle, "CreateBackbuffer") else {
        return STATUS_UNSUCCESSFUL;
    };

    let Some((handle, srgb_handle)) =
        metal::create_backbuffer(params.device_handle, params.width, params.height)
    else {
        error!(
            target: LOG_TARGET,
            "failed to create {}x{} backbuffer (samples={}) device={:#x} queue={:#x}",
            params.width,
            params.height,
            params.sample_count,
            params.device_handle.raw(),
            record.queue().raw(),
        );
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
            "failed to create the {}x multisampled companion of the {}x{} backbuffer; the \
             single-sample texture is released again; device={:#x} queue={:#x}",
            params.sample_count,
            params.width,
            params.height,
            params.device_handle.raw(),
            record.queue().raw(),
        );
        // Both handles are minted and neither has been handed back, so this
        // side owns their only copies; the view goes first, since it holds a
        // retain on its base.
        metal::destroy_texture(srgb_handle);
        metal::destroy_texture(handle.raw());
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
    metal::clear_new_color_textures(
        record.queue(),
        &[params.texture_handle, params.msaa_texture_handle],
        metal::OPAQUE_BLACK,
    );
    // debug, not info: it fires per-frame during a Reset-driven
    // window drag. The CreateDevice + AttachMetalLayer info
    // lines already cover the boot-time milestone.
    debug!(
        target: "mtld3d::unix::command",
        "created backbuffer {}x{} samples={}: texture {:#x} srgb {:#x} msaa {:#x} \
         msaa_srgb {:#x} device={:#x} queue={:#x}",
        params.width,
        params.height,
        params.sample_count,
        params.texture_handle.raw(),
        params.srgb_texture_handle.raw(),
        params.msaa_texture_handle.raw(),
        params.msaa_srgb_texture_handle.raw(),
        params.device_handle.raw(),
        record.queue().raw(),
    );
    STATUS_SUCCESS
}

pub extern "C" fn blit_texture_to_buffer_handler(args: *mut c_void) -> i32 {
    // SAFETY: unix-call handler params; PE side passes *const BlitTextureToBufferParams.
    let Some(params) = (unsafe { InPtr::<BlitTextureToBufferParams>::opt(args.cast()) }) else {
        return -1;
    };
    let Some(record) = device_record(params.record_handle, "BlitTextureToBuffer") else {
        return STATUS_UNSUCCESSFUL;
    };
    let blit_args = metal::BlitArgs {
        planes: params.planes,
        stencil_bytes_per_row: params.stencil_bytes_per_row,
        stencil_offset: params.stencil_offset,
        record: &record,
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
        debug!(
            target: LOG_TARGET,
            "created depth texture {}x{} samples={}: {:#x}",
            params.width,
            params.height,
            params.sample_count,
            handle.raw(),
        );
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
    let Some(record) = device_record(params.record_handle, "CreateColorTarget") else {
        return STATUS_UNSUCCESSFUL;
    };

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
            "failed to create the {}x multisampled companion of the {}x{} {:?} color target; \
             the single-sample texture is released again",
            params.sample_count, params.width, params.height, params.pixel_format
        );
        // Same ownership as the back buffer's companion failure above: both
        // minted handles are still this side's only copies.
        metal::destroy_texture(srgb_handle);
        metal::destroy_texture(handle.raw());
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
    metal::clear_new_color_textures(
        record.queue(),
        &[params.texture_handle, params.msaa_texture_handle],
        metal::TRANSPARENT_BLACK,
    );
    STATUS_SUCCESS
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
    destroy_resources_bulk(params.kind, slice);
    STATUS_SUCCESS
}

/// Destroy native resources without rebuilding a boundary request.
pub fn destroy_resources_bulk(kind: DestroyKind, slice: &[u64]) {
    if slice.is_empty() {
        return;
    }
    // The handles by value, so a later fault or ledger warning on one of
    // them can be matched to the destroy that carried it.
    debug!(
        target: LOG_TARGET,
        "DestroyResourcesBulk {:?} x{}: {slice:#x?}",
        kind,
        slice.len(),
    );
    match kind {
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
        DestroyKind::ComputePipeline => {
            for &h in slice {
                metal::depth_transfer::destroy_pipeline(h);
            }
        }
        DestroyKind::DepthStencilState => {
            for &h in slice {
                metal::destroy_depth_stencil_state(h);
            }
        }
    }
}

/// Resolve the public export's storage, without following a CF object pointer.
fn core_foundation_image() -> Option<identity::LoadedImage> {
    // The address of the exported variable belongs to CoreFoundation. Reading
    // its value instead would follow a CF object, and a Rust wrapper function
    // would identify the image the wrapper was linked into.
    let symbol = (&raw const kCFRunLoopCommonModes).cast();
    // SAFETY: CoreFoundation is a linked dependency and stays mapped throughout
    // this call, including the immutable Mach-O header and load commands.
    unsafe { identity::LoadedImage::for_symbol(symbol) }
}

#[cfg(test)]
mod tests;
