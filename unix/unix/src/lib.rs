use core::ffi::c_void;

use mtld3d_shared::Thunks;
use strum::{EnumCount, VariantArray};

mod crash;
mod handlers;
mod metal;

/// `log` target used by every call inside this crate.
///
/// d3d9.dll fires a one-shot `InitLogger` thunk on load — see
/// `handlers::init_logger_handler` — which registers `env_logger`; all
/// other handlers log through that backend. A handler that runs before
/// that thunk would silently no-op, which is why the PE side dispatches
/// `InitLogger` from its own `DllMain` before any other call.
const LOG_TARGET: &str = "mtld3d::unix";

#[unsafe(no_mangle)]
pub static __wine_unix_call_funcs: [UnixCallFn; Thunks::COUNT] = DISPATCH_TABLE;

#[unsafe(no_mangle)]
pub static __wine_unix_call_wow64_funcs: [UnixCallFn; Thunks::COUNT] = DISPATCH_TABLE;

type UnixCallFn = unsafe extern "C" fn(*mut c_void) -> i32;

const DISPATCH_TABLE: [UnixCallFn; Thunks::COUNT] = build_dispatch_table();

/// Wrap a handler in an `@autoreleasepool` so every dispatch call drains on return.
///
/// The pool catches any autoreleased Apple objects (most visibly
/// `MTLCommandBuffer`). Wine's unix-call dispatcher does not set up a
/// pool, so without this wrap autoreleased objects live until thread exit
/// and pin every resource they encoded — `bytesNoCopy` pages stay wired
/// and `newBufferWithBytesNoCopy:` eventually returns nil. Each macro
/// invocation defines a uniquely-scoped `extern "C"` wrapper so wrapping
/// every handler is two new tokens at the call site.
macro_rules! arp {
    ($inner:path) => {{
        extern "C" fn arp_wrap(args: *mut c_void) -> i32 {
            mtld3d_shared::crumb!(stringify!($inner), args as usize as u64);
            objc2::rc::autoreleasepool(|_| $inner(args))
        }
        arp_wrap as UnixCallFn
    }};
}

const fn dispatch(code: Thunks) -> UnixCallFn {
    match code {
        Thunks::InitLogger => arp!(handlers::init_logger_handler),
        Thunks::GetDeviceInfo => arp!(handlers::get_device_info_handler),
        Thunks::CreateCommandQueue => arp!(handlers::create_command_queue_handler),
        Thunks::AttachMetalLayer => arp!(handlers::attach_metal_layer_handler),
        Thunks::DestroyCommandQueue => arp!(handlers::destroy_command_queue_handler),
        Thunks::CreateBackbuffer => arp!(handlers::create_backbuffer_handler),
        Thunks::CreateRenderPipeline => arp!(handlers::create_render_pipeline_handler),
        Thunks::SubmitFrame => arp!(handlers::submit_frame_handler),
        Thunks::CreateDepthTexture => arp!(handlers::create_depth_texture_handler),
        Thunks::CreateColorTarget => arp!(handlers::create_color_target_handler),
        Thunks::CreateDepthStencilState => arp!(handlers::create_depth_stencil_state_handler),
        Thunks::CreateTexturesBatch => arp!(handlers::create_textures_batch_handler),
        Thunks::CreateSamplerState => arp!(handlers::create_sampler_state_handler),
        Thunks::CompileShaderLibrary => arp!(handlers::compile_shader_library_handler),
        Thunks::CreateBuffersBatch => arp!(handlers::create_buffers_batch_handler),
        Thunks::BlitTextureToBuffer => arp!(handlers::blit_texture_to_buffer_handler),
        Thunks::SetDisplaySyncEnabled => arp!(handlers::set_display_sync_enabled_handler),
        Thunks::DestroyResourcesBulk => arp!(handlers::destroy_resources_bulk_handler),
        Thunks::SetLayerDrawableSize => arp!(handlers::set_layer_drawable_size_handler),
        Thunks::WaitForGpuRetire => arp!(handlers::wait_for_gpu_retire_handler),
        Thunks::StartGpuCapture => arp!(handlers::start_gpu_capture_handler),
        Thunks::StopGpuCapture => arp!(handlers::stop_gpu_capture_handler),
        Thunks::GetPrimaryDisplayMode => arp!(handlers::get_primary_display_mode_handler),
        Thunks::EnsureClearQuadPipeline => arp!(handlers::ensure_clear_quad_pipeline_handler),
        Thunks::EnsureBlitPipeline => arp!(handlers::ensure_blit_pipeline_handler),
    }
}

const fn build_dispatch_table() -> [UnixCallFn; Thunks::COUNT] {
    extern "C" fn unimplemented_thunk(_args: *mut c_void) -> i32 {
        unimplemented!("Called unimplemented thunk.")
    }

    let mut table = [unimplemented_thunk as UnixCallFn; Thunks::COUNT];
    let variants = Thunks::VARIANTS;
    let mut i = 0;
    while i < variants.len() {
        table[variants[i] as usize] = dispatch(variants[i]);
        i += 1;
    }
    table
}
