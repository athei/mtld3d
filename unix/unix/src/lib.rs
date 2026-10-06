use core::ffi::c_void;

use mtld3d_shared::Thunks;
use strum::{EnumCount, VariantArray};

mod crash;
mod draw;
mod encoder;
mod encoder_service;
mod handlers;
mod hud_state;
mod log_file;
mod main_thread_checker;
mod metal;
mod shader_prewarm;
mod shader_programs;
mod stack_page;

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

/// Every native Rust allocation goes through snmalloc, the allocator the PE side uses.
///
/// Process-wide resource: a dylib has one global allocator. The native encoder,
/// submit and worker threads allocate and free per packet, and the default
/// macOS allocator cost the encoder about 5.5 us per packet more than snmalloc
/// in a matched streaming measurement. Only Rust's allocation calls change:
/// snmalloc exports `sn_rust_*` entry points, not `malloc` or `free`, so memory
/// Objective-C or C allocate is still freed by them, and no buffer this crate
/// hands to Metal is freed by Metal (`bytesNoCopy` wrappers carry no
/// deallocator).
#[global_allocator]
static ALLOCATOR: snmalloc_rs::SnMalloc = snmalloc_rs::SnMalloc;

/// Wrap a handler in an `@autoreleasepool` so every dispatch call drains on return.
///
/// The pool catches any autoreleased Apple objects (most visibly
/// `MTLCommandBuffer`). Wine's unix-call dispatcher does not set up a
/// pool, so without this wrap autoreleased objects live until thread exit
/// and pin every resource they encoded — `bytesNoCopy` pages stay wired
/// and `newBufferWithBytesNoCopy:` eventually returns nil. Each macro
/// invocation defines a uniquely-scoped `extern "C"` wrapper so wrapping
/// every handler is two new tokens at the call site. The thunks of
/// `macdrv::run_on_main_thread_sync` and `macdrv::run_on_main_thread_async`
/// are its main-thread twins: every closure the layer dispatches there runs
/// in a pool of its own for the same reason.
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
        Thunks::CreateShaderProgram => arp!(shader_programs::create_handler),
        Thunks::CancelShaderProgram => arp!(shader_programs::cancel_handler),
        Thunks::SubmitEncoderFrame => arp!(encoder_service::submit_handler),
        Thunks::EncoderControl => arp!(encoder_service::control_handler),
        Thunks::CreateEncoder => arp!(encoder_service::create_handler),
        Thunks::DestroyEncoder => arp!(encoder_service::destroy_handler),
        Thunks::InitLogger => arp!(handlers::init_logger_handler),
        Thunks::GetDeviceInfo => arp!(handlers::get_device_info_handler),
        Thunks::CreateCommandQueue => arp!(handlers::create_command_queue_handler),
        Thunks::AttachMetalLayer => arp!(handlers::attach_metal_layer_handler),
        Thunks::DestroyCommandQueue => arp!(handlers::destroy_command_queue_handler),
        Thunks::CreateBackbuffer => arp!(handlers::create_backbuffer_handler),
        Thunks::CreateDepthTexture => arp!(handlers::create_depth_texture_handler),
        Thunks::CreateColorTarget => arp!(handlers::create_color_target_handler),
        Thunks::BlitTextureToBuffer => arp!(handlers::blit_texture_to_buffer_handler),
        Thunks::DestroyResourcesBulk => arp!(handlers::destroy_resources_bulk_handler),
        Thunks::WriteLog => arp!(handlers::write_log_handler),
        Thunks::OpenLog => arp!(handlers::open_log_handler),
        Thunks::SetCursorOverlay => arp!(handlers::set_cursor_overlay_handler),
        Thunks::DetachMetalLayer => arp!(handlers::detach_metal_layer_handler),
        Thunks::SetPresentWaitPolicy => arp!(handlers::set_present_wait_policy_handler),
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
