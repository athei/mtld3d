mod blit;
mod buffer;
mod capture;
mod clear_quad;
mod command;
mod device;
pub mod handle;
mod macdrv;
mod null_texture;
mod pipeline;
mod present;
mod sampler;
mod shader;
mod texture;
mod upload_quad;
mod upscale;

pub use blit::ensure_blit_pipeline;
pub use buffer::{create_buffer, destroy_buffer};
pub use capture::{start_capture, stop_capture};
pub use clear_quad::ensure_clear_quad_pipeline;
pub use command::{BlitArgs, blit_texture_to_buffer, submit_frame, wait_for_gpu_retire};
pub use device::{create_command_queue, default_device_info, destroy_command_queue};
pub use macdrv::{
    PresentPacing, attach_metal_layer, declare_latency_critical_activity, set_display_sync_enabled,
};
pub use mtld3d_shared::perf::init_tracking_enabled;
pub use pipeline::{create_render_pipeline, destroy_render_pipeline};
pub use sampler::{create_sampler_state, destroy_sampler_state};
pub use shader::{compile_shader_library, destroy_function, destroy_library};
pub use texture::{
    create_backbuffer, create_color_target, create_depth_stencil_state, create_depth_texture,
    create_msaa_companion, create_texture, destroy_depth_stencil_state, destroy_texture,
};
pub use upload_quad::ensure_upload_pipeline;
pub use upscale::is_supported as upscale_is_supported;
