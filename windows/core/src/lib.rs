//! Pure-Rust d3d9 helpers shared by `d3d9.dll`.
//!
//! Host-testable: no COM, no `raw-dylib`, no `winecrt0`. Consumed by
//! `windows/d3d9` as an rlib.

/// `log` target for every mtld3d-core call site *except* `dxso::*` and `perf`.
///
/// Shares the COM layer's `"mtld3d::d3d9"` target — these modules were
/// carved out of `d3d9.dll` and from a user's perspective they're still
/// the d3d9 layer. The `dxso` submodule logs to `"mtld3d::dxso"`, `perf`
/// logs to `"mtld3d::perf"`.
const LOG_TARGET: &str = "mtld3d::d3d9";

pub mod api_lock;
pub mod app_profile;
pub mod buffer_backing;
pub mod buffer_rename;
pub mod caps;
pub mod config;
pub mod convert;
pub mod depth_stencil_state;
pub mod dirty_range;
pub mod dirty_rect;
pub mod display_mode;
pub mod dxso;
pub mod ff_state;
pub mod format;
pub mod format_probe;
pub mod fullscreen_resize;
pub mod gpu_caps;
pub mod ids;
pub mod level_authority;
pub mod multisample;
pub mod page_box;
pub mod page_box_pool;
pub mod passes;
pub mod perf;
pub mod pipeline_state;
pub mod pixel_convert;
pub mod pool;
pub mod present;
pub mod process_vertices;
pub mod readback;
pub mod render_scale;
pub mod sampler_state;
pub mod scratch;
pub mod shader_cache;
pub mod shader_compile_stats;
pub mod staging_coverage;
pub mod state_trace;
pub mod storage_policy;
pub mod streams;
pub mod stretch_rect;
pub mod surface_lock;
pub mod texture_flags;
pub mod texture_staging;
pub mod upload_pass;
pub mod upload_recovery;
pub mod visibility;
pub mod vs_draw;
