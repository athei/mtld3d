//! The end-to-end suite, one test binary so one Wine process runs it all.
//!
//! Every module here drives the real `d3d9.dll` through the shared
//! [`mtld3d_tests::Harness`], and the tests of all of them run on the
//! threads of this one process, each with its own device. The five files
//! that stay outside are the ones that need a process of their own:
//! `exit_code.rs` ends the process it runs in, `unload.rs` frees the
//! library, so nothing else may keep it mapped, `unload_after_device.rs`
//! frees it after a device has pinned it and ends the process it runs in,
//! `snmalloc_drift.rs` installs its own global allocator, and
//! `thread_exit.rs` reads the process-wide pool of allocators, which other
//! tests' threads would stir.
//! `COVERAGE.md` indexes the modules; this file only declares them.

mod a2b10g10r10;
mod a2r10g10b10;
mod async_compile;
mod bench;
mod bench_api_cost;
mod bench_buffers;
mod bench_clock;
mod bench_cold_start;
mod bench_crossing;
mod bench_frame_shape;
mod bench_query;
mod bench_shader_stutter;
mod bench_streaming;
mod bench_wow112;
mod buffers;
mod clip_planes;
mod d3d9ex;
mod d3d9ex_reset;
mod d3d9ex_resources;
mod d3dperf;
mod device;
mod draw;
mod dxt_volume;
mod dynamic_depth;
mod expand16;
mod failure_exits;
mod ff_vertex_pipeline;
mod float_filter;
mod format_agreement;
mod format_query;
mod implicit_surface;
mod lock_lifetime;
mod mrt;
mod msaa;
mod multi_device;
mod multithreaded;
mod no_color_passes;
mod non_uma;
mod one_off_passes;
mod packed10;
mod points;
mod present_split;
mod query;
mod range_fog;
mod render_scale;
mod render_states;
mod render_target;
mod resource_misc;
mod same_size_reset;
mod samplers;
mod sampling_views;
mod shaders;
mod smoke;
mod state_block;
mod streams;
mod subresource_identity;
mod table_fog;
mod texture_stages;
mod textures;
mod transforms_ff;
mod unix_encoder;
mod vertex_decl;
mod wide_stretch;
mod window_lifecycle;
mod wireframe;
