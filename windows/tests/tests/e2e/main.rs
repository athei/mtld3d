//! The end-to-end suite, one test binary so one Wine process runs it all.
//!
//! Every module here drives the real `d3d9.dll` through the shared
//! [`mtld3d_tests::Harness`], and the tests of all of them run on the
//! threads of this one process, each with its own device. The two files
//! that stay outside are the ones that need a process of their own:
//! `unload.rs` frees the library, so nothing else may keep it mapped, and
//! `snmalloc_drift.rs` installs its own global allocator. `COVERAGE.md`
//! indexes the modules; this file only declares them.

mod buffers;
mod clip_planes;
mod d3dperf;
mod device;
mod draw;
mod expand16;
mod float_filter;
mod implicit_surface;
mod mrt;
mod msaa;
mod multi_device;
mod multithreaded;
mod non_uma;
mod points;
mod query;
mod render_scale;
mod render_states;
mod render_target;
mod resource_misc;
mod samplers;
mod shaders;
mod smoke;
mod state_block;
mod streams;
mod subresource_identity;
mod texture_stages;
mod textures;
mod transforms_ff;
mod vertex_decl;
