//! Captured draw inputs recorded by the PE API and consumed by the native encoder.

pub use mtld3d_core::draw_data::{
    DepthScissorFlags, DepthStencilFlags, DrawOp, IndexSource, PsSource, RenderStateSnapshot,
    ScratchSlice, StageBinding, StreamBinding, VertexSource, VsSource, arena_alloc_bytes,
    build_alpha_ref_bytes,
};
