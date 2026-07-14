//! DXSO (D3D9 shader bytecode) parsing and → MSL translation.
//!
//! Platform-independent pure Rust. The `parser` submodule turns a `&[u32]` of
//! DXSO bytecode into a structured `DxsoProgram`; the MSL emitters
//! (`emit_vs_programmable` / `emit_ps_programmable` for DXSO and `emit_vs_ff`
//! / `emit_ps_ff` for fixed-function) each produce a standalone translation
//! unit containing a single `mtld3d_vs` or `mtld3d_ps` entry point. A render
//! pipeline can mix any VS source with any PS source.

mod emit;
mod ff;
mod ir;
mod opcode;
mod parser;

/// `log` target used by every call inside this module.
///
/// Consumed via the `log` crate macros (`warn!(target: LOG_TARGET, …)`) and
/// by `log_once_warn!`. No logger is registered here — mtld3d-core is an rlib
/// statically linked into `d3d9.dll`, which owns the `env_logger` init; tests
/// without a registered logger simply no-op.
pub const LOG_TARGET: &str = "mtld3d::dxso";

pub use emit::{
    DEFAULT_PS_ENTRY, DEFAULT_VS_ENTRY, EmitError, VariantFlags, VariantKey, emit_ps_programmable,
    emit_ps_programmable_named, emit_vs_programmable, emit_vs_programmable_named,
};
pub use ff::{
    FfPsKey, FfStage, FfVsFlags, FfVsKey, emit_ps_ff, emit_ps_ff_named, emit_vs_ff,
    emit_vs_ff_named, ff_attr_index_for_semantic,
};
pub use ir::{CmpFunc, DeclUsage, Declaration, DxsoError, DxsoProgram, RegKind, ShaderType};
pub use parser::parse;

#[cfg(test)]
mod emit_tests;
#[cfg(test)]
mod ff_tests;
#[cfg(test)]
mod parser_tests;
