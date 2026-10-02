//! Fixed-layout shader creation inputs and API metadata.
//!
//! Token and semantic buffers remain owned by the caller for the synchronous call.
//! Parsed programs stay in the native device until an ordered frame adopts them.

use bitflags::bitflags;

use crate::{Thunk, Thunks};

/// Shader stage requested by the API entry point.
#[repr(u32)]
#[derive(Debug, PartialEq, Eq)]
pub enum ShaderStage {
    Vertex = 0,
    Pixel = 1,
}

/// D3D declaration usage encoded in shader input semantics.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeclUsage {
    Position,
    BlendWeight,
    BlendIndices,
    Normal,
    PSize,
    Texcoord,
    Tangent,
    Binormal,
    TessFactor,
    PositionT,
    Color,
    Fog,
    Depth,
    Sample,
}

/// A shader input declaration in its original declaration order.
#[repr(C)]
pub struct ShaderInputSemantic {
    pub usage: DeclUsage,
    pub usage_index: u8,
    pub register_index: u16,
}

bitflags! {
    /// Shader properties needed synchronously by API state capture.
    #[repr(transparent)]
    pub struct ShaderUsage: u8 {
        const RELATIVE_CONST = 1 << 0;
        const INT_CONST = 1 << 1;
        const BOOL_CONST = 1 << 2;
        const BUMP_ENV = 1 << 3;
        const AUTOMATIC_FOG = 1 << 4;
        /// A `ps_3_0` reads an input semantic outside the fixed-function varyings.
        ///
        /// Such a draw's pixel variant records which of those semantics the
        /// bound vertex shader outputs (`VariantKey::linked_input_mask`).
        const LINKED_INPUTS = 1 << 5;
    }
}

/// Synchronous creation request and bounded API metadata output.
///
/// The caller initializes outputs to failure and retains both pointed-to buffers until return.
/// Only matching PE and native builds may exchange this typed wire record. The stage must be
/// a valid variant before forming a reference to this record. A native registration is published
/// only after validation and after verifying the semantic output capacity.
#[repr(C, align(8))]
pub struct CreateShaderProgramParams {
    pub runtime: u64,
    pub tokens_ptr: u64,
    pub semantics_ptr: u64,
    pub stage: ShaderStage,
    pub token_count: u32,
    pub semantic_capacity: u32,
    pub result: i32,
    pub program_id: u64,
    pub registration: u64,
    pub max_const_used: u32,
    pub semantic_count: u32,
    pub usage: ShaderUsage,
    pub color_out_mask: u8,
    pub padding: [u8; 6],
}

/// Cancel one registration that was not transferred to an admitted frame.
#[repr(C, align(8))]
pub struct CancelShaderProgramParams {
    pub runtime: u64,
    pub registration: u64,
}

impl Thunk for CreateShaderProgramParams {
    const CODE: u32 = Thunks::CreateShaderProgram as u32;
}

impl Thunk for CancelShaderProgramParams {
    const CODE: u32 = Thunks::CancelShaderProgram as u32;
}

const _: () = {
    assert!(size_of::<CancelShaderProgramParams>() == 16);
    assert!(align_of::<CancelShaderProgramParams>() == 8);
    assert!(size_of::<ShaderInputSemantic>() == 4);
    assert!(align_of::<ShaderInputSemantic>() == 2);
    assert!(size_of::<CreateShaderProgramParams>() == 72);
    assert!(align_of::<CreateShaderProgramParams>() == 8);
    assert!(core::mem::offset_of!(CreateShaderProgramParams, program_id) == 40);
    assert!(core::mem::offset_of!(CreateShaderProgramParams, usage) == 64);
};
