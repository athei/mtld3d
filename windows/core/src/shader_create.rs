//! Synchronous shader validation and metadata extraction.
//!
//! The caller owns failure reporting and HRESULT translation. Successful programs are parsed
//! once and remain owned by the runtime that performed creation.

use mtld3d_shared::shader_create::{ShaderInputSemantic, ShaderStage, ShaderUsage};

use crate::{
    dxso::{Declaration, DxsoError, DxsoProgram, RegKind, ShaderType, parse},
    ids::ProgramId,
};

/// Validated program and the metadata needed by the API wrapper.
pub struct ParsedShader {
    pub program: DxsoProgram,
    pub id: ProgramId,
    pub max_const_used: u32,
    pub usage: ShaderUsage,
    pub color_out_mask: u8,
    pub input_semantics: Vec<ShaderInputSemantic>,
}

/// Creation failures translated to INVALIDCALL by the API boundary.
#[derive(Debug)]
pub enum ShaderCreateError {
    Parse(DxsoError),
    WrongStage,
    InvalidPixelInput,
    ConstantRegisterLimit,
}

/// Parse once, validate the requested stage, and extract synchronous API metadata.
///
/// # Panics
///
/// Panics if the parser emits a semantic usage index wider than its four-bit DXSO field.
///
/// # Errors
///
/// Returns the parser error or the first stage-specific validation failure. Pixel input
/// validation precedes constant-register validation, matching the API creation contract.
pub fn parse_shader(
    stage: &ShaderStage,
    tokens: &[u32],
) -> Result<ParsedShader, ShaderCreateError> {
    let program = parse(tokens).map_err(ShaderCreateError::Parse)?;
    let expected = match stage {
        ShaderStage::Vertex => ShaderType::Vertex,
        ShaderStage::Pixel => ShaderType::Pixel,
    };
    if program.shader_type != expected {
        return Err(ShaderCreateError::WrongStage);
    }
    if expected == ShaderType::Pixel && program.has_invalid_pixel_input_decl() {
        return Err(ShaderCreateError::InvalidPixelInput);
    }
    if program.violates_constant_register_limits() {
        return Err(ShaderCreateError::ConstantRegisterLimit);
    }
    let max_const_used = program.max_const_reg().map_or(0, |m| u32::from(m) + 1);
    let mut usage = ShaderUsage::empty();
    usage.set(ShaderUsage::INT_CONST, program.uses_dynamic_int_constants());
    usage.set(
        ShaderUsage::BOOL_CONST,
        program.uses_dynamic_bool_constants(),
    );
    let mut input_semantics = Vec::new();
    let color_out_mask = match stage {
        ShaderStage::Vertex => {
            usage.set(
                ShaderUsage::RELATIVE_CONST,
                program.uses_relative_const_addressing(),
            );
            for decl in &program.declarations {
                if let Declaration::Semantic {
                    usage,
                    usage_index,
                    reg,
                } = decl
                    && reg.kind == RegKind::Input
                {
                    input_semantics.push(ShaderInputSemantic {
                        usage: *usage,
                        usage_index: u8::try_from(*usage_index)
                            .expect("DXSO usage index is a four-bit field"),
                        register_index: reg.index,
                    });
                }
            }
            0
        }
        ShaderStage::Pixel => {
            usage.set(ShaderUsage::AUTOMATIC_FOG, program.major < 3);
            usage.set(ShaderUsage::BUMP_ENV, program.uses_bump_env());
            program.color_out_mask()
        }
    };
    Ok(ParsedShader {
        id: ProgramId::from_tokens(tokens),
        program,
        max_const_used,
        usage,
        color_out_mask,
        input_semantics,
    })
}

#[cfg(test)]
mod tests;
