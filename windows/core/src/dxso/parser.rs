//! DXSO bytecode parser.
//!
//! Accepts Shader Model 1, 2, and 3 bytecode; versions outside that range
//! surface as a hard `UnsupportedShaderModel` error instead of silently
//! producing wrong IR.

use mtld3d_types::{
    D3DDECLUSAGE_BINORMAL, D3DDECLUSAGE_BLENDINDICES, D3DDECLUSAGE_BLENDWEIGHT, D3DDECLUSAGE_COLOR,
    D3DDECLUSAGE_DEPTH, D3DDECLUSAGE_FOG, D3DDECLUSAGE_NORMAL, D3DDECLUSAGE_POSITION,
    D3DDECLUSAGE_POSITIONT, D3DDECLUSAGE_PSIZE, D3DDECLUSAGE_SAMPLE, D3DDECLUSAGE_TANGENT,
    D3DDECLUSAGE_TESSFACTOR, D3DDECLUSAGE_TEXCOORD,
};

use super::{
    ir::{
        CmpFunc, DeclUsage, Declaration, DefConstant, DefIntConstant, DstMods, DstOperand,
        DxsoError, DxsoProgram, InstrFlags, Instruction, RegKind, Register, RelativeAddr,
        ShaderType, SrcModifier, SrcOperand, Swizzle, TextureType, WriteMask,
    },
    opcode::Opcode,
};

/// Parse DXSO bytecode into a [`DxsoProgram`] IR.
///
/// # Errors
///
/// Returns [`DxsoError::Truncated`] if the token stream ends mid-instruction;
/// [`DxsoError::InvalidHeader`] if the version word isn't a recognised VS/PS
/// token; [`DxsoError::UnsupportedShaderModel`] for SM versions outside the
/// SM1/SM2/SM3 range the IR models.
///
/// # Panics
///
/// Panics only if the `subroutines` entry for the label block currently being
/// parsed is missing. That is unreachable by construction: the entry is
/// inserted when the `label` instruction that opens the block is parsed.
pub fn parse(bytecode: &[u32]) -> Result<DxsoProgram, DxsoError> {
    let mut pos = 0;
    let first = *bytecode.get(pos).ok_or(DxsoError::Truncated)?;
    pos += 1;

    let shader_type = match (first >> 16) & 0xFFFF {
        0xFFFE => ShaderType::Vertex,
        0xFFFF => ShaderType::Pixel,
        _ => return Err(DxsoError::InvalidHeader),
    };
    let major = ((first >> 8) & 0xFF) as u8;
    let minor = (first & 0xFF) as u8;
    // SM1.x (vs_1_1 / ps_1_0..1_4), SM2.x, and SM3.0 are the models the IR +
    // emitter cover. Vertex shaders begin at 1.1 (there is no vs_1_0); pixel
    // shaders span 1.0..1.4. Reject anything else so an unmodelled stream
    // fails loudly rather than emitting garbage.
    let known = match (shader_type, major) {
        (_, 2 | 3) => true,
        (ShaderType::Vertex, 1) => minor == 1,
        (ShaderType::Pixel, 1) => minor <= 4,
        _ => false,
    };
    if !known {
        return Err(DxsoError::UnsupportedShaderModel { major, minor });
    }

    let mut declarations = Vec::new();
    let mut def_constants = Vec::new();
    let mut def_int_constants = Vec::new();
    let mut instructions = Vec::new();
    let mut subroutines: std::collections::BTreeMap<u32, Vec<Instruction>> =
        std::collections::BTreeMap::new();
    // None = currently parsing main; Some(label) = inside a
    // `label sN` / `ret` block, pushing into `subroutines[label]`.
    let mut current_label: Option<u32> = None;

    loop {
        let token = *bytecode.get(pos).ok_or(DxsoError::Truncated)?;
        pos += 1;

        let opcode_bits = (token & 0xFFFF) as u16;
        let opcode = Opcode::from_u16(opcode_bits).ok_or(DxsoError::UnknownOpcode(opcode_bits))?;

        match opcode {
            Opcode::End => break,
            Opcode::Comment => {
                let len = ((token >> 16) & 0x7FFF) as usize;
                pos = pos.checked_add(len).ok_or(DxsoError::Truncated)?;
                if pos > bytecode.len() {
                    return Err(DxsoError::Truncated);
                }
            }
            Opcode::Dcl => {
                let usage_token = *bytecode.get(pos).ok_or(DxsoError::Truncated)?;
                pos += 1;
                let dst_token = *bytecode.get(pos).ok_or(DxsoError::Truncated)?;
                pos += 1;
                let kind = decode_reg_type(dst_token)?;
                let reg = Register {
                    kind,
                    index: (dst_token & 0x7FF) as u16,
                };
                let decl = if kind == RegKind::Sampler {
                    Declaration::Sampler {
                        texture_type: decode_texture_type((usage_token >> 27) & 0xF),
                        reg,
                    }
                } else {
                    Declaration::Semantic {
                        usage: decode_decl_usage((usage_token & 0x1F) as u8)?,
                        usage_index: (usage_token >> 16) & 0xF,
                        reg,
                    }
                };
                declarations.push(decl);
            }
            Opcode::Def => {
                let dst_token = *bytecode.get(pos).ok_or(DxsoError::Truncated)?;
                pos += 1;
                let reg = Register {
                    kind: decode_reg_type(dst_token)?,
                    index: (dst_token & 0x7FF) as u16,
                };
                let mut value = [0.0f32; 4];
                for v in &mut value {
                    let bits = *bytecode.get(pos).ok_or(DxsoError::Truncated)?;
                    pos += 1;
                    *v = f32::from_bits(bits);
                }
                def_constants.push(DefConstant { reg, value });
            }
            Opcode::DefI => {
                let dst_token = *bytecode.get(pos).ok_or(DxsoError::Truncated)?;
                pos += 1;
                let reg = Register {
                    kind: decode_reg_type(dst_token)?,
                    index: (dst_token & 0x7FF) as u16,
                };
                let mut value = [0i32; 4];
                for v in &mut value {
                    let bits = *bytecode.get(pos).ok_or(DxsoError::Truncated)?;
                    pos += 1;
                    *v = bits.cast_signed();
                }
                def_int_constants.push(DefIntConstant { reg, value });
            }
            Opcode::DefB => {
                // Bool def constant: dst + 1 bool.
                pos = pos.checked_add(2).ok_or(DxsoError::Truncated)?;
                if pos > bytecode.len() {
                    return Err(DxsoError::Truncated);
                }
            }
            // ps_1_4 `phase` separates the two texture-addressing passes. It
            // carries no operands and only marks a boundary; our emitter lowers
            // instructions to sequential MSL statements with mutable per-texture
            // registers that carry across the boundary, so the boundary itself
            // is a no-op. Drop it (emit no instruction).
            Opcode::Phase => {}
            Opcode::Label => {
                // `label sN` — sN is a Label register (type 18). Index
                // is the subroutine id. Subsequent instructions until
                // the next `Ret` are this subroutine's body.
                let token_count = ((token >> 24) & 0xF) as usize;
                if token_count != 1 {
                    return Err(DxsoError::Truncated);
                }
                // `label` is SM3-only, so SM1 relative semantics never apply.
                let (label_src, consumed) = decode_src(bytecode, pos, false)?;
                pos += consumed;
                let label_id = u32::from(label_src.reg.index);
                subroutines.entry(label_id).or_default();
                current_label = Some(label_id);
            }
            Opcode::Ret => {
                // Closes the current subroutine block; a fall-through `ret`
                // past the end of main resumes main (some compilers emit one
                // before the first label).
                current_label = None;
            }
            _ => {
                // SM2.0 introduced the instruction-length field in bits 24-27
                // of the opcode token. SM1.x (vs_1_1 / ps_1_0..1_4) leaves
                // those bits zero, so for SM1 the operand count must instead be
                // derived from the parameter-token continuation flag: every
                // dst/src register token — including a relative-address
                // extension token — sets bit 31, while the next opcode token
                // and the END token clear it. Count that run for SM1; read the
                // length field directly for SM2+. (Without this, SM1
                // instructions parse with zero operands and the emitter indexes
                // an empty source list.)
                let token_count = if major >= 2 {
                    ((token >> 24) & 0xF) as usize
                } else {
                    let mut n = 0;
                    while bytecode.get(pos + n).is_some_and(|t| t & 0x8000_0000 != 0) {
                        n += 1;
                    }
                    n
                };
                let predicated = (token & (1 << 28)) != 0;
                // D3D9 packs the comparison-function selector for
                // ifc / breakc / setp into bits 16-23 of the
                // instruction token. Non-comparison opcodes leave
                // those bits zero (or carry unrelated control
                // bits the generic arm doesn't read).
                let cmp_func = match opcode {
                    Opcode::Ifc | Opcode::BreakC | Opcode::SetP => {
                        let raw = ((token >> 16) & 0xFF) as u8;
                        CmpFunc::from_raw(raw)
                    }
                    _ => None,
                };
                // SM2+ `texld` packs its control modifier in the same bits:
                // `D3DSI_TEXLD_PROJECT` (bit 16) selects `texldp` (divide the
                // coordinate by `.w` before sampling). Other opcodes leave it 0.
                let tex_projected = matches!(opcode, Opcode::TexLd) && (token & 0x0001_0000) != 0;
                // `D3DSI_COISSUE` (bit 30) marks a ps_1_x co-issued instruction.
                let coissue = (token & 0x4000_0000) != 0;
                let mut remaining = token_count;

                let dst = if opcode.has_destination() && remaining >= 1 {
                    let (d, consumed) = decode_dst(bytecode, pos)?;
                    pos += consumed;
                    remaining = remaining
                        .checked_sub(consumed)
                        .ok_or(DxsoError::Truncated)?;
                    Some(d)
                } else {
                    None
                };

                // SM2.x / SM3 predicated execution: bit 28 of the
                // instruction token signals an extra predicate source
                // operand sandwiched between dst and the regular
                // sources. Decode and split off so `srcs` only
                // contains the actual operands the instruction
                // consumes.
                let predicate = if predicated && remaining > 0 {
                    // Predication is SM2.x+; SM1 relative semantics don't apply.
                    let (p, consumed) = decode_src(bytecode, pos, false)?;
                    pos += consumed;
                    remaining = remaining
                        .checked_sub(consumed)
                        .ok_or(DxsoError::Truncated)?;
                    Some(p)
                } else {
                    None
                };

                let mut srcs = Vec::new();
                while remaining > 0 {
                    let (s, consumed) = decode_src(bytecode, pos, major < 2)?;
                    srcs.push(s);
                    pos += consumed;
                    remaining = remaining
                        .checked_sub(consumed)
                        .ok_or(DxsoError::Truncated)?;
                }

                let mut flags = InstrFlags::empty();
                flags.set(InstrFlags::PREDICATED, predicated);
                flags.set(InstrFlags::TEX_PROJECTED, tex_projected);
                flags.set(InstrFlags::COISSUE, coissue);
                let inst = Instruction {
                    opcode,
                    dst,
                    srcs,
                    flags,
                    predicate,
                    cmp_func,
                };
                match current_label {
                    Some(label) => subroutines
                        .get_mut(&label)
                        .expect("subroutine entry was created on Label")
                        .push(inst),
                    None => instructions.push(inst),
                }
            }
        }
    }

    Ok(DxsoProgram {
        shader_type,
        major,
        minor,
        declarations,
        def_constants,
        def_int_constants,
        instructions,
        subroutines,
    })
}

const fn decode_reg_type(token: u32) -> Result<RegKind, DxsoError> {
    let low = (token >> 28) & 0x7;
    let high = (token >> 11) & 0x3;
    let full = low | (high << 3);
    Ok(match full {
        0 => RegKind::Temp,
        1 => RegKind::Input,
        2 => RegKind::Const,
        3 => RegKind::Addr,
        4 => RegKind::RastOut,
        5 => RegKind::AttrOut,
        6 => RegKind::TexcoordOut,
        7 => RegKind::ConstInt,
        8 => RegKind::ColorOut,
        9 => RegKind::DepthOut,
        10 => RegKind::Sampler,
        11 => RegKind::Output,
        14 => RegKind::ConstBool,
        15 => RegKind::Loop,
        16 => RegKind::TempFloat16,
        17 => RegKind::MiscType,
        18 => RegKind::Label,
        19 => RegKind::Predicate,
        _ => return Err(DxsoError::UnknownRegisterType(full)),
    })
}

fn decode_dst(bc: &[u32], pos: usize) -> Result<(DstOperand, usize), DxsoError> {
    let tok = *bc.get(pos).ok_or(DxsoError::Truncated)?;
    if tok & (1 << 13) != 0 {
        return Err(DxsoError::UnsupportedRelativeAddressing);
    }
    let kind = decode_reg_type(tok)?;
    let write_mask = ((tok >> 16) & 0xF) as u8;
    let result_mod = (tok >> 20) & 0xF;
    let shift_raw = ((tok >> 24) & 0xF) as i8;
    let shift_scale = if shift_raw & 0x8 != 0 {
        shift_raw - 16
    } else {
        shift_raw
    };
    let mut mods = DstMods::empty();
    mods.set(DstMods::SATURATE, result_mod & 1 != 0);
    mods.set(DstMods::PARTIAL_PRECISION, result_mod & 2 != 0);
    mods.set(DstMods::CENTROID, result_mod & 4 != 0);
    Ok((
        DstOperand {
            reg: Register {
                kind,
                index: (tok & 0x7FF) as u16,
            },
            write_mask: WriteMask(write_mask),
            mods,
            shift_scale,
        },
        1,
    ))
}

fn decode_src(bc: &[u32], pos: usize, sm1: bool) -> Result<(SrcOperand, usize), DxsoError> {
    let tok = *bc.get(pos).ok_or(DxsoError::Truncated)?;
    let kind = decode_reg_type(tok)?;
    let swizzle = decode_swizzle_bits((tok >> 16) & 0xFF);
    let modifier = decode_src_modifier(((tok >> 24) & 0xF) as u8);
    let (rel_addr, consumed) = if tok & (1 << 13) != 0 {
        if sm1 {
            // SM1 relative addressing (`c[a0.x + N]`) sets the relative bit but
            // carries NO separate relative-address token — the index register
            // is implicitly `a0.x` (the only address register before vs_2_0).
            // SM2+ added the explicit token, decoded in the `else` below.
            let rel = RelativeAddr {
                reg: Register {
                    kind: RegKind::Addr,
                    index: 0,
                },
                swizzle: Swizzle([0; 4]),
            };
            (Some(rel), 1)
        } else {
            let rel_tok = *bc.get(pos + 1).ok_or(DxsoError::Truncated)?;
            let rel = RelativeAddr {
                reg: Register {
                    kind: decode_reg_type(rel_tok)?,
                    index: (rel_tok & 0x7FF) as u16,
                },
                swizzle: decode_swizzle_bits((rel_tok >> 16) & 0xFF),
            };
            (Some(rel), 2)
        }
    } else {
        (None, 1)
    };
    Ok((
        SrcOperand {
            reg: Register {
                kind,
                index: (tok & 0x7FF) as u16,
            },
            swizzle,
            modifier,
            rel_addr,
        },
        consumed,
    ))
}

const fn decode_swizzle_bits(bits: u32) -> Swizzle {
    Swizzle([
        (bits & 0x3) as u8,
        ((bits >> 2) & 0x3) as u8,
        ((bits >> 4) & 0x3) as u8,
        ((bits >> 6) & 0x3) as u8,
    ])
}

const fn decode_src_modifier(b: u8) -> SrcModifier {
    match b {
        1 => SrcModifier::Neg,
        2 => SrcModifier::Bias,
        3 => SrcModifier::BiasNeg,
        4 => SrcModifier::Sign,
        5 => SrcModifier::SignNeg,
        6 => SrcModifier::Comp,
        7 => SrcModifier::X2,
        8 => SrcModifier::X2Neg,
        9 => SrcModifier::Dz,
        10 => SrcModifier::Dw,
        11 => SrcModifier::Abs,
        12 => SrcModifier::AbsNeg,
        13 => SrcModifier::Not,
        _ => SrcModifier::None,
    }
}

const fn decode_decl_usage(b: u8) -> Result<DeclUsage, DxsoError> {
    Ok(match b {
        D3DDECLUSAGE_POSITION => DeclUsage::Position,
        D3DDECLUSAGE_BLENDWEIGHT => DeclUsage::BlendWeight,
        D3DDECLUSAGE_BLENDINDICES => DeclUsage::BlendIndices,
        D3DDECLUSAGE_NORMAL => DeclUsage::Normal,
        D3DDECLUSAGE_PSIZE => DeclUsage::PSize,
        D3DDECLUSAGE_TEXCOORD => DeclUsage::Texcoord,
        D3DDECLUSAGE_TANGENT => DeclUsage::Tangent,
        D3DDECLUSAGE_BINORMAL => DeclUsage::Binormal,
        D3DDECLUSAGE_TESSFACTOR => DeclUsage::TessFactor,
        D3DDECLUSAGE_POSITIONT => DeclUsage::PositionT,
        D3DDECLUSAGE_COLOR => DeclUsage::Color,
        D3DDECLUSAGE_FOG => DeclUsage::Fog,
        D3DDECLUSAGE_DEPTH => DeclUsage::Depth,
        D3DDECLUSAGE_SAMPLE => DeclUsage::Sample,
        _ => return Err(DxsoError::UnknownDeclUsage(b)),
    })
}

const fn decode_texture_type(bits: u32) -> TextureType {
    match bits {
        2 => TextureType::Texture2D,
        3 => TextureType::TextureCube,
        4 => TextureType::Texture3D,
        _ => TextureType::Unknown,
    }
}
