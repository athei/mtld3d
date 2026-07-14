//! DXSO intermediate representation, produced by `parser::parse` and consumed by the MSL emitter.
//!
//! Covers the subset of Shader Model 2.0 we actually need.

use super::opcode::Opcode;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShaderType {
    Vertex,
    Pixel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum RegKind {
    Temp,
    Input,
    Const,
    Addr,
    RastOut,
    AttrOut,
    TexcoordOut,
    ConstInt,
    ColorOut,
    DepthOut,
    Sampler,
    /// SM3 unified VS output register (`o0..o11`).
    ///
    /// Semantic comes from a matching `dcl_<usage> oN` declaration — unlike
    /// SM2 where the register kind itself encodes the slot
    /// (RastOut/AttrOut/TexcoordOut).
    Output,
    ConstBool,
    Loop,
    TempFloat16,
    MiscType,
    Label,
    Predicate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Register {
    pub kind: RegKind,
    pub index: u16,
}

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextureType {
    Unknown,
    Texture2D,
    TextureCube,
    Texture3D,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Declaration {
    /// A vertex/fragment input or output with a usage semantic (Position, Color, Texcoord, …).
    Semantic {
        usage: DeclUsage,
        usage_index: u32,
        reg: Register,
    },
    /// A PS sampler declaration carrying the expected texture dimensionality.
    Sampler {
        texture_type: TextureType,
        reg: Register,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DefConstant {
    pub reg: Register,
    pub value: [f32; 4],
}

/// Integer constant declared via `defi iN, x, y, z, w`.
///
/// Used to hold loop-counter limits (`loop aL, iN` reads `iN.x` for
/// iteration count, `iN.y` for initial `aL`, `iN.z` for step) and `rep`
/// counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DefIntConstant {
    pub reg: Register,
    pub value: [i32; 4],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SrcModifier {
    None,
    Neg,
    Bias,
    BiasNeg,
    Sign,
    SignNeg,
    Comp,
    X2,
    X2Neg,
    Dz,
    Dw,
    Abs,
    AbsNeg,
    Not,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Swizzle(pub [u8; 4]);

impl Swizzle {
    pub const IDENTITY: Self = Self([0, 1, 2, 3]);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WriteMask(pub u8);

impl WriteMask {
    pub const ALL: Self = Self(0b1111);
    pub const fn covers(self, component: u8) -> bool {
        self.0 & (1 << component) != 0
    }
}

/// Relative addressing operand (a0-style indexing).
///
/// Parsed but unsupported by the emitter, which rejects it at compile time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RelativeAddr {
    pub reg: Register,
    pub swizzle: Swizzle,
}

bitflags::bitflags! {
    /// Destination result modifiers (`D3DSPDM_*`), packed from token bits 20-23.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct DstMods: u8 {
        /// `_sat` — clamp the result to `[0, 1]`.
        const SATURATE = 1 << 0;
        /// `_pp` — partial-precision hint (parsed; the emitter ignores it).
        const PARTIAL_PRECISION = 1 << 1;
        /// `_centroid` — multisample centroid interpolation hint.
        ///
        /// Parsed; the emitter ignores it.
        const CENTROID = 1 << 2;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DstOperand {
    pub reg: Register,
    pub write_mask: WriteMask,
    pub mods: DstMods,
    pub shift_scale: i8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SrcOperand {
    pub reg: Register,
    pub swizzle: Swizzle,
    pub modifier: SrcModifier,
    pub rel_addr: Option<RelativeAddr>,
}

/// D3D9 comparison-function selector (`D3DSPC_*`).
///
/// Encoded in the instruction-token bits 16-23 of `ifc` / `breakc` /
/// `setp`. The numeric values match Microsoft's `D3DSHADER_COMPARISON`
/// enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CmpFunc {
    Gt = 1,
    Eq = 2,
    Ge = 3,
    Lt = 4,
    Ne = 5,
    Le = 6,
}

impl CmpFunc {
    #[must_use]
    pub const fn from_raw(raw: u8) -> Option<Self> {
        Some(match raw {
            1 => Self::Gt,
            2 => Self::Eq,
            3 => Self::Ge,
            4 => Self::Lt,
            5 => Self::Ne,
            6 => Self::Le,
            _ => return None,
        })
    }

    /// MSL operator for the comparison.
    #[must_use]
    pub const fn op(self) -> &'static str {
        match self {
            Self::Gt => ">",
            Self::Eq => "==",
            Self::Ge => ">=",
            Self::Lt => "<",
            Self::Ne => "!=",
            Self::Le => "<=",
        }
    }
}

bitflags::bitflags! {
    /// Per-instruction control-token flags.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct InstrFlags: u8 {
        /// SM2.x / SM3 predicated execution (control-token bit 28).
        ///
        /// The gating operand is carried separately in
        /// `Instruction::predicate`.
        const PREDICATED = 1 << 0;
        /// `texldp` — the `D3DSI_TEXLD_PROJECT` control modifier (bit 16) on a SM2+ `texld`.
        ///
        /// The coordinate is divided by its `.w` before sampling. Clear for
        /// every other opcode (and for plain `texld`).
        const TEX_PROJECTED = 1 << 1;
        /// `D3DSI_COISSUE` (bit 30): co-issued with the preceding instruction.
        ///
        /// The pairing is on the `ps_1_x` vector/scalar pipes. Only `cnd`
        /// honours it — a co-issued non-alpha `cnd` (`ps_1_1`..`1_3`) selects
        /// src1.
        const COISSUE = 1 << 2;
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct Instruction {
    pub opcode: Opcode,
    pub dst: Option<DstOperand>,
    pub srcs: Vec<SrcOperand>,
    pub flags: InstrFlags,
    /// The predicate source operand parsed when `InstrFlags::PREDICATED` is set.
    ///
    /// SM3 has a single predicate register `p0`, but the operand carries a
    /// swizzle (which lane(s) gate the write) and an optional `Not` modifier.
    /// `None` when the instruction is not predicated.
    pub predicate: Option<SrcOperand>,
    /// Comparison function for `ifc` / `breakc` / `setp` opcodes; `None` for everything else.
    ///
    /// Decoded from the instruction token's bits 16-23.
    pub cmp_func: Option<CmpFunc>,
}

#[derive(Debug)]
pub struct DxsoProgram {
    pub shader_type: ShaderType,
    pub major: u8,
    pub minor: u8,
    pub declarations: Vec<Declaration>,
    pub def_constants: Vec<DefConstant>,
    pub def_int_constants: Vec<DefIntConstant>,
    pub instructions: Vec<Instruction>,
    /// SM2.x / SM3 subroutines: `Label N` / `Ret` pairs.
    ///
    /// The body between them lives here keyed by label index, and the emitter
    /// inline-expands at each `Call sN` or `CallNz`. Empty for shaders that
    /// don't use subroutines.
    pub subroutines: std::collections::BTreeMap<u32, Vec<Instruction>>,
}

impl DxsoProgram {
    /// Whether a pixel shader declares an input register with a usage the D3D9 validator forbids.
    ///
    /// The only forbidden case is `dcl_position0 v#`: the rasterizer position
    /// (`POSITION`, usage index 0) arrives in the special `vPos` register, not
    /// a `v#` input, so declaring it on a general input is invalid and
    /// `CreatePixelShader` must reject it. Higher position indices
    /// (`dcl_position1 v#`, …) are ordinary user semantics and are valid: a
    /// `dcl_position0` pixel shader yields `D3DERR_INVALIDCALL` while a
    /// `dcl_position1` shader yields `S_OK`. Always `false` for a vertex
    /// shader.
    ///
    /// SM3 only: SM1/SM2 pixel-shader `dcl` tokens encode the semantic by
    /// register type (v# = color, t# = texcoord), so their usage bits must NOT
    /// be read as a forbidden POSITION — `dcl v0` in `ps_2_0` (usage bits 0) is a
    /// valid color input, not `dcl_position0`.
    #[must_use]
    pub fn has_invalid_pixel_input_decl(&self) -> bool {
        if self.shader_type != ShaderType::Pixel || self.major < 3 {
            return false;
        }
        self.declarations.iter().any(|d| {
            matches!(
                d,
                Declaration::Semantic {
                    usage: DeclUsage::Position,
                    usage_index: 0,
                    reg: Register {
                        kind: RegKind::Input,
                        ..
                    },
                }
            )
        })
    }

    /// Highest float constant register `cN` read by an instruction in the main body.
    ///
    /// Matrix instructions span multiple consecutive rows starting from their
    /// second source, so their range is expanded to cover the whole matrix.
    /// Only `instructions` is walked: a constant read solely from a subroutine
    /// body in `subroutines` is not counted, unlike [`Self::max_const_index`],
    /// which covers both. Returns `None` if the main body reads no constants
    /// from the buffer.
    #[must_use]
    pub fn max_const_reg(&self) -> Option<u16> {
        let mut max: Option<u16> = None;
        for inst in &self.instructions {
            let (mat_src_idx, mat_rows) = match inst.opcode {
                Opcode::M4x4 | Opcode::M3x4 => (Some(1usize), 4u16),
                Opcode::M4x3 | Opcode::M3x3 => (Some(1usize), 3u16),
                Opcode::M3x2 => (Some(1usize), 2u16),
                _ => (None, 1u16),
            };
            for (i, src) in inst.srcs.iter().enumerate() {
                if src.reg.kind != RegKind::Const {
                    continue;
                }
                let span = if mat_src_idx == Some(i) { mat_rows } else { 1 };
                let top = src.reg.index + span - 1;
                max = Some(max.map_or(top, |m| m.max(top)));
            }
        }
        max
    }

    /// Highest register index the shader touches in one constant register file.
    ///
    /// `ConstInt` → `iN`, `ConstBool` → `bN`. Scans instruction sources (main
    /// body and subroutines) plus any matching `defi` declaration. Float
    /// (`Const`) is handled by [`Self::max_const_reg`], which also expands
    /// matrix instructions. Returns `None` when the file is untouched.
    #[must_use]
    pub fn max_const_index(&self, kind: RegKind) -> Option<u16> {
        let src_max = self
            .instructions
            .iter()
            .chain(self.subroutines.values().flatten())
            .flat_map(|inst| inst.srcs.iter())
            .filter(|s| s.reg.kind == kind)
            .map(|s| s.reg.index)
            .max();
        let def_max = if kind == RegKind::ConstInt {
            self.def_int_constants.iter().map(|d| d.reg.index).max()
        } else {
            None
        };
        src_max.into_iter().chain(def_max).max()
    }

    /// Whether the shader addresses a constant register past its shader model's register file.
    ///
    /// `Create{Vertex,Pixel}Shader` must reject it with
    /// `D3DERR_INVALIDCALL`. Float files: 256 for any vertex shader (the
    /// hardware-vertex-processing `MaxVertexShaderConst`) and 8 / 32 / 224
    /// for pixel shader model 1 / 2 / 3. Integer (`iN`) and boolean (`bN`)
    /// files hold 16 registers each and only exist from `vs_2_0` / `ps_3_0`
    /// onward, so any integer or boolean register in an earlier model —
    /// notably a `ps_2_0` using `defi` / `defb` — is itself out of range.
    #[must_use]
    pub fn violates_constant_register_limits(&self) -> bool {
        let is_vertex = self.shader_type == ShaderType::Vertex;
        let float_limit: u16 = if is_vertex {
            256
        } else {
            match self.major {
                0 | 1 => 8,
                2 => 32,
                _ => 224,
            }
        };
        let int_bool_limit: u16 =
            if (is_vertex && self.major >= 2) || (!is_vertex && self.major >= 3) {
                16
            } else {
                0
            };
        let float_max = self
            .max_const_reg()
            .into_iter()
            .chain(self.def_constants.iter().map(|d| d.reg.index))
            .max();
        let over = |max: Option<u16>, limit: u16| max.is_some_and(|m| m >= limit);
        over(float_max, float_limit)
            || over(self.max_const_index(RegKind::ConstInt), int_bool_limit)
            || over(self.max_const_index(RegKind::ConstBool), int_bool_limit)
    }

    /// Whether any instruction reads a constant via relative addressing (`c[a0.<swiz> + N]`).
    ///
    /// Such shaders — notably `WoW` M2 skinning VSes that index a bone-matrix
    /// palette by per-vertex `BLENDINDICES` — read slots beyond what
    /// `max_const_reg` can see statically. Draws that bind these shaders must
    /// upload the full populated constant-buffer prefix (tracked on the d3d9
    /// side), not a statically-bounded prefix.
    #[must_use]
    pub fn uses_relative_const_addressing(&self) -> bool {
        self.instructions.iter().any(|inst| {
            inst.srcs
                .iter()
                .any(|s| s.reg.kind == RegKind::Const && s.rel_addr.is_some())
        })
    }

    /// Whether any instruction reads an integer-constant register (`iN`) not declared by `defi`.
    ///
    /// That is a *dynamic* integer constant fed by `SetVertexShaderConstantI`.
    /// `defi` constants are baked into the MSL as locals, but a dynamic `iN`
    /// (typically a `loop aL, iN` / `rep iN` counter) needs the runtime
    /// integer-constant buffer uploaded and bound; draws that bind such a
    /// shader gate that bind on this flag.
    #[must_use]
    pub fn uses_dynamic_int_constants(&self) -> bool {
        let defined: std::collections::BTreeSet<u16> =
            self.def_int_constants.iter().map(|d| d.reg.index).collect();
        self.instructions.iter().any(|inst| {
            inst.srcs
                .iter()
                .any(|s| s.reg.kind == RegKind::ConstInt && !defined.contains(&s.reg.index))
        })
    }

    /// Whether any instruction is a bump/environment-mapping op (`texbem`/`texbeml`/`bem`).
    ///
    /// Such shaders read the per-stage bump matrix + luminance scale/offset
    /// from a dedicated PS uniform; draws that bind them must upload that
    /// uniform (the d3d9 side gates the slot-12 bind on this).
    #[must_use]
    pub fn uses_bump_env(&self) -> bool {
        self.instructions
            .iter()
            .any(|inst| matches!(inst.opcode, Opcode::TexBem | Opcode::TexBemL | Opcode::Bem))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DxsoError {
    /// Bytecode ended before the program was complete.
    Truncated,
    /// First token was not recognizable as a shader header.
    InvalidHeader,
    /// Shader Model outside the SM1/SM2/SM3 range we model.
    ///
    /// Or an out-of-range minor, e.g. `vs_1_0` / `ps_1_5`.
    UnsupportedShaderModel { major: u8, minor: u8 },
    /// Opcode is a valid control-flow instruction we don't yet handle.
    UnsupportedControlFlow,
    /// Opcode byte doesn't map to any known SM1-3 opcode.
    UnknownOpcode(u16),
    /// Register type bits don't map to any known register kind.
    UnknownRegisterType(u32),
    /// Relative-addressed destination operand.
    ///
    /// Only relative source reads (`c[a0+N]`) are modelled.
    UnsupportedRelativeAddressing,
    /// Declaration usage byte doesn't map to any known D3DDECLUSAGE.
    UnknownDeclUsage(u8),
}
