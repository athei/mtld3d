//! DXSO opcode enumeration and classification.
//!
//! Numeric values match `D3DSIO_*` in the DirectX 9 SDK.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Opcode {
    Nop,
    Mov,
    Add,
    Sub,
    Mad,
    Mul,
    Rcp,
    Rsq,
    Dp3,
    Dp4,
    Min,
    Max,
    Slt,
    Sge,
    Exp,
    Log,
    Lit,
    Dst,
    Lrp,
    Frc,
    M4x4,
    M4x3,
    M3x4,
    M3x3,
    M3x2,
    Call,
    CallNz,
    Loop,
    Ret,
    EndLoop,
    Label,
    Dcl,
    Pow,
    Crs,
    Sgn,
    Abs,
    Nrm,
    SinCos,
    Rep,
    EndRep,
    If,
    Ifc,
    Else,
    EndIf,
    Break,
    BreakC,
    MovA,
    DefB,
    DefI,
    // SM1 (ps_1_x) legacy texture-addressing ops. `TexLd` (opcode 66) is
    // shared with SM2+ `texld`; the emitter branches on shader version to
    // pick SM1 `tex` vs SM2 `texld` semantics. The rest below are SM1-only.
    TexCoord,
    TexKill,
    TexLd,
    TexBem,
    TexBemL,
    TexReg2Ar,
    TexReg2Gb,
    TexM3x2Pad,
    TexM3x2Tex,
    TexM3x3Pad,
    TexM3x3Tex,
    TexM3x3Spec,
    TexM3x3VSpec,
    TexReg2Rgb,
    TexDp3Tex,
    TexM3x2Depth,
    TexDp3,
    TexM3x3,
    TexDepth,
    Bem,
    ExpP,
    LogP,
    Cnd,
    Def,
    Cmp,
    Dp2Add,
    Dsx,
    Dsy,
    TexLdD,
    SetP,
    TexLdL,
    BreakP,
    Phase,
    End,
    Comment,
}

impl Opcode {
    pub const fn from_u16(v: u16) -> Option<Self> {
        Some(match v {
            0 => Self::Nop,
            1 => Self::Mov,
            2 => Self::Add,
            3 => Self::Sub,
            4 => Self::Mad,
            5 => Self::Mul,
            6 => Self::Rcp,
            7 => Self::Rsq,
            8 => Self::Dp3,
            9 => Self::Dp4,
            10 => Self::Min,
            11 => Self::Max,
            12 => Self::Slt,
            13 => Self::Sge,
            14 => Self::Exp,
            15 => Self::Log,
            16 => Self::Lit,
            17 => Self::Dst,
            18 => Self::Lrp,
            19 => Self::Frc,
            20 => Self::M4x4,
            21 => Self::M4x3,
            22 => Self::M3x4,
            23 => Self::M3x3,
            24 => Self::M3x2,
            25 => Self::Call,
            26 => Self::CallNz,
            27 => Self::Loop,
            28 => Self::Ret,
            29 => Self::EndLoop,
            30 => Self::Label,
            31 => Self::Dcl,
            32 => Self::Pow,
            33 => Self::Crs,
            34 => Self::Sgn,
            35 => Self::Abs,
            36 => Self::Nrm,
            37 => Self::SinCos,
            38 => Self::Rep,
            39 => Self::EndRep,
            40 => Self::If,
            41 => Self::Ifc,
            42 => Self::Else,
            43 => Self::EndIf,
            44 => Self::Break,
            45 => Self::BreakC,
            46 => Self::MovA,
            47 => Self::DefB,
            48 => Self::DefI,
            64 => Self::TexCoord,
            65 => Self::TexKill,
            66 => Self::TexLd,
            67 => Self::TexBem,
            68 => Self::TexBemL,
            69 => Self::TexReg2Ar,
            70 => Self::TexReg2Gb,
            71 => Self::TexM3x2Pad,
            72 => Self::TexM3x2Tex,
            73 => Self::TexM3x3Pad,
            74 => Self::TexM3x3Tex,
            76 => Self::TexM3x3Spec,
            77 => Self::TexM3x3VSpec,
            78 => Self::ExpP,
            79 => Self::LogP,
            80 => Self::Cnd,
            81 => Self::Def,
            82 => Self::TexReg2Rgb,
            83 => Self::TexDp3Tex,
            84 => Self::TexM3x2Depth,
            85 => Self::TexDp3,
            86 => Self::TexM3x3,
            87 => Self::TexDepth,
            88 => Self::Cmp,
            89 => Self::Bem,
            90 => Self::Dp2Add,
            91 => Self::Dsx,
            92 => Self::Dsy,
            93 => Self::TexLdD,
            94 => Self::SetP,
            95 => Self::TexLdL,
            96 => Self::BreakP,
            0xFFFD => Self::Phase,
            0xFFFE => Self::Comment,
            0xFFFF => Self::End,
            _ => return None,
        })
    }

    /// True if this opcode's first operand token is in destination (dst) form.
    ///
    /// The generic instruction parser uses this to route that token through
    /// `decode_dst` (write mask) rather than `decode_src` (swizzle). Most such
    /// opcodes do write the register; `texkill`, `texdepth` and `texm3x2depth`
    /// only read it — they discard the fragment or write fragment depth.
    pub const fn has_destination(self) -> bool {
        matches!(
            self,
            Self::Mov
                | Self::MovA
                | Self::Add
                | Self::Sub
                | Self::Mad
                | Self::Mul
                | Self::Rcp
                | Self::Rsq
                | Self::Dp3
                | Self::Dp4
                | Self::Min
                | Self::Max
                | Self::Slt
                | Self::Sge
                | Self::Exp
                | Self::Log
                | Self::Lit
                | Self::Dst
                | Self::Lrp
                | Self::Frc
                | Self::M4x4
                | Self::M4x3
                | Self::M3x4
                | Self::M3x3
                | Self::M3x2
                | Self::Pow
                | Self::Crs
                | Self::Sgn
                | Self::Abs
                | Self::Nrm
                | Self::SinCos
                | Self::Cnd
                | Self::Cmp
                | Self::Dp2Add
                | Self::Dsx
                | Self::Dsy
                | Self::TexLd
                | Self::TexLdL
                | Self::TexLdD
                | Self::TexKill
                | Self::TexCoord
                | Self::TexBem
                | Self::TexBemL
                | Self::TexReg2Ar
                | Self::TexReg2Gb
                | Self::TexReg2Rgb
                | Self::TexM3x2Pad
                | Self::TexM3x2Tex
                | Self::TexM3x2Depth
                | Self::TexM3x3Pad
                | Self::TexM3x3Tex
                | Self::TexM3x3
                | Self::TexM3x3Spec
                | Self::TexM3x3VSpec
                | Self::TexDp3
                | Self::TexDp3Tex
                | Self::TexDepth
                | Self::Bem
                | Self::ExpP
                | Self::LogP
                | Self::SetP
        )
    }
}
