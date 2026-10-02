//! Shader model 3 vertex-to-pixel linkage by declaration semantic.
//!
//! An SM3 vertex shader names each output with a `dcl_<usage><index> oN.mask`
//! and an SM3 pixel shader each input with a `dcl_<usage><index> vN.mask`.
//! D3D9 links the two stages by semantic, not by register: the same semantic
//! may sit in different registers on the two sides, and several semantics may
//! share one register in disjoint lanes. A semantic keeps its lanes at their
//! component positions, so `TEXCOORD1` declared on `o1.zw` is lanes `zw` of
//! `TEXCOORD1` on both sides.
//!
//! Each semantic travels in a `Varyings` member named after it, so the two
//! stages, compiled separately, agree on the member without knowing each
//! other. The members every struct declares (the fixed-function set: the
//! position, sixteen texture coordinates, two colours, the secondary position
//! and fog) need nothing more. Every other semantic is an extra member, which
//! the vertex shader declares when it outputs it and the pixel shader only
//! when the bound vertex shader outputs it, since Metal rejects a pipeline
//! whose fragment function reads a member the vertex function does not write.
//! [`LinkInputs`] and [`SemanticSet`] are the two halves of that decision,
//! and `VariantKey::linked_input_mask` carries it to the pixel emitter.

use std::{collections::BTreeMap, fmt::Write};

use super::ir::{DeclUsage, Declaration, DstOperand, DxsoProgram, RegKind, ShaderType, WriteMask};

/// Most extra input semantics a pixel shader links to the vertex shader.
///
/// Bit `i` of `VariantKey::linked_input_mask` stands for the `i`-th entry of
/// [`LinkInputs`]. A pixel shader declaring more reads zero for the rest.
pub const MAX_LINKED_INPUTS: usize = 8;

/// The value a pixel shader reads for an input no vertex output supplies.
const ZERO: &str = "float4(0.0)";

/// The component names of lanes 0 to 3.
const LANE_NAMES: [char; 4] = ['x', 'y', 'z', 'w'];

/// One declaration semantic: a usage and its usage index.
// Copy: two bytes, stored in and read out of the fixed linkage arrays by value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Semantic {
    usage: DeclUsage,
    index: u8,
}

impl Semantic {
    /// The semantic of a `dcl` declaration; the usage index is a four-bit field.
    #[must_use]
    pub const fn new(usage: DeclUsage, index: u32) -> Self {
        Self {
            usage,
            index: (index & 0xF) as u8,
        }
    }

    /// A dense code below 224: the usage in the high nibble, the index in the low one.
    const fn code(self) -> u8 {
        ((self.usage as u8) << 4) | self.index
    }

    const fn from_code(code: u8) -> Self {
        let usage = match code >> 4 {
            0 => DeclUsage::Position,
            1 => DeclUsage::BlendWeight,
            2 => DeclUsage::BlendIndices,
            3 => DeclUsage::Normal,
            4 => DeclUsage::PSize,
            5 => DeclUsage::Texcoord,
            6 => DeclUsage::Tangent,
            7 => DeclUsage::Binormal,
            8 => DeclUsage::TessFactor,
            9 => DeclUsage::PositionT,
            10 => DeclUsage::Color,
            11 => DeclUsage::Fog,
            12 => DeclUsage::Depth,
            _ => DeclUsage::Sample,
        };
        Self {
            usage,
            index: code & 0xF,
        }
    }

    /// Whether the semantic travels in a member outside the set every `Varyings` declares.
    ///
    /// The clip position (`POSITION0`) and the point size (`PSIZE0`) are not
    /// members a pixel shader can read, and the rest of the fixed set is
    /// declared by every vertex and pixel struct.
    const fn is_extra(self) -> bool {
        !matches!(
            (self.usage, self.index),
            (DeclUsage::Texcoord, _)
                | (DeclUsage::Color | DeclUsage::Position, 0 | 1)
                | (DeclUsage::Fog | DeclUsage::PSize, 0)
        )
    }

    /// The `Varyings` member carrying the semantic, `<usage><index>` but for `position` and `fog`.
    fn member(self) -> String {
        let name = match self.usage {
            DeclUsage::Position if self.index == 0 => return "position".to_owned(),
            DeclUsage::Fog if self.index == 0 => return "fog".to_owned(),
            DeclUsage::Position => "position",
            DeclUsage::BlendWeight => "blendweight",
            DeclUsage::BlendIndices => "blendindices",
            DeclUsage::Normal => "normal",
            DeclUsage::PSize => "psize",
            DeclUsage::Texcoord => "texcoord",
            DeclUsage::Tangent => "tangent",
            DeclUsage::Binormal => "binormal",
            DeclUsage::TessFactor => "tessfactor",
            DeclUsage::PositionT => "positiont",
            DeclUsage::Color => "color",
            DeclUsage::Fog => "fog",
            DeclUsage::Depth => "depth",
            DeclUsage::Sample => "sample",
        };
        format!("{name}{}", self.index)
    }

    /// The value a vertex output lane of this semantic holds until the shader writes it.
    ///
    /// D3D9's defaults for an unwritten output: opaque white diffuse, no fog,
    /// the render-state point size, zero for everything else.
    const fn vertex_default(self) -> &'static str {
        match (self.usage, self.index) {
            (DeclUsage::Color | DeclUsage::Fog, 0) => "1.0",
            (DeclUsage::PSize, 0) => "vs_draw.point.x",
            _ => "0.0",
        }
    }
}

/// A set of semantics, one bit per `Semantic` code.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct SemanticSet([u64; 4]);

impl SemanticSet {
    /// The extra semantics an SM3 vertex shader outputs, each of which it emits as a member.
    ///
    /// Empty for every other vertex shader: SM1 and SM2 outputs are the fixed
    /// registers of the base set.
    #[must_use]
    pub fn vs_outputs(vs: &DxsoProgram) -> Self {
        let mut set = Self::default();
        if vs.shader_type != ShaderType::Vertex || vs.major != 3 {
            return set;
        }
        for element in elements(vs).filter(|e| is_output_kind(e.reg.0)) {
            if element.semantic.is_extra() {
                set.insert(element.semantic);
            }
        }
        set
    }

    /// Whether the set holds no semantic.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.iter().all(|w| *w == 0)
    }

    const fn insert(&mut self, semantic: Semantic) {
        let code = semantic.code();
        self.0[(code >> 6) as usize] |= 1 << (code & 63);
    }

    const fn contains(&self, semantic: Semantic) -> bool {
        let code = semantic.code();
        self.0[(code >> 6) as usize] & (1 << (code & 63)) != 0
    }

    /// The semantics in code order.
    fn iter(&self) -> impl Iterator<Item = Semantic> + '_ {
        (0..=u8::MAX)
            .filter(|code| self.0[usize::from(code >> 6)] & (1 << (code & 63)) != 0)
            .map(Semantic::from_code)
    }
}

/// The extra input semantics of an SM3 pixel shader, in first-declaration order.
///
/// The order fixes which bit of `VariantKey::linked_input_mask` stands for
/// which semantic, for the encoder that sets the bits and the emitter that
/// reads them.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct LinkInputs {
    codes: [u8; MAX_LINKED_INPUTS],
    len: u8,
}

impl LinkInputs {
    /// Collect the extra input semantics of `ps`; empty for anything but an SM3 pixel shader.
    #[must_use]
    pub fn ps_inputs(ps: &DxsoProgram) -> Self {
        let mut inputs = Self::default();
        if ps.shader_type != ShaderType::Pixel || ps.major != 3 {
            return inputs;
        }
        for element in elements(ps).filter(|e| e.reg.0 == RegKind::Input) {
            let semantic = element.semantic;
            if !semantic.is_extra() || inputs.position(semantic).is_some() {
                continue;
            }
            if usize::from(inputs.len) == MAX_LINKED_INPUTS {
                mtld3d_shared::log_once_warn!(target: super::LOG_TARGET,
                    "dxso: PS declares more than {MAX_LINKED_INPUTS} extra input semantics → \
                     the rest read zero"
                );
                break;
            }
            inputs.codes[usize::from(inputs.len)] = semantic.code();
            inputs.len += 1;
        }
        inputs
    }

    /// Whether the pixel shader reads no extra semantic.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The `linked_input_mask` for this pixel shader against a vertex shader's outputs.
    ///
    /// Bit `i` is set when the vertex shader outputs the `i`-th input.
    #[must_use]
    pub fn mask_against(&self, outputs: &SemanticSet) -> u8 {
        self.iter()
            .enumerate()
            .filter(|(_, semantic)| outputs.contains(*semantic))
            .fold(0, |mask, (i, _)| mask | (1 << i))
    }

    fn position(&self, semantic: Semantic) -> Option<usize> {
        self.codes[..usize::from(self.len)]
            .iter()
            .position(|code| *code == semantic.code())
    }

    fn iter(&self) -> impl Iterator<Item = Semantic> + '_ {
        self.codes[..usize::from(self.len)]
            .iter()
            .map(|code| Semantic::from_code(*code))
    }
}

/// One `dcl` of an input or output register with its semantic and lanes.
// Copy: a few bytes, iterated out of the declaration list by value.
#[derive(Clone, Copy)]
struct Element {
    semantic: Semantic,
    reg: (RegKind, u16),
    mask: WriteMask,
}

fn elements(program: &DxsoProgram) -> impl Iterator<Item = Element> + '_ {
    program.declarations.iter().filter_map(|decl| match decl {
        Declaration::Semantic {
            usage,
            usage_index,
            reg,
            mask,
        } => Some(Element {
            semantic: Semantic::new(*usage, *usage_index),
            reg: (reg.kind, reg.index),
            // A dcl without a mask covers the whole register.
            mask: if mask.0 == 0 { WriteMask::ALL } else { *mask },
        }),
        Declaration::Sampler { .. } => None,
    })
}

/// The register kinds an SM3 vertex shader's outputs arrive under.
///
/// The dcl carries the semantic whichever output kind the compiler picked:
/// some ship `TexcoordOut` (`D3DSPR_OUTPUT` shares its type 6), others
/// `Output`, `RastOut` or `AttrOut`, sometimes with overlapping indices, so
/// registers are keyed on the kind as well as the index.
pub const fn is_output_kind(kind: RegKind) -> bool {
    matches!(
        kind,
        RegKind::RastOut | RegKind::AttrOut | RegKind::TexcoordOut | RegKind::Output
    )
}

/// The `.xyzw` suffix selecting `mask`'s lanes, empty for all four.
fn lanes(mask: WriteMask) -> String {
    if mask == WriteMask::ALL {
        return String::new();
    }
    let mut s = String::from(".");
    for (c, name) in (0..4).zip(LANE_NAMES) {
        if mask.covers(c) {
            s.push(name);
        }
    }
    s
}

/// The first lane `mask` covers, 0 for x through 3 for w.
fn first_lane_index(mask: WriteMask) -> u8 {
    (0..4).find(|c| mask.covers(*c)).unwrap_or(0)
}

/// The first lane `mask` covers, as `x`..`w`.
fn first_lane(mask: WriteMask) -> char {
    LANE_NAMES[usize::from(first_lane_index(mask))]
}

/// How an SM3 vertex shader's output registers reach the `Varyings` members.
///
/// A register carrying one semantic is written straight into its member. A
/// register several semantics share is written into a staging local, and the
/// epilogue copies each semantic's lanes into its member, so packed
/// semantics neither collide nor lose their lanes.
pub struct VsOutputs {
    /// Write target per output register: a member, the point-size store, or a staging local.
    targets: BTreeMap<(RegKind, u16), String>,
    /// Registers several semantics share, each with its staging local and elements.
    staged: Vec<(String, Vec<Element>)>,
    /// Members beyond the base set, in code order.
    extras: Vec<Semantic>,
    /// The register holding `FOG0` and its lane.
    fog: Option<((RegKind, u16), WriteMask)>,
}

impl VsOutputs {
    /// Plan the outputs of `vs`; `None` for anything but an SM3 vertex shader.
    #[must_use]
    pub fn build(vs: &DxsoProgram) -> Option<Self> {
        if vs.shader_type != ShaderType::Vertex || vs.major != 3 {
            return None;
        }
        let mut by_reg: BTreeMap<(RegKind, u16), Vec<Element>> = BTreeMap::new();
        let mut fog = None;
        for element in elements(vs).filter(|e| is_output_kind(e.reg.0)) {
            if (element.semantic.usage, element.semantic.index) == (DeclUsage::Fog, 0) {
                fog = Some((element.reg, WriteMask(1 << first_lane_index(element.mask))));
            }
            by_reg.entry(element.reg).or_default().push(element);
        }
        let mut targets = BTreeMap::new();
        let mut staged = Vec::new();
        for (reg, elements) in by_reg {
            let direct = match elements.as_slice() {
                [only] => direct_target(*only),
                _ => None,
            };
            let target = direct.unwrap_or_else(|| {
                let local = format!("_o{}", staged.len());
                staged.push((local.clone(), elements));
                local
            });
            targets.insert(reg, target);
        }
        Some(Self {
            targets,
            staged,
            extras: SemanticSet::vs_outputs(vs).iter().collect(),
            fog,
        })
    }

    /// Write target per output register, for the destination path.
    #[must_use]
    pub const fn targets(&self) -> &BTreeMap<(RegKind, u16), String> {
        &self.targets
    }

    /// The members this vertex shader declares beyond the base set.
    #[must_use]
    pub fn extras(&self) -> &[Semantic] {
        &self.extras
    }

    /// Whether `dst` writes the lane that holds `FOG0`.
    #[must_use]
    pub fn writes_fog(&self, dst: DstOperand) -> bool {
        self.fog.is_some_and(|(reg, lane)| {
            reg == (dst.reg.kind, dst.reg.index) && dst.write_mask.0 & lane.0 != 0
        })
    }

    /// Zero the extra members and declare each staging local with its lanes' defaults.
    pub fn write_prologue(&self, out: &mut String) {
        for semantic in &self.extras {
            let _ = writeln!(out, "    out.{} = float4(0.0);", semantic.member());
        }
        for (local, elements) in &self.staged {
            let lane = |c: u8| {
                elements
                    .iter()
                    .find(|e| e.mask.covers(c))
                    .map_or("0.0", |e| e.semantic.vertex_default())
            };
            let _ = writeln!(
                out,
                "    float4 {local} = float4({}, {}, {}, {});",
                lane(0),
                lane(1),
                lane(2),
                lane(3)
            );
        }
    }

    /// Copy each staged semantic's lanes into its member at their own positions.
    pub fn write_epilogue(&self, out: &mut String) {
        for (local, elements) in &self.staged {
            for element in elements {
                let Element { semantic, mask, .. } = *element;
                let sel = lanes(mask);
                match (semantic.usage, semantic.index) {
                    (DeclUsage::PSize, 0) => {
                        let lane = first_lane(mask);
                        let _ = writeln!(out, "    _psize_storage.x = {local}.{lane};");
                    }
                    (DeclUsage::Fog, 0) => {
                        let lane = first_lane(mask);
                        let _ = writeln!(out, "    out.fog = float4({local}.{lane});");
                    }
                    _ => {
                        let member = semantic.member();
                        let _ = writeln!(out, "    out.{member}{sel} = {local}{sel};");
                    }
                }
            }
        }
    }
}

/// The member a register carrying only `element` is written into, or `None` to stage it.
///
/// `FOG0` and `PSIZE0` are scalars read from lane x of their store, so one
/// declared on another lane goes through a staging local, whose epilogue
/// moves the lane.
fn direct_target(element: Element) -> Option<String> {
    let scalar_off_x = first_lane(element.mask) != 'x';
    Some(match (element.semantic.usage, element.semantic.index) {
        (DeclUsage::PSize | DeclUsage::Fog, 0) if scalar_off_x => return None,
        (DeclUsage::PSize, 0) => "_psize_storage".to_owned(),
        _ => format!("out.{}", element.semantic.member()),
    })
}

/// How an SM3 pixel shader's input registers read the `Varyings` members.
///
/// A register carrying one semantic reads its member whole, lanes outside
/// the dcl mask included, as a title may read a lane its declaration leaves
/// out. A register several semantics share reads a prologue local assembled
/// lane by lane from the semantic that covers each lane.
pub struct PsInputs {
    /// Read expression per input register index.
    reads: BTreeMap<u16, String>,
    /// Prologue lines declaring the assembled locals of shared registers.
    locals: Vec<String>,
    /// Extra members the stage-in struct declares: the inputs the vertex shader outputs.
    extras: Vec<Semantic>,
}

impl PsInputs {
    /// Plan the inputs of `ps` given which of its extra semantics the bound vertex shader outputs.
    #[must_use]
    pub fn build(ps: &DxsoProgram, linked_input_mask: u8) -> Self {
        let mut plan = Self {
            reads: BTreeMap::new(),
            locals: Vec::new(),
            extras: Vec::new(),
        };
        let inputs = elements(ps).filter(|e| e.reg.0 == RegKind::Input);
        if ps.major != 3 {
            // SM1/SM2 `dcl vN` is structural: input N is diffuse or specular colour N.
            for element in inputs {
                let index = element.reg.1;
                plan.reads.insert(index, format!("in.color{index}"));
            }
            return plan;
        }
        let link = LinkInputs::ps_inputs(ps);
        plan.extras = link
            .iter()
            .enumerate()
            .filter(|(i, _)| linked_input_mask & (1 << i) != 0)
            .map(|(_, semantic)| semantic)
            .collect();
        let read = |semantic: Semantic| -> Option<String> {
            match (semantic.usage, semantic.index) {
                // The clip position and the point size are no varyings a
                // pixel shader can read; `dcl_position0 vN` is rejected at
                // creation and reads the rasterizer position here.
                (DeclUsage::Position, 0) => Some("in.position".to_owned()),
                (DeclUsage::PSize, 0) => None,
                _ if !semantic.is_extra() => Some(format!("in.{}", semantic.member())),
                _ => {
                    let position = link.position(semantic);
                    let linked = position.is_some_and(|i| linked_input_mask & (1 << i) != 0);
                    if position.is_none() {
                        mtld3d_shared::log_once_info_by!(target: super::LOG_TARGET,
                            key: u64::from(semantic.code()),
                            "dxso: PS input {} is past the first {MAX_LINKED_INPUTS} extra \
                             input semantics → reads zero",
                            semantic.member()
                        );
                    } else if !linked {
                        mtld3d_shared::log_once_info_by!(target: super::LOG_TARGET,
                            key: u64::from(semantic.code()),
                            "dxso: PS input {} has no vertex output → reads zero",
                            semantic.member()
                        );
                    }
                    linked.then(|| format!("in.{}", semantic.member()))
                }
            }
        };
        let mut by_reg: BTreeMap<u16, Vec<Element>> = BTreeMap::new();
        for element in inputs {
            by_reg.entry(element.reg.1).or_default().push(element);
        }
        for (index, elements) in by_reg {
            if let [only] = elements.as_slice() {
                let expr = read(only.semantic).unwrap_or_else(|| ZERO.to_owned());
                plan.reads.insert(index, expr);
                continue;
            }
            let lane = |c: u8| {
                let element = elements
                    .iter()
                    .find(|e| e.mask.covers(c))
                    .unwrap_or(&elements[0]);
                let name = LANE_NAMES[usize::from(c)];
                read(element.semantic).map_or_else(|| "0.0".to_owned(), |e| format!("{e}.{name}"))
            };
            let local = format!("_v{index}");
            plan.locals.push(format!(
                "    float4 {local} = float4({}, {}, {}, {});",
                lane(0),
                lane(1),
                lane(2),
                lane(3)
            ));
            plan.reads.insert(index, local);
        }
        plan
    }

    /// Read expression per input register index.
    #[must_use]
    pub const fn reads(&self) -> &BTreeMap<u16, String> {
        &self.reads
    }

    /// The members the stage-in struct declares beyond the base set.
    #[must_use]
    pub fn extras(&self) -> &[Semantic] {
        &self.extras
    }

    /// Declare the locals of registers several semantics share.
    pub fn write_prologue(&self, out: &mut String) {
        for line in &self.locals {
            out.push_str(line);
            out.push('\n');
        }
    }
}

/// Declare one `float4` member per extra semantic, appended to a `Varyings` struct.
pub fn write_extra_members(out: &mut String, extras: &[Semantic]) {
    for semantic in extras {
        let _ = writeln!(out, "    float4 {};", semantic.member());
    }
}

#[cfg(test)]
mod tests;
