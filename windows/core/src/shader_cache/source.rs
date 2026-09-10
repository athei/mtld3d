//! Retained programmable shader inputs for MSL regeneration.
//!
//! The explicit encoding carries DXSO tokens and every specialization input,
//! never Rust struct memory. Shader identities are shared with the live draw path.

use std::sync::Arc;

use super::{CachedKind, RecipeReader, ff_key_hash};
use crate::{
    dxso::{self, DxsoProgram, VariantFlags, VariantKey, VsSamplerKinds},
    ids::ProgramId,
};

/// DXSO and the stage-specific inputs needed to reproduce a compiled variant.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShaderSource {
    tokens: Arc<[u32]>,
    specialization: Specialization,
}

impl ShaderSource {
    #[must_use]
    pub fn vertex(
        program: &DxsoProgram,
        provided_input_mask: u16,
        clip_plane_count: u8,
        sampler_kinds: VsSamplerKinds,
    ) -> Self {
        Self {
            tokens: program.bytecode().clone(),
            specialization: Specialization::Vertex {
                provided_input_mask,
                clip_plane_count,
                sampler_kinds,
            },
        }
    }

    #[must_use]
    pub fn pixel(program: &DxsoProgram, variant: VariantKey) -> Self {
        Self {
            tokens: program.bytecode().clone(),
            specialization: Specialization::Pixel(variant),
        }
    }

    #[must_use]
    pub fn disk_key(&self) -> u64 {
        let id = ProgramId::from_tokens(&self.tokens);
        match self.specialization {
            Specialization::Vertex {
                provided_input_mask,
                clip_plane_count,
                sampler_kinds,
            } => vs_source_disk_key_programmable(
                id,
                provided_input_mask,
                clip_plane_count,
                sampler_kinds,
            ),
            Specialization::Pixel(variant) => ps_source_disk_key_programmable(id, variant),
        }
    }

    /// Reparse retained DXSO and emit the same variant using the current translator.
    ///
    /// # Errors
    ///
    /// Returns the parser or emitter diagnostic when this translator cannot rebuild the source.
    pub fn emit(&self, entry: &str) -> Result<String, String> {
        let program =
            dxso::parse(&self.tokens).map_err(|error| format!("DXSO parse: {error:?}"))?;
        let result = match self.specialization {
            Specialization::Vertex {
                provided_input_mask,
                clip_plane_count,
                sampler_kinds,
            } => dxso::emit_vs_programmable_named(
                &program,
                entry,
                provided_input_mask,
                clip_plane_count,
                sampler_kinds,
            ),
            Specialization::Pixel(variant) => {
                dxso::emit_ps_programmable_named(&program, variant, entry)
            }
        };
        result.map_err(|error| format!("MSL emission: {error:?}"))
    }

    pub(super) fn encode(&self, out: &mut Vec<u8>) {
        match self.specialization {
            Specialization::Vertex {
                provided_input_mask,
                clip_plane_count,
                sampler_kinds,
            } => {
                out.extend_from_slice(&provided_input_mask.to_le_bytes());
                out.extend_from_slice(&[
                    clip_plane_count,
                    sampler_kinds.volume_mask,
                    sampler_kinds.cube_mask,
                ]);
            }
            Specialization::Pixel(variant) => encode_variant(variant, out),
        }
        out.extend_from_slice(
            &u32::try_from(self.tokens.len())
                .expect("DXSO token count fits u32")
                .to_le_bytes(),
        );
        for token in self.tokens.iter() {
            out.extend_from_slice(&token.to_le_bytes());
        }
    }

    pub(super) fn decode(kind: CachedKind, reader: &mut RecipeReader<'_>) -> Option<Self> {
        let specialization = if kind.is_vertex() {
            let provided_input_mask = reader.u16()?;
            let clip_plane_count = reader.u8()?;
            let sampler_kinds = VsSamplerKinds {
                volume_mask: reader.u8()?,
                cube_mask: reader.u8()?,
            };
            if usize::from(clip_plane_count) > crate::vs_draw::MAX_CLIP_PLANES
                || (sampler_kinds.volume_mask | sampler_kinds.cube_mask) & !0x0F != 0
                || sampler_kinds.volume_mask & sampler_kinds.cube_mask != 0
            {
                return None;
            }
            Specialization::Vertex {
                provided_input_mask,
                clip_plane_count,
                sampler_kinds,
            }
        } else {
            Specialization::Pixel(decode_variant(reader)?)
        };
        let count = usize::try_from(reader.u32()?).ok()?;
        let bytes = reader.take(count.checked_mul(4)?)?;
        let tokens: Arc<[u32]> = bytes
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes(word.try_into().expect("four-byte token")))
            .collect();
        let header = *tokens.first()?;
        let pixel = match header >> 16 {
            0xFFFE => false,
            0xFFFF => true,
            _ => return None,
        };
        let major = u8::try_from((header >> 8) & 0xFF).ok()?;
        // Match the opcode-only END check used by shader creation and DXSO parsing.
        if CachedKind::from_programmable(major, pixel)? != kind || tokens.last()? & 0xFFFF != 0xFFFF
        {
            return None;
        }
        Some(Self {
            tokens,
            specialization,
        })
    }
}

/// Content identity of a programmable vertex shader and its emission inputs.
#[must_use]
pub fn vs_source_disk_key_programmable(
    vs_id: ProgramId,
    provided_input_mask: u16,
    clip_plane_count: u8,
    sampler_kinds: VsSamplerKinds,
) -> u64 {
    ff_key_hash(&(
        vs_id.raw(),
        provided_input_mask,
        clip_plane_count,
        sampler_kinds,
    ))
}

/// Content identity of a programmable pixel shader and its emission inputs.
#[must_use]
pub fn ps_source_disk_key_programmable(ps_id: ProgramId, variant: VariantKey) -> u64 {
    ff_key_hash(&(ps_id.raw(), variant))
}

// Clone is required by CacheEntry's round-trip and concurrent-writer test fixtures.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Specialization {
    Vertex {
        provided_input_mask: u16,
        clip_plane_count: u8,
        sampler_kinds: VsSamplerKinds,
    },
    Pixel(VariantKey),
}

fn encode_variant(variant: VariantKey, out: &mut Vec<u8>) {
    let VariantKey {
        alpha_func,
        fog_mode,
        fog_table_mode,
        depth_sampler_mask,
        depth_fetch_mask,
        volume_sampler_mask,
        cube_sampler_mask,
        tt_projected_mask,
        color_out_mask,
        sample_mask,
        flags,
    } = variant;
    out.extend_from_slice(&[alpha_func, fog_mode, fog_table_mode]);
    for mask in [
        depth_sampler_mask,
        depth_fetch_mask,
        volume_sampler_mask,
        cube_sampler_mask,
    ] {
        out.extend_from_slice(&mask.to_le_bytes());
    }
    out.extend_from_slice(&[tt_projected_mask, color_out_mask, sample_mask, flags.bits()]);
}

fn decode_variant(reader: &mut RecipeReader<'_>) -> Option<VariantKey> {
    Some(VariantKey {
        alpha_func: reader.u8()?,
        fog_mode: reader.u8()?,
        fog_table_mode: reader.u8()?,
        depth_sampler_mask: reader.u16()?,
        depth_fetch_mask: reader.u16()?,
        volume_sampler_mask: reader.u16()?,
        cube_sampler_mask: reader.u16()?,
        tt_projected_mask: reader.u8()?,
        color_out_mask: reader.u8()?,
        sample_mask: reader.u8()?,
        flags: VariantFlags::from_bits(reader.u8()?)?,
    })
}
